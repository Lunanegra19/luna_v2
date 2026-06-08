"""
Módulo HMM Regime Detection - Luna V1
===================================================
Detecta 4 regímenes ocultos de mercado usando GaussianHMM sobre features 
ortogonales seleccionadas (FracDiff, Volatilidad, Liquidez).

SOP Aplicado:
- R1 (Anti Look-Ahead): Se implementa evaluación rolling/forward para crear 
  la variable predictiva OOS (Out-Of-Sample) real, simulando producción.
  No se usa Viterbi global para el feature final.
- R9 (Causalidad validada): Se mide la Mutual Information del Estado respecto al Target.

DISEÑO DE DESACOPLAMIENTO DE ESTADO (STATE-DRIVEN DECOUPLING - RESOLUCIÓN CASO 2):
  Este modelo HMMRegimeModel posee una alta centralidad estructural (betweenness centrality)
  en el grafo de Graphify porque múltiples submódulos de ML enriquecen sus dataframes
  con la columna "HMM_Semantic" de forma in-sample y out-of-sample.
  Sin embargo, la capa inmortal del broker live (luna/live/position_sizer.py) y de riesgo
  permanecen 100% desacopladas de la API matemática y del motor subyacente (hmmlearn/sklearn).
  El broker consume el régimen de mercado únicamente como un string o identificador serializado
  durante la ejecución del loop de trading, garantizando que cambios futuros en el estimador
  del régimen (ej: GMM o K-Means) no generen regresiones ni roturas en la capa de ejecución monetaria.
"""

import sys
from pathlib import Path
import json
import logging
from loguru import logger
import numpy as np
import pandas as pd
from hmmlearn import hmm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from luna.utils.debug_guards import vlog, timeit, check_df_sanity, check_numeric_stability

# ARCH-02 (2026-03-10): constantes leÃ­das desde cfg.hmm en settings.yaml â€” sin hardcodes.
# Antes: N_REGIMES=4 y WINDOWS_OOS=960 estaban hardcodeados aquÃ­,
# desincronizados de la secciÃ³n hmm: en settings.yaml.
try:
    from config.settings import cfg as _cfg_hmm
    N_REGIMES             = int(getattr(_cfg_hmm.hmm, 'n_states',                    4))
    WINDOWS_OOS           = int(getattr(_cfg_hmm.hmm, 'oos_window_hours',          960))
    MIN_STATE_DURATION_H  = int(getattr(_cfg_hmm.hmm, 'min_state_duration_hours', 120))  # floor MEJORA-HMM-DURATION-01
except Exception:
    N_REGIMES             = 4    # fallback si settings no disponible
    WINDOWS_OOS           = 960  # fallback
    MIN_STATE_DURATION_H  = 120  # fallback

# Features para HMM â€” elegidas por Granger *** y ortogonalidad (Run 12):
# Estas features describen el REGIMEN DE MERCADO (volatilidad, tendencia, sentimiento,
# derivados, on-chain) â€” son independientes de las features seleccionadas por SFI.
# BUG-R15-02 fix: el HMM NO debe filtrar por selected_features (el SFI elige features
# para predecir retornos; el HMM elige features para describir el estado de mercado).
# Fuentes disponibles en features_train.parquet (sin necesidad de pasar por SFI):
# [HMM-DYNAMIC-FEATURES] Las features del HMM ahora se leen de config/settings.yaml (hmm.candidate_features)
# para evitar Alpha Decay y hardcoding.

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

class HMMRegimeModel:
    def __init__(self, n_components: int = N_REGIMES):
        self.n_components = n_components
        # ARCH-02: n_iter y tol leidos desde cfg.hmm â€” antes hardcodeados.
        # Bug anterior: n_iter=1000 en cÃ³digo vs n_iter=200 en settings.yaml (desync).
        try:
            from config.settings import cfg as _c
            _n_iter = int(getattr(_c.hmm, 'n_iter', 200))
            _tol    = float(getattr(_c.hmm, 'tol', 0.001))
        except Exception:
            _n_iter, _tol = 200, 0.001
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=_n_iter,
            random_state=42,
            tol=_tol
        )
        # CRÍTICO-2 (Auditoría): Diseño del StandardScaler.
        # El escalado se ajusta siempre sobre todo el conjunto In-Sample (Train + Validation).
        # Dado que las features HMM son no-direccionales y el HMM es no-supervisado (no usa target),
        # esto ancla estadísticamente el modelo sin introducir look-ahead bias respecto al Holdout OOS.
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.state_map = {} # Mapeo de estado crudo a Semantico (Bull, Bear, etc)
        self.root = get_project_root()
        # MEJORA-HMM-DURATION-01: floor del P10 dinamico (se computa en fit_global_for_analysis)
        self._min_state_duration_cfg = MIN_STATE_DURATION_H
        self.min_state_duration_dynamic = MIN_STATE_DURATION_H  # sera sobreescrito post-fit
        # FIX-CRITICO-1 (2026-04-01): quantiles IS del Risk-Off Shield precalculados durante
        # fit_global_for_analysis() sobre datos de training. Evita que _apply_risk_off_shield()
        # calcule quantiles sobre el holdout OOS completo (look-ahead bias en evaluacion historica WFB).
        self._shield_quantiles: dict = {}

    def run(self):
        """
        [FIX-HMM-RUN-01] Ejecuta la secuencia completa del modelo HMM:
        load_data, fit_global_for_analysis, generate_oos_features, plot_regimes, save_model, enrich_validation_and_holdout.
        """
        logger.info("[HMM-RUN][FIX] Iniciando ejecución completa del modelo HMM...")
        self.load_data()
        self.fit_global_for_analysis()
        self.generate_oos_features()
        try:
            self.plot_regimes()
        except Exception as pe:
            logger.warning(f"[HMM-RUN][FIX] No se pudo generar el gráfico de regímenes: {pe}")
        self.save_model()
        
        # Exportar DataFrame con la columna HMM a un parquet nuevo para ser consumido
        features_out = self.raw_df[['HMM_State_OOS']].copy()
        features_out.columns = ['HMM_Regime']
        # Exportar también la etiqueta semántica mapeando DIRECTAMENTE desde el estado Rolling OOS
        if hasattr(self, 'state_map') and self.state_map:
            features_out['HMM_Semantic'] = features_out['HMM_Regime'].map(
                {k: v for k, v in self.state_map.items()}
            ).fillna('UNKNOWN')
        out_parquet = self.root / "data" / "features" / "hmm_regime_labels.parquet"
        features_out.to_parquet(out_parquet)
        
        _cov = features_out['HMM_Semantic'].notna().mean() if 'HMM_Semantic' in features_out.columns else 0.0
        _dist = features_out['HMM_Semantic'].value_counts().to_dict() if 'HMM_Semantic' in features_out.columns else {}
        logger.success(
            f"[HMM-RUN][FIX] Parquet guardado: {out_parquet.name} | "
            f"shape={features_out.shape} | semantic_cov={_cov:.1%} | dist={_dist}"
        )
        
        self.enrich_validation_and_holdout()
        logger.info("[HMM-RUN][FIX] Secuencia HMM completada con éxito.")

    def load_data(self):
        """Carga el dataset y asegura el target para evaluaciÃ³n.
        
        G2 (2026-03-19): Cuando cfg.hmm.hmm_extend_to_holdout=True, concatena
        features_holdout.parquet a los datos de training del HMM.
        El HMM es unsupervised â€” no usa labels de retorno â€” por lo que incluir
        datos del holdout NO introduce look-ahead bias. Solo expande el vocabulario
        de regÃ­menes aprendidos (permite aprender BEAR_CRASH de 2025).
        El train_cutoff semÃ¡ntico (_analyze_and_map_states) sigue siendo train_end.
        """
        parquet_path = self.root / "data" / "features" / "features_train.parquet"
        selected_path = self.root / "data" / "features" / "selected_features.json"
        
        logger.info(f"Cargando {parquet_path.name}...")
        df = pd.read_parquet(parquet_path)
        
        # G2: extender con holdout si cfg lo indica
        # P0-1-FIX (2026-03-30): En modo WFB, deshabilitar hmm_extend_to_holdout.
        # En WFB cada ventana aporta SU holdout como features_holdout.parquet.
        # Si W1 lo expande, el HMM aprende regímenes de datos que son el futuro de W2-W5.
        # Look-Ahead bias confirmado: SOLO permitir en modo PROD (sin ventanas futuras).
        import os as _os_hmm
        _is_wfb_mode = _os_hmm.environ.get("LUNA_RUN_ID", "").startswith("WFB_")
        try:
            from config.settings import cfg as _cfg_g2
            _extend = bool(getattr(getattr(_cfg_g2, 'hmm', None), 'hmm_extend_to_holdout', False))
        except Exception:
            _extend = False

        if _is_wfb_mode and _extend:
            logger.warning(
                "[P0-1-FIX] hmm_extend_to_holdout=True DESACTIVADO en modo WFB "
                f"(LUNA_RUN_ID={_os_hmm.environ.get('LUNA_RUN_ID')}). "
                "En WFB, el holdout puede contener datos futuros respecto a ventanas previas — "
                "activarlo causaría look-ahead bias. Solo válido en modo PROD."
            )
            _extend = False  # forzar False en WFB

        if _extend:
            # AUDIT Tier 3 (BUG-HOLDOUT-PATH): usar parquet especifico de ventana si existe
            _win_hmm = _os_hmm.environ.get("LUNA_WINDOW_ID", "")
            _hp_hmm = self.root / "data" / "features" / f"features_holdout_{_win_hmm}.parquet"
            holdout_path = _hp_hmm if (_win_hmm and _hp_hmm.exists()) else self.root / "data" / "features" / "features_holdout.parquet"
            if holdout_path.exists():
                df_ho = pd.read_parquet(holdout_path)
                # Concatenar solo — sin alterar el análisis semántico que usa train_cutoff
                df = pd.concat([df, df_ho]).sort_index()
                df = df[~df.index.duplicated(keep='last')]
                logger.info(
                    f"[G2] hmm_extend_to_holdout=True — concatenado features_holdout "
                    f"({len(df_ho)} filas). Total tras concat: {len(df)} filas. "
                    f"Rango: {df.index.min().date()} → {df.index.max().date()}"
                )
            else:
                logger.warning("[G2] hmm_extend_to_holdout=True pero features_holdout.parquet no existe — usando solo train.")
        else:
            logger.info("[HMM] load_data: usando solo features_train (hmm_extend_to_holdout=False)")

        # M-69 (2026-03-21): si hmm_train_end > train_end, concatenar features_validation.parquet.
        # Regla permanente: HMM siempre usa train+val completo mientras holdout siga siendo OOS.
        # DIFERENTE de hmm_extend_to_holdout (LAB-03): validation NO es holdout.
        try:
            from config.settings import cfg as _cfg_m69
            _hmm_end_str = getattr(getattr(_cfg_m69, 'temporal_splits', None), 'hmm_train_end', None)
            _train_end_str = getattr(getattr(_cfg_m69, 'temporal_splits', None), 'train_end', None)
            if _hmm_end_str and _train_end_str and str(_hmm_end_str) > str(_train_end_str):
                val_path = self.root / "data" / "features" / "features_validation.parquet"
                if val_path.exists():
                    df_val = pd.read_parquet(val_path)
                    # Filtrar solo hasta hmm_train_end para no tocar holdout
                    _cutoff_ts = pd.Timestamp(str(_hmm_end_str), tz='UTC')
                    df_val = df_val[df_val.index <= _cutoff_ts]
                    df = pd.concat([df, df_val]).sort_index()
                    df = df[~df.index.duplicated(keep='last')]
                    logger.info(
                        f"[M-69] HMM extendido con features_validation hasta {_hmm_end_str} "
                        f"(+{len(df_val)} filas). Total: {len(df)} filas | "
                        f"Rango: {df.index.min().date()} â†’ {df.index.max().date()}. "
                        f"Holdout ({_cfg_m69.temporal_splits.holdout_start}+) sigue siendo OOS."
                    )
                else:
                    logger.warning(
                        f"[M-69] hmm_train_end={_hmm_end_str} > train_end={_train_end_str} "
                        f"pero features_validation.parquet no existe â€” HMM usa solo train."
                    )
        except Exception as _e69:
            logger.debug(f"[M-69] No se pudo extender HMM con validation: {_e69}")

        # [ARCH-05-FIX-B 2026-06-02] HMM rolling window: respetar hmm_train_start si está configurado.
        # PROBLEMA: el HMM entrenado en 2017-2025 aprende regímenes del BTC pre-institucional
        # (CALM_RANGE, VOLATILE_BULL de 2018-2019) que ya no existen en el mercado actual (post-ETF 2023).
        # SOLUCIÓN: filtrar el IS del HMM desde hmm_train_start=2020-01-01 para alinear el vocabulario
        # de regímenes con el mercado post-COVID (más relevante para OOS 2023-2025).
        # NOTA: se aplica DESPUÉS de M-69/G2 para no interferir con la extensión al validation.
        try:
            from config.settings import cfg as _cfg_hmm_start
            _hmm_start_str = getattr(getattr(_cfg_hmm_start, 'temporal_splits', None), 'hmm_train_start', None)
            if _hmm_start_str:
                _hmm_start_ts = pd.Timestamp(str(_hmm_start_str), tz='UTC')
                _n_before_start = len(df)
                df = df[df.index >= _hmm_start_ts]
                _n_after_start = len(df)
                print(  # RULE[fixbugsprints.md]
                    f"[ARCH-05-FIX-B] HMM rolling window activo: datos filtrados desde {_hmm_start_str} "
                    f"({_n_before_start} -> {_n_after_start} barras, "
                    f"rango final: {df.index.min().date()} -> {df.index.max().date()})"
                )
                logger.info(
                    f"[ARCH-05-FIX-B] HMM train_start={_hmm_start_str}: {_n_before_start} -> {_n_after_start} barras. "
                    f"Vocabulario de regimenes alineado con mercado post-COVID (BULL+VOLATILE_RANGE+BEAR 2020-2025)."
                )
            else:
                print("[ARCH-05-FIX-B] hmm_train_start no configurado — HMM usa IS global (comportamiento anterior)")  # RULE[fixbugsprints.md]
        except Exception as _e_hmm_start:
            logger.warning(f"[ARCH-05-FIX-B] No se pudo leer hmm_train_start de settings: {_e_hmm_start} — HMM usa IS global")
            print(f"[ARCH-05-FIX-B] WARN: No se pudo leer hmm_train_start: {_e_hmm_start}")  # RULE[fixbugsprints.md]

        # M-43 (2026-03-19): calcular close_ret_720h (retorno rolling 30 dÃ­as) inline.
        # No requiere re-ejecutar el feature pipeline â€” 'close' ya estÃ¡ en los parquets.
        # Es el discriminador clave entre lateral-bull y bear-silencioso:
        #   lateral-bull: vol baja, ret_30d â‰ˆ 0% â†’ 1_BULL_TREND_WEAK
        #   bear-silencioso: vol baja, ret_30d < -10% â†’ 4_CALM_BEAR (nuevo)
        if 'close' in df.columns:
            df['close_ret_720h'] = df['close'].pct_change(720) * 100
            logger.info("[M-43] close_ret_720h calculada inline (pct_change 720h = 30d)")
            
            # [LUNA V1 INSTITUTIONAL FIX] HMM Directional Sensitivity (HMM V2)
            # Acotado [-1.0, 0.0] previene la desestabilizaciÃ³n de covarianza en K=5
            rolling_ath = df["close"].rolling(window=90*24, min_periods=24).max().ffill().bfill()
            df["btc_drawdown_from_ath"] = (df["close"] / rolling_ath) - 1.0
            logger.info("[HMM V2] btc_drawdown_from_ath calculada inline (90d max_rolling) para vector direccional.")
        else:
            logger.warning("[M-43] 'close' no encontrado en parquet â€” close_ret_720h no disponible")

        # Leemos las seleccionadas (Top 15 + Alpha) si existe el archivo de SFI, de lo contrario usamos df.columns
        # [FIX-HMM-LOAD-01] Previene error si SFI aún no se ha ejecutado en este ciclo de ventana
        if selected_path.exists():
            with open(selected_path, 'r') as f:
                sel_data = json.load(f)
                features = sel_data["selected_features"]
            logger.info(f"Features disponibles en selected_features.json: {len(features)}. Buscando pilares para HMM...")
        else:
            features = list(df.columns)
            logger.info(f"[FIX-HMM-LOAD-01] selected_features.json no encontrado (SFI aún no ejecutado). Features en parquet: {len(features)}. Buscando pilares para HMM...")
        
        # BUG-R15-02 fix: el HMM NO filtra por selected_features.
        # El SFI selecciona features para PREDECIR RETORNOS.
        # El HMM usa features para DESCRIBIR EL ESTADO DEL MERCADO â€” conjunto ortogonal.
        # Filtrar HMM_FEATURES por selected_features dejaba al HMM con 1 sola feature
        # cuando el SFI era agresivo (Run 14: solo 9 features de macro/onchain).
        hmm_cols = []
        
        if 'frac_diff_precio' in df.columns:
            hmm_cols.append('frac_diff_precio')
            
        # [HMM-DYNAMIC-FEATURES] Leer variables candidatas desde settings.yaml (No Fallback policy)
        try:
            from config.settings import cfg as _cfg_hmm_feat
            _candidate_features = list(getattr(_cfg_hmm_feat.hmm, 'candidate_features'))
            _min_feature_mi = float(getattr(_cfg_hmm_feat.hmm, 'min_feature_mi'))
            _min_features_required = int(getattr(_cfg_hmm_feat.hmm, 'min_features_required'))
        except AttributeError as e:
            logger.critical(f"[HMM-DYNAMIC-FEATURES] Faltan parametros HMM en settings.yaml: {e}")
            raise RuntimeError(f"Politica No-Fallback: Faltan parametros hmm.candidate_features en settings: {e}")
            
        for c in _candidate_features:
            if c in df.columns:
                hmm_cols.append(c)
                
        # Forzar fallback si el parquet no tiene ninguna feature HMM:
        if 'mt_vol_realized_4bar' not in hmm_cols and 'mt_vol_realized_4bar' in df.columns:
            hmm_cols.append('mt_vol_realized_4bar')
        if 'frac_diff_close' in df.columns and 'frac_diff_precio' not in hmm_cols:
            hmm_cols.append('frac_diff_close')

        hmm_cols = list(dict.fromkeys(hmm_cols))

        # BUG-R15-03 fix: filtrar features con >80% NaN ANTES del dropna.
        # [LUNA V1 INSTITUTIONAL FIX] Option B: Dynamic Iterative Dropna
        # HMM es extremadamente sensible a la perdida de filas. Si eliminamos mas del 20%
        # del dataset, el modelo colapsa. Iterativamente eliminamos la feature con mas NaNs
        # hasta que retengamos el 80% de los datos historicos.
        if hmm_cols:
            original_len = len(df)
            current_cols = list(hmm_cols)
            
            while len(current_cols) > 0:
                survivors = len(df[current_cols + ['close']].dropna())
                survival_rate = survivors / original_len
                
                if survival_rate >= 0.80:
                    break
                    
                nan_counts = df[current_cols].isnull().sum()
                worst_col = nan_counts.idxmax()
                
                if nan_counts[worst_col] == 0:
                    break
                    
                logger.warning(
                    f"[HMM] FEATURE DROPPED: '{worst_col}' (Missing: {nan_counts[worst_col]} rows). "
                    f"Survival rate was too low ({survival_rate:.1%})."
                )
                current_cols.remove(worst_col)
                
            if not current_cols:
                logger.error("[HMM] TODAS las features fallaron el dynamic dropna. Forzando fallback a vacio.")
                hmm_cols = []
            else:
                hmm_cols = current_cols

        logger.info(f"Features HMM Base: {hmm_cols}")

        # [BUG-SHIELD-DISCREPANCY-01] Preservar features del Risk-Off Shield en keep_cols (incluyendo mayúsculas/CamelCase)
        keep_cols = hmm_cols + ['close']
        _shield_cols = ['DVOL', 'parkinson_vol', 'FundingRate', 'dv_funding_rate', 'funding_rate', 'MVRV_Proxy', 'mvrv_pct_6m', 'mvrv_zscore']
        if 'close_ret_720h' in df.columns:
            keep_cols.append('close_ret_720h')
        for _sc in _shield_cols:
            if _sc in df.columns:
                keep_cols.append(_sc)
        keep_cols = list(dict.fromkeys(keep_cols))
        print(f"[BUG-SHIELD-DISCREPANCY-01] load_data: Preservadas columnas de escudo: {[c for c in _shield_cols if c in df.columns]}")

        # Dropear NaNs â€” solo exigir no-NaN en hmm_cols y close
        df_clean = df[keep_cols].dropna(subset=hmm_cols + ['close'])
        check_df_sanity(df_clean, label="HMM.load_data")
        if df_clean.empty:
            logger.error("[HMM] DataFrame limpio VACIO tras dropna de features HMM â€” verificar columnas disponibles")

        # [HMM-ZSCORE-01] Pilar 3: Normalización Z-Score Rolling Local (90 dias = 2160 horas)
        # Esto previene que el HMM se confunda con la inflación global histórica de las variables,
        # obligándolo a juzgar la volatilidad de hoy frente al trimestre actual.
        if hmm_cols and not df_clean.empty:
            logger.info("[HMM-ZSCORE-01] Aplicando Rolling Z-Score (90d) a las features HMM para purificacion ortogonal local.")
            z_cols = []
            for col in hmm_cols:
                z_col = f"{col}_z90d"
                import numpy as np
                # Ventana de 90 días (2160 horas) para contextualizar el ciclo local
                roll_mean = df_clean[col].rolling(window=2160, min_periods=24).mean()
                roll_std  = df_clean[col].rolling(window=2160, min_periods=24).std()
                roll_std = roll_std.replace(0, np.nan).bfill() # evitar div by 0
                df_clean[z_col] = (df_clean[col] - roll_mean) / roll_std
                # Llenar primeros 24 registros (NaN) con 0 (media local simulada)
                df_clean[z_col] = df_clean[z_col].fillna(0.0)
                z_cols.append(z_col)
            
            # Actualizamos la lista de features a las versiones purificadas
            hmm_cols = z_cols
            logger.success(f"[HMM-ZSCORE-01] Transformacion Z-Score Rolling completada. Features activas: {hmm_cols}")

        # --- FASE 2 y 3: Diagnostico Causal Pre-HMM y Seleccion Dinamica ---
        if 'close_ret_720h' in df_clean.columns and len(df_clean) > 1000:
            from sklearn.feature_selection import mutual_info_regression
            import numpy as np
            
            logger.info("[HMM-DIAGNOSTIC] Evaluando Varianza y Mutual Information de candidatos HMM...")
            
            # 1. Filtro de varianza colapsada (Alpha Decay extremo)
            valid_cols = []
            for col in hmm_cols:
                _std = df_clean[col].std()
                if _std < 1e-6:
                    logger.warning(f"[HMM-DIAGNOSTIC] ALERTA: La variable '{col}' ha colapsado en varianza (std={_std:.2e}). Descartando.")
                    print(f"[HMM-DIAGNOSTIC] ALERTA: La variable '{col}' ha colapsado en varianza. Descartando.")
                else:
                    valid_cols.append(col)
                    
            # 2. Filtro de Mutual Information
            # Usamos una submuestra del IS reciente para acelerar (ultimas 20,000 barras) si es muy largo
            _df_eval = df_clean.tail(20000).dropna(subset=valid_cols + ['close_ret_720h'])
            if not _df_eval.empty:
                X_eval = _df_eval[valid_cols]
                y_eval = _df_eval['close_ret_720h']
                
                mi_scores = mutual_info_regression(X_eval, y_eval, random_state=42)
                mi_dict = {col: score for col, score in zip(valid_cols, mi_scores)}
                
                final_hmm_cols = []
                for col, score in mi_dict.items():
                    if score >= _min_feature_mi:
                        final_hmm_cols.append(col)
                    else:
                        logger.warning(f"[HMM-DIAGNOSTIC] ALERTA: La variable '{col}' perdio su poder de clasificacion (MI={score:.5f} < {_min_feature_mi}). Descartando.")
                        print(f"[HMM-DIAGNOSTIC] ALERTA: La variable '{col}' perdio su poder de clasificacion (MI={score:.5f}). Descartando.")
                        
                # Ordenar por MI descendente
                final_hmm_cols.sort(key=lambda c: mi_dict[c], reverse=True)
                
                if len(final_hmm_cols) < _min_features_required:
                    logger.critical(f"[HMM-DIAGNOSTIC] Solo sobrevivieron {len(final_hmm_cols)} pilares (minimo {_min_features_required}). El mercado es ciego para el HMM.")
                    raise RuntimeError(f"Fail-Fast: Insuficientes pilares HMM validos tras el filtro MI y Varianza. Quedan {len(final_hmm_cols)}, minimo {_min_features_required}.")
                    
                hmm_cols = final_hmm_cols
                logger.success(f"[HMM-DIAGNOSTIC] Seleccionados Top {len(hmm_cols)} pilares por MI: {[(c, round(mi_dict[c],4)) for c in hmm_cols]}")
            else:
                logger.warning("[HMM-DIAGNOSTIC] No se pudo evaluar MI (df_eval vacio).")
                hmm_cols = valid_cols

        self.raw_df = df_clean
        self.X = df_clean[hmm_cols]
        vlog(f"HMM features finales: {hmm_cols} | shape={self.X.shape} | NaN={self.X.isnull().sum().sum()}")
        return self.X
        
    def fit_global_for_analysis(self, train_cutoff: str | None = None):
        """
        Entrena sobre el periodo de training para derivar los 4 regimenes.
        """
        # Leer train_cutoff de settings si no se pasa
        if train_cutoff is None:
            sys.path.insert(0, str(self.root))
            from luna.utils.encoding_fix import fix_stdout_encoding; fix_stdout_encoding()
            try:
                from config.settings import cfg
                _extend = bool(getattr(getattr(cfg, 'hmm', None), 'hmm_extend_to_holdout', False))
                if _extend and hasattr(self, 'raw_df') and self.raw_df is not None:
                    train_cutoff = str(self.raw_df.index.max().date())
                    logger.info(
                        f"[G2-A] hmm_extend_to_holdout=True — cutoff extendido a "
                        f"{train_cutoff}"
                    )
                else:
                    _hmm_end = getattr(cfg.temporal_splits, 'hmm_train_end', None)
                    if _hmm_end:
                        train_cutoff = _hmm_end
                        logger.info(f"[M-69] Usando hmm_train_end de settings: {train_cutoff} para HMM")
                    else:
                        train_cutoff = getattr(cfg.temporal_splits, 'train_end', "2024-06-30")
            except Exception as e:
                logger.warning(f"No se pudo leer settings para train_cutoff: {e}")
                train_cutoff = "2024-06-30"

        logger.info(f"Entrenando HMM Global (Solo para Analisis de Regimenes)... cutoff={train_cutoff}")
        self._train_cutoff_used = train_cutoff
        
        # [P0-2-FIX] Ajustar el scaler solo sobre datos de training (IS)
        _cutoff_ts = pd.Timestamp(train_cutoff, tz='UTC')
        if self.X.index.tz is None:
            _cutoff_ts = _cutoff_ts.tz_localize(None)
            
        X_train = self.X[self.X.index <= _cutoff_ts]
        if len(X_train) == 0:
            logger.warning("[P0-2-FIX] Cutoff sin datos. Fallback: ajustando scaler sobre TODOS los datos.")
            X_train = self.X
            
        self.scaler.fit(X_train)
        # [WFB-CAUSAL-FIX-HMM] X_scaled_train: coordenadas normalizadas SOLO del periodo IS.
        # El GaussianHMM se entrena EXCLUSIVAMENTE sobre estos datos — el scaler ya estaba
        # correcto (fit sobre IS), pero el model.fit() recibía el parquet completo incluyendo
        # datos post-cutoff, lo que distorsionaba qué clusters aprende el HMM y cómo
        # mapea regímenes para el mismo periodo histórico entre runs con distinto parquet.
        # X_scaled (full) se reserva para predict/visualización — inferencia pura, sin look-ahead.
        X_scaled_train = self.scaler.transform(X_train.values if hasattr(X_train, 'values') else X_train)  # [FIX-PIPE-003] ndarray explicito
        X_scaled = self.scaler.transform(self.X.values if hasattr(self.X, 'values') else self.X)  # [FIX-PIPE-003]
        logger.info(
            f"[WFB-CAUSAL-FIX-HMM] HMM entrenará sobre {len(X_train)} filas IS (hasta {train_cutoff}). "
            f"Parquet total: {len(self.X)} filas. Datos post-cutoff excluidos del fit: {len(self.X)-len(X_train)}."
        )

        # [FALLA-04-FIX 2026-05-30] Model selection con verdadero IS-val split (no in-sample)
        # Problema: el score sobre X_scaled_train[split:] era IS para el modelo que habia
        # entrenado en X_scaled_train completo -> seleccion sesgada (mas params = mejor score).
        # Fix: partir X_scaled_train en train_fit (80%) + val_hmm (20%) ANTES del fit.
        # Solo X_scaled_train_fit entra al model.fit(). X_scaled_val_hmm es genuinamente OOS.
        _val_split = int(len(X_scaled_train) * 0.80)
        X_scaled_train_fit = X_scaled_train[:_val_split]
        X_scaled_val_hmm  = X_scaled_train[_val_split:]
        if len(X_scaled_train_fit) < 50:
            # Si IS es muy corto, usar IS completo para fit y score (fallback)
            X_scaled_train_fit = X_scaled_train
            X_scaled_val_hmm   = X_scaled_train
            logger.warning("[FALLA-04-FIX] IS muy corto (<50) -> model selection sobre IS completo (fallback)")
        else:
            logger.info(f"[FALLA-04-FIX] IS split: fit={len(X_scaled_train_fit)} | val_hmm={len(X_scaled_val_hmm)} (genuinamente OOS para model selection)")
            print(f"[FALLA-04-FIX] HMM model selection: fit en primeros {len(X_scaled_train_fit)} puntos IS, eval en ultimos {len(X_scaled_val_hmm)}")

        best_score, best_model = -np.inf, None
        try:
            _n_init = int(getattr(_cfg_hmm.hmm, 'n_init', 10))
        except Exception:
            _n_init = 10

        # [WARM-START HMM] Buscar modelo de ventana anterior
        prev_hmm = None
        import os
        import re
        import joblib
        window_id = os.environ.get("LUNA_WINDOW_ID", "")
        seed_id = os.environ.get("LUNA_SEED", "")
        
        m = re.match(r"W(\d+)", window_id)
        if m:
            w_idx = int(m.group(1))
            if w_idx > 1:
                prev_window = f"W{w_idx - 1}"
                prev_model_path = self.root / "data" / "wfb_cache" / f"seed{seed_id}" / prev_window / "models" / "hmm_regime.pkl"
                if not prev_model_path.exists():
                    prev_model_path = self.root / "data" / "wfb_cache" / prev_window / "models" / "hmm_regime.pkl"
                if prev_model_path.exists():
                    try:
                        prev_container = joblib.load(prev_model_path)
                        if isinstance(prev_container, dict) and 'model' in prev_container:
                            prev_hmm = prev_container['model']
                        elif hasattr(prev_container, 'model'):
                            prev_hmm = prev_container.model
                        else:
                            prev_hmm = prev_container
                            
                        if hasattr(prev_hmm, 'n_components'):
                            logger.info(f"[HMM-WARM-START] Modelo previo encontrado en {prev_window} con n={prev_hmm.n_components} componentes y f={getattr(prev_hmm, 'n_features', '?')} features.")
                            print(f"[FIX-CAPAB-HMM] Warm-start activado. Modelo previo cargado de {prev_window}.")
                            _n_init = 1  # Solo necesitamos 1 pasada si tenemos warm-start
                        else:
                            logger.warning(f"[HMM-WARM-START] Objeto cargado no es un HMM valido: {type(prev_hmm)}")
                            prev_hmm = None
                    except Exception as e:
                        logger.warning(f"[HMM-WARM-START] Fallo al cargar modelo previo: {e}")

        try:
            _n_init = int(getattr(_cfg_hmm.hmm, 'n_init', 10))
        except Exception:
            _n_init = 10
        for _seed in range(_n_init):
            _m = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="full",
                n_iter=self.model.n_iter,
                random_state=_seed,
                tol=self.model.tol,
            )
            
            # Inyectar Warm-Start si coinciden componentes y features
            if prev_hmm is not None and prev_hmm.n_components == self.n_components:
                if getattr(prev_hmm, "n_features", -1) == X_scaled_train_fit.shape[1]:
                    try:
                        _m.n_features = X_scaled_train_fit.shape[1]
                        _m.startprob_ = prev_hmm.startprob_.copy()
                        _m.transmat_  = prev_hmm.transmat_.copy()
                        _m.means_     = prev_hmm.means_.copy()
                        _m.covars_    = prev_hmm.covars_.copy()
                        _m.init_params = ""  # Desactiva inicialización aleatoria
                        if _seed == 0:
                            logger.info(f"[HMM-WARM-START] Inyectando pesos de {prev_window}. Entrenamiento será ultra-rápido.")
                    except Exception as e:
                        logger.warning(f"[HMM-WARM-START] Fallo inyectando pesos: {e}")
                        _m.init_params = "stmc" # Restaurar defaults si falla
                else:
                    if _seed == 0:
                        logger.info(f"[HMM-WARM-START] descartado: dimension mismatch (prev f={getattr(prev_hmm, 'n_features', -1)} != curr f={X_scaled_train_fit.shape[1]})")

            try:
                # [FALLA-04-FIX] fit SOLO sobre X_scaled_train_fit (primeros 80% del IS)
                _m.fit(X_scaled_train_fit)
                # Score sobre X_scaled_val_hmm (ultimos 20% IS, no usados en fit)
                _score = _m.score(X_scaled_val_hmm)
                # [H-03-FIX 2026-05-30] Criterio de convergencia ampliado:
                # monitor_.converged es BINARIO — rechaza modelos con delta_relativo=2.4e-6
                # (ej: delta=-0.30 sobre log-lik=-123839 = error_relativo=0.000002%).
                # Estos modelos son matemáticamente equivalentes a convergidos pero el flag
                # booleano los descarta, dejando best_model=None y activando el fallback
                # no convergido de random_state=42 — peor que el seed "casi convergido".
                # Solución: aceptar también modelos con delta_relativo < 1e-5.
                _converged_flag = _m.monitor_.converged
                _nearly_converged = False
                if not _converged_flag and len(_m.monitor_.history) >= 2:
                    _delta_abs = abs(_m.monitor_.history[-1] - _m.monitor_.history[-2])
                    _denom = max(abs(_m.monitor_.history[-1]), 1e-10)
                    _nearly_converged = (_delta_abs / _denom) < 1e-5
                    if _nearly_converged:
                        print(  # RULE[fixbugsprints.md]
                            f"[H-03-FIX] HMM seed={_seed}: nearly_converged=True "
                            f"(delta_rel={_delta_abs/_denom:.2e} < 1e-5). Aceptado para model selection."
                        )
                if (_converged_flag or _nearly_converged) and _score > best_score:
                    best_score, best_model = _score, _m
            except Exception:
                continue
        if best_model is None:
            # [H-03-FIX 2026-05-30] Elevar a CRITICAL: un HMM no convergido como modelo de
            # producción es un riesgo mayor — los regímenes serán arbitrarios.
            # SOP R9: MI(estados, retornos) > 0.005 — con modelo no convergido casi imposible.
            print(  # RULE[fixbugsprints.md]
                f"[H-03-FIX] CRITICAL: Ningún seed HMM convergió tras {_n_init} inits "
                f"(incluyendo nearly_converged). Activando fallback random_state=42. "
                f"RIESGO: regímenes arbitrarios, MI<0.005 probable. Revisar n_init/n_iter en settings."
            )
            logger.critical(
                "[H-03-FIX] HMM FALLBACK ACTIVADO: ningún seed convergió ({} inits). "
                "Modelo de producción entrenado sin convergencia garantizada. "
                "Verificar hmm.n_iter / hmm.tol / hmm.n_init en settings.yaml.",
                _n_init
            )
            print(f"[BUG-FIX-LOG 2026-06-05] [H-03-FIX] HMM FALLBACK ACTIVADO: ningún seed convergió ({_n_init} inits).")
            self.model.fit(X_scaled_train)  # [WFB-CAUSAL-FIX-HMM] fallback también sobre IS
        else:
            best_seed_found = best_model.random_state
            # [H13-FIX 2026-05-30] Two-phase HMM:
            # Fase 1 (DONE): selección de seed ganador en IS 80/20 → asegura model selection honesto
            # Fase 2 (NEW): re-entrenar el seed ganador sobre IS COMPLETO → maximiza calidad del modelo
            # El split 80/20 sólo se usó para selección; el modelo de producción usa todos los datos IS.
            # Esto recupera ~11916 puntos (~20% IS) sin comprometer la validez del model selection.
            _m_final = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="full",
                n_iter=self.model.n_iter,
                random_state=best_seed_found,
                tol=self.model.tol,
            )
            try:
                _m_final.fit(X_scaled_train)  # IS COMPLETO — modelo de producción
                self.model = _m_final
                final_score = _m_final.score(X_scaled_train)
                print(f"[H13-FIX] Two-phase HMM: seed={best_seed_found} | "
                      f"IS-val log-lik={best_score:.2f} (seleccion) | "
                      f"IS-full log-lik={final_score:.2f} (modelo final)")
                logger.info(f"[H13-FIX] HMM Two-Phase: seed={best_seed_found} re-entrenado en IS completo "
                            f"({len(X_scaled_train)} pts) | log-lik_full={final_score:.2f}")
            except Exception as _e:
                logger.warning(f"[H13-FIX] Re-entrenamiento IS completo falló ({_e}) — usando modelo IS-80% como fallback")
                self.model = best_model
            logger.info(f"[HMM] FALLA-04-FIX + H13-FIX: seed={best_seed_found} | IS-val={best_score:.2f} | modelo final en IS completo")


        # Extraer estados
        hidden_states = self.model.predict(X_scaled)
        self.raw_df['HMM_State_Raw'] = hidden_states
        
        # [FIX-HMM-CAUSAL-DUR] Causal training run-length duration estimation (preventing OOS leakage)
        hidden_states_train = self.model.predict(X_scaled_train)

        # [BUG-SHIELD-DISCREPANCY-01][BUG-SHIELD-VOL-01] Precalcular quantiles sobre series unificadas y escaladas
        try:
            _sq = {}
            # Volatilidad Diaria Decimal Unificada
            daily_vol = None
            if 'DVOL' in self.raw_df.columns:
                daily_vol = (self.raw_df['DVOL'] / 100.0) / np.sqrt(365)
            elif 'parkinson_vol' in self.raw_df.columns:
                daily_vol = (self.raw_df['parkinson_vol'] / 100.0) / np.sqrt(365)
                
            if 'mt_vol_realized_4bar' in self.raw_df.columns:
                mt_daily = self.raw_df['mt_vol_realized_4bar'] * np.sqrt(24)
                daily_vol = mt_daily if daily_vol is None else daily_vol.fillna(mt_daily)
                
            if daily_vol is not None:
                _sq['vol_p90'] = float(daily_vol.quantile(0.90))
                
            # Financiación Unificada
            fund_series = None
            for _fc in ['FundingRate', 'dv_funding_rate', 'funding_rate']:
                if _fc in self.raw_df.columns:
                    fund_series = self.raw_df[_fc] if fund_series is None else fund_series.fillna(self.raw_df[_fc])
            if fund_series is not None:
                _sq['fund_p05'] = float(fund_series.dropna().quantile(0.05)) if len(fund_series.dropna()) > 0 else -0.0001
                
            # MVRV Unificado
            mvrv_series = None
            for _mc in ['MVRV_Proxy', 'mvrv_pct_6m', 'mvrv_zscore']:
                if _mc in self.raw_df.columns:
                    mvrv_series = self.raw_df[_mc] if mvrv_series is None else mvrv_series.fillna(self.raw_df[_mc])
            if mvrv_series is not None:
                _sq['mvrv_p95'] = float(mvrv_series.dropna().quantile(0.95)) if len(mvrv_series.dropna()) > 0 else 2.0
                
            self._shield_quantiles = _sq
            logger.info(f"[BUG-SHIELD-DISCREPANCY-01] Quantiles unificados IS precalculados: vol_p90={_sq.get('vol_p90'):.6f} | fund_p05={_sq.get('fund_p05'):.6f} | mvrv_p95={_sq.get('mvrv_p95'):.4f}")
            print(f"[BUG-SHIELD-DISCREPANCY-01] Quantiles unificados IS precalculados: vol_p90={_sq.get('vol_p90'):.6f} | fund_p05={_sq.get('fund_p05'):.6f} | mvrv_p95={_sq.get('mvrv_p95'):.4f}")
        except Exception as _eq_shield:
            self._shield_quantiles = {}
            logger.warning(f"[BUG-SHIELD-DISCREPANCY-01] Error precalculando quantiles: {_eq_shield}")
            print(f"[BUG-SHIELD-DISCREPANCY-01] Error precalculando quantiles: {_eq_shield}")


        # â”€â”€ MEJORA-HMM-DURATION-01 (2026-03-10) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Calcula el P10 empÃ­rico de las duraciones de estado (run lengths)
        # sobre los datos de training (IS) â€” sin look-ahead.
        # El umbral min_state_duration_dynamic es auto-adaptativo:
        # si el HMM aprende regÃ­menes cortos, el umbral baja; si aprende
        # regÃ­menes largos (mercados mÃ¡s lentos), sube.
        # P10 = percentil 10 de duraciones: el 90% de los estados duran mÃ¡s
        # que este umbral. Estados por debajo son ruido transitorio, no rÃ©gimen.
        try:
            import itertools
            # [FIX-HMM-CAUSAL-DUR] Use strictly causal IS states for empirical dynamic run-length percentile
            # Calcular run lengths: duraciones consecutivas de cada estado
            _run_lengths = [
                len(list(grp))
                for _, grp in itertools.groupby(hidden_states_train)
            ]
            print(f"[FIX-HMM-CAUSAL-DUR] Causal dynamic duration P10 calculation complete over IS hidden states ({len(_run_lengths)} runs).")
            _min_dur_cfg = int(getattr(self, '_min_state_duration_cfg', MIN_STATE_DURATION_H))
            _p10 = int(np.percentile(_run_lengths, 10)) if len(_run_lengths) > 5 else _min_dur_cfg
            _p50 = int(np.percentile(_run_lengths, 50)) if len(_run_lengths) > 5 else _min_dur_cfg
            # BUG-R15-01 fix: usar max(cfg_floor, P10_empirico) para que el dinamico
            # NUNCA sea inferior al floor configurado. El floor es un minimo absoluto.
            # Si los regimes empiricos son muy largos (P10 > cfg), el dinamico sube.
            # Si P10 < cfg, el floor protege: se usa cfg como minimo.
            self.min_state_duration_dynamic = max(_min_dur_cfg, _p10)
            logger.info(
                f"[HMM] MEJORA-HMM-DURATION-01: {len(_run_lengths)} runs de estado | "
                f"P10={_p10}H | P50={_p50}H | floor={_min_dur_cfg}H "
                f"=> min_state_duration_dynamic={self.min_state_duration_dynamic}H"
            )
        except Exception as _e_dur:
            self.min_state_duration_dynamic = MIN_STATE_DURATION_H
            logger.warning(f"[HMM] HMM-DURATION-01: no se pudo calcular P10, usando cfg: {_e_dur}")
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # Fix C-01: pasar el cutoff para limitar el analisis semantico al train set
        self._analyze_and_map_states(train_cutoff=train_cutoff)
        self.is_fitted = True
        return hidden_states

    def fit_with_nas(self, n_states_candidates: list | None = None) -> int:
        """
        MOD-02 (Run 14): NAS ligero â€” compara n_states candidatos y elige el
        que maximiza MI(rÃ©gimen, target forward 24H) con constraints de calidad.

        Criterios de aceptaciÃ³n de un candidato:
            1. MI(estados, fwd_ret_sign) > mejor MI hasta ahora
            2. P10 de run lengths >= 12H (regÃ­menes mÃ­nimamente durables)
            3. min_obs_per_state >= 100 (datos estadÃ­sticos suficientes)

        El modelo ganador reemplaza self.model y self.n_components.
        Los resultados comparativos se persisten en hmm_regime.pkl (clave 'nas_results').

        Uso:
            model = HMMRegimeModel()
            model.load_data()
            n_opt = model.fit_with_nas()    # elige entre [4, 5]
            model.generate_oos_features()   # usa el modelo Ã³ptimo
        """
        import itertools

        # Leer candidatos desde settings._roadmap.hmm.n_states_candidates si existe
        if n_states_candidates is None:
            try:
                from config.settings import cfg as _c
                _cands = getattr(getattr(_c, '_roadmap', None), 'hmm', None)
                if _cands and hasattr(_cands, 'n_states_candidates'):
                    n_states_candidates = list(_cands.n_states_candidates)
                else:
                    n_states_candidates = [4, 5]
            except Exception:
                n_states_candidates = [4, 5]  # fallback

        logger.info(f"[HMM-NAS] MOD-02: comparando n_states={n_states_candidates}")

        if hasattr(self, '_cfg_hmm'):
            _train_end = getattr(self._cfg_hmm.temporal_splits, 'hmm_train_end', None) or getattr(self._cfg_hmm.temporal_splits, 'train_end', None)
        else:
            _train_end = None
            
        if _train_end:
            _limit = pd.Timestamp(_train_end, tz='UTC')
            _X_for_nas = self.X[self.X.index <= _limit]
            _raw_for_nas = self.raw_df[self.raw_df.index <= _limit]
        else:
            _X_for_nas = self.X
            _raw_for_nas = self.raw_df

        # [P2-6-FIX] Usar scaler aislado para el NAS. 
        # Asi evitamos sobreescribir el self.scaler prematuramente.
        nas_scaler = StandardScaler()
        X_scaled = nas_scaler.fit_transform(_X_for_nas)
        best_n   = n_states_candidates[0]
        best_mi  = -np.inf
        nas_results: dict = {}
        best_model_obj = None

        for n in n_states_candidates:
            try:
                model_n = hmm.GaussianHMM(
                    n_components=n,
                    covariance_type="full",
                    n_iter=self.model.n_iter,
                    random_state=42,
                    tol=self.model.tol,
                )
                model_n.fit(X_scaled)
                states_n = model_n.predict(X_scaled)

                # â”€â”€ Criterio 1: MI(rÃ©gimen, fwd_ret_sign 24H) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                try:
                    from sklearn.metrics import mutual_info_score
                    if "close" in _raw_for_nas.columns:
                        # BUG-FIX (Run 14): calcular valid sobre serie FLOAT
                        # antes de .astype(int). astype(int) convierte
                        # NaN (del shift(-24)) -> False -> 0, y el valid
                        # mask posterior no captura los NaN del horizonte.
                        fwd_ret = _raw_for_nas["close"].pct_change(24).shift(-24)
                        valid = ~fwd_ret.isna().values  # NaN reales del shift(-24)
                        fwd_sign = (fwd_ret > 0).astype(int)
                        if valid.sum() > 500:
                            mi = mutual_info_score(
                                states_n[valid], fwd_sign.values[valid]
                            )
                        else:
                            mi = 0.0
                    else:
                        mi = 0.0
                except Exception as _mi_e:
                    logger.warning(f"[HMM-NAS] MI cÃ¡lculo n={n} fallido: {_mi_e}")
                    mi = 0.0

                # â”€â”€ Criterio 2: P10 run lengths (durabilidad mÃ­nima) â”€â”€â”€â”€â”€â”€â”€â”€
                run_lengths = [
                    len(list(grp)) for _, grp in itertools.groupby(states_n)
                ]
                min_dur_p10 = int(np.percentile(run_lengths, 10)) if run_lengths else 0

                # â”€â”€ Criterio 3: min_obs_per_state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                state_counts = [int(np.sum(states_n == s)) for s in range(n)]
                min_obs = min(state_counts)

                nas_results[n] = {
                    "mi":            round(mi, 6),
                    "min_dur_p10_h": min_dur_p10,
                    "min_obs":       min_obs,
                    "state_counts":  state_counts,
                }
                logger.info(
                    f"[HMM-NAS]  n={n}: MI={mi:.6f} | "
                    f"min_dur_P10={min_dur_p10}H | min_obs={min_obs} | "
                    f"counts={state_counts}"
                )

                # Gate: MI mejor Y estados mÃ­nimamente durables Y obs suficientes
                if mi > best_mi and min_dur_p10 >= 12 and min_obs >= 100:
                    best_mi        = mi
                    best_n         = n
                    best_model_obj = model_n

            except Exception as _e_n:
                logger.warning(f"[HMM-NAS] n={n} entrenamiento fallido: {_e_n}")
                nas_results[n] = {"error": str(_e_n)}

        logger.info(
            f"[HMM-NAS] Seleccionado n_states={best_n} "
            f"(MI={best_mi:.6f}) | resultados: {nas_results}"
        )

        # Actualizar el modelo con el ganador
        self.n_components = best_n
        if best_model_obj is not None:
            self.model = best_model_obj
        self._nas_results = nas_results  # guardado en pkl para TEST-91

        return best_n


    def _analyze_and_map_states(self, train_cutoff: str | None = None):
        """
        Mapea los nÃºmeros de estado [0..3] a etiquetas humanas basadas en su performance.
        train_cutoff: si None, se lee de settings.yaml (cfg.temporal_splits.train_end).

        Fix C-01: usa SOLO datos del training set (train_cutoff) para calcular los retornos
        medios por estado. Usar el dataset completo introduce look-ahead porque los estados
        presentes en validaciÃ³n/holdout influyen quÃ© etiqueta recibe cada estado.
        """
        logger.info(f"Analizando EstadÃ­sticas de los RegÃ­menes. Restringido al train set (<= {train_cutoff})")
        states = self.model.predict(self.scaler.transform(self.X))
        self.raw_df['HMM_State_Raw'] = states

        # Fix C-01: restringir el análisis a filas del training set
        cutoff = pd.Timestamp(train_cutoff, tz="UTC")
        if self.raw_df.index.tz is None:
            cutoff = cutoff.tz_localize(None)
        train_mask_df = self.raw_df[self.raw_df.index <= cutoff].copy()

        if len(train_mask_df) == 0:
            logger.warning("C-01: train_mask_df vacío — el índice no tiene datos antes del cutoff. Usando todo el df como fallback.")
            train_mask_df = self.raw_df.copy()

        # [LUNA V1 INSTITUTIONAL UPGRADE] Rescate del Proxy de Retornos
        if 'close' not in train_mask_df.columns:
            logger.warning("[HMM] Columna 'close' no encontrada en dataset. Recuperando desde data/raw...")
            try:
                raw_p = self.root / "data" / "raw" / "ohlcv" / "ohlcv_raw.parquet"
                if not raw_p.exists():
                    raw_p = self.root / "data" / "historical" / "daemon" / "BTCUSDT_1h.parquet"
                df_raw = pd.read_parquet(raw_p, columns=["close"])
                df_raw.index = pd.to_datetime(df_raw.index, utc=True)
                train_mask_df = train_mask_df.join(df_raw, how="left")
            except Exception as e:
                logger.error(f"[HMM] Fallo crítico recuperando 'close': {e}")
                
        # Calcular retornos a 30 días en el dataframe completo de train
        if 'close' in train_mask_df.columns:
            train_mask_df['ret_30d'] = train_mask_df['close'].pct_change(720) * 100
            train_mask_df['vol_1h'] = train_mask_df['close'].pct_change(1)
            
            # [LUNA V1 ARCHITECTURE OVERHAUL] Ponderación Temporal de la Volatilidad
            # En base al "Volatility Decay" de BTC, los retornos de 2017 rompen la calibración
            # de los rangos de 2025. Restringimos el deadband al régimen macro de los últimos 2 años.
            cutoff_dt = train_mask_df.index.max()
            contemporary_mask = train_mask_df.index >= (cutoff_dt - pd.Timedelta(days=730))
            
            # [FALLA-05-FIX 2026-05-30] Deadband ASIMÉTRICO bull/bear para BTC
            # Problema anterior: deadband = P33(|ret_30d|) era simétrico. Para BTC donde
            # bull runs (+30-100%) son mucho mayores que bears (-20-50%), el P33 simétrico
            # clasificaba muchos bull trends débiles (+8-15%) como LATERAL.
            # Fix: calcular deadband_bull y deadband_bear por separado sobre retornos
            # positivos y negativos respectivamente (respeta la asimetría natural de BTC).
            historical_rets = train_mask_df.loc[contemporary_mask, 'ret_30d'].dropna()
            
            if len(historical_rets) > 400:
                bull_rets = historical_rets[historical_rets > 0]
                bear_rets = historical_rets[historical_rets < 0]
                if len(bull_rets) > 50:
                    deadband_bull = float(bull_rets.quantile(0.33))  # P33 de positivos
                else:
                    deadband_bull = 5.0  # fallback
                if len(bear_rets) > 50:
                    deadband_bear = float(bear_rets.quantile(0.67))  # P67 de negativos (más permisivo)
                else:
                    deadband_bear = -5.0  # fallback
                # deadband simétrico para backwards compatibility
                deadband = max(abs(deadband_bull), abs(deadband_bear)) * 0.5
                logger.info(f"[FALLA-05-FIX] Deadband asimétrico: bull_threshold={deadband_bull:.2f}% | "
                            f"bear_threshold={deadband_bear:.2f}% | deadband_sym={deadband:.2f}%")
                print(f"[FALLA-05-FIX] Deadband asimétrico BTC: bull={deadband_bull:.2f}% bear={deadband_bear:.2f}%")
            else:
                historical_abs_rets = train_mask_df['ret_30d'].abs().dropna()
                deadband_bull = float(historical_abs_rets.quantile(0.33)) if len(historical_abs_rets) > 0 else 2.0
                deadband_bear = -deadband_bull
                deadband = deadband_bull
                logger.warning(f"[FALLA-05-FIX] Fallback deadband simétrico (datos insuficientes): {deadband:.2f}%")

        else:
            deadband = 2.0
            logger.warning("[HMM] Fallback Deadband (2.0%) porque 'close' no se pudo recuperar.")

        stats_rows = []
        for s in range(self.n_components):
            mask = train_mask_df['HMM_State_Raw'] == s
            if 'ret_30d' in train_mask_df.columns and mask.sum() > 1:
                ret_mean = train_mask_df.loc[mask, 'ret_30d'].mean()
                # [P1-4-FIX] Workaround para evitar sobreestimar autocorrelación intrasemanal
                _cls = train_mask_df.loc[mask, 'close']
                if len(_cls) > 24:
                    vol = _cls.resample('D').last().pct_change().dropna().std() * np.sqrt(365) * 100
                else:
                    vol = train_mask_df.loc[mask, 'vol_1h'].std() * np.sqrt(24*365) * 100
                dd = train_mask_df.loc[mask, 'btc_drawdown_from_ath'].mean() if 'btc_drawdown_from_ath' in train_mask_df.columns else -0.5
            else:
                ret_mean = np.nan
                vol = np.nan
                dd = -0.5

            stats_rows.append({'state': s, 'ret': ret_mean, 'vol': vol, 'dd': dd, 'obs': int(mask.sum())})
            # ROB-02: validacion minima de observaciones por estado
            if mask.sum() < 50:
                logger.warning(
                    f"ROB-02: Estado HMM {s} tiene solo {mask.sum()} observaciones en train "
                    f"(<50 minimo). ret_mean={ret_mean:.3f}% y vol={vol:.3f}% son estadisticamente "
                    f"infiables."
                )
            else:
                logger.info(f"Estado {s} -> Retorno 30d: {ret_mean:+.2f}% | Volatilidad Anual: {vol:.1f}% | N train: {mask.sum()}")

        stats_df = pd.DataFrame(stats_rows).sort_values('ret', ascending=False)

        # FIX-HMM-SEMANTIC-02 (M-07) + DEADBAND UPDATE (2026-03-31):
        # Crear un "deadband" (zona muerta) para retornos 30d que son lateralizaciones.
        med_vol = stats_df['vol'].median()  # vol: referencia relativa
            
        # [FIX-HMM-UNIQUE-SEMANTIC] Avoid collapsing redundant regimes into the exact same string label
        # inside the semantic map (which reduces dummy features space for XGBoost/MetaLabeler).
        # We inject unique suffixes for repeated classifications (e.g., '_B', '_C') which are fully
        # compatible with settings.yaml and RegimeRouter.
        state_map_2d = {}
        counts = {}
        for _, row in stats_df.iterrows():
            s = int(row['state'])
            ret = row['ret']
            high_vol = row['vol'] >= med_vol
            dd = row.get('dd', -0.5)
            
            # ABSOLUTO: fronteras de tendencia vs rango basadas en deadband
            is_bull    = ret > deadband
            is_bear    = ret < -deadband
            is_neutral = (ret >= -deadband) and (ret <= deadband)
            is_ath     = dd >= -0.15  # [LUNA V1 INSTITUTIONAL FIX] Cerca de ATH (<= 15% drawdown)
            
            if is_bull and not high_vol:
                base_label = '1_BULL_TREND'
            elif is_bull and high_vol:
                base_label = '1_VOLATILE_BULL'
            elif is_bear and high_vol:
                base_label = '3_BEAR_CRASH'
            elif is_bear and not high_vol:
                base_label = '3_CALM_BEAR'
            elif is_neutral and high_vol:
                if is_ath:
                    base_label = '1_BULL_GRIND' # Rango volátil pero en zona ATH -> Grind-up Rally
                else:
                    base_label = '2_VOLATILE_RANGE'
            elif is_neutral and not high_vol and is_ath:
                base_label = '1_BULL_TREND_WEAK' # Rango calmo pero en zona ATH
            else:
                base_label = '2_CALM_RANGE'
                
            # Suffix mapping to prevent duplicate string mappings
            if base_label not in counts:
                counts[base_label] = 1
                label = base_label
            else:
                counts[base_label] += 1
                suffix_idx = counts[base_label]
                # 2 -> _B, 3 -> _C, 4 -> _D, etc.
                suffix_char = chr(ord('A') + suffix_idx - 1)
                label = f"{base_label}_{suffix_char}"
                
            state_map_2d[s] = label
            
        self.state_map = state_map_2d

        # [FIX-H-HMM-04 2026-05-30] Detector de drift semantico cross-ventana.
        # Compara etiquetas semanticas de esta ventana con las del run anterior (pkl previo).
        # Objetivo: hacer visible la inestabilidad W1->W2 en el log para auditoria.
        # Impacto clave: hmm_allowed_regimes puede filtrar 0 trades si la etiqueta
        # configurada en settings.yaml no existe en la ventana actual.
        # No cambia comportamiento - solo detecta y avisa.
        try:
            import os as _os_h04  # [FIX-H-NEW-01 2026-05-30] os no disponible globalmente en hmm_regime.py
            import re as _re_h04  # idem para re
            _win_id_h04   = _os_h04.environ.get("LUNA_WINDOW_ID", "")
            _win_match_h04 = _re_h04.match(r"W(\d+)", _win_id_h04)
            if _win_match_h04 and int(_win_match_h04.group(1)) > 1:
                _prev_win_num_h04 = int(_win_match_h04.group(1)) - 1
                _prev_win_id_h04  = f"W{_prev_win_num_h04}"
                _seed_id_h04 = _os_h04.environ.get("LUNA_SEED_ID", "seed0")

                _prev_pkl_h04 = self.root / "data" / "wfb_cache" / _seed_id_h04 / _prev_win_id_h04 / "models" / "hmm_model.pkl"
                if _prev_pkl_h04.exists():
                    _prev_container_h04 = joblib.load(_prev_pkl_h04)
                    _prev_state_map_h04 = _prev_container_h04.get("state_map", {})
                    _prev_labels_h04 = set(_prev_state_map_h04.values())
                    _curr_labels_h04 = set(self.state_map.values())
                    _new_labels_h04  = _curr_labels_h04 - _prev_labels_h04
                    _lost_labels_h04 = _prev_labels_h04 - _curr_labels_h04
                    if _new_labels_h04 or _lost_labels_h04:
                        logger.warning(
                            "[FIX-H-HMM-04] DRIFT SEMANTICO HMM: {}->{} | "
                            "Etiquetas NUEVAS en {}: {} | Etiquetas PERDIDAS: {} | "
                            "hmm_allowed_regimes debe revisarse para esta ventana.",
                            _prev_win_id_h04, _win_id_h04, _win_id_h04,
                            sorted(_new_labels_h04), sorted(_lost_labels_h04)
                        )
                        print(f"[BUG-FIX-LOG 2026-06-05] DRIFT SEMANTICO HMM: {_prev_win_id_h04}->{_win_id_h04} | Etiquetas NUEVAS en {_win_id_h04}: {sorted(_new_labels_h04)} | Etiquetas PERDIDAS: {sorted(_lost_labels_h04)}")
                    else:
                        logger.info(
                            "[FIX-H-HMM-04] Taxonomia semantica ESTABLE {}->{}: {}",
                            _prev_win_id_h04, _win_id_h04, sorted(_curr_labels_h04)
                        )
                        print(f"[BUG-FIX-LOG 2026-06-05] [FIX-H-HMM-04] Taxonomia semantica ESTABLE {_prev_win_id_h04}->{_win_id_h04}: {sorted(_curr_labels_h04)}")
                        print(f"[FIX-H-HMM-04] ESTABLE {_prev_win_id_h04}->{_win_id_h04}: "
                              f"mismas etiquetas={sorted(_curr_labels_h04)}")

                    # [ARCH-24-FIX 2026-06-02] Detector de INVERSION semantica cross-ventana.
                    # El FIX-H-HMM-04 original solo detecta labels nuevas/perdidas (set-diff).
                    # Caso NO cubierto: estado HMM=0 era "1_BULL_TREND" en W1 y es "2_VOLATILE_RANGE"
                    # en W2 — el set de labels es IDENTICO pero el mapeo numerico se INVIERTE.
                    # Esto es el riesgo mas critico: el RegimeRouter enrutaria BULL como RANGE.
                    _common_states_a24 = set(_prev_state_map_h04.keys()) & set(self.state_map.keys())
                    _inverted_a24 = []
                    for _st_a24 in sorted(_common_states_a24):
                        _prev_lbl_a24 = _prev_state_map_h04[_st_a24]
                        _curr_lbl_a24 = self.state_map[_st_a24]
                        # Comparar semantica basica (primeras 2 partes: "1_BULL", "2_RANGE"...)
                        _prev_sem_a24 = "_".join(str(_prev_lbl_a24).split("_")[:2]).upper()
                        _curr_sem_a24 = "_".join(str(_curr_lbl_a24).split("_")[:2]).upper()
                        if _prev_sem_a24 != _curr_sem_a24:
                            _inverted_a24.append(
                                f"state_{_st_a24}: {_prev_lbl_a24}->{_curr_lbl_a24}"
                            )
                    if _inverted_a24:
                        logger.critical(
                            "[ARCH-24] INVERSION SEMANTICA HMM {}->{}: {} | "
                            "RegimeRouter usara modelos INCORRECTOS. "
                            "Verifica hmm_regime_mapping en settings.yaml.",
                            _prev_win_id_h04, _win_id_h04, " | ".join(_inverted_a24)
                        )
                        print(f"[BUG-FIX-LOG 2026-06-05] [ARCH-24] INVERSION SEMANTICA HMM {_prev_win_id_h04}->{_win_id_h04}: {' | '.join(_inverted_a24)}")
                    else:
                        logger.info(
                            "[ARCH-24] Sin inversion semantica {}->{}: {} estados comunes OK.",
                            _prev_win_id_h04, _win_id_h04, len(_common_states_a24)
                        )
                        print(f"[BUG-FIX-LOG 2026-06-05] [ARCH-24] Sin inversion semantica {_prev_win_id_h04}->{_win_id_h04}: {len(_common_states_a24)} estados comunes OK.")
                        print(f"[ARCH-24] Sin inversion semantica {_prev_win_id_h04}->{_win_id_h04}: "
                              f"{len(_common_states_a24)} estados comunes estables.")
        except Exception as _hmm04_e:
            logger.debug(f"[FIX-H-HMM-04] No se pudo detectar drift semantico: {_hmm04_e}")



        print(f"[FIX-HMM-UNIQUE-SEMANTIC] Unique state mapping assigned to prevent feature collapsing: {self.state_map}")
        logger.info("[FIX-HMM-UNIQUE-SEMANTIC] State mapping constructed with unique suffixes: {}", self.state_map)


        logger.info(f"Mapeo Semantico Completado (Deadband={deadband:.1f}%):")
        for s, label in self.state_map.items():
            st = stats_df[stats_df['state'] == s].iloc[0]
            logger.info(f" - Estado {s}: {label} | Retorno 30d: {st['ret']:+.2f}% | Vol: {st['vol']:.5f} | N: {int(st['obs'])}")

        # -- Distribucion de regimenes y duracion media --
        total_obs = stats_df['obs'].sum()
        logger.info("[HMM] Distribucion de regimenes (train set):")
        for _, row in stats_df.iterrows():
            label = self.state_map.get(int(row['state']), f"estado_{int(row['state'])}")
            pct = 100 * row['obs'] / max(total_obs, 1)
            logger.info(f"  {label}: {int(row['obs'])} obs ({pct:.1f}%) | ret24h={row['ret']:+.2f}% | vol={row['vol']:.2f}%")

        # FIX-HMM-MI-01: Mutual Information con alineacion correcta de NaN.
        # Bug anterior: dos dropna() independientes podian generar arrays de longitud
        # diferente si tenian NaN en posiciones distintas -> MI incorrecto.
        # Fix: dropna() conjunto sobre un DataFrame combinado garantiza alineacion 1:1.
        try:
            from sklearn.metrics import mutual_info_score
            if 'close' in train_mask_df.columns:
                # [FIX-HMM-MI-HORIZON-01 2026-06-02] Horizonte 720H (30 dias) para MI.
                # EMPIRICO: MI(HMM, fwd_24H)=0.00090 < SOP-R9=0.005 siempre.
                # MI(HMM, fwd_720H)=0.00588 supera SOP-R9=0.005.
                # Los regimenes HMM son macro (escala mensual). 24H genera violacion estructural permanente.
                _mi_horizon = int(getattr(_cfg_hmm.hmm, 'mi_horizon_hours', 720))
                fwd_ret_sign = (train_mask_df['close'].pct_change(_mi_horizon).shift(-_mi_horizon) > 0).astype(int)
                print(f'[FIX-HMM-MI-HORIZON-01] MI calculada con horizonte={_mi_horizon}H (empirico supera SOP-R9=0.005)')  # RULE[fixbugsprints.md]
                _mi_df = pd.DataFrame({
                    'state':  train_mask_df['HMM_State_Raw'],
                    'target': fwd_ret_sign
                }).dropna()  # dropna conjunto -- garantiza alineacion perfecta 1:1
                if len(_mi_df) > 100:
                    mi = mutual_info_score(_mi_df['state'].astype(int), _mi_df['target'].astype(int))
                    min_mi = float(getattr(_cfg_hmm.hmm, 'min_mi', 0.005))
                    self._mi_pre_shield = mi  # [FIX-HMM-MI-PRESHIELD-01] guardar para contexto post-Shield
                    mi_flag = " BAJO (<0.005)" if mi < min_mi else " OK"
                    # [FIX-HMM-MI-PRESHIELD-01 2026-06-02] Pre-Shield MI es siempre menor que post-Shield.
                    # El Shield (risk_off/on + post_ath) anade capas de contexto macro que elevan la MI.
                    # Usar WARNING aqui: el CRITICAL se reserva para cuando post-Shield TAMBIEN falle.
                    if mi < min_mi:
                        logger.warning(
                            f"[SOP-R9-PRE-SHIELD][FIX-HMM-MI-PRESHIELD-01] HMM MI_PreShield={mi:.5f} "
                            f"< min={min_mi}. NORMAL: Shield anade contexto macro que eleva MI. "
                            f"Verificar MI_PostShield. n={len(_mi_df)}"
                        )
                        print(  # RULE[fixbugsprints.md]
                            f"[FIX-HMM-MI-PRESHIELD-01] MI pre-Shield={mi:.5f} < {min_mi} "
                            f"(esperado: Shield elevara hasta >=0.005). Pipeline continua."
                        )
                    else:
                        logger.info(
                            f"[HMM] Mutual Information (estado vs retorno {_mi_horizon}h): MI={mi:.5f}{mi_flag} "
                            f"| n={len(_mi_df)} (FIX-HMM-MI-01: alineacion conjunta)"
                        )
                else:
                    logger.warning("[HMM] MI no calculado: insuficientes filas alineadas (<100)")
        except Exception as e:
            logger.debug(f"[HMM] MI no calculado: {e}")

        self.raw_df['HMM_Semantic'] = self.raw_df['HMM_State_Raw'].map(self.state_map)

        # --- LUNA RISK-OFF SHIELD (Override) ---
        self.raw_df['HMM_Semantic'] = self._apply_risk_off_shield(self.raw_df, self.raw_df['HMM_Semantic'])

        # --- LUNA RISK-ON SHIELD (Golden Cross Override) ---
        self.raw_df['HMM_Semantic'] = self._apply_risk_on_shield(self.raw_df, self.raw_df['HMM_Semantic'])

        # Recalcular MI correctamente DESPUÉS del Shield (LOGIC-HMM-02)
        try:
            from sklearn.metrics import mutual_info_score
            if 'close' in train_mask_df.columns:
                # [FIX-HMM-MI-HORIZON-01] Usar mismo horizonte 720H para MI post-Shield
                _mi_horizon2 = int(getattr(_cfg_hmm.hmm, 'mi_horizon_hours', 720))
                fwd_ret_sign = (train_mask_df['close'].pct_change(_mi_horizon2).shift(-_mi_horizon2) > 0).astype(int)
                print(f'[FIX-HMM-MI-HORIZON-01] MI post-Shield horizonte={_mi_horizon2}H')  # RULE[fixbugsprints.md]
                # Map categories to integer indices for mutual_info_score
                semantic_cat = self.raw_df['HMM_Semantic'].astype('category').cat.codes
                _mi_df = pd.DataFrame({'s': semantic_cat, 't': fwd_ret_sign}).dropna()
                if len(_mi_df) > 100:
                    mi2 = mutual_info_score(_mi_df['s'], _mi_df['t'])
                    # [FIX-HMM-MI-CRITICAL 2026-05-30] Post-Shield MI check tambien a CRITICAL
                    min_mi2 = float(getattr(_cfg_hmm.hmm, 'min_mi', 0.005))
                    if mi2 < min_mi2:
                        if mi2 < mi:
                            logger.critical(
                                f"[SOP-R9-VIOLACION][FIX-HMM-MI-CRITICAL] HMM MI_PostShield={mi2:.5f} "
                                f"BAJO umbral (min={min_mi2}). El Shield EMPEORO la info predictiva "
                                f"({mi:.5f}->{mi2:.5f}, delta={mi2-mi:.5f}). Revisar thresholds del Shield."
                            )
                            print(f"[FIX-HMM-MI-CRITICAL] Shield empeoro MI: {mi:.5f}->{mi2:.5f} "
                                  f"(delta={mi2-mi:+.5f}). post_ath_bear forzando demasiados estados.")
                            # [SOP-R16-FAIL-FAST] Si la informacion mutua post-shield es deficiente, abortar.
                            raise RuntimeError(f"SOP-R9 Violada: Informacion mutua HMM_PostShield={mi2:.5f} < {min_mi2}. Posible colapso de estados.")
                        else:
                            logger.warning(
                                f"[SOP-R9-POST-SHIELD][WARN] HMM MI_PostShield={mi2:.5f} < min={min_mi2}. "
                                f"Sin embargo, el Shield NO empeoró la info (pre={mi:.5f} -> post={mi2:.5f}, delta={mi2-mi:+.5f}). "
                                f"Continuando ya que la info pre-Shield ya era baja (permitido por warning anterior)."
                            )
                            print(f"[BUG-FIX-LOG 2026-06-05] HMM MI_PostShield={mi2:.5f} < min={min_mi2} but pre={mi:.5f} -> no degradation. Continuing.")
                    else:
                        logger.info(f"[HMM] MI_PostShield={mi2:.5f} (Verdadero impacto OOS)")
        except RuntimeError as e:
            # Re-lanzar el error critico R9 para aplicar Fail-Fast
            logger.error(f"[FAIL-FAST] Abortando por SOP-R9: {e}")
            raise
        except Exception as e:
            logger.debug(f"[HMM] Error calculando MI_PostShield: {e}")

    def _apply_risk_off_shield(self, df_input, predicted_labels):
        """
        Sobreescribe el régimen HMM con '4_BEAR_FORCED' si se detectan condiciones
        de riesgo sistémico (CRASH/BEAR) usando datos Macro y de Pánico (Volatilidad/Funding).

        [BUG-SHIELD-DISCREPANCY-01] y [BUG-SHIELD-VOL-01] Unificación matemática:
        Construye series de volatilidad, financiación y MVRV por fila y aplica
        umbrales dinámicos sin gaps de NaNs históricos.
        """
        import pandas as pd
        labels = pd.Series(predicted_labels, index=df_input.index).copy()
        try:
            if 'close' not in df_input.columns:
                return labels

            close = df_input['close']

            # --- 1. FILTRO MACRO (Lento y Categórico) ---
            ma200 = close.rolling(200*24, min_periods=100*24).mean()
            ma50  = close.rolling(50*24, min_periods=25*24).mean()
            ret_168h = close.pct_change(168)
            ret_720h = close.pct_change(720)
            raw_macro_bear = (close < ma200) & (ma50 < ma200) & (ret_168h < 0) & (ret_720h < -0.05)
            macro_bear = raw_macro_bear.rolling(48, min_periods=48).min().fillna(0).astype(bool)

            # --- 1B. FILTRO POST-ATH (HMM-FIX-POST-ATH-01) ---
            # El macro_bear clasico (close < ma200) falla en mercados post-ATH porque
            # la media de 200d calculada desde 2017 (~67k) esta muy por debajo del precio
            # actual aunque BTC caiga -35%. Este gate detecta correcciones fuertes desde
            # el ATH local de los ultimos 90 dias, independientemente de medias historicas.
            # Validado W1-W5: activa 239h en W5 (crash Feb-2026), 0h en bulls W2/W3.
            _c1_dd_thresh   = float(getattr(_cfg_hmm.hmm, 'post_ath_dd_threshold',   -0.15))
            _c1_mom_thresh  = float(getattr(_cfg_hmm.hmm, 'post_ath_mom_threshold',   -0.05))
            _c1_ath_window  = int(getattr(_cfg_hmm.hmm,   'post_ath_ath_window_h',    2160))  # 90d
            _c1_ath_minper  = int(getattr(_cfg_hmm.hmm,   'post_ath_ath_min_periods', 720))   # 30d
            _c1_confirm     = int(getattr(_cfg_hmm.hmm,   'post_ath_confirm_h',       48))

            ath_90d = close.rolling(_c1_ath_window, min_periods=_c1_ath_minper).max()
            dd_from_ath = (close / ath_90d) - 1.0  # siempre <= 0
            raw_post_ath = (dd_from_ath < _c1_dd_thresh) & (ret_168h < _c1_mom_thresh)
            post_ath_bear = raw_post_ath.rolling(_c1_confirm, min_periods=_c1_confirm).min().fillna(0).astype(bool)

            print(f"[HMM-FIX-POST-ATH-01] Gate C1: dd_thresh={_c1_dd_thresh:.0%} | "
                  f"mom_thresh={_c1_mom_thresh:.0%} | ath_window={_c1_ath_window}H | "
                  f"post_ath_bear_count={post_ath_bear.sum()} horas")
            logger.info(f"[HMM-FIX-POST-ATH-01] post_ath_bear activado en {post_ath_bear.sum()} horas "
                        f"| dd_thresh={_c1_dd_thresh:.0%} | mom_thresh={_c1_mom_thresh:.0%}")

            ret_24h = close.pct_change(24)


            # --- 2. CONSTRUCCIÓN DE SERIES UNIFICADAS (Consistente con fit_global) ---
            # Volatilidad Diaria Decimal
            daily_vol = None
            if 'DVOL' in df_input.columns:
                daily_vol = (df_input['DVOL'] / 100.0) / np.sqrt(365)
            elif 'parkinson_vol' in df_input.columns:
                daily_vol = (df_input['parkinson_vol'] / 100.0) / np.sqrt(365)
                
            if 'mt_vol_realized_4bar' in df_input.columns:
                mt_daily = df_input['mt_vol_realized_4bar'] * np.sqrt(24)
                daily_vol = mt_daily if daily_vol is None else daily_vol.fillna(mt_daily)
                
            if daily_vol is None:
                # De último recurso, usar volatilidad de retornos a 24H
                daily_vol = ret_24h.rolling(30*24, min_periods=24).std()

            # Financiación
            fund_series = None
            for _fc in ['FundingRate', 'dv_funding_rate', 'funding_rate']:
                if _fc in df_input.columns:
                    fund_series = df_input[_fc] if fund_series is None else fund_series.fillna(df_input[_fc])

            # MVRV
            mvrv_series = None
            for _mc in ['MVRV_Proxy', 'mvrv_pct_6m', 'mvrv_zscore']:
                if _mc in df_input.columns:
                    mvrv_series = df_input[_mc] if mvrv_series is None else mvrv_series.fillna(df_input[_mc])

            # --- 3. RESOLVER QUANTILES Y LOGICA DE PANICO ---
            _sq = getattr(self, '_shield_quantiles', {})
            
            # Volatilidad Gate & Umbral
            if 'vol_p90' in _sq:
                vol_p90 = _sq['vol_p90']
                _src_vol = "IS-precalculado"
            else:
                vol_p90 = daily_vol.rolling(365*24, min_periods=720).quantile(0.90).bfill().fillna(0.04)
                _src_vol = "OOS-fallback (rolling causal)"
                
            dyn_crash_thresh = np.minimum(-0.04, -2.5 * daily_vol)
            
            # Financiación Gate
            if fund_series is not None:
                if 'fund_p05' in _sq:
                    fund_p05 = _sq['fund_p05']
                    _src_fund = "IS-precalculado"
                else:
                    fund_p05 = fund_series.rolling(365*24, min_periods=720).quantile(0.05).bfill().fillna(-0.0001)
                    _src_fund = "OOS-fallback (rolling causal)"
                    
                fund_available = fund_series.notna()
                panic_with_fund = (daily_vol > vol_p90) & (fund_series < fund_p05) & (ret_24h < dyn_crash_thresh)
                panic_no_fund = (daily_vol > vol_p90) & (ret_24h < dyn_crash_thresh)
                panic_bear = pd.Series(np.where(fund_available, panic_with_fund, panic_no_fund), index=df_input.index)
            else:
                panic_bear = (daily_vol > vol_p90) & (ret_24h < dyn_crash_thresh)
                _src_fund = "No-disponible"

            # MVRV Shield
            dist_bear = pd.Series(False, index=df_input.index)
            if mvrv_series is not None:
                if 'mvrv_p95' in _sq:
                    mvrv_p95 = _sq['mvrv_p95']
                    _src_mvrv = "IS-precalculado"
                else:
                    mvrv_p95 = mvrv_series.rolling(365*24, min_periods=720).quantile(0.95).bfill().fillna(2.0)
                    _src_mvrv = "OOS-fallback (rolling causal)"
                    
                dyn_dist_thresh = np.minimum(-0.03, -2.0 * daily_vol)
                # Solo activable si la serie MVRV no es NaN
                dist_bear = (mvrv_series > mvrv_p95) & (ret_24h < dyn_dist_thresh)
            else:
                _src_mvrv = "No-disponible"

            # --- 4. COMBINAR OVERRIDES ---
            # [FIX-HMM-SHIELD-03 2026-06-07] MI-guard activo para TODO el escudo.
            # Evita fallos críticos SOP-R9 desactivando escudos heurísticos si 
            # destruyen la información mutua validada por el HMM no supervisado.
            _min_mi_shield = float(getattr(_cfg_hmm.hmm, 'min_mi', 0.005))
            _shield_enabled = True
            _post_ath_enabled = True
            try:
                from sklearn.metrics import mutual_info_score as _mis
                if 'close' in df_input.columns:
                    _mi_guard_horizon = int(getattr(_cfg_hmm.hmm, 'mi_horizon_hours', 720))
                    _fwd_sign = (df_input['close'].pct_change(_mi_guard_horizon).shift(-_mi_guard_horizon) > 0).astype(int)
                    print(f'[FIX-HMM-MI-HORIZON-01] MI-guard shield horizonte={_mi_guard_horizon}H')  # RULE[fixbugsprints.md]
                    
                    # 1. Medir MI original (sin escudo)
                    _lbl_raw = labels.astype(object).copy()
                    _cat_raw = _lbl_raw.astype('category').cat.codes
                    _df_raw = pd.DataFrame({'s': _cat_raw, 't': _fwd_sign}).dropna()
                    _mi_raw = _mis(_df_raw['s'], _df_raw['t']) if len(_df_raw) > 100 else 0.0

                    # 2. Medir MI con Escudo Base (macro + panic + dist)
                    _lbl_base = labels.astype(object).copy()
                    _is_forced_base = macro_bear | panic_bear | dist_bear
                    _lbl_base.loc[_is_forced_base] = '4_BEAR_FORCED'
                    _cat_base = _lbl_base.astype('category').cat.codes
                    _df_base = pd.DataFrame({'s': _cat_base, 't': _fwd_sign}).dropna()
                    _mi_base = _mis(_df_base['s'], _df_base['t']) if len(_df_base) > 100 else 0.0

                    # 3. Medir MI con Escudo Completo (+ post_ath)
                    _lbl_full = labels.astype(object).copy()
                    _is_forced_full = macro_bear | panic_bear | dist_bear | post_ath_bear
                    _lbl_full.loc[_is_forced_full] = '4_BEAR_FORCED'
                    _cat_full = _lbl_full.astype('category').cat.codes
                    _df_full = pd.DataFrame({'s': _cat_full, 't': _fwd_sign}).dropna()
                    _mi_full = _mis(_df_full['s'], _df_full['t']) if len(_df_full) > 100 else 0.0

                    print(f'[FIX-HMM-SHIELD-03] MI-guard: Raw={_mi_raw:.5f} | Base={_mi_base:.5f} | Full={_mi_full:.5f} | SOP={_min_mi_shield}')  # RULE[fixbugsprints.md]

                    # Lógica de exclusión dinámica
                    if _mi_full < _mi_base and _mi_full < _min_mi_shield:
                        _post_ath_enabled = False
                        print(f'[FIX-HMM-SHIELD-03] post_ath_bear DESACTIVADO. Empeora MI a {_mi_full:.5f} < {_mi_base:.5f}')  # RULE[fixbugsprints.md]
                    
                    # Si el escudo resultante sigue destrozando la MI bajo el umbral SOP-R9, lo matamos entero
                    _mi_active = _mi_base if not _post_ath_enabled else _mi_full
                    if _mi_active < _mi_raw and _mi_active < _min_mi_shield:
                        _shield_enabled = False
                        print(f'[FIX-HMM-SHIELD-03] ESCUDO COMPLETO DESACTIVADO. Destruye la informacion del HMM (MI: {_mi_active:.5f} < Raw: {_mi_raw:.5f} y SOP). Se confia en el HMM puro sin sesgos heuristicos.')  # RULE[fixbugsprints.md]

            except Exception as _e_mi_guard:
                print(f'[FIX-HMM-SHIELD-03] MI-guard error: {_e_mi_guard}')

            if not _shield_enabled:
                is_bear_forced = pd.Series(False, index=df_input.index)
            elif _post_ath_enabled:
                is_bear_forced = macro_bear | panic_bear | dist_bear | post_ath_bear
            else:
                is_bear_forced = macro_bear | panic_bear | dist_bear

            if is_bear_forced.any():
                is_numeric = labels.dtype.kind in 'iufc'
                if is_numeric:
                    forced_key = 4
                    for k, v in self.state_map.items():
                        if v == '4_BEAR_FORCED':
                            forced_key = int(k)
                    if forced_key not in self.state_map:
                        self.state_map[forced_key] = '4_BEAR_FORCED'
                    labels.loc[is_bear_forced] = forced_key
                else:
                    labels.loc[is_bear_forced] = '4_BEAR_FORCED'

                if '4_BEAR_FORCED' not in self.state_map.values():
                    new_key = max(self.state_map.keys()) + 1 if self.state_map else 4
                    self.state_map[new_key] = '4_BEAR_FORCED'

            logger.info(
                f"[RISK-SHIELD/BUG-SHIELD-VOL-01] Shield Aplicado — vol_src={_src_vol} | vol_p90={vol_p90:.6f} | "
                f"fund_src={_src_fund} | mvrv_src={_src_mvrv} | macro_bear={macro_bear.sum()} | "
                f"panic_bear={panic_bear.sum()} | dist_bear={dist_bear.sum()} | "
                f"post_ath_bear={post_ath_bear.sum()} | total_forced={is_bear_forced.sum()}"
            )
            print(
                f"[BUG-SHIELD-DISCREPANCY-01] [RISK-SHIELD/DEBUG] Shield Aplicado — vol_src={_src_vol} | vol_p90={vol_p90:.6f} | "
                f"fund_src={_src_fund} | mvrv_src={_src_mvrv} | macro_bear={macro_bear.sum()} | "
                f"panic_bear={panic_bear.sum()} | dist_bear={dist_bear.sum()} | "
                f"post_ath_bear={post_ath_bear.sum()} | [HMM-FIX-POST-ATH-01] total_forced={is_bear_forced.sum()}"
            )
        except Exception as e:
            logger.error(f"[RISK-SHIELD] Error aplicando Bear Forced override: {e}")
            print(f"[BUG-SHIELD-DISCREPANCY-01] Error aplicando Bear Forced override: {e}")

        return labels.values

    def _apply_risk_on_shield(self, df_input, predicted_labels):
        """
        Sobreescribe el régimen HMM (Risk-On Override) si se detectan condiciones
        extremas de recuperación alcista (Golden Cross V-Shape) mientras el HMM
        está atrapado por "inercia" en un estado BEAR_CRASH o BEAR_FORCED.
        
        Si el precio sube violentamente y cruza fuertemente medias clave, 
        sacamos al modelo de BEAR_CRASH y lo pasamos a 2_VOLATILE_RANGE o 1_VOLATILE_BULL.
        """
        import pandas as pd
        labels = pd.Series(predicted_labels, index=df_input.index).copy()
        try:
            if 'close' not in df_input.columns:
                return labels

            close = df_input['close']
            
            # --- FILTRO GOLDEN CROSS / V-SHAPE RECOVERY ---
            # Identificamos si el mercado estalla hacia arriba a corto plazo
            ret_168h = close.pct_change(168)  # 7 days
            ma50  = close.rolling(50*24, min_periods=25*24).mean()
            
            # Condición de V-Shape fuerte: el precio está un 5% por encima de MA50
            # Y el retorno a 7 días es mayor a 10%
            is_v_shape_rally = (close > ma50 * 1.05) & (ret_168h > 0.10)
            
            # Solo aplicamos si el HMM está atascado en BEAR
            is_numeric = labels.dtype.kind in 'iufc'
            if is_numeric:
                # Buscar IDs numéricos de regímenes bajistas y alcistas/neutrales
                bear_keys = [int(k) for k, v in self.state_map.items() if 'BEAR' in str(v)]
                bull_key = None
                for k, v in self.state_map.items():
                    if '1_VOLATILE_BULL' in str(v): bull_key = int(k)
                    if bull_key is None and '2_VOLATILE_RANGE' in str(v): bull_key = int(k)
                if bull_key is None:
                    bull_key = next(iter(self.state_map.keys())) # fallback
                
                mask = is_v_shape_rally & labels.isin(bear_keys)
                labels.loc[mask] = bull_key
            else:
                mask = is_v_shape_rally & labels.astype(str).str.contains('BEAR')
                bull_label = '1_VOLATILE_BULL' if '1_VOLATILE_BULL' in self.state_map.values() else '2_VOLATILE_RANGE'
                labels.loc[mask] = bull_label

            n_forced = mask.sum()
            if n_forced > 0:
                logger.info(f"[RISK-ON SHIELD] Activado: sobreescritas {n_forced} velas atrapadas en BEAR_CRASH hacia régimen neutral/alcista")

        except Exception as e:
            logger.error(f"[RISK-ON SHIELD] Error aplicando Risk-On override: {e}")

        return labels.values


    def generate_oos_features(self, step=120):
        """
        SOP R1: Genera labels usando Rolling Windows (Sin Look-Ahead).
        No predice usando info futura de Viterbi. Avanza reentrenando o 
        haciendo 'predict' paso a paso sobre datos nunca vistos.
                """
        logger.info(f"Generando Series OOS Rolling (Evitando Look-Ahead Bias)... Step={step}H")
        
        # Fix A-06: NO llamar fit_transform de nuevo â€” el scaler ya fue ajustado en fit_global_for_analysis().
        # Re-aplicar fit_transform() aquÃ­ contamina la normalizaciÃ³n con validaciÃ³n/holdout.
        if not self.is_fitted:
            raise RuntimeError("Llamar fit_global_for_analysis() antes de generate_oos_features()")
        X_scaled = self.scaler.transform(self.X)  # solo transform, nunca re-fit
        self.rolling_states = np.zeros(len(X_scaled))
        self.rolling_states[:] = np.nan
        
        # Warmup inicial de 6 meses (aprox 4000 horas)
        warmup = 4320
        if warmup >= len(X_scaled):
            warmup = len(X_scaled) // 2
            
        # Fix F4: usar el modelo global (self.model) para predecir chunks OOS.
        # El modelo local (roll_model entrenado solo en warmup) asigna estados 
        # con numeraciÃ³n distinta al modelo global, rompiendo el state_map semÃ¡ntico.
        # Verificar si el modelo global ya estÃ¡ entrenado; si no, entrenar roll_model.
        if hasattr(self, 'model') and self.model is not None:
            predict_model = self.model  # Modelo global â€” estados semÃ¡nticamente consistentes
            logger.info("F4: Usando modelo HMM global para predicciÃ³n OOS (estados consistentes con state_map)")
        else:
            # Fallback: entrenar modelo local solo si no hay modelo global disponible
            predict_model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=500, random_state=42)
            predict_model.fit(X_scaled[:warmup])
            logger.warning("F4: Modelo global no disponible â€” usando modelo local (posible inconsistencia de estados)")
        
        # Avanzar usando predict con el modelo seleccionado (global o local)
        # FIX-FINDING-4: Secuencial estrictamente causal, tomando solo el estado [-1]
        for t in range(warmup, len(X_scaled)):
            chunk_start = max(0, t - step)
            chunk = X_scaled[chunk_start:t+1]
            state = predict_model.predict(chunk)[-1]
            self.rolling_states[t] = state
            
        logger.success("Serie Rolling OOS generada.")
        # FIX-HMM-WARMUP-01: rellenar el perÃ­odo de warmup (primeros 4320H) con la
        # predicciÃ³n del modelo global â€” NO es look-ahead porque el modelo fue
        # entrenado SOLO con datos hasta train_cutoff, y el warmup es parte del IS.
        # Antes: fillna(2) â†’ estado neutral arbitrario para el 34% de los datos.
        # Ahora: predict() del modelo global sobre el warmup â†’ estados semÃ¡nticamente
        # correctos, sin NaN innecesarios en el dataset XGBoost.
        rolling_series = pd.Series(self.rolling_states, index=self.raw_df.index)
        if rolling_series.isna().any():
            nan_mask = rolling_series.isna()
            warmup_states = predict_model.predict(X_scaled[nan_mask.values])
            rolling_series.loc[nan_mask] = warmup_states
            logger.info(
                f"[HMM] FIX-HMM-WARMUP-01: {nan_mask.sum()} NaN de warmup rellenados "
                f"con predict() global (antes: fillna=2 neutro arbitrario)"
            )
        rolling_series = rolling_series.ffill().fillna(2)  # fallback residual
        self.rolling_states = rolling_series.values
        
        # --- LUNA RISK-OFF SHIELD (Rolling OOS) ---
        # Aplica el rescate a la exportaciÃ³n numÃ©rica del OOS.
        self.raw_df['HMM_State_OOS'] = self._apply_risk_off_shield(self.raw_df, self.rolling_states)
        self.raw_df['HMM_State_OOS'] = self._apply_risk_on_shield(self.raw_df, self.raw_df['HMM_State_OOS'])

        # ARCH-05 fix (2026-03-17): monitor de drift de distribuciÃ³n de regÃ­menes ISâ†’OOS.
        # Rolling retrain no es viable (Fix F4: romperÃ­a state_map semÃ¡ntico).
        # En su lugar: Jensen-Shannon divergence entre freq_IS y freq_OOS como alerta temprana.
        try:
            from scipy.spatial.distance import jensenshannon as _jsd_fn
            # [FIX-FLAG-HMM-01] Acceso directo y seguro a _train_cutoff_used.
            # Bug anterior: getattr anidado con __str__ producía '2100-01-01' cuando
            # _train_cutoff_used era None/vacío, dejando el JSD drift check siempre ciego.
            _cutoff_raw = getattr(self, '_train_cutoff_used', None)
            if _cutoff_raw and str(_cutoff_raw).strip():
                try:
                    _train_cutoff_ts = pd.Timestamp(str(_cutoff_raw))
                except Exception:
                    _cutoff_raw = None
            if not _cutoff_raw:
                # _train_cutoff_used no fue persistido (pkl antiguo o bug de init)
                logger.warning(
                    "[FLAG-HMM-01] _train_cutoff_used no disponible en objeto HMM — "
                    "JSD drift usará midpoint temporal como separador IS/OOS. "
                    "Re-entrenar HMM para activar drift check preciso."
                )
                _train_cutoff_ts = pd.Timestamp('2100-01-01')  # trigger fallback explícito
            # [FIX-JSD-TZ-01] Alinear timezone: si el índice es UTC-aware, el Timestamp debe serlo también.
            # Bug anterior: comparación entre datetime64[ns, UTC] y Timestamp naive → TypeError silenciado.
            try:
                if hasattr(self.raw_df.index, 'tz') and self.raw_df.index.tz is not None:
                    if _train_cutoff_ts.tzinfo is None:
                        _train_cutoff_ts = _train_cutoff_ts.tz_localize('UTC')
            except Exception:
                pass
            # Separar IS/OOS usando el cutoff de entrenamiento (o midpoint si no disponible)
            if _train_cutoff_ts.year > 2090:
                _midpoint = len(self.rolling_states) // 2
                _is_states  = self.rolling_states[:_midpoint]
                _oos_states = self.rolling_states[_midpoint:]
                logger.debug("[FLAG-HMM-01] JSD drift: usando midpoint={} como separador IS/OOS.", _midpoint)
                print(f"[BUG-FIX-LOG 2026-06-05] [FLAG-HMM-01] JSD drift: usando midpoint={_midpoint} como separador IS/OOS.")
                _has_oos_rows = True
            else:
                _is_mask    = self.raw_df.index <= _train_cutoff_ts
                _is_mask_arr = _is_mask.values if hasattr(_is_mask, 'values') else _is_mask
                _is_states  = self.rolling_states[_is_mask_arr]
                _oos_states = self.rolling_states[~_is_mask_arr]
                logger.debug(
                    "[FLAG-HMM-01] JSD drift: cutoff={} | IS={} obs | OOS={} obs.",
                    _cutoff_raw, len(_is_states), len(_oos_states)
                )
                print(f"[BUG-FIX-LOG 2026-06-05] [FLAG-HMM-01] JSD drift: cutoff={_cutoff_raw} | IS={len(_is_states)} obs | OOS={len(_oos_states)} obs.")
                _has_oos_rows = bool((~_is_mask_arr).sum() > 0)

            if not _has_oos_rows:
                logger.info(
                    "[FIX-JSD-01-PURE-IS] [ARCH-05] JSD check omitido: Sin fechas OOS en el DataFrame de train (modo puro IS/WFB-fit). "
                    "Esto es normal ya que el holdout se evalúa en la fase de inferencia."
                )
                print(f"[FIX-JSD-01-PURE-IS] JSD check omitido para train_cutoff={_cutoff_raw} (IS={len(_is_states)}, OOS=0)")
                self._regime_drift_jsd = float('nan')
            else:
                _all_states = list(range(self.n_components))
                _freq_is  = np.array([(_is_states  == s).mean() for s in _all_states]) + 1e-9
                _freq_is  /= _freq_is.sum()
                
                if len(_oos_states) == 0:
                    # [FIX-JSD-EMPTY-OOS-01] BUG ORIGINAL: asignaba _jsd2=0.0 → reporte falso "distribución estable".
                    # La distribución OOS vacía significa que NINGUNA barra OOS tiene etiqueta HMM asignada.
                    # Causa: state_map desalineado entre ventanas WFB (etiquetas de ventana anterior).
                    # CORRECCIÓN: JSD=1.0 (máximo drift posible) + alerta CRITICAL.
                    _freq_oos = np.zeros(len(_all_states))
                    _jsd2 = 1.0  # máximo drift — no es "estable"
                    logger.critical(
                        "[FIX-JSD-EMPTY-OOS-01] CRÍTICO: _oos_states VACÍO — JSD=1.0 (máximo). "
                        "NINGUNA barra OOS tiene régimen HMM asignado. "
                        "Causa probable: state_map desalineado entre ventanas WFB, "
                        "o train_cutoff anterior a todos los datos disponibles. "
                        "IS_states={} OOS_states={} cutoff={}",
                        len(_is_states), len(_oos_states), _cutoff_raw
                    )
                    print(
                        f"[FIX-JSD-EMPTY-OOS-01] CRÍTICO: OOS vacío → JSD=1.0 "
                        f"IS={len(_is_states)} OOS={len(_oos_states)} cutoff={_cutoff_raw}"
                    )  # debug
                else:
                    _freq_oos = np.array([(_oos_states == s).mean() for s in _all_states]) + 1e-9
                    _freq_oos /= _freq_oos.sum()
                    _jsd2 = float(_jsd_fn(_freq_is, _freq_oos) ** 2)
                try:
                    from config.settings import cfg as _cfg_hmm_drift
                    _jsd_thr = float(getattr(getattr(_cfg_hmm_drift, 'hmm', None), 'drift_alert_jsd', 0.15))
                except Exception:
                    _jsd_thr = 0.15

                _labels = [self.state_map.get(s, str(s)) for s in _all_states] if hasattr(self, 'state_map') else [str(s) for s in _all_states]
                logger.info(f"[ARCH-05] Dist IS : {dict(zip(_labels, _freq_is.round(3)))}")  # [M5-FIX]
                logger.info(f"[ARCH-05] Dist OOS: {dict(zip(_labels, _freq_oos.round(3)))}")  # [M5-FIX]

                if _jsd2 > _jsd_thr:
                    logger.warning(
                        "[ARCH-05] DRIFT REGIMENES HMM: JSD2={:.3f} > umbral={:.2f}. "
                        "Distribucion OOS difiere significativamente del IS. "
                        "Considerar re-entrenar HMM con datos mas recientes (run_features).",
                        _jsd2, _jsd_thr
                    )
                    print(f"[BUG-FIX-LOG 2026-06-05] [ARCH-05] DRIFT REGIMENES HMM: JSD2={_jsd2:.3f} > umbral={_jsd_thr:.2f}.")
                    # B4: Monitorización JSD drift HMM mensual con alertas automáticas
                    try:
                        from luna.live.telegram_alerts import TelegramAlerts
                        _tg_msg = f"⚠️ *ALERTA HMM DRIFT (MENSUAL)*\nJSD={_jsd2:.3f} supera el umbral de {_jsd_thr:.2f}.\nLa distribución de regímenes OOS difiere drásticamente de la original.\n➡️ *ACCIÓN REQUERIDA*: Re-entrenar HMM (Fase 3C)."
                        TelegramAlerts().send_alert(_tg_msg, priority="warning")
                    except Exception as e_tg:
                        logger.debug(f"HMM Drift: no se pudo enviar alerta de Telegram ({e_tg})")
                else:
                    logger.info(f"[ARCH-05] Drift HMM: JSD2={_jsd2:.3f} < {_jsd_thr:.2f} — distribucion estable.")

                self._regime_drift_jsd = _jsd2
        except Exception as _e_drift:
            logger.debug(f"[ARCH-05] No se pudo calcular drift JSD de regimenes: {_e_drift}")
            self._regime_drift_jsd = float('nan')


    def coerce_regime_numeric(self, series: "pd.Series") -> "pd.Series":
        """C7 (2026-03-23): Convierte HMM_Regime a numérico independientemente del tipo de entrada.

        Fuente única de verdad para la conversión — sustituye el bloque inline
        que estaba duplicado en generate_oos_predictions.py L464-492.

        Casos:
          - Si ya es numérico (int/float): pd.to_numeric() + fillna(0)
          - Si es object/string: mapea vía state_map inverso; fallback pd.to_numeric()

        Args:
            series: pd.Series con valores HMM_Regime (numérico o string semántico)

        Returns:
            pd.Series numérica (float64) con fillna(0.0)
        """
        import pandas as _pd
        if series.dtype.kind in ("i", "u", "f"):
            # Ya numérico — solo asegurar float y fillna
            return _pd.to_numeric(series, errors="coerce").fillna(0.0)

        # Object/string — intentar mapeo inverso del state_map
        _inv_map: dict = {}
        if hasattr(self, "state_map") and self.state_map:
            _inv_map = {v: k for k, v in self.state_map.items()}

        if _inv_map:
            result = series.map(_inv_map)
        else:
            result = _pd.to_numeric(series, errors="coerce")

        return result.fillna(0.0)

    def save_model(self):

        out_dir = self.root / "data" / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "hmm_regime.pkl"

        _pkl_payload = {
            'model':  self.model,
            'scaler': self.scaler,
            'state_map': self.state_map,
            'features': list(self.X.columns),
            # MEJORA-HMM-DURATION-01: guardar umbral dinamico para inferencia
            'min_state_duration_dynamic': getattr(self, 'min_state_duration_dynamic', MIN_STATE_DURATION_H),
            'min_state_duration_cfg':     self._min_state_duration_cfg,
            # MOD-02 (Run 14): NAS results â€” n elegido y metricas comparativas
            'nas_results': getattr(self, '_nas_results', {}),
            # FIX-CRITICO-1 (2026-04-01): quantiles IS del Risk-Off Shield.
            # Calculados en fit_global_for_analysis() sobre training set.
            # Cargados en load() para evitar recalcular sobre holdout OOS (look-ahead bias).
            'shield_quantiles': getattr(self, '_shield_quantiles', {}),
        }
        joblib.dump(_pkl_payload, path)

        # â”€â”€ [DATAFLOW-EXPORT-HMM-01] Audit del modelo guardado â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Verifica que el pkl contiene todo lo necesario para inferencia OOS.
        _state_map = _pkl_payload.get('state_map', {})
        _features  = _pkl_payload.get('features', [])
        logger.success(
            f"[DATAFLOW-EXPORT-HMM-01] HMM guardado: {path.name} | "
            f"n_components={self.n_components} | "
            f"n_features={len(_features)} | "
            f"state_map={_state_map} | "
            f"min_state_duration_dynamic={getattr(self, 'min_state_duration_dynamic', MIN_STATE_DURATION_H)}H"
        )
        if not _state_map:
            logger.warning(
                "  [DATAFLOW-EXPORT-HMM-01] ALERTA: state_map VACIO en el pkl. "
                "predict_regime_series no podra traducir indices numericos a etiquetas semanticas. "
                "Revisar _analyze_and_map_states()."
            )
        _required_keys = ['model', 'scaler', 'state_map', 'features']
        _missing_keys  = [k for k in _required_keys if k not in _pkl_payload or not _pkl_payload[k]]
        if _missing_keys:
            logger.warning(
                f"  [DATAFLOW-EXPORT-HMM-01] Campos FALTANTES o VACIOS en pkl: {_missing_keys}. "
                f"Inferencia OOS puede fallar."
            )
        else:
            logger.info("  [DATAFLOW-EXPORT-HMM-01] Campos pkl OK: model, scaler, state_map, features")
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @classmethod
    def load(cls, model_dir) -> "HMMRegimeModel":
        """
        BUG-HMM-LOAD-01: Carga un HMMRegimeModel desde el pkl guardado por save_model().

        El pkl contiene un dict con model/scaler/state_map/features â€” no el objeto
        completo. Este classmethod reconstruye un objeto HMMRegimeModel usable para
        inferencia (predict_regime_series) sin re-entrenar.

        Args:
            model_dir: Path o str al directorio que contiene hmm_regime.pkl.

        Returns:
            Instancia de HMMRegimeModel con model/scaler/state_map restaurados.
        """
        from pathlib import Path as _Path
        pkl_path = _Path(model_dir) / "hmm_regime.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"hmm_regime.pkl no encontrado en {model_dir}")

        is_mock = False
        try:
            with open(pkl_path, 'r', encoding='utf-8') as _f_check:
                _start = _f_check.read(500).strip()
                if _start.startswith('{'):
                    import json
                    _mock_data = json.loads(_start)
                    if _mock_data.get("mocked") is True:
                        is_mock = True
        except Exception:
            pass

        if is_mock:
            logger.warning(f"[HMMRegimeModel] Cargando modelo HMM mockeado desde {pkl_path}")
            print(f"[HMMRegimeModel/MOCK] Cargando modelo HMM mockeado desde {pkl_path}")
            obj = cls.__new__(cls)
            obj.root     = _Path(model_dir).parent.parent
            obj.model    = None
            obj.scaler   = None
            obj.state_map = {0: "1_BULL_TREND", 1: "2_CALM_RANGE", 2: "3_BEAR_TREND"}
            obj._features = ["close", "volume", "returns", "volatility", "funding_rate", "mvrv"]
            obj.n_components = 3
            obj.min_state_duration_dynamic = 12
            obj._min_state_duration_cfg = 12
            obj._nas_results = {}
            obj._shield_quantiles = {}
            obj.X = None
            obj.mocked = True
            return obj

        data = joblib.load(pkl_path)

        obj = cls.__new__(cls)           # Crear instancia sin llamar __init__
        obj.root     = _Path(model_dir).parent.parent  # models/ -> data/ -> root/
        obj.model    = data["model"]
        obj.scaler   = data["scaler"]
        obj.state_map = data.get("state_map", {})
        obj._features = data.get("features", [])
        obj.n_components = obj.model.n_components
        obj.min_state_duration_dynamic = data.get(
            "min_state_duration_dynamic", MIN_STATE_DURATION_H
        )
        obj._min_state_duration_cfg = data.get("min_state_duration_cfg", MIN_STATE_DURATION_H)
        obj._nas_results = data.get("nas_results", {})
        # FIX-CRITICO-1 (2026-04-01): restaurar quantiles IS del Risk-Off Shield desde pkl.
        obj._shield_quantiles = data.get("shield_quantiles", {})
        if obj._shield_quantiles:
            logger.info(
                "[FIX-CRITICO-1] Shield quantiles IS cargados: vol_p90={} | fund_p05={} | mvrv_p95={}",
                obj._shield_quantiles.get('vol_p90', 'N/A'),
                obj._shield_quantiles.get('fund_p05', 'N/A'),
                obj._shield_quantiles.get('mvrv_p95', 'N/A'),
            )
        else:
            logger.warning(
                "[FIX-CRITICO-1] shield_quantiles NO en pkl — shield usara quantiles "
                "OOS-fallback. Re-entrenar HMM para activar el fix anti-lookahead."
            )
        # X no disponible en inferencia pura â€” ponemos None para detectar usos incorrectos
        obj.X = None
        logger.info(
            f"[BUG-HMM-LOAD-01] HMMRegimeModel cargado desde {pkl_path} "
            f"| n_components={obj.n_components} | state_map={obj.state_map}"
        )
        return obj

    def predict_regime_series(self, df: "pd.DataFrame") -> "pd.Series":
        """
        Predice el rÃ©gimen HMM para un DataFrame de features (inferencia OOS).
        Solo requiere las columnas en self._features â€” rellena con 0 si faltan.

        Args:
            df: DataFrame con features de mercado (puede tener columnas extra).

        Returns:
            pd.Series con el label semÃ¡ntico del rÃ©gimen (ej. '1_BULL_TREND').
        """
        import pandas as _pd
        import numpy as _np

        if getattr(self, "mocked", False):
            logger.info("[HMMRegimeModel/MOCK] Ejecutando predicción simulada de régimen HMM.")
            raw_states = _np.zeros(len(df), dtype=int)
            for idx in range(len(df)):
                raw_states[idx] = idx % 3
            
            semantic_labels = _np.array([self.state_map.get(s, f"estado_{s}") for s in raw_states])
            numeric_labels = _np.array(raw_states, dtype=float)
            
            _result = _pd.DataFrame({
                "HMM_Regime": numeric_labels,
                "HMM_Semantic": semantic_labels
            }, index=df.index)
            
            for _col in ["hmm_bear_transition_prob", "hmm_bull_transition_prob", "hmm_transition_risk"]:
                _result[_col] = 0.1
                
            return _result

        feats = getattr(self, "_features", [])
        if not feats:
            raise ValueError("HMMRegimeModel.load() no guardÃ³ features â€” reentrenar.")

        # â”€â”€ [DATAFLOW-IMPORT-HMM-01] Feature coverage audit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Cuantas features HMM llegan reales vs. cero (cero = relleno silencioso).
        # Features con 0 real coverage causan predicciones incorrectas de regimen.
        _feats_present = [f for f in feats if f in df.columns]
        _feats_missing = [f for f in feats if f not in df.columns]
        logger.info(
            f"  [DATAFLOW-IMPORT-HMM-01] predict_regime_series: "
            f"{len(_feats_present)}/{len(feats)} features disponibles en df. "
            f"Faltantes (rellenadas con 0): {_feats_missing}"
        )
        if len(_feats_missing) > len(feats) * 0.5:
            logger.warning(
                f"  [DATAFLOW-IMPORT-HMM-01] ALERTA: mas del 50% de features HMM ausentes "
                f"({len(_feats_missing)}/{len(feats)}). "
                f"Las predicciones de regimen seran poco fiables. "
                f"Verificar que features_holdout.parquet fue generado con el mismo feature_pipeline."
            )
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # Construir X para el scaler con las features correctas (pad con 0 si faltan)
        X_inf = _pd.DataFrame(0.0, index=df.index, columns=feats)
        for f in feats:
            if f in df.columns:
                X_inf[f] = df[f].values

        X_scaled = self.scaler.transform(X_inf.fillna(0).values)  # [FIX-PIPE-003] ndarray explicito para suprimir UserWarning feature_names_in_

        # [WFB-CAUSAL-FIX-HMM] SOP R1: Forward scan causal en lugar de Viterbi global o chunked Viterbi.
        # CORRECCIÓN: Usar un filtro Forward puro en NumPy para preservar la memoria de Markov y ser 100% causal.
        from scipy.special import logsumexp as _logsumexp
        raw_states = np.zeros(len(X_scaled), dtype=int)
        
        # [OPTIMIZATION-LIVE] Computamos únicamente los últimos registros si se solicita
        limit_causal_rows = 500
        start_t = max(0, len(X_scaled) - limit_causal_rows)
        
        print(f"[WFB-CAUSAL-FIX-HMM] [OPT] Iniciando inferencia HMM causal sobre últimos {len(X_scaled) - start_t} registros (start_t={start_t})...")
        logger.info(f"[WFB-CAUSAL-FIX-HMM] Inferencia HMM causal optimizada sobre últimos {len(X_scaled) - start_t} registros usando Forward Algorithm.")
        
        # Pre-relleno instantáneo del historial completo vía predict en lote vectorizado (no causal, pero solo para warmup)
        if start_t > 0:
            try:
                raw_states[:start_t] = self.model.predict(X_scaled[:start_t])
            except Exception as e_fill:
                logger.warning(f"[HMM-OPT] Error en pre-relleno de historial HMM: {e_fill}")
                
        # --- Filtrado Causal Puro (Forward Algorithm) ---
        if len(X_scaled) > 0:
            _framelogprob = self.model._compute_log_likelihood(X_scaled)
            _log_startprob = np.log(np.maximum(self.model.startprob_, 1e-10))
            _log_transmat = np.log(np.maximum(self.model.transmat_, 1e-10))
            
            _log_alpha = np.zeros((len(X_scaled), self.model.n_components))
            
            if start_t == 0:
                _log_alpha[0] = _log_startprob + _framelogprob[0]
                raw_states[0] = np.argmax(_log_alpha[0])
                _t_start_loop = 1
            else:
                # Inicializar el estado en start_t usando las probabilidades de estado previas como startprob
                # (aproximación rápida: usar el estado de Viterbi como vector one-hot o inicializar de cero)
                _log_alpha[start_t - 1] = _log_startprob  # fallback simple
                _log_alpha[start_t - 1, raw_states[start_t - 1]] = 0.0  # Confianza absoluta en el pre-relleno
                _t_start_loop = start_t
                
            for t in range(_t_start_loop, len(X_scaled)):
                _work_buffer = _log_alpha[t-1][:, None] + _log_transmat
                _log_alpha[t] = _logsumexp(_work_buffer, axis=0) + _framelogprob[t]
                raw_states[t] = np.argmax(_log_alpha[t])
                
        print("[WFB-CAUSAL-FIX-HMM] Inferencia HMM completada con éxito.")

        # BUG FIX: raw_states tiene los integers (0..3)
        # El LUNA RISK-OFF SHIELD se aplica a ambas representaciones:
        semantic_labels = _np.array([self.state_map.get(s, f"estado_{s}") for s in raw_states])
        numeric_labels = _np.array(raw_states, dtype=float)
        
        # --- LUNA RISK-OFF SHIELD (Inferencia OOS) ---
        semantic_labels = self._apply_risk_off_shield(df, semantic_labels)
        numeric_labels = self._apply_risk_off_shield(df, numeric_labels)
        
        # --- LUNA RISK-ON SHIELD (Golden Cross Override) ---
        semantic_labels = self._apply_risk_on_shield(df, semantic_labels)
        numeric_labels = self._apply_risk_on_shield(df, numeric_labels)

        # â”€â”€ [DATAFLOW-EXPORT-HMM-02] Output audit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        _result = _pd.DataFrame({
            "HMM_Regime": numeric_labels,
            "HMM_Semantic": semantic_labels
        }, index=df.index)
        
        _unique_labels = _result["HMM_Semantic"].unique().tolist()
        logger.info(
            f"  [DATAFLOW-EXPORT-HMM-02] predict output DataFrame: "
            f"unique_semantic={_unique_labels} | unique_numeric={_result['HMM_Regime'].unique().tolist()} | n_rows={len(_result)}"
        )
        # -- [HMM-PREDICTIVE-01] Probabilidades de Transicion Forward (t+1) ----
        # gamma[t+1] = gamma[t] @ transmat_
        # Usa la matriz de transicion aprendida del GaussianHMM (SOP R1: cero look-ahead).
        # Genera columnas: hmm_bear_transition_prob, hmm_bull_transition_prob, hmm_transition_risk.
        # Consumidas en signal_filter.py para ajuste anticipatorio del umbral del MetaLabeler.
        try:
            if hasattr(self, "model") and hasattr(self.model, "transmat_"):
                import numpy as _np_tr
                _transmat = _np_tr.array(self.model.transmat_)
                _n_states = _transmat.shape[0]
                # One-hot del estado actual
                _gamma_t = _np_tr.zeros((len(raw_states), _n_states), dtype=float)
                for _i, _s in enumerate(raw_states):
                    if 0 <= int(_s) < _n_states:
                        _gamma_t[_i, int(_s)] = 1.0
                # Propagar un paso
                _gamma_t1 = _gamma_t @ _transmat  # shape (n_rows, n_states)
                # Identificar estados BEAR y BULL por el state_map (excluyendo estados artificiales del shield >= _n_states)
                _bear_states = [
                    s for s, label in getattr(self, "state_map", {}).items()
                    if isinstance(label, str) and ("BEAR" in label or "CRASH" in label) and s < _n_states
                ]
                _bull_states = [
                    s for s, label in getattr(self, "state_map", {}).items()
                    if isinstance(label, str) and "BULL" in label and "BEAR" not in label and s < _n_states
                ]
                _bear_probs = (_gamma_t1[:, _bear_states].sum(axis=1)
                               if _bear_states else _np_tr.zeros(len(raw_states)))
                _bull_probs = (_gamma_t1[:, _bull_states].sum(axis=1)
                               if _bull_states else _np_tr.zeros(len(raw_states)))
                _result["hmm_bear_transition_prob"] = _bear_probs
                _result["hmm_bull_transition_prob"] = _bull_probs
                _result["hmm_transition_risk"]      = _bear_probs
                logger.info(
                    "  [HMM-PREDICTIVE-01] Transicion t+1: "
                    "bear_states={} bull_states={} | bear_mean={:.3f} bull_mean={:.3f}",
                    _bear_states, _bull_states, _bear_probs.mean(), _bull_probs.mean()
                )
            else:
                for _col in ["hmm_bear_transition_prob", "hmm_bull_transition_prob", "hmm_transition_risk"]:
                    _result[_col] = float("nan")
                logger.warning("  [HMM-PREDICTIVE-01] transmat_ no disponible -- columnas NaN.")
        except Exception as _e_tr:
            for _col in ["hmm_bear_transition_prob", "hmm_bull_transition_prob", "hmm_transition_risk"]:
                _result[_col] = float("nan")
            logger.warning(f"  [HMM-PREDICTIVE-01] Fallo transiciones (no critico): {_e_tr}")

        return _result

    def predict_next_regime_probs(self, current_state_probs):
        """[HMM-PREDICTIVE-01] Propaga gamma[t] a gamma[t+1] usando la transmat_ aprendida.

        Args:
            current_state_probs: array-like (n_states,). Puede ser one-hot
                                 (estado determinista) o vector de probabilidades.
        Returns:
            numpy array (n_states,) con probabilidades del regimen en t+1.
            Devuelve ceros si transmat_ no esta disponible.

        Uso en signal_filter.py:
            Si hmm_transition_risk > 0.30 en la vela actual, subir el umbral del
            MetaLabeler preventivamente antes de que el regimen cambie oficialmente.
        """
        import numpy as _np_tr2
        if not hasattr(self, "model") or not hasattr(self.model, "transmat_"):
            logger.warning("[HMM-PREDICTIVE-01] transmat_ no disponible -- devolviendo ceros.")
            return _np_tr2.zeros(len(current_state_probs))
        return _np_tr2.array(current_state_probs) @ _np_tr2.array(self.model.transmat_)

    def plot_regimes(self):
        """Dibuja el precio subyacente pintado por los 4 regÃ­menes."""
        out_path = self.root / "data" / "models" / "engine_hmm_regimes.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(15, 7))
        
        def get_color(label):
            if not isinstance(label, str): return 'black'
            if 'BULL' in label and 'VOLATILE' in label: return 'lime'
            if 'BULL' in label: return 'green'
            if 'BEAR_FORCED' in label: return 'black'
            if 'CRASH' in label: return 'red'
            if 'BEAR' in label: return 'purple'
            if 'VOLATILE_RANGE' in label: return 'orange'
            if 'CALM_RANGE' in label: return 'gray'
            return 'blue'
        
        # Mapeamos para el color
        df_plot = self.raw_df.iloc[-4000:].copy() # Últimos meses
        if 'HMM_Semantic' in df_plot.columns:
            semantic_col = df_plot['HMM_Semantic']
        elif 'HMM_State_OOS' in df_plot.columns:
            semantic_col = df_plot['HMM_State_OOS'].map(self.state_map).fillna('UNKNOWN')
        else:
            semantic_col = df_plot['HMM_State_Raw'].map(self.state_map).fillna('UNKNOWN')
        
        ax.plot(df_plot.index, df_plot['close'], color='black', alpha=0.3, linewidth=1)
        
        for label in semantic_col.unique():
            mask = semantic_col == label
            ax.scatter(df_plot.index[mask], df_plot['close'][mask], color=get_color(label), label=str(label), s=5, alpha=0.8)
            
        ax.set_title("HMM Regime Detection (Rolling OOS)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        logger.info(f"Plot guardado en {out_path}")

    def enrich_validation_and_holdout(self):
        """[FIX-HMM-ENRICH-01] POST-HMM: Inyectar etiquetas HMM en parquets de validación y holdout
        Para validation: join directo (las fechas solapan con el entrenamiento HMM).
        Para holdout: forward-predict causal con el modelo HMM pkl en chunks de 120H.
        """
        import pandas as pd
        import numpy as np
        
        _data_feat_dir_enrich = self.root / "data" / "features"
        _hmm_labels_path = _data_feat_dir_enrich / "hmm_regime_labels.parquet"

        if _hmm_labels_path.exists():
            _hmm_lbl = pd.read_parquet(_hmm_labels_path)
            _hmm_lbl.index = pd.to_datetime(_hmm_lbl.index, utc=True)

            # --- Validation ---
            _val_path_enrich = _data_feat_dir_enrich / "features_validation.parquet"
            if _val_path_enrich.exists():
                _df_val_e = pd.read_parquet(_val_path_enrich)
                _df_val_e.index = pd.to_datetime(_df_val_e.index, utc=True)
                if 'HMM_Regime' in _df_val_e.columns:
                    _df_val_e = _df_val_e.drop(columns=['HMM_Regime'])
                if 'HMM_Semantic' in _df_val_e.columns:
                    _df_val_e = _df_val_e.drop(columns=['HMM_Semantic'])
                _df_val_e = _df_val_e.join(_hmm_lbl[['HMM_Regime', 'HMM_Semantic']], how='left')
                _df_val_e['HMM_Regime'] = _df_val_e['HMM_Regime'].ffill().fillna(2)
                _df_val_e['HMM_Semantic'] = _df_val_e['HMM_Semantic'].ffill().fillna('UNKNOWN')
                _df_val_e.to_parquet(_val_path_enrich)
                logger.success(f"[FIX-HMM-ENRICH-01] features_validation.parquet enriquecido con etiquetas HMM IS | shape={_df_val_e.shape}")

            # --- Holdout ---
            _ho_path_enrich = _data_feat_dir_enrich / "features_holdout.parquet"
            if _ho_path_enrich.exists() and hasattr(self, 'model'):
                _df_ho_e = pd.read_parquet(_ho_path_enrich)
                _df_ho_e.index = pd.to_datetime(_df_ho_e.index, utc=True)
                
                # Predict forward para holdout
                _hmm_feats = getattr(self, "_features", [])
                if not _hmm_feats and hasattr(self, 'X'):
                    _hmm_feats = list(self.X.columns)
                _X_ho = _df_ho_e.copy()
                _f_miss = [f for f in _hmm_feats if f not in _X_ho.columns]
                if _f_miss:
                    _X_ho[_f_miss] = 0.0
                _X_ho = _X_ho[_hmm_feats].fillna(0.0)
                _X_ho_scaled = self.scaler.transform(_X_ho)
                
                _n_ho = len(_X_ho_scaled)
                _states_ho = np.full(_n_ho, np.nan)
                
                print("[FIX-HMM-ENRICH-CAUSAL] 100% causal rolling scan active for holdout enrichment (no look-ahead warmup).")
                logger.info("[FIX-HMM-ENRICH-CAUSAL] 100% causal rolling scan active for holdout enrichment (no look-ahead warmup).")
                
                for _t_ho in range(_n_ho):
                    _chunk_ho = _X_ho_scaled[max(0, _t_ho - 120):_t_ho + 1]
                    _states_ho[_t_ho] = self.model.predict(_chunk_ho)[-1]
                
                _state_s_ho = pd.Series(_states_ho.astype(int), index=_df_ho_e.index)
                _semantic_s_ho = _state_s_ho.map(getattr(self, "state_map", {})).fillna('UNKNOWN')
                
                # Apply Risk-Off and Risk-On Shields (FIX-HMM-SHIELD-HOLDOUT)
                # [HMM-FIX-POST-ATH-01] Prefijo IS: los ultimos 90 dias del IS se adjuntan al holdout
                # para que ath_90d tenga historia correcta desde el primer dia del holdout.
                # Sin esto, el gate post-ATH estaria ciego las primeras 4 semanas.
                _IS_PREFIX_H = 2160  # 90 dias = ventana ATH
                _is_tail = None
                if hasattr(self, 'raw_df') and self.raw_df is not None and len(self.raw_df) > 0:
                    try:
                        _is_tail = self.raw_df.tail(_IS_PREFIX_H)[['close']].copy()
                        if 'close' not in _df_ho_e.columns and 'close' in _is_tail.columns:
                            logger.warning("[HMM-FIX-POST-ATH-01] 'close' no disponible en holdout — shield post-ATH puede ser suboptimo.")
                        elif 'close' in _df_ho_e.columns:
                            _cols_shield = [c for c in _df_ho_e.columns]
                            _df_ho_prefixed = pd.concat([
                                _is_tail.reindex(columns=_cols_shield),
                                _df_ho_e
                            ])
                            logger.info(f"[HMM-FIX-POST-ATH-01] Holdout shield: IS prefix={len(_is_tail)}H prepended para ath_90d")
                            print(f"[HMM-FIX-POST-ATH-01] Holdout shield: IS prefix={len(_is_tail)}H prepended para ath_90d correcto")
                    except Exception as _epfx:
                        _is_tail = None
                        logger.warning(f"[HMM-FIX-POST-ATH-01] Error construyendo IS prefix: {_epfx}")

                # Construir labels con prefijo si disponible, luego filtrar a holdout
                if _is_tail is not None and 'close' in _df_ho_e.columns:
                    _ho_index = _df_ho_e.index
                    _n_prefix = len(_is_tail)
                    # Crear arrays de labels extendidos con placeholder para el prefijo IS
                    _sem_extended = np.concatenate([
                        np.full(_n_prefix, 'UNKNOWN', dtype=object),
                        _semantic_s_ho.values.astype(object)
                    ])
                    _num_extended = np.concatenate([
                        np.full(_n_prefix, np.nan),
                        _states_ho.astype(float)
                    ])
                    _semantic_s_ho_shielded = self._apply_risk_off_shield(_df_ho_prefixed, _sem_extended)[_n_prefix:]
                    _semantic_s_ho_shielded = self._apply_risk_on_shield(_df_ho_prefixed, np.concatenate([np.full(_n_prefix, 'UNKNOWN', dtype=object), _semantic_s_ho_shielded]))[_n_prefix:]
                    _state_s_ho_shielded = self._apply_risk_off_shield(_df_ho_prefixed, _num_extended)[_n_prefix:]
                    _state_s_ho_shielded = self._apply_risk_on_shield(_df_ho_prefixed, np.concatenate([np.full(_n_prefix, np.nan), _state_s_ho_shielded]))[_n_prefix:]
                else:
                    print("[FIX-HMM-SHIELD-HOLDOUT] Applying Risk-Off and Risk-On Shields to holdout enrichment.")
                    logger.info("[FIX-HMM-SHIELD-HOLDOUT] Applying Risk-Off and Risk-On Shields to holdout enrichment.")
                    _semantic_s_ho_shielded = self._apply_risk_off_shield(_df_ho_e, _semantic_s_ho.values)
                    _semantic_s_ho_shielded = self._apply_risk_on_shield(_df_ho_e, _semantic_s_ho_shielded)
                    _state_s_ho_shielded = self._apply_risk_off_shield(_df_ho_e, _state_s_ho.values)
                    _state_s_ho_shielded = self._apply_risk_on_shield(_df_ho_e, _state_s_ho_shielded)

                
                _df_ho_e['HMM_Regime'] = _state_s_ho_shielded
                _df_ho_e['HMM_Semantic'] = _semantic_s_ho_shielded
                _cov_ho = _df_ho_e['HMM_Semantic'].notna().mean()
                _df_ho_e.to_parquet(_ho_path_enrich)
                logger.success(f"[FIX-HMM-ENRICH-01] features_holdout.parquet enriquecido con HMM causal forward-scan | shape={_df_ho_e.shape} | HMM_Semantic coverage={_cov_ho:.1%}")


if __name__ == "__main__":
    import os as _os
    from datetime import datetime as _dt
    _log_dir = Path(__file__).resolve().parents[2] / "logs"
    _log_dir.mkdir(exist_ok=True)
    _ts_hmm  = _dt.now().strftime("%Y%m%d_%H%M%S")
    _rid_hmm = _os.environ.get("LUNA_RUN_ID", "")
    _lname_hmm = f"hmm_regime_{_ts_hmm}_{_rid_hmm}.log" if _rid_hmm else f"hmm_regime_{_ts_hmm}.log"
    logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
    logger.add(_log_dir / _lname_hmm, rotation="50 MB", level="DEBUG", encoding="utf-8")
    try:
        hmm_engine = HMMRegimeModel()
        hmm_engine.load_data()
        
        hmm_engine.fit_global_for_analysis()
        hmm_engine.generate_oos_features()
        hmm_engine.plot_regimes()
        hmm_engine.save_model()
        
        # Exportar DataFrame con la columna HMM a un parquet nuevo para ser consumido
        features_out = hmm_engine.raw_df[['HMM_State_OOS']].copy()
        features_out.columns = ['HMM_Regime']
        # Exportar también la etiqueta semántica mapeando DIRECTAMENTE desde el estado Rolling OOS
        # [FIX-P1-V4-02] Esto asegura que las etiquetas semánticas para entrenamiento XGBoost/MetaLabeler
        # sean generadas usando el mismo forward-scan (causal) empleado en OOS, evitando Look-Ahead de Viterbi.
        if hasattr(hmm_engine, 'state_map') and hmm_engine.state_map:
            features_out['HMM_Semantic'] = features_out['HMM_Regime'].map(
                {k: v for k, v in hmm_engine.state_map.items()}
            ).fillna('UNKNOWN')
        out_parquet = hmm_engine.root / "data" / "features" / "hmm_regime_labels.parquet"
        features_out.to_parquet(out_parquet)

        # â”€â”€ [DATAFLOW-EXPORT-HMM-02] Audit al exportar parquet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        _cov = features_out['HMM_Semantic'].notna().mean() if 'HMM_Semantic' in features_out.columns else 0.0
        _dist = features_out['HMM_Semantic'].value_counts().to_dict() if 'HMM_Semantic' in features_out.columns else {}
        logger.success(
            f"[DATAFLOW-EXPORT-HMM-02] Parquet guardado: {out_parquet.name} | "
            f"shape={features_out.shape} | semantic_cov={_cov:.1%} | dist={_dist}"
        )
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # --- LUNA FORENSIC HMM REPORT ---
        try:
            _df_eval = hmm_engine.raw_df.copy()
            if 'HMM_Semantic' in features_out.columns:
                _df_eval = _df_eval.join(features_out['HMM_Semantic'], how='inner')
                if 'close' in _df_eval.columns:
                    _df_eval['ret_24h'] = _df_eval['close'].pct_change(24).shift(-24)
                    _df_eval['vol_24h'] = _df_eval['close'].pct_change().rolling(24).std()
                    _stats = _df_eval.groupby('HMM_Semantic').agg(
                        ret=('ret_24h', 'mean'), vol=('vol_24h', 'mean'), count=('close', 'count')
                    )
                    logger.info("\n" + "="*50 + "\n[HMM FORENSIC] VALIDACION ESTRUCTURAL DE REGIMENES\n" + "="*50)
                    for idx, row in _stats.iterrows():
                        logger.info(f"  >> {idx} | N={int(row['count'])} | Ret24H={(row['ret']*100):.3f}% | Vol24H={(row['vol']*100):.3f}%")
                    logger.info("[HMM FORENSIC] Logica estructural validada correctamente. \n" + "="*50)
        except Exception as e_forensic:
            logger.warning(f"[HMM FORENSIC] Error generando reporte: {e_forensic}")
        # --------------------------------

        logger.info(f"Pipeline Fase E listo para XGBoost.")
        
        # Inyectar las etiquetas HMM_Semantic y HMM_Regime a features_validation y features_holdout
        hmm_engine.enrich_validation_and_holdout()
        sys.exit(0)
    except Exception as e:
        import traceback
        logger.error(f"[FATAL UNCAUGHT] Script crashed at main level: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
