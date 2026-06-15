"""
generate_oos_predictions.py -- Luna V1 v1.5
=======================================
Genera las predicciones Out-Of-Sample reales al aplicar el pipeline
entrenado completo sobre el periodo OOS purgado.

Pipeline de filtrado:
  1. XGBoost -> probabilidad base
  2. OOD Guard (Isolation Forest) -> bloquea barras anomalas off-distribution
  3. MetaLabelerV2 (RollingStats extractor + RF arbitro) -> filtro secundario
     de senales; solo pasan donde XGB Y MetaV2 confirman LONG

Salida: data/predictions/oos_trades.parquet
  Columnas: timestamp | return_pct | return_raw | tribe_mult | is_win | xgb_prob | meta_v2_prob | entry_time | exit_time

Fuente OOS (jerarquia):
  1. features_holdout.parquet (2025+ -- holdout real, nunca visto)
  2. features_validation.parquet (2024 -- semi-conocido, fallback)
  3. split 20% sobre features_train (sesgado -- ultimo recurso)
"""
import sys
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(get_project_root()))

from luna.utils.encoding_fix import fix_stdout_encoding; fix_stdout_encoding()

import json
import logging
import numpy as np

# Sistema de trazabilidad (pipeline_invariants)
try:
    from luna.utils.pipeline_invariants import check_oos_df, check_trades_df, check_config_consistency
    _INVARIANTS_AVAILABLE = True
except ImportError:
    _INVARIANTS_AVAILABLE = False

import pandas as pd


from loguru import logger

# ARCH-FIX-CALIB-01: importar clases top-level del calibrador para que
# joblib pueda deserializar metalabeler_v2_calibrator.joblib correctamente.
# Sin este import, Python no puede reconstruir _RFWithAdapter en este proceso.
# BUG-CALIB-CACHE-01 (2026-05-19): si este import falla silenciosamente,
# el calibrador deserializa como un objeto roto → xgb_prob_cal queda plano (std=0.0).
# El print debe ser VISIBLE en logs para detectarlo durante runs WFB.
try:
    from luna.models.calibrate_probabilities import (  # noqa: F401
        _RFWithAdapter, _IdentityWrapper, _TSAdapter
    )
    print("[BUG-CALIB-CACHE-01] OK: _RFWithAdapter, _IdentityWrapper, _TSAdapter importados correctamente para deserialización joblib.")
except ImportError as _calib_import_err:
    print(f"[BUG-CALIB-CACHE-01] CRITICO: No se pudo importar clases del calibrador: {_calib_import_err}. "
          "El metalabeler_v2_long_calibrator.joblib puede quedar plano (std=0.0). "
          "Verificar que calibrate_probabilities.py existe y tiene _RFWithAdapter como clase top-level.")

# [FIX-PLATT-DESER-01 2026-06-02] Importar PlattCalibrator para que joblib pueda deserializar
# xgboost_isotonic_calibrator_bull_long.joblib que contiene objetos PlattCalibrator.
# Sin este import: "Can't get attribute 'PlattCalibrator' on <module predict_oos>"
# Efecto del bug: calibrador bull NO cargado → xgb_prob_cal == xgb_prob_raw → WR degradado (32.9% vs 54%).
# PlattCalibrator esta definido en signal_filter.py y train_xgboost_v2.py (no en calibrate_probabilities).
# Mismo patron aplicado en regime_router.py (L12-30).
try:
    from luna.models.signal_filter import PlattCalibrator  # noqa: F401 - necesario para joblib.load()
    print("[FIX-PLATT-DESER-01] OK: PlattCalibrator importado desde signal_filter en predict_oos — joblib deserializa calibradores bull.")  # RULE[fixbugsprints.md]
except ImportError:
    try:
        from luna.models.train_xgboost_v2 import PlattCalibrator  # noqa: F401 - fallback
        print("[FIX-PLATT-DESER-01] OK: PlattCalibrator importado desde train_xgboost_v2 (fallback) en predict_oos.")  # RULE[fixbugsprints.md]
    except ImportError as _platt_err:
        print(f"[FIX-PLATT-DESER-01] WARN: PlattCalibrator no importable desde ningún módulo: {_platt_err}. Calibrador bull puede fallar.")


logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

# ── Log file propio del subproceso (trazabilidad por RUN_ID) ──────────────
import os as _os_oos
from datetime import datetime as _dt_oos
_log_dir_oos = get_project_root() / "logs"
_log_dir_oos.mkdir(exist_ok=True)
_ts_oos  = _dt_oos.now().strftime("%Y%m%d_%H%M%S")
_rid_oos = _os_oos.environ.get("LUNA_RUN_ID", "")
_lname_oos = f"generate_oos_{_ts_oos}_{_rid_oos}.log" if _rid_oos else f"generate_oos_{_ts_oos}.log"
logger.add(_log_dir_oos / _lname_oos, rotation="50 MB", level="DEBUG", encoding="utf-8")
# ─────────────────────────────────────────────────────────────────────────



# [V2-P1] TBM Dinamico HMM
HMM_TBM_PARAMS = {
    "1_VOLATILE_BULL":   {"sl": 2.5, "tp": 3.0},
    # [FIX-TBM-BULLGRIND-01 2026-06-02] 1_BULL_GRIND: regimen alcista de bajo momentum detectado en OOS 2025.
    # BTC en consolidacion lenta al alza. SL conservador, TP reducido vs BULL_TREND (menor direccionalidad).
    # Origen: HMM 5-estados genera este label semantico en datos post-cutoff 2025.
    "1_BULL_GRIND":      {"sl": 1.2, "tp": 2.0},
    # [LUNA-V2-FIX] Asimetría de Tendencia Optimizada: tp 2.0 -> 2.5 para BULL_TREND
    "1_BULL_TREND":      {"sl": 1.5, "tp": 2.5},
    "2_CALM_RANGE":      {"sl": 0.6, "tp": 1.2},
    "2_VOLATILE_RANGE":  {"sl": 1.2, "tp": 1.8},
    # [CRASH-FIX-TBM 2026-05-30] 3_CALM_BEAR: regimen de 5 estados generado por HMM (ret30d=-7%, vol=63%).
    # Ausencia causaba ValueError en get_hmm_tbm_params() -> crash de predict_oos.py.
    # Params: SL conservador, TP bajo — regimen bajista moderado (entre VOLATILE_RANGE y BEAR_CRASH).
    "3_CALM_BEAR":       {"sl": 1.0, "tp": 1.2},
    # FIX-I4 (2026-04-29): tp 2.5->1.5 — objetivo mas realista en regimen bajista.
    "3_BEAR_CRASH":      {"sl": 1.5, "tp": 1.5},
}
HMM_HORIZON_MAP = {
    # [CAPA-4-ARQUITECTURA Camino B] Reducción agresiva de la Barrera Vertical (VB)
    # Evita trades zombies con retorno negativo sistemático pasadas 48H.
    "1_VOLATILE_BULL":   48,
    "1_BULL_GRIND":      48,
    "1_BULL_TREND":      48,
    "2_CALM_RANGE":      24,
    "2_VOLATILE_RANGE":  24,
    "3_CALM_BEAR":       24,
    "3_BEAR_CRASH":      24,
}
_HMM_TBM_FALLBACK = {"sl": 0.8, "tp": 1.5}
_HMM_HORIZON_FALLBACK = 48

# --- RESOLVEDORES ROBUSTOS CONTRA SILENT REGIME FALLBACK ---
# [BUG-FIX-HMM-RESOLVER] (2026-05-20): Permite mapear variantes semánticas dinámicas
# (por ejemplo, '1_BULL_TREND_WEAK', '1_BULL_TREND_B', '1_VOLATILE_BULL_B') a su base HMM canonical,
# evitando que caigan silenciosamente en el fallback neutral/genérico _HMM_TBM_FALLBACK.
def get_hmm_tbm_params(regime_name: str) -> dict:
    """
    Resuelve de manera robusta los parámetros del Triple Barrier Method para un régimen HMM.
    Soporta coincidencia exacta, y si no, busca por prefijo base (ej. '1_BULL_TREND_WEAK' -> '1_BULL_TREND').
    """
    if not isinstance(regime_name, str):
        regime_name = str(regime_name)
    regime_str = regime_name.strip()
    
    if regime_str in HMM_TBM_PARAMS:
        return HMM_TBM_PARAMS[regime_str]
        
    # Intentar coincidencia por prefijo base
    for base_regime in HMM_TBM_PARAMS.keys():
        if regime_str.startswith(base_regime):
            print(f"[BUG-FIX-HMM-RESOLVER] OK: Regime '{regime_str}' mapeado dinámicamente a base '{base_regime}' para TBM params.")
            return HMM_TBM_PARAMS[base_regime]
            
    # Quitar sufijos comunes si no coincide prefijo directo
    import re as _re_regime
    _stripped = _re_regime.sub(r'_(WEAK|[B-D])$', '', regime_str)
    if _stripped in HMM_TBM_PARAMS:
        print(f"[BUG-FIX-HMM-RESOLVER] OK: Regime '{regime_str}' mapeado a '{_stripped}' (sufijo removido) para TBM params.")
        return HMM_TBM_PARAMS[_stripped]
        
    # --- NUEVA GUARDA FAIL-FAST CONTRA MAPEOS INCOMPLETOS EN REGÍMENES CONOCIDOS ---
    if regime_str.startswith(("1_", "2_", "3_")):
        err_msg = f"[CRITICAL-FALLBACK-ALERT] Régimen conocido '{regime_str}' no pudo ser resuelto a ninguna clave base de HMM_TBM_PARAMS!"
        print(f"[CRITICAL-FALLBACK-ALERT] {err_msg}")  # trazabilidad institucional RULE[fixbugsprints.md]
        raise ValueError(err_msg)
        
    print(f"[BUG-FIX-HMM-RESOLVER] WARN: Régimen '{regime_str}' no reconocido. Usando fallback _HMM_TBM_FALLBACK.")
    return _HMM_TBM_FALLBACK

def get_hmm_horizon(regime_name: str) -> int:
    """
    Resuelve de manera robusta el horizonte de tiempo máximo para un régimen HMM.
    """
    if not isinstance(regime_name, str):
        regime_name = str(regime_name)
    regime_str = regime_name.strip()
    
    if regime_str in HMM_HORIZON_MAP:
        return HMM_HORIZON_MAP[regime_str]
        
    for base_regime in HMM_HORIZON_MAP.keys():
        if regime_str.startswith(base_regime):
            return HMM_HORIZON_MAP[base_regime]
            
    import re as _re_regime
    _stripped = _re_regime.sub(r'_(WEAK|[B-D])$', '', regime_str)
    if _stripped in HMM_HORIZON_MAP:
        return HMM_HORIZON_MAP[_stripped]
        
    # --- NUEVA GUARDA FAIL-FAST CONTRA MAPEOS INCOMPLETOS EN REGÍMENES CONOCIDOS ---
    if regime_str.startswith(("1_", "2_", "3_")):
        err_msg = f"[CRITICAL-FALLBACK-ALERT] Régimen conocido '{regime_str}' no pudo ser resuelto a ninguna clave base de HMM_HORIZON_MAP!"
        print(f"[CRITICAL-FALLBACK-ALERT] {err_msg}")  # trazabilidad institucional RULE[fixbugsprints.md]
        raise ValueError(err_msg)
        
    return _HMM_HORIZON_FALLBACK


class OOSTradesGenerator:
    """
    Genera predicciones OOS reales simulando el pipeline productivo
    (XGBoost → Triple Barrier Method) sobre el holdout temporal.
    """

    def __init__(self):
        import os
        self.root = get_project_root()
        self.data_dir = self.root / "data"
        self.models_dir = self.data_dir / "models"

        # [WFB-ISOLATION] Redirigir a cache especifico de la ventana si estamos en WFB
        wfb_dir = os.environ.get("WFB_WINDOW_DIR")
        if wfb_dir:
            self.root = Path(wfb_dir)
            self.data_dir = self.root / "data"
            self.models_dir = self.data_dir / "models"
            logger.info("  [WFB-ISOLATION] Contexto OOS aislado detectado en: {}", self.root)
        else:
            _wid = os.environ.get("LUNA_WINDOW_ID", "")
            _seed = os.environ.get("LUNA_SEED", "42")
            if _wid:
                # [V2-FIX-DATAFLOW] En V2, hydrate_window_state restaura los modelos y las features
                # al workspace activo (data/models y data/features). No se debe leer de wfb_cache directamente.
                self.models_dir = self.data_dir / "models"
                logger.info("  [WFB-ISOLATION] Contexto OOS WFB: seed{}/{}, usando workspace activo", _seed, _wid)

    @staticmethod
    def _calc_btc_cycle_position(oos_close: "pd.Series", root: "Path") -> "pd.Series":
        """
        BUG-05 fix (2026-03-17): cÃ¡lculo centralizado de btc_cycle_position.

        Percentil del precio en la ventana rolling 365d (8760H).
        Antes se calculaba dos veces con cÃ³digo idÃ©ntico:
          - Bloque R21 (timing features, inside try)
          - LEGACY-04 GUARD (fallback si R21 fallÃ³)
        Ahora ambos bloques llaman a esta funciÃ³n â€” fuente Ãºnica de verdad.

        Carga features_train.parquet para proveer historia completa al rolling 365d.
        Fallback: solo datos holdout (rolling potencialmente incompleto en las primeras filas).
        """
        import pandas as _pd
        try:
            _train_close = _pd.read_parquet(
                root / "data" / "features" / "features_train.parquet",
                columns=["close"]
            )["close"]
            # [P2-7-FIX] drop_duplicates() sin subset no elimina duplicados de índice si el precio difiere.
            _full_close = _pd.concat([_train_close, oos_close]).sort_index()
            _full_close = _full_close[~_full_close.index.duplicated(keep='last')]
        except Exception:
            _full_close = oos_close  # fallback: solo holdout (rolling incompleto en primeras filas)

        _roll = _full_close.rolling(window=8760, min_periods=720)
        _rmin = _roll.min()
        _rmax = _roll.max()
        _rng  = (_rmax - _rmin).replace(0, float("nan"))
        _cycle = ((_full_close - _rmin) / _rng).clip(0.0, 1.0)
        return _cycle.reindex(oos_close.index)

    @staticmethod
    def _calc_btc_drawdown_from_ath(oos_close: "pd.Series", root: "Path", window_h: int = 90 * 24) -> "pd.Series":
        """
        P2-7-FIX (2026-03-30): cálculo centralizado de btc_drawdown_from_ath con historia completa.

        Mismo patrón que _calc_btc_cycle_position: concatena features_train.parquet para
        que el rolling ATH disponga de historia completa en las primeras filas del holdout.

        Sin este fix, el rolling de 90d (2160H) en las primeras barras del holdout usa
        solo las últimas 24H (min_periods) → ATH ≈ precio actual → drawdown ≈ 0.0
        (feature artificialmente optimista que degrada la discriminación del modelo en OOS).

        Args:
            oos_close:  Serie de precios close del periodo OOS/holdout.
            root:       Raíz del proyecto (para cargar features_train.parquet).
            window_h:   Ventana rolling en horas (default=2160H=90d, igual que feature_pipeline.py).

        Returns:
            Serie alineada al índice de oos_close con valores en (-inf, 0].
        """
        import pandas as _pd
        try:
            _train_close = _pd.read_parquet(
                root / "data" / "features" / "features_train.parquet",
                columns=["close"]
            )["close"]
            # [P2-7-FIX] drop_duplicates() sin subset no elimina duplicados de índice si el precio difiere.
            _full_close = _pd.concat([_train_close, oos_close]).sort_index()
            _full_close = _full_close[~_full_close.index.duplicated(keep='last')]
        except Exception:
            _full_close = oos_close  # fallback: solo holdout (ATH incompleto en primeras filas)

        min_periods_h = min(window_h, 720)  # mínimo 30d de historia antes de producir valores
        _ath_full = _full_close.rolling(window=window_h, min_periods=min_periods_h).max().clip(lower=1e-8)
        _drawdown = (_full_close / _ath_full) - 1.0
        return _drawdown.reindex(oos_close.index)

    def generate(self) -> bool:
        """
        Ejecuta la generación de trades OOS.
        Returns: True si se generaron trades correctamente, False si hubo error.
        """
        logger.info("Ã°Å¸â€ºÂ Ã¯Â¸Â Generando dataset de predicciones OOS reales...")

        # Ã¢â€â‚¬Ã¢â€â‚¬ Verificar existencia de archivos necesarios Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        features_path = self.data_dir / "features" / "features_train.parquet"
        xgb_model_path = self.models_dir / "xgboost_meta.model"
        xgb_sig_path   = self.models_dir / "xgboost_meta_signature.json"
        # ── Verificar existencia de archivos necesarios ──
        try:
            from config.settings import cfg as _cfg_xgb
            use_regime = getattr(_cfg_xgb.fase2, 'use_regime_agents', False)
            use_lgbm = getattr(_cfg_xgb.fase2, 'use_lgbm_ensemble', False)
        except Exception:
            use_regime = False
            use_lgbm = False

        if use_regime:
            required_files = [features_path]
            for model_name in ["bull", "range", "bear"]:
                for _d in ["long", "short"]:
                    required_files.append(self.models_dir / f"xgboost_meta_{model_name}_{_d}.model")
                    required_files.append(self.models_dir / f"xgboost_meta_{model_name}_{_d}_signature.json")
                    if use_lgbm:
                        required_files.append(self.models_dir / f"lgbm_meta_{model_name}_{_d}.model")
                        required_files.append(self.models_dir / f"lgbm_meta_{model_name}_{_d}_signature.json")
        else:
            required_files = [features_path, xgb_model_path, xgb_sig_path]

        for path in required_files:
            if not path.exists():
                if path == features_path:
                    logger.error(f"Archivo requerido no encontrado: {path}")
                    logger.error("Asegúrate de ejecutar el feature engineering primero:")
                    return False
                else:
                    logger.warning(f"Modelo no encontrado (se omitirá en el router): {path}")

        # ── Cargar features ──
        logger.info("Cargando features_train.parquet...")
        df = pd.read_parquet(features_path)
        logger.info(f"  Shape: {df.shape}")
        hmm_path = self.data_dir / "features" / "hmm_regime_labels.parquet"

        if use_regime:
            # MEJ-GEN-02: dict.fromkeys() para orden determinístico entre runs
            _all_regime_feats = []
            _optimal_threshold_per_regime_long = {}
            _optimal_threshold_per_regime_short = {}
            
            # Need to know the semantic to numeric mapping for XGBoost
            _sem_to_num = {}
            try:
                if hmm_path.exists():
                    _hmm_df = pd.read_parquet(hmm_path)
                    if "HMM_Semantic" in _hmm_df.columns and "HMM_Regime" in _hmm_df.columns:
                        for _, _row in _hmm_df.drop_duplicates(subset=["HMM_Semantic"]).iterrows():
                            _sem_to_num[str(_row["HMM_Semantic"])] = int(float(_row["HMM_Regime"]))
            except Exception:
                pass
                
            from config.settings import cfg as _cfg_regimes
            # [SOL3-CALM-BEAR-01 2026-06-01] Fallback actualizado: calm_bear separado de bear.
            _rm = getattr(_cfg_regimes.fase2, 'regimes_config', {
                "bull":      ["1_BULL_TREND", "1_VOLATILE_BULL"],
                "range":     ["2_CALM_RANGE", "2_VOLATILE_RANGE"],
                "calm_bear": ["3_CALM_BEAR", "3_CALM_BEAR_B", "3_CALM_BEAR_C", "3_CALM_BEAR_D"],
                "bear":      ["3_BEAR_CRASH", "3_BEAR_CRASH_B", "4_BEAR_FORCED"]
            })
            print("[SOL3-CALM-BEAR-01/PREDICT] regime_mapping cargado para predict_oos.")

            # [SOL3-CALM-BEAR-01 2026-06-01] calm_bear añadido al loop de carga de modelos
            for model_name in ["bull", "range", "calm_bear", "bear"]:
                for _d in ["long", "short"]:
                    p = self.models_dir / f"xgboost_meta_{model_name}_{_d}_signature.json"
                    if p.exists():
                        with open(p, "r") as f:
                            sig = json.load(f)
                            _all_regime_feats.extend(sig["features"])
                            
                            # Map the optimal_threshold for this agent to the regimes it handles
                            _thr = float(sig.get("optimal_threshold", 0.51))
                            _sem_list = _rm.get(model_name, [])
                            for _sem in _sem_list:
                                _num_id = _sem_to_num.get(_sem)
                                if _num_id is not None:
                                    if _d == "long":
                                        _optimal_threshold_per_regime_long[str(_num_id)] = _thr
                                    else:
                                        _optimal_threshold_per_regime_short[str(_num_id)] = _thr

            xgb_features = list(dict.fromkeys(_all_regime_feats))
            
            # Sintetizar el Signature maestro para signal_filter.py por direccin
            with open(self.models_dir / "xgboost_meta_long_signature.json", "w") as f:
                json.dump({"features": xgb_features, "optimal_threshold_per_regime": _optimal_threshold_per_regime_long}, f, indent=4)
            with open(self.models_dir / "xgboost_meta_short_signature.json", "w") as f:
                json.dump({"features": xgb_features, "optimal_threshold_per_regime": _optimal_threshold_per_regime_short}, f, indent=4)
                
            # Compatibilidad legacy
            with open(xgb_sig_path, "w") as f:
                json.dump({"features": xgb_features, "optimal_threshold_per_regime": _optimal_threshold_per_regime_long}, f, indent=4)
        else:
            with open(xgb_sig_path, "r") as f:
                sig = json.load(f)
            xgb_features = sig["features"]
            
        logger.info(f"  Features XGBoost (unión): {len(xgb_features)}")


        if hmm_path.exists():
            df_hmm = pd.read_parquet(hmm_path)
            _overlap = [c for c in df_hmm.columns if c in df.columns]
            if _overlap:
                logger.info(f"[FIX-HMM-JOIN] Eliminando columnas solapadas antes del join: {_overlap}")
                df_hmm = df_hmm.drop(columns=_overlap)
            df = df.join(df_hmm, how="left")
            logger.info("  HMM labels integrados.")
        else:
            logger.warning("  hmm_regime_labels.parquet no encontrado.")

        if "close" not in df.columns:
            logger.error("Columna close no encontrada.")
            return False

        try:
            if "FundingRate" in df.columns:
                df["timing_funding_acum8h"] = df["FundingRate"].ewm(span=8, min_periods=1).mean()
            if "close" in df.columns:
                _r24h_b = df["close"].pct_change(24)
                _r7d_b  = df["close"].pct_change(168)
                df["timing_momentum_div"] = _r24h_b - _r7d_b
            if "close" in df.columns and "volume" in df.columns:
                _r24h_abs_b  = df["close"].pct_change(24).abs()
                _vol_ma_b    = df["volume"].rolling(window=720, min_periods=48).mean()
                _vol_ratio_b = df["volume"] / (_vol_ma_b + 1e-6)
                # P2-3-FIX (2026-03-30): guardia mínima para evitar división por ratio cerca de cero.
                # Antes: _vol_ratio=0 y epsilon=1e-6 en denominador → valores >1e6 antes del clip.
                # Ahora: clamp _vol_ratio a [0.01, inf] para limitar el spike a 100x.
                _vol_ratio_b = _vol_ratio_b.clip(lower=0.01)
                df["timing_vol_divergence"] = (_r24h_abs_b / (_vol_ratio_b + 1e-6)).clip(upper=5.0)
            logger.info("  [R21] Features de timing calculadas en df base")
        except Exception as _e_base:
            logger.warning(f"  [R21] Error en timing: {_e_base}")

        available_feats = [f for f in xgb_features if f in df.columns]
        # HMM_Regime se incluye si el modelo lo requiere (nuevo modelo V10 lo usa como feature numerica).
        # HMM_Semantic se excluye siempre (string, solo para filtrar regimenes, no para XGBoost).
        _exclude = ["timestamp", "close", "entry_time", "HMM_Semantic", "HMM_State_Raw"]
        available_feats = [f for f in available_feats if f not in _exclude]
        # Guard: si HMM_Regime esta en xgb_features pero no en df.columns, añadir con 0.0
        if "HMM_Regime" in xgb_features and "HMM_Regime" not in df.columns:
            df["HMM_Regime"] = 0.0
            logger.warning("  [FIX-HMM-AVAIL] HMM_Regime ausente en df base -- inicializado a 0.0")
        if "HMM_Regime" in xgb_features and "HMM_Regime" not in available_feats:
            available_feats.append("HMM_Regime")
        missing_feats   = [f for f in xgb_features if f not in df.columns and f not in _exclude]
        if missing_feats:
            logger.warning(f"  missing features: {missing_feats[:5]}")
        # CORRECCIÃƒâ€œN: dropna solo sobre 'close' (precio requerido para TBM).
        # XGBoost maneja NaN nativamente en features Ã¢â‚¬â€ NO dropna agresivo.
        df_clean = df.dropna(subset=["close"]).copy()

        # Guard: rellenar con 0 solo features 100% NaN en el período OOS (imposibles)
        for feat in available_feats:
            if feat in df_clean.columns and df_clean[feat].isna().all():
                logger.warning(f"  Feature 100%% NaN en período OOS: {feat} Ã¢â‚¬â€ rellenando con 0")
                df_clean[feat] = 0.0

        logger.info(f"  Dataset limpio: {len(df_clean)} filas")

        # Ã¢â€â‚¬Ã¢â€â‚¬ Definir período OOS Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        # LOG-BUG-04 fix (2026-03-10): PRIORIDAD DE FUENTE OOS cambiada.
        # ANTES (HOLD-02): features_validation.parquet era el primario.
        #   Problema: validation cubre 2024-07-01..2024-12-31 Ã¢â‚¬â€ el modelo fue validado
        #   en ese periodo (semi-conocido). WR y DSR resultantes eran OPTIMISTAS FALSOS.
        # AHORA: features_holdout.parquet es el primario (2025-01-01+, nunca visto).
        #   features_validation.parquet pasa a ser fallback secundario.
        #   Split 20% sigue siendo el ÃƒÂºltimo recurso.
        # AUDIT GAP-01 FIX (2026-04-29): seleccionar features_holdout_W{N}.parquet si existe.
        # En modo WFB, LUNA_WINDOW_ID está inyectado por run_walkforward_pipeline_v2.py.
        # Cada ventana escribe su propio parquet en feature_pipeline.py (línea 1262).
        # Sin este fix, todas las ventanas leen el genérico (el último escrito → siempre W4 en run completo).
        import os as _os_gen
        _wid_gen = _os_gen.environ.get("LUNA_WINDOW_ID", "")
        _holdout_window_path = self.data_dir / "features" / f"features_holdout_{_wid_gen}.parquet"
        if _wid_gen and _holdout_window_path.exists():
            holdout_features_path = _holdout_window_path
            logger.info(
                "[AUDIT-GAP-01] WFB mode: usando features_holdout_{}.parquet (window-specific, {} filas)",
                _wid_gen, len(pd.read_parquet(holdout_features_path, columns=["close"]))
                if _holdout_window_path.exists() else "?",
            )
        else:
            holdout_features_path = self.data_dir / "features" / "features_holdout.parquet"
            if _wid_gen and not _holdout_window_path.exists():
                logger.warning(
                    "[AUDIT-GAP-01] features_holdout_{}.parquet no existe — usando genérico. "
                    "Regenerar feature_pipeline.py con --window-id {} para corregir.",
                    _wid_gen, _wid_gen,
                )
        val_features_path     = self.data_dir / "features" / "features_validation.parquet"

        if holdout_features_path.exists():
            logger.info("✅ LOG-BUG-04: Usando features_holdout.parquet como fuente OOS "
                        "(2025+ Ã¢â‚¬â€ holdout real, nunca visto por el modelo)")
            df_oos_raw = pd.read_parquet(holdout_features_path)
            
            # â”€â”€ [DATAFLOW-IMPORT-PRED] OOS Source Audit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            _start = df_oos_raw.index.min().date() if len(df_oos_raw) > 0 else "N/A"
            _end   = df_oos_raw.index.max().date() if len(df_oos_raw) > 0 else "N/A"
            logger.success(
                f"[DATAFLOW-IMPORT-PRED] Holdout cargado: shape={df_oos_raw.shape} | "
                f"fechas={_start} -> {_end}"
            )
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

            if "close" not in df_oos_raw.columns:
                logger.error("features_holdout.parquet no tiene columna 'close'. Abortando.")
                return False

            # Integrar HMM labels Ã¢â‚¬â€ P2-BUG-META-HMM-01
            # hmm_regime_labels.parquet solo cubre hasta train_cutoff (2024-06).
            # Para el holdout 2025 el join da todo NaN → MetaLabeler recibe 33 features
            # en vez de 37 → ValueError. Fix: predecir rÃƒÂ©gimen con HMMRegimeModel.load().
            if hmm_path.exists():
                df_hmm_hld = pd.read_parquet(hmm_path)
                df_oos_raw = df_oos_raw.copy()
                cols_to_drop = [c for c in df_hmm_hld.columns if c in df_oos_raw.columns]
                if cols_to_drop:
                    df_oos_raw = df_oos_raw.drop(columns=cols_to_drop)
                df_oos_raw = df_oos_raw.join(df_hmm_hld, how="left")

            # â”€â”€ [LOG-DIAG-HMM-01] HMM DATA AUDIT post-join â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Detecta type mismatch antes de que el filtro isin() falle silenciosamente.
            for _hcol in ["HMM_Regime", "HMM_Semantic"]:
                if _hcol in df_oos_raw.columns:
                    _hcov  = df_oos_raw[_hcol].notna().mean()
                    _hdtyp = df_oos_raw[_hcol].dtype
                    _hsamp = df_oos_raw[_hcol].dropna().head(3).tolist()
                    logger.info(
                        f"  [HMM-AUDIT] {_hcol}: dtype={_hdtyp} | cov={_hcov:.1%} | sample={_hsamp}"
                    )
                    # ALERTA critica: filtro hmm_allowed_regimes espera strings, no numeros
                    if _hcol == "HMM_Regime" and str(_hdtyp) in ("float64", "float32", "int64", "int32"):
                        _uniq = sorted(df_oos_raw[_hcol].dropna().unique().tolist())
                        logger.warning(
                            f"  [HMM-AUDIT] CRITICO: HMM_Regime es NUMERICO ({_hdtyp}). "
                            f"El filtro isin(string_labels) bloqueara el 100%% de senales. "
                            f"Valores unicos vistos: {_uniq}. "
                            f"Verificar que hmm_regime_labels.parquet o predict_regime_series devuelven HMM_Semantic (string)."
                        )
                else:
                    if _hcol == "HMM_Regime":
                        logger.warning(f"  [HMM-AUDIT] {_hcol}: NO EXISTE en df_oos_raw tras join.")
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

            # FIX-HMM-OOS-COVERAGE-01: activar predict_regime_series() si la cobertura
            # de HMM_Regime en el holdout es < 90%.
            # Antes: solo se activaba si HMM_Regime era 100% NaN/ausente.
            # Problema: si hmm_regime_labels.parquet cubre hasta 2024 y el holdout es
            # 2025-2026, el join da valores parciales â€” el perÃ­odo 2025+ queda NaN
            # y se rellena con 0 â†’ colapso de probabilidades XGBoost â†’ 0 seÃ±ales OOS.
            _hmm_cov = (df_oos_raw["HMM_Regime"].notna().mean()
                        if "HMM_Regime" in df_oos_raw.columns else 0.0)
            if _hmm_cov < 0.90:
                if _hmm_cov > 0.0:
                    logger.warning(
                        "  [FIX-HMM-OOS-COVERAGE-01] HMM_Regime cobertura=%.1f%% < 90%% "
                        "â€” el join con hmm_regime_labels.parquet no cubre el holdout completo. "
                        "Prediciendo via predict_regime_series() para garantizar cobertura total.",
                        _hmm_cov * 100
                    )
                else:
                    logger.warning("  [FIX-HMM-OOS-COVERAGE-01] HMM_Regime ausente en holdout â€” prediciendo via predict_regime_series().")
                try:
                    from luna.models.hmm_regime import HMMRegimeModel as _HMM
                    _hmm_pred  = _HMM.load(self.models_dir)
                    _hmm_df = _hmm_pred.predict_regime_series(df_oos_raw)
                    df_oos_raw["HMM_Regime"] = _hmm_df["HMM_Regime"]
                    df_oos_raw["HMM_Semantic"] = _hmm_df["HMM_Semantic"]
                    _hmm_serie = _hmm_df["HMM_Semantic"]
                    # â”€â”€ [LOG-DIAG-HMM-02] PREDICT AUDIT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    _pdtype = _hmm_serie.dtype if hasattr(_hmm_serie, "dtype") else type(_hmm_serie)
                    _psamp  = _hmm_serie.dropna().head(3).tolist() if hasattr(_hmm_serie, "dropna") else []
                    logger.info(
                        f"  [P2-FIX] HMM_Regime predicho via HMMRegimeModel.load() | "
                        f"dtype={_pdtype} | sample={_psamp}"
                    )
                    if str(_pdtype) in ("float64", "float32", "int64", "int32"):
                        logger.warning(
                            f"  [P2-FIX] ALERTA: predict_regime_series devuelve NUMERICO ({_pdtype}). "
                            f"El filtro HMM necesita strings semanticos. "
                            f"Se traducira via state_map en el bloque de filtrado (BUG-HMM-FILTER-01 fix)."
                        )
                except Exception as _he:
                    logger.warning(f"  [P2-FIX] HMMRegimeModel.load() fallido: {_he}")

            df_oos = df_oos_raw.dropna(subset=["close"]).copy()

            # [FIX-HMM-OOS-PROPAGATION] Asegurar que HMM_Regime numerico se propaga a df_oos
            # El dropna() preserva columnas del df, pero si HMM_Regime fue asignado via
            # predict_regime_series() DESPUES de construir df_oos_raw, necesitamos alinearlo.
            if "HMM_Regime" in df_oos_raw.columns and "HMM_Regime" in available_feats:
                _hmm_reg_aligned = df_oos_raw["HMM_Regime"].reindex(df_oos.index)
                if _hmm_reg_aligned.notna().mean() > 0.5:
                    df_oos["HMM_Regime"] = pd.to_numeric(_hmm_reg_aligned, errors="coerce").fillna(0.0)
                    logger.info(
                        "  [FIX-HMM-OOS-PROPAGATION] HMM_Regime propagado: cov={:.0f}% dtype={}",
                        df_oos["HMM_Regime"].notna().mean() * 100,
                        df_oos["HMM_Regime"].dtype,
                    )
                    print(f"[BUG-FIX-LOG 2026-06-05] HMM_Regime propagado: cov={df_oos['HMM_Regime'].notna().mean() * 100:.0f}% dtype={df_oos['HMM_Regime'].dtype}")
                else:
                    df_oos["HMM_Regime"] = 0.0
                    logger.warning("  [FIX-HMM-OOS-PROPAGATION] HMM_Regime sin cobertura en drift -- fallback 0.0")
            elif "HMM_Regime" in available_feats and "HMM_Regime" not in df_oos.columns:
                df_oos["HMM_Regime"] = 0.0
                logger.warning("  [FIX-HMM-OOS-PROPAGATION] HMM_Regime ausente en df_oos -- fallback 0.0")

            # [FIX-HMM-OOS-PROPAGATION-SEMANTIC] Asegurar que HMM_Semantic se propaga a df_oos
            # Si HMM_Semantic fue asignado via predict_regime_series() DESPUES de construir df_oos_raw,
            # o si HMM_Regime es numérico y necesitamos HMM_Semantic para el filtro.
            if "HMM_Semantic" in df_oos_raw.columns:
                _hmm_sem_aligned = df_oos_raw["HMM_Semantic"].reindex(df_oos.index)
                if _hmm_sem_aligned.notna().mean() > 0.5:
                    df_oos["HMM_Semantic"] = _hmm_sem_aligned
                    logger.info(
                        "  [FIX-HMM-OOS-PROPAGATION-SEMANTIC] HMM_Semantic propagado: cov={:.0f}% dtype={}",
                        df_oos["HMM_Semantic"].notna().mean() * 100,
                        df_oos["HMM_Semantic"].dtype,
                    )
                    print(f"[BUG-FIX-LOG 2026-06-05] HMM_Semantic propagado: cov={df_oos['HMM_Semantic'].notna().mean() * 100:.0f}% dtype={df_oos['HMM_Semantic'].dtype}")
                else:
                    df_oos["HMM_Semantic"] = "UNKNOWN" # Fallback si no hay cobertura
                    logger.warning("  [FIX-HMM-OOS-PROPAGATION-SEMANTIC] HMM_Semantic sin cobertura en drift -- fallback 'UNKNOWN'")
            elif "HMM_Regime" in df_oos.columns and str(df_oos["HMM_Regime"].dtype) in ("float64", "float32", "int64", "int32"):
                # Si HMM_Semantic no existe pero HMM_Regime es numérico, crear HMM_Semantic
                try:
                    from luna.models.hmm_regime import HMMRegimeModel as _HMM_SEM_CONV
                    _hmm_ref_conv = _HMM_SEM_CONV.load(self.models_dir)
                    df_oos["HMM_Semantic"] = df_oos["HMM_Regime"].map(_hmm_ref_conv.state_map).fillna("UNKNOWN")
                    logger.info(
                        "  [FIX-HMM-OOS-PROPAGATION-SEMANTIC] HMM_Semantic creado desde HMM_Regime numérico: cov={:.0f}% dtype={}",
                        df_oos["HMM_Semantic"].notna().mean() * 100,
                        df_oos["HMM_Semantic"].dtype,
                    )
                    print(f"[BUG-FIX-LOG 2026-06-05] HMM_Semantic creado desde HMM_Regime numérico: cov={df_oos['HMM_Semantic'].notna().mean() * 100:.0f}% dtype={df_oos['HMM_Semantic'].dtype}")
                except Exception as _e_sem_conv:
                    df_oos["HMM_Semantic"] = "UNKNOWN"
                    logger.warning(f"  [FIX-HMM-OOS-PROPAGATION-SEMANTIC] Fallback 'UNKNOWN' al crear HMM_Semantic: {_e_sem_conv}")
            else:
                df_oos["HMM_Semantic"] = "UNKNOWN"
                logger.warning("  [FIX-HMM-OOS-PROPAGATION-SEMANTIC] HMM_Semantic ausente y no se pudo crear -- fallback 'UNKNOWN'")

            for feat in available_feats:
                if feat in df_oos.columns and df_oos[feat].isna().all():
                    logger.warning("  Feature 100% NaN en holdout: {} Ã¢â‚¬â€ rellenando con 0", feat)
                    df_oos[feat] = 0.0

            # Ã¢â€â‚¬Ã¢â€â‚¬ [R21] Features de Timing en df_oos (holdout) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
            # Replicar exactamente el mismo cÃƒÂ¡lculo que en train_xgboost.load_dataset()
            try:
                # BUG-GEN-01 FIX (2026-04-06): Precargar 720h de historial de train para evitar
                # NaNs y values sesgados en las primeras filas del holdout (cold-start).
                # Sin este fix, EWM(span=8) y pct_change(168) sobre SOLO holdout dan NaN en la
                # primera semana OOS, degradando el 25% de la señal en ventanas de 3 meses.
                _timing_train_tail = pd.DataFrame()
                try:
                    _timing_train_path = self.root / "data" / "features" / "features_train.parquet"
                    _timing_cols = [c for c in ["close", "FundingRate", "volume"] if c in df_oos.columns]
                    if _timing_train_path.exists() and _timing_cols:
                        _timing_train_tail = pd.read_parquet(_timing_train_path, columns=_timing_cols).iloc[-720:]
                        logger.info(f"  [BUG-GEN-01] Timing cold-start: {len(_timing_train_tail)} filas de historial precargadas")
                except Exception as _e_timing_load:
                    logger.debug(f"  [BUG-GEN-01] No se pudo precargar historial timing: {_e_timing_load}")

                def _prepend(col: str) -> "pd.Series":
                    """Concatena historial de tren + holdout, calcula, luego realinea al índice OOS."""
                    if not _timing_train_tail.empty and col in _timing_train_tail.columns:
                        return pd.concat([_timing_train_tail[col], df_oos[col]]).sort_index()
                    return df_oos[col]

                if "FundingRate" in df_oos.columns:
                    _full_fr = _prepend("FundingRate")
                    df_oos["timing_funding_acum8h"] = _full_fr.ewm(span=8, min_periods=1).mean().reindex(df_oos.index)
                if "close" in df_oos.columns:
                    _full_close_t = _prepend("close")
                    _r24h = _full_close_t.pct_change(24)
                    _r7d  = _full_close_t.pct_change(168)
                    df_oos["timing_momentum_div"] = (_r24h - _r7d).reindex(df_oos.index)
                if "close" in df_oos.columns and "volume" in df_oos.columns:
                    _full_close_v = _prepend("close")
                    _full_volume  = _prepend("volume")
                    _r24h_abs   = _full_close_v.pct_change(24).abs()
                    _vol_ma     = _full_volume.rolling(window=720, min_periods=48).mean()
                    _vol_ratio  = (_full_volume / (_vol_ma + 1e-6)).clip(lower=0.01)
                    df_oos["timing_vol_divergence"] = (_r24h_abs / (_vol_ratio + 1e-6)).clip(upper=5.0).reindex(df_oos.index)
                # M-08a: Features de posiciÃ³n en ciclo BTC (igual que train_xgboost.py)
                # ARCH-06 fix (2026-03-17): feature_pipeline.py (Paso 7B/P4-1-1) ya calcula
                # btc_cycle_position y btc_drawdown_from_ath y las guarda en features_holdout.parquet.
                # Si ya estÃ¡n presentes en df_oos â†’ usarlas directamente (evitar recÃ¡lculo).
                # Fallback: calcular si faltan (compatibilidad retroactiva con parquets antiguos).
                if "close" in df_oos.columns:
                    _cycle_present = (
                        "btc_cycle_position" in df_oos.columns
                        and df_oos["btc_cycle_position"].notna().mean() > 0.90
                    )
                    _dd_present = (
                        "btc_drawdown_from_ath" in df_oos.columns
                        and df_oos["btc_drawdown_from_ath"].notna().mean() > 0.90
                    )

                    if _cycle_present and _dd_present:
                        # âœ… ARCH-06: columnas ya calculadas por feature_pipeline.py â†’ reusar
                        logger.info(
                            "  [ARCH-06] btc_cycle_position y btc_drawdown_from_ath ya presentes "
                            "en holdout parquet â€” reutilizando (no recalculadas). "
                            "cycle: min=%.3f max=%.3f NaN=%d",
                            df_oos["btc_cycle_position"].min(),
                            df_oos["btc_cycle_position"].max(),
                            df_oos["btc_cycle_position"].isna().sum()
                        )
                    else:
                        # Fallback: calcular desde histÃ³rico (parquets generados antes de ARCH-06)
                        logger.info(
                            "  [ARCH-06 FALLBACK] btc_cycle_position o btc_drawdown_from_ath "
                            "no en holdout parquet â€” calculando desde histÃ³rico "
                            "(regenerar pipeline para eliminar este fallback)."
                        )
                        # P2-7-FIX: usar _calc_btc_drawdown_from_ath() con historia de train prepend.
                        # Antes: rolling solo sobre holdout → ATH≈precio en primeras filas → drawdown≈0.
                        if not _dd_present:
                            df_oos["btc_drawdown_from_ath"] = self._calc_btc_drawdown_from_ath(
                                df_oos["close"], self.root, window_h=90*24
                            )
                        # btc_cycle_position: vÃ­a funciÃ³n centralizada (BUG-05 fix)
                        if not _cycle_present:
                            df_oos["btc_cycle_position"] = self._calc_btc_cycle_position(
                                df_oos["close"], self.root
                            )
                            logger.info(
                                "  [LEGACY-04/BUG-05] btc_cycle_position calculada (fallback): "
                                "min=%.3f max=%.3f NaN=%d",
                                df_oos["btc_cycle_position"].min(),
                                df_oos["btc_cycle_position"].max(),
                                df_oos["btc_cycle_position"].isna().sum()
                            )
                logger.info("  [R21+M08a] Timing + ciclo BTC procesadas en holdout OOS (ARCH-06)")
            except Exception as _e_rt:
                logger.warning("  [R21] Error calculando timing features en holdout: {}", _e_rt)
            # Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


            # Validación cruzada con settings.yaml: el holdout debe empezar en holdout_start
            # LOG-BUG-04: fechas fluyen settings.yaml → pipeline features → parquet → aquí
            try:
                from config.settings import cfg as _cfg_splits
                holdout_start_cfg = pd.Timestamp(_cfg_splits.temporal_splits.holdout_start, tz="UTC")
                holdout_start_data = df_oos.index.min()
                if holdout_start_data < holdout_start_cfg:
                    logger.warning(
                        "⚠️  LOG-BUG-04 ALERTA: features_holdout.parquet empieza en %s "
                        "pero settings.yaml.holdout_start=%s. "
                        "El parquet puede incluir datos del período de validación. "
                        "Re-generar el pipeline de features.",
                        holdout_start_data.date(), holdout_start_cfg.date()
                    )
                else:
                    logger.info(f"  ✅ Fechas validadas: holdout desde {holdout_start_data.date()} (settings: {holdout_start_cfg.date()})")
            except Exception as _e:
                logger.debug(f"  Validación holdout_start no disponible: {_e}")

            logger.info(f"  Período OOS holdout: {len(df_oos)} filas ({df_oos.index.min()} → {df_oos.index.max()})")


        elif val_features_path.exists():
            logger.warning("⚠️  LOG-BUG-04 FALLBACK: features_holdout.parquet no encontrado.")
            logger.warning("    Usando features_validation.parquet (2024 — semi-conocido). "
                           "Regenerar el pipeline para obtener holdout 2025.")
            df_oos_raw = pd.read_parquet(val_features_path)
            if "close" not in df_oos_raw.columns:
                logger.error("features_validation.parquet no tiene columna 'close'. Abortando.")
                return False

            # Integrar HMM labels si existen
            if hmm_path.exists():
                df_hmm_val = pd.read_parquet(hmm_path)
                df_oos_raw = df_oos_raw.join(df_hmm_val, how="left")

            # Guard: features 100% NaN en validation → rellenar con 0
            df_oos = df_oos_raw.dropna(subset=["close"]).copy()
            for feat in available_feats:
                if feat in df_oos.columns and df_oos[feat].isna().all():
                    logger.warning(f"  Feature 100%% NaN en validation: {feat} — rellenando con 0")
                    df_oos[feat] = 0.0

            # LOGIC-GEN-01 FIX (2026-04-06): Filtrar por holdout_start para evitar contaminar
            # trades con datos de periodos ya vistos por el modelo (anti-leakage defensivo).
            try:
                from config.settings import cfg as _cfg_val_cut
                _val_hstart_str = getattr(_cfg_val_cut.temporal_splits, 'holdout_start', None)
                if _val_hstart_str:
                    _val_hstart = pd.Timestamp(_val_hstart_str, tz="UTC")
                    _before = len(df_oos)
                    df_oos = df_oos[df_oos.index >= _val_hstart]
                    logger.warning(
                        "  [LOGIC-GEN-01] Validation fallback filtrado por holdout_start=%s: %d → %d filas",
                        _val_hstart_str, _before, len(df_oos)
                    )
            except Exception as _e_vcut:
                logger.debug(f"  [LOGIC-GEN-01] No se pudo aplicar corte holdout_start en fallback: {_e_vcut}")

            _oos_start = df_oos.index.min() if len(df_oos) > 0 else 'N/A'
            _oos_end   = df_oos.index.max() if len(df_oos) > 0 else 'N/A'
            logger.info(f"  Período OOS validation: {len(df_oos)} filas ({_oos_start} → {_oos_end})")

        else:
            logger.warning("⚠️  features_holdout.parquet y features_validation.parquet no encontrados.")
            logger.warning("    Fallback a split 20%% sobre features_train.parquet.")
            logger.warning("    Esto produce OOS sesgado al período mÃƒÂ¡s alcista (HOLD-02).")
            # ── Fallback: ÃƒÂºltimo 20% purgado (comportamiento v1.1) ─────────────
            total_rows = len(df_clean)
            # [FIX-03] purge_rows calculado dinamicamente (antes: 336 = 14d x 24H hardcodeado, asume velas 1H)
            # Ahora: int(vertical_barrier_hours * 1.5) proporcional al horizonte real del TBM
            try:
                from config.settings import cfg as _cfg_pr
                _vbh_pr = int(_cfg_pr.xgboost.vertical_barrier_hours)
            except Exception as e_vbh:
                raise RuntimeError(f"Falta vertical_barrier_hours en settings.yaml (SOP No-Fallback): {e_vbh}")
            purge_rows = int(_vbh_pr * 1.5)
            print('[FIX-03] Fallback OOS split: purge_rows=' + str(purge_rows) + ' filas (vbh=' + str(_vbh_pr) + 'H x1.5, antes hardcode 336)')
            # [FIX-SPLIT-PRED-01] Hacer ratio de split temporal train/val de fallback configurable
            try:
                from config.settings import cfg as _cfg_sp
                _val_ratio = float(getattr(_cfg_sp.metalabeler, 'val_split_ratio', 0.20))
                _train_ratio = 1.0 - _val_ratio
                print(f"[FIX-SPLIT-PRED-01] Fallback split temporal cargado: train_ratio={_train_ratio:.2f} (val_ratio={_val_ratio:.2f})")
            except Exception as _e_pred_sp:
                _train_ratio = 0.80
                print(f"[FIX-SPLIT-PRED-01] WARN: No se pudo leer metalabeler.val_split_ratio ({_e_pred_sp}). Usando fallback train_ratio={_train_ratio:.2f}")
            
            split_idx  = int(total_rows * _train_ratio) + purge_rows

            if split_idx >= total_rows:
                logger.error(f"Dataset insuficiente para OOS: purge_rows={purge_rows} filas (vbh_pr x1.5). [FIX-03]")  # antes: 336H hardcode
                return False

            df_oos = df_clean.iloc[split_idx:].copy()
            logger.info(f"  Periodo OOS (fallback split): {len(df_oos)} filas -> {df_oos.index.min()}")

        # ── Cargar y aplicar XGBoost y LightGBM ──
        # P3-V3-4: Uso de importlib.reload para asegurar leer el yaml de esta ventana WFB
        try:
            import importlib
            import config.settings as _cs_oos
            importlib.reload(_cs_oos)
            _cfg_xgb = _cs_oos.cfg
            use_regime = getattr(_cfg_xgb.fase2, 'use_regime_agents', False)
            use_lgbm2  = getattr(_cfg_xgb.fase2, 'use_lgbm_ensemble', False)
        except Exception:
            use_regime = False
            use_lgbm2  = False

        # [PIPELINE-INTEGRITY] Pre-window check: verificar artefactos antes de inferencia.
        # Detecta FIX-CALIB-BINARY-01, CAL_MISSING, MODEL_MISSING antes de que fallen
        # silenciosamente y contaminen todos los trades de la ventana.
        if use_regime:
            try:
                from luna.pipeline_integrity import PipelineIntegrityChecker as _PIC_pre
                _window_label_pre = getattr(self, '_window_id', 'UNKNOWN')
                _direction_pre    = getattr(self, 'direction', 'long')
                _PIC_pre.pre_window_check(_window_label_pre, self.models_dir, _direction_pre)
            except RuntimeError:
                raise  # FIX-CALIB-BINARY-01 detectado: propagar para detener la ventana
            except Exception as _e_pre:
                print(f"[PIPELINE-INTEGRITY] pre_window_check error (no critico): {_e_pre}")

        # Guard btc_cycle_position (aplica para ambos modos)
        if ("btc_cycle_position" in available_feats or use_regime) and "btc_cycle_position" not in df_oos.columns:
            logger.warning("  [LEGACY-04 GUARD] btc_cycle_position ausente tras bloque R21 — calculando via función centralizada (BUG-05)")
            df_oos["btc_cycle_position"] = self._calc_btc_cycle_position(df_oos["close"], self.root)
            _bcp = df_oos["btc_cycle_position"]
            logger.info(
                f"  [LEGACY-04 GUARD] btc_cycle_position: "
                f"min={_bcp.min():.3f} max={_bcp.max():.3f} NaN={_bcp.isna().sum()}"
            )

        # Guard HMM_Regime numeric coercion (reutilizable)
        if ("HMM_Regime" in available_feats or use_regime):
            if "HMM_Regime" not in df_oos.columns:
                try:
                    from luna.models.hmm_regime import HMMRegimeModel as _HMM2
                    _hmm_ref = _HMM2.load(self.models_dir)
                    _hmm2_df = _hmm_ref.predict_regime_series(df_oos)
                    df_oos["HMM_Regime"] = _hmm_ref.coerce_regime_numeric(_hmm2_df["HMM_Regime"])
                except Exception:
                    df_oos["HMM_Regime"] = 0.0
            else:
                try:
                    from luna.models.hmm_regime import HMMRegimeModel as _HMM3
                    _hmm_ref = _HMM3.load(self.models_dir)
                    df_oos["HMM_Regime"] = _hmm_ref.coerce_regime_numeric(df_oos["HMM_Regime"])
                except Exception:
                    df_oos["HMM_Regime"] = pd.to_numeric(df_oos["HMM_Regime"], errors="coerce").fillna(0.0)

        df_oos_base = df_oos.copy()
        all_xgb_baseline_records = []
        try:
            from config.settings import cfg as _cfg_dir
            _dmode = getattr(_cfg_dir.fase2, 'direction_mode', 'both')
            if _dmode == "both":
                directions_to_run = ["long", "short"]
            elif isinstance(_dmode, list):
                directions_to_run = _dmode
            else:
                directions_to_run = [_dmode]
        except Exception:
            directions_to_run = ["long", "short"] if use_regime else ["long"]
        all_trade_records = []

        # [V2-P3-DRIFT] PSI Monitor — Covariate Shift Detection
        # Llamar UNA VEZ sobre df_oos_base, ANTES del loop de inferencia.
        # Si hay >= 5 features con PSI > 0.25, se reduce el Kelly max_position al 50%.
        _psi_kelly_penalty = 1.0
        try:
            from luna.monitoring.feature_drift_monitor import run_drift_monitor
            _train_feat_path = self.data_dir / "features" / "features_train.parquet"
            if _train_feat_path.exists() and len(df_oos_base) > 0:
                # [FIX-NEW-06] Evitar doble lectura I/O y carga completa en RAM (~500MB).
                # Usar pyarrow para leer schema instantáneamente y luego cargar solo common_cols.
                import pyarrow.parquet as pq
                _schema_cols = pq.read_schema(_train_feat_path).names
                _common_cols = [c for c in available_feats if c in _schema_cols and c in df_oos_base.columns]
                if _common_cols:
                    _train_ref = pd.read_parquet(_train_feat_path, columns=_common_cols)
                    _train_ref_slim = _train_ref[_common_cols]
                    _oos_slim = df_oos_base[_common_cols]
                    _drift_result = run_drift_monitor(_train_ref_slim, _oos_slim, feature_cols=_common_cols)
                    _psi_kelly_penalty = _drift_result.get("kelly_penalty", 1.0)
                    if _psi_kelly_penalty < 1.0:
                        logger.warning(
                            "  [V2-P3-DRIFT] Kelly penalty={:.0f}% activo: {} features en drift CRÍTICO. "
                            "max_position se reduce proporcionalmente en todos los trades.",
                            _psi_kelly_penalty * 100, _drift_result.get("n_drifted", 0)
                        )
                        print(f"[BUG-FIX-LOG 2026-06-05] Kelly penalty={_psi_kelly_penalty * 100:.0f}% activo: {_drift_result.get('n_drifted', 0)} features en drift CRÍTICO.")
                else:
                    logger.info("  [V2-P3-DRIFT] Sin features comunes para PSI — omitiendo drift monitor.")
            else:
                logger.info("  [V2-P3-DRIFT] features_train.parquet no disponible — PSI omitido.")
        except Exception as _e_psi:
            logger.warning("  [V2-P3-DRIFT] PSI Monitor falló (no crítico): {}", _e_psi)

        for _direct in directions_to_run:
            df_oos = df_oos_base.copy()
            _pred_drift_penalty = 1.0

            # [DEGRADED-MODE] Leer agentes deshabilitados por Gate-G2 (si aplica)
            _g2_disabled_path = self.models_dir / "gate_g2_disabled_agents.json"
            _g2_disabled_agents: list = []
            if _g2_disabled_path.exists():
                try:
                    import json as _json_g2r
                    _g2_info = _json_g2r.loads(_g2_disabled_path.read_text(encoding="utf-8"))
                    _g2_disabled_agents = _g2_info.get("disabled_agents", [])
                    if _g2_disabled_agents:
                        logger.warning(
                            "[OOS-PREDS/DEGRADED] Agentes NO_OPERABLE leídos de Gate-G2: {} "
                            "→ RegimeRouter forzará CASH en esos regímenes.",
                            _g2_disabled_agents
                        )
                except Exception as _e_g2r:
                    logger.warning("[OOS-PREDS] Error leyendo gate_g2_disabled_agents.json: {}", _e_g2r)

            # ── [FIX-BULL-GATE-01 2026-06-01] Gate de calidad DSR para agente bull_long ──
            # MOTIVACION: WR < 50% en 48 ventanas historicas consecutivas (2 semanas, N seeds).
            # bull_long aprende media-reversion en IS BULL pero OOS BULL 2025 tiene pullbacks
            # mas profundos → SL hit antes que TP → perdidas sistematicas sin excepcion.
            # ACCION: Si DSR_CPCV_best <= bull_gate_min_dsr → aniadir 'bull' a disabled_agents.
            # El RegimeRouter ya tiene logica para forzar prob=0.0 en regimenes deshabilitados
            # → 0 trades bull_long esta ventana. Reutiliza infraestructura existente (no chapuza).
            # PARAMETRO: xgboost.bull_gate_min_dsr en settings.yaml (default=0.0, alineado R5 SOP).
            if _direct == "long" and "bull" not in _g2_disabled_agents:
                _bull_sig_path = self.models_dir / "xgboost_meta_bull_long_signature.json"
                if _bull_sig_path.exists():
                    try:
                        import json as _json_bull_gate
                        _bull_sig = _json_bull_gate.loads(_bull_sig_path.read_text(encoding="utf-8"))
                        # Usar dsr_cpcv_best (canonico FIX-DSR-MASK-01), fallback a dsr_oos
                        _bull_dsr = float(_bull_sig.get("dsr_cpcv_best", _bull_sig.get("dsr_oos", 0.0)))
                        # Leer umbral desde settings — No-Fallback silencioso: 0.0 es el default R5 explicito
                        try:
                            _min_bull_dsr = float(_cfg_xgb.xgboost.bull_gate_min_dsr)
                        except Exception as e_bull:
                            raise RuntimeError(f"Falta bull_gate_min_dsr en settings.yaml (SOP No-Fallback): {e_bull}")
                        if _bull_dsr <= _min_bull_dsr:
                            _g2_disabled_agents = list(_g2_disabled_agents) + ["bull"]
                            _bull_gate_msg = (
                                f"[FIX-BULL-GATE-01/SKIP] Agente 'bull_long': "
                                f"DSR_CPCV={_bull_dsr:.4f} <= {_min_bull_dsr:.4f} (bull_gate_min_dsr). "
                                f"→ bull DESACTIVADO esta ventana | 0 trades bull_long. "
                                f"Evidencia: WR<50% en 48 ventanas historicas sin excepcion. "
                                f"RegimeRouter asignara prob=0.0 a todos los bares BULL."
                            )
                            print(_bull_gate_msg)  # RULE[fixbugsprints.md]
                            logger.warning(_bull_gate_msg)
                        else:
                            print(
                                f"[FIX-BULL-GATE-01/PASS] bull_long DSR_CPCV={_bull_dsr:.4f} "
                                f"> {_min_bull_dsr:.4f} → bull_long ACTIVO esta ventana."
                            )
                    except Exception as _e_bull_gate:
                        print(f"[FIX-BULL-GATE-01/WARN] Error leyendo bull signature: {_e_bull_gate}. bull_long ACTIVO por defecto.")
                else:
                    print(
                        f"[FIX-BULL-GATE-01/INFO] xgboost_meta_bull_long_signature.json no encontrada "
                        f"en {self.models_dir}. bull_long ACTIVO (primera ventana o cache vacio)."
                    )

            # 1) INFERENCIA XGBOOST
            if use_regime:
                from luna.models.regime_router import RegimeRouter
                router_xgb = RegimeRouter(
                    self.models_dir,
                    agent_type="xgboost",
                    direction=_direct,
                    disabled_regimes=_g2_disabled_agents,
                )
                logger.info(f"Generando predicciones XGBoost (Multi-Agent OOS) sobre {len(df_oos)} filas...")
                xgb_probs_df = router_xgb.route_and_predict(df_oos)
                df_oos["xgb_prob"] = xgb_probs_df["raw"]
                df_oos["xgb_prob_cal"] = xgb_probs_df["calibrated"]

                # [FIX-PRED-DRIFT-SENTINEL 2026-06-13] Prediction Drift Sentinel (OOD Circuit Breaker)
                # Compara las predicciones calibradas de Holdout contra las de Validación.
                try:
                    import os as _os_sentinel
                    _window_id_s = _os_sentinel.environ.get("LUNA_WINDOW_ID", "UNK")
                    # Rutas para buscar la validación in-sample
                    _val_feat_path = self.root / "data" / "wfb_cache" / _window_id_s / "features" / f"features_validation_{_window_id_s}.parquet"
                    if not _val_feat_path.exists():
                        _val_feat_path = self.root / "data" / "wfb_cache" / _window_id_s / "features" / "features_validation.parquet"
                    
                    if _val_feat_path.exists():
                        _df_val_s = pd.read_parquet(_val_feat_path)
                        # Predecir con el mismo router para validación in-sample
                        _xgb_val_s = router_xgb.route_and_predict(_df_val_s)
                        _val_cal = _xgb_val_s["calibrated"].fillna(0.5).values
                        _hold_cal = df_oos["xgb_prob_cal"].fillna(0.5).values
                        
                        if len(_val_cal) > 0 and len(_hold_cal) > 0:
                            _num_buckets = 10
                            # [FIX-PRED-DRIFT-SENTINEL-FIX-PSI 2026-06-13] Usar binning uniforme de ancho fijo entre [0.0, 1.0]
                            # Previene el colapso del PSI causado por percentiles repetidos de IsotonicRegression
                            _buckets = np.linspace(0.0, 1.0, _num_buckets + 1)
                            if len(_buckets) >= 2:
                                _buckets[0] = -np.inf
                                _buckets[-1] = np.inf
                                _val_counts = np.histogram(_val_cal, bins=_buckets)[0]
                                _hold_counts = np.histogram(_hold_cal, bins=_buckets)[0]
                                _val_pct = _val_counts / len(_val_cal)
                                _hold_pct = _hold_counts / len(_hold_cal)
                                _val_pct = np.where(_val_pct == 0, 1e-4, _val_pct)
                                _hold_pct = np.where(_hold_pct == 0, 1e-4, _hold_pct)
                                _pred_psi = np.sum((_hold_pct - _val_pct) * np.log(_hold_pct / _val_pct))
                                
                                # [FIX-PRED-DRIFT-SENTINEL 2026-06-13] Carga dinámica de parámetros desde settings.yaml (No-Fallback)
                                _pred_min_psi = 0.08
                                _pred_max_psi = 0.20
                                if _cfg_xgb is not None:
                                    try:
                                        _pred_min_psi = float(getattr(_cfg_xgb.wfb, "pred_drift_min_psi", 0.08))
                                        _pred_max_psi = float(getattr(_cfg_xgb.wfb, "pred_drift_max_psi", 0.20))
                                    except Exception as _e_cfg:
                                        pass
                                
                                if _pred_psi >= _pred_max_psi:
                                    _pred_drift_penalty = 0.0
                                elif _pred_psi >= _pred_min_psi:
                                    _pred_drift_penalty = (_pred_max_psi - _pred_psi) / (_pred_max_psi - _pred_min_psi)
                                    
                                print(f"[FIX-PRED-DRIFT-SENTINEL] Window {_window_id_s} | Dirección {_direct} | Preds PSI = {_pred_psi:.4f} | Penalty = {_pred_drift_penalty:.1%} | Bounds=[{_pred_min_psi:.2f}, {_pred_max_psi:.2f}]")
                                logger.info(f"[FIX-PRED-DRIFT-SENTINEL] Window {_window_id_s} | Dirección {_direct} | Preds PSI = {_pred_psi:.4f} | Penalty = {_pred_drift_penalty:.1%} | Bounds=[{_pred_min_psi:.2f}, {_pred_max_psi:.2f}]")
                except Exception as _e_sent:
                    print(f"[FIX-PRED-DRIFT-SENTINEL] Error en Sentinel (no critico): {_e_sent}")
                    logger.debug(f"[FIX-PRED-DRIFT-SENTINEL] Error en Sentinel: {_e_sent}")

                # [FIX-CALIB-BINARY-01 DETECTION-3] Guard post-asignacion en predict_oos.
                # Verifica que xgb_prob_cal difiere de xgb_prob_raw antes de continuar.
                # Si son identicos con calibradores cargados -> bug activo -> CRITICAL.
                try:
                    _n_total_oos = len(df_oos)
                    _diff_oos    = (df_oos["xgb_prob_cal"].fillna(df_oos["xgb_prob"]) - df_oos["xgb_prob"]).abs()
                    _n_mod_oos   = (_diff_oos > 1e-6).sum()
                    _pct_mod_oos = _n_mod_oos / max(_n_total_oos, 1) * 100
                    _n_cals_ok   = len(router_xgb.isotonic_calibrators)
                    print(
                        f"[FIX-CALIB-BINARY-01/DETECTION-3] predict_oos calibracion audit | "
                        f"calibradores_cargados={_n_cals_ok} | "
                        f"barras_modificadas={_n_mod_oos}/{_n_total_oos} ({_pct_mod_oos:.1f}%) | "
                        f"diff_mean={float(_diff_oos.mean()):.4f} | "
                        f"{'OK' if _pct_mod_oos > 0.5 or _n_cals_ok == 0 else '*** ALERTA: cal==raw con calibradores cargados ***'}"
                    )
                    if _pct_mod_oos < 0.5 and _n_cals_ok > 0 and _n_total_oos > 100:
                        logger.critical(
                            "[FIX-CALIB-BINARY-01/DETECTION-3] CRITICAL: %d calibradores cargados "
                            "pero solo %.1f%% de barras tienen cal!=raw (diff_mean=%.6f). "
                            "Todos los trades OOS con xgb_prob_cal == xgb_prob_raw. "
                            "WR degradado vs historico. Causa: probs OOS fuera del rango del calibrador "
                            "(out_of_bounds clip) O apertura binaria incorrecta del .joblib.",
                            _n_cals_ok, _pct_mod_oos, float(_diff_oos.mean())
                        )
                except Exception as _e_det3:
                    print(f"[FIX-CALIB-BINARY-01/DETECTION-3] Error en audit (no critico): {_e_det3}")
            else:
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(xgb_model_path)
                logger.info("Generando predicciones XGBoost tradicional sobre OOS...")
                X_oos = df_oos[available_feats]

                # [OPT-INFERENCE] Force CPU device to bypass DMatrix PCIe transfer bottleneck in XGBoost 3.x
                try:
                    if hasattr(model, "set_params"):
                        model.set_params(device="cpu")
                except Exception:
                    pass

                xgb_probs = model.predict_proba(X_oos)[:, 1]
                df_oos["xgb_prob"] = xgb_probs

            # 2) INFERENCIA LIGHTGBM (P1-V3-1 FIX: Fuera de condicional use_regime XGBoost)
            # P3-V3-5 FIX: Discriminar adecuadamente el modelo a cargar para LGBM
            if use_lgbm2:
                if use_regime:
                    from luna.models.regime_router import RegimeRouter
                    router_lgbm = RegimeRouter(
                        self.models_dir,
                        agent_type="lightgbm",
                        direction=_direct,
                        disabled_regimes=_g2_disabled_agents,
                    )
                    logger.info(f"Generando predicciones LightGBM (Multi-Agent OOS) sobre {len(df_oos)} filas...")
                    lgbm_probs_df = router_lgbm.route_and_predict(df_oos)
                    df_oos["lgbm_prob"] = lgbm_probs_df["raw"]

                    # BUG-LGBM-SIG-01 FIX (2026-04-08): generar lgbm_meta_signature.json global.
                    # signal_filter.apply_model_threshold() busca '{prefix}_signature.json' (es decir,
                    # 'lgbm_meta_signature.json'). Este archivo NUNCA se creaba porque ensemble_lgbm.py
                    # guarda solo archivos individuales por régimen (lgbm_meta_bull_signature.json, etc.).
                    # Sin el archivo global, apply_model_threshold() cae al fallback de settings.yaml (0.51)
                    # en lugar de usar los thresholds calibrados por Optuna por régimen.
                    # Fix: consolidar las firmas individuales en un único lgbm_meta_signature.json con
                    # optimal_threshold_per_regime mapeado a IDs numéricos del HMM.
                    try:
                        from config.settings import cfg as _cfg_lgbm_sig
                        _rm = vars(_cfg_lgbm_sig.fase2.regime_mapping)  # {bull: [...], range: [...], bear: [...]}
                    except Exception:
                        # [SOL3-CALM-BEAR-01 2026-06-01] Fallback LGBM con calm_bear dedicado
                        _rm = {
                            "bull":      ["1_BULL_TREND", "1_VOLATILE_BULL", "1_BULL_GRIND", "1_BULL_TREND_WEAK", "1_BULL_TREND_B", "1_VOLATILE_BULL_B"],
                            "range":     ["2_CALM_RANGE", "2_VOLATILE_RANGE", "2_CALM_RANGE_B", "2_VOLATILE_RANGE_B"],
                            "calm_bear": ["3_CALM_BEAR", "3_CALM_BEAR_B", "3_CALM_BEAR_C", "3_CALM_BEAR_D"],
                            "bear":      ["3_BEAR_CRASH", "3_BEAR_CRASH_B", "4_BEAR_FORCED"],
                        }
                        print("[SOL3-CALM-BEAR-01/LGBM-FALLBACK] Usando fallback interno con calm_bear.")

                    # Mapa semántico→numérico del HMM (extraído del parquet si existe)
                    _sem_to_num = {}
                    try:
                        _hmm_parquet = self.root / "data" / "features" / "hmm_regime_labels.parquet"
                        if _hmm_parquet.exists():
                            _hmm_df = pd.read_parquet(_hmm_parquet, columns=["HMM_Regime", "HMM_Semantic"])
                            for _, _row in _hmm_df.drop_duplicates().iterrows():
                                _sem_to_num[str(_row["HMM_Semantic"])] = int(float(_row["HMM_Regime"]))
                    except Exception as _e_sem:
                        logger.debug("  [BUG-LGBM-SIG-01] No se pudo construir mapa semántico→numérico: {}", _e_sem)

                    _lgbm_per_regime_thresh = {}
                    _lgbm_thresholds_found = []
                    for _agent_name, _sem_list in _rm.items():
                        _agent_sig_path = self.models_dir / f"lgbm_meta_{_agent_name}_{_direct}_signature.json"
                        if not _agent_sig_path.exists():
                            # Fallback a nombre legacy sin direccion
                            _agent_sig_path = self.models_dir / f"lgbm_meta_{_agent_name}_signature.json"
                            
                        if _agent_sig_path.exists():
                            try:
                                _agent_sig = json.loads(_agent_sig_path.read_text(encoding="utf-8"))
                                _thr = float(_agent_sig.get("optimal_threshold", 0.51))
                                _lgbm_thresholds_found.append(_thr)
                                # Mapear cada régimen semántico de este agente a su threshold
                                for _sem in _sem_list:
                                    _num_id = _sem_to_num.get(_sem)
                                    if _num_id is not None:
                                        _lgbm_per_regime_thresh[str(_num_id)] = _thr
                            except Exception as _e_sig:
                                logger.debug("  [BUG-LGBM-SIG-01] Error leyendo firma {}: {}", _agent_sig_path.name, _e_sig)

                    # [P1-7-FIX] El threshold global debe ser la mediana (o media ponderada) para
                    # no exponerse al threshold más permisivo cuando un régimen nuevo aparece.
                    # Usar min() es peligroso porque si un régimen calmo tiene thresh=0.48 y aparece 
                    # un flash crash, por defecto aplicaremos 0.48 a pesar de desconocerlo.
                    _global_lgbm_thresh = np.median(_lgbm_thresholds_found) if _lgbm_thresholds_found else 0.51
                    _lgbm_global_sig = {
                        "optimal_threshold": _global_lgbm_thresh,
                        "optimal_threshold_per_regime": _lgbm_per_regime_thresh,
                        "source": "consolidado-desde-firmas-por-regimen (BUG-LGBM-SIG-01-FIX)",
                        "n_regimes": len(_lgbm_per_regime_thresh),
                    }
                    _lgbm_sig_out = self.models_dir / "lgbm_meta_signature.json"
                    _lgbm_sig_out.write_text(json.dumps(_lgbm_global_sig, indent=4), encoding="utf-8")
                    logger.info(
                        "  [BUG-LGBM-SIG-01-FIX] lgbm_meta_signature.json generado: "
                        "global_thresh={:.3f} | regimes mapeados={} | {}",
                        _global_lgbm_thresh, len(_lgbm_per_regime_thresh), _lgbm_per_regime_thresh
                    )

                else:
                    import joblib
                    _lgbm_meta_path = self.models_dir / "lgbm_meta.model"
                    _lgbm_sig_path = self.models_dir / "lgbm_meta_signature.json"
                    if _lgbm_meta_path.exists() and _lgbm_sig_path.exists():
                        logger.info("Generando predicciones LightGBM tradicional sobre OOS...")
                        with open(_lgbm_sig_path, 'r') as _f:
                            _lgbm_sig = json.load(_f)
                            _lgbm_feats = _lgbm_sig.get("features", available_feats)
                        _lgbm_model = joblib.load(_lgbm_meta_path)
                    
                        # Garantizar que las columnas de features esten (LightGBM no tolera features faltantes)
                        _X_lgbm = df_oos.copy()
                        for _f_lgbm in _lgbm_feats:
                            if _f_lgbm not in _X_lgbm.columns:
                                _X_lgbm[_f_lgbm] = 0.0
                        _X_lgbm = _X_lgbm[_lgbm_feats]
                    
                        lgbm_probs = _lgbm_model.predict_proba(_X_lgbm)[:, 1]
                        # [BUG-LGBM-YPRO-01 FIX] Detectar raw log-odds (Focal Loss) y aplicar sigmoide
                        if lgbm_probs.min() < 0 or lgbm_probs.max() > 1.0:
                            from scipy.special import expit
                            lgbm_probs = expit(lgbm_probs)
                        df_oos["lgbm_prob"] = lgbm_probs
                    else:
                        logger.warning("use_lgbm_ensemble=True, pero lgbm_meta.model o signature no existen. Omitiendo LGBM.")

            # ── Pipeline Iterativo de Filtros (SignalFilter) ──
            from luna.models.signal_filter import SignalFilter
            signal_pipeline = SignalFilter(self.models_dir)
        
            signal_mask = signal_pipeline.filter_signals(df_oos, available_feats, direction=_direct)
            n_signals = int(signal_mask.sum())
        
            if n_signals == 0:
                logger.warning(f"0 señales iniciales tras filtros para la dirección {_direct}.")
                continue

            # [POOL-SAVE-01] Guardar pool completo PRE-EMBARGO para BOTAU bidireccional.
            # Contiene todas las señales que pasaron XGB/MetaLabeler/HMM/Momentum,
            # con meta_v2_prob, xgb_prob, lgbm_prob — antes del filtro de embargo.
            # Permite testear BOTAU con pool sin re-run completo.
            # Se activa con env var LUNA_SAVE_SIGNAL_POOL=1
            # [FIX-POOL-SEED-01] Nombre incluye seed para evitar que seeds distintas
            # sobreescriban el pool de las anteriores.
            try:
                import os as _os_pool
                if _os_pool.environ.get("LUNA_SAVE_SIGNAL_POOL", "0") == "1":
                    _pool_df = df_oos[signal_mask].copy()
                    _pool_cols = [c for c in ["close", "xgb_prob", "lgbm_prob", "meta_v2_prob",
                                              "HMM_Regime", "HMM_Semantic"] if c in _pool_df.columns]
                    _pool_df = _pool_df[_pool_cols]
                    _wid_pool = _os_pool.environ.get("LUNA_WINDOW_ID", "UNK")
                    # Extraer seed de settings.yaml para aislar pools por seed
                    try:
                        import yaml as _yaml_pool
                        _cfg_pool = _yaml_pool.safe_load(
                            (self.root / "config" / "settings.yaml").read_text(encoding="utf-8")
                        )
                        _seed_pool = _cfg_pool.get("xgboost", {}).get("optuna_seed", "UNK")
                    except Exception:
                        _seed_pool = "UNK"
                    _pool_fname = f"signal_pool_{_wid_pool}_seed{_seed_pool}_{_direct}.parquet"
                    _pool_path = self.root / "data" / "predictions" / _pool_fname
                    _pool_df.to_parquet(_pool_path)
                    logger.success(
                        "[POOL-SAVE-01] Pool pre-embargo guardado: {} señales → {}",
                        len(_pool_df), _pool_path.name
                    )
            except Exception as _e_pool:
                logger.debug(f"[POOL-SAVE-01] No se pudo guardar pool: {_e_pool}")

            signal_times = signal_pipeline.apply_embargo(df_oos, signal_mask)
            n_signals = len(signal_times)
        
            if n_signals == 0:
                logger.warning(f"Sin señales tras aplicar embargo para la dirección {_direct}.")
                continue
            
            signal_pipeline.export_funnel_json(self.root / "data" / "reports")

            # [FIX-P1A-FUNNEL-SEED 2026-05-28] Copiar signal_funnel.json con key por seed
            # para que run_statistical_validation.py pueda leer el funnel acumulado correcto.
            # Problema anterior: LUNA_RUN_ID cambia entre ventanas de la misma seed (incluye timestamp)
            # → el acumulador FIX-FUNNEL-ACCUM-01 se resetea en cada ventana porque ve run_id distinto.
            # Fix: usar una clave estable seed-based (LUNA_SEED_KEY = seed_{optuna_seed}) como LUNA_RUN_ID
            # para la acumulación, independientemente de cuándo se lanzó el run.
            import os as _os_funnel_fix
            import shutil as _shutil_fix
            import json as _json_fix

            _run_id_env  = _os_funnel_fix.environ.get("LUNA_RUN_ID", "")
            _window_env  = _os_funnel_fix.environ.get("LUNA_WINDOW_ID", "UNK")

            # Leer seed desde settings para construir key estable
            try:
                from config.settings import cfg as _cfg_fnl
                _optuna_seed_fnl = int(getattr(_cfg_fnl.xgboost, 'optuna_seed', 0))
            except Exception:
                _optuna_seed_fnl = 0

            # key de acumulacion: incluye seed pero NO timestamp → estable entre ventanas
            _seed_funnel_key = f"seed{_optuna_seed_fnl}"

            # Si LUNA_RUN_ID aún no tiene el seed, construimos uno estable para esta seed
            if not _run_id_env or f"seed{_optuna_seed_fnl}" not in _run_id_env:
                _stable_run_id = _run_id_env or f"WFB_standalone_{_seed_funnel_key}"
                # No sobreescribir si ya hay un run_id bien formado con esta seed
                _os_funnel_fix.environ["LUNA_RUN_ID"] = _stable_run_id
                print(f"[FIX-P1A-FUNNEL-SEED] LUNA_RUN_ID seteado para acumulación funnel: {_stable_run_id}")
            else:
                _stable_run_id = _run_id_env

            # Crear copia con nombre por seed para que run_statistical_validation.py lo encuentre
            # Formato: signal_funnel_{run_id}.json (L516 de run_statistical_validation.py)
            try:
                _funnel_src  = self.root / "data" / "reports" / "signal_funnel.json"
                _funnel_dest = self.root / "data" / "reports" / f"signal_funnel_{_stable_run_id}.json"
                if _funnel_src.exists():
                    _shutil_fix.copy(_funnel_src, _funnel_dest)
                    # Verificar acumulación
                    with open(_funnel_dest, encoding="utf-8") as _fd:
                        _fd_data = _json_fix.load(_fd)
                    _n_win_fnl = _fd_data.get("n_windows_accumulated", 0)
                    _n_emb_fnl = _fd_data.get("after_embargo", "?")
                    print(f"[FIX-P1A-FUNNEL-SEED] Funnel copiado → {_funnel_dest.name} "
                          f"| n_windows={_n_win_fnl} | after_embargo={_n_emb_fnl} "
                          f"| W={_window_env} seed={_optuna_seed_fnl}")
            except Exception as _e_fnl_fix:
                print(f"[FIX-P1A-FUNNEL-SEED] WARN: no se pudo copiar funnel: {_e_fnl_fix}")


            # Ã¢â€â‚¬Ã¢â€â‚¬ Aplicar Triple Barrier Method Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
            logger.info(f"Aplicando Triple Barrier Method a {n_signals} señales...")
            from luna.features.tbm import apply_triple_barrier

            # BUG-A01 FIX (2026-03-17): leer tbm_min_return de settings.yaml â€” era hardcoded 0.005.
            # BUG-A04 FIX (2026-03-17): usar cfg.xgboost.vertical_barrier_hours (no temporal_splits.embargo_hours).
            # Antes: _vb_h leidÃ³ de temporal_splits.embargo_hours (96H embargo CV != horizonte trade TBM).
            try:
                from config.settings import cfg as _cfg
                _min_ret = float(_cfg.xgboost.tbm_min_return)
            except Exception as e_minret:
                raise RuntimeError(f"Falta tbm_min_return en settings.yaml (SOP No-Fallback): {e_minret}")
            try:
                from config.settings import cfg as _cfg_tbm
                _vb_h    = int(_cfg_tbm.xgboost.vertical_barrier_hours)
                _dyn_min = int(_cfg_tbm.xgboost.dynamic_horizon_min_h)
                _embargo = int(_cfg_tbm.sop.embargo_hours)
                _dynamic_barrier = bool(_cfg_tbm.xgboost.dynamic_barrier)
                _lin_decay = bool(_cfg_tbm.xgboost.linear_decay_pt)
                _pt_decay_frac = float(_cfg_tbm.xgboost.pt_decay_fraction)
                _funding_series_oos = df_oos["FundingRate"] if "FundingRate" in df_oos.columns else None
            except Exception as e_tbm2:
                raise RuntimeError(f"Faltan parametros de riesgo TBM en settings.yaml (SOP No-Fallback): {e_tbm2}")


            # [FIX-OBS-01] Barreras PT/SL dinámicas por trade (TBM Vectorizado)
            # Bug anterior: calculaba una única moda de HMM_Semantic para toda la ventana OOS
            # y aplicaba el mismo PT/SL a todos los trades, ignorando los cambios de régimen intra-ventana.
            if "HMM_Semantic" in df_oos.columns:
                _sem_series = df_oos["HMM_Semantic"].fillna("UNKNOWN").astype(str)
                # [BUG-FIX-HMM-RESOLVER] (2026-05-20): Usar resolvedor robusto con fallback dinámico
                _pt = _sem_series.map(lambda r: get_hmm_tbm_params(r)["tp"])
                _sl = _sem_series.map(lambda r: get_hmm_tbm_params(r)["sl"])
                
                # [CONF-SCALER-01] Escala Intra-Régimen por Confianza
                if "xgb_prob_cal" in df_oos.columns:
                    _prob_series = df_oos["xgb_prob_cal"].fillna(0.5).clip(0.5, 1.0)
                elif "meta_v2_prob" in df_oos.columns:
                    _prob_series = df_oos["meta_v2_prob"].fillna(0.5).clip(0.5, 1.0)
                elif "xgb_prob" in df_oos.columns:
                    _prob_series = df_oos["xgb_prob"].fillna(0.5).clip(0.5, 1.0)
                else:
                    _prob_series = pd.Series(0.5, index=df_oos.index)
                
                # Mapear probabilidad [0.5, 1.0] a escalar [0.7, 1.3]
                _conf_scaler = 0.7 + ((_prob_series - 0.5) / 0.5) * (1.3 - 0.7)
                _pt = _pt * _conf_scaler
                _sl = _sl * _conf_scaler
                
                # Mantenemos moda para max_horizon porque apply_triple_barrier espera escalar
                # [BUG-FIX-HMM-RESOLVER] (2026-05-20): Usar resolvedor robusto para horizontes dinámicos
                _dyn_max = int(df_oos.loc[df_oos.index >= signal_times[0], "HMM_Semantic"].dropna().map(
                    lambda r: get_hmm_horizon(r)
                ).mode().iloc[0] if not df_oos.loc[df_oos.index >= signal_times[0], "HMM_Semantic"].dropna().empty else _HMM_HORIZON_FALLBACK)
                
                # Variables locales para retrocompatibilidad downstream si es necesario
                _regime_now = "MIXED (Dynamic)"
                _pt_log, _sl_log = _pt.median(), _sl.median()
                logger.info(f"  [CONF-SCALER] Scaler min={_conf_scaler.min():.2f} max={_conf_scaler.max():.2f}")
                logger.info(f"  [DIAG] OOS TBM (Dinámico/Trade): PT(med)={_pt_log:.1f}x SL(med)={_sl_log:.1f}x VB={_vb_h}h min_ret={_min_ret:.4f} dyn={_dynamic_barrier} [{_dyn_min}h,{_dyn_max}h]")
            else:
                _regime_now = "UNKNOWN"
                # [BUG-FIX-HMM-RESOLVER] (2026-05-20): Usar resolvedores robustos en bloque fallback estático
                tbm_p = get_hmm_tbm_params(_regime_now)
                _dyn_max = get_hmm_horizon(_regime_now)
                _pt = tbm_p["tp"]
                _sl = tbm_p["sl"]
                logger.info(f"  [DIAG] OOS TBM (Estático): PT={_pt}x SL={_sl}x VB={_vb_h}h min_ret={_min_ret:.4f} dyn={_dynamic_barrier} [{_dyn_min}h,{_dyn_max}h]")

            # [FIX-HOLDING-CAP 2026-06-14] Cargar cap de simulacion/ejecucion si existe
            try:
                from config.settings import cfg as _cfg_exec
                _exec_cap = int(getattr(_cfg_exec.execution, 'execution_holding_cap_h', 0))
            except Exception:
                _exec_cap = 0

            if _exec_cap > 0:
                _vb_h = min(_vb_h, _exec_cap)
                _dyn_max = min(_dyn_max, _exec_cap)
                print(f"[FIX-HOLDING-CAP 2026-06-14] Truncando horizonte maximo de la simulacion a {_exec_cap}H (TBM original: {_cfg_tbm.xgboost.vertical_barrier_hours}H)")
                logger.info(f"  [EXECUTION-CAP] Truncando horizonte máximo de la simulación a {_exec_cap}H (TBM original: {_cfg_tbm.xgboost.vertical_barrier_hours}H)")
            tbm_result = apply_triple_barrier(
                price_series=df_oos["close"],
                event_times=signal_times,
                sides=pd.Series(1 if _direct=='long' else -1, index=signal_times),
                pt_sl_multiplier=[_pt, _sl],
                vertical_barrier_hours=_vb_h,
                min_return=_min_ret,              # BUG-A01-FIX: de settings
                dynamic_barrier=_dynamic_barrier,
                dynamic_horizon_min_h=_dyn_min,
                dynamic_horizon_max_h=_dyn_max,
                linear_decay_pt=_lin_decay,
                pt_decay_fraction=_pt_decay_frac,
                funding_series=_funding_series_oos,
            )

            # ── Construir DataFrame de trades con P3 drawdown stop ──
            # P3-FIX: detener trades cuando equity cae >max_dd_stop_pct desde el máximo.
            # Esto limita el MaxDD=99.8% que resulta de operar durante rachas negativas largas.
            try:
                from config.settings import cfg as _cfg_risk
                MAX_DD_STOP = float(_cfg_risk.position_sizer.max_dd_stop_pct) / 100.0
            except Exception as e_dd:
                raise RuntimeError(f"Falta max_dd_stop_pct en settings.yaml (SOP No-Fallback): {e_dd}")
            logger.info(f"  [P3-FIX] Drawdown stop activo: >{MAX_DD_STOP:.0%} desde máximo → pausa de trades")

            # ─ TRIBE-2 (M-14 Fix B3): usar Kelly Fraccional actualizado desde PositionSizer ─
            try:
                from luna.risk.kelly_sizer import build_kelly_sizer_from_settings
                _kelly_sizer_instance = build_kelly_sizer_from_settings()
            except Exception as e:
                logger.warning(f"  [P3-FIX B3] No se pudo cargar KellyPositionSizer: {e}")
                _kelly_sizer_instance = None

            trade_records = []
            # BUG-4 fix (M-14): capital multiplicativo (compounding real)
            # ANTES: running_equity += ret_net ← lineal, subestima DD real
            # AHORA: capital *= (1 + ret_kelly) ← compounding correcto
            capital        = 1.0
            peak_capital   = 1.0
            dd_stop_active = False
            dd_stop_time   = None
            # alias aditivo para métricas downstream (run_statistical_validation.py usa cumsum)
            running_equity = 0.0
            peak_equity    = 0.0

            # [H5-ROLL-SR-GATE 2026-06-03] Cargar parametros del Kelly Rolling Sharpe Gate
            # No-Fallback estricto: parametro critico de riesgo, ausencia -> RuntimeError
            try:
                from config.settings import cfg as _cfg_h5
                _h5_enabled   = bool(getattr(_cfg_h5.position_sizer, 'roll_sr_gate_enabled', None))
                _h5_window    = int(getattr(_cfg_h5.position_sizer, 'roll_sr_window', None))
                _h5_threshold = float(getattr(_cfg_h5.position_sizer, 'roll_sr_threshold', None))
                if _h5_enabled is None or _h5_window is None or _h5_threshold is None:
                    raise KeyError("Parametros H5 incompletos en position_sizer")
                print(f"[H5-ROLL-SR-GATE] Cargado: enabled={_h5_enabled} window={_h5_window} threshold={_h5_threshold}")
                logger.info(f"[H5-ROLL-SR-GATE] Kelly Rolling Sharpe Gate activo: window={_h5_window} trades, umbral_SR={_h5_threshold}")
            except Exception as _e_h5:
                _err_h5 = f"CRITICAL [H5-ROLL-SR-GATE]: Fallo cargando position_sizer.roll_sr_* de settings.yaml: {_e_h5}"
                print(_err_h5)
                logger.critical(_err_h5)
                raise RuntimeError(_err_h5) from _e_h5

            # Historial de retornos recientes (deque de tamano fijo) para rolling Sharpe causal
            from collections import deque as _deque
            _roll_sr_history = _deque(maxlen=_h5_window)  # retornos brutos de los ultimos N trades
            _h5_gates_applied = 0  # contador de trades silenciados por H5
            print(f"[H5-ROLL-SR-GATE] Inicializado: history=deque(maxlen={_h5_window}) | threshold={_h5_threshold}")

            # [SOP-COST-FIX 2026-06-05] Cargar costo transaccional (SOP R6) sin fallback
            try:
                from config.settings import cfg as _cfg_cost
                _GLOBAL_COST_RT = float(_cfg_cost.sop.cost_pct)
            except Exception as _e_cost:
                raise RuntimeError(f"CRITICAL: Falta cfg.sop.cost_pct en settings.yaml. Política No-Fallback: {_e_cost}")

            # [FIX-CONCURRENCY-CAP 2026-06-13] Cargar limites de concurrencia de settings
            try:
                from config.settings import cfg as _cfg_concur
                _c_base = int(_cfg_concur.position_sizer.max_concurrent_trades)
                _dynamic_concurrency = bool(_cfg_concur.position_sizer.dynamic_concurrency_enabled)
                print(f"[FIX-CONCURRENCY-CAP 2026-06-13] Cargado: base={_c_base} | dynamic={_dynamic_concurrency}")
                logger.info(f"[FIX-CONCURRENCY-CAP] Concurrencia base: {_c_base} | Dinamico: {_dynamic_concurrency}")
            except Exception as _e_concur:
                _err_msg = f"CRITICAL: Falta max_concurrent_trades o dynamic_concurrency_enabled en settings.yaml. Politica No-Fallback: {_e_concur}"
                print(_err_msg)
                logger.critical(_err_msg)
                raise RuntimeError(_err_msg) from _e_concur

            active_exits = []

            for t in signal_times:
                if t not in tbm_result.index:
                    continue
                row = tbm_result.loc[t]
                if pd.isna(row.get("ret", np.nan)):
                    continue

                # [FIX-CONCURRENCY-CAP 2026-06-13] Clean up expired exits
                active_exits = [ex for ex in active_exits if ex > t]

                # Determine HMM semantic label and regime-based concurrency cap
                _hmm_sem_t = str(df_oos.loc[t, "HMM_Semantic"]) if "HMM_Semantic" in df_oos.columns and t in df_oos.index else ""
                
                if _dynamic_concurrency:
                    # round(C_base * 1.5) for strong trends, max(1, round(C_base * 0.5)) for volatile/range/bear
                    _is_strong_trend = any(x in _hmm_sem_t for x in ["BULL_TREND", "BULL_GRIND", "1_BULL"])
                    if _is_strong_trend:
                        _max_concur = int(round(_c_base * 1.5))
                    else:
                        _max_concur = max(1, int(round(_c_base * 0.5)))
                else:
                    _max_concur = _c_base

                if len(active_exits) >= _max_concur:
                    print(f"[FIX-CONCURRENCY-CAP 2026-06-13] SKIP en {t} | Concurrencia actual={len(active_exits)} >= cap={_max_concur} (regime={_hmm_sem_t})")
                    logger.info(f"[FIX-CONCURRENCY-CAP] Skipped trade in {t} due to concurrency cap: {len(active_exits)} >= {_max_concur} (regime={_hmm_sem_t})")
                    continue

                # P3: drawdown stop — usando capital multiplicativo (BUG-4 fix M-14)
                current_dd_pct = (peak_capital - capital) / peak_capital
                if current_dd_pct > MAX_DD_STOP:  # BUG-A05 FIX: eliminar condicion peak>1.0
                    if not dd_stop_active:
                        logger.warning(f"  [P3-FIX] Drawdown stop activado en {t}: DD={current_dd_pct:.1%} > {MAX_DD_STOP:.0%} — omitiendo trades hasta recuperación")
                        dd_stop_active = True
                    continue  # ← omitir este trade

                # Reactivar si la equity se recuperó hasta ≤50% del max_dd_stop
                if dd_stop_active and current_dd_pct < MAX_DD_STOP * 0.5:
                    dd_stop_active = False
                    logger.info(f"  [P3-FIX] Drawdown stop desactivado en {t}: equity recuperada")

                # [V2-P3-DRIFT] sizing unificado mediante la clase oficial (DRY principle)
                ret_raw_tbm  = float(row["ret"])
                
                if _kelly_sizer_instance is not None:
                    # Empaquetamos la fila para que el método nativo evalue todo el conjunto
                    _row_df = df_oos.loc[[t]].copy()
                    # Mapear columnas esperadas por KellyPositionSizer
                    _row_df["xgb_prob"] = float(df_oos.loc[t, "xgb_prob_cal"]) if "xgb_prob_cal" in df_oos.columns and t in df_oos.index else float(df_oos.loc[t, "xgb_prob"]) if "xgb_prob" in df_oos.columns and t in df_oos.index else 0.5
                    _row_df["meta_prob"] = float(df_oos.loc[t, "meta_v2_prob"]) if "meta_v2_prob" in df_oos.columns and t in df_oos.index else 0.5
                    _row_df["hmm_regime"] = float(df_oos.loc[t, "HMM_Regime"]) if "HMM_Regime" in df_oos.columns and t in df_oos.index else np.nan
                    
                    _fractions = _kelly_sizer_instance.size_signals_dynamic(_row_df, prob_col="xgb_prob")
                    _eff_mult = float(_fractions.iloc[0]) * _psi_kelly_penalty * _pred_drift_penalty
                else:
                    _eff_mult = _psi_kelly_penalty * _pred_drift_penalty # Fallback de emergencia

                # [H5-ROLL-SR-GATE 2026-06-03] Gate: Kelly=0 si Rolling Sharpe < umbral
                # El rolling Sharpe se calcula sobre los retornos BRUTOS de los últimos N trades
                # (retorno antes de Kelly, para que el gate no sea sensible al tamaño de posición).
                # Causalidad garantizada: _roll_sr_history solo contiene trades YA ejecutados (t-1, t-2, ...).
                if _h5_enabled and len(_roll_sr_history) >= max(3, _h5_window // 2):
                    _h5_arr = list(_roll_sr_history)
                    _h5_mean = float(np.mean(_h5_arr))
                    _h5_std  = float(np.std(_h5_arr, ddof=1)) if len(_h5_arr) > 1 else 1e-8
                    _h5_roll_sr = _h5_mean / max(_h5_std, 1e-8)
                    if _h5_roll_sr < _h5_threshold:
                        _h5_gates_applied += 1
                        print(f"[H5-ROLL-SR-GATE] SILENCIADO Kelly en {t} | "
                              f"roll_SR={_h5_roll_sr:.4f} < umbral={_h5_threshold} | "
                              f"kelly_original={_eff_mult:.4f} → 0.0 | "
                              f"history_n={len(_h5_arr)} | mean_ret={_h5_mean*100:.4f}% | "
                              f"std={_h5_std*100:.4f}% | gates_total={_h5_gates_applied}")
                        logger.info(f"[H5-ROLL-SR-GATE] Kelly silenciado en {t}: roll_SR={_h5_roll_sr:.4f} < {_h5_threshold}")
                        _eff_mult = 0.0
                    else:
                        logger.debug(f"[H5-ROLL-SR-GATE] Kelly PERMITIDO en {t}: roll_SR={_h5_roll_sr:.4f} >= {_h5_threshold}")
                elif _h5_enabled:
                    logger.debug(f"[H5-ROLL-SR-GATE] Historial insuficiente ({len(_roll_sr_history)}/{_h5_window}), gate inactivo en {t}")

                # [P1-BEAR-CRASH-01 2026-05-29] Hard exclusion en regimen 3_BEAR_CRASH.
                # Evidencia auditoria 2026-05-30: 36 trades con Kelly=0 contaminaban el WR
                # (is_win evaluado sobre ret_bruto aunque posicion=0 → ruido estadistico puro).
                # [H2-FIX 2026-05-30] Cambiado de soft (kelly=0, trade loggeado) a hard exclusion
                # (continue antes del append). Impacto esperado: +0.5pp WR, N limpio sin fantasmas.
                _hmm_sem_t = str(df_oos.loc[t, "HMM_Semantic"]) if "HMM_Semantic" in df_oos.columns and t in df_oos.index else ""
                _is_bear_crash = "BEAR_CRASH" in _hmm_sem_t or "3_BEAR" in _hmm_sem_t
                if _is_bear_crash:
                    print(f"[H2-FIX][P1-BEAR-CRASH-01] HARD SKIP en {t} | regime={_hmm_sem_t} | kelly_original={_eff_mult:.4f} | trade NO registrado (hard exclusion)")
                    continue  # Hard exclusion: no registrar, no contaminar WR ni N de trades

                # [FIX-COST-RT-01] (2026-05-16): Costo RT proporcional a la posicion Kelly ejecutada.
                # BUG ANTERIOR: ret_kelly = ret_raw * kelly - 0.0025
                #   Descontaba 0.25% sobre el 100% del NOMINAL, aunque Kelly=5% solo ejecuta el 5%.
                #   Con Kelly=5%: costo efectivo era 0.25%/0.05=5x el retorno ajustado → destruia todo P&L.
                # FIX: ret_kelly = (ret_raw - cost_rt) * kelly
                #   El costo 0.25% se aplica al retorno bruto ANTES del escalado Kelly.
                #   Equivalente matematico: cost_efectivo = cost_rt * kelly (proporcional a posicion real).
                _COST_RT = _GLOBAL_COST_RT  # [SOP-COST-FIX] Valor leido de settings (No-Fallback)

                ret_kelly    = (ret_raw_tbm - _COST_RT) * _eff_mult  # [FIX-COST-RT-01] costo sobre posicion real
                ret_bruto    = ret_raw_tbm - _COST_RT                # retorno bruto sin Kelly (para is_win TBM puro)
                logger.debug(
                    "[FIX-COST-RT-01] Trade: ret_raw=%.4f%% | kelly=%.1f%% | "
                    "ret_bruto=%.4f%% | ret_kelly=%.4f%% | costo_efectivo=%.5f%%",
                    ret_raw_tbm * 100, _eff_mult * 100,
                    ret_bruto * 100, ret_kelly * 100, _COST_RT * _eff_mult * 100
                )

                # [Fase 4] SHAP Drift Monitor (Explicabilidad por trade ejecutado)
                top_shap_features = "N/A"
                if _eff_mult > 0.0 and use_regime:
                    try:
                        _agent_name = None
                        _semantic_t = str(df_oos.loc[t, "HMM_Semantic"]) if "HMM_Semantic" in df_oos.columns else ""
                        for _r_name, _r_permitted in router_xgb.regimes_config.items():
                            if _semantic_t in _r_permitted:
                                _agent_name = _r_name
                                break
                        if _agent_name and _agent_name in router_xgb.models:
                            import shap
                            _model = router_xgb.models[_agent_name]
                            _features = router_xgb.signatures[_agent_name]["features"]
                            _explainer = shap.TreeExplainer(_model)
                            _avail_feats = [f for f in _features if f in df_oos.columns]
                            _X_t = df_oos.loc[[t], _avail_feats].copy()
                            # Rellenar faltantes con 0 para que explainer no falle
                            for mf in [f for f in _features if f not in _avail_feats]:
                                _X_t[mf] = 0.0
                            _X_t = _X_t[_features].fillna(0)
                            
                            _shap_raw = _explainer.shap_values(_X_t)
                            if isinstance(_shap_raw, list):
                                _shap_vals = _shap_raw[1][0]  # Clase 1
                            else:
                                if len(_shap_raw.shape) == 3:
                                    _shap_vals = _shap_raw[0, 1, :]
                                else:
                                    _shap_vals = _shap_raw[0]
                                    
                            _abs_shap = np.abs(_shap_vals)
                            _top_3_idx = np.argsort(_abs_shap)[-3:][::-1]
                            _top_str = [f"{_features[i]} ({_shap_vals[i]:.3f})" for i in _top_3_idx if _abs_shap[i] > 0.001]
                            if _top_str:
                                top_shap_features = " | ".join(_top_str)
                                logger.info(f"  [SHAP] Trade en {t} (Kelly: {_eff_mult:.1%}) -> Top Drivers: {top_shap_features}")
                    except Exception as e_shap:
                        logger.warning(f"  [SHAP] Fallo al generar explicabilidad: {e_shap}")

                tribe_mult = _eff_mult  # [FIX-TRIBE-MULT-01] alias V1→V2: tribe_mult era el multiplicador Kelly del sistema TRIBE
                trade_records.append({
                    "timestamp":    t,
                    "direction":    _direct,
                    "return_pct":   ret_kelly,
                    "return_raw":   ret_bruto,
                    "tribe_mult":   tribe_mult,
                    # BUG-GEN-02 FIX (2026-04-06): is_win = victoria TBM bruta (modelo puro).
                    # is_win_kelly = victoria teniendo en cuenta el multiplicador Kelly tribal.
                    # Si tribe_mult < 1.0, un trade ganador puede tener return_pct < 0 (Kelly pierde).
                    # Gauntlet usa return_pct → is_win_kelly es la métrica relevante para Calmar/WR real.
                    "is_win":       bool(ret_bruto > 0),
                    "is_win_kelly": bool(ret_kelly > 0),
                    "xgb_prob":     float(df_oos.loc[t, "xgb_prob"]) if t in df_oos.index else np.nan,
                    "xgb_prob_cal": float(df_oos.loc[t, "xgb_prob_cal"]) if "xgb_prob_cal" in df_oos.columns and t in df_oos.index else np.nan,
                    "meta_v2_prob": float(df_oos.loc[t, "meta_v2_prob"]) if "meta_v2_prob" in df_oos.columns and t in df_oos.index else np.nan,
                    "lgbm_prob":    float(df_oos.loc[t, "lgbm_prob"]) if "lgbm_prob" in df_oos.columns and t in df_oos.index else np.nan,
                    # BUG-03 FIX: registrar si este trade entro con threshold de emergencia
                    "signal_threshold":      signal_pipeline.used_threshold if 'signal_pipeline' in locals() else np.nan,
                    "threshold_was_lowered": signal_pipeline.threshold_was_lowered if 'signal_pipeline' in locals() else False,
                    "filter_fallback_level": getattr(signal_pipeline, "filter_fallback_level", 0) if 'signal_pipeline' in locals() else 0,
                    # TEARSHEET-v5.0: tiempos de entrada/salida para Panel B (Holding Time)
                    "entry_time":   t,
                    "exit_time":    row.get("first_touch", pd.NaT) if hasattr(row, "get") else getattr(row, "first_touch", pd.NaT),
                    # FIX-REGIME-STATIC-01 (2026-04-29): hmm_regime por trade (no moda estática del OOS completo).
                    # Bug: _regime_now era la moda de TODO el holdout, asignando 3_BEAR_CRASH a todos los trades
                    # en W3+W4 aunque BTC pasó de $60K a $108K. Fix: leer HMM_Semantic del bar de entrada.
                    "hmm_regime":   str(df_oos.loc[t, "HMM_Semantic"]) if "HMM_Semantic" in df_oos.columns and t in df_oos.index else (_regime_now if '_regime_now' in locals() else 'UNKNOWN'),
                    "HMM_Semantic": str(df_oos.loc[t, "HMM_Semantic"]) if "HMM_Semantic" in df_oos.columns and t in df_oos.index else (_regime_now if '_regime_now' in locals() else 'UNKNOWN'),
                    "kelly_fraction_used": _eff_mult,
                    "ood_kl_distance": float(df_oos.loc[t, "ood_kl_distance"]) if "ood_kl_distance" in df_oos.columns and t in df_oos.index else np.nan,
                    "shap_drivers": top_shap_features,
                    "alpha_trigger": ",".join([c for c in ["alpha_golden_score", "alpha_genetic_score", "alpha_dtw_signal"] if c in df_oos.columns and t in df_oos.index and float(df_oos.loc[t, c]) > 0]),
                })
                # [FIX-CONCURRENCY-CAP 2026-06-13] Registrar la salida del trade activo
                _exit_t = row.get("first_touch", pd.NaT) if hasattr(row, "get") else getattr(row, "first_touch", pd.NaT)
                if pd.isnull(_exit_t):
                    _exit_t = t + pd.Timedelta(hours=int(_vb_h))
                active_exits.append(_exit_t)

                # BUG-4: compounding multiplicativo
                capital      = capital * (1.0 + ret_kelly)
                peak_capital = max(peak_capital, capital)
                # alias lineal para compatibilidad con downstream
                running_equity += ret_kelly
                peak_equity     = max(peak_equity, running_equity)

                # [H5-ROLL-SR-GATE] Actualizar historial con retorno BRUTO del trade ejecutado
                # Usamos ret_bruto (pre-Kelly) para que el gate sea invariante al tamaño de posición
                _roll_sr_history.append(float(ret_bruto))


            # ---> RESEARCH: XGBoost Only (Pre-MetaLabeler Baseline)
            try:
                logger.info(f"  [RESEARCH] Generando baseline XGBoost Puro (Sin MetaLabeler) para {_direct}...")
                xgb_times = signal_pipeline.apply_embargo(df_oos, signal_pipeline.last_xgb_mask)
                if len(xgb_times) > 0:
                    tbm_xgb = apply_triple_barrier(
                        price_series=df_oos["close"],
                        event_times=xgb_times,
                        sides=pd.Series(1 if _direct=='long' else -1, index=xgb_times),
                        pt_sl_multiplier=[_pt, _sl],
                        vertical_barrier_hours=_vb_h,
                        min_return=_min_ret,
                        dynamic_barrier=_dynamic_barrier,
                        dynamic_horizon_min_h=_dyn_min,
                        dynamic_horizon_max_h=_dyn_max if '_dyn_max' in locals() else _vb_h,
                        linear_decay_pt=_lin_decay,
                        pt_decay_fraction=_pt_decay_frac,
                    )
                    for t_x in xgb_times:
                        try:
                            if t_x in tbm_xgb.index:
                                r_row = tbm_xgb.loc[t_x]
                            else:
                                continue
                        except KeyError:
                            continue
                        
                        if isinstance(r_row, pd.DataFrame):
                            r_row = r_row.iloc[0]
                            print(f"[XGB-BASELINE-BUGFIX] r_row was DataFrame for {t_x}, taking first row")

                        _raw_ret = r_row.get("ret", float('nan'))
                        if pd.isna(_raw_ret) or type(_raw_ret).__name__ == "NaTType":
                            continue

                        try:
                            # [SOP-COST-FIX] Usar costo global unificado sin fallback
                            ret_bruto = float(_raw_ret) - _GLOBAL_COST_RT
                            print(f"[FIX-05] Trade cost aplicado: cost_rt={_GLOBAL_COST_RT:.4f}, ret_raw={float(_raw_ret):.4f}, ret_bruto={ret_bruto:.4f}")
                        except Exception:
                            continue

                        try:
                            _is_win_val = bool(ret_bruto > 0)
                        except:
                            _is_win_val = False

                        _ft = r_row.get("first_touch", pd.NaT)
                        _pt_val = r_row.get("pt", pd.NaT)
                        _sl_val = r_row.get("sl", pd.NaT)
                        _t1_val = r_row.get("t1", pd.NaT)
                        
                        _exit = "UNKNOWN"
                        if not pd.isna(_ft):
                            if not pd.isna(_pt_val) and _ft == _pt_val: _exit = "PT"
                            elif not pd.isna(_sl_val) and _ft == _sl_val: _exit = "SL"
                            elif not pd.isna(_t1_val) and _ft == _t1_val: _exit = "VB"

                        all_xgb_baseline_records.append({
                            "timestamp": pd.to_datetime(t_x, utc=True),
                            "direction": _direct,
                            "return_raw": ret_bruto,
                            "is_win": _is_win_val,
                            "exit_type": _exit,
                            # FIX-HOLDINGTIME-01 (2026-04-29): holding_time_hours calculado desde t1-entry.
                            # Bug: r_row.get('holding_time', 0.0) siempre devolvía 0.0 (clave inexistente).
                            # Fix: calcular directamente como (first_touch - entry).total_seconds()/3600.
                            "holding_time_hours": float(
                                (_ft - t_x).total_seconds() / 3600
                                if not pd.isna(_ft) and hasattr((_ft - t_x), 'total_seconds')
                                else 0.0
                            ),
                            "hmm_regime": str(df_oos.loc[t_x, "HMM_Semantic"]) if "HMM_Semantic" in df_oos.columns and t_x in df_oos.index else (_regime_now if '_regime_now' in locals() else 'UNKNOWN'),
                            "HMM_Semantic": str(df_oos.loc[t_x, "HMM_Semantic"]) if "HMM_Semantic" in df_oos.columns and t_x in df_oos.index else (_regime_now if '_regime_now' in locals() else 'UNKNOWN'),
                            "xgb_prob": float(df_oos.loc[t_x, "xgb_prob"]) if "xgb_prob" in df_oos.columns and t_x in df_oos.index else float('nan'),
                            "meta_v2_prob": float(df_oos.loc[t_x, "meta_v2_prob"]) if "meta_v2_prob" in df_oos.columns and t_x in df_oos.index else float('nan'),
                            "xgb_prob_cal": float(df_oos.loc[t_x, "xgb_prob_cal"]) if "xgb_prob_cal" in df_oos.columns and t_x in df_oos.index else float('nan'),
                            "exit_time": _ft
                        })
                        print(f"[XGB-BASELINE-BUGFIX] Trade baseline logeado: t={t_x}, dir={_direct}, ret_bruto={ret_bruto:.4f}")
            except Exception as e:
                logger.exception(f"  [RESEARCH] Fallo generando XGBoost baseline para {_direct}:")
                
            all_trade_records.extend(trade_records)

        # Guardar el baseline unificado (RESEARCH: XGBoost Only)
        if 'all_xgb_baseline_records' in locals() and all_xgb_baseline_records:
            try:
                out_dir = self.root / "data" / "predictions"
                out_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(all_xgb_baseline_records).set_index("timestamp").to_parquet(out_dir / "oos_trades_xgb_baseline.parquet")
                print(f"[XGB-BASELINE-BUGFIX] Guardado oos_trades_xgb_baseline.parquet con {len(all_xgb_baseline_records)} registros")
            except Exception as e:
                logger.warning(f"  [RESEARCH] Fallo guardando parquet de baseline: {e}")

        if not all_trade_records:
            logger.warning("0 señales iniciales tras filtros y TBM en TODAS las direcciones. Generando dataset OOS vacío.")
            out_path = self.root / "data" / "predictions" / "oos_trades.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["timestamp", "return_pct", "return_raw", "tribe_mult", "is_win", "is_win_kelly", "xgb_prob", "meta_v2_prob", "lgbm_prob", "signal_threshold", "threshold_was_lowered", "filter_fallback_level", "entry_time", "exit_time", "hmm_regime", "HMM_Semantic", "kelly_fraction_used", "ood_kl_distance", "alpha_trigger"]).set_index("timestamp").to_parquet(out_path, index=True)

            # [FIX-F04] Escribir signal_funnel.json con estado 'zero_signals' para que
            # Gate-G5 pueda validar la ventana. Sin este write, Gate-G5 queda ciego
            # y no puede distinguir "ventana sin señales esperadas" de "error de pipeline".
            import json as _json_f04, os as _os_f04
            _funnel_dir = self.root / "data" / "reports"
            _funnel_dir.mkdir(parents=True, exist_ok=True)
            _funnel_zero = {
                "status": "zero_signals",
                "n_trades": 0,
                "window_id": _os_f04.environ.get("LUNA_WINDOW_ID", "unknown"),
                "seed": _os_f04.environ.get("LUNA_SEED", "unknown"),
                "reason": "0 senales pasaron todos los filtros (XGBoost + MetaLabeler + HMM + Embargo)",
                "after_xgb": 0,
                "after_meta": 0,
                "after_hmm": 0,
                "after_embargo": 0,
                "disabled_agents": _os_f04.environ.get("LUNA_DISABLED_AGENTS", "none"),
            }
            _funnel_path = _funnel_dir / "signal_funnel.json"
            with open(_funnel_path, "w", encoding="utf-8") as _ff:
                _json_f04.dump(_funnel_zero, _ff, indent=2)
            print(f"[FIX-F04] signal_funnel.json escrito con estado 'zero_signals' para Gate-G5 | window={_funnel_zero['window_id']} seed={_funnel_zero['seed']}")
            logger.info("[FIX-F04] signal_funnel.json escrito con 'zero_signals' para Gate-G5 - ventana sin señales registrada correctamente.")

            # Copia con run_id para evitar race conditions en WFB multiprocess
            _run_id_f04 = _os_f04.environ.get("LUNA_RUN_ID", "")
            if _run_id_f04:
                import shutil as _shutil_f04
                try:
                    _shutil_f04.copy(_funnel_path, _funnel_dir / f"signal_funnel_{_run_id_f04}.json")
                except Exception:
                    pass
            return True

        df_trades = pd.DataFrame(all_trade_records)
        
        # [FIX-OBS-04] Resolver colisiones LONG/SHORT en el mismo timestamp
        if df_trades.duplicated(subset=["timestamp"]).any():
            df_trades = df_trades.sort_values(by="xgb_prob_cal", ascending=False).drop_duplicates(subset=["timestamp"], keep="first")
            logger.warning("[FIX-OBS-04] Resueltas colisiones LONG/SHORT en el mismo timestamp reteniendo la señal de mayor prob calibrada.")

        # BUG-02 FIX (2026-03-17): usar 'timestamp' como índice para que el WFV
        # tenga un DatetimeIndex real al leer el parquet. Antes: index=False → índice
        # entero 0,1,2 → _run_wfv devolvía start="0" end="12" en vez de fechas reales.
        df_trades = df_trades.set_index("timestamp")
        df_trades.index = pd.to_datetime(df_trades.index, utc=True)
        n_wins = int(df_trades["is_win"].sum())
        wr     = float(n_wins) / float(len(df_trades))

        # MEJORA-R12-02 (2026-03-10): MaxDrawdown como métrica diagnóstica.
        # CÃƒÂ¡lculo sobre equity curve acumulada (cumsum de return_pct neto).
        # MaxDD estÃƒÂ¡tico = peor caída desde mÃƒÂ¡ximo histórico en el período OOS.
        # No bloquea el run Ã¢â‚¬â€ solo diagnóstico para run_statistical_validation.
        # [M-3-FIX] MaxDrawdown en base a equity multiplicativa
        equity = (1 + df_trades["return_pct"]).cumprod()
        rolling_max = equity.cummax()
        drawdown = equity - rolling_max                       # negativo en caídas
        max_drawdown = float(drawdown.min())                  # max drawdown (< 0)
        df_trades["equity_curve"] = equity.values
        df_trades["drawdown"]     = drawdown.values

        # ── DATA-01 FILTER AUDIT (C7 fix, 2026-03-23, corrected 2026-05-20) ──────────────────
        # Antes: referenciaba variables eliminadas o desordenadas que no correspondían a la secuencia de filtros.
        # Ahora: lee de signal_pipeline.funnel_stats y calcula restas secuenciales exactas en orden de aplicación real:
        # XGBoost -> LightGBM -> OOD -> CVD -> HMM -> MetaLabeler -> Cash Shield -> Momentum -> Embargo.
        try:
            print("[FIX-DATA-01-AUDIT] Running corrected sequential filter audit calculations...")
            logger.info("[FIX-DATA-01-AUDIT] Iniciando auditoría del embudo de filtrado en orden secuencial...")
            _f = signal_pipeline.funnel_stats
            _n_raw   = _f.get("raw_oos_bars", len(df_oos))
            _n_xgb   = _f.get("after_xgb",   0)
            _n_lgbm  = _f.get("after_lgbm",  _n_xgb)
            _n_ood   = _f.get("after_ood",   _n_lgbm)
            _n_cvd   = _f.get("after_cvd",   _n_ood)
            _n_hmm   = _f.get("after_hmm",   _n_cvd)
            _n_meta  = _f.get("after_meta",  _n_hmm)
            _n_cash  = _f.get("after_cash_shield", _n_meta)
            _n_mom   = _f.get("after_momentum", _n_cash)
            _n_emb   = _f.get("after_embargo", len(df_trades))
            _n_final = len(df_trades)
            logger.info(
                (
                    "[DATA-01 FILTER AUDIT] Embudo de senales:\n"
                    "  Horas OOS:        %5d\n"
                    "  XGBoost:          %5d  (bloq: %d, %.0f%%)\n"
                    "  LightGBM:         %5d  (bloq: %d, %.0f%% vs XGB)\n"
                    "  OOD Guard:        %5d  (bloq: %d, %.0f%% vs LGBM)\n"
                    "  CVD Divergence:   %5d  (bloq: %d, %.0f%% vs OOD)\n"
                    "  HMM filter:       %5d  (bloq: %d, %.0f%% vs CVD)\n"
                    "  MetaLabelerV2:    %5d  (bloq: %d, %.0f%% vs HMM)\n"
                    "  Cash Shield:      %5d  (bloq: %d, %.0f%% vs Meta)\n"
                    "  Momentum:         %5d  (bloq: %d, %.0f%% vs Cash)\n"
                    "  Embargo:          %5d  (bloq: %d, %.0f%% vs Mom)\n"
                    "  TRADES FINALES:   %5d  (%.1f%% de XGB raw)  %s"
                ) % (
                    _n_raw,
                    _n_xgb,  _n_raw  - _n_xgb,  (1 - _n_xgb  / max(_n_raw,  1)) * 100,
                    _n_lgbm, _n_xgb  - _n_lgbm, (1 - _n_lgbm / max(_n_xgb,  1)) * 100,
                    _n_ood,  _n_lgbm - _n_ood,  (1 - _n_ood  / max(_n_lgbm, 1)) * 100,
                    _n_cvd,  _n_ood  - _n_cvd,  (1 - _n_cvd  / max(_n_ood,  1)) * 100,
                    _n_hmm,  _n_cvd  - _n_hmm,  (1 - _n_hmm  / max(_n_cvd,  1)) * 100,
                    _n_meta, _n_hmm  - _n_meta, (1 - _n_meta / max(_n_hmm,  1)) * 100,
                    _n_cash, _n_meta - _n_cash, (1 - _n_cash / max(_n_meta, 1)) * 100,
                    _n_mom,  _n_cash - _n_mom,  (1 - _n_mom  / max(_n_cash, 1)) * 100,
                    _n_emb,  _n_mom  - _n_emb,  (1 - _n_emb  / max(_n_mom,  1)) * 100,
                    _n_final, (_n_final / max(_n_xgb, 1)) * 100,
                    "OK" if _n_final >= 100 else f"INSUF ({_n_final}/100)"
                )
            )
        except Exception as _e_audit:
            logger.debug(f"[DATA-01 FILTER AUDIT] Error: {_e_audit}")
        # ─────────────────────────────────────────────────────────────────────

        logger.success(
            "✅ %d trades OOS | WR=%.1f%% | MeanRet=%.3f%% | MaxDD=%.2f%%" % (
                len(df_trades), wr * 100,
                df_trades["return_pct"].mean() * 100,
                max_drawdown * 100
            )
        )


        # [FIX-RAW-PROBS] Generar y guardar oos_raw_probs.parquet para ensamble
        try:
            print("[FIX-RAW-PROBS] Iniciando la generación de oos_raw_probs.parquet...")
            logger.info("[FIX-RAW-PROBS] Generando probabilidades de expertos para ensamble...")
            raw_probs_df = pd.DataFrame(index=df_oos_base.index)
            raw_probs_df.index.name = "timestamp"
            
            if use_regime:
                from luna.models.regime_router import RegimeRouter
                primary_dir = directions_to_run[0] if directions_to_run else "long"
                router = RegimeRouter(self.models_dir, agent_type="xgboost", direction=primary_dir)
                
                for agent_name in ["bull", "bear", "range"]:
                    col_name = f"prob_{agent_name}"
                    if agent_name in router.models:
                        model = router.models[agent_name]
                        sig = router.signatures[agent_name]
                        features_list = sig["features"]
                        
                        # Alinear features
                        available_feats_agent = [f for f in features_list if f in df_oos_base.columns]
                        missing_feats_agent   = [f for f in features_list if f not in df_oos_base.columns]
                        
                        X_subset = df_oos_base[available_feats_agent].copy()
                        for m_feat in missing_feats_agent:
                            X_subset[m_feat] = np.nan
                        X_subset = X_subset[features_list]
                        
                        # [OPT-INFERENCE] Force CPU device to bypass DMatrix PCIe transfer bottleneck in XGBoost 3.x
                        try:
                            if hasattr(model, "set_params"):
                                model.set_params(device="cpu")
                        except Exception:
                            pass

                        # [FIX-RAW-PROBS-FN-01 2026-05-28] Convertir a numpy para evitar error feature_names_in_
                        try:
                            _X_np_rp = X_subset.to_numpy(dtype=np.float32, na_value=np.nan)
                            print(f"[FIX-RAW-PROBS-FN-01] Agente '{agent_name}': numpy float32 shape={_X_np_rp.shape} | missing={len(missing_feats_agent)}")
                            probs = model.predict_proba(_X_np_rp)[:, 1]
                        except Exception as _rp_np_err:
                            print(f"[FIX-RAW-PROBS-FN-01] WARN numpy fallback para '{agent_name}': {_rp_np_err}")
                            probs = model.predict_proba(X_subset)[:, 1]
                        
                        # Calibrar si existe isotonic_calibrator
                        if agent_name in router.isotonic_calibrators:
                            try:
                                calibrated = router.isotonic_calibrators[agent_name].predict(probs)
                                calibrated = np.clip(calibrated, 0.0, 1.0)
                            except Exception:
                                calibrated = probs
                        else:
                            calibrated = probs
                            
                        raw_probs_df[col_name] = calibrated
                    else:
                        if router._baseline_model is not None and router._baseline_features:
                            X_base = df_oos_base[[f for f in router._baseline_features if f in df_oos_base.columns]].copy()
                            for _mf in [f for f in router._baseline_features if f not in df_oos_base.columns]:
                                X_base[_mf] = np.nan
                            X_base = X_base[router._baseline_features]
                            
                            # [OPT-INFERENCE] Force CPU
                            try:
                                if hasattr(router._baseline_model, "set_params"):
                                    router._baseline_model.set_params(device="cpu")
                            except Exception:
                                pass

                            # [FIX-RAW-PROBS-FN-01] baseline también necesita conversión numpy
                            try:
                                _X_base_np = X_base.to_numpy(dtype=np.float32, na_value=np.nan)
                                print(f"[FIX-RAW-PROBS-FN-01] baseline '{agent_name}': numpy float32 shape={_X_base_np.shape}")
                                raw_probs_df[col_name] = router._baseline_model.predict_proba(_X_base_np)[:, 1]
                            except Exception as _rp_bl_err:
                                print(f"[FIX-RAW-PROBS-FN-01] WARN baseline numpy fallback: {_rp_bl_err}")
                                raw_probs_df[col_name] = router._baseline_model.predict_proba(X_base)[:, 1]
                        else:
                            raw_probs_df[col_name] = 0.0
            else:
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(xgb_model_path)
                
                # [OPT-INFERENCE] Force CPU
                try:
                    if hasattr(model, "set_params"):
                        model.set_params(device="cpu")
                except Exception:
                    pass

                X_oos = df_oos_base[available_feats]
                probs = model.predict_proba(X_oos)[:, 1]
                raw_probs_df["prob_bull"] = probs
                raw_probs_df["prob_bear"] = 0.0
                raw_probs_df["prob_range"] = 0.0
                
            # [BUG-5 FIX] OOD Score Passthrough
            if "ood_kl_distance" in df_oos_base.columns:
                raw_probs_df["ood_score"] = df_oos_base["ood_kl_distance"]
            if "is_ood_outlier" in df_oos_base.columns:
                raw_probs_df["is_ood_outlier"] = df_oos_base["is_ood_outlier"]
            elif "ood_kl_distance" in df_oos_base.columns:
                # Fallback if binary column doesn't exist
                _ood_thresh = 0.95 
                try:
                    from config.settings import cfg as _cfg_ood
                    _ood_thresh = float(getattr(_cfg_ood.xgboost, 'ood_kl_threshold', 0.95))
                except Exception:
                    pass
                raw_probs_df["is_ood_outlier"] = df_oos_base["ood_kl_distance"] > _ood_thresh
                
            # [FIX-predict_oos.py] Guardar con index=True para conservar el DatetimeIndex timestamp y eliminar to_parquet con index=False
            print("[FIX-RAW-PROBS] Guardando oos_raw_probs.parquet con index=True (DatetimeIndex)...")
            raw_probs_out_dir = self.root / "data" / "predictions"
            raw_probs_out_dir.mkdir(exist_ok=True, parents=True)
            raw_probs_out_path = raw_probs_out_dir / "oos_raw_probs.parquet"
            raw_probs_df.to_parquet(raw_probs_out_path, index=True)
            print(f"[FIX-RAW-PROBS] oos_raw_probs.parquet guardado exitosamente en {raw_probs_out_path} con {len(raw_probs_df)} filas.")
            logger.success(f"[FIX-RAW-PROBS] oos_raw_probs.parquet guardado con éxito: {len(raw_probs_df)} filas.")
        except Exception as _e_raw_gen:
            print(f"[FIX-RAW-PROBS] ERROR al generar oos_raw_probs.parquet: {_e_raw_gen}")
            logger.error(f"[FIX-RAW-PROBS] Error en oos_raw_probs.parquet: {_e_raw_gen}")

        # ── Guardar parquet ──
        out_dir  = self.root / "data" / "predictions"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / "oos_trades.parquet"
        # INVARIANTS CAPA 2: verificar trades antes de guardar
        if _INVARIANTS_AVAILABLE:
            check_trades_df(df_trades, context="oos_trades_save")

        df_trades.to_parquet(out_path, index=True)  # BUG-02 FIX: índice=timestamp real
        logger.success(f"✅ Guardado en: {out_path}")

        # ─── Red flags SOP ───
        if wr > 0.70:
            logger.warning(f"🚨 RED FLAG SOP: WR={wr * 100:.1f}% > 70% ⚠️  verificar leakage o costos")
        if len(df_trades["return_pct"]) > 0:
            std_r = df_trades["return_pct"].std()
            if std_r > 1e-10:
                # FIX M-05: Annualization proper
                if len(df_trades) > 1:
                    days = (df_trades.index.max() - df_trades.index.min()).days
                    n_per_year = len(df_trades) / (days / 365.25) if days > 0 else len(df_trades) * 365.25
                else:
                    n_per_year = 0
                sh = (df_trades["return_pct"].mean() / std_r) * (n_per_year ** 0.5)
                if sh > 4.0:
                    logger.warning(f"🚨 RED FLAG SOP: Sharpe crudo={sh:.2f} > 4.0 ⚠️  revisar pipeline")


        # Guardar el baseline unificado (RESEARCH: XGBoost Only)
        if 'all_xgb_baseline_records' in locals() and all_xgb_baseline_records:
            try:
                out_dir = self.root / "data" / "predictions"
                out_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(all_xgb_baseline_records).set_index("timestamp").to_parquet(out_dir / "oos_trades_xgb_baseline.parquet")
            except Exception as e:
                logger.warning(f"  [RESEARCH] Fallo guardando parquet de baseline: {e}")
        
        return True


if __name__ == "__main__":
    import sys
    try:
        generator = OOSTradesGenerator()
        success = generator.generate()
        sys.exit(0 if success else 1)
    except Exception as e:
        from loguru import logger
        import traceback
        logger.error(f"[FATAL UNCAUGHT] Script crashed at main level: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

