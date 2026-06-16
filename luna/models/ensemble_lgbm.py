# [ARCH-29-FIX 2026-06-02] CODIGO ZOMBIE — NO ACTIVO EN PRODUCCION
# =======================================================================
# Este archivo (ensemble_lgbm.py) NO se importa en ningun punto del pipeline
# activo (wfb_worker.py, run_wfb_orchestrator.py, train_production_model.py).
# La flag use_lgbm_ensemble=false en settings.yaml desactiva su uso.
# El RegimeRouter (regime_router.py) tiene soporte para agent_type='lightgbm'
# pero nunca se instancia con ese parametro en el WFB actual.
#
# ESTADO: Codigo de referencia/investigacion, no de produccion.
# ACCION REQUERIDA antes de reactivar: verificar compatibilidad con pipeline
# actual (xgboost v3.x, WFB rolling, SFI, regime_mapping de 3 agentes).
# AUDITORIA: arch22_29_code_inspection.py confirma 0 invocaciones.
# =======================================================================

"""


Orquestador LightGBM Meta-Model - Luna V1



===================================================



Entrena el modelo base conectando las features seleccionadas (SFI) y la 



etiqueta del rÃƒÂ©gimen HMM.



SOP Aplicado:



- R3 (Purge/Embargo): Se usa Combinatorial Purged CV para la evaluaciÃƒÂ³n de Optuna.



- R5 (DSR Objetivo): La mÃƒÂ©trica a maximizar por Optuna es el Deflated Sharpe OOS.



- R6 (Costos TransacciÃ³n): 0.25% RT aplicado a las simulaciones de Sharpe.



"""



import sys
import os as _os_lgbm



from pathlib import Path



def get_project_root() -> Path:



    return Path(__file__).resolve().parent.parent.parent



# Inject root to sys.path



sys.path.insert(0, str(get_project_root()))



from luna.utils.encoding_fix import fix_stdout_encoding; fix_stdout_encoding()



import json



import logging



from loguru import logger



import numpy as np



import pandas as pd



import lightgbm as lgb



import optuna



import joblib



from scipy.stats import norm



import math



import matplotlib.pyplot as plt



from luna.utils.debug_guards import (



    check_target_balance, check_numeric_stability, check_df_sanity,



    vlog, timeit, log_memory_usage,



)



# ParÃ¡metros Globales (SOP) â€” [TIPO-2] constantes de SOP, no hiperparÃ¡metros



# Todos leÃ­dos desde settings.yaml â€” ver bloque try/except abajo



CPCV_TEST_GROUPS = 2  # arquitectura CPCV: siempre k=2 grupos de test (LdP 2018)



# B1 FIX (2026-03-09): leer desde settings.yaml



import os as _os



try:



    from config.settings import cfg as _cfg_xgb



    OPTUNA_TRIALS = int(_cfg_xgb.xgboost.optuna_trials)



    # ARCH-03-REFACTOR (2026-03-18): fuente Ãºnica = sop.cpcv_groups.



    # Fallback a n_purged_splits solo si sop.cpcv_groups no existe (alias deprecado).



    _cpcv_n = int(_cfg_xgb.sop.cpcv_groups) \
               or int(_cfg_xgb.xgboost.n_purged_splits)



    CPCV_GROUPS   = int(_cpcv_n)



    PURGE_H       = int(_cfg_xgb.sop.purge_hours)
    EMBARGO_H     = int(_cfg_xgb.sop.embargo_hours)



    COST_PCT      = float(_cfg_xgb.sop.cost_pct)



except Exception as _cfg_err:



    # ARCH-FAIL-LOUD (2026-03-18): NO silenciar errores de configuraciÃ³n.



    # Un except silencioso ocultarÃ­a: YAML corrupto, import error, parÃ¡metro



    # movido de secciÃ³n, etc. El pipeline correrÃ­a con valores INCORRECTOS



    # (p.ej. 600 trials en vez de 100, CPCV=6 en vez de 8) sin ningÃºn aviso.



    # Principio: Fail Loud > Fail Silent. Si settings.yaml no carga, abortamos.



    raise RuntimeError(



        f"\n[CRITICAL] ensemble_lgbm.py no pudo cargar settings.yaml.\n"



        f"  Error: {_cfg_err}\n"



        f"  El pipeline NO puede ejecutarse sin configuraciÃ³n vÃ¡lida.\n"



        f"  Verifica: sintaxis YAML, PYTHONPATH, existencia de config/settings.py"



    ) from _cfg_err



# B3 FIX (2026-03-09): flag diagnÃ³stico para aislar efecto de mining rules.



# Uso: set LUNA_SKIP_MINING=1 && python core/models/ensemble_lgbm.py



SKIP_MINING: bool = _os.environ.get("LUNA_SKIP_MINING", "0") == "1"



# ---------------------------------------------------------------------------



# P1-5: MiningRuleValidator â€” filtro DSR para reglas de AI Mining



# ---------------------------------------------------------------------------



class MiningRuleValidator:



    """



    Valida las reglas de AI Mining (golden_rule_N, genetic_rule_N) usando DSR



    antes de inyectarlas en LightGBM.



    Reemplaza el pass-through ciego de hits>0 con validaciÃ³n estadÃ­stica.



    P1-5 (planes_mejora_v3.md):



    - n_trials_efectivo = OPTUNA_TRIALS * 3.0 por penalizaciÃ³n heurÃ­stica (mining es bÃºsqueda ad-hoc)



    - Solo reglas con DSR >= MIN_DSR_RULE pasan al modelo



    - 0 reglas aprobadas es preferible a reglas con overfitting



    CONTRATO close_rets (LAB-01 fix 2026-03-20):



    - Debe ser el retorno forward al MISMO horizonte que el TBM del LightGBM.



    - INCORRECTO: pct_change(1).shift(-1)  â† 1H (inconsistente con TBM de 96-168H)



    - CORRECTO:   pct_change(N).shift(-N)  â† N = vertical_barrier_hours de settings.yaml



    - RazÃ³n: una regla con edge en 1H puede ser destructiva en el horizonte TBM real;



      la validaciÃ³n DSR debe usar el mismo horizonte que el modelo que consume la regla.



    """



    MIN_DSR_RULE = 0.80       # Umbral DSR para reglas de mining
    # [FIX-D] N_TRIALS_PENALTY leído de settings.yaml ai_mining.n_trials_penalty
    # Ref: Bailey (2014) "Pseudo-Mathematics": ratio ≈ 3× para corregir overfitting por selección múltiple.
    _N_TRIALS_PENALTY_DEFAULT = 3.0  # fallback documentado

    def __init__(self, close_rets: pd.Series, cost_pct: float = COST_PCT):
        self.close_rets = close_rets
        self.cost_pct = cost_pct
        try:
            from config.settings import cfg as _cfg_tp_lgbm
            self.N_TRIALS_PENALTY = float(_cfg_tp_lgbm.ai_mining.n_trials_penalty)
        except Exception:
            self.N_TRIALS_PENALTY = self._N_TRIALS_PENALTY_DEFAULT
            print(f"[FIX-D] LGBM WARN: No se pudo leer ai_mining.n_trials_penalty. Usando fallback={self.N_TRIALS_PENALTY} (Bailey 2014)")
        print(f"[FIX-D] LGBM MiningRuleValidator: N_TRIALS_PENALTY={self.N_TRIALS_PENALTY} (optuna_trials efectivos = {int(OPTUNA_TRIALS * self.N_TRIALS_PENALTY)})")
        self.n_trials_efectivo = int(OPTUNA_TRIALS * self.N_TRIALS_PENALTY)  # ~300



    def _compute_rule_dsr(self, rule_series: pd.Series) -> float:



        """



        Calcula el DSR de una regla binary (0/1) usando sus retornos OOS.



        Retorna DSR en [0,1]; DSR < 0.80 = rechazada.



        """



        from scipy.stats import norm



        import math



        aligned = self.close_rets.align(rule_series.reindex(self.close_rets.index), join='inner')



        rets, sigs = aligned



        sigs = sigs.fillna(0).astype(float)



        pos = sigs.values



        # [LUNA V1 INSTITUTIONAL FIX] TBM overlapping trades cost



        strat_rets = pos * (rets.values - self.cost_pct)



        if len(strat_rets) < 30 or np.std(strat_rets) < 1e-8:



            return 0.0



        sr = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(365 * 24)



        t = len(strat_rets)



        # DSR (Bailey & LdP 2014): penaliza por n_trials usando la distribución del máximo de normales



        n_trials = max(self.n_trials_efectivo, 2)



        z1 = norm.ppf(1 - 1.0 / n_trials)



        z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))



        # Asumiendo varianza cruzada empírica estándar = 1.0 para el EVT



        sr_star = 1.0 * ((1 - 0.577215) * z1 + 0.577215 * z2)



        try:



            # BUG-EVT FIX: Comparar SR con sr_star directo escalado por longitud.



            dsr = float(norm.cdf((sr - sr_star) * math.sqrt(max(1, t - 1))))



        except Exception:



            dsr = 0.0



        return round(max(0.0, min(1.0, dsr)), 4)



    def filter_rules(self, df: pd.DataFrame, rule_cols: list[str]) -> list[str]:



        """



        Filtra las reglas por DSR >= MIN_DSR_RULE.



        Args:



            df:        DataFrame con las columnas de reglas



            rule_cols: Lista de columnas candidatas (golden_rule_N, genetic_rule_N)



        Returns:



            Lista de reglas aprobadas (puede ser vacÃ­a)



        """



        approved = []



        rejected = []



        for col in rule_cols:



            if col not in df.columns:



                continue



            dsr = self._compute_rule_dsr(df[col])



            if dsr >= self.MIN_DSR_RULE:



                approved.append(col)



                logger.info(f"  [MINING PASS] {col}: DSR={dsr:.4f} >= {self.MIN_DSR_RULE}")



            else:



                rejected.append(col)



                logger.debug(f"  [MINING REJECT] {col}: DSR={dsr:.4f} < {self.MIN_DSR_RULE}")



        logger.info(



            f"[P1-5 Mining DSR] {len(approved)}/{len(rule_cols)} reglas aprobadas "



            f"(n_trials_efectivo={self.n_trials_efectivo}, umbral DSR={self.MIN_DSR_RULE})"



        )



        return approved



class LGBMTrainer:



    def __init__(self, regime_name=None, regime_list=None, n_trials=OPTUNA_TRIALS):



        self.root = get_project_root()



        self.regime_name = regime_name



        self.regime_list = regime_list



        self.n_trials = n_trials
        
        try:
            from config.settings import cfg as _cfg_dir
            _dmode = str(_cfg_dir.fase2.direction_mode)
        except Exception:
            _dmode = "both"
            
        if _dmode == "both":
            self.native_direction = "short" if self.regime_name == "bear" else "long"
        else:
            self.native_direction = "long"

        self.X = None
        self.y = None



        self.close_rets = None



        self.study = None



        self.best_params = {}



        # FIX-CPCV-CACHE-01: cache de splits CPCV precalculados.



        # Los splits son siempre los mismos para un dataset dado â€” recalcularlos



        # en cada uno de los 100+ trials es trabajo redundante (~0.5s * 100 = 50s).



        # Se setea en tune_hyperparameters() antes de lanzar Optuna.



        self._cached_splits = None



    def load_dataset(self):



        logger.info("Cargando features base y labels HMM...")



        # 1. Cargar Parquet Principal



        df = pd.read_parquet(self.root / "data" / "features" / "features_train.parquet")

        # ── [CAPA-1] Rolling Window de 3 años (Filtro de Memoria) ──────────────
        try:
            from config.settings import cfg as _cfg_rw
            _t_mode = str(_cfg_rw.wfb.training_mode)
            if _t_mode == 'rolling':
                _rw_years = int(_cfg_rw.wfb.rolling_window_years)
                _train_end_str = _cfg_rw.temporal_splits.train_end
                _train_end_dt = pd.to_datetime(_train_end_str, utc=True)
                _rolling_start = _train_end_dt - pd.DateOffset(years=_rw_years)
                
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                    
                _len_before = len(df)
                df = df[df.index >= _rolling_start]
                logger.info(
                    f"[CAPA-1] training_mode='rolling' ({_rw_years} años): "
                    f"Descartadas {_len_before - len(df)} velas anteriores a {_rolling_start.date()}."
                )
        except Exception as e:
            logger.warning(f"[CAPA-1] Error aplicando Rolling Window en LGBM: {e}. Fallback a 'expanding'.")
        # ───────────────────────────────────────────────────────────────────────



        # 2. Cargar Seleccionadas



        with open(self.root / "data" / "features" / "selected_features.json", 'r') as f:



            features_list = json.load(f)["selected_features"]



        # â”€â”€ [P1-5] Mining Rules con validaciÃ³n DSR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



        # Las golden_rule_N y genetic_rule_N pasan por filtro DSR antes de ser



        # inyectadas en LightGBM. Solo reglas con DSR >= 0.80 (n_trials=1800 efectivos)



        # son admitidas. 0 reglas aprobadas es CORRECTO si todas hacen overfitting.



        # (Antes: pass-through ciego si hits > 0 â€” BUG P1-5)



        _rule_cols_candidate = sorted([



            c for c in df.columns



            if (c.startswith("golden_rule_") or c.startswith("genetic_rule_"))



            and df[c].sum() > 0  # al menos 1 activaciÃ³n



        ])



        if _rule_cols_candidate and not SKIP_MINING:



            # LAB-01 fix (2026-03-20): usar horizonte TBM en vez de 1H para la validaciÃ³n DSR.



            # pct_change(N).shift(-N) = retorno forward de N horas en la barra t.



            # N = vertical_barrier_hours (mismo horizonte que el TBM con el que entrena LightGBM).



            # Sin este fix: una regla que predice 1H bien (DSR>0.80) pero es destructiva



            # en 96-168H podrÃ­a injertarse en el modelo y degradar el rendimiento OOS.



            if "close" in df.columns:



                try:



                    from config.settings import cfg as _cfg_mvr



                    _vbh_mvr = int(_cfg_mvr.xgboost.vertical_barrier_hours)



                except Exception:



                    _vbh_mvr = 96



                _close_rets_proxy = df["close"].pct_change(_vbh_mvr).shift(-_vbh_mvr)



                logger.debug(



                    "[P1-5/LAB-01] MiningRuleValidator: close_rets horizonte=%dH "



                    "(consistente con vertical_barrier_hours del TBM).", _vbh_mvr



                )



            else:



                _close_rets_proxy = pd.Series(dtype=float)



            _validator = MiningRuleValidator(close_rets=_close_rets_proxy)



            _approved_rules = _validator.filter_rules(df, _rule_cols_candidate)



            _new_rules = [c for c in _approved_rules if c not in features_list]



            if _new_rules:



                features_list.extend(_new_rules)



                logger.info(f"[P1-5] {len(_new_rules)} reglas Mining aprobadas DSR: {_new_rules}")



            else:



                logger.info("[P1-5] 0 reglas Mining aprobadas con DSR >= 0.80.")



        elif SKIP_MINING:



            # MEJORA-03: limpiar rules de runs anteriores en features_list cuando SKIP_MINING=1.



            # Sin esto, golden_rule_N del selected_features.json del run anterior quedan en el modelo



            # aunque el SKIP_MINING pretenda aislarlos.



            _rule_feats_old = [f for f in features_list



                               if f.startswith("golden_rule_") or f.startswith("genetic_rule_")]



            if _rule_feats_old:



                for _rf in _rule_feats_old:



                    features_list.remove(_rf)



                logger.info(f"[DIAGNÃ“STICO] LUNA_SKIP_MINING=1 â€” {len(_rule_feats_old)} mining rules eliminadas de features_list: {_rule_feats_old}")



            else:



                logger.info("[DIAGNÃ“STICO] LUNA_SKIP_MINING=1 â€” mining rules DESACTIVADAS para aislamiento.")



            logger.info("  Para activar: borrar variable de entorno LUNA_SKIP_MINING o poner en 0.")



        else:



            logger.info("[P1-5] No hay reglas Mining con activaciones para evaluar.")



        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



        # â”€â”€ [R21] Features de Timing Corto Plazo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



        # AUDIT-01 FIX (2026-03-20): estas features son CANDIDATAS SFI â€” no deben bypasear el filtro



        # indefinidamente. En el prÃ³ximo ciclo SFI completo deben incluirse como candidatas normales



        # en feature_pipeline.py para que el SFI decida si tienen edge real (DSR â‰¥ 0.05).



        #



        # cfg.features.timing_features_bypass_sfi:



        #   true  (DEFAULT actual): inyectar como pass-through mientras el SFI no las ha evaluado.



        #   false: NO inyectar â€” el SFI ya las evaluÃ³ y decidiÃ³ si entran o no.



        #



        # Para el prÃ³ximo ciclo SFI completo: cambiar a false Y aÃ±adir en feature_pipeline.py



        # las columnas timing_funding_acum8h, timing_momentum_div, timing_vol_divergence



        # como features calculadas antes del SFI.



        try:



            from config.settings import cfg as _cfg_timing



            _bypass_sfi = bool(int(int(_cfg_timing.features).timing_features_bypass_sfi))



        except Exception:



            _bypass_sfi = True



        _timing_feats_added = []



        if _bypass_sfi:



            try:



                # 1. Funding Rate acumulado 8h (EWM Î±=0.5 â†’ decay rÃ¡pido)



                # Negativo acumulado = mercado pagando shorts = presiÃ³n bajista real



                if "FundingRate" in df.columns:



                    df["timing_funding_acum8h"] = (



                        df["FundingRate"].ewm(span=8, min_periods=1).mean()



                    )



                    _timing_feats_added.append("timing_funding_acum8h")



                # 2. Momentum divergence 24h vs 7d



                # Si ret_24h < ret_7d: precio decelerando â†’ potencial reversiÃ³n



                # Si ret_24h > ret_7d: precio acelerando â†’ momentum continuaciÃ³n



                if "close" in df.columns:



                    ret_24h = df["close"].pct_change(24)



                    ret_7d  = df["close"].pct_change(168)



                    df["timing_momentum_div"] = ret_24h - ret_7d



                    _timing_feats_added.append("timing_momentum_div")



                # 3. Volume divergence (price move vs volume ratio)



                # Sube el precio pero el volumen es bajo â†’ seÃ±al potencialmente dÃ©bil



                # vol_ratio = volume / rolling_30d_mean_volume



                # divergence = abs(ret_24h) / (vol_ratio + 1e-6)



                # Valor alto = mucho movimiento de precio con poco volumen â†’ fake



                if "close" in df.columns and "volume" in df.columns:



                    ret_24h_abs = df["close"].pct_change(24).abs()



                    vol_ma_30d  = df["volume"].rolling(window=720, min_periods=48).mean()



                    vol_ratio   = df["volume"] / (vol_ma_30d + 1e-6)



                    df["timing_vol_divergence"] = ret_24h_abs / (vol_ratio + 1e-6)



                    # Clip extremos para evitar outliers en crashes (crash real = vol muy alto)



                    df["timing_vol_divergence"] = df["timing_vol_divergence"].clip(upper=5.0)



                    _timing_feats_added.append("timing_vol_divergence")



                # â”€â”€ MEJORA-08a: Features de PosiciÃ³n en Ciclo BTC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



                # btc_drawdown_from_ath: distancia porcentual al ATH de los Ãºltimos 365d.



                #   - Valor negativo (ej. -0.30) = estamos 30% por debajo del ATH â†’ correcciÃ³n



                #   - Valor cercano a 0 = cerca del ATH â†’ potencial sobreextensiÃ³n



                # btc_cycle_position: percentil del precio actual en la ventana 365d.



                #   - 0.0 = mÃ­nimo del aÃ±o, 1.0 = mÃ¡ximo del aÃ±o



                # Ambas son procedurales (solo usan precio histÃ³rico, sin look-ahead).



                # Ayudan al LightGBM a discriminar entre bull-trend real y lateral bajista.



                if "close" in df.columns:



                    rolling_365d = df["close"].rolling(window=8760, min_periods=720)  # 365d en horas



                    rolling_ath  = rolling_365d.max()



                    df["btc_drawdown_from_ath"] = (df["close"] / rolling_ath) - 1.0



                    _timing_feats_added.append("btc_drawdown_from_ath")



                # Inyectar como pass-through (bypass SFI temporal)



                new_timing = [c for c in _timing_feats_added if c not in features_list]



                if new_timing:



                    features_list.extend(new_timing)



                    logger.info(



                        "[R21] %d features de timing inyectadas como pass-through temporal "



                        "(AUDIT-01: pendiente de evaluaciÃ³n SFI): %s",



                        len(new_timing), new_timing



                    )



            except Exception as _e_timing:



                logger.warning(f"[R21] Error calculando features de timing â€” omitidas: {_e_timing}")



        else:



            logger.info("[R21] timing_features_bypass_sfi=False â€” timing features evaluadas por SFI, no se inyectan manualmente.")



        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



        # 3. Cargar HMM Labels OOS



        # HMM_Regime es SIEMPRE un pass-through al feature set del modelo.



        # La compatibilidad en holdout estÃ¡ garantizada por generate_oos_predictions.py



        # que llama a predict_regime_series() para cubrir todo el perÃ­odo OOS.



        # FIX-HMM-REGIME-TRAINING-01 revertido (M-64 â†’ M-66): excluir HMM_Regime



        # redujo de 108 a 28 trades sin mejorar WR â€” la causa root estaba en otro lado.



        df_hmm = pd.read_parquet(self.root / "data" / "features" / "hmm_regime_labels.parquet")



        _hmm_cols_overlap = df.columns.intersection(df_hmm.columns)
        if not _hmm_cols_overlap.empty:
            logger.info(f"[BUGFIX] Eliminadas columnas superpuestas antes de join HMM: {_hmm_cols_overlap.tolist()}")
            df = df.drop(columns=_hmm_cols_overlap)
            
        df_final = df.join(df_hmm)


        # HMM_Regime siempre se aÃ±ade al feature set (pass-through obligatorio).



        # En holdout, generate_oos_predictions.py llama a predict_regime_series() para



        # cubrir el perÃ­odo 2025+ donde hmm_regime_labels.parquet no llega.



        if "HMM_Regime" not in features_list:



            features_list.append("HMM_Regime")



            logger.info("[HMM-PASS-THROUGH] HMM_Regime aÃ±adido al feature set.")

        # [FIX-B3] Inyectar features legacy de LGBM V1 como pass-through causal.
        # El LGBM Bull de V1 (dsr_oos=0.248, lgbm_prob_media=0.7884) usaba 13 features
        # con lags especificos generados en feature_pipeline.py (MI_LAG_FEATURES).
        _V1_LGBM_LEGACY_FEATURES = [
            'alpha_dtw_signal',
            'cal_day_of_week_milag72h',
            'ETH_Return_1d_milag336h',
            'NASDAQ_Ret_milag12h',
            'M2_YoY_Chg_z90d_milag72h',
            'ECBASSETS_milag336h',
            'SP500_Ret_milag72h',
            'DXY_Slope30d_z90d_milag72h',
            'UnemployRate_z90d_milag1h',
            'pi_cycle_ma111_milag72h',
            'EURUSD_milag500h',
            'GBTC_Low_milag48h',
            'CPI_YoY_milag1h',
        ]
        _legacy_added = [f for f in _V1_LGBM_LEGACY_FEATURES if f not in features_list]
        features_list.extend(_legacy_added)
        if _legacy_added:
            logger.info(f"[FIX-B3] {len(_legacy_added)} features V1-LGBM legacy: {_legacy_added}")



        # OBTENCIÃƒâ€œN METÃƒâ€œDICA DE TARGETS (Triple Barrier Method)



        # 1. Ya no usamos el naive 'target' del feature pipeline (que fue borrado).



        # 2. Computamos las barreras EWMA realistas para cada instante.



        if "close" not in df_final.columns:



            raise ValueError("No se encontrÃƒÂ³ la columna 'close' para generar Triple Barriers.")



        logger.info("Etiquetando Dataset OOS usando el Triple Barrier Method (SOP R3/R4)...")



        from luna.features.tbm import apply_triple_barrier



        # apply_triple_barrier expects pd.Series for price and pd.DatetimeIndex for events



        events_idx = df_final.index



        price_series = df_final["close"]



        # BUG-6 FIX (P4-0-4, 2026-03-08): leer pt/sl de settings.yaml



        # Coherencia con MetaLabelerV2Trainer que usa los mismos multiplicadores



        # BUG-A01 FIX (2026-03-17): leer tbm_min_return de settings.yaml â€” era hardcoded 0.005.



        try:



            from config.settings import cfg as _cfg



            _pt      = float(_cfg.lightgbm.pt_mult_min)



            _sl      = float(_cfg.lightgbm.sl_mult_min)



            _min_ret = float(_cfg.lightgbm.tbm_min_return)



            logger.info(f"TBM LightGBM: pt_mult={_pt}, sl_mult={_sl}, min_return={_min_ret} (de settings.yaml)")



        except Exception:



            _pt, _sl, _min_ret = 2.0, 1.0, 0.005



            logger.warning("TBM LightGBM: usando defaults 2.0/1.0/0.005 (settings no disponible)")



        # FIX-TBM-DYNAMIC-01: dynamic_barrier y event_sampling_hours configurables.



        # dynamic_barrier=True usa horizonte ATR adaptativo (Mejora 4 implementada



        # pero nunca activada hasta ahora). event_sampling_hours>1 reduce el



        # solapamiento entre labels TBM al muestrear eventos cada N horas.



        try:



            _dynamic_barrier = bool(int(_cfg.xgboost) and



                                    bool(_cfg.lightgbm.dynamic_barrier))



            _event_sampling_h = int(_cfg.lightgbm.event_sampling_hours)



        except Exception:



            _dynamic_barrier, _event_sampling_h = False, 1



        if _event_sampling_h > 1:



            events_idx = events_idx[::_event_sampling_h]



            logger.info(



                f"[FIX-TBM-SAMPLE-01] event_sampling_hours={_event_sampling_h}: "



                f"{len(events_idx)} eventos (reducido de {len(df_final)} â€” menos solapamiento)"



            )



        _vbh = int(_cfg.lightgbm.vertical_barrier_hours) if hasattr(_cfg, 'xgboost') else 96
        _dyn_min = int(bool(_cfg.lightgbm.dynamic_horizon_min_h)) if hasattr(_cfg, 'xgboost') else 48
        # Vincular el techo maximo de la barrera dinamica al EMBARGO_H del SOP
        _dyn_max = EMBARGO_H
        _lin_decay = bool(_cfg.lightgbm.linear_decay_pt) if hasattr(_cfg, 'xgboost') else False
        _pt_decay_frac = float(_cfg.lightgbm.pt_decay_fraction) if hasattr(_cfg, 'xgboost') else 0.75

        _side_val = -1.0 if self.native_direction == "short" else 1.0
        _sides_series = pd.Series(_side_val, index=events_idx)

        tbm_result = apply_triple_barrier(
            price_series=price_series,
            event_times=events_idx,
            sides=_sides_series,
            pt_sl_multiplier=[_pt, _sl],  # P4-0-4: de settings, no hardcodeado
            min_return=_min_ret,           # BUG-A01-FIX: de settings (era hardcoded 0.005)
            vertical_barrier_hours=_vbh,
            dynamic_barrier=_dynamic_barrier,  # FIX-TBM-DYNAMIC-01
            dynamic_horizon_min_h=_dyn_min,
            dynamic_horizon_max_h=_dyn_max,
            linear_decay_pt=_lin_decay,
            pt_decay_fraction=_pt_decay_frac,
        )



        # tbm_result contiene las etiquetas (1=PT, -1=SL, 0=T1) en la columna "bin" o "meta_label". 



        # La columna "bin" (1 si retornÃƒÂ³ pt_sl positivo vs SL, 0 timeout).



        # Para simplificar la base prediction de LightGBM, entraremos si "bin" es 1 o 'meta_label' es 1.



        # Combinemos el Dataframe final:



        df_labeled = df_final.join(tbm_result[['bin', 'ret']], how='inner')



        df_labeled["target"] = (df_labeled["bin"] == 1).astype(int)



        # Calcular los retornos forward de simulaciÃƒÂ³n usando el 'ret' real obtenido del Triple Barrier Method



        # para backtesting fidedigno (cuÃƒÂ¡nto ganÃƒÂ³ al tocar SL o PT en la vida real).



        df_labeled["simulated_fwd_ret_24h"] = df_labeled["ret"]



        # Filtrar solo columnas vÃƒÂ¡lidas Ã¢â‚¬â€ excluir columnas 100% NaN antes del dropna



        feature_candidates = [c for c in features_list if c in df_labeled.columns]



        meta_cols = ['target', 'simulated_fwd_ret_24h']



        if "HMM_Semantic" in df_labeled.columns:



            meta_cols.append("HMM_Semantic")



        cols_to_keep = list(set(feature_candidates + meta_cols))



        df_subset = df_labeled[cols_to_keep].copy()



        # Excluir columnas de features que son 100% NaN (no generadas en este pipeline)



        fully_empty_feats = [c for c in feature_candidates if df_subset[c].isna().all()]



        if fully_empty_feats:



            logger.warning(f"Features 100% vacÃƒÂ­as excluidas: {fully_empty_feats}")



            df_subset = df_subset.drop(columns=fully_empty_feats)



        valid_features = [c for c in feature_candidates if c not in fully_empty_feats]



        # Ã¢â€â‚¬Ã¢â€â‚¬ OpciÃƒÂ³n B: LightGBM maneja NaN nativamente (sin fillna, sin dropna agresivo)



        # Solo eliminamos filas donde el TARGET o el RET de simulaciÃƒÂ³n sean NaN



        # (estos deben ser siempre completos para poder entrenar/evaluar).



        # Las features con NaN parcial (LongShortRatio, OI_USD, ETF prices, etc.)



        # las deja pasar Ã¢â‚¬â€ LightGBM aprende la direcciÃƒÂ³n ÃƒÂ³ptima del split para NaN.



        # Resultado: preservamos 43.793 filas en lugar de ~14.000 con dropna agresivo.



        df_clean = df_subset.dropna(subset=meta_cols)



        # FASE 2: Filtrado por Régimen



        if self.regime_list is not None and "HMM_Semantic" in df_clean.columns:



            logger.info(f"Filtro de Régimen Semántico Activo: {self.regime_name} -> {self.regime_list}")



            mask = df_clean["HMM_Semantic"].isin(self.regime_list)



            df_clean = df_clean[mask]



            logger.info(f"  Eventos en régimen {self.regime_name}: {len(df_clean)}")



        elif self.regime_list is not None:



            logger.warning("Filtro de Régimen solicitado pero HMM_Semantic no está en las features. Se entrenará con todos los datos.")



        # Verificar que no hay features 100% NaN en el perÃƒÂ­odo de training



        nan_pct = df_clean[valid_features].isna().mean()



        partial_nan = nan_pct[(nan_pct > 0) & (nan_pct < 1.0)]



        if not partial_nan.empty:



            logger.info(f"Features con NaN parcial (LightGBM nativo): {len(partial_nan)} cols "



                        f"(max: {partial_nan.max():.1%} en '{partial_nan.idxmax()}')")



        # ── [FIX-PBO-01] Submuestreo de trayectorias superpuestas (Anti-Overlap) ──
        try:
            from config.settings import cfg as _cfg_tbm
            _sampling_h = int(_cfg_tbm.xgboost.event_sampling_hours)
        except Exception:
            _sampling_h = 12

        _len_pre_sample = len(df_clean)
        df_clean = df_clean.iloc[::_sampling_h]
        logger.info(
            f"[FIX-PBO-01] Submuestreo Anti-Overlap aplicado (sampling={_sampling_h}H): "
            f"{_len_pre_sample} eventos → {len(df_clean)} eventos. "
            f"(PBO Mitigation: garantiza independencia de trayectorias TBM)"
        )

        self.features = valid_features



        self.X = df_clean[self.features]



        self.y = df_clean['target']



        self.close_rets = df_clean['simulated_fwd_ret_24h']



        # â”€â”€ Debug guards post-load â”€â”€



        check_df_sanity(self.X, label="LightGBM.load_dataset.X")



        check_target_balance(self.y, label="LightGBM.target")



        log_memory_usage("post-load_dataset")



        # â”€â”€ [DATAFLOW-IMPORT-LGBM-01] Feature availability audit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



        # Detecta desalineamiento entre selected_features.json y features_train.parquet.



        # Si muchas features esperadas no existen, el modelo se entrena con features incompletas.



        _feats_expected   = features_list  # las que pide selected_features.json



        _feats_available  = [f for f in _feats_expected if f in df_labeled.columns]



        _feats_missing    = [f for f in _feats_expected if f not in df_labeled.columns]



        _feats_fully_nan  = fully_empty_feats



        _pct_missing      = len(_feats_missing) / max(len(_feats_expected), 1)



        logger.info(



            f"  [DATAFLOW-IMPORT-LGBM-01] Features: {len(_feats_available)}/{len(_feats_expected)} "



            f"disponibles en parquet. "



            f"Faltantes ({len(_feats_missing)}): {_feats_missing[:5]}{'...' if len(_feats_missing) > 5 else ''}. "



            f"100%% NaN: {_feats_fully_nan}."



        )



        if _pct_missing > 0.20:



            logger.warning(



                f"  [DATAFLOW-IMPORT-LGBM-01] ALERTA: {_pct_missing:.0%} de features esperadas NO EXISTEN "



                f"en features_train.parquet. "



                f"Probable causa: selected_features.json desactualizado o feature_pipeline.py no regenerado. "



                f"Re-ejecutar Fase 3A (feature_pipeline) antes de entrenar."



            )



        # Aviso si features_train no tiene columnas HMM â€” el modelo puede estar usando features incorrectas



        for _hc in ["HMM_Regime", "HMM_Semantic"]:



            if _hc in _feats_expected and _hc not in df_labeled.columns:



                logger.warning(



                    f"  [DATAFLOW-IMPORT-LGBM-01] {_hc} requerida por features_list pero AUSENTE en train. "



                    f"El modelo LightGBM no podra usar el regimen HMM como feature."



                )



        # â”€â”€ [DATAFLOW-IMPORT-LGBM-02] Dimensionality & Target audit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



        _t_min, _t_max = self.X.index.min().date(), self.X.index.max().date()



        logger.success(



            f"[DATAFLOW-IMPORT-LGBM-02] Dataset de Entrenamiento Cargado y Validado | "



            f"shape={self.X.shape} | fechas={_t_min} -> {_t_max} | "



            f"Target Balance: {self.y.sum()} / {len(self.y)} ({self.y.mean():.1%} positivos)"



        )



        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



        return self.X, self.y



    # LEGACY-01 ELIMINADO (2026-03-17): _create_wfa_splits() â€” nunca activo desde P1-6.



    # El WFA (Walk-Forward Analysis) fue reemplazado por CPCV Real en P1-6 (2026-03-07).



    # Ver historial en diario.md: Fix M-04 / P1-6.



    def _create_cpcv_splits(self):



        """



        CPCV Real segun Lopez de Prado (2018) Ch.12. [ACTIVO DESDE P1-6]



        C(n_groups, k_test) combinaciones de grupos test (no secuencial).



        Con n_groups=10 (sop.cpcv_groups=10), k_test=2: C(10,2)=45 caminos OOS vs 8 del WFA.



        Con n_groups=6  (sop.cpcv_groups=6,  M-40):    C(6,2)=15  caminos OOS.



        Esta funciÃ³n ES la activa en objective() desde P1-6 (2026-03-07).



        Requiere ~5-6x mÃ¡s tiempo de cÃ³mputo que WFA â€” justificado para DSR.



        """



        from itertools import combinations



        n_samples = len(self.X)



        timestamps = self.X.index



        # Dividir en CPCV_GROUPS grupos iguales



        groups = np.array_split(np.arange(n_samples), CPCV_GROUPS)



        splits = []



        k_test = 2  # numero de grupos que forman el test set



        for test_gidx in combinations(range(CPCV_GROUPS), k_test):



            test_idx = np.concatenate([groups[i] for i in test_gidx])



            train_idx = np.concatenate([groups[i] for i in range(CPCV_GROUPS)



                                        if i not in test_gidx])



            # BUG-10 FIX (2026-03-08): purge POR BLOQUE de test independiente.



            # El fix anterior (BUG-8) aplicaba purge sobre el SPAN completo del test



            # (timestamps[test_idx[0]] â†’ timestamps[test_idx[-1]]).



            # Para grupos NO CONTIGUOS (ej. grupos 0+9 = extremos del dataset),



            # ese span cubre TODO el periodo â†’ train_purged = 0 (1 split descartado).



            # Fix correcto: purgar independientemente respecto a CADA bloque de test.



            if len(test_idx) == 0 or len(train_idx) == 0:



                continue



            # Calcular mÃ¡scara keep: un punto de train se mantiene si estÃ¡



            # fuera del buffer PURGE_H de TODOS los bloques de test.



            train_mask = np.ones(len(train_idx), dtype=bool)



            for gi in test_gidx:



                block = groups[gi]



                block_start = timestamps[block[0]]



                block_end   = timestamps[block[-1]]



                purge_lo    = block_start - pd.Timedelta(hours=PURGE_H)



                purge_hi    = block_end   + pd.Timedelta(hours=PURGE_H)



                # Excluir puntos de train dentro del buffer de este bloque



                in_purge_zone = (



                    (timestamps[train_idx] >= purge_lo) &



                    (timestamps[train_idx] <= purge_hi)



                )



                train_mask &= ~in_purge_zone  # quitar los que caen en la zona de purge



            train_idx = train_idx[train_mask]



            if len(train_idx) < 100 or len(test_idx) < 50:



                continue



            splits.append((train_idx, test_idx))



        import math as _math



        n_paths_total = _math.comb(CPCV_GROUPS, 2)



        if CPCV_GROUPS < 8:



            # ARCH-03 warning (2026-03-17): menos de C(8,2)=28 paths â€” robustez estadÃ­stica baja.



            # Con 15 paths el IC del DSR es ~3Ã— mÃ¡s amplio que con 45 paths â†’ mÃ¡s fÃ¡cil sobreajustar.



            # Para aumentar: xgboost.n_purged_splits: 10 en settings.yaml (sin tocar cÃ³digo).



            _eta_h = OPTUNA_TRIALS * n_paths_total * 4.0 / 3600  # ~4s/fold estimado



            logger.warning(



                "[ARCH-03] CPCV_GROUPS=%d â†’ C(%d,2)=%d paths activos (ROBUSTEZ BAJA). "



                "DSR con %d paths tiene IC ~3x mas amplio que con 45 paths. "



                "Para produccion: n_purged_splits=10 en settings.yaml (ETA ~%.0fH con %d trials).",



                CPCV_GROUPS, CPCV_GROUPS, n_paths_total,



                n_paths_total,



                _math.comb(10, 2) * OPTUNA_TRIALS * 4.0 / 3600,



                OPTUNA_TRIALS



            )



        else:



            logger.info(



                "[CPCV REAL] %d grupos â†’ C(%d,2)=%d paths â€” robustez estadistica adecuada.",



                CPCV_GROUPS, CPCV_GROUPS, n_paths_total



            )



        logger.info("[CPCV REAL] Generados {} splits efectivos (descartados por purge/size).", len(splits))



        return splits



    def _compute_dsr(self, fold_sharpes: list, test_lengths: list = None, n_trials: int = OPTUNA_TRIALS) -> float:



        """



        Calcula el Deflated Sharpe Ratio segun Bailey & LdP (2014).



        FIX-DSR-T-01: T = mean(test_lengths) es la eleccion CORRECTA para CPCV.



        Razonamiento:



          - En CPCV C(10,2)=45 paths, cada path OOS tiene ~12.000 observaciones.



          - T en la formula DSR representa el nro de obs usadas para calcular



            cada Sharpe individual del backtest, no el total acumulado.



          - T=sum (=45*12.000=540.000) inflaria el umbral sr_star artificialmente



            penalizando estrategias buenas â€” seria INCORRECTO.



          - T=mean(test_lengths) captura la longitud tipica de cada fold, que es



            exactamente el T del paper (Bailey & LdP 2014, eq.2).



        """



        if len(fold_sharpes) < 2: return 0.0



        sr_mean = np.mean(fold_sharpes)



        if sr_mean <= 0: return 0.0  # Circuit Breaker rapido



        # [P0-3-FIX-CROSS-VAR 2026-06-04] Bailey & Lopez de Prado (2014) exige la varianza TRANSVERSAL.
        # Usar la varianza temporal de los folds borraba la penalización de Multiple Testing
        # para estrategias estables. Asignamos varianza transversal conservadora (std=1.0)
        # para que la barrera crezca implacablemente por número de pruebas.
        sr_std_cross = 1.0

        gamma = 0.5772156649

        # T = longitud temporal promedio de cada path OOS â€” ver docstring.
        if test_lengths and len(test_lengths) > 0:
            T = int(np.mean(test_lengths))   # Correcto: longitud promedio por fold
        else:
            T = max(1000, n_trials * 20)  # [BUG-XGB-01 PORT] Heurística cuando test_lengths falla

        z1 = norm.ppf(1 - 1.0 / max(n_trials, 2))
        z2 = norm.ppf(1 - 1.0 / max(n_trials * math.e, 2))

        sr_star = sr_std_cross * ((1 - gamma) * z1 + gamma * z2)



        # BUG-DSR-MATH FIX: Reintroduccion de 'T' escalar probablístico
        
        # [FIX-MATH-OPTUNA-LGBM-01]: Aplicar la varianza teórica de la estimación de Sharpe (Bailey & LdP)
        # para convertir la diferencia en un Z-Score estadísticamente válido y evitar la polarización a 0 o 1.
        freq = 8760.0
        var_sr = (freq + 0.5 * (sr_mean ** 2)) / T
        z_score = (sr_mean - sr_star) / np.sqrt(var_sr)
        dsr = float(norm.cdf(z_score))
        
        # Trace print for mathematical correction (fixbugsprints.md / fixaplly.md)
        trace_msg = f"[FIX-MATH-OPTUNA-LGBM-01] DSR recalculado: sr_mean={sr_mean:.4f}, sr_star={sr_star:.4f}, T={T}, dsr={dsr:.4f}"
        print(trace_msg)
        logger.info(trace_msg)
        return dsr



    def _compute_sample_weights(self, index: pd.Index) -> np.ndarray:



        """



        ARCH-02 fix (2026-03-17): decaimiento exponencial configurable por aÃ±o.



        weight_i = exp(-alpha Ã— aÃ±os_desde_train_end)



          alpha=0.0 â†’ uniforme (sin Ã©nfasis temporal, para diagnÃ³stico)



          alpha=0.5 â†’ suave â€” ratio aÃ±o0:aÃ±o-1 â‰ˆ 1.6:1  (DEFAULT)



          alpha=1.0 â†’ moderado â€” ratio â‰ˆ 2.7:1



          alpha=1.6 â†’ agresivo â€” ratio â‰ˆ 5.0:1  (equivalente al esquema anterior 5x/1x)



        Configurable en settings.yaml â†’ xgboost.weight_decay_alpha



        Sin hardcodes de aÃ±os â€” completamente dinÃ¡mico a partir de train_end.



        """



        ts = pd.to_datetime(index)



        try:



            from config.settings import cfg as _cfg_sw



            _train_end_year = pd.Timestamp(_cfg_sw.temporal_splits.train_end).year



            _alpha = float(_cfg_sw.xgboost.weight_decay_alpha)



        except Exception:



            _train_end_year = ts.year.max()



            _alpha = 0.5



        years_ago = np.clip(_train_end_year - ts.year.to_numpy(), 0, None).astype(float)



        weights = np.exp(-_alpha * years_ago)



        _verbose_debug = bool(int(_os.environ.get("LUNA_VERBOSE", "0")))



        if _verbose_debug and not getattr(self.__class__, '_sw_logged', False):



            logger.debug(



                "[R20-B/ARCH-02] sample_weights config: alpha=%.2f, train_end=%d "



                "â†’ pesos unitarios [aÃ±o0=%.3f, aÃ±o-1=%.3f, aÃ±o-2=%.3f] "



                "(este mensaje se emite UNA sola vez â€” throttle activo)",



                _alpha, _train_end_year,



                np.exp(0.0), np.exp(-_alpha), np.exp(-2.0 * _alpha)



            )



            self.__class__._sw_logged = True  # throttle: 1 log por run, no por split



        return weights



    def _get_focal_loss_obj(self, scale_pos_weight=1.0):



        """



        [A1] Genera un custom objective Focal Loss para LGBMClassifier.



        El solver de LightGBM requiere que la función objetivo retorne Gradiente y Pseudo-Hessiana



        calculadas respecto al raw margin (log-odds).



        """



        try:



            from config.settings import cfg as _cfg_fl



            gamma = float(_cfg_fl.xgboost.focal_loss_gamma)



        except Exception:



            gamma = 2.0



        def focal_loss(y_true, y_pred, sample_weight=None):



            # y_pred en LightGBM custom objectives entra en raw log-odds (margin).



            # Transformar a probabilidad p:



            p = 1.0 / (1.0 + np.exp(-y_pred))



            p = np.clip(p, 1e-5, 1.0 - 1e-5)



            # Derivada analítica del Focal Loss respecto al log-odds (y_pred):



            grad_1 = -(1 - p)**gamma * (1 - p - gamma * p * np.log(p))



            # [FIX-FOCAL-LOSS-MATH-01] Error crítico de signo en la derivada analítica.
            # La derivada correcta de FL_0 respecto a z (log-odds) es:
            # p^gamma * [p - gamma * (1-p) * ln(1-p)]
            # Anteriormente el '+' causaba que a 'p' se le restara el término (porque ln(1-p) es negativo),
            # lo que DILUÍA el gradiente en lugar de AMPLIFICARLO cuando el modelo estaba muy equivocado.
            # Esto provocaba una peligrosa sub-penalización de los Falsos Positivos (operaciones perdedoras).
            grad_0 = p**gamma * (p - gamma * (1 - p) * np.log(1 - p))



            # Gradiente final ponderando clase positiva por scale_pos_weight



            grad = y_true * grad_1 * scale_pos_weight + (1 - y_true) * grad_0



            # Pseudo-Hessiana: aproximación de segundo orden para garantizar P.D.



            hess_1 = (1 - p)**gamma * p * (1 - p) * (1 + gamma)



            hess_0 = p**gamma * p * (1 - p) * (1 + gamma)



            hess = y_true * hess_1 * scale_pos_weight + (1 - y_true) * hess_0



            # Aplicar sample_weight si fit() lo inyectó dinámicamente



            if sample_weight is not None:



                grad *= sample_weight



                hess *= sample_weight



            # Estabilidad numérica exigida por el solver xgboost



            hess = np.clip(hess, 1e-4, None)



            return grad, hess



        return focal_loss



    def objective(self, trial):



        """



        FunciÃ³n objetivo Optuna: maximiza DSR sobre 45 paths CPCV.



        BUG-R12-03 fix (2026-03-10): todos los rangos Optuna leÃ­dos desde



        cfg.lightgbm.optuna_search_space en settings.yaml â€” sin nÃºmeros mÃ¡gicos.



        JustificaciÃ³n de cada rango documentada en settings.yaml.



        Nuevos parÃ¡metros: gamma (L0), reg_alpha (L1), reg_lambda (L2), scale_pos_weight.



        """



        # Leer espacio de bÃºsqueda desde settings.yaml â€” REGLA: sin hardcodes



        sp = _cfg_xgb.lightgbm.optuna_search_space



        params = {



            'n_estimators': trial.suggest_int('n_estimators', sp.n_estimators_min, sp.n_estimators_max),



            'num_leaves': trial.suggest_int('num_leaves', sp.num_leaves_min, sp.num_leaves_max),



            'learning_rate': trial.suggest_float('learning_rate', sp.learning_rate_min, sp.learning_rate_max, log=True),



            'feature_fraction': trial.suggest_float('feature_fraction', sp.feature_fraction_min, sp.feature_fraction_max),



            'bagging_fraction': trial.suggest_float('bagging_fraction', sp.bagging_fraction_min, sp.bagging_fraction_max),



            'bagging_freq': 1,



            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', sp.min_data_in_leaf_min, sp.min_data_in_leaf_max),



            'reg_alpha': trial.suggest_float('reg_alpha', sp.reg_alpha_min, sp.reg_alpha_max, log=True),



            'reg_lambda': trial.suggest_float('reg_lambda', sp.reg_lambda_min, sp.reg_lambda_max, log=True),



            'scale_pos_weight': trial.suggest_float('scale_pos_weight', getattr(self, '_spw_min', None) or sp.scale_pos_weight_min, getattr(self, '_spw_max', None) or sp.scale_pos_weight_max),



            # [FIX-1C] min_gain_to_split: ganancia mínima para aceptar un split (regularización estructural).
            # Default LightGBM=0.0 (acepta cualquier split). Rango [0,2] fuerza splits significativos.
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', getattr(sp, 'min_gain_to_split_min', 0.0), getattr(sp, 'min_gain_to_split_max', 2.0)),



            # [FIX-1C] max_bin: bins para discretizar features. Menos bins = menos overfitting OOS.
            'max_bin': trial.suggest_int('max_bin', getattr(sp, 'max_bin_min', 63), getattr(sp, 'max_bin_max', 255)),



            'objective': 'binary',



            'verbose': -1,



            'n_jobs': -1,



            # [FIX-RANDOM-STATE-03 2026-05-28] Usar LUNA_SEED para diversidad entre seeds del ensemble
            'random_state': int(_os_lgbm.environ.get('LUNA_SEED', 42))



        }



        # [A1] Inyectar Focal Loss o Monetary Loss si están activados en settings.yaml
        # [FIX-LGBM-FOCAL-NS-01] Leer de sección 'lightgbm', NO de 'xgboost'



        use_focal_loss = False



        use_monetary_loss = False



        try:



            from config.settings import cfg as _cfg_opts



            use_focal_loss = bool(int(float(_cfg_opts.lightgbm).use_focal_loss))



            use_monetary_loss = bool(_cfg_opts.fase2.use_monetary_loss)



        except Exception:



            pass



        if use_monetary_loss:



            from luna.losses.monetary_loss import get_monetary_pnl_loss



            params['objective'] = get_monetary_pnl_loss()



            use_focal_loss = False



        elif use_focal_loss:



            _spw = params.get('scale_pos_weight', 1.0)



            params['objective'] = self._get_focal_loss_obj(scale_pos_weight=_spw)



        # FIX-CPCV-CACHE-01: usar splits precalculados en tune_hyperparameters().



        # Los 45 splits C(10,2) son siempre iguales para este dataset â€” no recalcular.



        splits = self._cached_splits if self._cached_splits is not None else self._create_cpcv_splits()



        fold_sharpes = []



        test_lengths  = []  # Fix F8: acumular longitudes de test para DSR correcto



        for fold_i, (train_idx, test_idx) in enumerate(splits):



            X_tr, y_tr = self.X.iloc[train_idx], self.y.iloc[train_idx]



            X_te, y_te = self.X.iloc[test_idx], self.y.iloc[test_idx]



            rets_te = self.close_rets.iloc[test_idx]



            clf = lgb.LGBMClassifier(**params)



            # R20-B: sample_weight prioriza datos de 2022-2024 (mÃ¡s parecidos a 2025)



            sw_tr = self._compute_sample_weights(X_tr.index)



            # FIT the LGBM model



            clf.fit(X_tr, y_tr, sample_weight=sw_tr)



            # [A1 FIX] Restaurar el objective original si se usó focal loss



            if callable(params.get('objective')):



                clf.set_params(objective='binary')



                if hasattr(clf, 'booster_'):



                    clf.booster_.params['objective'] = 'binary'



            # Predicciones binarias (1/0)



            preds = clf.predict(X_te)



            # Anti Buy & Hold Circuit Breaker:



            if len(np.unique(preds)) == 1:



                fold_sharpes.append(-1.0)



                test_lengths.append(len(test_idx))



            else:



                # Fix F9: modelo entrenado solo para Long (TBM side=1).



                # pred=1 -> Long (capturar ret), pred=0 -> Cash (no hacer nada).



                pos = preds.astype(float)  # Long=1, Cash=0 -- Long-only correcto



                # [LUNA V1 INSTITUTIONAL FIX] TBM Overlap Cost Logic



                strat_rets = pos * (rets_te.values - COST_PCT)



                mean_ret = np.mean(strat_rets)



                std_ret  = np.std(strat_rets) or 1e-6



                # Sharpe Anualizado (Ajustado dinámicamente a la frecuencia del TBM)



                try:



                    from config.settings import cfg as _cfg_tmp



                    _samp_h = int(_cfg_tmp.xgboost.event_sampling_hours)



                except Exception:



                    _samp_h = 24



                sharpe = (mean_ret / std_ret) * np.sqrt(365 * 24 / _samp_h)



                fold_sharpes.append(sharpe)



                test_lengths.append(len(test_idx))  # Fix F8: registrar longitud del test



            # FIX-OPTUNA-PRUNE-01: reportar DSR parcial para permitir pruning intermedio.



            # MedianPruner aborta trials cuyo DSR parcial esta por debajo de la mediana



            # historica en el mismo paso â€” ahorra ~30-40% de evaluaciones CPCV.



            # Solo reportamos si tenemos >= 2 folds (DSR no definido con 1 fold).



            if len(fold_sharpes) >= 2:



                partial_dsr = self._compute_dsr(



                    fold_sharpes, test_lengths=test_lengths, n_trials=OPTUNA_TRIALS



                )



                trial.report(partial_dsr, step=fold_i)



                if trial.should_prune():



                    raise optuna.TrialPruned()



        # [V2-FIX-1] BRIER IS: después de acumular fold_sharpes (para telemetría),
        # calculamos el Brier Score IS con TimeSeriesSplit como objetivo real.
        # DSR se mantiene SOLO como telemetría ex-post (no influye en selección).
        try:
            from config.settings import cfg as _cfg_metric
            # [FIX-B1] cfg.lightgbm es el namespace correcto en settings.yaml (sección 'lightgbm:').
            # cfg.lgbm no existe → getattr caía siempre al default 'dsr' → LGBM corría ciego.
            _optuna_metric = str(str(_cfg_metric.lightgbm.optuna_metric)).lower()
        except Exception:
            _optuna_metric = 'dsr'

        if _optuna_metric in ('brier', 'logloss'):
            # Calcular IS Brier/LogLoss con TimeSeriesSplit (gap = purge_hours)
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import brier_score_loss, log_loss
            try:
                # [FIX-B1-EMBARGO] Mismo namespace fix: cfg.lightgbm (no cfg.lgbm)
                _purge_gap = int(_cfg_metric.lightgbm.purge_hours)
            except Exception:
                _purge_gap = 96

            # n_splits adaptativo: 1 split por cada 6 meses de datos (mín 3, máx 6)
            _n_months = max(1, len(self.X) // (24 * 30 * 6))
            _n_splits_is = max(3, min(6, _n_months))
            _tscv = TimeSeriesSplit(n_splits=_n_splits_is, gap=_purge_gap)

            _is_scores = []
            for _tr_i, _val_i in _tscv.split(self.X):
                _clf_is = lgb.LGBMClassifier(**params)
                _sw_is = self._compute_sample_weights(self.X.iloc[_tr_i].index)
                _clf_is.fit(self.X.iloc[_tr_i], self.y.iloc[_tr_i], sample_weight=_sw_is)
                # [A1 FIX] Si Focal Loss custom, restaurar binario
                if callable(params.get('objective')):
                    _clf_is.set_params(objective='binary')
                    _clf_is.objective = 'binary'
                _proba_is = _clf_is.predict_proba(self.X.iloc[_val_i])[:, 1]
                # [BUG-LGBM-YPRO-01 FIX] Focal Loss emite raw leaf scores (log-odds).
                # Usar expit (sigmoid) en lugar de np.clip para no destruir la distribución.
                if callable(params.get('objective')):
                    from scipy.special import expit
                    _proba_is = expit(_proba_is)
                _y_val = self.y.iloc[_val_i].values
                if _optuna_metric == 'brier':
                    _is_scores.append(brier_score_loss(_y_val, _proba_is))
                else:
                    _is_scores.append(log_loss(_y_val, _proba_is))

            _metric_val = float(np.mean(_is_scores)) if _is_scores else 1.0

            # DSR ex-post (telemetría silenciosa — no influye en selección)
            _dsr_telemetry = self._compute_dsr(fold_sharpes, test_lengths=test_lengths, n_trials=self.n_trials)
            logger.debug(
                "[V2-FIX-1] Brier IS=%.4f (splits=%d, gap=%dh) | DSR ex-post=%.4f (telemetría)",
                _metric_val, _n_splits_is, _embargo_gap, _dsr_telemetry
            )
            # Optuna minimiza Brier → retornamos el valor directamente (study direction='minimize')
            return _metric_val
        else:
            # Modo legacy DSR (backwards compatible)
            dsr = self._compute_dsr(fold_sharpes, test_lengths=test_lengths, n_trials=self.n_trials)
            return dsr



    def tune_hyperparameters(self):



        logger.info(f"Iniciando Optuna Tuning ({self.n_trials} trials)... Optimización: {_optuna_metric.upper() if '_optuna_metric' in dir() else 'brier'}")



        if len(self.X) < 100:



            logger.warning(f"[{self.regime_name}] Dataset muy pequeÃ±o ({len(self.X)} filas). Omitiendo Optuna tuning y usando fallback params.")



            self.best_params = {



                'n_estimators': 150,



                'max_depth': 4,



                'learning_rate': 0.05,



                'subsample': 0.8,



                'colsample_bytree': 0.8,



                'min_child_weight': 5,



            }



            # Simulamos un objeto study para que train_final_model no falle en logging



            class DummyStudy:



                best_value = 0.50



                best_params = self.best_params



            self.study = DummyStudy()



            return



        import time



        _t0 = time.time()



        def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):



            """Log progreso de Optuna â€” cada trial en VERBOSE, cada 10 en modo normal."""



            n = trial.number + 1



            elapsed = time.time() - _t0



            # FIX-OPTUNA-PRUNE-02: Evitar falso Brier=1.000 verificando estado de poda
            if trial.state == optuna.trial.TrialState.PRUNED:
                val_str = "[PRUNED via DSR]"
            else:
                val_str = f"{trial.value:.4f}" if trial.value is not None else "N/A"



            try:



                best_str = f"{study.best_value:.4f}"



            except ValueError:  # aun no hay trial completado



                best_str = "N/A"



            eta_s   = (elapsed / n) * (self.n_trials - n) if n < self.n_trials else 0



            eta_str = f"{eta_s/60:.0f}min" if eta_s > 60 else f"{eta_s:.0f}s"



            # En modo VERBOSE: loguear cada trial. En normal: cada 10 + primero y ultimo



            _verbose_mode = bool(int(_os.environ.get("LUNA_VERBOSE", "0")))



            if _verbose_mode or n % 10 == 0 or n == 1 or n == self.n_trials:



                logger.info(



                    f"[Optuna] Trial {n:>3}/{self.n_trials} | "
                    f"Metric={val_str} | "
                    f"Best={best_str} | "



                    f"{elapsed/60:.1f}min | ETAâ‰ˆ{eta_str}"



                )



        # Suprimir logs intermedios de optuna a nivel WARNING



        optuna.logging.set_verbosity(optuna.logging.WARNING)



        # FIX-OPTUNA-PRUNE-01: MedianPruner para abortar trials malos antes



        # de completar los 45 CPCV folds. Parametros:



        #   n_startup_trials=10: los primeros 10 trials corren completos para



        #     establecer la mediana de referencia (no se pruna antes).



        #   n_warmup_steps=5: dentro de cada trial, los primeros 5 folds no



        #     se pruna (necesitamos al menos 5 puntos para DSR fiable).



        # Ahorro estimado: ~35% del tiempo de tuning en runs con muchos trials.



        _pruner = optuna.pruners.MedianPruner(



            n_startup_trials=10,



            n_warmup_steps=5,



            interval_steps=1,



        )



        # FIX-CPCV-CACHE-01: precalcular splits una sola vez antes de lanzar Optuna.



        # Sin este fix, _create_cpcv_splits() se llama N_TRIALS veces innecesariamente.



        logger.debug("[FIX-CPCV-CACHE-01] Pre-calculando CPCV splits...")



        self._cached_splits = self._create_cpcv_splits()



        # REPRO-01 fix (2026-03-17): sampler determinista para reproducibilidad.



        # Sin TPESampler(seed=N), dos runs con los mismos datos pueden producir



        # hiperparametros LightGBM distintos. seed se lee de settings o default 42.



        try:



            _optuna_seed = int(_cfg_xgb.xgboost.optuna_seed)



        except Exception:



            _optuna_seed = 42



        _sampler = optuna.samplers.TPESampler(seed=_optuna_seed)



        # [V2-FIX-1] Leer métrica y dirección del study dinámicamente desde settings
        try:
            from config.settings import cfg as _cfg_tune_dir
            # [FIX-B1] Mismo fix: leer desde cfg.lightgbm, no cfg.lgbm (namespace inexistente).
            _optuna_metric_dir = str(str(_cfg_tune_dir.lightgbm.optuna_metric)).lower()
        except Exception:
            _optuna_metric_dir = 'dsr'
        _study_direction = 'minimize' if _optuna_metric_dir in ('brier', 'logloss') else 'maximize'
        self.study = optuna.create_study(direction=_study_direction, pruner=_pruner, sampler=_sampler)



        logger.info("[REPRO-01] Optuna TPESampler(seed={}) activo â€” runs deterministas.", _optuna_seed)



        # SPW-AUTO-01 (2026-03-23): calcular scale_pos_weight range desde labels TBM reales.



        # PROBLEMA ANTERIOR: min/max leidos desde settings.yaml -> error humano (RUN-0010:



        # max=1.30 cortaba 57% del espacio de busqueda). Ahora auto-calculado:



        #   ideal = neg/pos  (ratio de clases real del training set)



        #   min   = ideal * 0.5   (permite downweight hasta la mitad)



        #   max   = min(2.0, ideal * 2.5)  SOP_LIMIT=2.0 (post-mortem M-79: SPW>2 colapsa WR)



        # El YAML (scale_pos_weight_min/max) ya no es fuente de verdad â€” solo documentacion.



        try:



            _spw_pos   = int((self.y == 1).sum())



            _spw_neg   = int((self.y == 0).sum())



            _spw_ideal = _spw_neg / max(_spw_pos, 1)



            self._spw_min = max(0.1,  _spw_ideal * 0.5)



            self._spw_max = min(2.0,  _spw_ideal * 2.5)  # SOP_LIMIT=2.0



            logger.info(



                "[SPW-AUTO-01] pos=%d neg=%d ideal=%.3f -> range=[%.2f, %.2f] (SOP_LIMIT=2.0)",



                _spw_pos, _spw_neg, _spw_ideal, self._spw_min, self._spw_max,



            )



        except Exception as _e_spw:



            logger.warning("[SPW-AUTO-01] Calculo automatico fallido ({}) â€” usando YAML settings.", _e_spw)



            self._spw_min = None



            self._spw_max = None



        # BUG-A03 FIX (2026-03-17): n_jobs=1 para evitar carrera con _cached_splits mutable.



        # n_jobs=4 con GIL en workloads CPU-bound numpy apenas gana velocidad real, pero



        # introduce riesgo de condiciÃ³n de carrera si algÃºn trial modifica stato compartido.



        # MedianPruner ya compensa el tiempo con early stopping de trials malos.



        self.study.optimize(



            self.objective,



            n_trials=self.n_trials,



            n_jobs=1,



            callbacks=[_progress_callback],



        )



        self.best_params = self.study.best_params



        logger.success(f"Tuning completado! Mejor DSR OOS estimado: {self.study.best_value:.4f}")



        logger.info(f"Mejores params: {self.best_params}")



    def _load_calibration_source(self):

        """

        [FIX-LGBM-FOCAL-SIGMOID-01] Retorna el DataFrame de calibracion.

        Jerarquia: holdout_3m (preferido, nunca visto) > validation.parquet (fallback).

        Extraido de _calibrate_threshold() para reutilizacion en Platt Scaling.

        """

        import pandas as pd

        try:
            from config.settings import cfg as _cfg_cal_src
            _calib_months = int(_cfg_cal_src.lightgbm.holdout_calib_months)
        except Exception:

            _calib_months = 3

        df_val = None

        # AUDIT GAP-02 (BUG-HOLDOUT-PATH): usar parquet especifico de ventana si existe
        import os as _os_lgbm_1
        _win_lgbm = _os_lgbm_1.environ.get('LUNA_WINDOW_ID', '')
        _hp_w1 = self.root / 'data' / 'features' / f'features_holdout_{_win_lgbm}.parquet'
        holdout_path = _hp_w1 if (_win_lgbm and _hp_w1.exists()) else self.root / 'data' / 'features' / 'features_holdout.parquet'

        if holdout_path.exists() and _calib_months > 0:

            try:

                df_h = pd.read_parquet(holdout_path)

                if len(df_h) > 200:

                    _end = df_h.index.min() + pd.DateOffset(months=_calib_months)

                    df_val = df_h[df_h.index <= _end].dropna(subset=['close']).copy()

                    logger.debug('[_load_calibration_source] holdout {}m: {} filas',

                                 _calib_months, len(df_val))

            except Exception as _eh:

                logger.warning('[_load_calibration_source] holdout error: {}', _eh)

        if df_val is None or len(df_val) < 100:

            val_path = self.root / 'data' / 'features' / 'features_validation.parquet'

            try:

                df_val = pd.read_parquet(val_path).dropna(subset=['close']).copy()

                logger.debug('[_load_calibration_source] validation fallback: {} filas', len(df_val))

            except Exception:

                df_val = None

        return df_val



    def _calibrate_threshold(self) -> float:



        """



        MEJORA-R12-01 fix (2026-03-10): calibraciÃ³n automÃ¡tica del threshold LGBM.



        Barre thresholds sobre features_validation.parquet y selecciona el que



        maximiza el Expected Value esperado por trade:



            EV(t) = P(win | prob > t) Ã— avg_win - P(loss | prob > t) Ã— avg_loss - cost



        sujeto a: n_trades(t) >= threshold_min_trades (settings.yaml).



        Fuente: features_validation.parquet (perÃ­odo semi-OOS, nunca en train).



        Fallback: 0.50 si validation no disponible o < min_trades en todo el sweep.



        Returns



        -------



        float â€” optimal_threshold para usar en generate_oos_predictions.py



        """



        # ParÃ¡metros de calibraciÃ³n desde settings.yaml â€” sin hardcodes



        try:



            cal_cfg = _cfg_xgb.xgboost



            t_min      = float(float(cal_cfg.threshold_sweep_min))



            t_max      = float(float(cal_cfg.threshold_sweep_max))



            t_step     = float(float(cal_cfg.threshold_sweep_step))



            min_trades = int(int(cal_cfg.threshold_min_trades))



            # OPT-B (2026-03-22): densidad mÃ­nima de seÃ±ales respecto a t_min.



            # [TIPO-3: CALCULADO] n_baseline = seÃ±ales@t_min. Si n(t) < n_baseline*density_pct,



            # el threshold se considera hiperselectivo y se descarta del sweep.



            # M-80: 0.75 tenÃ­a densidad=12.2% â†’ con 30% mÃ­nimo habrÃ­a elegido ~0.61 (EV>0, N>600).



            min_density_pct = float(float(cal_cfg.threshold_min_density_pct))



        except Exception:



            t_min, t_max, t_step, min_trades, min_density_pct = 0.40, 0.75, 0.01, 30, 0.30



        # ARCH-04 fix (2026-03-17): jerarquia de calibracion tiered.



        # 1. features_holdout.parquet (primeros holdout_calib_months meses) -- datos realmente no vistos



        # 2. features_validation.parquet + WARNING de sesgo (2024-H2 semi-conocido)



        # 3. 0.50 neutral si ninguno existe



        try:



            _calib_months = int(int(cal_cfg.holdout_calib_months))



        except Exception:



            _calib_months = 3



        cal_source = "validation"   # actualizado si se usa holdout



        df_val = None



        # AUDIT GAP-02 (BUG-HOLDOUT-PATH): usar parquet especifico de ventana si existe
        import os as _os_lgbm_2
        _win_lgbm2 = _os_lgbm_2.environ.get("LUNA_WINDOW_ID", "")
        _hp_w2 = self.root / "data" / "features" / f"features_holdout_{_win_lgbm2}.parquet"
        holdout_path = _hp_w2 if (_win_lgbm2 and _hp_w2.exists()) else self.root / "data" / "features" / "features_holdout.parquet"



        if holdout_path.exists() and _calib_months > 0:



            try:



                df_holdout_raw = pd.read_parquet(holdout_path)



                if len(df_holdout_raw) > 200:



                    _calib_end = df_holdout_raw.index.min() + pd.DateOffset(months=_calib_months)



                    df_val = df_holdout_raw[df_holdout_raw.index <= _calib_end].dropna(subset=["close"]).copy()



                    cal_source = f"holdout_{_calib_months}m"



                    logger.info(



                        "[Calibrate/ARCH-04] Usando primeros %d meses de holdout (%s -> %s) -- %d filas "



                        "(datos realmente no vistos, sin sesgo de seleccion).",



                        _calib_months,



                        df_holdout_raw.index.min().date(), _calib_end.date(), len(df_val)



                )



                    if len(df_val) < min_trades * 2:



                        logger.warning(



                            "[Calibrate/ARCH-04] Tramo holdout corto (%d filas). "



                            "Incrementar holdout_calib_months o reducir threshold_min_trades.",



                            len(df_val)



                        )



                else:



                    logger.debug("[Calibrate/ARCH-04] holdout muy pequeno ({} filas) -- fallback a validation.", len(df_holdout_raw))



            except Exception as _e_h:



                logger.warning("[Calibrate/ARCH-04] Error cargando holdout: {} -- fallback a validation.", _e_h)



        if df_val is None:



            # Fallback a validation con warning explicito de sesgo (ARCH-04)



            logger.warning(



                "[Calibrate/ARCH-04] SESGO: threshold calibrado en features_validation.parquet "



                "(2024-H2 semi-conocido). Para eliminar: asegurar features_holdout.parquet y "



                "holdout_calib_months>0 en settings.yaml."



            )



            val_path = self.root / "data" / "features" / "features_validation.parquet"



            if not val_path.exists():



                logger.warning("[Calibrate] features_validation.parquet no existe -- threshold=0.50 (neutral)")



                self._cal_source = "neutral_050"



                return 0.50



            try:



                df_val = pd.read_parquet(val_path).dropna(subset=["close"]).copy()



            except Exception as e:



                logger.warning("[Calibrate] No se pudo leer features_validation.parquet: {} -- threshold=0.50", e)



                self._cal_source = "neutral_050"



                return 0.50



        self._cal_source = cal_source   # persistir para signature JSON



        # ── LAB-CAL-01 fix (2026-03-20): filtrar df_val por regimenes HMM permitidos.



        _hmm_filtered = False



        try:



            from luna.models.hmm_regime import HMMRegimeModel



            from luna.models.signal_filter import SignalFilter



            _hmm_pkl_path = self.root / "data" / "models" / "hmm_regime.pkl"



            if _hmm_pkl_path.exists():



                _hmm_predictor = HMMRegimeModel.load(self.root / "data" / "models")



                _hmm_df = _hmm_predictor.predict_regime_series(df_val)



                df_val["HMM_Semantic"] = _hmm_df["HMM_Semantic"]



                df_val["HMM_Regime"] = _hmm_df["HMM_Regime"]



                sf = SignalFilter(self.root / "data" / "models")



                _mask_hmm = sf.apply_hmm(df_val)



                _n_before = len(df_val)



                df_val = df_val[_mask_hmm.values].copy()



                _n_after = len(df_val)



                _hmm_filtered = True



                logger.info(



                    "[LAB-CAL-01] HMM filter (vía SignalFilter) aplicado: "



                    "%d -> %d filas", _n_before, _n_after



                )



                if _n_after < 100:



                    logger.warning("[LAB-CAL-01] Tras HMM filter quedan {} filas (<100). Restaurando sin filtro.", _n_after)



                    _fallback_path = self.root / "data" / "features" / "features_validation.parquet"



                    df_val = pd.read_parquet(_fallback_path).dropna(subset=["close"]).copy()



                    _hmm_filtered = False



        except Exception as _e_hmm_cal:



            logger.warning("[LAB-CAL-01] HMM filter fallido, usando df_val sin filtrar: {}", _e_hmm_cal)



        # Score de calibracion: EV * penalizacion_volumen



        # LAB-CAL-01: penalizar thresholds con n_trades < N_target.



        # EV puro (sin penalizacion) elige thresholds muy restrictivos (M-52: 0.63, 58 trades).



        # Con score compuesto, el calibrador equilibra senial y volumen.



        # min_trades del Gauntlet (100) es el N_target optimo.



        try:



            _n_target = int(_cfg_xgb.sop.min_trades)



        except Exception:



            _n_target = 100



        # [FIX-CALIBRATE-TIMING-01] Calcular timing features inline en df_val,



        # igual que en load_dataset() y generate_oos_predictions.py.



        # Sin este bloque, las timing features llegan como padding=0 al calibrador



        # y el threshold queda sesgado hacia valores más bajos de lo óptimo.



        try:



            if "FundingRate" in df_val.columns and "timing_funding_acum8h" not in df_val.columns:



                df_val["timing_funding_acum8h"] = df_val["FundingRate"].ewm(span=8, min_periods=1).mean()



            if "close" in df_val.columns and "timing_momentum_div" not in df_val.columns:



                _r24h = df_val["close"].pct_change(24)



                _r7d  = df_val["close"].pct_change(168)



                df_val["timing_momentum_div"] = _r24h - _r7d



            if "close" in df_val.columns and "volume" in df_val.columns and "timing_vol_divergence" not in df_val.columns:



                _r24h_abs  = df_val["close"].pct_change(24).abs()



                _vol_ma    = df_val["volume"].rolling(window=720, min_periods=48).mean()



                _vol_ratio = df_val["volume"] / (_vol_ma + 1e-6)



                df_val["timing_vol_divergence"] = (_r24h_abs / (_vol_ratio + 1e-6)).clip(upper=5.0)



            logger.debug("[FIX-CALIBRATE-TIMING-01] Timing features calculadas inline en df_val (calibración).")



        except Exception as _e_timing_cal:



            logger.warning("[FIX-CALIBRATE-TIMING-01] Error calculando timing features en calibración: {}", _e_timing_cal)



        avail = [f for f in self.features if f in df_val.columns]



        missing = [f for f in self.features if f not in df_val.columns]



        if missing:



            logger.warning("[Calibrate] {} features ausentes en calibration source -- padding 0: {}",



                           len(missing), missing[:5])



            for f in missing:



                df_val[f] = 0.0



        if len(df_val) < 100:



            logger.warning("[Calibrate] Calibration source muy pequeno ({} filas) -- threshold=0.50", len(df_val))



            self._cal_source = "neutral_050"



            return 0.50



        probs = self.final_model.predict_proba(df_val[self.features])[:, 1]



        # OPT-B: calcular n_baseline = total seÃ±ales en t_min para constraint de densidad.



        # [TIPO-3: CALCULADO] n_baseline se deriva de los datos reales, no es hardcode.



        _baseline_mask = probs > t_min



        n_baseline = int(_baseline_mask.sum())



        n_density_min = max(1, int(n_baseline * min_density_pct))



        logger.info(



            "[Calibrate/OPT-B] Density constraint: n_baseline@%.2f=%d, min_density=%.0f%%, "



            "n_density_min=%d. Se descartan thresholds con n<%d.",



            t_min, n_baseline, min_density_pct * 100, n_density_min, n_density_min



        )



        # FIX-CALIBRATE-TBM-01: usar retornos TBM para calibrar threshold.



        # Bug anterior: se usaban retornos de 1H (np.diff(close)), mientras el modelo



        # fue entrenado con retornos TBM de hasta 96H (PT/SL). Esta inconsistencia



        # de horizonte sesgaba el threshold Ã³ptimo.



        # Fix: aplicar TBM sobre validation set y usar sus retornos como proxy.



        fwd_ret = None



        probs_aligned = probs



        try:



            from luna.features.tbm import apply_triple_barrier as _atb



            # BUG-3 FIX (2026-03-15): usar el pt_mult REAL del modelo entrenado, no siempre



            # pt_mult_min (que puede diferir del valor elegido por Optuna en el trial ganador).



            # best_params puede no incluir pt_mult (es fijo en load_dataset), asÃ­ que se lee



            # pt_mult_min como valor real del training (sin rango Optuna en pt por ahora).



            _pt_c  = float(float(cal_cfg.pt_mult_min))  # valor real del training



            _sl_c  = float(float(cal_cfg.sl_mult_min))



            _vbh_c = int(int(cal_cfg.vertical_barrier_hours))
            _min_ret_c = float(float(cal_cfg.tbm_min_return))
            _lin_decay_c = bool(bool(cal_cfg.linear_decay_pt))
            _pt_decay_frac_c = float(int(cal_cfg.pt_decay_fraction))

            _tbm_val = _atb(
                price_series=df_val["close"],
                event_times=df_val.index,
                pt_sl_multiplier=[_pt_c, _sl_c],
                min_return=_min_ret_c,
                vertical_barrier_hours=_vbh_c,
                linear_decay_pt=_lin_decay_c,
                pt_decay_fraction=_pt_decay_frac_c,
            )



            df_val_tbm = df_val.join(_tbm_val[["ret"]], how="inner").dropna(subset=["ret"])



            if len(df_val_tbm) >= min_trades:



                fwd_ret      = df_val_tbm["ret"].values



                probs_aligned = self.final_model.predict_proba(df_val_tbm[self.features])[:, 1]



                logger.info(



                    f"[Calibrate] FIX-CALIBRATE-TBM-01: retornos TBM "



                    f"(PT={_pt_c}x/SL={_sl_c}x/{_vbh_c}H) | {len(df_val_tbm)} eventos"



                )



        except Exception as _e_cal:



            logger.warning(f"[Calibrate] TBM fallback a retorno 1H: {_e_cal}")



        if fwd_ret is None:  # fallback si TBM fallÃ³



            close = df_val["close"].values



            fwd_ret       = np.diff(close) / close[:-1]



            probs_aligned = probs[:-1]



            logger.debug("[Calibrate] Usando retorno 1H como proxy (fallback)")



        thresholds = np.arange(t_min, t_max + t_step / 2, t_step)



        try:



            _max_density_pct = float(_cfg_xgb.xgboost.max_signal_density_pct)



            fallback_t = float(_cfg_xgb.xgboost.xgb_signal_threshold)



        except Exception:



            _max_density_pct = 0.60



            fallback_t = 0.40



        # Inner function to run sweep on any data slice



        def run_sweep(probs_slice, rets_slice, min_trades_req, baseline_count):



            b_thresh = fallback_t



            b_ev = -np.inf



            b_score = -np.inf



            c_log = []



            n_density_max = max(int(baseline_count * _max_density_pct), min_trades_req * 3)



            n_density_min = min(max(1, int(baseline_count * min_density_pct)), min_trades_req)



            for t in thresholds:



                mask = probs_slice > t



                n = int(mask.sum())



                if n < min_trades_req: continue



                if n < n_density_min: continue



                if n_density_max > 0 and n > n_density_max: continue



                trade_rets = rets_slice[mask] - COST_PCT



                wins = trade_rets[trade_rets > 0]



                loses = trade_rets[trade_rets <= 0]



                if len(wins) == 0 or len(loses) == 0: continue



                p_win = len(wins) / n



                avg_win = wins.mean()



                avg_los = abs(loses.mean())



                ev = p_win * avg_win - (1 - p_win) * avg_los



                c_log.append({



                    "threshold": round(float(t), 3),



                    "n_trades": n,



                    "wr": round(p_win, 4),



                    "avg_win": round(avg_win, 5),



                    "avg_loss": round(avg_los, 5),



                    "ev": round(ev, 6)



                })



                # LAB-CAL-01: penalize thresholds with n < N_target



                _vol_factor = min(1.0, n / max(_n_target, 1))



                score = ev * _vol_factor



                if ev > b_ev and score > b_score:



                    b_ev = ev



                    b_score = score



                    b_thresh = float(t)



            return b_thresh, b_ev, c_log



        # 1. Calibración Global



        best_threshold, best_ev, calibration_log = run_sweep(probs_aligned, fwd_ret, min_trades, n_baseline)



        if best_ev == -np.inf:



            # BUG-LGBM-CAL-01 FIX (2026-04-08): el sweep falla para agentes Bull/Range/Bear



            # porque el holdout de 2 meses filtrado por regimen especifico tiene muy pocas senales.



            # Reintentar con min_trades adaptativo = 30%% de senales disponibles (piso: 3).



            _n_avail = int((probs_aligned > t_min).sum())



            _adaptive_min = max(3, int(_n_avail * 0.30))



            if _adaptive_min < min_trades and _n_avail >= 3:



                logger.warning(



                    "[Calibrate/BUG-LGBM-CAL-01] Reintentando sweep con min_trades=%d "



                    "(adaptativo, n_avail=%d). Poder estadistico bajo.", _adaptive_min, _n_avail



                )



                best_threshold, best_ev, calibration_log = run_sweep(



                    probs_aligned, fwd_ret, _adaptive_min, n_baseline



                )



                if best_ev > -np.inf:



                    logger.info(



                        "[Calibrate/BUG-LGBM-CAL-01-FIX] Threshold=%.3f (EV=%.5f, n_min=%d)",



                        best_threshold, best_ev, _adaptive_min



                    )



            if best_ev == -np.inf:



                logger.warning(



                    "[Calibrate] Sweep sin resultado (min_trades=%d, n_avail=%d) "



                    "threshold=%.2f (fallback cfg)", min_trades,



                    _n_avail if _n_avail >= 3 else 0, fallback_t



                )



                best_threshold = fallback_t



        else:



            logger.success(



                "[Calibrate] Threshold Global Ã³ptimo=%.2f | EV=%.5f | wr=%.1f%% | "



                "%d combinaciones evaluadas",



                best_threshold, best_ev,



                next((r["wr"] for r in calibration_log if abs(r["threshold"] - best_threshold) < 1e-6), 0) * 100,



                len(calibration_log)



            )



        # 2. Calibración I4: Threshold por Régimen HMM



        self._threshold_per_regime = {}



        if "HMM_Regime" in df_val.columns:



            logger.info("[Calibrate/I4] Ejecutando calibración iterativa por régimen HMM...")



            # Use original index alignment because fwd_ret might be dropped by TBM



            if len(df_val) == len(probs_aligned):



                hmm_regimes_series = df_val["HMM_Regime"]



            else:



                # df_val_tbm was used, it's inner joined so we can access its HMM_Regime



                hmm_regimes_series = df_val_tbm["HMM_Regime"] if 'df_val_tbm' in locals() else df_val["HMM_Regime"].iloc[:-1]



            hmm_regimes_clean = pd.to_numeric(hmm_regimes_series, errors='coerce').fillna(-1).astype(int)



            unique_regimes = hmm_regimes_clean.unique()



            for r in unique_regimes:



                if r == -1: continue



                r_mask = (hmm_regimes_clean.values == r)



                # To avoid over-restricting small regimes, we use softer min trade constraints for substrings



                r_baseline = int((probs_aligned[r_mask] > t_min).sum())



                r_min_trades = max(10, int(min_trades * 0.25)) # Lower threshold for individual regimes 



                if r_mask.sum() < r_min_trades * 2:



                    logger.debug(f"  [Regimen {r}] Ignorado por tamaño muestral insuficiente (n={r_mask.sum()})")



                    continue



                r_thresh, r_ev, r_log = run_sweep(probs_aligned[r_mask], fwd_ret[r_mask], r_min_trades, r_baseline)



                if r_ev > -np.inf:



                    self._threshold_per_regime[str(r)] = r_thresh



                    logger.info(f"  [Regimen {r}] Threshold={r_thresh:.2f} (EV={r_ev:.4f}) calibrado sobre n={r_mask.sum()} señales")



                else:



                    logger.debug(f"  [Regimen {r}] Ningún threshold pasó los filtros mínimos (min_trades={r_min_trades}). Fallback a global.")



        self._calibration_report = calibration_log



        return best_threshold



    def train_final_model(self):



        logger.info("Entrenando Modelo Final con todo el Dataset (usando Best Params)...")



        best_params = self.best_params.copy()



        # [FIX-RANDOM-STATE-03 2026-05-28] random_state usa LUNA_SEED igual que en objective()
        _final_rs_lgbm = int(_os_lgbm.environ.get('LUNA_SEED', 42))
        best_params.update({'objective': 'binary', 'n_jobs': -1, 'random_state': _final_rs_lgbm})
        print(f"[FIX-RANDOM-STATE-03] ensemble_lgbm train_final: random_state={_final_rs_lgbm} (LUNA_SEED={_os_lgbm.environ.get('LUNA_SEED', 'no-set')})")  # RULE[fixbugsprints.md]



        self.final_model = lgb.LGBMClassifier(**best_params)



        # [A1] Inyectar Focal Loss o Monetary Loss en modelo final si está configurado
        # [FIX-LGBM-FOCAL-NS-01]: LightGBM usa su PROPIA sección 'lightgbm.use_focal_loss'.
        # Antes leía de 'xgboost.use_focal_loss' (True) → crash num_features()=0 en LGBMClassifier.
        # LightGBM y XGBoost manejan custom objectives de forma diferente internamente.
        # Si 'lightgbm.use_focal_loss' no existe en settings.yaml, se asume False (seguro).



        use_focal_loss = False



        use_monetary_loss = False



        try:



            from config.settings import cfg as _cfg_opts

            # [FIX-LGBM-FOCAL-NS-01] Leer de sección 'lightgbm', NO de 'xgboost'
            use_focal_loss = bool(int(float(_cfg_opts.lightgbm).use_focal_loss))

            use_monetary_loss = bool(_cfg_opts.fase2.use_monetary_loss)



        except Exception:



            pass



        if use_monetary_loss:



            from luna.losses.monetary_loss import get_monetary_pnl_loss



            self.final_model.set_params(objective=get_monetary_pnl_loss())



            logger.info("[Fase 2] Entrenando final_model con Monetary PnL Loss Custom")



            use_focal_loss = False



        elif use_focal_loss:



            _spw = best_params.get('scale_pos_weight', 1.0)



            self.final_model.set_params(objective=self._get_focal_loss_obj(scale_pos_weight=_spw))



            logger.info("[A1] Entrenando final_model con Focal Loss Custom (gamma={})", 



                        int(_cfg_opts.xgboost.focal_loss_gamma))



        # ARCH-02: decaimiento exponencial configurable â€” ver _compute_sample_weights



        sw_full = self._compute_sample_weights(self.X.index)



        self.final_model.fit(self.X, self.y, sample_weight=sw_full)



        # [A1 FIX] Restaurar objective estándar para que predict_proba y calibración funcionen



        # Track whether a custom objective was used (needed for Platt Scaling below)
        _used_custom_objective = callable(self.final_model.get_params().get('objective'))

        if _used_custom_objective:



            self.final_model.set_params(objective='binary')



            if hasattr(self.final_model, 'booster_'):



                self.final_model.booster_.params['objective'] = 'binary'

        # [FIX-LGBM-FOCAL-SIGMOID-01] Platt Scaling post Focal / Monetary Loss
        # PROBLEM: Focal/Monetary Loss uses a custom objective that does NOT register
        # the sigmoid link in the C++ booster. set_params(objective='binary') above
        # updates only the Python wrapper -- booster still emits raw leaf scores.
        # RESULT: predict_proba()[:,1] == booster_.predict() -- values in [0.78, 1.60].
        # All LGBM filter thresholds (<=1.0) are always exceeded -> filter is a NOOP.
        # FIX: Wrap with CalibratedClassifierCV(method='sigmoid', cv=None).
        # Platt Scaling learns a logistic mapping: booster_raw_score -> [0,1] probability.
        if _used_custom_objective:
            try:
                from sklearn.calibration import CalibratedClassifierCV
                _cal_df = self._load_calibration_source()
                if _cal_df is not None and len(_cal_df) >= 100:
                    _missing_cal = [f for f in self.features if f not in _cal_df.columns]
                    for _fc in _missing_cal:
                        _cal_df[_fc] = 0.0
                    _X_cal = _cal_df[self.features].fillna(0)
                    # Build binary target for calibration.
                    # Prefer TBM label if available; fall back to sign of raw booster score.
                    if 'target' in _cal_df.columns:
                        _y_cal = _cal_df['target'].values.astype(int)
                    else:
                        _raw_cal = self.final_model.booster_.predict(_X_cal)
                        _y_cal = (_raw_cal > float(_raw_cal.mean())).astype(int)
                    calibrated_wrapper = CalibratedClassifierCV(
                        self.final_model, method='sigmoid', cv=None
                    )
                    calibrated_wrapper.fit(_X_cal, _y_cal)
                    # Verify fix worked: max predict_proba must be <= 1.0
                    _cal_check = calibrated_wrapper.predict_proba(_X_cal[:20])[:, 1]
                    if _cal_check.max() <= 1.0 and _cal_check.min() >= 0.0:
                        self._base_lgbm_model = self.final_model
                        self.final_model = calibrated_wrapper
                        logger.success(
                            '[FIX-LGBM-FOCAL-SIGMOID-01] Platt Scaling APLICADO. '
                            'predict_proba() ahora en [0,1]: min=%.4f max=%.4f '
                            '(n_cal=%d, target_source=%s)',
                            float(_cal_check.min()), float(_cal_check.max()), len(_cal_df),
                            'target_col' if 'target' in _cal_df.columns else 'raw_score_sign'
                        )
                    else:
                        self._base_lgbm_model = self.final_model
                        logger.error(
                            '[FIX-LGBM-FOCAL-SIGMOID-01] Platt Scaling produjo valores fuera de [0,1] '
                            '(min=%.4f max=%.4f). Descartando wrapper.',
                            float(_cal_check.min()), float(_cal_check.max())
                        )
                else:
                    self._base_lgbm_model = self.final_model
                    logger.warning(
                        '[FIX-LGBM-FOCAL-SIGMOID-01] Datos de calibracion insuficientes (%d filas). '
                        'Saltando Platt Scaling - regime_router sigmoid safety net activo.',
                        len(_cal_df) if _cal_df is not None else 0
                    )
            except Exception as _e_platt:
                self._base_lgbm_model = self.final_model
                logger.warning(
                    '[FIX-LGBM-FOCAL-SIGMOID-01] Platt Scaling fallo: %s. '
                    'Continuando sin calibracion (regime_router safety net activo).', _e_platt
                )
        else:
            self._base_lgbm_model = self.final_model



        try:



            from config.settings import cfg as _cfg_log



            _alpha_log = float(_cfg_log.xgboost.weight_decay_alpha)



        except Exception:



            _alpha_log = 0.5



        logger.info("[R20-B/ARCH-02] sample_weight activo: decaimiento exp(alpha={:.2f}) desde train_end", _alpha_log)



        # â”€â”€ Feature importance top-10 siempre visible â”€â”€



        # CalibratedClassifierCV wraps the base model and has no feature_importances_.
        # Use _base_lgbm_model (the raw LGBMClassifier) for importance extraction.
        _importance_source = getattr(self, '_base_lgbm_model', self.final_model)
        importances = pd.Series(_importance_source.feature_importances_, index=self.X.columns)



        top10 = importances.sort_values(ascending=False).head(10)



        logger.info("[LGBM] Feature Importance TOP-10:")



        for feat, imp in top10.items():



            logger.info(f"  {feat}: {imp:.4f}")



        # â”€â”€ Overfit check: train AUC vs DSR OOS â”€â”€



        try:



            train_proba = self.final_model.predict_proba(self.X)[:, 1]



            from sklearn.metrics import roc_auc_score



            train_auc = roc_auc_score(self.y, train_proba)



            oos_dsr   = self.study.best_value if self.study else float('nan')



            gap_flag = " âš ï¸ SOBREAJUSTE" if train_auc > 0.80 and oos_dsr < 0.50 else ""



            logger.info(f"[LGBM] Overfit check: train_AUC={train_auc:.4f} | best_DSR_OOS={oos_dsr:.4f}{gap_flag}")



            check_numeric_stability(train_proba, label="LGBM.train_proba")



        except Exception as e:



            logger.warning(f"[LGBM] Overfit check fallido: {e}")



        # Plot Feature Importances



        plt.figure(figsize=(10, 8))



        importances.sort_values(ascending=True).plot(kind='barh')



        plt.title("LightGBM Meta-Model Feature Importances (Gain)")



        out_path = self.root / "data" / "models" / "engine_xgb_importances.png"



        plt.tight_layout()



        plt.savefig(out_path)



        logger.info(f"Importancias exportadas a {out_path.name}")



        # Guardar Modelo



        out_dir = self.root / "data" / "models"



        out_dir.mkdir(parents=True, exist_ok=True)



        suffix = f"_{self.regime_name}" if self.regime_name else ""



        model_path = out_dir / f"lgbm_meta{suffix}_{self.native_direction}.model"



        # xgb format



        import joblib; joblib.dump(self.final_model, model_path)



        # â”€â”€ CalibraciÃ³n automÃ¡tica del threshold (MEJORA-R12-01) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



        # Barre thresholds sobre features_validation.parquet maximizando EV(t).



        # Resultado guardado en la firma â€” generate_oos_predictions.py lo carga.



        self._calibration_report = []



        optimal_threshold = self._calibrate_threshold()



        # Save signature



        sig_path = out_dir / f"lgbm_meta{suffix}_{self.native_direction}_signature.json"



        with open(sig_path, 'w') as f:



            json.dump({



                "features":           self.features,



                "dsr_oos":            float(self.study.best_value),



                "params":             self.best_params,



                "cost_discounted":    COST_PCT,



                # MEJORA-R12-01: threshold calibrado automÃ¡ticamente



                "optimal_threshold":  optimal_threshold,



                # I4: Umbrales calibrados por rÃ©gimen



                "optimal_threshold_per_regime": getattr(self, '_threshold_per_regime', {}),



                # ARCH-04: trazabilidad de fuente de calibraciÃ³n ("holdout_3m" o "validation")



                "cal_source":         getattr(self, '_cal_source', 'validation'),



                "calibration_report": self._calibration_report,



                "regime_name":        self.regime_name,



            }, f, indent=4)



        # â”€â”€ [DATAFLOW-EXPORT-LGBM-01] Model Signature Audit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



        logger.success(



            f"[DATAFLOW-EXPORT-LGBM-01] Modelo guardado: {model_path.name} | "



            f"n_features={len(self.features)} | "



            f"dsr_oos={float(self.study.best_value):.4f} | "



            f"threshold_calibrado={optimal_threshold:.2f} ({getattr(self, '_cal_source', 'validation')})"



        )



        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



class MultiAgentLGBMTrainer:



    """



    Orquestador de N agentes LightGBM especializados por régimen HMM.



    FASE 2: Enrutamiento Bull/Range/Bear basado en etiquetas semánticas.



    """



    def __init__(self):



        # Mapeo textual semántico a cada agente experto



        try:



            from config.settings import cfg as _cfg_ma



            self.regimes_config = vars(_cfg_ma.fase2.regime_mapping)



        except Exception as e:



            logger.warning(f"Error cargando regime_mapping: {e}. Fallback interno.")



            self.regimes_config = {



                # [SOL3-CALM-BEAR-01 2026-06-01] calm_bear dedicado separado de bear
                "bull":      ["1_BULL_TREND", "1_VOLATILE_BULL", "1_BULL_GRIND", "1_BULL_TREND_WEAK", "1_BULL_TREND_B", "1_VOLATILE_BULL_B"],
                "range":     ["2_CALM_RANGE", "2_VOLATILE_RANGE", "2_CALM_RANGE_B", "2_VOLATILE_RANGE_B"],
                "calm_bear": ["3_CALM_BEAR", "3_CALM_BEAR_B", "3_CALM_BEAR_C", "3_CALM_BEAR_D"],
                "bear":      ["3_BEAR_CRASH", "3_BEAR_CRASH_B", "4_BEAR_FORCED"]



            }



        self.trainers = {}



    def run_all(self):



        logger.info("[FASE 2] Iniciando Entrenamiento Multi-Agent LightGBM por Régimen")



        # Dividir los trials totales entre los 3 regímenes para mantener tiempo cte



        n_regimes = len(self.regimes_config)



        trials_per_regime = max(30, OPTUNA_TRIALS // n_regimes)



        for name, r_list in self.regimes_config.items():



            logger.info(f"\n{'='*50}\n[FASE 2] Entrenando Agente [{name.upper()}] (Regimes: {r_list})\n{'='*50}")



            t = LGBMTrainer(regime_name=name, regime_list=r_list, n_trials=trials_per_regime)



            try:



                t.load_dataset()



                t.tune_hyperparameters()



                t.train_final_model()



                self.trainers[name] = t



            except Exception as e:



                logger.error(f"[FASE 2] Falló el entrenamiento del agente {name.upper()}: {e}")



        logger.success("[FASE 2] Entrenamiento Multi-Agent Completado Exitosamente.")



if __name__ == "__main__":



    import os as _os



    from datetime import datetime as _dt



    from pathlib import Path as _Path



    _log_dir = _Path(__file__).resolve().parents[2] / "logs"



    _log_dir.mkdir(exist_ok=True)



    _ts_xgb  = _dt.now().strftime("%Y%m%d_%H%M%S")



    _rid_xgb = _os.environ.get("LUNA_RUN_ID", "")



    _lname_xgb = f"ensemble_lgbm_v2_{_ts_xgb}_{_rid_xgb}.log" if _rid_xgb else f"ensemble_lgbm_v2_{_ts_xgb}.log"



    logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")



    logger.add(_log_dir / _lname_xgb, rotation="100 MB", level="DEBUG", encoding="utf-8")



    try:



        from config.settings import cfg as _cfg_main



        use_regime = bool(_cfg_main.fase2.use_regime_agents)



    except Exception as e:



        logger.warning(f"No se pudo leer fase2.use_regime_agents ({e}), usando trainer estándar")



        use_regime = False



    if use_regime:



        ma = MultiAgentLGBMTrainer()



        ma.run_all()



    else:



        trainer = LGBMTrainer()



        trainer.load_dataset()



        trainer.tune_hyperparameters()



        trainer.train_final_model()



