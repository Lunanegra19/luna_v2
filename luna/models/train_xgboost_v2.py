"""
Orquestador XGBoost Meta-Model - Luna V1
===================================================
Entrena el modelo base conectando las features seleccionadas (SFI) y la 
etiqueta del rÃƒÂ©gimen HMM.

SOP Aplicado:
- R3 (Purge/Embargo): Se usa Combinatorial Purged CV para la evaluaciÃƒÂ³n de Optuna.
- R5 (DSR Objetivo): La mÃƒÂ©trica a maximizar por Optuna es el Deflated Sharpe OOS.
- R6 (Costos TransacciÃ³n): 0.25% RT aplicado a las simulaciones de Sharpe.
"""

import sys
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
import xgboost as xgb
import optuna
import joblib
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
from luna.utils.debug_guards import (
    check_target_balance, check_numeric_stability, check_df_sanity,
    vlog, timeit, log_memory_usage,
)

class ModelDegradationError(Exception):
    """Excepción para abortar el entrenamiento de un solo agente y forzar modo CASH."""
    pass

# ── [LUNA-V2-CALIB] Platt Calibrator drop-in wrapper for joblib/pickle serialization ──
class PlattCalibrator:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression as _LR
        # [FIX-RANDOM-STATE-01] PlattCalibrator LR: random_state=0 (determinista)
        self.model = _LR(C=1e6, random_state=0)
        self.X_thresholds_ = []

    def fit(self, X, y):
        X_2d = X.reshape(-1, 1) if X.ndim == 1 else X
        self.model.fit(X_2d, y)
        return self

    def predict(self, T):
        T_2d = T.reshape(-1, 1) if T.ndim == 1 else T
        return self.model.predict_proba(T_2d)[:, 1]



# ── [FIX-TEMPCAL-DESER-01 2026-06-01] Temperature Scaling Calibrator ────────
# ANTES: clase definida aquí → joblib la serializaba como '__main__.TemperatureCalibrator'
#        → AttributeError al deserializar en regime_router.py (namespace distinto)
#        → 33 errores / 5 FATALs directas (seeds: 100, 1337, 2026, 27243, 44085)
# AHORA: importada desde luna.models.calibrators → serializada como
#        'luna.models.calibrators.TemperatureCalibrator' → importable desde cualquier script
# REFERENCIA: luna/models/calibrators.py
from luna.models.calibrators import TemperatureCalibrator  # [FIX-TEMPCAL-DESER-01]
print("[FIX-TEMPCAL-DESER-01] TemperatureCalibrator importada desde luna.models.calibrators — deserialización OK")  # RULE[fixbugsprints.md]


# ParÃ¡metros Globales (SOP) â€” [TIPO-2] constantes de SOP, no hiperparÃ¡metros

# Todos leídos desde settings.yaml — ver bloque try/except abajo
CPCV_TEST_GROUPS = 2  # arquitectura CPCV: siempre k=2 grupos de test (LdP 2018)
# B1 FIX (2026-03-09): leer desde settings.yaml
import os as _os
try:
    from config.settings import cfg as _cfg_xgb
    
    # ── [CRITICAL-LUNA-V2] Fail-Loud Configuration Validation ──
    # Ensure all required Luna v2 settings exist and violate no limits.
    if not hasattr(_cfg_xgb, "xgboost"):
        raise AttributeError("settings.yaml missing required 'xgboost' section")
    if not hasattr(_cfg_xgb, "sop"):
        raise AttributeError("settings.yaml missing required 'sop' section")
        
    # Strictly validate Optuna space bounds
    if not hasattr(_cfg_xgb.xgboost, "optuna_search_space"):
        raise AttributeError("settings.yaml missing required 'xgboost.optuna_search_space'")
    _sp = _cfg_xgb.xgboost.optuna_search_space
    
    if float(_sp.min_child_weight_min) is None:
        raise AttributeError("settings.yaml missing 'xgboost.optuna_search_space.min_child_weight_min'")
    if float(_sp.min_child_weight_min) < 1.0:
        raise ValueError(f"min_child_weight_min ({_sp.min_child_weight_min}) must be strictly >= 1 in Luna v2.")  # [FIX-REG-01 2026-05-31] Reducido de 30 a 1 (el floor real es evitar MCW=0, no sobreregularizar)
        
    if int(_sp.max_depth_max) is None:
        raise AttributeError("settings.yaml missing 'xgboost.optuna_search_space.max_depth_max'")
    if int(_sp.max_depth_max) > 4:
        raise ValueError(f"max_depth_max ({_sp.max_depth_max}) must be strictly <= 4 in Luna v2.")
        
    # Strictly validate Calibration parameters
    if float(_cfg_xgb.xgboost.calibration_min_samples_isotonic) is None:
        raise AttributeError("settings.yaml missing 'xgboost.calibration_min_samples_isotonic'")
    if int(_cfg_xgb.xgboost.calibration_min_samples_isotonic) < 1000:
        raise ValueError(f"calibration_min_samples_isotonic must be >= 1000 in Luna v2 to prevent calibration collapse.")
        
    if getattr(_cfg_xgb.xgboost, "calibration_fallback_method", None) is None:
        raise AttributeError("settings.yaml missing 'xgboost.calibration_fallback_method'")
    if getattr(_cfg_xgb.xgboost, "calibration_fallback_method") != "sigmoid":
        raise ValueError("calibration_fallback_method must be 'sigmoid' in Luna v2 Platt scaling fallback.")
        
    # Strictly validate Embargo parameters
    if getattr(_cfg_xgb.xgboost, "embargo_dynamic_decay", None) is None:
        raise AttributeError("settings.yaml missing 'xgboost.embargo_dynamic_decay'")
    if getattr(_cfg_xgb.xgboost, "embargo_decay_atr_lookback", None) is None:
        raise AttributeError("settings.yaml missing 'xgboost.embargo_decay_atr_lookback'")
    if getattr(_cfg_xgb.xgboost, "embargo_low_density_threshold", None) is None:
        raise AttributeError("settings.yaml missing 'xgboost.embargo_low_density_threshold'")
    if getattr(_cfg_xgb.xgboost, "embargo_min_hours", None) is None:
        raise AttributeError("settings.yaml missing 'xgboost.embargo_min_hours'")
        
    OPTUNA_TRIALS = int(_cfg_xgb.xgboost.optuna_trials)
    _cpcv_n = int(_cfg_xgb.sop.cpcv_groups) \
               or int(_cfg_xgb.xgboost.n_purged_splits)
    CPCV_GROUPS   = int(_cpcv_n)
    PURGE_H       = int(_cfg_xgb.sop.purge_hours)
    EMBARGO_H     = int(_cfg_xgb.sop.embargo_hours)
    COST_PCT      = float(_cfg_xgb.sop.cost_pct)
    
    print("[LUNA-V2-CONFIG] STRICT CONF VALIDATION: Passed! All settings.yaml keys validated as critical and compliant.")
    logger.info("[LUNA-V2-CONFIG] STRICT CONF VALIDATION: Passed!")
except Exception as _cfg_err:
    # ARCH-FAIL-LOUD (2026-03-18): NO silenciar errores de configuración.
    # Un except silencioso ocultaría: YAML corrupto, import error, parámetro
    # movido de sección, etc. El pipeline correría con valores INCORRECTOS
    # (p.ej. 600 trials en vez de 100, CPCV=6 en vez de 8) sin ningún aviso.
    # Principio: Fail Loud > Fail Silent. Si settings.yaml no carga, abortamos.
    raise RuntimeError(
        f"\n[CRITICAL-LUNA-V2] train_xgboost.py no pudo cargar o validar settings.yaml.\n"
        f"  Error: {_cfg_err}\n"
        f"  El pipeline NO puede ejecutarse sin configuración válida y conforme a las reglas Luna v2.\n"
        f"  Verifica: sintaxis YAML, PYTHONPATH, existencia de config/settings.py"
    ) from _cfg_err

# B3 FIX (2026-03-09): flag diagnÃ³stico para aislar efecto de mining rules.
# Uso: set LUNA_SKIP_MINING=1 && python core/models/train_xgboost.py
SKIP_MINING: bool = _os.environ.get("LUNA_SKIP_MINING", "0") == "1"


# ---------------------------------------------------------------------------
# P1-5: MiningRuleValidator â€” filtro DSR para reglas de AI Mining
# ---------------------------------------------------------------------------
class MiningRuleValidator:
    """
    Valida las reglas de AI Mining (golden_rule_N, genetic_rule_N) usando DSR
    antes de inyectarlas en XGBoost.

    Reemplaza el pass-through ciego de hits>0 con validaciÃ³n estadÃ­stica.

    P1-5 (planes_mejora_v3.md):
    - n_trials_efectivo = OPTUNA_TRIALS * 3.0 por penalizaciÃ³n heurÃ­stica (mining es bÃºsqueda ad-hoc)
    - Solo reglas con DSR >= MIN_DSR_RULE pasan al modelo
    - 0 reglas aprobadas es preferible a reglas con overfitting

    CONTRATO close_rets (LAB-01 fix 2026-03-20):
    - Debe ser el retorno forward al MISMO horizonte que el TBM del XGBoost.
    - INCORRECTO: usar pct_change con shift de 1 barra (inconsistente con TBM de 96-168H)
    - CORRECTO:   pct_change(N).shift(-N)  â†  N = vertical_barrier_hours de settings.yaml
    - RazÃ³n: una regla con edge en 1H puede ser destructiva en el horizonte TBM real;
      la validaciÃ³n DSR debe usar el mismo horizonte que el modelo que consume la regla.
    """
    MIN_DSR_RULE = 0.80       # Umbral DSR para reglas de mining
    # [FIX-D] N_TRIALS_PENALTY leído de settings.yaml ai_mining.n_trials_penalty
    # Ref: Bailey (2014) "Pseudo-Mathematics": ratio ≈ 3× para corregir overfitting por selección múltiple.
    # Antes: 3.0 hardcodeado como atributo de clase sin referencia
    _N_TRIALS_PENALTY_DEFAULT = 3.0

    def __init__(self, close_rets: pd.Series, cost_pct: float = COST_PCT):
        self.close_rets = close_rets
        self.cost_pct = cost_pct
        # Leer n_trials_penalty de cfg
        try:
            from config.settings import cfg as _cfg_tp
            self.N_TRIALS_PENALTY = float(_cfg_tp.ai_mining.n_trials_penalty)
        except Exception:
            self.N_TRIALS_PENALTY = self._N_TRIALS_PENALTY_DEFAULT
            print(f"[FIX-D] WARN: No se pudo leer ai_mining.n_trials_penalty. Usando fallback={self.N_TRIALS_PENALTY} (Bailey 2014)")  # debug
        print(f"[FIX-D] MiningRuleValidator: N_TRIALS_PENALTY={self.N_TRIALS_PENALTY} (optuna_trials efectivos = {int(OPTUNA_TRIALS * self.N_TRIALS_PENALTY)})")  # debug
        self.n_trials_efectivo = int(OPTUNA_TRIALS * self.N_TRIALS_PENALTY)  # ~1800

    def _compute_rule_dsr(self, rule_series: pd.Series) -> float:
        """
        Calcula el DSR de una regla binary (0/1) usando sus retornos pseudo-OOS.
        Retorna DSR en [0,1]; DSR < 0.80 = rechazada.

        LOGIC-XGB-01 FIX (2026-04-06): DSR calculado sobre el 20% final del periodo
        de training (pseudo-OOS interno). Antes se usaba el dataset completo (IS),
        lo que permitía a reglas overfittadas obtener DSR=1.0 in-sample.
        Ahora: 80% últimas barras = IS (fit de la regla), 20% últimas = pseudo-OOS (DSR).

        BUG-DSR-NAN-01 FIX (2026-05-05): El early-exit usaba np.std(strat_rets) < 1e-8,
        pero cuando rets_oos contiene NaN (los últimos N values de pct_change(N).shift(-N)),
        np.std devuelve NaN. En Python: NaN < 1e-8 = False, lo que bypasea el early-exit.
        Luego SR=NaN, norm.cdf(NaN)=NaN, y max(0, min(1, NaN)) = 1.0 en Python.
        Resultado: reglas con CERO trades OOS obtenían DSR=1.0 trivialmente (falso positivo).
        Fix: dropna() ANTES del early-exit, operando solo sobre retornos válidos.
        """
        from scipy.stats import norm
        import math

        aligned = self.close_rets.align(rule_series.reindex(self.close_rets.index), join='inner')
        rets, sigs = aligned
        sigs = sigs.fillna(0).astype(float)

        # LOGIC-XGB-01: reservar último 20% como pseudo-OOS para el DSR
        n_total = len(rets)
        n_oos   = max(30, int(n_total * 0.20))
        rets_oos = rets.iloc[-n_oos:]
        sigs_oos = sigs.iloc[-n_oos:]

        # [V2-MATH-FIX] Evaluar DSR sobre los TRADES reales, no sobre las BARRAS (donde >99% son 0.0)
        # Mantener los ceros infla artificialmente los grados de libertad t = 10,000 en lugar de t = 5.
        trade_rets = rets_oos.values[sigs_oos.values == 1] - self.cost_pct
        trade_rets_clean = trade_rets[~np.isnan(trade_rets)]
        n_trades_clean = len(trade_rets_clean)

        if n_trades_clean < 5 or np.std(trade_rets_clean) < 1e-8:
            logger.debug(
                "[V2-MATH-FIX] Regla rechazada: n_trades={t}, std={std:.2e} "
                "(Insuficientes grados de libertad o varianza cero)",
                t=n_trades_clean, 
                std=np.std(trade_rets_clean) if n_trades_clean > 0 else 0.0
            )
            return 0.0

        # Raw Sharpe Ratio por trade (no anualizado ni inflado por N barras)
        sr_raw = np.mean(trade_rets_clean) / np.std(trade_rets_clean)
        t = n_trades_clean

        # Calculamos el t-statistic de la regla (nivel de significancia estadistica real)
        t_stat = sr_raw * math.sqrt(max(1, t - 1))

        # DSR (Bailey & LdP 2014): penaliza por n_trials usando la distribucion del maximo de normales
        n_trials = max(self.n_trials_efectivo, 2)
        z1 = norm.ppf(1 - 1.0 / n_trials)
        z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
        
        # sr_star (Expected Maximum) calculado para t-statistics (varianza = 1.0)
        t_stat_star = 1.0 * ((1 - 0.577215) * z1 + 0.577215 * z2)

        try:
            # BUG-EVT FIX: Comparar T-Stat con T-Stat_Star directo.
            # No se debe multiplicar de nuevo por sqrt(t-1) porque t_stat y t_stat_star
            # ya estan en el espacio de la distribucion normal estandar.
            dsr_raw = float(norm.cdf(t_stat - t_stat_star))
            if np.isnan(dsr_raw):
                logger.warning("[BUG-DSR-NAN-01] DSR calculado es NaN → 0.0")
                return 0.0
            dsr = dsr_raw
            
            if dsr >= self.MIN_DSR_RULE:
                logger.debug(f"[V2-MATH-FIX] Regla aprobada: n_trades={t}, t_stat={t_stat:.2f} > t_stat_star={t_stat_star:.2f} -> DSR={dsr:.4f}")
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


class XGBoostTrainer:
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
        elif _dmode == "long":
            self.native_direction = "long"
        elif _dmode == "short":
            self.native_direction = "short"
        else:
            self.native_direction = "long"
            
        self.X = None
        self.y = None
        self.close_rets = None
        self._spw_ideal = 1.0
        self._base_rate_is = 0.50
        self.study = None
        self.best_params = {}
        # FIX-CPCV-CACHE-01: cache de splits CPCV precalculados.
        # Los splits son siempre los mismos para un dataset dado â€” recalcularlos
        # en cada uno de los 100+ trials es trabajo redundante (~0.5s * 100 = 50s).
        # Se setea en tune_hyperparameters() antes de lanzar Optuna.
        self._cached_splits = None
        # IDEA-G: acumular feature importances por fold CPCV para análisis de estabilidad.
        # Lista de dicts {feature: importance_gain} por cada fold del mejor trial.
        self._fold_importances: list = []
        
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
            logger.warning(f"[CAPA-1] Error aplicando Rolling Window: {e}. Fallback a 'expanding'.")
        # ───────────────────────────────────────────────────────────────────────


        # 2. Cargar Seleccionadas
        with open(self.root / "data" / "features" / "selected_features.json", 'r') as f:
            features_list = json.load(f)["selected_features"]
        # [OOF-CALIB-V2 2026-06-03] Deduplicar features_list para evitar columnas duplicadas
        # que causan ValueError (ambiguous truth value of a Series) en check_df_sanity
        _len_before_dedup = len(features_list)
        features_list = list(dict.fromkeys(features_list))
        if len(features_list) < _len_before_dedup:
            print(f"[OOF-CALIB-V2] Deduplicado features_list: {_len_before_dedup} -> {len(features_list)} features")

        # â”€â”€ [P1-5] Mining Rules con validaciÃ³n DSR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Las golden_rule_N y genetic_rule_N pasan por filtro DSR antes de ser
        # inyectadas en XGBoost. Solo reglas con DSR >= 0.80 (n_trials=1800 efectivos)
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
            # N = vertical_barrier_hours (mismo horizonte que el TBM con el que entrena XGBoost).
            # Sin este fix: una regla que predice 1H bien (DSR>0.80) pero es destructiva
            # en 96-168H podrÃ­a injertarse en el modelo y degradar el rendimiento OOS.
            if "close" in df.columns:
                try:
                    from config.settings import cfg as _cfg_mvr
                    _vbh_mvr = int(_cfg_mvr.xgboost.vertical_barrier_hours)  # [FIX-01] fallback unificado a 72H (igual que predict_oos y settings.yaml)
                except Exception:
                    _vbh_mvr = 72  # [FIX-01] Antes era 96H → inconsistente con predict_oos (168H) y settings.yaml (72H)
                    print(f"[FIX-01] WARN: No se pudo leer xgboost.vertical_barrier_hours de cfg. Usando fallback={_vbh_mvr}H")  # debug
                _close_rets_proxy = df["close"].pct_change(_vbh_mvr).shift(-_vbh_mvr)
                logger.debug(
                    f"[P1-5/LAB-01] MiningRuleValidator: close_rets horizonte={_vbh_mvr}H "
                    f"(consistente con vertical_barrier_hours del TBM)."
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
            _bypass_sfi = bool(getattr(_cfg_timing.features, "timing_features_bypass_sfi", True))
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
                    # P2-N2-FIX (2026-03-30): propagar guardia clip(lower=0.01) desde P2-3.
                    # Sin esto train y OOS tienen distribuciones distintas para timing_vol_divergence.
                    vol_ratio   = vol_ratio.clip(lower=0.01)
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
                # Ayudan al XGBoost a discriminar entre bull-trend real y lateral bajista.
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
        # [FIX-HMM-JOIN-01] El feature pipeline (Paso 3B) ya integra HMM_Regime en
        # features_train.parquet. Si hacemos join() con hmm_regime_labels.parquet que
        # tambien tiene HMM_Regime → ValueError: columns overlap but no suffix specified.
        # Fix: eliminar del df_hmm las columnas que ya existen en df (evita duplicados).
        _hmm_cols_overlap = [c for c in df_hmm.columns if c in df.columns]
        if _hmm_cols_overlap:
            logger.info(
                f"[FIX-HMM-JOIN-01] Columnas ya presentes en features_train.parquet "
                f"eliminadas de hmm_regime_labels antes del join (evita colision): "
                f"{_hmm_cols_overlap}"
            )
            print(f"[XGB][FIX-HMM-JOIN-01] Eliminando columnas duplicadas del join HMM: {_hmm_cols_overlap}")  # debug
            df_hmm = df_hmm.drop(columns=_hmm_cols_overlap)
        df_final = df.join(df_hmm)
        # HMM_Regime siempre se aÃ±ade al feature set (pass-through obligatorio).
        # En holdout, generate_oos_predictions.py llama a predict_regime_series() para
        # cubrir el perÃ­odo 2025+ donde hmm_regime_labels.parquet no llega.
        if "HMM_Regime" not in features_list:
            features_list.append("HMM_Regime")
            logger.info("[HMM-PASS-THROUGH] HMM_Regime aÃ±adido al feature set.")


        # [OOD-GUARD] Detección genérica de features con distribución degenerada en OOS
        # Compara la distribución de cada feature entre training y validation set.
        # Features cuya varianza colapsa en OOS (std_ratio < 5%, >95% valor constante,
        # <3 valores únicos) se excluyen AUTOMÁTICAMENTE — sin hardcodear nombres.
        #
        # RAZONAMIENTO ARQUITECTURAL: features como KMeans_Tribe_ID o Master_Causal_Signal
        # tienen varianza real en training (calculadas sobre el mismo set), pero en OOS
        # colapsan a un valor constante porque dependen de modelos no actualizables
        # en producción (K-Means offline, Bayesian Engine con priors de training).
        # Cuando XGBoost recibe estas features constantes, devuelve la misma probabilidad
        # para todas las filas OOS → señales estáticas, 0 trades útiles.
        #
        # El guard detecta CUALQUIER feature con este patrón, sin importar el nombre.
        # Si en el futuro otra feature sufre el mismo problema, será bloqueada automáticamente.
        try:
            from luna.utils.ood_feature_guard import filter_ood_features as _ood_filter
            # [ARCH-25-FIX-A 2026-06-02] OOD Feature Guard usa IS propio del agente
            # como X_oos en lugar de features_validation.parquet (100% BULL, KL=8.33).
            # PROBLEMA: features RANGE/BEAR tienen alta varianza en su IS pero baja en
            # BULL-only validation -> falsos positivos OOD -> features validas bloqueadas.
            # SOLUCION: ultimo 20% IS del agente como X_oos. Fallback: val.parquet (<200 barras).
            _regime_name_ood = str(getattr(self, 'regime_name', '') or '')
            _df_train_ood = df_final
            _df_oos_ood   = None
            _oos_source   = 'none'

            # Ultimo 20% del IS como X_oos (mismo regimen que el agente)
            _n_is = len(_df_train_ood)
            _split_20pct = int(_n_is * 0.8)
            _df_is_recent = _df_train_ood.iloc[_split_20pct:]
            if len(_df_is_recent) >= 200:
                _df_oos_ood = _df_is_recent
                _oos_source = f'IS_reciente_20pct_{_regime_name_ood}_N{len(_df_is_recent)}'
                print(  # RULE[fixbugsprints.md]
                    f'[ARCH-25-FIX-A] OOD Guard X_oos=IS reciente propio '
                    f'N={len(_df_is_recent)} regimen={_regime_name_ood}'
                )
            else:
                _val_path = self.root / 'data' / 'features' / 'features_validation.parquet'
                if _val_path.exists():
                    _df_oos_ood = pd.read_parquet(_val_path)
                    _oos_source = 'features_validation_parquet_fallback'
                    print(  # RULE[fixbugsprints.md]
                        f'[ARCH-25-FIX-A] OOD Guard FALLBACK a validation.parquet '
                        f'(IS reciente insuf: {len(_df_is_recent)} < 200)'
                    )

            if _df_oos_ood is not None:
                _feats_to_check = [f for f in features_list
                                   if f in _df_train_ood.columns and f in _df_oos_ood.columns]
                _agent_ctx = str(getattr(self, 'regime_name', 'XGBoost') or 'XGBoost')
                _valid_feats, _ood_reports = _ood_filter(
                    X_train=_df_train_ood[_feats_to_check],
                    X_oos=_df_oos_ood[_feats_to_check],
                    context=f'XGBoost/{_agent_ctx}[{_oos_source}]',
                )
                _not_checked = [f for f in features_list if f not in _feats_to_check]
                features_list = _valid_feats + _not_checked
                _n_blocked = sum(1 for r in _ood_reports if r.blocked)
                if _n_blocked > 0:
                    _blocked_names = [r.feature for r in _ood_reports if r.blocked]
                    logger.warning(
                        f'[OOD-GUARD] {_n_blocked} features bloqueadas [{_oos_source}]: '
                        f'{_blocked_names} -> features_list={len(features_list)}'
                    )
            else:
                logger.debug('[OOD-GUARD] sin fuente OOS disponible -- guard omitido.')
        except Exception as _ood_err:
            logger.warning(f'[OOD-GUARD] Error: {_ood_err}')



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
        # BUG-A01 FIX (2026-03-17): leer tbm_min_return de settings.yaml — era hardcoded 0.005.
        try:
            from config.settings import cfg as _cfg
            _pt      = float(_cfg.xgboost.pt_mult_min)
            _sl      = float(_cfg.xgboost.sl_mult_min)
            _min_ret = float(_cfg.xgboost.tbm_min_return)
            logger.info(f"TBM XGBoost: pt_mult={_pt}, sl_mult={_sl}, min_return={_min_ret} (de settings.yaml)")
        except Exception:
            _pt, _sl, _min_ret = 2.0, 1.0, 0.005
            logger.warning("TBM XGBoost: usando defaults 2.0/1.0/0.005 (settings no disponible)")

        # TBM-REGIME-01 (2026-05-05): Perfil TBM adaptativo por régimen del agente.
        # RAZONAMIENTO: la tasa base (Base Rate) del TBM varía drásticamente por régimen:
        #   BULL  : TBM 1.5x/0.8x → Base Rate ~45% → debajo del umbral MetaLabeler (0.50)
        #   RANGE : TBM simétrico 1.0x/1.0x → Base Rate ~52% → por encima del umbral
        #   BEAR  : TBM conservador 0.5x/2.0x → muy pocas señales (mercado adverso)
        # Sin este ajuste, XGBoost aprende correctamente (prob~0.45) pero el MetaLabeler
        # veta el 100% de señales porque exige prob>0.50. No es fallo del modelo, es aritmética.
        # Garantía causal: self.regime_name es el agente IS actual, sin look-ahead.
        if self.regime_name is not None:
            try:
                _regime_profiles_raw = None
                if getattr(self, "native_direction", None) == "short":
                    _regime_profiles_raw = getattr(_cfg.xgboost, "regime_tbm_profiles_short", None)
                if _regime_profiles_raw is None:
                    _regime_profiles_raw = getattr(_cfg.xgboost, "regime_tbm_profiles", None)
                    
                if _regime_profiles_raw is not None:
                    # [TBM-REGIME-01 FIX] _Namespace recursivo → convertir a dict plano con vars()
                    # sin esto, la iteración y .get() fallan silenciosamente → fallback a global
                    try:
                        _regime_profiles = vars(_regime_profiles_raw)
                    except TypeError:
                        _regime_profiles = dict(_regime_profiles_raw)

                    # Normalizar clave: "1_bull_trend" → buscar perfil cuya key sea prefijo del régimen
                    _rkey_raw = str(self.regime_name).lower()
                    _matched_profile_ns = None
                    _matched_key = None
                    for _pk, _pv in _regime_profiles.items():
                        if _rkey_raw == str(_pk).lower() or _rkey_raw.startswith(str(_pk).lower()):
                            _matched_profile_ns = _pv
                            _matched_key = str(_pk).lower()
                            break

                    if _matched_profile_ns is not None:
                        # Convertir el sub-Namespace del perfil a dict también
                        try:
                            _prof_dict = vars(_matched_profile_ns)
                        except TypeError:
                            _prof_dict = dict(_matched_profile_ns)

                        _pt_new = float(_prof_dict.get('pt_mult_min', _pt))
                        _sl_new = float(_prof_dict.get('sl_mult_min', _sl))
                        logger.info(
                            f"[TBM-REGIME-01] Agente '{self.regime_name}' → perfil '{_matched_key}': "
                            f"pt={_pt_new:.2f}x (era {_pt:.2f}x), sl={_sl_new:.2f}x (era {_sl:.2f}x)"
                        )
                        _pt = _pt_new
                        _sl = _sl_new
                    else:
                        logger.info(
                            f"[TBM-REGIME-01] Agente '{self.regime_name}': sin perfil específico "
                            f"→ usando global pt={_pt:.2f}x, sl={_sl:.2f}x"
                        )
                else:
                    logger.debug("[TBM-REGIME-01] regime_tbm_profiles no configurado → TBM global activo")
            except Exception as _e_tbm_regime:
                logger.warning("[TBM-REGIME-01] Error leyendo perfil régimen: {} → usando global", _e_tbm_regime)
                print(f"[BUG-FIX-LOG 2026-06-05] Error leyendo perfil régimen: {_e_tbm_regime} → usando global")

        # FIX-TBM-DYNAMIC-01: dynamic_barrier y event_sampling_hours configurables.
        # dynamic_barrier=True usa horizonte ATR adaptativo (Mejora 4 implementada
        # pero nunca activada hasta ahora). event_sampling_hours>1 reduce el
        # solapamiento entre labels TBM al muestrear eventos cada N horas.
        try:
            _dynamic_barrier = bool(hasattr(_cfg, "xgboost") and
                                    bool(getattr(_cfg.xgboost, "dynamic_barrier", False)))
            _event_sampling_h = int(_cfg.xgboost.event_sampling_hours)
        except Exception:
            _dynamic_barrier, _event_sampling_h = False, 1

        if _event_sampling_h > 1:
            # [P2-4-FIX] Subsampling aleatorio uniforme en lugar de secuencial para evitar sesgos
            n_samples = max(1, len(events_idx) // _event_sampling_h)
            # [FIX-RANDOM-STATE-01b 2026-05-28] Usar LUNA_SEED para diversificar event sampling entre seeds
            _rng_seed = int(_os.environ.get('LUNA_SEED', 42))
            _rng = np.random.default_rng(_rng_seed)
            print(f"[FIX-RANDOM-STATE-01b] TBM event_sampling rng seed={_rng_seed} (LUNA_SEED={_os.environ.get('LUNA_SEED', 'no-set')})")  # RULE[fixbugsprints.md]
            _chosen = _rng.choice(len(events_idx), size=n_samples, replace=False)
            events_idx = events_idx[_chosen].sort_values()
            logger.info(
                f"[FIX-TBM-SAMPLE-01] event_sampling_hours={_event_sampling_h}: "
                f"{len(events_idx)} eventos (reducido de {len(df_final)} — menos solapamiento)"
            )

        try:
            _vbh = int(_cfg.xgboost.vertical_barrier_hours)
            _dyn_min = int(_cfg.xgboost.dynamic_horizon_min_h)
            _dyn_max = int(_cfg.xgboost.dynamic_horizon_max_h)
        except Exception as e:
            raise RuntimeError(f"Faltan parametros de horizonte TBM en settings.yaml (SOP No-Fallback): {e}")
        _lin_decay = bool(_cfg.xgboost.linear_decay_pt) if hasattr(_cfg, 'xgboost') else False
        _pt_decay_frac = float(_cfg.xgboost.pt_decay_fraction) if hasattr(_cfg, 'xgboost') else 0.75
        
        _side_val = -1.0 if self.native_direction == "short" else 1.0
        _sides_series = pd.Series(_side_val, index=events_idx)

        _funding_series = df_final["FundingRate"] if "FundingRate" in df_final.columns else None

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
            funding_series=_funding_series,
        )
        
        # tbm_result contiene las etiquetas (1=PT, -1=SL, 0=T1) en la columna "bin" o "meta_label". 
        # La columna "bin" (1 si retornó pt_sl positivo vs SL, 0 timeout).
        # Para simplificar la base prediction de XGBoost, entraremos si "bin" es 1 o 'meta_label' es 1.
        # Combinemos el Dataframe final:
        df_labeled = df_final.join(tbm_result[['bin', 'ret', 'holding_time_hours']], how='inner')
        self.holding_time_hours = df_labeled['holding_time_hours']
        df_labeled["target"] = (df_labeled["bin"] == 1).astype(int)
        
        # Calcular los retornos forward de simulación usando el 'ret' real obtenido del Triple Barrier Method
        # para backtesting fidedigno (cuánto ganó al tocar SL o PT en la vida real).
        df_labeled["simulated_fwd_ret_24h"] = df_labeled["ret"]
        
        # Filtrar solo columnas válidas — excluir columnas 100% NaN antes del dropna
        feature_candidates = [c for c in features_list if c in df_labeled.columns]
        meta_cols = ['target', 'simulated_fwd_ret_24h']
        if "HMM_Semantic" in df_labeled.columns:
            meta_cols.append("HMM_Semantic")
        cols_to_keep = list(set(feature_candidates + meta_cols))
        
        df_subset = df_labeled[cols_to_keep].copy()
        
        # Excluir columnas de features que son 100% NaN (no generadas en este pipeline)
        fully_empty_feats = [c for c in feature_candidates if df_subset[c].isna().all()]
        if fully_empty_feats:
            logger.warning(f"Features 100% vacías excluidas: {fully_empty_feats}")
            df_subset = df_subset.drop(columns=fully_empty_feats)
        
        valid_features = [c for c in feature_candidates if c not in fully_empty_feats]

        # — Opción B: XGBoost maneja NaN nativamente (sin fillna, sin dropna agresivo)
        # Solo eliminamos filas donde el TARGET o el RET de simulación sean NaN
        # (estos deben ser siempre completos para poder entrenar/evaluar).
        # Las features con NaN parcial (LongShortRatio, OI_USD, ETF prices, etc.)
        # las deja pasar — XGBoost aprende la dirección óptima del split para NaN.
        # Resultado: preservamos 43.793 filas en lugar de ~14.000 con dropna agresivo.
        df_clean = df_subset.dropna(subset=meta_cols)
        
        # FASE 2: Filtrado por Régimen
        if self.regime_list is not None and "HMM_Semantic" in df_clean.columns:
            logger.info(f"Filtro de Régimen Semántico Activo: {self.regime_name} -> {self.regime_list}")
            mask = df_clean["HMM_Semantic"].isin(self.regime_list)
            df_clean = df_clean[mask]
            n_eventos = len(df_clean)
            logger.info(f"  Eventos en régimen {self.regime_name}: {n_eventos}")
            # [BUG-RANGE-01 FIX] Guardia explícita: si el filtro HMM deja 0 eventos el dataset
            # está vacío y cualquier operación posterior (.max(), .isna().mean()) lanzará
            # ValueError: zero-size array. Mejor fallar aquí con mensaje claro para que
            # run_all() lo capture y loguee correctamente en lugar de un crash críptico.
            if n_eventos == 0:
                # [FIX-REGIME-POOL-01] Si el régimen específico tiene 0 eventos,
                # activar modo universal: entrenar sobre TODOS los datos IS con
                # HMM_Semantic como feature contextual en lugar de omitir el agente.
                # Esto evita que el signal funnel colapse a 0 señales cuando el régimen
                # OOS difiere del IS (ej: VOLATILE_RANGE ausente en 2024+).
                print(
                    f"[FIX-REGIME-POOL-01] Agente '{self.regime_name}': 0 eventos en régimen "
                    f"{self.regime_list} — activando modo UNIVERSAL (todos los datos IS). "
                    f"HMM_Semantic se usará como feature contextual. "
                    f"Etiquetas disponibles: {df_clean['HMM_Semantic'].value_counts().to_dict()}"
                )
                logger.warning(
                    f"[FIX-REGIME-POOL-01] Agente '{self.regime_name}': régimen no representado "
                    f"en IS — modo universal activado (N_total={len(df_clean)}). "
                    f"HMM_Semantic injertada como feature contextual."
                )
                # Modo universal: usar df_clean completo (sin filtro de régimen)
                # HMM_Semantic se queda como feature si está en valid_features
                df_clean = df_subset.dropna(subset=meta_cols)  # reset al dataset completo
                n_eventos = len(df_clean)
                self._universal_mode = True  # flag para diagnóstico

            # Loguear N por agente siempre (diagnóstico obligatorio)
            _universal = getattr(self, '_universal_mode', False)
            print(
                f"[AUDIT-REGIME-N] Agente='{self.regime_name}' | "
                f"n_train={n_eventos} | universal_mode={_universal} | "
                f"regime_list={self.regime_list}"
            )
            logger.info(
                f"[AUDIT-REGIME-N] Agente='{self.regime_name}' n_train={n_eventos} "
                f"universal_mode={_universal}"
            )

            # [SOP-R8-GATE] Gate estadístico: SOP Rule R8 exige mínimo 30 trades para
            # inferencia estadística confiable. Con n_train < min_trades el modelo XGBoost
            # es degenerado (min_child_weight=20 > n_total/2) y produce probabilidades ≈ base_rate.
            # NOTA: En modo universal, el N es siempre suficiente — el gate solo se aplica
            # cuando el régimen fue filtrado y tiene pocos eventos genuinos.
            try:
                from config.settings import cfg as _cfg_sop_gate
                _sop_min = int(_cfg_sop_gate.sop.paper_min_trades)
            except Exception:
                _sop_min = 30
            if n_eventos < _sop_min and not getattr(self, '_universal_mode', False):
                _msg = (
                    f"[BUG-RANGE-01] [SOP-R8-GATE] Agente '{self.regime_name}': "
                    f"n_train={n_eventos} < {_sop_min} (sop.paper_min_trades). "
                    f"Modelo degenerado — insuficiente evidencia estadística (SOP Rule R8). "
                    f"Agente omitido para esta ventana."
                )
                print(f"[SOP-R8-GATE] {_msg}")
                raise ValueError(_msg)
        elif self.regime_list is not None:
            logger.warning("Filtro de Régimen solicitado pero HMM_Semantic no está en las features. Se entrenará con todos los datos.")

        # Verificar que no hay features 100% NaN en el período de training
        nan_pct = df_clean[valid_features].isna().mean()

        # [FIX-NAN-BEAR-01] Descartar dinamicamente features con >40% de NaNs en este subset de regimen
        # Previene que XGBoost asuma que la ausencia de dato es informacion estructural valida.
        high_nan = nan_pct[nan_pct > 0.40]
        if not high_nan.empty:
            _dropped_cols = high_nan.index.tolist()
            logger.warning(f"  [FIX-NAN-BEAR-01] Descartando {len(_dropped_cols)} features con >40% NaN en regimen {self.regime_name}: {_dropped_cols[:5]}...")
            valid_features = [f for f in valid_features if f not in _dropped_cols]
            nan_pct = nan_pct.drop(_dropped_cols)

        partial_nan = nan_pct[(nan_pct > 0) & (nan_pct < 1.0)]
        if not partial_nan.empty:
            logger.info(f"Features con NaN parcial (XGBoost nativo): {len(partial_nan)} cols "
                        f"(max: {partial_nan.max():.1%} en '{partial_nan.idxmax()}')")

        self.features = valid_features
        self.X = df_clean[self.features]
        self.y = df_clean['target']
        self.close_rets = df_clean['simulated_fwd_ret_24h']

        # — Debug guards post-load —
        check_df_sanity(self.X, label="XGBoost.load_dataset.X")
        check_target_balance(self.y, label="XGBoost.target")
        log_memory_usage("post-load_dataset")

        # —— [DATAFLOW-IMPORT-XGB-01] Feature availability audit ————————————————————————
        # Detecta desalineamiento entre selected_features.json y features_train.parquet.
        # Si muchas features esperadas no existen, el modelo se entrena con features incompletas.
        _feats_expected   = features_list  # las que pide selected_features.json
        _feats_available  = [f for f in _feats_expected if f in df_labeled.columns]
        _feats_missing    = [f for f in _feats_expected if f not in df_labeled.columns]
        _feats_fully_nan  = fully_empty_feats
        _pct_missing      = len(_feats_missing) / max(len(_feats_expected), 1)
        logger.info(
            f"  [DATAFLOW-IMPORT-XGB-01] Features: {len(_feats_available)}/{len(_feats_expected)} "
            f"disponibles en parquet. "
            f"Faltantes ({len(_feats_missing)}): {_feats_missing[:5]}{'...' if len(_feats_missing) > 5 else ''}. "
            f"100%% NaN: {_feats_fully_nan}."
        )
        if _pct_missing > 0.20:
            logger.warning(
                f"  [DATAFLOW-IMPORT-XGB-01] ALERTA: {_pct_missing:.0%} de features esperadas NO EXISTEN "
                f"en features_train.parquet. "
                f"Probable causa: selected_features.json desactualizado o feature_pipeline.py no regenerado. "
                f"Re-ejecutar Fase 3A (feature_pipeline) antes de entrenar."
            )
        # Aviso si features_train no tiene columnas HMM — el modelo puede estar usando features incorrectas
        for _hc in ["HMM_Regime", "HMM_Semantic"]:
            if _hc in _feats_expected and _hc not in df_labeled.columns:
                logger.warning(
                    f"  [DATAFLOW-IMPORT-XGB-01] {_hc} requerida por features_list pero AUSENTE en train. "
                    f"El modelo XGBoost no podra usar el regimen HMM como feature."
                )
        # — [DATAFLOW-IMPORT-XGB-02] Dimensionality & Target audit —————————————————————————
        _t_min, _t_max = self.X.index.min().date(), self.X.index.max().date()
        logger.success(
            f"[DATAFLOW-IMPORT-XGB-02] Dataset de Entrenamiento Cargado y Validado | "
            f"shape={self.X.shape} | fechas={_t_min} -> {_t_max} | "
            f"Target Balance: {self.y.sum()} / {len(self.y)} ({self.y.mean():.1%} positivos)"
        )
        # ——————————————————————————————————————————————————————————————————————————————————————
        # [REGIME-DIST-01] Distribucion IS + WR IS por regimen HMM.
        # self.X y self.y disponibles. Detecta regimenes degenerados antes de Optuna.
        try:
            _hmm_col_rd = None
            if "HMM_Semantic" in self.X.columns and self.X["HMM_Semantic"].notna().any():
                _hmm_col_rd = "HMM_Semantic"
            elif "HMM_Regime" in self.X.columns and self.X["HMM_Regime"].notna().any():
                _hmm_col_rd = "HMM_Regime"
            if _hmm_col_rd:
                _rc     = self.X[_hmm_col_rd].value_counts()
                _tot    = len(self.X)
                _wr_reg = self.y.groupby(self.X[_hmm_col_rd].values).mean()
                logger.info(
                    "[REGIME-DIST-01] Distribucion IS (agente={} | n_total={}):",
                    self.regime_name or "global", _tot
                )
                for _rn, _nr in _rc.items():
                    _pct  = _nr / _tot * 100
                    _wr   = float(_wr_reg.get(_rn, float("nan"))) * 100
                    _flag = " [!!] <200 MUESTRAS" if _nr < 200 else (
                            " [!!] WR>70% (overfitting?)" if _wr > 70 else " [OK]"
                    )
                    logger.info(
                        "[REGIME-DIST-01]   {:30s} n={:5d} ({:5.1f}%) WR_IS={:.1f}%  {}",
                        str(_rn), _nr, _pct, _wr, _flag
                    )
                _escasos = [str(r) for r, n in _rc.items() if n < 200]
                if _escasos:
                    logger.warning(
                        "[REGIME-DIST-01] ALERTA: {} regimen(es) con <200 muestras -> riesgo modelo degenerado: {}",
                        len(_escasos), _escasos
                    )
            else:
                logger.info("[REGIME-DIST-01] HMM no disponible en X -- distribucion omitida.")
        except Exception as _e_rd2:
            logger.warning("[REGIME-DIST-01] Diagnostico regimen fallido (no bloqueante): {}", _e_rd2)

        return self.X, self.y

        
    # LEGACY-01 ELIMINADO (2026-03-17): _create_wfa_splits() — nunca activo desde P1-6.
    # El WFA (Walk-Forward Analysis) fue reemplazado por CPCV Real en P1-6 (2026-03-07).
    # Ver historial en diario.md: Fix M-04 / P1-6.

    def _create_cpcv_splits(self):
        """
        CPCV Real segun Lopez de Prado (2018) Ch.12. [ACTIVO DESDE P1-6]
        C(n_groups, k_test) combinaciones de grupos test (no secuencial).
        Con n_groups=10 (sop.cpcv_groups=10), k_test=2: C(10,2)=45 caminos OOS vs 8 del WFA.
        Con n_groups=6  (sop.cpcv_groups=6,  M-40):    C(6,2)=15  caminos OOS.

        Esta función ES la activa en objective() desde P1-6 (2026-03-07).
        Requiere ~5-6x más tiempo de cómputo que WFA — justificado para DSR.
        """
        from itertools import combinations
        n_samples = len(self.X)
        timestamps = self.X.index

        # [MEJORA-CPCV-01] CPCV Dinámico para evitar caer en pocos splits
        if n_samples < 2000:
            cpcv_groups_din = self._compute_optimal_cpcv_groups(n_samples, PURGE_H)
            logger.info(f"[MEJORA-CPCV-01] Dataset pequeño ({n_samples} eventos). Reduciendo CPCV_GROUPS de {CPCV_GROUPS} a {cpcv_groups_din} para maximizar splits válidos.")
            current_cpcv_groups = cpcv_groups_din
        else:
            current_cpcv_groups = CPCV_GROUPS

        # Dividir en current_cpcv_groups grupos iguales
        groups = np.array_split(np.arange(n_samples), current_cpcv_groups)

        # [FIX-P0] min_train_size proporcional al dataset total para evitar descartar splits en regimenes minoritarios
        min_train_size = max(50, int(len(self.X) * 0.10))
        splits = []
        k_test = 2  # numero de grupos que forman el test set
        for test_gidx in combinations(range(current_cpcv_groups), k_test):
            test_idx = np.concatenate([groups[i] for i in test_gidx])
            train_idx = np.concatenate([groups[i] for i in range(current_cpcv_groups)
                                        if i not in test_gidx])

            # BUG-10 FIX (2026-03-08): purge POR BLOQUE de test independiente.
            # El fix anterior (BUG-8) aplicaba purge sobre el SPAN completo del test
            # (timestamps[test_idx[0]] → timestamps[test_idx[-1]]).
            # Para grupos NO CONTIGUOS (ej. grupos 0+9 = extremos del dataset),
            # ese span cubre TODO el periodo → train_purged = 0 (1 split descartado).
            # Fix correcto: purgar independientemente respecto a CADA bloque de test.
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue

            # Calcular máscara keep: un punto de train se mantiene si está
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

            if len(train_idx) < min_train_size or len(test_idx) < 50:
                continue
            splits.append((train_idx, test_idx))

        import math as _math
        n_paths_total = _math.comb(current_cpcv_groups, 2)
        if current_cpcv_groups < 8:
            # ARCH-03 warning (2026-03-17): menos de C(8,2)=28 paths — robustez estadística baja.
            # Con 15 paths el IC del DSR es ~3× más amplio que con 45 paths → más fácil sobreajustar.
            # Para aumentar: xgboost.n_purged_splits: 10 en settings.yaml (sin tocar código).
            _eta_h = OPTUNA_TRIALS * n_paths_total * 4.0 / 3600  # ~4s/fold estimado
            logger.warning(
                "[ARCH-03] CPCV_GROUPS=%d → C(%d,2)=%d paths activos (ROBUSTEZ BAJA). "
                "DSR con %d paths tiene IC ~3x mas amplio que con 45 paths. "
                "Para produccion: n_purged_splits=10 en settings.yaml (ETA ~%.0fH con %d trials).",
                current_cpcv_groups, current_cpcv_groups, n_paths_total,
                n_paths_total,
                _math.comb(10, 2) * OPTUNA_TRIALS * 4.0 / 3600,
                OPTUNA_TRIALS
            )
        else:
            logger.info(
                "[CPCV REAL] %d grupos → C(%d,2)=%d paths — robustez estadistica adecuada.",
                current_cpcv_groups, current_cpcv_groups, n_paths_total
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
            penalizando estrategias buenas — seria INCORRECTO.
          - T=mean(test_lengths) captura la longitud tipica de cada fold, que es
            exactamente el T del paper (Bailey & LdP 2014, eq.2).
        """
        if len(fold_sharpes) < 2: return 0.0

        sr_mean = np.mean(fold_sharpes)
        # [BUG-C2 FIX] Retornar 0.0 en lugar de float('nan') cuando sr_mean<=0.
        # float('nan') pasado a Optuna trial.report() causa que MedianPruner corte
        # trials válidos y no active correctly should_prune(). DSR es [0,1] — 0.0
        # es el valor correcto para estrategias con Sharpe negativo (rechazadas).
        if sr_mean <= 0: return 0.0  # [BUG-C2 FIX] era: float("nan")

        # [P0-3-FIX-CROSS-VAR 2026-06-04] Bailey & Lopez de Prado (2014) exige la varianza TRANSVERSAL.
        # Usar la varianza temporal de los folds borraba la penalización de Multiple Testing
        # para estrategias estables. Asignamos varianza transversal conservadora (std=1.0)
        # para que la barrera crezca implacablemente por número de pruebas.
        sr_std_cross = 1.0
        gamma = 0.5772156649

        # T = longitud temporal promedio de cada path OOS — ver docstring.
        if test_lengths and len(test_lengths) > 0:
            T = int(np.mean(test_lengths))   # Correcto: longitud promedio por fold
        else:
            # BUG-XGB-01 FIX (2026-04-06): El fallback T=n_trials era matemáticamente
            # incorrecto. n_trials=600 (trials Optuna) != T (obs por fold ~12000).
            # Consecuencia: sr_star inflado/deflado según caso, DSR distorsionado.
            # Fix: estimado conservador basado en tamaño mínimo esperado de fold CPCV.
            # En CPCV C(G,2), cada fold test tiene ~N*2/G samples (G=10 → 20% de N).
            # Sin N disponible en este scope, estimamos N≥6000 (mínimo razonable luna).
            T = max(1000, n_trials * 20)  # nunca n_trials a secas
            logger.warning(
                "[BUG-XGB-01] DSR: test_lengths vacío — usando T estimado conservador=%d. "
                "Verificar que CPCV no descartó todos los splits por purge excesivo.", T
            )

        z1 = norm.ppf(1 - 1.0 / max(n_trials, 2))
        z2 = norm.ppf(1 - 1.0 / max(n_trials * math.e, 2))

        sr_star = sr_std_cross * ((1 - gamma) * z1 + gamma * z2)
        
        # [FIX-MATH-OPTUNA-01-V2]: Restablecer división por T y escalar constante por F=8760.0
        # para que el Z-Score sea matemáticamente válido sin polarizarse.
        freq = 8760.0
        var_sr = (freq + 0.5 * (sr_mean ** 2)) / T
        z_score = (sr_mean - sr_star) / np.sqrt(var_sr)
        dsr = float(norm.cdf(z_score))
        
        # Trace print for mathematical correction (fixbugsprints.md / fixaplly.md)
        trace_msg = f"[FIX-MATH-OPTUNA-01-V2] DSR recalculado: sr_mean={sr_mean:.4f}, sr_star={sr_star:.4f}, T={T}, dsr={dsr:.4f}"
        print(trace_msg)  # debug
        logger.info(trace_msg)
        return dsr

    def _compute_sample_weights(self, index: pd.Index) -> np.ndarray:
        """
        ARCH-02 fix (2026-03-17): decaimiento exponencial configurable por año.

        weight_i = exp(-alpha × años_desde_train_end)

          alpha=0.0 → uniforme (sin énfasis temporal, para diagnóstico)
          alpha=0.5 → suave — ratio año0:año-1 ≈ 1.6:1  (DEFAULT)
          alpha=1.0 → moderado — ratio ≈ 2.7:1
          alpha=1.6 → agresivo — ratio ≈ 5.0:1  (equivalente al esquema anterior 5x/1x)

        Configurable en settings.yaml → xgboost.weight_decay_alpha
        Sin hardcodes de años — completamente dinámico a partir de train_end.
        """
        # [BUG-C1 FIX] Guard correcto: isinstance en lugar de hasattr(.year).
        # pd.DatetimeIndex SIEMPRE tiene .year, por lo que el guard original nunca activaba.
        # Si el índice es entero, pd.to_datetime() lo convierte a epoch Unix (1970s)
        # → years_ago = 2024-1970 = 54 → exp(-0.5*54) ≈ 0 → todos los pesos à 0.
        ts = pd.to_datetime(index, errors='coerce', utc=True)
        if not isinstance(ts, pd.DatetimeIndex) or ts.isna().all():
            logger.warning("[BUG-C1] Índice no es DatetimeIndex válido. Pesos uniformes.")
            return np.ones(len(index))
        if ts.min().year < 2000:
            logger.warning(
                "[BUG-C1] Índice con fechas pre-2000 (%s) — posiblemente epoch Unix. Pesos uniformes.",
                ts.min()
            )
            return np.ones(len(index))
        try:
            from config.settings import cfg as _cfg_sw
            _train_end_year = pd.Timestamp(_cfg_sw.temporal_splits.train_end).year
            
            # [FIX-HMM-AMNESIA 2026-06-14] Prevenir amnesia estructural en agentes HMM raros
            # Si es un agente de régimen específico, el decaimiento temporal destruye su
            # memoria de estados pasados. Usamos hmm_weight_decay (por defecto 0.0).
            if getattr(self, 'regime_name', None) is not None and getattr(self, '_universal_mode', False) is False:
                _alpha = float(_cfg_sw.xgboost.hmm_weight_decay)
                if not getattr(self.__class__, '_hmm_decay_logged', False):
                    logger.info(f"[FIX-HMM-AMNESIA] Agente '{self.regime_name}': desactivando decaimiento temporal (alpha={_alpha}) para proteger memoria de Markov.")
                    self.__class__._hmm_decay_logged = True
            else:
                _alpha = float(_cfg_sw.xgboost.weight_decay_alpha)
        except Exception:
            _train_end_year = ts.year.max()
            _alpha = 0.5

        years_ago = np.clip(_train_end_year - ts.year.to_numpy(), 0, None).astype(float)
        weights = np.exp(-_alpha * years_ago)

        # [HOLDING-TIME-PENALTY] Penalización proporcional para Shorts (Mejora 3)
        # El mercado baja por el ascensor. Shorts largos implican riesgo de Short Squeeze.
        _dir = getattr(self, 'direction_mode', getattr(self, 'native_direction', 'both'))
        if _dir == 'short' and hasattr(self, 'holding_time_hours'):
            from config.settings import cfg as _cfg_ht
            penalty_factor = float(_cfg_ht.xgboost.short_holding_time_penalty)
                
            _ht = self.holding_time_hours.loc[index].values
            # Penalty inverso al holding_time. Más tiempo retenido = menos peso = la loss penaliza este trade.
            _ht_penalty = np.exp(-penalty_factor * _ht)
            weights *= _ht_penalty
            
            _verbose_debug = bool(int(_os.environ.get("LUNA_VERBOSE", "0")))
            if _verbose_debug and not getattr(self.__class__, '_ht_logged', False):
                logger.debug(f"[SHORT-HT-PENALTY] Aplicando penalización de Holding Time a Shorts. factor={penalty_factor}. Min multiplier: {np.min(_ht_penalty):.2f}x")
                self.__class__._ht_logged = True

        _verbose_debug = bool(int(_os.environ.get("LUNA_VERBOSE", "0")))
        if _verbose_debug and not getattr(self.__class__, '_sw_logged', False):
            logger.debug(
                "[R20-B/ARCH-02] sample_weights config: alpha=%.2f, train_end=%d "
                "→ pesos unitarios [año0=%.3f, año-1=%.3f, año-2=%.3f] "
                "(este mensaje se emite UNA sola vez — throttle activo)",
                _alpha, _train_end_year,
                np.exp(0.0), np.exp(-_alpha), np.exp(-2.0 * _alpha)
            )
            self.__class__._sw_logged = True  # throttle: 1 log por run, no por split
        return weights

    def _get_focal_loss_obj(self, scale_pos_weight=1.0, gamma: float | None = None):
        """
        [A1] Genera un custom objective Focal Loss para XGBClassifier.
        El solver de XGBoost requiere que la función objetivo retorne Gradiente y Pseudo-Hessiana
        calculadas respecto al raw margin (log-odds).

        [P1-FIX] gamma ahora puede ser pasado explícitamente (desde el trial Optuna),
        o leído del cfg como fallback. Esto garantiza que gamma sea validado CPCV.
        """
        if gamma is None:
            try:
                from config.settings import cfg as _cfg_fl
                gamma = float(_cfg_fl.xgboost.focal_loss_gamma)
            except Exception:
                gamma = 2.0

        def focal_loss(y_true, y_pred, sample_weight=None):
            # y_pred en XGBoost custom objectives entra en raw log-odds (margin).
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
        Función objetivo Optuna: maximiza DSR sobre 45 paths CPCV.

        Función objetivo Optuna: maximiza DSR sobre 45 paths CPCV.

        BUG-R12-03 fix (2026-03-10): todos los rangos Optuna leídos desde
        cfg.xgboost.optuna_search_space en settings.yaml — sin números mágicos.
        Justificación de cada rango documentada en settings.yaml.
        Nuevos parámetros: gamma (L0), reg_alpha (L1), reg_lambda (L2), scale_pos_weight.
        """
        # Leer espacio de búsqueda desde settings.yaml — REGLA: sin hardcodes
        sp = _cfg_xgb.xgboost.optuna_search_space


        # ── [LUNA-V2-REGULARIZATION] Strict search space bounds to prevent overfitting ──
        # We enforce strict regularizers as per Luna v2 architecture.
        # min_child_weight MUST be strictly at least 30 to prevent leaf-level overfitting.
        # max_depth is capped at 4.
        
        # [REGULARIZATION-DYN-01] Forzado matemático de underfitting por régimen
        try:
            _dyn_reg = _cfg_xgb.xgboost.dynamic_regime_regularization
            # Check if self.regime_name exists in yaml, if not error out
            _regime_caps = getattr(_dyn_reg, self.regime_name)
            _md_cap = int(_regime_caps.max_depth_cap)
            _gamma_fl = float(_regime_caps.gamma_floor)
        except Exception as e:
            raise RuntimeError(f"Falta el bloque dynamic_regime_regularization en settings.yaml o el régimen '{self.regime_name}' no está definido. (Política No-Fallback): {e}")

        _dyn_md_min = min(sp.max_depth_min, _md_cap)
        _dyn_md_max = min(sp.max_depth_max, _md_cap)
        _dyn_gamma_min = max(sp.gamma_min, _gamma_fl)
        _dyn_gamma_max = max(sp.gamma_max, _gamma_fl)

        logger.info(
            f"[REGULARIZATION-DYN-01] Optuna restringido dinámicamente para régimen '{self.regime_name}': "
            f"max_depth=[{_dyn_md_min}, {_dyn_md_max}], gamma=[{_dyn_gamma_min}, {_dyn_gamma_max}]"
        )

        _mcw_min   = sp.min_child_weight_min
        _mcw_max   = sp.min_child_weight_max

        # [OOF-CALIB-V2 2026-06-03] Cota inferior adaptativa de min_child_weight
        # Si el dataset es pequeno, permitir min_child_weight mas bajos (minimo 2)
        _n_train_agent = len(self.X)
        _mcw_min_adaptive = int(max(2, min(_mcw_min, _n_train_agent // 100)))
        if _mcw_min_adaptive < _mcw_min:
            print(
                f"[OOF-CALIB-V2] min_child_weight_min adaptativo: {_mcw_min} -> {_mcw_min_adaptive} "
                f"(n_train={_n_train_agent}) -- permite que el arbol crezca en datasets chicos"
            )
            logger.info(
                "[OOF-CALIB-V2] n_train=%d -> min_child_weight_min=%d (era %d)",
                _n_train_agent, _mcw_min_adaptive, _mcw_min
            )
            _mcw_min = _mcw_min_adaptive

        # [FIX-CRIT-02-MCW-ADAPTIVE 2026-05-30] Adaptar min_child_weight_max al n_train del AGENTE.
        # Con n_train=766 (bear) y min_child_weight_max=100, ningun arbol puede crecer:
        # depth=4 -> 766/16=47 samples/hoja < MCW=100 -> todos los arboles son triviales.
        # El floor: MCW_max = min(MCW_max_settings, max(10, n_train // 20))
        # Con n=766: min(100, max(10, 38)) = 38 -> depth=4: 766/16=47 >= 38 -> arboles crecen.
        _n_train_agent = len(self.X)
        _mcw_max_adaptive = min(_mcw_max, max(10, _n_train_agent // 20))
        if _mcw_max_adaptive < _mcw_max:
            print(  # RULE[fixbugsprints.md]
                f"[FIX-CRIT-02-MCW-ADAPTIVE] min_child_weight_max reducido: {_mcw_max} -> {_mcw_max_adaptive}"
                f" (n_train={_n_train_agent}, n_train//20={_n_train_agent//20}) -- previene model collapse"
            )
            logger.warning(
                "[FIX-CRIT-02-MCW-ADAPTIVE] n_train=%d pequeno -> min_child_weight_max=%d (era %d) "
                "-- prevencion de model collapse activa",
                _n_train_agent, _mcw_max_adaptive, _mcw_max
            )
            _mcw_max = _mcw_max_adaptive
        
        if trial.number == 0:
            print(f"[LUNA-V2-REGULARIZATION] STRICT BOUNDS active | min_child_weight range=[{_mcw_min}, {_mcw_max}], max_depth in [{sp.max_depth_min}, {sp.max_depth_max}], gamma_max={_dyn_gamma_max}")
            print(  # [FIX-REG-01 2026-05-31] Trazabilidad de nuevos bounds anti-sobreregularizacion
                f"[FIX-REG-01] Bounds activos: "
                f"lr_min={sp.learning_rate_min} (era 0.005) | "
                f"reg_alpha_max={sp.reg_alpha_max} (era 50.0) | "
                f"mcw_min={_mcw_min} (era 30) | "
                f"mcw_max={_mcw_max} (era 100) | "
                f"n_train={_n_train_agent}"
            )
            logger.info(f"[LUNA-V2-REGULARIZATION] STRICT BOUNDS active | min_child_weight range=[{_mcw_min}, {_mcw_max}], max_depth in [{sp.max_depth_min}, {sp.max_depth_max}], gamma_max={_dyn_gamma_max}")
            logger.info(
                "[FIX-REG-01] lr_min=%.4f | reg_alpha_max=%.2f | mcw=[%d,%d] | n_train=%d",
                sp.learning_rate_min, sp.reg_alpha_max, _mcw_min, _mcw_max, _n_train_agent
            )

        params = {
            'max_depth':        trial.suggest_int(
                'max_depth', _dyn_md_min, _dyn_md_max),
            'learning_rate':    trial.suggest_float(
                'learning_rate', sp.learning_rate_min, sp.learning_rate_max, log=True),
            'subsample':        trial.suggest_float(
                'subsample', sp.subsample_min, sp.subsample_max),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', sp.colsample_bytree_min, sp.colsample_bytree_max),
            'min_child_weight': trial.suggest_int(
                'min_child_weight', _mcw_min, _mcw_max),
            # Regularización (BUG-R12-03): ausentes en versión anterior
            'gamma':            trial.suggest_float(
                'gamma', _dyn_gamma_min, _dyn_gamma_max),
            'reg_alpha':        trial.suggest_float(
                'reg_alpha', sp.reg_alpha_min, sp.reg_alpha_max, log=True),
            'reg_lambda':       trial.suggest_float(
                'reg_lambda', sp.reg_lambda_min, sp.reg_lambda_max, log=True),
            'scale_pos_weight': trial.suggest_float(
                # SPW-AUTO-01: rango calculado desde labels TBM reales (ver tune_hyperparameters)
                # Fallback a YAML solo si _spw_min no esta seteado (e.g. llamada directa sin tuning)
                'scale_pos_weight',
                getattr(self, '_spw_min', None) or sp.scale_pos_weight_min,
                getattr(self, '_spw_max', None) or sp.scale_pos_weight_max,
            ),
            # Fijos — no son hiperparámetros de modelo, son de arquitectura
            'objective':    'binary:logistic',
            'tree_method':  'hist',
            'n_jobs':       -1,
            'random_state': 42
        }

        # [A1] Inyectar Focal Loss o Monetary Loss si están activados en settings.yaml
        use_focal_loss = False
        use_monetary_loss = False
        try:
            from config.settings import cfg as _cfg_opts
            use_focal_loss = bool(_cfg_opts.xgboost.use_focal_loss)
            use_monetary_loss = bool(_cfg_opts.fase2.use_monetary_loss)
        except Exception:
            pass

        if use_monetary_loss:
            from luna.losses.monetary_loss import get_monetary_pnl_loss
            params['objective'] = get_monetary_pnl_loss()
            use_focal_loss = False
        elif use_focal_loss:
            _spw = params.get('scale_pos_weight', 1.0)
            # [P1-FIX] focal_loss_gamma ahora entra al espacio Optuna para validación CPCV.
            # Se lee el rango desde settings.yaml → optuna_search_space.focal_loss_gamma_min/max
            # Si el rango no existe en settings, usa default [0.5, 4.0] (rango empírico seguro).
            try:
                _fl_gamma_min = float(_cfg_opts.xgboost.optuna_search_space.focal_loss_gamma_min)
                _fl_gamma_max = float(_cfg_opts.xgboost.optuna_search_space.focal_loss_gamma_max)
            except Exception:
                _fl_gamma_min, _fl_gamma_max = 0.5, 4.0
            _fl_gamma = trial.suggest_float('focal_loss_gamma', _fl_gamma_min, _fl_gamma_max)
            
            # [HEURISTIC PRUNING] Gamma vs SPW Coupling
            # Focal Loss degenerates the minority class if gamma is very high but SPW is low.
            # Reject absurd combinations mathematically to optimize convergence.
            _spw_ideal = self._spw_ideal
            if _fl_gamma > 2.0 and _spw < _spw_ideal:
                raise optuna.TrialPruned("Degenerate: High focal gamma requires higher SPW.")
            if _fl_gamma < 1.0 and _spw > _spw_ideal * 1.5:
                raise optuna.TrialPruned("Degenerate: Low focal gamma does not require excessive SPW.")
                
            params['objective'] = self._get_focal_loss_obj(scale_pos_weight=_spw, gamma=_fl_gamma)

        # FIX-CPCV-CACHE-01: usar splits precalculados en tune_hyperparameters().
        # LOS 45 SPLITS AHORA SOLO SE USAN AL FINAL DEL SCRIPT PARA TELEMETRIA EX-POST.
        # OPTUNA SOLO USA TIMESERIESSPLIT (5 folds) PARA EVALUACION Y PRUNING.
        try:
            from config.settings import cfg as _cfg_metric
            _optuna_metric = str(str(_cfg_metric.xgboost.optuna_metric)).lower()
            _purge_gap = int(_cfg_metric.sop.purge_hours)
        except Exception as e:
            raise RuntimeError(f"Falta purge_hours o parametros metricos en settings.yaml (SOP No-Fallback): {e}")

        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import brier_score_loss, log_loss
        import numpy as np
        import xgboost as xgb

        # [FIX-CRIT-01-NSPLITS 2026-05-30] n_splits adaptativo al tamanio del agente.
        # BUG: _n_months = len(X)//4320 = 766//4320 = 0 -> n_splits=3 pero era coincidencia correcta.
        # El problema real: con n=766 y n_splits=3, cada fold_test tiene ~255 filas
        # PERO el DSR con T=255 y n_trials=1-5 puede ser cercano a 0 si los trials son pocos.
        # Fix: calcular n_splits explicitamente por tamanio del agente, no por meses calendario.
        # Regla: n<2000 -> 3 splits (T_test~33%), n<5000 -> 5 splits, n>=5000 -> 6 splits.
        _n_train_for_splits = len(self.X)
        if _n_train_for_splits < 2000:
            _n_splits_is = 3
        elif _n_train_for_splits < 5000:
            _n_splits_is = 5
        else:
            _n_splits_is = 6

        # [FIX-CV-CRASH 2026-06-22] Calcular test_size seguro para evitar ValueError de TimeSeriesSplit
        _test_size_explicit = max(5, (_n_train_for_splits - _purge_gap) // (_n_splits_is + 1))
        while _test_size_explicit * (_n_splits_is + 1) + _purge_gap > _n_train_for_splits and _n_splits_is > 1:
            _n_splits_is -= 1
            _test_size_explicit = max(5, (_n_train_for_splits - _purge_gap) // (_n_splits_is + 1))

        if trial.number == 0:
            print(  # RULE[fixbugsprints.md]
                f"[FIX-CRIT-01-NSPLITS] n_splits_is={_n_splits_is} test_size={_test_size_explicit} para n_train={_n_train_for_splits}"
                f" | purge_gap={_purge_gap}H"
            )
            logger.info(
                "[FIX-CRIT-01-NSPLITS] Optuna TimeSeriesSplit: n_splits=%d | test_size=%d | n_train=%d | gap=%dH",
                _n_splits_is, _test_size_explicit, _n_train_for_splits, _purge_gap
            )
        _tscv = TimeSeriesSplit(n_splits=_n_splits_is, gap=_purge_gap, test_size=_test_size_explicit)

        _is_scores = []
        _proba_is_std = []  # [FIX-PROB-CLUSTERING]
        fold_sharpes = []
        test_lengths = []
        best_iters = []
        _naive_scores = []

        for fold_i, (_tr_i, _val_i) in enumerate(_tscv.split(self.X)):
            _clf_is = xgb.XGBClassifier(**params)
            _clf_is.set_params(n_estimators=2000, early_stopping_rounds=50)
            
            _es_idx = int(len(_tr_i) * 0.8)
            _tr_inner_i = _tr_i[:_es_idx]
            _es_i = _tr_i[_es_idx:]
            
            _sw_inner = self._compute_sample_weights(self.X.iloc[_tr_inner_i].index)
            _sw_es = self._compute_sample_weights(self.X.iloc[_es_i].index)
            
            _y_train_inner = self.y.iloc[_tr_inner_i]
            _y_val_inner = self.y.iloc[_val_i]
            
            # [FIX-SINGLE-CLASS-FOLD 2026-06-17] Prevenir ValueError(Expected: [0], got [1]) si fold extremo tiene solo una clase
            # Si el fold de entrenamiento tiene solo 1 clase, XGBoost no puede entrenar binary:logistic.
            # Lo mismo si el fold de validacion no tiene ambas clases para metricas como AUC/Logloss
            if len(np.unique(_y_train_inner)) < 2 or len(np.unique(_y_val_inner)) < 2:
                logger.debug(
                    f"[FIX-SINGLE-CLASS-FOLD] Fold {fold_i} ignorado: clases únicas insuficientes "
                    f"(train={len(np.unique(_y_train_inner))}, val={len(np.unique(_y_val_inner))})"
                )
                continue
                
            _clf_is.fit(
                self.X.iloc[_tr_inner_i], _y_train_inner, 
                sample_weight=_sw_inner,
                eval_set=[(self.X.iloc[_es_i], self.y.iloc[_es_i])],
                sample_weight_eval_set=[_sw_es],
                verbose=False
            )
            best_iters.append(_clf_is.best_iteration)

            if callable(params.get('objective')):
                _clf_is.set_params(objective='binary:logistic')
                _clf_is.get_booster().set_param({'objective': 'binary:logistic'})
            
            _proba_is = _clf_is.predict_proba(self.X.iloc[_val_i])[:, 1]
            _proba_is_std.append(np.std(_proba_is))  # [FIX-PROB-CLUSTERING]
            _y_val = self.y.iloc[_val_i].values
            
            # Calcular métricas principales (siempre calcular ambas para telemetría correcta)
            _brier_fold = brier_score_loss(_y_val, _proba_is)
            _logloss_fold = log_loss(_y_val, _proba_is)
            
            if _optuna_metric == 'brier':
                _is_scores.append(_brier_fold)
            elif _optuna_metric == 'logloss':
                _is_scores.append(_logloss_fold)
            else:
                _is_scores.append(0.0) # Dummy fallback
                
            if getattr(self, '_brier_scores', None) is None: self._brier_scores = []
            self._brier_scores.append(_brier_fold)

            # [FIX-IDEA-A-01] Calcular Naive Brier Score para el fold exacto
            _naive_p = _y_val.mean() if len(_y_val) > 0 else 0.50
            _naive_scores.append(_naive_p * (1 - _naive_p))

            # Calcular Sharpe para telemetria DSR
            # [ARCH-04-FIX-A 2026-06-02] Threshold alineado con deployment (sweep_min).
            # ANTES: 0.5 hardcoded - Optuna optimizaba para umbral que nunca se usa en prod.
            # AHORA: threshold_sweep_min - DSR telemetria refleja rendimiento OOS real.
            # El Brier principal no cambia (no usa threshold).
            try:
                from config.settings import cfg as _cfg_04a
                _optuna_deploy_thr = float(_cfg_04a.xgboost.threshold_sweep_min)
            except Exception:
                _optuna_deploy_thr = 0.45  # fallback conservador
            if not getattr(self, '_arch04_printed', False):
                print(  # RULE[fixbugsprints.md]
                    f'[ARCH-04-FIX-A] Optuna telemetria DSR threshold={_optuna_deploy_thr:.3f} '
                    f'(alineado con sweep_min={_optuna_deploy_thr:.3f}, antes=0.5)'
                )
                self._arch04_printed = True
            _preds_bin = np.where(_proba_is > _optuna_deploy_thr, 1.0, 0.0)
            if len(np.unique(_preds_bin)) == 1:
                test_lengths.append(len(_val_i))
            else:
                _rets_te = self.close_rets.iloc[_val_i].values
                # [FIX-MATH-OPTUNA-03 2026-06-07] Calcular Sharpe SOLO sobre trades reales, no sobre ceros
                trade_rets = _rets_te[_preds_bin == 1.0] - COST_PCT
                trade_rets_clean = trade_rets[~np.isnan(trade_rets)]
                
                # Exigir un mínimo de 3 trades por fold para evitar varianza cero o DSR distorsionado
                if len(trade_rets_clean) < 3 or np.std(trade_rets_clean) < 1e-8:
                    fold_sharpes.append(0.0)
                    test_lengths.append(len(_val_i))
                else:
                    mean_ret = np.mean(trade_rets_clean)
                    std_ret  = np.std(trade_rets_clean)
                    # No multiplicamos por sqrt(365*24) aquí: el test_lengths (n_trades) actuará
                    # como tamaño de muestra t en _compute_dsr (Bailey). Anualizar aquí distorsiona el T-Stat.
                    sharpe = mean_ret / std_ret
                    fold_sharpes.append(sharpe)
                    test_lengths.append(len(trade_rets_clean))  # N = número de apuestas (trades) reales

            # Reportar a Optuna para pruning
            # MedianPruner abortará trials ineficientes ahorrando 50% de cómputo extra.
            _dir = getattr(self.study, "direction", None)
            
            # Si optuna_metric es DSR, prunamos usando DSR parcial
            if _optuna_metric == 'dsr':
                if len(fold_sharpes) >= 2:
                    partial_dsr = self._compute_dsr(fold_sharpes, test_lengths=test_lengths, n_trials=OPTUNA_TRIALS)
                    if np.isnan(partial_dsr): partial_dsr = 0.0
                    _reported_val = -partial_dsr if _dir == optuna.study.StudyDirection.MINIMIZE else partial_dsr
                    trial.report(_reported_val, step=fold_i)
            else:
                # Si Brier/Logloss, prunamos directamente con la métrica IS parcial
                _reported_val = float(np.mean(_is_scores))
                # La métrica IS se asume que se debe MINIMIZAR (brier, logloss)
                if _dir == optuna.study.StudyDirection.MAXIMIZE:
                    _reported_val = -_reported_val
                trial.report(_reported_val, step=fold_i)

            if trial.should_prune():
                raise optuna.TrialPruned()

        # Al finalizar los 5 folds temporales, calculamos telemetría global
        _metric_val = float(np.mean(_is_scores)) if _is_scores else 1.0
        
        # [FIX-PROB-CLUSTERING] Detectar colapso de conviccion (asfixia del arbol)
        _mean_std = float(np.mean(_proba_is_std)) if _proba_is_std else 0.0
        _prob_clustering_penalty = False
        try:
            _proba_std_min = float(_cfg_xgb.stat.xgb_proba_std_min)
        except Exception as e:
            raise RuntimeError(f"[CRITICAL-SOP] Falta stat.xgb_proba_std_min en settings.yaml: {e}")
        if _mean_std < _proba_std_min:
            _prob_clustering_penalty = True
            print(f"[BUG-FIX-LOG 2026-06-14] [CLUSTERING-PENALTY] Activo: IS mean_std {_mean_std:.6f} < threshold {_proba_std_min:.6f}")
        
        _n_trials_local = max(2, trial.number + 1)
        _dsr_telemetry = self._compute_dsr(fold_sharpes, test_lengths=test_lengths, n_trials=_n_trials_local)

        if np.isnan(_dsr_telemetry):
            _telemetry_str = "DSR ex-post=N/A (sr_mean<=0)"
        elif _dsr_telemetry > 0:
            _telemetry_str = f"DSR ex-post={_dsr_telemetry:.4f} (telemetria, n={_n_trials_local})"
        else:
            _telemetry_str = f"DSR ex-post=~0.0000 (sr positivo pero DSR~0, n={_n_trials_local})"

        _dsr_safe = _dsr_telemetry if not np.isnan(_dsr_telemetry) and _dsr_telemetry > 0 else 0.0
        
        # Métrica compuesta o legacy
        if _optuna_metric in ('brier', 'logloss'):
            # [FIX-MATH-OPTUNA-03]: dsr_safe ya esta en escala [0.0, 1.0]. Dividirlo por 100.0
            # aniquilaba matematicamente la penalizacion sobre Brier/Logloss.
            _composite_val = (0.7 * _metric_val) - (0.3 * _dsr_safe)
            
            # [FIX-PROB-CLUSTERING] Castigo letal si Optuna hizo trampa
            if _prob_clustering_penalty:
                _composite_val = 1.0
                logger.debug(f"[V2-FIX-1] Penalizacion de Clustering Activa: STD={_mean_std:.4f}. Comp forzado a 1.0")
            else:
                logger.debug(
                    f"[V2-FIX-1] {_optuna_metric.upper()} IS={_metric_val:.4f} -> Comp={_composite_val:.4f} (splits={_n_splits_is}, gap={_purge_gap}h) | {_telemetry_str}"
                )
        else:
            _composite_val = _dsr_safe
            if _prob_clustering_penalty:
                _composite_val = -1.0 # Castigo en maximizacion
            logger.debug(f"[V2-FIX-1] DSR IS={_composite_val:.4f} (splits={_n_splits_is}, gap={_purge_gap}h) | {_telemetry_str}")

        trial.set_user_attr('naive_is', float(np.mean(_naive_scores)) if _naive_scores else 0.50)
        trial.set_user_attr('dsr_telemetry', _dsr_telemetry)
        trial.set_user_attr('metric_is', _metric_val)
        
        # [FIX-BRIER-LOGLOSS-MIXUP] Guardar Brier puro explícitamente para GATE-G2
        _mean_brier = float(np.mean(getattr(self, '_brier_scores', [1.0])))
        trial.set_user_attr('brier_is', _mean_brier)
        # Limpiar state para el siguiente trial
        if hasattr(self, '_brier_scores'): del self._brier_scores
        
        trial.set_user_attr('composite_loss', _composite_val)
        trial.set_user_attr('mean_best_iter', float(np.mean(best_iters)) if best_iters else 100)

        return _composite_val

    def _load_warmstart_params(self, wfb_cache_dir: "Path | None") -> list:
        """
        IDEA-E (2026-05-07): Carga los mejores parámetros de ventanas WFB anteriores
        del mismo agente para inicializar Optuna con conocimiento previo.

        Solo se usan signatures de ventanas ANTERIORES (causalmente seguro).
        El warm start no modifica el training data — solo el espacio de búsqueda de Optuna.
        Si no hay signatures previas, retorna lista vacía (Optuna explora desde cero).
        """
        if wfb_cache_dir is None or self.regime_name is None:
            return []
        wfb_cache_dir = Path(wfb_cache_dir)
        warmstart = []
        # Buscar signatures de W1..W5 para este agente
        for w_idx in range(1, 6):
            w_dir = wfb_cache_dir / f"W{w_idx}" / "models"
            sig_path = w_dir / f"xgboost_meta_{self.regime_name}_long_signature.json"
            if not sig_path.exists():
                continue
            try:
                import json as _json_ws
                sig = _json_ws.loads(sig_path.read_text(encoding="utf-8"))
                params = sig.get("params", {})
                if params:
                    warmstart.append(params)
                    logger.info(
                        "[IDEA-E] WarmStart: cargados params de W{}/{}: n_est={} lr={} depth={}",
                        w_idx, self.regime_name,
                        params.get("n_estimators"), params.get("learning_rate"),
                        params.get("max_depth")
                    )
                    print(f"[BUG-FIX-LOG 2026-06-05] WarmStart: cargados params de W{w_idx}/{self.regime_name}: n_est={params.get('n_estimators')} lr={params.get('learning_rate')} depth={params.get('max_depth')}")
            except Exception as _e_ws:
                logger.debug("[IDEA-E] WarmStart: no se pudo leer {}: {}", sig_path.name, _e_ws)
                print(f"[BUG-FIX-LOG 2026-06-05] WarmStart: no se pudo leer {sig_path.name}: {_e_ws}")
        return warmstart


    @staticmethod
    def _compute_optimal_cpcv_groups(n_samples: int, embargo_h: int) -> int:
        """
        IDEA-B (2026-05-07): Calcula el número óptimo de grupos CPCV según el tamaño
        del dataset del régimen, garantizando suficiente cobertura estadística.

        Para el agente BEAR (5k muestras) CPCV_GROUPS=10 genera 45 paths pero
        la mayoría se descartan por el filtro de purge → splits efectivos ≈ 3.
        Con CPCV_GROUPS=5 (C(5,2)=10 paths) todos los splits son viables.

        Criterio: test_size ≥ 50 muestras && train_size ≥ 10% del dataset.
        """
        min_test = 50
        min_train_pct = 0.10
        embargo_factor = max(1, embargo_h // 24)  # dias de purge ≈ puntos descartados

        for g in range(15, 4, -1):
            test_size_approx  = (n_samples // g) * 2  # k=2 grupos de test
            # Restar buffer de purge x2 extremos del train
            train_size_approx = n_samples - test_size_approx - embargo_factor * 4
            if (test_size_approx >= min_test and
                    train_size_approx >= n_samples * min_train_pct):
                return g
        return 5  # fallback mínimo absoluto

    def tune_hyperparameters(self):
        logger.info(f"Iniciando Optuna Tuning ({self.n_trials} trials)... OptimizaciÃ³n: Deflated Sharpe")

        if len(self.X) < 500:
            logger.warning(f"[{self.regime_name}] Dataset minúsculo ({len(self.X)} filas < 500). Omitiendo Optuna tuning y usando fallback ultra-robusto (MEDIO-7).")
            self.best_params = {
                'n_estimators': 100,
                'max_depth': 2,
                'learning_rate': 0.05,
                'subsample': 0.6,
                'colsample_bytree': 0.8,
                'min_child_weight': 20,
                'gamma': 1.0,
                'reg_alpha': 0.5,
                'reg_lambda': 1.5,
            }
            # Simulamos un objeto study para que train_final_model no falle en logging
            class DummyTrial:
                user_attrs = {
                    'mean_best_iter': 100,
                    'dsr_telemetry': 0.50,
                    'metric_is': 0.15,
                    'naive_is': 0.15
                }
            class DummyStudy:
                best_value = 0.50
                best_params = self.best_params
                best_trial = DummyTrial()
                direction = "maximize"
            self.study = DummyStudy()
            print(f"[FIX-DUMMY-STUDY] Instanciado DummyStudy con DummyTrial (n={len(self.X)}) para evitar AttributeError.")
            logger.info(f"[FIX-DUMMY-STUDY] Instanciado DummyStudy con DummyTrial (n={len(self.X)}) para evitar AttributeError.")
            return

        import time
        import psutil
        _t0 = time.time()
        _process = psutil.Process(_os.getpid())

        def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            """Log progreso de Optuna — cada trial en VERBOSE, cada 10 en modo normal."""
            n = trial.number + 1
            elapsed = time.time() - _t0
            trial_duration = trial.duration.total_seconds() if trial.duration else 0.0
            mem_mb = _process.memory_info().rss / 1024 / 1024
            
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
                    f"{elapsed/60:.1f}min | ETA≈{eta_str} | "
                    f"RAM={mem_mb:.0f}MB | t={trial_duration:.1f}s"
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
        
        # REPRO-01 fix (2026-03-17): sampler determinista para reproducibilidad.
        # [FIX-RANDOM-STATE-01 2026-05-28] Priorizar LUNA_SEED del entorno sobre cfg.optuna_seed.
        # Con optuna_seed fijo en settings todas las seeds WFB exploraban el mismo espacio
        # de hiperparámetros — sin diversidad en el ensemble.
        _luna_seed_env = _os.environ.get('LUNA_SEED', '')
        if _luna_seed_env.isdigit():
            _optuna_seed = int(_luna_seed_env)
            print(f"[FIX-RANDOM-STATE-01] TPESampler seed={_optuna_seed} (desde LUNA_SEED env)")  # RULE[fixbugsprints.md]
        else:
            try:
                from config.settings import cfg as _cfg_xgb
                _optuna_seed = int(_cfg_xgb.xgboost.optuna_seed)
                print(f"[FIX-RANDOM-STATE-01] TPESampler seed={_optuna_seed} (desde cfg)")  # RULE[fixbugsprints.md]
            except Exception:
                _optuna_seed = 42
                print(f"[FIX-RANDOM-STATE-01] TPESampler seed={_optuna_seed} (fallback)")  # RULE[fixbugsprints.md]
        _sampler = optuna.samplers.TPESampler(seed=_optuna_seed)
        # [V2-FIX-1] Leer métrica y dirección del study dinámicamente desde settings
        try:
            from config.settings import cfg as _cfg_tune_dir
            _optuna_metric_dir = str(str(_cfg_tune_dir.xgboost.optuna_metric)).lower()
        except Exception:
            _optuna_metric_dir = 'dsr'
        _study_direction = 'minimize' if _optuna_metric_dir in ('brier', 'logloss') else 'maximize'
        self.study = optuna.create_study(direction=_study_direction, pruner=_pruner, sampler=_sampler)
        logger.info(
            f"[REPRO-01] Optuna TPESampler(seed={_optuna_seed}) activo — runs deterministas. "
            f"[V2-FIX-1] Métrica={_optuna_metric_dir.upper()}, direction={_study_direction}"
        )

        # [P4-WARM-START] Encolar parámetros de W_t-1 (conocimiento previo)
        try:
            from config.settings import cfg as _cfg_ws
            _ws_enabled = bool(_cfg_ws.xgboost.warm_start_enabled)
        except Exception:
            _ws_enabled = False

        self._ws_enabled = False
        self._ws_count = 0

        if _ws_enabled:
            _wfb_cache_dir = _os.environ.get("LUNA_WFB_CACHE_DIR", "")
            if not _wfb_cache_dir:
                _seed = _os.environ.get("LUNA_SEED", "")
                if _seed:
                    _wfb_cache_dir = self.root / "data" / "wfb_cache" / f"seed{_seed}"
            
            if _wfb_cache_dir:
                _ws_params = self._load_warmstart_params(_wfb_cache_dir)
                if _ws_params:
                    # Encolamos los priors para que Optuna los pruebe primero y construya el TPE rápido
                    for p in _ws_params:
                        # Filtrar solo hiperparámetros que estén en el espacio de búsqueda para evitar ValueError
                        _clean_p = {k: v for k, v in p.items() if k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda", "scale_pos_weight", "focal_loss_gamma"]}
                        if _clean_p:
                            self.study.enqueue_trial(_clean_p)
                            self._ws_count += 1
                    self._ws_enabled = True
                    logger.success(f"[P4-WARM-START] {self._ws_count} prior trials encolados exitosamente desde {_wfb_cache_dir}")


        # SPW-AUTO-01 (2026-03-23): calcular scale_pos_weight range desde labels TBM reales.
        # PROBLEMA ANTERIOR: min/max leidos desde settings.yaml -> error humano (RUN-0010:
        # max=1.30 cortaba 57% del espacio de busqueda). Ahora auto-calculado:
        #   ideal = neg/pos  (ratio de clases real del training set)
        #   min   = ideal * 0.5   (permite downweight hasta la mitad)
        #   max   = min(2.0, ideal * 2.5)  SOP_LIMIT=2.0 (post-mortem M-79: SPW>2 colapsa WR)
        # El YAML (scale_pos_weight_min/max) ya no es fuente de verdad — solo documentacion.
        try:
            _spw_pos   = int((self.y == 1).sum())
            _spw_neg   = int((self.y == 0).sum())
            # [SPW-FIX 2026-06-16] Bloqueamos scale_pos_weight en 1.0 absoluto para preservar el edge de XGBoost
            # y evitar que Optuna destruya el ranking natural de las probabilidades penalizando la clase mayoritaria.
            self._spw_ideal = 1.0
            self._spw_min = 1.0
            self._spw_max = 1.0
            
            logger.info(
                f"[SPW-FIX] pos={_spw_pos} neg={_spw_neg} | SPW bloqueado a 1.0 absoluto para prevenir Asfixia de Edge"
            )
        except Exception as _e_spw:
            logger.warning("[SPW-AUTO-01] Calculo automatico fallido ({}) — usando YAML settings.", _e_spw)
            self._spw_min = None
            self._spw_max = None

        # MEJ-XGB-01 (2026-04-06): Análisis de seguridad de n_jobs:
        # _cached_splits es SOLO LECTURA en objective() — no hay escritura concurrente.
        # El GIL de Python NO protege workloads CPU-bound (numpy), pero en Optuna con
        # n_jobs>1 los procesos son independientes (spawn), no threads → sin race condition.
        # MedianPruner funciona correctamente en modo multi-process.
        # n_jobs=1: referencia segura conservadora (sin cambio de comportamiento).
        # n_jobs=2: reduciría ~50% el tiempo de tuning (600 trials → efectivo 300 en paralelo).
        # Para activar: cambiar n_jobs=2 en settings.yaml → optuna_n_jobs cuando se añada ese param.
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=1,   # safe: ver MEJ-XGB-01 analysis arriba para activar n_jobs=2
            callbacks=[_progress_callback],
        )

        
        self.best_params = self.study.best_params
        
        # [FAIL-FAST] Validar colapso matemático total (Asfixia del árbol)
        if hasattr(self.study, "best_value"):
            _best_val = float(self.study.best_value)
            _metric = _optuna_metric_dir
            if _metric == 'dsr' and _best_val <= 0.0000:
                logger.error(f"[FAIL-FAST] Optuna colapsado: Mejor DSR OOS = {_best_val:.4f}. Modelo estéril. Forzando Modo Degradado.")
                raise ModelDegradationError(f"Optuna colapsado (DSR={_best_val:.4f})")
            # En Brier, un score superior a 0.25 (random guessing en binario balanceado)
            # o igual al naive prior, indica que no aprendió. Pero para ser conservadores
            # con el Fail-Fast, abortamos si TODOS los trials fueron PRUNED.
            pruned_trials = [t for t in self.study.trials if t.state.name == "PRUNED"]
            if len(pruned_trials) == len(self.study.trials) and len(self.study.trials) > 0:
                logger.error("[FAIL-FAST] Optuna colapsado: TODOS los trials fueron podados. Asfixia por parámetros. Forzando Modo Degradado.")
                raise ModelDegradationError("Optuna colapsado (Todos podados)")

        logger.success(f"Tuning completado! Mejor {_optuna_metric_dir} OOS estimado: {self.study.best_value:.4f}")
        logger.info(f"Mejores params: {self.best_params}")
        
    def _calibrate_threshold(self) -> float:
        """
        MEJORA-R12-01 fix (2026-03-10): calibración automática del threshold XGB.

        Barre thresholds sobre features_validation.parquet y selecciona el que
        maximiza el Expected Value esperado por trade:
            EV(t) = P(win | prob > t) × avg_win - P(loss | prob > t) × avg_loss - cost
        sujeto a: n_trades(t) >= threshold_min_trades (settings.yaml).

        Fuente: features_validation.parquet (período semi-OOS, nunca en train).
        Fallback: 0.50 si validation no disponible o < min_trades en todo el sweep.

        Returns
        -------
        float — optimal_threshold para usar en generate_oos_predictions.py
        """
        # Parámetros de calibración desde settings.yaml — sin hardcodes

        # [H-RANGE-01-FIX 2026-05-30] Guard final_model=None: gate pre-fit abortó el entrenamiento.
        # predict_proba sobre None lanzaría AttributeError. Retornar threshold neutro.
        if self.final_model is None:
            print(  # RULE[fixbugsprints.md]
                f"[H-RANGE-01-FIX] _calibrate_threshold: final_model=None agente='{self.regime_name}'"
                f" → threshold=0.56 (base_rate+margen). 0 trades esperados."
            )
            logger.warning(
                "[H-RANGE-01-FIX] _calibrate_threshold: final_model=None agente=%s → skip.",
                self.regime_name
            )
            return 0.56

        try:
            cal_cfg = _cfg_xgb.xgboost
            t_min      = float(float(cal_cfg.threshold_sweep_min))
            t_max      = float(float(cal_cfg.threshold_sweep_max))
            t_step     = float(float(cal_cfg.threshold_sweep_step))
            min_trades = int(int(cal_cfg.threshold_min_trades))
            # OPT-B (2026-03-22): densidad mínima de señales respecto a t_min.
            # [TIPO-3: CALCULADO] n_baseline = señales@t_min. Si n(t) < n_baseline*density_pct,
            # el threshold se considera hiperselectivo y se descarta del sweep.
            # M-80: 0.75 tenía densidad=12.2% — con 30% mínimo habría elegido ~0.61 (EV>0, N>600).
            min_density_pct = float(float(cal_cfg.threshold_min_density_pct))
        except Exception:
            t_min, t_max, t_step, min_trades, min_density_pct = 0.40, 0.75, 0.01, 30, 0.30
            print(f"[FIX-C] WARN: No se pudo leer calibracion de cfg. Usando fallbacks: min_density_pct={0.30}, t_min={0.40}, t_max={0.75}")  # debug

        # [FIX-130] [ARCH-04]: calibración threshold con jerarquía holdout-first
        # Si holdout_calib_months > 0 y features_holdout.parquet existe, cargamos ese archivo
        # y filtramos los primeros 'holdout_calib_months' meses del set de holdout para calibrar.
        # Caso contrario, caemos a features_validation.parquet.
        try:
            holdout_calib_months = int(_cfg_xgb.xgboost.holdout_calib_months)
        except Exception:
            holdout_calib_months = 0

        df_val = None
        cal_source = "validation"
        
        holdout_path = self.root / "data" / "features" / "features_holdout.parquet"
        val_path = self.root / "data" / "features" / "features_validation.parquet"
        
        if holdout_calib_months > 0 and holdout_path.exists():
            try:
                df_holdout_full = pd.read_parquet(holdout_path).dropna(subset=["close"]).copy()
                if not df_holdout_full.empty:
                    df_holdout_full = df_holdout_full.sort_index()
                    # Si el índice no es DatetimeIndex, asegurar conversión o usar columna
                    if not isinstance(df_holdout_full.index, pd.DatetimeIndex):
                        if "timestamp" in df_holdout_full.columns:
                            df_holdout_full = df_holdout_full.set_index("timestamp")
                        df_holdout_full.index = pd.to_datetime(df_holdout_full.index, utc=True)
                        df_holdout_full = df_holdout_full.sort_index()
                    
                    first_date = df_holdout_full.index.min()
                    limit_date = first_date + pd.DateOffset(months=holdout_calib_months)
                    df_val = df_holdout_full[df_holdout_full.index <= limit_date].copy()
                    cal_source = f"holdout_{holdout_calib_months}m"
                    print(f"[FIX-130] [ARCH-04] Jerarquía Holdout-First activa. Cargando {cal_source} de {holdout_path} ({len(df_val)} filas).")  # debug
                    logger.info(f"[Calibrate] Jerarquía holdout-first activa: cargados {holdout_calib_months} meses desde holdout para calibración.")
            except Exception as e_holdout:
                print(f"[FIX-130] [ARCH-04] Error leyendo features_holdout.parquet, cayendo a validation: {e_holdout}")  # debug
                logger.warning(f"[Calibrate] Fallo al leer holdout, cayendo a validation: {e_holdout}")
        
        if df_val is None:
            # Fallback a validation.parquet
            if not val_path.exists():
                logger.warning("[Calibrate] features_validation.parquet no existe -- threshold=0.50 (neutral)")
                self._cal_source = "neutral_050"
                return 0.50
            try:
                df_val = pd.read_parquet(val_path).dropna(subset=["close"]).copy()
                cal_source = "validation"
                print(f"[FIX-130] [ARCH-04] Jerarquía Holdout-First cayendo a features_validation.parquet ({len(df_val)} filas).")  # debug
                logger.info(f"[Calibrate] Cargado features_validation.parquet con {len(df_val)} filas para calibración.")
            except Exception as e:
                logger.warning("[Calibrate] No se pudo leer features_validation.parquet: {} -- threshold=0.50", e)
                self._cal_source = "neutral_050"
                return 0.50

        self._cal_source = cal_source   # persistir para signature JSON

        # ── LAB-CAL-01 fix (2026-03-20): filtrar df_val por regimenes HMM permitidos.
        _hmm_filtered = False
        try:
            from luna.models.hmm_regime import HMMRegimeModel
            import os
            # [Fix P2] Detectar si estamos en WFB para usar la ruta de caché dinámica
            _wfb_window = os.environ.get("LUNA_WINDOW_ID", "")
            _wfb_seed = os.environ.get("LUNA_SEED", "")
            
            if _wfb_window and _wfb_window != "PROD" and _wfb_seed:
                _hmm_model_dir = self.root / "data" / "wfb_cache" / f"seed{_wfb_seed}" / _wfb_window / "models"
            else:
                _hmm_model_dir = self.root / "data" / "models"
                if _wfb_window == "PROD":
                    print(f"[BUGFIX-PROD-HMM-CALIB 2026-06-20] HMM model directory resolved to {_hmm_model_dir} for production calibration (LUNA_WINDOW_ID={_wfb_window}, LUNA_SEED={_wfb_seed}).")
                    logger.info(f"[BUGFIX-PROD-HMM-CALIB 2026-06-20] HMM model directory resolved to {_hmm_model_dir} for production calibration.")
                
            _hmm_pkl_path = _hmm_model_dir / "hmm_regime.pkl"
            
            if _hmm_pkl_path.exists():
                _hmm_predictor = HMMRegimeModel.load(_hmm_model_dir)
                _hmm_df = _hmm_predictor.predict_regime_series(df_val)
                df_val["HMM_Semantic"] = _hmm_df["HMM_Semantic"]
                df_val["HMM_Regime"] = _hmm_df["HMM_Regime"]
                
                # [Fix P2] Filtrar df_val *exclusivamente* por los regímenes de este agente
                _n_before = len(df_val)
                if self.regime_list:
                    _mask_hmm = df_val["HMM_Semantic"].isin(self.regime_list)
                    df_val = df_val[_mask_hmm].copy()
                _n_after = len(df_val)
                
                _hmm_filtered = True
                logger.info(
                    f"[LAB-CAL-01] HMM filter (intra-agent) aplicado para {self.regime_name}: "
                    f"{_n_before} -> {_n_after} filas"
                )
                
                # [Fix P2] Protección anti-Dead-Cat Bounce. Si tras filtrar quedan muy pocas
                # velas, calibrar el threshold en ese bloque sobreajustará el EV.
                if _n_after < 300:
                    # ── [FIX-CAL-DYN-01] Umbral de Fallback Dinámico In-Sample ──
                    # Calcular el Base Rate In-Sample para que el fallback respete
                    # la densidad natural del régimen (P(y=1)), más un margen conservador.
                    _base_rate_is = float(self.y.mean()) if len(self.y) > 0 else 0.45
                    _dyn_fallback = min(0.65, max(0.50, round(_base_rate_is + 0.08, 2)))
                    logger.warning(
                        f"[LAB-CAL-01] Tras HMM filter quedan {_n_after} filas (<300). Evitando overfitting a ruido. "
                        f"Threshold dinámico fijado en {_dyn_fallback:.2f} (BaseRate={_base_rate_is:.2f} + margen)."
                    )
                    self._cal_source = f"fallback_dynamic_{_dyn_fallback:.2f}"
                    return _dyn_fallback
        except Exception as _e_hmm_cal:
            logger.warning("[LAB-CAL-01] HMM filter fallido, usando df_val sin filtrar: {}", _e_hmm_cal)

        # Score de calibracion: EV * penalizacion_volumen
        # LAB-CAL-01: penalizar thresholds con n_trades < N_target.
        # EV puro (sin penalizacion) elige thresholds muy restrictivos (M-52: 0.63, 58 trades).
        # Con score compuesto, el calibrador equilibra senial y volumen.
        # min_trades del Gauntlet (100) es el N_target optimo.
        try:
            _n_target = int(_cfg_xgb.stat.min_trades)
        except Exception as e:
            raise RuntimeError(f"Falta stat.min_trades en settings.yaml. Política No-Fallback: {e}") from e

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
            # [2026-05-08 FORENSIC-FIX] Leer pt/sl del perfil de régimen específico,
            # igual que en load_dataset() → TBM-REGIME-01. Bug anterior: usaba el
            # valor global (sl=1.2x) aunque el agente bull fue entrenado con sl=1.5x.
            _pt_c = float(float(cal_cfg.pt_mult_min))
            _sl_c = float(float(cal_cfg.sl_mult_min))
            try:
                _regime_profiles = getattr(cal_cfg, "regime_tbm_profiles", None)
                if _regime_profiles is not None and self.regime_name is not None:
                    _rkey = str(self.regime_name).lower()
                    for _pk in vars(_regime_profiles).keys():
                        if _rkey == str(_pk).lower() or _rkey.startswith(str(_pk).lower()):
                            _prof = getattr(_regime_profiles, _pk)
                            _pt_c = float(getattr(_prof, 'pt_mult_min', _pt_c))
                            _sl_c = float(getattr(_prof, 'sl_mult_min', _sl_c))
                            logger.debug(
                                "[FORENSIC-FIX] Calibrator TBM regime '{}' → pt={:.2f}x sl={:.2f}x",
                                _pk, _pt_c, _sl_c
                            )
                            print(f"[BUG-FIX-LOG 2026-06-05] Calibrator TBM regime '{_pk}' -> pt={_pt_c:.2f}x sl={_sl_c:.2f}x")
                            break
            except Exception as _e_regime_cal:
                logger.debug("[FORENSIC-FIX] No se pudo leer regime_tbm_profiles en calibrador: {}", _e_regime_cal)
                print(f"[BUG-FIX-LOG 2026-06-05] No se pudo leer regime_tbm_profiles en calibrador: {_e_regime_cal}")
            _vbh_c = int(int(cal_cfg.vertical_barrier_hours))
            # BUG-XGB-02 FIX (2026-04-06): leer tbm_min_return desde settings en lugar de
            # hardcode 0.005. En training se usa 0.003 (tbm_min_return de settings.yaml).
            # La discrepancia causaba que la calibración descartase ~40% más de eventos
            # y por tanto eligiese umbrales sesgados hacia señales de alta volatilidad.
            _tbm_min_c = float(float(cal_cfg.tbm_min_return))
            _lin_decay_c = bool(bool(cal_cfg.linear_decay_pt))
            _pt_decay_frac_c = float(int(cal_cfg.pt_decay_fraction))
            logger.debug(
                "[BUG-XGB-02] Calibración TBM: min_return={:.4f} (de settings, consistente con training).",
                _tbm_min_c
            )
            print(f"[BUG-FIX-LOG 2026-06-05] Calibración TBM: min_return={_tbm_min_c:.4f}")

            _tbm_val = _atb(
                price_series=df_val["close"],
                event_times=df_val.index,
                pt_sl_multiplier=[_pt_c, _sl_c],
                min_return=_tbm_min_c,
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

        # FIX-SWEEP-ADAPTIVE-01 (2026-03-29): Auto-ajuste del rango de sweep desde la
        # distribucion real de probabilidades del modelo. El rango hardcodeado en settings.yaml
        # puede no coincidir con la distribucion del modelo re-entrenado:
        #   - Si t_min > P95(probs) → el sweep evalua 0 thresholds validos → fallback.
        #   - Si t_max < P5(probs)  → el sweep evalua 0 thresholds validos → fallback.
        # Solucion: ajustar t_min/t_max dinamicamente desde los percentiles reales.
        # Los settings actuan como COTA INFERIOR de t_min (nunca bajar de 0.50)
        # y COTA SUPERIOR de t_max (nunca subir de 0.99). El rango real es:
        #   t_min_eff = max(t_min_cfg, percentil(probs, 5))   [al menos P5 del modelo]
        #   t_max_eff = min(t_max_cfg, percentil(probs, 99) + 0.03)  [hasta P99+margen]
        try:
            _p5_probs  = float(np.percentile(probs_aligned, 5))
            _p99_probs = float(np.percentile(probs_aligned, 99))
            _p50_probs = float(np.percentile(probs_aligned, 50))

            # t_min efectivo: max(config_min, P5_modelo) pero nunca < 0.50 ni > P50
            t_min_eff = max(t_min, _p5_probs)
            t_min_eff = min(t_min_eff, _p50_probs, 0.95)  # no puede superar la mediana

            # t_max efectivo: P99 + 3pp de margen, sin exceder 0.99 ni el config_max
            t_max_eff = min(t_max, _p99_probs + 0.03, 0.99)
            t_max_eff = max(t_max_eff, t_min_eff + t_step * 5)  # garantizar al menos 5 pasos

            if abs(t_min_eff - t_min) > 0.001 or abs(t_max_eff - t_max) > 0.001:
                logger.info(
                    "[FIX-SWEEP-ADAPTIVE-01] Rango adaptado desde distribucion real del modelo: "
                    "[{:.3f}, {:.3f}] cfg -> [{:.3f}, {:.3f}] efectivo "
                    "(P5={:.3f}, P50={:.3f}, P99={:.3f})",
                    t_min, t_max, t_min_eff, t_max_eff,
                    _p5_probs, _p50_probs, _p99_probs
                )
                print(f"[BUG-FIX-LOG 2026-06-05] Rango adaptado: [{t_min:.3f}, {t_max:.3f}] cfg -> [{t_min_eff:.3f}, {t_max_eff:.3f}] efectivo (P5={_p5_probs:.3f}, P50={_p50_probs:.3f}, P99={_p99_probs:.3f})")
            else:
                logger.debug(
                    "[FIX-SWEEP-ADAPTIVE-01] Rango cfg coincide con distribucion (P5={:.3f} P99={:.3f}) — sin ajuste.",
                    _p5_probs, _p99_probs
                )
                print(f"[BUG-FIX-LOG 2026-06-05] Rango cfg coincide con distribucion (P5={_p5_probs:.3f} P99={_p99_probs:.3f}) — sin ajuste.")

            thresholds = np.arange(t_min_eff, t_max_eff + t_step / 2, t_step)

        except Exception as _e_adapt:
            logger.warning("[FIX-SWEEP-ADAPTIVE-01] Error en adaptacion de rango: {} — usando rango cfg.", _e_adapt)
            # thresholds ya definido arriba con el rango de cfg, se queda como esta

        
        try:
            _max_density_pct = float(_cfg_xgb.xgboost.max_signal_density_pct)
            fallback_t = float(_cfg_xgb.xgboost.xgb_signal_threshold)
        except Exception:
            _max_density_pct = 0.60
            fallback_t = 0.40
            print(f"[FIX-C] WARN: No se pudo leer max_signal_density_pct de cfg. Usando fallback={0.60}")  # debug
            
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
                
                # [V2-MATH-FIX] Un umbral sin ganancias (len(wins)==0) se salta, pero no uno sin perdidas!
                if len(wins) == 0: continue
                
                p_win = len(wins) / n
                avg_win = wins.mean()
                avg_los = abs(loses.mean()) if len(loses) > 0 else 0.0
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
                
                # [FIX-B] ev_tolerance_pct leído de cfg (antes: +0.010 hardcodeado)
                # Justificación: XGBoost es Weak Learner; MetaLabeler aporta EV downstream.
                # Permitir EV negativo hasta -ev_tolerance en el weak learner aislado.
                try:
                    from config.settings import cfg as _cfg_ev
                    _ev_tol = float(_cfg_ev.xgboost.ev_tolerance_pct)
                    print(f"[BUG-FIX-LOG 2026-06-05] ev_tolerance_pct cargada exitosamente: {_ev_tol:.4f}")
                except Exception as e_ev:
                    raise RuntimeError(f"Falta ev_tolerance_pct en settings.yaml (SOP No-Fallback): {e_ev}")
                ev_adjusted = ev + _ev_tol
                print(f"[FIX-B] EV calibrador: ev={ev:.4f}, ev_tol={_ev_tol:.4f}, ev_adjusted={ev_adjusted:.4f} (threshold={t:.3f})")  # debug
                
                # REGLA DE HIERRO RELAJADA: Permitir hasta -ev_tolerance_pct de pérdida en el weak learner

                if ev_adjusted <= 0:
                    continue
                    
                score = ev_adjusted * _vol_factor
                    
                # [V2-MATH-FIX] Solo requerimos superar el mejor score (que incluye la penalizacion por volumen).
                if score > b_score:
                    b_ev = ev
                    b_score = score
                    b_thresh = float(t)
            
            return b_thresh, b_ev, c_log

        # 1. Calibración Global
        best_threshold, best_ev, calibration_log = run_sweep(probs_aligned, fwd_ret, min_trades, n_baseline)

        if best_ev == -np.inf:
            logger.warning(
                f"[Calibrate] Sweep global OOS sin resultado valido (min_trades={min_trades}). "
                f"Asignando fallback cfg {fallback_t:.2f} (In-sample fallback deshabilitado por riesgo de overfitting)."
            )
            best_threshold = fallback_t
        else:
            logger.success(
                "[Calibrate] Threshold Global óptimo={:.2f} | EV={:.5f} | wr={:.1f}% | "
                "{} combinaciones evaluadas",
                best_threshold, best_ev,
                next((r["wr"] for r in calibration_log if abs(r["threshold"] - best_threshold) < 1e-6), 0) * 100,
                len(calibration_log)
            )
            print(f"[BUG-FIX-LOG 2026-06-05] [Calibrate] Threshold Global óptimo={best_threshold:.2f} | EV={best_ev:.5f} | wr={next((r['wr'] for r in calibration_log if abs(r['threshold'] - best_threshold) < 1e-6), 0) * 100:.1f}% | {len(calibration_log)} combinaciones evaluadas")

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
                # [FIX-F] r_min_trades lee threshold_min_trades de cfg (antes: max(10, int(min_trades*0.25)) magic)
                # El 25% del min_trades global es el mínimo para un régimen individual — conservador pero no arbitrario.
                try:
                    from config.settings import cfg as _cfg_rmt
                    _min_t_base = int(_cfg_rmt.xgboost.threshold_min_trades)
                except Exception:
                    _min_t_base = min_trades  # fallback al ya leído
                r_min_trades = max(5, int(_min_t_base * 0.25))
                print(f"[FIX-F] Régimen {r}: r_min_trades={r_min_trades} (25% de threshold_min_trades={_min_t_base})")  # debug
                
                if r_mask.sum() < r_min_trades * 2:
                    logger.debug(f"  [Regimen {r}] Ignorado por tamaño muestral insuficiente (n={r_mask.sum()})")
                    continue
                
                r_thresh, r_ev, r_log = run_sweep(probs_aligned[r_mask], fwd_ret[r_mask], r_min_trades, r_baseline)
                if r_ev > -np.inf:
                    self._threshold_per_regime[str(r)] = r_thresh
                    logger.info(f"  [Regimen {r}] Threshold={r_thresh:.2f} (EV={r_ev:.4f}) calibrado sobre n={r_mask.sum()} seales")
                else:
                    logger.debug(f"  [Regimen {r}] Ningún threshold pasó los filtros mínimos (min_trades={r_min_trades}). Fallback a global.")

        self._calibration_report = calibration_log
        return best_threshold

    def train_final_model(self):
        logger.info("Entrenando Modelo Final con todo el Dataset (usando Best Params)...")
        best_params = self.best_params.copy()
        # [FIX-RANDOM-STATE-01 2026-05-28] random_state usa LUNA_SEED igual que en objective()
        _final_rs = int(_os.environ.get('LUNA_SEED', 42))
        best_params.update({'objective': 'binary:logistic', 'tree_method': 'hist', 'n_jobs': -1, 'random_state': _final_rs})
        print(f"[FIX-RANDOM-STATE-01] train_final_model: random_state={_final_rs} (LUNA_SEED={_os.environ.get('LUNA_SEED', 'no-set')})")  # RULE[fixbugsprints.md]

        # [H-RANGE-01-FIX 2026-05-30] Gate pre-fit: abortar entrenamiento si n_train < min_viable_train_samples.
        # Con n=114 filas, depth=2, min_child_weight=20: 114/4=28 < 20 -> solo raiz = prob constante.
        # El MODEL COLLAPSE es deterministico con estas condiciones. Abortar antes evita:
        #   - Instanciar XGBClassifier (no hay beneficio con n < umbral)
        #   - Collapse detection post-fit (redundante y costoso)
        #   - Calibracion isotonica que produce warning 1441->1 filas
        # El pipeline downstream maneja self.final_model=None como 0 trades (correcto).
        try:
            from config.settings import cfg as _cfg_gate
            _min_viable = int(_cfg_gate.xgboost.min_viable_train_samples)
        except Exception as e_minviable:
            raise RuntimeError(f"Falta min_viable_train_samples en settings.yaml (SOP No-Fallback): {e_minviable}")
        _n_train_gate = len(self.X)
        if _n_train_gate < _min_viable:
            print(  # RULE[fixbugsprints.md]
                f"[H-RANGE-01-FIX] GATE PRE-FIT: agente='{self.regime_name}' n_train={_n_train_gate} < "
                f"min_viable={_min_viable} → entrenamiento abortado (MODEL COLLAPSE deterministico). "
                f"Señal OOS = base_rate → 0 trades (correcto)."
            )
            logger.warning(
                "[H-RANGE-01-FIX] GATE PRE-FIT abortado: agente={} n_train={} < min_viable={} "
                "→ MODEL COLLAPSE deterministico. Retornando modelo nulo.",
                self.regime_name, _n_train_gate, _min_viable
            )
            print(f"[BUG-FIX-LOG 2026-06-05] [H-RANGE-01-FIX] GATE PRE-FIT abortado: agente={self.regime_name} n_train={_n_train_gate} < min_viable={_min_viable} → MODEL COLLAPSE deterministico.")
            # Configurar modelo nulo para que el pipeline downstream no falle
            self.final_model = None
            self.threshold = 0.56  # base_rate + margen estandar
            self.dsr_cpcv_best = 0.5
            return


        _best_iter = 200
        if hasattr(self, 'study') and self.study is not None:
            if hasattr(self.study, 'best_trial') and self.study.best_trial is not None:
                _best_iter = self.study.best_trial.user_attrs.get('mean_best_iter', 200)
                print(f"[FIX-DUMMY-STUDY] n_estimators óptimo recuperado de study.best_trial: {_best_iter}")  # debug
                logger.info(f"[FIX-DUMMY-STUDY] n_estimators óptimo recuperado de study.best_trial: {_best_iter}")
            else:
                print("[FIX-DUMMY-STUDY] No se detectó best_trial en self.study. Usando fallback de n_estimators = 200")  # debug
                logger.info("[FIX-DUMMY-STUDY] No se detectó best_trial en self.study. Usando fallback de n_estimators = 200")
        else:
            print("[FIX-DUMMY-STUDY] No se detectó self.study. Usando fallback de n_estimators = 200")  # debug
            logger.info("[FIX-DUMMY-STUDY] No se detectó self.study. Usando fallback de n_estimators = 200")
        # [FIX-XGB-NEST-FLOOR] Aplicar floor institucional en n_estimators para prevenir
        # el modelo nulo: cuando Optuna elige hiperparámetros con alta regularización
        # (gamma alto, min_child_weight alto), el primer árbol no puede mejorar y el
        # early_stopping para en best_iteration=1. Con 1 árbol, el modelo predice la tasa
        # base constante (~0.511) en todas las filas, destruyendo la discriminación.
        # El floor de 100 garantiza un mínimo institucional de complejidad.
        try:
            from config.settings import cfg as _cfg_nest
            _nest_floor = int(_cfg_nest.xgboost.n_estimators_min_floor)
        except Exception:
            _nest_floor = 100
        _best_iter_raw = int(_best_iter)
        _best_iter_floored = max(_nest_floor, _best_iter_raw)
        if _best_iter_floored > _best_iter_raw:
            logger.warning(
                f"[FIX-XGB-NEST-FLOOR] n_estimators={_best_iter_raw} (early stopping prematuro detectado) "
                f"→ elevado al floor institucional de {_best_iter_floored}. "
                f"Causa probable: gamma/min_child_weight demasiado altos en el trial Optuna elegido."
            )
        best_params['n_estimators'] = _best_iter_floored
        logger.info(f"[Early Stopping] n_estimators óptimo calculado = {_best_iter_raw} → final = {_best_iter_floored}")

        self.final_model = xgb.XGBClassifier(**best_params)

        # [A1] Inyectar Focal Loss o Monetary Loss en modelo final si está configurado
        use_focal_loss = False
        use_monetary_loss = False
        try:
            from config.settings import cfg as _cfg_opts
            use_focal_loss = bool(_cfg_opts.xgboost.use_focal_loss)
            use_monetary_loss = bool(_cfg_opts.fase2.use_monetary_loss)
        except Exception:
            pass

        # ── [LOSS-TRACE-01] Traza del objetivo de entrenamiento ───────────────
        # Permite verificar en los logs exactamente qué loss function se usa
        # en cada etapa: TRAIN vs INFERENCE. Un mismatch causa Brier > 0.25.
        # ─────────────────────────────────────────────────────────────────────
        if use_monetary_loss:
            from luna.losses.monetary_loss import get_monetary_pnl_loss
            self.final_model.set_params(objective=get_monetary_pnl_loss())
            logger.info("[Fase 2] Entrenando final_model con Monetary PnL Loss Custom")
            logger.info(
                "[LOSS-TRACE-01] TRAIN objective: MonetaryPnLLoss (custom callable) | "
                "INFERENCE: binary:logistic (restauracion post-fit) | "
                "MISMATCH: SI -> probabilidades NO calibradas"
            )
            use_focal_loss = False
        elif use_focal_loss:
            _spw = best_params.get('scale_pos_weight', 1.0)
            _gamma_opt = best_params.pop('focal_loss_gamma', int(_cfg_opts.xgboost.focal_loss_gamma))
            self.final_model.set_params(objective=self._get_focal_loss_obj(scale_pos_weight=_spw, gamma=_gamma_opt))
            logger.info("[A1] Entrenando final_model con Focal Loss Custom (gamma={})", _gamma_opt)
            logger.warning(
                "[LOSS-TRACE-01] TRAIN objective: FocalLoss(gamma={:.1f}, spw={:.2f}) [CUSTOM CALLABLE] | "
                "INFERENCE: binary:logistic (restauracion post-fit) | "
                "MISMATCH: SI -> raw_margins entrenados con FL, sigmoid aplicado con LL -> "
                "probabilidades descalibradas (Brier > 0.25 esperado)",
                _gamma_opt, _spw
            )
        else:
            logger.info(
                "[LOSS-TRACE-01] TRAIN objective: binary:logistic (nativo XGB) | "
                "INFERENCE: binary:logistic | "
                "MISMATCH: NO -> sigmoid(raw_margin) produce probabilidad calibrada -> Brier < 0.25 esperado"
            )

        # ARCH-02: decaimiento exponencial configurable — ver _compute_sample_weights
        sw_full = self._compute_sample_weights(self.X.index)

        # ══════════════════════════════════════════════════════════════════════
        # [GUARDIAN-09] Target Leakage Guardian (Look-Ahead Bias)
        # ══════════════════════════════════════════════════════════════════════
        try:
            from config.settings import cfg as _cfg_g
            _corr_leak_thresh = float(_cfg_g.debug.corr_leakage_threshold)
        except Exception as e:
            raise RuntimeError(f"[CRITICAL-SOP] Falta debug.corr_leakage_threshold en settings.yaml: {e}")
            
        try:
            if not self.X.empty and not self.y.empty:
                _corrs = self.X.corrwith(self.y).abs()
                _max_corr = _corrs.max()
                if pd.notna(_max_corr) and _max_corr > _corr_leak_thresh:
                    _leaky_feat = _corrs.idxmax()
                    logger.error(
                        f"[GUARDIAN-09] Target Leakage DETECTADO: La feature '{_leaky_feat}' "
                        f"tiene una correlación de {_max_corr:.4f} (> {_corr_leak_thresh:.4f}) con el Target. "
                        f"Esto es un flagrante Look-Ahead Bias. Abortando."
                    )
                    print(f"[GUARDIAN-09] FATAL: Leakage en '{_leaky_feat}' (corr={_max_corr:.4f}). Forzando Modo Degradado.")
                    raise ModelDegradationError(f"Target Leakage ({_leaky_feat})")
        except SystemExit:
            raise
        except Exception as _e_g9:
            logger.warning(f"[GUARDIAN-09] Fallo al verificar Target Leakage: {_e_g9}")

        # ══════════════════════════════════════════════════════════════════════
        # [GUARDIAN-06] Target Imbalance (TBM) Guardian
        # ══════════════════════════════════════════════════════════════════════
        try:
            from config.settings import cfg as _cfg_g
            _tbm_min_class_pct = float(_cfg_g.xgboost.guardian_tbm_min_class_pct)
            _hmm_min_mi = float(_cfg_g.xgboost.guardian_hmm_min_mi)
        except Exception as e:
            raise RuntimeError(f"[CRITICAL-SOP] Faltan params de G-06/G-08 en settings.yaml: {e}")
            
        try:
            _y_mean = self.y.mean()
            _minority_class_pct = min(_y_mean, 1 - _y_mean)
            if _minority_class_pct < _tbm_min_class_pct:
                logger.warning(
                    f"[GUARDIAN-06] Target Imbalance DETECTADO: Clase minoritaria es {_minority_class_pct:.2%} "
                    f"(< {_tbm_min_class_pct:.2%}). Las barreras dinámicas (TBM) fallaron. "
                    f"Agente={self.regime_name}. Estableciendo modelo nulo y omitiendo fit."
                )
                print(f"[GUARDIAN-06] WARNING: TBM Degeneration. Clase minoritaria {_minority_class_pct:.2%} < {_tbm_min_class_pct:.2%}. Omitiendo agente.")
                self.final_model = None
                return self
        except Exception as _e_g6:
            logger.warning(f"[GUARDIAN-06] Fallo al verificar TBM Imbalance: {_e_g6}")

        # ══════════════════════════════════════════════════════════════════════
        # [GUARDIAN-08] HMM Mutual Information Guardian (SOP R9)
        # NOTA: Comprobado globalmente en hmm_regime.py. El cálculo local por agente es erróneo y se ha desactivado.
        # ══════════════════════════════════════════════════════════════════════
        pass
        logger.info(
            "[LOSS-TRACE-01] Iniciando fit: n_train={} | n_features={} | n_estimators={} | "
            "scale_pos_weight={:.3f} | use_focal_loss={}",
            len(self.X), len(self.features),
            best_params.get('n_estimators', '?'),
            best_params.get('scale_pos_weight', 1.0),
            use_focal_loss
        )
        self.final_model.fit(self.X, self.y, sample_weight=sw_full)

        # [A1 FIX] Restaurar objective estándar para que predict_proba y calibración funcionen
        _obj_before_restore = self.final_model.get_params().get('objective')
        _had_custom_obj = callable(_obj_before_restore)
        if _had_custom_obj:
            self.final_model.set_params(objective='binary:logistic')
            self.final_model.get_booster().set_param({'objective': 'binary:logistic'})
            logger.warning(
                "[LOSS-TRACE-01] POST-FIT RESTAURACION: objective CUSTOM -> binary:logistic. "
                "Los arboles internos tienen raw_margins optimizados para el loss custom. "
                "predict_proba usara sigmoid(raw_margin) de binary:logistic -> MISMATCH CONFIRMADO. "
                "Esto causa Brier > Brier_naive. Fix: use_focal_loss=false en settings.yaml."
            )
        else:
            logger.info(
                "[LOSS-TRACE-01] POST-FIT: objective='{}' (nativo, sin restauracion necesaria). "
                "predict_proba produce probabilidades calibradas correctamente.",
                _obj_before_restore
            )
        try:
            from config.settings import cfg as _cfg_log
            _alpha_log = float(_cfg_log.xgboost.weight_decay_alpha)
        except Exception:
            _alpha_log = 0.5
        logger.info("[R20-B/ARCH-02] sample_weight activo: decaimiento exp(alpha={:.2f}) desde train_end", _alpha_log)


        # ── [MC-PFI-01] Monte Carlo Permutation Feature Importance (In-Sample) ──
        logger.info("[MC-PFI-01] Ejecutando Permutation Feature Importance (50 iteraciones MC) sobre In-Sample...")
        from sklearn.metrics import brier_score_loss
        try:
            _y_pred_base = self.final_model.predict_proba(self.X)[:, 1]
            _base_brier = brier_score_loss(self.y, _y_pred_base)
            
            _pfi_results = []
            for _f_idx, _f_name in enumerate(self.features):
                _brier_shuffled_list = []
                _X_shuffled = self.X.copy()
                for _mc_iter in range(50):
                    np.random.seed(42 + _mc_iter)
                    _X_shuffled.iloc[:, _f_idx] = np.random.permutation(_X_shuffled.iloc[:, _f_idx].values)
                    _y_pred_shuf = self.final_model.predict_proba(_X_shuffled)[:, 1]
                    _brier_shuffled_list.append(brier_score_loss(self.y, _y_pred_shuf))
                
                _mean_brier_shuf = np.mean(_brier_shuffled_list)
                # Si el Brier sube (peor), la feature era importante. Si el Brier baja o queda igual, era ruido.
                _importance = _mean_brier_shuf - _base_brier
                _pfi_results.append((_f_name, _importance))
            
            _pfi_results.sort(key=lambda x: x[1]) # Orden ascendente (menor a mayor importance)
            _noise_features = [f for f, imp in _pfi_results if imp <= 0]
            
            print(f"[MC-PFI-01] Evaluados {len(self.features)} features con MC-PFI.")
            if _noise_features:
                print(f"[ALERTA MC-PFI] Detectadas {len(_noise_features)} variables Shadow/Noise (Importance <= 0):")
                for _nf in _noise_features[:10]:
                    print(f"  - {_nf}")
                if len(_noise_features) > 10:
                    print(f"  ... y {len(_noise_features) - 10} más.")
                logger.warning(f"[MC-PFI-01] Shadow/Noise features detectados: {_noise_features}")
            else:
                print("[MC-PFI-01] Todas las variables mostraron aporte predictivo (Importance > 0).")
        except Exception as _pfi_err:
            logger.error(f"[MC-PFI-01] Error ejecutando Permutation Feature Importance: {_pfi_err}")
            print(f"[MC-PFI-01] Error MC-PFI: {_pfi_err}")


        # ─── Feature importance top-10 siempre visible ───———
        importances = pd.Series(self.final_model.feature_importances_, index=self.X.columns)
        top10 = importances.sort_values(ascending=False).head(10)
        logger.info("[XGB] Feature Importance TOP-10:")
        for feat, imp in top10.items():
            logger.info(f"  {feat}: {imp:.4f}")

        # ══════════════════════════════════════════════════════════════════════
        # [GUARDIAN-04] SHAP/Gini Monopolization (Agotamiento de Features)
        # Si una variable copa > X% de la importancia, el modelo es extremadamente
        # frágil y ciego al resto del Data Lake.
        # ══════════════════════════════════════════════════════════════════════
        try:
            from config.settings import cfg as _cfg_g
            _max_monopoly = float(_cfg_g.xgboost.guardian_shap_max_monopoly)
        except Exception as e:
            raise RuntimeError(f"[CRITICAL-SOP] Falta xgboost.guardian_shap_max_monopoly en settings.yaml: {e}")

        try:
            if len(importances) > 0:
                _max_imp = float(importances.max())
                _max_feat = importances.idxmax()
                if _max_imp > _max_monopoly:
                    logger.error(
                        f"[GUARDIAN-04] SHAP Monopolization DETECTADO: '{_max_feat}' tiene "
                        f"{_max_imp:.1%} de la importancia total (> {_max_monopoly:.1%}). Modelo frágil/ciego. Abortando."
                    )
                    print(f"[GUARDIAN-04] FATAL: Monopolio de feature '{_max_feat}' ({_max_imp:.1%}) > {_max_monopoly:.1%}. Forzando Modo Degradado.")
                    raise ModelDegradationError(f"SHAP Monopolization ({_max_feat})")
        except SystemExit:
            raise
        except Exception as _e_g4:
            logger.warning(f"[GUARDIAN-04] Fallo al verificar monopolización: {_e_g4}")

        # ══════════════════════════════════════════════════════════════════════
        # [FIX-COLLAPSE-DETECT-01] Diagnóstico de Model Collapse post-fit (IS).
        # Detecta std_raw(IS) < 0.01 → modelo colapsa a probabilidad constante.
        # Esto es el precursor del colapso que genera 0 señales en OOS.
        # Si el FIX-REGIME-POOL-01 está activo (modo universal), esto NO debería
        # ocurrir — si ocurre, indica un problema más profundo con los datos.
        # ══════════════════════════════════════════════════════════════════════
        try:
            _is_proba = self.final_model.predict_proba(self.X)[:, 1]
            _std_is = float(np.std(_is_proba))
            _mean_is = float(np.mean(_is_proba))
            _min_is  = float(_is_proba.min())
            _max_is  = float(_is_proba.max())
            _IS_COLLAPSE = _std_is < 0.01
            _universal_active = getattr(self, '_universal_mode', False)
            print(
                f"[FIX-COLLAPSE-DETECT-01] POST-FIT IS | agente={self.regime_name or 'global'} "
                f"dir={self.native_direction} | "
                f"std_IS={_std_is:.6f} mean={_mean_is:.4f} range=[{_min_is:.4f},{_max_is:.4f}] | "
                f"universal_mode={_universal_active} | "
                f"{'🚨 MODEL COLLAPSE DETECTADO — señales OOS serán 0' if _IS_COLLAPSE else '✅ señal IS detectada'}"
            )
            if _IS_COLLAPSE:
                logger.error(
                    "[FIX-COLLAPSE-DETECT-01] MODEL COLLAPSE en IS: agente={} dir={} "
                    "std_IS={:.6f} — el modelo produce prob constante={:.4f}. "
                    "Causa: regularización excesiva (gamma/min_child_weight muy alto) o "
                    "dataset IS sin variabilidad. "
                    "universal_mode={} | n_train={} | n_features={}. "
                    "ACCIÓN RECOMENDADA: aumentar n_estimators o reducir regularización.",
                    self.regime_name or "global", self.native_direction,
                    _std_is, _mean_is, _universal_active, len(self.X), len(self.features)
                )
                print(f"[BUG-FIX-LOG 2026-06-05] [FIX-COLLAPSE-DETECT-01] MODEL COLLAPSE en IS: agente={self.regime_name or 'global'} dir={self.native_direction} std_IS={_std_is:.6f} — el modelo produce prob constante={_mean_is:.4f}.")
            else:
                logger.info(
                    "[FIX-COLLAPSE-DETECT-01] OK: std_IS={:.4f} mean={:.4f} range=[{:.4f},{:.4f}] "
                    "agente={} dir={}",
                    _std_is, _mean_is, _min_is, _max_is,
                    self.regime_name or "global", self.native_direction
                )
                print(f"[BUG-FIX-LOG 2026-06-05] [FIX-COLLAPSE-DETECT-01] OK: std_IS={_std_is:.4f} mean={_mean_is:.4f} range=[{_min_is:.4f},{_max_is:.4f}]")
        except Exception as _e_cd01:
            logger.warning("[FIX-COLLAPSE-DETECT-01] Diagnóstico fallido (no bloqueante): {}", _e_cd01)
            print(f"[BUG-FIX-LOG 2026-06-05] Diagnóstico de collapse fallido: {_e_cd01}")
        # ══════════════════════════════════════════════════════════════════════

        # ——— Overfit check: train AUC vs DSR OOS ———
        try:
            train_proba = self.final_model.predict_proba(self.X)[:, 1]
            from sklearn.metrics import roc_auc_score
            train_auc = roc_auc_score(self.y, train_proba)
            oos_dsr   = self.study.best_value if self.study else float('nan')
            gap_flag = " ⚠️  SOBREAJUSTE" if train_auc > 0.80 and oos_dsr < 0.50 else ""
            logger.info(f"[XGB] Overfit check: train_AUC={train_auc:.4f} | best_DSR_OOS={oos_dsr:.4f}{gap_flag}")
            check_numeric_stability(train_proba, label="XGB.train_proba")
        except Exception as e:
            logger.warning(f"[XGB] Overfit check fallido: {e}")

        # Plot Feature Importances
        plt.figure(figsize=(10, 8))
        importances.sort_values(ascending=True).plot(kind='barh')
        plt.title("XGBoost Meta-Model Feature Importances (Gain)")
        out_path = self.root / "data" / "models" / "engine_xgb_importances.png"
        plt.tight_layout()
        plt.savefig(out_path)
        logger.info(f"Importancias exportadas a {out_path.name}")
        # IDEA-G: Calcular FI por tipo (gain/weight/cover) para la signature
        try:
            _booster_ig = self.final_model.get_booster()
            _fi_raw_gain   = _booster_ig.get_score(importance_type='gain')
            _fi_raw_weight = _booster_ig.get_score(importance_type='weight')
            _fi_raw_cover  = _booster_ig.get_score(importance_type='cover')
            self._fi_gain_top20   = dict(sorted(_fi_raw_gain.items(),   key=lambda x: -x[1])[:20])
            self._fi_weight_top20 = dict(sorted(_fi_raw_weight.items(), key=lambda x: -x[1])[:20])
            self._fi_cover_top20  = dict(sorted(_fi_raw_cover.items(),  key=lambda x: -x[1])[:20])
            if len(self._fold_importances) >= 3:
                _fi_sets_g = [{k for k, _ in sorted(d.items(), key=lambda x: -x[1])[:10]}
                              for d in self._fold_importances if d]
                self._stable_fi = sorted(set.intersection(*_fi_sets_g)) if _fi_sets_g else sorted(self._fi_gain_top20)
            else:
                self._stable_fi = sorted(self._fi_gain_top20)
            logger.info("[IDEA-G] FI calculadas: gain={} weight={} stable={}",
                        len(self._fi_gain_top20), len(self._fi_weight_top20), len(self._stable_fi))
            print(f"[BUG-FIX-LOG 2026-06-05] [IDEA-G] FI calculadas: gain={len(self._fi_gain_top20)} weight={len(self._fi_weight_top20)} stable={len(self._stable_fi)}")
        except Exception as _e_ig:
            logger.warning("[IDEA-G] FI no calculadas: {}", _e_ig)
            print(f"[BUG-FIX-LOG 2026-06-05] FI no calculadas: {_e_ig}")
            self._fi_gain_top20 = self._fi_weight_top20 = self._fi_cover_top20 = {}
            self._stable_fi = []
        # IDEA-A: calcular base_rate IS para brier_adaptive_gate
        self._base_rate_is = float(self.y.mean()) if self.y is not None and len(self.y) > 0 else 0.50
        if self._base_rate_is <= 0.01 or self._base_rate_is >= 0.99:
            _brier_adaptive_gate = None  # NO_OPERABLE
            logger.warning(
                "[IDEA-A][FIX-IDEA-A-01] base_rate_IS=%.3f -> agente degenerado (sin muestras del regimen). "
                "brier_adaptive_gate=None (NO_OPERABLE). No se calculara gate de calibracion.",
                self._base_rate_is
            )
        else:
            try:
                _naive_is_fold = self.study.best_trial.user_attrs.get("naive_is")
            except Exception:
                _naive_is_fold = None
            
            if _naive_is_fold is not None:
                _brier_naive_true = _naive_is_fold
                logger.info(f"[FIX-IDEA-A-01] Usando Naive Brier ({_brier_naive_true:.4f}) alineado a los folds de Optuna, "
                            f"en lugar del global ({self._base_rate_is * (1 - self._base_rate_is):.4f})")
            else:
                _brier_naive_true = self._base_rate_is * (1 - self._base_rate_is)

            # [FIX-G2-BRIER-MARGIN-01 2026-06-02] Margen adaptativo por regimen para Gate G2.
            # ANTES: 0.030 (range) y 0.025 (otros) hardcodeados — viola Politica No-Fallback.
            # AHORA: se leen de settings.yaml stat.brier_margin_range / stat.brier_margin_default.
            # Evidencia: 28 seeds nocturnas, Brier CALM_BEAR umbral a 1.0 sigma (0.025 insuficiente).
            # margin=0.035 cubre media+1.5sigma sin riesgo de overfitting (10pp bajo Brier_random=0.25).
            _regime_str_gate = str(self.regime_name).lower() if self.regime_name else ""
            try:
                from config.settings import cfg as _cfg_brier
                _stat_brier = getattr(_cfg_brier, 'stat', None)
                if _stat_brier is None:
                    raise KeyError("seccion 'stat' ausente en settings.yaml")
                if not hasattr(_stat_brier, 'brier_margin_range'):
                    raise KeyError("stat.brier_margin_range ausente en settings.yaml")
                if not hasattr(_stat_brier, 'brier_margin_default'):
                    raise KeyError("stat.brier_margin_default ausente en settings.yaml")
                _margin_range   = float(_stat_brier.brier_margin_range)
                _margin_default = float(_stat_brier.brier_margin_default)
            except (AttributeError, KeyError, TypeError) as _e_bm:
                _err_bm = (
                    f"[FIX-G2-BRIER-MARGIN-01] CRITICAL: No se pudo leer stat.brier_margin_range/"
                    f"stat.brier_margin_default de settings.yaml: {_e_bm}. "
                    "Aniadir ambos parametros a la seccion stat antes de continuar."
                )
                print(_err_bm)
                logger.critical(_err_bm)
                raise RuntimeError(_err_bm) from _e_bm
            _brier_margin = _margin_range if _regime_str_gate == 'range' else _margin_default
            _brier_adaptive_gate = round(_brier_naive_true + _brier_margin, 4)
            print(f"[FIX-G2-BRIER-MARGIN-01] Brier margin para '{_regime_str_gate or 'global'}': {_brier_margin:.4f} (range={_margin_range} default={_margin_default}) | Gate adaptativo: {_brier_adaptive_gate:.4f}")  # RULE[fixbugsprints.md]
            logger.info(f"[FIX-G2-BRIER-MARGIN-01] Brier margin agente='{_regime_str_gate or 'global'}': {_brier_margin:.4f} | Gate={_brier_adaptive_gate:.4f} (naive={_brier_naive_true:.4f})")
        self._brier_adaptive_gate = _brier_adaptive_gate
        # ══════════════════════════════════════════════════════════════════════
        # [CAL-DIAG-01] Diagnóstico de Calibración XGBoost post-fit
        # Regla: Brier aleatorio = base_rate*(1-base_rate) ≈ 0.25 para WR~50%.
        # Un XGB bien calibrado tiene Brier_val < Brier_naive (random baseline).
        # Si Brier_val > Brier_naive → el modelo predice PEOR que el azar → ALERTA.
        # Ejecuta SIEMPRE — se imprime en el log de cada ventana WFB.
        # ══════════════════════════════════════════════════════════════════════
        try:
            _val_path_cd = self.root / "data" / "features" / "features_validation.parquet"
            _cal_ok = False
            if _val_path_cd.exists():
                _df_cd = pd.read_parquet(_val_path_cd).dropna(subset=["close"]).copy()
                _feats_cd = [f for f in self.features if f in _df_cd.columns]
                _missing_cd = [f for f in self.features if f not in _df_cd.columns]
                for _mf in _missing_cd:
                    _df_cd[_mf] = 0.0

                # Generar target TBM en validation para medir calibración real
                from luna.features.tbm import apply_triple_barrier as _atb_cd
                try:
                    from config.settings import cfg as _cfg_cd
                    _pt_cd = float(_cfg_cd.xgboost.pt_mult_min)
                    _sl_cd = float(_cfg_cd.xgboost.sl_mult_min)
                    _vb_cd = int(_cfg_cd.xgboost.vertical_barrier_hours)
                    _mr_cd = float(_cfg_cd.xgboost.tbm_min_return)
                except Exception as e_cd:
                    raise RuntimeError(f"Faltan parametros TBM en settings.yaml (SOP No-Fallback): {e_cd}")

                _tbm_cd = _atb_cd(
                    price_series=_df_cd["close"],
                    event_times=_df_cd.index,
                    pt_sl_multiplier=[_pt_cd, _sl_cd],
                    min_return=_mr_cd,
                    vertical_barrier_hours=_vb_cd,
                )
                _df_cd = _df_cd.join(_tbm_cd[["bin"]], how="inner").dropna(subset=["bin"])
                if len(_df_cd) >= 50:
                    _y_cd = (_df_cd["bin"] == 1).astype(float).values
                    _p_cd = self.final_model.predict_proba(_df_cd[self.features].fillna(0))[:, 1]

                    _brier_val   = float(np.mean((_p_cd - _y_cd) ** 2))
                    _brier_naive = float(np.mean(_y_cd) * (1 - float(np.mean(_y_cd))))  # random baseline
                    _brier_is    = float(np.mean((
                        self.final_model.predict_proba(self.X.fillna(0))[:, 1] - self.y.values
                    ) ** 2))
                    _wr_val      = float(_y_cd.mean()) * 100
                    _mean_prob   = float(_p_cd.mean())
                    _overconf    = _mean_prob - float(_y_cd.mean())  # >0 = sobreconfiado, <0 = subconfiado
                    _is_better   = _brier_val < _brier_naive

                    # Reliability por bins (5 buckets)
                    _bins_cd = np.linspace(0, 1, 6)
                    _rel_log = []
                    for _bi in range(len(_bins_cd) - 1):
                        _lo, _hi = _bins_cd[_bi], _bins_cd[_bi + 1]
                        _mask_bi = (_p_cd >= _lo) & (_p_cd < _hi)
                        _n_bi = int(_mask_bi.sum())
                        if _n_bi >= 5:
                            _wr_bi = float(_y_cd[_mask_bi].mean()) * 100
                            _mp_bi = float(_p_cd[_mask_bi].mean()) * 100
                            _rel_log.append(f"  [{_lo:.1f}-{_hi:.1f}] n={_n_bi:3d} pred={_mp_bi:.1f}% WR={_wr_bi:.1f}%")

                    _flag = "PASS" if _is_better else "FAIL"
                    _icon = "[OK]" if _is_better else "[!!]"
                    logger.info(
                        "[CAL-DIAG-01] {} XGB Calibracion post-fit ({} | {} | thr=0.5):",
                        _icon, self.regime_name or "global", self.native_direction
                    )
                    logger.info(
                        "[CAL-DIAG-01]   Brier_IS={:.4f} | Brier_VAL={:.4f} | Brier_naive={:.4f} | {} {}",
                        _brier_is, _brier_val, _brier_naive, _flag,
                        "(XGB MEJOR QUE RANDOM)" if _is_better else "(XGB PEOR QUE RANDOM - DESCALIBRADO)"
                    )
                    logger.info(
                        "[CAL-DIAG-01]   WR_real={:.1f}% | avg_prob={:.3f} | overconf={:+.3f} | n_val={}",
                        _wr_val, _mean_prob, _overconf, len(_df_cd)
                    )
                    if _rel_log:
                        logger.info("[CAL-DIAG-01]   Reliability diagram (pred_prob -> WR_real):")
                        for _rl in _rel_log:
                            logger.info("[CAL-DIAG-01]  {}", _rl)

                    if not _is_better:
                        logger.warning(
                            "[CAL-DIAG-01] ALERTA: XGB Brier_VAL={:.4f} > Brier_naive={:.4f}. "
                            "El modelo predice PEOR que el azar en validation. "
                            "Posibles causas: Focal Loss activo, overfitting IS, features no causales. "
                            "use_focal_loss={} | n_features={} | n_train={}",
                            _brier_val, _brier_naive, use_focal_loss,
                            len(self.features), len(self.X)
                        )
                    _cal_ok = True
                else:
                    logger.warning("[CAL-DIAG-01] Validation insuficiente tras TBM ({} muestras < 50). Diagnostico omitido.", len(_df_cd))
            else:
                logger.warning("[CAL-DIAG-01] features_validation.parquet no existe. Diagnostico de calibracion omitido.")
        except Exception as _e_cal_diag:
            logger.warning("[CAL-DIAG-01] Diagnostico de calibracion fallido (no bloqueante): {}", _e_cal_diag)
        # ══════════════════════════════════════════════════════════════════════


        out_dir = self.root / "data" / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = f"_{self.regime_name}" if self.regime_name else ""
        model_path = out_dir / f"xgboost_meta{suffix}_{self.native_direction}.model"

        # xgb format
        # [H-RANGE-01-FIX 2026-05-30] Guard: si gate pre-fit abortó (final_model=None),
        # no hay booster que serializar. Guardamos un marcador NULL_MODEL para el RegimeRouter.
        if self.final_model is None:
            import json as _json_null
            _null_marker = out_dir / f"xgboost_meta{suffix}_{self.native_direction}.NULL_MODEL"
            _null_marker.write_text(_json_null.dumps({
                "null_model": True,
                "regime": self.regime_name,
                "direction": self.native_direction,
                "reason": f"GATE_PRE_FIT: n_train < min_viable_train_samples",
                "threshold": self.threshold,
                "dsr": 0.5
            }), encoding="utf-8")
            print(  # RULE[fixbugsprints.md]
                f"[H-RANGE-01-FIX] save_model: final_model=None agente='{self.regime_name}' "
                f"→ NULL_MODEL guardado en {_null_marker.name}. 0 señales OOS."
            )
            logger.warning(
                "[H-RANGE-01-FIX] save_model: NULL_MODEL guardado agente={}. 0 señales.",
                self.regime_name
            )
            print(f"[BUG-FIX-LOG 2026-06-05] [H-RANGE-01-FIX] save_model: NULL_MODEL guardado agente={self.regime_name}. 0 señales.")
            return
        self.final_model.save_model(model_path)
        print(f"[FIX-ISOTONIC-CAL-01] save_model OK: {model_path.name} — iniciando calibracion isotonica")  # debug


        # =====================================================================
        # [FIX-ISOTONIC-CAL-01] Calibrador Isotónico por agente.
        # PROBLEMA (2026-05-18): RegimeRouter carga xgboost_isotonic_calibrator_{suffix}.joblib
        # pero nunca se generaba -> xgb_prob_cal == xgb_prob (gap sobreconfianza +0.130).
        # SOLUCION original: features_validation.parquet. LIMITACION: 100% BULL (KL=8.33).
        #
        # [ARCH-05-FIX-D 2026-06-02] MEJORA: usar IS del régimen propio del agente.
        # PROBLEMA: features_validation.parquet es 100% BULL -> calibrador RANGE/BEAR
        # entrenado con 0 barras de su régimen -> colapso del calibrador.
        # SOLUCIÓN: isotónico se ajusta sobre último 30% del IS propio (self.X filtrado por régimen).
        # RANGE: ~5.400 barras VOLATILE_RANGE; BEAR: ~1.250 barras BEAR.
        # THRESHOLD SWEEP (Optuna -> _calibrate_threshold) NO CAMBIA: sigue usando validation.parquet.
        # FALLBACK a features_validation.parquet si IS del régimen < 300 barras.
        # =====================================================================
        try:
            import joblib as _jbl
            import traceback as _tb_iso
            from sklearn.isotonic import IsotonicRegression as _IR
            _val_parquet = self.root / "data" / "features" / "features_validation.parquet"
            _iso_ok = False

            # [ARCH-05-FIX-D] Seleccionar fuente de calibración isotónica
            _df_v = None
            _iso_cal_source = "unknown"

            if self.X is not None and self.y is not None and len(self.X) >= 300:
                _n_is_total = len(self.X)
                _n_cal_block = int(_n_is_total * 0.30)  # último 30% del IS del régimen
                _X_cal_is = self.X.iloc[-_n_cal_block:].copy()
                _y_cal_is_raw = self.y.iloc[-_n_cal_block:] if hasattr(self.y, 'iloc') else self.y[-_n_cal_block:]

                if len(_X_cal_is) >= 300:
                    _df_v = _X_cal_is.copy()
                    _df_v["bin"] = _y_cal_is_raw.values if hasattr(_y_cal_is_raw, 'values') else _y_cal_is_raw
                    _iso_cal_source = f"IS_regimen_propio_last30pct_n{len(_df_v)}"
                    print(  # RULE[fixbugsprints.md]
                        f"[ARCH-05-FIX-D] Isotonic cal usando IS propio '{self.regime_name}': "
                        f"{len(_df_v)} barras (ultimo 30% de {_n_is_total} IS). "
                        f"Sustituye validation.parquet (100% BULL, KL=8.33)."
                    )
                    logger.info(
                        f"[ARCH-05-FIX-D] Isotonic cal source=IS_regimen_propio | "
                        f"agente={self.regime_name} | n_cal={len(_df_v)} | n_is={_n_is_total}"
                    )
                else:
                    print(f"[ARCH-05-FIX-D] IS regimen {self.regime_name} insuf ({len(_X_cal_is)}<300) — fallback val.parquet")  # RULE[fixbugsprints.md]
                    logger.warning(f"[ARCH-05-FIX-D] IS regimen insuficiente ({len(_X_cal_is)}<300) agente={self.regime_name}")
            else:
                _is_n = len(self.X) if self.X is not None else 0
                print(f"[ARCH-05-FIX-D] self.X insuf (n={_is_n}<300) agente={self.regime_name} — fallback val.parquet")  # RULE[fixbugsprints.md]

            # Fallback: usar features_validation.parquet (comportamiento anterior)
            if _df_v is None:
                if _val_parquet.exists():
                    _df_v = pd.read_parquet(_val_parquet)
                    _iso_cal_source = "features_validation_parquet_fallback"
                    _missing_feats = [f for f in self.features if f not in _df_v.columns]
                    for _mf in _missing_feats:
                        _df_v[_mf] = 0.0
                    _df_v = _df_v.dropna(subset=["close"]).copy()
                    print(f"[ARCH-05-FIX-D] FALLBACK val.parquet: {len(_df_v)} filas agente={self.regime_name}")  # RULE[fixbugsprints.md]
                    logger.info(f"[ARCH-05-FIX-D] Fallback: features_validation.parquet | n={len(_df_v)}")

            if _df_v is not None:
                from config.settings import cfg as _cfg_iso

                # [ARCH-05-FIX-D] Si la fuente es features_validation.parquet (fallback),
                # necesitamos aplicar TBM para generar 'bin'. Si es IS propio, 'bin' ya viene de self.y.
                if not _iso_cal_source.startswith("IS_regimen"):
                    from luna.features.tbm import apply_triple_barrier as _atb
                    # ── [CRITICAL-LUNA-V2] Load TBM parameters strictly from configuration ──
                    try:
                        _pt  = float(getattr(_cfg_iso.xgboost, "pt_mult_min"))
                        _sl  = float(getattr(_cfg_iso.xgboost, "sl_mult_min"))
                        _vbh = int(getattr(_cfg_iso.xgboost, "vertical_barrier_hours"))
                        _mr  = float(getattr(_cfg_iso.xgboost, "tbm_min_return"))
                    except Exception as _cfg_load_err:
                        raise RuntimeError(
                            f"[CRITICAL-LUNA-V2] Error al cargar params TBM para calibracion: {_cfg_load_err}"
                        ) from _cfg_load_err

                    try:
                        _tbm_labels = _atb(
                            price_series=_df_v["close"],
                            event_times=_df_v.index,
                            pt_sl_multiplier=[_pt, _sl],
                            min_return=_mr,
                            vertical_barrier_hours=_vbh,
                        )
                        _df_v = _df_v.join(_tbm_labels[["bin"]], how="inner").dropna(subset=["bin"])
                    except Exception as _e_tbm:
                        logger.warning("[FIX-ISOTONIC-CAL-01] TBM sobre val fallido: {} — ret proxy", _e_tbm)
                        print(f"[FIX-ISOTONIC-CAL-01] TBM fallido: {_e_tbm}")  # RULE[fixbugsprints.md]
                        _df_v["bin"] = (_df_v["close"].pct_change(int(_vbh)).shift(-int(_vbh)) > 0).astype(float)
                        _df_v = _df_v.dropna(subset=["bin"])

                _n_val = len(_df_v)
                if _n_val >= 40:
                    _y_val = (_df_v["bin"] == 1).astype(float).values
                    _X_val = _df_v[self.features].fillna(0)
                    
                    if _iso_cal_source.startswith("IS_regimen"):
                        # [OOF-CALIB-V2 2026-06-03] Evitar overfitting del calibrador
                        # usando predicciones Out-of-Fold para el ultimo 30% del IS.
                        print(f"[OOF-CALIB-V2] Generando predicciones OOF (Out-of-Fold) para calibrar agente={self.regime_name or 'global'}...")
                        logger.info(f"[OOF-CALIB-V2] Generando predicciones OOF para calibracion agente={self.regime_name or 'global'}")
                        
                        from sklearn.model_selection import TimeSeriesSplit
                        from sklearn.base import clone
                        
                        _oof_preds = np.zeros(len(self.X))
                        _oof_mask = np.zeros(len(self.X), dtype=bool)
                        
                        try:
                            from config.settings import cfg as _cfg_oof
                            _purge_gap = int(_cfg_oof.sop.purge_hours)
                        except Exception as e_emb:
                            raise RuntimeError(f"Falta purge_hours en cfg.sop (SOP No-Fallback): {e_emb}")
                            
                        _n_splits_cal = 5 if len(self.X) >= 500 else 3
                        _tscv_cal = TimeSeriesSplit(n_splits=_n_splits_cal, gap=_purge_gap)
                        
                        # Generar predicciones OOF para cada split
                        for _train_idx, _test_idx in _tscv_cal.split(self.X):
                            _model_clone = clone(self.final_model)
                            
                            # Si se usaba custom objective, copiar
                            _model_clone.set_params(objective=_obj_before_restore)
                            
                            # Obtener sub-datasets
                            _X_tr, _y_tr = self.X.iloc[_train_idx], self.y.iloc[_train_idx]
                            _X_te = self.X.iloc[_test_idx]
                            
                            # Obtener sample weights correspondientes a _train_idx
                            _sw_tr = sw_full[_train_idx]
                            
                            # [FIX-SINGLE-CLASS-FOLD-OOF 2026-06-17] Prevenir ValueError
                            if len(np.unique(_y_tr)) < 2:
                                print(f"[OOF-CALIB-V2] Fold ignorado: solo 1 clase en fold train.")
                                _oof_preds[_test_idx] = float(_y_tr.iloc[0])
                                _oof_mask[_test_idx] = True
                                continue
                                
                            # Entrenar clon
                            _model_clone.fit(_X_tr, _y_tr, sample_weight=_sw_tr)
                            
                            # Si tenia custom objective, restaurar
                            if _had_custom_obj:
                                _model_clone.set_params(objective='binary:logistic')
                                _model_clone.get_booster().set_param({'objective': 'binary:logistic'})
                                
                            # Predecir sobre test fold
                            _oof_preds[_test_idx] = _model_clone.predict_proba(_X_te)[:, 1]
                            _oof_mask[_test_idx] = True
                            
                        # Usar predicciones OOF correspondientes a los ultimos _n_cal_block elementos
                        _p_raw = np.zeros(_n_cal_block)
                        for _idx_offset in range(_n_cal_block):
                            _orig_idx = len(self.X) - _n_cal_block + _idx_offset
                            if _oof_mask[_orig_idx]:
                                _p_raw[_idx_offset] = _oof_preds[_orig_idx]
                            else:
                                # Fallback individual in-sample
                                _p_raw[_idx_offset] = self.final_model.predict_proba(_X_val.iloc[[_idx_offset]])[0, 1]
                                
                        print(f"[OOF-CALIB-V2] Predicciones OOF generadas. N={_n_cal_block} | std_oof={np.std(_p_raw):.6f}")
                        logger.info(f"[OOF-CALIB-V2] Predicciones OOF generadas exitosamente. N={_n_cal_block} | std_oof={np.std(_p_raw):.6f}")
                        
                    else:
                        # Si es features_validation.parquet (already out-of-sample), usar predicciones directas
                        _p_raw = self.final_model.predict_proba(_X_val)[:, 1]
                    # ══════════════════════════════════════════════════════════════════════
                    # [GUARDIAN-07] Probability Clustering Guardian (Conviction)
                    # ══════════════════════════════════════════════════════════════════════
                    try:
                        from config.settings import cfg as _cfg_g
                        _min_iqr = float(_cfg_g.xgboost.guardian_min_p_iqr)
                    except Exception as e:
                        raise RuntimeError(f"[CRITICAL-SOP] Falta xgboost.guardian_min_p_iqr en settings.yaml: {e}")
                    
                    try:
                        if len(_p_raw) >= 50:
                            _p_p25 = np.percentile(_p_raw, 25)
                            _p_p75 = np.percentile(_p_raw, 75)
                            _iqr = _p_p75 - _p_p25
                            if _iqr < _min_iqr:
                                logger.error(
                                    f"[GUARDIAN-07] Probability Clustering DETECTADO: IQR de predicciones es {_iqr:.4f} "
                                    f"(< {_min_iqr:.4f}). El modelo perdió convicción y clusteriza en torno a la media. Abortando."
                                )
                                print(f"[GUARDIAN-07] FATAL: XGBoost perdió convicción (IQR={_iqr:.4f} < {_min_iqr:.4f}). Forzando Modo Degradado.")
                                raise ModelDegradationError(f"Probability Clustering (IQR={_iqr:.4f})")
                    except SystemExit:
                        raise
                    except Exception as _e_g7:
                        logger.warning(f"[GUARDIAN-07] Fallo al verificar Probability Clustering: {_e_g7}")

                    # ══════════════════════════════════════════════════════════════════════
                    # [GUARDIAN-01] Spearman Rank-Order Quintiles (Poder de Clasificación)
                    # Si el Top 20% de predicciones de más alta confianza tiene peor Win Rate
                    # que el Bottom 20%, la "confianza" es ruido y el modelo no ordena.
                    # ══════════════════════════════════════════════════════════════════════
                    try:
                        from config.settings import cfg as _cfg_g
                        _min_rank_samples = int(_cfg_g.xgboost.guardian_rank_order_min_samples)
                    except Exception as e:
                        raise RuntimeError(f"[CRITICAL-SOP] Falta xgboost.guardian_rank_order_min_samples en settings.yaml: {e}")

                    try:
                        if len(_p_raw) >= _min_rank_samples:
                            _df_rank = pd.DataFrame({"p": _p_raw, "y": _y_val})
                            _df_rank["quintile"] = pd.qcut(_df_rank["p"], 5, labels=False, duplicates='drop')
                            _q_stats = _df_rank.groupby("quintile")["y"].mean()
                            if len(_q_stats) > 1:
                                _bot_q = _q_stats.index.min()
                                _top_q = _q_stats.index.max()
                                _top_wr = _q_stats[_top_q]
                                _bot_wr = _q_stats[_bot_q]
                                print(f"[BUG-FIX-LOG 2026-06-14] [GUARDIAN-01] Rank-Order Win Rates (n={len(_p_raw)}): Q{_bot_q} (Bottom)={_bot_wr:.1%} | Q{_top_q} (Top)={_top_wr:.1%}")
                                logger.info("[BUG-FIX-LOG 2026-06-14] [GUARDIAN-01] Rank-Order Win Rates (n={}): Q{} (Bottom)={:.1%} | Q{} (Top)={:.1%}", len(_p_raw), _bot_q, _bot_wr, _top_q, _top_wr)
                                if _top_wr <= _bot_wr and _top_wr < 0.50:
                                    logger.error(
                                        f"[GUARDIAN-01] Rank-Order FALLIDO: Top WR ({_top_wr:.1%}) <= Bottom WR ({_bot_wr:.1%}). "
                                        f"El modelo no sabe ordenar trades (probabilidades aleatorias). Abortando."
                                    )
                                    print(f"[GUARDIAN-01] FATAL: Top WR ({_top_wr:.1%}) <= Bottom WR ({_bot_wr:.1%}). Forzando Modo Degradado.")
                                    raise ModelDegradationError(f"Rank-Order Inverted (Top={_top_wr:.1%} <= Bot={_bot_wr:.1%})")
                        else:
                            print(f"[BUG-FIX-LOG 2026-06-14] [GUARDIAN-01] Rank-Order OMITIDO: len(_p_raw)={len(_p_raw)} < guardian_rank_order_min_samples={_min_rank_samples}")
                            logger.warning(
                                "[BUG-FIX-LOG 2026-06-14] [GUARDIAN-01] Rank-Order OMITIDO: len(_p_raw)={} < min_samples={}",
                                len(_p_raw), _min_rank_samples
                            )
                    except SystemExit:
                        raise
                    except Exception as _e_g1:
                        logger.warning(f"[GUARDIAN-01] Fallo al calcular Rank-Order: {_e_g1}")

                    # ══════════════════════════════════════════════════════════════════════
                    # [GUARDIAN-05] Desintegración de Distribución (OOD Covariate Shift)
                    # ══════════════════════════════════════════════════════════════════════
                    try:
                        from config.settings import cfg as _cfg_g
                        _ood_kl_thresh = float(_cfg_g.xgboost.guardian_ood_kl_threshold)
                        _ood_max_fails = int(_cfg_g.xgboost.guardian_ood_max_failures)
                    except Exception as e:
                        raise RuntimeError(f"[CRITICAL-SOP] Falta guardian_ood_kl_threshold/guardian_ood_max_failures en settings.yaml: {e}")

                    try:
                        if len(_X_val) >= 50 and len(self.X) >= 50:
                            import scipy.stats as _stats
                            _kl_failures = 0
                            _importances = pd.Series(self.final_model.feature_importances_, index=self.X.columns)
                            _top_feats_kl = _importances.sort_values(ascending=False).head(5).index
                            for _f_kl in _top_feats_kl:
                                _p_dist, _ = np.histogram(self.X[_f_kl].dropna(), bins=10, density=True)
                                _q_dist, _ = np.histogram(_X_val[_f_kl].dropna(), bins=10, density=True)
                                _p_dist = np.where(_p_dist == 0, 1e-6, _p_dist)
                                _q_dist = np.where(_q_dist == 0, 1e-6, _q_dist)
                                _kl_div = _stats.entropy(_p_dist, _q_dist)
                                if _kl_div > _ood_kl_thresh:
                                    _kl_failures += 1
                            if _kl_failures >= _ood_max_fails:
                                logger.error(
                                    f"[GUARDIAN-05] Covariate Shift EXTREMO: {_kl_failures} de las 5 features "
                                    f"top están OOD (KL > {_ood_kl_thresh}). El modelo no generalizará. Abortando."
                                )
                                print(f"[GUARDIAN-05] FATAL: OOD Explosion detectada. KL > {_ood_kl_thresh} en {_kl_failures}/5 features top. Forzando Modo Degradado.")
                                raise ModelDegradationError(f"OOD Covariate Shift ({_kl_failures}/5 failures)")
                    except SystemExit:
                        raise
                    except Exception as _e_g5:
                        logger.warning(f"[GUARDIAN-05] Fallo al verificar Covariate Shift: {_e_g5}")

                    # ── [LUNA-V2-CALIB] Parámetros de calibración desde settings ──
                    try:
                        _min_samples_iso = int(getattr(_cfg_iso.xgboost, "calibration_min_samples_isotonic"))
                        _fallback_method = getattr(_cfg_iso.xgboost, "calibration_fallback_method")
                    except Exception as _cfg_load_err:
                        raise RuntimeError(
                            f"[CRITICAL-LUNA-V2] Error al cargar parámetros de calibración desde settings.yaml: {_cfg_load_err}"
                        ) from _cfg_load_err

                    # ─────────────────────────────────────────────────────────────────
                    # [FIX-CALIB-TEMP-01] PRE-CHECK: diagnóstico de varianza raw ANTES
                    # de intentar cualquier calibración.
                    #
                    # CASO A — MODEL COLLAPSE (std_raw < 1e-3):
                    #   El modelo XGBoost produce probabilidades prácticamente idénticas.
                    #   Causa típica: régimen (ej. "bear") ausente del período de validación.
                    #   El XGBoost converge al base-rate sin discriminación.
                    #   Ningún calibrador (isotónico, Platt, Temperature) puede crear
                    #   varianza donde no existe — la transformación monotona preserva std=0.
                    #   Acción: raw fallback inmediato. El threshold sweep producirá EV≈0
                    #   → 0 trades → comportamiento estadísticamente correcto.
                    #
                    # CASO B — CALIBRATOR COLLAPSE (std_raw ok, isotónica colapsa):
                    #   El modelo tiene señal (std_raw > 1e-3) pero la distribución de
                    #   breakpoints es irregular → IsotonicRegression produce 1 anchor.
                    #   Temperature Scaling rescata la señal (garantía matemática).
                    # ─────────────────────────────────────────────────────────────────
                    _std_raw   = float(np.std(_p_raw))
                    _range_raw = float(_p_raw.max() - _p_raw.min())
                    _MODEL_COLLAPSE = _std_raw < 1e-3

                    print(
                        f"[FIX-CALIB-TEMP-01] PRE-CHECK raw probs | "
                        f"agente={self.regime_name or 'global'} dir={self.native_direction} | "
                        f"n_val={_n_val} std_raw={_std_raw:.6f} range_raw={_range_raw:.6f} | "
                        f"{'⚠ MODEL COLLAPSE → raw fallback directo' if _MODEL_COLLAPSE else '✓ señal detectada → intentar calibración'}"
                    )  # RULE[fixbugsprints.md]

                    if _MODEL_COLLAPSE:
                        # CASO A: colapso del modelo — ningún calibrador puede ayudar
                        logger.error(
                            "[GUARDIAN-02] MODEL COLLAPSE agente={} dir={} | "
                            "std(p_raw)={:.2e} range={:.6f} — XGBoost produce probs sin varianza. "
                            "Causa típica: régimen ausente en val period (distribución OOD). "
                            "Ningún calibrador puede crear señal inexistente. Abortando.",
                            self.regime_name or "global", self.native_direction,
                            _std_raw, _range_raw
                        )
                        print(f"[GUARDIAN-02] FATAL: Model Collapse pre-calibración. std={_std_raw:.2e}. Forzando Modo Degradado.")
                        raise ModelDegradationError(f"Model Collapse pre-cal (std={_std_raw:.2e})")
                    else:
                        # CASO B+ : hay señal — intentar cascada de calibración
                        # [FIX-ISOTONIC-BLINDNESS-01] Nivel 1: PlattCalibrator incondicional
                        # Isotonic (Step-Function) fue desactivado porque causaba "ceguera" en colas OOS.
                        print(f"[LUNA-V2-CALIB] n_val={_n_val}. Usando PlattCalibrator (LogisticRegression) para preservar diferenciación en colas OOS.")
                        logger.info(f"[LUNA-V2-CALIB] n_val={_n_val}. Usando PlattCalibrator (LogisticRegression) para preservar diferenciación en colas OOS.")
                        _iso = PlattCalibrator()

                        _iso.fit(_p_raw, _y_val)
                        _p_cal = _iso.predict(_p_raw)

                        _brier_r  = float(np.mean((_p_raw - _y_val) ** 2))
                        _brier_c  = float(np.mean((_p_cal - _y_val) ** 2))
                        _gap_r    = float(_p_raw.mean()) - float(_y_val.mean())
                        _gap_c    = float(_p_cal.mean()) - float(_y_val.mean())
                        _improve  = (_brier_r - _brier_c) / max(_brier_r, 1e-9) * 100
                        _iso_std  = float(np.std(_p_cal))
                        _n_anchors = len(getattr(_iso, 'X_thresholds_', []))
                        _iso_range = float(_p_cal.max() - _p_cal.min())

                        print(
                            f"[BUG-CALIB-XGB-01] SANITY CHECK L1 | "
                            f"agente={self.regime_name or 'global'} dir={self.native_direction} | "
                            f"std_cal={_iso_std:.6f} anchors={_n_anchors} range=[{_p_cal.min():.4f},{_p_cal.max():.4f}]"
                        )

                        _L1_DEGENERATE = (_iso_std < 1e-4) or (_n_anchors <= 2 and _iso_range < 1e-4)

                        if not _L1_DEGENERATE:
                            # Nivel 1 funcionó — guardar y continuar
                            logger.info(
                                "[FIX-ISOTONIC-CAL-01] Calibrador L1 OK | agente={} dir={} n_val={} | "
                                "Brier {:.4f}->{:.4f} ({:+.1f}%) | gap_overconf {:+.3f}->{:+.3f} | "
                                "std_cal={:.4f} anchors={}",
                                self.regime_name or "global", self.native_direction,
                                _n_val, _brier_r, _brier_c, _improve, _gap_r, _gap_c,
                                _iso_std, _n_anchors
                            )
                            print(
                                f"[FIX-ISOTONIC-CAL-01] agente={self.regime_name or 'global'} "
                                f"dir={self.native_direction} n={_n_val} | "
                                f"Brier {_brier_r:.4f}->{_brier_c:.4f} ({_improve:+.1f}%) | "
                                f"gap {_gap_r:+.3f}->{_gap_c:+.3f} | std_cal={_iso_std:.4f} anchors={_n_anchors}"
                            )
                            _iso_path = out_dir / f"xgboost_isotonic_calibrator{suffix}_{self.native_direction}.joblib"
                            _jbl.dump(_iso, _iso_path)
                            _iso_ok = True
                            logger.info("[FIX-ISOTONIC-CAL-01] Calibrador L1 GUARDADO: {}", _iso_path.name)
                            print(f"[FIX-ISOTONIC-CAL-01] GUARDADO: {_iso_path.name}")

                        else:
                            # Nivel 1 degenerado — intentar Nivel 2: Temperature Scaling
                            # (Guo et al. 2017 — garantía: si std(p_raw)>0, std(p_cal)>0 siempre)
                            print(
                                f"[FIX-CALIB-TEMP-01] L1 degenerado (std={_iso_std:.2e}, anchors={_n_anchors}). "
                                f"Intentando Temperature Scaling (L2)... std_raw={_std_raw:.6f}"
                            )
                            logger.info(
                                "[FIX-CALIB-TEMP-01] L1 degenerado agente={} — escalando a Temperature Scaling L2.",
                                self.regime_name or "global"
                            )

                            try:
                                _temp_cal = TemperatureCalibrator()
                                _temp_cal.fit(_p_raw, _y_val)
                                _p_temp = _temp_cal.predict(_p_raw)
                                _std_temp = float(np.std(_p_temp))
                                _brier_temp = float(np.mean((_p_temp - _y_val) ** 2))
                                _improve_temp = (_brier_r - _brier_temp) / max(_brier_r, 1e-9) * 100

                                print(
                                    f"[FIX-CALIB-TEMP-01] Temperature Scaling L2 | "
                                    f"agente={self.regime_name or 'global'} dir={self.native_direction} | "
                                    f"T={_temp_cal.temperature:.4f} std_temp={_std_temp:.6f} | "
                                    f"Brier {_brier_r:.4f}->{_brier_temp:.4f} ({_improve_temp:+.1f}%)"
                                )  # RULE[fixbugsprints.md]

                                if _std_temp >= 1e-3:
                                    # Temperature Scaling rescató la señal — guardar
                                    logger.info(
                                        "[FIX-CALIB-TEMP-01] Temperature Scaling L2 OK | "
                                        "agente={} dir={} T={:.4f} std_temp={:.4f} | "
                                        "Brier {:.4f}->{:.4f} ({:+.1f}%)",
                                        self.regime_name or "global", self.native_direction,
                                        _temp_cal.temperature, _std_temp,
                                        _brier_r, _brier_temp, _improve_temp
                                    )
                                    _iso_path = out_dir / f"xgboost_isotonic_calibrator{suffix}_{self.native_direction}.joblib"
                                    _jbl.dump(_temp_cal, _iso_path)
                                    _iso_ok = True
                                    print(f"[FIX-CALIB-TEMP-01] TemperatureCalibrator GUARDADO: {_iso_path.name}")
                                else:
                                    # También colapsa — model collapse más sutil (el pre-check era limítrofe)
                                    logger.error(
                                        "[GUARDIAN-02] Temperature Scaling L2 también colapsa | "
                                        "agente={} dir={} T={:.4f} std_temp={:.2e}. "
                                        "Señal insuficiente incluso con T-Scaling. Abortando.",
                                        self.regime_name or "global", self.native_direction,
                                        _temp_cal.temperature, _std_temp
                                    )
                                    print(f"[GUARDIAN-02] FATAL: L2 también colapsa std_temp={_std_temp:.2e} → Forzando Modo Degradado.")
                                    raise ModelDegradationError(f"Model Collapse post-cal (std={_std_temp:.2e})")

                            except Exception as _e_temp:
                                logger.warning(
                                    "[FIX-CALIB-TEMP-01] ERROR en Temperature Scaling L2: {} — Raw fallback.",
                                    _e_temp
                                )
                                print(f"[FIX-CALIB-TEMP-01] ERROR L2: {_e_temp} → Raw fallback.")
                                _iso_ok = False

                else:
                    logger.warning(
                        "[FIX-ISOTONIC-CAL-01] n_val={} < 40 — calibrador NO generado agente={}",
                        _n_val, self.regime_name or "global"
                    )
                    print(f"[FIX-ISOTONIC-CAL-01] SKIP n_val={_n_val}<40 agente={self.regime_name or 'global'}")  # debug
            else:
                logger.warning("[FIX-ISOTONIC-CAL-01] features_validation.parquet NO existe — sin calibrador")
                print(f"[FIX-ISOTONIC-CAL-01] SKIP: {_val_parquet} no existe")  # debug

            if not _iso_ok:
                print(f"[FIX-ISOTONIC-CAL-01] ADVERTENCIA: Sin calibrador agente={self.regime_name or 'global'} dir={self.native_direction}")  # debug

        except Exception as _e_iso:
            _tb_str = _tb_iso.format_exc() if '_tb_iso' in dir() else str(_e_iso)
            logger.error("[FIX-ISOTONIC-CAL-01] ERROR no bloqueante: {}\n{}", _e_iso, _tb_str)
            print(f"[FIX-ISOTONIC-CAL-01] ERROR: {_e_iso}\n{_tb_str}")  # debug
        # ===================================================================== FIN ISOTONIC


        # â”€â”€ CalibraciÃ³n automÃ¡tica del threshold (MEJORA-R12-01) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Barre thresholds sobre features_validation.parquet maximizando EV(t).
        # Resultado guardado en la firma â€” generate_oos_predictions.py lo carga.
        self._calibration_report = []
        optimal_threshold = self._calibrate_threshold()

        # [FIX-DSR-MASK-01] Extraer métricas reales con distinción entre fuentes:
        # - dsr_cpcv_best: study.best_value → DSR CPCV promediado, la métrica canónica.
        #   SIEMPRE positivo cuando el tuning converge (es el Best optimizado por Optuna).
        # - dsr_oos_telemetry: user_attrs del best_trial → puede ser NaN si el trial
        #   no guardó atributos (ej. trial podado). Fallback a -1.0 para detección.
        # BUG ANTERIOR: dsr_oos en la firma usaba dsr_telemetry → NaN → -1.0, enmascarando
        # el DSR real (0.25). Resultado: dashboards y _compare_run_metrics mostraban -1.0
        # aunque el modelo tuviese señal predictiva confirmada por CPCV.
        # [FIX-DUMMY-STUDY] Acceso seguro a study y best_trial
        if hasattr(self, 'study') and self.study is not None and hasattr(self.study, 'best_trial') and self.study.best_trial is not None:
            best_trial = self.study.best_trial
            dsr_telemetry = best_trial.user_attrs.get("dsr_telemetry", float("nan"))
            
            # [FIX-BRIER-LOGLOSS-MIXUP] Extraer Brier puro, no la metrica de Optuna que podría ser LogLoss
            brier_raw     = best_trial.user_attrs.get("brier_is", best_trial.user_attrs.get("metric_is", float("nan")))
            
            print("[FIX-DUMMY-STUDY] Cargada telemetría real desde study.best_trial.")  # debug
            logger.info("[FIX-DUMMY-STUDY] Cargada telemetría real desde study.best_trial.")
        else:
            best_trial = None
            dsr_telemetry = float("nan")
            brier_raw     = float("nan")
            print("[FIX-DUMMY-STUDY] Fallback: No se detectó best_trial en self.study. Telemetría mockeada.")  # debug
            logger.info("[FIX-DUMMY-STUDY] Fallback: No se detectó best_trial en self.study. Telemetría mockeada.")
            
        dsr_cpcv_best    = float(self.study.best_value) if (hasattr(self, 'study') and self.study is not None and hasattr(self.study, 'best_value')) else 0.50  # canónico — siempre fiable
        dsr_oos_legacy   = dsr_telemetry if not np.isnan(dsr_telemetry) else -1.0  # retrocompat
        if np.isnan(brier_raw):
            brier_raw = 1.0  # [FIX-BRIER-LOGLOSS-MIXUP] Fallback conservador para brier

        # Save signature
        sig_path = out_dir / f"xgboost_meta{suffix}_{self.native_direction}_signature.json"
        with open(sig_path, 'w') as f:
            json.dump({
                "features":           self.features,
                # [FIX-DSR-MASK-01] dsr_oos ahora es el CPCV Best (canónico), no la telemetría
                # del trial individual (que puede ser NaN→-1.0 por pruning o fallo de attrs).
                "dsr_oos":            dsr_cpcv_best,
                "dsr_cpcv_best":      dsr_cpcv_best,
                "dsr_oos_telemetry":  dsr_oos_legacy,
                "xgb_brier_raw":      brier_raw,
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
            # IDEA-G: Feature importances para diagnostico de regimen (2026-05-07)
            "feature_importances":        getattr(self, "_fi_gain_top20", {}),
            "feature_importances_weight": getattr(self, "_fi_weight_top20", {}),
            "feature_importances_cover":  getattr(self, "_fi_cover_top20", {}),
            "stable_features_all_folds":  getattr(self, "_stable_fi", []),
            # [P4-WARM-START] Telemetría de adopción
            "warm_start_used":            getattr(self, "_ws_enabled", False),
            "warm_start_count":           getattr(self, "_ws_count", 0),
            "best_trial_number":          self.study.best_trial.number if (hasattr(self, "study") and hasattr(self.study, "best_trial") and self.study.best_trial) else -1,
            # IDEA-A: Brier gate adaptativo al regimen actual (2026-05-07)
            # [FIX-IDEA-A-01] Usa _brier_adaptive_gate calculado arriba (None si NO_OPERABLE)
            "target_base_rate":    self._base_rate_is,
            "brier_naive":         round(self._base_rate_is * (1 - self._base_rate_is), 4),
            "brier_adaptive_gate": getattr(self, "_brier_adaptive_gate", None),
            }, f, indent=4)


        # â”€â”€ [DATAFLOW-EXPORT-XGB-01] Model Signature Audit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        logger.success(
            f"[DATAFLOW-EXPORT-XGB-01] Modelo guardado: {model_path.name} | "
            f"n_features={len(self.features)} | "
            f"dsr_cpcv_best={dsr_cpcv_best:.4f} | dsr_oos_telemetry={dsr_oos_legacy:.4f} | "
            f"threshold_calibrado={optimal_threshold:.2f} ({getattr(self, '_cal_source', 'validation')})"
        )
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        
class MultiAgentXGBoostTrainer:
    """
    Orquestador de N agentes XGBoost especializados por régimen HMM.
    FASE 2: Enrutamiento Bull/Range/Bear basado en etiquetas semánticas.
    """
    def __init__(self):
        # Mapeo textual semántico a cada agente experto (Centralizado en settings.yaml)
        try:
            from config.settings import cfg as _cfg_ma
            self.regimes_config = vars(_cfg_ma.fase2.regime_mapping)
        except Exception as e:
            from loguru import logger
            logger.warning(f"Error cargando regime_mapping: {e}. Fallback interno.")
            # [SOL3-CALM-BEAR-01 2026-06-01] Fallback actualizado: calm_bear dedicado.
            # Separar CALM_BEAR del agente bear genérico mejora la señal en períodos de bajada suave.
            # El agente 'bear' cubre solo CRASH/FORCED; calm_bear cubre CALM_BEAR variants.
            self.regimes_config = {
                "bull":      ["1_BULL_TREND", "1_VOLATILE_BULL", "1_BULL_GRIND", "1_BULL_TREND_WEAK", "1_BULL_TREND_B", "1_VOLATILE_BULL_B"],
                "range":     ["2_CALM_RANGE", "2_VOLATILE_RANGE", "2_CALM_RANGE_B", "2_VOLATILE_RANGE_B"],
                "calm_bear": ["3_CALM_BEAR", "3_CALM_BEAR_B", "3_CALM_BEAR_C", "3_CALM_BEAR_D"],
                "bear":      ["3_BEAR_CRASH", "3_BEAR_CRASH_B", "3_BEAR_CRASH_C", "4_BEAR_FORCED"]
            }
            print("[SOL3-CALM-BEAR-01/FALLBACK] MultiAgentXGBoost usando fallback con agente calm_bear.")
        self.trainers = {}
        
    def run_all(self):
        logger.info("[FASE 2] Iniciando Entrenamiento Multi-Agent XGBoost por Régimen")
        # Dividir los trials totales entre los 3 regímenes para mantener tiempo cte
        n_regimes = len(self.regimes_config)
        trials_per_regime = max(30, OPTUNA_TRIALS // n_regimes)

        # [BUG-RANGE-01 FIX] Auto-reconciliar etiquetas HMM reales contra el regime_mapping.
        # El HMM genera etiquetas dinámicas (ej: '2_VOLATILE_RANGE') que pueden no estar
        # enumeradas en settings.yaml:fase2.regime_mapping si el mapping fue escrito para
        # un HMM con n_states distinto. Expandimos cada agente con las etiquetas reales
        # que contengan su prefijo semántico ('1_' = bull, '2_' = range, '3_'/'4_' = bear).
        try:
            from pathlib import Path as _Path
            import pyarrow.parquet as _pq_ra
            # HMM_Semantic está en hmm_regime_labels.parquet, no en features_train.parquet
            _feat_base = _Path(__file__).resolve().parents[2] / "data" / "features"
            _hmm_candidates = [
                _feat_base / "hmm_regime_labels.parquet",
                _feat_base / "features_validation.parquet",  # fallback: también contiene HMM_Semantic
            ]
            _feat_path = next((p for p in _hmm_candidates if p.exists()), None)
            if _feat_path is not None:
                _df_ra = _pq_ra.read_table(str(_feat_path), columns=["HMM_Semantic"]).to_pandas()
                _real_labels = set(_df_ra["HMM_Semantic"].dropna().unique().tolist())
                logger.info(f"[BUG-RANGE-01 FIX] Etiquetas HMM reales en {_feat_path.name}: {sorted(_real_labels)}")

                # Prefijos semánticos por agente
                _prefix_map = {
                    "bull":  ["1_"],
                    "range": ["2_"],
                    "bear":  ["3_", "4_"],
                }
                _reconciled = {}
                for _agent, _cfg_list in self.regimes_config.items():
                    _prefixes = _prefix_map.get(_agent, [])
                    # Unión: los del config + los reales que coincidan por prefijo
                    _auto = [l for l in _real_labels if any(l.startswith(p) for p in _prefixes)]
                    _merged = sorted(set(_cfg_list) | set(_auto))
                    if set(_merged) != set(_cfg_list):
                        _added = sorted(set(_merged) - set(_cfg_list))
                        logger.info(f"[BUG-RANGE-01 FIX] Agente '{_agent}': añadidas etiquetas reales {_added}")
                    _reconciled[_agent] = _merged
                self.regimes_config = _reconciled
            else:
                logger.warning("[BUG-RANGE-01 FIX] features_train.parquet no encontrado — usando regime_mapping de settings.yaml sin reconciliar")
        except Exception as _e_ra:
            logger.warning(f"[BUG-RANGE-01 FIX] Auto-reconciliación HMM falló (no bloquea): {_e_ra}")

        for name, r_list in self.regimes_config.items():
            logger.info(f"\n{'='*50}\n[FASE 2] Entrenando Agente [{name.upper()}] (Regimes: {r_list})\n{'='*50}")
            t = XGBoostTrainer(regime_name=name, regime_list=r_list, n_trials=trials_per_regime)
            try:
                t.load_dataset()
                t.tune_hyperparameters()
                t.train_final_model()
                self.trainers[name] = t
            except ModelDegradationError as e:
                logger.error(f"[MODO DEGRADADO] Agente {name.upper()} omitido por fallo de guardián: {e}")
                print(f"[GUARDIAN/DEGRADADO] Agente {name.upper()} no superó validación: {e}")
            except ValueError as e:
                # [BUG-RANGE-01] Captura explícita de 0-eventos / SOP-R8-GATE.
                # Con FIX-REGIME-POOL-01 activo, el modo universal debería evitar la mayoría
                # de estos casos. Si aún llega aquí es un fallo genuino (n < 30 incluso en IS).
                import traceback
                if "0 eventos" in str(e) or "BUG-RANGE-01" in str(e):
                    logger.warning(f"[FASE 2] Agente {name.upper()} omitido tras FIX-REGIME-POOL-01: {e}")
                    print(
                        f"[BUG-RANGE-01/SKIP-FINAL] Agente {name.upper()} omitido incluso en modo universal: "
                        f"{str(e)[:200]}. regime_list={r_list}. RegimeRouter usara fallback global."
                    )
                else:
                    logger.error(f"[FASE 2] Falló el entrenamiento del agente {name.upper()}: {e}\n{traceback.format_exc()}")
                    print(f"[FASE 2] ERROR agente {name.upper()}: {e}")
            except Exception as e:
                import traceback
                logger.error(f"[FASE 2] Falló el entrenamiento del agente {name.upper()}: {e}\n{traceback.format_exc()}")

        n_trained = len(self.trainers)
        n_skipped = len(self.regimes_config) - n_trained
        print(
            f"[FIX-REGIME-POOL-01/SUMMARY] Agentes entrenados: {n_trained}/{len(self.regimes_config)} "
            f"| Omitidos: {n_skipped} | Trained: {list(self.trainers.keys())}"
        )
        logger.info(
            f"[FIX-REGIME-POOL-01/SUMMARY] Multi-Agent: {n_trained} entrenados, {n_skipped} omitidos."
        )
        if n_trained == 0:
            print(
                "[FIX-REGIME-POOL-01/CRITICAL] NINGÚN agente entrenado — signal funnel colapsará. "
                "Revisar distribución de regímenes en IS y configuración de regime_mapping."
            )
            logger.error(
                "[FIX-REGIME-POOL-01/CRITICAL] 0 agentes entrenados — todos los funnels generarán 0 señales."
            )
        logger.success("[FASE 2] Entrenamiento Multi-Agent Completado Exitosamente.")


if __name__ == "__main__":
    import os as _os
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    _log_dir = _Path(__file__).resolve().parents[2] / "logs"
    _log_dir.mkdir(exist_ok=True)
    _ts_xgb  = _dt.now().strftime("%Y%m%d_%H%M%S")
    _rid_xgb = _os.environ.get("LUNA_RUN_ID", "")
    _lname_xgb = f"train_xgboost_v2_{_ts_xgb}_{_rid_xgb}.log" if _rid_xgb else f"train_xgboost_v2_{_ts_xgb}.log"
    logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
    logger.add(_log_dir / _lname_xgb, rotation="100 MB", level="DEBUG")

    try:
        try:
            from config.settings import cfg as _cfg_main
            use_regime = bool(_cfg_main.fase2.use_regime_agents)
        except Exception as e:
            logger.warning(f"No se pudo leer fase2.use_regime_agents ({e}), usando trainer estándar")
            use_regime = False

        if use_regime:
            ma = MultiAgentXGBoostTrainer()
            ma.run_all()
            if len(ma.trainers) == 0:
                logger.error("[FASE 2 FATAL] Todos los agentes fallaron. Abortando.")
                sys.exit(1)
        else:
            trainer = XGBoostTrainer()
            trainer.load_dataset()
            trainer.tune_hyperparameters()
            trainer.train_final_model()
            if getattr(trainer, 'xgb_model', None) is None:
                sys.exit(1)
        sys.exit(0)
    except Exception as e:
        import traceback
        logger.error(f"[FATAL UNCAUGHT] Script crashed at main level: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)



