"""
calibrate_probabilities.py — Luna V1
======================================
Calibración de probabilidades del pipeline MetaLabelerV2.

Calibra el Random Forest árbitro del MetaLabelerV2 usando
CalibratedClassifierCV (isotonic o platt) sobre el validation set.

Reescrito en P4-0 (2026-03-08) para MetaLabelerV2 (LSTM extractor + RF árbitro).
El script anterior usaba MetaLabelerBiLSTM (v1 — OBSOLETO, eliminado del pipeline).

Pipeline de calibración:
  1. Cargar MetaLabelerV2 entrenado (LSTM pesos + RF árbitro)
  2. Cargar features_validation.parquet (período OOS real)
  3. Generar embeddings LSTM sobre validation + XGBoost OOS probs + HMM context
  4. Calibrar el RF con CalibratedClassifierCV (método beta o platt)
  5. Guardar el calibrador en el MetaLabelerV2 y re-persistir

SOP aplicado:
- R1: calibración SOLO sobre validation set (OOS temporal)
- R10: calibración beta/platt como capa final del pipeline
"""

import sys
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

ROOT = get_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import joblib
import argparse
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from luna.utils.debug_guards import check_model_sanity, check_numeric_stability, timeit
# [I2-FIX] Temperature Scaling como fallback cuando Isotónico Y Platt colapsan
# Módulo documentado en ROADMAP_ARQUITECTURA_DINAMICA_V2.md §7
from luna.calibration.temperature_scaler import TemperatureScaler as _TemperatureScaler


class _TSAdapter:
    """[I2-FIX] Wrapper pickleable de TemperatureScaler para _RFWithAdapter.
    Clase top-level requerida para joblib.dump (las clases locales no son pickleable).
    """
    def __init__(self, scaler: "_TemperatureScaler"):
        self._scaler = scaler

    def predict_proba(self, raw: "np.ndarray") -> "np.ndarray":
        raw1d = raw.ravel() if raw.ndim > 1 else raw
        cal = self._scaler.calibrate(raw1d)
        return np.column_stack([1 - cal, cal])


class _IdentityWrapper:
    """
    P6-FIX: wrapper top-level del RF sin calibración para evitar PicklingError.
    pickle requiere que las clases sean importables por nombre desde el módulo.
    Se usa cuando la calibración empeora el Brier (skip-if-worse).
    """
    def __init__(self, rf):
        self._rf = rf

    def predict_proba(self, X):
        return self._rf.predict_proba(X)

    def predict(self, X):
        return self._rf.predict(X)


class _RFWithAdapter:
    """
    ARCH-FIX-CALIB-01: combina RF nativo + calibrador separado (pickleable).
    Flujo: X_combined → RF.predict_proba → raw_probs → adapter → calibrated_probs.
    El RF nunca se re-entrena ni transforma por la calibración.
    El adapter es un objeto sklearn ligero (LogisticRegression o IsotonicRegression)
    ajustado SOLO sobre raw_probs del RF en validation.
    """
    def __init__(self, rf, adapter, method: str):
        self._rf      = rf
        self._adapter = adapter   # LR o IsotonicRegression
        self._method  = method

    def predict_proba(self, X):
        raw = self._rf.predict_proba(X)[:, 1]
        if hasattr(self._adapter, 'predict_proba'):  # LogisticRegression
            cal = self._adapter.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:                                         # IsotonicRegression
            cal = self._adapter.predict(raw)
        # Devolver array (n, 2) estándar sklearn
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class MetaLabelerV2Calibrator:
    """
    Calibra el RF árbitro del MetaLabelerV2 usando CalibratedClassifierCV.

    El MetaLabelerV2 genera probabilidades con el RF nativo (predict_proba).
    La calibración ajusta esas probabilidades para que sean fieles a la
    frecuencia empírica de éxito en el validation set OOS.

    Nota: la calibración se entrena SIEMPRE sobre features_validation.parquet,
    nunca sobre el train set (SOP R1: causalidad estricta).
    """

    def __init__(self, direction: str = "long"):
        self.root = ROOT
        self.direction = direction
        self.models_dir = ROOT / "data" / "models"
        self.features_dir = ROOT / "data" / "features"

    @staticmethod
    def _make_identity_calibrator(rf_model, X: np.ndarray, y: np.ndarray):
        """
        P6-FIX: devuelve _IdentityWrapper(rf) — clase top-level para evitar PicklingError.
        Usado cuando la calibración empeora el Brier (skip-if-worse).
        """
        return _IdentityWrapper(rf_model)

    def calibrate(self, method: str = "isotonic") -> None:
        """
        Args:
            method: 'isotonic' (más flexible, preferable con n>1000)
                    'sigmoid' / 'platt' (equivalentes, para n<1000)
        """
        logger.info("=" * 60)
        logger.info(f"Calibrando MetaLabelerV2 (método: {method})")
        logger.info("=" * 60)

        # 0. Normalizar nombres
        if method == "platt":
            method = "sigmoid"

        # 1. Verificar archivos requeridos
        v2_config_path = self.models_dir / f"metalabeler_v2_{self.direction}_config.json"
        v2_lstm_path   = self.models_dir / f"metalabeler_v2_{self.direction}_lstm.pt"
        v2_rf_path     = self.models_dir / f"metalabeler_v2_{self.direction}_rf.joblib"
        val_path       = self.features_dir / "features_validation.parquet"
        
        # Override WFB para features
        import os
        if 'LUNA_WFB_WINDOW_DIR' in os.environ:
            w_id = os.environ['LUNA_WFB_WINDOW_DIR'][-1]
            alt_path = self.root / "data" / "features" / f"features_validation_W{w_id}.parquet"
            if alt_path.exists():
                val_path = alt_path
                logger.info(f"Usando dataset de validacion especifico de W{w_id}: {val_path}")
        
        try:
            from config.settings import cfg as _cfg_xgb
            use_regime = bool(_cfg_xgb.fase2.use_regime_agents)
        except Exception:
            use_regime = False

        xgb_model_path = self.models_dir / "xgboost_meta.model"
        xgb_sig_path   = self.models_dir / "xgboost_meta_signature.json"
        
        required_files = [v2_config_path, v2_rf_path, val_path]
        if use_regime:
            for model_name in ["bull", "range", "bear"]:
                for direction in ["long", "short"]: # Nueva arquitectura multi-agente direccional
                    # Es posible que algunos regímenes no tengan cortos, chequeamos si existe antes de requerir?
                    # Lo ideal es dejar que RegimeRouter maneje la carga. Skip archivo estricto aquí para agentes 
                    pass
        else:
            required_files.extend([xgb_model_path, xgb_sig_path])

        for p in required_files:
            if not p.exists():
                logger.error(f"Archivo requerido no encontrado: {p}")
                logger.error("Ejecutar primero: python scripts/run_features_and_training.py --only-train")
                return False

        # 2. Cargar MetaLabelerV2
        from luna.models.train_metalabeler_v2 import MetaLabelerV2, SEQ_LEN, HMM_N_STATES
        v2_config = json.loads(v2_config_path.read_text(encoding="utf-8"))
        try:
            model = MetaLabelerV2.load(self.models_dir, direction_mode=self.direction)
            logger.info(f"MetaLabelerV2 cargado: input_dim={v2_config.get('input_dim')}, "
                        f"hidden={v2_config.get('lstm_hidden')}")
        except Exception as e:
            logger.error(f"No se pudo cargar MetaLabelerV2: {e}")
            return

        seq_len  = v2_config.get("seq_len", SEQ_LEN)
        hmm_ctx  = v2_config.get("hmm_context", False)
        
        try:
            from config.settings import cfg
            n_states = int(cfg.hmm.n_states)
        except Exception as e:
            raise RuntimeError(f"Fallo leyendo cfg.hmm.n_states: {e}")

        # 3. Cargar validation set
        df_val = pd.read_parquet(val_path)
        
        # [ARCH-25 FIX] Split validation (1441 bars) into val_A (Isotonic/OOD) and val_B (Threshold Sweep)
        # MetaLabeler/Isotonic calibrator ONLY evaluates on val_A (the first half) to prevent leakage.
        half = len(df_val) // 2
        df_val = df_val.iloc[:half].copy()
        
        logger.info(f"Validation (val_A split): {df_val.shape} | {df_val.index.min()} → {df_val.index.max()}")

        # [FIX-P2-TIMING] Calcular timing features in-line para calibrador (2026-03-26)
        # Garantiza que el XGBoost tenga las 30 features en validation, de lo contrario
        # predeciría 0 señales > 0.51, omitiendo la calibración del MetaLabeler.
        if "FundingRate" in df_val.columns:
            df_val["timing_funding_acum8h"] = df_val["FundingRate"].ewm(span=8, min_periods=1).mean()
        if "close" in df_val.columns:
            _r24h = df_val["close"].pct_change(24)
            _r7d  = df_val["close"].pct_change(168)
            df_val["timing_momentum_div"] = _r24h - _r7d
        if "close" in df_val.columns and "volume" in df_val.columns:
            _r24h_abs   = df_val["close"].pct_change(24).abs()
            _vol_ma     = df_val["volume"].rolling(window=720, min_periods=48).mean()
            _vol_ratio  = df_val["volume"] / (_vol_ma + 1e-6)
            df_val["timing_vol_divergence"] = (_r24h_abs / (_vol_ratio + 1e-6)).clip(upper=5.0)

        # [A2] Calcular btc_drawdown_from_ath en caso de que validation no lo posea (compatibilidad OOS)
        # P2-7-FIX (2026-03-30): prepend historia de train para evitar warm-up bias en primeras filas.
        # Antes: rolling 365d solo sobre validation → primeras filas tienen ATH≈precio actual → drawdown≈0.
        if "close" in df_val.columns:
            try:
                _features_dir = self.features_dir
                _train_close = pd.read_parquet(
                    _features_dir / "features_train.parquet",
                    columns=["close"]
                )["close"]
                _full_close_val = pd.concat([_train_close, df_val["close"]]).sort_index().drop_duplicates()
            except Exception as _e_dd:
                # P2-N5-FIX (2026-03-30): log explícito del fallback
                logger.warning(
                    f"  [P2-N5] Fallback: No se pudo cargar features_train.parquet para historia ATH ({_e_dd}). "
                    f"El rolling 90d ATH en validation tendra history vacia en las primeras barras."
                )
                _full_close_val = df_val["close"]  # fallback: solo validation

            _rolling_ath_full = _full_close_val.rolling(window=90*24, min_periods=720).max().clip(lower=1e-8)
            _drawdown_full = (_full_close_val / _rolling_ath_full) - 1.0
            # Alinear al índice de df_val (descartando las filas de train prepend)
            df_val["btc_drawdown_from_ath"] = _drawdown_full.reindex(df_val.index)

        # 4. Cargar HMM labels si el modelo fue entrenado con contexto HMM
        hmm_path = self.features_dir / "hmm_regime_labels.parquet"
        if hmm_ctx:
            if "HMM_Regime" not in df_val.columns and hmm_path.exists():
                df_hmm = pd.read_parquet(hmm_path)
                # Rellenar NaNs para evitar TypeError al convertir a int/float
                df_hmm["HMM_Regime"] = df_hmm["HMM_Regime"].ffill().bfill()
                # [FIX-CALIB-JOIN] Evitar ValueError si hay solapamiento de otras columnas (ej. HMM_Semantic)
                _overlap = [c for c in df_hmm.columns if c in df_val.columns]
                if _overlap:
                    logger.info(f"[FIX-CALIB-JOIN] Eliminando columnas solapadas antes del join: {_overlap}")
                    df_hmm = df_hmm.drop(columns=_overlap)
                df_val = df_val.join(df_hmm, how="left")
            elif "HMM_Regime" in df_val.columns:
                logger.info("HMM_Regime ya se encuentra en df_val (hidratado previamente). Saltando join de parquet.")

            
            # [FIX-HMM-CALIB-STATES]: En vez de `pd.get_dummies`, forzar estrictamente `n_states`
            # columnas para evitar mismatch si un estado no se observó en validation.
            # FIX-HMM-ALIGN-01 (2026-03-26): estandarizar nombres HMM_OH_x (sin floats) igual que en train y OOS.
            n_states_total = n_states + 1
            hmm_dummies = []
            for s in range(n_states_total):
                col = f"HMM_OH_{s}"
                df_val[col] = (df_val["HMM_Regime"].fillna(-1).astype(int) == s).astype(float)
                hmm_dummies.append(col)
                
            logger.info(f"HMM context generado estrictamente ({n_states_total} estados): {hmm_dummies}")
        elif hmm_ctx and not hmm_path.exists():
            logger.warning("Modelo entrenado con HMM context pero hmm_regime_labels.parquet no encontrado.")
            hmm_ctx = False

        # 5. Cargar XGBoost y su firma de features (DEBE ir antes de 5b)
        if use_regime:
            _all_regime_feats = []
            _optimal_threshold_per_regime = {}
            _global_thresh_list = []
            
            try:
                from config.settings import cfg as _cfg_xgb_sig
                _rm = vars(_cfg_xgb_sig.fase2.regime_mapping)
            except Exception:
                _rm = {
                    "bull":  ["1_BULL_TREND", "1_VOLATILE_BULL", "1_BULL_GRIND", "1_BULL_TREND_WEAK", "1_BULL_TREND_B", "1_VOLATILE_BULL_B"],
                    "range": ["2_CALM_RANGE", "2_VOLATILE_RANGE", "2_CALM_RANGE_B", "2_VOLATILE_RANGE_B"],
                    "bear":  ["3_CALM_BEAR", "3_BEAR_CRASH", "4_BEAR_FORCED"],
                }

            _sem_to_num = {}
            # [BUG-A3 / FIX-2A-CAL] Construir _sem_to_num desde HMM.state_map (PRIORIDAD).
            # El parquet hmm_regime_labels.parquet solo cubre hasta train_cutoff — exactamente
            # el mismo bug que Fix-2A corrigió en generate_oos_predictions_v2.py.
            # Sin este fix, los thresholds XGB per-régimen usan IDs numéricos incompletos
            # (solo los del training set), haciendo Fix-2A parcialmente inefectivo.
            try:
                from luna.models.hmm_regime import HMMRegimeModel as _HMM_CAL_2A
                _hmm_cal_model = _HMM_CAL_2A.load(self.models_dir)
                _sem_to_num = {str(v): int(k) for k, v in _hmm_cal_model.state_map.items()}
                logger.info("  [BUG-A3/CAL] _sem_to_num desde HMM.state_map: {} estados: {}",
                            len(_sem_to_num), list(_sem_to_num.keys()))
            except Exception as _e_a3:
                logger.warning("  [BUG-A3/CAL] Fallo cargando HMM.state_map, fallback a parquet: {}", _e_a3)
                # Fallback al comportamiento anterior (parquet incompleto)
                if hmm_ctx and hmm_path.exists():
                    try:
                        _hmm_df_map = pd.read_parquet(hmm_path, columns=["HMM_Regime", "HMM_Semantic"])
                        for _, _row in _hmm_df_map.drop_duplicates().iterrows():
                            _sem_to_num[str(_row["HMM_Semantic"])] = int(float(_row["HMM_Regime"]))
                    except Exception as _e_parq:
                        logger.debug("  [BUG-A3/CAL] Fallback parquet también falló: {}", _e_parq)

            for model_name, _sem_list in _rm.items():
                for direction in ["long", "short"]:
                    p = self.models_dir / f"xgboost_meta_{model_name}_{direction}_signature.json"
                    if not p.exists(): continue
                    with open(p, "r") as f:
                        sig = json.load(f)
                        _all_regime_feats.extend(sig["features"])
                        _thr = float(sig["optimal_threshold"])
                        _global_thresh_list.append(_thr)
                        # Asignamos el umbral al HMM_Regime, pero nota: Multi-Agent ahora separa long/short
                        # Calibrate probs hoy no distingue el umbral largo/corto, así que usaremos el mínimo o el promedio para el sweep global
                        for _s in _sem_list:
                            if _s in _sem_to_num:
                                # Overwrite or keep minimum to be conservative
                                key = str(_sem_to_num[_s])
                                if key in _optimal_threshold_per_regime:
                                    _optimal_threshold_per_regime[key] = min(_optimal_threshold_per_regime[key], _thr)
                                else:
                                    _optimal_threshold_per_regime[key] = _thr

            xgb_features_all = list(dict.fromkeys(_all_regime_feats))
            if not _global_thresh_list:
                raise RuntimeError("CRITICAL: _global_thresh_list vacío. No se encontraron firmas XGBoost válidas con optimal_threshold.")
            
            xgb_sig = {
                "features": xgb_features_all,
                "optimal_threshold": min(_global_thresh_list),
                "optimal_threshold_per_regime": _optimal_threshold_per_regime
            }
        else:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(xgb_model_path)
            with open(xgb_sig_path) as f:
                xgb_sig = json.load(f)
            xgb_features_all = xgb_sig["features"]  # todas las features del modelo (sin filtrar)

        # 5b. Añadir HMM_Regime al df_val si el XGBoost lo requiere (fue entrenado con él)
        if "HMM_Regime" not in df_val.columns or df_val["HMM_Regime"].isna().all() or "HMM_Semantic" not in df_val.columns:
            hmm_model_path_pkl = self.models_dir / "hmm_regime.pkl"
            if hmm_model_path_pkl.exists():
                try:
                    from luna.models.hmm_regime import HMMRegimeModel
                    hmm_predictor = HMMRegimeModel.load(self.models_dir)
                    _hmm_df = hmm_predictor.predict_regime_series(df_val)
                    df_val["HMM_Regime"] = _hmm_df["HMM_Regime"]
                    df_val["HMM_Semantic"] = _hmm_df["HMM_Semantic"]
                    logger.info(f"HMM_Regime/Semantic predicho en df_val: {df_val['HMM_Regime'].value_counts().to_dict()}")
                except Exception as e:
                    logger.warning(f"HMM predict fallido: {e}. Usando HMM_Regime=0.")
                    df_val["HMM_Regime"] = 0
                    df_val["HMM_Semantic"] = "UNKNOWN"
            else:
                logger.warning("HMM model no encontrado, usando HMM_Regime=0 como fallback.")
                df_val["HMM_Regime"] = 0
                df_val["HMM_Semantic"] = "UNKNOWN"

        # Convertir HMM_Regime a one-hot (HMM_OH_0..n_states) si el MetaLabeler usó HMM context
        if hmm_ctx and "HMM_Regime" in df_val.columns and not any(c.startswith("HMM_OH_") for c in df_val.columns):
            try:
                from config.settings import cfg
                n_states = int(cfg.hmm.n_states)
            except Exception as e:
                raise RuntimeError(f"Fallo leyendo cfg.hmm.n_states: {e}")
            n_states_total = n_states + 1  # Risk-Off Shield agrega estado n_states
            for s in range(n_states_total):
                df_val[f"HMM_OH_{s}"] = (pd.to_numeric(df_val["HMM_Regime"], errors='coerce').fillna(-1).astype(int) == s).astype(float)
            logger.info(f"HMM one-hot generado: {[f'HMM_OH_{s}' for s in range(n_states_total)]}")
            hmm_ctx = True  # ahora disponible


        # 5c. Determinar features XGBoost disponibles (tolerar missing con fillna(0))
        missing_xgb = [c for c in xgb_features_all if c not in df_val.columns]
        if missing_xgb:
            logger.warning(f"Features XGBoost no disponibles en val (se usará 0): {missing_xgb}")
            for mc in missing_xgb:
                df_val[mc] = 0
        xgb_features = xgb_features_all  # todas garantizadas en df_val


        # 6. Generar etiquetas TBM en validation
        # [FIX-P2-TBM-SYNC] (2026-05-03): sincronizar calibrador con TBM DINÁMICO del MetaLabeler.
        # BUG ORIGINAL: el calibrador usaba TBM estático (vertical_barrier_hours fijo) mientras
        # el MetaLabeler entrenaba con _apply_dynamic_tbm() basado en ATR (barreras 24-96H).
        # El mismo trade podía ser ganador en entrenamiento y perdedor en calibración (o viceversa)
        # si el mercado revertía entre el cierre dinámico (ATR) y el cierre fijo del calibrador.
        # FIX: el calibrador ahora usa apply_triple_barrier con dynamic_atr=True, idéntico al MetaLabeler.
        from luna.features.tbm import apply_triple_barrier
        try:
            from config.settings import cfg
            pt_mult     = float(cfg.xgboost.pt_mult_min)
            sl_mult     = float(cfg.xgboost.sl_mult_min)
            _tbm_min_return = float(cfg.xgboost.tbm_min_return)
            # [FIX-P2-TBM-SYNC] Usar rangos dinámicos ATR idénticos al MetaLabeler
            _vb_min_h   = int(cfg.xgboost.dynamic_horizon_min_h)
            _vb_max_h   = int(cfg.xgboost.dynamic_horizon_max_h)
            _use_dynamic_atr = True
            _lin_decay_c = bool(cfg.xgboost.linear_decay_pt)
            _pt_decay_frac_c = float(cfg.xgboost.pt_decay_fraction)
        except Exception as _e_tbm:
            _err_msg = f"CRITICAL: Fallo leyendo config TBM para Calibrador. Política No-Fallback: {_e_tbm}"
            logger.critical(_err_msg)
            raise RuntimeError(_err_msg) from _e_tbm
        logger.info(
            f"[FIX-P2-TBM-SYNC] Calibrador TBM: PT={pt_mult}x SL={sl_mult}x "
            f"dynamic_ATR=True barrier=[{_vb_min_h}H-{_vb_max_h}H] min_return={_tbm_min_return} "
            f"[Sincronizado con MetaLabeler _apply_dynamic_tbm()]"
        )

        _side_val = -1.0 if self.direction == "short" else 1.0
        _sides_series = pd.Series(_side_val, index=df_val.index)

        tbm = apply_triple_barrier(
            price_series=df_val["close"],
            event_times=df_val.index,
            sides=_sides_series,
            pt_sl_multiplier=[pt_mult, sl_mult],
            min_return=_tbm_min_return,
            vertical_barrier_hours=_vb_max_h,
            dynamic_barrier=True,
            dynamic_horizon_min_h=_vb_min_h,
            dynamic_horizon_max_h=_vb_max_h,
            linear_decay_pt=_lin_decay_c,
            pt_decay_fraction=_pt_decay_frac_c,
        )

        df_val = df_val.join(tbm[["bin", "ret"]], how="inner")
        df_val["target"] = (df_val["bin"] == 1).astype(int)
        df_val = df_val.dropna(subset=["target", "ret"])

        if len(df_val) < 50:
            logger.error(f"Validation set insuficiente tras TBM: {len(df_val)} muestras < 50")
            return

        y = df_val["target"].values

        # 7. Construir secuencias LSTM
        n_input = v2_config.get("input_dim")
        # FIX seq_features (2026-03-09): cargar lista exacta de features del LSTM
        # desde el config guardado en entrenamiento. Antes: seq_feat_candidates = xgb_features[:n_input]
        # (orden XGB alfabético) != features SFI del LSTM → embeddings inválidos.
        seq_features_saved = v2_config.get("seq_features", [])
        if seq_features_saved:
            # Pad con 0 features ausentes en validación (eg. mining rules no en val parquet)
            missing_seq = [f for f in seq_features_saved if f not in df_val.columns]
            if missing_seq:
                logger.warning(f"Calibrador: {len(missing_seq)} seq_features ausentes en val → pad 0: {missing_seq[:5]}")
                for f in missing_seq:
                    df_val[f] = 0.0
            seq_feat_candidates = seq_features_saved  # orden exacto del entrenamiento
            logger.info(f"Calibrador: usando seq_features del config ({len(seq_feat_candidates)} features)")
        else:
            # Fallback retrocompatible para configs antiguos sin seq_features
            seq_feat_candidates = [c for c in xgb_features if c in df_val.columns]
            if n_input and n_input != len(seq_feat_candidates):
                seq_feat_candidates = seq_feat_candidates[:n_input]
            logger.warning(f"Calibrador: seq_features no en config — fallback a xgb_features[:{len(seq_feat_candidates)}]")

        X_raw = df_val[seq_feat_candidates].fillna(0).values
        X_seq_list, seq_indices = [], []
        for i in range(seq_len, len(X_raw)):
            X_seq_list.append(X_raw[i - seq_len:i])
            seq_indices.append(i)

        if len(X_seq_list) < 50:
            logger.error(f"Secuencias insuficientes para calibrar: {len(X_seq_list)} < 50")
            return

        X_seq = np.array(X_seq_list)
        y_seq = y[seq_indices]

        # Ajustar input_dim al LSTM si hay mismatch
        if X_seq.shape[2] != v2_config.get("input_dim", X_seq.shape[2]):
            expected = v2_config["input_dim"]
            actual   = X_seq.shape[2]
            if actual > expected:
                X_seq = X_seq[:, :, :expected]
            elif actual < expected:
                pad = np.zeros((X_seq.shape[0], X_seq.shape[1], expected - actual))
                X_seq = np.concatenate([X_seq, pad], axis=2)
            logger.info(f"Calibrador: ajustado input_dim {actual} → {expected}")

        # 8. XGBoost probs en validation (OOS, no CPCV — ya es OOS real)
        # NOTA: df_val proviene de build_regime_validation (<= train_end). 
        # No se debe aplicar embargo relativo a train_end porque df_val es IS.
        _seq_indices_calib = seq_indices

        if use_regime:
            from luna.models.regime_router import RegimeRouter
            router_xgb = RegimeRouter(self.models_dir, agent_type="xgboost", direction=self.direction)
            _df_for_xgb = df_val.iloc[_seq_indices_calib].copy()
            xgb_probs = router_xgb.route_and_predict(_df_for_xgb)["raw"].values
        else:
            xgb_probs = xgb_model.predict_proba(df_val.iloc[_seq_indices_calib][xgb_features].fillna(0))[:, 1]
            
        # Actualizar y_seq y X_seq para alinear con los índices filtrados
        y_seq = y[_seq_indices_calib]
        X_seq = X_seq[[seq_indices.index(i) for i in _seq_indices_calib]]

        # 9. HMM context si aplica
        hmm_onehot = None
        if hmm_ctx:
            hmm_dummies = [c for c in df_val.columns if c.startswith("HMM_OH_")]
            if hmm_dummies:
                # LOGIC-CALIB-01 FIX (2026-04-06): usar _seq_indices_calib (post-embargo)
                # en lugar de seq_indices (todos). Tras el filtro de embargo, X_seq y y_seq
                # tienen len(_seq_indices_calib) filas. Usar seq_indices aquí produces shape
                # mismatch en np.hstack([embeddings, xgb_probs, hmm_onehot]).
                hmm_onehot = df_val[[c for c in hmm_dummies]].values[_seq_indices_calib].astype(float)
                logger.info(f"HMM context: {len(hmm_dummies)} columnas, {hmm_onehot.shape}")
            else:
                logger.warning("HMM context esperado pero no hay columnas HMM_OH_* en validation")

        # 10. Extraer embeddings LSTM y construir X_combined para el RF
        from luna.models.train_metalabeler_v2 import MetaLabelerV2
        embeddings = model.extractor.extract_embeddings(X_seq)
        parts = [embeddings, xgb_probs.reshape(-1, 1)]
        if hmm_onehot is not None:
            parts.append(hmm_onehot)
        X_combined = np.hstack(parts)
        logger.info(f"X_combined para calibración: {X_combined.shape} ({len(y_seq)} muestras)")

        # 11. Calibrar sobre raw probs del RF — calibrador INDEPENDIENTE
        # ARCH-FIX-CALIB-01: CalibratedClassifierCV(rf, cv=None) wrappea y
        # transforma el RF completo, produciendo un modelo distinto que no
        # generaliza bien a distribuciones OOS diferentes a validation.
        # Fix correcto: (a) RF nativo produce raw_probs, (b) calibrador separado
        # aprende solo la transformación raw_probs → proba_calibrada.
        # El RF original nunca se modifica ni re-entrena con la calibración.
        raw_probs_train = model.rf.predict_proba(X_combined)[:, 1]

        # [FIX-P1-CALIB-PASSTHROUGH] (2026-05-03): Desacoplar pool de calibración del threshold XGBoost.
        # BUG ORIGINAL: cuando XGBoost threshold=0.9 filtraba TODAS las señales de validation,
        # el calibrador entraba en 'skipped_insufficient_signals' y colapsaba TODAS las
        # probabilidades del RF a una constante (prob=mean, std=0.000). Con std=0, el MetaLabeler
        # no podía discriminar ninguna señal y el pipeline emitía 0 trades (silencio total).
        # FIX: cuando el pool XGBoost está vacío, calibrar sobre TODA la distribución RF (pass-through
        # isotónico), preservando la varianza discriminativa del MetaLabeler. El threshold XGBoost
        # sigue aplicándose al momento de DECISIÓN en generate_oos, no en calibración.
        _xgb_opt_t_global = float(xgb_sig["optimal_threshold"])
        _xgb_opt_t_regime = xgb_sig.get("optimal_threshold_per_regime", {})
        if "HMM_Regime" in df_val.columns:
            hmm_regimes_val = df_val["HMM_Regime"].iloc[_seq_indices_calib].fillna(-1).astype(int).values
            _xgb_thresh_array = np.full(len(xgb_probs), _xgb_opt_t_global)
            for r_str, r_thresh in _xgb_opt_t_regime.items():
                _xgb_thresh_array[hmm_regimes_val == int(r_str)] = r_thresh
            _xgb_mask_cal_pre = xgb_probs > _xgb_thresh_array
        else:
            _xgb_mask_cal_pre = xgb_probs > _xgb_opt_t_global
        _n_xgb_pass_pre = int(_xgb_mask_cal_pre.sum())

        if _n_xgb_pass_pre < 50:
            logger.warning(
                f"[FIX-P1-CALIB-PASSTHROUGH] Solo {_n_xgb_pass_pre} señales pasan XGBoost threshold={_xgb_opt_t_global:.3f}. "
                f"Pool de calibración vacío — usando isotonic pass-through sobre distribución RF completa ({len(raw_probs_train)} muestras). "
                f"Esto preserva la varianza discriminativa del MetaLabeler (evita colapso std=0)."
            )
            # [FIX-P1] Calibrar sobre TODA la distribución RF sin filtro XGBoost.
            # El threshold de decisión XGBoost se aplica DESPUÉS en generate_oos_predictions_v2.py.
            # [FIX-ISOTONIC-BLINDNESS-01] Cambiamos IsotonicRegression (Step-Function) por Platt (Regresión Logística).
            # Isotonic agrupa probabilidades extremas OOS en un solo escalón (1.000), destruyendo el 
            # Position Sizer. Platt usa una sigmoide continua que extrapola asimétricamente las colas.
            from sklearn.linear_model import LogisticRegression
            _adapter_p1 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            _adapter_p1.fit(raw_probs_train.reshape(-1, 1), y_seq)
            _cal_check_p1 = _adapter_p1.predict_proba(raw_probs_train.reshape(-1, 1))[:, 1]
            _p1_std = float(np.std(_cal_check_p1))
            if _p1_std < 1e-4:
                logger.warning(
                    f"[FIX-P1-CALIB-PASSTHROUGH] Platt pass-through también colapsó (std={_p1_std:.6f}). "
                    f"RF tiene distribución uniforme — señal insuficiente en esta ventana. Usando identity."
                )
                method = "skipped_insufficient_signals"
                _adapter = None
            else:
                logger.info(
                    f"[FIX-P1-CALIB-PASSTHROUGH] Platt pass-through exitoso: std={_p1_std:.4f} "
                    f"(pool_size={len(raw_probs_train)}). MetaLabeler mantiene varianza discriminativa."
                )
                method = "platt_full_passthrough"
                _adapter = _adapter_p1

        if method in ["isotonic", "platt"]:
            # [FIX-ISOTONIC-BLINDNESS-01] Forzamos Platt (LogisticRegression) incluso si se pidió isotonic
            from sklearn.linear_model import LogisticRegression
            _adapter = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            _adapter.fit(raw_probs_train.reshape(-1, 1), y_seq)
            _cal_check = _adapter.predict_proba(raw_probs_train.reshape(-1, 1))[:, 1]
            
            _brier_raw_train = float(np.mean((raw_probs_train - y_seq) ** 2))
            _brier_iso_train = float(np.mean((_cal_check - y_seq) ** 2))
            _improvement = (_brier_raw_train - _brier_iso_train) / max(_brier_raw_train, 1e-8) * 100
            
            if _improvement < 2.0 or np.std(_cal_check) < 1e-4:
                logger.warning(
                    f"[I2-FIX] Calibrador isotónico no mejora Brier > 2% o colapsa (imp={_improvement:.1f}%, std={np.std(_cal_check):.6f}). "
                    "Activando Temperature Scaling (T=0.5, sharpening)."
                )
                _ts_scaler = _TemperatureScaler(temperature=0.5)
                _ts_probs  = _ts_scaler.calibrate(raw_probs_train)
                _brier_ts  = float(np.mean((_ts_probs - y_seq) ** 2))
                
                if _brier_ts < _brier_raw_train:
                    logger.info(
                        "[I2-FIX] TS mejora Brier: {:.4f} -> {:.4f} (imp={:.1f}%). std raw_probs_ts={:.4f}. Usando TS.",
                        _brier_raw_train, _brier_ts,
                        (_brier_raw_train - _brier_ts) / max(_brier_raw_train, 1e-8) * 100,
                        float(np.std(_ts_probs))
                    )
                    _adapter = _TSAdapter(_ts_scaler)
                    method = "temperature_scaling_T05"
                else:
                    logger.warning("[I2-FIX] TS no mejora Brier. Fallback a LogisticRegression.")
                    from sklearn.linear_model import LogisticRegression
                    _adapter = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                    _adapter.fit(raw_probs_train.reshape(-1, 1), y_seq)
                    method = "platt_fallback"
                    
        elif method in ["sigmoid", "beta"]:
            from sklearn.linear_model import LogisticRegression
            _adapter = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            _adapter.fit(raw_probs_train.reshape(-1, 1), y_seq)

        logger.info(f"Calibrador separado (method={method}) entrenado sobre {len(y_seq)} raw_probs del RF.")

        # 12. Métricas: Brier Score pre/post calibración
        raw_probs = model.rf.predict_proba(X_combined)[:, 1]
        
        # Calcular cal_probs usando adapter separado (no un wrapper del RF completo)
        if method == "skipped_insufficient_signals":
            cal_probs = raw_probs
        elif hasattr(_adapter, 'predict_proba'):  # LogisticRegression
            cal_probs = _adapter.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        else:  # IsotonicRegression
            cal_probs = _adapter.predict(raw_probs)
            
        brier_raw = float(np.mean((raw_probs - y_seq) ** 2))
        brier_cal = float(np.mean((cal_probs - y_seq) ** 2))

        # ── Guards de calidad post-calibración ──
        check_numeric_stability(cal_probs, label="Calibrator.cal_probs")
        check_model_sanity(y_seq, cal_probs, label="Calibrator.post", threshold=0.5)

        if brier_cal > brier_raw:
            logger.warning(
                f"[CALIB] ⚠️  Calibración EMPEORÓ el Brier: {brier_raw:.4f} → {brier_cal:.4f} ({method}). "
                f"P6-FIX: se usará el RF sin calibrar (_IdentityWrapper) para no degradar la señal."
            )
            calibrated_rf = self._make_identity_calibrator(model.rf, X_combined, y_seq)
            brier_cal = brier_raw
            method = f"{method}_skipped_worse"
        else:
            # Crear wrapper pickleable RF + adapter separado
            calibrated_rf = _RFWithAdapter(model.rf, _adapter, method)

        logger.success(f"Calibración completada: Brier {brier_raw:.4f} → {brier_cal:.4f} ({method})")

        # 13. Persistir el calibrador en el modelo y re-guardar
        model.calibrator = calibrated_rf
        ruta_calibrator = self.models_dir / f"metalabeler_v2_{self.direction}_calibrator.joblib"

        # [BUG-CALIB-CACHE-01] GUARD ANTI-DEGENERACIÓN DEL METALABELER CALIBRADOR
        # Verificar que el calibrador produce salida con varianza antes de guardarlo.
        # Un calibrador degenerado (std≈0) destruiría todas las ventanas futuras del WFB.
        try:
            import numpy as _np_cal_guard
            _test_probs_guard = _np_cal_guard.linspace(0.1, 0.9, 80)
            if hasattr(calibrated_rf, 'predict_proba') and not hasattr(calibrated_rf, '_rf'):
                _guard_out = calibrated_rf.predict_proba(_test_probs_guard.reshape(-1, 1))[:, 1]
            elif hasattr(calibrated_rf, 'predict') and not hasattr(calibrated_rf, '_rf'):
                _guard_out = calibrated_rf.predict(_test_probs_guard)
            elif hasattr(calibrated_rf, '_adapter'):
                _adpt_g = calibrated_rf._adapter
                if hasattr(_adpt_g, 'predict_proba'):
                    _guard_out = _adpt_g.predict_proba(_test_probs_guard.reshape(-1, 1))[:, 1]
                elif hasattr(_adpt_g, 'predict'):
                    _guard_out = _adpt_g.predict(_test_probs_guard)
                else:
                    _guard_out = _test_probs_guard  # no testeable → asumir sano
            else:
                _guard_out = _test_probs_guard  # _IdentityWrapper u otro — asumir sano

            _guard_std = float(_np_cal_guard.std(_guard_out))
            _guard_type = type(calibrated_rf).__name__

            print(
                f"[BUG-CALIB-CACHE-01] SANITY CHECK MetaLabeler calibrador | "
                f"dir={self.direction} tipo={_guard_type} | "
                f"std_output={_guard_std:.6f} method={method}"
            )

            if _guard_std < 1e-4:
                logger.critical(
                    "[BUG-CALIB-CACHE-01] CALIBRADOR METALABELER DEGENERADO — std_output={:.8f}. "
                    "NO SE GUARDA para evitar contaminar caché WFB. dir={} method={}. "
                    "Causa probable: RF con muy pocos datos de calibración o distribución de señales plana.",
                    _guard_std, self.direction, method
                )
                print(
                    f"[BUG-CALIB-CACHE-01] *** CALIBRADOR METALABELER RECHAZADO *** | "
                    f"dir={self.direction} std={_guard_std:.8f} — NO GUARDADO. "
                    f"Forzar recalibración con más datos de validación."
                )
                return False  # NO guardar un calibrador degenerado
            else:
                logger.info(
                    "[BUG-CALIB-CACHE-01] Calibrador MetaLabeler SANO: std_output={:.4f} tipo={} dir={}",
                    _guard_std, _guard_type, self.direction
                )
                print(f"[BUG-CALIB-CACHE-01] MetaLabeler calibrador SANO: std={_guard_std:.4f} tipo={_guard_type}")
        except Exception as _e_guard_meta:
            logger.warning(
                "[BUG-CALIB-CACHE-01] No se pudo verificar sanidad del calibrador MetaLabeler: {}. "
                "Procediendo con guardado (riesgo asumido).", _e_guard_meta
            )
            print(f"[BUG-CALIB-CACHE-01] WARNING: No se pudo verificar calibrador MetaLabeler: {_e_guard_meta}")

        joblib.dump(calibrated_rf, ruta_calibrator)
        logger.success(f"Calibrador guardado: {ruta_calibrator}")


        # 14. Actualizar firma
        sig = json.loads(v2_config_path.read_text(encoding="utf-8"))
        sig["calibration_method"]  = method
        sig["brier_raw"]           = brier_raw
        sig["brier_calibrated"]    = brier_cal
        sig["calibrated"]          = True
        v2_config_path.write_text(json.dumps(sig, indent=2), encoding="utf-8")
        logger.info(f"metalabeler_v2_config.json actualizado con calibración.")

        # ── META-CAL-01: EV-sweep del threshold MetaLabeler ──────────────────
        # Objetivo: calibrar el threshold de decision del MetaLabeler automaticamente.
        # Analogia con LAB-CAL-01 del XGBoost: barre thresholds sobre cal_probs
        # filtradas por XGBoost (xgb_prob > optimal_threshold del XGBoost).
        # Poblacion de calibracion = senales que pasan XGBoost Y van al MetaLabeler.
        # OOS seguro: todo sobre validation set (nunca training ni holdout).
        optimal_meta_threshold = None
        try:
            # Parametros del sweep desde settings.yaml
            from config.settings import cfg as _cfg_meta_cal
            _mt_min    = float(_cfg_meta_cal.metalabeler.meta_sweep_min)
            _mt_max    = float(_cfg_meta_cal.metalabeler.meta_sweep_max)
            _mt_step   = float(_cfg_meta_cal.metalabeler.meta_sweep_step)
            _mt_min_tr = int(_cfg_meta_cal.metalabeler.meta_min_trades)
            _xgb_opt_t_global = float(xgb_sig["optimal_threshold"])
            _xgb_opt_t_regime = xgb_sig.get("optimal_threshold_per_regime", {})
            try:
                COST_PCT_META = float(_cfg_meta_cal.sop.cost_pct)
            except Exception as e_cost:
                raise RuntimeError(f"Falta cfg.sop.cost_pct en settings.yaml. Política No-Fallback (SOP R6): {e_cost}")

            # Mascara XGBoost adaptada para Idea I4 (Regímenes HMM)
            # Primero buscamos si HMM_Regime está disponible en validation (creado en L158-180)
            if "HMM_Regime" in df_val.columns:
                hmm_regimes_val = df_val["HMM_Regime"].iloc[_seq_indices_calib].fillna(-1).astype(int).values
                _xgb_thresh_array = np.full(len(xgb_probs), _xgb_opt_t_global)
                
                # Asignar threshold específico según el régimen
                for r_str, r_thresh in _xgb_opt_t_regime.items():
                    r_int = int(r_str)
                    _xgb_thresh_array[hmm_regimes_val == r_int] = r_thresh
                    
                _xgb_mask_cal = xgb_probs > _xgb_thresh_array
            else:
                _xgb_mask_cal = xgb_probs > _xgb_opt_t_global

            _n_xgb_pass   = int(_xgb_mask_cal.sum())
            logger.info(
                f"[META-CAL-01] {_n_xgb_pass} senales pasan XGBoost (umbral dinámico I4) en validation "
                f"— poblacion de calibracion MetaLabeler."
            )

            if _n_xgb_pass >= _mt_min_tr:
                _cal_probs_xgb = cal_probs[_xgb_mask_cal]  # meta probs filtradas por XGB
                _y_xgb         = y_seq[_xgb_mask_cal]       # etiquetas correspondientes

                # Retorno forward 1-paso para EV (mismo proxy que XGBoost calibrador)
                # Usar retorno real de la barrera TBM como proxy del retorno del trade
                # y_seq == 1 implica que el PT se tocó primero → usamos avg_return por clase
                _fwd_ret_win  = 0.0       # placeholder: retorno medio ganadores
                _fwd_ret_loss = 0.0       # placeholder: retorno medio perdedores

                # Para el EV usamos win/loss rate directo (mas estable con pocos datos)
                _best_meta_ev    = -np.inf
                _best_meta_score = -np.inf
                _best_meta_t     = float(_cfg_meta_cal.metalabeler.meta_filter_threshold)
                try:
                    _n_target_meta   = int(_cfg_meta_cal.stat.min_trades)
                except Exception as e_mt:
                    raise RuntimeError(f"Falta stat.min_trades en settings.yaml. Política No-Fallback: {e_mt}") from e_mt
                _meta_log        = []

                for _mt in np.arange(_mt_min, _mt_max + _mt_step / 2, _mt_step):
                    _mask_mt  = _cal_probs_xgb > _mt
                    _n_mt     = int(_mask_mt.sum())
                    if _n_mt < _mt_min_tr:
                        continue
                    _wins  = int(_y_xgb[_mask_mt].sum())
                    _wr    = _wins / _n_mt
                    _lr    = 1.0 - _wr
                    
                    # [LUNA V1 INSTITUTIONAL FIX] EV Matching OOS Strategy
                    # Previously it subtracted absolute cost (0.0015) from proxy units (1.0).
                    # Now it scales the proxy back to absolute returns before subtracting cost.
                    try:
                        from config.settings import cfg
                        _pt_m = float(cfg.xgboost.pt_mult_min)
                        _sl_m = float(cfg.xgboost.sl_mult_min)
                        _m_ret = float(cfg.xgboost.tbm_min_return)
                    except Exception as _e_cfg:
                        raise RuntimeError(f"Falta config TBM en settings. Política No-Fallback: {_e_cfg}") from _e_cfg
                        
                    _ev_proxy = _wr * _pt_m - _lr * _sl_m
                    _ev       = _ev_proxy * _m_ret - COST_PCT_META
                    
                    _vol_f = min(1.0, _n_mt / max(_n_target_meta, 1))
                    _score = _ev * _vol_f
                    _meta_log.append({"threshold": round(float(_mt), 3), "n": _n_mt,
                                       "wr": round(_wr, 4), "ev": round(_ev, 6)})
                    if _ev > _best_meta_ev and _score > _best_meta_score:
                        _best_meta_ev    = _ev
                        _best_meta_score = _score
                        _best_meta_t     = float(_mt)

                if _best_meta_ev > 0:
                    optimal_meta_threshold = _best_meta_t
                    logger.success(
                        f"[META-CAL-01] optimal_meta_threshold={optimal_meta_threshold:.2f} "
                        f"(EV={_best_meta_ev:.4f} sobre {_n_xgb_pass} senales XGB-filtradas)"
                    )
                else:
                    logger.warning(
                        f"[META-CAL-01] EV-sweep produjo EV negativo o invalido (EV={_best_meta_ev:.4f}). "
                        "Para evitar Starvation Overfitting de validation, se usara fallback manual."
                    )
                
                # ── Calibración por Régimen HMM (Idea I4 / Fase 7) ──
                optimal_meta_threshold_per_regime = {}
                if "HMM_Regime" in df_val.columns:
                    logger.info("[META-CAL-01/I4] Ejecutando calibración MetaLabeler iterativa por régimen HMM...")
                    hmm_regimes_val = df_val["HMM_Regime"].iloc[_seq_indices_calib].fillna(-1).astype(int).values
                    _hmm_val_xgb = hmm_regimes_val[_xgb_mask_cal]
                    
                    for r in np.unique(_hmm_val_xgb):
                        if r < 0: continue
                        
                        r_mask = _hmm_val_xgb == r
                        r_probs = _cal_probs_xgb[r_mask]
                        r_y = _y_xgb[r_mask]
                        r_n = len(r_probs)
                        
                        # [BUG-LGBM-CAL-01 FIX análogo] adaptativo min_trades para regímenes
                        r_min_trades = max(3, int(r_n * 0.30))
                        r_min_trades = min(r_min_trades, max(10, int(_mt_min_tr * 0.25)))

                        if r_n < r_min_trades:
                            logger.debug(f"  [Regimen {r}] Ignorado por tamaño muestral ({r_n} < {r_min_trades})")
                            continue
                            
                        r_best_t = None
                        r_best_ev = -np.inf
                        r_best_score = -np.inf
                        
                        for _mt in np.arange(_mt_min, _mt_max + _mt_step / 2, _mt_step):
                            _m_mt = r_probs > _mt
                            _n_mt = int(_m_mt.sum())
                            if _n_mt < r_min_trades: continue
                            
                            _wins = int(r_y[_m_mt].sum())
                            _wr = _wins / _n_mt
                            _lr = 1.0 - _wr
                            
                            _ev_proxy = _wr * _pt_m - _lr * _sl_m
                            _ev = _ev_proxy * _m_ret - COST_PCT_META
                            
                            # Mismo scoring que global
                            _vol_f = min(1.0, _n_mt / max(_n_target_meta//4, 1))
                            _score = _ev * _vol_f
                            
                            if _ev > r_best_ev and _score > r_best_score:
                                r_best_ev = _ev
                                r_best_score = _score
                                r_best_t = float(_mt)
                                
                        if r_best_ev > 0:
                            optimal_meta_threshold_per_regime[str(r)] = r_best_t
                            logger.info(f"  [Regimen {r}] Threshold={r_best_t:.2f} (EV={r_best_ev:.4f}, n={r_n}) calibrado")
                        else:
                            logger.debug(f"  [Regimen {r}] EV={r_best_ev:.4f} <= 0. Fallback global para evitar Starvation.")

            else:
                # [BUG-FIX-05] Modo reducido: intentar EV-sweep con al menos 5 señales
                # (antes el mínimo era meta_min_trades=20, que sistémicamente bloqueaba
                # el sweep en periodos de crash donde el XGBoost pasa pocas señales)
                if _n_xgb_pass >= 5:
                    logger.warning(
                        "[META-CAL-01] Solo {} señales pasan XGBoost (<min={}). "
                        "[BUG-FIX-05] Intentando EV-sweep en modo reducido (min_trades=3).",
                        _n_xgb_pass, _mt_min_tr
                    )
                    print(f"[BUG-FIX-LOG 2026-06-05] [META-CAL-01] Solo {_n_xgb_pass} señales pasan XGBoost (<min={_mt_min_tr}). Intentando modo reducido.")
                    _cal_probs_xgb = cal_probs[_xgb_mask_cal]
                    _y_xgb = y_seq[_xgb_mask_cal]
                    _best_meta_ev = -np.inf
                    _best_meta_score = -np.inf
                    _best_meta_t = float(_cfg_meta_cal.metalabeler.meta_filter_threshold)
                    _n_target_meta = max(5, _n_xgb_pass)
                    for _mt in np.arange(_mt_min, _mt_max + _mt_step / 2, _mt_step):
                        _mask_mt = _cal_probs_xgb > _mt
                        _n_mt = int(_mask_mt.sum())
                        if _n_mt < 3: continue
                        _wins = int(_y_xgb[_mask_mt].sum())
                        _wr = _wins / _n_mt
                        _lr = 1.0 - _wr
                        try:
                            from config.settings import cfg
                            _pt_m = float(cfg.xgboost.pt_mult_min)
                            _sl_m = float(cfg.xgboost.sl_mult_min)
                            _m_ret = float(cfg.xgboost.tbm_min_return)
                        except Exception as _e_cfg:
                            raise RuntimeError(f"Falta config TBM en settings. Política No-Fallback: {_e_cfg}") from _e_cfg
                        _ev_proxy = _wr * _pt_m - _lr * _sl_m
                        _ev = _ev_proxy * _m_ret - COST_PCT_META
                        _vol_f = min(1.0, _n_mt / max(_n_target_meta, 1))
                        _score = _ev * _vol_f
                        if _ev > _best_meta_ev and _score > _best_meta_score:
                            _best_meta_ev = _ev
                            _best_meta_score = _score
                            _best_meta_t = float(_mt)
                    if _best_meta_ev > 0:
                        optimal_meta_threshold = _best_meta_t
                        logger.success(
                            "[META-CAL-01] [MODO-REDUCIDO] optimal_meta_threshold={:.2f} (EV={:.4f}, n={})",
                            optimal_meta_threshold, _best_meta_ev, _n_xgb_pass
                        )
                        print(f"[BUG-FIX-LOG 2026-06-05] [META-CAL-01] [MODO-REDUCIDO] optimal_meta_threshold={optimal_meta_threshold:.2f} (EV={_best_meta_ev:.4f}, n={_n_xgb_pass})")
                    else:
                        logger.warning(
                            "[META-CAL-01] EV-sweep modo reducido: EV={:.4f} <= 0. Usando threshold manual.",
                            _best_meta_ev
                        )
                        print(f"[BUG-FIX-LOG 2026-06-05] [META-CAL-01] EV-sweep modo reducido: EV={_best_meta_ev:.4f} <= 0. Usando threshold manual.")
                else:
                    logger.warning(
                        "[META-CAL-01] Solo {} señales pasan XGBoost (<min={}, <5). "
                        "EV-sweep MetaLabeler omitido — usando threshold manual.",
                        _n_xgb_pass, _mt_min_tr
                    )
                    print(f"[BUG-FIX-LOG 2026-06-05] [META-CAL-01] Solo {_n_xgb_pass} señales pasan XGBoost (<min={_mt_min_tr}, <5). Usando threshold manual.")
        except Exception as _e_meta_cal:
            logger.warning(f"[META-CAL-01] EV-sweep MetaLabeler fallido: {_e_meta_cal}")
            import traceback; logger.debug(traceback.format_exc())

        # 15. Persistir calibrador + firma
        calib_sig_path = self.models_dir / f"calibrator_{self.direction}_signature.json"
        _meta_thresh_final = optimal_meta_threshold
        if _meta_thresh_final is None:
            try:
                from config.settings import cfg as _cfg_mt_fb
                _meta_thresh_final = float(_cfg_mt_fb.metalabeler.meta_filter_threshold)
            except Exception:
                _meta_thresh_final = 0.40
            logger.info(
                "[MEJ-CALIB-01] EV-sweep no produjo threshold — usando manual de settings: {:.2f}",
                _meta_thresh_final
            )
            print(f"[BUG-FIX-LOG 2026-06-05] [MEJ-CALIB-01] EV-sweep no produjo threshold — usando manual: {_meta_thresh_final:.2f}")

        # [P2-META-01] dynamic_is aplicado al calibrador (mismo que train_metalabeler_v2.py)
        # Cuando el EV-sweep no encuentra un threshold positivo y cae al fallback manual,
        # usamos la base rate de cal_probs como proxy IS para calcular un floor más informado.
        try:
            from config.settings import cfg as _cfg_dynmeta_cal
            _thresh_mode_cal   = str(int(_cfg_dynmeta_cal.metalabeler.meta_v2_threshold_mode)).lower()
            _edge_pct_cal      = float(_cfg_dynmeta_cal.metalabeler.meta_v2_dynamic_edge_pct)
            _floor_abs_cal     = float(_cfg_dynmeta_cal.metalabeler.meta_v2_min_prob)
        except Exception:
            _thresh_mode_cal = 'fixed'
            _edge_pct_cal, _floor_abs_cal = 0.05, 0.38

        if _thresh_mode_cal == 'dynamic_is':
            # Proxy IS: base rate media de las probabilidades calibradas del RF (sin sesgo OOS)
            _is_base_rate_cal = float(np.mean(y_seq)) if len(y_seq) > 0 else 0.50
            _dyn_floor_cal = max(_floor_abs_cal, _is_base_rate_cal * (1.0 + _edge_pct_cal))
            _dyn_floor_cal = min(_dyn_floor_cal, 0.65)  # cap absoluto

            # [P2-META-01 SAFETY-CAP] El calibrador isotónico puede comprimir todas las probs
            # por debajo de 0.50 (isotonic_full_passthrough con señales escasas).
            # Si el floor IS supera el percentil 90 de cal_probs, el threshold es inalcanzable.
            # Solución: el floor nunca puede superar p90(cal_probs) × 0.95 (margen del 5%).
            try:
                _p90_cal_probs = float(np.percentile(cal_probs, 90))
                _max_cal_probs = float(np.max(cal_probs))
                _reachable_cap = _p90_cal_probs * 0.95

                print(
                    f"[FIX-META-THRESHOLD-01] DIAGNÓSTICO: p90={_p90_cal_probs:.3f} "
                    f"max={_max_cal_probs:.3f} floor_abs={_floor_abs_cal:.3f} "
                    f"dyn_floor={_dyn_floor_cal:.3f} reachable_cap={_reachable_cap:.3f}"
                )  # debug
                logger.info(
                    "[FIX-META-THRESHOLD-01] DIAGNÓSTICO: p90={:.3f} max={:.3f} "
                    "floor_abs={:.3f} dyn_floor={:.3f} reachable_cap={:.3f}",
                    _p90_cal_probs, _max_cal_probs, _floor_abs_cal, _dyn_floor_cal, _reachable_cap
                )

                if _dyn_floor_cal > _reachable_cap and _reachable_cap > _floor_abs_cal:
                    # floor IS supera el p90: bajarlo al cap alcanzable
                    logger.info(
                        "[FIX-META-THRESHOLD-01] Safety-cap normal: IS-floor={:.3f} > p90*0.95={:.3f}. "
                        "Bajando floor al cap alcanzable (p90={:.3f}).",
                        _dyn_floor_cal, _reachable_cap, _p90_cal_probs,
                    )
                    _dyn_floor_cal = _reachable_cap
                    print(f"[FIX-META-THRESHOLD-01] floor bajado a {_reachable_cap:.3f} (p90*0.95)")  # debug

                elif _reachable_cap <= _floor_abs_cal:
                    # [FIX-META-THRESHOLD-01] BUG ORIGINAL: usaba _floor_abs_cal (inalcanzable)
                    # → producía CERO señales cuando max_prob < floor_abs.
                    # CORRECCIÓN: usar p90*0.90 como mejor threshold alcanzable.
                    # Si ni siquiera p90*0.90 produce señales, es una ventana sin alpha real.
                    _emergency_thresh = _p90_cal_probs * 0.90
                    logger.warning(
                        "[FIX-META-THRESHOLD-01] RANGO COMPRIMIDO DETECTADO: "
                        "p90(cal_probs)*0.95={:.3f} <= floor_abs={:.3f} — "
                        "max_prob={:.3f}. BUG ORIGINAL usaba floor_abs (INALCANZABLE). "
                        "CORRECCIÓN: usando p90*0.90={:.3f} como threshold de emergencia. "
                        "Si max_prob<{:.3f}, esta ventana no tendrá señales (alpha insuficiente).",
                        _reachable_cap, _floor_abs_cal, _max_cal_probs,
                        _emergency_thresh, _emergency_thresh,
                    )
                    print(
                        f"[FIX-META-THRESHOLD-01] RANGO COMPRIMIDO: floor_abs={_floor_abs_cal:.3f} "
                        f"INALCANZABLE (max={_max_cal_probs:.3f}). "
                        f"USANDO emergency_thresh={_emergency_thresh:.3f} (p90*0.90)"
                    )  # debug
                    _dyn_floor_cal = _emergency_thresh

            except Exception as _e_cap:
                logger.debug("[FIX-META-THRESHOLD-01] Safety-cap skipped: {}", _e_cap)
                print(f"[FIX-META-THRESHOLD-01] Safety-cap exception: {_e_cap}")  # debug

            if _meta_thresh_final < _dyn_floor_cal:
                logger.info(
                    "[P2-META-01/CAL] dynamic_is: thresh_fallback={:.3f} < IS-floor={:.3f} "
                    "(IS_base_rate={:.3f}, edge={:.0%}). Subiendo al floor.",
                    _meta_thresh_final, _dyn_floor_cal, _is_base_rate_cal, _edge_pct_cal,
                )
                _meta_thresh_final = _dyn_floor_cal
            else:
                logger.info(
                    "[P2-META-01/CAL] dynamic_is: thresh={:.3f} >= IS-floor={:.3f}. OK.",
                    _meta_thresh_final, _dyn_floor_cal,
                )

        # Fallback local var si la excepción ocurrió antes
        if 'optimal_meta_threshold_per_regime' not in locals():
            optimal_meta_threshold_per_regime = {}

        _calib_sig_data = {
            "calibration_method":       method,
            "brier_score_raw":          brier_raw,
            "brier_score_calibrated":   brier_cal,
            "mejora_pct":               float((brier_raw - brier_cal) / max(brier_raw, 1e-8) * 100),
            "model":                    "MetaLabelerV2_RF",
            "n_samples_val":            len(y_seq),
            "optimal_meta_threshold":   _meta_thresh_final,
            "optimal_meta_threshold_per_regime": optimal_meta_threshold_per_regime
        }
        logger.info("[MEJ-CALIB-01] optimal_meta_threshold={:.2f} guardado en calibrator_signature.json",
                    _meta_thresh_final)

        calib_sig_path.write_text(json.dumps(_calib_sig_data, indent=4), encoding="utf-8")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibración probabilidades Luna V1 (MetaLabelerV2)")
    parser.add_argument(
        "--method", choices=["isotonic", "sigmoid", "platt", "beta"], default="isotonic",
        help="Método de calibración (default: isotonic)"
    )
    parser.add_argument(
        "--direction", choices=["long", "short"], default="long",
        help="Dirección a calibrar (long o short)"
    )
    args = parser.parse_args()

    import os as _os
    from datetime import datetime as _dt
    _log_dir = ROOT / "logs"
    _log_dir.mkdir(exist_ok=True)
    _ts_cb  = _dt.now().strftime("%Y%m%d_%H%M%S")
    _rid_cb = _os.environ.get("LUNA_RUN_ID", "")
    _lname_cb = f"calibrate_probabilities_{_ts_cb}_{_rid_cb}.log" if _rid_cb else f"calibrate_probabilities_{_ts_cb}.log"
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add(_log_dir / _lname_cb, rotation="50 MB", level="DEBUG", encoding="utf-8")

    method = "sigmoid" if args.method in ["platt", "beta"] else args.method
    if args.method in ["platt", "beta"]:
        logger.info(f"Método '{args.method}' mapeado a 'sigmoid' (sklearn CalibratedClassifierCV).")

    calib = MetaLabelerV2Calibrator(direction=args.direction)
    
    if 'LUNA_WFB_WINDOW_DIR' in _os.environ:
        from pathlib import Path
        calib.models_dir = Path(_os.environ['LUNA_WFB_WINDOW_DIR']) / "models"
        logger.info(f"Usando WFB Window Dir para calibracion: {calib.models_dir}")

    success = calib.calibrate(method=method)
    if success is False:
        sys.exit(1)
    sys.exit(0)
