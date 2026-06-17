import sys
import os
import time
import json
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger
from luna.utils.debug_guards import check_numeric_stability, vlog, timeit, check_invariant

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Fetchers y Features
from luna.data.data_collector import DataCollector
from luna.features.feature_pipeline import FeaturePipeline

# Modelos — P4-0-5 FIX (Auditoría CODE-02):
# MetaLabelerV2 (LSTM-32 extractor + RF-300 árbitro) es el modelo que
# produce el training. BiLSTMv1 fue el modelo anterior (deprecado).
from luna.models.train_metalabeler_v2 import MetaLabelerV2      # P4-0-5: V2 activo
# XGBoostTrainer no se instancia en inferencia live — el modelo se carga con xgb.load_model()
# Referencia de entrenamiento: core.models.train_xgboost_v2

# HMM Model
try:
    from luna.models.hmm_regime import HMMRegimeModel
except ImportError:
    class HMMRegimeModel: pass

class LunaLiveInference:
    """
    Cerebro predictivo Luna V1 en producción (Pipeline Continua).
    Responsable de:
     1. Obtener la última data de los fetchers.
     2. Validar que la Macro Data no esté crónicamente atascada (Stale Check 14d).
     3. Pasar la data por el Master Feature Pipeline.
     4. Predecir Estado HMM → Predecir Dirección XGBoost → Filtrar con MetaLabelerV2.
     5. Calibración integrada en MetaLabelerV2 (RF con Platt Scaling).
    """

    def __init__(self):
        # Fix A-03: los modelos se guardan en data/models/, no en models/
        self.models_dir = PROJECT_ROOT / "data" / "models"
        self.data_dir = PROJECT_ROOT / "data"

        print("[Inferencia] Cargando artefactos de entrenamiento...")

        # 1. HMM
        self.hmm_model = self._load_model("hmm_pipeline.pkl")

        # 2. XGBoost — Fix A-03: usar xgb.load_model(.model), no joblib.load(.pkl)
        self.xgb_model = self._load_xgb("xgboost_meta.model")

        # 3. MetaLabelerV2 (LSTM-32 + RF-300) - P4-0-5 FIX (era BiLSTMv1 incorrecto)
        self.metalabeler_v2_long = self._load_metalabeler_v2("long")
        self.metalabeler_v2_short = self._load_metalabeler_v2("short")

        # 4. Calibrador legacy (Platt externo) — solo si MetaLabelerV2 no está disponible
        self.calibrator = self._load_model("platt_scaler.pkl")

        # 5. Features seleccionadas
        self.selected_features = self._load_selected_features()
        
    def _load_model(self, filename: str):
        path = self.models_dir / filename
        if path.exists():
            return joblib.load(path)
        logger.warning(f"[Live] Missing artifact: {filename}")
        return None

    def _load_xgb(self, filename: str):
        """Fix A-03: carga XGBoost con su formato nativo (.model), no joblib."""
        import xgboost as xgb
        path = self.models_dir / filename
        if not path.exists():
            logger.warning(f"[Live] Missing XGBoost model: {filename}")
            return None
        try:
            model = xgb.XGBClassifier()
            model.load_model(str(path))
            logger.info(f"[Live] XGBoost cargado desde {filename}")
            return model
        except Exception as e:
            logger.error(f"[Live] XGBoost load_model error: {e}")
            return None

    def _load_metalabeler_v2(self, direction_mode: str) -> MetaLabelerV2 | None:
        """P4-0-5: Carga MetaLabelerV2 (LSTM-32 + RF-300) usando su API nativa.
        Reemplaza _load_bilstm que cargaba BiLSTMv1 (modelo diferente al que entrena el pipeline).
        """
        model_dir = self.models_dir
        config_path = model_dir / f"metalabeler_v2_{direction_mode}_config.json"
        lstm_path = model_dir / f"metalabeler_v2_{direction_mode}_lstm.pt"
        rf_path = model_dir / f"metalabeler_v2_{direction_mode}_rf.joblib"

        if not all(p.exists() for p in [config_path, lstm_path, rf_path]):
            missing = [p.name for p in [config_path, lstm_path, rf_path] if not p.exists()]
            logger.warning(f"[Live] MetaLabelerV2 ({direction_mode}) artefactos faltantes: {missing}. Usando HOLD por defecto.")
            return None
        try:
            model = MetaLabelerV2.load(model_dir, direction_mode=direction_mode)
            logger.info(f"[Live] MetaLabelerV2 ({direction_mode}) cargado (LSTM-32 + RF-300) desde {model_dir}")
            return model
        except Exception as e:
            logger.error(f"[Live] MetaLabelerV2 ({direction_mode}) load error: {e}")
            return None

    def _load_selected_features(self) -> list:
        feat_path = self.data_dir / "features" / "selected_features.json"
        if feat_path.exists():
            with open(feat_path, 'r') as f:
                data = json.load(f)
                return data.get("selected_features", [])
        return []

    def _load_bilstm_signature(self, direction_mode: str = "long") -> dict:
        """Carga la firma del BiLSTM para conocer seq_feature_names, tribe_dim, n_regime_dim."""
        sig_path = self.models_dir / f"metalabeler_{direction_mode}_signature.json"
        if sig_path.exists():
            with open(sig_path, 'r') as f:
                return json.load(f)
        return {}

    def _stale_data_watchdog(self, raw_macro_df: pd.DataFrame) -> tuple[bool, str]:
        """
        [DEFENSA INSTITUCIONAL - Luna v2 Fase 15]
        Evita que predecamos un Theso muerto debido a APIs rotas o feriados extendidos the la FED (FRED).
        Si la Thea MVRV, o la Macro esta congelada por mas the 14 dias, VETA todo operador.
        """
        if raw_macro_df is None or raw_macro_df.empty:
            return True, "No se theMacro Data. Asumiendo Stale Data Critical Failure."
            
        last_date = raw_macro_df.dropna(how='all').index.max()
        if pd.isna(last_date):
            return True, "Last macro date is NaT."
            
        # Asumiendo `last_date` The tz-aware
        now = pd.Timestamp.now('UTC')  # [FIX-PIPE-002] utcnow() deprecated en Pandas 2.x
        if last_date.tz is None:
            last_date = last_date.tz_localize('UTC')
        else:
             last_date = last_date.tz_convert('UTC')
             
        age_days = (now - last_date).days
        
        if age_days > 14:
            return True, f"STALE DATA KILL-SWITCH: Data theal con {age_days} Theas the retraso (>14 Theas permitidos)."
            
        return False, f"Data fresca (Edad: {age_days} Theas)"

    def predict_cycle(self) -> dict:
        """
        Flujo maestro the theiccion 24/7. Responde con un Thte estandar unificado.
        """
        print(f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] 🧠 Iniciando Inferencia V1 LUNA...")
        
        # 1. Fetchers
        print("  -> Recolectando Datos Frescos (APIs Externas)...")
        try:
            DataCollector().build(mode='incremental')  # Fix: era collect_all_data() inexistente
        except Exception as e:
            print(f"  [!] Fallo de Red en Fetchers, usando datos en cache: {e}")
            
        # 2. Stale Check
        macro_path = self.data_dir / "raw" / "macro" / "macro_features.parquet"
        if macro_path.exists():
            macro_raw = pd.read_parquet(macro_path)
            is_stale, stale_msg = self._stale_data_watchdog(macro_raw)
            if is_stale:
                 print(f"  [!] {stale_msg}")
                 return {"action": "HOLD", "confidence": 0.0, "reason": stale_msg, "xgb_prob": 0.0, "regime": 1}
        
        # 3. Pipeline de Transformacion
        print("  -> Ejecutando Pipeline de Ingenieria de Features...")
        try:
            # BUG-LIVE-02 FIX: skip_fracdiff debe ser False. El modelo XGBoost/MetaLabeler 
            # asume la existencia de features fraccionalmente diferenciadas (ej. close_fd).
            # BUG-LIVE-01 FIX: live_mode=True para evitar el truncation temporal del training_end.
            result = FeaturePipeline().run(skip_fracdiff=False, skip_sfi=True, live_mode=True)
            df = result.get('live', None)
            if df is not None and hasattr(df, 'sort_index'):
                df = df.sort_index()
                # Save live features to disk for dashboard telemetry
                try:
                    out_dir = PROJECT_ROOT / "data" / "features"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(out_dir / "features_live.parquet")
                    print(f"🌙 [LIVE-INFERENCE-SAVE] [MEJORA-FEAT-TELEMETRY] Features en vivo guardadas exitosamente en: {out_dir / 'features_live.parquet'} | Filas: {len(df)} | Fecha Máxima: {df.index.max()}")
                except Exception as save_err:
                    print(f"⚠️ [LIVE-INFERENCE-WARN] No se pudo guardar features_live.parquet: {save_err}")
        except Exception as e:
            msg = f"Feature Pipeline error: {e}"
            print(f"  [!] {msg}")
            return {"action": "HOLD", "confidence": 0.0, "reason": msg, "xgb_prob": 0.0, "regime": 1}
            
        if df is None or df.empty:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Pipeline df vacio.", "xgb_prob": 0.0, "regime": 1}

        # 4. Inferencia de Regime (HMM)
        regime = 1  # Neutral Default
        print("  -> Proyeccion HMM...")
        if self.hmm_model:
            try:
                # Fix F15: activar predicción HMM real (antes estaba mockeada).
                # La interfaz HMMRegimeModel.predict espera el último vector de features.
                scaler = self.hmm_model.get('scaler') if isinstance(self.hmm_model, dict) else getattr(self.hmm_model, 'scaler', None)
                model  = self.hmm_model.get('model')  if isinstance(self.hmm_model, dict) else getattr(self.hmm_model, 'model', self.hmm_model)
                feats  = self.hmm_model.get('features', self.selected_features) if isinstance(self.hmm_model, dict) else self.selected_features
                if model is not None and feats:
                    hmm_input = df[[f for f in feats if f in df.columns]].iloc[-1:].values
                    if scaler is not None:
                        hmm_input = scaler.transform(hmm_input)
                    regime = int(model.predict(hmm_input)[0])
                    print(f"     [HMM] Regime: {regime}")
            except Exception as e:
                print(f"  [!] HMM falló: {e} — usando regime=1 (neutral)")
                
        # 5. Inferencia Direccional (XGBoost)
        xgb_prob = 0.5
        features_x = df[self.selected_features].iloc[-1:] if self.selected_features else df.iloc[-1:]
        
        print("  -> Inferencia The XGBoost...")
        if self.xgb_model:
            try:
                # Seleccion the columnas theactas Therenamiento
                xgb_cols = self.xgb_model.feature_names_in_
                x_input = df[xgb_cols].iloc[-1:]
                xgb_prob = self.xgb_model.predict_proba(x_input)[0][1]
            except Exception as e:
                 print(f"  [!] Theoost thelló: {e}")
                 
        direction = "LONG" if xgb_prob >= 0.5 else "SHORT"
        check_numeric_stability(np.array([xgb_prob]), label="Live.xgb_prob")
        logger.info(f"[Live] XGB: {direction} | prob={xgb_prob:.4f}")

        # 6. MetaLabelerV2 — LSTM-32 + RF-300 (P4-0-5)
        meta_prob_raw = 1.0  # Default si no hay MetaLabelerV2 (pass-through)
        seq_len_cfg = 48     # SEQ_LEN en train_metalabeler_v2.py (alineado con settings.yaml)
        
        active_metalabeler = self.metalabeler_v2_long if direction == "LONG" else self.metalabeler_v2_short
        direction_lower = direction.lower()
        
        if active_metalabeler and len(df) >= seq_len_cfg:
            print(f"  -> MetaLabelerV2 ({direction}) filtrando señal XGBoost...")
            try:
                sig = self._load_bilstm_signature(direction_mode=direction_lower)
                seq_feat_names = sig.get("seq_feature_names", self.selected_features)
                try:
                    from config.settings import cfg
                    n_states = int(cfg.hmm.n_states)
                except Exception as e:
                    raise RuntimeError(f"Fallo leyendo cfg.hmm.n_states: {e}")
                use_hmm_context = sig.get("hmm_context", False)

                # Secuencias temporales (N=1, seq_len, n_features)
                avail_seq = [f for f in seq_feat_names if f in df.columns]
                seq_arr = df[avail_seq].iloc[-seq_len_cfg:].values.astype(np.float32)
                if seq_arr.shape[1] < len(seq_feat_names):
                    pad = np.zeros((seq_len_cfg, len(seq_feat_names) - seq_arr.shape[1]), dtype=np.float32)
                    seq_arr = np.concatenate([seq_arr, pad], axis=1)
                X_seq = seq_arr[np.newaxis, ...]   # (1, seq_len, n_feat)

                # XGBoost prob como array (N=1,)
                xgb_arr = np.array([xgb_prob], dtype=np.float32)

                # HMM regime one-hot (P4-0-2 context)
                hmm_arr = None
                if use_hmm_context:
                    regime_oh = np.zeros((1, n_states), dtype=np.float32)
                    regime_oh[0, min(regime, n_states - 1)] = 1.0
                    hmm_arr = regime_oh

                meta_prob_raw = float(active_metalabeler.predict_proba(
                    X_seq, xgb_arr, hmm_regime=hmm_arr
                )[0])
                check_numeric_stability(np.array([meta_prob_raw]), label="Live.meta_prob")
                check_invariant(0 <= meta_prob_raw <= 1, f"meta_prob fuera de [0,1]: {meta_prob_raw}")
                logger.info(f"[Live] MetaLabelerV2 ({direction}) meta_prob={meta_prob_raw:.4f}")
            except Exception as e:
                logger.error(f"[Live] MetaLabelerV2 ({direction}) fallo: {e}")
                meta_prob_raw = 0.0  # HOLD por seguridad ante error

        # 7. Calibrador (integrado en MetaLabelerV2 durante training)
        # Si MetaLabelerV2 no está disponible, caemos al Platt Scaler legacy
        final_confidence = meta_prob_raw
        if active_metalabeler is None and self.calibrator:
            try:
                final_confidence = self.calibrator.predict_proba([[meta_prob_raw]])[0][1]
            except Exception as e:
                final_confidence = 1.0 / (1.0 + np.exp(-3.0 * meta_prob_raw + 1.5))

        print(f"     [MetaLabelerV2] meta_prob={meta_prob_raw:.4%} | final={final_confidence:.4%}")

        # Fix F17: calcular historical_vol con ventana real, no como multiplicador fijo.
        # Antes: historical_vol = current_vol * 1.5 (siempre ratio=1.5, vol targeting nunca dinámico).
        # Ahora: current_vol = std de últimos 7 días (168H), historical_vol = media rolling de 30d (720H).
        pct_changes = df['close'].pct_change()
        current_vol    = float(pct_changes.tail(7 * 24).std())
        historical_vol = float(pct_changes.rolling(30 * 24).std().iloc[-1])
        # Guard: si historical_vol es NaN (dataset muy corto), usar fallback
        if np.isnan(historical_vol) or historical_vol <= 0:
            historical_vol = current_vol if current_vol > 0 else 1e-6

        if final_confidence < 0.50:
             return {
                 "action": "HOLD",
                 "confidence": final_confidence,
                 "xgb_prob": xgb_prob,
                 "regime": regime,
                 "reason": f"El MetaLabeler veta la señal XGBoost ({direction}).",
                 "current_vol": current_vol,
                 "historical_vol": historical_vol,
                 "mvrv_zscore": df['mvrv'].iloc[-1] if 'mvrv' in df.columns else 1.0,
                 "funding_rate": df['funding_rate'].iloc[-1] if 'funding_rate' in df.columns else 0.0
             }
             
        # 8. Guard pre-ejecución por Tribe (Mejora M4)
        # Si la señal XGBoost es débil (0.50-0.58) Y la tribu es NEUTRAL,
        # la probabilidad real de alpha genuino es muy baja — bloqueamos.
        try:
            from luna.features.alpha_rules import NEUTRAL_TRIBES
            # Detectar tribu activa en la última barra
            tribe_col = None
            if 'KMeans_Tribe_ID' in df.columns:
                tribe_col = 'KMeans_Tribe_ID'
            elif 'K_Shape_Cluster_ID' in df.columns:
                tribe_col = 'K_Shape_Cluster_ID'

            if tribe_col is not None:
                current_tribe = int(df[tribe_col].iloc[-1])
                CONFIDENCE_MIN_LIMIT = 0.58  # Señales por debajo = débiles
                if current_tribe in NEUTRAL_TRIBES and xgb_prob < CONFIDENCE_MIN_LIMIT:
                    reason = (f"Guard M4: Señal bloqueada (Tribe {current_tribe} NEUTRAL "
                              f"| XGB={xgb_prob:.2%} < {CONFIDENCE_MIN_LIMIT:.0%})")
                    print(f"  [!] {reason}")
                    return {
                        "action": "HOLD",
                        "confidence": final_confidence,
                        "xgb_prob": xgb_prob,
                        "regime": regime,
                        "reason": reason,
                        "current_vol": current_vol,
                        "historical_vol": historical_vol,
                        "mvrv_zscore": df['mvrv'].iloc[-1] if 'mvrv' in df.columns else 1.0,
                        "funding_rate": df['funding_rate'].iloc[-1] if 'funding_rate' in df.columns else 0.0
                    }
        except Exception as e:
            print(f"  [!] Guard M4 no pudo ejecutarse: {e} — ignorando filtro tribal")

        return {
             "action": direction,
             "confidence": final_confidence,
             "xgb_prob": xgb_prob,
             "regime": regime,
             "reason": f"Unanimidad Validada (XGB: {xgb_prob:.2%} | LSTM: {final_confidence:.2%})",
             "price": df['close'].iloc[-1],
             "current_vol": current_vol,
             "historical_vol": historical_vol,
             "mvrv_zscore": df['mvrv'].iloc[-1] if 'mvrv' in df.columns else 1.0,
             "funding_rate": df['funding_rate'].iloc[-1] if 'funding_rate' in df.columns else 0.0
        }

if __name__ == "__main__":
    live_brain = LunaLiveInference()
    decision = live_brain.predict_cycle()
    print("\n🏁 DECISION FINAL:")
    print(json.dumps(decision, indent=4))
