"""
OOD Guard (Isolation Forest Anti-Cisnes Negros) - Luna V1
===================================================
Aprende la topologia N-dimensional de los mercados historicos usando
Features OOS. Si las lecturas actuales se salen de esa distribucion,
este modelo levantara la bandera de Anomaly (1) para bloquear el trading.

SOP Aplicado:
- Evita que el pipeline (MetaLabelerV2 + XGBoost) opere en condiciones de
  mercado Out-of-Distribution (fuera de lo observado en el training set).
- P4-0-3 PENDIENTE: contamination dinamico calibrado con crisis historicas.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import json
from loguru import logger

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

# Asegurar path para imports locales
root = get_project_root()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

class OODGuardTrainer:
    def __init__(self):
        self.root = get_project_root()
        self.model_path = self.root / "data" / "models" / "ood_guard.pkl"
        self.sig_path = self.root / "data" / "models" / "ood_guard_signature.json"
        
    def train(self):
        logger.info("Entrenando OOD Guard (Isolation Forest)...")
        
        # Cargar training data
        df = pd.read_parquet(self.root / "data" / "features" / "features_train.parquet")
        with open(self.root / "data" / "features" / "selected_features.json", 'r') as f:
            _feat_json = json.load(f)
            features = _feat_json["selected_features"]
            # P1-N3-FIX (2026-03-30): incluir pass_through_features para que el OOD Guard
            # sea entrenado con el mismo feature space que XGBoost.
            # Sin este fix: OOD Guard usa solo SFI features → distribución N-dim diverge
            # del XGBoost que también recibe HMM_Regime, KMeans_Tribe_ID, etc.
            _pass_through = _feat_json.get("pass_through_features", [])
            # [FIX-OOD-PASSTHROUGH-01] SOP V10: Las variables estructurales (passthrough)
            # como close_fd estan matematicamente blindadas. Incluirlas en el OOD Guard 
            # provoca censura empirica (bloqueo por Covariate Shift).
            # Por tanto, el OOD Guard evaluara UNICAMENTE las variables SFI.
            features = list(dict.fromkeys(features))  # Solo SFI, excluimos _pass_through
            logger.info(
                f"[FIX-OOD-PASSTHROUGH-01] OOD Guard: {len(features)} features SFI puros. "
                f"Ignorando {len(_pass_through)} pass_through para evitar censura estructural."
            )

        use_cols = [c for c in features if c in df.columns]
        # LOGIC-OOD-01 FIX (2026-04-06): usar fillna(0) en lugar de dropna().
        # En producción (signal_filter.apply_ood), el OOD Guard recibe X con fillna(0).
        # El entrenamiento con dropna() aprendía la distribución de vectores COMPLETOS,
        # mientras que en producción llegaban vectores con 0s en posiciones NaN
        # (ETF gaps, macro lag, mining rules ausentes).
        # Esto causaba falsos positivos OOD (bloqueo injustificado de señales válidas)
        # porque el IsolationForest nunca había visto ceros en esas dimensiones.
        # Fix: entrenar con el mismo fillna(0) que se usa en inferencia.
        X = df[use_cols].fillna(0)


        if len(X) < 1000:
            logger.error("Dataset insuficiente para modelar distribución normal.")
            # P3-N2-FIX (2026-03-30): sys.exit(1) en lugar de return silencioso.
            # Un return aquí deja ood_guard.pkl del ciclo anterior en disco (stale).
            # El orchestrator consideraria el training "exitoso" y continuaría con
            # un guard desactualizado potencialmente de varias ventanas WFB atrás.
            sys.exit(1)

        # M-17-FIX (2026-03-16): parámetros leídos de settings.yaml — elimina hardcodes.
        # Antes: contamination=0.03 hardcodeado → bloqueaba barras del holdout 2025.
        # Ahora: ood_guard.contamination en settings.yaml = 0.10 (configurable).
        try:
            from config.settings import cfg as _cfg_ood
            _contamination  = float(getattr(_cfg_ood.ood_guard, 'contamination',  0.10))
            _n_estimators   = int(getattr(_cfg_ood.ood_guard,   'n_estimators',   200))
            _random_state   = int(getattr(_cfg_ood.ood_guard,   'random_state',   42))
        except Exception:
            _contamination, _n_estimators, _random_state = 0.10, 200, 42
            logger.warning("OOD Guard: settings.yaml no disponible — usando fallback contamination=0.10")

        logger.info(
            f"OOD Guard config: contamination={_contamination} | n_estimators={_n_estimators}"
        )
        model = IsolationForest(
            n_estimators=_n_estimators,
            max_samples='auto',
            contamination=_contamination,
            random_state=_random_state,
            n_jobs=-1
        )
        
        logger.info(f"Modelando nube multi-dimensional con {len(use_cols)} variables sobre {len(X)} horas de mercado...")
        model.fit(X)
        
        # Save model
        joblib.dump(model, self.model_path)
        
        # Save threshold mapping reference para depuración
        scores = model.decision_function(X)
        _pct = int(_contamination * 100)  # percentile consistente con contamination
        threshold = np.percentile(scores, _pct)
        
        with open(self.sig_path, 'w') as f:
            # MEJ-OOD-01 FIX (2026-04-06): añadir hash del selected_features.json y timestamp.
            # Sin esta información, es imposible saber si el OOD Guard está desactualizado
            # respecto al XGBoost (que puede haberse re-entrenado con un SFI diferente).
            import hashlib as _hashlib
            import datetime as _datetime
            _sfi_path = self.root / "data" / "features" / "selected_features.json"
            _sfi_hash = _hashlib.md5(_sfi_path.read_bytes()).hexdigest()[:12] if _sfi_path.exists() else "unknown"
            json.dump({
                "features_tracked": use_cols,
                "n_features": len(use_cols),
                "n_samples": len(X),
                "contamination": _contamination,
                "anomaly_score_threshold": threshold,
                "trained_at": _datetime.datetime.now().isoformat(),
                "sfi_hash": _sfi_hash,  # MEJ-OOD-01: para detectar drift de features
                "fillna_strategy": "fillna(0)",  # LOGIC-OOD-01: consistente con apply_ood()
                "description": "Si score(X) < threshold_referencia, el vector X es una anomalía OOD."
            }, f, indent=4)
            
        logger.success("OOD Guard entrenado y persistido. Detectará Cisnes Negros en Live.")
        logger.info("[MEJ-OOD-01] SFI hash={} | features={} | Fill strategy: fillna(0)", _sfi_hash, len(use_cols))



if __name__ == "__main__":
    logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
    trainer = OODGuardTrainer()
    trainer.train()
