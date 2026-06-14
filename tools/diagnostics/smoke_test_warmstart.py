import sys
from pathlib import Path
import json
import os

base_dir = Path("c:/Users/Usuario/Desktop/ia/luna_v2")
sys.path.append(str(base_dir))

# Mock environment to pretend we are running WFB
os.environ["LUNA_SEED"] = "999"

from luna.models.train_xgboost_v2 import XGBoostTrainer

def test_warm_start():
    print("--- [TEST] Iniciando Smoke Test de Warm Start ---")
    # 1. Crear directorios mock de WFB cache
    wfb_cache = base_dir / "data" / "wfb_cache" / "seed999"
    w1_models = wfb_cache / "W1" / "models"
    w1_models.mkdir(parents=True, exist_ok=True)
    
    # 2. Escribir signature mock de W1
    mock_signature = {
        "params": {
            "n_estimators": 404,
            "max_depth": 3,
            "learning_rate": 0.042,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "gamma": 0.5,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "focal_loss_gamma": 1.5,
            "spurious_param": "should_be_ignored"
        }
    }
    with open(w1_models / "xgboost_meta_bull_long_signature.json", "w", encoding="utf-8") as f:
        json.dump(mock_signature, f)
        
    print("✓ Mocks creados")

    try:
        # 3. Inicializar el modelo
        model = XGBoostTrainer(
            regime_name="bull",
            n_trials=2
        )
        model.root = base_dir  # Set root path explicitly for the mock
        
        # Mock dataset para que no aborte por falta de datos
        import pandas as pd
        import numpy as np
        model.X = pd.DataFrame({"feat1": np.random.randn(500), "feat2": np.random.randn(500)})
        model.y = pd.Series(np.random.randint(0, 2, size=500))
        model.close_rets = pd.Series(np.random.randn(500) * 0.01)
        model.features = ["feat1", "feat2"]
        
        print("✓ Ejecutando tune_hyperparameters...")
        # 4. Lanzar tune_hyperparameters
        model.tune_hyperparameters()
        
        # 5. Verificar que el trial 0 tiene los parametros de W1
        trials = model.study.trials
        if len(trials) > 0:
            first_trial = trials[0]
            print(f"Trial 0 Params: {first_trial.params}")
            if abs(first_trial.params.get("learning_rate", 0) - 0.042) < 1e-6:
                print("✓ EXITOSO: El Trial 0 ha utilizado los params de W1 inyectados por Warm Start!")
            else:
                print("❌ ERROR: El Trial 0 NO tiene los params de W1.")
        
    finally:
        # Cleanup
        import shutil
        if (base_dir / "data" / "wfb_cache" / "seed999").exists():
            shutil.rmtree(base_dir / "data" / "wfb_cache" / "seed999")

if __name__ == "__main__":
    test_warm_start()
