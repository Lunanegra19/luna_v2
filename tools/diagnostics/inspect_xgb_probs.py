import pandas as pd
from pathlib import Path
import json
import warnings
import numpy as np
import joblib

warnings.filterwarnings("ignore")
ROOT = Path("g:/Mi unidad/ia/luna_v2")

import sys
sys.path.insert(0, str(ROOT))
from luna.models.regime_router import RegimeRouter

window_id = "W3"
seed = 42

models_dir = ROOT / "data" / "wfb_cache" / f"seed{seed}" / window_id / "models"
holdout_path = ROOT / "data" / "features" / f"features_holdout_{window_id}.parquet"

df_oos = pd.read_parquet(holdout_path)
df_oos.index = pd.to_datetime(df_oos.index, utc=True)
df_oos = df_oos.sort_index()

print("Holdout shape:", df_oos.shape)
print("HMM_Regime distribution:")
print(df_oos["HMM_Regime"].value_counts(dropna=False))

router_xgb = RegimeRouter(models_dir, agent_type="xgboost", direction="long")
xgb_probs_df = router_xgb.route_and_predict(df_oos)

print("\nPredicted probabilities summary:")
print(xgb_probs_df.describe())

# Check how they map in SignalFilter thresholds
xgb_sig_path = models_dir / "xgboost_meta_bull_long_signature.json"
if not xgb_sig_path.exists():
    xgb_sig_path = models_dir / "xgboost_meta_long_signature.json"

with open(xgb_sig_path, "r", encoding="utf-8") as f:
    xgb_sig = json.load(f)
available_feats = xgb_sig.get("features", [])

print("\nFirst 10 predicted probabilities:")
print(xgb_probs_df.head(10))

# Miremos si hay un archivo de calibrador
calib_sig_path = models_dir / "calibrator_long_signature.json"
if calib_sig_path.exists():
    with open(calib_sig_path) as csf:
        print("\nCalibrator signature:")
        print(json.load(csf))
