import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Añadir el root del proyecto al sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from luna.models.hmm_regime import HMMRegimeModel
import joblib
from loguru import logger

def test_hmm_oos_blindness():
    # 1. Load the model
    model_dir = root_dir / "data" / "models"
    model_path = model_dir / "hmm_regime.pkl"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
        
    logger.info(f"Loading HMM model from {model_dir}")
    hmm = HMMRegimeModel.load(str(model_dir))
    
    # Check features expected by the model
    feats = getattr(hmm, "_features", [])
    logger.info(f"Features expected by HMM: {feats}")
    
    # 2. Create a dummy dataframe mimicking OOS data (without z90d suffixes)
    dates = pd.date_range(start="2025-01-01", periods=10, freq="1h", tz="UTC")
    df_oos = pd.DataFrame(index=dates)
    
    # Add raw features that exist before z-score transform
    if any("close_fd" in f for f in feats):
        df_oos["close_fd"] = np.random.randn(10) * 0.05
    if any("mt_vol_realized_4bar" in f for f in feats):
        df_oos["mt_vol_realized_4bar"] = np.abs(np.random.randn(10) * 0.1)
    
    # 3. Predict regimes
    logger.info("Predicting regimes on dummy OOS DataFrame (which lacks _z90d features)...")
    res = hmm.predict_regime_series(df_oos)
    
    logger.info("Prediction result:")
    print(res.head())
    
if __name__ == "__main__":
    test_hmm_oos_blindness()
