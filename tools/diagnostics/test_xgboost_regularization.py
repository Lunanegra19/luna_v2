"""
Diagnostic script to test the XGBoost regularization and calibration hypothesis.
Evaluates:
1. Low Regularization (Overfitting): min_child_weight=1, max_depth=5 (Current settings)
2. High Regularization (Proposed): min_child_weight=30, max_depth=3/4
Evaluates on training and validation sets to measure:
- Brier Score vs Naive Baseline (lower is better, Brier > naive = bad/uncalibrated).
- Isotonic Calibrator degeneration (whether calibrated probabilities collapse to a flat distribution, std < 1e-4).
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(get_project_root()))

from luna.features.tbm import apply_triple_barrier

def run_experiment():
    root = get_project_root()
    print("=== HYPOTHESIS TEST: XGBOOST REGULARIZATION & CALIBRATION DEGENERATION ===")
    
    # 1. Load datasets
    print("\nLoading datasets...")
    df_train = pd.read_parquet(root / "data" / "features" / "features_train.parquet")
    df_val = pd.read_parquet(root / "data" / "features" / "features_validation.parquet")
    
    with open(root / "data" / "features" / "selected_features.json", 'r') as f:
        selected_features = json.load(f)["selected_features"]
    
    # Ensure HMM_Regime is in features if present
    if "HMM_Regime" not in selected_features:
        selected_features.append("HMM_Regime")
        
    df_hmm = pd.read_parquet(root / "data" / "features" / "hmm_regime_labels.parquet")
    
    # Avoid duplicate column join error
    overlap_cols = [c for c in df_hmm.columns if c in df_train.columns]
    if overlap_cols:
        df_hmm = df_hmm.drop(columns=overlap_cols)
        
    df_train = df_train.join(df_hmm)
    
    # Apply HMM_Semantic in validation if needed, else we can do a standard check
    # Let's align features
    features = [f for f in selected_features if f in df_train.columns and f in df_val.columns]
    print(f"Total features matched: {len(features)}")
    
    # 2. Run TBM to generate targets
    print("\nGenerating TBM targets...")
    pt = 1.8
    sl = 1.5
    vbh = 72
    min_ret = 0.003
    
    # Train targets
    tbm_train = apply_triple_barrier(
        price_series=df_train["close"],
        event_times=df_train.index,
        pt_sl_multiplier=[pt, sl],
        min_return=min_ret,
        vertical_barrier_hours=vbh
    )
    df_train_labeled = df_train.join(tbm_train[['bin', 'ret']], how='inner')
    df_train_labeled["target"] = (df_train_labeled["bin"] == 1).astype(int)
    
    # Validation targets
    tbm_val = apply_triple_barrier(
        price_series=df_val["close"],
        event_times=df_val.index,
        pt_sl_multiplier=[pt, sl],
        min_return=min_ret,
        vertical_barrier_hours=vbh
    )
    df_val_labeled = df_val.join(tbm_val[['bin', 'ret']], how='inner')
    df_val_labeled["target"] = (df_val_labeled["bin"] == 1).astype(int)
    
    # Clean up NaNs in targets
    df_train_clean = df_train_labeled.dropna(subset=['target'])
    df_val_clean = df_val_labeled.dropna(subset=['target'])
    
    X_train = df_train_clean[features].fillna(0)
    y_train = df_train_clean['target']
    
    X_val = df_val_clean[features].fillna(0)
    y_val = df_val_clean['target']
    
    naive_brier = y_val.mean() * (1 - y_val.mean())
    print(f"Train samples: {len(X_train)} (Positive rate: {y_train.mean():.2%})")
    print(f"Validation samples: {len(X_val)} (Positive rate: {y_val.mean():.2%})")
    print(f"Naive Brier Baseline (random guessing): {naive_brier:.4f}")
    
    # Configurations to test
    configs = [
        {
            "name": "Current Overfitted (Low Regularization)",
            "params": {
                "max_depth": 5,
                "min_child_weight": 1,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_estimators": 200,
                "objective": "binary:logistic",
                "random_state": 42,
                "n_jobs": -1
            }
        },
        {
            "name": "Proposed Regularized (High Regularization)",
            "params": {
                "max_depth": 3,
                "min_child_weight": 30,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_estimators": 200,
                "objective": "binary:logistic",
                "random_state": 42,
                "n_jobs": -1
            }
        }
    ]
    
    for config in configs:
        print(f"\nEvaluating: {config['name']}")
        print(f"Parameters: max_depth={config['params']['max_depth']}, min_child_weight={config['params']['min_child_weight']}")
        
        # Train model
        model = xgb.XGBClassifier(**config["params"])
        model.fit(X_train, y_train)
        
        # Predict probabilities
        p_train = model.predict_proba(X_train)[:, 1]
        p_val = model.predict_proba(X_val)[:, 1]
        
        # Calculate Brier Scores
        brier_train = brier_score_loss(y_train, p_train)
        brier_val = brier_score_loss(y_val, p_val)
        
        print(f"  - Train Brier: {brier_train:.4f}")
        print(f"  - Val Brier:   {brier_val:.4f} (Naive baseline: {naive_brier:.4f})")
        if brier_val < naive_brier:
            print("    [PASS] Validation Brier is BETTER than random baseline.")
        else:
            print("    [FAIL] Validation Brier is WORSE than random baseline (No Signal/Descalibrated).")
            
        # Check Isotonic Calibration
        print("  - Testing Isotonic Calibration...")
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(p_val, y_val)
        p_cal = iso.predict(p_val)
        
        cal_std = np.std(p_cal)
        cal_range = [p_cal.min(), p_cal.max()]
        n_anchors = len(getattr(iso, 'X_thresholds_', []))
        brier_cal = brier_score_loss(y_val, p_cal)
        
        print(f"    * Calibrated Brier: {brier_cal:.4f}")
        print(f"    * Calibrated std:   {cal_std:.6f}")
        print(f"    * Calibrated range: [{cal_range[0]:.4f}, {cal_range[1]:.4f}]")
        print(f"    * Calibrated anchors: {n_anchors}")
        
        is_degenerate = (cal_std < 1e-4) or (n_anchors <= 2 and (cal_range[1] - cal_range[0]) < 1e-4)
        if is_degenerate:
            print("    * [DEGENERATE ALERT] Calibrator collapsed to a flat line! std is too low.")
        else:
            print("    * [STABLE] Calibrator is working correctly and has spread.")

if __name__ == "__main__":
    run_experiment()
