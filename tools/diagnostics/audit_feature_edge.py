import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import json
from loguru import logger
from scipy.stats import spearmanr
from luna.features.tbm import apply_triple_barrier

def main():
    root = Path(__file__).resolve().parent.parent.parent
    df = pd.read_parquet(root / "data" / "features" / "features_train.parquet")
    
    with open(root / "data" / "features" / "selected_features.json", 'r') as f:
        feats = json.load(f)["selected_features"]
        
    print(f"Total features to check: {len(feats)}")
    
    # Label with TBM
    events_idx = df.index
    price_series = df["close"]
    sides_series = pd.Series(1.0, index=events_idx) # Assume long-only for univariate test
    
    tbm_result = apply_triple_barrier(
        price_series=price_series,
        event_times=events_idx,
        sides=sides_series,
        pt_sl_multiplier=[2.0, 1.0],
        min_return=0.005,
        vertical_barrier_hours=72,
        dynamic_barrier=False,
    )
    
    df_labeled = df.join(tbm_result[['bin', 'ret']], how='inner')
    target_ret = df_labeled['ret']
    
    print("\n--- INFORMATION COEFFICIENT (SPEARMAN) ---")
    results = []
    for f in feats:
        if f in df_labeled.columns:
            feat_series = df_labeled[f].fillna(0)
            rho, pval = spearmanr(feat_series, target_ret)
            results.append((f, rho, pval))
    
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for f, rho, pval in results:
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "ns"))
        print(f"{f:<30} | Rho: {rho:+.4f} | P-val: {pval:.4f} {sig}")
        
    # Test linear baseline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler
    
    X = df_labeled[feats].fillna(0)
    y = df_labeled['bin']
    
    print(f"\n--- BASE RATE ---")
    print(f"Win Rate: {y.mean():.2%}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    tscv = TimeSeriesSplit(n_splits=5)
    
    auc_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc')
    
    print(f"\n--- LINEAR BASELINE (Logistic Regression) ---")
    print(f"ROC AUC (5-fold TimeSeriesCV): {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")
    
if __name__ == "__main__":
    main()
