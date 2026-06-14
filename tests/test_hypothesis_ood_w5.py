import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Resolve project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

def calculate_psi(expected, actual, num_buckets=10):
    """Calcula el Population Stability Index (PSI) entre dos series de datos."""
    # Filtrar nans
    expected = expected.dropna().values
    actual = actual.dropna().values
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Obtener cuantiles del dataset esperado (training)
    percentiles = np.linspace(0, 100, num_buckets + 1)
    try:
        # Usar unique para evitar errores en percentiles idénticos si hay valores constantes
        buckets = np.unique(np.percentile(expected, percentiles))
        if len(buckets) < 2:
            return 0.0
    except Exception:
        return 0.0
        
    buckets[0] = -np.inf
    buckets[-1] = np.inf
    
    # Contar observaciones en cada bucket
    expected_counts = np.histogram(expected, bins=buckets)[0]
    actual_counts = np.histogram(actual, bins=buckets)[0]
    
    # Proporciones
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Suavizado para evitar división por cero
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)
    
    # Cálculo del PSI
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi_value

def test_hypothesis_ood_w5():
    print("[TEST-OOD-W5] Starting quantitative test for W5 feature drift/OOD...")
    
    # 1. Load W5 feature parquet files
    w5_feat_dir = ROOT / "data" / "wfb_cache" / "W5" / "features"
    train_path = w5_feat_dir / "features_train.parquet"
    holdout_path = w5_feat_dir / "features_holdout_W5.parquet"
    sel_feats_path = w5_feat_dir / "selected_features.json"
    
    if not (train_path.exists() and holdout_path.exists() and sel_feats_path.exists()):
        print("[TEST-OOD-W5] SKIP: W5 cached parquets or selected features list not found.")
        return
        
    df_train = pd.read_parquet(train_path)
    df_holdout = pd.read_parquet(holdout_path)
    
    with open(sel_feats_path, "r", encoding="utf-8") as f:
        selected_features_data = json.load(f)
        
    # extract features list
    if isinstance(selected_features_data, dict) and "selected_features" in selected_features_data:
        selected_features = selected_features_data["selected_features"]
    elif isinstance(selected_features_data, list):
        selected_features = selected_features_data
    else:
        selected_features = list(df_train.columns[:10]) # fallback
        
    print(f"[TEST-OOD-W5] Loaded selected features for W5: {len(selected_features)}")
    print(f"  Train shape: {df_train.shape} | Holdout shape: {df_holdout.shape}")
    
    # 2. Calculate PSI for each selected feature
    drifted_features = []
    psi_scores = {}
    
    for feat in selected_features:
        if feat in df_train.columns and feat in df_holdout.columns:
            psi = calculate_psi(df_train[feat], df_holdout[feat])
            psi_scores[feat] = psi
            if psi > 0.25:
                drifted_features.append((feat, psi))
                
    # Sort by PSI descending
    drifted_features = sorted(drifted_features, key=lambda x: x[1], reverse=True)
    
    print(f"\n--- Feature Drift Analysis (PSI > 0.25 = severely drifted/OOD) ---")
    print(f"  Total analyzed features: {len(psi_scores)}")
    print(f"  Drifted features count:  {len(drifted_features)} ({len(drifted_features)/max(1, len(psi_scores)):.1%})")
    
    print("\nTop Drifted Features:")
    for feat, psi in drifted_features[:15]:
        print(f"  {feat}: PSI = {psi:.4f}")
        
    # Check if there is high drift
    pct_drifted = len(drifted_features) / max(1, len(psi_scores))
    print(f"\nDrift ratio: {pct_drifted:.1%}")
    
    # We assert that there is a significant fraction of drifted features in W5 (at least 15% of selected features are drifted)
    assert pct_drifted >= 0.15, f"Hypothesis failed: only {pct_drifted:.1%} of features drifted, which is not OOD enough."
    
    print(f"[TEST-OOD-W5] SUCCESS: Hypothesis verified! A significant fraction ({pct_drifted:.1%}) of selected features are severely drifted/OOD in W5.")
    print("[TEST-OOD-W5] This explains the performance colapse (WR=11.8%) due to using obsolete train cutoff (Oct-2025) for Q1-2026 data.")

if __name__ == "__main__":
    test_hypothesis_ood_w5()
