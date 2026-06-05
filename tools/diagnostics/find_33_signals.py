import pandas as pd
import numpy as np

def find_signals():
    path = "g:/Mi unidad/ia/luna_v2/data/reports/wfb/oos_raw_probs_W4_seed42.parquet"
    df = pd.read_parquet(path)
    print("=== OOS RAW PROBS W4 ===")
    print(df.describe())
    
    # Let's count how many times each column is non-zero
    print("\nNon-zero counts:")
    for col in df.columns:
        nonzero = (df[col] > 0).sum()
        print(f" - {col}: {nonzero}")
        
    # Let's see the unique values or value counts of prob_bull when it's > 0
    bull_active = df[df['prob_bull'] > 0]
    print(f"\nActive bull rows: {len(bull_active)}")
    
    # What are the thresholds? Let's check settings.yaml for the xgboost threshold
    # Let's read settings.yaml
    import yaml
    with open("g:/Mi unidad/ia/luna_v2/config/settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    xgb_thr = settings.get('xgboost', {}).get('proba_threshold', 0.5)
    print(f"\nSettings xgboost proba_threshold: {xgb_thr}")
    
    # How many active bull rows cross this threshold?
    crossed = df[df['prob_bull'] >= xgb_thr]
    print(f"Active bull rows crossing threshold {xgb_thr}: {len(crossed)}")
    
    # Let's print out all rows where prob_bull is active, sorted by timestamp
    print("\nAll active bull rows:")
    print(bull_active[['prob_bull', 'prob_bear', 'prob_range']])

if __name__ == "__main__":
    find_signals()
