import pandas as pd
import numpy as np
import sys

def test_thresholds():
    # Attempt to load raw probabilities
    try:
        df = pd.read_parquet("g:/Mi unidad/ia/luna_v2/data/predictions/oos_raw_probs.parquet")
    except Exception as e:
        print(f"Error loading oos_raw_probs.parquet: {e}")
        return

    print(f"Loaded raw probabilities: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    if 'meta_prob' not in df.columns:
        # Fallback if the column is different
        meta_col = [c for c in df.columns if 'meta' in c.lower() or 'prob' in c.lower()]
        if not meta_col:
            print("No meta_prob column found.")
            return
        prob_col = meta_col[0]
    else:
        prob_col = 'meta_prob'

    if 'Target_TBM_Bin' not in df.columns and 'target' not in df.columns:
        print("Warning: no target column found. Using simulated hit rates if 'return' exists.")
        
    ret_col = 'Return' if 'Return' in df.columns else ('return' if 'return' in df.columns else None)
    target_col = 'Target_TBM_Bin' if 'Target_TBM_Bin' in df.columns else ('target' if 'target' in df.columns else None)

    if not target_col and not ret_col:
        print("Cannot calculate win rate, no target or return columns.")
        return

    print(f"\n--- TESTING HYPOTHESIS: LOWERING PERCENTILE ---")
    print(f"Probability Column used: {prob_col}")
    
    percentiles = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60]
    
    results = []
    for p in percentiles:
        threshold = df[prob_col].quantile(p)
        df_filtered = df[df[prob_col] >= threshold]
        
        n_trades = len(df_filtered)
        if n_trades == 0:
            continue
            
        if target_col:
            wins = df_filtered[target_col].sum()
            wr = wins / n_trades * 100
        else:
            wins = (df_filtered[ret_col] > 0).sum()
            wr = wins / n_trades * 100
            
        total_ret = df_filtered[ret_col].sum() * 100 if ret_col else 0.0
        # Applying a simple 0.25% cost per trade
        net_ret = total_ret - (n_trades * 0.25)
        
        results.append({
            'Percentil': f"{p:.2f}",
            'Prob_Threshold': f"{threshold:.4f}",
            'Trades': n_trades,
            'WinRate(%)': f"{wr:.1f}%",
            'GrossRet(%)': f"{total_ret:.2f}%",
            'NetRet(%)': f"{net_ret:.2f}%"
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    test_thresholds()
