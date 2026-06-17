import pandas as pd
import numpy as np

try:
    df = pd.read_parquet('data/predictions/oos_trades_xgb_baseline.parquet')
    
    percentiles_to_test = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    print(f"Total Base Signals: {len(df)}")
    base_win_rate = df['is_win'].mean() * 100
    base_ret = df['return_raw'].sum() * 100
    print(f"Base Win Rate: {base_win_rate:.2f}% | Base Return: {base_ret:.2f}%")
    print("-" * 50)
    
    for p in percentiles_to_test:
        threshold = df['meta_v2_prob'].quantile(p)
        df_filtered = df[df['meta_v2_prob'] >= threshold]
        
        n_trades = len(df_filtered)
        if n_trades > 0:
            win_rate = df_filtered['is_win'].mean() * 100
            ret_sum = df_filtered['return_raw'].sum() * 100
            print(f"Percentile {p*100:.0f}% (Prob >= {threshold:.4f}) | Trades: {n_trades:<4} | WinRate: {win_rate:.2f}% | Return: {ret_sum:.2f}%")
        else:
            print(f"Percentile {p*100:.0f}% | 0 trades")
except Exception as e:
    print(e)
