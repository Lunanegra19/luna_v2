import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
predictions_dir = _ROOT / "data" / "predictions"

def audit():
    print("="*90)
    print("      DIAGNOSTIC AUDIT: ENSEMBLE PORTFOLIO DISCREPANCY DETECTOR      ")
    print("="*90)
    
    unified_path = predictions_dir / "unified_ensemble_trades_raw.parquet"
    portfolio_path = predictions_dir / "ensemble_portfolio_trades.parquet"
    
    if not unified_path.exists() or not portfolio_path.exists():
        print("ERROR: Parquet files not found in data/predictions.")
        return
        
    df_raw = pd.read_parquet(unified_path)
    df_port = pd.read_parquet(portfolio_path)
    
    print(f"Unified Raw Trades (df_raw): {len(df_raw)} rows")
    print(f"Ensemble Portfolio Trades (df_port): {len(df_port)} rows")
    
    # Analyze individual seeds in df_raw
    print("\n--- INDIVIDUAL SEEDS ANALYSIS IN DF_RAW ---")
    for seed, group in df_raw.groupby('seed'):
        n = len(group)
        mean_ret = group['return_pct'].mean()
        sum_ret = group['return_pct'].sum()
        std_ret = group['return_pct'].std()
        wr = group['is_win'].mean()
        
        days = (group.index.max() - group.index.min()).days
        n_per_year = n / (days / 365.25) if days > 0 else n * 365.25
        sharpe = (mean_ret / std_ret) * (n_per_year ** 0.5) if std_ret > 1e-9 else 0.0
        
        print(f"Seed {seed:4d}: Trades={n:3d} | WR={wr*100:5.2f}% | Avg Ret={mean_ret*100:8.5f}% | Sum Ret={sum_ret*100:8.5f}% | Std={std_ret*100:8.5f}% | Sharpe={sharpe:6.4f}")
        
    # Analyze df_port
    print("\n--- ENSEMBLE PORTFOLIO ANALYSIS (df_port) ---")
    n_port = len(df_port)
    mean_ret_port = df_port['return_pct'].mean()
    sum_ret_port = df_port['return_pct'].sum()
    std_ret_port = df_port['return_pct'].std()
    wr_port = df_port['is_win'].mean()
    
    days_port = (df_port.index.max() - df_port.index.min()).days
    n_per_year_port = n_port / (days_port / 365.25) if days_port > 0 else n_port * 365.25
    sharpe_port = (mean_ret_port / std_ret_port) * (n_per_year_port ** 0.5) if std_ret_port > 1e-9 else 0.0
    
    print(f"Portfolio: Trades={n_port:3d} | WR={wr_port*100:5.2f}% | Avg Ret={mean_ret_port*100:8.5f}% | Sum Ret={sum_ret_port*100:8.5f}% | Std={std_ret_port*100:8.5f}% | Sharpe={sharpe_port:6.4f}")
    
    # Check if there are duplicate indices in df_raw
    print("\n--- COLLISIONS ANALYSIS ---")
    collisions = df_raw.index.value_counts()
    print(f"Timestamps with 1 seed active: {len(collisions[collisions == 1])}")
    print(f"Timestamps with 2 seeds active: {len(collisions[collisions == 2])}")
    print(f"Timestamps with 3 seeds active: {len(collisions[collisions == 3])}")
    print(f"Timestamps with 4 seeds active: {len(collisions[collisions == 4])}")
    print(f"Timestamps with 5 seeds active: {len(collisions[collisions == 5])}")
    
    # Check what the average return is for collisions vs non-collisions
    df_raw_with_counts = df_raw.copy()
    df_raw_with_counts['count'] = df_raw_with_counts.index.map(collisions)
    
    print("\n--- RETURN BY SEED COLLISION COUNT ---")
    for count, group in df_raw_with_counts.groupby('count'):
        print(f"Active Seeds = {count}: Trades={len(group):3d} | Avg Ret={group['return_pct'].mean()*100:8.5f}% | WR={group['is_win'].mean()*100:5.2f}%")
        
    print("\n--- DETAILED BREAKDOWN PER SEED: UNIQUE VS SHARED ---")
    for seed, group in df_raw_with_counts.groupby('seed'):
        unique_trades = group[group['count'] == 1]
        shared_trades = group[group['count'] > 1]
        
        n_uniq = len(unique_trades)
        mean_ret_uniq = unique_trades['return_pct'].mean() * 100 if n_uniq > 0 else 0.0
        sum_ret_uniq = unique_trades['return_pct'].sum() * 100 if n_uniq > 0 else 0.0
        wr_uniq = unique_trades['is_win'].mean() * 100 if n_uniq > 0 else 0.0
        
        n_shar = len(shared_trades)
        mean_ret_shar = shared_trades['return_pct'].mean() * 100 if n_shar > 0 else 0.0
        sum_ret_shar = shared_trades['return_pct'].sum() * 100 if n_shar > 0 else 0.0
        wr_shar = shared_trades['is_win'].mean() * 100 if n_shar > 0 else 0.0
        
        print(f"Seed {seed:4d} | Unique Trades: N={n_uniq:2d}, WR={wr_uniq:5.2f}%, AvgRet={mean_ret_uniq:8.5f}%, SumRet={sum_ret_uniq:8.5f}%")
        print(f"          | Shared Trades: N={n_shar:2d}, WR={wr_shar:5.2f}%, AvgRet={mean_ret_shar:8.5f}%, SumRet={sum_ret_shar:8.5f}%")
        print(f"          | Total: N={len(group):2d}, SumRet={group['return_pct'].sum()*100:8.5f}%")
        print("-" * 80)
        
    print("\n--- SAMPLE OF MULTI-SEED COLLISION TRADES ---")
    multi_seed_idx = collisions[collisions > 1].index
    if len(multi_seed_idx) > 0:
        sample_idx = multi_seed_idx[:3]
        for idx in sample_idx:
            print(f"\nTimestamp: {idx}")
            trades_at_idx = df_raw.loc[[idx]]
            print(trades_at_idx[['seed', 'direction', 'return_pct', 'is_win', 'wfb_window']].to_string())
            print(f"Combined Portfolio Trade return_pct: {df_port.loc[idx, 'return_pct']*100:.5f}% | is_win: {df_port.loc[idx, 'is_win']}")
            
if __name__ == "__main__":
    audit()
