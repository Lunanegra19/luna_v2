import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path("g:/Mi unidad/ia/luna_v2")
reports_wfb = ROOT / "data" / "reports" / "wfb"

print("=== DEEP INSPECTION OF SEED 1337 TRADES ===")
files = sorted(reports_wfb.glob("oos_trades_W*_seed1337.parquet"))

for f in files:
    print(f"\nFile: {f.name}")
    df = pd.read_parquet(f)
    print(f"  Total Trades: {len(df)}")
    if len(df) > 0:
        print(f"  Columns: {list(df.columns)}")
        
        # Check target, return, is_win, etc.
        cols = [c for c in ["is_win", "return_raw", "return_pct", "ret_bruto", "ret", "xgb_prob_cal", "meta_v2_prob"] if c in df.columns]
        print(df[cols].to_string())
        
        # Compute exact win rate
        if "is_win" in df.columns:
            print(f"  Calculated WR from 'is_win': {df['is_win'].mean() * 100:.2f}%")
        if "return_raw" in df.columns:
            print(f"  Calculated WR from 'return_raw > 0': {(df['return_raw'] > 0).mean() * 100:.2f}%")
        if "return_pct" in df.columns:
            print(f"  Calculated WR from 'return_pct > 0': {(df['return_pct'] > 0).mean() * 100:.2f}%")
            
        print("  Statistical stats of return:")
        if "return_raw" in df.columns:
            print(f"    Mean Return: {df['return_raw'].mean() * 100:.4f}%")
            print(f"    Std Return: {df['return_raw'].std() * 100:.4f}%")
            print(f"    Skewness: {df['return_raw'].skew():.4f}")
            print(f"    Kurtosis: {df['return_raw'].kurt():.4f}")
