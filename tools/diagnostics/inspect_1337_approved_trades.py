import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path("g:/Mi unidad/ia/luna_v2")
run_dir = ROOT / "data" / "runs" / "WFB_20260521_033115_seed1337" / "seed1337"

print("=== INSPECTION OF APPROVED SEED 1337 RUN (20260521_033115) ===")

for w in ["W2", "W3", "W4", "W5"]:
    p = run_dir / w / "oos_trades.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        print(f"\nWindow: {w}")
        print(f"  Total Trades: {len(df)}")
        cols = [c for c in ["is_win", "return_raw", "return_pct", "xgb_prob_cal", "meta_v2_prob"] if c in df.columns]
        print(df[cols].to_string())
        if "is_win" in df.columns:
            print(f"  Calculated WR from 'is_win': {df['is_win'].mean() * 100:.2f}%")
        if "return_raw" in df.columns:
            print(f"  Calculated WR from 'return_raw > 0': {(df['return_raw'] > 0).mean() * 100:.2f}%")
    else:
        print(f"\nWindow: {w} - NO TRADES (or empty parquet)")
