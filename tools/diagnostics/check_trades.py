import pandas as pd
from pathlib import Path

wfb_dir = Path("data/reports/wfb")
files = sorted(wfb_dir.glob("oos_trades_W*_seed88611.parquet"))

for f in files:
    try:
        df = pd.read_parquet(f)
        w_num = f.stem.split('_')[2]
        print(f"--- {w_num} ---")
        print(f"Total Trades: {len(df)}")
        if len(df) > 0:
            print(f"Win Rate: {(df['is_win'] == 1).mean() * 100:.2f}%")
            print(f"Mean Return: {df['return_pct'].mean():.4f}%")
            print(f"Max Drawdown: {df['drawdown'].min():.4f}%")
        print("")
    except Exception as e:
        print(f"Error reading {f.name}: {e}\n")
