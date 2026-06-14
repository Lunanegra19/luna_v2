import pandas as pd
import os, glob
p = 'c:/Users/Usuario/Desktop/ia/luna_v2/data/reports/wfb'
files = glob.glob(p + '/oos_trades_W*.parquet')
results = []
for f in files:
    if 'baseline' not in f and 'flag' not in f:
        df = pd.read_parquet(f)
        results.append({
            'file': os.path.basename(f),
            'trades': len(df),
            'wr': df['is_win'].mean()*100 if len(df) > 0 else 0
        })

for r in sorted(results, key=lambda x: x['file']):
    print(f"{r['file']}: {r['trades']} trades, WR: {r['wr']:.2f}%")
