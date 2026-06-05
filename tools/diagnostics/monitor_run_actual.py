from pathlib import Path
import pandas as pd
import os, time

wfb = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')
parquets = list(wfb.glob('oos_trades_W*_seed*.parquet'))
now = time.time()
today_files = [f for f in parquets if (now - f.stat().st_mtime) < 7200]

seeds_data = {}
for f in today_files:
    parts = f.stem.split('_')
    seed = int(next(p.replace('seed','') for p in parts if p.startswith('seed')))
    win  = next(p for p in parts if p.startswith('W'))
    df = pd.read_parquet(f)
    if seed not in seeds_data:
        seeds_data[seed] = {}
    seeds_data[seed][win] = len(df)

print(f'Seeds con trades en ultimas 2h: {len(seeds_data)}')
print()
print(f'{"Seed":>8} | W1  | W2  | W3  | W4  | W5  | Total')
print('-'*52)
total_trades = 0
for seed, wins in sorted(seeds_data.items()):
    row = f'{seed:>8} |'
    t = 0
    for w in ['W1','W2','W3','W4','W5']:
        n = wins.get(w, '-')
        t += n if isinstance(n, int) else 0
        row += f' {str(n):>3} |'
    row += f' {t:>4}'
    total_trades += t
    print(row)
print(f'TOTAL trades nuevos esta run: {total_trades}')
