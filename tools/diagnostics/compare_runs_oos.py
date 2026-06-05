"""
Comparativa OOS final — usa columna return_pct confirmada
"""
import sys, pandas as pd, numpy as np, pathlib
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
base = pathlib.Path('g:/Mi unidad/ia/luna_v2/data/runs')

configs = [
    ('ANT s95209', base / 'WFB_20260601_115308_seed95209/seed95209'),
    ('ANT s74482', base / 'WFB_20260601_131309_seed74482/seed74482'),
    ('NEW s42',    base / 'WFB_20260601_132627_seed42/seed42'),
    ('NEW s100',   base / 'WFB_20260601_141627_seed100/seed100'),
]

def stats(fp):
    if not fp.exists():
        return None
    try:
        df = pd.read_parquet(fp)
    except Exception:
        return None
    if len(df) == 0:
        return None
    v  = df['return_pct'].values
    eq = np.cumsum(v)
    n  = len(v)
    wr  = float((v > 0).sum() / n * 100)
    ret = float(v.sum() * 100)
    dd  = float((eq - np.maximum.accumulate(eq)).min() * 100)
    sr  = float(v.mean() / (v.std() + 1e-9) * np.sqrt(252)) if n > 1 else 0.0
    return n, wr, ret, dd, sr

print('=' * 65)
print('COMPARATIVA OOS W1/W2/W3  |  ANT=run anterior  NEW=run nueva')
print('=' * 65)
print()

for w in ['W1', 'W2', 'W3']:
    print(f'  {w}')
    print(f'  {"Label":<12} | {"N":>4} | {"WR%":>5} | {"ret%":>7} | {"MaxDD%":>7} | {"Sharpe":>6}')
    print('  ' + '-'*55)
    found = 0
    for label, sdir in configs:
        r = stats(sdir / w / 'oos_trades.parquet')
        if r is None:
            continue
        n, wr, ret, dd, sr = r
        found += 1
        tag = '>>>' if 'NEW' in label else '   '
        wr_s  = f'{wr:5.1f}%'
        ret_s = f'{ret:+7.3f}%'
        dd_s  = f'{dd:+7.3f}%'
        sr_s  = f'{sr:6.2f}'
        print(f'  {tag} {label:<10} | {n:>4} | {wr_s} | {ret_s} | {dd_s} | {sr_s}')
    if found == 0:
        print('    sin datos')
    print()
