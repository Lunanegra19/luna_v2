"""
Inspección profunda de parquets de la run 20260602
"""
import pathlib, pandas as pd, numpy as np

pqs = list(pathlib.Path('g:/Mi unidad/ia/luna_v2/data/runs').rglob('WFB_20260602*/*/oos_trades.parquet'))
pqs = sorted(pqs, key=lambda p: p.stat().st_mtime)

print(f'Total parquets encontrados: {len(pqs)}')
print()

for pq in pqs:
    df = pd.read_parquet(pq)
    path_label = f'{pq.parts[-4]}/{pq.parts[-3]}/{pq.parts[-2]}'
    print(f'=== {path_label} ===')
    print(f'  N={len(df)}')
    if len(df) == 0:
        print('  VACIO')
        continue

    # Rango de fechas
    for col in ['entry_time', 'exit_time', 'timestamp']:
        if col in df.columns:
            print(f'  {col}: {df[col].min()} -> {df[col].max()}')
            break
    if df.index.name in ['entry_time', 'timestamp']:
        print(f'  index ({df.index.name}): {df.index.min()} -> {df.index.max()}')

    # Regimenes
    if 'hmm_regime' in df.columns:
        print(f'  regimes: {df.hmm_regime.value_counts().to_dict()}')

    # Retornos
    wr = (df['return_pct'] > 0).mean()
    print(f'  WR={wr:.1%} | return_pct: min={df.return_pct.min():.5f} max={df.return_pct.max():.5f} mean={df.return_pct.mean():.5f}')

    # xgb_prob
    if 'xgb_prob_cal' in df.columns:
        print(f'  xgb_prob_cal: min={df.xgb_prob_cal.min():.3f} mean={df.xgb_prob_cal.mean():.3f} max={df.xgb_prob_cal.max():.3f}')
    print()
