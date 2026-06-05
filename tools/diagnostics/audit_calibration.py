import pandas as pd
import numpy as np
from pathlib import Path

all_files = list(Path('data/reports/wfb').glob('oos_trades_W*_seed88611.parquet'))
all_dfs = []
for f in sorted(all_files):
    df = pd.read_parquet(f)
    df['window'] = f.stem.split('_')[2]
    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)
print('=== ANALISIS DE CALIBRACION OOS (TRADES REALES) ===')
print(f'Total trades: {len(df_all)}')
wr = df_all['is_win'].mean()
print(f'Win Rate real: {wr*100:.1f}%')
print()
print('Columnas disponibles:', list(df_all.columns))
print()

# Distribucion de probabilidades XGBoost usadas
if 'xgb_prob' in df_all.columns:
    xgb_avg = df_all['xgb_prob'].mean()
    xgb_std = df_all['xgb_prob'].std()
    xgb_min = df_all['xgb_prob'].min()
    xgb_max = df_all['xgb_prob'].max()
    print(f'XGB Prob media: {xgb_avg:.3f}')
    print(f'XGB Prob std:   {xgb_std:.3f}')
    print(f'XGB Prob rango: [{xgb_min:.3f} - {xgb_max:.3f}]')
    overconf = xgb_avg - wr
    print(f'GAP (avg_prob - WR_real): {overconf:+.3f}  <- OVERCONFIDENCE INDICATOR')
    print()
    print('=== RELIABILITY BINS (xgb_prob -> WR_real) ===')
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (df_all['xgb_prob'] >= lo) & (df_all['xgb_prob'] < hi)
        n = int(mask.sum())
        if n > 0:
            bin_wr = df_all.loc[mask, 'is_win'].mean()
            avg_p = df_all.loc[mask, 'xgb_prob'].mean()
            gap = avg_p - bin_wr
            print(f'  [{lo:.1f}-{hi:.1f}] n={n:3d} | WR={bin_wr*100:.1f}% | avg_prob={avg_p:.3f} | gap={gap:+.3f}')

print()
if 'kelly_fraction_used' in df_all.columns:
    kf_mean = df_all['kelly_fraction_used'].mean()*100
    kf_std = df_all['kelly_fraction_used'].std()*100
    print(f'=== KELLY FRACTION ===')
    print(f'Media: {kf_mean:.2f}%')
    print(f'Std:   {kf_std:.2f}%')
    print()

if 'hmm_regime' in df_all.columns:
    print('=== TRADES POR REGIMEN HMM ===')
    print(df_all.groupby('hmm_regime')['is_win'].agg(['count','mean']).rename(columns={'count':'n','mean':'WR'}))
    print()

# Analisis por ventana
print('=== EVOLUCION POR VENTANA ===')
for w, gdf in df_all.groupby('window'):
    wwin = gdf['is_win'].mean()
    n = len(gdf)
    ret_mean = gdf['return_pct'].mean() if 'return_pct' in gdf.columns else 0
    print(f'{w}: n={n} | WR={wwin*100:.1f}% | MeanRet={ret_mean:.4f}%')
