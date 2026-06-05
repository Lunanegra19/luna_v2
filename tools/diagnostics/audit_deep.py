import pandas as pd
import numpy as np
from pathlib import Path

dfs = [pd.read_parquet(f) for f in sorted(Path('data/reports/wfb').glob('oos_trades_W*_seed88611.parquet'))]
df = pd.concat(dfs, ignore_index=True)

print('=== RAW vs CALIBRADA ===')
raw_mean = df['xgb_prob'].mean()
wr_real = df['is_win'].mean()
gap_raw = raw_mean - wr_real
print(f'XGB raw prob media:    {raw_mean:.3f}')
if 'xgb_prob_cal' in df.columns:
    cal_mean = df['xgb_prob_cal'].mean()
    gap_cal = cal_mean - wr_real
    print(f'XGB calibrada media:   {cal_mean:.3f}')
    print(f'Gap calibrada:         {gap_cal:+.3f}')
print(f'WR real:               {wr_real:.3f}')
print(f'Gap raw:               {gap_raw:+.3f}  <- SOBRECONFIANZA DEL MODELO RAW')
print()

print('=== KELLY FRACTION POR REGIMEN HMM ===')
if 'hmm_regime' in df.columns and 'kelly_fraction_used' in df.columns:
    for reg, gdf in df.groupby('hmm_regime'):
        kf = gdf['kelly_fraction_used'].mean()*100
        wrate = gdf['is_win'].mean()*100
        n = len(gdf)
        print(f'  {reg}: kelly={kf:.2f}% | WR={wrate:.1f}% | n={n}')
print('  -> Kelly NO varia por incertidumbre del regimen? Confirmar.')
print()

print('=== DURACION REAL DE TRADES (barrera vertical en vivo) ===')
if 'entry_time' in df.columns and 'exit_time' in df.columns:
    df['entry_dt'] = pd.to_datetime(df['entry_time'])
    df['exit_dt'] = pd.to_datetime(df['exit_time'])
    df['duracion_h'] = (df['exit_dt'] - df['entry_dt']).dt.total_seconds() / 3600
    dur_mean = df['duracion_h'].mean()
    dur_std = df['duracion_h'].std()
    dur_min = df['duracion_h'].min()
    dur_max = df['duracion_h'].max()
    print(f'  Duracion media: {dur_mean:.1f}h')
    print(f'  Duracion std:   {dur_std:.1f}h')
    print(f'  Duracion rango: [{dur_min:.1f}h - {dur_max:.1f}h]')
    wins = df[df['is_win']==1]['duracion_h']
    losses = df[df['is_win']==0]['duracion_h']
    print(f'  Duracion WINS:  {wins.mean():.1f}h (std={wins.std():.1f}h)')
    print(f'  Duracion LOSS:  {losses.mean():.1f}h (std={losses.std():.1f}h)')
    print(f'  -> TBM vol-adjusted implica mayor variacion en std y rango que 72-96H fijos')
    pct_96h = (df['duracion_h'] == 96).mean() * 100
    pct_72h = (df['duracion_h'] == 72).mean() * 100
    print(f'  Trades exactamente 96H: {pct_96h:.1f}%')
    print(f'  Trades exactamente 72H: {pct_72h:.1f}%')
