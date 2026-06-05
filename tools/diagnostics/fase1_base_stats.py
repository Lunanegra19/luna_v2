"""
=============================================================================
FASE 1 — CARGA Y ESTADÍSTICAS BASE
Protocolo diagnóstico_cuantitativo.md — Obligatorio antes de formular hipótesis
=============================================================================
"""
import sys, json, warnings
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from collections import Counter

wfb_dir = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')

# ─── Carga estándar (patrón del protocolo) ───────────────────────────────────
all_dfs = []
for f in sorted(wfb_dir.glob('oos_trades_W*_seed*.parquet')):
    try:
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue
        parts = f.stem.split('_')
        df['seed']   = int(next(p.replace('seed','') for p in parts if p.startswith('seed')))
        df['window'] = next(p for p in parts if p.startswith('W'))
        all_dfs.append(df)
    except Exception as e:
        print(f'[ERROR carga] {f.name}: {e}')

df_all = pd.concat(all_dfs, ignore_index=True)
N = len(df_all)

SEP = '═'*72

# ─── Estadísticas base (protocolo §FASE1) ────────────────────────────────────
print(SEP)
print('FASE 1 — ESTADÍSTICAS BASE COMPLETAS')
print(SEP)

def base_stats(df, label=''):
    n    = len(df)
    if n == 0:
        return
    ret  = df['return_pct'].dropna()
    wins = ret[ret > 0]
    loss = ret[ret <= 0]
    wr   = float((ret > 0).mean())
    aw   = float(wins.mean())   if len(wins) > 0 else float('nan')
    al   = float(loss.mean())   if len(loss) > 0 else float('nan')
    pl   = abs(aw/al)           if not np.isnan(aw) and al != 0 else float('nan')
    ev   = wr * aw + (1-wr) * al if not np.isnan(aw) and not np.isnan(al) else float('nan')
    pcts = ret.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).values

    print(f'\n[{label}] N={n}')
    print(f'  WR={wr*100:.2f}% | AvgWin={aw*100:+.4f}% | AvgLoss={al*100:+.4f}% | P/L={pl:.3f} | EV={ev*100:+.5f}%')
    print(f'  Ret: mean={ret.mean()*100:+.4f}% std={ret.std()*100:.4f}%')
    print(f'  Percentiles [p10,p25,p50,p75,p90]: {[f"{x*100:+.4f}%" for x in pcts]}')
    # Calibración
    if 'xgb_prob' in df.columns and 'xgb_prob_cal' in df.columns:
        delta = df['xgb_prob_cal'] - df['xgb_prob']
        q_c   = df['xgb_prob_cal'].quantile([0.10,0.25,0.50,0.75,0.90]).values
        print(f'  Δcal: mean={delta.mean():+.4f} std={delta.std():.4f} | '
              f'xgb_prob_cal pcts: {[f"{x:.4f}" for x in q_c]}')
    return dict(n=n, wr=wr, aw=aw, al=al, pl=pl, ev=ev, mean=ret.mean(), std=ret.std())

# Global
base_stats(df_all, 'GLOBAL')

# Por ventana
for w in ['W1','W2','W3']:
    base_stats(df_all[df_all['window']==w], f'Ventana {w}')

print()
print(SEP)
print('FASE 1 — ESTADÍSTICAS DE VARIABLES DE INTERÉS PARA HIPÓTESIS')
print(SEP)

# Variable xgb_prob_cal: distribución completa
cal = df_all['xgb_prob_cal'].dropna()
raw = df_all['xgb_prob'].dropna()
print(f'\nxgb_prob_raw: mean={raw.mean():.4f} std={raw.std():.4f} '
      f'min={raw.min():.4f} max={raw.max():.4f}')
print(f'xgb_prob_cal: mean={cal.mean():.4f} std={cal.std():.4f} '
      f'min={cal.min():.4f} max={cal.max():.4f}')

# OOD
ood = df_all['ood_kl_distance'].dropna()
print(f'\nood_kl_distance: mean={ood.mean():.4f} std={ood.std():.4f} '
      f'p25={ood.quantile(0.25):.4f} p75={ood.quantile(0.75):.4f}')

# seeds con trades
seeds_ok = sorted(df_all['seed'].unique())
print(f'\nSeeds con trades: {len(seeds_ok)} → {seeds_ok}')

# Distribución de N trades por seed×ventana
counts = df_all.groupby(['seed','window']).size().reset_index(name='n_trades')
print(f'\nDistribución N trades por (seed, window):')
print(f'  mean={counts.n_trades.mean():.1f} std={counts.n_trades.std():.1f} '
      f'min={counts.n_trades.min()} max={counts.n_trades.max()}')
# Detección de clones (mismo N)
n_50_51 = counts[counts['n_trades'].isin([50,51])]
print(f'  Pares con N=50 o N=51: {len(n_50_51)} (posibles clones)')

print()
print('[FASE 1 COMPLETADA] — Datos listos para formulación de hipótesis')
