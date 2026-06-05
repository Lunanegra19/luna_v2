import json, glob, os

print('[ERROR D] Investigacion MaxDD fraccion vs porcentaje')
print()

# Buscar todos los verdicts de la batch del 21 mayo
files = glob.glob(r'G:\Mi unidad\ia\luna_v2\data\reports\2026-05-21_*.json')
files.sort()
print(f'Verdicts encontrados: {len(files)}')
print()

for fpath in files:
    fname = os.path.basename(fpath)
    with open(fpath) as f:
        v = json.load(f)
    
    metrics = v.get('metrics', {})
    flags   = v.get('flags', {})
    sop     = v.get('sop_thresholds', {})
    
    mdd_pct     = metrics.get('max_drawdown_pct')
    pass_dd     = flags.get('pass_dd')
    max_dd_thr  = sop.get('max_drawdown_pct')
    approved    = v.get('deploy_approved')
    n_trades    = metrics.get('total_trades')
    
    if mdd_pct is not None:
        # Verificar si hay inconsistencia: max_dd > threshold pero pass_dd=True
        is_bug = (mdd_pct is not None and max_dd_thr is not None and 
                  mdd_pct > max_dd_thr and pass_dd == True)
        marker = '*** BUG ***' if is_bug else ''
        print(f'{fname[:60]}')
        print(f'  n_trades={n_trades} MaxDD={mdd_pct:.1f}% CUTOFF = {max_dd_thr}% pass_dd={pass_dd} approved={approved} {marker}')
    else:
        print(f'{fname[:60]} -> sin metrics.max_drawdown_pct')
    
print()
print('[ERROR D] Comprobando la logica de calculo de max_dd en statistical_audit.py...')
print()

# Simular el calculo con retornos de ejemplo
import numpy as np

# Caso 1: retornos como fraccion (correcto)
returns_frac = np.array([0.05, -0.10, 0.08, -0.12, 0.04])
cum = (1 + returns_frac).cumprod()
peaks = np.maximum.accumulate(cum)
dd_frac = float(abs(np.min((cum - peaks) / peaks)))
print(f'Retornos como fraccion: max_dd={dd_frac:.4f} ({dd_frac*100:.1f}%)')

# Caso 2: retornos como porcentaje (bug potencial)
returns_pct = np.array([5.0, -10.0, 8.0, -12.0, 4.0])
cum2 = (1 + returns_pct).cumprod()
peaks2 = np.maximum.accumulate(cum2)
dd_pct = float(abs(np.min((cum2 - peaks2) / peaks2)))
print(f'Retornos como porcentaje (bug): max_dd={dd_pct:.4f} ({dd_pct*100:.1f}%)')
print(f'  -> Si returns_pct se pasa como fraccion, max_dd seria ENORME (~{dd_pct:.0f})')
print()

# Verificar que tipo de retornos tienen los trades
import pandas as pd
import glob as glb

trade_files = glb.glob(r'G:\Mi unidad\ia\luna_v2\data\oos_trades\*seed1337*.parquet')
trade_files.sort()
if trade_files:
    tf = trade_files[-1]  # el mas reciente
    trades = pd.read_parquet(tf)
    print(f'Archivo trades: {os.path.basename(tf)}')
    print(f'Columnas: {list(trades.columns)}')
    if 'return_pct' in trades.columns:
        print(f'return_pct stats: min={trades.return_pct.min():.4f} max={trades.return_pct.max():.4f} mean={trades.return_pct.mean():.4f}')
        print(f'Tipo esperado: {"FRACCION (OK)" if abs(trades.return_pct.max()) < 5 else "POSIBLE PORCENTAJE (BUG)"}')
    if 'pnl' in trades.columns:
        print(f'pnl stats: min={trades.pnl.min():.4f} max={trades.pnl.max():.4f}')
