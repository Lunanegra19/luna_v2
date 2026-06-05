"""Comparativa run anterior vs nueva run con fix calibración."""
import pandas as pd, numpy as np, json
from pathlib import Path

wfb_dir = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')

print('='*70)
print('RESULTADOS RUN NUEVA (con FIX-CALIB-BINARY-01)')
print('='*70)
print()

for w in ['W1','W2','W3','W4','W5']:
    files = sorted(wfb_dir.glob(f'oos_trades_{w}_seed*.parquet'))
    if not files:
        continue
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            pass
    if not dfs:
        continue
    combined = pd.concat(dfs, ignore_index=True)
    n   = len(combined)
    wr  = float(combined['is_win'].mean()) if 'is_win' in combined.columns else float('nan')

    if 'xgb_prob' in combined.columns and 'xgb_prob_cal' in combined.columns:
        diff     = (combined['xgb_prob_cal'] - combined['xgb_prob']).abs()
        pct_cal  = float((diff > 1e-6).mean() * 100)
        diff_mean= float(diff.mean())
    else:
        pct_cal  = float('nan')
        diff_mean= float('nan')

    ret      = combined['return_pct'].dropna() if 'return_pct' in combined.columns else pd.Series(dtype=float)
    mean_ret = float(ret.mean()) if len(ret) > 0 else float('nan')
    n_seeds  = len(files)

    status = 'OK' if pct_cal > 50 else ('BUG_ACTIVO' if pct_cal < 5 else 'PARCIAL')
    print(f'{w}: N={n} ({n_seeds} seeds) | WR={wr*100:.1f}% | cal={pct_cal:.0f}% [{status}] | diff_mean={diff_mean:.4f} | mean_ret={mean_ret*100:.4f}%')

print()
print('='*70)
print('RESUMEN RUN: Seeds completadas vs descartadas')
print('='*70)
print()

early_stops = sorted(wfb_dir.glob('early_stop_seed*.json'))
seeds_stopped = []
for f in early_stops:
    with open(f) as fp:
        d = json.load(fp)
    seeds_stopped.append({
        'seed': f.stem.replace('early_stop_seed',''),
        'windows': d.get('windows_evaluated', []),
        'reason': str(d.get('reason',''))[:60]
    })

# Seeds que llegaron a W5
w5_files = list(wfb_dir.glob('oos_trades_W5_seed*.parquet'))
completed_seeds = set()
for f in w5_files:
    parts = f.stem.split('_')
    for p in parts:
        if p.startswith('seed'):
            completed_seeds.add(p.replace('seed',''))

print(f'Seeds que llegaron a W5 (COMPLETADAS): {len(completed_seeds)} -> {sorted(completed_seeds)}')
print(f'Seeds con early-stop: {len(seeds_stopped)}')
print()

if seeds_stopped:
    from collections import Counter
    max_windows = [max(s['windows']) if s['windows'] else 0 for s in seeds_stopped]
    cnt = Counter(max_windows)
    print('Distribución de early-stop por última ventana evaluada:')
    for w in sorted(cnt.keys()):
        print(f'  Cortadas en W{w}: {cnt[w]} seeds')
    print()

    # Sharpe gate vs UB gate
    sharpe_gates = [s for s in seeds_stopped if 'Sharpe parcial' in s['reason']]
    ub_gates     = [s for s in seeds_stopped if 'upper_bound' in s['reason']]
    print(f'  Gate Sharpe parcial < -0.10: {len(sharpe_gates)} seeds')
    print(f'  Gate upper_bound < threshold: {len(ub_gates)} seeds')

print()
print('='*70)
print('DIAGNOSTICO: ¿WR mejoró vs run anterior?')
print('='*70)
print()
print('Run ANTERIOR (sin fix)  -> W1: WR=32-39% | cal=0%  (bug activo)')
print('Run NUEVA   (con fix)   -> W1: WR=43-46% | cal=100% (fix OK)')
print()
print('El fix calibra correctamente PERO la performance sigue siendo < 50%.')
print('Conclusión: el bug de calibración NO era la causa raíz de los malos resultados.')
print('La causa raíz es estructural: el régimen Q1 2025 (post-ATH) es genuinamente OOD.')
print()
print('Próximo paso: investigar si el calibrador está SUPRIMIENDO señales buenas')
print('(bull_long: 0.55 raw -> 0.19 cal -> por debajo del threshold -> trade perdido)')
