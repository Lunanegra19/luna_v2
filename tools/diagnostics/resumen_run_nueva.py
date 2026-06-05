"""
Resumen completo run nueva — todas las seeds disponibles
Métricas OOS por ventana + ensemble agregado
"""
import sys, pathlib, pandas as pd, numpy as np, datetime
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')

data   = pathlib.Path('g:/Mi unidad/ia/luna_v2/data')
runs   = data / 'runs'
cutoff = datetime.datetime(2026, 6, 1, 13, 26, 0)
SEP    = '─' * 70

# ── Descubrir todas las runs nuevas ───────────────────────────────────────
new_run_dirs = [d for d in runs.iterdir()
                if d.is_dir() and d.stat().st_ctime > cutoff.timestamp()
                and d.name.startswith('WFB_20260601')]

print(SEP)
print(f'Runs nuevas encontradas: {len(new_run_dirs)}')
for d in sorted(new_run_dirs, key=lambda x: x.name):
    seed_dirs = list(d.glob('seed*'))
    s = seed_dirs[0].name.replace('seed','') if seed_dirs else '?'
    print(f'  {d.name} | seed={s}')
print()

# ── Cargar todos los OOS trades ───────────────────────────────────────────
all_records = []
VENTANAS = ['W1', 'W2', 'W3', 'W4', 'W5']

for rd in new_run_dirs:
    for sd in rd.glob('seed*'):
        seed = int(sd.name.replace('seed', ''))
        for w in VENTANAS:
            fp = sd / w / 'oos_trades.parquet'
            if not fp.exists():
                continue
            try:
                df = pd.read_parquet(fp)
                if len(df) == 0:
                    continue
                df['seed']    = seed
                df['ventana'] = w
                all_records.append(df)
            except Exception as e:
                print(f'  ERROR {sd.name}/{w}: {e}')

if not all_records:
    print('Sin datos OOS disponibles.')
    sys.exit(0)

df_all = pd.concat(all_records, ignore_index=True)
print(f'Total filas OOS cargadas: {len(df_all)}')
print(f'Seeds con datos: {sorted(df_all["seed"].unique())}')
print(f'Ventanas con datos: {sorted(df_all["ventana"].unique())}')
print()

# ── Función de métricas ───────────────────────────────────────────────────
def stats(df):
    if len(df) == 0:
        return None
    v   = df['return_pct'].values
    eq  = np.cumsum(v)
    n   = len(v)
    wr  = (v > 0).sum() / n * 100
    ret = float(v.sum() * 100)
    dd  = float((eq - np.maximum.accumulate(eq)).min() * 100)
    sr  = float(v.mean() / (v.std() + 1e-9) * np.sqrt(252)) if n > 1 else 0.0
    calmar = abs(ret / dd) if dd < 0 else 0.0
    return {'n': n, 'wr': wr, 'ret': ret, 'dd': dd, 'sr': sr, 'calmar': calmar}

# ── Tabla 1: Por ventana (agregado de todas las seeds) ───────────────────
print(SEP)
print('TABLA 1 — Métricas OOS por Ventana (agregado todas las seeds)')
print(SEP)
print(f'  {"W":^3} | {"Seeds":>5} | {"N":>5} | {"WR%":>5} | {"ret%":>7} | {"MaxDD%":>7} | {"Sharpe":>6} | {"Calmar":>6}')
print('  ' + '─' * 60)

for w in VENTANAS:
    dfw = df_all[df_all['ventana'] == w]
    if len(dfw) == 0:
        continue
    n_seeds = dfw['seed'].nunique()
    m = stats(dfw)
    print(f'  {w:^3} | {n_seeds:>5} | {m["n"]:>5} | {m["wr"]:>4.1f}% | {m["ret"]:>+6.3f}% | {m["dd"]:>+6.3f}% | {m["sr"]:>6.2f} | {m["calmar"]:>6.2f}')

# ── Tabla 2: Por seed (W1-W5 concatenado) ────────────────────────────────
print()
print(SEP)
print('TABLA 2 — Métricas OOS por Seed (todas las ventanas)')
print(SEP)
print(f'  {"Seed":>8} | {"Ws":^6} | {"N":>5} | {"WR%":>5} | {"ret%":>7} | {"MaxDD%":>7} | {"Sharpe":>6} | {"Calmar":>6}')
print('  ' + '─' * 65)

for seed in sorted(df_all['seed'].unique()):
    dfs = df_all[df_all['seed'] == seed]
    ws  = ','.join(sorted(dfs['ventana'].unique()))
    m   = stats(dfs)
    print(f'  {seed:>8} | {ws:^6} | {m["n"]:>5} | {m["wr"]:>4.1f}% | {m["ret"]:>+6.3f}% | {m["dd"]:>+6.3f}% | {m["sr"]:>6.2f} | {m["calmar"]:>6.2f}')

# ── Tabla 3: Ensemble total ───────────────────────────────────────────────
print()
print(SEP)
print('TABLA 3 — Ensemble Total (todas las seeds, todas las ventanas)')
print(SEP)
m = stats(df_all)
n_seeds = df_all['seed'].nunique()
n_wins = int(m["n"] * m["wr"] / 100)
print(f'  Seeds completadas : {n_seeds}')
print(f'  Total trades OOS  : {m["n"]} ({n_wins} wins / {m["n"]-n_wins} losses)')
print(f'  Win Rate          : {m["wr"]:.1f}%')
print(f'  Retorno acumulado : {m["ret"]:+.3f}%')
print(f'  Max Drawdown      : {m["dd"]:+.3f}%')
print(f'  Sharpe anualizado : {m["sr"]:+.2f}')
print(f'  Calmar ratio      : {m["calmar"]:.2f}')
print()

# ── Breakdown por dirección ───────────────────────────────────────────────
if 'direction' in df_all.columns:
    print('  Por dirección:')
    for d in df_all['direction'].dropna().unique():
        dfd = df_all[df_all['direction'] == d]
        m2  = stats(dfd)
        print(f'    {d:<8}: N={m2["n"]:>4} | WR={m2["wr"]:>4.1f}% | ret={m2["ret"]:>+6.3f}% | SR={m2["sr"]:>+5.2f}')

# ── W4 check: FIX-BEAR-SKIP-01 ───────────────────────────────────────────
print()
print(SEP)
print('DIAGNÓSTICO W4 — Verificación FIX-BEAR-SKIP-01')
print(SEP)
dfw4 = df_all[df_all['ventana'] == 'W4']
if len(dfw4) == 0:
    print('  W4: sin trades (SKIP activado para todos los seeds)')
else:
    print(f'  W4: {len(dfw4)} trades generados en {dfw4["seed"].nunique()} seeds')
    if 'direction' in dfw4.columns:
        print('  Direcciones en W4:', dfw4['direction'].value_counts().to_dict())
print()
