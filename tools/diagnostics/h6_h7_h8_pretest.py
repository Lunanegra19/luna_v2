"""
h6_h7_h8_rigorous_pretest.py
============================
Revision en profundidad de H6, H7, H8 antes de cualquier implementacion.
Regla: estudiar en profundidad, probar, confirmar que son mejoras reales.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\reports\wfb')
SEP = "=" * 70

def load_all():
    dfs = []
    for f in sorted(DATA.glob('oos_trades_W*_seed*.parquet')):
        stem = f.stem; wid = stem.split('_')[2]; seed = int(stem.split('_seed')[1])
        df = pd.read_parquet(f)
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index, utc=True)
        df['_seed'] = seed; df['_window'] = wid
        dfs.append(df)
    return pd.concat(dfs).sort_index()

def build_portfolio(df_all, min_seeds=3, embargo_h=96):
    df_all = df_all.copy()
    df_all['bucket'] = df_all.index.floor('2h')
    bkt = df_all.groupby('bucket')['_seed'].nunique()
    df_all['consensus'] = df_all['bucket'].map(bkt)
    dfc = df_all[df_all['consensus'] >= min_seeds]
    if len(dfc) == 0:
        return pd.DataFrame()
    dfp = dfc.groupby('bucket').agg(
        return_pct=('return_pct','mean'),
        is_win=('is_win','max'),
        window=('_window','first'),
        consensus=('consensus','first'),
        signal_CUTOFF = ('signal_threshold','mean'),
        xgb_prob_cal=('xgb_prob_cal','mean'),
        alpha_trigger=('alpha_trigger','first'),
    ).sort_index()
    sel, lt = [], None
    for ts in dfp.index:
        if lt is None or (ts - lt).total_seconds() / 3600 >= embargo_h:
            sel.append(ts); lt = ts
    return dfp.loc[sel].copy()

df = load_all()

# =============================================================================
print(SEP)
print("H6 — REVISION EN PROFUNDIDAD: Subir consensus de 3 a 4")
print(SEP)

print("\n[H6.1] Por que consensus 2,3,4 dan exactamente el mismo portfolio?")
print("       Inspeccionando la dominancia del embargo 96H:\n")

for ms in [2, 3, 4, 5, 6]:
    df['bucket'] = df.index.floor('2h')
    bkt = df.groupby('bucket')['_seed'].nunique()
    df['consensus'] = df['bucket'].map(bkt)
    dfc = df[df['consensus'] >= ms]
    n_buckets = dfc.groupby('bucket').ngroups
    dfp = dfc.groupby('bucket').agg(return_pct=('return_pct','mean'), is_win=('is_win','max')).sort_index()
    sel, lt = [], None
    for ts in dfp.index:
        if lt is None or (ts - lt).total_seconds() / 3600 >= 96:
            sel.append(ts); lt = ts
    print(f"  consensus>={ms}: {n_buckets} buckets candidatos -> {len(sel)} seleccionados (embargo 96H)")

print()
print("[H6.2] Buckets eliminados al subir de 3->4:")
df['bucket'] = df.index.floor('2h')
bkt = df.groupby('bucket')['_seed'].nunique()
df['consensus'] = df['bucket'].map(bkt)

dfc3 = df[df['consensus'] >= 3].groupby('bucket').agg(
    return_pct=('return_pct','mean'), is_win=('is_win','max'),
    window=('_window','first'), consensus=('consensus','first')
).sort_index()
dfc4 = df[df['consensus'] >= 4].groupby('bucket').agg(
    return_pct=('return_pct','mean'), is_win=('is_win','max'),
    window=('_window','first'), consensus=('consensus','first')
).sort_index()

only_in_3 = dfc3.index.difference(dfc4.index)
print(f"  Buckets solo en consensus>=3 (n={len(only_in_3)}):")
for ts in only_in_3:
    row = dfc3.loc[ts]
    print(f"    {ts} | consensus={row['consensus']} | win={'W' if row['is_win'] else 'L'} "
          f"| ret={row['return_pct']*100:.4f}% | window={row['window']}")

print()
print("[H6.3] Esos buckets excluidos ¿llegarian al portfolio final tras embargo?")
# Reconstruir portfolio con embargo para ver cuales de esos buckets habrian pasado
port3 = build_portfolio(df, min_seeds=3, embargo_h=96)
port4 = build_portfolio(df, min_seeds=4, embargo_h=96)

port3_idx = set(port3.index)
port4_idx = set(port4.index)
diff_3to4 = port3_idx.symmetric_difference(port4_idx)
print(f"  Portfolio consensus>=3: {len(port3)} trades")
print(f"  Portfolio consensus>=4: {len(port4)} trades")
print(f"  Diferencia (trades distintos): {len(diff_3to4)}")

if len(diff_3to4) == 0:
    print("\n  HALLAZGO CRITICO H6: Subir de 3->4 NO cambia ningun trade del portfolio.")
    print("  El embargo 96H ya filtra los buckets con consensus=3.")
    print("  H6 es una no-operacion: DESCARTAR como mejora.")
else:
    print(f"\n  Trades que cambian:")
    for ts in sorted(diff_3to4):
        if ts in port3_idx:
            print(f"    ELIMINADO: {ts} ret={port3.loc[ts,'return_pct']*100:.4f}% win={port3.loc[ts,'is_win']}")
        else:
            print(f"    NUEVO:     {ts} ret={port4.loc[ts,'return_pct']*100:.4f}% win={port4.loc[ts,'is_win']}")

# =============================================================================
print()
print(SEP)
print("H7 — REVISION EN PROFUNDIDAD: Desactivar alpha_golden_score")
print(SEP)

print("\n[H7.1] Muestra completa de trades con alpha_golden_score:")
ags = df[df['alpha_trigger'].str.contains('alpha_golden_score', na=False)].copy()
print(f"  Total trades: {len(ags)}")
print(f"  Por ventana: W1={len(ags[ags['_window']=='W1'])} W3={len(ags[ags['_window']=='W3'])} W4={len(ags[ags['_window']=='W4'])}")
print()
print(f"  {'Fecha':<22} {'Ventana':>7} {'Seed':>7} {'Win':>5} {'ret%':>9} {'consensus':>10}")
for ts, row in ags.iterrows():
    print(f"  {str(ts)[:19]:<22} {row['_window']:>7} {int(row['_seed']):>7}  "
          f"{'W' if row['is_win'] else 'L':>4} {row['return_raw']*100:>9.4f}% "
          f"{int(row['consensus']) if not pd.isna(row['consensus']) else '?':>10}")

print()
print("[H7.2] Test de significancia binomial:")
n_ags = len(ags); wins_ags = ags['is_win'].sum()
from scipy.stats import binomtest
p_binom = binomtest(int(wins_ags), int(n_ags), 0.5, alternative='less').pvalue
print(f"  {wins_ags} wins en {n_ags} trades | p(binom, H0: WR>=50%): {p_binom:.4f}")
print(f"  -> {'SIGNIFICATIVO (p<0.05) — hay evidencia de que este trigger destruye valor' if p_binom < 0.05 else 'No significativo — muestra insuficiente'}")

print()
print("[H7.3] Analisis de si son los MISMOS timestamps o seeds distintas:")
ags_timestamps = ags.index.unique()
print(f"  Timestamps unicos con golden_score: {len(ags_timestamps)}")
print(f"  Seeds distintas con golden_score: {ags['_seed'].nunique()}")
print(f"  ¿Son los mismos momentos en diferentes seeds?")
for ts in sorted(ags_timestamps):
    seeds_ts = ags[ags.index == ts]['_seed'].tolist()
    print(f"    {ts}: seeds={[int(s) for s in seeds_ts]}")

print()
print("[H7.4] ¿Donde se controla alpha_golden_score en el pipeline?")
print("  Buscar en settings.yaml la habilitacion de este trigger...")

import subprocess
result = subprocess.run(
    ['powershell','-Command','Select-String -Path luna\\* -Pattern golden_score -Recurse | Select-Object -First 20'],
    capture_output=True, text=True, cwd=r'g:\Mi unidad\ia\luna_v2'
)
for line in (result.stdout or result.stderr).splitlines()[:20]:
    print(f"  {line}")

print()
print("[H7.5] ¿Llega alguno de estos 7 trades al portfolio ensemble?")
port3_idx = set(port3.index)
ags_buckets = (ags.index.floor('2h'))
overlap = [b for b in ags_buckets if b in port3_idx]
print(f"  Buckets golden_score que llegaron al portfolio: {len(overlap)}")
for b in overlap:
    print(f"    {b}: ret={port3.loc[b,'return_pct']*100:.4f}% win={port3.loc[b,'is_win']}")
if len(overlap) == 0:
    print("  HALLAZGO H7: Estos 7 trades NO llegan al portfolio ensemble.")
    print("  El embargo/consenso ya los filtra. Desactivar golden_score en settings")
    print("  solo afecta a la ejecucion en produccion single-seed, NO al ensemble actual.")

# =============================================================================
print()
print(SEP)
print("H8 — REVISION EN PROFUNDIDAD: Subir signal_threshold")
print(SEP)

print("\n[H8.1] Distribucion de signal_threshold en los 545 trades:")
st = df['signal_threshold']
print(f"  min={st.min():.4f} | p25={st.quantile(0.25):.4f} | median={st.median():.4f} "
      f"| p75={st.quantile(0.75):.4f} | max={st.max():.4f}")
print(f"  Todos iguales? std={st.std():.6f} | nunique={st.nunique()}")
print()
print(f"  Primeros 10 valores unicos de signal_threshold:")
for v in sorted(st.unique())[:10]:
    n_with = (st == v).sum()
    wr_with = df[st == v]['is_win'].mean() * 100
    print(f"    {v:.6f}: {n_with:>4} trades WR={wr_with:.1f}%")

print()
print("[H8.2] Distribucion por ventana — ¿el threshold cambia entre ventanas?")
for win in ['W1','W3','W4']:
    dw = df[df['_window'] == win]['signal_threshold']
    if len(dw) == 0: continue
    print(f"  {win}: min={dw.min():.4f} med={dw.median():.4f} max={dw.max():.4f} std={dw.std():.4f}")

print()
print("[H8.3] Correlacion detallada threshold vs retorno por ventana:")
for win in ['GLOBAL','W3','W4']:
    if win == 'GLOBAL':
        d = df
    else:
        d = df[df['_window'] == win]
    rho, p = stats.spearmanr(d['signal_threshold'], d['return_raw'])
    rho_w, p_w = stats.spearmanr(d['signal_threshold'], d['is_win'].astype(float))
    print(f"  {win}: Spearman(thr, ret_raw)=rho={rho:.4f} p={p:.4f} | "
          f"Spearman(thr, is_win)=rho={rho_w:.4f} p={p_w:.4f}")

print()
print("[H8.4] Simulacion: ¿que pasaria si excluimos trades con threshold < X?")
print(f"  {'MinThr':>8} {'N_trades':>9} {'WR%':>7} {'RetMed%':>10} {'RetTot%':>10} {'Alcanza30?':>11}")
for thr_pct in [0.0, 0.50, 0.52, 0.55, 0.57, 0.60, 0.63, 0.65, 0.70]:
    filtered = df[df['signal_threshold'] >= thr_pct]
    n = len(filtered)
    if n == 0: continue
    wr = filtered['is_win'].mean() * 100
    ret_med = filtered['return_raw'].mean() * 100
    ret_tot = filtered['return_raw'].sum() * 100
    ok30 = "SI" if n >= 30 else "NO"
    flag = " <- BASE" if thr_pct == 0.0 else ""
    print(f"  {thr_pct:>8.2f} {n:>9} {wr:>7.1f}% {ret_med:>10.4f}% {ret_tot:>10.4f}% {ok30:>11}{flag}")

print()
print("[H8.5] El threshold se puede controlar externamente?")
print("  Buscando donde se genera signal_threshold en el codigo...")
result2 = subprocess.run(
    ['powershell','-Command','Select-String -Path luna\\* -Pattern signal_threshold -Recurse -Include *.py | Select-Object -First 25'],
    capture_output=True, text=True, cwd=r'g:\Mi unidad\ia\luna_v2'
)
for line in (result2.stdout or result2.stderr).splitlines()[:25]:
    print(f"  {line}")

print()
print("[H8.6] Riesgo de overfitting al subir threshold:")
print("  El threshold se calcula POR TRADE (no es parametro global).")
print("  Si subimos el threshold minimo, estamos haciendo post-hoc selection.")
print("  La pregunta clave: ¿el threshold es deterministico o estocástico?")
print("  Si deterministico: igual en re-run. Si estocástico: overfitting.")
print()
# ¿El threshold varia entre seeds para el mismo timestamp?
common_ts = df.groupby(df.index.floor('2h'))['signal_threshold'].std()
print(f"  STD de signal_threshold entre seeds en mismo bucket:")
print(f"  media={common_ts.mean():.6f} | max={common_ts.max():.6f}")
if common_ts.max() < 0.001:
    print("  -> DETERMINISTICO: el threshold es el mismo para todas las seeds en el mismo momento")
    print("     Esto confirma que es una propiedad del mercado/modelo, no del seed.")
else:
    print("  -> ESTOCASTICO: varia entre seeds. Riesgo de overfitting.")

print()
print(SEP)
print("VEREDICTOS PRELIMINARES")
print(SEP)
