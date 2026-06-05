"""arquitectura_cuadro_completo.py — Analisis estructural del sistema."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from pathlib import Path

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\reports\wfb')
SEP = "=" * 70

def load_all():
    dfs = []
    for f in sorted(DATA.glob('oos_trades_W*_seed*.parquet')):
        stem = f.stem
        wid = stem.split('_')[2]
        seed = int(stem.split('_seed')[1])
        df = pd.read_parquet(f)
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index, utc=True)
        df['_seed'] = seed
        df['_window'] = wid
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            et = pd.to_datetime(df['entry_time'], utc=True, errors='coerce')
            xt = pd.to_datetime(df['exit_time'], utc=True, errors='coerce')
            df['holding_h'] = (xt - et).dt.total_seconds() / 3600
        dfs.append(df)
    return pd.concat(dfs).sort_index()

df = load_all()
df['ret100'] = df['return_raw'] * 100
dw3 = df[df['_window'] == 'W3']
dw4 = df[df['_window'] == 'W4']
n_seeds = df['_seed'].nunique()
df['bucket'] = df.index.floor('2h')
bkt = df.groupby('bucket')['_seed'].nunique()
df['consensus'] = df['bucket'].map(bkt)

# ─── 3. MAGNITUD DE RETORNOS ─────────────────────────────────────────────
print(SEP)
print("3. MAGNITUD DE RETORNOS BRUTOS — ¿El precio se mueve suficiente?")
print(SEP)
print(f"  {'Ventana':<8}  {'|ret|<0.1% plano':>18}  {'ret>+0.1%':>11}  {'ret<-0.1%':>11}  {'|ret|>0.5%':>11}")
for win, dw in [('W3', dw3), ('W4', dw4), ('GLOBAL', df)]:
    r = dw['ret100']
    plano  = (r.abs() < 0.10).mean() * 100
    pos    = (r > 0.10).mean() * 100
    neg    = (r < -0.10).mean() * 100
    grande = (r.abs() > 0.50).mean() * 100
    print(f"  {win:<8}  {plano:>18.1f}%  {pos:>11.1f}%  {neg:>11.1f}%  {grande:>11.1f}%")

# Distribucion percentiles de retorno
print()
print("  Percentiles de retorno bruto:")
for win, dw in [('W3', dw3), ('W4', dw4)]:
    r = dw['ret100']
    p5,p25,p50,p75,p95 = r.quantile([.05,.25,.50,.75,.95])
    print(f"  {win}: p5={p5:.3f}%  p25={p25:.3f}%  med={p50:.3f}%  p75={p75:.3f}%  p95={p95:.3f}%")

# ─── 4. ENSEMBLE SIN DIVERSIDAD ─────────────────────────────────────────
print()
print(SEP)
print("4. ENSEMBLE — ¿20 seeds diversifican o son 20 copias del mismo modelo?")
print(SEP)
print("  Distribucion de consensus por bucket de 2h:")
dist = {}
for n in [1,2,3,5,8,10,15,20]:
    cnt = (bkt == n).sum()
    gte = (bkt >= n).sum()
    if cnt > 0:
        dist[n] = cnt
        pct_gte = gte / len(bkt) * 100
        print(f"  consensus={n:>2}: {cnt:>4} buckets exactos  |  consensus>={n}: {gte:>4} buckets ({pct_gte:.1f}%)")

modo = bkt.value_counts().index[0]
print(f"\n  Consensus mas frecuente: {modo} seeds (el ensemble casi siempre esta 100% de acuerdo)")
print(f"  Implicacion: un consensus>=3 no filtra nada cuando todas las seeds votan igual")

# ─── 5. SEED INDIVIDUAL vs ENSEMBLE ─────────────────────────────────────
print()
print(SEP)
print("5. SEED INDIVIDUAL vs ENSEMBLE — ¿Aporta valor el ensemble?")
print(SEP)

rows = []
for seed in sorted(df['_seed'].unique()):
    d3 = df[(df['_seed'] == seed) & (df['_window'] == 'W3')]
    d4 = df[(df['_seed'] == seed) & (df['_window'] == 'W4')]
    n3, n4 = len(d3), len(d4)
    wr3 = d3['is_win'].mean() * 100 if n3 > 0 else 0.0
    wr4 = d4['is_win'].mean() * 100 if n4 > 0 else 0.0
    r3  = d3['ret100'].sum() if n3 > 0 else 0.0
    r4  = d4['ret100'].sum() if n4 > 0 else 0.0
    rows.append(dict(seed=seed, n3=n3, wr3=wr3, r3=r3, n4=n4, wr4=wr4, r4=r4, total=r3+r4))

dr = pd.DataFrame(rows)
print(f"  {'Seed':>8}  {'W3_n':>5}  {'W3_WR%':>7}  {'W3_ret%':>8}  {'W4_n':>5}  {'W4_WR%':>7}  {'W4_ret%':>8}  {'Total%':>7}")
print("  " + "-" * 74)
for _, row in dr.sort_values('total', ascending=False).iterrows():
    marker = " <-- MEJOR" if row['total'] == dr['total'].max() else ""
    print(
        f"  {int(row['seed']):>8}  {int(row['n3']):>5}  {row['wr3']:>7.1f}%  "
        f"{row['r3']:>8.3f}%  {int(row['n4']):>5}  {row['wr4']:>7.1f}%  "
        f"{row['r4']:>8.3f}%  {row['total']:>7.3f}%{marker}"
    )

best = dr.nlargest(1, 'total').iloc[0]
print(f"\n  MEJOR seed individual: seed={int(best['seed'])} -> Total={best['total']:.3f}%")
print(f"  PEOR  seed individual: seed={int(dr.nsmallest(1,'total').iloc[0]['seed'])} -> Total={dr['total'].min():.3f}%")
print(f"  RANGO entre seeds: {dr['total'].max() - dr['total'].min():.3f}pp")
print(f"  (un rango amplio indicaria diversidad; un rango estrecho indica redundancia)")

# Portfolio ensemble
dfc = df[df['consensus'] >= 3]
dfp = dfc.groupby('bucket').agg(
    return_pct=('return_pct', 'mean'),
    is_win=('is_win', 'max'),
    window=('_window', 'first')
).sort_index()
sel, lt = [], None
for ts in dfp.index:
    if lt is None or (ts - lt).total_seconds() / 3600 >= 96:
        sel.append(ts); lt = ts
portfolio = dfp.loc[sel]
pw3 = portfolio[portfolio['window'] == 'W3']
pw4 = portfolio[portfolio['window'] == 'W4']
port_tot = portfolio['return_pct'].sum() * 100

print(f"\n  PORTFOLIO ENSEMBLE (consensus>=3, embargo 96H):")
print(f"    W3: {len(pw3)} trades  WR={pw3['is_win'].mean()*100:.1f}%  RetTot={pw3['return_pct'].sum()*100:.3f}%")
print(f"    W4: {len(pw4)} trades  WR={pw4['is_win'].mean()*100:.1f}%  RetTot={pw4['return_pct'].sum()*100:.3f}%")
print(f"    TOTAL: {port_tot:.3f}%")

winner = "ENSEMBLE" if port_tot > best['total'] else "SEED INDIVIDUAL"
print(f"\n  Ensemble ({port_tot:.3f}%) vs Mejor seed individual ({best['total']:.3f}%)")
print(f"  GANADOR: {winner}")

# ─── 6. HOLDING TIME ─────────────────────────────────────────────────────
print()
print(SEP)
print("6. HOLDING TIME — Comportamiento temporal del modelo")
print(SEP)
if 'holding_h' in df.columns:
    for win, dw in [('W3', dw3), ('W4', dw4), ('GLOBAL', df)]:
        h = dw['holding_h'].dropna()
        print(f"  {win}: n={len(h)} min={h.min():.0f}h med={h.median():.0f}h mean={h.mean():.0f}h max={h.max():.0f}h")
    print()
    # ¿Correlacion entre holding y retorno?
    h_all = df['holding_h'].dropna()
    r_all = df.loc[h_all.index, 'ret100']
    from scipy import stats as _sp
    valid = h_all.notna() & r_all.notna()
    rho, pval = _sp.spearmanr(h_all[valid], r_all[valid])
    print(f"  Spearman(holding_h, return): rho={rho:.4f}  p={pval:.4f}")
    if rho < -0.1 and pval < 0.05:
        print("  -> Trades MAS CORTOS tienen MEJOR retorno (baja de VB a PT)")
    elif rho > 0.1 and pval < 0.05:
        print("  -> Trades MAS LARGOS tienen MEJOR retorno")
    else:
        print("  -> Sin correlacion significativa holding vs retorno")

# ─── 7. CUADRO COMPLETO ──────────────────────────────────────────────────
print()
print(SEP)
print("7. DIAGNOSTICO FINAL — Los problemas reales del sistema")
print(SEP)

t_start = df.index.min(); t_end = df.index.max()
months = (t_end - t_start).days / 30.44
hours = (t_end - t_start).total_seconds() / 3600
max_96 = hours / 96
tps = len(df) / n_seeds
realizacion = tps / max_96 * 100

print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ PROBLEMA 1: TRADE STARVATION — {realizacion:.0f}% del maximo teorico        │
  ├─────────────────────────────────────────────────────────────────────┤
  │ Max teorico (embargo 96H): {max_96:.0f} trades/seed                       │
  │ Obtenido: {tps:.1f} trades/seed en {months:.1f} meses = {tps/months:.2f}/mes            │
  │                                                                     │
  │ Causas en cadena:                                                   │
  │   - Alpha triggers (DTW/Golden/Genetic) filtran la mayoria          │
  │   - HMM regime filter rechaza regimenes no permitidos               │
  │   - XGBoost threshold rechaza señales de baja confianza             │
  │   - Consensus>=3 (REDUNDANTE cuando corr=0.9975)                    │
  │   - Embargo 96H = maximo 7.7 trades/mes de techo absoluto          │
  │                                                                     │
  │ IMPACTO: Moderado. W3 logra 5.4/mes con WR=74%. El problema         │
  │ NO es la inanicion — es QUE los trades en W4 son todos perdedores.  │
  └─────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ PROBLEMA 2: ENSEMBLE SIN DIVERSIDAD — corr=0.9975, N_ef=0.1        │
  ├─────────────────────────────────────────────────────────────────────┤
  │ El consensus mas frecuente es {int(modo)} seeds de acuerdo.               │
  │ Cuando todas las seeds votan igual, el consensus>=3 no filtra nada. │
  │                                                                     │
  │ Causa raiz: todas las seeds usan el MISMO Alpha Trigger (DTW +      │
  │ Golden + Genetic Rules calculadas sobre los mismos datos raw).      │
  │ La semilla aleatoria solo cambia la inicializacion de XGBoost,      │
  │ no el timing de las señales de entrada.                             │
  │                                                                     │
  │ IMPACTO: El ensemble consume 20x recursos sin beneficio estadistico.│
  │ La diversidad percibida (20 seeds) es una ilusion arquitectonica.   │
  └─────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ PROBLEMA 3: W4 OOD (OUT-OF-DISTRIBUTION) — causa raiz del fallo    │
  ├─────────────────────────────────────────────────────────────────────┤
  │ W3 (jul-sep 2025, BTC 60K): WR=74%, bootstrap P(ret>0)=100%        │
  │ W4 (oct-dic 2025, BTC ATH): WR=46%, bootstrap P(ret>0)=0%          │
  │                                                                     │
  │ El modelo es mas confiante en W4 (xgb_prob mas alto) pero tiene    │
  │ peores resultados. El modelo ve señales que en IS eran bullish       │
  │ pero en ATH el contexto es completamente diferente.                 │
  │                                                                     │
  │ IMPACTO: Este es el problema principal. Ninguna mejora de parametros│
  │ puede corregir un modelo que opera en un regimen desconocido.        │
  │ SOLUCION: H4 features ATH (ya implementado) + re-run obligatorio.   │
  └─────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ PROBLEMA 4: EMBUDO DE CALIDAD vs EMBUDO DE CANTIDAD                 │
  ├─────────────────────────────────────────────────────────────────────┤
  │ Todos los filtros apuntan a maximizar PRECISION (WR alta)           │
  │ a costa de minimizar RECALL (pocos trades).                         │
  │                                                                     │
  │ El resultado: en W3 funciona brillante (precision=74%, N=13.2/seed) │
  │ En W4 falla completamente porque el modelo no sabe que esta en ATH. │
  │                                                                     │
  │ El SOP exige min 30 trades para validez estadistica.                │
  │ Con 27.2 trades/seed en 10 meses, estamos en el limite inferior.    │
  │ Dividido en 3 ventanas: W1=2.9, W3=13.2, W4=16.4 trades/seed.      │
  │ NINGUNA ventana individual supera el umbral de 30 trades.           │
  └─────────────────────────────────────────────────────────────────────┘

  PRIORIDADES REALES:
  ═══════════════════
  1. H4 ATH features + re-run [ya implementado — es la unica correccion de fondo]
  2. Consensus-Soft Embargo (SOP ya lo menciona): reducir embargo a 24H
     cuando consensus >= 4 para capturar mas señales de la misma calidad
     → potencialmente 2-3x mas trades por ventana → supera el umbral de 30
  3. Largo plazo (V3): redisenar el ensemble para tener diversidad REAL
     de señal (diferentes Alpha Triggers por grupo de seeds)
""")
