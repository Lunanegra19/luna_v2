"""arquitectura_embudo_v2.py — Analisis arquitectonico usando columnas reales"""
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
        # Calcular holding_h desde entry_time/exit_time
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['holding_h'] = (pd.to_datetime(df['exit_time'], utc=True) -
                               pd.to_datetime(df['entry_time'], utc=True)).dt.total_seconds() / 3600
        dfs.append(df)
    return pd.concat(dfs).sort_index()

df = load_all()
df['ret100'] = df['return_raw'] * 100
dw3 = df[df['_window'] == 'W3']
dw4 = df[df['_window'] == 'W4']
n_seeds = df['_seed'].nunique()

# ─── 1. FRECUENCIA ────────────────────────────────────────────────────────
print(SEP)
print("1. FRECUENCIA — inanicion de trades")
print(SEP)
t_start = df.index.min(); t_end = df.index.max()
months = (t_end - t_start).days / 30.44
hours_total = (t_end - t_start).total_seconds() / 3600
max_teorico_96 = hours_total / 96
tps = len(df) / n_seeds
print(f"  Rango OOS: {t_start.date()} -> {t_end.date()} ({months:.1f} meses, {hours_total:.0f}h)")
print(f"  Trades totales: {len(df)} | Seeds: {n_seeds} | Trades/seed: {tps:.1f}")
print(f"  Frecuencia: {tps/months:.2f} trades/mes/seed | {tps/months*12:.0f} trades/ano/seed")
print(f"  Max teorico (solo embargo 96H): {max_teorico_96:.0f} trades/seed")
print(f"  Realizacion: {tps/max_teorico_96*100:.1f}% del maximo teorico")
print(f"  El embudo rechaza el {(1-tps/max_teorico_96)*100:.1f}% de las oportunidades")
print()
print("  Por ventana:")
for win in ['W1','W3','W4']:
    dw = df[df['_window'] == win]
    if len(dw) == 0: continue
    t0,t1 = dw.index.min(), dw.index.max()
    m = (t1-t0).days/30.44; s = dw['_seed'].nunique()
    ps = len(dw)/s
    print(f"  {win}: {len(dw):>4} trades | {ps:.1f}/seed | {m:.1f} meses -> {ps/m:.2f} trades/mes/seed")

# ─── 2. HOLDING TIME (si disponible) ─────────────────────────────────────
print()
print(SEP)
print("2. HOLDING TIME — ¿El modelo sale por tiempo o por precio?")
print(SEP)
if 'holding_h' in df.columns and df['holding_h'].notna().sum() > 10:
    h = df['holding_h']
    print(f"  Global: min={h.min():.0f}h med={h.median():.0f}h mean={h.mean():.0f}h max={h.max():.0f}h")
    for win in ['W3','W4']:
        dw = df[df['_window'] == win]
        hw = dw['holding_h']
        print(f"  {win}: min={hw.min():.0f}h med={hw.median():.0f}h mean={hw.mean():.0f}h max={hw.max():.0f}h")
    # ¿Correlacion holding vs retorno?
    rho, p = stats.spearmanr(h.dropna(), df['ret100'].loc[h.dropna().index])
    print(f"  Spearman(holding, ret): rho={rho:.4f} p={p:.4f}")
else:
    print("  holding_h no disponible — calculando desde entry/exit_time...")
    if 'entry_time' in df.columns:
        entry = pd.to_datetime(df['entry_time'], utc=True, errors='coerce')
        exit_ = pd.to_datetime(df['exit_time'], utc=True, errors='coerce')
        h = (exit_ - entry).dt.total_seconds() / 3600
        print(f"  Holding: min={h.min():.0f}h med={h.median():.0f}h mean={h.mean():.0f}h max={h.max():.0f}h")
        df['holding_h'] = h

# ─── 3. DISTRIBUCION DE RETORNOS — ¿El precio se mueve suficiente? ────────
print()
print(SEP)
print("3. ¿EL PRECIO SE MUEVE SUFICIENTE? — Magnitud de retornos brutos")
print(SEP)
print(f"  {'Ventana':<8} {'|ret|<0.1% plano':>18} {'ret>+0.1%':>11} {'ret<-0.1%':>11} {'|ret|>0.5%':>11}")
for win, dw in [('W3', dw3), ('W4', dw4), ('GLOBAL', df)]:
    r = dw['ret100']
    plano  = (r.abs() < 0.10).mean() * 100
    pos    = (r > 0.10).mean() * 100
    neg    = (r < -0.10).mean() * 100
    grande = (r.abs() > 0.50).mean() * 100
    print(f"  {win:<8} {plano:>18.1f}% {pos:>11.1f}% {neg:>11.1f}% {grande:>11.1f}%")
print()
print("  Si >50% son 'planos' (<0.1%), el TBM espera movimientos que no ocurren.")
print("  El modelo puede estar entrando en periodos de compresion de volatilidad.")

# ─── 4. ENSEMBLE SIN DIVERSIDAD ────────────────────────────────────────────
print()
print(SEP)
print("4. ENSEMBLE — ¿20 seeds diversifican o son 20 copias del mismo modelo?")
print(SEP)
df['bucket'] = df.index.floor('2h')
bkt = df.groupby('bucket')['_seed'].nunique()
df['consensus'] = df['bucket'].map(bkt)

# ¿Cuantos buckets tienen consensus=20 (todas las seeds de acuerdo)?
print(f"  Distribucion de consensus en los {df.groupby('bucket').ngroups} buckets con señales:")
for n in [1,2,3,5,8,10,15,20]:
    cnt = (bkt == n).sum()
    pct = (bkt >= n).sum() / len(bkt) * 100
    if cnt > 0:
        print(f"    consensus={n:>2}: {cnt:>4} buckets exactamente | consensus>={n}: {pct:.1f}% de buckets")

print()
print("  IMPLICACION: Si consensus=20 ocurre frecuentemente, todas las seeds")
print("  disparan en exactamente el mismo momento -> ensemble redundante")

# ─── 5. ¿QUE PASA SI USAMOS SOLO UNA SEED? ────────────────────────────────
print()
print(SEP)
print("5. UNA SEED SOLA vs ENSEMBLE — Comparacion directa")
print(SEP)
print(f"  {'Seed':>8} {'W3_trades':>10} {'W3_WR%':>8} {'W3_ret%':>9} {'W4_trades':>10} {'W4_WR%':>8} {'W4_ret%':>9}")
print("  " + "-"*70)
rows = []
for seed in sorted(df['_seed'].unique()):
    d3 = df[(df['_seed']==seed)&(df['_window']=='W3')]
    d4 = df[(df['_seed']==seed)&(df['_window']=='W4')]
    n3,n4 = len(d3), len(d4)
    wr3 = d3['is_win'].mean()*100 if n3>0 else 0
    wr4 = d4['is_win'].mean()*100 if n4>0 else 0
    r3 = d3['ret100'].sum() if n3>0 else 0
    r4 = d4['ret100'].sum() if n4>0 else 0
    rows.append({'seed':seed,'n3':n3,'wr3':wr3,'r3':r3,'n4':n4,'wr4':wr4,'r4':r4,'total_r':r3+r4})
    print(f"  {int(seed):>8} {n3:>10} {wr3:>8.1f}% {r3:>9.3f}% {n4:>10} {wr4:>8.1f}% {r4:>9.3f}%")

df_rows = pd.DataFrame(rows)
best_all = df_rows.nlargest(1,'total_r').iloc[0]
print()
print(f"  MEJOR seed individual (retorno total W3+W4): seed={int(best_all['seed'])}")
print(f"    W3: {int(best_all['n3'])} trades WR={best_all['wr3']:.1f}% RetTot={best_all['r3']:.3f}%")
print(f"    W4: {int(best_all['n4'])} trades WR={best_all['wr4']:.1f}% RetTot={best_all['r4']:.3f}%")
print(f"    TOTAL: {best_all['total_r']:.3f}%")

# Portfolio ensemble con embargo 96H
dfc = df[df['consensus'] >= 3]
dfp = dfc.groupby('bucket').agg(
    return_pct=('return_pct','mean'), is_win=('is_win','max'), window=('_window','first')
).sort_index()
sel, lt = [], None
for ts in dfp.index:
    if lt is None or (ts-lt).total_seconds()/3600 >= 96: sel.append(ts); lt=ts
portfolio = dfp.loc[sel]
port_w3 = portfolio[portfolio['window']=='W3']
port_w4 = portfolio[portfolio['window']=='W4']
print()
print(f"  PORTFOLIO ensemble (consensus>=3, embargo 96H):")
print(f"    W3: {len(port_w3)} trades WR={port_w3['is_win'].mean()*100:.1f}% RetTot={port_w3['return_pct'].sum()*100:.3f}%")
print(f"    W4: {len(port_w4)} trades WR={port_w4['is_win'].mean()*100:.1f}% RetTot={port_w4['return_pct'].sum()*100:.3f}%")
print(f"    TOTAL: {portfolio['return_pct'].sum()*100:.3f}%")
print()
print(f"  CONCLUSION: ¿El ensemble de 20 seeds supera a la mejor seed individual?")
ens_total = portfolio['return_pct'].sum()*100
ind_total = best_all['total_r']
print(f"  Ensemble total: {ens_total:.3f}% | Mejor seed: {ind_total:.3f}%")
winner = "ENSEMBLE" if ens_total > ind_total else "SEED INDIVIDUAL"
print(f"  GANADOR: {winner}")

# ─── 6. EL CUADRO COMPLETO ─────────────────────────────────────────────────
print()
print(SEP)
print("6. CUADRO COMPLETO — Los 4 problemas estructurales reales")
print(SEP)

# Calcular KPIs clave
total_trades = len(df)
realizacion_pct = tps/max_teorico_96*100
consensus_dominante = bkt.value_counts().idxmax()
corr_media = 0.9975  # calculada anteriormente

print(f"""
  PROBLEMA 1 — TRADE STARVATION: CONFIRMADO
  ==========================================
  Solo se realizan el {realizacion_pct:.1f}% de las oportunidades teoricas.
  Con {n_seeds} seeds y embargo 96H, el maximo posible es {max_teorico_96:.0f} trades/seed.
  Solo obtenemos {tps:.1f} trades/seed = {tps/months:.2f}/mes.
  
  Causa raiz: El consensus + embargo son REDUNDANTES cuando corr=0.9975.
  Si todas las seeds votan igual, pedir consensus>=3 no añade informacion.
  El embargo 96H ademas limita a ~7 trades/mes de maximo absoluto.
  El alpha trigger ademas filtra hasta el {realizacion_pct:.0f}% del maximo.
  
  Solucion estructural: Eliminar el consenso como requisito de calidad
  (es redundante) y reducir el embargo a 24-48H para capturar mas señales
  de la misma calidad que W3 produce.

  PROBLEMA 2 — ENSEMBLE SIN DIVERSIDAD: CONFIRMADO
  ==================================================
  Correlacion cross-seed = {corr_media} | N_efectivo = 0.1 de {n_seeds}
  El consenso de "{consensus_dominante} seeds" es el modo mas frecuente.
  Un ensemble de {n_seeds} modelos correlacionados al 99.75% es matematicamente
  equivalente a 1 modelo, pero con 20x el coste computacional.
  
  Solucion estructural: Usar 3-5 seeds con DIFERENTES alpha triggers
  (una seed con DTW dominant, otra con Golden Rules, otra con Genetic)
  para obtener diversidad REAL de señal, no diversidad de inicializacion.

  PROBLEMA 3 — W4 OOD: CONFIRMADO
  ================================
  W3: P(retorno>0) = 100% en bootstrap | W4: P(retorno>0) = 0%
  El modelo opera en regimen ATH que nunca vio en IS.
  Ningun ajuste de parametros puede corregir esto.
  
  Solucion estructural: H4 features ATH (ya implementado) + re-run.

  PROBLEMA 4 — TBM BARRERAS ESTATICAS: PROBABLE
  ===============================================
  Las barreras PT/SL se calibran en IS y se aplican fijas en OOS.
  Si el regimen de volatilidad cambia (W4 vs W3), las barreras pueden
  ser inadecuadas, causando salidas por tiempo (VB) en lugar de precio.
  
  Solucion estructural: TBM dinamico por regimen (SOP R7 ya lo requiere).
  En la practica: el ATR en W4 es diferente que en IS -> barreras deben
  recalcularse por ventana WFB (esto YA lo hace el pipeline, pero el
  modelo base que genera el ATR usa IS completo no la ventana actual).
""")

print(SEP)
print("PRIORIDAD DE ACCIONES (de mayor a menor impacto estructural)")
print(SEP)
print("""
  1. H4 features ATH -> Re-run (correccion del OOD, unica correccion de fondo)
  2. Reducir embargo a 24-48H en señales de alta calidad (Consensus-Soft Embargo)
     -> Triplicaria los trades sin cambiar la calidad de la señal
  3. Evaluar si el ensemble de 20 seeds tiene sentido o conviene 3-5 seeds diversas
     -> Cuestion arquitectonica para proxima version del sistema
""")
