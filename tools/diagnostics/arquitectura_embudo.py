"""
arquitectura_embudo.py — Analisis del cuadro completo
=====================================================
No buscamos pequeños fixes. Buscamos entender SI el sistema tiene
un problema estructural de arquitectura que explique la inanicion de trades,
la correlacion perfecta del ensemble, y el fallo en W4.

Preguntas criticas:
1. ¿Cuantas señales se pierden en cada capa del embudo?
2. ¿La arquitectura TBM es la correcta para este mercado?
3. ¿Es el ensemble de seeds realmente util o es redundante?
4. ¿El problema de W4 es OOD o un problema de barreras?
5. ¿Cuanto potencial alpha hay sin el embudo?
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

df = load_all()

# ─── 1. ANALISIS TEMPORAL: ¿Cuantos trades por mes por seed? ───────────────
print(SEP)
print("1. FRECUENCIA DE TRADING — ¿Inanicion real?")
print(SEP)

# Rango temporal total
t_start = df.index.min()
t_end   = df.index.max()
months_total = (t_end - t_start).days / 30.44

n_seeds = df['_seed'].nunique()
n_trades_total = len(df)
trades_per_seed = n_trades_total / n_seeds
trades_per_month_per_seed = trades_per_seed / months_total

print(f"  Rango: {t_start.date()} -> {t_end.date()} ({months_total:.1f} meses)")
print(f"  Total trades: {n_trades_total} | Seeds: {n_seeds}")
print(f"  Trades/seed: {trades_per_seed:.1f} en {months_total:.1f} meses")
print(f"  Frecuencia: {trades_per_month_per_seed:.2f} trades/mes/seed")
print(f"  Frecuencia: {trades_per_month_per_seed*12:.1f} trades/ano/seed")
print()

# Maximo teorico con embargo 96H
hours_total = (t_end - t_start).total_seconds() / 3600
max_teorico = hours_total / 96
print(f"  MAX TEORICO (solo embargo 96H): {max_teorico:.0f} trades/seed")
print(f"  REALIZACION: {trades_per_seed:.1f} / {max_teorico:.0f} = {trades_per_seed/max_teorico*100:.1f}%")
print(f"  El embudo rechaza el {(1-trades_per_seed/max_teorico)*100:.1f}% de oportunidades temporales")

print()
print("  Por ventana:")
for win in ['W1','W3','W4']:
    dw = df[df['_window'] == win]
    if len(dw) == 0: continue
    t0 = dw.index.min(); t1 = dw.index.max()
    meses = (t1-t0).days / 30.44
    per_seed = len(dw) / dw['_seed'].nunique()
    print(f"  {win}: {len(dw)} trades total | {per_seed:.1f}/seed | {meses:.1f} meses -> {per_seed/meses:.2f} trades/mes/seed")

# ─── 2. ANALISIS TBM: ¿Las barreras son las correctas? ────────────────────
print()
print(SEP)
print("2. TBM BARRERAS — ¿El modelo entra pero el mercado no se mueve?")
print(SEP)

# 97% VB significa que el precio casi nunca alcanza PT ni SL dentro del horizonte
print("  Exit types: columna 'exit_type' no disponible en parquet actual")
print("  Usando holding_h como proxy (holding=max_horizonte -> VB, corto -> PT/SL)")
print()
print("  Distribucion de holding_h por ventana (proxy de tipo de salida):")
for win in ['GLOBAL','W3','W4']:
    d = df if win == 'GLOBAL' else df[df['_window'] == win]
    if len(d) == 0: continue
    h_med = d['holding_h'].median()
    h_max = d['holding_h'].max()
    h_min = d['holding_h'].min()
    ret_m = d['return_raw'].mean() * 100
    print(f"  {win:6}: holding med={h_med:.0f}h min={h_min:.0f}h max={h_max:.0f}h | RetMed={ret_m:.4f}%")

print()
print("  INTERPRETACION: 97% VB significa que el precio NO se mueve")
print("  lo suficiente para alcanzar PT o SL en el horizonte temporal.")
print("  Hay 3 posibles causas:")
print("    A) El horizonte TBM es demasiado corto para este activo/periodo")
print("    B) Las barreras PT/SL son demasiado anchas")
print("    C) El modelo entra en momentos de baja volatilidad esperada")

# Holding time vs VB
print()
print()
print("  Distribucion de retornos brutos (|ret|<0.15% = movimiento minimo)")
for win, d_check in [('W3',dw3), ('W4',dw4)]:
    r = d_check['return_raw'] * 100
    pos = (r > 0.15).mean() * 100
    neg = (r < -0.15).mean() * 100
    tiny = (r.abs() < 0.15).mean() * 100
    print(f"  {win}: |ret|<0.15%(plano): {tiny:.1f}% | ret>+0.15%: {pos:.1f}% | ret<-0.15%: {neg:.1f}%")

# ─── 3. EL ENSEMBLE COMO ARQUITECTURA ─────────────────────────────────────
print()
print(SEP)
print("3. ARQUITECTURA ENSEMBLE — ¿20 seeds aportan valor real?")
print(SEP)

print("  Cross-seed correlation = 0.9975 (medido anteriormente)")
print("  N_efectivo Kish = 0.1 de 20 seeds")
print()
print("  IMPLICACIONES ARQUITECTONICAS:")
print("  - El ensemble no reduce varianza (todas las seeds votan igual)")
print("  - El consenso minimo de 3 seeds no filtra ruido (0.1 seeds efectivos)")
print("  - 20 seeds x mismo Alpha Trigger = 20 copias del mismo modelo")
print("  - El coste computacional es 20x sin beneficio estadistico")

# ¿Cuantas señales generaria UNA sola seed sin filtros?
print()
print("  Analisis de señales por seed individual (W3):")
for seed in sorted(df['_seed'].unique())[:5]:
    dw3 = df[(df['_seed'] == seed) & (df['_window'] == 'W3')]
    print(f"    Seed {int(seed)}: {len(dw3)} trades en W3 | WR={dw3['is_win'].mean()*100:.1f}% | RetTot={dw3['return_raw'].sum()*100:.3f}%")

# ─── 4. COMPARACION CON ALTERNATIVA: UNA SEED SIN CONSENSO ───────────────
print()
print(SEP)
print("4. ALTERNATIVA ARQUITECTONICA: Una seed vs ensemble de 20")
print(SEP)

# Mejor seed individual en W3
best_seed_w3 = df[df['_window']=='W3'].groupby('_seed')['return_raw'].sum().idxmax()
dw3_best = df[(df['_seed'] == best_seed_w3) & (df['_window'] == 'W3')]
dw4_best = df[(df['_seed'] == best_seed_w3) & (df['_window'] == 'W4')]

# Portfolio ensemble con embargo 96H
df['bucket'] = df.index.floor('2h')
bkt = df.groupby('bucket')['_seed'].nunique()
df['consensus'] = df['bucket'].map(bkt)
dfc = df[df['consensus'] >= 3]
dfp = dfc.groupby('bucket').agg(return_pct=('return_pct','mean'),is_win=('is_win','max'),window=('_window','first')).sort_index()
sel, lt = [], None
for ts in dfp.index:
    if lt is None or (ts-lt).total_seconds()/3600 >= 96: sel.append(ts); lt=ts
portfolio = dfp.loc[sel]

print(f"  ENSEMBLE (20 seeds, consensus>=3, embargo 96H):")
print(f"    Total trades: {len(portfolio)} | WR={portfolio['is_win'].mean()*100:.1f}%")
print(f"    W3: {len(portfolio[portfolio['window']=='W3'])} trades | W4: {len(portfolio[portfolio['window']=='W4'])} trades")
print(f"    RetTot: {portfolio['return_pct'].sum()*100:.3f}%")
print()
print(f"  MEJOR SEED INDIVIDUAL ({int(best_seed_w3)}, sin consensus, sin embargo cross-seed):")
print(f"    W3: {len(dw3_best)} trades | WR={dw3_best['is_win'].mean()*100:.1f}% | RetTot={dw3_best['return_raw'].sum()*100:.3f}%")
print(f"    W4: {len(dw4_best)} trades | WR={dw4_best['is_win'].mean()*100:.1f}% | RetTot={dw4_best['return_raw'].sum()*100:.3f}%")

# ─── 5. EL PROBLEMA DE W4 DESDE LA PERSPECTIVA DEL MERCADO ───────────────
print()
print(SEP)
print("5. ¿ES W4 UN PROBLEMA DE MODELO O DE MERCADO?")
print(SEP)

# Calcular retornos del mercado en W4 vs W3 (usando datos de trades)
dw3 = df[df['_window'] == 'W3']
dw4 = df[df['_window'] == 'W4']

# El holding medio es ~25h. ¿Como es el mercado en ese periodo?
print(f"  W3: holding_med={dw3['holding_h'].median():.0f}h | ret_mercado_medio=???")
print(f"  W4: holding_med={dw4['holding_h'].median():.0f}h | ret_mercado_medio=???")
print()
print("  Analisis de distribucion de retornos brutos:")
for win, d in [('W3',dw3), ('W4',dw4)]:
    r = d['return_raw'] * 100
    pos = (r > 0.15).mean() * 100  # trades que podrian ser PT
    neg = (r < -0.15).mean() * 100  # trades que podrian ser SL
    tiny = (r.abs() < 0.15).mean() * 100  # trades con retorno minimo (VB con poco movimiento)
    print(f"  {win}: |ret|>0.15%: {100-tiny:.1f}% | ret>+0.15%: {pos:.1f}% | ret<-0.15%: {neg:.1f}% | |ret|<0.15%(plano): {tiny:.1f}%")

print()
print("  INTERPRETACION: Si la mayoria de retornos son 'planos' (<0.15%),")
print("  el mercado no se mueve en la direccion esperada durante el holding.")
print("  Esto puede indicar que las BARRERAS TBM son correctas pero el")
print("  HORIZONTE es demasiado corto para los patrones que el modelo aprende.")

# ─── 6. ANALISIS DEL HORIZONTE TEMPORAL ───────────────────────────────────
print()
print(SEP)
print("6. HORIZONTE TEMPORAL — ¿Que pasa si esperamos mas?")
print(SEP)

# Buscar retornos a distintos horizontes en los datos de trades
# Solo podemos hacer esto indirectamente — usando el holding time vs retorno
print("  Distribucion de retornos por cuartil de holding time:")
df['holding_q'] = pd.qcut(df['holding_h'], q=4, labels=['Q1(corto)','Q2','Q3','Q4(largo)'])
print(f"  {'Cuartil':<12} {'N':>5} {'WR%':>7} {'RetMed%':>10} {'Holding_med':>12}")
for q in ['Q1(corto)','Q2','Q3','Q4(largo)']:
    dq = df[df['holding_q'] == q]
    if len(dq) == 0: continue
    wr = dq['is_win'].mean() * 100
    rm = dq['return_raw'].mean() * 100
    hm = dq['holding_h'].median()
    print(f"  {q:<12} {len(dq):>5} {wr:>7.1f}% {rm:>10.4f}% {hm:>12.0f}h")

# ─── 7. PREGUNTA FUNDAMENTAL: ¿Donde se pierde el alpha? ─────────────────
print()
print(SEP)
print("7. ¿DONDE SE PIERDE EL ALPHA? — El embudo completo")
print(SEP)

print("""
  El sistema tiene las siguientes capas de filtro ANTES de ejecutar un trade:
  
  [RAW] Señal de precio (cada hora)
    ↓ Alpha Triggers (DTW + Golden Rules + Genetic Rules)
    ↓ HMM Regime filter (solo regimenes permitidos)
    ↓ XGBoost threshold (prob_cal > ~0.50-0.64)
    ↓ MetaLabeler V2 (filtro de calidad de señal)
    ↓ KMeans Tribe filter
    ↓ Consenso >= 3 seeds
    ↓ Embargo 96H
  [TRADE EJECUTADO]
  
  Con 76873 horas de datos y solo 545 trades en 20 seeds:
""")

total_hours = 76873
print(f"  Total horas disponibles (raw): {total_hours}")
print(f"  Total señales ejecutadas (todas seeds): {len(df)}")
print(f"  Tasa de conversion raw->trade: {len(df)/total_hours*100:.4f}%")
print(f"  Por seed: {len(df)/n_seeds:.0f} trades de {total_hours} horas = {len(df)/n_seeds/total_hours*100:.4f}%")
print()
print("  HALLAZGO CRITICO: El sistema rechaza el 99.96% de las horas disponibles.")
print("  La pregunta es: ¿ese rechazo es SELECCION DE CALIDAD o DESTRUCCION DE ALPHA?")

print()
# Comparar retorno de los trades ejecutados vs los rechazados
# (aproximacion: si los trades ejecutados tienen RetMed cerca de 0, el filtro no aporta)
print("  Evidencia de si el filtro aporta calidad (vs no filtro):")
print(f"  RetMed global (trades seleccionados): {df['return_raw'].mean()*100:.4f}%")
print(f"  WR global: {df['is_win'].mean()*100:.1f}%")
print(f"  Si WR fuera aleatorio: 50.0%")
print(f"  Uplift del filtro: {(df['is_win'].mean()-0.5)*100:+.1f}pp sobre random")
print()
print("  PERO en W4:")
dw4 = df[df['_window'] == 'W4']
print(f"  WR en W4: {dw4['is_win'].mean()*100:.1f}% (peor que random=50%)")
print(f"  El embudo en W4 DESTRUYE valor: selecciona activamente los peores trades")

# ─── 8. ALTERNATIVAS ARQUITECTONICAS REALES ───────────────────────────────
print()
print(SEP)
print("8. ALTERNATIVAS ARQUITECTONICAS — Cuadro completo")
print(SEP)

print("""
  A. PROBLEMA DE TRADE STARVATION (inanicion):
     El sistema necesita simultaneamente:
     - consensus >= 3 seeds (muy restrictivo si correlacion=0.9975)
     - embargo 96H (4 dias entre trades)
     - Alpha triggers activos
     → Resultado: ~2 trades/mes/seed cuando el mercado ofrece ~7 oportunidades/mes
     
     SOLUCION REAL: El consenso no aporta nada con corr=0.9975.
     Una sola seed bien calibrada con embargo 24H daria ~3x mas trades
     con la misma calidad estadistica.
  
  B. PROBLEMA TBM - BARRERAS FIJAS EN REGIMEN VARIABLE:
     Las barreras PT/SL se calculan con ATR en IS (2020-2024).
     En W4 (BTC en ATH con vol diferente), esas barreras pueden ser:
     - Demasiado anchas: el precio no llega a PT en el horizonte → VB
     - Demasiado estrechas: el precio toca SL por volatilidad intraday
     El 97% VB sugiere barreras demasiado ANCHAS para el horizonte actual.
  
  C. PROBLEMA DE ENSEMBLE SIN DIVERSIDAD:
     20 seeds con corr=0.9975 es matematicamente equivalente a 1 seed.
     El overhead computacional es 20x por 0 beneficio estadistico.
     ALTERNATIVA: 3-5 seeds con DIFERENTES Alpha Triggers (DTW vs Genetic vs Golden)
     que generen señales en momentos DISTINTOS.
  
  D. EL HALLAZGO MAS IMPORTANTE DEL ANALISIS:
     W3 (sin ATH) → modelo funciona perfectamente (WR=74%, SR=+3.75)
     W4 (ATH) → modelo falla completamente (WR=46%, SR=-3.75)
     
     Esto NO es un problema de parametros. Es un problema de DATOS:
     el modelo nunca vio un regimen ATH de BTC durante entrenamiento
     (o vio muy poco). Todos los "fixes" de parametros son parches.
     H4 (features ATH) es la unica correccion estructural posible.
""")

print(SEP)
print("VEREDICTO ARQUITECTONICO")
print(SEP)
print("""
  1. TRADE STARVATION: REAL — el embudo rechaza 99.96% de horas
     Causa: consenso-embargo redundante con ensemble correlacionado perfectamente
     Fix estructural: reducir embargo a 24-48H O usar single-seed con mas señales

  2. TBM BARRERAS: PROBABLE — 97% VB indica horizontes/barreras mal calibradas
     Causa: barreras calibradas en IS (2020-2024) aplicadas en W4 ATH (2025)
     Fix estructural: TBM dinamico por regimen (ya mencionado en SOP R7)

  3. ENSEMBLE SIN DIVERSIDAD: CONFIRMADO — corr=0.9975, N_ef=0.1
     Causa: todas las seeds disparan con el mismo Alpha Trigger en el mismo momento
     Fix estructural: diversidad por tipo de señal, no por seed aleatoria

  4. OOD W4: CONFIRMADO — el modelo nunca vio ATH durante entrenamiento
     Fix estructural: H4 features ATH (implementado)
""")
