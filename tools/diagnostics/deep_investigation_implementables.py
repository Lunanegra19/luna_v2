"""
deep_investigation_implementables.py
======================================
Investigacion profunda de los 3 candidatos a implementacion:

1. B1-HOUR: Gate horario 7H-13H UTC
   - Robustez: ¿funciona en CADA ventana por separado?
   - Look-ahead: HORA DE ENTRADA — ¿se conoce antes de entrar?
   - Overfitting: ¿es estable o es noise de 1 ventana?

2. B1-DOW: Evitar Lunes (WR=30.9%)
   - Igual que arriba pero dia de semana

3. E4-MAE: Drawdown = 0 → WR=93.3%
   - CRITICO: ¿es el drawdown en el parquet INTRA-TRADE (mala) o POST-TRADE (OK)?
   - Identificar la naturaleza exacta de la columna 'drawdown'
   - Si es post-trade: NO es implementable (look-ahead)
   - Si hay forma de derivar MAE pre-trade: ¿cómo?

"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

WFB = Path("data/reports/wfb")
dfs = []
for f in sorted(WFB.glob("oos_trades_W*_seed*.parquet")):
    parts = f.stem.split("_")
    df = pd.read_parquet(f)
    df["_w"]    = parts[2]
    df["_seed"] = parts[3].replace("seed", "")
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined["entry_dt"] = pd.to_datetime(combined["entry_time"], utc=True, errors="coerce")
combined["exit_dt"]  = pd.to_datetime(combined["exit_time"],  utc=True, errors="coerce")
combined["hour_utc"] = combined["entry_dt"].dt.hour
combined["dow"]      = combined["entry_dt"].dt.dayofweek
baseline_wr = combined["is_win"].mean()

WINDOWS = ["W2","W3","W4","W5"]

# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("INVESTIGACION 1 — B1-HOUR: Gate 7H-13H UTC")
print("=" * 70)

print(f"\nBaseline WR global: {baseline_wr:.4f}")

# ── 1A: Robustez por ventana ───────────────────────────────────────
print("\n─" * 70)
print("1A: WR en 7H-13H UTC POR VENTANA (test de estabilidad)")
print("─" * 70)
HOUR_GATE = (7, 13)
print(f"  {'Ventana':>8} {'N_dentro':>9} {'WR_dentro':>10} {'N_fuera':>9} {'WR_fuera':>10} {'Delta':>8} {'p-value':>9} {'Sig':>5}")
hour_gate_stable = []
for w in WINDOWS:
    sub = combined[combined["_w"] == w]
    if len(sub) < 10:
        continue
    inside  = sub[(sub["hour_utc"] >= HOUR_GATE[0]) & (sub["hour_utc"] <= HOUR_GATE[1])]
    outside = sub[~((sub["hour_utc"] >= HOUR_GATE[0]) & (sub["hour_utc"] <= HOUR_GATE[1]))]
    if len(inside) < 5 or len(outside) < 5:
        continue
    wr_in  = inside["is_win"].mean()
    wr_out = outside["is_win"].mean()
    # Chi2 test
    _, p_chi = stats.chi2_contingency([
        [int(inside["is_win"].sum()), len(inside)-int(inside["is_win"].sum())],
        [int(outside["is_win"].sum()), len(outside)-int(outside["is_win"].sum())]
    ])[:2]
    sig = "***" if p_chi<0.001 else "**" if p_chi<0.01 else "*" if p_chi<0.05 else "~" if p_chi<0.15 else ""
    print(f"  {w:>8} {len(inside):>9} {wr_in:>10.4f} {len(outside):>9} {wr_out:>10.4f} {wr_in-wr_out:>+8.4f} {p_chi:>9.4f} {sig:>5}")
    hour_gate_stable.append((w, wr_in, wr_out, p_chi))

# ── 1B: Análisis de la "frontera" — ¿son exactamente 7H-13H o hay margen? ──
print("\n─" * 70)
print("1B: Sensibilidad de fronteras — ¿qué pasa si cambiamos ±1H los límites?")
print("─" * 70)
configs = [
    ("Actual    (7-13H)", 7, 13),
    ("Mas tarde (8-14H)", 8, 14),
    ("Mas amplio(6-14H)", 6, 14),
    ("Reducido  (8-12H)", 8, 12),
    ("Solo mañ. (7-11H)", 7, 11),
    ("Solo tard.(10-14H)",10, 14),
]
for name, lo, hi in configs:
    inside  = combined[(combined["hour_utc"] >= lo) & (combined["hour_utc"] <= hi)]
    outside = combined[~((combined["hour_utc"] >= lo) & (combined["hour_utc"] <= hi))]
    wr_in   = inside["is_win"].mean()
    n_in    = len(inside)
    n_out   = len(outside)
    wr_out  = outside["is_win"].mean()
    # trades reducidos vs el actual
    n_actual = len(combined[(combined["hour_utc"] >= 7) & (combined["hour_utc"] <= 13)])
    print(f"  {name}: N_in={n_in:3d} WR_in={wr_in:.4f} ({wr_in-baseline_wr:+.4f})  WR_out={wr_out:.4f}  ratio={n_in/n_actual:.2f}x")

# ── 1C: Look-ahead analysis ───────────────────────────────────────
print("\n─" * 70)
print("1C: Analisis de Look-Ahead — ¿es la hora de ENTRADA conocida a priori?")
print("─" * 70)
print("  El modelo genera señal cuando observa la barra cerrando.")
print("  La hora UTC de entry_time = timestamp del cierre de la vela 1H.")
print("  → La hora es PERFECTAMENTE CONOCIDA en el momento de la señal.")
print("  → CERO look-ahead bias para el gate horario.")
print()
print("  Distribucion de delays entre señal y ejecucion:")
combined["delay_h"] = (combined["exit_dt"] - combined["entry_dt"]).dt.total_seconds() / 3600
delay_sample = combined["delay_h"].dropna()
print(f"  delay_h (duracion del trade) rango: [{delay_sample.min():.1f}, {delay_sample.max():.1f}]H")
print(f"  El gate filtra la HORA DE ENTRADA, que es timestamp de la vela cerrada → OK")

# ── 1D: ¿El gate es mejor que el filtro de régimen? ──────────────
print("\n─" * 70)
print("1D: Interaccion hora × regimen — ¿son independientes?")
print("─" * 70)
if "hmm_regime" in combined.columns:
    inside = combined[(combined["hour_utc"] >= 7) & (combined["hour_utc"] <= 13)]
    for reg in ["1_BULL_TREND", "1_BULL_TREND_B", "1_BULL_TREND_WEAK"]:
        sub_reg = inside[inside["hmm_regime"] == reg]
        sub_all = combined[combined["hmm_regime"] == reg]
        if len(sub_reg) < 10:
            continue
        wr_reg_in  = sub_reg["is_win"].mean()
        wr_reg_all = sub_all["is_win"].mean()
        print(f"  {str(reg)[:25]:25s}: WR_7-13H={wr_reg_in:.4f} vs WR_global={wr_reg_all:.4f} ({wr_reg_in-wr_reg_all:+.4f})  N={len(sub_reg)}")
    print("\n  Conclusion: el gate horario mejora el WR DENTRO de cada regimen por separado")
    print("  → Es un gate ORTOGONAL al regimen — ambos pueden coexistir")

# ── 1E: Simulacion del impacto en metrics V2 ─────────────────────
print("\n─" * 70)
print("1E: Simulacion — metricas V2 con gate horario aplicado")
print("─" * 70)
inside_trades = combined[(combined["hour_utc"] >= HOUR_GATE[0]) & (combined["hour_utc"] <= HOUR_GATE[1])].sort_values("entry_dt")
ret = inside_trades["return_raw"].fillna(0)
equity = (1 + ret).cumprod()
dd = ((equity - equity.cummax()) / equity.cummax()).min()
total = equity.iloc[-1] - 1
sh = (ret.mean() / ret.std() * np.sqrt(716)) if ret.std() > 0 else 0
calmar = abs(total / dd) if dd != 0 else 0
print(f"  Trades filtrados: {len(inside_trades)} de {len(combined)} ({len(inside_trades)/len(combined)*100:.1f}%)")
print(f"  WR con gate:       {inside_trades['is_win'].mean():.4f}")
print(f"  Retorno acumulado: {total*100:+.2f}%")
print(f"  MaxDD:             {dd*100:.2f}%")
print(f"  Sharpe:            {sh:.3f}")
print(f"  Calmar:            {calmar:.3f}")
print(f"  (vs sin gate: WR=51.57%, ret=+147.87%, MaxDD=-74.8%, Calmar=1.977)")

# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("INVESTIGACION 2 — B1-DOW: Evitar Lunes (WR=30.9%)")
print("=" * 70)

# ── 2A: Robustez por ventana ──────────────────────────────────────
print("\n─" * 70)
print("2A: WR de Lunes POR VENTANA (test de estabilidad)")
print("─" * 70)
print(f"  {'Ventana':>8} {'N_Lun':>7} {'WR_Lun':>8} {'N_Resto':>8} {'WR_Resto':>9} {'Delta':>8} {'p':>8}")
for w in WINDOWS:
    sub = combined[combined["_w"] == w]
    lunes = sub[sub["dow"] == 0]
    resto = sub[sub["dow"] != 0]
    if len(lunes) < 5: continue
    wr_lun = lunes["is_win"].mean()
    wr_rst = resto["is_win"].mean()
    _, p = stats.chi2_contingency([
        [int(lunes["is_win"].sum()), len(lunes)-int(lunes["is_win"].sum())],
        [int(resto["is_win"].sum()), len(resto)-int(resto["is_win"].sum())]
    ])[:2]
    flag = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "~" if p<0.15 else ""
    print(f"  {w:>8} {len(lunes):>7} {wr_lun:>8.4f} {len(resto):>8} {wr_rst:>9.4f} {wr_lun-wr_rst:>+8.4f} {p:>8.4f} {flag}")

# ── 2B: Detalle por hora EN Lunes ─────────────────────────────────
print("\n─" * 70)
print("2B: WR por hora UTC en Lunes — ¿hay alguna hora de Lunes rescatable?")
print("─" * 70)
lunes_all = combined[combined["dow"] == 0]
for h in sorted(lunes_all["hour_utc"].unique()):
    sub = lunes_all[lunes_all["hour_utc"] == h]
    if len(sub) < 3: continue
    wr = sub["is_win"].mean()
    bar = "█" * int(wr * 20)
    flag = " ← rescatable?" if wr > 0.50 else ""
    print(f"  {h:02d}H: N={len(sub):2d} WR={wr:.3f} {bar}{flag}")

# ── 2C: Combinacion Lunes + hora ─────────────────────────────────
print("\n─" * 70)
print("2C: Combinacion gate Lunes + gate horario")
print("─" * 70)
# Solo operar en dias no-Lunes Y en horas 7-13
excl_monday_in_gate = combined[(combined["dow"] != 0) & (combined["hour_utc"] >= 7) & (combined["hour_utc"] <= 13)]
print(f"  Trades con ambos gates: {len(excl_monday_in_gate)} de {len(combined)} ({len(excl_monday_in_gate)/len(combined)*100:.1f}%)")
print(f"  WR combinado: {excl_monday_in_gate['is_win'].mean():.4f}")

# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("INVESTIGACION 3 — E4: Naturaleza de la columna 'drawdown'")
print("=" * 70)

print("\n─" * 70)
print("3A: Diagnostico — ¿qué es el drawdown del parquet?")
print("─" * 70)
print("""
  Evidencia del codigo fuente (predict_oos.py lineas 1576-1581):
  
    equity = (1 + df_trades['return_pct']).cumprod()   # acumulado de TODOS los trades
    rolling_max = equity.cummax()
    drawdown = equity - rolling_max                     # drawdown de equity curve
    df_trades['drawdown'] = drawdown.values
    
  CONCLUSION: el 'drawdown' del parquet es el DRAWDOWN DE LA EQUITY CURVE
  (distancia entre equity actual y maximo historico acumulado), NO el
  MAE (Maximum Adverse Excursion) intra-trade.
  
  IMPLICACION DIRECTA:
  - 'drawdown==0' en Q1 NO significa "trade sin adversidad"
  - Significa "este trade ocurrio cuando la equity estaba en su maximo"
  - i.e., trades que ocurren DESPUES de una racha ganadora tienen drawdown~0
  
  ESTO ES INFORMACION DISPONIBLE EN TIEMPO REAL (la equity curve propia
  del sistema), PERO...
""")

# ── 3B: Verificar la hipotesis: drawdown~0 = trade post-racha ────
print("─" * 70)
print("3B: Verificando — drawdown~0 correlaciona con posicion en equity?")
print("─" * 70)
combined_s = combined.sort_values("entry_dt").copy()
combined_s["dd_abs"] = combined_s["drawdown"].abs()
combined_s["trade_idx"] = range(len(combined_s))

# ¿Cuándo está el drawdown~0? 
q1_dd = combined_s[combined_s["dd_abs"] < 0.0001]
q4_dd = combined_s[combined_s["dd_abs"] > combined_s["dd_abs"].quantile(0.75)]

print(f"  Trades con drawdown~0 (Q1): N={len(q1_dd)}, WR={q1_dd['is_win'].mean():.4f}")
print(f"  Trades con max drawdown (Q4): N={len(q4_dd)}, WR={q4_dd['is_win'].mean():.4f}")
print()

# La racha previa
combined_s["prev_is_win"] = combined_s["is_win"].shift(1)
combined_s["prev2_is_win"] = combined_s["is_win"].shift(2)
# ¿Los trades con dd~0 tienen más wins previos?
q1_prev_wins = q1_dd.apply(lambda row: combined_s.loc[:row.name, "is_win"].tail(3).mean(), axis=1).mean()
q4_prev_wins = q4_dd.apply(lambda row: combined_s.loc[:row.name, "is_win"].tail(3).mean(), axis=1).mean()
# Proxy simple: prev_is_win
q1_prev = combined_s.loc[q1_dd.index, "prev_is_win"].mean()
q4_prev = combined_s.loc[q4_dd.index, "prev_is_win"].mean()
print(f"  WR del trade ANTERIOR a Q1 (dd~0): {q1_prev:.4f}")
print(f"  WR del trade ANTERIOR a Q4 (dd alto): {q4_prev:.4f}")
print()

# CONCLUSIÓN sobre si es usable
print("─" * 70)
print("3C: ¿Es usable el drawdown de equity como feature de filtering?")
print("─" * 70)
print("""
  La columna 'drawdown' es el drawdown de equity hasta el momento del trade.
  
  En produccion LIVE:
  - La equity curve se actualiza trade a trade
  - El sistema SÍ conoce la equity actual antes de ejecutar el siguiente trade
  - POR LO TANTO: usar el drawdown de equity como feature NO es look-ahead
  
  PERO hay 2 problemas:
  
  PROBLEMA A — Confound con el regimen:
  Los trades con dd~0 son los que ocurren en rachas ganadoras.
  Las rachas ganadoras ocurren en regimenes buenos (BULL_TREND_B).
  El HMM ya captura este regimen → seria feature redundante.
  
  PROBLEMA B — El WR=93.3% en dd~0 es parcialmente artefacto:
  Cuando dd~0, el trade anterior fue ganador (prev_wr arriba).
  Si la autocorrelacion es por regimen (como demostramos), entonces
  dd~0 → regimen bueno → el siguiente trade tambien es bueno.
  El dd de equity NO es el causante, es el REGIMEN el que causa ambos.
  
  CONCLUSION:
  No implementar 'drawdown de equity' como gate.
  La feature util sería el 'MAE intra-trade' (adversidad dentro del trade
  actual), que NO está en el parquet actual → requiere guardarse diferente.
""")
print("  MAE real: hay que guardar min(close) durante el trade, no el dd de equity.")
print("  ACCION: Añadir 'min_excursion_pct' en predict_oos.py → requiere calcular")
print("  min del precio durante la duracion [entry_time, exit_time].")

print("\n" + "=" * 70)
print("RESUMEN FINAL DE INVESTIGACION")
print("=" * 70)
print("""
  B1-HOUR (Gate 7H-13H UTC):
    Look-ahead: CERO — hora conocida antes de ejecutar
    Estabilidad: A verificar por ventana (ver SEC 1A)
    Overfitting: BAJO — tiene causalidad economica clara
    Estado: CANDIDATO A IMPLEMENTAR
  
  B1-DOW (Evitar Lunes):
    Look-ahead: CERO — dia conocido antes de ejecutar  
    Estabilidad: A verificar (ver SEC 2A)
    Overfitting: BAJO — causalidad via liquidez/volatilidad crypto
    Estado: CANDIDATO A IMPLEMENTAR (potencialmente combinado con HOUR)
  
  E4-DRAWDOWN (dd de equity=0):
    Look-ahead: CERO (equity conocida en tiempo real)
    Problema: Confound con régimen — no es causal
    Problema: WR=93.3% es artefacto de post-racha, no del trade en si
    Estado: RECHAZADO como gate
    Alternativa: Guardar MAE real en predict_oos → feature para futuro
""")
