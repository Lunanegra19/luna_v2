"""
test_E1_kelly_vs_return.py + test_G1_edge_decay.py
===================================================
E1: Kelly fraction usado vs retorno real — ¿está bien calibrado?
G1: Edge decay rolling 30d — ¿el modelo se degrada con el tiempo?
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
    df = pd.read_parquet(f)
    df["_w"] = f.stem.split("_")[2]
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined["entry_dt"] = pd.to_datetime(combined["entry_time"], utc=True, errors="coerce")
combined_sorted = combined.sort_values("entry_dt").reset_index(drop=True)

print("=" * 65)
print("TEST E1 — Kelly fraction vs Retorno Real")
print("=" * 65)

valid_e1 = combined.dropna(subset=["kelly_fraction_used","return_raw","is_win"])
valid_e1 = valid_e1[valid_e1["kelly_fraction_used"] > 0].copy()
baseline_wr = combined["is_win"].mean()

print(f"\nTrades con kelly>0: {len(valid_e1)} / {len(combined)}")
print(f"Kelly media: {valid_e1['kelly_fraction_used'].mean():.4f}")
print(f"Kelly rango: [{valid_e1['kelly_fraction_used'].min():.4f}, {valid_e1['kelly_fraction_used'].max():.4f}]")

# Correlacion Kelly vs WR
rho_wr, p_wr = stats.spearmanr(valid_e1["kelly_fraction_used"], valid_e1["is_win"])
rho_ret, p_ret = stats.spearmanr(valid_e1["kelly_fraction_used"], valid_e1["return_raw"])
print(f"\nSpearman kelly vs is_win:    rho={rho_wr:.4f}  p={p_wr:.4f}")
print(f"Spearman kelly vs return_raw: rho={rho_ret:.4f}  p={p_ret:.4f}")

# Por cuartil de kelly
print("\nWR por cuartil de kelly_fraction_used:")
try:
    valid_e1["kelly_q"] = pd.qcut(valid_e1["kelly_fraction_used"], q=4,
                                   labels=["Q1_bajo","Q2","Q3","Q4_alto"], duplicates="drop")
except ValueError:
    # Si hay pocos valores únicos, usar corte por valor
    valid_e1["kelly_q"] = pd.cut(valid_e1["kelly_fraction_used"],
                                  bins=4, labels=["Q1_bajo","Q2","Q3","Q4_alto"])
for q in ["Q1_bajo","Q2","Q3","Q4_alto"]:
    sub = valid_e1[valid_e1["kelly_q"] == q]
    if len(sub) == 0:
        continue
    lo  = sub["kelly_fraction_used"].min()
    hi  = sub["kelly_fraction_used"].max()
    wr  = sub["is_win"].mean()
    ret = sub["return_raw"].mean()
    print(f"  {q}: N={len(sub):3d} | kelly=[{lo:.4f},{hi:.4f}] | WR={wr:.4f} ({wr-baseline_wr:+.3f}) | ret_raw={ret:+.5f}")

# Efecto del kelly en el retorno ponderado
valid_e1["weighted_return"] = valid_e1["kelly_fraction_used"] * valid_e1["return_raw"]
print(f"\nRetorno ponderado por Kelly: {valid_e1['weighted_return'].mean():.6f}")
print(f"Retorno sin ponderar:         {valid_e1['return_raw'].mean():.6f}")
print(f"  → Kelly {'mejora' if valid_e1['weighted_return'].mean() > valid_e1['return_raw'].mean() else 'no mejora'} el retorno ponderado")

# Costo efectivo
if "return_pct" in combined.columns:
    valid_cost = combined.dropna(subset=["return_raw","return_pct"]).copy()
    valid_cost["cost_effective"] = valid_cost["return_raw"] - valid_cost["return_pct"]
    print(f"\nCosto efectivo por trade:")
    print(f"  Media: {valid_cost['cost_effective'].mean():.5f} ({valid_cost['cost_effective'].mean()*100:.3f}%)")
    print(f"  Asumido en SOP: 0.150%")
    assumed = 0.0015
    actual  = valid_cost["cost_effective"].mean()
    if abs(actual) > assumed * 1.5:
        print(f"  ⚠ DISCREPANCIA: costo real {actual*100:.3f}% vs asumido {assumed*100:.3f}%")
    else:
        print(f"  OK: costo real ≈ asumido")

print(f"\nVEREDICTO E1:")
if p_wr < 0.05 and rho_wr > 0:
    print(f"  CONFIRMADA: Kelly mayor → WR mayor (rho={rho_wr:.3f}, p={p_wr:.4f})")
    print(f"  El Kelly está correctamente calibrado")
elif p_wr < 0.05 and rho_wr < 0:
    print(f"  INVERTIDA: Kelly mayor → WR MENOR (rho={rho_wr:.3f}, p={p_wr:.4f})")
    print(f"  ⚠ El Kelly asigna más capital a señales peores — revisar calibración")
else:
    print(f"  DESCARTADA: Kelly no correlaciona con WR (p={p_wr:.4f})")
    print(f"  El sizing es independiente de la calidad de la señal")

# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("TEST G1 — Edge Decay Rolling")
print("=" * 65)

valid_g1 = combined_sorted.dropna(subset=["entry_dt","is_win"]).copy()
valid_g1 = valid_g1.set_index("entry_dt").sort_index()

# Rolling 30d WR
rolling_wr = valid_g1["is_win"].rolling("30D").mean()
rolling_n  = valid_g1["is_win"].rolling("30D").count()

# Filtrar solo donde hay N >= 15
valid_rolling = rolling_wr[rolling_n >= 15]

print(f"\nPuntos de WR rolling (30d, min_n=15): {len(valid_rolling)}")

if len(valid_rolling) > 10:
    # Tendencia: regresion lineal de WR_rolling vs tiempo
    t_numeric = (valid_rolling.index - valid_rolling.index.min()).total_seconds() / (30*24*3600)
    slope, intercept, r_val, p_lin, se = stats.linregress(t_numeric, valid_rolling.values)
    print(f"Regresion lineal WR_rolling(30d) vs tiempo:")
    print(f"  Pendiente: {slope:+.4f} WR/mes   (negativa = decay)")
    print(f"  R²: {r_val**2:.4f}   p: {p_lin:.4f}")
    print(f"  WR proyectado al inicio: {intercept:.4f}")
    print(f"  WR proyectado al final:  {intercept + slope*t_numeric.max():.4f}")

    if p_lin < 0.05 and slope < 0:
        print(f"\n  → EDGE DECAY CONFIRMADO: -{abs(slope):.3f} WR/mes")
        print(f"     Al ritmo actual, el edge se agotará en {abs(intercept/slope):.0f} meses adicionales")
    elif slope < -0.01:
        print(f"\n  → TENDENCIA NEGATIVA: -{abs(slope):.3f} WR/mes (p={p_lin:.3f})")
    else:
        print(f"\n  → Sin decay estadisticamente significativo (p={p_lin:.3f})")

# WR por cuartil temporal (Q1=primero, Q4=último)
valid_g1["t_rank"] = pd.qcut(range(len(valid_g1)), q=4,
                              labels=["Q1_primero","Q2","Q3","Q4_ultimo"])
print(f"\nWR por cuartil temporal:")
for q in ["Q1_primero","Q2","Q3","Q4_ultimo"]:
    sub = valid_g1[valid_g1["t_rank"] == q]
    t_lo = sub.index.min().date()
    t_hi = sub.index.max().date()
    wr   = sub["is_win"].mean()
    print(f"  {q}: N={len(sub):3d} | {t_lo} → {t_hi} | WR={wr:.4f} ({wr-baseline_wr:+.3f})")

# Regresion por ventanas
print(f"\nWR por ventana (evidencia de decay):")
window_stats = []
for w in ["W2","W3","W4","W5"]:
    sub = valid_g1[valid_g1["_w"] == w]
    if len(sub) < 10:
        continue
    wr = sub["is_win"].mean()
    t_mid = sub.index.mean()
    window_stats.append((w, t_mid, wr, len(sub)))
    print(f"  {w}: N={len(sub):3d} | WR={wr:.4f} | fecha_media={t_mid.date()}")

if len(window_stats) > 2:
    t_vals = [(ws[1] - window_stats[0][1]).total_seconds()/(30*24*3600) for ws in window_stats]
    wr_vals = [ws[2] for ws in window_stats]
    slope_w, intercept_w, r_w, p_w, _ = stats.linregress(t_vals, wr_vals)
    print(f"\n  Regresion ventanas: slope={slope_w:+.4f}/mes  p={p_w:.4f}")
    if p_w < 0.05 and slope_w < 0:
        print(f"  → DECAY CONFIRMADO en ventanas: -{abs(slope_w):.3f} WR/mes")
    else:
        print(f"  → Decay en ventanas no significativo (p={p_w:.3f})")

print(f"\nVEREDICTO G1:")
if p_lin < 0.05 and slope < 0:
    print(f"  CONFIRMADO: Edge decay de {abs(slope):.3f} WR/mes")
    print(f"  ACCION: Establecer criterio de reentrenamiento cada X meses")
elif slope < -0.005:
    print(f"  TENDENCIA: Posible decay ({slope:+.4f}/mes) — monitorear")
    print(f"  ACCION: Vigilar el WR rolling en la próxima run")
else:
    print(f"  DESCARTADO: Sin decay estadisticamente significativo")
    print(f"  El modelo mantiene su edge en el periodo analizado")
