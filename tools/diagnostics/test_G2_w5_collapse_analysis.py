"""
test_G2_w5_collapse_analysis.py
================================
H-G2: ¿Por qué W5 (Q1-2026) tiene WR=36.8%?
¿Es decay continuo, salto abrupto, o evento específico?

Busca:
  1. Distribución temporal del WR dentro de W5
  2. Comparación de features/probabilidades W5 vs W3 (mejor ventana)
  3. Deteccion de structural break en la serie OOS completa
  4. Diferencias en regime distribution
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

print("=" * 65)
print("TEST G2 — Analisis Colapso W5 (Q1-2026, WR=36.8%)")
print("=" * 65)

# ── Cargar datos ──────────────────────────────────────────────────
WFB = Path("data/reports/wfb")
dfs = []
for f in sorted(WFB.glob("oos_trades_W*_seed*.parquet")):
    df = pd.read_parquet(f)
    df["_w"] = f.stem.split("_")[2]
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined["entry_dt"] = pd.to_datetime(combined["entry_time"], utc=True, errors="coerce")
combined["exit_dt"]  = pd.to_datetime(combined["exit_time"],  utc=True, errors="coerce")
combined["duration_h"] = (combined["exit_dt"] - combined["entry_dt"]).dt.total_seconds() / 3600

# ── SEC 1: WR por ventana (baseline) ─────────────────────────────
print("\n─" + "─" * 64)
print("SEC 1: WR por ventana — el problema de W5")
print("─" * 65)
for w in ["W1","W2","W3","W4","W5"]:
    sub = combined[combined["_w"] == w]
    if len(sub) == 0:
        continue
    wr = sub["is_win"].mean()
    t_min = sub["entry_dt"].min()
    t_max = sub["entry_dt"].max()
    bar = "█" * int(wr * 30)
    print(f"  {w}: WR={wr:.3f} N={len(sub):3d} | {t_min.date()} → {t_max.date()} | {bar}")

# ── SEC 2: WR rolling mensual sobre OOS completo ──────────────────
print("\n─" + "─" * 64)
print("SEC 2: WR rolling 30d — decay continuo o salto abrupto?")
print("─" * 65)
oos_sorted = combined.dropna(subset=["entry_dt","is_win"]).sort_values("entry_dt")
oos_sorted = oos_sorted.set_index("entry_dt")

# Rolling mensual
rolling_wr = oos_sorted["is_win"].rolling("30D").mean()
rolling_n  = oos_sorted["is_win"].rolling("30D").count()

# Imprimir por mes
monthly = oos_sorted["is_win"].resample("MS").agg(["mean","count"])
monthly.columns = ["WR","N"]
print(f"  {'Mes':<12} {'N':>5} {'WR':>7} {'Barra'}")
last_wr = None
for idx, row in monthly.iterrows():
    if row["N"] < 5:
        continue
    wr = row["WR"]
    n  = int(row["N"])
    delta = f"({wr-last_wr:+.3f})" if last_wr is not None else "       "
    bar = "█" * int(wr * 25)
    change = "⚠ CAIDA" if last_wr is not None and wr < last_wr - 0.05 else ""
    print(f"  {str(idx.date()):<12} {n:>5} {wr:>7.3f} {delta} {bar} {change}")
    last_wr = wr

# ── SEC 3: Test de cambio estructural (Chow test aproximado) ─────
print("\n─" + "─" * 64)
print("SEC 3: Deteccion de structural break")
print("─" * 65)

# Dividir OOS en dos mitades y comparar WR
mid_date = oos_sorted.index.min() + (oos_sorted.index.max() - oos_sorted.index.min()) / 2
early = oos_sorted[oos_sorted.index < mid_date]["is_win"]
late  = oos_sorted[oos_sorted.index >= mid_date]["is_win"]

print(f"  Corte temporal: {mid_date.date()}")
print(f"  EARLY ({oos_sorted.index.min().date()} → {mid_date.date()}): N={len(early)} WR={early.mean():.4f}")
print(f"  LATE  ({mid_date.date()} → {oos_sorted.index.max().date()}):  N={len(late)}  WR={late.mean():.4f}")

chi2, p_chi2 = stats.chi2_contingency([
    [int(early.sum()), len(early) - int(early.sum())],
    [int(late.sum()),  len(late)  - int(late.sum())]
])[:2]
print(f"  Chi² test early vs late: chi2={chi2:.3f}  p={p_chi2:.4f}")
if p_chi2 < 0.05:
    print(f"  → STRUCTURAL BREAK CONFIRMADO: la segunda mitad del OOS es significativamente peor")
else:
    print(f"  → No hay break estructural estadisticamente significativo")

# ── SEC 4: Buscar el mes exacto del quiebre ───────────────────────
print("\n─" + "─" * 64)
print("SEC 4: Localizacion del quiebre — Chow rolling")
print("─" * 65)
print("  (Probando cada mes como posible punto de quiebre...)")

months = monthly[monthly["N"] >= 10].index.tolist()
best_chi2, best_date = 0, None
for cut_date in months:
    before = oos_sorted[oos_sorted.index < cut_date]["is_win"]
    after  = oos_sorted[oos_sorted.index >= cut_date]["is_win"]
    if len(before) < 20 or len(after) < 20:
        continue
    c2, p2 = stats.chi2_contingency([
        [int(before.sum()), len(before) - int(before.sum())],
        [int(after.sum()),  len(after) - int(after.sum())]
    ])[:2]
    marker = " ← MAX BREAK" if c2 > best_chi2 else ""
    if c2 > best_chi2:
        best_chi2, best_date = c2, cut_date
    print(f"  Corte en {str(cut_date.date())}: chi2={c2:.3f} p={p2:.4f}{marker}")

if best_date:
    before_best = oos_sorted[oos_sorted.index < best_date]["is_win"]
    after_best  = oos_sorted[oos_sorted.index >= best_date]["is_win"]
    print(f"\n  MEJOR CORTE: {best_date.date()}")
    print(f"    Antes:  N={len(before_best)} WR={before_best.mean():.4f}")
    print(f"    Despues: N={len(after_best)}  WR={after_best.mean():.4f}")
    print(f"    Delta WR: {after_best.mean()-before_best.mean():+.4f}")

# ── SEC 5: Comparación W3 (mejor) vs W5 (peor) ───────────────────
print("\n─" + "─" * 64)
print("SEC 5: Comparacion feature-level W3 (WR=52%) vs W5 (WR=36.8%)")
print("─" * 65)
w3 = combined[combined["_w"] == "W3"]
w5 = combined[combined["_w"] == "W5"]

numeric_cols = ["meta_v2_prob","xgb_prob","xgb_prob_cal","tribe_mult",
                "ood_kl_distance","kelly_fraction_used","duration_h"]

print(f"  {'Feature':<25} {'W3 media':>10} {'W5 media':>10} {'Delta':>10} {'p-value':>10} {'Sig'}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*5}")
for col in numeric_cols:
    if col not in w3.columns or col not in w5.columns:
        continue
    v3 = w3[col].dropna()
    v5 = w5[col].dropna()
    if len(v3) < 5 or len(v5) < 5:
        continue
    _, p = stats.mannwhitneyu(v3, v5, alternative="two-sided")
    d3 = v3.mean()
    d5 = v5.mean()
    delta = d5 - d3
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {col:<25} {d3:>10.4f} {d5:>10.4f} {delta:>+10.4f} {p:>10.4f} {sig}")

# ── SEC 6: Regime distribution W3 vs W5 ──────────────────────────
print("\n─" + "─" * 64)
print("SEC 6: Distribucion de regimenes W3 vs W5")
print("─" * 65)
if "hmm_regime" in combined.columns:
    for reg in sorted(combined["hmm_regime"].dropna().unique()):
        n3 = (w3["hmm_regime"] == reg).sum()
        n5 = (w5["hmm_regime"] == reg).sum()
        pct3 = n3 / len(w3) * 100 if len(w3) > 0 else 0
        pct5 = n5 / len(w5) * 100 if len(w5) > 0 else 0
        if n3 + n5 < 5:
            continue
        delta_pct = pct5 - pct3
        print(f"  {str(reg)[:28]:30s}: W3={pct3:5.1f}% (N={n3:3d}) W5={pct5:5.1f}% (N={n5:3d}) delta={delta_pct:+5.1f}pp")

# ── SEC 7: OOD check — son los datos de W5 mas out-of-distribution?
print("\n─" + "─" * 64)
print("SEC 7: OOD distance W3 vs W5")
print("─" * 65)
ood3 = w3["ood_kl_distance"].dropna()
ood5 = w5["ood_kl_distance"].dropna()
if len(ood3) > 5 and len(ood5) > 5:
    _, p_ood = stats.mannwhitneyu(ood3, ood5, alternative="two-sided")
    print(f"  OOD_KL W3: media={ood3.mean():.4f} med={ood3.median():.4f}")
    print(f"  OOD_KL W5: media={ood5.mean():.4f} med={ood5.median():.4f}")
    print(f"  MW p={p_ood:.4f} {'→ W5 MAS OOD (datos mas anómalos)' if ood5.mean() > ood3.mean() and p_ood < 0.05 else '→ Sin diferencia significativa'}")

# ── VEREDICTO ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VEREDICTO G2")
print("=" * 65)
w5_wr = combined[combined["_w"] == "W5"]["is_win"].mean()
w3_wr = combined[combined["_w"] == "W3"]["is_win"].mean()
print(f"  WR W5={w5_wr:.4f} vs W3={w3_wr:.4f} (delta={w5_wr-w3_wr:+.4f})")
chi2_35, p_35 = stats.chi2_contingency([
    [int(w3["is_win"].sum()), len(w3) - int(w3["is_win"].sum())],
    [int(w5["is_win"].sum()), len(w5) - int(w5["is_win"].sum())]
])[:2]
print(f"  Chi² W3 vs W5: chi2={chi2_35:.3f}  p={p_35:.4f}")
if p_35 < 0.05:
    print(f"  → CONFIRMADO: W5 es estadisticamente diferente de W3")
    if best_date:
        print(f"  → Quiebre localizado en: {best_date.date()}")
    print(f"  → Ver SEC 5 para features mas distintas entre ventanas")
else:
    print(f"  → Diferencia W3/W5 no estadisticamente significativa con N actual")
