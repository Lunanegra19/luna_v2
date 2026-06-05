"""
arch02_n_statistical_power.py
================================
ARCH-02: N estadístico insuficiente — bucle pseudo-diagnóstico

PROTOCOLO:
  FASE 1: Medir N real de trades por régimen en las últimas runs
  FASE 2: Calcular poder estadístico real con ese N
  FASE 3: Estimar N necesario para diferentes niveles de confianza
  FASE 4: Verificar si el WR observado tiene p-value significativo
  FASE 5: Counterfactual — ¿cuántas runs serían necesarias para alcanzar N suficiente?

USO: python tools/diagnostics/arch02_n_statistical_power.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binomtest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

PREDICTIONS_DIR = ROOT / "data" / "predictions"
RUNS_DIR        = ROOT / "data" / "runs"

print("=" * 70)
print("[ARCH-02] DIAGNOSTICO: N Estadistico y Poder de las Pruebas")
print("=" * 70)

# ── Cargar todos los trades ────────────────────────────────────────────────────
all_dfs = []
for f in sorted(PREDICTIONS_DIR.glob("oos_trades_seed*.parquet")):
    try:
        df = pd.read_parquet(f)
        df["seed"] = int(f.stem.replace("oos_trades_seed", ""))
        all_dfs.append(df)
    except:
        pass

df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

if df_all.empty:
    print("  [ERROR] No se encontraron trades")
    sys.exit(1)

ret_col = next((c for c in ["return_pct", "return_raw"] if c in df_all.columns), None)
if ret_col:
    df_all["ret"] = df_all[ret_col] / (100.0 if df_all[ret_col].abs().median() > 1 else 1.0)

regime_col = next((c for c in ["hmm_regime", "HMM_Semantic", "regime"] if c in df_all.columns), None)
if regime_col:
    df_all["regime"] = df_all[regime_col].astype(str)
else:
    df_all["regime"] = "ALL"

print(f"  Total trades cargados: {len(df_all):,} | Seeds: {df_all['seed'].nunique()}")

# ── FASE 1: N por régimen y seed ───────────────────────────────────────────────
print("\n[FASE 1] N POR REGIMEN (todas las seeds)")
print("-" * 60)

regime_n = df_all.groupby("regime").agg(
    N=("ret", "count"),
    WR=("ret", lambda x: (x > 0).mean()),
    seeds=("seed", "nunique")
).reset_index()
regime_n["WR_pct"] = (regime_n["WR"] * 100).round(1)
print(regime_n[["regime", "N", "WR_pct", "seeds"]].to_string(index=False))

# ── FASE 2: N por ventana WFB (single seed) ───────────────────────────────────
print("\n[FASE 2] N POR VENTANA WFB (seed 42 — ultima run)")
print("-" * 60)

wfb_col = next((c for c in ["wfb_window", "window"] if c in df_all.columns), None)
if wfb_col:
    seed_42 = df_all[df_all["seed"] == 42] if 42 in df_all["seed"].values else df_all[df_all["seed"] == df_all["seed"].mode()[0]]
    wfb_n = seed_42.groupby([wfb_col, "regime"]).agg(
        N=("ret", "count"),
        WR=("ret", lambda x: round((x > 0).mean() * 100, 1))
    ).reset_index()
    print(wfb_n.to_string(index=False))
else:
    print("  [WARN] No hay columna de ventana WFB en el parquet")

# ── FASE 3: Poder estadístico ──────────────────────────────────────────────────
print("\n[FASE 3] PODER ESTADISTICO — N necesario para detectar mejoras")
print("-" * 60)

from math import ceil

def n_for_binomial(p0, p1, alpha=0.05, power=0.80):
    """N mínimo para detectar mejora WR de p0 → p1 con potencia dada."""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta  = norm.ppf(power)
    p_mean  = (p0 + p1) / 2
    n = ((z_alpha * np.sqrt(2 * p_mean * (1 - p_mean)) +
          z_beta  * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) /
         (p1 - p0)) ** 2
    return ceil(n)

print("  Mejora WR detectable vs N:")
print(f"  {'WR base':>8} {'WR objetivo':>12} {'N (poder 80%)':>15} {'N (poder 95%)':>15}")
for p0, p1 in [(0.50, 0.55), (0.50, 0.60), (0.50, 0.65), (0.55, 0.60), (0.55, 0.65)]:
    n80 = n_for_binomial(p0, p1, power=0.80)
    n95 = n_for_binomial(p0, p1, power=0.95)
    print(f"  {p0:.0%}     {p1:.0%}        {n80:>15,}  {n95:>15,}")

# ── FASE 4: p-value del WR observado ──────────────────────────────────────────
print("\n[FASE 4] P-VALUES POR REGIMEN (binomial test WR > 50%)")
print("-" * 60)

for _, row in regime_n.iterrows():
    n = int(row["N"])
    wr = row["WR"]
    n_wins = int(n * wr)
    result = binomtest(n_wins, n, 0.5, alternative="greater")
    sig = "✓ SIGNIFICATIVO p<0.05" if result.pvalue < 0.05 else "✗ NO significativo"
    ci_low, ci_high = result.proportion_ci(confidence_level=0.95)
    print(f"  {row['regime']:30s}: N={n:5,} WR={wr*100:.1f}%  p={result.pvalue:.4f}  IC95=[{ci_low*100:.1f}%, {ci_high*100:.1f}%]  {sig}")

# ── FASE 5: Counterfactual — cuántas runs para N suficiente ───────────────────
print("\n[FASE 5] COUNTERFACTUAL — Runs necesarias para N estadístico valido")
print("-" * 60)

# N promedio por run y régimen
n_seeds = df_all["seed"].nunique()
for _, row in regime_n.iterrows():
    n_total = int(row["N"])
    n_por_seed = n_total / max(n_seeds, 1)
    n_needed_80 = n_for_binomial(0.55, 0.60, power=0.80)  # escenario conservador
    runs_needed = ceil(n_needed_80 / max(n_por_seed, 0.1))
    print(f"  {row['regime']:30s}: N={n_total:5,} ({n_por_seed:.1f}/seed) "
          f"→ N_needed={n_needed_80} → {runs_needed} seeds adicionales")

print("\n" + "=" * 70)
print("RESUMEN ARCH-02")
print("=" * 70)
print(f"  N total multi-seed: {len(df_all):,} trades en {n_seeds} seeds")
print(f"  N por seed: {len(df_all)/n_seeds:.1f} trades/seed promedio")
print(f"  N para detectar WR 55%→60% (p<0.05, poder=80%): {n_for_binomial(0.55, 0.60):,} trades")
print(f"  N para detectar WR 50%→55% (p<0.05, poder=80%): {n_for_binomial(0.50, 0.55):,} trades")

print("\n[ARCH-02] Diagnostico completado.")
