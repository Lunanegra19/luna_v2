"""
test_F3_B4_diagnostico_autocorrelacion.py
==========================================
El resultado de B4 es sospechoso:
  WR tras WIN: 87% vs WR tras LOSS: 14% → delta=73pp
  rho Lag-1 = 0.728 — esto es ANORMALMENTE alto

Hipotesis:
  a) Trades de la misma seed en la misma ventana son cronologicamente
     ordenados y provienen del mismo modelo — si el modelo gana hoy,
     gana mañana (correlacion de series temporales misma seed)
  b) Hay repeticion de trades: el mismo trade aparece en multiples seeds
  c) El momentum_3trades = retorno acumulado de los 3 trades previos
     que puede incluir trades de DIFERENTES seeds — mezcla temporal

Objetivo: distinguir autocorrelacion REAL (el mercado tiene momentum)
de artefacto de POOL de seeds (mismo periodo temporal = mismo resultado)
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
combined["exit_dt"]  = pd.to_datetime(combined["exit_time"], utc=True, errors="coerce")
combined = combined.sort_values("entry_dt").reset_index(drop=True)
baseline_wr = combined["is_win"].mean()

print("=" * 65)
print("DIAGNOSTICO B4/F3 — Origen de la autocorrelacion")
print("=" * 65)

# ── Test 1: Autocorrelacion POR SEED individual ───────────────────
print("\n─" * 65)
print("SEC 1: Autocorrelacion lag-1 POR SEED INDIVIDUAL")
print("─" * 65)
print("  (Si la autocorr viene del mercado, debe existir en CADA seed)")
print(f"  {'Seed':>10} {'N':>5} {'rho_lag1':>10} {'p':>8} {'WR_tras_W':>10} {'WR_tras_L':>10}")
any_real_autocorr = False
for seed in sorted(combined["_seed"].unique()):
    sub = combined[combined["_seed"] == seed].sort_values("entry_dt")
    if len(sub) < 20:
        continue
    wins = sub["is_win"].values
    rho = pd.Series(wins.astype(float)).autocorr(lag=1)
    n   = len(wins)
    # t-test para la autocorrelacion
    t_stat = rho * np.sqrt((n-2) / (1 - rho**2)) if abs(rho) < 1 else np.inf
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

    # WR tras win/loss dentro de la misma seed
    wr_aw = sum(wins[i]==1 and wins[i+1]==1 for i in range(n-1)) / max(sum(wins[i]==1 for i in range(n-1)), 1)
    wr_al = sum(wins[i]==0 and wins[i+1]==1 for i in range(n-1)) / max(sum(wins[i]==0 for i in range(n-1)), 1)

    flag = " *** REAL" if p_val < 0.01 else " *" if p_val < 0.05 else ""
    if p_val < 0.05:
        any_real_autocorr = True
    print(f"  {seed:>10} {n:>5} {rho:>+10.4f} {p_val:>8.4f} {wr_aw:>10.4f} {wr_al:>10.4f}{flag}")

print(f"\n  Autocorrelacion real en alguna seed: {'SI — momentum existe' if any_real_autocorr else 'NO — es artefacto del pool'}")

# ── Test 2: Trades con mismo entry_time en diferentes seeds ──────
print("\n─" * 65)
print("SEC 2: Solapamiento temporal entre seeds")
print("─" * 65)
combined["entry_dt_floor"] = combined["entry_dt"].dt.floor("4h")
overlap = combined.groupby("entry_dt_floor")["_seed"].nunique()
print(f"  Buckets 4H con >=2 seeds operando simultaneamente: {(overlap>=2).sum()} / {len(overlap)}")
print(f"  Buckets con TODAS las seeds: {(overlap==combined['_seed'].nunique()).sum()}")
print(f"  Buckets con 1 sola seed: {(overlap==1).sum()}")

# Cuando múltiples seeds operan en el mismo período, ¿coinciden en is_win?
print("\n  Correlacion de is_win entre seeds en mismo bucket 4H:")
pivot_wins = combined.pivot_table(index="entry_dt_floor", columns="_seed",
                                   values="is_win", aggfunc="mean")
# Seeds con mas datos
main_seeds = combined["_seed"].value_counts().head(5).index.tolist()
for s in main_seeds[:3]:
    if s in pivot_wins.columns:
        for s2 in main_seeds[:3]:
            if s2 != s and s2 in pivot_wins.columns:
                common = pivot_wins[[s, s2]].dropna()
                if len(common) >= 10:
                    rho_w, p_w = stats.pearsonr(common[s], common[s2])
                    print(f"  {s} vs {s2}: rho={rho_w:+.4f}  p={p_w:.4f}  N={len(common)}")

# ── Test 3: Momentum real con precio BTC ─────────────────────────
print("\n─" * 65)
print("SEC 3: Verificacion — ¿el momentum correlaciona con is_win realmente?")
print("─" * 65)
# El momentum_3trades calculado incluye trades de DISTINTAS seeds mezcladas
# Calcular momentum DENTRO de cada seed por separado
for seed in sorted(combined["_seed"].unique())[:5]:
    sub = combined[combined["_seed"] == seed].sort_values("entry_dt").copy()
    if len(sub) < 15:
        continue
    sub["mom3"] = sub["return_raw"].rolling(3, min_periods=3).sum().shift(1)
    valid = sub.dropna(subset=["mom3","is_win"])
    if len(valid) < 10:
        continue
    rho, p = stats.spearmanr(valid["mom3"], valid["is_win"])
    print(f"  Seed {seed:>6}: N={len(valid):3d} momentum_3 rho={rho:+.4f}  p={p:.4f}  {'*' if p<0.05 else ''}")

# ── Test 4: Patron temporal del momentum ─────────────────────────
print("\n─" * 65)
print("SEC 4: Causa real de momentum_3 = rho 0.56")
print("─" * 65)
# El momentum_3trades en el pool mezclado refleja que
# cuando el mercado es bueno, TODAS las seeds ganan a la vez
# → los 3 trades previos (de seeds distintas) ganaron porque el mercado era alcista
# → el trade actual también gana por el mismo mercado alcista
# Esto es CORRELACION CON EL REGIMEN, no autocorrelacion causal

# Comprobar: momentum_3 predice hmm_regime?
if "hmm_regime" in combined.columns:
    combined_s = combined.sort_values("entry_dt").copy()
    combined_s["mom3_pool"] = combined_s["return_raw"].rolling(3, min_periods=3).sum().shift(1)
    valid_reg = combined_s.dropna(subset=["mom3_pool","hmm_regime"])
    regime_dummies = pd.get_dummies(valid_reg["hmm_regime"])
    print("  Correlacion momentum_3 (pool) con cada regimen HMM:")
    for col in regime_dummies.columns:
        rho_r, p_r = stats.spearmanr(valid_reg["mom3_pool"], regime_dummies[col])
        print(f"    {str(col)[:30]:30s}: rho={rho_r:+.4f}  p={p_r:.4f}")

print("\n" + "=" * 65)
print("CONCLUSION DIAGNOSTICO")
print("=" * 65)
if any_real_autocorr:
    print("  AUTOCORRELACION REAL en seeds individuales")
    print("  → El mercado tiene momentum que el modelo puede explotar")
    print("  ACCION: Añadir feature 'drawdown_rolling_equity' al meta-modelo")
else:
    print("  ARTEFACTO CONFIRMADO:")
    print("  El momentum_3 del pool mezclado refleja el REGIMEN, no causalidad")
    print("  → Cuando el mercado es alcista, TODAS las seeds ganan a la vez")
    print("  → El trade siguiente también gana por el mismo régimen")
    print("  El rho=0.728 en B4 es un artefacto de la correlacion temporal de seeds")
    print("  ACCION: No implementar filtro de momentum — es espurio")
    print("  La feature ya capturada es hmm_regime (que sí tiene información real)")
