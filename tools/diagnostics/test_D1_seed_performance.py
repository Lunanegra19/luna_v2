"""
test_D1_seed_performance.py
============================
H-D1: ¿Hay seeds sistemáticamente mejores que otras?
- ¿Alguna seed tiene WR consistentemente superior en TODAS las ventanas?
- ¿Cuál es la correlación de retornos entre seeds? (independencia)
- ¿Hay seeds que sean especialistas por régimen?
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations

print("=" * 65)
print("TEST D1 — Análisis de Performance por Seed")
print("=" * 65)

WFB = Path("data/reports/wfb")
dfs = []
for f in sorted(WFB.glob("oos_trades_W*_seed*.parquet")):
    parts = f.stem.split("_")
    # format: oos_trades_W2_seed42  → parts = ['oos','trades','W2','seed42']
    df = pd.read_parquet(f)
    df["_w"]    = parts[2]                          # W2, W3, ...
    df["_seed"] = parts[3].replace("seed", "")      # 42, 100, ...
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined["entry_dt"] = pd.to_datetime(combined["entry_time"], utc=True, errors="coerce")

seeds = sorted(combined["_seed"].unique())
windows = sorted(combined["_w"].unique())
baseline_wr = combined["is_win"].mean()

print(f"\nSeeds encontradas: {seeds}")
print(f"Ventanas: {windows}")
print(f"WR baseline global: {baseline_wr:.4f}")
print(f"N total trades: {len(combined)}")

# ── SEC 1: WR por seed (global) ───────────────────────────────────
print("\n─" * 65)
print("SEC 1: WR global por seed — ranking")
print("─" * 65)
print(f"  {'Seed':>10} {'N':>5} {'WR':>7} {'vs_base':>8} {'Wins':>5} {'Barra'}")
seed_global = {}
for seed in seeds:
    sub = combined[combined["_seed"] == seed]
    wr  = sub["is_win"].mean()
    n   = len(sub)
    seed_global[seed] = (n, wr)

for seed, (n, wr) in sorted(seed_global.items(), key=lambda x: -x[1][1]):
    bar = "█" * int(wr * 25)
    flag = " ⭐" if wr > baseline_wr + 0.05 else " ⚠" if wr < baseline_wr - 0.05 else ""
    print(f"  {seed:>10} {n:>5} {wr:>7.4f} {wr-baseline_wr:>+8.4f} {int(wr*n):>5} {bar}{flag}")

# ANOVA entre seeds
seed_groups = [combined[combined["_seed"] == s]["is_win"].values for s in seeds]
kw_stat, kw_p = stats.kruskal(*seed_groups)
print(f"\n  Kruskal-Wallis entre seeds: H={kw_stat:.3f}  p={kw_p:.4f}")
if kw_p < 0.05:
    print(f"  → SIGNIFICATIVO: hay seeds con WR estadisticamente diferente")
else:
    print(f"  → No hay diferencia estadisticamente significativa entre seeds")

# ── SEC 2: WR por seed × ventana (tabla cruzada) ─────────────────
print("\n─" * 65)
print("SEC 2: WR por (seed, ventana) — tabla cruzada")
print("─" * 65)
header = f"  {'Seed':>10} " + " ".join(f"{w:>8}" for w in windows) + "  MEDIA"
print(header)

seed_w_table = {}
for seed in seeds:
    row_vals = []
    for w in windows:
        sub = combined[(combined["_seed"] == seed) & (combined["_w"] == w)]
        wr = sub["is_win"].mean() if len(sub) >= 5 else float("nan")
        row_vals.append(wr)
    seed_w_table[seed] = row_vals
    vals_str = " ".join(f"{v:>8.3f}" if not np.isnan(v) else "     ---" for v in row_vals)
    mean_wr = np.nanmean(row_vals)
    star = " ⭐" if mean_wr > baseline_wr + 0.04 else ""
    print(f"  {seed:>10} {vals_str}  {mean_wr:.3f}{star}")

# ── SEC 3: Consistencia — ¿cuántas ventanas gana cada seed? ──────
print("\n─" * 65)
print("SEC 3: Consistencia por seed (ventanas con WR > baseline)")
print("─" * 65)
for seed in seeds:
    vals = seed_w_table[seed]
    wins_above = sum(1 for v in vals if not np.isnan(v) and v > baseline_wr)
    n_valid    = sum(1 for v in vals if not np.isnan(v))
    consistency = wins_above / n_valid if n_valid > 0 else 0
    bar = "█" * wins_above + "░" * (n_valid - wins_above)
    print(f"  {seed:>10}: {wins_above}/{n_valid} ventanas WR>baseline  [{bar}]  {'CONSISTENTE' if consistency >= 0.67 else ''}")

# ── SEC 4: Correlación entre seeds (independencia) ────────────────
print("\n─" * 65)
print("SEC 4: Correlacion de retornos entre seeds (independencia)")
print("─" * 65)

# Construir pivot: filas=entry_time (hourly bucket), cols=seed, vals=return_raw
combined["hour_bucket"] = combined["entry_dt"].dt.floor("4h")
pivot = combined.pivot_table(index="hour_bucket", columns="_seed",
                              values="return_raw", aggfunc="mean")

print(f"  Buckets temporales comunes (4H): {pivot.dropna().shape[0]}")
print(f"  Matriz de correlacion (Pearson):")

all_rhos = []
for s1, s2 in combinations(seeds, 2):
    if s1 in pivot.columns and s2 in pivot.columns:
        common = pivot[[s1, s2]].dropna()
        if len(common) < 10:
            continue
        rho, p = stats.pearsonr(common[s1], common[s2])
        all_rhos.append(rho)
        bar = "█" * int(abs(rho) * 20)
        sign = "+" if rho > 0 else "-"
        print(f"  {s1:>10} vs {s2:>10}: rho={rho:+.3f}  p={p:.4f}  {sign}{bar}")

if all_rhos:
    print(f"\n  Correlacion media entre seeds: {np.mean(all_rhos):.4f}")
    print(f"  Correlacion max: {max(all_rhos):.4f}  min: {min(all_rhos):.4f}")
    if np.mean(all_rhos) < 0.15:
        print(f"  → SEEDS INDEPENDIENTES: correlacion baja — el ensemble aporta diversificacion real")
    elif np.mean(all_rhos) < 0.35:
        print(f"  → Correlacion moderada — diversificacion parcial")
    else:
        print(f"  → Alta correlacion — seeds redundantes, el ensemble no diversifica bien")

# ── SEC 5: Especialización por régimen ───────────────────────────
print("\n─" * 65)
print("SEC 5: Especializacion por regimen HMM por seed")
print("─" * 65)
if "hmm_regime" in combined.columns:
    regimes = combined["hmm_regime"].dropna().unique()
    for reg in regimes:
        sub_reg = combined[combined["hmm_regime"] == reg]
        if len(sub_reg) < 30:
            continue
        print(f"\n  Regimen: {reg} (N={len(sub_reg)}, WR_global={sub_reg['is_win'].mean():.3f})")
        seed_wr_reg = {}
        for seed in seeds:
            s = sub_reg[sub_reg["_seed"] == seed]
            if len(s) >= 10:
                seed_wr_reg[seed] = (len(s), s["is_win"].mean())
        for seed, (n, wr) in sorted(seed_wr_reg.items(), key=lambda x: -x[1][1]):
            flag = " ⭐ ESPECIALISTA" if wr > sub_reg["is_win"].mean() + 0.10 else ""
            print(f"    {seed:>10}: N={n:3d} WR={wr:.4f}{flag}")

# ── SEC 6: ¿Hay una "super seed"? ────────────────────────────────
print("\n─" * 65)
print("SEC 6: Identificacion de super-seed")
print("─" * 65)
print("  (Seed con WR > baseline en >= 3 de 4 ventanas)")
super_seeds = []
for seed in seeds:
    vals = [v for v in seed_w_table[seed] if not np.isnan(v)]
    above = sum(1 for v in vals if v > baseline_wr)
    if above >= 3 and len(vals) >= 3:
        mean_wr = np.mean(vals)
        super_seeds.append((seed, above, len(vals), mean_wr))
        print(f"  ⭐ SUPER-SEED {seed}: {above}/{len(vals)} ventanas > baseline | WR_media={mean_wr:.4f}")

if not super_seeds:
    print(f"  No hay super-seeds. Ninguna seed supera el baseline en >=3 ventanas.")
    print(f"  → El ensemble de seeds es necesario: ninguna seed individual domina")

# ── SEC 7: Retorno acumulado por seed ────────────────────────────
print("\n─" * 65)
print("SEC 7: Retorno acumulado (equity) por seed")
print("─" * 65)
for seed in seeds:
    sub = combined[combined["_seed"] == seed].sort_values("entry_dt")
    if len(sub) < 10:
        continue
    cumret = sub["return_raw"].cumsum().iloc[-1]
    maxdd  = (sub["return_raw"].cumsum() - sub["return_raw"].cumsum().cummax()).min()
    print(f"  {seed:>10}: ret_cum={cumret:+.4f}  maxDD={maxdd:.4f}  calmar={abs(cumret/maxdd):.2f}" if maxdd != 0 else f"  {seed:>10}: ret_cum={cumret:+.4f}  maxDD=0")

# ── VEREDICTO ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VEREDICTO D1")
print("=" * 65)
if kw_p < 0.05:
    print(f"  CONFIRMADA: Hay seeds estadisticamente diferentes (KW p={kw_p:.4f})")
    if super_seeds:
        print(f"  Super-seeds identificadas: {[s[0] for s in super_seeds]}")
        print(f"  ACCION: Ponderar mas estas seeds en el ensemble")
    else:
        print(f"  Diferencia estadistica pero sin super-seed clara")
        print(f"  ACCION: Mantener ensemble igualmente ponderado")
elif kw_p < 0.15:
    print(f"  TENDENCIA: Diferencia marginal entre seeds (KW p={kw_p:.4f})")
else:
    print(f"  DESCARTADA: Seeds son estadisticamente equivalentes (KW p={kw_p:.4f})")
    print(f"  ACCION: Mantener ensemble igualmente ponderado — todas las seeds contribuyen igual")
if all_rhos:
    print(f"\n  Independencia entre seeds: rho_media={np.mean(all_rhos):.4f}")
    print(f"  {'Diversificacion ALTA' if np.mean(all_rhos) < 0.15 else 'Diversificacion MEDIA' if np.mean(all_rhos) < 0.35 else 'Diversificacion BAJA'}")
