"""
tools/diagnostics/audit_wr_distribution.py
Investiga si la WR del 52.9% global es estadisticamente significativa
y si seed42 (68%) es un outlier o representa el edge real del sistema.
"""
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats

PREDS = Path("g:/Mi unidad/ia/luna_v2/data/predictions")

all_seeds = []
for p in sorted(PREDS.glob("oos_trades_seed*.parquet")):
    try:
        df = pd.read_parquet(p)
        seed = p.stem.replace("oos_trades_seed", "")
        if "is_win" not in df.columns and "return_pct" not in df.columns:
            continue
        wins = df["is_win"].sum() if "is_win" in df.columns else (df["return_pct"] > 0).sum()
        n = len(df)
        wr = wins / n if n > 0 else 0
        ev = df["return_pct"].mean() if "return_pct" in df.columns else 0
        ev_raw = df["return_raw"].mean() if "return_raw" in df.columns else 0
        # Regimenes
        regimes = df["hmm_regime"].value_counts().to_dict() if "hmm_regime" in df.columns else {}
        windows = df["wfb_window"].value_counts().to_dict() if "wfb_window" in df.columns else {}
        all_seeds.append({
            "seed": seed, "n": n, "wins": wins, "wr": wr,
            "ev_pct": ev, "ev_raw": ev_raw,
            "regimes": regimes, "windows": windows
        })
    except Exception as e:
        print(f"Error {p.name}: {e}")

df_seeds = pd.DataFrame(all_seeds).sort_values("wr", ascending=False)

print("=== DISTRIBUCION WR POR SEED ===")
print(f"{'Seed':<12} {'N':>5} {'WR':>7} {'EV_pct':>10} {'EV_raw':>10} Regimenes")
print("-" * 80)
for _, row in df_seeds.iterrows():
    reg_str = str(row["regimes"])[:35]
    print(f"  {row['seed']:<10} {row['n']:>5} {row['wr']:>7.1%} {row['ev_pct']:>10.5f} {row['ev_raw']:>10.5f}  {reg_str}")

print()
print("=== ESTADISTICAS GLOBALES ===")
total_wins = df_seeds["wins"].sum()
total_n = df_seeds["n"].sum()
global_wr = total_wins / total_n
global_ev = df_seeds["ev_pct"].mean()
print(f"  Total trades    : {total_n}")
print(f"  Total wins      : {total_wins}")
print(f"  WR global       : {global_wr:.3%}")
print(f"  EV medio (pct)  : {global_ev:.6f}")
print(f"  WR median/seed  : {df_seeds['wr'].median():.3%}")
print(f"  WR std/seed     : {df_seeds['wr'].std():.3%}")
print(f"  WR min/seed     : {df_seeds['wr'].min():.3%}")
print(f"  WR max/seed     : {df_seeds['wr'].max():.3%}")

print()
print("=== TEST ESTADISTICO: ES 52.9% SIGNIFICATIVAMENTE > 50%? ===")
# Test binomial global
binom_global = stats.binomtest(int(total_wins), int(total_n), 0.50, alternative="greater")
print(f"  H0: WR <= 50%  |  H1: WR > 50%")
print(f"  Global: WR={global_wr:.3%} n={total_n} p-value={binom_global.pvalue:.4f}")
print(f"  => {'RECHAZAR H0 (edge real)' if binom_global.pvalue < 0.05 else 'NO RECHAZAR H0 (no hay edge demostrable)'}")

print()
print("=== TEST ESTADISTICO SEED42: ES 68% UN OUTLIER? ===")
seed42 = df_seeds[df_seeds["seed"] == "42"].iloc[0]
n42, w42 = int(seed42["n"]), int(seed42["wins"])
binom_42 = stats.binomtest(w42, n42, 0.50, alternative="greater")
binom_42_vs_global = stats.binomtest(w42, n42, global_wr, alternative="greater")
print(f"  seed42: n={n42} WR={seed42['wr']:.1%}")
print(f"  vs H0=50%:          p={binom_42.pvalue:.4f} => {'SIGNIFICATIVO' if binom_42.pvalue < 0.05 else 'NO SIGNIFICATIVO'}")
print(f"  vs H0=global({global_wr:.1%}): p={binom_42_vs_global.pvalue:.4f} => {'OUTLIER SIGNIFICATIVO' if binom_42_vs_global.pvalue < 0.05 else 'no es outlier estadistico'}")

# Z-score de seed42 dentro de la distribucion de seeds
z_score = (seed42["wr"] - df_seeds["wr"].mean()) / df_seeds["wr"].std()
print(f"  Z-score seed42 vs dist. seeds: {z_score:.2f} sigma")
print(f"  => {'outlier (>2sigma)' if abs(z_score) > 2 else 'dentro de rango normal (<2sigma)'}")

print()
print("=== ANALISIS POR REGIMEN ===")
# Agregar por regimen
regime_agg = {}
for _, row in df_seeds.iterrows():
    for reg, cnt in row["regimes"].items():
        if reg not in regime_agg:
            regime_agg[reg] = {"n": 0, "seed_count": 0}
        regime_agg[reg]["n"] += cnt
        regime_agg[reg]["seed_count"] += 1
for reg, d in sorted(regime_agg.items(), key=lambda x: -x[1]["n"]):
    print(f"  {reg}: {d['n']} trades en {d['seed_count']} seeds")

print()
print("=== ANALISIS POR VENTANA ===")
window_agg = {}
for _, row in df_seeds.iterrows():
    for w, cnt in row["windows"].items():
        window_agg[w] = window_agg.get(w, 0) + cnt
for w, cnt in sorted(window_agg.items()):
    print(f"  {w}: {cnt} trades totales")
