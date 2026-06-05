import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

RUNS = Path(r"g:\Mi unidad\ia\luna_v2\data\runs")

def load_raw_probs(window):
    dfs = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        seed_label = run_dir.name.split("seed")[-1] if "seed" in run_dir.name else None
        if seed_label is None:
            continue
        seed_subdir = run_dir / seed_label
        if not seed_subdir.exists():
            subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
            seed_subdir = subdirs[0] if subdirs else None
        if seed_subdir:
            pq = seed_subdir / window / "oos_raw_probs.parquet"
            if pq.exists():
                df = pd.read_parquet(pq)
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                df.index = pd.to_datetime(df.index, utc=True)
                dfs.append(df)
    if dfs:
        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep="first")]
        return combined.sort_index()
    return pd.DataFrame()

print("Cargando W3 y W4...")
df_w3 = load_raw_probs("W3")
df_w4 = load_raw_probs("W4")
print(f"W3: {len(df_w3)} barras | W4: {len(df_w4)} barras")
print(f"W3 periodo: {df_w3.index.min().date()} -> {df_w3.index.max().date()}")
print(f"W4 periodo: {df_w4.index.min().date()} -> {df_w4.index.max().date()}")
print()

# Regimen dominante
df_w4["regime"] = df_w4[["prob_bull", "prob_bear", "prob_range"]].idxmax(axis=1)
counts = df_w4["regime"].value_counts()
total = len(df_w4)
print("=== REGIMEN DOMINANTE W4 (Oct-Dic 2025) ===")
for reg, n in counts.items():
    pct = n / total * 100
    if "bull" in reg:
        status = "BLOQUEADO gate 0.20"
    elif "bear" in reg:
        status = "SIN MODELO -> 0 seniales"
    else:
        status = "VIABLE para trading"
    print(f"  {reg:<15}: {n:5d}/{total} ({pct:5.1f}%)  [{status}]")
print()

# Prob_range estadisticas
print("=== PROB_RANGE ESTADISTICAS ===")
for name, df in [("W3", df_w3), ("W4", df_w4)]:
    pr = df["prob_range"]
    q10 = pr.quantile(0.10)
    q90 = pr.quantile(0.90)
    pct_zero = 100 * (pr == 0).sum() / len(pr)
    pct_above = 100 * (pr > 0.5).sum() / len(pr)
    print(f"  {name}: media={pr.mean():.4f}  mediana={pr.median():.4f}  std={pr.std():.4f}")
    print(f"       p10={q10:.4f}  p90={q90:.4f}")
    print(f"       prob_range==0: {pct_zero:.1f}%  |  prob_range>0.5: {pct_above:.1f}%")
print()

# Tests estadisticos
ks, p_ks = stats.ks_2samp(df_w3["prob_range"], df_w4["prob_range"])
t, p_t = stats.ttest_ind(df_w3["prob_range"], df_w4["prob_range"])
print(f"  KS-test W3 vs W4 prob_range: KS={ks:.4f}, p={p_ks:.6f}")
print(f"  t-test  W3 vs W4 prob_range: t={t:.3f},  p={p_t:.6f}")
print()

# Evolucion mensual
df_w4["mes"] = df_w4.index.month
print("=== EVOLUCION MENSUAL PROB_RANGE EN W4 ===")
for mes, grp in df_w4.groupby("mes"):
    nombre = {10: "Oct-2025", 11: "Nov-2025", 12: "Dic-2025"}.get(mes, str(mes))
    pr = grp["prob_range"]
    pct_zero = 100 * (pr == 0).sum() / len(pr)
    print(f"  {nombre}: media={pr.mean():.4f}  max={pr.max():.4f}  pct_zero={pct_zero:.0f}%  N={len(pr)}")
print()

# RESULTADO
pct_range = 100 * (df_w4["regime"] == "prob_range").sum() / total
pr_w4_media = df_w4["prob_range"].mean()
pr_w3_media = df_w3["prob_range"].mean()
print("=" * 55)
print("RESULTADO H-W4-REGIME")
print("=" * 55)
print(f"  RANGE dominante en W4:  {pct_range:.1f}%")
print(f"  prob_range media W4:    {pr_w4_media:.4f}")
print(f"  prob_range media W3:    {pr_w3_media:.4f}")
print(f"  Caida W3->W4:           {pr_w3_media - pr_w4_media:+.4f}")
print(f"  KS p-value:             {p_ks:.6f}")
if pct_range < 5:
    print()
    print(">>> H-W4-REGIME: CONFIRMADA")
    print(f">>> W4 tiene solo {pct_range:.1f}% de barras con RANGE dominante")
    print(">>> El HMM predice BULL+BEAR en Oct-Dic 2025")
    print(">>> -> 0 trades esperados (gate 0.20 bloquea BULL, sin modelo BEAR)")
elif pct_range < 20:
    print(f">>> H-W4-REGIME: PARCIALMENTE CONFIRMADA ({pct_range:.1f}% RANGE)")
else:
    print(f">>> H-W4-REGIME: DESCARTADA ({pct_range:.1f}% RANGE > umbral 5%)")
print()
print("[FIX-DIAG-H-W4-01] Test completado sobre datos reales W4.")
