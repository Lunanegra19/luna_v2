"""
audit_meta_distribution.py
===========================
Analisis de la distribucion de meta_v2_prob por ventana para calibrar el
threshold dinamico optimo sin look-ahead bias.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

WFB = Path("data/reports/wfb")
dfs = []
for f in sorted(WFB.glob("oos_trades_W*_seed*.parquet")):
    df = pd.read_parquet(f)
    w = f.stem.split("_")[2]
    df["_w"] = w
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f"Total: {len(combined)} trades | WR_baseline={combined['is_win'].mean():.4f}\n")

print("=" * 60)
print("DISTRIBUCION meta_v2_prob por ventana")
print("=" * 60)
for w in ["W1","W2","W3","W4","W5"]:
    sub = combined[combined["_w"]==w]["meta_v2_prob"].dropna()
    if len(sub) == 0:
        continue
    print(f"{w}: N={len(sub):4d} | p25={sub.quantile(0.25):.4f}  p50={sub.quantile(0.50):.4f}  "
          f"p75={sub.quantile(0.75):.4f}  p90={sub.quantile(0.90):.4f}  "
          f"range=[{sub.min():.4f},{sub.max():.4f}]")

print()
print("=" * 60)
print("WR por cuartil (global)")
print("=" * 60)
combined["q_meta"] = pd.qcut(combined["meta_v2_prob"], q=4,
                              labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
for q in ["Q1","Q2","Q3","Q4"]:
    sub = combined[combined["q_meta"]==q]
    lo = sub["meta_v2_prob"].min()
    hi_v = sub["meta_v2_prob"].max()
    print(f"  {q}: N={len(sub):4d} | WR={sub['is_win'].mean():.4f} | "
          f"prob=[{lo:.4f},{hi_v:.4f}]")

print()
print("=" * 60)
print("WR por cuartil x ventana (delta del threshold dinamico)")
print("=" * 60)
for w in ["W1","W2","W3","W4","W5"]:
    sub = combined[combined["_w"]==w]
    if len(sub) < 20:
        continue
    p50 = sub["meta_v2_prob"].quantile(0.50)
    p75 = sub["meta_v2_prob"].quantile(0.75)
    low = sub[sub["meta_v2_prob"] < p50]
    hi  = sub[sub["meta_v2_prob"] >= p75]
    if len(hi) == 0:
        continue
    delta = hi["is_win"].mean() - low["is_win"].mean()
    print(f"  {w}: p50={p50:.4f} p75={p75:.4f} | "
          f"WR_low={low['is_win'].mean():.3f}(N={len(low)}) | "
          f"WR_p75+={hi['is_win'].mean():.3f}(N={len(hi)}) | "
          f"delta={delta:+.3f}")

print()
print("=" * 60)
print("SIMULACION de threshold causal rolling (SIN look-ahead)")
print("El threshold de cada ventana se calcula SOLO con datos anteriores")
print("=" * 60)

# Simular: para cada ventana OOS, el threshold se calcula con
# la distribucion de probs de las ventanas ANTERIORES (causal estricto)
windows_order = ["W1","W2","W3","W4","W5"]
cumulative_probs = []
results = []

for i, w in enumerate(windows_order):
    sub = combined[combined["_w"]==w]
    if len(sub) == 0:
        continue
    sub_probs = sub["meta_v2_prob"].dropna()

    if len(cumulative_probs) >= 20:
        # Threshold calculado SOLO con probs de ventanas ANTERIORES
        hist_probs = pd.Series(cumulative_probs)
        dyn_thresh_p50 = hist_probs.quantile(0.50)
        dyn_thresh_p60 = hist_probs.quantile(0.60)
        dyn_thresh_p75 = hist_probs.quantile(0.75)

        # Aplicar cada threshold y ver el WR resultante
        for thr_name, thr_val in [("p50", dyn_thresh_p50),
                                    ("p60", dyn_thresh_p60),
                                    ("p75", dyn_thresh_p75),
                                    ("static_0.38", 0.38),
                                    ("static_0.63", 0.63)]:
            filtered = sub[sub["meta_v2_prob"] >= thr_val]
            n_filtered = len(filtered)
            wr_filtered = filtered["is_win"].mean() if n_filtered > 0 else float("nan")
            pct_kept = n_filtered / len(sub) * 100
            results.append({
                "window": w,
                "threshold_name": thr_name,
                "threshold_val": round(thr_val, 4),
                "N_kept": n_filtered,
                "pct_kept": round(pct_kept, 1),
                "WR": round(wr_filtered, 4) if not pd.isna(wr_filtered) else None
            })
    else:
        print(f"  {w}: Datos previos insuficientes ({len(cumulative_probs)} probs) — no hay threshold causal aun")

    cumulative_probs.extend(sub_probs.tolist())

if results:
    df_res = pd.DataFrame(results)
    pivot = df_res.pivot_table(index="threshold_name", columns="window",
                                values=["WR","N_kept"], aggfunc="first")
    print("\nWR por threshold causal rolling:")
    for thr_name in ["static_0.38","p50","p60","p75","static_0.63"]:
        row = df_res[df_res["threshold_name"]==thr_name]
        parts = []
        for _, r in row.iterrows():
            parts.append(f"{r['window']}:WR={r['WR']:.3f}({r['N_kept']})")
        avg_wr = row["WR"].mean()
        avg_n  = row["N_kept"].mean()
        print(f"  {thr_name:15s} | avg_WR={avg_wr:.4f} avg_N={avg_n:.0f} | {' | '.join(parts)}")

print()
print("=" * 60)
print("VEREDICTO: mejor threshold dinamico")
print("=" * 60)

# Test estadistico: p60 causal vs static 0.38
if results:
    df_res = pd.DataFrame(results)
    wrs_static = df_res[df_res["threshold_name"]=="static_0.38"]["WR"].dropna()
    wrs_p60    = df_res[df_res["threshold_name"]=="p60"]["WR"].dropna()
    wrs_p75    = df_res[df_res["threshold_name"]=="p75"]["WR"].dropna()

    for name, wrs in [("p60 causal", wrs_p60), ("p75 causal", wrs_p75)]:
        if len(wrs) > 1 and len(wrs_static) > 1:
            t, p = stats.ttest_rel(wrs.values, wrs_static.values)
            print(f"  {name} vs static_0.38: avg_WR={wrs.mean():.4f} vs {wrs_static.mean():.4f} | "
                  f"t={t:.3f} p={p:.4f} ({'mejora' if t > 0 else 'peor'})")
        else:
            print(f"  {name}: N insuficiente para t-test apareado ({len(wrs)} ventanas)")
