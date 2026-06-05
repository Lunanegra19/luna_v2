import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import binomtest, mannwhitneyu
import datetime

BASE = Path(r"g:\Mi unidad\ia\luna_v2")
RUNS = BASE / "data" / "runs"
cutoff = datetime.datetime(2026, 6, 1, 22, 0, 0).timestamp()
overnight_runs = [d for d in sorted(RUNS.iterdir()) if d.is_dir() and d.stat().st_mtime >= cutoff]

all_trades = []
for run in overnight_runs:
    seed_label = run.name.split("seed")[-1] if "seed" in run.name else None
    if seed_label is None:
        continue
    seed_sub = run / seed_label
    if not seed_sub.exists():
        subs = [d for d in run.iterdir() if d.is_dir()]
        seed_sub = subs[0] if subs else None
    if seed_sub is None:
        continue
    for w in ["W1", "W2", "W3", "W4", "W5"]:
        pq = seed_sub / w / "oos_trades.parquet"
        if pq.exists():
            try:
                df = pd.read_parquet(pq)
                df["seed"] = seed_label
                df["window"] = w
                all_trades.append(df)
            except Exception:
                pass

df_all = pd.concat(all_trades, ignore_index=True)
df_bull  = df_all[df_all["hmm_regime"].str.contains("BULL", na=False)]
df_range = df_all[df_all["hmm_regime"].str.contains("RANGE", na=False)]

n = len(df_bull)
n_wins = int(df_bull["is_win"].sum())
wr = df_bull["is_win"].mean()
r = df_bull["return_pct"] * 100

print("=== H-BULL-WEAK: WR=42% en BULL_TREND (W1) — testeable con N=45? ===")
print(f"N trades BULL: {n}  Wins: {n_wins}  WR: {wr*100:.1f}%")
print(f"EV por trade:  {r.mean():+.4f}%   Std: {r.std():.4f}%")
print()

res_less = binomtest(n_wins, n, 0.5, alternative="less")
res_two  = binomtest(n_wins, n, 0.5, alternative="two-sided")
print(f"binom_test (alt=less):      p = {res_less.pvalue:.4f}")
print(f"binom_test (alt=two-sided): p = {res_two.pvalue:.4f}")
print()

if res_less.pvalue < 0.05:
    print(">>> H-BULL-WEAK: CONFIRMADA (p<0.05)")
    print(">>> WR<50% estadisticamente significativo. Sistema pierde en BULL.")
elif res_less.pvalue < 0.10:
    print(f">>> H-BULL-WEAK: SUGESTIVA (p={res_less.pvalue:.3f} < 0.10)")
    print(">>> Tendencia negativa, N insuficiente para p<0.05")
else:
    print(f">>> H-BULL-WEAK: NO CONCLUYENTE (p={res_less.pvalue:.3f})")
    print(f">>> WR=42% puede ser varianza natural con N={n}")

print()
print("Distribucion xgb_prob_cal en trades BULL (confianza del modelo):")
if "xgb_prob_cal" in df_bull.columns:
    for p_thr in [0.50, 0.55, 0.60, 0.65]:
        sub = df_bull[df_bull["xgb_prob_cal"] >= p_thr]
        if len(sub) > 0:
            wr_sub = sub["is_win"].mean()
            print(f"  prob_cal >= {p_thr}: N={len(sub):2d}  WR={wr_sub*100:.0f}%")
print()

print("Analisis OOD KL-distance BULL vs RANGE:")
if "ood_kl_distance" in df_bull.columns:
    bull_ood  = df_bull["ood_kl_distance"].dropna()
    range_ood = df_range["ood_kl_distance"].dropna()
    if len(bull_ood) > 0 and len(range_ood) > 0:
        print(f"  BULL  OOD media={bull_ood.mean():.4f}  std={bull_ood.std():.4f}  N={len(bull_ood)}")
        print(f"  RANGE OOD media={range_ood.mean():.4f}  std={range_ood.std():.4f}  N={len(range_ood)}")
        stat, p_ood = mannwhitneyu(bull_ood, range_ood, alternative="greater")
        print(f"  Mann-Whitney (BULL_OOD > RANGE_OOD): p={p_ood:.4f}")
        if p_ood < 0.05:
            print("  >>> BULL tiene OOD sistematicamente MAYOR -> shift de distribucion en W1")
        else:
            print("  >>> OOD no distingue BULL de RANGE estadisticamente")
    else:
        print("  Datos OOD insuficientes")
else:
    print("  ood_kl_distance no disponible en trades")

print()
print("=== DECISION SOBRE HIPOTESIS ADICIONALES PRE-RERUN ===")
print(f"H-BULL-WEAK: p={res_less.pvalue:.3f} (alt=less)")
if res_less.pvalue < 0.05:
    print("  -> CONFIRMADA: implica que el gate 0.20 deberia ser mas estricto en BULL")
    print("  -> O que el agente BULL necesita re-entrenamiento con mas datos IS")
else:
    print("  -> No concluyente: N=45 insuficiente para testear WR<50% con p<0.05")
    print("  -> RECOMENDACION: no hay hipotesis adicional que valga la pena testear")
    print("     con los datos actuales. Proceder al RERUN con los 3 fixes aplicados.")
