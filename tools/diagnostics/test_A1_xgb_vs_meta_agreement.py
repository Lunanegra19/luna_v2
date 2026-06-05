"""
test_A1_xgb_vs_meta_agreement.py
==================================
H-A1: Cuando XGB y Meta divergen, ¿cuál tiene razón?
  delta = xgb_prob_cal - meta_v2_prob
  delta > 0: XGB más optimista que Meta
  delta < 0: Meta más optimista que XGB

Hipótesis: El desacuerdo entre modelos es señal de incertidumbre.
Los trades donde ambos coinciden fuertemente deberían tener WR más alto.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

print("=" * 65)
print("TEST A1 — Acuerdo XGB vs MetaLabeler")
print("=" * 65)

WFB = Path("data/reports/wfb")
dfs = []
for f in sorted(WFB.glob("oos_trades_W*_seed*.parquet")):
    df = pd.read_parquet(f)
    df["_w"] = f.stem.split("_")[2]
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)

valid = combined.dropna(subset=["xgb_prob_cal","meta_v2_prob","is_win"])
valid = valid.copy()
valid["delta"] = valid["xgb_prob_cal"] - valid["meta_v2_prob"]
valid["agreement_abs"] = valid["delta"].abs()
baseline_wr = valid["is_win"].mean()
print(f"\nN trades validos: {len(valid)} | WR baseline: {baseline_wr:.4f}")
print(f"Delta media: {valid['delta'].mean():.4f}  std: {valid['delta'].std():.4f}")
print(f"XGB > Meta en {(valid['delta'] > 0).sum()} trades ({(valid['delta']>0).mean()*100:.1f}%)")
print(f"Meta > XGB en {(valid['delta'] < 0).sum()} trades ({(valid['delta']<0).mean()*100:.1f}%)")

# ── SEC 1: WR por cuartil de delta ────────────────────────────────
print("\n─" * 65)
print("SEC 1: WR por cuartil de delta (XGB - Meta)")
print("─" * 65)
valid["delta_q"] = pd.qcut(valid["delta"], q=4,
                            labels=["Q1_meta_gana","Q2","Q3","Q4_xgb_gana"],
                            duplicates="drop")
for q in ["Q1_meta_gana","Q2","Q3","Q4_xgb_gana"]:
    sub = valid[valid["delta_q"] == q]
    lo  = sub["delta"].min()
    hi  = sub["delta"].max()
    wr  = sub["is_win"].mean()
    print(f"  {q}: N={len(sub):3d} | delta=[{lo:+.3f},{hi:+.3f}] | WR={wr:.4f} ({wr-baseline_wr:+.3f}pp)")

# ── SEC 2: WR por banda de delta ──────────────────────────────────
print("\n─" * 65)
print("SEC 2: WR por banda de delta — interpretacion directa")
print("─" * 65)
bands = [
    ("Meta mucho mas optimista", -1.0, -0.10),
    ("Meta moderadamente mas optimista", -0.10, -0.03),
    ("Acuerdo (zona neutra)", -0.03, +0.03),
    ("XGB moderadamente mas optimista", +0.03, +0.10),
    ("XGB mucho mas optimista", +0.10, +1.0),
]
for name, lo, hi in bands:
    sub = valid[(valid["delta"] >= lo) & (valid["delta"] < hi)]
    if len(sub) < 5:
        continue
    wr = sub["is_win"].mean()
    bar = "█" * int(wr * 25)
    print(f"  {name[:38]:38s}: N={len(sub):3d} WR={wr:.3f} {bar}")

# ── SEC 3: Acuerdo (concordancia de magnitud) ─────────────────────
print("\n─" * 65)
print("SEC 3: Acuerdo absoluto — |delta| como medida de desacuerdo")
print("─" * 65)
print("  Hipotesis: acuerdo fuerte (|delta| bajo) = mayor WR")
valid["agree_q"] = pd.qcut(valid["agreement_abs"], q=4,
                            labels=["Q1_acuerdo","Q2","Q3","Q4_desacuerdo"],
                            duplicates="drop")
for q in ["Q1_acuerdo","Q2","Q3","Q4_desacuerdo"]:
    sub = valid[valid["agree_q"] == q]
    lo  = sub["agreement_abs"].min()
    hi  = sub["agreement_abs"].max()
    wr  = sub["is_win"].mean()
    print(f"  {q}: N={len(sub):3d} |delta|=[{lo:.3f},{hi:.3f}] | WR={wr:.4f} ({wr-baseline_wr:+.3f}pp)")

# Spearman: |delta| vs is_win
rho, p_rho = stats.spearmanr(valid["agreement_abs"], valid["is_win"])
print(f"\n  Spearman |delta| vs is_win: rho={rho:.4f}  p={p_rho:.4f}")
if p_rho < 0.05:
    direction = "MAYOR DESACUERDO → WR MENOR" if rho < 0 else "MAYOR DESACUERDO → WR MAYOR"
    print(f"  → SIGNIFICATIVO: {direction}")

# ── SEC 4: Quien tiene razon cuando divergen ─────────────────────
print("\n─" * 65)
print("SEC 4: Cuando divergen fuertemente, ¿quien acierta?")
print("─" * 65)
CUTOFF = 0.05
xgb_wins_meta = valid[valid["delta"] > threshold]   # XGB más optimista
meta_wins_xgb = valid[valid["delta"] < -threshold]  # Meta más optimista
neutral       = valid[valid["delta"].abs() <= threshold]

print(f"  XGB mas optimista (N={len(xgb_wins_meta)}): WR={xgb_wins_meta['is_win'].mean():.4f}")
print(f"  Meta mas optimista (N={len(meta_wins_xgb)}): WR={meta_wins_xgb['is_win'].mean():.4f}")
print(f"  Zona neutra (N={len(neutral)}): WR={neutral['is_win'].mean():.4f}")

if len(xgb_wins_meta) > 10 and len(meta_wins_xgb) > 10:
    _, p_comp = stats.chi2_contingency([
        [int(xgb_wins_meta["is_win"].sum()), len(xgb_wins_meta)-int(xgb_wins_meta["is_win"].sum())],
        [int(meta_wins_xgb["is_win"].sum()), len(meta_wins_xgb)-int(meta_wins_xgb["is_win"].sum())]
    ])[:2]
    print(f"\n  Chi² XGB-optimista vs Meta-optimista: p={p_comp:.4f}")
    if p_comp < 0.05:
        winner = "XGB" if xgb_wins_meta["is_win"].mean() > meta_wins_xgb["is_win"].mean() else "Meta"
        print(f"  → {winner} tiene razon estadisticamente cuando divergen")

# ── SEC 5: Por regimen — ¿cambia quien gana? ─────────────────────
print("\n─" * 65)
print("SEC 5: Acuerdo por regimen HMM — ¿cambia el patrón?")
print("─" * 65)
if "hmm_regime" in valid.columns:
    for reg in valid["hmm_regime"].dropna().unique():
        sub = valid[valid["hmm_regime"] == reg]
        if len(sub) < 20:
            continue
        delta_reg = sub["delta"].mean()
        wr_high_xgb = sub[sub["delta"] > 0.03]["is_win"].mean() if (sub["delta"] > 0.03).sum() > 5 else float("nan")
        wr_high_meta = sub[sub["delta"] < -0.03]["is_win"].mean() if (sub["delta"] < -0.03).sum() > 5 else float("nan")
        print(f"  {str(reg)[:28]:30s}: N={len(sub):3d} delta_media={delta_reg:+.4f} | "
              f"WR_xgb_opt={wr_high_xgb:.3f} WR_meta_opt={wr_high_meta:.3f}")

# ── SEC 6: Correlacion xgb vs meta probs ─────────────────────────
print("\n─" * 65)
print("SEC 6: Correlacion XGB_cal vs Meta (¿son redundantes?)")
print("─" * 65)
r, p_r = stats.pearsonr(valid["xgb_prob_cal"], valid["meta_v2_prob"])
print(f"  Pearson(xgb_prob_cal, meta_v2_prob) = {r:.4f}  p = {p_r:.4f}")
if r > 0.7:
    print(f"  → ALTA correlacion ({r:.3f}): los modelos son REDUNDANTES")
elif r > 0.3:
    print(f"  → Correlacion moderada ({r:.3f}): los modelos aportan informacion complementaria")
else:
    print(f"  → Baja correlacion ({r:.3f}): los modelos son INDEPENDIENTES — el meta tiene valor real")

print(f"\n  Distribucion de meta_v2_prob: [{valid['meta_v2_prob'].min():.3f}, {valid['meta_v2_prob'].max():.3f}]")
print(f"  Distribucion de xgb_prob_cal: [{valid['xgb_prob_cal'].min():.3f}, {valid['xgb_prob_cal'].max():.3f}]")

# ── VEREDICTO ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VEREDICTO A1")
print("=" * 65)
rho_agree, p_agree = stats.spearmanr(valid["agreement_abs"], valid["is_win"])
best_q_wr = max([valid[valid["delta_q"]==q]["is_win"].mean() for q in ["Q1_meta_gana","Q4_xgb_gana"]])
if p_agree < 0.05 and rho_agree < 0:
    print(f"  CONFIRMADA: Acuerdo entre modelos predice WR (rho={rho_agree:.3f}, p={p_agree:.4f})")
    print(f"  → Zona de desacuerdo alto = WR inferior")
    print(f"  ACCION: Añadir |delta| = |xgb_prob_cal - meta_v2_prob| como feature al meta-modelo")
elif p_agree < 0.15:
    print(f"  TENDENCIA: Acuerdo marginalmente predictivo (p={p_agree:.4f})")
    print(f"  ACCION: Monitorear en proxima run.")
else:
    print(f"  DESCARTADA: El acuerdo entre modelos no predice WR (p={p_agree:.4f})")
    print(f"  Los modelos son complementarios, no redundantes.")
