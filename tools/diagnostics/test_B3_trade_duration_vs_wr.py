"""
test_B3_trade_duration_vs_wr.py
================================
H-B3: ¿La duración del trade predice si gana o pierde?
Hipótesis A: Trades rápidos (<24H) = alta convicción del modelo → WR mayor
Hipótesis B: Trades largos (>72H) = tiempo barrier hit = señal débil → WR menor

Sin join externo — solo entry_time / exit_time del parquet.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

print("=" * 65)
print("TEST B3 — Duración del Trade vs Win Rate")
print("=" * 65)

# ── Cargar datos ──────────────────────────────────────────────────
WFB = Path("data/reports/wfb")
dfs = []
for f in sorted(WFB.glob("oos_trades_W*_seed*.parquet")):
    df = pd.read_parquet(f)
    df["_w"] = f.stem.split("_")[2]
    df["_seed"] = f.stem.split("_")[4] if len(f.stem.split("_")) > 4 else "?"
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)

# Calcular duración
combined["entry_dt"] = pd.to_datetime(combined["entry_time"], utc=True, errors="coerce")
combined["exit_dt"]  = pd.to_datetime(combined["exit_time"],  utc=True, errors="coerce")
combined["duration_h"] = (combined["exit_dt"] - combined["entry_dt"]).dt.total_seconds() / 3600

valid = combined.dropna(subset=["duration_h", "is_win"])
print(f"\nTrades válidos con duración: {len(valid)} / {len(combined)}")
print(f"Duración media: {valid['duration_h'].mean():.1f}H  "
      f"mediana: {valid['duration_h'].median():.1f}H  "
      f"max: {valid['duration_h'].max():.0f}H")

# ── SEC 1: Distribución de duraciones ─────────────────────────────
print("\n" + "─" * 65)
print("SEC 1: Distribución de duraciones (wins vs losses)")
print("─" * 65)
wins   = valid[valid["is_win"] == 1]["duration_h"]
losses = valid[valid["is_win"] == 0]["duration_h"]
print(f"  WINS   (N={len(wins)}):   media={wins.mean():.1f}H  med={wins.median():.1f}H  std={wins.std():.1f}H")
print(f"  LOSSES (N={len(losses)}): media={losses.mean():.1f}H  med={losses.median():.1f}H  std={losses.std():.1f}H")

ks_stat, ks_p = stats.ks_2samp(wins.values, losses.values)
mw_stat, mw_p = stats.mannwhitneyu(wins.values, losses.values, alternative="two-sided")
print(f"\n  KS test (distribuciones distintas): stat={ks_stat:.4f}  p={ks_p:.4f}")
print(f"  Mann-Whitney U:                     stat={mw_stat:.0f}  p={mw_p:.4f}")
if mw_p < 0.05:
    direction = "WINS son MAS CORTOS" if wins.median() < losses.median() else "WINS son MAS LARGOS"
    print(f"  → SIGNIFICATIVO: {direction}")
else:
    print(f"  → No significativo (p={mw_p:.3f})")

# ── SEC 2: WR por cuartil de duración ─────────────────────────────
print("\n" + "─" * 65)
print("SEC 2: Win Rate por cuartil de duración")
print("─" * 65)
valid["dur_q"] = pd.qcut(valid["duration_h"], q=4, labels=["Q1_corto","Q2","Q3","Q4_largo"], duplicates="drop")
for q in ["Q1_corto","Q2","Q3","Q4_largo"]:
    sub = valid[valid["dur_q"] == q]
    lo  = sub["duration_h"].min()
    hi  = sub["duration_h"].max()
    wr  = sub["is_win"].mean()
    print(f"  {q}: N={len(sub):3d} | rango=[{lo:.0f}H,{hi:.0f}H] | WR={wr:.4f} ({wr*100-51.6:+.1f}pp vs baseline)")

# ── SEC 3: Bandas específicas ──────────────────────────────────────
print("\n" + "─" * 65)
print("SEC 3: Bandas de duración — WR por hora exacta")
print("─" * 65)
bands = [(0, 12), (12, 24), (24, 48), (48, 72), (72, 96), (96, 200)]
for lo, hi in bands:
    sub = valid[(valid["duration_h"] >= lo) & (valid["duration_h"] < hi)]
    if len(sub) < 5:
        continue
    wr = sub["is_win"].mean()
    bar = "█" * int(wr * 20)
    print(f"  [{lo:3d}H-{hi:3d}H): N={len(sub):3d} | WR={wr:.3f} {bar}")

# ── SEC 4: Correlación Spearman ────────────────────────────────────
print("\n" + "─" * 65)
print("SEC 4: Correlación Spearman (duration vs is_win)")
print("─" * 65)
rho, p_rho = stats.spearmanr(valid["duration_h"], valid["is_win"])
print(f"  Spearman rho = {rho:.4f}  p = {p_rho:.4f}")
if p_rho < 0.05:
    direction = "trades MAS CORTOS ganan mas" if rho < 0 else "trades MAS LARGOS ganan mas"
    print(f"  → SIGNIFICATIVO: {direction}")
else:
    print(f"  → No significativo")

# ── SEC 5: Duración por ventana (stability check) ─────────────────
print("\n" + "─" * 65)
print("SEC 5: Duración media por ventana (estabilidad del patrón)")
print("─" * 65)
for w in ["W1","W2","W3","W4","W5"]:
    sub = valid[valid["_w"] == w]
    if len(sub) < 10:
        continue
    sub_w = sub[sub["is_win"] == 1]["duration_h"]
    sub_l = sub[sub["is_win"] == 0]["duration_h"]
    delta = sub_w.mean() - sub_l.mean() if len(sub_l) > 0 else float("nan")
    print(f"  {w}: N={len(sub):3d} | dur_wins={sub_w.mean():.1f}H dur_losses={sub_l.mean():.1f}H | delta={delta:+.1f}H")

# ── SEC 6: Duración por régimen ────────────────────────────────────
print("\n" + "─" * 65)
print("SEC 6: Duración media por régimen HMM")
print("─" * 65)
if "hmm_regime" in valid.columns:
    for reg in valid["hmm_regime"].dropna().unique():
        sub = valid[valid["hmm_regime"] == reg]
        wr  = sub["is_win"].mean()
        dur = sub["duration_h"].mean()
        print(f"  {str(reg)[:30]:30s}: N={len(sub):3d} | WR={wr:.3f} | dur_media={dur:.1f}H")

# ── VEREDICTO FINAL ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VEREDICTO B3")
print("=" * 65)

# Calcular si hay efecto en alguna banda
best_wr, best_band = 0, ""
for lo, hi in [(0,24),(24,48),(48,96)]:
    sub = valid[(valid["duration_h"] >= lo) & (valid["duration_h"] < hi)]
    if len(sub) >= 30:
        wr = sub["is_win"].mean()
        if wr > best_wr:
            best_wr = wr
            best_band = f"{lo}-{hi}H"

if mw_p < 0.05:
    print(f"  CONFIRMADA: duración predice WR (MW p={mw_p:.4f})")
    print(f"  Mejor banda: {best_band} con WR={best_wr:.4f}")
    print(f"  ACCION: Revisar vertical_barrier_hours en settings.yaml")
    print(f"  RIESGO OVERFITTING: BAJO (efecto con respaldo teórico de TBM)")
elif mw_p < 0.15:
    print(f"  TENDENCIA: duración marginalmente predictiva (MW p={mw_p:.4f})")
    print(f"  ACCION: Monitorear en próxima run. No implementar gate todavía.")
else:
    print(f"  DESCARTADA: duración no predice WR (MW p={mw_p:.4f})")
    print(f"  El vertical barrier actual es apropiado.")
    print(f"  No modificar vertical_barrier_hours.")
