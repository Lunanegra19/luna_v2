"""
test_B1_hour_of_entry.py
=========================
H-B1: ¿A qué hora UTC del día son más rentables los trades?
BTC/USDT es 24/7 pero tiene microestructura horaria conocida:
  - Asia open: ~00-02 UTC
  - Europa open: ~07-09 UTC
  - US open: ~13-15 UTC
  - US close/Asia overlap: ~21-23 UTC

Riesgo overfitting: MODERADO — implementar solo si hay cluster de ≥3 horas
consecutivas con WR>55% y N≥30 por banda.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

print("=" * 65)
print("TEST B1 — Hora de Entrada vs Win Rate")
print("=" * 65)

WFB = Path("data/reports/wfb")
dfs = []
for f in sorted(WFB.glob("oos_trades_W*_seed*.parquet")):
    df = pd.read_parquet(f)
    df["_w"] = f.stem.split("_")[2]
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined["entry_dt"] = pd.to_datetime(combined["entry_time"], utc=True, errors="coerce")
combined["hour_utc"] = combined["entry_dt"].dt.hour
combined["dow"]      = combined["entry_dt"].dt.dayofweek  # 0=Lunes
combined["month"]    = combined["entry_dt"].dt.month

valid = combined.dropna(subset=["hour_utc","is_win"]).copy()
baseline_wr = valid["is_win"].mean()
print(f"\nN trades: {len(valid)} | WR baseline: {baseline_wr:.4f}")

# ── SEC 1: WR por hora UTC ─────────────────────────────────────────
print("\n─" * 65)
print("SEC 1: WR por hora UTC (0-23)")
print("─" * 65)
print(f"  {'Hora UTC':>8} {'N':>5} {'WR':>7} {'vs base':>8}  Barra")
hour_stats = []
for h in range(24):
    sub = valid[valid["hour_utc"] == h]
    if len(sub) < 3:
        continue
    wr = sub["is_win"].mean()
    hour_stats.append((h, len(sub), wr))
    bar_len = int(wr * 30)
    bar = "█" * bar_len
    flag = " ⬆" if wr > 0.58 else " ⬇" if wr < 0.44 else ""
    print(f"  {h:>8}H   {len(sub):>5} {wr:>7.3f} {wr-baseline_wr:>+8.3f}  {bar}{flag}")

# ANOVA: ¿hay diferencia significativa entre horas?
hour_groups = [valid[valid["hour_utc"] == h]["is_win"].values
               for h in range(24) if len(valid[valid["hour_utc"] == h]) >= 5]
if len(hour_groups) > 1:
    kw_stat, kw_p = stats.kruskal(*hour_groups)
    print(f"\n  Kruskal-Wallis (diferencia entre horas): H={kw_stat:.3f}  p={kw_p:.4f}")
    if kw_p < 0.05:
        print(f"  → SIGNIFICATIVO: hay horas con WR estadisticamente diferente")
    else:
        print(f"  → No significativo: la hora no predice WR")

# ── SEC 2: Bandas de mercado ──────────────────────────────────────
print("\n─" * 65)
print("SEC 2: WR por sesion de mercado")
print("─" * 65)
sessions = {
    "Asia Open    (00-04 UTC)": (0, 3),
    "Asia Close   (04-08 UTC)": (4, 7),
    "Europa Open  (07-10 UTC)": (7, 9),
    "Europa Mid   (10-13 UTC)": (10, 12),
    "US Open      (13-17 UTC)": (13, 16),
    "US Mid       (17-21 UTC)": (17, 20),
    "US Late/Over (21-23 UTC)": (21, 23),
}
session_results = []
for name, (lo, hi) in sessions.items():
    sub = valid[(valid["hour_utc"] >= lo) & (valid["hour_utc"] <= hi)]
    if len(sub) < 5:
        continue
    wr = sub["is_win"].mean()
    session_results.append((name, len(sub), wr))
    bar = "█" * int(wr * 25)
    flag = " ⬆⬆" if wr > 0.60 else " ⬆" if wr > 0.55 else " ⬇" if wr < 0.45 else ""
    print(f"  {name}: N={len(sub):3d} WR={wr:.4f} ({wr-baseline_wr:+.4f}) {bar}{flag}")

# ── SEC 3: Top horas significativas ──────────────────────────────
print("\n─" * 65)
print("SEC 3: Identificacion de horas outlier (test binomial)")
print("─" * 65)
print(f"  (Buscando horas con WR significativamente > {baseline_wr:.3f})")
for h, n, wr in sorted(hour_stats, key=lambda x: -x[2]):
    if n < 10:
        continue
    # Binomial: H0: p=baseline_wr, H1: p > baseline_wr
    from scipy.stats import binomtest
    result = binomtest(int(wr * n), n, baseline_wr, alternative="greater")
    p_bin = result.pvalue
    star = "***" if p_bin < 0.001 else "**" if p_bin < 0.01 else "*" if p_bin < 0.05 else ""
    if p_bin < 0.15:  # mostrar solo los marginalmente significativos
        print(f"  Hora {h:02d}H: N={n:3d} WR={wr:.4f} p_binomial={p_bin:.4f} {star}")

# ── SEC 4: Clusters temporales ────────────────────────────────────
print("\n─" * 65)
print("SEC 4: Clusters de horas consecutivas con WR alto")
print("─" * 65)
# Definir "buenas horas" como WR > baseline + 5pp
threshold_good = baseline_wr + 0.05
good_hours = [h for h, n, wr in hour_stats if wr > threshold_good and n >= 10]
print(f"  Horas con WR > {threshold_good:.3f} y N>=10: {sorted(good_hours)}")

# Buscar clusters (horas consecutivas)
if good_hours:
    clusters = []
    current_cluster = [good_hours[0]]
    for i in range(1, len(good_hours)):
        if good_hours[i] - good_hours[i-1] <= 2:  # ≤2H de gap
            current_cluster.append(good_hours[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [good_hours[i]]
    clusters.append(current_cluster)
    for cluster in clusters:
        if len(cluster) >= 2:
            lo_h, hi_h = min(cluster), max(cluster)
            sub = valid[(valid["hour_utc"] >= lo_h) & (valid["hour_utc"] <= hi_h)]
            wr_cluster = sub["is_win"].mean()
            print(f"  CLUSTER {lo_h}H-{hi_h}H: N={len(sub)} WR={wr_cluster:.4f} → {'ACCIONABLE' if len(sub)>=30 else 'N insuficiente'}")
        else:
            print(f"  Hora aislada {cluster[0]}H — no forma cluster")

# ── SEC 5: Dia de semana vs WR ────────────────────────────────────
print("\n─" * 65)
print("SEC 5: Dia de semana vs Win Rate")
print("─" * 65)
dias = ["Lun","Mar","Mie","Jue","Vie","Sab","Dom"]
for d in range(7):
    sub = valid[valid["dow"] == d]
    if len(sub) < 5:
        continue
    wr = sub["is_win"].mean()
    bar = "█" * int(wr * 25)
    print(f"  {dias[d]}: N={len(sub):3d} WR={wr:.4f} ({wr-baseline_wr:+.4f}) {bar}")

dow_groups = [valid[valid["dow"] == d]["is_win"].values
              for d in range(7) if len(valid[valid["dow"] == d]) >= 10]
if len(dow_groups) > 1:
    kw_d, kw_dp = stats.kruskal(*dow_groups)
    print(f"\n  Kruskal-Wallis (diferencia entre dias): H={kw_d:.3f}  p={kw_dp:.4f}")

# ── SEC 6: Interaccion hora x regimen ─────────────────────────────
print("\n─" * 65)
print("SEC 6: Mejor hora por regimen HMM")
print("─" * 65)
if "hmm_regime" in valid.columns:
    for reg in valid["hmm_regime"].dropna().unique():
        sub_reg = valid[valid["hmm_regime"] == reg]
        if len(sub_reg) < 30:
            continue
        # Mejor hora en este regimen
        best_h, best_wr_reg, best_n = -1, 0, 0
        for h in range(24):
            sub_h = sub_reg[sub_reg["hour_utc"] == h]
            if len(sub_h) >= 10 and sub_h["is_win"].mean() > best_wr_reg:
                best_h = h
                best_wr_reg = sub_h["is_win"].mean()
                best_n = len(sub_h)
        overall_wr = sub_reg["is_win"].mean()
        if best_h >= 0:
            print(f"  {str(reg)[:28]:30s}: WR_global={overall_wr:.3f} | mejor_hora={best_h:02d}H WR={best_wr_reg:.3f} (N={best_n})")

# ── VEREDICTO ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VEREDICTO B1")
print("=" * 65)
if kw_p < 0.05:
    print(f"  CONFIRMADA: La hora predice WR (Kruskal-Wallis p={kw_p:.4f})")
    actionable = [(h, n, wr) for h, n, wr in hour_stats
                  if wr > threshold_good and n >= 30]
    if actionable:
        print(f"  Horas accionables (N>=30, WR>{threshold_good:.3f}):")
        for h, n, wr in actionable:
            print(f"    {h:02d}H: N={n} WR={wr:.4f}")
        print(f"  ACCION: Añadir gate horario — solo operar en horas {[h for h,n,wr in actionable]}")
    else:
        print(f"  Diferencia significativa pero sin horas con N>=30 suficiente para gate")
        print(f"  ACCION: Acumular N en proxima run antes de implementar gate")
elif kw_p < 0.15:
    print(f"  TENDENCIA: Hora marginalmente predictiva (p={kw_p:.4f})")
    print(f"  ACCION: Monitorear en proxima run")
else:
    print(f"  DESCARTADA: La hora de entrada no predice WR (p={kw_p:.4f})")
    print(f"  El mercado crypto 24/7 es eficiente en terminos horarios")
