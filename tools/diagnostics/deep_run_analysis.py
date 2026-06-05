"""
deep_run_analysis.py - Analisis profundo de la run WFB activa
Ingenieria inversa: que tienen en comun los trades ganadores/perdedores
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
SEP = "-" * 70
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

wfb_dir = Path("data/reports/wfb")
print("=" * 70)
print("  ANÁLISIS PROFUNDO RUN WFB — Ingeniería inversa de trades")
print("=" * 70)

# ─── 1. CARGAR TODOS LOS TRADES ───────────────────────────────────────
all_trades = []
trades_by_seed_window = {}

for f in sorted(wfb_dir.glob("oos_trades_W*_seed*.parquet")):
    parts = f.stem.split("_")
    window = parts[2]   # W1..W5
    seed = parts[3].replace("seed", "")
    df = pd.read_parquet(f)
    df["_window"] = window
    df["_seed"] = seed
    df["_window_n"] = int(window[1:])
    all_trades.append(df)
    trades_by_seed_window[(seed, window)] = df

if not all_trades:
    print("Sin trades disponibles.")
    exit()

combined = pd.concat(all_trades, ignore_index=True)
print(f"\nTotal trades cargados: {len(combined):,}")
print(f"Seeds únicas: {combined['_seed'].nunique()}")
print(f"Columnas disponibles: {list(combined.columns)}")

# ─── 2. MÉTRICAS GLOBALES ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("2. MÉTRICAS GLOBALES")
print("─" * 70)

wr_global = combined["is_win"].mean() * 100
n_total = len(combined)
n_wins = combined["is_win"].sum()
n_losses = n_total - n_wins

ret = combined["return_pct"]
avg_ret_win  = ret[combined["is_win"] == 1].mean() * 100
avg_ret_loss = ret[combined["is_win"] == 0].mean() * 100
rr = abs(avg_ret_win / avg_ret_loss) if avg_ret_loss != 0 else 0

print(f"  WR global:       {wr_global:.1f}%")
print(f"  N trades:        {n_total} (wins={n_wins}, losses={n_losses})")
print(f"  AvgRet ganador:  {avg_ret_win:.4f}%")
print(f"  AvgRet perdedor: {avg_ret_loss:.4f}%")
print(f"  Ratio Retorno:   {rr:.2f}x")
print(f"  Kelly estimado:  {(wr_global/100 - (1-wr_global/100)/rr)*100:.1f}%")

# ─── 3. POR VENTANA ───────────────────────────────────────────────────
print("\n" + "─" * 70)
print("3. MÉTRICAS POR VENTANA")
print("─" * 70)
print(f"  {'Ventana':<8} {'N':>5} {'WR%':>6} {'AvgRet%':>9} {'Seeds':>6} {'WR_std':>8}")
for w in ["W1", "W2", "W3", "W4", "W5"]:
    sub = combined[combined["_window"] == w]
    if len(sub) == 0:
        continue
    wr = sub["is_win"].mean() * 100
    ar = sub["return_pct"].mean() * 100
    n_seeds = sub["_seed"].nunique()
    # WR std entre seeds
    seed_wrs = sub.groupby("_seed")["is_win"].mean() * 100
    wr_std = seed_wrs.std()
    flag = " ← COLAPSO" if w == "W5" and wr < 40 else (" ← MEJOR" if wr > 58 else "")
    print(f"  {w:<8} {len(sub):>5} {wr:>6.1f} {ar:>9.4f} {n_seeds:>6} {wr_std:>8.1f}{flag}")

# ─── 4. ANÁLISIS TEMPORAL — ¿A QUÉ HORA GANA EL MODELO? ─────────────
print("\n" + "─" * 70)
print("4. ANÁLISIS TEMPORAL (hora del día UTC)")
print("─" * 70)
if "timestamp" in combined.columns:
    combined["ts"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined["hour"] = combined["ts"].dt.hour
    combined["dow"] = combined["ts"].dt.dayofweek  # 0=lunes

    hourly = combined.groupby("hour").agg(
        N=("is_win", "count"),
        WR=("is_win", lambda x: x.mean() * 100),
        AvgRet=("return_pct", lambda x: x.mean() * 100)
    ).reset_index()
    print(f"  {'Hora':>5} {'N':>5} {'WR%':>6} {'AvgRet%':>9}")
    for _, row in hourly.iterrows():
        bar = "█" * int(row["WR"] / 5)
        flag = " ← BUENA" if row["WR"] > 60 else (" ← MALA" if row["WR"] < 40 else "")
        print(f"  {int(row['hour']):>5}h {int(row['N']):>5} {row['WR']:>6.1f} {row['AvgRet']:>9.4f}  {bar}{flag}")

    print()
    dow_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    dow_stats = combined.groupby("dow").agg(
        N=("is_win", "count"),
        WR=("is_win", lambda x: x.mean() * 100)
    ).reset_index()
    print("  Día de semana:")
    for _, row in dow_stats.iterrows():
        print(f"    {dow_names[int(row['dow'])]}: N={int(row['N']):<4} WR={row['WR']:.1f}%")

# ─── 5. ANÁLISIS POR RÉGIMEN HMM ──────────────────────────────────────
print("\n" + "─" * 70)
print("5. ANÁLISIS POR RÉGIMEN HMM")
print("─" * 70)
regime_col = None
for c in ["regime", "hmm_regime", "HMM_Semantic", "hmm_semantic", "regime_at_entry"]:
    if c in combined.columns:
        regime_col = c
        break

if regime_col:
    print(f"  (usando columna: {regime_col})")
    reg_stats = combined.groupby(regime_col).agg(
        N=("is_win", "count"),
        WR=("is_win", lambda x: x.mean() * 100),
        AvgRet=("return_pct", lambda x: x.mean() * 100)
    ).reset_index().sort_values("WR", ascending=False)
    print(f"  {'Régimen':<25} {'N':>5} {'WR%':>6} {'AvgRet%':>9}")
    for _, row in reg_stats.iterrows():
        flag = " ← EVITAR" if row["WR"] < 45 else (" ← MEJOR" if row["WR"] > 60 else "")
        print(f"  {str(row[regime_col]):<25} {int(row['N']):>5} {row['WR']:>6.1f} {row['AvgRet']:>9.4f}{flag}")
else:
    print("  [No hay columna de régimen en los trades]")
    print(f"  Columnas disponibles: {[c for c in combined.columns if 'reg' in c.lower() or 'hmm' in c.lower()]}")

# ─── 6. DIRECCIÓN (LONG/SHORT) ────────────────────────────────────────
print("\n" + "─" * 70)
print("6. DIRECCIÓN DE TRADES")
print("─" * 70)
dir_col = None
for c in ["direction", "side", "trade_direction", "signal"]:
    if c in combined.columns:
        dir_col = c
        break

if dir_col:
    dir_stats = combined.groupby(dir_col).agg(
        N=("is_win", "count"),
        WR=("is_win", lambda x: x.mean() * 100),
        AvgRet=("return_pct", lambda x: x.mean() * 100)
    ).reset_index()
    for _, row in dir_stats.iterrows():
        print(f"  {str(row[dir_col]):<10}: N={int(row['N']):<5} WR={row['WR']:.1f}%  AvgRet={row['AvgRet']:.4f}%")
else:
    print("  [Sin columna de dirección]")

# ─── 7. INGENIERÍA INVERSA — ¿QUÉ FEATURES PREDICEN EL WIN? ──────────
print("\n" + "─" * 70)
print("7. INGENIERÍA INVERSA — Features numéricas correlacionadas con WIN")
print("─" * 70)
numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
skip = ["is_win", "return_pct", "_window_n", "_seed"]
feature_cols = [c for c in numeric_cols if c not in skip and not c.startswith("_")]

if feature_cols:
    corrs = {}
    for col in feature_cols:
        try:
            r = combined[["is_win", col]].dropna()
            if len(r) > 20 and r[col].std() > 1e-8:
                corr = r["is_win"].corr(r[col])
                if not np.isnan(corr):
                    corrs[col] = corr
        except Exception:
            pass

    corrs_sorted = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top 15 features con mayor correlación con is_win:")
    for col, c in corrs_sorted[:15]:
        direction = "↑ win si sube" if c > 0 else "↓ win si baja"
        print(f"  {col:<35} r={c:+.4f}  {direction}")

    # Separar wins vs losses para top features
    print()
    print("  Diferencia media (wins - losses) en top 5 features:")
    for col, c in corrs_sorted[:5]:
        wins_mean  = combined[combined["is_win"] == 1][col].mean()
        losses_mean = combined[combined["is_win"] == 0][col].mean()
        print(f"  {col:<35} wins={wins_mean:.4f}  losses={losses_mean:.4f}  Δ={wins_mean-losses_mean:+.4f}")
else:
    print("  Sin features numéricas adicionales en los trades")

# ─── 8. DURACIÓN DE TRADES ────────────────────────────────────────────
print("\n" + "─" * 70)
print("8. DURACIÓN DE TRADES")
print("─" * 70)
dur_col = None
for c in ["duration_h", "holding_hours", "duration", "bars_held"]:
    if c in combined.columns:
        dur_col = c
        break

if dur_col:
    for win_val, label in [(1, "GANADORES"), (0, "PERDEDORES")]:
        sub = combined[combined["is_win"] == win_val][dur_col].dropna()
        if len(sub) > 0:
            print(f"  {label}: media={sub.mean():.1f}h  median={sub.median():.1f}h  p10={sub.quantile(0.1):.1f}h  p90={sub.quantile(0.9):.1f}h")
else:
    print("  [Sin columna de duración]")
    dur_candidates = [c for c in combined.columns if "dur" in c.lower() or "hour" in c.lower() or "hold" in c.lower()]
    print(f"  Candidatas: {dur_candidates}")

# ─── 9. CONSISTENCIA INTER-SEED ───────────────────────────────────────
print("\n" + "─" * 70)
print("9. CONSISTENCIA INTER-SEED (¿Qué seeds son estables?)")
print("─" * 70)
print(f"  {'Seed':<8} {'W1':>6} {'W2':>6} {'W3':>6} {'W4':>6} {'W5':>6} {'Rango':>7} {'Estable?':>9}")
for s in sorted(combined["_seed"].unique()):
    row_vals = []
    for w in ["W1", "W2", "W3", "W4", "W5"]:
        sub = combined[(combined["_seed"] == s) & (combined["_window"] == w)]
        if len(sub) >= 3:
            row_vals.append((w, sub["is_win"].mean() * 100))
    if len(row_vals) >= 2:
        wrs = [v[1] for v in row_vals]
        rng = max(wrs) - min(wrs)
        stable = "✅ ESTABLE" if rng < 20 else ("⚠️  INESTABLE" if rng > 30 else "OK")
        vals_str = ""
        for w in ["W1", "W2", "W3", "W4", "W5"]:
            found = [v[1] for v in row_vals if v[0] == w]
            vals_str += f"{found[0]:>6.0f}" if found else f"{'—':>6}"
        print(f"  {s:<8} {vals_str} {rng:>7.0f} {stable:>9}")

# ─── 10. TRADES COMPARTIDOS (consenso multi-seed) ────────────────────
print("\n" + "─" * 70)
print("10. CONSENSO MULTI-SEED — Trades en la misma hora por múltiples seeds")
print("─" * 70)
if "timestamp" in combined.columns:
    combined["ts_hour"] = combined["ts"].dt.floor("H")
    consensus = combined.groupby(["_window", "ts_hour"]).agg(
        n_seeds=("_seed", "nunique"),
        WR_consenso=("is_win", lambda x: x.mean() * 100),
        N=("is_win", "count")
    ).reset_index()

    for min_seeds in [4, 3, 2]:
        sub = consensus[consensus["n_seeds"] >= min_seeds]
        if len(sub) > 0:
            print(f"\n  Horas con >= {min_seeds} seeds coincidentes: {len(sub)} horas")
            print(f"    WR medio en esas horas: {sub['WR_consenso'].mean():.1f}%")
            print(f"    WR baseline (todas):    {wr_global:.1f}%")
            diff = sub['WR_consenso'].mean() - wr_global
            print(f"    Mejora por consenso:    {diff:+.1f}pp")
            # Por ventana
            for w in ["W1", "W2", "W3", "W4", "W5"]:
                sw = sub[sub["_window"] == w]
                if len(sw) > 0:
                    print(f"      {w}: {len(sw)} horas consenso | WR={sw['WR_consenso'].mean():.1f}%")

# ─── 11. DISTRIBUCIÓN DE RETORNOS ────────────────────────────────────
print("\n" + "─" * 70)
print("11. DISTRIBUCIÓN DE RETORNOS")
print("─" * 70)
ret_all = combined["return_pct"] * 100
print(f"  Media:   {ret_all.mean():.4f}%")
print(f"  Mediana: {ret_all.median():.4f}%")
print(f"  Std:     {ret_all.std():.4f}%")
print(f"  P5:      {ret_all.quantile(0.05):.4f}%")
print(f"  P95:     {ret_all.quantile(0.95):.4f}%")
print(f"  Min:     {ret_all.min():.4f}%")
print(f"  Max:     {ret_all.max():.4f}%")
print(f"  Skew:    {ret_all.skew():.3f}  (>0 = cola derecha positiva)")
print(f"  Kurtosis: {ret_all.kurtosis():.3f}  (>3 = colas pesadas)")

# Trades extremos
big_wins  = combined[combined["return_pct"] > 0.005].sort_values("return_pct", ascending=False)
big_loss  = combined[combined["return_pct"] < -0.005].sort_values("return_pct")
print(f"\n  Trades > +0.5%: {len(big_wins)}")
print(f"  Trades < -0.5%: {len(big_loss)}")

print("\n" + "=" * 70)
print("  FIN DEL ANÁLISIS")
print("=" * 70)
