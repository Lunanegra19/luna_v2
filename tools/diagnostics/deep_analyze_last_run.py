"""
tools/diagnostics/deep_analyze_last_run.py
Analisis profundo de los resultados de la ultima run:
- Que gates bloquearon cada ventana
- Distribucion de returns por ventana y seed
- Analisis de regimenes HMM
- Por que W5 de seeds 42/100/777 no aparece
- Patron de trades: hora, win/loss, retorno
[DEEP-ANALYZE-RUN 2026-05-30]
"""
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import datetime

ROOT    = Path(r"g:\Mi unidad\ia\luna_v2")
wfb_dir = ROOT / "data" / "reports" / "wfb"

LAST_RUN_SEEDS = [1337, 2025, 12751, 28020, 30915, 34324, 43610, 77542]
# + las 3 incompletas para comparar
INCOMPLETAS    = [42, 100, 777]
ALL_INTEREST   = LAST_RUN_SEEDS + INCOMPLETAS

# ═══════════════════════════════════════════════════════════════════
# 1. ANALISIS DE GATES: que rechazo cada ventana
# ═══════════════════════════════════════════════════════════════════
print("=" * 80)
print("1. ANALISIS DE GATES POR SEED Y VENTANA")
print("=" * 80)

gate_summary = defaultdict(dict)

for seed in ALL_INTEREST:
    for win in ["W1", "W2", "W3", "W4", "W5"]:
        # Leer el gate mas alto disponible (G5 > G4 > G2)
        for gnum in ["G5", "G4", "G2"]:
            gfile = wfb_dir / f"gate_{gnum}_{win}_seed{seed}.json"
            if gfile.exists():
                with open(gfile) as f:
                    data = json.load(f)
                verdict = data.get("verdict", data.get("decision", "?"))
                reason  = data.get("reason", data.get("rejection_reason", ""))
                n_trades = data.get("n_trades", data.get("trades", "?"))
                gate_summary[seed][win] = {
                    "gate": gnum,
                    "verdict": verdict,
                    "reason": str(reason)[:60],
                    "trades": n_trades
                }
                break

for seed in ALL_INTEREST:
    label = "(COMPLETA)" if seed in LAST_RUN_SEEDS else "(INCOMPLETA)"
    print(f"\nSeed {seed} {label}:")
    for win in ["W1", "W2", "W3", "W4", "W5"]:
        if win in gate_summary[seed]:
            g = gate_summary[seed][win]
            print(f"  {win} [{g['gate']}]: {g['verdict']:12} | trades={g['trades']:4} | {g['reason'][:55]}")
        else:
            parquet = list(wfb_dir.glob(f"oos_trades_{win}_seed{seed}.parquet"))
            empty   = list(wfb_dir.glob(f"oos_trades_{win}_seed{seed}_EMPTY.flag"))
            if parquet:
                print(f"  {win}: [sin gate JSON] -> parquet con trades existe")
            elif empty:
                print(f"  {win}: [sin gate JSON] -> EMPTY flag")
            else:
                print(f"  {win}: [NO HAY ARCHIVOS] -> ventana no ejecutada o perdida")

# ═══════════════════════════════════════════════════════════════════
# 2. ANALISIS DE TRADES: distribucion de retornos y regimenes
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("2. ANALISIS DE TRADES POR SEED")
print("=" * 80)

all_trades_list = []
seed_stats = {}

for seed in ALL_INTEREST:
    dfs = []
    for f in sorted(wfb_dir.glob(f"oos_trades_W*_seed{seed}.parquet")):
        try:
            df = pd.read_parquet(f)
            win = re.search(r"(W\d)", f.stem).group(1)
            df["window"] = win
            df["seed"]   = seed
            dfs.append(df)
        except Exception as e:
            print(f"  ERROR leyendo {f.name}: {e}")

    if not dfs:
        print(f"Seed {seed}: 0 trades")
        continue

    df_seed = pd.concat(dfs).sort_index() if dfs else pd.DataFrame()
    all_trades_list.append(df_seed)

    n = len(df_seed)
    if n == 0:
        continue

    wr = df_seed["is_win"].mean() if "is_win" in df_seed.columns else float("nan")
    ret_mean = df_seed["return_pct"].mean() * 100 if "return_pct" in df_seed.columns else float("nan")
    ret_std  = df_seed["return_pct"].std()  * 100 if "return_pct" in df_seed.columns else float("nan")
    max_win  = df_seed["return_pct"].max()  * 100 if "return_pct" in df_seed.columns else float("nan")
    max_loss = df_seed["return_pct"].min()  * 100 if "return_pct" in df_seed.columns else float("nan")

    # Desglose por ventana
    by_win = df_seed.groupby("window").agg(
        n=("return_pct", "count"),
        wr=("is_win", "mean"),
        ret=("return_pct", "mean")
    ).reset_index()

    # Regimenes HMM si existe la columna
    regimes = {}
    if "hmm_regime" in df_seed.columns:
        regimes = df_seed["hmm_regime"].value_counts().head(3).to_dict()

    # Direcciones (long/short)
    directions = {}
    if "direction" in df_seed.columns:
        directions = df_seed["direction"].value_counts().to_dict()

    seed_stats[seed] = {
        "n": n, "wr": wr, "ret_mean": ret_mean,
        "ret_std": ret_std, "max_win": max_win, "max_loss": max_loss,
        "regimes": regimes, "directions": directions, "by_win": by_win
    }

    label = "(COMPLETA)" if seed in LAST_RUN_SEEDS else "(INCOMPLETA)"
    print(f"\nSeed {seed} {label}: {n} trades | WR={wr:.1%} | ret_medio={ret_mean:.4f}%")
    print(f"  Rango: [{max_loss:.3f}%, {max_win:.3f}%] | std={ret_std:.4f}%")
    print(f"  Direcciones: {directions}")
    print(f"  Regimenes HMM: {regimes}")
    print(f"  Por ventana:")
    for _, row in by_win.iterrows():
        print(f"    {row['window']}: {int(row['n']):3d} trades | WR={row['wr']:.1%} | ret={row['ret']*100:.4f}%")

# ═══════════════════════════════════════════════════════════════════
# 3. ANALISIS TEMPORAL: cuando se generan las senales
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("3. ANALISIS TEMPORAL DE SENALES (mes y hora del dia)")
print("=" * 80)

if all_trades_list:
    df_all = pd.concat(all_trades_list)
    if not df_all.index.tz:
        df_all.index = pd.to_datetime(df_all.index, utc=True)
    else:
        df_all.index = df_all.index.tz_convert("UTC")

    df_all["month"] = df_all.index.month
    df_all["hour"]  = df_all.index.hour

    print("\nDistribucion por mes:")
    by_month = df_all.groupby("month").agg(
        n=("return_pct","count"),
        wr=("is_win","mean"),
        ret=("return_pct","mean")
    )
    for m, row in by_month.iterrows():
        bar = "#" * int(row["n"])
        print(f"  Mes {int(m):2d}: {int(row['n']):4d} trades | WR={row['wr']:.1%} | ret={row['ret']*100:.4f}% {bar[:40]}")

    print("\nDistribucion por hora UTC:")
    by_hour = df_all.groupby("hour").agg(
        n=("return_pct","count"),
        wr=("is_win","mean")
    )
    for h, row in by_hour.iterrows():
        bar = "#" * int(row["n"])
        print(f"  {int(h):02d}h: {int(row['n']):4d} trades | WR={row['wr']:.1%} {bar[:30]}")

# ═══════════════════════════════════════════════════════════════════
# 4. HIPOTESIS: por que la run se interrumpio (seeds 42/100/777)
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("4. DIAGNOSTICO: SEEDS INCOMPLETAS (42, 100, 777 - sin W5)")
print("=" * 80)

# Ver si hay gate G4/G5 de W5 para estas seeds (si existen, W5 se ejecuto)
for seed in INCOMPLETAS:
    print(f"\nSeed {seed}:")
    # Buscar cualquier archivo con W5 y este seed
    w5_files = list(wfb_dir.glob(f"*W5*seed{seed}*"))
    if w5_files:
        for f in w5_files:
            mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  ENCONTRADO: {f.name} ({mtime.strftime('%H:%M')})")
    else:
        print(f"  SIN ARCHIVOS W5 -> ventana nunca iniciada o no hay gate/parquet")
    # Ver cuando termino el ultimo archivo de esta seed
    all_seed_files = list(wfb_dir.glob(f"*seed{seed}*"))
    if all_seed_files:
        ultimo = max(all_seed_files, key=lambda f: f.stat().st_mtime)
        mtime  = datetime.datetime.fromtimestamp(ultimo.stat().st_mtime)
        print(f"  Ultimo archivo: {ultimo.name} ({mtime.strftime('%Y-%m-%d %H:%M')})")
        # Cuanto tiempo despues del ultimo archivo de seed 42 aparece el de 43610
        # para estimar si fue una terminacion o un crash
