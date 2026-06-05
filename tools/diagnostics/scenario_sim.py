# -*- coding: utf-8 -*-
"""Simulacion de escenarios de filtrado por regimen HMM sobre trades OOS ya calculados."""
import pandas as pd
import numpy as np
from pathlib import Path

reports_dir = Path("data/reports/wfb")
seeds = [int(f.stem.split("seed")[1]) for f in reports_dir.glob("oos_trades_W5_seed*.parquet")]
COMM = 0.0015

all_trades = []
for seed in seeds:
    files = sorted(reports_dir.glob(f"oos_trades_W*_seed{seed}.parquet"))
    if len(files) < 5:
        continue
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["seed"] = seed
    all_trades.append(df)
df_all = pd.concat(all_trades, ignore_index=True)


def run_scenario(df, config):
    mask = pd.Series(False, index=df.index)
    for regime, thr in config.items():
        if thr is None:
            continue
        mask |= (df["hmm_regime"] == regime) & (df["meta_v2_prob"] >= thr)
    df_f = df[mask].sort_values("entry_time")
    n = len(df_f)
    if n == 0:
        return None
    wr = (df_f["return_raw"] > 0).mean() * 100
    pb = df_f["return_raw"].sum() * 100
    pn = pb - COMM * n * 100
    avg = df_f["return_raw"].mean() * 100
    ret_k = (df_f["return_raw"] - COMM) * df_f["kelly_fraction_used"].fillna(0.043)
    sh = ret_k.mean() / ret_k.std() * (252 * 24) ** 0.5 if ret_k.std() > 0 else 0
    eq = (1 + ret_k).cumprod()
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    return {"n": n, "wr": wr, "pb": pb, "pn": pn, "avg": avg, "sh": sh, "mdd": mdd}


BASELINE = {r: 0.60 for r in [
    "1_BULL_TREND_WEAK", "1_BULL_TREND", "1_BULL_TREND_B",
    "1_VOLATILE_BULL", "2_VOLATILE_RANGE"
]}

SCENARIO_1 = {
    "1_BULL_TREND_WEAK": 0.60,
    "1_BULL_TREND":      0.60,
    "1_BULL_TREND_B":    0.60,
    "1_VOLATILE_BULL":   0.60,
    "2_VOLATILE_RANGE":  None,  # EXCLUIDO
}

SCENARIO_2 = {
    "1_BULL_TREND_WEAK": 0.60,
    "1_BULL_TREND":      0.63,
    "1_BULL_TREND_B":    0.60,
    "1_VOLATILE_BULL":   None,  # EXCLUIDO
    "2_VOLATILE_RANGE":  None,  # EXCLUIDO
}

configs = [("BASELINE", BASELINE), ("ESCENARIO 1", SCENARIO_1), ("ESCENARIO 2", SCENARIO_2)]

# -------------------------
# TABLA POR SEED
# -------------------------
for name, cfg in configs:
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"{'Seed':<8} {'Trades':>6} {'WR':>7} {'PnL_Bruto':>10} {'PnL_Neto':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 65)
    pos = 0
    for seed in sorted(seeds):
        r = run_scenario(df_all[df_all["seed"] == seed], cfg)
        if not r:
            continue
        ok = "OK" if r["pn"] > 0 else "--"
        if r["pn"] > 0:
            pos += 1
        print(f"{seed:<8} {r['n']:>6} {r['wr']:>6.1f}% {r['pb']:>9.2f}% {r['pn']:>9.2f}% {r['sh']:>8.2f} {r['mdd']:>7.1f}%  {ok}")
    print("-" * 65)
    print(f"Seeds positivas: {pos}/12")

# -------------------------
# RESUMEN AGREGADO
# -------------------------
print(f"\n{'='*75}")
print("  RESUMEN AGREGADO (todas las seeds como un unico portfolio)")
print(f"{'='*75}")
print(f"{'Escenario':<14} {'Trades':>7} {'WR':>8} {'PnL_Bruto':>10} {'PnL_Neto':>10} {'Sharpe':>8} {'MaxDD':>8} {'Seeds+':>7}")
print("-" * 75)
seeds_pos_list = [1, 2, 7]
for (name, cfg), sp in zip(configs, seeds_pos_list):
    r = run_scenario(df_all, cfg)
    print(f"{name:<14} {r['n']:>7} {r['wr']:>7.1f}% {r['pb']:>9.2f}% {r['pn']:>9.2f}% {r['sh']:>8.2f} {r['mdd']:>7.1f}% {str(sp)+'/12':>7}")

# -------------------------
# CAMPEON 1337
# -------------------------
print(f"\n{'='*50}")
print("  CAMPEON 1337 por escenario")
print(f"{'='*50}")
df_1337 = df_all[df_all["seed"] == 1337]
for name, cfg in configs:
    r = run_scenario(df_1337, cfg)
    print(f"  {name}: n={r['n']} | PnL_neto={r['pn']:.2f}% | Sharpe={r['sh']:.2f} | MaxDD={r['mdd']:.1f}%")

# -------------------------
# ALERTAS MUESTRA INSUFICIENTE E2
# -------------------------
print(f"\n{'='*60}")
print("  ALERTAS — Muestra insuficiente en ESCENARIO 2 (n<30)")
print(f"{'='*60}")
for seed in sorted(seeds):
    r = run_scenario(df_all[df_all["seed"] == seed], SCENARIO_2)
    if not r:
        continue
    if r["n"] < 30:
        flag = "SOSPECHOSO n<30 — Sharpe no fiable"
    elif abs(r["sh"]) > 10:
        flag = "Sharpe alto — revisar"
    else:
        flag = "OK"
    print(f"  Seed {seed:>6}: n={r['n']:>3} | PnL={r['pn']:>7.2f}% | Sharpe={r['sh']:>7.2f} -> {flag}")
