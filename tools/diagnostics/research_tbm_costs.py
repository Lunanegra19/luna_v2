"""
Investigacion de TBM, Costes y Regimenes sobre la baseline de 20 semillas de esta madrugada.
Analizamos el impacto de los regimenes HMM en la rentabilidad y resiliencia frente a costes.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

runs_dir = Path("data/runs")
# Buscar los directorios de la run de esta madrugada (las carpetas creadas entre 00:00 y 05:00 del 11 de junio)
run_folders = [d for d in runs_dir.iterdir() if d.is_dir() and "WFB_20260611_0" in d.name]

all_trades = []
for r in run_folders:
    # Buscar todos los oos_trades.parquet de las ventanas
    parquets = list(r.rglob("oos_trades.parquet"))
    for p in parquets:
        df = pd.read_parquet(p)
        all_trades.append(df)

if not all_trades:
    print("No se encontraron trades de la baseline.")
    exit()

df_all = pd.concat(all_trades, ignore_index=True)
print(f"Total trades baseline cargados: {len(df_all)}")

# Analisis por regimen HMM
print("\n--- RENDIMIENTO POR REGIMEN HMM ---")
grouped = df_all.groupby("HMM_Semantic")
res = []
for regime, group in grouped:
    n = len(group)
    wr = group["is_win"].mean() * 100
    ret_mean = group["return_pct"].mean() * 100
    ret_sum = group["return_pct"].sum() * 100
    std = group["return_pct"].std() * 100
    sharpe = (group["return_pct"].mean() / group["return_pct"].std() * np.sqrt(365*24)) if group["return_pct"].std() > 0 else 0
    res.append({
        "Regime": regime,
        "N": n,
        "WR": f"{wr:.1f}%",
        "RetMean": f"{ret_mean:.2f}%",
        "RetSum": f"{ret_sum:.1f}%",
        "Sharpe": f"{sharpe:.2f}"
    })

res_df = pd.DataFrame(res).sort_values("Sharpe", ascending=False)
print(res_df.to_string(index=False))

# Analisis de Costes / Slippage
print("\n--- SIMULACION DE SENSIBILIDAD A COSTES ADICIONALES ---")
# Actualmente asumimos ~0.25% cost/slippage base. Vamos a añadir penalizaciones.
cost_scenarios = [0.0, 0.05, 0.10, 0.20, 0.30] # Extra penalization %
for cost in cost_scenarios:
    cost_dec = cost / 100.0
    # Penalizamos cada trade con el coste extra (asumiendo que se resta del return)
    sim_ret = df_all["return_pct"] - cost_dec
    sim_wr = (sim_ret > 0).mean() * 100
    sim_sharpe = (sim_ret.mean() / sim_ret.std() * np.sqrt(365*24)) if sim_ret.std() > 0 else 0
    print(f"Coste Extra: +{cost:.2f}% | WR: {sim_wr:4.1f}% | Sharpe: {sim_sharpe:4.2f}")

print("\n--- CONCLUSIONES PRELIMINARES DE SLIPPAGE ---")
print("Si el slippage real supera nuestra estimacion en un 0.10%, el Sharpe podria caer.")
