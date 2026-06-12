"""Analiza el resultado de la mini-run H3 (5 seeds) vs el baseline."""
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Las semillas que corrimos
test_seeds = [42, 100, 777, 1337, 2025]
data_dir = Path("data/predictions")

print("========================================================================")
print("ANALISIS MINI-RUN H3 CAUSAL (Gate KL_Q75_IS)")
print("Semillas testeadas:", test_seeds)
print("========================================================================")

total_trades = 0
total_ret = 0.0
all_trades = []

for seed in test_seeds:
    f = data_dir / f"oos_trades_seed{seed}.parquet"
    if not f.exists():
        print(f"Seed {seed:4} -> NO HAY DATOS (Probablemente prune temprano)")
        continue
        
    df = pd.read_parquet(f)
    n = len(df)
    
    # Calcular métricas básicas
    wr = df["is_win"].mean() * 100
    ret = df["return_pct"].sum() * 100
    
    # Sharpe anualizado (aproximado usando freq horaria si hay 1 barra, pero como son trades discretos usamos std simple)
    # Asumiendo 1 trade = 1 barra para simplificar
    std = df["return_pct"].std()
    sharpe = (df["return_pct"].mean() / std * np.sqrt(365 * 24)) if std > 0 else 0
    
    # MaxDD
    df["cum_ret"] = (1 + df["return_pct"]).cumprod()
    df["rolling_max"] = df["cum_ret"].cummax()
    df["drawdown"] = df["cum_ret"] / df["rolling_max"] - 1.0
    maxdd = df["drawdown"].min() * 100
    
    print(f"Seed {seed:4} -> {n:4} trades | WR: {wr:4.1f}% | Ret: {ret:+6.1f}% | Sharpe: {sharpe:4.2f} | MaxDD: {maxdd:6.1f}%")
    
    total_trades += n
    all_trades.append(df)

print("-" * 72)
if all_trades:
    df_all = pd.concat(all_trades, ignore_index=True)
    wr = df_all["is_win"].mean() * 100
    ret = df_all["return_pct"].sum() * 100
    std = df_all["return_pct"].std()
    sharpe = (df_all["return_pct"].mean() / std * np.sqrt(365 * 24)) if std > 0 else 0
    
    df_all = df_all.sort_values(by="close_time") if "close_time" in df_all.columns else df_all
    df_all["cum_ret"] = (1 + df_all["return_pct"]).cumprod()
    df_all["rolling_max"] = df_all["cum_ret"].cummax()
    df_all["drawdown"] = df_all["cum_ret"] / df_all["rolling_max"] - 1.0
    maxdd = df_all["drawdown"].min() * 100
    
    print(f"TOTAL H3 -> {total_trades:4} trades | WR: {wr:4.1f}% | Ret: {ret:+6.1f}% | Sharpe: {sharpe:4.2f} | MaxDD: {maxdd:6.1f}%")
else:
    print("No hay trades para analizar.")

print("========================================================================")
