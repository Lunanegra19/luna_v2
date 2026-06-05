import pandas as pd
import numpy as np
import os
import glob

base_path = r"g:\Mi unidad\ia\luna_v2"
pred_dir = os.path.join(base_path, "data", "predictions")

# Buscar todos los parquets de trades OOS de runs previas
parquet_files = glob.glob(os.path.join(pred_dir, "oos_trades_seed*.parquet"))

print(f"Cargando {len(parquet_files)} archivos de trades históricos OOS...")

dfs = []
for f in parquet_files:
    try:
        _df = pd.read_parquet(f)
        if 'xgb_prob_cal' in _df.columns and 'return_pct' in _df.columns and 'is_win' in _df.columns:
            dfs.append(_df)
    except Exception as e:
        pass

if not dfs:
    print("No se encontraron trades históricos válidos.")
    exit(1)

df_all = pd.concat(dfs, ignore_index=True)

print("\n==== ANÁLISIS BASE GLOBAL (LUNA V1/V2 Histórico) ====")
base_winrate = df_all['is_win'].mean()
base_return = df_all['return_pct'].sum()
print(f"Total Trades Históricos: {len(df_all)}")
print(f"WinRate Base Global: {base_winrate:.2%}")
print(f"Retorno Total Bruto (Flat Sizing): {base_return:.4%}")

print("\n==== SIMULACIÓN IDEA 3: FRACTIONAL KELLY GLOBAL ====")

capital = 10000.0  # Flat
capital_kelly = 10000.0  # Kelly

# Usamos half-kelly acotado al 20%
def calculate_kelly(prob_win, reward_risk=1.5):
    b = reward_risk
    p = float(prob_win)
    if pd.isna(p): return 0.05
    q = 1.0 - p
    kelly_f = p - (q / b)
    return max(0.01, min(kelly_f * 0.5, 0.20)) # Min 1%, Max 20%, Half-Kelly

df_all['kelly_f'] = df_all['xgb_prob_cal'].apply(lambda x: calculate_kelly(x))

# Ordenar por fecha si el indice fuera datetime, pero concat reseteó.
# Simularemos secuencialmente como si fuera la curva global
for idx, row in df_all.iterrows():
    # Base flat sizing (5% fijo)
    pnl_base = capital * 0.05 * float(row['return_pct']) * 10 
    capital += pnl_base
    
    # Kelly Sizing
    f = row['kelly_f']
    pnl_kelly = capital_kelly * f * float(row['return_pct']) * 10
    capital_kelly += pnl_kelly
    
    # Evitar bancarrota total en la simulación
    if capital < 0: capital = 0.01
    if capital_kelly < 0: capital_kelly = 0.01

print(f"\nResultados tras {len(df_all)} trades secuenciales apalancados x10:")
print(f"-> Capital Final Flat Sizing (5% fijo): ${capital:,.2f}")
print(f"-> Capital Final Fractional Kelly (Dinámico): ${capital_kelly:,.2f}")

diff_pct = ((capital_kelly - capital) / capital) * 100
print(f"\nMejora Neta del Capital: {diff_pct:+.2f}%")
