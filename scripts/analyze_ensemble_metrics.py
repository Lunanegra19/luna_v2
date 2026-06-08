import pandas as pd
import numpy as np

df = pd.read_parquet("data/predictions/ensemble_portfolio_trades.parquet")
print("Total rows:", len(df))
print(df.columns)
print(df)

# Calculemos el retorno nominal acumulado y compuesto
returns = df["return_pct"]
nom_ret = returns.sum()
comp_ret = (1.0 + returns).prod() - 1.0

# Win rate
wr = (df["is_win"] > 0).mean()

# Drawdown
cum_prod = (1.0 + returns).cumprod()
peak = cum_prod.cummax()
dd = (cum_prod - peak) / peak
max_dd = dd.min()

print(f"Nominal Return: {nom_ret * 100:.4f}%")
print(f"Compound Return: {comp_ret * 100:.4f}%")
print(f"Win Rate: {wr * 100:.2f}%")
print(f"Max Drawdown: {max_dd * 100:.4f}%")

# Sharpe Ratio
n_trades = len(df)
if n_trades > 1:
    std_r = returns.std()
    days = (df.index.max() - df.index.min()).days
    n_per_year = n_trades / (days / 365.25) if days > 0 else n_trades * 365.25
    sharpe = (returns.mean() / std_r) * (n_per_year ** 0.5) if std_r > 0 else 0
    calmar = comp_ret / abs(max_dd) if abs(max_dd) > 0 else 0
else:
    sharpe = 0
    calmar = 0

print(f"Annualized Sharpe: {sharpe:.4f}")
print(f"Calmar Ratio: {calmar:.4f}")

# Contar cuántos trades tuvieron consenso >= 4
if "consensus_count" in df.columns:
    print("Consensus counts distribution:")
    print(df["consensus_count"].value_counts())
