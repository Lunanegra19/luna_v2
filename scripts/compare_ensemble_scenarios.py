import pandas as pd
import numpy as np

# Cargar ambos datasets
df_raw = pd.read_parquet("data/predictions/unified_ensemble_trades_raw.parquet")
df_ens = pd.read_parquet("data/predictions/ensemble_portfolio_trades.parquet")

def compute_metrics(df, label):
    returns = df["return_pct"]
    nom_ret = returns.sum()
    comp_ret = (1.0 + returns).prod() - 1.0
    wr = (df["is_win"] > 0).mean()
    
    cum_prod = (1.0 + returns).cumprod()
    peak = cum_prod.cummax()
    dd = (cum_prod - peak) / peak
    max_dd = dd.min()
    
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
        
    print(f"=== {label} ===")
    print(f"Total Trades: {n_trades}")
    print(f"Win Rate: {wr * 100:.2f}%")
    print(f"Nominal Return: {nom_ret * 100:.4f}%")
    print(f"Compound Return: {comp_ret * 100:.4f}%")
    print(f"Max Drawdown: {max_dd * 100:.4f}%")
    print(f"Annualized Sharpe: {sharpe:.4f}")
    print(f"Calmar Ratio: {calmar:.4f}")
    print()

compute_metrics(df_raw, "SIN FILTRO DE CONSENSO (18 semillas operando libremente)")
compute_metrics(df_ens, "CON FILTRO DE CONSENSO (Consenso >= 3 semillas, 2H bucket)")
