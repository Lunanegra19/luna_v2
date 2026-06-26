import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

_ROOT = Path(r"C:\Users\Usuario\Desktop\ia\luna_v2")
predictions_dir = _ROOT / "data" / "predictions"

def evaluate_ensemble(direction):
    files = list(predictions_dir.glob(f"oos_trades_seed*_{direction}.parquet"))
    
    seeds = set()
    dfs = []
    
    for f in files:
        df = pd.read_parquet(f)
        if not df.empty:
            df['seed'] = f.stem.split('_')[2].replace('seed', '')
            seeds.add(df['seed'].iloc[0])
            
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index, utc=True)
            dfs.append(df)
            
    if not dfs:
        return pd.DataFrame(), 0
        
    df_all = pd.concat(dfs).sort_index()
    n_seeds = len(seeds)
    threshold = max(2, int(n_seeds / 3))
    
    print(f"[{direction.upper()}] Encontradas {n_seeds} semillas completadas. Threshold de consenso (1/3) = {threshold}")
    
    # Bucket consensus 1h
    df_all['consensus_bucket'] = df_all.index.floor('1h')
    bucket_unique_seeds = df_all.groupby('consensus_bucket')['seed'].nunique().rename('consensus_count')
    df_all['consensus_count'] = df_all['consensus_bucket'].map(bucket_unique_seeds)
    
    df_filtered = df_all[df_all['consensus_count'] >= threshold].copy()
    
    agg_dict = {
        'return_pct': 'mean',
        'direction': 'first'
    }
    
    df_portfolio = df_filtered.groupby('consensus_bucket').agg(agg_dict).sort_index()
    df_portfolio['is_win'] = (df_portfolio['return_pct'] > 0).astype(float)
    
    return df_portfolio, n_seeds

# Evaluar LONGS
df_longs, n_longs = evaluate_ensemble("long")

# Evaluar SHORTS 
df_shorts, n_shorts = evaluate_ensemble("short")

# Combinar ambos (cruce de ensambles)
print("\n--- CRUZANDO ENSAMBLES (LONGS + SHORTS) ---")
if not df_longs.empty and not df_shorts.empty:
    df_combined = pd.concat([df_longs, df_shorts]).sort_index()
    
    # Resolver colisiones en el mismo bucket (si long y short ocurren en la misma hora)
    # En caso de colisión, si las señales son contrarias, se pueden anular o sumar los retornos.
    # Sumaremos retornos.
    df_final = df_combined.groupby(df_combined.index).agg({
        'return_pct': 'sum',
        'direction': lambda x: 'neutral' if len(x) > 1 else x.iloc[0]
    })
    df_final = df_final[df_final['return_pct'] != 0].copy()
    df_final['is_win'] = (df_final['return_pct'] > 0).astype(float)
    
    n_trades = len(df_final)
    wr = df_final['is_win'].mean() * 100
    ret_mean = df_final['return_pct'].mean() * 100
    
    # Sharpe Anualizado
    std_r = df_final['return_pct'].std()
    days = (df_final.index.max() - df_final.index.min()).days if n_trades > 1 else 0
    n_per_year = n_trades / (days / 365.25) if days > 0 else n_trades * 365.25
    sharpe = (df_final['return_pct'].mean() / std_r) * (n_per_year ** 0.5) if std_r > 0 else 0
    
    # Max DD
    cum_returns = np.cumsum(df_final['return_pct'].values)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = running_max - cum_returns
    max_dd = np.max(drawdowns) * 100
    
    print(f"Total Trades Consolidados: {n_trades}")
    print(f"Win Rate: {wr:.2f}%")
    print(f"Retorno Medio por Trade: {ret_mean:.4f}%")
    print(f"Sharpe Ratio Anualizado: {sharpe:.4f}")
    print(f"Max Drawdown (Nominal): {max_dd:.2f}%")
    print(f"Calmar Ratio: {((ret_mean * n_per_year) / max_dd) if max_dd > 0 else 0:.4f}")
    
else:
    print("Faltan datos de long o short para cruzar.")
