import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("SIMULADOR DE HARD VOTING EN ENSEMBLE (OOS)")
    print("=" * 60)
    
    data_dir = Path("data/predictions")
    files = list(data_dir.glob("oos_trades_seed*.parquet"))
    
    if not files:
        print("No se encontraron archivos oos_trades_seed.")
        sys.exit(1)
        
    print(f"[*] Cargando {len(files)} archivos de trades (semillas que pasaron sus umbrales locales)...")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df['_seed'] = int(f.stem.split('seed')[1])
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
            dfs.append(df)
        except Exception as e:
            print(f"Error cargando {f.name}: {e}")
            
    df_all = pd.concat(dfs)
    print(f"[*] Total trades individuales agrupados: {len(df_all)}")
    
    # Agrupar por timestamp exacto (Hard Voting)
    print("\n[*] Calculando superposicion de semillas por cada Timestamp...")
    
    overlap = df_all.groupby(level=0).agg(
        n_seeds=('_seed', 'nunique'),
        is_win=('is_win', 'mean') # Si el 100% de las semillas que entraron en ese timestamp dicen 'win', la media sera 1
    )
    
    total_unique_timestamps = len(overlap)
    print(f"[*] Timestamps unicos donde al menos 1 semilla disparo: {total_unique_timestamps}")
    
    print("\n" + "=" * 60)
    print("RESULTADOS DE UMBRALES DE CONSENSO (HARD VOTING)")
    print("=" * 60)
    print(f"{'Consenso (N seeds)':>18} | {'Trades':>8} | {'Win Rate (aprox)':>18}")
    print("-" * 60)
    
    for n_req in range(1, 10):
        subset = overlap[overlap['n_seeds'] >= n_req]
        n_trades = len(subset)
        if n_trades == 0:
            break
            
        wr = subset['is_win'].mean() * 100
        print(f"{n_req:>18} | {n_trades:>8} | {wr:>17.1f}%")

    print("=" * 60)

if __name__ == "__main__":
    main()
