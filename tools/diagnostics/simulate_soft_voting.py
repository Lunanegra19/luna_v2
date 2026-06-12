import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Forzar codificación utf-8 para la salida estándar en Windows
sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("SIMULADOR DE SOFT VOTING EN ENSEMBLE (OOS)")
    print("=" * 60)
    
    data_dir = Path("data/reports/wfb")
    if not data_dir.exists():
        data_dir = Path("data/predictions")
        
    files = list(data_dir.glob("oos_raw_probs_W*_seed*.parquet"))
    if not files:
        print("No se encontraron archivos oos_raw_probs.")
        sys.exit(1)
        
    print(f"[*] Cargando {len(files)} archivos de probabilidades raw...")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
            
            parts = f.stem.split('_')
            seed = int(parts[-1].replace('seed', ''))
            df['_seed'] = seed
            dfs.append(df)
        except Exception as e:
            print(f"Error cargando {f.name}: {e}")
            
    df_all = pd.concat(dfs)
    print(f"[*] Total filas raw cargadas: {len(df_all)}")
    print(f"[*] Columnas disponibles: {df_all.columns.tolist()}")
    
    prob_col = 'xgb_prob_cal' if 'xgb_prob_cal' in df_all.columns else ('prob_cal' if 'prob_cal' in df_all.columns else None)
    
    if not prob_col:
        # Fallback: buscar cualquier columna que contenga prob
        prob_cols = [c for c in df_all.columns if 'prob' in c]
        if prob_cols:
            prob_col = prob_cols[0]
            print(f"[*] Usando columna de probabilidad detectada: {prob_col}")
        else:
            print("ERROR: No se encuentra columna de probabilidad.")
            sys.exit(1)
            
    print("\n[*] Agrupando por timestamp para calcular la Probabilidad Media (Soft Voting)...")
    
    ensemble_probs = df_all.groupby(level=0).agg(
        mean_prob=(prob_col, 'mean'),
        active_seeds=('_seed', 'nunique')
    )
    
    total_timestamps = len(ensemble_probs)
    print(f"[*] Timestamps unicos evaluados: {total_timestamps}")
    
    print("\n" + "=" * 60)
    print("RESULTADOS DE LA SIMULACION DE UMBRALES (SOFT VOTING)")
    print("=" * 60)
    print(f"{'Umbral':>8} | {'Trades Disparados':>18} | {'Multiplicador (vs 35)':>22}")
    print("-" * 60)
    
    base_trades = 35 
    
    for thresh in [0.50, 0.505, 0.51, 0.52, 0.55]:
        trades_triggered = (ensemble_probs['mean_prob'] >= thresh).sum()
        mult = trades_triggered / base_trades if base_trades > 0 else 0
        print(f"{thresh:>8.3f} | {trades_triggered:>18} | {mult:>21.1f}x")
            
    print("\n(Multiplicador calculado asumiendo el umbral sobre la media de las semillas vivas)")
    print("=" * 60)

if __name__ == "__main__":
    main()
