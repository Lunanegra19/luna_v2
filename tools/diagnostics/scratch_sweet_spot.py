import pandas as pd
import numpy as np

# Cargar trades raw de TODAS las semillas del ensemble de anoche
df = pd.read_parquet('data/predictions/unified_ensemble_trades_raw.parquet')

# Convertir HMM_Semantic a grandes grupos
def categorize_regime(x):
    if pd.isna(x): return 'NEUTRAL'
    x = str(x).upper()
    if 'BULL' in x: return 'BULL'
    if 'BEAR' in x: return 'BEAR'
    return 'NEUTRAL'

df['regime_group'] = df['HMM_Semantic'].apply(categorize_regime)

print("="*60)
print(f"ANÁLISIS DE SWEET SPOT - ENSEMBLE (Total Trades Base: {len(df)})")
print("="*60)

# Queremos ver que pasa al subir el umbral (simulando rolling percentiles a nivel macro)
percentiles_to_test = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

for group in ['BULL', 'BEAR', 'NEUTRAL']:
    print(f"\n--- Régimen: {group} ---")
    df_g = df[df['regime_group'] == group]
    if len(df_g) == 0:
        continue
    
    # Probabilidad base para calcular percentiles macro
    probs = df_g['xgb_prob'].dropna()
    
    print(f"Trades Base: {len(df_g)} | Win Rate Base: {(df_g['is_win'] == True).mean()*100:.1f}%")
    
    for pct in percentiles_to_test:
        thresh = np.percentile(probs, pct * 100)
        df_filtered = df_g[df_g['xgb_prob'] >= thresh]
        wr = (df_filtered['is_win'] == True).mean() * 100
        n_trades = len(df_filtered)
        
        # Para el ensemble necesitamos q queden suficientes trades. 
        # Si de 5 semillas quedan 150 trades brutos, son unos 30 por semilla.
        status = "✅ OK" if wr >= 50.0 else "❌ BAJO"
        
        print(f"  P{pct*100:.0f} (Thresh >= {thresh:.3f}): WR = {wr:.1f}% | Trades = {n_trades} | {status}")

print("\n" + "="*60)
print("SIMULACIÓN GLOBAL COMBINANDO LOS MEJORES PERCENTILES")
print("="*60)
