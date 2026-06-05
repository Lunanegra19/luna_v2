import pandas as pd
import numpy as np
import os
import json

# Rutas
base_path = r"g:\Mi unidad\ia\luna_v2"
w1_oos_parquet = os.path.join(base_path, "data", "reports", "wfb", "oos_trades_W1_seed42.parquet")

if not os.path.exists(w1_oos_parquet):
    print(f"Error: {w1_oos_parquet} no encontrado.")
    exit(1)

df = pd.read_parquet(w1_oos_parquet)

print("==== ANÁLISIS BASE (LUNA V2 ACTUAL) ====")
base_winrate = df['is_win'].mean()
base_return = df['return_pct'].sum()
print(f"Total Trades: {len(df)}")
print(f"WinRate Base: {base_winrate:.2%}")
print(f"Retorno Total (Flat Sizing, sin comisiones): {base_return:.4%}")

# Idea 1: MetaLabeler Ensembling (Soft Voting)
# Asumimos que si la media de las probabilidades es mayor a cierto umbral, operamos.
print("\n==== MEJORA 1: METALABELER ENSEMBLING ====")
if 'meta_v2_prob' in df.columns and 'lgbm_prob' in df.columns:
    df['ensemble_prob'] = (df['meta_v2_prob'].astype(float) + df['lgbm_prob'].astype(float)) / 2.0
    
    # Filtramos donde ensemble > 0.50
    df_ensemble = df[df['ensemble_prob'] > 0.50]
    ens_winrate = df_ensemble['is_win'].mean() if len(df_ensemble) > 0 else 0
    ens_return = df_ensemble['return_pct'].sum() if len(df_ensemble) > 0 else 0
    
    print(f"Trades retenidos: {len(df_ensemble)} / {len(df)}")
    if len(df_ensemble) < len(df):
        print(f"Trades filtrados (evitados): {len(df) - len(df_ensemble)}")
    print(f"Nuevo WinRate (Ensemble): {ens_winrate:.2%}")
    print(f"Nuevo Retorno Total: {ens_return:.4%}")
else:
    print("No se encontraron las probabilidades de múltiples modelos para ensembling.")

# Idea 2: Fractional Kelly (Position Sizing Dinámico)
print("\n==== MEJORA 2: FRACTIONAL KELLY (DINÁMICO) ====")
# Frmula Kelly simple: K = W - ((1 - W) / R)
# Para WFB, estimaremos WinRate global histórico en 0.55 y un R (Reward/Risk) de 1.5 a modo de ejemplo.
# Usaremos la probabilidad calibrada para ajustar la fracción.

capital = 10000.0  # Capital inicial base
capital_kelly = 10000.0

def calculate_kelly(prob_win, reward_risk=1.5):
    # prob_win viene del calibrador
    b = reward_risk
    p = prob_win
    q = 1.0 - p
    kelly_f = p - (q / b)
    # Acotar Kelly (Half-Kelly o Fractional Kelly)
    fraction = max(0.0, min(kelly_f, 0.20)) # Cap max 20%
    return fraction * 0.5 # Half-Kelly

df['kelly_f'] = df['xgb_prob_cal'].apply(lambda x: calculate_kelly(x))

for idx, row in df.iterrows():
    # Base flat sizing (asignamos un 5% por trade)
    pnl_base = 10000.0 * 0.05 * row['return_pct'] * 10  # Apalancamiento x10 hipotético
    capital += pnl_base
    
    # Kelly Sizing
    f = row['kelly_f']
    pnl_kelly = capital_kelly * f * row['return_pct'] * 10
    capital_kelly += pnl_kelly

print(f"Capital Final Flat Sizing (5% fijo, 10x): ${capital:.2f}")
print(f"Capital Final Fractional Kelly (Dinámico, 10x): ${capital_kelly:.2f}")

print("\nConclusión: El test aislado finalizó.")
