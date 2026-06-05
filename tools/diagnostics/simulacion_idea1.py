import pandas as pd
import numpy as np
import os

base_path = r"g:\Mi unidad\ia\luna_v2"
w1_oos_parquet = os.path.join(base_path, "data", "reports", "wfb", "oos_trades_W1_seed42.parquet")

df = pd.read_parquet(w1_oos_parquet)

print("==== ANÁLISIS BASE (LUNA V2 ACTUAL) ====")
base_winrate = df['is_win'].mean()
base_return = df['return_pct'].sum()
print(f"Total Trades: {len(df)}")
print(f"WinRate Base: {base_winrate:.2%}")
print(f"Retorno Total (Flat Sizing): {base_return:.4%}")

print("\n==== SIMULACIÓN IDEA 1: ENSEMBLING HÍBRIDO (XGBoost + MetaLabeler) ====")
# Como lgbm_prob no está disponible, creamos un Soft-Voting entre el Strong Learner (XGBoost) y el MetaLabeler
# Le damos un peso de 40% a la macro-convicción del XGB y 60% a la micro-precisión del MetaLabeler
df['ensemble_prob'] = (df['xgb_prob_cal'].astype(float) * 0.4) + (df['meta_v2_prob'].astype(float) * 0.6)

# Probemos con un umbral más estricto del 65% de convicción combinada
umbral_ensemble = 0.65
df_ensemble = df[df['ensemble_prob'] > umbral_ensemble]

ens_winrate = df_ensemble['is_win'].mean() if len(df_ensemble) > 0 else 0
ens_return = df_ensemble['return_pct'].sum() if len(df_ensemble) > 0 else 0

print(f"Umbral de Soft-Voting Híbrido: > {umbral_ensemble:.0%}")
print(f"Trades retenidos: {len(df_ensemble)} / {len(df)}")
print(f"Trades filtrados (evitados): {len(df) - len(df_ensemble)}")

if len(df_ensemble) > 0:
    # Vamos a ver cuántas de las operaciones evitadas eran perdedoras
    df_filtrados = df[df['ensemble_prob'] <= umbral_ensemble]
    evitadas_perdedoras = len(df_filtrados[df_filtrados['is_win'] == False])
    evitadas_ganadoras = len(df_filtrados[df_filtrados['is_win'] == True])
    print(f"De los {len(df_filtrados)} trades filtrados: {evitadas_perdedoras} eran perdedores y {evitadas_ganadoras} ganadores.")
    print(f"-> Nuevo WinRate (Ensemble): {ens_winrate:.2%} (Mejora: {(ens_winrate - base_winrate)*100:+.2f}%)")
    print(f"-> Nuevo Retorno Total: {ens_return:.4%}")
else:
    print("Ningún trade superó el umbral.")
