"""mi_horizon_analysis.py - Analisis empirico de MI HMM vs distintos horizontes forward"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

df = pd.read_parquet('data/features/features_train.parquet')
print(f'IS: {len(df)} filas')
print(f'HMM_Regime unique: {sorted(df["HMM_Regime"].dropna().unique()[:8].tolist())}')
print(f'HMM_Semantic unique: {sorted(df["HMM_Semantic"].dropna().unique().tolist())}')
print()

# Calcular MI con distintos horizontes forward
print('=== MI(HMM_Regime, fwd_return) por horizonte ===')
for horizon in [24, 48, 96, 168, 336, 720]:
    fwd = (df['close'].pct_change(horizon).shift(-horizon) > 0).astype(int)
    hmm_cat = df['HMM_Regime'].astype('category').cat.codes
    mi_df = pd.DataFrame({'s': hmm_cat, 't': fwd}).dropna()
    if len(mi_df) > 100:
        mi = mutual_info_score(mi_df['s'], mi_df['t'])
        flag = ' *** SUPERA SOP-R9=0.005' if mi >= 0.005 else f' (< 0.005, viola SOP-R9)'
        print(f'  horizon={horizon:4d}H: MI={mi:.5f}{flag}')

print()
# Duracion media de regimenes
print('=== Duracion media regimenes HMM ===')
regime_durations = []
prev = None
count = 0
for v in df['HMM_Regime'].dropna():
    if v != prev:
        if prev is not None and count > 0:
            regime_durations.append(count)
        count = 1
        prev = v
    else:
        count += 1
if regime_durations:
    print(f'Duracion media: {np.mean(regime_durations):.0f}H ({np.mean(regime_durations)/24:.1f} dias)')
    print(f'Duracion mediana: {np.median(regime_durations):.0f}H ({np.median(regime_durations)/24:.1f} dias)')
    print(f'Duracion max: {max(regime_durations):.0f}H ({max(regime_durations)/24:.1f} dias)')
    print(f'Horizonte optimo para MI: {int(np.median(regime_durations))}H')
    print()
    # Distribucion por regimen
    print('Distribucion de regimenes:')
    print(df['HMM_Semantic'].value_counts())
