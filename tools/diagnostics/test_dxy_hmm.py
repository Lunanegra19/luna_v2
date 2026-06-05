"""
Test de verificacion DXY-HMM-01 — features DXY condicionadas al regimen HMM.
[DXY-HMM-01 2026-06-03]
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

print('[TEST DXY-HMM-01] Verificacion logica de features condicionales')
print()

# Simular DataFrame con las columnas necesarias
np.random.seed(42)
n = 500
dates = pd.date_range('2021-01-01', periods=n, freq='h', tz='UTC')
df_test = pd.DataFrame({
    'DXY_Ret':    np.random.normal(0, 0.002, n),
    'DXY_Zscore': np.random.normal(0, 1.0,   n),
    'HMM_Regime': np.random.choice([0, 1, 2, 3, 4], n),
    'close':      np.cumsum(np.random.normal(100, 50, n)) + 30000,
}, index=dates)

hmm_dist = dict(df_test['HMM_Regime'].value_counts().sort_index())
print(f'  DataFrame test: {df_test.shape} | HMM dist: {hmm_dist}')

# Simular bloque DXY-HMM-01
_has_dxy_ret  = 'DXY_Ret' in df_test.columns
_has_dxy_z    = 'DXY_Zscore' in df_test.columns
_has_hmm      = 'HMM_Regime' in df_test.columns
_n_states_dxh = 5

assert _has_dxy_ret and _has_hmm and _has_dxy_z

_dxy_ret     = df_test['DXY_Ret'].ffill()
_hmm_reg     = df_test['HMM_Regime'].fillna(df_test['HMM_Regime'].median())
_regime_norm = (_hmm_reg / (_n_states_dxh - 1)) * 2 - 1   # [0,4] -> [-1, +1]

df_test['DXY_HMM_cond']     = _dxy_ret * (-_regime_norm)
_bull_flag = (_hmm_reg >= _n_states_dxh // 2).astype(float)
df_test['DXY_HMM_bull_neg'] = _dxy_ret * _bull_flag * (-1.0)
df_test['DXY_HMM_interact'] = df_test['DXY_Zscore'].ffill() * _regime_norm

print()
print('[DXY-HMM-01] Features generadas:')
for feat in ['DXY_HMM_cond', 'DXY_HMM_bull_neg', 'DXY_HMM_interact']:
    s = df_test[feat]
    print(f'  {feat:22s}: N={s.notna().sum()} mean={s.mean():.5f} std={s.std():.5f} range=[{s.min():.4f},{s.max():.4f}]')

# Verificacion semantica
bear_mask = df_test['HMM_Regime'] == 0
bull_mask = df_test['HMM_Regime'] == 4
bear_bull_neg_mean = df_test.loc[bear_mask, 'DXY_HMM_bull_neg'].mean()
bull_bull_neg_mean = df_test.loc[bull_mask, 'DXY_HMM_bull_neg'].mean()
bear_regime_norm   = float(_regime_norm[bear_mask].mean())
bull_regime_norm   = float(_regime_norm[bull_mask].mean())

print()
print('[DXY-HMM-01] Verificacion semantica:')
print(f'  HMM=0 (crash): DXY_HMM_bull_neg mean={bear_bull_neg_mean:.6f} (debe ser 0.0)')
print(f'  HMM=4 (bull):  DXY_HMM_bull_neg mean={bull_bull_neg_mean:.6f} (debe ser != 0)')
print(f'  HMM=0 (crash): regime_norm = {bear_regime_norm:.3f} (debe ser -1.0)')
print(f'  HMM=4 (bull):  regime_norm = {bull_regime_norm:.3f} (debe ser +1.0)')

assert abs(bear_bull_neg_mean) < 0.0001, f'DXY_HMM_bull_neg en crash debe ser 0, got {bear_bull_neg_mean}'
bull_std = float(df_test.loc[bull_mask, 'DXY_HMM_bull_neg'].std())
assert bull_std > 0.0005, f'DXY_HMM_bull_neg en bull debe tener std>0 (activo), got std={bull_std}'
print(f'  HMM=4 (bull):  DXY_HMM_bull_neg std={bull_std:.5f} (debe ser > 0 -- feature activa)')

assert abs(bear_regime_norm - (-1.0)) < 0.01, f'regime_norm crash debe ser -1, got {bear_regime_norm}'
assert abs(bull_regime_norm - (1.0))  < 0.01, f'regime_norm bull debe ser +1, got {bull_regime_norm}'

# Correlacion — debe ser != +-1.0
corr_cond = df_test['DXY_HMM_cond'].corr(df_test['DXY_Ret'])
corr_bull = df_test['DXY_HMM_bull_neg'].corr(df_test['DXY_Ret'])
print()
print(f'  corr(DXY_HMM_cond, DXY_Ret)     = {corr_cond:.3f}  (debe ser != +-1.0)')
print(f'  corr(DXY_HMM_bull_neg, DXY_Ret) = {corr_bull:.3f}  (debe ser != +-1.0)')
assert abs(corr_cond) < 0.99,  'DXY_HMM_cond es identico a DXY_Ret -- error logica'
assert abs(corr_bull) < 0.99,  'DXY_HMM_bull_neg es identico a DXY_Ret -- error logica'

# Verificar settings
from config.settings import cfg
wl_macro  = list(getattr(cfg.features, 'sfi_macro_features', []) or [])
wl_stable = list(getattr(cfg.features, 'sfi_macro_stable_features', []) or [])

for feat in ['DXY_HMM_cond', 'DXY_HMM_bull_neg', 'DXY_HMM_interact']:
    assert feat in wl_macro,  f'{feat} no en sfi_macro_features'
    assert feat in wl_stable, f'{feat} no en sfi_macro_stable_features'

print()
print('[DXY-HMM-01] TODAS LAS VERIFICACIONES OK')
print('  3 features DXY condicionales generadas correctamente')
print('  Semantica verificada: bull_neg=0 en crash, !=0 en bull')
print('  Registradas en sfi_macro_features (cuota) y sfi_macro_stable_features (boost)')
print(f'  sfi_macro_features total: {len(wl_macro)} | sfi_macro_stable_features: {len(wl_stable)}')
