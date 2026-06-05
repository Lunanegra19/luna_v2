"""
Test final de los 3 features implementados:
- DXY-HMM-01: DXY condicional al regimen HMM
- EXCHANGE-FLOW-01: Exchange Net Flows reales/proxy
- LTH-SUPPLY-01: LTH Supply Change via CoinMetrics
[2026-06-03]
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r'g:\Mi unidad\ia\luna_v2')

print('='*65)
print('[TEST FINAL] Verificacion de los 3 features OOS-estables')
print('='*65)
print()

# ─── 1. DXY-HMM-01 ────────────────────────────────────────────────
print('── TEST 1: DXY-HMM-01 (pipeline feature_pipeline.py) ──────')
from luna.features.feature_pipeline import FeaturePipeline
fp = FeaturePipeline.__new__(FeaturePipeline)  # sin __init__ complejo
import inspect
src = inspect.getsource(fp.apply_derived_features.__func__ if hasattr(fp.apply_derived_features, '__func__') else type(fp).apply_derived_features)
assert 'DXY-HMM-01' in src, 'DXY-HMM-01 no en apply_derived_features'
assert 'DXY_HMM_cond' in src
assert 'DXY_HMM_bull_neg' in src
assert 'DXY_HMM_interact' in src
print('[OK] DXY-HMM-01 implementado en apply_derived_features')

# ─── 2. EXCHANGE-FLOW-01 ──────────────────────────────────────────
print()
print('── TEST 2: EXCHANGE-FLOW-01 (fetch_onchain.py) ─────────────')
from luna.data.fetch_onchain import OnchainFetcher
f = OnchainFetcher()
assert hasattr(f, 'fetch_cryptoquant_netflow')
# Verificar que el metodo se llama en build_onchain_dataset
import inspect as insp
builder_src = insp.getsource(f.build_onchain_dataset)
assert 'fetch_cryptoquant_netflow' in builder_src, 'netflow no integrado en builder'
assert 'EXCHANGE-FLOW-01' in builder_src
print('[OK] fetch_cryptoquant_netflow integrado en build_onchain_dataset')

# Ejecutar sin API key (nivel 3)
old_key = os.environ.pop('CRYPTOQUANT_API_KEY', None)
df_nf = f.fetch_cryptoquant_netflow(start='2023-01-01')
if old_key: os.environ['CRYPTOQUANT_API_KEY'] = old_key
assert not df_nf.empty, 'Exchange NetFlow DataFrame vacio'
expected_cols = ['Exchange_NetFlow', 'Exchange_NetFlow_7dEMA',
                 'Exchange_NetFlow_z30d', 'Exchange_NetFlow_Accum30d',
                 'Exchange_Outflow_Signal']
for col in expected_cols:
    assert col in df_nf.columns, f'{col} no en DataFrame'
print(f'[OK] Exchange NetFlow: {df_nf.shape} | cols: {list(df_nf.columns)}')

# ─── 3. LTH-SUPPLY-01 ─────────────────────────────────────────────
print()
print('── TEST 3: LTH-SUPPLY-01 (fetch_onchain.py + CoinMetrics) ──')
assert hasattr(f, 'fetch_lth_supply_proxy')
lth_src = insp.getsource(f.fetch_lth_supply_proxy)
assert 'CoinMetrics' in lth_src
assert 'SplyExNtv' in lth_src
assert 'LTH-SUPPLY-01' in lth_src
assert 'fetch_lth_supply_proxy' in builder_src, 'LTH no integrado en builder'
print('[OK] fetch_lth_supply_proxy implementado con CoinMetrics')

df_lth = f.fetch_lth_supply_proxy(start='2023-01-01')
assert not df_lth.empty, 'LTH Supply DataFrame vacio'
expected_lth = ['NonEx_Supply', 'LTH_Supply_Change_30d',
                'LTH_Accum_Signal', 'NonEx_Supply_z90d']
for col in expected_lth:
    assert col in df_lth.columns, f'{col} no en LTH DataFrame'
print(f'[OK] LTH Supply: {df_lth.shape} | cols: {list(df_lth.columns)}')
last_change = float(df_lth['LTH_Supply_Change_30d'].dropna().iloc[-1])
print(f'[OK] LTH_Supply_Change_30d actual: {last_change:+.0f} BTC (>0 = acumulacion)')

# ─── 4. Settings verificacion ──────────────────────────────────────
print()
print('── TEST 4: settings.yaml — whitelists completas ────────────')
from config.settings import cfg

wl_macro    = list(getattr(cfg.features, 'sfi_macro_features', []) or [])
wl_onchain  = list(getattr(cfg.features, 'sfi_onchain_features', []) or [])
wl_stable   = list(getattr(cfg.features, 'sfi_macro_stable_features', []) or [])

# DXY-HMM
for f_name in ['DXY_HMM_cond', 'DXY_HMM_bull_neg', 'DXY_HMM_interact']:
    assert f_name in wl_macro, f'{f_name} no en sfi_macro_features'
    assert f_name in wl_stable, f'{f_name} no en sfi_macro_stable_features'
print(f'[OK] DXY-HMM-01: 3 features en sfi_macro_features + sfi_macro_stable_features')

# Exchange Flow
for f_name in ['Exchange_NetFlow_z30d', 'Exchange_NetFlow_Accum30d', 'Exchange_Outflow_Signal']:
    assert f_name in wl_onchain, f'{f_name} no en sfi_onchain_features'
    assert f_name in wl_stable, f'{f_name} no en sfi_macro_stable_features'
print(f'[OK] EXCHANGE-FLOW-01: features en sfi_onchain + sfi_macro_stable_features')

# LTH
for f_name in ['LTH_Supply_Change_30d', 'LTH_Accum_Signal', 'NonEx_Supply_z90d']:
    assert f_name in wl_onchain, f'{f_name} no en sfi_onchain_features'
    assert f_name in wl_stable, f'{f_name} no en sfi_macro_stable_features'
print(f'[OK] LTH-SUPPLY-01: features en sfi_onchain + sfi_macro_stable_features')

print()
print('='*65)
print('[RESUMEN FINAL] Todos los tests OK')
print(f'  sfi_macro_features:   {len(wl_macro)} features')
print(f'  sfi_onchain_features: {len(wl_onchain)} features')
print(f'  sfi_macro_stable (boost): {len(wl_stable)} features')
print()
print('  DXY-HMM-01:        3 features | apply_derived_features (Paso 7B)')
print('  EXCHANGE-FLOW-01:  5 features | fetch_onchain.py (3 niveles fallback)')
print('  LTH-SUPPLY-01:     5 features | fetch_onchain.py (CoinMetrics gratis)')
print('='*65)
