"""Find actual column names for dashboard feature groups"""
import pandas as pd

df = pd.read_parquet('/root/luna_v2/data/features/features_live.parquet')

cols = list(df.columns)
ohlcv = [c for c in cols if c in ['close','open','high','low','volume','Close','Open','High','Low','Volume']]
hmm_cols = [c for c in cols if 'hmm' in c.lower() or 'regime' in c.lower() or c.startswith('HMM')]
ret_cols = sorted([c for c in cols if 'return' in c.lower() or 'ret_' in c.lower()])[:10]
atr_cols = [c for c in cols if 'atr' in c.lower()]
vol_cols = [c for c in cols if 'volatil' in c.lower() or 'vol_' in c.lower()][:8]

print(f"OHLCV: {ohlcv}")
print(f"Return cols: {ret_cols}")
print(f"ATR cols: {atr_cols}")
print(f"Vol cols: {vol_cols}")
print(f"HMM/Regime cols ({len(hmm_cols)}): {hmm_cols}")
print(f"\nLast row sample:")
print(df[ohlcv + hmm_cols].iloc[-1])
