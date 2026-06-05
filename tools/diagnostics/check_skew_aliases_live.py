"""Quick check for SKEW aliases in current features_live.parquet"""
import pandas as pd

df = pd.read_parquet('/root/luna_v2/data/features/features_live.parquet')
cols = set(df.columns)

aliases = {
    'FundingRate_EMA3': ['funding_ema_3'],
    'FundingRate_Pct90d': ['dv_funding_pct_90d', 'funding_pct_90d'],
    'OI_Open_USD': ['Coinglass_oi_open'],
    'OI_High_USD': ['Coinglass_oi_high'],
    'OI_Low_USD': ['Coinglass_oi_low'],
    'ETF_Flow_Proxy': ['etf_flow_proxy'],
    'dv_etf_flow_proxy': [],
}

print(f"Last index: {df.index[-1]}")
print(f"Total cols: {len(cols)}")
print("\n=== ALIAS STATUS IN CURRENT LIVE PARQUET ===")
for train_col, live_equiv in aliases.items():
    in_live = train_col in cols
    equiv_in_live = [c for c in live_equiv if c in cols]
    print(f"  {train_col}: {'OK ✓' if in_live else 'MISSING ✗'} | live equiv: {equiv_in_live}")
