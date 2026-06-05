"""Definitive check of ETF/OI/Funding aliases in current live parquet"""
import pandas as pd

df = pd.read_parquet('/root/luna_v2/data/features/features_live.parquet')
print(f"Last index: {df.index[-1]}")
print(f"Total cols: {len(df.columns)}")
print(f"\nETF cols: {sorted([c for c in df.columns if 'ETF' in c or 'etf' in c])}")
print(f"\nFunding cols: {sorted([c for c in df.columns if 'Funding' in c or 'funding' in c])}")
print(f"\nOI cols: {sorted([c for c in df.columns if c.startswith('OI_') or 'oi_' in c.lower()])}")
print(f"\nFundingRate_EMA3: {'OK' if 'FundingRate_EMA3' in df.columns else 'MISSING'}")
print(f"ETF_Flow_Proxy: {'OK' if 'ETF_Flow_Proxy' in df.columns else 'MISSING'}")
print(f"dv_etf_flow_proxy: {'OK' if 'dv_etf_flow_proxy' in df.columns else 'MISSING'}")
print(f"OI_Open_USD: {'OK' if 'OI_Open_USD' in df.columns else 'MISSING'}")
