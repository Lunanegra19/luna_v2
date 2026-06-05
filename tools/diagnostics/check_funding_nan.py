"""Check funding_ema_3 and FundingRate_EMA3 in features_live.parquet"""
import pandas as pd
import sys

df = pd.read_parquet('/root/luna_v2/data/features/features_live.parquet')
print(f"Last index: {df.index[-1]}")
print(f"Total cols: {len(df.columns)}")

for col in ['funding_ema_3', 'FundingRate_EMA3', 'FundingRate', 'dv_funding_rate']:
    if col in df.columns:
        s = df[col]
        nan_pct = s.isna().mean() * 100
        last_val = s.dropna().iloc[-1] if not s.dropna().empty else 'ALL NaN'
        print(f"  {col}: NaN={nan_pct:.1f}% | last_val={last_val}")
    else:
        print(f"  {col}: MISSING from parquet")

print("\n=== dropna(all) survivors ===")
# simulate dropna all
df_test = df.dropna(axis=1, how='all')
for col in ['funding_ema_3', 'FundingRate_EMA3']:
    print(f"  {col} survives dropna(all): {col in df_test.columns}")
