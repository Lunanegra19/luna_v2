"""Check missing features against training parquet to determine if they're real or aspirational"""
import pandas as pd

try:
    train = pd.read_parquet('/root/luna_v2/data/features/features_train.parquet')
    train_cols = set(train.columns)
    print(f"features_train.parquet: {len(train_cols)} columns")
except Exception as e:
    print(f"Can't load features_train: {e}")
    train_cols = set()

live = pd.read_parquet('/root/luna_v2/data/features/features_live.parquet')
live_cols = set(live.columns)

missing_groups = {
    "OHLCV": ["returns_1h", "returns_24h", "atr_14h", "volatility_24h"],
    "FundingRate": ["FundingRate_EMA3", "FundingRate_Pct90d"],
    "OpenInterest": ["OI_Open_USD", "OI_High_USD", "OI_Low_USD"],
    "ETFFlows": ["ETF_Flow_Proxy", "dv_etf_flow_proxy"],
}

print("\n=== MISSING FEATURES: Train vs Live ===")
for group, features in missing_groups.items():
    print(f"\n{group}:")
    for f in features:
        in_train = f in train_cols if train_cols else None
        in_live = f in live_cols
        status = "BOTH MISSING" if not in_train and not in_live else \
                 "IN TRAIN, MISSING LIVE (real bug!)" if in_train and not in_live else \
                 "IN LIVE only (dashboard bug?)" if not in_train and in_live else "BOTH PRESENT"
        print(f"  {'TRAIN?' if in_train is None else 'TRAIN:'+str(in_train):15} LIVE:{str(in_live):5} | {f} → {status}")

# Also check similar column names that might exist
print("\n=== Closest matches in live parquet ===")
all_search = ["return", "atr", "volatility", "FundingRate_EMA", "FundingRate_Pct", 
              "OI_Open", "OI_High", "OI_Low", "ETF_Flow", "dv_etf"]
for s in all_search:
    matches = [c for c in live_cols if s in c]
    if matches:
        print(f"  {s}: {matches[:5]}")
