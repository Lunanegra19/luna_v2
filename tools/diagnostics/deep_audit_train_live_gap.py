"""Deep audit of train vs live mismatch for critical features"""
import pandas as pd

train = pd.read_parquet('/root/luna_v2/data/features/features_train.parquet')
live = pd.read_parquet('/root/luna_v2/data/features/features_live.parquet')
train_cols = set(train.columns)
live_cols = set(live.columns)

# Features IN TRAIN but not LIVE
in_train_not_live = sorted(train_cols - live_cols)
print(f"=== IN TRAIN but NOT LIVE ({len(in_train_not_live)} features) ===")
for f in in_train_not_live:
    print(f"  {f}")

print(f"\n=== IN LIVE but NOT TRAIN ({len(live_cols - train_cols)} features) ===")
for f in sorted(live_cols - train_cols)[:20]:
    print(f"  {f}")

# Check FundingRate columns in both
print("\n=== FundingRate cols ===")
print("Train:", sorted([c for c in train_cols if 'Funding' in c or 'funding' in c]))
print("Live: ", sorted([c for c in live_cols if 'Funding' in c or 'funding' in c]))

# Check OI columns in both
print("\n=== OI cols ===")
print("Train:", sorted([c for c in train_cols if c.startswith('OI_')]))
print("Live: ", sorted([c for c in live_cols if c.startswith('OI_')]))

# Check ETF cols in both
print("\n=== ETF cols ===")
print("Train:", sorted([c for c in train_cols if 'ETF' in c or 'etf' in c]))
print("Live: ", sorted([c for c in live_cols if 'ETF' in c or 'etf' in c]))

# Check OHLCV-related columns
print("\n=== OHLCV-related cols in live ===")
ohlcv_live = [c for c in live_cols if any(s in c.lower() for s in ['return', 'atr', 'volatil', 'close', 'open', 'high', 'low', 'volume'])]
print(sorted(ohlcv_live)[:30])
