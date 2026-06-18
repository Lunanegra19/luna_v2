import pandas as pd

# Load master probs
df_probs = pd.read_parquet('data/predictions/master_ensemble_probs.parquet')
print(f"Total timestamps in master probs: {len(df_probs)}")

# Load the actual returns (from raw trades to match timestamps, or from oos_trades)
# Since we just need returns, let's load any oos_trades for its return_pct
# Or we can just read raw trades and deduplicate by timestamp
df_raw = pd.read_parquet('data/predictions/unified_ensemble_trades_raw.parquet')
# Group by timestamp (entry_time) to get the actual return for that hour
df_ret = df_raw[['return_pct']].groupby(df_raw.index).first()

# Join
df = df_probs.join(df_ret, how='inner')
df['is_win'] = df['return_pct'] > 0

print(f"Matched timestamps with returns: {len(df)}")

# Test different thresholds for the ensemble mean probability
for thresh in [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]:
    df_long = df[df['prob_bull'] >= thresh]
    if len(df_long) > 0:
        wr = df_long['is_win'].mean() * 100
        print(f"Ensemble Prob Bull >= {thresh:.2f} | Trades: {len(df_long):<4} | Win Rate: {wr:.1f}%")
    else:
        print(f"Ensemble Prob Bull >= {thresh:.2f} | Trades: 0")
