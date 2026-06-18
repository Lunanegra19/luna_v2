import pandas as pd

# Load raw trades
df_raw = pd.read_parquet('data/predictions/unified_ensemble_trades_raw.parquet')

# Convert index if necessary
if not isinstance(df_raw.index, pd.DatetimeIndex):
    if 'entry_time' in df_raw.columns:
        df_raw['entry_time'] = pd.to_datetime(df_raw['entry_time'])
        df_raw.set_index('entry_time', inplace=True)
    else:
        df_raw.index = pd.to_datetime(df_raw.index)

df_raw = df_raw.sort_index()

# FILTER ONLY APPROVED SEEDS
approved_seeds = [1337, 2025, 42]
df_raw = df_raw[df_raw['seed'].isin(approved_seeds)]

# Group by entry_time
consensus_series = df_raw.groupby(df_raw.index)['seed'].nunique()
# IMPORTANTE: Usar el is_win del dataframe original, no return_pct > 0
is_win_series = df_raw.groupby(df_raw.index)['is_win'].first()

df_agg = pd.DataFrame({
    'consensus': consensus_series,
    'is_win': is_win_series
})

print(f"Total Unique Trades (Solo 3 seeds, No embargo): {len(df_agg)}")
print(f"Win Rate Base (No embargo, Consensus >= 1): {df_agg['is_win'].mean()*100:.1f}%")

def simulate_ensemble(consensus_threshold, embargo_hours):
    df_filtered = df_agg[df_agg['consensus'] >= consensus_threshold].copy()
    
    selected_indices = []
    last_time = None
    
    for ts, row in df_filtered.iterrows():
        if last_time is None:
            selected_indices.append(ts)
            last_time = ts
        else:
            delta_h = (ts - last_time).total_seconds() / 3600.0
            if delta_h >= embargo_hours:
                selected_indices.append(ts)
                last_time = ts
                
    df_final = df_filtered.loc[selected_indices]
    if len(df_final) == 0:
        return 0, 0.0
    
    wr = df_final['is_win'].mean() * 100
    return len(df_final), wr

print("="*50)
print("ENSEMBLE GRID SEARCH (SOLO SEEDS APROBADAS)")
print("="*50)

embargos = [0, 24, 48, 72, 96, 120]
consensus_levels = [1, 2, 3]

for c in consensus_levels:
    print(f"\n--- CONSENSUS >= {c} SEEDS ---")
    for e in embargos:
        trades, wr = simulate_ensemble(c, e)
        status = "✅ OK" if wr >= 50.0 and trades >= 30 else "❌ REJECT"
        print(f"  Embargo {e:3}H | Trades: {trades:<3} | WR: {wr:5.1f}% | {status}")
