import pandas as pd
import numpy as np

def simulate_w4_candidates():
    # 1. Load raw probs
    probs_path = "g:/Mi unidad/ia/luna_v2/data/reports/wfb/oos_raw_probs_W4_seed42.parquet"
    print(f"Loading raw probs from {probs_path}...")
    df_probs = pd.read_parquet(probs_path)
    print(f"Loaded raw probs: shape={df_probs.shape}")
    
    # 2. Load OHLCV raw data
    ohlcv_path = "g:/Mi unidad/ia/luna_v2/data/raw/ohlcv/ohlcv_raw.parquet"
    print(f"Loading OHLCV from {ohlcv_path}...")
    df_ohlcv = pd.read_parquet(ohlcv_path)
    print(f"Loaded OHLCV: shape={df_ohlcv.shape}")
    
    # Ensure index is datetime with timezone
    if not isinstance(df_probs.index, pd.DatetimeIndex):
        df_probs.index = pd.to_datetime(df_probs.index)
    if df_probs.index.tz is None:
        df_probs.index = df_probs.index.tz_localize('UTC')
        
    if not isinstance(df_ohlcv.index, pd.DatetimeIndex):
        df_ohlcv.index = pd.to_datetime(df_ohlcv.index)
    if df_ohlcv.index.tz is None:
        df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
        
    # Join probs with close price
    df_joined = df_probs.join(df_ohlcv[['open', 'high', 'low', 'close']], how='inner')
    print(f"Joined shape: {df_joined.shape}")
    
    # Calculate ATR (rolling 14 hours) on df_ohlcv, then join
    df_ohlcv['high_low'] = df_ohlcv['high'] - df_ohlcv['low']
    df_ohlcv['high_close'] = (df_ohlcv['high'] - df_ohlcv['close'].shift()).abs()
    df_ohlcv['low_close'] = (df_ohlcv['low'] - df_ohlcv['close'].shift()).abs()
    df_ohlcv['tr'] = df_ohlcv[['high_low', 'high_close', 'low_close']].max(axis=1)
    df_ohlcv['atr'] = df_ohlcv['tr'].rolling(14).mean()
    
    df_joined = df_joined.join(df_ohlcv['atr'], how='inner')
    
    # Let's inspect the candidate signals
    # In W4, what was the signal threshold for XGB?
    # Let's print out rows where prob_bull or prob_bear shows active values
    print("\nNon-zero prob_bull rows:")
    bull_candidates = df_joined[df_joined['prob_bull'] > 0.0]
    print(f"Count of non-zero prob_bull: {len(bull_candidates)}")
    print(bull_candidates.head(10))
    
    # Let's write a standard TBM simulator for these candidate timestamps
    # For each candidate, we simulate a trade:
    # - Start: timestamp t
    # - Entry Price: close at t
    # - ATR at t: atr_t
    # - Horizontal parameters: tp_mult = 1.8, sl_mult = 1.5, horizon = 72 hours
    # - Let's simulate with and without linear decay
    
    results = []
    
    # We will simulate for all timestamps where prob_bull > 0
    trade_idx = 0
    for ts, row in bull_candidates.iterrows():
        entry_price = row['close']
        atr_val = row['atr']
        prob_val = row['prob_bull']
        
        # Determine the future path
        future_idx = df_ohlcv.index.get_indexer([ts])[0]
        if future_idx == -1:
            continue
            
        future_bars = df_ohlcv.iloc[future_idx:future_idx + 73]
        if len(future_bars) < 2:
            continue
            
        # TBM thresholds
        tp_barrier = entry_price + (1.8 * atr_val)
        sl_barrier = entry_price - (1.5 * atr_val)
        
        # Simulate with static TBM (No Decay)
        pnl_static = None
        exit_ts_static = None
        reason_static = "horizon"
        
        # Simulate with Linear Decay (TP decays to 25% of its initial ATR mult at 72h)
        # i.e., tp_mult decays from 1.8 to 0.45
        pnl_decay = None
        exit_ts_decay = None
        reason_decay = "horizon"
        
        for i in range(1, len(future_bars)):
            bar = future_bars.iloc[i]
            cur_ts = future_bars.index[i]
            high_p = bar['high']
            low_p = bar['low']
            close_p = bar['close']
            
            # 1. Static TBM
            if pnl_static is None:
                if high_p >= tp_barrier:
                    pnl_static = (tp_barrier - entry_price) / entry_price
                    exit_ts_static = cur_ts
                    reason_static = "tp"
                elif low_p <= sl_barrier:
                    pnl_static = (sl_barrier - entry_price) / entry_price
                    exit_ts_static = cur_ts
                    reason_static = "sl"
            
            # 2. Linear Decay TBM
            if pnl_decay is None:
                elapsed_hours = i
                # Decay factor: decays linearly from 1.0 down to 0.25 at 72 hours
                decay_factor = 1.0 - (0.75 * (elapsed_hours / 72.0))
                decay_factor = max(0.25, decay_factor)
                current_tp_barrier = entry_price + (1.8 * atr_val * decay_factor)
                
                if high_p >= current_tp_barrier:
                    pnl_decay = (current_tp_barrier - entry_price) / entry_price
                    exit_ts_decay = cur_ts
                    reason_decay = "tp"
                elif low_p <= sl_barrier:
                    pnl_decay = (sl_barrier - entry_price) / entry_price
                    exit_ts_decay = cur_ts
                    reason_decay = "sl"
                    
        # If horizon reached without hitting barriers
        if pnl_static is None:
            final_close = future_bars.iloc[-1]['close']
            pnl_static = (final_close - entry_price) / entry_price
            exit_ts_static = future_bars.index[-1]
            
        if pnl_decay is None:
            final_close = future_bars.iloc[-1]['close']
            pnl_decay = (final_close - entry_price) / entry_price
            exit_ts_decay = future_bars.index[-1]
            
        results.append({
            'timestamp': ts,
            'prob': prob_val,
            'pnl_static': pnl_static,
            'reason_static': reason_static,
            'pnl_decay': pnl_decay,
            'reason_decay': reason_decay
        })
        
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        print("\n=== SUMMARY OF SIMULATED TRADES ===")
        print(f"Total simulated: {len(df_res)}")
        
        print("\n--- STATIC TBM (NO DECAY) ---")
        win_rate_static = (df_res['pnl_static'] > 0).mean()
        mean_pnl_static = df_res['pnl_static'].mean()
        sum_pnl_static = df_res['pnl_static'].sum()
        tp_count_static = (df_res['reason_static'] == "tp").sum()
        sl_count_static = (df_res['reason_static'] == "sl").sum()
        horizon_count_static = (df_res['reason_static'] == "horizon").sum()
        print(f"Win Rate: {win_rate_static*100:.2f}%")
        print(f"Mean PnL: {mean_pnl_static*100:.4f}%")
        print(f"Sum PnL: {sum_pnl_static*100:.4f}%")
        print(f"Exits: TP={tp_count_static}, SL={sl_count_static}, Horizon={horizon_count_static}")
        
        print("\n--- LINEAR DECAY TBM ---")
        win_rate_decay = (df_res['pnl_decay'] > 0).mean()
        mean_pnl_decay = df_res['pnl_decay'].mean()
        sum_pnl_decay = df_res['pnl_decay'].sum()
        tp_count_decay = (df_res['reason_decay'] == "tp").sum()
        sl_count_decay = (df_res['reason_decay'] == "sl").sum()
        horizon_count_decay = (df_res['reason_decay'] == "horizon").sum()
        print(f"Win Rate: {win_rate_decay*100:.2f}%")
        print(f"Mean PnL: {mean_pnl_decay*100:.4f}%")
        print(f"Sum PnL: {sum_pnl_decay*100:.4f}%")
        print(f"Exits: TP={tp_count_decay}, SL={sl_count_decay}, Horizon={horizon_count_decay}")
        
        # Save to csv for inspection
        df_res.to_csv("g:/Mi unidad/ia/luna_v2/tools/dumps/w4_signal_simulation.csv", index=False)
        print("\nDetailed trade log saved to tools/dumps/w4_signal_simulation.csv")
        
        # Let's print out a few trades
        print("\nFirst 10 trade simulations:")
        print(df_res.head(10))
    else:
        print("\nNo trade simulations generated.")

if __name__ == "__main__":
    simulate_w4_candidates()
