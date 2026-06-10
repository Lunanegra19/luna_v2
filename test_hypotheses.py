import pandas as pd
from pathlib import Path
import json

predictions_dir = Path('C:/Users/Usuario/Desktop/ia/luna_v2/data/predictions')
raw_trades_path = predictions_dir / 'unified_ensemble_trades_raw.parquet'
master_probs_path = predictions_dir / 'master_ensemble_probs.parquet'

print('--- HYPOTHESIS 1: SIN METALABELER ---')
if master_probs_path.exists():
    df_probs = pd.read_parquet(master_probs_path)
    print(f'Master probs columns: {list(df_probs.columns)}')
    if 'prob_xgb' in df_probs.columns or 'xgb_prob' in df_probs.columns:
        prob_col = 'prob_xgb' if 'prob_xgb' in df_probs.columns else 'xgb_prob'
        # Simulate trades based on raw XGB prob > threshold
        threshold = 0.50
        df_xgb_trades = df_probs[df_probs[prob_col] > threshold].copy()
        print(f'Raw XGB Trades (Prob > {threshold}): {len(df_xgb_trades)}')
        if 'return_pct' in df_xgb_trades.columns:
            # Adjust cost if return_pct already has 0.15% deducted.
            # Usually return_pct is after cost. Let's assume it has 0.15%.
            # We want to see it at 0.25% (-0.10%).
            wr_015 = (df_xgb_trades['return_pct'] > 0).mean()
            wr_025 = ((df_xgb_trades['return_pct'] - 0.0010) > 0).mean()
            ret_015 = df_xgb_trades['return_pct'].mean()
            ret_025 = (df_xgb_trades['return_pct'] - 0.0010).mean()
            print(f'Raw XGB WR (0.15% fee): {wr_015:.2%}, Avg Ret: {ret_015:.4%}')
            print(f'Raw XGB WR (0.25% fee): {wr_025:.2%}, Avg Ret: {ret_025:.4%}')
else:
    print('No master_ensemble_probs.parquet found.')

print('\n--- HYPOTHESIS 2: EMBARGOS Y COSTOS EN ENSEMBLE ---')
if raw_trades_path.exists():
    df_all_trades = pd.read_parquet(raw_trades_path)
    df_all_trades['consensus_bucket'] = df_all_trades.index.floor('2h')
    
    # Calculate consensus count
    bucket_unique_seeds = df_all_trades.groupby('consensus_bucket')['seed'].nunique()
    df_all_trades['consensus_count'] = df_all_trades['consensus_bucket'].map(bucket_unique_seeds)
    
    # Filter consensus >= 2
    df_filtered = df_all_trades[df_all_trades['consensus_count'] >= 2].copy()
    
    agg_dict = {
        'return_pct': 'mean',
        'direction': 'first',
        'wfb_window': 'first'
    }
    df_port = df_filtered.groupby('consensus_bucket').agg(agg_dict).sort_index()
    
    print(f'Total consensus trades before embargo: {len(df_port)}')
    
    for emb_h in [0, 24, 48, 72, 96, 168]:
        for fee_extra in [0.0, 0.0010]: # 0.0 means 0.15% total, 0.0010 means 0.25% total
            fee_label = '0.15%' if fee_extra == 0.0 else '0.25%'
            df_port['ret_adjusted'] = df_port['return_pct'] - fee_extra
            df_port['is_win'] = (df_port['ret_adjusted'] > 0).astype(float)
            
            selected_indices = []
            last_time = None
            for ts, row in df_port.iterrows():
                if last_time is None:
                    selected_indices.append(ts)
                    last_time = ts
                else:
                    delta_h = (ts - last_time).total_seconds() / 3600.0
                    if delta_h >= emb_h:
                        selected_indices.append(ts)
                        last_time = ts
            
            df_final = df_port.loc[selected_indices]
            n_trades = len(df_final)
            wr = df_final['is_win'].mean() if n_trades > 0 else 0
            ret_mean = df_final['ret_adjusted'].mean() if n_trades > 0 else 0
            print(f'Embargo: {emb_h}H | Fee: {fee_label} | Trades: {n_trades} | WR: {wr:.2%} | Avg Ret: {ret_mean*100:.4f}%')
else:
    print('No unified_ensemble_trades_raw.parquet found.')

