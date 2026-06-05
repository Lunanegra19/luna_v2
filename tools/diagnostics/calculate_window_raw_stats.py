import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
predictions_dir = _ROOT / "data" / "predictions"

def calculate_window_raw_stats():
    unified_path = predictions_dir / "unified_ensemble_trades_raw.parquet"
    if not unified_path.exists():
        print("ERROR: unified_ensemble_trades_raw.parquet not found.")
        return
        
    df_raw = pd.read_parquet(unified_path)
    collisions = df_raw.index.value_counts()
    df_raw_counts = df_raw.copy()
    df_raw_counts['consensus_count'] = df_raw_counts.index.map(collisions)
    df_filtered_raw = df_raw_counts[df_raw_counts['consensus_count'] >= 3]
    
    df_port = df_filtered_raw.groupby(df_filtered_raw.index).agg({
        'return_raw': 'mean',
        'is_win': 'max',
        'wfb_window': 'first'
    }).sort_index()
    
    windows = sorted(df_port['wfb_window'].unique())
    
    print(f"{'Ventana':<8} | {'N':<3} | {'WinRate':<7} | {'AvgWin':<7} | {'AvgLoss':<7} | {'R:R':<6} | {'RetRaw':<8} | {'RetComp':<8} | {'MaxDD':<7} | {'Kelly':<7} | {'Comp x10':<9} | {'MaxDD x10':<9} | {'Comp x20':<9} | {'MaxDD x20':<9}")
    print("-" * 145)
    
    for w in windows:
        group = df_port[df_port['wfb_window'] == w]
        n_trades = len(group)
        if n_trades == 0:
            continue
            
        wr = group['is_win'].mean() * 100
        rets = pd.Series(group['return_raw'].values)
        
        normal_return = rets.sum() * 100
        comp_return = ((1 + rets).prod() - 1) * 100
        
        # Max Drawdown
        cum_series = (1 + rets).cumprod()
        peaks = cum_series.cummax()
        drawdowns = (cum_series - peaks) / peaks
        max_dd = drawdowns.min() * 100 if not drawdowns.empty else 0.0
        
        # Kelly Sizer
        pos_rets = rets[rets > 0]
        neg_rets = rets[rets < 0]
        
        avg_win = pos_rets.mean() if not pos_rets.empty else 0.0
        avg_loss = abs(neg_rets.mean()) if not neg_rets.empty else 0.0
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 1e-10 else 0.0
        p = wr / 100.0
        q = 1.0 - p
        
        optimal_kelly = (p * win_loss_ratio - q) / win_loss_ratio if win_loss_ratio > 0 else 0.0
        
        # Simular Half-Kelly x10 y x20
        half_kelly = max(0.0, optimal_kelly * 0.5)
        
        # x10 leverage
        mult_10 = half_kelly * 10
        lev_rets_10 = rets * mult_10
        comp_10 = ((1 + lev_rets_10).prod() - 1) * 100
        cum_10 = (1 + lev_rets_10).cumprod()
        peaks_10 = cum_10.cummax()
        dd_10 = (cum_10 - peaks_10) / peaks_10
        max_dd_10 = dd_10.min() * 100 if not dd_10.empty else 0.0
        
        # x20 leverage
        mult_20 = half_kelly * 20
        lev_rets_20 = rets * mult_20
        comp_20 = ((1 + lev_rets_20).prod() - 1) * 100
        cum_20 = (1 + lev_rets_20).cumprod()
        peaks_20 = cum_20.cummax()
        dd_20 = (cum_20 - peaks_20) / peaks_20
        max_dd_20 = dd_20.min() * 100 if not dd_20.empty else 0.0
        
        print(f"{w:<8} | {n_trades:<3} | {wr:6.2f}% | {avg_win*100:6.3f}% | {avg_loss*100:6.3f}% | {win_loss_ratio:5.3f} | {normal_return:7.4f}% | {comp_return:7.4f}% | {max_dd:6.3f}% | {optimal_kelly:6.4f} | {comp_10:8.4f}% | {max_dd_10:8.4f}% | {comp_20:8.4f}% | {max_dd_20:8.4f}%")

if __name__ == "__main__":
    calculate_window_raw_stats()
