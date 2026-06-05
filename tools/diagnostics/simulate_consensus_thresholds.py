import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
predictions_dir = _ROOT / "data" / "predictions"

def simulate():
    print("="*90)
    print("      DIAGNOSTIC SIMULATION: MULTI-SEED CONSENSUS THRESHOLDS      ")
    print("="*90)
    
    unified_path = predictions_dir / "unified_ensemble_trades_raw.parquet"
    if not unified_path.exists():
        print("ERROR: unified_ensemble_trades_raw.parquet not found.")
        return
        
    df_raw = pd.read_parquet(unified_path)
    
    # Calculate collision counts
    collisions = df_raw.index.value_counts()
    
    # Map counts to raw trades
    df_raw_counts = df_raw.copy()
    df_raw_counts['consensus_count'] = df_raw_counts.index.map(collisions)
    
    # Define thresholds to simulate
    thresholds = [1, 2, 3, 4, 5]
    
    for t in thresholds:
        print(f"\n=================== SIMULATION FOR CONSENSUS THRESHOLD >= {t} ===================")
        
        # Filter raw trades where at least t seeds agree
        df_filtered_raw = df_raw_counts[df_raw_counts['consensus_count'] >= t]
        
        if df_filtered_raw.empty:
            print(f"No trades with consensus >= {t}")
            continue
            
        # Group by timestamp to form the portfolio trade series
        df_port = df_filtered_raw.groupby(df_filtered_raw.index).agg({
            'return_pct': 'mean',
            'is_win': 'max',
            'direction': 'first',
            'wfb_window': 'first'
        }).sort_index()
        
        n_trades = len(df_port)
        if n_trades == 0:
            print("No portfolio trades after grouping.")
            continue
            
        # Calculate statistics
        wr = df_port['is_win'].mean() * 100
        mean_ret = df_port['return_pct'].mean()
        std_ret = df_port['return_pct'].std()
        
        # Normal (arithmetic) sum of returns
        normal_return = df_port['return_pct'].sum() * 100
        
        # Compound return
        comp_return = ((1 + df_port['return_pct']).prod() - 1) * 100
        
        # Max Drawdown (sequential trades)
        cum_series = (1 + df_port['return_pct']).cumprod()
        peaks = cum_series.cummax()
        drawdowns = (cum_series - peaks) / peaks
        max_dd = drawdowns.min() * 100 if not drawdowns.empty else 0.0
        
        # Annualized Sharpe (approximated on trades)
        sharpe = 0.0
        if std_ret > 1e-10:
            days = (df_port.index.max() - df_port.index.min()).days
            n_per_year = n_trades / (days / 365.25) if days > 0 else n_trades * 365.25
            sharpe = (mean_ret / std_ret) * (n_per_year ** 0.5)
            
        # Kelly Sizer & Position Sizing
        # Optimal Kelly f* = (p * R - q) / R
        # where p is win probability, R is win/loss ratio (avg win / avg loss), q = 1 - p
        pos_rets = df_port[df_port['return_pct'] > 0]['return_pct']
        neg_rets = df_port[df_port['return_pct'] < 0]['return_pct']
        
        avg_win = pos_rets.mean() if not pos_rets.empty else 0.0
        avg_loss = abs(neg_rets.mean()) if not neg_rets.empty else 0.0
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 1e-10 else 0.0
        
        p = wr / 100.0
        q = 1.0 - p
        
        if win_loss_ratio > 0:
            optimal_kelly = (p * win_loss_ratio - q) / win_loss_ratio
        else:
            optimal_kelly = 0.0
            
        print(f"Total Portfolio Trades: {n_trades}")
        print(f"Win Rate:               {wr:.2f}%")
        print(f"Normal return:          {normal_return:.4f}%")
        print(f"Compound return:        {comp_return:.4f}%")
        print(f"Max Drawdown:           {max_dd:.4f}%")
        print(f"Sharpe Ratio (Anual):   {sharpe:.4f}")
        print(f"Avg Win:                {avg_win*100:.5f}%")
        print(f"Avg Loss:               -{avg_loss*100:.5f}%")
        print(f"R:R Real Ratio:         {win_loss_ratio:.4f}")
        print(f"Optimal Kelly f*:       {optimal_kelly:.4f} (Fraction of equity)")
        
        # Leverage Simulation (at x5 and x10 optimal sizing)
        print("\n--- POSITION SIZER SIMULATION (LEVERAGE x5 AND x10) ---")
        for lev in [5, 10]:
            # Scale each return_pct by the leverage level and optimal kelly fraction (half Kelly of 0.5)
            half_kelly = max(0.0, optimal_kelly * 0.5)
            effective_mult = half_kelly * lev
            
            # Simulated return series with leverage
            lev_rets = df_port['return_pct'] * effective_mult
            
            # Compound return with leverage
            lev_comp_return = ((1 + lev_rets).prod() - 1) * 100
            
            # Max DD with leverage
            lev_cum_series = (1 + lev_rets).cumprod()
            lev_peaks = lev_cum_series.cummax()
            lev_drawdowns = (lev_cum_series - lev_peaks) / lev_peaks
            lev_max_dd = lev_drawdowns.min() * 100 if not lev_drawdowns.empty else 0.0
            
            # Sharpe is unchanged by linear scaling, but standard deviation scales
            lev_std_ret = lev_rets.std()
            
            print(f"Leverage x{lev:2d} (Half-Kelly mult={effective_mult:.3f}) | Comp Ret={lev_comp_return:8.4f}% | Max DD={lev_max_dd:8.4f}%")
            
        # Display breakdown by temporal window (W2 to W5)
        print("\n--- PERFORMANCE BY TEMPORAL WINDOW ---")
        window_stats = []
        for w, group in df_port.groupby('wfb_window'):
            n_w = len(group)
            wr_w = group['is_win'].mean() * 100 if n_w > 0 else 0.0
            comp_w = ((1 + group['return_pct']).prod() - 1) * 100
            cum_series_w = (1 + group['return_pct']).cumprod()
            peaks_w = cum_series_w.cummax()
            dd_w = (cum_series_w - peaks_w) / peaks_w
            max_dd_w = dd_w.min() * 100 if not dd_w.empty else 0.0
            
            window_stats.append({
                'Window': w,
                'Trades': n_w,
                'Win Rate (%)': wr_w,
                'Comp Ret (%)': comp_w,
                'Max DD (%)': max_dd_w
            })
        df_win = pd.DataFrame(window_stats)
        print(df_win.to_string(index=False))

if __name__ == "__main__":
    simulate()
