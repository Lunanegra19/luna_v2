import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
predictions_dir = _ROOT / "data" / "predictions"

def calculate_window_stats():
    print("="*90)
    print("   LUNA V2 - CALCULADORA PREMIUM DE ESTADÍSTICAS POR VENTANA (RULE-WINDOWSTATS)   ")
    print("="*90)
    
    unified_path = predictions_dir / "unified_ensemble_trades_raw.parquet"
    if not unified_path.exists():
        print("ERROR: unified_ensemble_trades_raw.parquet not found.")
        return
        
    df_raw = pd.read_parquet(unified_path)
    
    # Calcular colisiones de timestamps
    collisions = df_raw.index.value_counts()
    df_raw_counts = df_raw.copy()
    df_raw_counts['consensus_count'] = df_raw_counts.index.map(collisions)
    
    # Filtrar por Consensus Gate >= 3
    df_filtered_raw = df_raw_counts[df_raw_counts['consensus_count'] >= 3]
    
    # Agrupar por timestamp para formar el portafolio consolidado
    df_port = df_filtered_raw.groupby(df_filtered_raw.index).agg({
        'return_pct': 'mean',
        'is_win': 'max',
        'direction': 'first',
        'wfb_window': 'first'
    }).sort_index()
    
    # Desglosar por ventana
    windows = sorted(df_port['wfb_window'].unique())
    
    for w in windows:
        group = df_port[df_port['wfb_window'] == w]
        n_trades = len(group)
        if n_trades == 0:
            continue
            
        wr = group['is_win'].mean() * 100
        mean_ret = group['return_pct'].mean()
        std_ret = group['return_pct'].std()
        
        normal_return = group['return_pct'].sum() * 100
        comp_return = ((1 + group['return_pct']).prod() - 1) * 100
        
        # Max Drawdown
        cum_series = (1 + group['return_pct']).cumprod()
        peaks = cum_series.cummax()
        drawdowns = (cum_series - peaks) / peaks
        max_dd = drawdowns.min() * 100 if not drawdowns.empty else 0.0
        
        # Sharpe Anualizado
        sharpe = 0.0
        if std_ret > 1e-10:
            days = (group.index.max() - group.index.min()).days
            n_per_year = n_trades / (days / 365.25) if days > 0 else n_trades * 365.25
            sharpe = (mean_ret / std_ret) * (n_per_year ** 0.5)
            
        # Kelly Sizer
        pos_rets = group[group['return_pct'] > 0]['return_pct']
        neg_rets = group[group['return_pct'] < 0]['return_pct']
        
        avg_win = pos_rets.mean() if not pos_rets.empty else 0.0
        avg_loss = abs(neg_rets.mean()) if not neg_rets.empty else 0.0
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 1e-10 else 0.0
        p = wr / 100.0
        q = 1.0 - p
        
        if win_loss_ratio > 0:
            optimal_kelly = (p * win_loss_ratio - q) / win_loss_ratio
        else:
            optimal_kelly = 0.0
            
        print(f"\n=================== ESTADÍSTICAS DETALLADAS VENTANA {w} (N = {n_trades}) ===================")
        print(f"Régimen y Período:      {'Transición Macro' if w == 'W1' else 'Lateralidad / Bull Grinds' if w == 'W2' else 'Bull Trend Estructurada' if w == 'W3' else 'Bearish / Q4 Crash' if w == 'W4' else 'Ciega post-ATH Volátil'}")
        print(f"Win Rate:               {wr:.2f}%")
        print(f"Retorno Normal:         {normal_return:.4f}%")
        print(f"Retorno Compuesto:      {comp_return:.4f}%")
        print(f"Max Drawdown:           {max_dd:.4f}%")
        print(f"Sharpe Ratio (Anual):   {sharpe:.4f}")
        print(f"Avg Win / Loss:         {avg_win*100:.5f}% / -{avg_loss*100:.5f}%")
        print(f"Ratio R:R Real:         {win_loss_ratio:.4f}")
        print(f"Optimal Kelly f*:       {optimal_kelly:.4f} ({optimal_kelly*100:.2f}% de la equity)")
        
        # Simular Position Sizing (Half-Kelly x5 y x10)
        half_kelly = max(0.0, optimal_kelly * 0.5)
        for lev in [5, 10]:
            effective_mult = half_kelly * lev
            lev_rets = group['return_pct'] * effective_mult
            lev_comp_return = ((1 + lev_rets).prod() - 1) * 100
            
            lev_cum_series = (1 + lev_rets).cumprod()
            lev_peaks = lev_cum_series.cummax()
            lev_drawdowns = (lev_cum_series - lev_peaks) / lev_peaks
            lev_max_dd = lev_drawdowns.min() * 100 if not lev_drawdowns.empty else 0.0
            
            print(f"  Apalancamiento x{lev:2d} (Half-Kelly mult={effective_mult:.3f}) | Comp Ret={lev_comp_return:8.4f}% | Max DD={lev_max_dd:8.4f}%")

if __name__ == "__main__":
    calculate_window_stats()
