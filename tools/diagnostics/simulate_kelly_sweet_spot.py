import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
predictions_dir = _ROOT / "data" / "predictions"

def simulate_sweet_spot():
    unified_path = predictions_dir / "unified_ensemble_trades_raw.parquet"
    if not unified_path.exists():
        print("ERROR: unified_ensemble_trades_raw.parquet not found.")
        return
        
    df_raw = pd.read_parquet(unified_path)
    collisions = df_raw.index.value_counts()
    df_raw_counts = df_raw.copy()
    df_raw_counts['consensus_count'] = df_raw_counts.index.map(collisions)
    df_filtered_raw = df_raw_counts[df_raw_counts['consensus_count'] >= 3]
    
    # Portafolio unificado consensuado
    df_port = df_filtered_raw.groupby(df_filtered_raw.index).agg({
        'return_pct': 'mean',
        'is_win': 'max',
        'wfb_window': 'first'
    }).sort_index()
    
    returns = df_port['return_pct'].values
    n_trades = len(returns)
    
    print(f"Loaded {n_trades} consensus trades for sweet spot simulation.\n")
    print(f"{'Exposición Nominal Total':<26} | {'Retorno Compuesto':<18} | {'Max Drawdown':<14} | {'Calmar Ratio':<14} | {'Sharpe Ratio':<12}")
    print("-" * 95)
    
    # Probar diferentes multiplicadores de exposición nominal (Margen % de Kelly * Apalancamiento Plataforma)
    # Por ejemplo, si f* = 29.12%, entonces un multiplicador total de 1.0 es Full-Kelly, 0.5 es Half-Kelly, etc.
    # Pero el usuario pregunta: mix de apalancamiento en la plataforma (ej. x5, x10) + % del Kelly (ej. Half-Kelly, Quarter-Kelly)
    # Exposición Nominal Total = (Fracción de Kelly) * (Apalancamiento de Plataforma)
    # Si Kelly = 29.12%, y operamos Half-Kelly (14.56%) con apalancamiento de plataforma x10: Exposición nominal = 1.456 veces la equity.
    
    # Probaremos diferentes Exposiciones Nominales Totales directas
    results = []
    exposures = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0]
    
    for exp in exposures:
        # Retornos de la cuenta
        account_rets = returns * exp
        cum_series = (1 + account_rets).cumprod()
        comp_return = (cum_series[-1] - 1) * 100 if len(cum_series) > 0 else 0.0
        
        # Max Drawdown
        peaks = pd.Series(cum_series).cummax()
        drawdowns = (pd.Series(cum_series) - peaks) / peaks
        max_dd = drawdowns.min() * 100 if not drawdowns.empty else 0.0
        
        # Sharpe Ratio Anualizado
        std_ret = account_rets.std()
        mean_ret = account_rets.mean()
        sharpe = 0.0
        if std_ret > 1e-10:
            days = (df_port.index.max() - df_port.index.min()).days
            n_per_year = n_trades / (days / 365.25) if days > 0 else n_trades * 365.25
            sharpe = (mean_ret / std_ret) * (n_per_year ** 0.5)
            
        calmar = comp_return / abs(max_dd) if abs(max_dd) > 1e-10 else 100.0
        
        results.append({
            "exp": exp,
            "ret": comp_return,
            "dd": max_dd,
            "calmar": calmar,
            "sharpe": sharpe
        })
        
        # Encontrar equivalencia en mix (Fracción de Kelly * Apalancamiento plataforma = exp)
        # Ejemplo 1: Half-Kelly (14.56%) * x10 Leverage = exp 1.456
        # Ejemplo 2: Quarter-Kelly (7.28%) * x10 Leverage = exp 0.728
        # Ejemplo 3: Full-Kelly (29.12%) * x20 Leverage = exp 5.824
        
        print(f"Exposición Nominal {exp:4.1f}x | {comp_return:16.4f}% | {max_dd:12.4f}% | {calmar:12.4f} | {sharpe:10.4f}")

if __name__ == "__main__":
    simulate_sweet_spot()
