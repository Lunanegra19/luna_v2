import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
predictions_dir = _ROOT / "data" / "predictions"

def recalculate_real_metrics():
    print("="*95)
    print("   LUNA V2 - AUDITORÍA DE ESCALA MATEMÁTICA Y EXPOSICIÓN REAL (KELLY & LEVERAGE)   ")
    print("="*95)
    
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
    
    # Portafolio unificado consensuado:
    # Agrupamos por timestamp y promediamos return_raw (retorno del activo neto de comisiones)
    # y return_pct (retorno ya escalado por el sizer individual de ~3.6% de cada semilla)
    df_port = df_filtered_raw.groupby(df_filtered_raw.index).agg({
        'return_raw': 'mean',
        'return_pct': 'mean',
        'is_win': 'max',
        'wfb_window': 'first'
    }).sort_index()
    
    n_trades = len(df_port)
    returns_raw = df_port['return_raw'].values
    returns_pct = df_port['return_pct'].values
    
    print(f"Trades consensuados cargados: {n_trades} trades únicos (Consensus >= 3).")
    print(f"Estadísticas de Retornos Brutos del Activo (return_raw) neto de comisiones:")
    print(f"  - Media por trade:  {returns_raw.mean()*100:.4f}%")
    print(f"  - Desviación Estd:   {returns_raw.std()*100:.4f}%")
    print(f"  - Peor pérdida:     {returns_raw.min()*100:.4f}%")
    print(f"  - Mejor ganancia:    {returns_raw.max()*100:.4f}%")
    
    print(f"\nEstadísticas de Retornos de las Semillas (return_pct) con Sizer de Semilla (~3.6% de capital):")
    print(f"  - Media por trade:  {returns_pct.mean()*100:.4f}%")
    print(f"  - Desviación Estd:   {returns_pct.std()*100:.4f}%")
    
    # --- CÁLCULO DEL KELLY REAL SOBRE EL ACTIVO SUBYACENTE (return_raw) ---
    wr_raw = df_port['is_win'].mean()
    pos_rets = df_port[df_port['return_raw'] > 0]['return_raw']
    neg_rets = df_port[df_port['return_raw'] < 0]['return_raw']
    avg_win = pos_rets.mean() if not pos_rets.empty else 0.0
    avg_loss = abs(neg_rets.mean()) if not neg_rets.empty else 0.0
    
    win_loss_ratio = avg_win / avg_loss if avg_loss > 1e-10 else 0.0
    p = wr_raw
    q = 1.0 - p
    
    # Kelly óptimo sobre el activo directo
    optimal_kelly_raw = (p * win_loss_ratio - q) / win_loss_ratio if win_loss_ratio > 0 else 0.0
    print(f"\nCálculo del Kelly Óptimo Real sobre el Activo Subyacente (return_raw):")
    print(f"  - Win Rate:         {wr_raw*100:.2f}%")
    print(f"  - Ratio R:R Real:   {win_loss_ratio:.4f}")
    print(f"  - Kelly f* Óptimo:  {optimal_kelly_raw:.4f} ({optimal_kelly_raw*100:.2f}% de la equity)")
    
    # --- COMPARACIÓN DE ESCALAS Y SIMULACIONES DE RENTABILIDAD ---
    # Si aplicamos Half-Kelly (14.56% de la equity por trade)
    half_kelly = optimal_kelly_raw * 0.5 # 14.56%
    
    print("\n" + "="*95)
    print("   SIMULACIÓN CORRECTA DE RENTABILIDAD DE LA CUENTA (Apalancamiento de Plataforma sobre Capital Real)")
    print("   Lógica: Exposición por trade = Fracción de Kelly * Apalancamiento de Plataforma")
    print("   Retorno por trade de la cuenta = return_raw * (Fracción de Kelly * Apalancamiento)")
    print("="*95)
    print(f"{'Escenario':<30} | {'Exposición Real':<15} | {'Retorno Normal':<16} | {'Retorno Compuesto':<18} | {'Max Drawdown':<12}")
    print("-"*95)
    
    # 1. Sin apalancamiento, con Sizer individual de semilla (Línea Base anterior)
    cum_pct = (1 + returns_pct).cumprod()
    ret_pct_comp = (cum_pct[-1] - 1) * 100 if len(cum_pct) > 0 else 0.0
    peaks_pct = pd.Series(cum_pct).cummax()
    dd_pct = (pd.Series(cum_pct) - peaks_pct) / peaks_pct
    max_dd_pct = dd_pct.min() * 100 if not dd_pct.empty else 0.0
    print(f"{'Base Semillas (Sizer 3.6% x1)':<30} | {'0.036x (3.6%)':<15} | {returns_pct.sum()*100:14.4f}% | {ret_pct_comp:16.4f}% | {max_dd_pct:10.4f}%")
    
    # Simulamos el escalamiento sobre return_raw
    scenarios = [
        {"name": "Half-Kelly (14.56%) x1 Lever", "fraction": half_kelly, "lever": 1},
        {"name": "Half-Kelly (14.56%) x2 Lever", "fraction": half_kelly, "lever": 2},
        {"name": "Half-Kelly (14.56%) x5 Lever", "fraction": half_kelly, "lever": 5},
        {"name": "Half-Kelly (14.56%) x10 Lever", "fraction": half_kelly, "lever": 10},
        {"name": "Half-Kelly (14.56%) x20 Lever", "fraction": half_kelly, "lever": 20},
        {"name": "Full-Kelly (29.12%) x10 Lever", "fraction": optimal_kelly_raw, "lever": 10},
        {"name": "Full-Kelly (29.12%) x20 Lever", "fraction": optimal_kelly_raw, "lever": 20},
    ]
    
    for sc in scenarios:
        frac = sc["fraction"]
        lev = sc["lever"]
        total_exp = frac * lev # Exposición real sobre el activo
        
        # Retornos de la cuenta
        account_rets = returns_raw * total_exp
        cum_series = (1 + account_rets).cumprod()
        comp_return = (cum_series[-1] - 1) * 100 if len(cum_series) > 0 else 0.0
        
        # Max Drawdown
        peaks = pd.Series(cum_series).cummax()
        drawdowns = (pd.Series(cum_series) - peaks) / peaks
        max_dd = drawdowns.min() * 100 if not drawdowns.empty else 0.0
        
        print(f"{sc['name']:<30} | {total_exp:13.3f}x | {account_rets.sum()*100:14.4f}% | {comp_return:16.4f}% | {max_dd:10.4f}%")
        
    print("="*95)
    print("\n>>> DETECCIÓN DEL ERROR DE ESCALA EN EL SCRIPT ANTERIOR:")
    print("El script 'simulate_window_stats_premium.py' multiplicaba la columna 'return_pct' por (Half-Kelly * Leverage).")
    print("Como 'return_pct' YA tenía incorporada la fracción de Kelly de la semilla (~3.6%), el script calculaba:")
    print("  Retorno = return_raw * 3.67% * 14.56% * Leverage = return_raw * 0.0053 * Leverage.")
    print("Para Leverage x20, el multiplicador efectivo sobre return_raw era de solo 0.1068x (exposición del 10.68%).")
    print("Es decir, ¡se simulaba una cuenta que operaba con el 10.68% de su capital real por trade en lugar del 291%!")
    print("Por eso el retorno total reportado a x20 era de solo un ~4.44% en lugar del ~120% real apalancado.")
    print("="*95)

if __name__ == "__main__":
    recalculate_real_metrics()
