"""
tools/diagnostics/audit_seeds_run.py
====================================
Script de diagnóstico para realizar auditoría institucional profunda de los resultados
de las 5 semillas (42, 100, 777, 1337, 2025) y sus respectivas ventanas (W1 a W5).
Analiza retornos, ratios R:R, Max Drawdown, duraciones e inestabilidades de cada semilla.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Alinear path del proyecto
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def analyze_trades():
    print("="*90)
    print("      LUNA V2 - AUDITORÍA INSTITUCIONAL PROFUNDA DE SEMILLAS (WFB ENSEMBLE)      ")
    print("="*90)
    
    wfb_dir = _ROOT / "data" / "reports" / "wfb"
    if not wfb_dir.exists():
        print(f"ERROR: Directorio no encontrado {wfb_dir}")
        return
        
    trade_files = list(wfb_dir.glob("oos_trades_W*_seed*.parquet"))
    if not trade_files:
        print("ERROR: No se encontraron archivos de trades en data/reports/wfb")
        return
        
    print(f"Detectados {len(trade_files)} archivos de trades. Procesando...")
    
    # Lista para acumular todos los trades
    all_trades = []
    
    for f in trade_files:
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue
            
            # Extraer meta de nombre de archivo: oos_trades_W{window}_seed{seed}.parquet
            stem = f.stem
            parts = stem.split("_")
            window = parts[2] # W1, W2, etc.
            seed = int(parts[3].replace("seed", ""))
            
            df['window'] = window
            df['seed'] = seed
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index, utc=True)
            all_trades.append(df)
        except Exception as e:
            print(f"Error procesando {f.name}: {e}")
            
    if not all_trades:
        print("No se pudieron cargar trades válidos.")
        return
        
    df_trades = pd.concat(all_trades).sort_index()
    print(f"Total de trades cargados: {len(df_trades)} de todas las semillas y ventanas.\n")
    
    # 1. Análisis global por Semilla y Ventana
    print("--- 1. RENDIMIENTO DETALLADO POR SEMILLA Y VENTANA ---")
    results = []
    
    # Agrupar por semilla y ventana
    grouped = df_trades.groupby(['seed', 'window'])
    for (seed, window), group in grouped:
        n_trades = len(group)
        wr = group['is_win'].mean() * 100 if 'is_win' in group.columns else 0.0
        
        # Calcular retornos
        ret_col = 'return_pct' if 'return_pct' in group.columns else 'ret'
        if ret_col not in group.columns:
            # Intentar deducir si hay otra columna
            cols = [c for c in group.columns if 'ret' in c or 'profit' in c]
            if cols:
                ret_col = cols[0]
            else:
                ret_col = None
                
        if ret_col:
            mean_ret = group[ret_col].mean() * 100
            cum_ret = (1 + group[ret_col]).prod() - 1
            cum_ret_pct = cum_ret * 100
            
            # Max Drawdown de la serie de trades (secuencial)
            cum_series = (1 + group[ret_col]).cumprod()
            peaks = cum_series.cummax()
            drawdowns = (cum_series - peaks) / peaks
            max_dd_pct = drawdowns.min() * 100 if not drawdowns.empty else 0.0
            
            # Sharpe Ratio de los trades (anualizado aproximado)
            std_ret = group[ret_col].std()
            if std_ret > 1e-9:
                days = (group.index.max() - group.index.min()).days
                n_per_year = n_trades / (days / 365.25) if days > 0 else n_trades
                sharpe = (group[ret_col].mean() / std_ret) * (n_per_year ** 0.5)
            else:
                sharpe = 0.0
                
            # Profit Factor & Ratio R:R (Ganancia media / Pérdida media)
            pos_rets = group[group[ret_col] > 0][ret_col]
            neg_rets = group[group[ret_col] < 0][ret_col]
            
            avg_win_pct = pos_rets.mean() * 100 if not pos_rets.empty else 0.0
            avg_loss_pct = neg_rets.mean() * 100 if not neg_rets.empty else 0.0
            
            rr_ratio = abs(avg_win_pct / avg_loss_pct) if abs(avg_loss_pct) > 1e-9 else 0.0
            profit_factor = pos_rets.sum() / abs(neg_rets.sum()) if abs(neg_rets.sum()) > 1e-9 else (100.0 if not pos_rets.empty else 0.0)
            
            results.append({
                "Seed": seed,
                "Window": window,
                "Trades": n_trades,
                "WR (%)": wr,
                "Cum Ret (%)": cum_ret_pct,
                "Avg Ret (%)": mean_ret,
                "Max DD (%)": max_dd_pct,
                "Sharpe": sharpe,
                "Avg Win (%)": avg_win_pct,
                "Avg Loss (%)": avg_loss_pct,
                "R:R Real": rr_ratio,
                "Profit Factor": profit_factor
            })
            
    df_res = pd.DataFrame(results)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    print(df_res.to_string(index=False))
    print("\n" + "-"*90 + "\n")
    
    # 2. Análisis por Ventana Agregando Semillas (Comportamiento Estructural del Mercado)
    print("--- 2. ANÁLISIS COMPARTIDO POR VENTANA TEMPORAL (Consistencia Macroeconómica) ---")
    window_res = []
    for window, group in df_trades.groupby('window'):
        n_trades = len(group)
        wr = group['is_win'].mean() * 100 if 'is_win' in group.columns else 0.0
        ret_col = 'return_pct' if 'return_pct' in group.columns else 'ret'
        
        if ret_col in group.columns:
            mean_ret = group[ret_col].mean() * 100
            # Simular promedio por trade consolidado
            pos_rets = group[group[ret_col] > 0][ret_col]
            neg_rets = group[group[ret_col] < 0][ret_col]
            
            avg_win_pct = pos_rets.mean() * 100 if not pos_rets.empty else 0.0
            avg_loss_pct = neg_rets.mean() * 100 if not neg_rets.empty else 0.0
            rr_ratio = abs(avg_win_pct / avg_loss_pct) if abs(avg_loss_pct) > 1e-9 else 0.0
            
            # Ver qué semillas operaron en esta ventana
            active_seeds = group['seed'].unique().tolist()
            
            window_res.append({
                "Window": window,
                "Total Trades": n_trades,
                "WR (%)": wr,
                "Avg Ret per Trade (%)": mean_ret,
                "Avg Win (%)": avg_win_pct,
                "Avg Loss (%)": avg_loss_pct,
                "R:R Real": rr_ratio,
                "Semillas Activas": active_seeds
            })
            
    df_win_res = pd.DataFrame(window_res)
    print(df_win_res.to_string(index=False))
    print("\n" + "-"*90 + "\n")
    
    # 3. Análisis de Distribución de Direcciones (Long vs Short)
    print("--- 3. BALANCE DE DIRECCIONALIDAD (Long vs Short) ---")
    if 'direction' in df_trades.columns:
        dir_res = []
        for direction, group in df_trades.groupby('direction'):
            n_trades = len(group)
            wr = group['is_win'].mean() * 100 if 'is_win' in group.columns else 0.0
            ret_col = 'return_pct' if 'return_pct' in group.columns else 'ret'
            mean_ret = group[ret_col].mean() * 100 if ret_col in group.columns else 0.0
            dir_res.append({
                "Direction": direction,
                "Trades": n_trades,
                "WR (%)": wr,
                "Avg Ret (%)": mean_ret
            })
        print(pd.DataFrame(dir_res).to_string(index=False))
    else:
        print("La columna 'direction' no está disponible en los parquets de trades.")
    print("\n" + "-"*90 + "\n")

    # 4. Diagnóstico de la ventana destructiva W4 y la inactiva W5
    print("--- 4. ENFOQUE CRÍTICO: DETECTANDO LA ASIMETRÍA DE W4 Y EL VETO DE W5 ---")
    w4_trades = df_trades[df_trades['window'] == 'W4']
    if not w4_trades.empty:
        print(f"Trades en W4: {len(w4_trades)}")
        ret_col = 'return_pct' if 'return_pct' in w4_trades.columns else 'ret'
        if ret_col in w4_trades.columns:
            worst_trade = w4_trades[ret_col].min() * 100
            best_trade = w4_trades[ret_col].max() * 100
            print(f"  Peor trade en W4: {worst_trade:.4f}%")
            print(f"  Mejor trade en W4: {best_trade:.4f}%")
            # Distribución de retornos en W4
            losses_below_limit = len(w4_trades[w4_trades[ret_col] <= -0.05])
            print(f"  Trades con pérdidas extremas (<= -5%): {losses_below_limit} de {len(w4_trades)} ({losses_below_limit/len(w4_trades)*100:.2f}%)")
            
            # Ver qué semillas sufrieron más
            for seed, g in w4_trades.groupby('seed'):
                s_ret = g[ret_col].mean() * 100
                s_dd = ((1 + g[ret_col]).cumprod() - (1 + g[ret_col]).cumprod().cummax()).min() * 100
                print(f"    Semilla {seed} en W4: {len(g)} trades | Ret Medio: {s_ret:.4f}% | Max DD: {s_dd:.4f}%")
                
    w5_trades = df_trades[df_trades['window'] == 'W5']
    if not w5_trades.empty:
        print(f"\nTrades en W5: {len(w5_trades)}")
        for seed, g in w5_trades.groupby('seed'):
            print(f"    Semilla {seed} en W5: {len(g)} trades | WR: {g['is_win'].mean()*100 if 'is_win' in g.columns else 0:.2f}%")
    else:
        print("\nNo se registraron trades en W5 a nivel agregado de parquets (Gated / Empty).")
        
    print("="*90)

if __name__ == "__main__":
    analyze_trades()
