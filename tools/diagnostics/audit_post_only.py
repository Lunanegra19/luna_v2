"""
tools/diagnostics/audit_post_only.py
====================================
Script de auditoría para simular la tasa de llenado (fill rate) de órdenes
Post-Only (Limit) sobre el portafolio consolidado del ensamble de Luna V2.

Compara el timestamp de cada trade con la vela horaria correspondiente para ver
si el precio fluctuó en la dirección opuesta al inicio de la vela, lo que habría
garantizado el llenado de una orden pasiva (maker) limitada al precio de apertura.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def run_audit():
    logger.info("Iniciando auditoría de órdenes Post-Only...")
    
    trades_path = _ROOT / "data" / "predictions" / "ensemble_portfolio_trades.parquet"
    ohlcv_path = _ROOT / "data" / "raw" / "ohlcv" / "ohlcv_raw.parquet"
    
    if not trades_path.exists():
        logger.error(f"No se encontró el archivo de trades en: {trades_path}")
        return
    if not ohlcv_path.exists():
        logger.error(f"No se encontró el archivo de precios en: {ohlcv_path}")
        return
        
    # 1. Cargar datos
    df_trades = pd.read_parquet(trades_path)
    df_ohlcv = pd.read_parquet(ohlcv_path)
    
    logger.info(f"Cargados {len(df_trades)} trades del portafolio consolidado.")
    logger.info(f"Cargadas {len(df_ohlcv)} velas horarias de precio.")
    
    # Asegurar formato de timestamps con zona horaria UTC
    if 'timestamp' in df_trades.columns:
        df_trades = df_trades.set_index('timestamp')
    df_trades.index = pd.to_datetime(df_trades.index, utc=True)
    
    if 'timestamp' in df_ohlcv.columns:
        df_ohlcv = df_ohlcv.set_index('timestamp')
    df_ohlcv.index = pd.to_datetime(df_ohlcv.index, utc=True)
    
    # Ordenar índices
    df_trades = df_trades.sort_index()
    df_ohlcv = df_ohlcv.sort_index()
    
    # 2. Alinear y cruzar datos
    # Como la señal se ejecuta al cierre de la vela T, la orden se envía para la vela T+1.
    # El timestamp del trade suele ser el cierre de la vela T. Vamos a buscar la vela de entrada (T+1).
    filled_results = []
    
    # Definir umbrales de offset para simular (porcentaje de margen con respecto al Open)
    offsets = [0.0, 0.0001, 0.0002, 0.0005, 0.0010]  # 0%, 0.01%, 0.02%, 0.05%, 0.10%
    
    # Almacén de estadísticas para imprimir
    stats_summary = []
    
    for offset in offsets:
        filled_count = 0
        missed_count = 0
        total_valid = 0
        
        filled_trades_returns = []
        original_returns = []
        
        for ts, row in df_trades.iterrows():
            # Buscar la vela de ejecución. Si el trade está marcado en 'ts', la ejecución ocurre en la vela que abre en 'ts'.
            # A veces hay una pequeña discrepancia de minutos, por lo que buscamos la vela más cercana o exacta.
            if ts in df_ohlcv.index:
                candle = df_ohlcv.loc[ts]
            else:
                # Buscar la vela más cercana en el futuro cercano (máximo 1 hora)
                matches = df_ohlcv.index[(df_ohlcv.index >= ts) & (df_ohlcv.index < ts + pd.Timedelta(hours=1))]
                if len(matches) > 0:
                    candle = df_ohlcv.loc[matches[0]]
                else:
                    continue
            
            total_valid += 1
            op = float(candle['open'])
            hi = float(candle['high'])
            lo = float(candle['low'])
            cl = float(candle['close'])
            
            direction = str(row.get('direction', 'BUY')).upper()
            is_buy = ('BUY' in direction or direction == '1' or row.get('direction') == 1)
            
            ret_orig = float(row.get('return_pct', 0.0))
            original_returns.append(ret_orig)
            
            if is_buy:
                # Para un BUY (Long), colocamos orden límite pasiva en: Open * (1 - offset)
                limit_price = op * (1.0 - offset)
                # Se llena si la fluctuación a la baja (low) de esa vela cruzó nuestro precio límite
                # Usamos menor estricto (<) para ser sumamente conservadores (garantiza llenado en la cola)
                if lo < limit_price:
                    filled_count += 1
                    # Si entramos más barato, nuestro retorno real mejora debido al "positive slippage"
                    slippage_gain = (op - limit_price) / op
                    filled_trades_returns.append(ret_orig + slippage_gain)
                else:
                    missed_count += 1
            else:
                # Para un SELL (Short), colocamos orden límite pasiva en: Open * (1 + offset)
                limit_price = op * (1.0 + offset)
                # Se llena si la fluctuación al alza (high) de esa vela cruzó nuestro precio límite
                if hi > limit_price:
                    filled_count += 1
                    # Si entramos más alto en el short, nuestro retorno real mejora
                    slippage_gain = (limit_price - op) / op
                    filled_trades_returns.append(ret_orig + slippage_gain)
                else:
                    missed_count += 1
                    
        fill_rate = (filled_count / total_valid) * 100 if total_valid > 0 else 0.0
        avg_ret_filled = np.mean(filled_trades_returns) * 100 if len(filled_trades_returns) > 0 else 0.0
        avg_ret_orig = np.mean(original_returns) * 100 if len(original_returns) > 0 else 0.0
        
        stats_summary.append({
            "offset": offset,
            "total": total_valid,
            "filled": filled_count,
            "missed": missed_count,
            "fill_rate": fill_rate,
            "avg_ret_filled": avg_ret_filled,
            "avg_ret_orig": avg_ret_orig,
            "cum_ret_filled": (np.prod([1 + r for r in filled_trades_returns]) - 1) * 100 if len(filled_trades_returns) > 0 else 0.0,
            "cum_ret_orig": (np.prod([1 + r for r in original_returns]) - 1) * 100 if len(original_returns) > 0 else 0.0,
        })
        
    # Imprimir reporte
    print("\n" + "="*80)
    print("   AUDITORÍA CUANTITATIVA: SIMULACIÓN DE EJECUCIÓN POST-ONLY (LIMIT)")
    print("="*80)
    print(f"Total de trades auditados en el portafolio consolidado: {len(df_trades)}")
    print("-"*80)
    print(f"{'Offset Límite':<15} | {'Trades':<8} | {'Llenados':<8} | {'Tasa Fill':<10} | {'Ret. Medio':<12} | {'Ret. Compuesto':<15}")
    print("-"*80)
    for s in stats_summary:
        offset_pct = f"{s['offset']*100:.3f}%"
        fill_pct = f"{s['fill_rate']:.2f}%"
        ret_f = f"{s['avg_ret_filled']:.4f}%"
        cum_f = f"{s['cum_ret_filled']:.2f}%"
        print(f"{offset_pct:<15} | {s['total']:<8} | {s['filled']:<8} | {fill_pct:<10} | {ret_f:<12} | {cum_f:<15}")
    print("-"*80)
    print("Nota: La simulación utiliza un criterio estricto y conservador (Low < Limit / High > Limit).")
    print("Un offset del 0.000% representa colocar la orden límite exactamente al precio de apertura (Open).")
    print("Un offset del 0.010% al 0.050% busca 'pescar' fluctuaciones defensivas con descuento.")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_audit()
