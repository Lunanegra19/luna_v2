import pandas as pd
import numpy as np
from pathlib import Path

wfb_dir = Path("g:/Mi unidad/ia/luna_v2/data/reports/wfb")

def analyze_trade_details():
    dfs = []
    for w in [1, 2, 3]:
        p = wfb_dir / f"oos_trades_W{w}_seed42.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df['window'] = f"W{w}"
            dfs.append(df)
            
    if not dfs:
        print("No trades found.")
        return
        
    trades = pd.concat(dfs).reset_index()
    
    print("================================================================")
    print("  ANÁLISIS DETALLADO DE DISTRIBUCIÓN DE RETORNOS - SEMILLA 42")
    print("================================================================\n")
    
    # Estadísticas descriptivas de los retornos (return_pct)
    ret_pct = trades['return_pct'] * 100
    print("Estadísticas de Retornos por Trade (%):")
    print(f"  Media (EV)      : {ret_pct.mean():.4f}%")
    print(f"  Desv. Estándar  : {ret_pct.std():.4f}%")
    print(f"  Mínimo (Peor)   : {ret_pct.min():.4f}%")
    print(f"  Máximo (Mejor)  : {ret_pct.max():.4f}%")
    print(f"  Mediana         : {ret_pct.median():.4f}%")
    print(f"  Skewness        : {ret_pct.skew():.4f}")
    
    # Asimetría Ganancias vs Pérdidas
    gains = ret_pct[ret_pct > 0]
    losses = ret_pct[ret_pct < 0]
    
    print("\nGanancias vs Pérdidas:")
    print(f"  Trades Ganadores: {len(gains)} (WR = {len(gains)/len(trades)*100:.1f}%)")
    print(f"  Trades Perdedores: {len(losses)} (LR = {len(losses)/len(trades)*100:.1f}%)")
    if len(gains) > 0:
        print(f"  Media Ganancias : {gains.mean():.4f}%")
    if len(losses) > 0:
        print(f"  Media Pérdidas  : {losses.mean():.4f}%")
    if len(gains) > 0 and len(losses) > 0:
        ratio_avg = gains.mean() / abs(losses.mean())
        print(f"  Ratio R:R Real  : {ratio_avg:.4f} (Media Ganancia / Media Pérdida)")
        
    # Analizar si las pérdidas son sistemáticamente grandes
    print("\nDistribución por cuantiles de retornos (%):")
    q = ret_pct.quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    for quant, val in q.items():
        print(f"  Q_{quant*100:02.0f}%: {val:.4f}%")

    # Duración de los trades
    if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
        trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        trades['exit_time'] = pd.to_datetime(trades['exit_time'])
        trades['duration_hours'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600
        
        print("\nEstadísticas de Duración de Trades (Horas):")
        print(f"  Media           : {trades['duration_hours'].mean():.1f} horas")
        print(f"  Mínima          : {trades['duration_hours'].min():.1f} horas")
        print(f"  Máxima          : {trades['duration_hours'].max():.1f} horas")
        
        print("\nDuración Ganadores vs Perdedores:")
        print(f"  Duración Media Ganadores  : {trades[trades['return_pct'] > 0]['duration_hours'].mean():.1f} horas")
        print(f"  Duración Media Perdedores : {trades[trades['return_pct'] < 0]['duration_hours'].mean():.1f} horas")
        
    # Verificar si el tamaño de Kelly o tribe_mult influye
    print("\nTamaño de la Posición y Multiplicador:")
    if 'tribe_mult' in trades.columns:
        print(f"  Valores únicos de tribe_mult: {trades['tribe_mult'].unique().tolist()}")
    if 'kelly_fraction_used' in trades.columns:
        print(f"  Media kelly_fraction_used   : {trades['kelly_fraction_used'].mean():.4f}")
        print(f"  Máxima kelly_fraction_used  : {trades['kelly_fraction_used'].max():.4f}")
        
    # Ver los 5 peores trades
    print("\nLos 5 peores trades de la muestra:")
    peores = trades.sort_values('return_pct').head(5)
    for idx, row in peores.iterrows():
        print(f"  Fecha: {row['entry_time']} | Ventana: {row['window']} | Ret: {row['return_pct']*100:.4f}% | Prob Base: {row['xgb_prob']:.3f} | Prob Meta: {row['meta_v2_prob']:.3f} | Regime: {row['hmm_regime']}")

    # Ver los 5 mejores trades
    print("\nLos 5 mejores trades de la muestra:")
    mejores = trades.sort_values('return_pct', ascending=False).head(5)
    for idx, row in mejores.iterrows():
        print(f"  Fecha: {row['entry_time']} | Ventana: {row['window']} | Ret: {row['return_pct']*100:.4f}% | Prob Base: {row['xgb_prob']:.3f} | Prob Meta: {row['meta_v2_prob']:.3f} | Regime: {row['hmm_regime']}")

if __name__ == "__main__":
    analyze_trade_details()
