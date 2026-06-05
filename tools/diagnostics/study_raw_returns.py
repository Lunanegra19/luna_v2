import pandas as pd
from pathlib import Path

wfb_dir = Path("g:/Mi unidad/ia/luna_v2/data/reports/wfb")
dfs = []
for w in [1, 2, 3]:
    p = wfb_dir / f"oos_trades_W{w}_seed42.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df['window'] = f"W{w}"
        dfs.append(df)
        
if dfs:
    trades = pd.concat(dfs).reset_index()
    print("================================================================")
    print("  ESTADÍSTICAS DEL RETORNO BRUTO REAL (return_raw * 100)")
    print("================================================================\n")
    
    # return_raw en porcentaje
    ret_raw = trades['return_raw'] * 100
    print(f"  Media (EV Bruto neto costos) : {ret_raw.mean():.4f}%")
    print(f"  Mínimo (Peor Trade)         : {ret_raw.min():.4f}%")
    print(f"  Máximo (Mejor Trade)        : {ret_raw.max():.4f}%")
    print(f"  Desviación Estándar         : {ret_raw.std():.4f}%")
    
    gains = ret_raw[ret_raw > 0]
    losses = ret_raw[ret_raw < 0]
    print(f"\n  Ganancias Promedio          : {gains.mean():.4f}% (n={len(gains)})")
    print(f"  Pérdidas Promedio           : {losses.mean():.4f}% (n={len(losses)})")
    print(f"  Ratio R:R Bruto Real        : {gains.mean() / abs(losses.mean()):.4f}")
    
    # Kelly multipliers
    print(f"\n  Kelly Multiplier Promedio   : {trades['tribe_mult'].mean():.4f}")
else:
    print("No trades found.")
