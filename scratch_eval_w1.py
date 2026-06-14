import pandas as pd
df = pd.read_parquet('C:/Users/Usuario/Desktop/ia/luna_v2/data/runs/WFB_20260614_201151_seed42/seed42/W1/oos_trades.parquet')
print(f'Retorno Simple Acumulado: {df["return_pct"].sum()*100:.2f}%')
if 'equity_curve' in df.columns:
    eq = df['equity_curve'].iloc[-1]
    print(f'Retorno Compuesto (Equity): {(eq-1)*100:.2f}%')
if 'drawdown' in df.columns:
    print(f'Max Drawdown: {df["drawdown"].min()*100:.2f}%')
