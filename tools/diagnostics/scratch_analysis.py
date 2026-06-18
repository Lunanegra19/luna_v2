import pandas as pd

df = pd.read_parquet('data/predictions/ensemble_portfolio_trades.parquet')
print(f"Total trades: {len(df)}")

win_rate_global = (df['ret'] > 0).mean() * 100
print(f"Win Rate Global: {win_rate_global:.2f}%")

if 'prob' in df.columns:
    df_high = df[df['prob'] > df['prob'].quantile(0.50)]
    wr_high = (df_high['ret'] > 0).mean() * 100
    print(f"Win Rate (Top 50% Prob): {wr_high:.2f}%, Trades: {len(df_high)}")

    df_top = df[df['prob'] > df['prob'].quantile(0.70)]
    wr_top = (df_top['ret'] > 0).mean() * 100
    print(f"Win Rate (Top 30% Prob): {wr_top:.2f}%, Trades: {len(df_top)}")
else:
    print("Columna 'prob' no encontrada. Columnas:", df.columns.tolist())
