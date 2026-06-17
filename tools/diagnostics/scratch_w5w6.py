import pandas as pd
import os

run_dir = 'data/runs/WFB_20260617_201619_seed42/seed42'

for w in ['W5', 'W6']:
    trades_path = f'{run_dir}/{w}/oos_trades.parquet'
    base_path = f'{run_dir}/{w}/oos_trades_xgb_baseline.parquet'

    print(f'\n=== {w} ===')
    if os.path.exists(trades_path):
        df = pd.read_parquet(trades_path)
        n = len(df)
        print(f'Trades filtrados (MetaLabeler): {n}')
        if n > 0:
            wr = df['is_win'].mean() * 100
            ret = df['return_raw'].sum() * 100
            print(f'Win Rate: {wr:.2f}%')
            print(f'Return Raw Acumulado: {ret:.2f}%')
            cols = ['entry_time', 'return_raw', 'is_win', 'HMM_Semantic']
            available = [c for c in cols if c in df.columns]
            print(df[available].to_string())
        else:
            print('  (Sin trades en esta ventana)')
    else:
        print(f'  [NO DISPONIBLE AUN] {trades_path}')

    if os.path.exists(base_path):
        df_b = pd.read_parquet(base_path)
        wr_b = df_b['is_win'].mean() * 100
        ret_b = df_b['return_raw'].sum() * 100
        print(f'Base XGBoost: {len(df_b)} signals | WR: {wr_b:.2f}% | Return: {ret_b:.2f}%')
    else:
        print(f'  [NO DISPONIBLE AUN] xgb_baseline')
