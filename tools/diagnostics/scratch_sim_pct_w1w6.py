import pandas as pd
import os

wfb_dir = 'data/reports/wfb'
seed = '42'

# Cargar todas las ventanas disponibles (W1 a W10)
all_dfs = []
windows_found = []
for w in ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11', 'W12', 'W13', 'W14', 'W15']:
    path = f'{wfb_dir}/oos_trades_xgb_baseline_{w}_seed{seed}.parquet'
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df['window'] = w
        all_dfs.append(df)
        windows_found.append(w)
        print(f'  [{w}] {len(df)} base signals | WR: {df["is_win"].mean()*100:.1f}% | Return: {df["return_raw"].sum()*100:.2f}%')
    else:
        print(f'  [{w}] NO DISPONIBLE — {path}')

if not all_dfs:
    print('ERROR: No se encontraron ventanas.')
    exit(1)

df = pd.concat(all_dfs)
print(f'\n=== SIMULACION W1 a {windows_found[-1]} ===')
print(f'Total Base Signals: {len(df)}')
print(f'Base Win Rate (Sin filtro): {df["is_win"].mean()*100:.2f}%')
print(f'Base Return Acumulado: +{df["return_raw"].sum()*100:.2f}%')
print('-' * 70)

percentiles_to_test = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
for p in percentiles_to_test:
    threshold = df['meta_v2_prob'].quantile(p)
    df_f = df[df['meta_v2_prob'] >= threshold]
    n = len(df_f)
    if n > 0:
        wr = df_f['is_win'].mean() * 100
        ret = df_f['return_raw'].sum() * 100
        print(f'Percentile {p*100:.0f}% (Prob >= {threshold:.4f}) | Trades: {n:<4} | WinRate: {wr:.2f}% | Return: +{ret:.2f}%')
    else:
        print(f'Percentile {p*100:.0f}% | 0 trades')

print()
print('=== DESGLOSE POR VENTANA ===')
for w in windows_found:
    d = df[df['window'] == w]
    print(f'{w}: {len(d)} signals | WR: {d["is_win"].mean()*100:.1f}% | Return: {d["return_raw"].sum()*100:.2f}%')

