import pandas as pd
import glob

print('--- BUSCANDO BUG 0.0 EN RAW PROBS ---')
raw_files = glob.glob('data/reports/wfb/oos_raw_probs_W*_seed42.parquet')
if not raw_files:
    raw_files = glob.glob('data/predictions/oos_raw_probs_W*_seed42.parquet')
if not raw_files:
    print('No se encontraron raw_probs. Buscando en todos los directorios...')
    raw_files = glob.glob('**/*raw_probs*.parquet', recursive=True)

for f in raw_files[:3]:
    try:
        df = pd.read_parquet(f)
        print(f'\nArchivo: {f}')
        cols = [c for c in ['xgb_prob', 'xgb_prob_cal'] if c in df.columns]
        for col in cols:
            print(f'{col}: media={df[col].mean():.4f}, min={df[col].min():.4f}, max={df[col].max():.4f}')
        if not cols:
            print('Columnas de xgb no encontradas en raw_probs.')
    except Exception as e:
        print(f'Error leyendo {f}: {e}')

print('\n--- PRUEBA HOLDING TIME CAP <= 24H ---')
trade_files = glob.glob('data/predictions/oos_trades_W*_seed42.parquet')
if trade_files:
    dfs = [pd.read_parquet(f) for f in trade_files]
    df_trades = pd.concat(dfs, ignore_index=True)
    if 'entry_time' in df_trades.columns and 'exit_time' in df_trades.columns:
        df_trades['duration_h'] = (pd.to_datetime(df_trades['exit_time']) - pd.to_datetime(df_trades['entry_time'])).dt.total_seconds() / 3600
        
        df_fast = df_trades[df_trades['duration_h'] <= 24.0]
        df_slow = df_trades[df_trades['duration_h'] > 24.0]
        
        tot_ret = df_trades['return_raw'].sum() * 100
        fast_ret = df_fast['return_raw'].sum() * 100
        slow_ret = df_slow['return_raw'].sum() * 100
        wr_fast = df_fast['is_win'].mean() * 100
        wr_slow = df_slow['is_win'].mean() * 100
        
        print(f'Trades totales: {len(df_trades)}')
        print(f'Retorno nominal actual: {tot_ret:.2f}%')
        
        print(f'\nSi hubieramos forzado el cierre a las 24h (Simulacion aproximada descartando lentos):')
        print(f'Trades <= 24h: {len(df_fast)} (Retorno: {fast_ret:.2f}%)')
        print(f'Trades > 24h: {len(df_slow)} (Retorno: {slow_ret:.2f}%)')
        print(f'Win Rate rapido: {wr_fast:.1f}%, Win Rate lento: {wr_slow:.1f}%')
