import pandas as pd
import json
from pathlib import Path

base_runs = Path('g:/Mi unidad/ia/luna_v2/data/runs')
reports_dir = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb') # and reports

seeds = [42, 100, 777, 1337, 2025, 53929, 59100]
results = []

for seed in seeds:
    seed_stats = {'Seed': seed, 'Total_Trades': 0, 'Global_WR': 0, 'Global_Ret(%)': 0}
    dfs = []
    
    # Buscamos en TODOS los WFB_* para sacar el parquet más reciente de cada ventana
    for w in ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']:
        # Buscar el parquet de reporte canónico si existe
        canonic_f = Path(f'g:/Mi unidad/ia/luna_v2/data/reports/wfb/oos_trades_{w}_seed{seed}.parquet')
        trades = 0
        wr = 0.0
        ret = 0.0
        if canonic_f.exists():
            try:
                df = pd.read_parquet(canonic_f)
                trades = len(df)
                if trades > 0:
                    wr = df.is_win.mean()
                    ret = df.return_pct.sum() * 100
                    dfs.append(df)
            except Exception:
                pass
        seed_stats[f'{w}_Trades'] = trades
        seed_stats[f'{w}_WR(%)'] = wr * 100
        seed_stats[f'{w}_Ret(%)'] = ret
        
    if dfs:
        all_df = pd.concat(dfs)
        seed_stats['Total_Trades'] = len(all_df)
        seed_stats['Global_WR(%)'] = all_df.is_win.mean() * 100
        seed_stats['Global_Ret(%)'] = all_df.return_pct.sum() * 100
    
    reports_dir = Path('g:/Mi unidad/ia/luna_v2/data/reports')
    verdicts = list(reports_dir.glob(f'*_seed{seed}_FINAL_statistical_verdict.json'))
    seed_stats['DSR_Veredicto'] = 'NaN'
    seed_stats['Ret_Veredicto(%)'] = 'NaN'
    if verdicts:
        latest_verdict = sorted(verdicts)[-1]
        try:
            with open(latest_verdict, 'r', encoding='utf-8') as vf:
                v = json.load(vf)
                dsr = v.get('statistical_audit', {}).get('dsr', 0)
                r = v.get('metrics', {}).get('total_return_pct', 0)
                seed_stats['DSR_Veredicto'] = f"{dsr:.2f}"
                seed_stats['Ret_Veredicto(%)'] = f"{r:.2f}"
        except Exception:
            pass
            
    results.append(seed_stats)

df_res = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(df_res.to_string(index=False))
