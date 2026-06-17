import pandas as pd
import numpy as np
from pathlib import Path

def calc_kelly_and_metrics(parquet_path, kelly_fraction=0.25, psi_penalty=0.5):
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        return {'trades': 0}
        
    df = df.copy()
    
    # Kelly Base
    p = df['xgb_prob_cal'].clip(0.01, 0.99)
    q = 1 - p
    b = 1.0 # default odd
    kelly_raw = p - (q / b)
    kelly_raw = kelly_raw.clip(0.0, 1.0)
    
    # Fractions
    df['kelly_fraction_used'] = kelly_raw * kelly_fraction * psi_penalty * 1.0 # pred_drift=1.0 now
    
    # Returns
    df['_ret_bruto'] = df['return_raw'] * df['tribe_mult']
    df['return_pct'] = df['_ret_bruto'] * df['kelly_fraction_used']
    df['is_win_kelly'] = df['return_pct'] > 0
    
    # Compounding
    ret_pct = df['return_pct'].values
    cum_returns = np.cumprod(1 + ret_pct)
    total_ret = cum_returns[-1] - 1.0 if len(cum_returns) > 0 else 0.0
    
    # MaxDD
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    
    wins = df['is_win_kelly'].sum()
    total = len(df)
    
    return {
        'trades': total,
        'win_rate': (wins / total * 100) if total > 0 else 0.0,
        'ret_comp': total_ret * 100,
        'max_dd': max_dd * 100,
        'calmar': (total_ret / abs(max_dd)) if max_dd < 0 else float('inf')
    }

for p in Path('data/predictions').glob('oos_trades_*.parquet'):
    if 'baseline' in p.name or 'raw' in p.name: continue
    res = calc_kelly_and_metrics(p)
    print(f'\n--- {p.name} ---')
    print(f'Trades: {res.get("trades", 0)}')
    if res.get("trades", 0) > 0:
        print(f'Win Rate: {res["win_rate"]:.1f}%')
        print(f'Ret Comp: {res["ret_comp"]:.2f}%')
        print(f'Max DD:   {res["max_dd"]:.2f}%')
        print(f'Calmar:   {res["calmar"]:.2f}')
