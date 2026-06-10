import pandas as pd
import numpy as np

df = pd.read_parquet('c:/Users/Usuario/Desktop/ia/luna_v2/data/predictions/unified_ensemble_trades_raw.parquet')

def simulate_threshold(df, col, threshold):
    filtered = df[df[col] >= threshold].copy()
    n_trades = len(filtered)
    if n_trades == 0:
        return 0, 0.0, 0.0, 0.0
    wins = filtered[filtered['return_raw'] > 0]
    win_rate = len(wins) / n_trades
    net_ret = filtered['return_raw'].sum()
    net_kelly = filtered['return_pct'].sum()
    return n_trades, win_rate, net_ret, net_kelly

print("\n=== LGBM PROB SWEEP ===")
results = []
for th in np.arange(0.40, 0.65, 0.02):
    n, wr, nr, nk = simulate_threshold(df, 'lgbm_prob', th)
    results.append({'threshold': th, 'n_trades': n, 'win_rate': wr*100, 'net_ret_unleveraged': nr*100, 'net_kelly': nk*100})
print(pd.DataFrame(results).to_string(index=False, float_format='%.3f'))
