import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

portfolio = pd.read_parquet('data/predictions/ensemble_portfolio_trades.parquet')
probs = pd.read_parquet('data/predictions/master_ensemble_probs.parquet')

oos_dir = Path('data/archive/W1_seed100/features_prev')
files = sorted(oos_dir.glob('features_holdout_*.parquet'))
dfs = []
for f in files:
    try:
        df = pd.read_parquet(f, columns=['close'])
        dfs.append(df)
    except:
        pass
price_df = pd.concat(dfs).sort_index()
price_df = price_df[~price_df.index.duplicated(keep='first')]

days = 263
fees = 0.0004
threshold = 0.55

bear_signals = probs[probs['prob_bear'] >= threshold][['prob_bear']]
bull_signals = probs[probs['prob_bull'] >= threshold]
print(f"Senales BAJISTAS (prob_bear >= {threshold}): {len(bear_signals)}")
print(f"Senales ALCISTAS (prob_bull >= {threshold}): {len(bull_signals)}")

# Shorts: ganan si precio baja en 24H
short_results = []
for ts, row in bear_signals.iterrows():
    exit_ts = ts + pd.Timedelta(hours=24)
    future = price_df[price_df.index >= exit_ts]
    if future.empty:
        continue
    try:
        p_entry = price_df.asof(ts)['close']
    except:
        continue
    p_exit = future.iloc[0]['close']
    ret_short = (p_entry / p_exit) - 1 - fees  # SHORT
    short_results.append({'ts': ts, 'ret': ret_short})

df_shorts = pd.DataFrame(short_results).set_index('ts')

comp_short = (np.prod(1 + df_shorts['ret']) - 1) * 100
ann_short = ((1 + comp_short/100)**(365/days) - 1) * 100
wr_short = (df_shorts['ret'] > 0).mean() * 100
print()
print(f"=== H-C: SHORTS por prob_bear >= {threshold} (hold 24H, Maker 0.04%) ===")
print(f"Total trades Short: {len(df_shorts)}")
print(f"Win Rate: {wr_short:.1f}%")
print(f"Retorno Total (9 meses): {comp_short:+.2f}%")
print(f"Retorno Anualizado: {ann_short:+.1f}%")

# Longs (24H Maker)
long_results = []
for entry_ts, row in portfolio.iterrows():
    exit_ts = entry_ts + pd.Timedelta(hours=24)
    future = price_df[price_df.index >= exit_ts]
    if future.empty:
        continue
    try:
        p_entry = price_df.asof(entry_ts)['close']
    except:
        continue
    p_exit = future.iloc[0]['close']
    long_results.append((p_exit / p_entry) - 1 - fees)

long_series = pd.Series(long_results)

comp_long = (np.prod(1 + long_series) - 1) * 100
ann_long = ((1 + comp_long/100)**(365/days) - 1) * 100

# Combinado
all_series = pd.concat([long_series.reset_index(drop=True), df_shorts['ret'].reset_index(drop=True)])
comp_comb = (np.prod(1 + all_series) - 1) * 100
ann_comb = ((1 + comp_comb/100)**(365/days) - 1) * 100

print()
print("=== PORTAFOLIO COMBINADO: Longs + Shorts ===")
print(f"Solo LONGS  ({len(long_series)} trades): WR={(long_series>0).mean()*100:.0f}% | Total={comp_long:+.2f}% | Ann={ann_long:+.1f}%")
print(f"Solo SHORTS ({len(df_shorts)} trades): WR={wr_short:.0f}% | Total={comp_short:+.2f}% | Ann={ann_short:+.1f}%")
print(f"COMBINADO   ({len(all_series)} trades): WR={(all_series>0).mean()*100:.0f}% | Total={comp_comb:+.2f}% | Ann={ann_comb:+.1f}%")
print()

# Resumen final comparativo de TODOS los escenarios sin re-run
print("=" * 70)
print("RESUMEN TOTAL: Escenarios testeables sin re-run (datos OOS existentes)")
print("=" * 70)

btc_rets_all = price_df['close'].pct_change().dropna()
oos_start = pd.Timestamp('2025-04-01', tz='UTC')
is_rets = btc_rets_all[btc_rets_all.index < oos_start]
rolling_vol_is = is_rets.rolling(24).std().dropna()
vol_p30 = rolling_vol_is.quantile(0.30)
vol_p70 = rolling_vol_is.quantile(0.70)
rolling_vol_all = btc_rets_all.rolling(24).std()

def sim_dynamic_hold_vol(portfolio, price_df, rolling_vol, vol_low, vol_high, fees=0.0004):
    results = []
    for entry_ts, row in portfolio.iterrows():
        past_vol = rolling_vol[rolling_vol.index <= entry_ts].dropna()
        current_vol = float(past_vol.iloc[-1]) if not past_vol.empty else vol_high + 1
        hold_h = 24 if current_vol < vol_low else (6 if current_vol > vol_high else 12)
        exit_ts = entry_ts + pd.Timedelta(hours=hold_h)
        future = price_df[price_df.index >= exit_ts]
        if future.empty:
            results.append(np.nan)
            continue
        try:
            p_entry = price_df.asof(entry_ts)['close']
        except:
            results.append(np.nan)
            continue
        p_exit = future.iloc[0]['close']
        results.append((p_exit / p_entry) - 1 - fees)
    return pd.Series(results, index=portfolio.index)

r_dyn = sim_dynamic_hold_vol(portfolio, price_df, rolling_vol_all, vol_p30, vol_p70).dropna()
c_dyn = (np.prod(1 + r_dyn) - 1) * 100
a_dyn = ((1 + c_dyn/100)**(365/days) - 1) * 100

print(f"Actual (1H Spot Taker):               WR=27% | Total=-5.4%  | Ann=-7.5%")
print(f"P1+P2+P3 Hold24H Futuros Maker:       WR=62% | Total={comp_long:+.2f}% | Ann={ann_long:+.1f}%")
print(f"P1+P2+P3 + Dynamic Hold vol:          WR={int((r_dyn>0).mean()*100)}% | Total={c_dyn:+.2f}% | Ann={a_dyn:+.1f}%")
print(f"P1+P2+P3 + H-C Shorts combinado:      WR={(all_series>0).mean()*100:.0f}% | Total={comp_comb:+.2f}% | Ann={ann_comb:+.1f}%")
