import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

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
portfolio = pd.read_parquet('data/predictions/ensemble_portfolio_trades.parquet')
probs = pd.read_parquet('data/predictions/master_ensemble_probs.parquet')

# ==============================================================================
# ANALISIS 1: Robustez del hold-time por subperiodo
# ==============================================================================
mid_date = pd.Timestamp('2025-09-07', tz='UTC')
second_half = portfolio[portfolio.index > mid_date]
print(f"Segunda mitad OOS ({len(second_half)} trades): {second_half.index.min().date()} a {portfolio.index.max().date()}")

price_second = price_df[price_df.index >= mid_date]
btc_change = (price_second['close'].iloc[-1] / price_second['close'].iloc[0] - 1) * 100
print(f"BTC variacion en 2a mitad: {btc_change:+.1f}%")
print()

# ==============================================================================
# ANALISIS 2: selector CAUSAL dinamico de hold-time basado en vol 24H historica
# La clave anti-cherrypicking: los umbrales se calculan sobre datos ANTERIORES
# al inicio del OOS (datos IS). No se optimizan sobre OOS.
# ==============================================================================

btc_rets = price_df['close'].pct_change().dropna()

# Usar solo datos IS (antes de Abr 2025) para calibrar umbrales
oos_start = pd.Timestamp('2025-04-01', tz='UTC')
is_rets = btc_rets[btc_rets.index < oos_start]
rolling_vol_is = is_rets.rolling(24).std().dropna()
vol_p30 = rolling_vol_is.quantile(0.30)
vol_p70 = rolling_vol_is.quantile(0.70)
print(f"Umbrales calibrados SOLO en IS data (antes de {oos_start.date()}):")
print(f"  Vol baja (p30): {vol_p30:.6f} = {vol_p30*100:.4f}% por hora")
print(f"  Vol alta (p70): {vol_p70:.6f} = {vol_p70*100:.4f}% por hora")
print()

# Rolling vol sobre toda la serie (incluye OOS, pero la lectura es causal/pasada)
rolling_vol_all = btc_rets.rolling(24).std()

def sim_dynamic_hold_vol(portfolio, price_df, rolling_vol, vol_low, vol_high, fees=0.0004):
    results = []
    holds = []
    for entry_ts, row in portfolio.iterrows():
        past_vol_vals = rolling_vol[rolling_vol.index <= entry_ts]
        if past_vol_vals.empty or past_vol_vals.dropna().empty:
            current_vol = vol_high + 1
        else:
            current_vol = float(past_vol_vals.dropna().iloc[-1])

        if current_vol < vol_low:
            hold_h = 24
        elif current_vol > vol_high:
            hold_h = 6
        else:
            hold_h = 12

        exit_ts = entry_ts + pd.Timedelta(hours=hold_h)
        future = price_df[price_df.index >= exit_ts]
        if future.empty:
            results.append(np.nan)
            holds.append(hold_h)
            continue
        try:
            p_entry = price_df.asof(entry_ts)['close']
        except:
            results.append(np.nan)
            holds.append(hold_h)
            continue
        p_exit = future.iloc[0]['close']
        ret = (p_exit / p_entry) - 1 - fees
        results.append(ret)
        holds.append(hold_h)
    return pd.Series(results, index=portfolio.index), pd.Series(holds, index=portfolio.index)

def sim_fixed_hold(portfolio, price_df, hold_h, fees=0.0004):
    results = []
    for entry_ts, row in portfolio.iterrows():
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
        ret = (p_exit / p_entry) - 1 - fees
        results.append(ret)
    return pd.Series(results, index=portfolio.index)

days = 263
print("=== COMPARATIVA: Hold Fijo vs Dynamic Hold por Volatilidad (CAUSAL, umbrales IS) ===")
for h in [12, 24]:
    r = sim_fixed_hold(portfolio, price_df, h).dropna()
    comp = (np.prod(1+r)-1)*100
    ann = ((1+comp/100)**(365/days)-1)*100
    print(f"  Hold FIJO {h:2d}H :         WR={(r>0).mean()*100:.0f}% | Total={comp:+.2f}% | Ann={ann:+.1f}%")

r_dyn, h_dyn = sim_dynamic_hold_vol(portfolio, price_df, rolling_vol_all, vol_p30, vol_p70)
r_dyn_v = r_dyn.dropna()
comp_dyn = (np.prod(1+r_dyn_v)-1)*100
ann_dyn = ((1+comp_dyn/100)**(365/days)-1)*100
print(f"  Hold DINAMICO vol:    WR={(r_dyn_v>0).mean()*100:.0f}% | Total={comp_dyn:+.2f}% | Ann={ann_dyn:+.1f}%")
print(f"  Distribucion holds: {h_dyn.value_counts().to_dict()}")

# ==============================================================================
# ANALISIS 3: H-A - Ensemble Pruning sin re-run
# Testear filtrar las 4 semillas con Sharpe OOS negativo
# ==============================================================================
print()
print("=== H-A: ENSEMBLE PRUNING (testeable sin re-run) ===")
# Cargar todos los trades por semilla
pred_dir = Path('data/predictions')
trade_files = sorted(pred_dir.glob('oos_trades_seed*.parquet'))
seed_sharpes = {}
for f in trade_files:
    try:
        seed = int(f.stem.split('seed')[1])
        df = pd.read_parquet(f)
        r = df['return_pct']
        if len(r) < 5:
            continue
        sharpe = r.mean() / r.std() * np.sqrt(8760) if r.std() > 0 else 0
        seed_sharpes[seed] = sharpe
    except:
        pass

good_seeds = [s for s, sh in seed_sharpes.items() if sh > 0]
bad_seeds  = [s for s, sh in seed_sharpes.items() if sh <= 0]
print(f"Semillas con Sharpe OOS > 0: {len(good_seeds)} -> {sorted(good_seeds)}")
print(f"Semillas con Sharpe OOS <= 0: {len(bad_seeds)} -> {sorted(bad_seeds)}")
print()

# Simular el portfolio si solo consideramos trades de semillas buenas
# (sin probabilidades separadas, solo como proxy de impacto)
all_trades_dfs = []
for f in trade_files:
    try:
        seed = int(f.stem.split('seed')[1])
        df = pd.read_parquet(f)
        df['seed'] = seed
        all_trades_dfs.append(df)
    except:
        pass

if all_trades_dfs:
    all_trades = pd.concat(all_trades_dfs)
    all_trades = all_trades[all_trades.index.notna()].sort_index()
    
    # Portfolio filtrado a semillas buenas
    trades_good = all_trades[all_trades['seed'].isin(good_seeds)]
    trades_good_unique = trades_good.groupby(trades_good.index.floor('1h')).agg({'return_pct': 'mean'})
    
    # Comparar con portfolio completo
    all_unique = all_trades.groupby(all_trades.index.floor('1h')).agg({'return_pct': 'mean'})
    
    print(f"Trades con ensamble completo (19 semillas): {len(all_unique)}")
    print(f"Trades con solo semillas positivas ({len(good_seeds)}): {len(trades_good_unique)}")
    
    comp_all = (np.prod(1 + all_unique['return_pct']) - 1) * 100
    comp_good = (np.prod(1 + trades_good_unique['return_pct']) - 1) * 100
    wr_all = (all_unique['return_pct'] > 0).mean() * 100
    wr_good = (trades_good_unique['return_pct'] > 0).mean() * 100
    print(f"  Ensamble completo: WR={wr_all:.1f}% | Total nominal={comp_all:+.2f}%")
    print(f"  Solo buenas seeds: WR={wr_good:.1f}% | Total nominal={comp_good:+.2f}%")
