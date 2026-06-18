"""
Reporte Máximo Detalle: Por Seed y Por Ventana
"""
import pandas as pd
import numpy as np
import os

PRED_DIR = "data/predictions"
REPORT_DIR = "data/reports/wfb"

# ─────────────────────────────────────────
# 1. CARGA DE DATOS CRUDOS
# ─────────────────────────────────────────
df_raw = pd.read_parquet(f"{PRED_DIR}/unified_ensemble_trades_raw.parquet")
if not isinstance(df_raw.index, pd.DatetimeIndex):
    df_raw.index = pd.to_datetime(df_raw.index)
df_raw = df_raw.sort_index()

df_ensemble = pd.read_parquet(f"{PRED_DIR}/ensemble_portfolio_trades.parquet")
if not isinstance(df_ensemble.index, pd.DatetimeIndex):
    df_ensemble.index = pd.to_datetime(df_ensemble.index)
df_ensemble = df_ensemble.sort_index()

print(f"Trades RAW totales: {len(df_raw)}")
print(f"Trades ENSEMBLE finales: {len(df_ensemble)}")
print(f"Seeds en RAW: {sorted(df_raw['seed'].unique())}")
print(f"Columnas RAW: {list(df_raw.columns)}")
print(f"Columnas ENSEMBLE: {list(df_ensemble.columns)}")
print()

# ─────────────────────────────────────────
# 2. MÉTRICAS POR SEED (COMPLETO)
# ─────────────────────────────────────────
def calc_metrics(df, label=""):
    if len(df) == 0:
        return {}
    
    # is_win detection: usar columna o calcular desde return_pct
    if 'is_win' in df.columns:
        wins = df['is_win'].astype(bool)
    else:
        wins = df['return_pct'] > 0
    
    wr = wins.mean() * 100
    n_trades = len(df)
    mean_ret = df['return_pct'].mean() * 100
    total_ret = df['return_pct'].sum() * 100
    
    # Sharpe (retorno acumulado anualizado / std)
    ret_series = df['return_pct'].dropna()
    if ret_series.std() > 0:
        sharpe = (ret_series.mean() / ret_series.std()) * np.sqrt(8760)  # hourly
    else:
        sharpe = 0.0
    
    # Drawdown
    equity = (1 + ret_series).cumprod()
    roll_max = equity.expanding().max()
    dd_series = (equity - roll_max) / roll_max
    max_dd = dd_series.min() * 100
    
    # Calmar ratio
    cumulative_ret = (equity.iloc[-1] - 1) * 100 if len(equity) > 0 else 0
    calmar = cumulative_ret / abs(max_dd) if max_dd != 0 else 0
    
    # Profit Factor
    gains = ret_series[ret_series > 0].sum()
    losses = abs(ret_series[ret_series < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')
    
    # Avg Win / Avg Loss
    avg_win = ret_series[ret_series > 0].mean() * 100 if (ret_series > 0).any() else 0
    avg_loss = ret_series[ret_series < 0].mean() * 100 if (ret_series < 0).any() else 0
    
    # Direcciones
    if 'direction' in df.columns:
        n_long = (df['direction'] == 'long').sum()
        n_short = (df['direction'] == 'short').sum()
    else:
        n_long = n_short = 0
    
    # Regimenes
    if 'HMM_Semantic' in df.columns:
        regimes = df['HMM_Semantic'].value_counts().to_dict()
    else:
        regimes = {}
    
    # Rolling WR
    wins_int = wins.astype(int)
    rolling_wr_last = wins_int.rolling(20, min_periods=5).mean().iloc[-1] * 100 if len(wins_int) >= 5 else wr

    return {
        'n_trades': n_trades,
        'win_rate': wr,
        'total_return_pct': total_ret,
        'cumulative_ret': cumulative_ret,
        'mean_ret_per_trade': mean_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'profit_factor': pf,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'n_long': n_long,
        'n_short': n_short,
        'rolling_wr_last20': rolling_wr_last,
        'regimes': regimes
    }


# ─────────────────────────────────────────
# PRINT FUNCIÓN
# ─────────────────────────────────────────
def print_metrics(m, title):
    pf = m['profit_factor']
    pf_str = f"{pf:.3f}" if pf != float('inf') else "∞"
    print(f"""
┌── {title}
│  Trades        : {m['n_trades']}    (Long: {m['n_long']} | Short: {m['n_short']})
│  Win Rate      : {m['win_rate']:.2f}%
│  Total Retorno : {m['total_return_pct']:.4f}%  (Cumulativo: {m['cumulative_ret']:.4f}%)
│  Ret/Trade     : {m['mean_ret_per_trade']:.5f}%
│  Sharpe        : {m['sharpe']:.4f}
│  Max Drawdown  : {m['max_dd']:.4f}%
│  Calmar Ratio  : {m['calmar']:.3f}
│  Profit Factor : {pf_str}
│  Avg Win       : {m['avg_win_pct']:.5f}%   Avg Loss: {m['avg_loss_pct']:.5f}%
│  WR Rolling 20 : {m['rolling_wr_last20']:.2f}%""")
    if m['regimes']:
        top3 = sorted(m['regimes'].items(), key=lambda x: -x[1])[:3]
        for r, c in top3:
            print(f"│  Régimen       : {r} = {c} trades")
    print("└" + "─"*60)

# ─────────────────────────────────────────
# 3. MÉTRICAS POR SEED
# ─────────────────────────────────────────
print("=" * 65)
print("SECCIÓN A: MÉTRICAS POR SEED (TODAS LAS VENTANAS COMBINADAS)")
print("=" * 65)

seeds = sorted(df_raw['seed'].unique())
seed_results = {}
for seed in seeds:
    df_s = df_raw[df_raw['seed'] == seed]
    m = calc_metrics(df_s, f"SEED {seed}")
    seed_results[seed] = m
    print_metrics(m, f"SEED {seed}")

# ─────────────────────────────────────────
# 4. MÉTRICAS POR VENTANA WFB
# ─────────────────────────────────────────
print("\n" + "=" * 65)
print("SECCIÓN B: MÉTRICAS POR VENTANA WFB (TODAS LAS SEEDS COMBINADAS)")
print("=" * 65)

if 'wfb_window' in df_raw.columns:
    windows = sorted(df_raw['wfb_window'].unique())
    for win in windows:
        df_w = df_raw[df_raw['wfb_window'] == win]
        m = calc_metrics(df_w)
        print_metrics(m, f"VENTANA {win}")

# ─────────────────────────────────────────
# 5. MATRIZ CRUZADA: SEED x VENTANA
# ─────────────────────────────────────────
print("\n" + "=" * 65)
print("SECCIÓN C: MATRIZ CRUZADA SEED x VENTANA (Win Rate %)")
print("=" * 65)

if 'wfb_window' in df_raw.columns and 'is_win' in df_raw.columns:
    df_raw['is_win_bool'] = df_raw['is_win'].astype(bool)
    pivot_wr = df_raw.pivot_table(values='is_win_bool', index='seed', columns='wfb_window', aggfunc='mean') * 100
    pivot_n  = df_raw.pivot_table(values='is_win_bool', index='seed', columns='wfb_window', aggfunc='count')
    
    print("\nWin Rate % por Seed x Ventana:")
    print(pivot_wr.round(1).to_string())
    print("\nNúmero de Trades por Seed x Ventana:")
    print(pivot_n.to_string())

# ─────────────────────────────────────────
# 6. ENSEMBLE FINAL (POST EMBARGO)
# ─────────────────────────────────────────
print("\n" + "=" * 65)
print("SECCIÓN D: PORTAFOLIO ENSEMBLE FINAL (POST-EMBARGO)")
print("=" * 65)

m_ens = calc_metrics(df_ensemble, "ENSEMBLE FINAL")
print_metrics(m_ens, "ENSEMBLE FINAL (66 trades post-embargo)")

if 'wfb_window' in df_ensemble.columns:
    print("\nEnsemble por Ventana:")
    for win in sorted(df_ensemble['wfb_window'].unique()):
        df_ew = df_ensemble[df_ensemble['wfb_window'] == win]
        m = calc_metrics(df_ew)
        print_metrics(m, f"ENSEMBLE {win}")

print("\n" + "=" * 65)
print("FIN DEL REPORTE EXHAUSTIVO")
print("=" * 65)
