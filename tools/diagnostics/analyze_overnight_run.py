"""
Script de análisis de la run nocturna 2026-06-02
Objetivo: verificar hipótesis H_BULL_ZERO, H_CALM_BEAR_ACC, H_DTW_W3W4
"""
import pandas as pd
import os
import glob
import numpy as np
from scipy import stats

base = r'g:\Mi unidad\ia\luna_v2\data\predictions'
files = sorted(glob.glob(os.path.join(base, 'oos_trades_seed*.parquet')))
print(f'=== ANÁLISIS RUN NOCTURNA 2026-06-02 ===')
print(f'Total archivos OOS trades encontrados: {len(files)}')
print()

all_trades = []
for f in files:
    try:
        df = pd.read_parquet(f)
        seed = os.path.basename(f).replace('oos_trades_seed','').replace('.parquet','')
        df['seed'] = seed
        all_trades.append(df)
    except Exception as e:
        print(f'ERROR {f}: {e}')

if not all_trades:
    print('No hay trades')
    exit()

combined = pd.concat(all_trades, ignore_index=True)
print(f'Total trades combinados: {len(combined)}')
print(f'Seeds: {combined["seed"].nunique()}')
print(f'Columnas disponibles: {list(combined.columns)}')
print()

# --- WinRate global ---
wr = combined['is_win'].mean() * 100
print(f'=== WINRATE GLOBAL: {wr:.2f}% | N={len(combined)} ===')

# --- Por hmm_regime ---
print('\n--- Breakdown por hmm_regime ---')
if 'hmm_regime' in combined.columns:
    regime_stats = combined.groupby('hmm_regime')['is_win'].agg(['count','mean'])
    regime_stats.columns = ['N', 'WR']
    regime_stats['WR'] = regime_stats['WR'] * 100
    print(regime_stats.to_string())
else:
    print('  [WARNING] columna hmm_regime no encontrada')

# --- Por alpha_trigger (DTW vs no-DTW) ---
print('\n--- Breakdown por alpha_trigger (DTW) ---')
if 'alpha_trigger' in combined.columns:
    atrig_stats = combined.groupby('alpha_trigger')['is_win'].agg(['count','mean'])
    atrig_stats.columns = ['N', 'WR']
    atrig_stats['WR'] = atrig_stats['WR'] * 100
    print(atrig_stats.to_string())
    
    # Separar DTW vs no-DTW para CALM_BEAR específicamente
    if 'hmm_regime' in combined.columns:
        calm_bear = combined[combined['hmm_regime'].str.contains('CALM|calm|bear|BEAR', case=False, na=False)]
        if len(calm_bear) > 0:
            print(f'\n--- CALM_BEAR: DTW vs no-DTW ---')
            dtw_stats = calm_bear.groupby('alpha_trigger')['is_win'].agg(['count','mean'])
            dtw_stats.columns = ['N', 'WR']
            dtw_stats['WR'] = dtw_stats['WR'] * 100
            print(dtw_stats.to_string())
        else:
            print('\n  [INFO] No hay trades CALM_BEAR en esta muestra')
else:
    print('  [WARNING] columna alpha_trigger no encontrada')

# --- Retorno ---
print('\n--- Retornos ---')
r = combined['return_pct']
print(f'Retorno total acumulado: {r.sum():.6f}')
print(f'Retorno medio por trade: {r.mean():.6f}')
print(f'Retorno mediano: {r.median():.6f}')

wins = combined[combined['is_win'] == 1]['return_pct']
losses = combined[combined['is_win'] == 0]['return_pct']
print(f'avg_win:  {wins.mean():.6f}%  (N={len(wins)})')
print(f'avg_loss: {losses.mean():.6f}%  (N={len(losses)})')
if len(losses) > 0 and losses.mean() != 0:
    rr = abs(wins.mean() / losses.mean())
    print(f'R:R ratio: {rr:.3f}')

# --- MaxDrawdown ---
equity = (1 + r).cumprod()
rolling_max = equity.cummax()
drawdowns = (equity - rolling_max) / rolling_max
max_dd = drawdowns.min() * 100
print(f'MaxDrawdown: {max_dd:.2f}%')

# --- Sharpe aproximado ---
if r.std() > 0:
    sharpe = (r.mean() / r.std()) * np.sqrt(252 * 24)  # hourly bars
    print(f'Sharpe anualizado (aprox): {sharpe:.3f}')

# --- Hipótesis H_BULL_ZERO ---
print('\n=== H_BULL_ZERO: ¿BULL = 0 trades? ===')
if 'hmm_regime' in combined.columns:
    bull_trades = combined[combined['hmm_regime'].str.contains('BULL|bull', case=False, na=False)]
    print(f'Trades en regímenes BULL: {len(bull_trades)}')
    if len(bull_trades) == 0:
        print('✅ H_BULL_ZERO CONFIRMADA: 0 trades BULL - Gate DX-BULL-GATE-02 funciona')
    else:
        print(f'❌ H_BULL_ZERO FALLIDA: hay {len(bull_trades)} trades BULL')
        print(bull_trades[['hmm_regime','is_win','return_pct']].head(10))

# --- Hipótesis H_CALM_BEAR_ACC ---
print('\n=== H_CALM_BEAR_ACC: Acumulación CALM_BEAR ===')
if 'hmm_regime' in combined.columns:
    calm = combined[combined['hmm_regime'].str.contains('CALM|3_CALM|calm', case=False, na=False)]
    print(f'Trades CALM_BEAR en esta run: {len(calm)}')
    if len(calm) > 0:
        wr_calm = calm['is_win'].mean() * 100
        print(f'WR CALM_BEAR: {wr_calm:.1f}%')
        # Test binomial
        from scipy.stats import binom_test
        try:
            p = binom_test(int(calm['is_win'].sum()), len(calm), 0.5, alternative='greater')
            print(f'p-value binomial (WR>50%): {p:.5f}')
        except Exception as e:
            n_wins = int(calm['is_win'].sum())
            p = stats.binomtest(n_wins, len(calm), 0.5, alternative='greater').pvalue
            print(f'p-value binomial (WR>50%): {p:.5f}')

# --- Por ventana si hay info ---
print('\n--- Distribución por seed (muestra) ---')
seed_stats = combined.groupby('seed')['is_win'].agg(['count','mean'])
seed_stats.columns = ['N', 'WR']
seed_stats['WR'] = seed_stats['WR'] * 100
print(seed_stats.to_string())

print('\n=== FIN ANÁLISIS ===')
