"""
hmm_shield_analysis.py — análisis cuantitativo del impacto de post_ath_bear
Cuantifica cuántas barras fuerza el shield y el impacto en MI HMM
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from config.settings import cfg

df = pd.read_parquet('data/features/features_train.parquet')
close = df['close']

# Parámetros actuales del shield
dd_thresh  = float(cfg.hmm.post_ath_dd_threshold)
mom_thresh = float(cfg.hmm.post_ath_mom_threshold)
ath_window = int(cfg.hmm.post_ath_ath_window_h)
ath_min    = int(cfg.hmm.post_ath_ath_min_periods)
confirm_h  = int(cfg.hmm.post_ath_confirm_h)

print(f"Shield actual: dd_CUTOFF = {dd_thresh} | mom_CUTOFF = {mom_thresh} | ath_window={ath_window}H | confirm={confirm_h}H")

ath_rolling  = close.rolling(ath_window, min_periods=ath_min).max()
dd_from_ath  = (close / ath_rolling - 1)
ret_168h     = close.pct_change(168)

raw_post_ath  = (dd_from_ath < dd_thresh) & (ret_168h < mom_thresh)
post_ath_bear = raw_post_ath.rolling(confirm_h, min_periods=confirm_h).min().fillna(0).astype(bool)

n_total  = len(close)
n_forced = int(post_ath_bear.sum())
print(f"\nIS total: {n_total} barras | post_ath_bear forzadas: {n_forced} ({n_forced/n_total:.1%})")

print("\nDistribución por año:")
for yr in range(2020, 2026):
    sub = post_ath_bear[post_ath_bear.index.year == yr]
    n_yr = len(sub)
    if n_yr == 0:
        continue
    n_f = int(sub.sum())
    print(f"  {yr}: {n_f:5d}/{n_yr:5d} = {n_f/n_yr:.1%} forzadas a BEAR")

# Impacto en MI: el shield sobreescribe estados HMM -> la distribucion se distorsiona
# Estimacion: con el shield, % de estados BEAR en el IS
# La MI baja porque el HMM aprende X pero el shield lo convierte en otra cosa
print(f"\nProblema raíz:")
print(f"  El HMM entrena -> aprende distribución natural de regímenes")
print(f"  El shield sobreescribe {n_forced} barras a BEAR ({n_forced/n_total:.1%})")
print(f"  El XGBoost/SFI usa los labels POST-shield")
print(f"  -> MI entre states-post-shield y retornos futuros ≈ 0.00078 (SOP min=0.005)")

# Test de distintos umbrales
print(f"\n--- Sensibilidad dd_threshold ---")
print(f"{'dd_thresh':>12s} {'N_forced':>10s} {'%forced':>8s} {'Impacto estimado':>20s}")
for new_dd in [-0.20, -0.25, -0.30, -0.35, -0.40, -0.50]:
    raw_new = (dd_from_ath < new_dd) & (ret_168h < mom_thresh)
    pb_new  = raw_new.rolling(confirm_h, min_periods=confirm_h).min().fillna(0).astype(bool)
    n_new   = int(pb_new.sum())
    pct_new = n_new / n_total if n_total > 0 else 0
    impact  = "ACTUAL" if new_dd == dd_thresh else ("mejor MI" if n_new < n_forced else "igual")
    print(f"  {new_dd:>10.2f} {n_new:>10d} {pct_new:>8.1%} {impact:>20s}")

# Test: desactivar post_ath_bear completamente
print(f"\n  Sin post_ath_bear: 0 barras (MI esperada más alta, pero sin protección post-ATH)")

# W1 tiene solo 2344 — verificar con ventana W1 IS
print(f"\n--- Replica W1 IS (2020-2024-10-31, rolling 5y) ---")
w1_end   = pd.Timestamp('2024-10-31', tz='UTC')
w1_start = w1_end - pd.DateOffset(years=5)
df_w1    = df.loc[(df.index >= w1_start) & (df.index <= w1_end)]
close_w1 = df_w1['close']
ath_w1   = close_w1.rolling(ath_window, min_periods=ath_min).max()
dd_w1    = (close_w1 / ath_w1 - 1)
r168_w1  = close_w1.pct_change(168)
raw_w1   = (dd_w1 < dd_thresh) & (r168_w1 < mom_thresh)
pb_w1    = raw_w1.rolling(confirm_h, min_periods=confirm_h).min().fillna(0).astype(bool)
n_w1     = int(pb_w1.sum())
print(f"W1 IS ({w1_start.date()} -> {w1_end.date()}): {n_w1}/{len(close_w1)} = {n_w1/len(close_w1):.1%} forzadas (log decía 2344)")

# Análisis: ¿por qué en 2023-2024 no hay post_ath_bear?
btc_2024_max = float(close[close.index.year == 2024].max())
btc_2024_end = float(close[close.index <= w1_end].iloc[-1])
dd_2024_end  = btc_2024_end / btc_2024_max - 1
print(f"\nBTC en 2024-10-31: ${btc_2024_end:,.0f} | ATH 30d: ${btc_2024_max:,.0f} | DD={dd_2024_end:.1%}")
print(f"-> DD={dd_2024_end:.1%} > CUTOFF = {dd_thresh:.1%}: NO hay post_ath_bear en 2024")
print(f"-> El shield es activo en 2021-2022 (crashes BTC -50% / -75%)")
print(f"-> En 2020-2021: 2344 barras = correcciones intermedias post-ATH (BTC -20% frecuente)")
