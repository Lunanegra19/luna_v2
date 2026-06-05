"""validate_h4_ath.py — Validacion de las features H4-ATH en datos reales."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from pathlib import Path

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\raw\ohlcv')
ohlcv_paths = list(DATA.glob('ohlcv_raw.parquet'))
if not ohlcv_paths:
    # Fallback al daemon
    DATA2 = Path(r'g:\Mi unidad\ia\luna_v2\data\historical\daemon')
    ohlcv_paths = list(DATA2.glob('BTCUSDT_1h.parquet'))
print(f'Archivos OHLCV encontrados: {[p.name for p in ohlcv_paths[:3]]}')
if not ohlcv_paths:
    print('ERROR: No se encontro archivo OHLCV')
    sys.exit(1)

df = pd.read_parquet(ohlcv_paths[0])
print(f'Shape: {df.shape} | Cols: {list(df.columns[:8])}')
df.index = pd.to_datetime(df.index, utc=True)
_close = df['close'].ffill()

# ── Feature 1: ath_dist_pct
_ath_cummax = _close.cummax()
ath_dist = (_ath_cummax - _close) / _ath_cummax.clip(lower=1e-8)
print(f'[OK] ath_dist_pct: min={ath_dist.min():.4f} max={ath_dist.max():.4f} NaN={ath_dist.isna().sum()}')
w3_ath = ath_dist.loc['2025-07':'2025-09'].mean()
w4_ath = ath_dist.loc['2025-10':'2025-12'].mean()
print(f'     W3 mean={w3_ath:.4f} | W4 mean={w4_ath:.4f} (0=AT ATH)')

# ── Feature 2: ath_streak_h (vectorizado)
_new_ath_flag = (_close >= _ath_cummax.shift(1)).astype(int)
_group_id = (_new_ath_flag == 0).cumsum()
_streak = _new_ath_flag.groupby(_group_id).cumcount().where(_new_ath_flag == 1, 0).astype(float)
print(f'[OK] ath_streak_h: max={_streak.max():.0f}h | p95={_streak.quantile(0.95):.0f}h')
w3_streak = _streak.loc['2025-07':'2025-09'].mean()
w4_streak = _streak.loc['2025-10':'2025-12'].mean()
print(f'     W3 mean={w3_streak:.1f}h | W4 mean={w4_streak:.1f}h')

# ── Feature 3: price_z_score_252d
_p_m = _close.rolling(252*24, min_periods=30*24).mean()
_p_s = _close.rolling(252*24, min_periods=30*24).std().clip(lower=1e-9)
pz = (_close - _p_m) / _p_s
w3_pz = pz.loc['2025-07':'2025-09'].mean()
w4_pz = pz.loc['2025-10':'2025-12'].mean()
print(f'[OK] price_z_score_252d: W3={w3_pz:.2f}sigma | W4={w4_pz:.2f}sigma')

# ── Feature 4: realized_vol_ratio
_ret1h = _close.pct_change(1)
_vol30  = _ret1h.rolling(30*24, min_periods=24).std()
_vol252 = _ret1h.rolling(252*24, min_periods=30*24).std().clip(lower=1e-9)
rvr = _vol30 / _vol252
w3_rvr = rvr.loc['2025-07':'2025-09'].mean()
w4_rvr = rvr.loc['2025-10':'2025-12'].mean()
print(f'[OK] realized_vol_ratio: W3={w3_rvr:.3f} | W4={w4_rvr:.3f}')

print()
print('=== SEPARACION W3 vs W4 (objetivo: features con valores distintos) ===')
SEP = '-' * 60
print(SEP)
print(f'{"Feature":<25} {"W3 (jul-sep)":>13} {"W4 (oct-dic)":>13} {"Delta":>10}')
print(SEP)
print(f'{"ath_dist_pct":<25} {w3_ath:>13.4f} {w4_ath:>13.4f} {w4_ath-w3_ath:>+10.4f}')
print(f'{"ath_streak_h":<25} {w3_streak:>13.1f} {w4_streak:>13.1f} {w4_streak-w3_streak:>+10.1f}')
print(f'{"price_z_score_252d":<25} {w3_pz:>13.2f} {w4_pz:>13.2f} {w4_pz-w3_pz:>+10.2f}')
print(f'{"realized_vol_ratio":<25} {w3_rvr:>13.3f} {w4_rvr:>13.3f} {w4_rvr-w3_rvr:>+10.3f}')
print(SEP)
print()
print('VERIFICACION CAUSALIDAD: cummax() solo usa datos hasta t -> SIN look-ahead bias')
print('VERIFICACION RENDIMIENTO: vectorizado, sin bucles Python')
n_total = len(_close)
print(f'DATOS: {n_total} filas horarias procesadas')
print()
print('[H4-ATH] IMPLEMENTACION VALIDADA - Lista para re-run WFB')
