"""
alpha_rules.py — GENERADO AUTOMÁTICAMENTE por export_alpha_rules.py
Timestamp: 2026-06-05 06:24 UTC
Golden Rules: 15  |  Genetic Rules: 7
DO NOT EDIT MANUALLY — se sobreescribe semanalmente con run_weekly_mining.py
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────
# GOLDEN STORM RULES (Master Pattern Engine)
# ──────────────────────────────────────────────────────────────────

GOLDEN_RULES: list[dict] = [
    {
        'type':        'golden_storm',
        'pandas_eval': '(VIX <= 15.0025) & (NASDAQ_Ret >= 0.0102) & (Master_Causal_Signal >= 0.2333)',
        'win_rate':    100.0,
        'ev_pct':      4.05,
        'description': 'IF VIX <= 15.0025 AND NASDAQ_Ret >= 0.0102 AND Master_Causal_Signal >= 0.2333',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(Fed_Net_Liquidity <= 6516689.6270) & (DangerZone >= 0.6075) & (MVRV_Proxy <= -0.9473)',
        'win_rate':    100.0,
        'ev_pct':      2.21,
        'description': 'IF Fed_Net_Liquidity <= 6516689.6270 AND DangerZone >= 0.6075 AND MVRV_Proxy <= -0.9473',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(DangerZone >= 0.6075) & (MVRV_Proxy <= -0.9473) & (KMeans_Tribe_ID <= 1.0000)',
        'win_rate':    100.0,
        'ev_pct':      2.21,
        'description': 'IF DangerZone >= 0.6075 AND MVRV_Proxy <= -0.9473 AND KMeans_Tribe_ID <= 1.0000',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(FedFundsRate >= 5.3300) & (DangerZone >= 0.6075) & (MVRV_Proxy <= -0.9473)',
        'win_rate':    100.0,
        'ev_pct':      2.43,
        'description': 'IF FedFundsRate >= 5.3300 AND DangerZone >= 0.6075 AND MVRV_Proxy <= -0.9473',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(Fed_Net_Liquidity <= 6516689.6270) & (WEI <= 1.5600) & (MVRV_Proxy <= -0.9473)',
        'win_rate':    100.0,
        'ev_pct':      28.45,
        'description': 'IF Fed_Net_Liquidity <= 6516689.6270 AND WEI <= 1.5600 AND MVRV_Proxy <= -0.9473',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(Fed_Net_Liquidity <= 6516689.6270) & (SP500_AboveMA200 <= 0.0000) & (MVRV_Proxy <= -0.9473)',
        'win_rate':    100.0,
        'ev_pct':      28.45,
        'description': 'IF Fed_Net_Liquidity <= 6516689.6270 AND SP500_AboveMA200 <= 0.0000 AND MVRV_Proxy <= -0.9473',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(Fed_Net_Liquidity <= 6516689.6270) & (MVRV_Proxy <= -0.9473) & (SSR <= 1.6093)',
        'win_rate':    100.0,
        'ev_pct':      28.45,
        'description': 'IF Fed_Net_Liquidity <= 6516689.6270 AND MVRV_Proxy <= -0.9473 AND SSR <= 1.6093',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(WEI <= 1.5600) & (MVRV_Proxy <= -0.9473) & (Master_Causal_Signal >= 0.2333)',
        'win_rate':    100.0,
        'ev_pct':      28.45,
        'description': 'IF WEI <= 1.5600 AND MVRV_Proxy <= -0.9473 AND Master_Causal_Signal >= 0.2333',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(Fed_Net_Liquidity <= 6516689.6270) & (FundingRate >= 0.0001) & (MVRV_Proxy <= -0.9473)',
        'win_rate':    100.0,
        'ev_pct':      2.59,
        'description': 'IF Fed_Net_Liquidity <= 6516689.6270 AND FundingRate >= 0.0001 AND MVRV_Proxy <= -0.9473',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(FundingRate >= 0.0001) & (MVRV_Proxy <= -0.9473) & (KMeans_Tribe_ID <= 1.0000)',
        'win_rate':    100.0,
        'ev_pct':      2.59,
        'description': 'IF FundingRate >= 0.0001 AND MVRV_Proxy <= -0.9473 AND KMeans_Tribe_ID <= 1.0000',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(Fed_Net_Liquidity <= 6516689.6270) & (MVRV_Proxy <= -0.9473) & (Whale_Vol_ZScore <= -0.8373)',
        'win_rate':    98.8,
        'ev_pct':      5.89,
        'description': 'IF Fed_Net_Liquidity <= 6516689.6270 AND MVRV_Proxy <= -0.9473 AND Whale_Vol_ZScore <= -0.8373',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(VIX <= 15.0025) & (Whale_Vol_ZScore <= -0.8373) & (Master_Causal_Signal >= 0.2333)',
        'win_rate':    98.4,
        'ev_pct':      0.27,
        'description': 'IF VIX <= 15.0025 AND Whale_Vol_ZScore <= -0.8373 AND Master_Causal_Signal >= 0.2333',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(WEI <= 1.5600) & (SSR <= 1.6093) & (KMeans_Tribe_ID <= 1.0000)',
        'win_rate':    98.2,
        'ev_pct':      13.76,
        'description': 'IF WEI <= 1.5600 AND SSR <= 1.6093 AND KMeans_Tribe_ID <= 1.0000',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(FedFundsRate >= 5.3300) & (MVRV_Proxy <= -0.9473) & (Whale_Vol_ZScore <= -0.8373)',
        'win_rate':    97.9,
        'ev_pct':      2.49,
        'description': 'IF FedFundsRate >= 5.3300 AND MVRV_Proxy <= -0.9473 AND Whale_Vol_ZScore <= -0.8373',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(VIX <= 15.0025) & (DVOL <= 45.4000) & (Whale_Vol_ZScore <= -0.8373)',
        'win_rate':    96.9,
        'ev_pct':      1.13,
        'description': 'IF VIX <= 15.0025 AND DVOL <= 45.4000 AND Whale_Vol_ZScore <= -0.8373',
    },
]

# ──────────────────────────────────────────────────────────────────
# GENETIC RULES (Deep Discovery Engine — AG 20 generaciones)
# ──────────────────────────────────────────────────────────────────

GENETIC_RULES: list[dict] = [
    {
        'type':        'genetic_rule',
        'pandas_eval': '(Master_Causal_Signal >= 0.3232) & (CPI_YoY <= 1.3738)',
        'win_rate':    73.0,
        'ev_pct':      1.32,
        'description': '`Master_Causal_Signal >= 0.3232`<br>**AND** `CPI_YoY <= 1.3738`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(CPI_YoY <= 1.3738) & (Master_Causal_Signal >= 0.3232)',
        'win_rate':    73.0,
        'ev_pct':      1.32,
        'description': '`CPI_YoY <= 1.3738`<br>**AND** `Master_Causal_Signal >= 0.3232`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.5492) & (CPI_YoY <= 1.9281) & (FearGreed <= 20.0000)',
        'win_rate':    69.5,
        'ev_pct':      1.58,
        'description': '`SSR <= 0.5492`<br>**AND** `CPI_YoY <= 1.9281`<br>**AND** `FearGreed <= 20.0000`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(FearGreed <= 20.0000) & (CPI_YoY <= 1.9281)',
        'win_rate':    68.4,
        'ev_pct':      1.32,
        'description': '`FearGreed <= 20.0000`<br>**AND** `CPI_YoY <= 1.9281`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(CPI_YoY <= 1.3738)',
        'win_rate':    63.8,
        'ev_pct':      0.76,
        'description': '`CPI_YoY <= 1.3738`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.3900)',
        'win_rate':    63.0,
        'ev_pct':      0.4,
        'description': '`SSR <= 0.3900`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(FearGreed >= 75.0000) & (CPI_YoY <= 1.9281)',
        'win_rate':    62.7,
        'ev_pct':      0.82,
        'description': '`FearGreed >= 75.0000`<br>**AND** `CPI_YoY <= 1.9281`',
    },
]

# ──────────────────────────────────────────────────────────────────
# CAUSAL VARIABLES (Advanced Engine — Granger*** + TE_net > 0)
# ──────────────────────────────────────────────────────────────────

CAUSAL_VARS: list[str] = ['SSR', 'DeFi_WBTC_TVL', 'KMeans_Tribe_ID', 'FearGreed', 'MVRV_Proxy', 'FundingRate', 'Stablecoin_Cap', 'DangerZone', 'CPI_YoY', 'Master_Causal_Signal']

# ──────────────────────────────────────────────────────────────────
# DTW FRACTAL PROBABILITY
# ──────────────────────────────────────────────────────────────────

DTW_BULL_PROB: float = 0.4  # P(BTC sube en 24H | análogos históricos)

# ──────────────────────────────────────────────────────────────────
# K-MEANS TRIBE BIAS
# ──────────────────────────────────────────────────────────────────

TRIBE_BIAS: dict[int, str] = {0: 'LARGA', 1: 'NEUTRAL', 2: 'NEUTRAL', 3: 'NEUTRAL'}

# ──────────────────────────────────────────────────────────────────
# K-MEANS TRIBE WIN-RATE MAP (M3 — actualizado semanalmente)
# ──────────────────────────────────────────────────────────────────

TRIBE_WR_MAP: dict[int, float] = {0: 0.647, 1: 0.526, 2: 0.526, 3: 0.447}

LARGA_TRIBES   = frozenset({0})
NEUTRAL_TRIBES = frozenset({1, 2, 3})

ALL_RULES: list[dict] = GOLDEN_RULES + GENETIC_RULES

# ──────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL — Cálculo de señales alpha
# ──────────────────────────────────────────────────────────────────

def get_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula señales alpha a partir de las reglas descubiertas por AI Mining.
    Llamada en el paso 9 de feature_pipeline.py.

    Args:
        df: DataFrame con todas las features crudas (post-lag, post-zscore)

    Returns:
        df con columnas adicionales:
          - alpha_golden_score  : suma de Golden Rules activas (0 a N)
          - alpha_genetic_score : suma de Genetic Rules activas (0 a N)
          - alpha_combined      : señal combinada [-1, 1]
          - alpha_dtw_signal    : señal DTW dinámica: dtw_direction × tanh(mom_24H × 20) ∈ [-1,1]
          - alpha_tribe_bias    : sesgo de tribu (-1/0/1)
    """
    df = df.copy()

    # ── Golden Storm Score ──────────────────────────────────────────
    golden_score = pd.Series(0.0, index=df.index)
    for rule in GOLDEN_RULES:
        try:
            mask = df.eval(rule['pandas_eval'], engine='python')
            golden_score += mask.astype(float) * (rule['win_rate'] / 100.0)
        except Exception:
            pass
    # Normalizar al número de reglas activas máximo
    if GOLDEN_RULES:
        golden_score /= len(GOLDEN_RULES)
    df['alpha_golden_score'] = golden_score.clip(0, 1)

    # ── Genetic Score ───────────────────────────────────────────────
    genetic_score = pd.Series(0.0, index=df.index)
    for rule in GENETIC_RULES:
        try:
            mask = df.eval(rule['pandas_eval'], engine='python')
            genetic_score += mask.astype(float) * (rule['win_rate'] / 100.0)
        except Exception:
            pass
    if GENETIC_RULES:
        genetic_score /= len(GENETIC_RULES)
    df['alpha_genetic_score'] = genetic_score.clip(0, 1)

    # ── Combined Alpha ──────────────────────────────────────────────
    # Ponderado: Golden 60% | Genetic 40%
    combined = 0.6 * df['alpha_golden_score'] + 0.4 * df['alpha_genetic_score']
    # Centrar en 0: [0,1] → [-1,1]
    df['alpha_combined'] = (combined * 2 - 1).clip(-1, 1)

    # ── DTW Fractal Signal (dinámica horaria) ────────────────────────
    # 1. Dirección del bias DTW: +1 si bull, -1 si bear
    # 2. Momentum a 24H del precio actual como amplitud dinámica
    # tanh escala el momentum a [-1,1] de forma suave
    dtw_direction = 1.0 if DTW_BULL_PROB >= 0.5 else -1.0
    if 'close' in df.columns:
        mom_24h = df['close'].pct_change(24).fillna(0.0)
    else:
        mom_24h = pd.Series(0.0, index=df.index)
    df['alpha_dtw_signal'] = (dtw_direction * np.tanh(mom_24h * 20)).clip(-1, 1)

    # ── Tribe Bias ──────────────────────────────────────────────────
    # Soporta K_Shape_Cluster_ID (kshape_engine) y KMeans_Tribe_ID (cluster_pattern_engine)
    bias_map = {
        tid: (1 if bias == 'LARGA' else -1 if bias == 'CORTA' else 0)
        for tid, bias in TRIBE_BIAS.items()
    }
    # P3-3-FIX: KMeans_Tribe_ID es la columna primaria (K_Shape decommisionado)
    _tribe_col = None
    if 'KMeans_Tribe_ID' in df.columns:
        _tribe_col = 'KMeans_Tribe_ID'
    elif 'K_Shape_Cluster_ID' in df.columns:  # legacy fallback solo
        _tribe_col = 'K_Shape_Cluster_ID'
    if _tribe_col:
        df['alpha_tribe_bias'] = df[_tribe_col].map(bias_map).fillna(0).astype(float)
    else:
        df['alpha_tribe_bias'] = 0.0

    # ── [PASS-THROUGH] Golden/Genetic Rule binaries ─────────────────
    # Genera golden_rule_0..N y genetic_rule_0..M como columnas 0/1.
    # Son pass-through: bypasan SFI y van directo a XGBoost.
    # COL ALIASES: mapea nombres del Mining a nombres en features_train.
    _COL_ALIASES = {
        # Yield Curve: Mining usa nombre FRED, features_train usa nombre propio
        'YieldCurve_10Y3M': 'yield_curve_spread',
        'T10Y2Y':           'yield_curve_spread',
        # Onchain: active_addresses_7d_ma -> ActiveAddresses_7d
        'active_addresses_7d_ma': 'ActiveAddresses_7d',
        # SSR: Mining usa 'SSR' (raw), features_train tiene 'SSR_ZScore'
        'SSR':              'SSR_ZScore',
    }
    _df2 = df.copy()
    for _orig, _alias in _COL_ALIASES.items():
        if _orig not in _df2.columns and _alias in _df2.columns:
            _df2[_orig] = _df2[_alias]
    # FearGreed: Mining usa escala raw 0-100; pipeline lo normaliza a 0-1.
    # Convertimos FearGreed_Normalized -> FearGreed (0-100) para que
    # los thresholds del Mining (ej: 84) sean comparables.
    if 'FearGreed' not in _df2.columns and 'FearGreed_Normalized' in _df2.columns:
        _df2['FearGreed'] = _df2['FearGreed_Normalized'] * 100.0
    # NASDAQ_Ret: Mining usa retorno porcentual de NASDAQ 1H.
    # features_train tiene 'NASDAQ' como precio raw -> calculamos ret.
    if 'NASDAQ_Ret' not in _df2.columns and 'NASDAQ' in _df2.columns:
        _df2['NASDAQ_Ret'] = _df2['NASDAQ'].pct_change(1).fillna(0.0)
    # Whale_Vol_ZScore: proxy basado en Tx_Volume (BTC en cadena) si disponible,
    # o volume OHLCV como fallback.
    if 'Whale_Vol_ZScore' not in _df2.columns:
        _vsrc = _df2['Tx_Volume'] if 'Tx_Volume' in _df2.columns else _df2.get('volume', None)
        if _vsrc is not None:
            _vol = _vsrc.replace(0, float('nan'))
            _roll = _vol.rolling(window=90*24, min_periods=720)
            _df2['Whale_Vol_ZScore'] = ((_vol - _roll.mean()) / _roll.std().replace(0, 1)).ffill().fillna(0).clip(-4, 4)
    for _i, _rule in enumerate(GOLDEN_RULES):
        _col = f'golden_rule_{_i}'
        try:
            _mask = _df2.eval(_rule['pandas_eval'], engine='python')
            df[_col] = _mask.astype(float).fillna(0.0)
        except Exception:
            df[_col] = 0.0
    for _i, _rule in enumerate(GENETIC_RULES):
        _col = f'genetic_rule_{_i}'
        try:
            _mask = _df2.eval(_rule['pandas_eval'], engine='python')
            df[_col] = _mask.astype(float).fillna(0.0)
        except Exception:
            df[_col] = 0.0

    return df


def get_rule_summary() -> dict:
    """Resumen de reglas disponibles para logging/monitoring."""
    return {
        'golden_rules':    len(GOLDEN_RULES),
        'genetic_rules':   len(GENETIC_RULES),
        'causal_vars':     len(CAUSAL_VARS),
        'dtw_bull_prob':   DTW_BULL_PROB,
        'tribe_bias_keys': list(TRIBE_BIAS.keys()),
        'tribe_wr_map':    TRIBE_WR_MAP,
    }


# ──────────────────────────────────────────────────────────────────
# MEJORA M3 — Features derivadas de Tribe para XGBoost
# Llamada en feature_pipeline.py paso 9B
# ──────────────────────────────────────────────────────────────────

def apply_tribe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    M3: Convierte KMeans_Tribe_ID/K_Shape_Cluster_ID en features numéricas:
      - tribe_wr_historical : WR histórico continuo (0.0-1.0)
      - tribe_in_larga      : 1.0 si tribu LARGA, 0.0 si no
      - tribe_wr_zscore     : Z-Score rolling 90d del WR
    """
    tribe_col = None
    # P3-3-FIX: KMeans_Tribe_ID es la columna primaria (K_Shape decommisionado)
    if 'KMeans_Tribe_ID' in df.columns:
        tribe_col = 'KMeans_Tribe_ID'
    elif 'K_Shape_Cluster_ID' in df.columns:  # legacy fallback solo
        tribe_col = 'K_Shape_Cluster_ID'
    if tribe_col is None:
        return df
    df = df.copy()
    df['tribe_wr_historical'] = df[tribe_col].map(TRIBE_WR_MAP).fillna(0.5)
    df['tribe_in_larga'] = df[tribe_col].isin(LARGA_TRIBES).astype(float)
    _wr_roll = df['tribe_wr_historical'].rolling(window=90 * 24, min_periods=24)
    df['tribe_wr_zscore'] = (
        (df['tribe_wr_historical'] - _wr_roll.mean()) /
        (_wr_roll.std().replace(0, 1))
    ).clip(-3, 3)
    return df
