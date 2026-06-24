"""
alpha_rules.py — GENERADO AUTOMÁTICAMENTE por export_alpha_rules.py
Timestamp: 2026-06-24 00:08 UTC
Golden Rules: 15  |  Genetic Rules: 10
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
        'pandas_eval': '(VIX >= 27.4700) & (NASDAQ_Ret >= 0.0095) & (FearGreed >= 56.0000)',
        'win_rate':    97.7,
        'ev_pct':      7.42,
        'description': 'IF VIX >= 27.4700 AND NASDAQ_Ret >= 0.0095 AND FearGreed >= 56.0000',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(VIX >= 27.4700) & (DXY <= 93.3300) & (NASDAQ_Ret >= 0.0095)',
        'win_rate':    94.6,
        'ev_pct':      8.99,
        'description': 'IF VIX >= 27.4700 AND DXY <= 93.3300 AND NASDAQ_Ret >= 0.0095',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(VIX >= 27.4700) & (Whale_Vol_ZScore <= -0.8164) & (Master_Causal_Signal >= 0.4464)',
        'win_rate':    88.1,
        'ev_pct':      4.81,
        'description': 'IF VIX >= 27.4700 AND Whale_Vol_ZScore <= -0.8164 AND Master_Causal_Signal >= 0.4464',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(T10Y2Y >= 0.7900) & (VIX >= 27.4700) & (MVRV_Proxy >= 1.3782)',
        'win_rate':    83.5,
        'ev_pct':      9.39,
        'description': 'IF T10Y2Y >= 0.7900 AND VIX >= 27.4700 AND MVRV_Proxy >= 1.3782',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(VIX >= 27.4700) & (DXY <= 93.3300) & (MVRV_Proxy >= 1.3782)',
        'win_rate':    81.9,
        'ev_pct':      6.73,
        'description': 'IF VIX >= 27.4700 AND DXY <= 93.3300 AND MVRV_Proxy >= 1.3782',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(DXY <= 93.3300) & (NASDAQ_Ret >= 0.0095) & (Master_Causal_Signal >= 0.4464)',
        'win_rate':    80.6,
        'ev_pct':      3.86,
        'description': 'IF DXY <= 93.3300 AND NASDAQ_Ret >= 0.0095 AND Master_Causal_Signal >= 0.4464',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(T10Y2Y >= 0.7900) & (NASDAQ_Ret >= 0.0095) & (MVRV_Proxy >= 1.3782)',
        'win_rate':    80.3,
        'ev_pct':      6.33,
        'description': 'IF T10Y2Y >= 0.7900 AND NASDAQ_Ret >= 0.0095 AND MVRV_Proxy >= 1.3782',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(DXY <= 93.3300) & (NASDAQ_Ret >= 0.0095) & (MVRV_Proxy >= 1.3782)',
        'win_rate':    76.3,
        'ev_pct':      4.85,
        'description': 'IF DXY <= 93.3300 AND NASDAQ_Ret >= 0.0095 AND MVRV_Proxy >= 1.3782',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(VIX >= 27.4700) & (MVRV_Proxy >= 1.3782) & (FearGreed >= 56.0000)',
        'win_rate':    76.3,
        'ev_pct':      3.92,
        'description': 'IF VIX >= 27.4700 AND MVRV_Proxy >= 1.3782 AND FearGreed >= 56.0000',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(VIX >= 27.4700) & (SP500_AboveMA200 >= 1.0000) & (MVRV_Proxy >= 1.3782)',
        'win_rate':    75.9,
        'ev_pct':      3.94,
        'description': 'IF VIX >= 27.4700 AND SP500_AboveMA200 >= 1.0000 AND MVRV_Proxy >= 1.3782',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(Fed_Net_Liquidity <= 5730394.0000) & (NASDAQ_Ret >= 0.0095) & (FearGreed >= 56.0000)',
        'win_rate':    72.3,
        'ev_pct':      1.56,
        'description': 'IF Fed_Net_Liquidity <= 5730394.0000 AND NASDAQ_Ret >= 0.0095 AND FearGreed >= 56.0000',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(NASDAQ_Ret >= 0.0095) & (FearGreed >= 56.0000) & (Master_Causal_Signal >= 0.4464)',
        'win_rate':    72.1,
        'ev_pct':      3.96,
        'description': 'IF NASDAQ_Ret >= 0.0095 AND FearGreed >= 56.0000 AND Master_Causal_Signal >= 0.4464',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(NASDAQ_Ret >= 0.0095) & (MVRV_Proxy >= 1.3782) & (Master_Causal_Signal >= 0.4464)',
        'win_rate':    71.1,
        'ev_pct':      3.49,
        'description': 'IF NASDAQ_Ret >= 0.0095 AND MVRV_Proxy >= 1.3782 AND Master_Causal_Signal >= 0.4464',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(DXY <= 93.3300) & (NASDAQ_Ret >= 0.0095) & (SSR <= 0.9505)',
        'win_rate':    71.0,
        'ev_pct':      0.44,
        'description': 'IF DXY <= 93.3300 AND NASDAQ_Ret >= 0.0095 AND SSR <= 0.9505',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(Fed_Net_Liquidity <= 5730394.0000) & (DXY <= 93.3300) & (NASDAQ_Ret >= 0.0095)',
        'win_rate':    70.8,
        'ev_pct':      1.58,
        'description': 'IF Fed_Net_Liquidity <= 5730394.0000 AND DXY <= 93.3300 AND NASDAQ_Ret >= 0.0095',
    },
]

# ──────────────────────────────────────────────────────────────────
# GENETIC RULES (Deep Discovery Engine — AG 20 generaciones)
# ──────────────────────────────────────────────────────────────────

GENETIC_RULES: list[dict] = [
    {
        'type':        'genetic_rule',
        'pandas_eval': '(Tx_Fees_USD <= 197567.9614) & (SSR <= 0.6276) & (NASDAQ_Ret <= -0.0059)',
        'win_rate':    85.7,
        'ev_pct':      0.75,
        'description': '`Tx_Fees_USD <= 197567.9614`<br>**AND** `SSR <= 0.6276`<br>**AND** `NASDAQ_Ret <= -0.0059`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.6276) & (Tx_Fees_USD <= 197567.9614) & (NASDAQ_Ret <= -0.0059)',
        'win_rate':    85.7,
        'ev_pct':      0.75,
        'description': '`SSR <= 0.6276`<br>**AND** `Tx_Fees_USD <= 197567.9614`<br>**AND** `NASDAQ_Ret <= -0.0059`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(T10Y2Y <= -0.1000) & (Tx_Fees_USD <= 197567.9614)',
        'win_rate':    70.8,
        'ev_pct':      0.9,
        'description': '`T10Y2Y <= -0.1000`<br>**AND** `Tx_Fees_USD <= 197567.9614`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(Tx_Fees_USD <= 197567.9614) & (T10Y2Y <= -0.1000)',
        'win_rate':    70.8,
        'ev_pct':      0.9,
        'description': '`Tx_Fees_USD <= 197567.9614`<br>**AND** `T10Y2Y <= -0.1000`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.4166) & (YieldCurve_10Y3M >= 1.0800) & (SSR <= 0.4166)',
        'win_rate':    66.0,
        'ev_pct':      0.93,
        'description': '`SSR <= 0.4166`<br>**AND** `YieldCurve_10Y3M >= 1.0800`<br>**AND** `SSR <= 0.4166`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(Master_Causal_Signal <= -0.1621) & (NASDAQ_Ret >= 0.0088) & (FearGreed <= 27.0000)',
        'win_rate':    64.1,
        'ev_pct':      1.14,
        'description': '`Master_Causal_Signal <= -0.1621`<br>**AND** `NASDAQ_Ret >= 0.0088`<br>**AND** `FearGreed <= 27.0000`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(Master_Causal_Signal <= -0.1621) & (NASDAQ_Ret >= 0.0088) & (MVRV_Proxy <= -0.8954)',
        'win_rate':    62.6,
        'ev_pct':      1.01,
        'description': '`Master_Causal_Signal <= -0.1621`<br>**AND** `NASDAQ_Ret >= 0.0088`<br>**AND** `MVRV_Proxy <= -0.8954`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.6276) & (T10Y2Y >= 0.5600)',
        'win_rate':    62.2,
        'ev_pct':      0.83,
        'description': '`SSR <= 0.6276`<br>**AND** `T10Y2Y >= 0.5600`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.6276) & (Tx_Fees_USD <= 197567.9614)',
        'win_rate':    62.1,
        'ev_pct':      0.31,
        'description': '`SSR <= 0.6276`<br>**AND** `Tx_Fees_USD <= 197567.9614`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(Tx_Fees_USD <= 197567.9614)',
        'win_rate':    59.7,
        'ev_pct':      0.26,
        'description': '`Tx_Fees_USD <= 197567.9614`',
    },
]

# ──────────────────────────────────────────────────────────────────
# CAUSAL VARIABLES (Advanced Engine — Granger*** + TE_net > 0)
# ──────────────────────────────────────────────────────────────────

CAUSAL_VARS: list[str] = ['SSR', 'DeFi_WBTC_TVL', 'MVRV_Proxy', 'FearGreed', 'Stablecoin_Cap', 'Master_Causal_Signal', 'NASDAQ_Ret', 'DVOL', 'FundingRate', 'DangerZone']

# ──────────────────────────────────────────────────────────────────
# DTW FRACTAL PROBABILITY
# ──────────────────────────────────────────────────────────────────

DTW_BULL_PROB: float = 0.6  # P(BTC sube en 24H | análogos históricos)

# ──────────────────────────────────────────────────────────────────
# K-MEANS TRIBE BIAS
# ──────────────────────────────────────────────────────────────────

TRIBE_BIAS: dict[int, str] = {0: 'LARGA', 2: 'NEUTRAL', 1: 'NEUTRAL', 3: 'CORTA'}

# ──────────────────────────────────────────────────────────────────
# K-MEANS TRIBE WIN-RATE MAP (M3 — actualizado semanalmente)
# ──────────────────────────────────────────────────────────────────

TRIBE_WR_MAP: dict[int, float] = {0: 0.652, 2: 0.518, 1: 0.527, 3: 0.439}

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
