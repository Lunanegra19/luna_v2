"""
alpha_rules.py — GENERADO AUTOMÁTICAMENTE por export_alpha_rules.py
Timestamp: 2026-06-16 04:33 UTC
Golden Rules: 15  |  Genetic Rules: 11
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
        'pandas_eval': '(FedFundsRate >= 4.6400) & (SP500_AboveMA200 <= 0.0000) & (MVRV_Proxy <= -0.7110)',
        'win_rate':    100.0,
        'ev_pct':      1.77,
        'description': 'IF FedFundsRate >= 4.6400 AND SP500_AboveMA200 <= 0.0000 AND MVRV_Proxy <= -0.7110',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(YieldCurve_10Y3M <= -0.3200) & (SP500_AboveMA200 <= 0.0000) & (MVRV_Proxy <= -0.7110)',
        'win_rate':    100.0,
        'ev_pct':      1.77,
        'description': 'IF YieldCurve_10Y3M <= -0.3200 AND SP500_AboveMA200 <= 0.0000 AND MVRV_Proxy <= -0.7110',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(FedFundsRate >= 4.6400) & (WEI <= 1.9500) & (MVRV_Proxy <= -0.7110)',
        'win_rate':    99.0,
        'ev_pct':      1.06,
        'description': 'IF FedFundsRate >= 4.6400 AND WEI <= 1.9500 AND MVRV_Proxy <= -0.7110',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(YieldCurve_10Y3M <= -0.3200) & (WEI <= 1.9500) & (MVRV_Proxy <= -0.7110)',
        'win_rate':    99.0,
        'ev_pct':      1.06,
        'description': 'IF YieldCurve_10Y3M <= -0.3200 AND WEI <= 1.9500 AND MVRV_Proxy <= -0.7110',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(Fed_Net_Liquidity >= 6319853.3600) & (WEI <= 1.9500) & (MVRV_Proxy <= -0.7110)',
        'win_rate':    99.0,
        'ev_pct':      1.06,
        'description': 'IF Fed_Net_Liquidity >= 6319853.3600 AND WEI <= 1.9500 AND MVRV_Proxy <= -0.7110',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(DVOL >= 57.3200) & (MVRV_Proxy <= -0.7110) & (Whale_Vol_ZScore >= 0.5172)',
        'win_rate':    97.9,
        'ev_pct':      0.85,
        'description': 'IF DVOL >= 57.3200 AND MVRV_Proxy <= -0.7110 AND Whale_Vol_ZScore >= 0.5172',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(T10Y2Y >= 0.4900) & (DVOL >= 57.3200) & (MVRV_Proxy <= -0.7110)',
        'win_rate':    95.8,
        'ev_pct':      1.16,
        'description': 'IF T10Y2Y >= 0.4900 AND DVOL >= 57.3200 AND MVRV_Proxy <= -0.7110',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(T10Y2Y >= 0.4900) & (WEI <= 1.9500) & (VIX >= 18.0900)',
        'win_rate':    89.1,
        'ev_pct':      6.13,
        'description': 'IF T10Y2Y >= 0.4900 AND WEI <= 1.9500 AND VIX >= 18.0900',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(DVOL >= 57.3200) & (MVRV_Proxy <= -0.7110) & (FearGreed <= 30.0000)',
        'win_rate':    88.0,
        'ev_pct':      0.95,
        'description': 'IF DVOL >= 57.3200 AND MVRV_Proxy <= -0.7110 AND FearGreed <= 30.0000',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(FedFundsRate >= 4.6400) & (SP500_AboveMA200 <= 0.0000) & (NASDAQ_Ret <= -0.0039)',
        'win_rate':    87.7,
        'ev_pct':      1.63,
        'description': 'IF FedFundsRate >= 4.6400 AND SP500_AboveMA200 <= 0.0000 AND NASDAQ_Ret <= -0.0039',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(FedFundsRate >= 4.6400) & (VIX >= 18.0900) & (SP500_AboveMA200 <= 0.0000)',
        'win_rate':    87.6,
        'ev_pct':      1.57,
        'description': 'IF FedFundsRate >= 4.6400 AND VIX >= 18.0900 AND SP500_AboveMA200 <= 0.0000',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(FedFundsRate >= 4.6400) & (DVOL >= 57.3200) & (FearGreed <= 30.0000)',
        'win_rate':    87.4,
        'ev_pct':      1.32,
        'description': 'IF FedFundsRate >= 4.6400 AND DVOL >= 57.3200 AND FearGreed <= 30.0000',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(YieldCurve_10Y3M <= -0.3200) & (DVOL >= 57.3200) & (FearGreed <= 30.0000)',
        'win_rate':    87.4,
        'ev_pct':      1.32,
        'description': 'IF YieldCurve_10Y3M <= -0.3200 AND DVOL >= 57.3200 AND FearGreed <= 30.0000',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(DXY <= 97.7700) & (NASDAQ_Ret <= -0.0039) & (SSR <= 0.4755)',
        'win_rate':    87.3,
        'ev_pct':      1.3,
        'description': 'IF DXY <= 97.7700 AND NASDAQ_Ret <= -0.0039 AND SSR <= 0.4755',
    },
    {
        'type':        'golden_storm',
        'pandas_eval': '(DVOL >= 57.3200) & (MVRV_Proxy <= -0.7110)',
        'win_rate':    86.9,
        'ev_pct':      1.11,
        'description': 'IF DVOL >= 57.3200 AND MVRV_Proxy <= -0.7110',
    },
]

# ──────────────────────────────────────────────────────────────────
# GENETIC RULES (Deep Discovery Engine — AG 20 generaciones)
# ──────────────────────────────────────────────────────────────────

GENETIC_RULES: list[dict] = [
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.4104) & (Wiki_BTC_Views <= 7937.0000) & (Stablecoin_Cap <= 3773375538.0000)',
        'win_rate':    84.8,
        'ev_pct':      0.88,
        'description': '`SSR <= 0.4104`<br>**AND** `Wiki_BTC_Views <= 7937.0000`<br>**AND** `Stablecoin_Cap <= 3773375538.0000`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.6140) & (Whale_Vol_ZScore <= -0.7132) & (SSR <= 0.4104) & (Stablecoin_Cap <= 3773375538.0000)',
        'win_rate':    81.7,
        'ev_pct':      1.06,
        'description': '`SSR <= 0.6140`<br>**AND** `Whale_Vol_ZScore <= -0.7132`<br>**AND** `SSR <= 0.4104`<br>**AND** `Stablecoin_Cap <= 377337',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.6140) & (Whale_Vol_ZScore <= -0.7132) & (SSR <= 0.4104)',
        'win_rate':    80.6,
        'ev_pct':      1.14,
        'description': '`SSR <= 0.6140`<br>**AND** `Whale_Vol_ZScore <= -0.7132`<br>**AND** `SSR <= 0.4104`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.4104) & (Whale_Vol_ZScore <= -0.7132)',
        'win_rate':    80.6,
        'ev_pct':      1.14,
        'description': '`SSR <= 0.4104`<br>**AND** `Whale_Vol_ZScore <= -0.7132`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.4104) & (VIX <= 13.2000) & (Stablecoin_Cap <= 3773375538.0000)',
        'win_rate':    79.4,
        'ev_pct':      0.46,
        'description': '`SSR <= 0.4104`<br>**AND** `VIX <= 13.2000`<br>**AND** `Stablecoin_Cap <= 3773375538.0000`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.6140) & (Whale_Vol_ZScore <= -0.7132) & (MVRV_Proxy <= -0.8180) & (DeFi_WBTC_TVL <= 5071106.0000)',
        'win_rate':    72.9,
        'ev_pct':      0.23,
        'description': '`SSR <= 0.6140`<br>**AND** `Whale_Vol_ZScore <= -0.7132`<br>**AND** `MVRV_Proxy <= -0.8180`<br>**AND** `DeFi_WBTC_TVL <=',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(MVRV_Proxy <= -0.8180) & (DeFi_WBTC_TVL <= 5071106.0000)',
        'win_rate':    68.2,
        'ev_pct':      0.44,
        'description': '`MVRV_Proxy <= -0.8180`<br>**AND** `DeFi_WBTC_TVL <= 5071106.0000`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(SSR <= 0.4104)',
        'win_rate':    65.2,
        'ev_pct':      0.56,
        'description': '`SSR <= 0.4104`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(MVRV_Proxy >= 2.3148) & (Whale_Vol_ZScore >= 0.5852) & (DXY <= 92.5300) & (DXY <= 96.5000)',
        'win_rate':    63.1,
        'ev_pct':      0.88,
        'description': '`MVRV_Proxy >= 2.3148`<br>**AND** `Whale_Vol_ZScore >= 0.5852`<br>**AND** `DXY <= 92.5300`<br>**AND** `DXY <= 96.5000`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(FearGreed >= 75.0000) & (active_addresses_7d_ma >= 670563.2857)',
        'win_rate':    59.4,
        'ev_pct':      0.67,
        'description': '`FearGreed >= 75.0000`<br>**AND** `active_addresses_7d_ma >= 670563.2857`',
    },
    {
        'type':        'genetic_rule',
        'pandas_eval': '(Wiki_BTC_Views <= 6737.0000)',
        'win_rate':    57.3,
        'ev_pct':      0.12,
        'description': '`Wiki_BTC_Views <= 6737.0000`',
    },
]

# ──────────────────────────────────────────────────────────────────
# CAUSAL VARIABLES (Advanced Engine — Granger*** + TE_net > 0)
# ──────────────────────────────────────────────────────────────────

CAUSAL_VARS: list[str] = ['SSR', 'DeFi_WBTC_TVL', 'MVRV_Proxy', 'FearGreed', 'Stablecoin_Cap', 'Master_Causal_Signal', 'DangerZone', 'FundingRate', 'Whale_Vol_ZScore']

# ──────────────────────────────────────────────────────────────────
# DTW FRACTAL PROBABILITY
# ──────────────────────────────────────────────────────────────────

DTW_BULL_PROB: float = 0.6  # P(BTC sube en 24H | análogos históricos)

# ──────────────────────────────────────────────────────────────────
# K-MEANS TRIBE BIAS
# ──────────────────────────────────────────────────────────────────

TRIBE_BIAS: dict[int, str] = {3: 'LARGA', 1: 'NEUTRAL', 2: 'NEUTRAL', 0: 'NEUTRAL'}

# ──────────────────────────────────────────────────────────────────
# K-MEANS TRIBE WIN-RATE MAP (M3 — actualizado semanalmente)
# ──────────────────────────────────────────────────────────────────

TRIBE_WR_MAP: dict[int, float] = {3: 0.643, 1: 0.523, 2: 0.526, 0: 0.448}

LARGA_TRIBES   = frozenset({3})
NEUTRAL_TRIBES = frozenset({0, 1, 2})

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
