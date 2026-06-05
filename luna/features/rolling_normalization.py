"""
rolling_normalization.py
========================
Luna V1 — Feature Engineering
Rolling Z-Score 90d para todas las variables macroeconómicas.

REGLA R14: Anti-Concept Drift obligatorio.
Motivación: Los umbrales estáticos ('UnemployRate >= 4.47') detectados en Luna v2
Fase 15 se vuelven OBSOLETOS cuando el régimen macro cambia. Con Rolling Z-Score,
el threshold se adapta al contexto histórico reciente.

Sin este módulo: el modelo usaría valores absolutos de CPI, FedFunds, etc. que
cambian de rango en distintos ciclos económicos (ej: CPI 2% en 2019 vs 9% en 2022).

Uso:
    normalizer = RollingZScoreNormalizer()
    df = normalizer.transform(df, columns=MACRO_COLUMNS)
    # Genera df['FedFundsRate_z90d'], df['CPI_YoY_z90d'], etc.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

from config.settings import cfg
from luna.utils.debug_guards import vlog, timeit, log_dataframe_transition, check_df_sanity

# Columnas macro que DEBEN normalizarse (R14)
MACRO_COLUMNS = [
    "FedFundsRate",
    "CPI_YoY",
    "UnemployRate",
    "WEI",
    "M2_USA_raw",
    "M2_YoY_Chg",
    "M2_MoM_Chg",
    "M2_China_raw",
    "M2_China_YoY",
    "YieldCurve_10Y3M",
    "YieldCurve_30Y10Y",
    "T10Y2Y",
    "VIX",
    "VIX_ZScore",
    "DXY",
    "DXY_Slope30d",
    "DXY_Pct6m",
    "Spread_VIX_MOVE",
    "SP500",
    "SP500_AboveMA200",   # [FIX-NAMES-01 2026-06-02] era SP500_vs_MA200 (nombre legacy, no existe en parquet)
    "RUSSELL2000",
    "Gold",
    "Oil",
    "TGA_USD",
    "RRP_USD",
    "MVRV_Proxy",
    "Stablecoin_Cap",
    "Stablecoin_Cap_Delta",
    "Total_TVL_USD",
    "DeFi_WBTC_TVL",
    "ETH_Price",
    "Puell_Multiple",
    "FearGreed",
    # [FIX-NAMES-01 2026-06-02] ActiveAddresses -> active_addresses (100% NaN, data lake sin sync)
    # [FIX-NAMES-01 2026-06-02] Hashrate -> hashrate_th (100% NaN, data lake sin sync)
    # Mantenemos los nombres reales: el normalizer ignora columnas ausentes gracefully.
    "OI_BTC",
    "OI_USD",
]
# [AUDIT-A3 FIX 2026-05-08] Features macro de baja frecuencia con cambios de regimen amplios.
# FedFundsRate: 0%->5% en 6 meses (2022) -> z-score > 5sigma es ESPERADO, no un error.
# CPI_YoY: 2%-9%-2% en 2 ciclos -> mismo comportamiento.
# Estas features usan clip adaptativo de +-7sigma en lugar del global +-5sigma.
MACRO_WIDE_CLIP_COLS = {
    "FedFundsRate",   # ciclos de subidas/bajadas rapidas de la Fed
    "CPI_YoY",        # inflacion: regimenes de 2%-9% en el mismo dataset
    "T10Y2Y",         # curva de yields: inversiones extremas en crisis
    "YieldCurve_10Y3M",  # mismo caso que T10Y2Y
    "YieldCurve_30Y10Y",
}



class RollingZScoreNormalizer:
    """
    Aplica Rolling Z-Score de ventana 90d a las variables macro.
    Genera columnas '<col>_z90d' sin eliminar las originales (el pipeline SFI
    puede seleccionar cuál versión aporta más Sharpe OOS).
    """

    def __init__(self, window_days: Optional[int] = None):
        self.window = window_days or cfg.features.rolling_zscore_window  # 90

    def transform(
        self,
        df: pd.DataFrame,
        columns: list[str] = MACRO_COLUMNS,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Aplica Rolling Z-Score a las columnas especificadas que existan en df.

        Args:
            df: DataFrame con índice temporal (hourly o daily)
            columns: lista de columnas a normalizar
            inplace: si True, reemplaza la columna original; si False, añade '_z90d'

        Returns:
            DataFrame con columnas normalizadas añadidas (o reemplazadas si inplace)
        """
        df_before = df  # guard verbose
        result = df.copy()
        window_pts = self.window * 24  # Convertir a puntos si es 1H

        # Usar ventana en puntos si el índice es horario, en días si es diario
        freq_detected = self._detect_frequency(df)
        window_pts = self.window * 24 if freq_detected == "1H" else self.window

        vlog(f"RollingZScore: shape={df.shape} | window={self.window}d ({window_pts}pts) | freq={freq_detected}")

        normalized_count = 0
        missing_cols = []
        zero_var_cols  = []
        heavy_clip_cols = []

        with timeit("RollingZScore.transform"):
            for col in columns:
                if col not in df.columns:
                    missing_cols.append(col)
                    continue
                if df[col].isna().all():
                    logger.debug(f"Rolling Z-Score [{col}]: columna vacía, saltando.")
                    continue

                roll_mean = df[col].rolling(window=window_pts, min_periods=window_pts // 2).mean()
                roll_std  = df[col].rolling(window=window_pts, min_periods=window_pts // 2).std()

                # Detectar varianza ~0 (columna constante o casi constante)
                mean_std = roll_std.mean()
                if mean_std < 1e-6:
                    zero_var_cols.append(col)
                    vlog(f"  [{col}] varianza CERO (mean_std={mean_std:.2e}) — se aplica z-score con eps", "WARNING")

                z_score = (df[col] - roll_mean) / (roll_std + 1e-8)  # +eps evita /0

                # [AUDIT-A3 FIX] Clip adaptativo: features macro con regimenes amplios -> +-7sigma
                _clip_bound = 7.0 if col in MACRO_WIDE_CLIP_COLS else 5.0

                # Detectar clipping agresivo (muchos valores fuera del bound)
                n_clipped_out = int((z_score.abs() > _clip_bound).sum())
                clip_pct = 100 * n_clipped_out / max(len(z_score.dropna()), 1)
                _clip_warn_threshold = 2.0 if col in MACRO_WIDE_CLIP_COLS else 1.0
                if clip_pct > _clip_warn_threshold:
                    heavy_clip_cols.append(f"{col}:{clip_pct:.1f}%")

                new_col = col if inplace else f"{col}_z90d"
                result[new_col] = z_score.clip(-_clip_bound, _clip_bound)
                normalized_count += 1

        if missing_cols:
            logger.debug(f"Rolling Z-Score: {len(missing_cols)} columnas no encontradas: {missing_cols[:5]}...")
        if zero_var_cols:
            logger.warning(f"Rolling Z-Score ⚠️  Columnas con varianza ~0: {zero_var_cols}")
        if heavy_clip_cols:
            logger.warning(f"Rolling Z-Score ⚠️  Clipping >1% fuera de ±5σ: {heavy_clip_cols}")


        logger.info(
            f"Rolling Z-Score (window={self.window}d): "
            f"{normalized_count}/{len(columns)} columnas normalizadas → sufijo '_z90d' | "
            f"missing={len(missing_cols)} | zero_var={len(zero_var_cols)} | heavy_clip={len(heavy_clip_cols)}"
        )
        log_dataframe_transition(df_before, result, "RollingZScore")
        return result

    def _detect_frequency(self, df: pd.DataFrame) -> str:
        """Detecta si el DataFrame es 1H o diario — robusto a gaps iniciales."""
        # LOGIC-ROLL-01 FIX (2026-04-06): usar mediana de los primeros 20 diffs en
        # lugar de solo las primeras 2 filas. Si el par inicial tiene un gap accidental
        # (mantenimiento Binance, datos faltantes), diff[0]>3600s → detecta erroneamente
        # "1D" y aplica window=90 en lugar de window=90*24=2160, produciendo Z-scores
        # con varianza 24x demasiado alta.
        if len(df) < 2:
            return "1H"  # default seguro para luna (siempre horario)
        diffs = pd.Series(df.index[:min(21, len(df))]).diff().dropna()
        if diffs.empty:
            return "1H"
        median_diff_s = diffs.median().total_seconds()
        return "1H" if median_diff_s <= 3600 else "1D"


    def get_z90d_columns(self, df: pd.DataFrame) -> list[str]:
        """Retorna lista de columnas '_z90d' generadas."""
        return [c for c in df.columns if c.endswith("_z90d")]
