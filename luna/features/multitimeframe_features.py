"""
multitimeframe_features.py — Luna V1
======================================
Genera features multi-timeframe desde los datos OHLCV:
  - 15min VWAP intra-hora (si disponible)
  - 4H momentum y alineación multi-TF
  - Volatilidad intra-hora realizada

Llamado en el PASO 5 de feature_pipeline.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger


class MultitimeframeFeatures:
    """
    Features de múltiples timeframes derivadas de OHLCV 1H base.

    Si los datos 15min están disponibles en el DataFrame, los usa.
    Si no, aproxima desde los datos 1H mediante técnicas estadísticas.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_4h_momentum(df)
        df = self._add_mt_alignment(df)
        df = self._add_intra_volatility(df)
        df = self._add_vwap_proxy(df)
        return df

    # ── 4H Momentum ─────────────────────────────────────────────────────────

    def _add_4h_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum en ventana de 4H: retorno acumulado last 4 barras.
        Proxy para la tendencia de corto plazo (no usa datos futuros).
        """
        if "close" not in df.columns:
            return df
        df["mt_momentum_4h"] = df["close"].pct_change(4)
        logger.debug("mt_momentum_4h calculado")
        return df

    # ── MT Alignment Score ──────────────────────────────────────────────────

    def _add_mt_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alineación multi-TF: ¿cuántos timeframes apuntan en la misma dirección?
        Score [-1, 1]: +1 = todos bullish, -1 = todos bearish.

        Timeframes simulados desde la 1H base:
        - 1H:   pct_change(1)  > 0
        - 4H:   pct_change(4)  > 0
        - 24H:  pct_change(24) > 0
        """
        if "close" not in df.columns:
            return df
        sig_1h  = (df["close"].pct_change(1)  > 0).astype(float) * 2 - 1
        sig_4h  = (df["close"].pct_change(4)  > 0).astype(float) * 2 - 1
        sig_24h = (df["close"].pct_change(24) > 0).astype(float) * 2 - 1
        df["mt_mt_alignment"] = (sig_1h + sig_4h + sig_24h) / 3.0
        logger.debug("mt_mt_alignment calculado")
        return df

    # ── Volatilidad Realizada Intra-Hora ────────────────────────────────────

    def _add_intra_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatilidad realizada en ventana de 4 barras (proxy de intra-hora).
        Usando high/low si disponibles (Parkinson estimator), sino std de retornos.
        """
        if "high" in df.columns and "low" in df.columns:
            # Parkinson (1980): estimador de volatilidad intra-barra
            hl_ratio = np.log(df["high"] / df["low"].clip(lower=1e-9))
            park = (hl_ratio ** 2) / (4 * np.log(2))
            df["mt_vol_realized_4bar"] = park.rolling(4).mean().apply(np.sqrt)
        elif "close" in df.columns:
            df["mt_vol_realized_4bar"] = df["close"].pct_change().rolling(4).std()
        logger.debug("mt_vol_realized_4bar calculado")
        return df

    # ── VWAP Proxy desde 1H ─────────────────────────────────────────────────

    def _add_vwap_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VWAP proxy definido como desviación del precio respecto al VWAP 24H.
        vwap_deviation > 0 → precio por encima del VWAP (sobrecompra relativa)
        """
        if "close" not in df.columns or "volume" not in df.columns:
            return df
        typical_price = df["close"]
        if "high" in df.columns and "low" in df.columns:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
        pv = typical_price * df["volume"]
        vwap_24 = pv.rolling(24).sum() / df["volume"].rolling(24).sum().clip(lower=1e-9)
        df["mt_vwap_deviation_proxy"] = (df["close"] - vwap_24) / vwap_24.abs().clip(lower=1e-9)
        logger.debug("mt_vwap_deviation_proxy calculado")
        return df
