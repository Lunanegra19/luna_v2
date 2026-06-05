import pandas as pd
import numpy as np
from loguru import logger

def validate_eth_lag(btc_close: pd.Series, eth_close: pd.Series, configured_lag_days: int) -> int:
    """
    Calcula el lag optimo entre ETH y BTC usando correlacion cruzada de retornos
    en una ventana rodante (sobre el ultimo año de datos disponibles).
    Retorna el lag optimo en dias. Lanza warning si difiere de configured_lag_days por > 2.
    """
    try:
        # Usar sólo el ulimo año de datos para medir el regimen actual
        cutoff = btc_close.index.max() - pd.Timedelta(days=365)
        btc_recent = btc_close[btc_close.index >= cutoff]
        eth_recent = eth_close[eth_close.index >= cutoff]
        
        # Retornos de 24H rodante por hora
        btc_ret = btc_recent.pct_change(24).dropna()
        eth_ret = eth_recent.pct_change(24).dropna()
        
        # Alinear indices
        common_idx = btc_ret.index.intersection(eth_ret.index)
        btc_ret = btc_ret.loc[common_idx]
        eth_ret = eth_ret.loc[common_idx]
        
        if len(btc_ret) < 30 * 24: # Menos de 1 mes de overlap
            return configured_lag_days

        corrs = {}
        # Lags desde 0 hasta 21 dias (pasos de 1 dia)
        for lag_days in range(22):
            # Desplazar ETH hacia el futuro para ver si el ETH del pasado predice el BTC actual
            eth_shifted = eth_ret.shift(lag_days * 24)
            
            valid_idx = eth_shifted.dropna().index
            if len(valid_idx) > 100:
                corr = btc_ret.loc[valid_idx].corr(eth_shifted.loc[valid_idx])
                corrs[lag_days] = corr

        if not corrs:
            return configured_lag_days
            
        optimal_lag_days = max(corrs, key=corrs.get)
        max_corr = corrs[optimal_lag_days]
        
        # Validar discrepancia (P3.1)
        if abs(optimal_lag_days - configured_lag_days) > 2:
            logger.warning(
                f"[P3.1] ⚠️ REGIME DRIFT: ETH Lag Optimo detectado en {optimal_lag_days} dias "
                f"(corr={max_corr:.3f}), pero settings.yaml usa {configured_lag_days} dias. "
                "Considera recalibrar 'eth_lag_days' si esta discrepancia persiste."
            )
        else:
            logger.debug(f"[P3.1] ETH Lag OK: Optimo={optimal_lag_days}d vs Config={configured_lag_days}d (corr={max_corr:.3f})")
            
        return optimal_lag_days
    except Exception as e:
        logger.error(f"[P3.1] Error calculando ETH_Lag optimo: {e}")
        return configured_lag_days
