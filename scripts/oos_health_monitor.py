"""
oos_health_monitor.py
======================
Módulo institucional para la monitorización de degradación (Drift) en Out-Of-Sample.
Calcula el CUSUM (Cumulative Sum Control Chart) y la degradación del Sharpe Ratio
para alertar cuando la equidad de la estrategia empieza a desvanecerse.

Referencia: Hawkins & Olwell (1998)
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def calculate_cusum(returns_series: pd.Series, target: float = 0.0, threshold: float = 4.0) -> dict:
    """
    Calcula el CUSUM direccional negativo (monitoriza las caídas persistentes 
    por debajo del rendimiento objetivo).

    Args:
        returns_series: pd.Series con los retornos de los trades.
        target: Retorno promedio esperado por trade (0.0 = sin pérdida neta).
        threshold: Umbral CUSUM para disparar la alarma (típicamente 4.0 - 5.0).
        
    Returns:
        dict: {
            "max_drift": float (máxima desviación acumulada negativa detectada),
            "trigger": bool (si el drift superó el threshold),
            "current_drift": float (desviación actual al final de la serie)
        }
    """
    if len(returns_series) == 0:
        return {"max_drift": 0.0, "trigger": False, "current_drift": 0.0}

    # Asumimos que queremos alertar cuando el retorno *cae* por debajo del target
    # CUSUM Negativo: S_t = max(0, S_{t-1} + (target - r_t))
    
    # Estandarizamos los retornos para que el umbral (e.g. 4.0 desviaciones) tenga sentido
    std = returns_series.std()
    if std == 0 or pd.isna(std):
        return {"max_drift": 0.0, "trigger": False, "current_drift": 0.0}

    # Desviación normalizada vs target (positivo = malo, estamos perdiendo respecto al target)
    deviations = (target - returns_series) / std
    
    cusum = np.maximum.accumulate(deviations.cumsum()) - deviations.cumsum() + deviations
    # Esta es una aproximacion rapida: la iterativa real es S = max(0, S + dev)
    
    S = 0.0
    max_drift = 0.0
    for dev in deviations:
        S = max(0.0, S + dev)
        max_drift = max(max_drift, S)

    trigger = max_drift >= threshold
    
    if trigger:
        logger.warning(f"[CUSUM] Alarma de degradación de equidad disparada. Max Drift: {max_drift:.2f} >= {threshold}")

    return {
        "max_drift": float(max_drift),
        "trigger": bool(trigger),
        "current_drift": float(S)
    }

def calculate_sharpe_degradation(returns_series: pd.Series, freq: str = "W", critical_weeks: int = 2) -> dict:
    """
    Verifica si el Sharpe Ratio se ha degradado significativamente en los periodos más recientes
    comparado con el Sharpe histórico.

    Args:
        returns_series: pd.Series con los retornos y un pd.DatetimeIndex.
        freq: Frecuencia de remuestreo (W = Semanal, M = Mensual).
        critical_weeks: Número de periodos recientes a evaluar como "recent".
        
    Returns:
        dict: {
            "recent_sharpe": float,
            "historical_sharpe": float,
            "trigger": bool (True si la degradación es crítica)
        }
    """
    if len(returns_series) < 10 or not isinstance(returns_series.index, pd.DatetimeIndex):
        # Fallback silencioso si no hay DatetimeIndex o data insuficiente
        return {"recent_sharpe": 0.0, "historical_sharpe": 0.0, "trigger": False}

    try:
        # Agrupamos por frecuencia y calculamos Sharpe básico
        grouped = returns_series.resample(freq).apply(lambda x: (x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else 0)
        
        if len(grouped) <= critical_weeks:
             return {"recent_sharpe": 0.0, "historical_sharpe": 0.0, "trigger": False}
             
        historical = grouped.iloc[:-critical_weeks]
        recent = grouped.iloc[-critical_weeks:]
        
        hist_sharpe = historical.mean()
        rec_sharpe = recent.mean()
        
        # Trigger si el sharpe reciente es negativo Y menor a la mitad del histórico
        trigger = rec_sharpe < 0 and rec_sharpe < (hist_sharpe * 0.5)
        
        return {
            "recent_sharpe": float(rec_sharpe),
            "historical_sharpe": float(hist_sharpe),
            "trigger": bool(trigger)
        }
    except Exception as e:
        logger.debug(f"[Sharpe Degradation] Error interno: {e}")
        return {"recent_sharpe": 0.0, "historical_sharpe": 0.0, "trigger": False}
