"""
core/calibration/temperature_scaler.py
[V2-P6] Temperature Scaling — Calibración XGB cuando Isotónico falla

Problema documentado (§7 ROADMAP_ARQUITECTURA_DINAMICA_V2.md):
    W1 OOS real: XGB_prob ganadores=0.591, perdedores=0.593 → separación=0.002.
    El Calibrador Isotónico no puede ajustar curvas útiles sobre una distribución
    tan comprimida → retorna NaN o empeora el Brier.

Solución — Temperature Scaling (T < 1.0: Sharpening):
    T < 1.0: SHARPENING → separa probabilidades concentradas (nuestro caso).
             Convierte {0.57, 0.59, 0.61} en {0.45, 0.59, 0.73} con T=0.5.
    T > 1.0: SOFTENING  → comprime hacia 0.5 (modelos overconfident).

Matemáticas:
    logit  = log(p / (1-p))       # prob → log-odds
    logit' = logit / T            # escalar por temperatura
    p'     = sigmoid(logit')      # log-odds → prob ajustada

Gate de activación:
    - Si mejora Brier del isotónico < MIN_BRIER_IMPROVEMENT_PCT (2%):
      activar Temperature Scaling con T=0.5 (sharpening).
    - Si T-Scaling tampoco mejora: mantener raw probs (fail-safe).

Referencias:
    - Guo et al. (2017) "On Calibration of Modern Neural Networks", ICML.
    - Platt (1999) "Probabilistic Outputs for Support Vector Machines".
"""
from __future__ import annotations

import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)  # type: ignore[assignment]


# Constante: si ISO no mejora más de este % → activar Temperature Scaling
MIN_BRIER_IMPROVEMENT_PCT: float = 2.0


class TemperatureScaler:
    """Ajusta la confianza de un clasificador mediante escalado de temperatura.

    Operación:
        - T < 1.0 (Sharpening): separa probs concentradas cerca de 0.5/0.59.
          Útil cuando XGBoost es underconfident (nuestro caso).
        - T > 1.0 (Softening):  comprime probs hacia 0.5.
          Útil cuando el modelo está sobreajustado (raro en XGB con CPCV).

    Ejemplo para T=0.5 (sharpening):
        p=0.57 → logit=-0.241 → logit/T=-0.482 → p'=0.382
        p=0.59 → logit=-0.041 → logit/T=-0.082 → p'=0.479
        p=0.62 → logit=+0.489 → logit/T=+0.978 → p'=0.727
    """

    def __init__(self, temperature: float = 0.5):
        """
        Args:
            temperature: T de escalado. Default 0.5 (sharpening para modelos
                         underconfident con probs concentradas en ~0.59).
        """
        if temperature <= 0:
            raise ValueError(f"temperature debe ser > 0. Recibido: {temperature}")
        self.T = temperature

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Aplica Temperature Scaling vectorialmente.

        Args:
            probs: Array de probabilidades en (0, 1).

        Returns:
            Array de probabilidades calibradas, mismo shape.
        """
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        logits  = np.log(probs / (1.0 - probs))  # sigmoid inverse
        logits_ = logits / self.T                 # temperatura
        return 1.0 / (1.0 + np.exp(-logits_))    # sigmoid


def apply_temperature_scaling_gate(
    raw_probs:     np.ndarray,
    y_true:        np.ndarray,
    iso_probs:     np.ndarray | None = None,
    brier_raw:     float | None = None,
    brier_iso:     float | None = None,
    temperature:   float = 0.5,
    min_improvement_pct: float = MIN_BRIER_IMPROVEMENT_PCT,
) -> tuple[np.ndarray, str, float]:
    """Gate: activa Temperature Scaling si ISO no mejoró suficientemente.

    Lógica:
        1. Calcular Brier raw si no se pasó.
        2. Calcular mejora ISO = (brier_raw - brier_iso) / brier_raw × 100.
        3. Si mejora < min_improvement_pct → aplicar TemperatureScaler(T=0.5).
        4. Si T-Scaling tampoco mejora → mantener raw_probs.

    Args:
        raw_probs:          Probabilidades crudas XGBoost (sin calibrar).
        y_true:             Labels binarios reales (0/1).
        iso_probs:          Probabilidades calibradas por ISO (None si ISO falló).
        brier_raw:          Brier Score de raw_probs (calculado si None).
        brier_iso:          Brier Score de iso_probs (calculado si None).
        temperature:        Temperatura para TemperatureScaler.
        min_improvement_pct: Umbral mínimo de mejora ISO para no activar TS.

    Returns:
        (final_probs, method_used, brier_final)
        method: "isotonic" | "temperature_scaling" | "raw"
    """
    y = np.asarray(y_true, dtype=float)
    p_raw = np.asarray(raw_probs, dtype=float)

    if brier_raw is None:
        brier_raw = float(np.mean((p_raw - y) ** 2))

    # Calcular mejora ISO
    iso_improvement = 0.0
    if iso_probs is not None and brier_iso is not None:
        iso_improvement = (brier_raw - brier_iso) / max(brier_raw, 1e-8) * 100.0

    if iso_improvement >= min_improvement_pct and iso_probs is not None:
        # ISO mejora suficiente → usar ISO
        logger.info(
            f"[V2-P6-TEMPSCALE] ISO suficiente: mejora={iso_improvement:.1f}% "
            f">= {min_improvement_pct:.0f}% — usando isotónico."
        )
        return iso_probs, "isotonic", brier_iso

    # ISO insuficiente → probar Temperature Scaling
    ts = TemperatureScaler(temperature=temperature)
    p_ts = ts.calibrate(p_raw)
    brier_ts = float(np.mean((p_ts - y) ** 2))

    if brier_ts < brier_raw:
        _mejora_ts = (brier_raw - brier_ts) / max(brier_raw, 1e-8) * 100.0
        logger.info(
            f"[V2-P6-TEMPSCALE] Temperature Scaling activado (T={temperature}): "
            f"Brier {brier_raw:.4f} → {brier_ts:.4f} ({_mejora_ts:.1f}% mejora). "
            f"ISO improvement fue {iso_improvement:.1f}% < {min_improvement_pct:.0f}% umbral."
        )
        return p_ts, "temperature_scaling", brier_ts
    else:
        logger.warning(
            f"[V2-P6-TEMPSCALE] Ningún calibrador mejora (raw={brier_raw:.4f} "
            f"ts={brier_ts:.4f}). Manteniendo probabilidades crudas."
        )
        return p_raw, "raw", brier_raw
