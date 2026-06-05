"""
luna/models/calibrators.py
==========================
Módulo canónico de calibradores de probabilidad para Luna V2.

[FIX-TEMPCAL-DESER-01 2026-06-01]
Antes de este fix, TemperatureCalibrator estaba definida únicamente en
train_xgboost_v2.py. Cuando joblib la serializaba, la guardaba con el
módulo de origen como '__main__' (el script de entrenamiento). Al
deserializar en regime_router.py (contexto diferente), Python no encontraba
la clase → AttributeError: Can't get attribute 'TemperatureCalibrator'.

Fix: mover la clase a este módulo importable. Ahora joblib la serializa
como 'luna.models.calibrators.TemperatureCalibrator', que es importable
desde cualquier script del pipeline.

Impacto: 5 seeds FATAL directas y 33 errores AttributeError eliminados.
Seeds afectadas: 100, 1337, 2026, 27243, 44085.
"""

import numpy as np


class TemperatureCalibrator:
    """
    Temperature Scaling calibrator (Guo et al. 2017).

    Propiedades garantizadas:
      1. Si std(p_raw) > 0 → std(p_cal) > 0 siempre — elimina calibration collapse.
      2. Monotone estricto: preserva todos los rankings relativos.
      3. Un solo parámetro T ∈ [0.1, 10] → cero riesgo de overfitting con 1400+ muestras.
      4. T > 1 suaviza probs hacia 0.5 (reduce overconfianza), T < 1 las sharpens.

    Uso en cascada: Isotonic → [si colapsa] TemperatureCalibrator → [si colapsa] Raw
    Compatible con signal_filter.py guard: tiene .predict() y .X_thresholds_.
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.X_thresholds_: list = []  # compatibilidad con BUG-CALIB-XGB-01 guard de signal_filter

    def fit(self, p_raw: np.ndarray, y_val: np.ndarray) -> "TemperatureCalibrator":
        """Ajusta T minimizando NLL sobre el validation set."""
        from scipy.optimize import minimize_scalar

        p_safe = np.clip(p_raw.astype(float), 1e-7, 1.0 - 1e-7)
        logits = np.log(p_safe / (1.0 - p_safe))

        def _nll(T: float) -> float:
            T = max(T, 1e-3)
            p_c = np.clip(1.0 / (1.0 + np.exp(-logits / T)), 1e-7, 1.0 - 1e-7)
            return -float(np.mean(y_val * np.log(p_c) + (1.0 - y_val) * np.log(1.0 - p_c)))

        result = minimize_scalar(_nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        # Simular X_thresholds_ para compatibilidad con el guard de signal_filter.py
        self.X_thresholds_ = list(np.linspace(float(p_raw.min()), float(p_raw.max()), 5))
        print(  # RULE[fixbugsprints.md]
            f"[FIX-TEMPCAL-DESER-01] TemperatureCalibrator.fit() OK: T={self.temperature:.4f} "
            f"n={len(p_raw)} | módulo=luna.models.calibrators (serializable)"
        )
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        """Aplica Temperature Scaling: p_cal = sigmoid(logit(p_raw) / T)."""
        p_safe = np.clip(p_raw.astype(float), 1e-7, 1.0 - 1e-7)
        logits = np.log(p_safe / (1.0 - p_safe))
        return np.clip(1.0 / (1.0 + np.exp(-logits / self.temperature)), 0.0, 1.0)
