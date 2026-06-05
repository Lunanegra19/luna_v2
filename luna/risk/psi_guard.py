"""
psi_guard.py
============
Luna V2 — Capa 3: PSI-Guard con Cooldown y Gestión Dinámica de Riesgo.

PROBLEMA QUE RESUELVE
---------------------
El drift estructural del mercado (e.g., irrupción de ETFs en Q1 2025, PSI>8.0)
actualmente produce Cash Shield total durante 3 meses. No existe gradación de
respuesta: o se opera al 100% o se cierra todo.

SOLUCIÓN
--------
Un monitor continuo de PSI con tres niveles de respuesta graduada:
    1. PSI < psi_alert_threshold (0.25): NORMAL — operar sin restricciones.
    2. PSI ∈ [alert, halt]: ALERTA — reducir Kelly Fraction al 50%, notificar Telegram.
    3. PSI > psi_halt_threshold (0.50): HALT — Cash Shield global + notificación urgente.

El EWMA del PSI (no el valor instantáneo) previene falsos positivos por días de
baja liquidez o microestructura ruidosa.

DISEÑO DE CAUSALIDAD (SOP R1)
-------------------------------
- El PSI se calcula comparando IS distribution vs ventana rodante OOS reciente.
- La ventana de referencia IS se fija en train_end (no se actualiza con datos OOS).
- El EWMA suaviza el PSI con span configurable (default: 168H = 7 días).

USO
---
from luna.validation.psi_guard import PSIGuard

guard = PSIGuard.from_config("config/settings.yaml")
result = guard.evaluate(df_oos_window, df_is_reference)
# result.action in ['NORMAL', 'ALERT', 'HALT']
# result.kelly_multiplier: 1.0 | 0.5 | 0.0
# result.psi_raw, result.psi_ewma
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class PSIGuardResult:
    """Resultado de la evaluación del PSIGuard."""
    action: str                      # 'NORMAL' | 'ALERT' | 'HALT'
    kelly_multiplier: float          # 1.0 | 0.5 | 0.0
    psi_raw: float                   # PSI calculado sobre la ventana actual
    psi_ewma: float                  # PSI suavizado (usado para decisión)
    n_features_checked: int          # Nº de features evaluadas
    drifted_features: list[str]      # Features con PSI individual > umbral
    timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now("UTC"))
    message: str = ""

    @property
    def is_operational(self) -> bool:
        return self.action != "HALT"

    @property
    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "kelly_multiplier": self.kelly_multiplier,
            "psi_raw": round(self.psi_raw, 4),
            "psi_ewma": round(self.psi_ewma, 4),
            "n_features_checked": self.n_features_checked,
            "drifted_features": self.drifted_features,
            "timestamp": str(self.timestamp),
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class PSIGuard:
    """
    Monitor continuo de drift de distribución de features mediante PSI.

    Proporciona respuesta graduada al concepto drift:
        - NORMAL: operar sin restricciones (kelly_multiplier=1.0)
        - ALERT: reducir exposición (kelly_multiplier=0.5)
        - HALT: Cash Shield total (kelly_multiplier=0.0)
    """

    # Thresholds PSI según literatura de riesgo financiero (industria estándar)
    PSI_STABLE: float = 0.10         # PSI < 0.10: distribución estable
    PSI_MODERATE: float = 0.25       # PSI ∈ [0.10, 0.25]: cambio moderado
    PSI_CRITICAL: float = 0.50       # PSI > 0.50: cambio estructural grave

    N_BINS: int = 10                 # Bins para cálculo de PSI (deciles)

    def __init__(
        self,
        alert_threshold: float = 0.25,
        halt_threshold: float = 0.50,
        ewma_span_hours: int = 168,
        cooldown_hours: int = 336,   # 2 semanas mínimas entre re-trainings
        feature_cols: Optional[list[str]] = None,
    ):
        self.alert_threshold = alert_threshold
        self.halt_threshold = halt_threshold
        self.ewma_span_hours = ewma_span_hours
        self.cooldown_hours = cooldown_hours
        self.feature_cols = feature_cols

        self._psi_history: list[float] = []
        self._last_halt_time: Optional[pd.Timestamp] = None
        self._ewma_state: Optional[float] = None

        logger.info(
            f"[PSI-GUARD] Inicializado | alert_threshold={alert_threshold} | "
            f"halt_threshold={halt_threshold} | ewma_span={ewma_span_hours}H | "
            f"cooldown={cooldown_hours}H"
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "PSIGuard":
        """Factory desde settings.yaml con claves wfb.psi_*."""
        try:
            import yaml  # type: ignore
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            wfb_cfg = cfg.get("wfb", {})
            return cls(
                alert_threshold=wfb_cfg.get("psi_alert_threshold", 0.25),
                halt_threshold=wfb_cfg.get("psi_halt_threshold", 0.50),
                ewma_span_hours=wfb_cfg.get("psi_ewma_span", 168),
                cooldown_hours=wfb_cfg.get("psi_cooldown_hours", 336),
            )
        except Exception as e:
            logger.warning(f"[PSI-GUARD] No se pudo cargar config ({e}). Usando defaults.")
            return cls()

    # -----------------------------------------------------------------------
    # Cálculo de PSI
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_psi_single(
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-6,
    ) -> float:
        """
        Calcula el PSI (Population Stability Index) para una sola variable.

        PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)

        Parámetros
        ----------
        expected : np.ndarray — distribución de referencia (IS / training).
        actual   : np.ndarray — distribución actual (OOS / producción).
        n_bins   : int — número de bins (deciles = 10).
        epsilon  : float — suavizado para evitar log(0).

        Retorna
        -------
        float : PSI ≥ 0. Valores altos indican drift.
        """
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)

        # Eliminar NaNs e Infs
        expected = expected[np.isfinite(expected)]
        actual = actual[np.isfinite(actual)]

        if len(expected) < 10 or len(actual) < 10:
            return 0.0  # Insuficientes datos — PSI indefinido

        # Calcular breakpoints sobre la distribución de referencia (IS)
        breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        breakpoints = np.unique(breakpoints)  # Eliminar duplicados por baja varianza

        if len(breakpoints) < 3:
            return 0.0  # Distribución degenerada (e.g., columna constante)

        # Calcular proporciones en cada bin
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        expected_pct = expected_counts / (expected_counts.sum() + epsilon)
        actual_pct = actual_counts / (actual_counts.sum() + epsilon)

        # Suavizado para evitar log(0)
        expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
        actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(np.clip(psi, 0.0, 100.0))

    def compute_psi_multivariate(
        self,
        df_reference: pd.DataFrame,
        df_actual: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
    ) -> tuple[float, dict[str, float]]:
        """
        PSI multivariante: promedio del PSI individual de cada feature.

        Retorna
        -------
        psi_mean : float — PSI promedio sobre todas las features.
        psi_per_feature : dict — PSI individual por feature.
        """
        cols = feature_cols or self.feature_cols
        if cols is None:
            # Auto-detectar columnas numéricas presentes en ambos DataFrames
            common = df_reference.select_dtypes(include=[np.number]).columns.intersection(
                df_actual.select_dtypes(include=[np.number]).columns
            ).tolist()
            # [FIX-H] Intentar usar las features seleccionadas por importancia XGBoost (selected_features.json)
            # Antes: cols[:50] usaba las primeras 50 en orden de columna — sin relación con importancia del modelo
            _selected_loaded = False
            try:
                import json as _json_psi
                from pathlib import Path as _Path_psi
                _root_psi = Path(__file__).resolve().parent.parent.parent
                _sel_path = _root_psi / "data" / "features" / "selected_features.json"
                if _sel_path.exists():
                    _sel = _json_psi.loads(_sel_path.read_text(encoding="utf-8"))
                    _sel_feats = _sel.get("selected_features", [])
                    # Intersectar con las columnas disponibles en ambos DataFrames
                    cols = [f for f in _sel_feats if f in common]
                    if cols:
                        print(f"[FIX-H] PSI Guard: usando {len(cols)} features de selected_features.json (importancia XGBoost)")
                        _selected_loaded = True
            except Exception as _e_psi:
                pass  # silencioso — fallback abajo
            if not _selected_loaded:
                cols = common[:50]  # fallback: primeras 50 en orden de columna
                print(f"[FIX-H] PSI Guard: selected_features.json no disponible. Usando primeras {len(cols)} columnas numéricas (fallback)")

        if not cols:
            logger.warning("[PSI-GUARD] No hay columnas numéricas comunes para calcular PSI.")
            return 0.0, {}

        psi_per_feature: dict[str, float] = {}
        for col in cols:
            if col not in df_reference.columns or col not in df_actual.columns:
                continue
            psi_val = self.compute_psi_single(
                df_reference[col].values,
                df_actual[col].values,
                n_bins=self.N_BINS,
            )
            psi_per_feature[col] = psi_val

        psi_mean = float(np.mean(list(psi_per_feature.values()))) if psi_per_feature else 0.0
        return psi_mean, psi_per_feature

    # -----------------------------------------------------------------------
    # EWMA del PSI
    # -----------------------------------------------------------------------

    def _update_ewma(self, psi_raw: float) -> float:
        """Actualiza el EWMA del PSI con el nuevo valor observado."""
        alpha = 2.0 / (self.ewma_span_hours + 1)
        if self._ewma_state is None:
            self._ewma_state = psi_raw
        else:
            self._ewma_state = alpha * psi_raw + (1 - alpha) * self._ewma_state
        return self._ewma_state

    # -----------------------------------------------------------------------
    # Evaluación principal
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        df_oos_window: pd.DataFrame,
        df_is_reference: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
    ) -> PSIGuardResult:
        """
        Evalúa el drift de distribución y determina la acción a tomar.

        Parámetros
        ----------
        df_oos_window   : Ventana OOS actual (últimas N horas de datos en producción).
        df_is_reference : Dataset IS de referencia (distribución de entrenamiento).
        feature_cols    : Features a evaluar. Si None, auto-detecta.

        Retorna
        -------
        PSIGuardResult con acción, kelly_multiplier y métricas de drift.
        """
        # Calcular PSI multivariante
        psi_raw, psi_per_feature = self.compute_psi_multivariate(
            df_reference=df_is_reference,
            df_actual=df_oos_window,
            feature_cols=feature_cols,
        )

        # Actualizar EWMA
        psi_ewma = self._update_ewma(psi_raw)
        self._psi_history.append(psi_raw)

        # Identificar features con drift individual alto
        drifted = [
            f for f, v in psi_per_feature.items()
            if v > self.alert_threshold
        ]

        n_checked = len(psi_per_feature)

        # --- Determinar acción basada en PSI EWMA (no el valor instantáneo) ---
        if psi_ewma >= self.halt_threshold:
            action = "HALT"
            kelly_mult = 0.0
            message = (
                f"🚨 HALT: PSI_EWMA={psi_ewma:.4f} ≥ halt_threshold={self.halt_threshold}. "
                f"Cash Shield activado. Features con drift alto: {drifted[:5]}"
            )
            logger.error(f"[PSI-GUARD] {message}")

        elif psi_ewma >= self.alert_threshold:
            action = "ALERT"
            kelly_mult = 0.5
            message = (
                f"⚠️  ALERT: PSI_EWMA={psi_ewma:.4f} ≥ alert_threshold={self.alert_threshold}. "
                f"Kelly reducido al 50%. Features drifted: {drifted[:5]}"
            )
            logger.warning(f"[PSI-GUARD] {message}")

        else:
            action = "NORMAL"
            kelly_mult = 1.0
            message = (
                f"✅ NORMAL: PSI_EWMA={psi_ewma:.4f} < alert_threshold={self.alert_threshold}. "
                f"Distribución estable."
            )
            logger.info(f"[PSI-GUARD] {message}")

        return PSIGuardResult(
            action=action,
            kelly_multiplier=kelly_mult,
            psi_raw=psi_raw,
            psi_ewma=psi_ewma,
            n_features_checked=n_checked,
            drifted_features=drifted,
            message=message,
        )

    def save_state(self, path: str | Path) -> None:
        """Persiste el estado del EWMA para reinicios."""
        state = {
            "ewma_state": self._ewma_state,
            "psi_history": self._psi_history[-1000:],  # Últimas 1000 observaciones
            "last_halt_time": str(self._last_halt_time) if self._last_halt_time else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        logger.debug(f"[PSI-GUARD] Estado persistido en {path}")

    def load_state(self, path: str | Path) -> None:
        """Carga el estado persistido (para reinicio tras crash)."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._ewma_state = state.get("ewma_state")
            self._psi_history = state.get("psi_history", [])
            halt_str = state.get("last_halt_time")
            self._last_halt_time = pd.Timestamp(halt_str) if halt_str else None
            logger.info(f"[PSI-GUARD] Estado cargado. EWMA={self._ewma_state:.4f}")
        except FileNotFoundError:
            logger.info("[PSI-GUARD] Sin estado previo — iniciando desde cero.")
        except Exception as e:
            logger.warning(f"[PSI-GUARD] Error cargando estado ({e}). Iniciando desde cero.")
