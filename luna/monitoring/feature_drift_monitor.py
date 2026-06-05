"""
core/monitoring/feature_drift_monitor.py
[V2-P3] Monitor de Feature Drift — Hallazgo Crítico #3 del ROADMAP_ARQUITECTURA_DINAMICA_V2.md

Population Stability Index (PSI) para detección de Covariate Shift entre ventanas WFB.

Evidencia forense:
- Mempool_HashRate:  drift 1.46σ entre W1 y W2 → predicciones contaminadas
- btc_trend_regime: drift 0.82σ entre W1 y W2 → sesgo de dirección

Umbrales PSI estándar (industria financiera):
- PSI < 0.10:  Distribución estable              → OK
- PSI 0.10-0.25: Cambio leve — monitorizar       → WARNING
- PSI > 0.25:  Cambio severo — intervención       → ALERT: reducir Kelly 50%
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)  # type: ignore[assignment]

# ── Umbrales PSI ──────────────────────────────────────────────────────────────
PSI_STABLE   = 0.10   # Por debajo: OK
PSI_WARNING  = 0.25   # Entre PSI_STABLE y PSI_WARNING: monitorizar
PSI_CRITICAL = 0.25   # Por encima: drift severo → reducir Kelly

# Número de features en drift severo que activa la penalización Kelly
DRIFT_KELLY_PENALTY_N_FEATURES = 5
DRIFT_KELLY_PENALTY_FACTOR     = 0.50   # Reducir max_position al 50% si drift severo


def compute_psi(
    expected: pd.Series,
    actual: pd.Series,
    buckets: int = 10,
) -> float:
    """Calcula el Population Stability Index entre dos distribuciones.

    PSI = Σ (P_expected - P_actual) × ln(P_expected / P_actual)

    Args:
        expected: distribución de referencia (train set).
        actual:   distribución actual (OOS / nueva ventana).
        buckets:  número de cuantiles para el histograma.

    Returns:
        float: PSI. 0.0 si los datos son degenerados (serie constante).
    """
    expected = expected.dropna().values
    actual   = actual.dropna().values

    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    # Usar cuantiles del expected como bins (más robusto que bins uniformes)
    breaks = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
    if len(breaks) < 3:
        return 0.0  # Serie casi constante → PSI indefinido

    e_counts, _ = np.histogram(expected, bins=breaks)
    a_counts, _ = np.histogram(actual,   bins=breaks)

    # Convertir a proporciones (clip para evitar log(0))
    e_pct = np.clip(e_counts / len(expected), 1e-4, None)
    a_pct = np.clip(a_counts / len(actual),   1e-4, None)

    psi = float(np.sum((e_pct - a_pct) * np.log(e_pct / a_pct)))
    return psi


def run_drift_monitor(
    train_features: pd.DataFrame,
    oos_features:   pd.DataFrame,
    feature_cols:   Optional[list] = None,
    psi_threshold:  float = PSI_CRITICAL,
    buckets:        int   = 10,
) -> dict:
    """Calcula PSI para todas las features y emite alertas si hay drift severo.

    Integración V2-P3: llamar al inicio de OOSTradesGenerator.generate()
    después de cargar df_oos, ANTES de cualquier predicción.

    Args:
        train_features: DataFrame del training set (distribución referencia).
        oos_features:   DataFrame del periodo OOS actual.
        feature_cols:   Lista de columnas a evaluar. None = todas las comunes.
        psi_threshold:  PSI por encima del cual se considera drift severo.
        buckets:        Número de cuantiles para el histograma PSI.

    Returns:
        dict con keys:
            - 'results':       {col: {'psi': float, 'status': str}}
            - 'n_drifted':     int — features con PSI > threshold
            - 'kelly_penalty': float — factor por el que multiplicar max_position (1.0 = sin penalización)
            - 'worst_features': list — top 5 features por PSI
    """
    # Determinar columnas a evaluar
    if feature_cols is None:
        common_cols = [c for c in train_features.columns if c in oos_features.columns]
    else:
        common_cols = [c for c in feature_cols if c in train_features.columns and c in oos_features.columns]

    # Excluir columnas no numéricas y constantes
    numeric_cols = []
    for col in common_cols:
        try:
            if pd.api.types.is_numeric_dtype(train_features[col]) and train_features[col].std() > 0:
                numeric_cols.append(col)
        except Exception:
            pass

    results = {}
    for col in numeric_cols:
        try:
            psi = compute_psi(train_features[col], oos_features[col], buckets=buckets)
            if psi < PSI_STABLE:
                status = "STABLE"
            elif psi < PSI_WARNING:
                status = "WARNING"
            else:
                status = "CRITICAL"
            results[col] = {"psi": round(psi, 4), "status": status}
        except Exception as e:
            logger.debug(f"  [PSI] Error calculando PSI para {col}: {e}")

    # Ordenar por PSI descendente
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["psi"], reverse=True))

    n_critical = sum(1 for v in results.values() if v["status"] == "CRITICAL")
    n_warning  = sum(1 for v in results.values() if v["status"] == "WARNING")
    n_stable   = sum(1 for v in results.values() if v["status"] == "STABLE")
    worst_5    = list(sorted_results.keys())[:5]

    # ── LOG ENRIQUECIDO — visible durante WFB en tiempo real ──────────────────
    logger.info(
        f"  [V2-P3-DRIFT] PSI Monitor: {len(results)} features evaluadas | "
        f"STABLE={n_stable} | WARNING={n_warning} | CRITICAL={n_critical}"
    )

    if n_critical > 0:
        logger.warning(
            f"  [V2-P3-DRIFT] {n_critical} features con drift CRITICO (PSI > {psi_threshold:.2f}):"
        )
        for col in worst_5:
            s = results[col]
            logger.warning(f"    {col} -> PSI={s['psi']:.4f} [{s['status']}]")
    elif n_warning > 0:
        logger.info(f"  [V2-P3-DRIFT] {n_warning} features con drift LEVE (PSI 0.10-0.25) - monitorizar.")
        for col in worst_5[:3]:
            s = results[col]
            logger.info(f"    {col} -> PSI={s['psi']:.4f} [{s['status']}]")
    else:
        logger.info(
            f"  [V2-P3-DRIFT] Todas las features ESTABLES (PSI < {PSI_STABLE:.2f}). "
            f"Sin Covariate Shift detectado."
        )

    # ── Penalización Kelly si hay drift severo ─────────────────────────────────
    kelly_penalty = 1.0
    if n_critical >= DRIFT_KELLY_PENALTY_N_FEATURES:
        kelly_penalty = DRIFT_KELLY_PENALTY_FACTOR
        logger.warning(
            f"  [V2-P3-DRIFT] KELLY PENALTY ACTIVA: {n_critical} features en CRITICAL "
            f"(>= {DRIFT_KELLY_PENALTY_N_FEATURES}). "
            f"max_position reducido al {kelly_penalty:.0%} para compensar distribuciones contaminadas."
        )
    elif n_critical > 0:
        # Penalizacion parcial proporcional al numero de features en drift
        kelly_penalty = max(0.5, 1.0 - (n_critical / max(len(results), 1)) * 2)
        logger.warning(f"  [V2-P3-DRIFT] KELLY PENALTY PARCIAL: {kelly_penalty:.0%} -> max_position reducido.")

    return {
        "results":        sorted_results,
        "n_drifted":      n_critical,
        "n_warning":      n_warning,
        "n_stable":       n_stable,
        "kelly_penalty":  kelly_penalty,
        "worst_features": worst_5,
    }
