"""
regime_stratified_validator.py
==============================
Luna V2 — Capa 2: Validación Condicional por Régimen.

PROBLEMA QUE RESUELVE
---------------------
En el WFO cronológico clásico, el set de validación corresponde a los últimos
N meses de calendario. Si esos meses tuvieron un régimen diferente al del OOS,
la calibración de umbrales es inválida (e.g., W1: val=lateral Nov-Dic, OOS=Bull+Crash).

SOLUCIÓN
--------
Construir el set de validación extrayendo las últimas `val_samples_per_regime`
velas de cada régimen HMM dentro del periodo IS (train_end estricto), garantizando:
    1. Causalidad total (ningún dato posterior a train_end incluido).
    2. Representatividad de cada régimen independiente del calendario.
    3. Fallback cronológico si algún régimen tiene < min_samples.

INTEGRIDAD CAUSAL (SOP Luna v2 R1)
-------------------------------
- El filtro temporal `df.loc[:train_end]` es OBLIGATORIO antes de cualquier `.tail()`.
- Se respeta el embargo_hours entre la última muestra del val y el inicio del OOS.
- No se mezclan datos del holdout en ningún caso.

USO
---
from luna.validation.regime_stratified_validator import build_regime_val_set

df_val = build_regime_val_set(
    df=df_is,                        # DataFrame IS completo (ya filtrado hasta train_end)
    regime_col='HMM_Regime',
    train_end='2024-10-31',
    samples_per_regime=300,          # settings.yaml: wfb.val_samples_per_regime
    min_samples_per_regime=50,       # settings.yaml: wfb.val_min_samples_per_regime
    embargo_hours=96,                # settings.yaml: sop.embargo_hours
    fallback_months=2,               # Meses cronológicos si no hay datos suficientes
)
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLES_PER_REGIME: int = 300
_DEFAULT_MIN_SAMPLES: int = 50
_DEFAULT_FALLBACK_MONTHS: int = 2
_DEFAULT_EMBARGO_HOURS: int = 96

# Regímenes semánticos estándar del sistema HMM de Luna V1
_KNOWN_REGIMES_PRIORITY = [
    # Alta prioridad: regímenes operables frecuentes
    "1_BULL_TREND", "1_BULL_TREND_B", "1_BULL_TREND_C", "1_BULL_TREND_D",
    "1_VOLATILE_BULL", "1_VOLATILE_BULL_B", "1_VOLATILE_BULL_C", "1_VOLATILE_BULL_D",
    "1_BULL_TREND_WEAK", "1_BULL_GRIND",
    "2_CALM_RANGE", "2_CALM_RANGE_B", "2_CALM_RANGE_C", "2_CALM_RANGE_D",
    "2_VOLATILE_RANGE", "2_VOLATILE_RANGE_B", "2_VOLATILE_RANGE_C", "2_VOLATILE_RANGE_D",
    # Media prioridad: regímenes bear (menos muestras históricas disponibles)
    "3_CALM_BEAR", "3_CALM_BEAR_B", "3_CALM_BEAR_C", "3_CALM_BEAR_D",
    "3_BEAR_CRASH", "3_BEAR_CRASH_B", "3_BEAR_CRASH_C", "3_BEAR_CRASH_D",
    "4_BEAR_FORCED",
]


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def build_regime_val_set(
    df: pd.DataFrame,
    regime_col: str = "HMM_Regime",
    train_end: Optional[str] = None,
    samples_per_regime: int = _DEFAULT_SAMPLES_PER_REGIME,
    min_samples_per_regime: int = _DEFAULT_MIN_SAMPLES,
    embargo_hours: int = _DEFAULT_EMBARGO_HOURS,
    fallback_months: int = _DEFAULT_FALLBACK_MONTHS,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Construye un set de validación estratificado por régimen HMM.

    El set de validación resultante contiene las últimas `samples_per_regime`
    velas de cada régimen presente en el periodo IS, garantizando representatividad
    de cada tipo de mercado para calibración de umbrales.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame IS completo, con índice DatetimeTZAware (UTC) y columna de régimen.
    regime_col : str
        Nombre de la columna HMM de régimen semántico.
    train_end : str, optional
        Fecha de corte IS en formato 'YYYY-MM-DD'. Si None, se usa max(df.index).
        CRÍTICO: se aplica como filtro duro para garantizar causalidad.
    samples_per_regime : int
        Número de velas a extraer por régimen (últimas cronológicamente dentro del IS).
    min_samples_per_regime : int
        Mínimo absoluto por régimen. Si no se alcanza, se usa fallback cronológico.
    embargo_hours : int
        Horas de embargo que deben quedar FUERA del val (antes del OOS).
        Las últimas `embargo_hours` del IS se excluyen del val set.
    fallback_months : int
        Meses cronológicos a usar como fallback si ningún régimen tiene suficientes datos.
    random_state : int
        Semilla para reproducibilidad (no se usa shuffle, solo para logging).

    Retorna
    -------
    pd.DataFrame
        Set de validación estratificado. El índice es temporal (UTC). Sin duplicados.

    Raises
    ------
    ValueError
        Si df está vacío o no contiene la columna de régimen.
    """
    if df is None or df.empty:
        raise ValueError("[REGIME-VAL] df está vacío. No se puede construir el set de validación.")
    if regime_col not in df.columns:
        raise ValueError(f"[REGIME-VAL] Columna de régimen '{regime_col}' no encontrada en df. "
                         f"Columnas disponibles: {list(df.columns)[:10]}")

    # --- 1. Filtro causal duro (SOP R1) ---
    if train_end is not None:
        cutoff = pd.Timestamp(train_end, tz="UTC")
        df_is = df[df.index <= cutoff].copy()
        logger.info(f"[REGIME-VAL] Filtro causal aplicado: {len(df)} → {len(df_is)} filas "
                    f"(corte={train_end})")
    else:
        df_is = df.copy()
        logger.warning("[REGIME-VAL] train_end no especificado — usando todo df como IS. "
                       "Verificar causalidad manualmente.")

    if df_is.empty:
        logger.error("[REGIME-VAL] df_is vacío tras filtro de train_end. "
                     "Usando fallback cronológico.")
        return _chronological_fallback(df, fallback_months)

    # --- 2. Excluir embargo (últimas N horas del IS no se usan en validación) ---
    if embargo_hours > 0:
        embargo_cutoff = df_is.index.max() - pd.Timedelta(hours=embargo_hours)
        df_is_no_embargo = df_is[df_is.index <= embargo_cutoff].copy()
        n_embargoed = len(df_is) - len(df_is_no_embargo)
        logger.info(f"[REGIME-VAL] Embargo aplicado: {n_embargoed} velas excluidas "
                    f"(últimas {embargo_hours}H antes del OOS)")
        df_is = df_is_no_embargo

    if df_is.empty:
        logger.error("[REGIME-VAL] df_is vacío tras embargo. Fallback cronológico.")
        return _chronological_fallback(df, fallback_months)

    # --- 3. Construcción estratificada por régimen ---
    present_regimes = df_is[regime_col].dropna().unique().tolist()
    # Ordenar por prioridad conocida, poniendo los desconocidos al final
    ordered_regimes = [r for r in _KNOWN_REGIMES_PRIORITY if r in present_regimes]
    unknown_regimes = [r for r in present_regimes if r not in _KNOWN_REGIMES_PRIORITY]
    ordered_regimes += unknown_regimes

    logger.info(f"[REGIME-VAL] Regímenes encontrados: {len(present_regimes)} — "
                f"Conocidos={len(ordered_regimes)-len(unknown_regimes)}, "
                f"Desconocidos={len(unknown_regimes)}")

    val_chunks: list[pd.DataFrame] = []
    regime_stats: dict[str, dict] = {}
    fallback_regimes: list[str] = []

    for regime in ordered_regimes:
        regime_df = df_is[df_is[regime_col] == regime]
        n_available = len(regime_df)

        if n_available < min_samples_per_regime:
            logger.warning(f"[REGIME-VAL]   Régimen '{regime}': {n_available} velas disponibles "
                           f"< mínimo {min_samples_per_regime} → marcado para fallback cronológico")
            fallback_regimes.append(regime)
            regime_stats[regime] = {"n_available": n_available, "n_sampled": 0, "mode": "SKIPPED"}
            continue

        # Extraer las últimas N velas (las más recientes dentro del IS)
        n_sample = min(samples_per_regime, n_available)
        chunk = regime_df.iloc[-n_sample:]
        val_chunks.append(chunk)
        regime_stats[regime] = {
            "n_available": n_available,
            "n_sampled": n_sample,
            "date_min": chunk.index.min().strftime("%Y-%m-%d"),
            "date_max": chunk.index.max().strftime("%Y-%m-%d"),
            "mode": "STRATIFIED",
        }
        logger.debug(f"[REGIME-VAL]   '{regime}': {n_sample} velas "
                     f"[{regime_stats[regime]['date_min']} → {regime_stats[regime]['date_max']}]")

    # --- 4. Logging del resumen de estratificación ---
    total_stratified = sum(s["n_sampled"] for s in regime_stats.values())
    logger.info(f"[REGIME-VAL] Resumen estratificación: "
                f"{len(val_chunks)} regímenes con datos, "
                f"{len(fallback_regimes)} sin datos suficientes, "
                f"total velas={total_stratified}")

    # --- 5. Fallback cronológico si no hay suficientes regímenes ---
    if not val_chunks:
        logger.error("[REGIME-VAL] 0 chunks estratificados. Usando fallback cronológico completo.")
        return _chronological_fallback(df_is, fallback_months)

    # --- 6. Concatenar y deduplicar ---
    df_val = pd.concat(val_chunks, axis=0)
    df_val = df_val[~df_val.index.duplicated(keep="last")]
    df_val = df_val.sort_index()

    # --- 7. Verificación de salud del set de validación ---
    _validate_output(df_val, regime_col, train_end, regime_stats)

    logger.info(
        f"[REGIME-VAL] ✅ Set de validación estratificado: shape={df_val.shape} | "
        f"fechas=[{df_val.index.min().date()} → {df_val.index.max().date()}] | "
        f"regímenes={df_val[regime_col].nunique()}"
    )
    return df_val


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _chronological_fallback(df: pd.DataFrame, fallback_months: int) -> pd.DataFrame:
    """Fallback: últimos N meses cronológicos (comportamiento legacy)."""
    cutoff = df.index.max() - pd.DateOffset(months=fallback_months)
    df_fallback = df[df.index >= cutoff].copy()
    logger.warning(
        f"[REGIME-VAL] [FALLBACK-CHRONOLOGICAL] Usando últimos {fallback_months} meses: "
        f"shape={df_fallback.shape} | "
        f"[{df_fallback.index.min().date() if not df_fallback.empty else 'N/A'} → "
        f"{df_fallback.index.max().date() if not df_fallback.empty else 'N/A'}]"
    )
    return df_fallback


def _validate_output(
    df_val: pd.DataFrame,
    regime_col: str,
    train_end: Optional[str],
    regime_stats: dict,
) -> None:
    """Sanity checks sobre el set de validación resultante."""
    issues: list[str] = []

    if df_val.empty:
        issues.append("df_val vacío.")

    if train_end is not None:
        cutoff = pd.Timestamp(train_end, tz="UTC")
        future_leaks = df_val[df_val.index > cutoff]
        if not future_leaks.empty:
            issues.append(
                f"LOOK-AHEAD DETECTADO: {len(future_leaks)} filas posteriores a train_end={train_end}. "
                "ABORTAR ENTRENAMIENTO."
            )

    if regime_col in df_val.columns:
        val_regimes = set(df_val[regime_col].dropna().unique())
        n_regimes = len(val_regimes)
        if n_regimes < 2:
            issues.append(f"Solo {n_regimes} régimen(es) en val_set — calibración sesgada.")

    if issues:
        for issue in issues:
            if "LOOK-AHEAD" in issue:
                logger.error(f"[REGIME-VAL] ❌ {issue}")
                raise RuntimeError(f"[REGIME-VAL] Violación de causalidad: {issue}")
            else:
                logger.warning(f"[REGIME-VAL] ⚠️  {issue}")
    else:
        skipped = [k for k, v in regime_stats.items() if v["mode"] == "SKIPPED"]
        if skipped:
            logger.warning(f"[REGIME-VAL]   Regímenes omitidos por pocas muestras: {skipped}")
