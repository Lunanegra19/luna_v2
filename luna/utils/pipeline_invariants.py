"""
pipeline_invariants.py — Luna V1
=================================
Sistema de trazabilidad en dos capas:

  CAPA 1 — Config Invariants (para pre_flight_check.py):
    Detecta inconsistencias en settings.yaml ANTES de lanzar el pipeline.
    Función: check_config_consistency(cfg) -> list[str]

  CAPA 2 — Runtime Invariants (para logs en tiempo de ejecución):
    Detecta anomalías en DataFrames y flujos de datos DURANTE el pipeline.
    Funciones: check_trades_df(), check_oos_df(), check_hmm_config(), etc.

Historial de bugs detectables con este módulo:
  BUG-01: n_trials_total=600 != optuna_trials=100 → DSR inflado artificialmente
  BUG-02: oos_trades sin DatetimeIndex → WFV reporta posiciones, no fechas
  BUG-03: threshold baja a 0.45 silenciosamente → señales no calibradas en OOS
  BUG-04: hmm_allowed_regimes con enteros → filtro inestable entre runs

Uso:
  # En pre_flight_check.py (antes de correr):
  from luna.utils.pipeline_invariants import check_config_consistency
  errors = check_config_consistency(cfg)

  # En generate_oos_predictions.py (en runtime):
  from luna.utils.pipeline_invariants import check_trades_df, check_oos_df
  check_oos_df(df_oos)
  check_trades_df(df_trades)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import pandas as pd
import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CAPA 1 — Config Invariants
# ─────────────────────────────────────────────────────────────────────────────


# AUDIT Tier 3 (BUG-HOLDOUT-PATH) helper — inserted by audit patch
def _get_holdout_path(data_dir: "pathlib.Path") -> "pathlib.Path":
    """Devuelve features_holdout_{window_id}.parquet si existe, si no el genérico."""
    import os as _os_pi
    _w = _os_pi.environ.get("LUNA_WINDOW_ID", "")
    _specific = data_dir / f"features_holdout_{_w}.parquet"
    if _w and _specific.exists():
        return _specific
    return data_dir / "features_holdout.parquet"


def check_config_consistency(cfg) -> list[str]:
    """
    Verifica la consistencia interna de la configuración en settings.yaml.
    Retorna una lista de strings de error. Lista vacía = OK.

    Llamar desde pre_flight_check.py antes de lanzar el pipeline.

    Errores detectados:
      - BUG-01: stat.n_trials_total != xgboost.optuna_trials
      - BUG-04: hmm_allowed_regimes contiene enteros (inestable entre runs)
      - ARCH-01: verificar que n_trials en statistical_audit apunta a optuna_trials
      - DATA-01: min_trades demasiado bajo para el framework estadístico
    """
    errors = []

    # ── BUG-01: n_trials_total debe sincronizarse con optuna_trials ──────────
    try:
        n_trials_stat  = int(getattr(cfg.stat, 'n_trials_total', -1)) if getattr(cfg, 'stat', None) else -1
        n_trials_optuna = int(getattr(cfg.xgboost, 'optuna_trials', -1)) if getattr(cfg, 'xgboost', None) else -1
        if n_trials_stat > 0 and n_trials_optuna > 0 and n_trials_stat != n_trials_optuna:
            errors.append(
                f"BUG-01: stat.n_trials_total={n_trials_stat} != xgboost.optuna_trials={n_trials_optuna}. "
                f"El DSR penalizará como si se probaran {n_trials_stat} hipótesis, "
                f"pero Optuna solo corre {n_trials_optuna}. Sincronizar ambos valores."
            )
    except Exception as e:
        errors.append(f"BUG-01: no se pudo verificar n_trials ({e})")

    # ── BUG-04: hmm_allowed_regimes debe contener strings semánticos ─────────
    try:
        _hmm_cfg = cfg.metalabeler if cfg else None
        _allowed = _hmm_cfg.hmm_allowed_regimes if _hmm_cfg else None
        if _allowed is not None:
            int_entries = [x for x in _allowed if isinstance(x, int)]
            if int_entries:
                errors.append(
                    f"BUG-04: hmm_allowed_regimes contiene enteros: {int_entries}. "
                    f"El HMM reasigna índices en cada run — usar etiquetas semánticas: "
                    f"['1_BULL_TREND', '2_VOLATILE_BULL', '3_BEAR_CRASH', '4_CALM_BEAR']."
                )
    except Exception as e:
        errors.append(f"BUG-04: no se pudo verificar hmm_allowed_regimes ({e})")

    # ── DATA-01: min_trades mínimo para estadística válida ───────────────────
    try:
        min_trades = int(cfg.stat.min_trades)
        if min_trades < 50:
            errors.append(
                f"DATA-01: stat.min_trades={min_trades} < 50. "
                f"Con menos de 50 trades el test binomial tiene poder estadístico < 40%. "
                f"Valor recomendado: >= 100."
            )
    except Exception as e:
        errors.append(f"DATA-01: no se pudo leer stat.min_trades ({e}). Política No-Fallback activa.")

    # ── SEGURIDAD: xgb_min_signals_threshold no puede ser menor que threshold_sweep_min ──
    try:
        xgb_cfg  = getattr(cfg, 'xgboost', None)
        t_sweep  = float(float(xgb_cfg.threshold_sweep_min))
        t_min_em = float(float(xgb_cfg.xgb_min_signals_threshold))
        if t_min_em < t_sweep:
            errors.append(
                f"BUG-03: xgb_min_signals_threshold={t_min_em} < threshold_sweep_min={t_sweep}. "
                f"El umbral de emergencia no puede ser menor que el mínimo del sweep de calibración."
            )
    except Exception:
        pass

    return errors


def check_config_consistency_or_raise(cfg) -> str:
    """
    Versión que lanza ValueError si hay errores críticos.
    Para usar en __main__ de scripts que requieren consistencia total.
    """
    errors = check_config_consistency(cfg)
    if errors:
        msg = "\n  ".join(errors)
        raise ValueError(
            f"[pipeline_invariants] {len(errors)} inconsistencias de configuración:\n  {msg}"
        )
    return f"Config OK — {len(errors)} errores"


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 2 — Runtime Invariants
# ─────────────────────────────────────────────────────────────────────────────

def check_trades_df(df: pd.DataFrame, context: str = "oos_trades") -> None:
    """
    Verifica invariantes del DataFrame de trades OOS.
    Emite logger.warning() para cada anomalía detectada.
    No lanza excepciones — solo alerta para no interrumpir el pipeline.

    Llamar justo ANTES de guardar oos_trades.parquet y DESPUÉS de cargarlo
    en run_statistical_validation.py.

    Anomalías detectadas:
      - BUG-02: índice no es DatetimeIndex → WFV inútil
      - BUG-03: threshold_was_lowered → señales con calidad inferior en OOS
      - DATA-01: n_trades < 30 → todo el framework estadístico es inválido
      - DATA-02: win_rate < 0.35 → señal de mismatch de régimen real
    """
    tag = f"[INVARIANT:{context}]"

    # BUG-02: DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning(
            "{} índice NO es DatetimeIndex ({}) — WFV reportará posiciones de fila "
            "en vez de fechas reales. Guardar con index=True y timestamp como índice.",
            tag, type(df.index).__name__
        )
    else:
        # Verificar que las fechas son razonables (no en el pasado lejano ni futuro)
        try:
            min_date = df.index.min()
            max_date = df.index.max()
            if min_date.year < 2019:
                logger.warning("{} fecha mínima {} < 2019 — posible error de timezone o índice corrupto.", tag, min_date.date())
            if max_date.year > 2030:
                logger.warning("{} fecha máxima {} > 2030 — posible timestamp corrupto.", tag, max_date.date())
        except Exception:
            pass

    # BUG-03: threshold_was_lowered
    if "threshold_was_lowered" in df.columns:
        n_lowered = int(df["threshold_was_lowered"].sum())
        if n_lowered > 0:
            logger.warning(
                "{} {} / {} trades ({:.0f}%) entraron con threshold de emergencia (threshold_was_lowered=True). "
                "Estas señales no fueron aprobadas por el calibrador EV. "
                "Bajar xgb_min_signals_threshold en settings.yaml o revisar el modelo.",
                tag, n_lowered, len(df), 100 * n_lowered / max(len(df), 1)
            )

    # DATA-01: volumen mínimo de trades
    n = len(df)
    if n < 30:
        logger.warning(
            "{} solo {} trades — insuficiente para CUALQUIER test estadístico. "
            "El DSR, PBO y test binomial son inútiles con n < 30. "
            "Revisar el embudo de filtros (HMM, MetaLabeler, OOD, threshold).",
            tag, n
        )
    elif n < 100:
        logger.warning(
            "{} {} trades < 100 mínimo SOP. El test binomial tiene poder < 60%. "
            "Resultados estadísticos son orientativos, no concluyentes.",
            tag, n
        )

    # DATA-02: win rate anómalamente bajo
    if "is_win" in df.columns and n >= 10:
        wr = float(df["is_win"].mean())
        if wr < 0.30:
            logger.warning(
                "{} win_rate={:.1f}% < 30% — el modelo está prediciendo el lado incorrecto. "
                "Con PT={}/SL={} un WR < 33% es peor que random. "
                "Investigar mismatch de régimen 2025 vs training.",
                tag, wr * 100, "1.5x", "1.0x"
            )
        elif wr > 0.70:
            logger.warning(
                "{} win_rate={:.1f}% > 70% — posible leakage o costos no aplicados. "
                "Verificar que se descuentan costos round-trip.",
                tag, wr * 100
            )

    # DATA-03: Sharpe anómalamente extremo
    if "return_pct" in df.columns and n >= 10:
        std_r = df["return_pct"].std()
        if std_r > 1e-10:
            sharpe = (df["return_pct"].mean() / std_r) * np.sqrt(365 * 24)
            if sharpe < -5:
                logger.warning(
                    "{} Sharpe={:.2f} < -5 — pérdidas concentradas temporalmente. "
                    "Probable que varios stop-loss consecutivos en el mismo período. "
                    "Revisar el WFV por ventana temporal para identificar el período problemático.",
                    tag, sharpe
                )


def check_oos_df(df: pd.DataFrame, context: str = "df_oos") -> None:
    """
    Verifica invariantes del DataFrame de features OOS (antes de generar trades).
    Emite warnings para anomalías de datos que pueden distorsionar el pipeline.

    Anomalías detectadas:
      - HMM_Regime ausente o todo NaN → filtro HMM no tendrá efecto
      - xgb_prob ausente → modelo no corrió correctamente
      - Demasiados NaN en features críticas
      - Rango de fechas inconsistente con holdout_start de settings
    """
    tag = f"[INVARIANT:{context}]"
    n = len(df)

    # HMM_Regime
    if "HMM_Regime" not in df.columns:
        logger.warning(
            "{} HMM_Regime ausente en df_oos — el filtro HMM [H3] no tendrá efecto. "
            "Verificar que hmm_regime.pkl existe y se cargó correctamente.",
            tag
        )
    elif df["HMM_Regime"].isna().mean() > 0.50:
        nan_pct = df["HMM_Regime"].isna().mean() * 100
        logger.warning(
            "{} HMM_Regime es {:.0f}% NaN — filtro HMM inefectivo en la mayoría de barras.",
            tag, nan_pct
        )

    # xgb_prob
    if "xgb_prob" not in df.columns:
        logger.warning(
            "{} xgb_prob ausente en df_oos — el modelo XGBoost no generó predicciones.",
            tag
        )
    elif df["xgb_prob"].isna().mean() > 0.20:
        logger.warning(
            "{} xgb_prob tiene {:.0f}% NaN — muchas barras sin predicción XGB.",
            tag, df["xgb_prob"].isna().mean() * 100
        )

    # Rango temporal mínimo: holdout debe ser >= 2025
    if isinstance(df.index, pd.DatetimeIndex) and n > 0:
        try:
            from config.settings import cfg
            holdout_start = pd.Timestamp(cfg.temporal_splits.holdout_start, tz="UTC")
            df_start = df.index.min()
            if df_start.tz is None:
                df_start = df_start.tz_localize("UTC")
            if df_start < holdout_start:
                logger.warning(
                    "{} datos OOS comienzan en {} pero holdout_start={} — "
                    "se están usando datos de validation (semi-conocidos) como OOS. "
                    "Generar features_holdout.parquet con datos post-{}.",
                    tag, df_start.date(), holdout_start.date(), holdout_start.date()
                )
        except Exception:
            pass


def check_hmm_config(cfg, state_map: dict | None = None) -> list[str]:
    """
    Verifica la consistencia de la configuración HMM.
    Si state_map está disponible (del pkl), verifica que hmm_allowed_regimes
    mapea a etiquetas reales del modelo actual.

    Retorna lista de warnings. Lista vacía = OK.
    """
    warnings_out = []
    _hmm_cfg  = cfg.metalabeler if cfg else None
    _allowed  = _hmm_cfg.hmm_allowed_regimes if _hmm_cfg else None

    if _allowed is None:
        return []  # null = pass-through, no hay nada que verificar

    # Verificar que son strings
    int_entries = [x for x in _allowed if isinstance(x, int)]
    if int_entries:
        warnings_out.append(
            f"hmm_allowed_regimes contiene enteros {int_entries} — inestable entre runs. "
            f"Usar etiquetas semánticas como '1_BULL_TREND'."
        )

    # Si tenemos el state_map actual, verificar que las etiquetas existen
    if state_map and all(isinstance(x, str) for x in _allowed):
        existing_labels = set(state_map.values())
        invalid = [x for x in _allowed if x not in existing_labels]
        if invalid:
            warnings_out.append(
                f"hmm_allowed_regimes contiene etiquetas no existentes en el modelo actual: {invalid}. "
                f"Etiquetas disponibles: {sorted(existing_labels)}. "
                f"Re-entrenar o ajustar hmm_allowed_regimes."
            )

    return warnings_out


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 3 — Feature Integrity (degeneración de señales)
# ─────────────────────────────────────────────────────────────────────────────

def check_frozen_features(
    df: pd.DataFrame,
    context: str = "features",
    min_days_frozen: int = 7,
    max_constant_pct: float = 0.95,
    exclude_cols: list | None = None,
    raise_on_critical: bool = False,
) -> list[str]:
    """
    [FIX-VIX-ZSCORE-01] Detecta features degeneradas o "congeladas" en un DataFrame.

    Un feature está degenerado si:
      - Tiene un único valor en toda la serie (std == 0)
      - Tiene el mismo valor durante más de `min_days_frozen` días consecutivos
      - Más del `max_constant_pct`% de sus valores son idénticos

    Motivación: VIX_Zscore estuvo congelado en 0.6020 durante 354 días (2025-04-01 a
    2026-03-20) sin ser detectado. El XGBoost recibía siempre el mismo valor para esa
    feature crítica, degradando silenciosamente todas las predicciones del holdout.

    Args:
        df: DataFrame con features (index debe ser DatetimeIndex para detección de días).
        context: Nombre del contexto para el log (ej: "features_holdout").
        min_days_frozen: Días mínimos con el mismo valor para considerarse congelado.
        max_constant_pct: Fracción máxima de valores idénticos (0.95 = 95%).
        exclude_cols: Columnas a ignorar (ej: flags binarias que son casi siempre 0).
        raise_on_critical: Si True, lanza ValueError cuando se detectan columnas críticas.

    Returns:
        Lista de strings con los problemas detectados. Lista vacía = OK.

    Uso:
        from luna.utils.pipeline_invariants import check_frozen_features
        issues = check_frozen_features(holdout_df, context="features_holdout")
        for issue in issues:
            logger.warning(issue)
    """
    tag = f"[FROZEN-FEAT:{context}]"
    issues: list[str] = []

    _exclude = set(exclude_cols or [])

    # Columnas binarias/flag o estáticas de red conocidas — no reportar como degeneradas
    _binary_ok = {
        "HMM_Regime", "VIX_Regime", "VIX_Spike", "Macro_Risk_On",
        "Recession_Signal", "SP500_AboveMA200",
        "macro_ohl_tension_z", "tribe_wr_zscore", "meta_oracle_score", 
        "KMeans_Tribe_ID", "Master_Causal_Signal", "Whale_Activity_Flag", "GBTC_Volume", "alpha_genetic_score", "dv_funding_rate", "cal_is_fomc_week", "cal_is_expiry_week", "cal_quarter_end", "vix_regime", "USDCNY", "FedFundsRate_z90d"
    }
    _exclude |= _binary_ok

    # Ignorar variables macroeconómicas (lógico que se congelen > 14 días)
    # [M1-FIX] Ampliado: cubrir prefijos en minúscula y strings exactos documentados
    _monthly_prefixes = (
        "M2_", "CPI_", "Inflation_", "PCE", "NFP",
        "mc_", "yield_curve", "YieldCurve",
        "m2_", "unemploy_", "Macro_Risk_",   # prefijos minúscula + derivadas
        "oc_unemploy_", "mc_unemploy_",       # variantes derivadas mc_/oc_
    )
    _macro_monthly_ok = {c for c in df.columns if c.startswith(_monthly_prefixes)}

    # Features individuales de baja frecuencia que no tienen un prefijo consistente
    _exact_low_freq = {
        "T10Y2Y",              # Treasury 10Y-2Y spread — publicado diariamente pero no cambia en horas
        "UnemployRate",        # Tasa desempleo — mensual
        "Unemploy_Rate",       # alias
        "unemploy_rate_raw",   # raw mensual
        "GBTC_Premium_Raw",    # semanal aproximado
        "ms_cvd_spot_vs_perps",# depende de Futures_Volume (API a veces 100% NaN)
        "Macro_Risk_Score",    # índice compuesto de publicación irregular
        "Recession_Signal",    # binario raro — casi siempre 0
        "hash_ribbon_signal",  # señal binaria de baja frecuencia (~1/mes)
    }
    _exclude |= _macro_monthly_ok | _exact_low_freq

    # Determinar si el index tiene frecuencia horaria o diaria
    is_hourly = False
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        median_gap = pd.Series(df.index).diff().dropna().median()
        is_hourly = median_gap < pd.Timedelta("2h")

    # Filas por día aproximadas
    rows_per_day = 24 if is_hourly else 1
    min_rows_frozen = min_days_frozen * rows_per_day

    for col in df.columns:
        if col in _exclude:
            continue

        series = df[col].dropna()
        if len(series) < 10:
            continue  # Serie demasiado corta para diagnosticar

        # ── Check 1: única única (std == 0) ─────────────────────────────
        if series.nunique() == 1:
            val = series.iloc[0]
            val_str = f"{val:.6g}" if isinstance(val, (int, float)) and not isinstance(val, bool) else str(val)
            issues.append(
                f"{tag} '{col}': UN SOLO VALOR en toda la serie ({val_str}). "
                f"Feature completamente degenerada — no aporta información al modelo. "
                f"Revisar el pipeline de construcción de esta columna."
            )
            logger.warning(
                "{} '{}': UN SOLO VALOR ({}) en {} filas — feature degenerada.",
                tag, col, val_str, len(series)
            )
            continue  # no hacer más checks si ya es constante total

        # ── Check 2: racha máxima de valor idéntico ──────────────────────
        # Detecta periodos donde el valor no cambia (run-length encoding)
        value_changes = series != series.shift(1)
        run_ids = value_changes.cumsum()
        run_lengths = run_ids.map(run_ids.value_counts())

        max_run = int(run_lengths.max())
        if max_run >= min_rows_frozen:
            # Identificar el período de la racha más larga
            longest_run_id = run_ids[run_lengths == max_run].iloc[0]
            frozen_mask = run_ids == longest_run_id
            frozen_series = series[frozen_mask]
            frozen_val = frozen_series.iloc[0]
            days_frozen = max_run / rows_per_day

            frozen_start = frozen_series.index.min()
            frozen_end   = frozen_series.index.max()

            issues.append(
                f"{tag} '{col}': congelado en {frozen_val:.6g} durante "
                f"{days_frozen:.0f} dias ({frozen_start.date()} -> {frozen_end.date()}). "
                f"Racha maxima: {max_run} filas. Verificar fetch incremental y rolling windows."
            )
            logger.warning(
                "{} '{}': congelado en {:.6g} durante {:.0f} dias ({} -> {}). "
                "Posible bug en fetch incremental o rolling window calculado sobre delta.",
                tag, col, frozen_val, days_frozen,
                frozen_start.date(), frozen_end.date()
            )

        # ── Check 3: porcentaje de valores más frecuente ─────────────────
        most_common_pct = series.value_counts(normalize=True).iloc[0]
        if most_common_pct >= max_constant_pct and series.nunique() > 1:
            top_val = series.value_counts().index[0]
            issues.append(
                f"{tag} '{col}': {most_common_pct*100:.1f}% de filas con valor {top_val:.6g}. "
                f"Feature casi constante — señal muy débil para el modelo."
            )
            logger.warning(
                "{} '{}': {:.1f}% de filas con mismo valor ({:.6g}) — casi constante.",
                tag, col, most_common_pct * 100, top_val
            )

    if issues:
        logger.warning(
            "{} RESUMEN: {} features con problemas de integridad detectados. "
            "Los modelos ML pueden verse afectados silenciosamente.",
            tag, len(issues)
        )
        if raise_on_critical:
            # En modo crítico, solo se lanza si hay features con 1 solo valor (total degeneración)
            total_degenerate = [i for i in issues if "UN SOLO VALOR" in i]
            if total_degenerate:
                raise ValueError(
                    f"[FROZEN-FEAT] {len(total_degenerate)} features completamente degeneradas:\n"
                    + "\n".join(total_degenerate)
                )
    else:
        logger.info("{} OK — ninguna feature degenerada o congelada detectada.", tag)

    return issues
