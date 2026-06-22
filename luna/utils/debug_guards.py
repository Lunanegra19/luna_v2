"""
debug_guards.py
===============
Luna V1 — Sistema de Diagnóstico Centralizado

Controla el nivel de verbosidad mediante:
  - cfg.debug.verbose (settings.yaml → debug.verbose: true)
  - env var LUNA_VERBOSE=1 (sobreescribe el YAML)

Todas las funciones son NO-OPs silenciosos cuando verbose=False → cero
overhead en producción. Los guards de invariantes DUROS (check_invariant)
siempre están activos pues protegen contra errores matemáticos críticos.

Uso típico:
    from luna.utils.debug_guards import vlog, check_df_sanity, check_invariant, timeit

    with timeit("Fase FracDiff"):
        df = apply_fracdiff(df)

    check_df_sanity(df, "post-fracdiff")
    check_invariant(0 <= kelly <= 1, f"Kelly fuera de [0,1]: {kelly}")
"""
from __future__ import annotations

import os
import time
import math
import contextlib
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Inicialización del flag verbose
# ---------------------------------------------------------------------------

def _load_verbose() -> bool:
    """
    Prioridad:
      1. LUNA_VERBOSE env var (=1 → True, =0 → False)
      2. cfg.debug.verbose (settings.yaml)
      3. False (default seguro)
    """
    env = os.environ.get("LUNA_VERBOSE", "").strip()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    try:
        from config.settings import cfg
        return bool(int(getattr(cfg.debug), "verbose", False))
    except Exception:
        return False


VERBOSE: bool = _load_verbose()

def _cfg_val(key: str, default: Any) -> Any:
    """Lee un valor de cfg.debug.key con fallback seguro."""
    try:
        from config.settings import cfg
        return int(getattr(cfg.debug), key, default)
    except Exception:
        return default


NAN_THRESHOLD_PCT:   float = _cfg_val("nan_threshold_pct",   5.0)
CORR_LEAKAGE_THR:   float = _cfg_val("corr_leakage_threshold", 0.95)
OUTLIER_SIGMA:      float = _cfg_val("outlier_sigma", 6.0)
LOG_TIMING:         bool  = _cfg_val("log_timing", True)

# ---------------------------------------------------------------------------
# Helpers de bajo nivel
# ---------------------------------------------------------------------------

def vlog(msg: str, level: str = "DEBUG") -> None:
    """Log condicional — solo si VERBOSE está activo."""
    if VERBOSE:
        getattr(logger, level.lower())(f"[VERBOSE] {msg}")


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


# ---------------------------------------------------------------------------
# 1. check_df_sanity — inspección general de un DataFrame
# ---------------------------------------------------------------------------

def check_df_sanity(
    df: pd.DataFrame,
    label: str = "",
    *,
    raise_on_all_nan: bool = False,
) -> None:
    """
    Analiza shape, NaN, inf y dtypes de un DataFrame.
    Siempre loguea el resumen a INFO; si VERBOSE, añade detalles columna a columna.
    """
    tag = f"[{label}]" if label else ""
    n_rows, n_cols = df.shape
    n_nan   = int(df.isnull().sum().sum())
    n_inf   = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
    pct_nan = 100 * n_nan / max(n_rows * n_cols, 1)

    level = "WARNING" if (pct_nan > NAN_THRESHOLD_PCT or n_inf > 0) else "INFO"
    getattr(logger, level.lower())(
        f"[SANITY]{tag} shape=({n_rows}×{n_cols}) | "
        f"NaN={n_nan} ({_fmt_pct(pct_nan)}) | "
        f"inf={n_inf} | "
        f"idx=[{df.index.min()} → {df.index.max()}]"
    )

    if raise_on_all_nan and n_nan == n_rows * n_cols:
        raise ValueError(f"[SANITY]{tag} DataFrame completamente vacío (todo NaN)")

    if not VERBOSE:
        return

    # Detalle por columna
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    bad_cols = []
    for col in numeric_cols:
        col_nan  = df[col].isnull().sum()
        col_inf  = np.isinf(df[col].replace([None], np.nan).dropna()).sum()
        col_pct  = 100 * col_nan / max(n_rows, 1)
        if col_pct > NAN_THRESHOLD_PCT or col_inf > 0:
            bad_cols.append(f"  ├ {col}: NaN={col_nan} ({_fmt_pct(col_pct)}), inf={col_inf}")

    if bad_cols:
        logger.warning(f"[VERBOSE][SANITY]{tag} Columnas problemáticas ({len(bad_cols)}):")
        for line in bad_cols[:20]:  # cap a 20 para no inundar
            logger.warning(line)
    else:
        logger.debug(f"[VERBOSE][SANITY]{tag} Todas las columnas limpias ✓")


# ---------------------------------------------------------------------------
# 2. check_distributions — outliers y correlaciones sospechosas
# ---------------------------------------------------------------------------

def check_distributions(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    label: str = "",
) -> None:
    """
    Detecta outliers (>OUTLIER_SIGMA σ) y correlaciones muy altas (posible leakage).
    Solo activo con VERBOSE.
    """
    if not VERBOSE:
        return

    tag  = f"[{label}]" if label else ""
    cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()

    # Outliers
    outlier_report = []
    zero_var_cols  = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 10:
            continue
        std = s.std()
        if std == 0 or math.isnan(std):
            zero_var_cols.append(col)
            continue
        n_out = int((((s - s.mean()) / std).abs() > OUTLIER_SIGMA).sum())
        if n_out > 0:
            outlier_report.append(f"  ├ {col}: {n_out} outliers (>{OUTLIER_SIGMA}σ)")

    if zero_var_cols:
        logger.warning(f"[VERBOSE][DIST]{tag} Columnas con varianza CERO: {zero_var_cols}")
    if outlier_report:
        logger.warning(f"[VERBOSE][DIST]{tag} Outliers detectados:")
        for line in outlier_report[:15]:
            logger.warning(line)

    # Correlaciones sospechosas (leakage proxy)
    num_df = df[cols].select_dtypes(include=[np.number]).dropna(how="all")
    if num_df.shape[1] > 1:
        try:
            corr = num_df.corr().abs()
            # Upper triangle sin diagonal
            mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
            high_corr = [
                f"  ├ {corr.columns[i]} ↔ {corr.columns[j]}: r={corr.iloc[i,j]:.3f}"
                for i in range(corr.shape[0])
                for j in range(corr.shape[1])
                if mask[i, j] and corr.iloc[i, j] >= CORR_LEAKAGE_THR
            ]
            if high_corr:
                logger.warning(
                    f"[VERBOSE][DIST]{tag} {len(high_corr)} pares con corr≥{CORR_LEAKAGE_THR} "
                    f"(posible leakage/redundancia):"
                )
                for line in high_corr[:10]:
                    logger.warning(line)
        except Exception as exc:
            logger.debug(f"[VERBOSE][DIST]{tag} No se pudo calcular correlaciones: {exc}")


# ---------------------------------------------------------------------------
# 3. check_target_balance — class imbalance
# ---------------------------------------------------------------------------

def check_target_balance(y: pd.Series | np.ndarray, label: str = "") -> None:
    """Reporta distribución de clases. Siempre activo."""
    tag   = f"[{label}]" if label else ""
    arr   = np.asarray(y)
    n     = len(arr)
    vals, counts = np.unique(arr[~np.isnan(arr.astype(float))], return_counts=True)
    parts = [f"clase {int(v)}={c} ({100*c/n:.1f}%)" for v, c in zip(vals, counts)]
    ratio = counts.max() / max(counts.min(), 1)
    level = "WARNING" if ratio > 3.0 else "INFO"
    getattr(logger, level.lower())(
        f"[TARGET]{tag} n={n} | {' | '.join(parts)} | imbalance ratio={ratio:.2f}x"
    )


# ---------------------------------------------------------------------------
# 4. check_temporal_split — solape entre splits
# ---------------------------------------------------------------------------

def check_temporal_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    holdout: pd.DataFrame,
    label: str = "",
) -> None:
    """Verifica que los 3 splits no se solapen. Siempre activo."""
    tag = f"[{label}]" if label else ""
    t_end = train.index.max()
    v_start = val.index.min()
    v_end   = val.index.max()
    h_start = holdout.index.min()

    issues = []
    if v_start <= t_end:
        issues.append(f"SOLAPE train/val: val_start={v_start} <= train_end={t_end}")
    if h_start <= v_end:
        issues.append(f"SOLAPE val/holdout: holdout_start={h_start} <= val_end={v_end}")

    for issue in issues:
        logger.error(f"[SPLIT]{tag} ⚠️  {issue}")

    logger.info(
        f"[SPLIT]{tag} train={len(train)} ({train.index.min().date()}→{t_end.date()}) | "
        f"val={len(val)} ({v_start.date()}→{v_end.date()}) | "
        f"holdout={len(holdout)} ({h_start.date()}→{holdout.index.max().date()}) | "
        f"{'⚠️ SOLAPE' if issues else '✅ OK'}"
    )


# ---------------------------------------------------------------------------
# 5. check_model_sanity — métricas básicas de un clasificador
# ---------------------------------------------------------------------------

def check_model_sanity(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label: str = "",
    threshold: float = 0.5,
) -> None:
    """Calcula Brier, distribución de probabilidades y accuracy. Siempre activo."""
    tag = f"[{label}]" if label else ""
    try:
        from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
        brier = brier_score_loss(y_true, y_pred_proba)
        auc   = roc_auc_score(y_true, y_pred_proba)
        acc   = accuracy_score(y_true, (y_pred_proba >= threshold).astype(int))
        p_mean, p_std = float(np.mean(y_pred_proba)), float(np.std(y_pred_proba))
        p_min,  p_max = float(np.min(y_pred_proba)),  float(np.max(y_pred_proba))

        level = "WARNING" if brier > 0.25 or auc < 0.52 else "INFO"
        getattr(logger, level.lower())(
            f"[MODEL]{tag} Brier={brier:.4f} | AUC={auc:.4f} | Acc@{threshold}={acc:.3f} | "
            f"prob [mean={p_mean:.3f} std={p_std:.3f} min={p_min:.3f} max={p_max:.3f}]"
        )

        if VERBOSE:
            # Histograma en texto (10 bins)
            counts, edges = np.histogram(y_pred_proba, bins=10, range=(0, 1))
            hist_str = " | ".join(
                f"{edges[i]:.1f}-{edges[i+1]:.1f}:{counts[i]}"
                for i in range(len(counts))
            )
            logger.debug(f"[VERBOSE][MODEL]{tag} Distribución prob: {hist_str}")
    except Exception as exc:
        logger.warning(f"[MODEL]{tag} No se pudieron calcular métricas: {exc}")


# ---------------------------------------------------------------------------
# 6. check_kelly — invariante matemático del position sizer
# ---------------------------------------------------------------------------

def check_kelly(fraction: float, label: str = "") -> None:
    """Valida la fracción Kelly. CRITICAL si >0.5 (Doble Kelly). Siempre activo.
    [FIX-KELLY-SANITY-01 2026-05-31]
    """
    tag = f"[{label}]" if label else ""
    if math.isnan(fraction) or math.isinf(fraction):
        _msg = (
            f"[FIX-KELLY-SANITY-01/CRITICAL] Kelly INVALIDO{tag}: {fraction} (NaN o inf). "
            f"Error matematico en calculo de win_rate/avg_win/avg_loss. LA RUN SE DETIENE."
        )
        print(_msg)
        logger.critical(_msg)
        raise RuntimeError(_msg)
    
    if fraction > 1.0:
        # Más de Full Kelly: Destrucción matemática garantizada.
        _msg = (
            f"[FIX-KELLY-SANITY-01/CRITICAL] Kelly EXCESIVO{tag}: f*={fraction:.4f} > 1.0 "
            f"(Sobre Full-Kelly). Según la teoría de Kelly, superar el óptimo empuja al sistema a EV negativa. "
            f"Verificar win_rate/avg_win/avg_loss en el position sizer. LA RUN SE DETIENE."
        )
        logger.critical(_msg)
        raise RuntimeError(_msg)
    elif fraction > 0.5:
        _msg = (
            f"[FIX-KELLY-SANITY-01/WARNING] Kelly elevado{tag}: f*={fraction:.4f} > 0.5. "
            f"Asumiendo entorno ESMA Retail (Apalancamiento Max x2). "
            f"Si el apalancamiento es institucional (x10+), ESTO ES EXTREMADAMENTE PELIGROSO."
        )
        logger.warning(
            "[FIX-KELLY-SANITY-01] Kelly elevado %s: f*=%.4f > 0.50", tag, fraction
        )
    
    if not (0.0 <= fraction <= 1.0):
        logger.error(f"[KELLY]{tag} fracción FUERA de [0,1]: {fraction:.4f}")
    else:
        logger.debug(f"[KELLY]{tag} fracción Kelly={fraction:.4f} OK")




# ---------------------------------------------------------------------------
# 7. check_numeric_stability — arrays genéricos
# ---------------------------------------------------------------------------

def check_numeric_stability(arr: np.ndarray, label: str = "") -> None:
    """Detecta NaN, inf y overflow float32. Siempre activo."""
    tag   = f"[{label}]" if label else ""
    arr_f = np.asarray(arr, dtype=float)
    n_nan = int(np.isnan(arr_f).sum())
    n_inf = int(np.isinf(arr_f).sum())
    # Overflow float32 si valores > 3.4e38
    n_of  = int((np.abs(arr_f) > 3.4e38).sum())

    issues = []
    if n_nan > 0:  issues.append(f"NaN={n_nan}")
    if n_inf > 0:  issues.append(f"inf={n_inf}")
    if n_of  > 0:  issues.append(f"overflow_f32={n_of}")

    if issues:
        logger.error(f"[NUMERIC]{tag} ⚠️  {', '.join(issues)} en array shape={arr_f.shape}")
    elif VERBOSE:
        logger.debug(f"[VERBOSE][NUMERIC]{tag} OK — shape={arr_f.shape}")


# ---------------------------------------------------------------------------
# 8. check_invariant — guard duro siempre activo
# ---------------------------------------------------------------------------

def check_invariant(condition: bool, msg: str) -> None:
    """
    Equivalente a assert pero siempre activo (no desactivable con -O).
    Si la condición falla: loguea ERROR (no lanza excepción para no romper producción).
    """
    if not condition:
        logger.error(f"[INVARIANT] ⚠️  VIOLACIÓN: {msg}")


# ---------------------------------------------------------------------------
# 9. timeit — context manager de timing siempre activo (si LOG_TIMING)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def timeit(label: str):
    """
    Mide el tiempo de ejecución de un bloque.
    Activo si cfg.debug.log_timing=true (default).

    Uso:
        with timeit("Fase FracDiff"):
            df = apply_fracdiff(df)
    """
    if not LOG_TIMING:
        yield
        return
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        logger.info(f"[TIMING] {label}: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# 10. log_dataframe_transition — antes/después de una transformación
# ---------------------------------------------------------------------------

def log_dataframe_transition(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    step_label: str,
) -> None:
    """
    Loguea cambios de shape y NaN entre dos versiones del mismo DataFrame.
    Solo activo con VERBOSE.
    """
    if not VERBOSE:
        return
    r_before, c_before = df_before.shape
    r_after,  c_after  = df_after.shape
    nan_before = int(df_before.isnull().sum().sum())
    nan_after  = int(df_after.isnull().sum().sum())

    new_cols = [c for c in df_after.columns if c not in df_before.columns]
    drop_cols = [c for c in df_before.columns if c not in df_after.columns]

    logger.debug(
        f"[VERBOSE][TRANSITION] ── {step_label} ──\n"
        f"  shape: ({r_before}×{c_before}) → ({r_after}×{c_after})\n"
        f"  NaN:   {nan_before} → {nan_after} (Δ={nan_after - nan_before:+d})\n"
        f"  +cols ({len(new_cols)}): {new_cols[:10]}\n"
        f"  -cols ({len(drop_cols)}): {drop_cols[:10]}"
    )


# ---------------------------------------------------------------------------
# 11. log_memory_usage — RSS del proceso
# ---------------------------------------------------------------------------

def log_memory_usage(label: str = "") -> None:
    """Loguea el uso de memoria RSS del proceso actual. Solo con VERBOSE."""
    if not VERBOSE:
        return
    try:
        import psutil, os as _os
        proc = psutil.Process(_os.getpid())
        rss_mb = proc.memory_info().rss / 1024 / 1024
        tag = f"[{label}]" if label else ""
        logger.debug(f"[VERBOSE][MEM]{tag} RSS={rss_mb:.1f} MB")
    except ImportError:
        pass  # psutil opcional


# ---------------------------------------------------------------------------
# 12. log_series_stats — estadísticas de una Serie numérica
# ---------------------------------------------------------------------------

def log_series_stats(s: pd.Series, label: str = "") -> None:
    """Loguea mean/std/min/max de una Serie. Solo con VERBOSE."""
    if not VERBOSE:
        return
    tag = f"[{label}]" if label else ""
    vals = s.dropna()
    if len(vals) == 0:
        logger.warning(f"[VERBOSE][STATS]{tag} Serie vacía")
        return
    logger.debug(
        f"[VERBOSE][STATS]{tag} n={len(vals)} | "
        f"mean={vals.mean():.4f} | std={vals.std():.4f} | "
        f"min={vals.min():.4f} | max={vals.max():.4f} | "
        f"NaN={s.isnull().sum()}"
    )
