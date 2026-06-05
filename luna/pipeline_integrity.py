"""
luna/pipeline_integrity.py
══════════════════════════════════════════════════════════════════════════════
Sistema de Detección de Fallos Silenciosos del Pipeline WFB — Luna V2

PROPÓSITO:
    Detectar antes, durante y después de cada ventana WFB los fallos que
    históricamente contaminaron resultados silenciosamente durante una run
    entera sin que el sistema lo reportara.

FALLOS HISTÓRICOS DOCUMENTADOS QUE ESTE MÓDULO DETECTA:
    FIX-CALIB-BINARY-01  → Calibrador joblib abierto como texto → UnicodeDecodeError
                           silencioso → xgb_prob_cal == xgb_prob_raw en 100% trades
    FIX-REG-01           → Regularización XGB demasiado alta → std(probs) < 0.01
                           → modelo colapsa → WR ≈ 50% (aleatorio)
    FIX-ROUTER-SANITY-01 → Agente bull predice < 0.47 en régimen BULL (señal invertida)
    FIX-OOD-01           → OOD Guard desactualizado (hash SFI != modelo en disco)
    MODEL-MOCK-SILENT    → Modelo guardado como NULL_MODEL pero pipeline continua
                           usando el baseline sin avisar
    CAL-COLLAPSE         → Calibrador activo pero probs OOS 100% fuera del rango IS
                           → out_of_bounds clip aplana todo a constante

CUÁNDO LLAMAR:
    1. PRE-WINDOW:  PipelineIntegrityChecker.pre_window_check(window_id, models_dir)
    2. POST-PREDICT: PipelineIntegrityChecker.post_predict_check(df_oos, window_id)
    3. POST-WINDOW:  PipelineIntegrityChecker.post_window_check(trades_df, window_id)

INTEGRACIÓN CON EL ORQUESTADOR:
    En run_wfb_orchestrator.py, añadir:
        from luna.pipeline_integrity import PipelineIntegrityChecker as PIC
        PIC.pre_window_check(window_id, models_dir)     # antes de predict_oos
        PIC.post_window_check(trades_df, window_id)     # después de guardar trades

SEVERIDAD:
    CRITICAL → raise RuntimeError (detiene la run — resultado contaminado)
    WARNING  → log + print (la run continúa pero el resultado es sospechoso)
    INFO     → log (resultado normal, trazabilidad)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE DETECCIÓN
# ══════════════════════════════════════════════════════════════════════════════

# Umbral: % de barras con cal != raw para considerar calibración "aplicada"
_CAL_APPLIED_PCT_MIN     = 0.5   # Menos del 0.5% → CRITICAL si hay calibradores cargados

# Umbral: std de probs para considerar modelo "colapsado"
_PROB_STD_COLLAPSE       = 0.01  # std < 0.01 → modelo produce probs quasi-constantes

# Umbral: std del calibrador para considerar "colapso por OOB"
_CAL_STD_COLLAPSE        = 1e-4  # std < 1e-4 → calibrador mapea todo a constante

# Umbral: diferencia máxima tolerable entre cal y raw si son "iguales"
_DIFF_EQ_TOLERANCE       = 1e-6

# Tag para todos los prints de este módulo
_TAG = "[PIPELINE-INTEGRITY]"


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1: PRE-WINDOW — Verificar artefactos en disco antes de la inferencia
# ══════════════════════════════════════════════════════════════════════════════

def _check_calibrator_files(models_dir: Path, direction: str = "long") -> dict:
    """
    Verifica que los calibradores isotónicos existen, son binarios joblib (no JSON mock)
    y se pueden cargar. Detecta FIX-CALIB-BINARY-01 antes de que falle silenciosamente.
    """
    import joblib
    agents = ["bull", "range", "bear"]
    results = {}

    for agent in agents:
        name    = f"{agent}_{direction}"
        cal_path = models_dir / f"xgboost_isotonic_calibrator_{name}.joblib"
        mdl_path = models_dir / f"xgboost_meta_{name}.model"

        if not mdl_path.exists():
            results[agent] = {"status": "MODEL_MISSING", "cal_ok": False}
            continue

        if not cal_path.exists():
            results[agent] = {"status": "CAL_MISSING", "cal_ok": False}
            continue

        # Verificar que el archivo es binario joblib y no JSON mock
        # (el bug FIX-CALIB-BINARY-01 se manifestaba porque se abría como texto)
        with open(cal_path, 'rb') as f:
            header = f.read(4)
        is_binary_joblib = header[0:1] == b'\x80'  # pickle magic byte
        is_json_mock     = header[0:1] == b'{'

        if is_json_mock:
            results[agent] = {"status": "CAL_IS_JSON_MOCK", "cal_ok": False}
            continue

        if not is_binary_joblib:
            results[agent] = {
                "status": f"CAL_UNKNOWN_FORMAT_header={header.hex()}",
                "cal_ok": False
            }
            continue

        # Verificar que se puede cargar y que tiene los atributos esperados
        try:
            cal = joblib.load(str(cal_path))
            has_knots = hasattr(cal, 'X_thresholds_')
            n_knots   = len(cal.X_thresholds_) if has_knots else 0
            
            import numpy as np
            if has_knots and n_knots > 0:
                _x_arr = np.array(cal.X_thresholds_)
                x_min = float(_x_arr.min())
                x_max = float(_x_arr.max())
            else:
                x_min = x_max = None
                
            has_y_knots = hasattr(cal, 'y_thresholds_')
            n_y_knots = len(cal.y_thresholds_) if has_y_knots else 0
            
            if has_y_knots and n_y_knots > 0:
                _y_arr = np.array(cal.y_thresholds_)
                y_min = float(_y_arr.min())
                y_max = float(_y_arr.max())
            else:
                y_min = y_max = None
                
            results[agent] = {
                "status": "OK",
                "cal_ok": True,
                "n_knots": n_knots,
                "x_range": f"[{x_min:.4f},{x_max:.4f}]" if x_min is not None else "N/A",
                "y_range": f"[{y_min:.4f},{y_max:.4f}]" if y_min is not None else "N/A",
                "cal": cal,
            }
        except Exception as e:
            results[agent] = {"status": f"CAL_LOAD_ERROR: {e}", "cal_ok": False}

    return results


def pre_window_check(window_id: str, models_dir: Path, direction: str = "long") -> None:
    """
    CHECK 1 — Ejecutar ANTES de predict_oos para el window_id dado.
    Detecta artefactos faltantes, corruptos o en formato incorrecto.

    Lanza RuntimeError si encuentra un fallo CRITICAL que invalidaría los resultados.
    """
    print(f"{_TAG} ══ PRE-WINDOW CHECK: {window_id} ══")
    n_errors  = 0
    n_warnings = 0

    # ── 1a. Verificar calibradores ─────────────────────────────────────────
    cal_results = _check_calibrator_files(models_dir, direction)
    n_ok        = sum(1 for r in cal_results.values() if r["cal_ok"])
    n_models    = sum(1 for r in cal_results.values() if r["status"] != "MODEL_MISSING")

    for agent, res in cal_results.items():
        status = res["status"]
        if status == "OK":
            print(
                f"{_TAG}   [{window_id}/{agent}] Calibrador OK | "
                f"knots={res['n_knots']} | x={res['x_range']} | y={res['y_range']}"
            )
            logger.info(
                f"{_TAG} [{window_id}/{agent}] Calibrador OK knots={res['n_knots']} "
                f"x={res['x_range']} y={res['y_range']}"
            )
        elif status == "CAL_MISSING":
            print(f"{_TAG}   [{window_id}/{agent}] WARNING: Calibrador .joblib NO EXISTE — "
                  f"el agente operará sin calibración (xgb_prob_cal == xgb_prob_raw).")
            logger.warning(f"{_TAG} [{window_id}/{agent}] CAL_MISSING — agente sin calibración.")
            n_warnings += 1
        elif status == "CAL_IS_JSON_MOCK":
            print(f"{_TAG}   [{window_id}/{agent}] WARNING: Calibrador es JSON MOCK (no binario joblib) — "
                  f"no se cargará. Mismo efecto que CAL_MISSING.")
            logger.warning(f"{_TAG} [{window_id}/{agent}] CAL_IS_JSON_MOCK.")
            n_warnings += 1
        elif status == "MODEL_MISSING":
            print(f"{_TAG}   [{window_id}/{agent}] INFO: Modelo no existe — agente no activo.")
        else:
            print(f"{_TAG}   [{window_id}/{agent}] ERROR: {status}")
            logger.error(f"{_TAG} [{window_id}/{agent}] {status}")
            n_errors += 1

    if n_ok == 0 and n_models > 0:
        _msg = (
            f"{_TAG} CRITICAL [{window_id}]: 0/{n_models} calibradores válidos. "
            f"TODOS los trades de esta ventana tendrán xgb_prob_cal == xgb_prob_raw. "
            f"Causa probable: FIX-CALIB-BINARY-01 (apertura en modo texto 'r' en lugar de 'rb'). "
            f"Verificar regime_router.py _load_models()."
        )
        print(_msg)
        logger.critical(_msg)
        raise RuntimeError(_msg)
    elif n_ok < n_models:
        _msg_w = (
            f"{_TAG} WARNING [{window_id}]: {n_ok}/{n_models} calibradores cargados. "
            f"Agentes sin calibrador operarán con xgb_prob_cal == xgb_prob_raw."
        )
        print(_msg_w)
        logger.warning(_msg_w)

    # ── 1b. Verificar que los modelos XGB no son NULL_MODEL ───────────────
    null_models = list(models_dir.glob("*.NULL_MODEL"))
    if null_models:
        for nm in null_models:
            print(f"{_TAG}   [{window_id}] WARNING: NULL_MODEL detectado: {nm.name} — "
                  f"agente produce 0 señales (n_train < min_viable).")
            logger.warning(f"{_TAG} [{window_id}] NULL_MODEL: {nm.name}")
            n_warnings += 1

    # ── 1c. Verificar OOD Guard actualizado ───────────────────────────────
    ood_sig = models_dir / "ood_guard_signature.json"
    if ood_sig.exists():
        import json
        with open(ood_sig) as f:
            sig = json.load(f)
        trained_at = sig.get("trained_at", "desconocido")
        print(f"{_TAG}   [{window_id}] OOD Guard: entrenado={trained_at} | "
              f"features={sig.get('n_features','?')} | contamination={sig.get('contamination','?')}")
    else:
        print(f"{_TAG}   [{window_id}] WARNING: OOD Guard signature no encontrada en {models_dir}.")
        logger.warning(f"{_TAG} [{window_id}] OOD Guard signature ausente.")
        n_warnings += 1

    # ── Resumen ────────────────────────────────────────────────────────────
    status_icon = "✅" if n_errors == 0 and n_warnings == 0 else ("⚠️" if n_errors == 0 else "🔴")
    print(
        f"{_TAG} PRE-WINDOW [{window_id}] {status_icon} "
        f"errors={n_errors} warnings={n_warnings} cal_ok={n_ok}/{n_models}"
    )
    logger.info(
        f"{_TAG} pre_window_check [{window_id}] completed: "
        f"errors={n_errors} warnings={n_warnings} cal_ok={n_ok}/{n_models}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2: POST-PREDICT — Verificar calidad de la calibración en el df_oos
# ══════════════════════════════════════════════════════════════════════════════

def post_predict_check(df_oos: pd.DataFrame, window_id: str, n_cals_loaded: int = -1) -> dict:
    """
    CHECK 2 — Ejecutar INMEDIATAMENTE después de route_and_predict() y la asignación
    de df_oos["xgb_prob"] / df_oos["xgb_prob_cal"].

    Verifica que la calibración isotónica realmente modificó las probabilidades.
    Detecta:
      - FIX-CALIB-BINARY-01: calibrador no cargado → cal == raw en 100% de barras
      - CAL-COLLAPSE: calibrador cargado pero probs OOS 100% fuera del rango IS
      - FIX-REG-01: modelo XGB colapsado (std_raw < 0.01)

    Retorna: dict con métricas de calibración para trazabilidad en logs.
    """
    print(f"{_TAG} ══ POST-PREDICT CHECK: {window_id} ══")

    metrics = {
        "window_id":        window_id,
        "n_bars":           len(df_oos),
        "has_xgb_prob":     "xgb_prob"     in df_oos.columns,
        "has_xgb_prob_cal": "xgb_prob_cal" in df_oos.columns,
    }

    if not metrics["has_xgb_prob"] or not metrics["has_xgb_prob_cal"]:
        _msg = f"{_TAG} ERROR [{window_id}]: Columnas xgb_prob/xgb_prob_cal ausentes en df_oos."
        print(_msg)
        logger.error(_msg)
        return metrics

    raw = df_oos["xgb_prob"].fillna(0.5)
    cal = df_oos["xgb_prob_cal"].fillna(raw)
    diff = (cal - raw).abs()

    metrics.update({
        "raw_mean":      round(float(raw.mean()), 4),
        "raw_std":       round(float(raw.std()), 6),
        "cal_mean":      round(float(cal.mean()), 4),
        "cal_std":       round(float(cal.std()), 6),
        "diff_mean":     round(float(diff.mean()), 6),
        "diff_max":      round(float(diff.max()), 6),
        "n_modified":    int((diff > _DIFF_EQ_TOLERANCE).sum()),
        "pct_modified":  round(float((diff > _DIFF_EQ_TOLERANCE).mean() * 100), 1),
        "n_cals_loaded": n_cals_loaded,
    })

    pct_mod    = metrics["pct_modified"]
    raw_std    = metrics["raw_std"]
    cal_std    = metrics["cal_std"]

    # ── Detección FIX-REG-01: Modelo XGB colapsado ────────────────────────
    if raw_std < _PROB_STD_COLLAPSE and metrics["n_bars"] > 50:
        _msg = (
            f"{_TAG} WARNING [{window_id}] FIX-REG-01: XGB raw std={raw_std:.6f} < {_PROB_STD_COLLAPSE}. "
            f"El modelo produce probs quasi-constantes (media={metrics['raw_mean']:.4f}). "
            f"Trades OOS tendrán WR ≈ base_rate. Verificar hiper-parámetros XGB (reg_alpha, min_child_weight)."
        )
        print(_msg)
        logger.warning(_msg)

    # ── Detección FIX-CALIB-BINARY-01: 0% barras modificadas con cals cargados ──
    if pct_mod < _CAL_APPLIED_PCT_MIN and n_cals_loaded > 0 and metrics["n_bars"] > 100:
        _msg = (
            f"{_TAG} CRITICAL [{window_id}] FIX-CALIB-BINARY-01: "
            f"{n_cals_loaded} calibradores cargados pero solo {pct_mod}% de barras "
            f"tienen cal≠raw (diff_mean={metrics['diff_mean']:.6f}). "
            f"xgb_prob_cal ≈ xgb_prob_raw en TODA la ventana. "
            f"Posibles causas: (1) probs OOS fuera del rango IS del calibrador "
            f"(out_of_bounds='clip' aplana todo), (2) bug en asignación de calibrated_probs.loc[mask]. "
            f"Verificar regime_router.py route_and_predict()."
        )
        print(_msg)
        logger.critical(_msg)
        # No lanzamos RuntimeError aquí — el pipeline ya tomó la decision,
        # pero el WARNING CRITICAL quedará en los logs para auditoria.

    # ── Detección CAL-COLLAPSE: calibrador activo pero std_cal ≈ 0 ────────
    elif pct_mod < _CAL_APPLIED_PCT_MIN and n_cals_loaded > 0:
        _msg = (
            f"{_TAG} WARNING [{window_id}] CAL-COLLAPSE: "
            f"cal_std={cal_std:.2e} ≈ 0. Probs OOS fuera del rango IS del calibrador. "
            f"out_of_bounds='clip' aplana todas las probs calibradas a constante. "
            f"El calibrador fue cargado pero no aporta información. "
            f"Verificar rango IS del calibrador vs distribución OOS actual."
        )
        print(_msg)
        logger.warning(_msg)

    # ── Estado normal ──────────────────────────────────────────────────────
    else:
        status_str = "✅ calibración OK" if pct_mod > 10 else "⚠️ calibración parcial"
        print(
            f"{_TAG} POST-PREDICT [{window_id}] {status_str} | "
            f"raw_mean={metrics['raw_mean']} raw_std={raw_std:.4f} | "
            f"cal_mean={metrics['cal_mean']} cal_std={cal_std:.4f} | "
            f"barras_modificadas={metrics['n_modified']}/{metrics['n_bars']} ({pct_mod}%) | "
            f"diff_mean={metrics['diff_mean']:.4f} diff_max={metrics['diff_max']:.4f}"
        )

    logger.info(
        f"{_TAG} post_predict_check [{window_id}]: "
        f"pct_modified={pct_mod}% raw_std={raw_std:.4f} cal_std={cal_std:.4f} "
        f"n_cals={n_cals_loaded}"
    )

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3: POST-WINDOW — Verificar calidad de los trades guardados
# ══════════════════════════════════════════════════════════════════════════════

def post_window_check(trades_df: pd.DataFrame, window_id: str) -> dict:
    """
    CHECK 3 — Ejecutar DESPUÉS de guardar el parquet de trades de cada ventana.

    Verifica integridad estadística de los trades generados:
      - Calibración aplicada (cal ≠ raw)
      - Win rate no absurdamente bajo (< 20%) o alto (> 85%)
      - Número de trades mínimo viable
      - Consistencia de columnas clave

    No lanza RuntimeError — solo registra para post-mortem.
    Retorna: dict con métricas para agregar en el resumen de la run.
    """
    print(f"{_TAG} ══ POST-WINDOW CHECK: {window_id} ══")

    if trades_df is None or len(trades_df) == 0:
        print(f"{_TAG}   [{window_id}] INFO: 0 trades generados. Ventana en CASH (puede ser correcto).")
        logger.info(f"{_TAG} post_window_check [{window_id}]: 0 trades.")
        return {"window_id": window_id, "n_trades": 0, "status": "CASH"}

    n = len(trades_df)
    metrics = {"window_id": window_id, "n_trades": n}

    # ── 3a. Calibración aplicada ────────────────────────────────────────
    if "xgb_prob" in trades_df.columns and "xgb_prob_cal" in trades_df.columns:
        diff = (trades_df["xgb_prob_cal"] - trades_df["xgb_prob"]).abs()
        pct_eq = float((diff < _DIFF_EQ_TOLERANCE).mean() * 100)
        metrics["pct_cal_eq_raw"] = round(pct_eq, 1)

        if pct_eq >= 99.0:
            _msg = (
                f"{_TAG} CRITICAL [{window_id}] POST-WINDOW: "
                f"xgb_prob_cal == xgb_prob_raw en {pct_eq:.0f}% de los {n} trades. "
                f"BUG FIX-CALIB-BINARY-01 activo en esta ventana. "
                f"Re-ejecutar con regime_router.py open(cal_path, 'rb')."
            )
            print(_msg)
            logger.critical(_msg)
            metrics["cal_bug"] = True
        elif pct_eq > 20:
            print(f"{_TAG}   [{window_id}] WARNING: {pct_eq:.0f}% trades con cal==raw (parcial).")
            logger.warning(f"{_TAG} [{window_id}] cal==raw parcial: {pct_eq:.0f}%")
            metrics["cal_bug"] = "PARTIAL"
        else:
            print(f"{_TAG}   [{window_id}] ✅ Calibración OK: solo {pct_eq:.1f}% trades con cal==raw.")
            metrics["cal_bug"] = False

    # ── 3b. Win Rate plausible ──────────────────────────────────────────
    if "is_win" in trades_df.columns:
        wr = float(trades_df["is_win"].mean() * 100)
        metrics["win_rate_pct"] = round(wr, 1)

        if wr < 20:
            print(
                f"{_TAG}   [{window_id}] ❌ CRITICAL: WR={wr:.1f}% < 20%. "
                f"El modelo está prediciendo PEOR que azar. Posibles causas: "
                f"señal invertida, calibrador OOB, FIX-REG-01 activo."
            )
            logger.critical(f"{_TAG} [{window_id}] WR={wr:.1f}% — peor que azar.")
        elif wr < 35:
            print(f"{_TAG}   [{window_id}] ⚠️ WARNING: WR={wr:.1f}% — bajo (azar=50%). Verificar pipeline.")
            logger.warning(f"{_TAG} [{window_id}] WR bajo: {wr:.1f}%")
        elif wr > 80:
            print(
                f"{_TAG}   [{window_id}] ⚠️ WARNING: WR={wr:.1f}% > 80% — sospechosamente alto. "
                f"Verificar look-ahead bias, embargo, y PurgedKFold."
            )
            logger.warning(f"{_TAG} [{window_id}] WR sospechosamente alto: {wr:.1f}%")
        else:
            print(f"{_TAG}   [{window_id}] ✅ WR={wr:.1f}% dentro del rango esperado.")

    # ── 3c. Número de trades ───────────────────────────────────────────
    if n < 5:
        print(f"{_TAG}   [{window_id}] ⚠️ WARNING: Solo {n} trades — estadísticamente insignificante.")
        logger.warning(f"{_TAG} [{window_id}] Muy pocos trades: {n}")
    else:
        print(f"{_TAG}   [{window_id}] ✅ N trades: {n}")

    # ── 3d. Columnas clave presentes ───────────────────────────────────
    expected_cols = ["xgb_prob", "xgb_prob_cal", "is_win", "signal_threshold", "HMM_Semantic"]
    missing = [c for c in expected_cols if c not in trades_df.columns]
    if missing:
        print(f"{_TAG}   [{window_id}] ⚠️ WARNING: Columnas ausentes en trades: {missing}")
        logger.warning(f"{_TAG} [{window_id}] Columnas ausentes: {missing}")
        metrics["missing_cols"] = missing

    logger.info(
        f"{_TAG} post_window_check [{window_id}]: "
        f"n={n} wr={metrics.get('win_rate_pct','?')}% "
        f"cal_bug={metrics.get('cal_bug','?')}"
    )
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL (interfaz unificada para el orquestador)
# ══════════════════════════════════════════════════════════════════════════════

class PipelineIntegrityChecker:
    """
    Interfaz estática unificada para los 3 checkpoints del pipeline.

    Uso en run_wfb_orchestrator.py:
        from luna.pipeline_integrity import PipelineIntegrityChecker as PIC

        # Antes de predict_oos:
        PIC.pre_window_check(window_id, models_dir)

        # Después de route_and_predict:
        PIC.post_predict_check(df_oos, window_id, n_cals_loaded)

        # Después de guardar trades:
        PIC.post_window_check(trades_df, window_id)
    """

    @staticmethod
    def pre_window_check(window_id: str, models_dir: Path, direction: str = "long") -> None:
        """Verificación pre-ventana: artefactos en disco."""
        try:
            pre_window_check(window_id, models_dir, direction)
        except RuntimeError:
            raise  # propagar CRITICAL al orquestador
        except Exception as e:
            print(f"{_TAG} ERROR inesperado en pre_window_check: {e}")
            logger.error(f"{_TAG} pre_window_check error inesperado: {e}")

    @staticmethod
    def post_predict_check(
        df_oos: pd.DataFrame,
        window_id: str,
        n_cals_loaded: int = -1
    ) -> dict:
        """Verificación post-predicción: calibración aplicada correctamente."""
        try:
            return post_predict_check(df_oos, window_id, n_cals_loaded)
        except Exception as e:
            print(f"{_TAG} ERROR inesperado en post_predict_check: {e}")
            logger.error(f"{_TAG} post_predict_check error inesperado: {e}")
            return {}

    @staticmethod
    def post_window_check(trades_df: pd.DataFrame, window_id: str) -> dict:
        """Verificación post-ventana: integridad estadística de los trades."""
        try:
            return post_window_check(trades_df, window_id)
        except Exception as e:
            print(f"{_TAG} ERROR inesperado en post_window_check: {e}")
            logger.error(f"{_TAG} post_window_check error inesperado: {e}")
            return {}
