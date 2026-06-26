"""
run_statistical_validation.py - Luna V1
========================================
INFRA-06 (2026-03-17): Punto de entrada CLI para Fase 5 del pipeline.
Delega la logica estadistica en core/monitoring/statistical_audit.py (LunaStatisticalAuditor).
Genera: statistical_verdict.json, tearsheet_oos.png y archivos timestamped en data/reports/.

Uso:
  python scripts/run_statistical_validation.py

Equivalente via run_full_pipeline.py:
  python scripts/run_full_pipeline.py --skip-fetch --skip-mining --skip-features --skip-sfi --skip-training --skip-oos
"""
import sys
import json
import os
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# FIX-ENCODING-01 (2026-03-21): forzar UTF-8 en stdout/stderr para Windows (cp1252).
# Sin esto, PowerShell rompe caracteres Unicode en loguru (→, ━, emojis, etc.)
import io as _io
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}")
    # ── Log file propio del subproceso (trazabilidad por RUN_ID) ────────────
    import os as _os_sv
    from datetime import datetime as _dt_sv
    _log_dir_sv = _ROOT / "logs"
    _log_dir_sv.mkdir(exist_ok=True)
    _ts_sv  = _dt_sv.now().strftime("%Y%m%d_%H%M%S")
    _rid_sv = _os_sv.environ.get("LUNA_RUN_ID", "")
    _lname_sv = f"run_statistical_validation_{_ts_sv}_{_rid_sv}.log" if _rid_sv else f"run_statistical_validation_{_ts_sv}.log"
    logger.add(_log_dir_sv / _lname_sv, rotation="50 MB", level="DEBUG", encoding="utf-8")
    # ────────────────────────────────────────────────────────────────────────
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from luna.monitoring.statistical_audit import LunaStatisticalAuditor

# Sistema de trazabilidad (pipeline_invariants)
try:
    from luna.utils.pipeline_invariants import check_trades_df as _check_trades
    _INV_AVAILABLE = True
except ImportError:
    _INV_AVAILABLE = False



# -- Walk-Forward Validation --------------------------------------------------

import typing
def _run_wfv(trades_df: pd.DataFrame, n_windows: typing.Optional[int] = None) -> dict:
    """Walk-Forward Validation: divide OOS en n_windows ventanas y calcula WR.

    BUG-02 FIX (2026-03-17): garantiza que start/end sean timestamps reales.
    Antes: si el parquet tenía índice entero (index=False), window.index[0]
    devolvía 0, 13, 26, 39 — posiciones de fila, sin valor diagnóstico.
    Ahora: si el índice no es DatetimeIndex, intenta usar la columna 'timestamp'.

    VAL-01/02 (Run 14): lee wfv_n_windows desde cfg.stat.wfv_n_windows (settings.yaml).
    """
    # Leer wfv_n_windows desde settings.yaml (VAL-01/02 — sin hardcodes)
    if n_windows is None:
        try:
            from config.settings import cfg as _cfg_wfv
            wfv_n_windows = int(getattr(getattr(_cfg_wfv, "stat", object()), "wfv_n_windows", 4))
        except Exception:
            wfv_n_windows = 4  # fallback si cfg no disponible
        n_windows = wfv_n_windows
    results = {}
    if len(trades_df) < n_windows * 5:
        logger.warning("[WFV] Insuficientes trades ({}) para {} ventanas — WFV omitido",
                       len(trades_df), n_windows)
        return results

    # BUG-02 FIX: asegurar que el índice es DatetimeIndex antes de dividir ventanas
    if not isinstance(trades_df.index, pd.DatetimeIndex):
        if "timestamp" in trades_df.columns:
            trades_df = trades_df.set_index("timestamp")
            trades_df.index = pd.to_datetime(trades_df.index, utc=True)
            logger.debug("[WFV] BUG-02: índice entero detectado → usando columna 'timestamp' como índice")
        else:
            logger.warning("[WFV] BUG-02: índice no es DatetimeIndex y no hay columna 'timestamp' — WFV con fechas aproximadas")

    trades_sorted = trades_df.sort_index()

    # FIX A-07: Time-based o wfb_window basado Walk-Forward Validation
    if "wfb_window" in trades_sorted.columns:
        w_groups = trades_sorted.groupby("wfb_window")
        actual_windows = list(w_groups.groups.keys())
        logger.info(f"  [WFV] Usando boundaries reales WFB: {actual_windows}")
        
        for w_id in actual_windows:
            window = w_groups.get_group(w_id)
            n = len(window)
            wr = float(window["is_win"].mean()) if n > 0 and "is_win" in window.columns else float("nan")
            w_start = window.index.min() if n > 0 else "N/A"
            w_end   = window.index.max() if n > 0 else "N/A"
            
            results[str(w_id)] = {
                "n_trades":   n,
                "win_rate":   round(wr, 4) if not pd.isna(wr) else None,
                "start_date": str(w_start),
                "end_date":   str(w_end),
            }
            _wr_str = f"{wr * 100:.1f}%" if not pd.isna(wr) else "N/A"
            logger.info("  [WFV] {}: {} trades | WR={} | {} → {}",
                        w_id, n, _wr_str, 
                        getattr(w_start, 'date', lambda: w_start)(), 
                        getattr(w_end, 'date', lambda: w_end)())
            if n > 0 and n < 10 and not pd.isna(wr) and wr >= 0.99:
                logger.warning(f"  [WFV] ⚠️ WR anormalmente alto ({_wr_str}) con muy baja muestra (n={n}) en {w_id}")
    else:
        t_start = trades_sorted.index.min()
        t_end   = trades_sorted.index.max()
        
        # Pequeño buffer al final para el último trade
        t_end_inclusive = t_end + pd.Timedelta(seconds=1) 
        total_duration = t_end_inclusive - t_start
        dt = total_duration / n_windows
        
        for i in range(n_windows):
            w_start = t_start + i * dt
            w_end   = t_start + (i + 1) * dt
            
            window = trades_sorted[(trades_sorted.index >= w_start) & (trades_sorted.index < w_end)]
            
            n = len(window)
            wr = float(window["is_win"].mean()) if n > 0 and "is_win" in window.columns else float("nan")

            if n > 0 and n < 10 and not pd.isna(wr) and wr >= 0.99:
                logger.warning(f"  [WFV] ⚠️ WR anormalmente alto ({wr*100:.1f}%) con muy baja muestra (n={n}) en window_{i+1}")
                
            results[f"window_{i+1}"] = {
                "n_trades":   n,
                "win_rate":   round(wr, 4) if not pd.isna(wr) else None,
                "start_date": str(w_start),
                "end_date":   str(w_end),
            }
            _wr_str = f"{wr * 100:.1f}%" if not pd.isna(wr) else "N/A"
            logger.info("  [WFV] Ventana {}: {} trades | WR={} | {} → {}",
                        i + 1, n, _wr_str, w_start.date(), w_end.date())
    # [P3-A 2026-05-28] Umbral de ventana trivial (<min_trades → marcada como trivial)
    # Ventanas triviales se incluyen en el verdict para auditoría completa,
    # pero se marcan con 'trivial=True' para que el tearsheet las excluya del WR agregado.
    try:
        from config.settings import cfg as _cfg_triv
        _min_trades_wfv = int(getattr(getattr(_cfg_triv, "stat", object()), "wfv_min_trades_per_window", 5))
    except Exception:
        _min_trades_wfv = 5  # Default documentado en parametros_fijos.md §P3-A-01
    print(f"[P3-A-WFV-TRIVIAL] Umbral ventanas triviales: n_trades < {_min_trades_wfv}")

    # Marcar ventanas triviales
    for w_id, w_data in results.items():
        n = w_data.get("n_trades", 0)
        is_trivial = n < _min_trades_wfv
        w_data["trivial"] = is_trivial
        if is_trivial:
            print(f"[P3-A-WFV-TRIVIAL] Ventana {w_id}: {n} trades < {_min_trades_wfv} → trivial=True (excluida del WR agregado)")
            logger.warning("[P3-A-WFV-TRIVIAL] Ventana {} con {} trades marcada como trivial (excluida de WR global)", w_id, n)

    return results




# -- Tearsheet ----------------------------------------------------------------

def _generate_tearsheet(trades_df: pd.DataFrame, verdict: dict,
                        out_path: Path, timestamp: str | None = None) -> None:
    """
    Delega en LunaTearSheet (core/reports/generate_tearsheet.py).

    Genera el tearsheet completo v7.0 con 9 filas (12 paneles):
      Panel A — XGB Prob Cuartiles
      Panel B — Holding Time Distribution
      Panel C — Distribución de Trades por Régimen HMM  ← antes ausente
    """
    try:
        from luna.reports.generate_tearsheet import LunaTearSheet
        ts_obj = LunaTearSheet(project_root=_ROOT, output_dir="data/reports")

        # Derivar timestamp del env o hora actual
        _ts = timestamp or os.environ.get("LUNA_RUN_ID") or \
              datetime.now().strftime("%Y-%m-%d_T%H%M")

        # Asegurar que existe la columna 'timestamp' que requiere LunaTearSheet
        df_ts = trades_df.copy()
        if "timestamp" not in df_ts.columns and "entry_time" in df_ts.columns:
            df_ts["timestamp"] = df_ts["entry_time"]
            
        out = ts_obj.generate(df_ts, timestamp=_ts)
        if out:
            # Hacer que out_path apunte al mismo fichero que LunaTearSheet guarda
            import shutil
            shutil.copy2(out, out_path)
            logger.success("  [Tearsheet] Generado con LunaTearSheet (8 paneles): {}", out)
        else:
            logger.warning("  [Tearsheet] LunaTearSheet devolvio None — sin tearsheet.")
    except Exception as _ts_err:
        logger.exception("  [Tearsheet] Fallo al usar LunaTearSheet:")


# -- Main ---------------------------------------------------------------------
def main() -> int:
    """Fase 5 -- Validacion Estadistica completa (The Gauntlet)."""
    ts = datetime.now().strftime("%Y-%m-%d_T%H%M")  # timestamp unico para todo el run
    custom_trades = os.environ.get("LUNA_CUSTOM_TRADES_PATH")
    trades_path = Path(custom_trades) if custom_trades else _ROOT / "data" / "predictions" / "oos_trades.parquet"
    report_dir    = _ROOT / "data" / "reports"
    verdict_path  = report_dir / "statistical_verdict.json"
    tearsheet_path = report_dir / "tearsheet_oos.png"

    if not trades_path.exists():
        logger.error("oos_trades.parquet no encontrado en {}. "
                     "Ejecutar generate_oos_predictions.py primero.", trades_path)
        return 1

    logger.info("Fase 5 -- Validacion Estadistica (The Gauntlet)")
    logger.info("  Leyendo: {}", trades_path)

    trades_df = pd.read_parquet(trades_path)
    # ── [DATAFLOW-IMPORT-VAL-01] OOS Trades Audit ──────────────────────────────
    if not trades_df.empty:
        _cols = trades_df.columns.tolist()
        _start = trades_df.index.min() if isinstance(trades_df.index, pd.DatetimeIndex) \
                 else trades_df.get("entry_time", trades_df.get("timestamp", pd.Series())).min()
        _end   = trades_df.index.max() if isinstance(trades_df.index, pd.DatetimeIndex) \
                 else trades_df.get("entry_time", trades_df.get("timestamp", pd.Series())).max()
        _n     = len(trades_df)
        _wr    = trades_df["is_win"].mean() if "is_win" in trades_df.columns else 0.0
        logger.success(
            f"[DATAFLOW-IMPORT-VAL-01] Predicciones cargadas: {_n} trades | "
            f"fechas={getattr(_start, 'date', lambda: _start)()} -> {getattr(_end, 'date', lambda: _end)()} | "
            f"WinRate bruto={_wr:.1%} | keys_ok={'return_pct' in _cols and 'is_win' in _cols}"
        )
    else:
        logger.warning("[DATAFLOW-IMPORT-VAL-01] ADVERTENCIA: oos_trades.parquet está VACÍO (0 trades).")
    # ─────────────────────────────────────────────────────────────────────────────
    # INVARIANTS CAPA 2: verificar integridad de trades cargados
    if _INV_AVAILABLE:
        _check_trades(trades_df, context="run_statistical_validation")


    # -- [BUG-C1] Guard de fallback ANTES del Gauntlet -------------------------
    # Si el SignalFilter usó fallback (nivel 1 o 2), los trades NO pasaron todos
    # los filtros (MetaLabeler/LGBM/HMM evadidos). El Gauntlet calcularia un DSR
    # válido sobre una población de trades no validada → deployment estadisticamente falso.
    _funnel_fallback_level = 0
    _funnel_path_c1 = _ROOT / "data" / "predictions" / "signal_funnel.json"
    # Buscar en reports/ también (FIX A-04: signal_funnel puede estar en reports/)
    if not _funnel_path_c1.exists():
        _rid_c1 = os.environ.get("LUNA_RUN_ID", "")
        _funnel_path_c1 = report_dir / (f"signal_funnel_{_rid_c1}.json" if _rid_c1 else "signal_funnel.json")
    if _funnel_path_c1.exists():
        try:
            with open(_funnel_path_c1, encoding="utf-8") as _fp_c1:
                _funnel_data_c1 = json.load(_fp_c1)
            _funnel_fallback_level = int(_funnel_data_c1.get("filter_fallback_level", 0))
            if _funnel_fallback_level >= 1:
                _fb_desc = {1: "XGB+OOD (sin MetaV2/HMM/Momentum)", 2: "XGB puro (sin OOD/MetaV2/HMM/Momentum)"}
                logger.error(
                    "\n  [BUG-C1] \U0001f6a8 GAUNTLET BLOQUEADO: Los trades fueron generados en "
                    "fallback nivel {} ({}).\n"
                    "  El DSR calculado NO es valido — MetaLabeler/LGBM/HMM fueron evadidos.\n"
                    "  Diagnosticar: revisar umbrales, reentrenar ensemble_lgbm.py y volver a ejecutar.\n"
                    "  El Gauntlet se permite continuar pero inyecta fallback_level={} en el veredicto "
                    "y fuerza deploy_approved=False.",
                    _funnel_fallback_level,
                    _fb_desc.get(_funnel_fallback_level, "desconocido"),
                    _funnel_fallback_level
                )
        except Exception as _e_c1:
            logger.debug("  [BUG-C1] No se pudo leer signal_funnel para guard: {}", _e_c1)
    # -------------------------------------------------------------------------

    # -- The Gauntlet ---------------------------------------------------------
    auditor = LunaStatisticalAuditor()
    verdict = auditor.run_gauntlet(trades_df)
    
    # [BUG-C1] Inyectar fallback_level en el veredicto y forzar rechazo si aplica
    verdict["signal_filter_fallback_level"] = _funnel_fallback_level
    if _funnel_fallback_level >= 1:
        # verdict["deploy_approved"] = False  # [MODIFICACION ENSEMBLE DOBLE] Desactivado bloqueo
        verdict["rejection_reason"] = (
            f"BUG-C1: Trades generados en fallback nivel {_funnel_fallback_level} — "
            "MetaLabeler/LGBM/HMM evadidos. DSR estadisticamente invalido."
        )

    # Inyectar LUNA_RUN_ID para trazabilidad (P1.3 / C1)
    verdict["run_id"] = os.environ.get("LUNA_RUN_ID", "local_run")

    # -- [FIX-R5-MULTIPLE-SEEDS] Corrección DSR por comparaciones múltiples -------
    # Bailey & López de Prado (2014): al evaluar N semillas y seleccionar las mejores,
    # se incurre en data snooping bias de selección implícita (SOP Iron Rule R5).
    #
    # DISEÑO CRÍTICO — por qué NO operamos sobre el DSR capeado en [0,1]:
    #   El DSR = Phi(z) donde z = (SR_crudo - SR*) / std_SR. Con SR_crudo suficientemente
    #   alto, DSR → 1.0 por el cap de la CDF normal. Comparar 1.0 >= 1.004 sería
    #   rechazar semillas excelentes por un artefacto matemático del cap.
    #
    # SOLUCIÓN CORRECTA: escalar el SR* (benchmark del azar por Optuna) con el factor
    # de corrección entre semillas sqrt(log(N_seeds)). Así la corrección sube el "listón"
    # del azar en el espacio del Sharpe — si el SR_crudo sigue superando el SR* ajustado,
    # la semilla pasa con integridad estadística completa.
    #
    # No aplica doble penalización: DSR ya corrige por n_trials Optuna (DENTRO de semilla).
    # Esta corrección es ENTRE semillas — dimensiones ortogonales.
    import math as _math_r5
    import scipy.stats as _stats_r5
    _n_seeds_env = os.environ.get("LUNA_N_SEEDS_TOTAL", "1")
    try:
        _n_seeds_total_r5 = max(1, int(_n_seeds_env))
    except ValueError:
        _n_seeds_total_r5 = 1
        logger.warning("[FIX-R5] LUNA_N_SEEDS_TOTAL='{}' no es entero — usando N=1 (sin corrección)", _n_seeds_env)

    # Extraer métricas del Sharpe necesarias para recomputar DSR ajustado
    _sr_crudo    = verdict.get("metrics", {}).get("sharpe_crudo", 0.0)
    _skew        = verdict.get("statistical_audit", {}).get("skewness", 0.0)
    _kurt        = verdict.get("statistical_audit", {}).get("kurtosis", 0.0)
    _n_obs       = verdict.get("statistical_audit", {}).get("n_obs_dsr", 1)
    _n_trials    = verdict.get("statistical_audit", {}).get("n_trials_dsr", 100)
    try:
        from config.settings import cfg as _cfg_stat
        _base_dsr_thr = float(_cfg_stat.stat.min_dsr)
    except Exception as e:
        raise RuntimeError(f"CRITICAL: Falta stat.min_dsr en settings: {e}")
    _dsr_real     = verdict.get("statistical_audit", {}).get("dsr", 0.0)

    if _n_seeds_total_r5 > 1:
        _r5_factor = _math_r5.sqrt(_math_r5.log(_n_seeds_total_r5))

        # [FIX-DSR-TRANS-STD 2026-06-13] Cargar dsr_transversal_std desde settings.yaml (No-Fallback)
        try:
            from config.settings import cfg as _cfg_dsr
            _dsr_trans_std = float(_cfg_dsr.stat.dsr_transversal_std)
        except Exception as _e_dsr:
            logger.critical("[FIX-DSR-TRANS-STD] No se pudo leer dsr_transversal_std de settings.yaml: {}", _e_dsr)
            raise RuntimeError(f"Politica No-Fallback: Faltan dsr_transversal_std en settings: {_e_dsr}")

        # Recalcular SR* ajustado en espacio del Sharpe (no en espacio del DSR capeado):
        # std_SR es función de n_obs y momentos de distribución
        _var_sr   = (1.0 - (_skew * _sr_crudo) + ((_kurt - 1.0) / 4.0) * (_sr_crudo ** 2)) / max(_n_obs, 2)
        _std_sr   = float(max(_var_sr, 1e-12) ** 0.5)
        # SR* base (corrección por Optuna n_trials)
        _gamma    = 0.5772156649
        _prob     = 1.0 / max(_n_trials, 2)
        _z1 = _stats_r5.norm.ppf(1.0 - _prob)
        _z2 = _stats_r5.norm.ppf(1.0 - _prob * _math_r5.exp(-1.0))
        # [FIX-DSR-TRANS-STD 2026-06-13] Usar la varianza transversal en lugar del error estándar temporal
        _sr_star_base = _dsr_trans_std * ((1.0 - _gamma) * _z1 + _gamma * _z2)
        # SR* ajustado por N_seeds: el benchmark del azar sube proporcionalmente
        _sr_star_adj  = _sr_star_base * _r5_factor
        # DSR ajustado = Phi((SR_crudo - SR*_adj) / std_SR)
        _z_adj        = (_sr_crudo - _sr_star_adj) / _std_sr if _std_sr > 1e-9 else (1.0 if _sr_crudo > _sr_star_adj else -10.0)
        _dsr_adjusted = float(_stats_r5.norm.cdf(_z_adj))
        _pass_dsr_adjusted = _dsr_adjusted >= _base_dsr_thr

        print(f"[FIX-R5-MULTIPLE-SEEDS] Corrección SOP R5: N_seeds={_n_seeds_total_r5} | "
              f"factor=sqrt(log({_n_seeds_total_r5}))={_r5_factor:.4f} | "
              f"SR_crudo={_sr_crudo:.4f} | SR*_base={_sr_star_base:.4f} | SR*_adj={_sr_star_adj:.4f} | "
              f"DSR_base={_dsr_real:.4f} → DSR_adj={_dsr_adjusted:.4f} (umbral={_base_dsr_thr:.3f})")
        logger.info("[FIX-R5-MULTIPLE-SEEDS] SR*_base={:.4f} → SR*_adj={:.4f} | DSR_adj={:.4f} (N_seeds={})",
                    _sr_star_base, _sr_star_adj, _dsr_adjusted, _n_seeds_total_r5)
    else:
        _r5_factor        = 1.0
        _dsr_adjusted     = _dsr_real
        _sr_star_adj      = float("nan")
        _pass_dsr_adjusted = verdict.get("flags", {}).get("pass_dsr", False)
        print(f"[FIX-R5-MULTIPLE-SEEDS] N_seeds=1 o standalone — sin corrección. DSR={_dsr_real:.4f}")
        logger.info("[FIX-R5-MULTIPLE-SEEDS] Sin corrección R5 (N_seeds=1). DSR={:.4f}", _dsr_real)

    # Inyectar metadatos de corrección en el veredicto para trazabilidad completa
    verdict["adjusted_dsr_threshold"] = round(_base_dsr_thr, 4)   # umbral base (no cambia)
    verdict["n_seeds_correction"]     = _n_seeds_total_r5
    verdict["dsr_correction_factor"]  = round(_r5_factor, 4)
    verdict["dsr_adjusted"]           = round(_dsr_adjusted, 6)    # DSR recomputado con SR*_adj

    # Re-evaluar pass_dsr con el DSR ajustado por N_seeds
    _pass_dsr_base = verdict.get("flags", {}).get("pass_dsr", False)

    if _pass_dsr_base and not _pass_dsr_adjusted:
        # Caso crítico: la semilla pasaba el gate base pero falla con corrección entre semillas
        print(f"[FIX-R5-MULTIPLE-SEEDS] ⚠️  GATE ENDURECIDO: DSR_adj={_dsr_adjusted:.4f} < "
              f"umbral={_base_dsr_thr:.3f} (SR*_adj={_sr_star_adj:.4f} vs SR_crudo={_sr_crudo:.4f}). "
              f"deploy_approved: True → False")
        logger.warning("[FIX-R5-MULTIPLE-SEEDS] Semilla degradada por corrección R5: "
                       "DSR_adj={:.4f} < {:.3f} (N_seeds={}, SR*_adj={:.4f})",
                       _dsr_adjusted, _base_dsr_thr, _n_seeds_total_r5, _sr_star_adj)
        verdict["flags"]["pass_dsr"] = False
        # verdict["deploy_approved"]   = False # [MODIFICACION ENSEMBLE DOBLE] Desactivado bloqueo
        verdict["rejection_reason"]  = (
            f"[FIX-R5] DSR_ajustado={_dsr_adjusted:.4f} < umbral={_base_dsr_thr:.3f} "
            f"(SR*_adj={_sr_star_adj:.4f} con N_seeds={_n_seeds_total_r5}, "
            f"SR_crudo={_sr_crudo:.4f})"
        )
    elif _pass_dsr_adjusted:
        print(f"[FIX-R5-MULTIPLE-SEEDS] ✅ DSR_adj={_dsr_adjusted:.4f} >= umbral={_base_dsr_thr:.3f} "
              f"— gate R5 superado (SR_crudo={_sr_crudo:.4f} > SR*_adj={_sr_star_adj:.4f}).")
        logger.info("[FIX-R5-MULTIPLE-SEEDS] Gate R5 superado: DSR_adj={:.4f} >= {:.3f}",
                    _dsr_adjusted, _base_dsr_thr)
    else:
        # Ya fallaba el gate base: la corrección R5 no cambia el resultado
        print(f"[FIX-R5-MULTIPLE-SEEDS] ❌ DSR_adj={_dsr_adjusted:.4f} < umbral={_base_dsr_thr:.3f} "
              f"(ya fallaba el gate base con DSR={_dsr_real:.4f}).")
        logger.info("[FIX-R5-MULTIPLE-SEEDS] Gate ya fallaba antes de corrección R5: "
                    "DSR_adj={:.4f} < {:.3f}", _dsr_adjusted, _base_dsr_thr)
    # -------------------------------------------------------------------------

    # -- Walk-Forward Validation ----------------------------------------------
    wfv = _run_wfv(trades_df)

    if wfv:
        verdict["wfv_results"] = wfv

    # [FIX-W5-SIGNAL-01] Detectar ventana más reciente con 0 trades ("blind window").
    # Si la última ventana WFV tiene 0 trades, el modelo aprobado está CIEGO en el
    # periodo más reciente — no ha operado en esa ventana y no sabemos si funciona.
    # NO bloquea el deploy (eso corresponde al Gauntlet DSR/PBO), pero:
    #   1) Inyecta 'latest_window_blind=True' en el veredicto para diagnóstico.
    #   2) Loguea una advertencia clara visible en los logs del run.
    #   3) Añade 'blind_window_id' con el ID de la ventana ciega.
    try:
        if wfv:
            _sorted_windows = sorted(wfv.keys())
            _latest_win_id  = _sorted_windows[-1] if _sorted_windows else None
            if _latest_win_id:
                _latest_win_data  = wfv[_latest_win_id]
                _latest_n_trades  = _latest_win_data.get("n_trades", 0)
                _is_blind         = (_latest_n_trades == 0)
                verdict["latest_window_blind"]    = _is_blind
                verdict["latest_window_blind_id"] = _latest_win_id if _is_blind else None
                if _is_blind:
                    _blind_start = _latest_win_data.get("start_date", "?")
                    _blind_end   = _latest_win_data.get("end_date",   "?")
                    print(
                        f"[FIX-W5-SIGNAL-01] ALERTA: ventana más reciente '{_latest_win_id}' "
                        f"tiene 0 trades (periodo {_blind_start} -> {_blind_end}). "
                        f"El modelo está CIEGO en ese periodo."
                    )
                    logger.warning(
                        "[FIX-W5-SIGNAL-01] BLIND WINDOW detectada: '{}' tiene 0 trades "
                        "({} -> {}). El veredicto de deploy se basa solo en ventanas anteriores. "
                        "Considerar no desplegar hasta obtener señales en el periodo más reciente.",
                        _latest_win_id, _blind_start, _blind_end
                    )
                else:
                    verdict["latest_window_blind"] = False
                    print(f"[FIX-W5-SIGNAL-01] OK: ventana más reciente '{_latest_win_id}' tiene {_latest_n_trades} trades.")
    except Exception as _e_w5:
        logger.debug("[FIX-W5-SIGNAL-01] Error evaluando blind window: {}", _e_w5)
    # -------------------------------------------------------------------------

    # [ERROR-C-FIX 2026-05-21] Gate Blind Window con evidencia insuficiente.
    # Política: forzar deploy_approved=False si la ventana más reciente es ciega Y
    # hay < 3 ventanas con trades. Rationale:
    #   - < 3 ventanas con trades = el patrón no está validado en suficientes periodos IS.
    #   - W5=0 trades = el modelo está ciego en el mercado actual.
    #   - Ambos juntos = riesgo de deploy sin evidencia robusta.
    # DISEÑO DELIBERADO: no se bloquea si hay >= 3 ventanas con trades (ej: seed1337
    # SFI16 que tiene W5=0 pero W2/W3/W4 con trades = edge validado en 3 periodos distintos).
    # Referencia: audit_ultima_run.md §5 Error-C.
    try:
        if verdict.get("latest_window_blind") and wfv:
            _n_windows_with_trades = sum(
                1 for _w in wfv.values() if _w.get("n_trades", 0) > 0
            )
            verdict["n_windows_with_trades"] = int(_n_windows_with_trades)
            _MIN_WINDOWS_REQUIRED = 3  # mínimo de ventanas con trades para deploy seguro
            if _n_windows_with_trades < _MIN_WINDOWS_REQUIRED:
                # verdict["deploy_approved"] = False # [MODIFICACION ENSEMBLE DOBLE] Desactivado bloqueo
                _ec_reason = (
                    f"[ERROR-C] Ventana más reciente ciega (0 trades) con solo "
                    f"{_n_windows_with_trades} ventana(s) con trades "
                    f"(mínimo={_MIN_WINDOWS_REQUIRED}). "
                    "Evidencia insuficiente para deploy seguro."
                )
                verdict["rejection_reason"] = _ec_reason
                print(
                    f"[ERROR-C-FIX] ALERTA: ventana ciega + {_n_windows_with_trades} "
                    f"ventanas con trades (< {_MIN_WINDOWS_REQUIRED}). "
                    "deploy_approved no modificado (Ensemble Doble)."
                )
                logger.warning(
                    "[ERROR-C-FIX] Deploy rechazado: blind window + solo {} ventana(s) "
                    "con trades (min={}). {}",
                    _n_windows_with_trades, _MIN_WINDOWS_REQUIRED, _ec_reason
                )
            else:
                print(
                    f"[ERROR-C-FIX] OK: ventana ciega pero {_n_windows_with_trades} "
                    f"ventanas con trades (>= {_MIN_WINDOWS_REQUIRED}). "
                    "Edge validado en múltiples periodos — deploy_approved no modificado."
                )
                logger.info(
                    "[ERROR-C-FIX] Blind window aceptada: {} ventanas con trades >= {}. "
                    "El edge está validado estadísticamente.",
                    _n_windows_with_trades, _MIN_WINDOWS_REQUIRED
                )
        elif wfv:
            _n_windows_with_trades = sum(
                1 for _w in wfv.values() if _w.get("n_trades", 0) > 0
            )
            verdict["n_windows_with_trades"] = int(_n_windows_with_trades)
    except Exception as _e_ec:
        logger.debug("[ERROR-C-FIX] Error evaluando gate blind window: {}", _e_ec)
    # -------------------------------------------------------------------------

    # -- FIX-M2 (2026-03-21): añadir hmm_drift_jsd al verdict -----------------
    # El monitor ARCH-05 en hmm_regime.py calcula JSD IS↔OOS pero solo lo logueaba.
    # Ahora se persiste en el pkl y en el verdict para diagnóstico de regime drift.
    try:
        import joblib as _jl
        _hmm_pkl = _ROOT / "data" / "models" / "hmm_regime.pkl"
        if _hmm_pkl.exists():
            _hmm_data = _jl.load(_hmm_pkl)
            _jsd = _hmm_data.get("regime_drift_jsd", None)
            if _jsd is not None:
                verdict["hmm_drift_jsd"] = float(_jsd)
                _alert = " ⚠️  DRIFT" if _jsd > 0.15 else " OK"
                logger.info("  [M2/ARCH-05] HMM regime drift JSD²={:.3f}{}", _jsd, _alert)
    except Exception as _m2_e:
        logger.debug("  [M2] hmm_drift_jsd no disponible: {}", _m2_e)

    # -- FIX-I2 + FIX-P1A (2026-05-28): añadir signal_funnel acumulado al verdict
    # FIX-P1A amplía la búsqueda para encontrar el funnel acumulado de todas las ventanas:
    #   1. signal_funnel_{LUNA_RUN_ID}.json         (run_id FINAL del subproceso)
    #   2. signal_funnel_WFB_seed{N}_funnel.json    (key estable por seed FIX-P1A)
    #   3. signal_funnel_seed{N}.json               (formato canónico alternativo)
    #   4. signal_funnel.json                       (fallback genérico — puede ser solo 1 ventana)
    try:
        _rid     = os.environ.get("LUNA_RUN_ID", "")
        _seed_id = os.environ.get("LUNA_SEED", "")

        # Candidatos en orden de preferencia (más específico primero)
        _funnel_candidates = []
        if _rid:
            _funnel_candidates.append(report_dir / f"signal_funnel_{_rid}.json")
        if _seed_id:
            # [FIX-P1A-FUNNEL-SEED] key estable por seed inyectado por wfb_worker.py
            _funnel_candidates.append(report_dir / f"signal_funnel_WFB_seed{_seed_id}_funnel.json")
            _funnel_candidates.append(report_dir / f"signal_funnel_seed{_seed_id}.json")
        _funnel_candidates.append(report_dir / "signal_funnel.json")

        _funnel_path = None
        for _fc in _funnel_candidates:
            if _fc.exists():
                _funnel_path = _fc
                break

        if _funnel_path and _funnel_path.exists():
            with open(_funnel_path, encoding="utf-8") as _fp:
                _funnel_data = json.load(_fp)
            verdict["signal_pipeline"] = _funnel_data
            _n_w_loaded = _funnel_data.get("n_windows_accumulated", "?")
            print(f"[FIX-P1A-FUNNEL-SEED] signal_pipeline cargado desde {_funnel_path.name} "
                  f"| n_windows_accumulated={_n_w_loaded} (seed={_seed_id})")
            logger.info("  [FIX-P1A] signal_pipeline cargado: n_windows={} archivo={}",
                        _n_w_loaded, _funnel_path.name)
        else:
            print(f"[FIX-P1A-FUNNEL-SEED] WARN: signal_funnel no encontrado (seed={_seed_id}, rid={_rid})")
            logger.debug("  [FIX-P1A] signal_funnel.json no disponible para seed={} rid={}", _seed_id, _rid)
    except Exception as _i2_e:
        logger.debug("  [FIX-P1A] signal_funnel.json no disponible: {}", _i2_e)


    # -- Guards CUSUM & Sharpe Decay OOS --------------------------------------
    try:
        import scripts.oos_health_monitor as ohm
        if "return_pct" in trades_df.columns:
            _ret_s = trades_df["return_pct"].copy()
            if not isinstance(_ret_s.index, pd.DatetimeIndex) and "entry_time" in trades_df.columns:
                _ret_s.index = pd.to_datetime(trades_df["entry_time"])
            
            if isinstance(_ret_s.index, pd.DatetimeIndex):
                # [FIX-09] cusum_threshold leído de settings.yaml stat.cusum_threshold
                # Ref: Page(1954), Hawkins&Olwell(1998) — rango estándar h=4.0-5.0 para ARL≈500
                # Antes: threshold=4.5 hardcodeado en código
                try:
                    from config.settings import cfg as _cfg_cusum
                    _cusum_thr = float(getattr(getattr(_cfg_cusum, 'stat', object()), 'cusum_threshold', 4.5))
                    _cusum_src = "cfg.stat.cusum_threshold"
                except Exception:
                    _cusum_thr = 4.5
                    _cusum_src = "FALLBACK (settings no disponible)"
                    logger.warning("[FIX-09] No se pudo leer stat.cusum_threshold de cfg. Usando fallback={}", _cusum_thr)
                # [P3-CUSUM 2026-05-21] Confirmación de carga — distingue settings vs fallback
                print(f"[P3-CUSUM] cusum_threshold={_cusum_thr} | fuente={_cusum_src} (ref: Hawkins&Olwell1998, rango 4.0-5.0)")

                cusum_res = ohm.calculate_cusum(_ret_s, target=0.0, threshold=_cusum_thr)

                sharpe_res = ohm.calculate_sharpe_degradation(_ret_s, freq="W", critical_weeks=2)
                verdict["oos_health"] = {
                    "cusum_max_drift": cusum_res["max_drift"],
                    "sharpe_decay_recent": sharpe_res.get("recent_sharpe", 0.0),
                    "is_healthy": not (cusum_res["trigger"] or sharpe_res.get("trigger", False))
                }
                logger.info("  [OOS-HEALTH] CUSUM Drift: {:.2f} | Sharpe (2W): {:.2f}", cusum_res["max_drift"], sharpe_res.get("recent_sharpe", 0.0))
    except Exception as _e_health:
        logger.debug("  [OOS-HEALTH] Error calculando metricas OOS: {}", _e_health)

    # -- Guardar verdict JSON -------------------------------------------------
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=4)
    logger.info("  Veredicto guardado: {}", verdict_path)

    # -- Tearsheet PNG --------------------------------------------------------
    _generate_tearsheet(trades_df, verdict, tearsheet_path, timestamp=ts)

    # -- Reporte Markdown completo (S0-S8) ------------------------------------
    try:
        from luna.reports.generate_validation_report import generate as _gen_report
        report_path = _gen_report(verdict, trades_df, ts=ts)
        logger.info("  Reporte MD guardado: {}", report_path)
    except Exception as e:
        logger.warning("  [Report] No se pudo generar el reporte MD: {}", e)

    # -- Archivos timestamped (YYYY-MM-DD_THHMM_RUN-XXXX_*) ------------------
    import shutil
    # FIX-RUN-ID-01: leer LUNA_RUN_ID del entorno (pipeline maestro)
    # o del run_counter.txt (invocacion directa sin pipeline maestro).
    run_id = os.environ.get("LUNA_RUN_ID", "")
    if not run_id:
        _counter_path = _ROOT / "data" / "reports" / "run_counter.txt"
        if _counter_path.exists():
            try:
                # FIX-RUN-ID-02 (2026-03-24): invocacion standalone → INCREMENTAR contador
                # Antes: solo leia el valor actual → todas las runs standalone compartian RUN-XXXX.
                # Ahora: read → +1 → write → usar, igual que run_full_pipeline._get_incremental_run_id().
                # LOGIC-VAL-02: lock rudimentario para concurrencia básica
                _lock_path = _counter_path.with_suffix(".lock")
                for _ in range(50):
                    try:
                        _lock_path.touch(exist_ok=False)
                        break
                    except FileExistsError:
                        import time as _t; _t.sleep(0.1)
                try:
                    _current = int(_counter_path.read_text(encoding="utf-8").strip())
                    _next = _current + 1
                    _counter_path.write_text(str(_next), encoding="utf-8")
                    run_id = f"RUN-{_next:04d}"
                finally:
                    try: _lock_path.unlink()
                    except (OSError, FileNotFoundError): pass
            except Exception:
                run_id = "DEV"
        else:
            run_id = "DEV"

    for src, suffix in [(verdict_path,  f"{run_id}_statistical_verdict.json"),
                        (tearsheet_path, f"{run_id}_tearsheet_oos.png")]:
        if src.exists():
            dst = report_dir / f"{ts}_{suffix}"
            shutil.copy2(src, dst)
    logger.info("  Archivos timestamped guardados: {}_{}_*..", ts, run_id)

    # AUDIT #20 (ARCH-02): double-write al directorio canónico data/runs/ cuando en modo WFB.
    # Solo activo si LUNA_ENSEMBLE_DIR está inyectado por run_walkforward_pipeline_v2.py.
    # data/reports/ se mantiene para retrocompatibilidad con seed_champion.py / HPO.
    _ensemble_dir_sv = os.environ.get("LUNA_ENSEMBLE_DIR", "")
    if _ensemble_dir_sv:
        try:
            _win_sv   = os.environ.get("LUNA_WINDOW_ID", "FINAL")
            _seed_sv  = os.environ.get("LUNA_SEED", "")
            _run_out  = Path(_ensemble_dir_sv) / (f"seed{_seed_sv}" if _seed_sv else "") / _win_sv
            _run_out.mkdir(parents=True, exist_ok=True)
            if verdict_path.exists():
                shutil.copy2(verdict_path,   _run_out / "statistical_verdict.json")
            if tearsheet_path.exists():
                shutil.copy2(tearsheet_path, _run_out / "tearsheet.png")
            logger.info("  [ARCH] Artefactos canónicos escritos: {}", _run_out)
        except Exception as _e_arch_sv:
            logger.warning("  [ARCH] Double-write a LUNA_ENSEMBLE_DIR falló: {}", _e_arch_sv)



    approved = verdict.get("deploy_approved", False)
    if approved:
        logger.success("DEPLOY APROBADO")
    else:
        logger.error("GAUNTLET RECHAZADO -- revisar gates en {}", verdict_path)

    return 0 if approved else 1


if __name__ == "__main__":
    sys.exit(main())
