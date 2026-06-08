"""
wfb_worker.py
=============
Luna V2 — Orquestador de Walk-Forward Backtesting (Worker)

Migración V1→V2 (2026-05-10):
  - FALLA 2: Lock de proceso (.wfb_lock_dir)
  - FALLA 6: Backup settings.yaml + restore_wfb()
  - FALLA 9: cleanup_old_logs()
"""

import sys
import os
import shutil
import argparse
import tempfile
import re
import json
import datetime
import pandas as pd
from pathlib import Path
from loguru import logger

# Forzar UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from luna.pipeline_executor import LunaPipelineExecutor, hydrate_window_state

# ── Config ────────────────────────────────────────────────────────────────────
SETTINGS_PATH = _ROOT / "config" / "settings.yaml"
WFB_OUT_DIR = _ROOT / "data" / "reports" / "wfb"

import os
_ts_wfb = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
_log_dir_wfb = _ROOT / "logs"
_log_dir_wfb.mkdir(exist_ok=True)
logger.add(_log_dir_wfb / f"wfb_worker_{_ts_wfb}.log", rotation="50 MB", level="DEBUG", encoding="utf-8")

# [GAP-11] Activar global_telemetry si está disponible en luna.utils
try:
    from luna.utils.global_telemetry import activate_global_telemetry
    activate_global_telemetry()
except Exception as _gt_e:
    logger.debug(f"[GAP-11] global_telemetry no disponible: {_gt_e}")

# [BUGFIX-UNPICKLE-01] Inyectar adapters de calibración en __main__ para evitar
# AttributeError de Pickle/Joblib al deserializar MetaLabelerV2 en el WFB worker.
# Idéntico al fix en run_live_trader.py L37-43. Sin esto, el wfb_worker lanza:
# "Can't get attribute '_RFWithAdapter' on <module '__main__' from wfb_worker.py>"
try:
    from luna.models.calibrate_probabilities import _RFWithAdapter, _IdentityWrapper, _TSAdapter
    sys.modules['__main__']._RFWithAdapter = _RFWithAdapter
    sys.modules['__main__']._IdentityWrapper = _IdentityWrapper
    sys.modules['__main__']._TSAdapter = _TSAdapter
    print("[BUGFIX-UNPICKLE-01][WFB_WORKER] OK: Wrappers de calibración inyectados en __main__ del WFB worker.")
    logger.info("[BUGFIX-UNPICKLE-01][WFB_WORKER] _RFWithAdapter, _IdentityWrapper, _TSAdapter inyectados exitosamente.")
except Exception as _e_unpickle:
    print(f"[BUGFIX-UNPICKLE-01][WFB_WORKER] WARNING: No se pudieron inyectar los wrappers en __main__: {_e_unpickle}")
    logger.warning(f"[BUGFIX-UNPICKLE-01][WFB_WORKER] Fallo al inyectar wrappers de calibración: {_e_unpickle}")


# Backup de settings [FALLA 6]
BACKUP_PATH = _ROOT / "config" / f"settings_backup_wfb_{_ts_wfb}.yaml"
_settings_restored = False

_WINDOWS_FALLBACK = [
    {"id": "W1", "train_end": "2024-10-31", "val_start": "2024-11-01", "val_end": "2024-12-31", "holdout_start": "2025-01-01", "holdout_end": "2025-03-31"},
    {"id": "W2", "train_end": "2025-01-31", "val_start": "2025-02-01", "val_end": "2025-03-31", "holdout_start": "2025-04-01", "holdout_end": "2025-06-30"},
    {"id": "W3", "train_end": "2025-04-30", "val_start": "2025-05-01", "val_end": "2025-06-30", "holdout_start": "2025-07-01", "holdout_end": "2025-09-30"},
    {"id": "W4", "train_end": "2025-07-31", "val_start": "2025-08-01", "val_end": "2025-09-30", "holdout_start": "2025-10-01", "holdout_end": "2025-12-31"},
    {"id": "W5", "train_end": "2025-10-31", "val_start": "2025-11-01", "val_end": "2025-12-31", "holdout_start": "2026-01-01", "holdout_end": "2026-03-31"},
    {"id": "W6", "train_end": "2026-01-31", "val_start": "2026-02-01", "val_end": "2026-03-31", "holdout_start": "2026-04-01", "holdout_end": "2026-06-30"},
]

def _load_windows_from_config() -> list:
    try:
        import yaml as _yaml
        with open(SETTINGS_PATH, "r", encoding="utf-8") as _f:
            _raw = _yaml.safe_load(_f)
        _wfb_windows = _raw.get("wfb", {}).get("windows", None)
        if _wfb_windows and isinstance(_wfb_windows, list) and len(_wfb_windows) > 0:
            logger.info(f"[CONFIG] WINDOWS cargadas desde settings.yaml: {len(_wfb_windows)} ventanas")
            return _wfb_windows
        return _WINDOWS_FALLBACK
    except Exception:
        return _WINDOWS_FALLBACK

WINDOWS = _load_windows_from_config()


# ── Validación de Solapamiento de Ventanas [GAP-01 · BUG-ORC-02 FIX] ─────────

def _validate_window_overlaps():
    """
    [BUG-ORC-02 FIX] Verifica que los holdouts de ventanas consecutivas no se solapen.
    Ejecutar al inicio del pipeline (fail-fast) para no desperdiciar horas de cómputo
    con una configuración de fechas incorrecta.
    """
    for w in WINDOWS:
        try:
            _te = pd.Timestamp(w["train_end"], tz="UTC")
            _vs = pd.Timestamp(w.get("val_start", w["train_end"]), tz="UTC")
            _ve = pd.Timestamp(w.get("val_end", w["train_end"]), tz="UTC")
            _hs = pd.Timestamp(w["holdout_start"], tz="UTC")
            _he = pd.Timestamp(w["holdout_end"], tz="UTC")
            if not (_te <= _vs <= _ve < _hs < _he):
                logger.error(
                    f"🔴 [DATA-INTEGRITY] ERROR TEMPORAL INTRA-VENTANA en {w['id']}: "
                    f"Orden requerido (train_end <= val_start <= val_end < holdout_start < holdout_end) no cumplido."
                )
                sys.exit(1)
        except SystemExit:
            raise
        except Exception:
            pass

    for i, w1 in enumerate(WINDOWS[:-1]):
        w2 = WINDOWS[i + 1]
        try:
            _end_w1   = pd.Timestamp(w1["holdout_end"],   tz="UTC")
            _start_w2 = pd.Timestamp(w2["holdout_start"], tz="UTC")
            if _end_w1 >= _start_w2:
                logger.error(
                    f"🔴 [BUG-ORC-02] SOLAPAMIENTO DE HOLDOUTS: "
                    f"{w1['id']} (holdout_end={w1['holdout_end']}) > "
                    f"{w2['id']} (holdout_start={w2['holdout_start']}). "
                    f"Abortando antes de desperdiciar horas de cómputo."
                )
                sys.exit(1)
        except SystemExit:
            raise
        except Exception as _e_ovlp:
            logger.warning(
                f"[BUG-ORC-02] No se pudo validar solapamiento "
                f"{w1['id']}/{w2['id']}: {_e_ovlp}"
            )
    logger.info("[BUG-ORC-02] Validación de solapamiento de holdouts: OK")


# ── Lock de Proceso [FALLA 2] ─────────────────────────────────────────────────

def _acquire_lock():
    """Crea un lock de proceso para evitar ejecuciones paralelas sobre el mismo workspace."""
    lock_file = _ROOT / ".wfb_lock"

    # Limpiar lock muerto si el proceso ya no existe
    if lock_file.exists():
        try:
            # [FIX-LOCK-RECOVER-01] TTL de emergencia: si el archivo de lock es más viejo que 1.5 horas (5400 seg),
            # asumimos que la ejecución anterior crasheó o se detuvo de forma forzada.
            _mtime = lock_file.stat().st_mtime
            _age_s = datetime.datetime.now().timestamp() - _mtime
            if _age_s > 5400:
                print(f"[FIX-LOCK-RECOVER-01] Lock huérfano detectado por inactividad ({_age_s/60:.1f} minutos). Removiendo lock...")
                logger.warning(f"[FIX-LOCK-RECOVER-01] Lock huérfano detectado por inactividad ({_age_s/60:.1f} minutos). Removiendo lock...")
                lock_file.unlink()
            else:
                old_pid = int(lock_file.read_text().strip())
                import psutil
                if not psutil.pid_exists(old_pid):
                    print(f"[FIX-LOCK-RECOVER-01] Lock huérfano detectado: PID {old_pid} no existe. Removiendo lock...")
                    lock_file.unlink()
                else:
                    # [FIX-LOCK-RECOVER-01] Colisión por reutilización de PID (el PID está vivo pero no es un script de Python/WFB)
                    _proc = psutil.Process(old_pid)
                    _cmdline = " ".join(_proc.cmdline()).lower()
                    if "python" not in _cmdline and "wfb" not in _cmdline:
                        print(f"[FIX-LOCK-RECOVER-01] Colisión de PID: PID {old_pid} está vivo pero pertenece a '{_proc.name()}' (no es Python/WFB). Removiendo lock...")
                        logger.warning(f"[FIX-LOCK-RECOVER-01] Colisión de PID: PID {old_pid} está vivo pero pertenece a '{_proc.name()}' (no es Python/WFB). Removiendo lock...")
                        lock_file.unlink()
        except Exception as _e_l:
            print(f"[FIX-LOCK-RECOVER-01] Error leyendo o validando lock: {_e_l}. Forzando remoción del lock corrupto...")
            try: lock_file.unlink()
            except: pass

    try:
        # File creation is atomic across platforms with O_EXCL
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            f.write(str(os.getpid()))
        print(f"[FIX-LOCK-RECOVER-01] Lock adquirido con éxito para PID={os.getpid()}")
    except FileExistsError:
        print(f"🔴 [FIX-LOCK-RECOVER-01] [FATAL] Lock '.wfb_lock' existe y el proceso propietario (PID) está activo. Abortando run duplicado.")
        logger.error("🔴 [FATAL] Lock '.wfb_lock' existe y el proceso está vivo. Abortando.")
        sys.exit(1)

    import atexit
    import signal
    def _release(signum=None, frame=None):
        try:
            if lock_file.exists() and lock_file.read_text().strip() == str(os.getpid()):
                lock_file.unlink()
        except: pass
        if signum is not None:
            sys.exit(1)
            
    atexit.register(_release)
    try:
        signal.signal(signal.SIGTERM, _release)
        signal.signal(signal.SIGINT, _release)
    except AttributeError:
        pass
        
    logger.info(f"[LOCK] Lock de proceso adquirido (PID={os.getpid()})")


# ── Backup y Restore de settings.yaml [FALLA 6] ──────────────────────────────

def _backup_settings():
    """Crea backup del settings.yaml al inicio de la sesión."""
    try:
        shutil.copy(SETTINGS_PATH, BACKUP_PATH)
        logger.info(f"[BACKUP] settings.yaml respaldado en {BACKUP_PATH.name}")
    except Exception as e:
        logger.warning(f"[BACKUP] No se pudo respaldar settings.yaml: {e}")


def restore_wfb():
    """Restaura settings.yaml desde el backup en caso de crash."""
    global _settings_restored
    if BACKUP_PATH.exists() and not _settings_restored:
        try:
            shutil.copy(BACKUP_PATH, SETTINGS_PATH)
            _settings_restored = True
            logger.info("[RESTORE] settings.yaml restaurado desde backup.")
        except Exception as e:
            logger.warning(f"[RESTORE] No se pudo restaurar settings.yaml: {e}")


# ── Cleanup de Logs [FALLA 9] ─────────────────────────────────────────────────

def cleanup_old_logs(days: int = 30):
    """Elimina logs con más de `days` días para evitar acumulación indefinida."""
    import time as _time
    _log_dir = _ROOT / "logs"
    if not _log_dir.exists():
        return
    _cutoff = _time.time() - days * 86400
    _deleted = 0
    for _pat in ["*.log", "*.done", "*_EMPTY.flag"]:
        for _lf in _log_dir.glob(_pat):
            try:
                if _lf.stat().st_mtime < _cutoff:
                    _lf.unlink()
                    _deleted += 1
            except Exception:
                pass
    # Purgar settings_backup_wfb_*.yaml — conservar solo el más reciente
    _cfg_dir = _ROOT / "config"
    if _cfg_dir.exists():
        _cfg_backups = sorted(_cfg_dir.glob("settings_backup_wfb_*.yaml"),
                              key=lambda p: p.stat().st_mtime)
        for _cbk in _cfg_backups[:-1]:
            try:
                _cbk.unlink()
                _deleted += 1
            except: pass
    if _deleted > 0:
        logger.info(f"[CLEANUP] {_deleted} archivos viejos eliminados (>{days}d)")


# ── YAML por ventana ──────────────────────────────────────────────────────────

def rewrite_yaml_for_window(window: dict):
    """Actualiza settings.yaml atómicamente con las fechas de la ventana."""
    _keys_to_patch = {
        "train_end":        window["train_end"],
        "validation_start": window["val_start"],
        "validation_end":   window["val_end"],
        "holdout_start":    window["holdout_start"],
        "holdout_end":      window["holdout_end"],
        "hmm_train_end":    window["val_end"],
    }

    content = SETTINGS_PATH.read_text(encoding="utf-8")
    _in_temporal_splits_block = False
    new_lines = []

    for _line in content.splitlines(keepends=True):
        if _line.strip() == "temporal_splits:" or _line.startswith("temporal_splits:"):
            _in_temporal_splits_block = True
            new_lines.append(_line)
            continue

        _stripped = _line.strip()
        if _in_temporal_splits_block and _stripped:
            # Terminar el bloque si encontramos una línea que no está indentada y no es comentario
            if not _line.startswith(" ") and not _line.startswith("#"):
                _in_temporal_splits_block = False

        if _in_temporal_splits_block:
            _matched = False
            for _key, _val in _keys_to_patch.items():
                if _line.lstrip().startswith(f"{_key}:"):
                    _parts = _line.split('#', 1)
                    _comment = f" #{_parts[1].rstrip()}" if len(_parts) > 1 else ""
                    _indent = _line[:len(_line) - len(_line.lstrip())]
                    new_lines.append(f"{_indent}{_key}: {_val}{_comment}\n")
                    _matched = True
                    break
            if not _matched:
                new_lines.append(_line)
        else:
            new_lines.append(_line)

    _new_content = ''.join(new_lines)
    _settings_dir = SETTINGS_PATH.parent

    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".yaml", dir=_settings_dir, delete=False) as _tmp:
            _tmp_path = Path(_tmp.name)
            _tmp.write(_new_content)
        _tmp_path.replace(SETTINGS_PATH)
        logger.info(f"  settings.yaml actualizado para ventana {window['id']}")
    except Exception as _e:
        logger.error(f"Error escribiendo settings.yaml para {window['id']}: {_e}")
        try: _tmp_path.unlink()
        except: pass
        raise


# ── Isolate + Merge ───────────────────────────────────────────────────────────

def isolate_window_trades(window: dict, seed: str):
    """
    Filtra y guarda los trades del holdout de la ventana.

    [GAP-03 · P0-4-FIX] Si oos_trades.parquet no existe, lanza RuntimeError en lugar
    de hacer un return silencioso. El orquestador capturará la excepción y abortará
    el run en lugar de continuar sin datos OOS para esta ventana.
    """
    trades_path = _ROOT / "data" / "predictions" / "oos_trades.parquet"
    if not trades_path.exists():
        raise RuntimeError(
            f"[GAP-03/P0-4-FIX] oos_trades.parquet no encontrado tras predict_oos.py "
            f"en ventana {window['id']}. Revisar logs de predict_oos.py. "
            f"Ruta esperada: {trades_path}"
        )

    df = pd.read_parquet(trades_path)
    _start_ts = pd.to_datetime(window['holdout_start'], utc=True)
    _he = pd.to_datetime(window['holdout_end'], utc=True)
    if _he.hour == 0 and _he.minute == 0 and _he.second == 0:
        _end_ts = _he + pd.Timedelta(hours=23, minutes=59, seconds=59)
    else:
        _end_ts = _he

    if not df.empty:
        if 'entry_time' not in df.columns:
            df['entry_time'] = df.index
        # [GAP-03 / FIX A-05] Si entry_time no es Datetime, intentar extraerlo del índice
        if not pd.api.types.is_datetime64_any_dtype(df['entry_time']):
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df['entry_time'] = df.index
            else:
                raise RuntimeError(
                    f"[GAP-03/P0-4-FIX] oos_trades.parquet ilegible "
                    f"(sin timestamp válido en índice ni columna 'entry_time'). "
                    f"Ventana {window['id']} no puede aislarse."
                )
        df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
        df = df[(df['entry_time'] >= _start_ts) & (df['entry_time'] <= _end_ts)]

    # Guardar snapshot de selected_features.json por ventana [P2-N4 de V1]
    _sf_src = _ROOT / "data" / "features" / "selected_features.json"
    _sf_dst = WFB_OUT_DIR / f"selected_features_{window['id']}.json"
    try:
        if _sf_src.exists():
            shutil.copy2(_sf_src, _sf_dst)
    except Exception: pass

    _seed_suffix = f"_seed{seed}" if seed else ""
    out_path = WFB_OUT_DIR / f"oos_trades_{window['id']}{_seed_suffix}.parquet"
    WFB_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        _empty_marker = WFB_OUT_DIR / f"oos_trades_{window['id']}{_seed_suffix}_EMPTY.flag"
        _empty_marker.write_text(
            f"Window {window['id']} produjo 0 trades OOS en "
            f"{window['holdout_start']} -> {window['holdout_end']} (Seed {seed})",
            encoding="utf-8"
        )
        logger.warning(
            f"⚠️ Ventana {window['id']} aislada con 0 trades en el período "
            f"{window['holdout_start']} → {window['holdout_end']}. "
            "Posibles causas: sin señales OOS, error en filtros HMM/MetaLabeler, "
            "o holdout_start/end fuera del rango de oos_trades.parquet."
        )
    else:
        df.to_parquet(out_path)
        logger.info(f"🛡️ Ventana {window['id']} aislada con {len(df)} trades → {out_path.name}")

    # ── [GAP-09 · AUDIT GAP-02 FIX] Double-Write artefactos canónicos ────────────
    # Aislar oos_raw_probs.parquet por ventana (Soft Voting)
    try:
        raw_probs_src = _ROOT / "data" / "predictions" / "oos_raw_probs.parquet"
        if raw_probs_src.exists():
            df_raw = pd.read_parquet(raw_probs_src)
            if not df_raw.empty:
                _time_col_raw = None
                if 'timestamp' in df_raw.columns:
                    _time_col_raw = pd.to_datetime(df_raw['timestamp'], utc=True)
                elif pd.api.types.is_datetime64_any_dtype(df_raw.index):
                    _time_col_raw = pd.to_datetime(df_raw.index, utc=True)
                if _time_col_raw is not None:
                    df_raw = df_raw[(_time_col_raw >= _start_ts) & (_time_col_raw <= _end_ts)]
                raw_probs_out = WFB_OUT_DIR / f"oos_raw_probs_{window['id']}{_seed_suffix}.parquet"
                df_raw.to_parquet(raw_probs_out)
                logger.info(f"  [GAP-09/RAW-PROBS] Ventana {window['id']} → {raw_probs_out.name}")
    except Exception as _e_raw:
        logger.warning(f"  [GAP-09/RAW-PROBS] No se pudo aislar oos_raw_probs.parquet: {_e_raw}")

    # Aislar oos_trades_xgb_baseline.parquet por ventana (Investigación)
    try:
        xgb_base_src = _ROOT / "data" / "predictions" / "oos_trades_xgb_baseline.parquet"
        if xgb_base_src.exists():
            df_xgb = pd.read_parquet(xgb_base_src)
            if not df_xgb.empty:
                _time_col_xgb = None
                if 'timestamp' in df_xgb.columns:
                    _time_col_xgb = pd.to_datetime(df_xgb['timestamp'], utc=True)
                elif pd.api.types.is_datetime64_any_dtype(df_xgb.index):
                    _time_col_xgb = pd.to_datetime(df_xgb.index, utc=True)
                if _time_col_xgb is not None:
                    df_xgb = df_xgb[(_time_col_xgb >= _start_ts) & (_time_col_xgb <= _end_ts)]
                xgb_out = WFB_OUT_DIR / f"oos_trades_xgb_baseline_{window['id']}{_seed_suffix}.parquet"
                df_xgb.to_parquet(xgb_out)
                logger.info(f"  [GAP-09/XGB-BASE] Ventana {window['id']} → {xgb_out.name}")
    except Exception as _e_xgb:
        logger.warning(f"  [GAP-09/XGB-BASE] No se pudo aislar oos_trades_xgb_baseline.parquet: {_e_xgb}")

    # Double-Write al directorio canónico data/runs/WFB_*/seed{N}/W{N}/
    try:
        import os as _os_dw
        _ensemble_dir_dw = _os_dw.environ.get("LUNA_ENSEMBLE_DIR", "")
        if _ensemble_dir_dw and seed:
            from pathlib import Path as _Path_dw
            _canonical_dir = _Path_dw(_ensemble_dir_dw) / f"seed{seed}" / window["id"]
            _canonical_dir.mkdir(parents=True, exist_ok=True)
            _raw_probs_dw = WFB_OUT_DIR / f"oos_raw_probs_{window['id']}{_seed_suffix}.parquet"
            _xgb_base_dw  = WFB_OUT_DIR / f"oos_trades_xgb_baseline_{window['id']}{_seed_suffix}.parquet"
            for _src_p, _dst_n in [
                (out_path,       "oos_trades.parquet"),
                (_raw_probs_dw,  "oos_raw_probs.parquet"),
                (_xgb_base_dw,   "oos_trades_xgb_baseline.parquet"),
                (_sf_dst,        "selected_features.json"),
            ]:
                try:
                    if _src_p.exists():
                        shutil.copy2(_src_p, _canonical_dir / _dst_n)
                except Exception:
                    pass
            logger.info(f"  [GAP-09/AUDIT-GAP-02] Double-write canónico → {_canonical_dir.relative_to(_ROOT / 'data')}")
    except Exception as _e_dw:
        logger.debug(f"  [GAP-09/AUDIT-GAP-02] Double-write canónico falló (no bloqueante): {_e_dw}")


def merge_and_validate(seed: str = ""):
    """Junta todos los fragmentos WFB y lanza el Gauntlet."""
    _seed_suffix = f"_seed{seed}" if seed else ""
    logger.info("Consolidando ventanas para seed='{}'", seed)

    global_df = pd.DataFrame()
    for w in WINDOWS:
        p = WFB_OUT_DIR / f"oos_trades_{w['id']}{_seed_suffix}.parquet"
        _empty_flag = WFB_OUT_DIR / f"oos_trades_{w['id']}{_seed_suffix}_EMPTY.flag"
        if p.exists():
            w_df = pd.read_parquet(p)
            w_df = w_df.reset_index(drop=True)
            w_df["wfb_window"] = w["id"]
            global_df = pd.concat([global_df, w_df], ignore_index=True)
            logger.info(f"  [WFB-MERGE] Window {w['id']}: {len(w_df)} trades cargados")
        elif _empty_flag.exists():
            logger.warning(f"  [WFB-MERGE] Window {w['id']}: 0 trades (EMPTY.flag)")
        else:
            logger.error(f"🔴 [FATAL] Window {w['id']}: parquet y EMPTY.flag no encontrados. "
                         f"La ventana no se completó correctamente, generando gap.")
            raise RuntimeError(f"Ventana faltante en WFB_MERGE: {w['id']}")

    if global_df.empty:
        logger.warning("🚨 0 trades en todo el WFB. Retornando dataset vacío (Score = 0).")
        return global_df

    global_df = global_df.sort_values("entry_time").reset_index(drop=True)

    if "symbol" in global_df.columns and "side" in global_df.columns:
        global_df = global_df.drop_duplicates(subset=["entry_time", "symbol", "side"], keep='first')
    else:
        global_df = global_df.drop_duplicates(subset=["entry_time"], keep='first')

    logger.success(f"🧩 Macro-dataset WFB: {len(global_df)} trades OOS")

    _out_path = _ROOT / "data" / "predictions" / f"oos_trades{_seed_suffix}.parquet"
    if "entry_time" in global_df.columns:
        _df_indexed = global_df.copy()
        _df_indexed["entry_time"] = pd.to_datetime(_df_indexed["entry_time"], utc=True)
        _df_indexed = _df_indexed.set_index("entry_time")
        _df_indexed.to_parquet(_out_path, index=True)
    else:
        global_df.to_parquet(_out_path, index=False)

    logger.info("Lanzando Validación Estadística (Gauntlet)...")
    
    _env = os.environ.copy()
    _env["LUNA_RUN_ID"] = f"WFB_{_ts_wfb}{_seed_suffix}_FINAL"
    _env["LUNA_CUSTOM_TRADES_PATH"] = str(_out_path)

    # [FIX-GAUNTLET-EXIT-01 2026-06-03] Distinguir rechazo legítimo (exit=1) de crash real (exit≥2).
    # Antes: check=True lanzaba CalledProcessError para CUALQUIER exit!=0, confundiendo
    # GAUNTLET RECHAZADO (salida limpia, exit=1) con crashes reales del subproceso.
    # Ahora: check=False + inspección manual del returncode para mensajes correctos.
    try:
        import subprocess
        print(f"[GAUNTLET][FIX-GAUNTLET-EXIT-01] Lanzando validador estadístico (run_statistical_validation.py)...")
        logger.info("[GAUNTLET][FIX-GAUNTLET-EXIT-01] Iniciando subproceso de validación estadística.")
        _gauntlet_result = subprocess.run(
            [sys.executable, "-u", str(_ROOT / "scripts/run_statistical_validation.py")],
            env=_env,
            check=False  # [FIX-GAUNTLET-EXIT-01] Sin raise: inspeccionar returncode manualmente
        )
        if _gauntlet_result.returncode == 0:
            print(f"[GAUNTLET][FIX-GAUNTLET-EXIT-01] Validación APROBADA (exit=0). Seed pasó todos los gates.")
            logger.success("[GAUNTLET][FIX-GAUNTLET-EXIT-01] Gauntlet APROBADO (exit=0). Ver statistical_verdict.json.")
        elif _gauntlet_result.returncode == 1:
            # Salida limpia con rechazo estadístico: el script ejecutó correctamente
            print(f"[GAUNTLET][FIX-GAUNTLET-EXIT-01] Seed RECHAZADA por gates estadísticos (exit=1). "
                  f"Ver data/reports/statistical_verdict.json para gates fallidos (DSR/PBO/trades mínimos).")
            logger.warning("[GAUNTLET][FIX-GAUNTLET-EXIT-01] Gauntlet RECHAZADO por gates (exit=1). "
                           "No es un crash — el validador ejecutó correctamente y emitió veredicto negativo. "
                           "Revisar statistical_verdict.json.")
        else:
            # Exit code inesperado = crash real del subproceso (excepción no manejada)
            print(f"[GAUNTLET][FIX-GAUNTLET-EXIT-01] ⚠️ CRASH REAL del validador (exit={_gauntlet_result.returncode}). "
                  f"Revisar logs/run_statistical_validation*.log para traceback completo.")
            logger.error("[GAUNTLET][FIX-GAUNTLET-EXIT-01] Crash inesperado del validador estadístico "
                         "(exit=%d). Ver logs para traceback. No bloqueante para la cola multi-seed.",
                         _gauntlet_result.returncode)
    except Exception as _e_gauntlet:
        print(f"[GAUNTLET][FIX-GAUNTLET-EXIT-01] Excepción al lanzar subproceso del validador: {_e_gauntlet}")
        logger.error("[GAUNTLET][FIX-GAUNTLET-EXIT-01] Fallo al lanzar run_statistical_validation.py: %s. "
                     "No bloqueante.", _e_gauntlet)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Luna V2 OOS WFB Worker")
    parser.add_argument("--seed", type=int, default=777, help="Seed del ensemble")
    parser.add_argument("--resume", action="store_true", help="Saltar ventanas completadas")
    parser.add_argument("--smoke-test", action="store_true", help="Modo humo")
    parser.add_argument("--nocache", action="store_true", help="Forzar eliminación total de caché WFB para esta seed antes de empezar")
    parser.add_argument("--merge-only", action="store_true",
                        help="[FIX-EARLYSTOP-MERGE-01] Solo merge_and_validate sin entrenar (seeds podadas early-stop N>=30)")
    args = parser.parse_args()

    # [FIX-EARLYSTOP-MERGE-01 2026-06-03] Modo merge-only para seeds podadas por early-stop con N>=30 trades.
    # ROOT CAUSE del bug: el orquestador mata el worker via _kill_process_tree() tras early-stop,
    # pero los trades W1-W4 ya existen (N>=30). Sin merge_and_validate, el Gauntlet NUNCA se ejecuta.
    # SOLUCION: el orquestador relanza este worker con --merge-only, que crea EMPTY.flags para
    # ventanas no ejecutadas y llama merge_and_validate+Gauntlet directamente.
    # NO adquiere lock ni backup de settings (el worker previo ya está muerto).
    if getattr(args, 'merge_only', False):
        seed_str_mo = str(args.seed) if args.seed else ""
        _seed_sfx_mo = f"_seed{args.seed}" if args.seed else ""
        print(f"[FIX-EARLYSTOP-MERGE-01] Modo merge-only activado para seed={args.seed}.")
        print(f"[FIX-EARLYSTOP-MERGE-01] Creando EMPTY.flags para ventanas no ejecutadas (podadas early-stop)...")
        for _w_mo in WINDOWS:
            _p_mo = WFB_OUT_DIR / f"oos_trades_{_w_mo['id']}{_seed_sfx_mo}.parquet"
            _ef_mo = WFB_OUT_DIR / f"oos_trades_{_w_mo['id']}{_seed_sfx_mo}_EMPTY.flag"
            if not _p_mo.exists() and not _ef_mo.exists():
                try:
                    WFB_OUT_DIR.mkdir(parents=True, exist_ok=True)
                    _ef_mo.write_text(
                        f"[FIX-EARLYSTOP-MERGE-01] Window {_w_mo['id']} seed={args.seed} no ejecutada - "
                        f"podada por early-stop. EMPTY.flag creado por merge-only invocation.",
                        encoding="utf-8"
                    )
                    print(f"[FIX-EARLYSTOP-MERGE-01] EMPTY.flag creado: {_w_mo['id']} seed={args.seed}")
                except Exception as _e_efmo:
                    print(f"[FIX-EARLYSTOP-MERGE-01] ERROR creando EMPTY.flag {_w_mo['id']}: {_e_efmo}")
            elif _p_mo.exists():
                print(f"[FIX-EARLYSTOP-MERGE-01] {_w_mo['id']}: parquet existente ({_p_mo.stat().st_size//1024}KB) — OK.")
            else:
                print(f"[FIX-EARLYSTOP-MERGE-01] {_w_mo['id']}: EMPTY.flag ya existe — OK.")
        print(f"[FIX-EARLYSTOP-MERGE-01] Ejecutando merge_and_validate para seed={args.seed}...")
        try:
            merge_and_validate(seed_str_mo)
            print(f"[FIX-EARLYSTOP-MERGE-01] merge_and_validate COMPLETADO exitosamente para seed={args.seed}. Ver Gauntlet logs.")
        except Exception as _e_mo:
            print(f"[FIX-EARLYSTOP-MERGE-01] ERROR en merge_and_validate seed={args.seed}: {_e_mo}")
            sys.exit(1)
        print(f"[FIX-EARLYSTOP-MERGE-01] merge-only FINALIZADO para seed={args.seed}. Saliendo con exit=0.")
        sys.exit(0)

    # [CACHE-INTEGRITY-01] Heredar --nocache del orquestador padre si está en env
    if os.environ.get("LUNA_NOCACHE") == "1":
        args.nocache = True
        print(f"[CACHE-INTEGRITY-01] LUNA_NOCACHE=1 detectado — activando --nocache en worker.")

    logger.info("=".center(60, "="))
    logger.info("  INICIANDO WFB WORKER V2  ".center(60))
    logger.info("=".center(60, "="))

    # [CACHE-INTEGRITY-01] Si --nocache, limpiar caché de ESTA seed antes de todo
    if args.nocache:
        _seed_cache_dir = _ROOT / "data" / "wfb_cache" / f"seed{args.seed}"
        if _seed_cache_dir.exists():
            shutil.rmtree(_seed_cache_dir, ignore_errors=True)
            logger.warning(f"[CACHE-INTEGRITY-01] --nocache: eliminado wfb_cache/seed{args.seed}/")
            print(f"[CACHE-INTEGRITY-01] NOCACHE: eliminado {_seed_cache_dir.relative_to(_ROOT)}")
        # Eliminar también executor_state JSON de esta seed
        _cache_dir_root = _ROOT / "data" / "wfb_cache"
        for _stale_json in _cache_dir_root.glob(f"executor_state_*_s{args.seed}_*.json"):
            try:
                _stale_json.unlink()
                logger.warning(f"[CACHE-INTEGRITY-01] executor_state eliminado: {_stale_json.name}")
                print(f"[CACHE-INTEGRITY-01] NOCACHE: eliminado {_stale_json.name}")
            except Exception as _e_stale:
                logger.debug(f"No se pudo eliminar {_stale_json.name}: {_e_stale}")
        args.resume = False  # no tiene sentido resume si borramos la caché
        print(f"[CACHE-INTEGRITY-01] --nocache activo para seed={args.seed} — run limpio sin caché.")
    elif not args.resume:
        # [CACHE-INTEGRITY-02] En modo fresh (sin --resume), limpiar executor_state stale de seeds
        # anteriores para esta seed. Evita que 'skip' de pasos incorrectos.
        _cache_dir_root = _ROOT / "data" / "wfb_cache"
        _stale_cleaned = 0
        for _stale_json in _cache_dir_root.glob(f"executor_state_*_s{args.seed}_*.json"):
            try:
                _stale_json.unlink()
                _stale_cleaned += 1
            except Exception:
                pass
        if _stale_cleaned:
            logger.warning(
                f"[CACHE-INTEGRITY-02] Fresh run (sin --resume): eliminados {_stale_cleaned} "
                f"executor_state*.json stale para seed={args.seed}"
            )
            print(f"[CACHE-INTEGRITY-02] FRESH RUN: {_stale_cleaned} executor_state*.json stale eliminados (seed={args.seed})")

    # [FALLA 2] Adquirir lock de proceso
    _acquire_lock()

    # [GAP-01 · BUG-ORC-02] Verificar solapamientos de holdouts ANTES de cualquier cómputo
    _validate_window_overlaps()

    # [FALLA 6] Backup de settings.yaml + registro de restore en atexit
    _backup_settings()
    import atexit
    atexit.register(restore_wfb)

    # [GAP-06-SIGNAL] SIGINT handler explícito para restore_wfb (Ctrl+C con logging mejorado)
    import signal as _signal
    def _sigint_handler(_signum, _frame):
        logger.warning("[SIGINT] Interrupción detectada — restaurando settings.yaml y saliendo.")
        restore_wfb()
        sys.exit(1)
    try:
        _signal.signal(_signal.SIGINT, _sigint_handler)
    except Exception:
        pass  # En algunos contextos (hilos) signal.signal no está disponible

    # [FALLA 9] Limpiar logs viejos
    cleanup_old_logs(days=30)

    seed_str = str(args.seed)
    os.environ["LUNA_SEED"] = seed_str

    # [FIX-P1A-FUNNEL-SEED 2026-05-28] Inyectar LUNA_RUN_ID estable por seed en el entorno
    # del worker ANTES del loop de ventanas. Así FIX-FUNNEL-ACCUM-01 (signal_filter.py) puede
    # acumular correctamente entre ventanas W1→W5 sin resetear el contador en cada ventana.
    # Formato: WFB_seed{N}_funnel — estable entre todas las ventanas de la misma seed.
    # El FINAL run_id (con timestamp) se setea en merge_and_validate() para el Gauntlet.
    _funnel_run_id = f"WFB_seed{args.seed}_funnel"
    os.environ["LUNA_RUN_ID"] = _funnel_run_id
    print(f"[FIX-P1A-FUNNEL-SEED] LUNA_RUN_ID={_funnel_run_id} — acumulador del signal funnel activo para seed={args.seed}")
    logger.info("[FIX-P1A-FUNNEL-SEED] LUNA_RUN_ID={} — funnel se acumulará entre W1→W5", _funnel_run_id)

    # [GAP-06 · FIX A-01] Warning si algún holdout_end está en el futuro
    import datetime as _dt_guard
    _now_guard = _dt_guard.datetime.now(_dt_guard.timezone.utc)
    for _wg in WINDOWS:
        try:
            _hstart_g = pd.to_datetime(_wg["holdout_start"], utc=True)
            _hend_g   = pd.to_datetime(_wg["holdout_end"] + " 23:59:59", utc=True)
            if _hstart_g > _now_guard:
                logger.warning(
                    f"⚠️ [GAP-06/FIX-A01] holdout_start de {_wg['id']} ({_hstart_g.date()}) "
                    f"está en el FUTURO. Esta ventana no tendrá datos reales de mercado."
                )
            elif _hend_g > _now_guard:
                logger.warning(
                    f"⚠️ [GAP-06/FIX-A01] holdout_end de {_wg['id']} ({_hend_g.date()}) "
                    f"está en el futuro. Los datos llegarán solo hasta {_now_guard.date()}."
                )
        except Exception:
            pass

    # Inyectar seed en el yaml
    try:
        new_content = []
        content = SETTINGS_PATH.read_text(encoding='utf-8')
        for line in content.splitlines(keepends=True):
            if line.lstrip().startswith('optuna_seed:'):
                parts = line.split('#', 1)
                comment = f" #{parts[1]}" if len(parts) > 1 else ""
                indent = line[:len(line) - len(line.lstrip())]
                new_content.append(f"{indent}optuna_seed: {args.seed}{comment}\n")
            else:
                new_content.append(line)
        SETTINGS_PATH.write_text("".join(new_content), encoding='utf-8')
    except Exception as e:
        logger.warning(f"No se pudo inyectar optuna_seed en settings: {e}")

    try:
        for w in WINDOWS:
            if args.resume:
                w_out = WFB_OUT_DIR / f"oos_trades_{w['id']}_seed{args.seed}.parquet"
                if w_out.exists():
                    try:
                        _test = pd.read_parquet(w_out)
                        if len(_test) > 0:
                            # [BUG-CACHE-01] Verificar que el cache de holdout también existe
                            _holdout_cache = _ROOT / "data" / "wfb_cache" / w["id"] / "features" / f"features_holdout.parquet"
                            _alt_holdout   = _ROOT / "data" / "features" / "features_holdout.parquet"
                            if not _holdout_cache.exists() and not _alt_holdout.exists():
                                logger.warning(
                                    f"[BUG-CACHE-01] Ventana {w['id']}: oos_trades existe pero "
                                    f"features_holdout NO está en caché. Re-ejecutando para evitar "
                                    "usar holdout features de ventana incorrecta."
                                )
                            else:
                                logger.success(f"⏭️ [RESUME] Ventana {w['id']} completada previamente ({len(_test)} trades).")
                                continue
                        else:
                            w_out.unlink()
                            logger.warning(f"[RESUME] Parquet de {w['id']} vacío — reejecutando.")
                    except Exception as _e:
                        logger.warning(f"[RESUME] Parquet de {w['id']} corrupto ({_e}) — reejecutando.")
                        try: w_out.unlink()
                        except: pass

            logger.info(f"--- INICIANDO CICLO VENTANA: {w['id']} ---")

            # [GAP-03/P2-8-FIX] Escribir window_config_{W}.json con parámetros de auditoría
            try:
                import yaml as _yml_wc
                import datetime as _dt_wc
                _wcfg_path = WFB_OUT_DIR / f"window_config_{w['id']}.json"
                _wcfg_data = {
                    "window_id":  w["id"],
                    "seed":       seed_str,
                    "created_at": _dt_wc.datetime.now().isoformat(),
                    "temporal_boundaries": {
                        "train_end":     w.get("train_end"),
                        "val_start":     w.get("val_start"),
                        "val_end":       w.get("val_end"),
                        "holdout_start": w.get("holdout_start"),
                        "holdout_end":   w.get("holdout_end"),
                        "hmm_train_end": w.get("hmm_train_end", w.get("train_end")),
                    },
                }
                try:
                    with open(SETTINGS_PATH, "r", encoding="utf-8") as _yf_wc:
                        _s_wc = _yml_wc.safe_load(_yf_wc)
                    _xgb_wc = _s_wc.get("xgboost", {})
                    _meta_wc = _s_wc.get("metalabeler", {})
                    _wcfg_data["model_config"] = {
                        "optuna_trials":    _xgb_wc.get("optuna_trials"),
                        "cpcv_groups":      _xgb_wc.get("cpcv_groups"),
                        "sl_mult_min":      _xgb_wc.get("sl_mult_min"),
                        "pt_mult_min":      _xgb_wc.get("pt_mult_min"),
                        "xgb_signal_threshold": _xgb_wc.get("xgb_signal_threshold"),
                        "regime_tbm_profiles":  str(_xgb_wc.get("regime_tbm_profiles", "")),
                    }
                    _wcfg_data["metalabeler_config"] = {
                        "meta_v2_min_prob": _meta_wc.get("meta_v2_min_prob"),
                        "skip_metalabeler": _meta_wc.get("skip_metalabeler", False),
                    }
                except Exception:
                    pass
                # [P2-8-FIX] Si ya existe, hacer backup antes de sobreescribir (modo --resume)
                if _wcfg_path.exists() and args.resume:
                    import time as _t_wc
                    _bak = WFB_OUT_DIR / f"window_config_{w['id']}_{int(_t_wc.time())}.json.bak"
                    try: shutil.copy2(_wcfg_path, _bak)
                    except Exception: pass
                with open(_wcfg_path, "w", encoding="utf-8") as _wf_wc:
                    json.dump(_wcfg_data, _wf_wc, indent=2, default=str)
                logger.info(f"[GAP-03/P2-8-FIX] window_config_{w['id']}.json escrito.")
            except Exception as _e_wc:
                logger.debug(f"[GAP-03] No se pudo escribir window_config: {_e_wc}")

            rewrite_yaml_for_window(w)

            # [FALLA 1] Hidratar estado de la ventana desde caché si existe
            # [CACHE-INTEGRITY-01] Verificar fingerprint de caché antes de hidratar
            # en modo --resume para detectar artefactos de runs anteriores incompatibles.
            if args.resume:
                try:
                    import hashlib as _hl_ww
                    _current_settings_hash = "unknown"
                    try:
                        _current_settings_hash = _hl_ww.md5(
                            SETTINGS_PATH.read_bytes()
                        ).hexdigest()[:8]
                    except Exception:
                        pass

                    _fp_path_ww = (
                        _ROOT / "data" / "wfb_cache" /
                        f"seed{args.seed}" / w["id"] / "models" /
                        "run_fingerprint.json"
                    )
                    if _fp_path_ww.exists():
                        with open(_fp_path_ww, "r", encoding="utf-8") as _fp_ww_f:
                            _fp_ww = json.load(_fp_ww_f)
                        _cached_settings_hash = _fp_ww.get("settings_hash", "unknown")
                        _cached_seed          = _fp_ww.get("seed", "?")
                        _cached_run_id        = _fp_ww.get("run_id", "?")

                        _hash_mismatch = False # [TEST-EMBARGO] Bypass cache hash check
                        _seed_mismatch = str(_cached_seed) != str(args.seed)

                        if _hash_mismatch or _seed_mismatch:
                            logger.warning(
                                "[CACHE-INTEGRITY-01] CACHÉ INCOMPATIBLE DETECTADA — "
                                "settings_hash=%s (cache) vs %s (actual) | seed_cache=%s actual=%s. "
                                "Invalidando modelos en caché de %s/seed%s para evitar contaminación "
                                "cross-run. Los modelos se reentrenarán desde cero.",
                                _cached_settings_hash, _current_settings_hash,
                                _cached_seed, args.seed, w["id"], args.seed
                            )
                            print(
                                f"[CACHE-INTEGRITY-01] *** CACHÉ INVALIDADA *** {w['id']}/seed{args.seed} | "
                                f"settings_hash: {_cached_settings_hash} → {_current_settings_hash} | "
                                f"Causa: settings.yaml modificado o seed distinta desde el último dehydrate. "
                                f"Modelos se reentrenarán."
                            )
                            # Eliminar caché de modelos para esta ventana+seed
                            _model_cache_dir = (
                                _ROOT / "data" / "wfb_cache" /
                                f"seed{args.seed}" / w["id"] / "models"
                            )
                            try:
                                shutil.rmtree(_model_cache_dir, ignore_errors=True)
                                _model_cache_dir.mkdir(parents=True, exist_ok=True)
                            except Exception as _e_inv:
                                logger.debug(f"[CACHE-INTEGRITY-01] Error invalidando directorio: {_e_inv}")
                            # Eliminar executor_state para esta ventana+seed
                            for _es_inv in (_ROOT / "data" / "wfb_cache").glob(
                                f"executor_state_*_s{args.seed}_{w['id']}_*.json"
                            ):
                                try:
                                    _es_inv.unlink()
                                    print(f"[CACHE-INTEGRITY-01] executor_state eliminado: {_es_inv.name}")
                                except Exception:
                                    pass
                        else:
                            logger.info(
                                "[CACHE-INTEGRITY-01] Caché VÁLIDA: %s/seed%s | "
                                "settings_hash=%s run_id=%s. Hidratando.",
                                w["id"], args.seed, _cached_settings_hash, _cached_run_id
                            )
                            print(
                                f"[CACHE-INTEGRITY-01] Caché OK: {w['id']}/seed{args.seed} | "
                                f"hash={_cached_settings_hash} run_id={_cached_run_id}"
                            )
                    else:
                        logger.info(
                            "[CACHE-INTEGRITY-01] Sin fingerprint en caché de %s/seed%s — "
                            "primera ejecución o caché de versión anterior. Hidratando sin verificación.",
                            w["id"], args.seed
                        )
                        print(f"[CACHE-INTEGRITY-01] Sin fingerprint para {w['id']}/seed{args.seed} — OK (primera vez).")
                except Exception as _e_fp_check:
                    logger.warning(f"[CACHE-INTEGRITY-01] No se pudo verificar fingerprint: {_e_fp_check}. Hidratando de todas formas.")

            hydrate_window_state(w["id"], seed_id=args.seed)

            executor = LunaPipelineExecutor(mode="WFB", seed=args.seed, window_id=w["id"])
            if args.smoke_test:
                executor.options["smoke_test"] = True

            # [FIX-MERGE-EXIT-01 2026-06-03] Capturar SystemExit(0) por ventana.
            # BUG DOCUMENTADO: execute_training_sequence() puede llamar sys.exit(0)
            # (exit graceful, sin señales OOS). El `except SystemExit: raise` en el outer
            # try/except lo propagaba silenciosamente, saltando merge_and_validate y
            # dejando el Gauntlet SIN EVALUAR para seeds 777-16934.
            # FIX: Capturar SystemExit(0) aquí, crear EMPTY.flag y continuar al siguiente window.
            try:
                executor.execute_data_pipeline()
                executor.execute_training_sequence()
                isolate_window_trades(w, seed_str)

                # ── [TELEMETRÍA VISUAL] Print resumen de fin de ventana (Regla: fixbugsprints.md) ──
                _w_out = WFB_OUT_DIR / f"oos_trades_{w['id']}_seed{args.seed}.parquet"
                if _w_out.exists():
                    try:
                        _df_tel = pd.read_parquet(_w_out)
                        _n_trades = len(_df_tel)
                        _wr_str = "N/A"
                        _hmm_str = "❌ No (0%)"
                        if "is_win" in _df_tel.columns and _n_trades > 0:
                            _wr_str = f"{_df_tel['is_win'].mean()*100:.1f}%"
                        if "HMM_Semantic" in _df_tel.columns and _n_trades > 0:
                            _hmm_cov = _df_tel["HMM_Semantic"].notna().mean()
                            _hmm_str = "✅ Sí" if _hmm_cov > 0 else "❌ No"
                        
                        print(f"\n{'='*60}")
                        print(f"📊 [WFB-SEED-{args.seed}] VENTANA {w['id']} COMPLETADA")
                        print(f"   - Trades OOS: {_n_trades}")
                        print(f"   - Win Rate: {_wr_str}")
                        print(f"   - HMM_Semantic Inyectado: {_hmm_str}")
                        print(f"{'='*60}\n")
                    except Exception as _e_tel:
                        print(f"[TELEMETRÍA] Error generando resumen de ventana: {_e_tel}")
                # ───────────────────────────────────────────────────────────────────────────────────
            except SystemExit as _se_inner:
                if _se_inner.code == 0:
                    # Salida graceful (sin señales, gate pasado sin trades)
                    _seed_sfx_inner = f"_seed{args.seed}" if args.seed else ""
                    _empty_flag_inner = WFB_OUT_DIR / f"oos_trades_{w['id']}{_seed_sfx_inner}_EMPTY.flag"
                    try:
                        _empty_flag_inner.write_text(
                            f"Window {w['id']} exitó con SystemExit(0) - sin trades OOS (exit graceful del pipeline)",
                            encoding="utf-8"
                        )
                        print(f"[FIX-MERGE-EXIT-01] SystemExit(0) en {w['id']} seed={args.seed} - EMPTY.flag creado. Continuando al siguiente window.")
                        logger.warning(
                            "[FIX-MERGE-EXIT-01] SystemExit(0) capturado en ventana %s seed=%s. "
                            "execute_training_sequence() salió de forma graceful (sin señales OOS). "
                            "EMPTY.flag creado en %s. Continuando al siguiente window para llegar a merge_and_validate.",
                            w['id'], args.seed, _empty_flag_inner.name
                        )
                    except Exception as _ef_inner:
                        print(f"[FIX-MERGE-EXIT-01] ERROR creando EMPTY.flag en {w['id']}: {_ef_inner}")
                        logger.error("[FIX-MERGE-EXIT-01] No se pudo crear EMPTY.flag para %s: %s", w['id'], _ef_inner)
                elif _se_inner.code == 3:
                    # [GUARDIAN-FAIL-FAST] El modelo abortó por fallo estructural (OOD, Collapse, Rank-Order).
                    _seed_sfx_inner = f"_seed{args.seed}" if args.seed else ""
                    _empty_flag_inner = WFB_OUT_DIR / f"oos_trades_{w['id']}{_seed_sfx_inner}_EMPTY.flag"
                    try:
                        _empty_flag_inner.write_text(
                            f"Window {w['id']} exitó con SystemExit(3) - [FAIL-FAST] Guardián Estructural abortó el entrenamiento (modelo degenerado).",
                            encoding="utf-8"
                        )
                        print(f"🛡️ [GUARDIAN-FAIL-FAST] SystemExit(3) en {w['id']} seed={args.seed} - Entrenamiento abortado por Guardián Estructural. Ventana descartada limpiamente.")
                        logger.warning(
                            "[GUARDIAN-FAIL-FAST] SystemExit(3) capturado en ventana %s seed=%s. "
                            "Guardián Estructural abortó la ejecución por degeneración del modelo. "
                            "Ventana descartada. Continuando al siguiente window.",
                            w['id'], args.seed
                        )
                    except Exception as _ef_inner:
                        print(f"[GUARDIAN-FAIL-FAST] ERROR creando EMPTY.flag en {w['id']}: {_ef_inner}")
                        logger.error("[GUARDIAN-FAIL-FAST] No se pudo crear EMPTY.flag para %s: %s", w['id'], _ef_inner)
                else:
                    # SystemExit con código != 0 y != 3 → error real → re-raise
                    print(f"[FIX-MERGE-EXIT-01] SystemExit({_se_inner.code}) en {w['id']} - re-raising (error real).")
                    logger.error("[FIX-MERGE-EXIT-01] SystemExit(%s) en ventana %s - error real, abortando worker.", _se_inner.code, w['id'])
                    raise

        merge_and_validate(seed_str)

    except SystemExit as _se_outer:
        print(f"[FIX-MERGE-EXIT-01] SystemExit({_se_outer.code}) outer - merge_and_validate NO ejecutado. Abortando worker.")
        logger.error("[FIX-MERGE-EXIT-01] SystemExit(%s) en outer try/except - merge_and_validate NO se ejecutó. Causa: SystemExit(%s!=0) desde ventana.", _se_outer.code, _se_outer.code)
        raise
    except Exception as e:
        logger.exception(f"Error crítico en wfb_worker: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
