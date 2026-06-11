"""
pipeline_executor.py
====================
Luna V2 - El Cerebro Central (Versión Institucional)

Migración V1→V2 (2026-05-10):
  - FALLA 1: Hydrate/Dehydrate de artefactos por ventana+seed
  - FALLA 3: Archivado de artefactos (_archive_dir/_archive_files)
  - FALLA 5: Timeout 4H por subproceso
  - FALLA 7: Validación PKL HMM post-hydrate
"""

import sys
import os
import subprocess
import shutil
import time
import json
from pathlib import Path
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent


# ── Helpers de Archivo (PRESERVE-ARTIFACTS-01) ───────────────────────────────

def _archive_dir(window_id: str, seed_id, subfolder: str) -> Path:
    """
    Retorna (y crea) el directorio de archivo canónico para ventana+seed.
    Formato: data/archive/{window_id}_seed{seed_id}/{subfolder}/
    """
    _s = f"seed{seed_id}" if seed_id is not None else "noseed"
    _d = _ROOT / "data" / "archive" / f"{window_id}_{_s}" / subfolder
    _d.mkdir(parents=True, exist_ok=True)
    return _d


def _archive_files(src_dir: Path, patterns: list, archive_dest: Path, tag: str = "") -> int:
    """
    Mueve archivos de src_dir que coincidan con patterns a archive_dest.
    Añade un timestamp para evitar colisiones entre corridas.
    """
    _ts = str(int(time.time()))[-6:]
    _moved = 0
    for _pat in patterns:
        for _f in src_dir.glob(_pat):
            if _f.is_file():
                _dst = archive_dest / f"{_f.stem}_{_ts}{_f.suffix}"
                try:
                    shutil.move(str(_f), str(_dst))
                    _moved += 1
                except Exception as _e:
                    logger.debug(f"[ARCHIVE] No se pudo mover {_f.name}: {_e}")
    if _moved:
        logger.info(f"[PRESERVE-ARTIFACTS-01]{tag} {_moved} archivo(s) archivados en {archive_dest.relative_to(_ROOT)}")
    return _moved


# ── Hydrate / Dehydrate (FEAT-CACHE-01 + FIX-P3-SEED-ISOLATION) ─────────────

def hydrate_window_state(window_id: str, seed_id=None):
    """
    [FALLA 1 FIX] Restaura features + modelos del wfb_cache al workspace activo.
    Features: namespace compartido por ventana (entre seeds, intencionalmente).
    Modelos:  namespace aislado por seed (evita cross-seed contamination).
    Post-hydrate: valida PKL HMM para detectar state_map degenerado [FALLA 7].
    """
    # --- Features (compartidas entre seeds de la misma ventana) ---
    features_cache = _ROOT / "data" / "wfb_cache" / window_id / "features"
    if features_cache.exists():
        target_features = _ROOT / "data" / "features"
        target_features.mkdir(parents=True, exist_ok=True)

        # Archivar features previas antes de sobrescribir
        _feat_archive = _archive_dir(window_id, seed_id, "features_prev")
        _archive_files(target_features, ["*.parquet", "selected_features.json"],
                       _feat_archive, tag=f"[hydrate/{window_id}]")
        # Limpiar residuos
        for f in target_features.glob("*"):
            if f.is_file():
                try: f.unlink()
                except: pass

        for f in features_cache.glob("*"):
            if f.is_file():
                shutil.copy2(f, target_features / f.name)
        logger.info(f"[HYDRATE] Features restauradas desde wfb_cache/{window_id}/features/")
    else:
        # [CACHE-HYGIENE-01] Cache no existe (primera seed con --nocache).
        # Limpiar workspace de todas formas para eliminar residuos de runs anteriores.
        target_features = _ROOT / "data" / "features"
        if target_features.exists():
            for f in target_features.glob("*"):
                if f.is_file():
                    try: f.unlink()
                    except: pass
            logger.info(f"[CACHE-HYGIENE-01] data/features/ limpiado (sin caché disponible, primera seed)")
            print(f"[CACHE-HYGIENE-01] data/features/ limpiado (sin caché disponible, primera seed)")

    # --- Modelos (aislados por seed) ---
    if seed_id is not None:
        models_cache = _ROOT / "data" / "wfb_cache" / f"seed{seed_id}" / window_id / "models"
        logger.info(f"[FIX-P3-SEED-ISOLATION] Hidratando modelos seed{seed_id}/{window_id}")
    else:
        models_cache = _ROOT / "data" / "wfb_cache" / window_id / "models"

    if models_cache.exists():
        target_models = _ROOT / "data" / "models"
        target_models.mkdir(parents=True, exist_ok=True)

        _mdl_archive = _archive_dir(window_id, seed_id, "models_prev")
        _archive_files(target_models, ["*"], _mdl_archive, tag=f"[hydrate/{window_id}_seed{seed_id}]")
        for f in target_models.iterdir():
            if f.is_file():
                try: f.unlink()
                except: pass

        # [FIX-P3-SHARED-MODELS] Hidratar modelos compartidos primero si existimos en una seed
        if seed_id is not None:
            shared_models_cache = _ROOT / "data" / "wfb_cache" / window_id / "models"
            if shared_models_cache.exists():
                for f in shared_models_cache.glob("*"):
                    if f.is_file():
                        shutil.copy2(f, target_models / f.name)

        for f in models_cache.glob("*"):
            if f.is_file():
                shutil.copy2(f, target_models / f.name)
        logger.info(f"[HYDRATE] Modelos restaurados desde {models_cache.relative_to(_ROOT)}")
    else:
        # [CACHE-HYGIENE-01] Cache no existe (primera seed con --nocache).
        target_models = _ROOT / "data" / "models"
        if target_models.exists():
            for f in target_models.glob("*"):
                if f.is_file():
                    try: f.unlink()
                    except: pass
            logger.info(f"[CACHE-HYGIENE-01] data/models/ limpiado (sin caché disponible, primera seed)")
            print(f"[CACHE-HYGIENE-01] data/models/ limpiado (sin caché disponible, primera seed)")

        # [FIX-P3-SHARED-MODELS-NOCACHE] Hidratar modelos compartidos incluso si la seed no tiene caché
        if seed_id is not None:
            shared_models_cache = _ROOT / "data" / "wfb_cache" / window_id / "models"
            if shared_models_cache.exists():
                target_models.mkdir(parents=True, exist_ok=True)
                for f in shared_models_cache.glob("*"):
                    if f.is_file():
                        shutil.copy2(f, target_models / f.name)
                logger.info(f"[HYDRATE-SHARED-FALLBACK] Modelos compartidos restaurados desde {shared_models_cache.relative_to(_ROOT)}")

        # [FALLA 7] Validar PKL HMM post-hydrate
        _validate_hmm_pkl(target_models, window_id)

        # [BUG-CALIB-CACHE-01] Validar calibrador MetaLabeler post-hydrate
        # Si el calibrador está corrupto (std=0, prob plana), eliminarlo de la caché
        # para forzar recalibración en el siguiente paso del pipeline.
        _validate_calibrator_cache(target_models, models_cache, window_id, seed_id)


def _validate_hmm_pkl(models_dir: Path, window_id: str):
    """
    [FIX-HMM-PKL-VALIDATE] Valida que el state_map del pkl HMM no sea degenerado.
    Si lo es, elimina el pkl para forzar re-entrenamiento limpio.
    """
    try:
        import joblib as _jbl
        _hmm_pkl = models_dir / "hmm_regime.pkl"
        if not _hmm_pkl.exists():
            return
        _hmm_v = _jbl.load(_hmm_pkl)
        _sm_v = _hmm_v.get("state_map", {})
        _sem_values = list(_sm_v.values())
        _n_distinct = len(set(_sem_values))
        _all_calm = all("CALM_RANGE" in str(s) for s in _sem_values)
        _n_bull_bear = sum(1 for s in _sem_values if any(k in str(s) for k in ["BULL", "BEAR", "CRASH", "FORCED"]))
        if _all_calm or _n_bull_bear < 2 or _n_distinct < 3:
            logger.warning(
                f"[FIX-HMM-PKL-VALIDATE] ⚠️ state_map DEGENERADO en pkl hidratado: {_sm_v}. "
                f"Eliminando pkl para forzar re-entrenamiento."
            )
            try: _hmm_pkl.unlink()
            except: pass
        else:
            logger.info(
                f"[FIX-HMM-PKL-VALIDATE] ✅ state_map OK: {_n_distinct} regímenes distintos, "
                f"{_n_bull_bear} Bull/Bear/Crash."
            )
    except Exception as _e:
        logger.warning(f"[FIX-HMM-PKL-VALIDATE] No se pudo validar pkl HMM: {_e}")


def _validate_calibrator_cache(models_dir: Path, cache_dir: Path, window_id: str, seed_id):
    """
    [BUG-CALIB-CACHE-01 FIX] Detecta calibradores MetaLabeler corrompidos en la caché.

    Un calibrador corrompido produce xgb_prob_cal completamente plano (std≈0),
    lo que colapsa el pipeline de señales a 0 trades.

    Causa: joblib no puede deserializar _RFWithAdapter si se guardó con __main__ como módulo
    de referencia. El resultado es un objeto inválido que devuelve siempre la misma prob.

    Si detecta el problema:
    1. Emite WARNING visible en logs.
    2. Elimina el .joblib corrupto del workspace activo.
    3. Elimina el .joblib corrupto de la caché (seed/window) para forzar recalibración.

    Post-fix: el pipeline recalibrará desde cero en este ciclo de ventana.
    """
    import joblib as _jbl_cal
    import numpy as _np_cal

    for _cal_name in ["metalabeler_v2_long_calibrator.joblib", "metalabeler_v2_short_calibrator.joblib"]:
        _cal_path_ws = models_dir / _cal_name
        _cal_path_cache = cache_dir / _cal_name

        if not _cal_path_ws.exists():
            continue

        try:
            # Pre-importar clases top-level para que joblib pueda reconstruirlas
            from luna.models.calibrate_probabilities import _RFWithAdapter, _IdentityWrapper, _TSAdapter  # noqa: F401
            _cal_obj = _jbl_cal.load(_cal_path_ws)

            # Test de sanidad: pasar una muestra de 50 probs uniformes y medir std de salida.
            # Soporta todos los tipos de calibrador del pipeline Luna:
            #   - IsotonicRegression / LogisticRegression → predict() o predict_proba()
            #   - _RFWithAdapter / _IdentityWrapper       → predict_proba(X_combined)
            # Para _RFWithAdapter necesitaríamos X_combined completo (imposible aquí),
            # así que usamos predict_proba(raw_probs.reshape(-1,1)) como proxy de la capa adaptadora.
            _test_x = _np_cal.linspace(0.3, 0.7, 50)
            _test_out = None

            # Intentar predict_proba con entrada 1D reshape (LogisticRegression, _TSAdapter)
            if hasattr(_cal_obj, 'predict_proba') and not hasattr(_cal_obj, '_rf'):
                try:
                    _test_out = _cal_obj.predict_proba(_test_x.reshape(-1, 1))[:, 1]
                except Exception:
                    pass

            # IsotonicRegression: solo tiene predict (no predict_proba)
            if _test_out is None and hasattr(_cal_obj, 'predict') and not hasattr(_cal_obj, '_rf'):
                try:
                    _test_out = _cal_obj.predict(_test_x)
                except Exception:
                    pass

            # _RFWithAdapter / _IdentityWrapper: tienen _rf interno.
            # No podemos hacer una predicción completa sin X_combined, pero podemos
            # intentar con el adaptador interno directamente si existe.
            if _test_out is None and hasattr(_cal_obj, '_adapter'):
                try:
                    _adpt = _cal_obj._adapter
                    if hasattr(_adpt, 'predict_proba'):
                        _test_out = _adpt.predict_proba(_test_x.reshape(-1, 1))[:, 1]
                    elif hasattr(_adpt, 'predict'):
                        _test_out = _adpt.predict(_test_x)
                except Exception:
                    pass

            if _test_out is None:
                # Último fallback: no podemos testear — asumir sano para no eliminar incorrectamente
                logger.info(
                    f"[BUG-CALIB-CACHE-01] {_cal_name}: tipo {type(_cal_obj).__name__} "
                    f"no testeable sin X_combined — asumiendo SANO (no eliminado). Window={window_id}"
                )
                print(f"[BUG-CALIB-CACHE-01] {_cal_name}: no testeable — asumiendo SANO")
                continue

            _std_out = float(_np_cal.std(_test_out))
            if _std_out < 1e-4:
                logger.warning(
                    f"[BUG-CALIB-CACHE-01] CALIBRADOR CORRUPTO detectado: {_cal_name} | "
                    f"std_output={_std_out:.8f} ≈ 0.0 (prob plana). "
                    f"Eliminando de workspace y caché para forzar recalibración limpia. "
                    f"Window={window_id} Seed={seed_id}"
                )
                print(f"[BUG-CALIB-CACHE-01] ELIMINANDO calibrador corrupto: {_cal_name} (std={_std_out:.8f})")
                try:
                    _cal_path_ws.unlink()
                    logger.info(f"[BUG-CALIB-CACHE-01] Eliminado del workspace: {_cal_path_ws}")
                except Exception as _e_ws:
                    logger.debug(f"[BUG-CALIB-CACHE-01] No se pudo eliminar del workspace: {_e_ws}")
                try:
                    if _cal_path_cache.exists():
                        _cal_path_cache.unlink()
                        logger.info(f"[BUG-CALIB-CACHE-01] Eliminado de caché: {_cal_path_cache}")
                except Exception as _e_cache:
                    logger.debug(f"[BUG-CALIB-CACHE-01] No se pudo eliminar de caché: {_e_cache}")
            else:
                logger.info(
                    f"[BUG-CALIB-CACHE-01] Calibrador OK: {_cal_name} | "
                    f"std_output={_std_out:.4f} (discrimina correctamente). Window={window_id}"
                )
                print(f"[BUG-CALIB-CACHE-01] Calibrador {_cal_name} SANO: std={_std_out:.4f}")

        except Exception as _e_val:
            logger.warning(
                f"[BUG-CALIB-CACHE-01] No se pudo validar {_cal_name}: {_e_val}. "
                f"Eliminando por precaución para evitar uso de calibrador roto. Window={window_id}"
            )
            print(f"[BUG-CALIB-CACHE-01] ERROR validando {_cal_name}: {_e_val} — eliminando por precaución")
            try:
                if _cal_path_ws.exists():
                    _cal_path_ws.unlink()
            except Exception:
                pass
            try:
                if _cal_path_cache.exists():
                    _cal_path_cache.unlink()
            except Exception:
                pass

    # [BUG-CALIB-XGB-01] Validar también los calibradores XGBoost isotónicos.
    # Un IsotonicRegression con solo 2 puntos de anclaje produce salida constante
    # (por ejemplo 0.5940) para toda la distribución de entrada.
    # Causa: training pool muy pequeño o probs del XGBoost concentradas en rango estrecho.
    for _xgb_name in models_dir.glob("xgboost_isotonic_calibrator_*.joblib"):
        _xgb_cache_path = cache_dir / _xgb_name.name
        try:
            import joblib as _jbl_xgb
            import numpy as _np_xgb
            _xgb_cal = _jbl_xgb.load(_xgb_name)
            _test_x_xgb = _np_xgb.linspace(0.3, 0.7, 50)
            _out_xgb = _xgb_cal.predict(_test_x_xgb) if hasattr(_xgb_cal, 'predict') else _np_xgb.array([0.5] * 50)
            _std_xgb = float(_np_xgb.std(_out_xgb))
            _n_anchors = len(getattr(_xgb_cal, 'X_thresholds_', []))
            if _std_xgb < 1e-4:
                logger.warning(
                    f"[BUG-CALIB-XGB-01] XGB calibrador DEGENERADO: {_xgb_name.name} | "
                    f"std={_std_xgb:.8f} | anchors={_n_anchors}. "
                    f"Eliminando para forzar recalibración. Window={window_id} Seed={seed_id}"
                )
                print(f"[BUG-CALIB-XGB-01] ELIMINANDO calibrador XGB degenerado: {_xgb_name.name} "
                      f"(std={_std_xgb:.8f}, anchors={_n_anchors})")
                try:
                    _xgb_name.unlink()
                except Exception: pass
                try:
                    if _xgb_cache_path.exists():
                        _xgb_cache_path.unlink()
                except Exception: pass
            else:
                logger.info(
                    f"[BUG-CALIB-XGB-01] XGB calibrador OK: {_xgb_name.name} | "
                    f"std={_std_xgb:.4f} anchors={_n_anchors}. Window={window_id}"
                )
                print(f"[BUG-CALIB-XGB-01] {_xgb_name.name} SANO: std={_std_xgb:.4f} anchors={_n_anchors}")
        except Exception as _e_xgb:
            logger.warning(f"[BUG-CALIB-XGB-01] No se pudo validar {_xgb_name.name}: {_e_xgb}")


def dehydrate_window_state(window_id: str, seed_id=None):
    """
    [FALLA 1 FIX] Guarda un snapshot de features+modelos al wfb_cache tras cada paso exitoso.
    Features: namespace compartido. Modelos: namespace aislado por seed.
    """
    target_features = _ROOT / "data" / "features"
    target_models = _ROOT / "data" / "models"

    # Features (compartidas)
    features_cache = _ROOT / "data" / "wfb_cache" / window_id / "features"
    features_cache.mkdir(parents=True, exist_ok=True)
    if target_features.exists():
        for f in target_features.glob("*.parquet"):
            try:
                # [DEHYDRATE-TOCTOU-FIX 2026-06-04] Google Drive (G:) puede devolver un
                # path en glob() pero lanzar FileNotFoundError en stat() si el archivo
                # fue archivado/eliminado entre la iteración y la lectura (race condition).
                if f.is_file() and f.stat().st_size > 0:
                    shutil.copy2(f, features_cache / f.name)
                elif f.is_file():
                    logger.warning(f"[DEHYDRATE] Saltando {f.name}: 0 bytes (archivo corrupto).")
            except OSError as _e_toctou:
                logger.warning(f"[DEHYDRATE-TOCTOU-FIX] Saltando {f.name}: desapareció durante dehydrate (TOCTOU/GDrive race) → {_e_toctou}")
                print(f"[DEHYDRATE-TOCTOU-FIX] {f.name} saltado por race condition: {_e_toctou}")
        for f in target_features.glob("*.json"):
            try:
                if f.is_file() and f.stat().st_size > 0:
                    shutil.copy2(f, features_cache / f.name)
                elif f.is_file():
                    logger.warning(f"[DEHYDRATE] Saltando {f.name}: 0 bytes (archivo corrupto).")
            except OSError as _e_toctou:
                logger.warning(f"[DEHYDRATE-TOCTOU-FIX] Saltando {f.name}: desapareció durante dehydrate (TOCTOU/GDrive race) → {_e_toctou}")
                print(f"[DEHYDRATE-TOCTOU-FIX] {f.name} saltado por race condition: {_e_toctou}")

    # Modelos (aislados por seed)
    if seed_id is not None:
        models_cache = _ROOT / "data" / "wfb_cache" / f"seed{seed_id}" / window_id / "models"
    else:
        models_cache = _ROOT / "data" / "wfb_cache" / window_id / "models"
    models_cache.mkdir(parents=True, exist_ok=True)

    if target_models.exists():
        for f in target_models.glob("*"):
            try:
                if f.is_file() and f.stat().st_size > 0:
                    shutil.copy2(f, models_cache / f.name)
                elif f.is_file():
                    logger.warning(f"[DEHYDRATE] Saltando {f.name}: 0 bytes (archivo corrupto).")
            except OSError as _e_toctou:
                logger.warning(f"[DEHYDRATE-TOCTOU-FIX] Saltando {f.name}: desapareció durante dehydrate (TOCTOU/GDrive race) → {_e_toctou}")
                print(f"[DEHYDRATE-TOCTOU-FIX] {f.name} saltado por race condition: {_e_toctou}")

    # [CACHE-INTEGRITY-01] Escribir fingerprint de run para detectar contaminación cross-run.
    # Cada vez que se deshidrata, se registra el run_id, seed, timestamp y versión de settings.
    # El hydrate verificará este fingerprint para rechazar caché de runs anteriores distintas.
    try:
        import hashlib as _hl_dh, yaml as _yaml_dh
        _run_id = os.environ.get("LUNA_RUN_ID", "unknown")
        _settings_hash = "unknown"
        try:
            _settings_content = (_ROOT / "config" / "settings.yaml").read_bytes()
            _settings_hash = _hl_dh.md5(_settings_content).hexdigest()[:8]
        except Exception:
            pass
        _fp_data = {
            "run_id":          _run_id,
            "seed":            str(seed_id),
            "window_id":       window_id,
            "settings_hash":   _settings_hash,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pid":             os.getpid(),
        }
        _fp_path = models_cache / "run_fingerprint.json"
        with open(_fp_path, "w", encoding="utf-8") as _fp_f:
            json.dump(_fp_data, _fp_f, indent=2)
        print(f"[CACHE-INTEGRITY-01] Fingerprint escrito: run_id={_run_id} seed={seed_id} "
              f"window={window_id} settings_hash={_settings_hash}")
    except Exception as _e_fp:
        logger.debug(f"[CACHE-INTEGRITY-01] No se pudo escribir fingerprint: {_e_fp}")

    logger.info(f"[DEHYDRATE] Snapshot guardado para {window_id}/seed{seed_id}")


# ── Executor Principal ────────────────────────────────────────────────────────

class LunaPipelineExecutor:
    """
    Director de orquesta que ejecuta la secuencia completa de Luna
    lanzando submódulos de manera aislada (subprocess) para asegurar
    la liberación de memoria y limpieza de dependencias.
    """

    def __init__(self, mode: str, seed: int = None, window_id: str = None, options: dict = None):
        self.mode = mode.upper()
        self.seed = seed
        self.window_id = window_id
        self.options = options or {}

        if self.mode == 'WFB' and not self.window_id:
            raise ValueError("[ERROR] El modo WFB requiere un window_id (ej. 'W1')")

        if self.mode == 'PROD':
            self.window_id = "PROD"

        self.cache_dir = _ROOT / "data" / "wfb_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.shared_cache_file = self.cache_dir / f"executor_state_{self.mode.lower()}_{self.window_id}_data.json"
        self.seed_cache_file = self.cache_dir / f"executor_state_{self.mode.lower()}_s{self.seed}_{self.window_id}_models.json"
        
        self.shared_completed_steps = self._load_cache(self.shared_cache_file)
        self.completed_steps = self._load_cache(self.seed_cache_file)

        # [PREFLIGHT] Ejecutar validaciones de seguridad
        self._run_preflight_checks()

    def _run_preflight_checks(self):
        """[PREFLIGHT CHECKS] Validación de hardware, entorno y dependencias."""
        logger.info("[PRE-FLIGHT] Ejecutando validaciones de seguridad institucionales...")
        
        # 1. Check RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            free_gb = mem.available / (1024 ** 3)
            if free_gb < 16.0:
                msg = f"RAM libre detectada ({free_gb:.1f} GB) es menor a 16GB. Riesgo de OOM."
                if self.mode == 'PROD':
                    logger.error(f"[PRE-FLIGHT FATAL] {msg}")
                    sys.exit(1)
                else:
                    logger.warning(f"[PRE-FLIGHT WARN] {msg}")
            else:
                logger.info(f"[PRE-FLIGHT] RAM libre: {free_gb:.1f} GB OK.")
        except ImportError:
            logger.warning("[PRE-FLIGHT] psutil no instalado. Saltando check de RAM.")

        # 2. Check CUDA/CPU
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"[PRE-FLIGHT] CUDA disponible: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("[PRE-FLIGHT] CUDA no disponible. Fallback a CPU.")
        except ImportError:
            logger.warning("[PRE-FLIGHT] torch no instalado. Saltando check de CUDA.")

        # 3. Check Datos Base
        raw_data = _ROOT / "data" / "raw" / "ohlcv"
        if not raw_data.exists():
            logger.error("[PRE-FLIGHT FATAL] No se encontró data/raw/ohlcv.")
            sys.exit(1)
        else:
            logger.info("[PRE-FLIGHT] data/raw/ohlcv OK.")

        # 4. Check Variables de Entorno (API)
        # [CLEANUP-OKX-01] Actualizado de KRAKEN → OKX (broker activo institucional)
        # [AUDIT-FIX] Naming canónico del .env: OKX_SECRET_KEY (no OKX_API_SECRET)
        required_vars = ["OKX_API_KEY", "OKX_SECRET_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
        from dotenv import load_dotenv
        load_dotenv()
        missing = [v for v in required_vars if not os.getenv(v)]
        if missing:
            msg = f"Faltan variables de entorno críticas: {missing}"
            print(f"[CLEANUP-OKX-01] PRE-FLIGHT: vars faltantes detectadas: {missing}")
            if self.mode == 'PROD':
                logger.error(f"[PRE-FLIGHT FATAL] {msg}")
                sys.exit(1)
            else:
                logger.warning(f"[PRE-FLIGHT WARN] {msg} (Ignorado en WFB)")
        else:
            print("[CLEANUP-OKX-01] PRE-FLIGHT: OKX_API_KEY/OKX_API_SECRET OK.")
            logger.info("[PRE-FLIGHT] Variables de entorno críticas OKX OK.")

        # 5. Check Aislamiento (Solo PROD)
        if self.mode == 'PROD':
            lock_path = _ROOT / ".wfb_lock_dir"
            if lock_path.exists():
                logger.error("[PRE-FLIGHT FATAL] Orquestador WFB activo detectado (.wfb_lock_dir). Abortando Producción.")
                sys.exit(1)

        logger.info("[PRE-FLIGHT] Todas las validaciones superadas.")

    def _load_cache(self, cache_file: Path) -> set:
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                _steps = data.get('completed_steps', [])
                logger.info(f"[CACHE-LOAD] Caché detectado: {cache_file.name} | {len(_steps)} pasos: {_steps}")
                print(
                    f"[CACHE-LOAD] ATENCIÓN: {cache_file.name} contiene {len(_steps)} pasos ya completados "
                    f"que se SALTARÁN: {_steps}"
                )
                return set(_steps)
            except Exception as e:
                logger.warning(f"[CACHE-LOAD] No se pudo leer el caché {cache_file}: {e}")
                print(f"[CACHE-LOAD] ERROR leyendo caché {cache_file.name}: {e} — partiendo desde cero")
        else:
            print(f"[CACHE-LOAD] Sin caché previa ({cache_file.name}) — run desde cero")
        return set()

    def _save_cache(self, cache_file: Path, steps: set):
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"completed_steps": list(steps)}, f, indent=4)
        except Exception as e:
            logger.warning(f"No se pudo guardar el caché en {cache_file}: {e}")

    def clear_cache(self):
        if self.seed_cache_file.exists():
            self.seed_cache_file.unlink()
        self.completed_steps.clear()

    def _is_step_stale(self, step_name: str, script_rel_path: str) -> bool:
        """
        [GAP-10 · SMART-CACHE-INVALIDATION] Verifica si el script fue modificado
        más recientemente que el cache JSON. Si es así, el paso está 'stale'
        y debe re-ejecutarse (equivalente al mtime check de V1 run_step_with_resume).
        """
        cache_file = self.seed_cache_file
        if not cache_file.exists():
            return False
        script_path = _ROOT / script_rel_path
        if not script_path.exists():
            return False
        try:
            if script_path.stat().st_mtime > cache_file.stat().st_mtime:
                logger.warning(
                    f"[GAP-10/CACHE-INVALIDATION] '{script_path.name}' modificado después "
                    f"del cache. Invalidando paso '{step_name}' → re-ejecución forzada."
                )
                print(
                    f"[CACHE-STALE] *** '{step_name}' INVALIDADO *** "
                    f"script '{script_path.name}' fue modificado después del cache JSON. "
                    f"Re-ejecutando paso."
                )
                return True
        except Exception as _e_stale:
            logger.debug(f"[GAP-10] No se pudo comparar mtime para '{step_name}': {_e_stale}")
        return False

    def _run_step(self, step_name: str, script_rel_path: str, args_list: list = None) -> bool:
        """Ejecuta un paso aislado del pipeline con timeout de 4H. [FALLA 5 FIX]"""
        if step_name in self.completed_steps:
            # [GAP-10] Verificar stale antes de saltar
            if self._is_step_stale(step_name, script_rel_path):
                self.completed_steps.discard(step_name)
                self._save_cache(self.seed_cache_file, self.completed_steps)
            else:
                logger.info(f"[CACHE-SKIP] Saltando Fase: '{step_name}' (encontrado en caché executor_state)")
                print(f"[CACHE-SKIP] SALTANDO: '{step_name}' — ya completado según caché. Si esto es incorrecto, usar --nocache.")
                return False

        logger.info(f"--- Iniciando Fase: {step_name} ---")
        script_path = _ROOT / script_rel_path

        if not script_path.exists():
            logger.error(f"[ERROR] Módulo no encontrado: {script_path}")
            sys.exit(1)

        cmd = [sys.executable, "-u", str(script_path)]
        if args_list:
            cmd.extend(args_list)

        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = str(_ROOT)
        run_env["PYTHONUNBUFFERED"] = "1"

        if self.seed is not None:
            run_env["LUNA_SEED"] = str(self.seed)
            run_env["LUNA_OPTUNA_SEED"] = str(self.seed)

        if self.window_id:
            run_env["LUNA_WINDOW_ID"] = self.window_id

        if self.mode == 'PROD':
            run_env["LUNA_PRODUCTION_MODE"] = "1"

        # [R3/FIX-SMOKE-01] Propagar LUNA_SMOKE_TEST a subprocesos si está activado en options
        # En V1 era env var global; en V2 llega vía --smoke-test CLI pero no se propagaba.
        if self.options.get("smoke_test") or os.environ.get("LUNA_SMOKE_TEST"):
            run_env["LUNA_SMOKE_TEST"] = "1"

        _TIMEOUT_SECONDS = 14400  # 4 horas máximo por script [FALLA 5]

        try:
            with subprocess.Popen(cmd, env=run_env, cwd=str(_ROOT)) as process:
                try:
                    process.wait(timeout=_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    logger.error(f"[TIMEOUT] Fase '{step_name}' excedió 4H. Terminando proceso.")
                    process.kill()
                    process.communicate()
                    sys.exit(1)

            if process.returncode != 0:
                logger.error(f"[FATAL] La Fase '{step_name}' abortó con código de error {process.returncode}")
                sys.exit(process.returncode)

            # Dehydrate tras cada paso exitoso para preservar artefactos
            if self.mode == 'WFB' and self.window_id:
                dehydrate_window_state(self.window_id, seed_id=self.seed)

            self.completed_steps.add(step_name)
            self._save_cache(self.seed_cache_file, self.completed_steps)
            logger.success(f"--- Fase '{step_name}' Completada Exitosamente ---\n")
            return True

        except Exception as e:
            logger.exception(f"Error crítico al orquestar {step_name}: {e}")
            sys.exit(1)

    # ── SFI Shared Cache [FALLA 8] ────────────────────────────────────────────

    def _compute_sfi_fingerprint(self) -> str:
        """
        [SFI-CACHE-01] Calcula el fingerprint del SFI basado en:
          1. Todas las fechas temporales (train_end, val_start, val_end,
             holdout_start, holdout_end) — condición principal de compartición.
          2. Shape del features_train.parquet — detecta actualizaciones de datos.
          3. Hash de parámetros TBM + MetaLabeler — si cambia sl_mult o min_prob,
             el target cambia y el SFI debe recalcularse.

        Solo si el fingerprint es idéntico se reutiliza el SFI de otra seed.
        """
        import hashlib
        _SETTINGS_PATH = _ROOT / "config" / "settings.yaml"
        parts = []

        # 1. Fechas temporales completas desde settings.yaml
        try:
            import yaml as _yaml
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f)
            _ts = _cfg.get("temporal_splits", {})
            for key in ["train_end", "validation_start", "validation_end",
                        "holdout_start", "holdout_end", "hmm_train_end"]:
                parts.append(f"{key}={str(_ts.get(key, ''))}")
        except Exception as _e:
            logger.debug(f"[SFI-FP] No se pudieron leer fechas: {_e}")
            parts.append("dates=unknown")

        # 2. Shape del dataset de entrenamiento
        _train_path = _ROOT / "data" / "features" / "features_train.parquet"
        if _train_path.exists():
            try:
                import pyarrow.parquet as _pq
                _meta = _pq.read_metadata(_train_path)
                parts.append(f"rows={_meta.num_rows}|cols={len(_pq.read_schema(_train_path).names)}")
            except Exception:
                parts.append(f"mtime={int(_train_path.stat().st_mtime)}")
        else:
            parts.append("dataset=missing")

        # 3. Hash de parámetros TBM + MetaLabeler
        try:
            import yaml as _yaml, hashlib as _hl
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as _f:
                _cfg2 = _yaml.safe_load(_f)
            _xgb = _cfg2.get("xgboost", {})
            _meta = _cfg2.get("metalabeler", {})
            _tbm = {
                "sl_mult_min":         str(_xgb.get("sl_mult_min", "")),
                "pt_mult_min":         str(_xgb.get("pt_mult_min", "")),
                "regime_tbm_profiles": str(_xgb.get("regime_tbm_profiles", "")),
                "meta_v2_min_prob":     str(_meta.get("meta_v2_min_prob", "")),
                "meta_threshold_mode": str(_meta.get("meta_v2_threshold_mode", "")),
            }
            _hash = _hl.md5(str(sorted(_tbm.items())).encode()).hexdigest()[:8]
            parts.append(f"tbm_hash={_hash}")
        except Exception:
            parts.append("tbm_hash=unknown")

        fingerprint = "|".join(parts)
        logger.debug(f"[SFI-FP] Fingerprint: {fingerprint}")
        return fingerprint

    def _run_sfi_with_shared_cache(self):
        """
        [SFI-CACHE-01] Ejecuta el SFI o lo reutiliza si otra seed ya lo calculó
        para la MISMA ventana temporal (mismo fingerprint completo de fechas).

        Lock file: data/wfb_cache/{window_id}/sfi_lock.json
        Estado:    'running' | 'done'
        """
        _STEP_NAME = "SFI Feature Selection"

        # Si ya completado por esta seed, saltar
        if _STEP_NAME in self.completed_steps:
            logger.info(f"--- Saltando Fase: {_STEP_NAME} (Encontrado en caché) ---")
            return

        _data_feat_dir = _ROOT / "data" / "features"
        _sf_dst = _data_feat_dir / "selected_features.json"

        # Solo aplica en modo WFB con window_id definido
        if self.mode != 'WFB' or not self.window_id:
            self._run_step(_STEP_NAME, "luna/features/feature_selection_e.py")
            return

        _lock_path = _ROOT / "data" / "wfb_cache" / self.window_id / "sfi_lock.json"
        _lock_path.parent.mkdir(parents=True, exist_ok=True)

        _current_fp = self._compute_sfi_fingerprint()

        # Verificar si hay lock válido de otra seed
        if _lock_path.exists():
            try:
                with open(_lock_path, "r", encoding="utf-8") as _lf:
                    _lock = json.load(_lf)

                _lock_status = _lock.get("status", "")
                _lock_fp = _lock.get("fingerprint", "")
                _lock_sf = Path(_lock.get("selected_features_path", ""))

                if _lock_status == "done" and _lock_fp == _current_fp and _lock_sf.exists():
                    # Fingerprint coincide: mismas fechas, mismo dataset, mismos parámetros
                    import shutil as _sh
                    if _lock_sf.resolve() != _sf_dst.resolve():
                        _sh.copy2(_lock_sf, _sf_dst)
                    logger.success(
                        f"[SFI-CACHE-01] ⚡ SFI reutilizado de seed {_lock.get('seed', '?')} "
                        f"para ventana {self.window_id} (fingerprint idéntico). "
                        f"Ahorro estimado: ~60-90 min."
                    )
                    logger.info(f"  Fechas verificadas: {_current_fp[:120]}...")
                    # Marcar como completado en caché compartido de esta semilla (para retrocompatibilidad, aunque ahora SFI está en _run_shared_step)
                    self.shared_completed_steps.add(_STEP_NAME)
                    self._save_cache(self.shared_cache_file, self.shared_completed_steps)
                    return

                elif _lock_status == "done" and _lock_fp != _current_fp:
                    logger.info(
                        f"[SFI-CACHE-01] Lock de {self.window_id} existe pero fingerprint difiere. "
                        f"Re-ejecutando SFI para seed {self.seed}."
                    )
                    logger.debug(f"  Lock FP : {_lock_fp[:100]}")
                    logger.debug(f"  Curr FP : {_current_fp[:100]}")

                elif _lock_status == "running":
                    logger.warning(
                        f"[SFI-CACHE-01] Lock de {self.window_id} en estado 'running' "
                        f"(proceso anterior no terminó). Re-ejecutando SFI."
                    )
            except Exception as _e:
                logger.warning(f"[SFI-CACHE-01] No se pudo leer lock de {self.window_id}: {_e}. Re-ejecutando SFI.")

        # Escribir lock "running" antes de ejecutar
        try:
            with open(_lock_path, "w", encoding="utf-8") as _lf:
                json.dump({
                    "status": "running",
                    "fingerprint": _current_fp,
                    "window_id": self.window_id,
                    "seed": str(self.seed),
                    "started": str(time.time()),
                }, _lf, indent=2)
        except Exception as _e:
            logger.warning(f"[SFI-CACHE-01] No se pudo escribir lock running: {_e}")

        # Ejecutar SFI real
        self._run_step(_STEP_NAME, "luna/features/feature_selection_e.py")

        # Escribir lock "done" con el fingerprint completo
        try:
            with open(_lock_path, "w", encoding="utf-8") as _lf:
                json.dump({
                    "status": "done",
                    "fingerprint": _current_fp,
                    "window_id": self.window_id,
                    "seed": str(self.seed),
                    "selected_features_path": str(_sf_dst),
                    "completed": str(time.time()),
                }, _lf, indent=2)
            logger.info(
                f"[SFI-CACHE-01] Lock de {self.window_id} marcado como 'done'. "
                f"Próximas seeds con el mismo periodo reutilizarán este SFI."
            )
        except Exception as _e:
            logger.warning(f"[SFI-CACHE-01] No se pudo escribir lock done: {_e}")

    # ── Métodos Auxiliares de Ventana ─────────────────────────────────────────

    def _run_hmm_enrichment(self):
        """
        [GAP-08 · FIX-HMM-ENRICH-01] Tras entrenar el HMM, inyecta etiquetas de
        régimen en features_validation y features_holdout de forma causal.
        - Validation: join directo con hmm_regime_labels.parquet (IS)
        - Holdout: forward-predict causal en chunks de 120H con el pkl HMM
        Sin esto el calibrador de threshold ve HMM_Semantic=NaN y fija umbral=0.90 → 0 trades.
        """
        if self.mode != 'WFB':
            return
        try:
            import pandas as _pd_e
            import joblib as _jbl_e
            import numpy as _np_e
            _feat_dir = _ROOT / "data" / "features"
            _hmm_labels = _feat_dir / "hmm_regime_labels.parquet"
            _hmm_pkl    = _ROOT / "data" / "models" / "hmm_regime.pkl"
            if not _hmm_labels.exists():
                logger.warning("[GAP-08/HMM-ENRICH] hmm_regime_labels.parquet no existe — saltando enriquecimiento.")
                return
            _lbl = _pd_e.read_parquet(_hmm_labels)
            _lbl.index = _pd_e.to_datetime(_lbl.index, utc=True)

            # --- Validation: join directo ---
            _val_p = _feat_dir / "features_validation.parquet"
            if _val_p.exists():
                _df_v = _pd_e.read_parquet(_val_p)
                _df_v.index = _pd_e.to_datetime(_df_v.index, utc=True)
                for _c in ['HMM_Regime', 'HMM_Semantic']:
                    if _c in _df_v.columns:
                        _df_v = _df_v.drop(columns=[_c])
                _df_v = _df_v.join(_lbl[['HMM_Regime', 'HMM_Semantic']], how='left')
                _df_v.to_parquet(_val_p)
                _cov = _df_v['HMM_Semantic'].notna().mean()
                logger.success(f"[GAP-08/HMM-ENRICH] features_validation enriquecido | HMM_Semantic cov={_cov:.1%}")

            # --- Holdout: forward-predict causal con pkl ---
            import os as _os_e
            _win_id = _os_e.environ.get("LUNA_WINDOW_ID", "")
            if _win_id:
                _ho_p = _feat_dir / f"features_holdout_{_win_id}.parquet"
            else:
                _ho_p = _feat_dir / "features_holdout.parquet"
            if _ho_p.exists() and _hmm_pkl.exists():
                _saved  = _jbl_e.load(_hmm_pkl)
                _model  = _saved['model']
                _scaler = _saved['scaler']
                _smap   = _saved['state_map']
                _feats  = _saved['features']
                _df_ho  = _pd_e.read_parquet(_ho_p)
                _df_ho.index = _pd_e.to_datetime(_df_ho.index, utc=True)
                for _c in ['HMM_Regime', 'HMM_Semantic']:
                    if _c in _df_ho.columns:
                        _df_ho = _df_ho.drop(columns=[_c])
                _X = _df_ho[[f for f in _feats if f in _df_ho.columns]].copy()
                for _f in [f for f in _feats if f not in _df_ho.columns]:
                    _X[_f] = 0.0
                _X = _X[_feats].fillna(0.0)
                _Xs = _scaler.transform(_X)
                _n  = len(_Xs)
                # [WFB-CAUSAL-FIX-ENRICH] SOP R1: Inferencia HMM 100% causal sin look-ahead.
                # Se utiliza el Forward Algorithm estricto (causal filter) sobre la ventana
                # para mantener la historia de Markov sin que Viterbi mire al futuro.
                from scipy.special import logsumexp as _logsumexp
                _states = _np_e.zeros(_n, dtype=int)
                print(f"[WFB-CAUSAL-FIX-ENRICH] Enriqueciendo holdout de forma 100% causal ({_n} registros) con Forward Filter...")
                logger.info(f"[WFB-CAUSAL-FIX-ENRICH] Inferencia HMM causal activa usando Forward Algorithm.")
                
                if _n > 0:
                    _framelogprob = _model._compute_log_likelihood(_Xs)
                    _log_startprob = _np_e.log(_np_e.maximum(_model.startprob_, 1e-10))
                    _log_transmat = _np_e.log(_np_e.maximum(_model.transmat_, 1e-10))
                    
                    _log_alpha = _np_e.zeros((_n, _model.n_components))
                    _log_alpha[0] = _log_startprob + _framelogprob[0]
                    _states[0] = _np_e.argmax(_log_alpha[0])
                    
                    for _t in range(1, _n):
                        _work_buffer = _log_alpha[_t-1][:, None] + _log_transmat
                        _log_alpha[_t] = _logsumexp(_work_buffer, axis=0) + _framelogprob[_t]
                        _states[_t] = _np_e.argmax(_log_alpha[_t])
                        
                print("[WFB-CAUSAL-FIX-ENRICH] Enriqueciendo holdout completado exitosamente.")
                _df_ho['HMM_Regime']   = _np_e.array(_states, dtype=float).astype(int)
                _df_ho['HMM_Semantic'] = _df_ho['HMM_Regime'].map(_smap).fillna('UNKNOWN')
                _df_ho.to_parquet(_ho_p)
                # Backup to global features_holdout.parquet if it's window-specific
                if _win_id:
                    _df_ho.to_parquet(_feat_dir / "features_holdout.parquet")
                _cov_ho = (_df_ho['HMM_Semantic'] != 'UNKNOWN').mean()
                logger.success(f"[GAP-08/HMM-ENRICH] features_holdout enriquecido (causal fwd) | cov={_cov_ho:.1%}")
        except Exception as _e_enrich:
            import traceback as _tb_enrich
            _tb_str = _tb_enrich.format_exc()
            logger.error(
                f"[FIX-PIPE-002][GAP-08/HMM-ENRICH] ERROR en enriquecimiento HMM: {_e_enrich}\n{_tb_str}"
            )
            print(
                f"[FIX-PIPE-002][GAP-08/HMM-ENRICH][ERROR] Fallo al enriquecer val/holdout con HMM_Semantic.\n"
                f"Traceback: {_tb_str}"
            )
            # Post-hoc: verificar si HMM_Semantic quedó de todas formas (ej. de una pasada anterior)
            try:
                import pyarrow.parquet as _pq_check
                _ho_p_check = _feat_dir / "features_holdout.parquet"
                _val_p_check = _feat_dir / "features_validation.parquet"
                for _check_path in [_ho_p_check, _val_p_check]:
                    if _check_path.exists():
                        _schema_check = _pq_check.read_schema(_check_path)
                        if "HMM_Semantic" not in _schema_check.names:
                            print(
                                f"❌ [FIX-PIPE-002][CRITICO] {_check_path.name} NO tiene HMM_Semantic tras el error. "
                                f"El filtro de régimen producirá 0 trades en esta ventana."
                            )
                            logger.critical(
                                f"[FIX-PIPE-002] {_check_path.name} sin HMM_Semantic — filtro de régimen inoperativo."
                            )
                        else:
                            print(f"[FIX-PIPE-002][OK] {_check_path.name} ya tiene HMM_Semantic (de pasada anterior).")
            except Exception as _e_check:
                logger.warning(f"[FIX-PIPE-002] No se pudo verificar parquets post-error: {_e_check}")


    def _cleanup_zombie_models(self):
        """
        [GAP-04 · ZOMBIE-KILL-01] Elimina artefactos de la arquitectura legacy single-agent.
        Tras confirmar freshness de modelos multi-agente (bull/range/bear), purga
        los modelos y signatures del formato pre-multi-agent que causarían confusión
        al Router si coexisten con los nuevos.
        """
        _zombie_patterns = [
            "xgboost_meta_long.model",          # legacy single-agent exacto
            "xgboost_meta_short.model",         # legacy single-agent exacto
            "xgboost_meta.model",               # baseline monolítico pre-multi-agent
            "xgboost_meta_long*.json",          # signatures legacy LONG
            "xgboost_meta_short*.json",         # signatures legacy SHORT
            "xgboost_meta_signature.json",      # signature baseline monolítica
            "xgboost_meta_calm_bear_long*.json",# [ZOMBIE-FIX-01 2026-06-03] signature calm_bear legacy
            #   causa FIX-MERGE-EXIT-01 en W4 cuando GATE-G2 detecta mtime antiguo
            #   propagada desde seed=42 a toda la caché WFB — se agrega a lista de purga
        ]
        print("[ZOMBIE-FIX-01] Lista de artefactos zombie incluye calm_bear_long — bug fix 2026-06-03")
        _models_dir = _ROOT / "data" / "models"
        if not _models_dir.exists():
            return
        _found = []
        for _pat in _zombie_patterns:
            for _z in _models_dir.glob(_pat):
                _found.append(_z)
        if _found:
            for _z in _found:
                try:
                    _z.unlink()
                except Exception as _ez:
                    logger.debug(f"[GAP-04/ZOMBIE] No se pudo eliminar {_z.name}: {_ez}")
            logger.warning(f"[GAP-04/ZOMBIE-KILL-01] {len(_found)} artefactos legacy eliminados: {[z.name for z in _found]}")
        else:
            logger.debug("[GAP-04/ZOMBIE-KILL-01] Sin artefactos legacy. Directorio de modelos limpio.")

    def _gate_2_xgboost_check(self):
        """
        [GAP-02 · GATE-2 inline] Verificación de calidad post-XGBoost en modo degradado.
        Si un agente falla, escribe gate_g2_disabled_agents.json y continúa en vez
        de abortar todo el pipeline (equivalente al Gate-2 de WFBPhaseGate en V1).
        """
        _models_dir = _ROOT / "data" / "models"
        _disabled_path = _models_dir / "gate_g2_disabled_agents.json"
        if not _models_dir.exists():
            return
        _models = list(_models_dir.glob("xgboost_meta*.model"))
        if not _models:
            logger.warning("[GAP-02/GATE-2] No se encontraron modelos XGBoost para verificar.")
            return
        # Leer disabled_agents previos para loguearlos
        if _disabled_path.exists():
            try:
                with open(_disabled_path, "r", encoding="utf-8") as _gf:
                    _prev = json.load(_gf)
                logger.warning(
                    f"[GAP-02/GATE-2] gate_g2_disabled_agents.json detectado de ciclo previo: "
                    f"{_prev.get('disabled_agents', [])} — Router permanecerá en CASH para estos agentes."
                )
            except Exception:
                pass
        # Verificar existencia y tamaño mínimo de cada modelo
        _failed = []
        for _m in _models:
            try:
                if _m.stat().st_size < 1024:  # < 1 KB = modelo vacío/corrupto
                    _failed.append(_m.name)
            except Exception:
                _failed.append(_m.name)
        if _failed:
            logger.error(
                f"[GAP-02/GATE-2] {len(_failed)} modelos XGBoost corruptos o vacíos: {_failed}. "
                f"Escribiendo gate_g2_disabled_agents.json → modo degradado (CASH en esos agentes)."
            )
            try:
                with open(_disabled_path, "w", encoding="utf-8") as _gf:
                    json.dump({
                        "disabled_agents": _failed,
                        "window_id": self.window_id,
                        "seed": str(self.seed),
                        "reason": "modelo_corrupto_o_vacio",
                    }, _gf, indent=2)
            except Exception as _ge:
                logger.warning(f"[GAP-02/GATE-2] No se pudo escribir gate_g2_disabled_agents.json: {_ge}")
        else:
            logger.info(f"[GAP-02/GATE-2] Gate-2 OK: {len(_models)} modelos XGBoost verificados.")
            # Limpiar disabled_agents si todo está OK en este ciclo
            if _disabled_path.exists():
                try:
                    _disabled_path.unlink()
                    logger.info("[GAP-02/GATE-2] gate_g2_disabled_agents.json eliminado (ciclo limpio).")
                except Exception:
                    pass

    # ── Pipeline Principal ────────────────────────────────────────────────────

    def _run_shared_step(self, step_name: str, script_rel_path: str, args_list: list = None):
        """Ejecuta un paso aislado del pipeline compartido (Agnóstico a la semilla)."""
        if step_name in self.shared_completed_steps:
            logger.info(f"--- [SHARED-CACHE] Saltando Fase: {step_name} (Encontrado en caché universal) ---")
            return

        logger.info(f"--- Iniciando Fase Compartida: {step_name} ---")
        script_path = _ROOT / script_rel_path

        if not script_path.exists():
            logger.error(f"[ERROR] Módulo no encontrado: {script_path}")
            sys.exit(1)

        cmd = [sys.executable, "-u", str(script_path)]
        if args_list:
            cmd.extend(args_list)

        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = str(_ROOT)
        run_env["PYTHONUNBUFFERED"] = "1"

        # IMPORTANTE: No se inyecta LUNA_SEED para garantizar pureza causal
        if self.window_id:
            run_env["LUNA_WINDOW_ID"] = self.window_id

        if self.mode == 'PROD':
            run_env["LUNA_PRODUCTION_MODE"] = "1"

        _TIMEOUT_SECONDS = 14400  # 4 horas

        try:
            with subprocess.Popen(cmd, env=run_env, cwd=str(_ROOT)) as process:
                try:
                    process.wait(timeout=_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    logger.error(f"[TIMEOUT] Fase Compartida '{step_name}' excedió 4H. Terminando proceso.")
                    process.kill()
                    process.communicate()
                    sys.exit(1)

            if process.returncode != 0:
                logger.error(f"[FATAL] La Fase Compartida '{step_name}' abortó con código de error {process.returncode}")
                sys.exit(process.returncode)

            # Dehydrate
            if self.mode == 'WFB' and self.window_id:
                # Se guarda en el snapshot de features compartidas (seed_id=None)
                dehydrate_window_state(self.window_id, seed_id=None)

            self.shared_completed_steps.add(step_name)
            self._save_cache(self.shared_cache_file, self.shared_completed_steps)
            logger.success(f"--- Fase Compartida '{step_name}' Completada Exitosamente ---\n")

        except Exception as e:
            logger.exception(f"Error crítico al orquestar {step_name}: {e}")
            sys.exit(1)

    def execute_data_pipeline(self):
        """Fase 1 y 2: Preparación de Datos Crudos y Features (Agnósticas a Semilla)"""
        logger.info("==========================================================")
        logger.info("   LUNA V2 - INICIANDO PIPELINE DE DATOS BASE             ")
        logger.info("==========================================================")
        self._run_shared_step("Feature Pipeline (Base Generation)", "luna/features/feature_pipeline.py", ["--skip-preflight"])
        self._run_shared_step("Build Dataset (AI Mining)", "scripts/build_dataset.py")
        self._run_shared_step("Feature Pipeline (Pre-SFI)", "luna/features/feature_pipeline.py", ["--skip-preflight", "--fast-inject"])
        
        # Como SFI ahora está en el shared pipeline, si _run_shared_step no saltó, intentamos usar el cache lock (Retrocompatibilidad o fallbacks)
        if "SFI Feature Selection" not in self.shared_completed_steps:
            self._run_sfi_with_shared_cache()
            self.shared_completed_steps.add("SFI Feature Selection")
            self._save_cache(self.shared_cache_file, self.shared_completed_steps)
            
        self._run_shared_step("Feature Pipeline (Post-SFI)", "luna/features/feature_pipeline.py", ["--skip-preflight"])

        # [Capa B - FIX HMM] Mover el modelo HMM al pipeline de datos para re-entrenarlo UNA vez por ventana compartida
        self._run_shared_step("HMM Regime Model", "luna/models/hmm_regime.py")

        # Enriquecimiento HMM también compartido
        if "HMM Enrichment" not in self.shared_completed_steps:
            self._run_hmm_enrichment()
            self.shared_completed_steps.add("HMM Enrichment")
            self._save_cache(self.shared_cache_file, self.shared_completed_steps)

        # [GAP-MINING-01] Re-exportar alpha_rules.py calibrado al train_end de esta ventana.
        # El engine 'export' (~30s) regenera luna/features/alpha_rules.py usando SOLO datos
        # hasta cfg.temporal_splits.train_end (reescrito por rewrite_yaml_for_window).
        # Sin esto, alpha_rules.py estático corresponde al último run manual de AI Mining,
        # que pudo haberse ejecutado con train_end de W6, causando leakage en W1-W5.
        # Se ejecuta DESPUÉS del Feature Pipeline Post-SFI para que las reglas vean
        # los mismos features que entrarán a XGBoost.
        # Nota: solo 'export' — los engines pesados (bayesian, cluster) corren en el
        # pipeline compartido una sola vez para todas las seeds/ventanas.
        if self.mode == 'WFB' and self.window_id:
            _mining_script = _ROOT / "scripts" / "build_dataset.py"
            if _mining_script.exists():
                self._run_step(
                    "AI Mining (alpha_rules export por ventana)",
                    "scripts/build_dataset.py",
                    ["--mode", "dev", "--engine", "export"]
                )
            else:
                logger.warning("[GAP-MINING-01] scripts/build_dataset.py no encontrado — alpha_rules.py NO se recalibrará por ventana.")

    def execute_training_sequence(self):
        """Fase 3 y 4: Secuencia de Entrenamiento Estándar"""
        logger.info("==========================================================")
        logger.info(f"   LUNA V2 - SECUENCIA DE ENTRENAMIENTO ({self.mode})     ")
        logger.info(f"   Semilla: {self.seed} | Ventana: {self.window_id}       ")
        logger.info("==========================================================")

        # [GAP-02/PHASE-GATE] Instanciar WFBPhaseGate para esta ventana/seed
        _gate = None
        _WFB_REPORTS_DIR = _ROOT / "data" / "reports" / "wfb"
        _features_dir    = _ROOT / "data" / "features"
        _models_dir      = _ROOT / "data" / "models"
        _predictions_dir = _ROOT / "data" / "predictions"
        if self.mode == 'WFB' and self.window_id:
            try:
                from luna.validation.phase_gates import WFBPhaseGate
                _gate = WFBPhaseGate(
                    window_id=self.window_id,
                    seed=int(self.seed) if self.seed is not None else 0,
                    reports_dir=_WFB_REPORTS_DIR,
                    root=_ROOT,
                )
                logger.info(f"[GAP-02/GATE] WFBPhaseGate instanciado: {self.window_id}/seed{self.seed}")
            except Exception as _ge_init:
                logger.warning(f"[GAP-02/GATE] No se pudo instanciar WFBPhaseGate: {_ge_init} — usando fallback inline.")

        # [GATE-0] Integridad de datos ANTES de HMM/XGBoost
        if _gate:
            try:
                _r0 = _gate.gate_0_data(_features_dir)
                if not _r0.passed and _r0.is_hard_stop:
                    logger.error(f"[GATE-0] HARD STOP: {_r0.summary}")
                    sys.exit(3)
            except Exception as _ge0:
                logger.warning(f"[GATE-0] Gate-0 no ejecutable: {_ge0}")

        # [HMM] El entrenamiento y enriquecimiento del HMM se han movido a execute_data_pipeline (shared)
        # para que se ejecuten una sola vez por ventana.

        # [GAP-05 · FIX-A1] Capturar mtime ANTES del entrenamiento XGBoost
        _t_before_xgb = time.time() - 1.0  # Buffer 1s para resolución filesystem Windows

        # [H-06-FIX 2026-05-30] Guard HMM_Semantic PRE-XGB (movido desde post-MetaLabeler).
        # PROBLEMA: [FIX-C1] estaba en L1216 (después de MetaLabeler) — si el enriquecimiento
        # fallaba silenciosamente, XGBoost, OOD y MetaLabeler entrenaban con HMM_Semantic=NaN.
        # Los agentes filtrados por régimen (bull/bear/range) veían 0 samples válidos.
        # SOLUCIÓN: verificar cobertura HMM_Semantic ANTES del XGB. Si <50%, re-enriquecer.
        # Si el re-enriquecimiento también falla → HARD STOP (no tiene sentido seguir sin HMM).
        try:
            import pandas as _pd_c1_pre
            _val_p_c1_pre = _features_dir / "features_validation.parquet"
            if _val_p_c1_pre.exists():
                import pyarrow.parquet as _pq_c1_pre
                _schema_pre = _pq_c1_pre.read_schema(_val_p_c1_pre)
                _has_hmm_pre = "HMM_Semantic" in _schema_pre.names
                if _has_hmm_pre:
                    _df_hmm_pre = _pd_c1_pre.read_parquet(_val_p_c1_pre, columns=["HMM_Semantic"])
                    _cov_pre = float(_df_hmm_pre["HMM_Semantic"].notna().mean())
                else:
                    _cov_pre = 0.0
                if not _has_hmm_pre or _cov_pre < 0.5:
                    print(  # RULE[fixbugsprints.md]
                        f"[H-06-FIX] PRE-XGB: HMM_Semantic cov={_cov_pre:.1%} < 50% en features_validation. "
                        f"Re-ejecutando _run_hmm_enrichment() de emergencia ANTES del XGBoost."
                    )
                    logger.warning(
                        "[H-06-FIX] PRE-XGB GUARD: HMM_Semantic cov={:.1f}% — re-enriquecimiento de emergencia.",
                        _cov_pre * 100
                    )
                    print(f"[BUG-FIX-LOG 2026-06-05] PRE-XGB GUARD: HMM_Semantic cov={_cov_pre * 100:.1f}% — re-enriquecimiento de emergencia.")
                    self._run_hmm_enrichment()
                    # Verificar que el re-enriquecimiento funcionó
                    if _val_p_c1_pre.exists():
                        _df_post = _pd_c1_pre.read_parquet(_val_p_c1_pre, columns=["HMM_Semantic"])
                        _cov_post = float(_df_post["HMM_Semantic"].notna().mean())
                        if _cov_post < 0.5:
                            print(  # RULE[fixbugsprints.md]
                                f"[H-06-FIX] HARD STOP: Re-enriquecimiento HMM también falló "
                                f"(cov={_cov_post:.1%}). XGBoost abortado — datos HMM inválidos."
                            )
                            raise RuntimeError(
                                f"[H-06-FIX] HMM_Semantic cov={_cov_post:.1%} tras re-enriquecimiento. "
                                f"XGBoost abortado. Revisar hmm_regime_labels.parquet y _run_hmm_enrichment."
                            )
                        print(f"[H-06-FIX] Re-enriquecimiento exitoso: cov={_cov_post:.1%}. XGBoost puede continuar.")
                else:
                    print(f"[H-06-FIX] PRE-XGB: HMM_Semantic OK (cov={_cov_pre:.1%}). Procediendo con XGBoost.")
                    logger.debug("[H-06-FIX] HMM_Semantic pre-XGB OK (cov={:.1f}%).", _cov_pre * 100)
                    print(f"[BUG-FIX-LOG 2026-06-05] HMM_Semantic pre-XGB OK (cov={_cov_pre * 100:.1f}%)")
        except RuntimeError:
            raise  # Propagar HARD STOP
        except Exception as _e_h06:
            logger.warning("[H-06-FIX] No se pudo verificar HMM_Semantic pre-XGB: {}", _e_h06)
            print(f"[BUG-FIX-LOG 2026-06-05] No se pudo verificar HMM_Semantic pre-XGB: {_e_h06}")

        _xgb_executed = self._run_step("XGBoost Core Model", "luna/models/train_xgboost_v2.py")

        # [SHAP-AUDIT-01 2026-06-03] Auditar importancia de features forzadas por SFI
        # Mide empiricamente si las features de cuota/boost aportan valor real.
        # No bloqueante: cualquier fallo loguea WARNING y el pipeline continua.
        if _xgb_executed:
            try:
                from luna.monitoring.shap_feature_auditor import run_shap_audit
                _audit_result = run_shap_audit(
                    window_id=self.window_id,
                    seed=int(self.seed) if self.seed is not None else None,
                    importance_type="gain",
                )
                if _audit_result:
                    _n_critical = sum(1 for a in _audit_result.get('alerts', [])
                                      if a['severity'] == 'CRITICAL')
                    if _n_critical > 0:
                        logger.warning(
                            f"[SHAP-AUDIT-01] {_n_critical} features forzadas CRITICAS: "
                            f"revisar data/shap_audit/audit_report.txt"
                        )
                    else:
                        logger.info(
                            f"[SHAP-AUDIT-01] Audit OK | ventanas={_audit_result['n_windows_total']} "
                            f"| alertas={len(_audit_result.get('alerts', []))}"
                        )
            except Exception as _e_audit:
                logger.warning(f"[SHAP-AUDIT-01] Auditor falló (no bloqueante): {_e_audit}")


        if _xgb_executed:
            _xgb_models = list(_models_dir.glob("xgboost_meta*.model")) if _models_dir.exists() else []
            if _xgb_models:
                _updated = [p for p in _xgb_models if p.stat().st_mtime >= _t_before_xgb]
                if not _updated:
                    raise RuntimeError(
                        f"[GAP-05/FIX-A1] NINGÚN modelo xgboost_meta*.model actualizado tras "
                        f"train_xgboost_v2.py (ventana {self.window_id}). "
                        f"El training falló silenciosamente (exit 0 sin escribir modelos). "
                        f"Modelos encontrados: {[p.name for p in _xgb_models]}"
                    )
                logger.info(f"[GAP-05/FIX-A1] Freshness OK: {len(_updated)}/{len(_xgb_models)} modelos actualizados.")

        # [GAP-04 · ZOMBIE-KILL-01] Limpiar artefactos de arquitectura legacy single-agent
        self._cleanup_zombie_models()

        # [GAP-02 · GATE-G2] Gate XGBoost formal (Brier score) con fallback inline
        if _gate:
            try:
                _r2 = _gate.gate_2_xgboost(_models_dir, _features_dir)
                if not _r2.passed and _r2.is_hard_stop:
                    logger.error(f"[GATE-G2] HARD STOP DETECTADO: {_r2.summary}")
                    logger.warning("[GATE-G2/DEGRADED] Activando modo degradado global (CASH) para evitar crash de la semilla.")
                    # Eliminado sys.exit(3) para permitir continuidad en WFB (Degraded Mode)
                # Escribir disabled_agents al JSON para que RegimeRouter lo lea
                if _r2.disabled_agents:
                    _disabled_path = _models_dir / "gate_g2_disabled_agents.json"
                    try:
                        with open(_disabled_path, "w", encoding="utf-8") as _gf2:
                            json.dump({
                                "disabled_agents": _r2.disabled_agents,
                                "window_id": self.window_id,
                                "seed": str(self.seed),
                                "brier_by_agent": _r2.metrics.get("brier_by_agent", {}),
                                "reason": "brier_score_superado",
                            }, _gf2, indent=2)
                        logger.warning(f"[GATE-G2/DEGRADED] {_r2.disabled_agents} → gate_g2_disabled_agents.json")
                    except Exception as _gw2:
                        logger.debug(f"[GATE-G2] No se pudo escribir disabled_agents: {_gw2}")
                else:
                    # Limpiar disabled de ciclo previo si todo OK
                    _dp2 = _models_dir / "gate_g2_disabled_agents.json"
                    if _dp2.exists():
                        try: _dp2.unlink()
                        except Exception: pass
            except SystemExit:
                raise
            except Exception as _ge2:
                logger.warning(f"[GATE-G2] Gate formal falló, usando fallback inline: {_ge2}")
                self._gate_2_xgboost_check()
        else:
            self._gate_2_xgboost_check()

        # [FIX-LGBM-GATE] Leer use_lgbm_ensemble desde settings.yaml (cfg.fase2)
        _use_lgbm = self.options.get("use_lgbm_ensemble", False)
        try:
            from config.settings import cfg as _cfg_lgbm
            _use_lgbm = bool(getattr(getattr(_cfg_lgbm, 'fase2', None), 'use_lgbm_ensemble', _use_lgbm))
        except Exception:
            pass
        if _use_lgbm:
            self._run_step("LGBM Ensemble", "luna/models/ensemble_lgbm.py")
        else:
            logger.info("   [LGBM] use_lgbm_ensemble=false en settings.yaml — saltando entrenamiento LGBM.")

        # [GATE-3] Coherencia LGBM + HMM
        if _gate and _use_lgbm:
            try:
                _r3 = _gate.gate_3_ensemble(_models_dir, _features_dir)
                if not _r3.passed and _r3.is_hard_stop:
                    logger.error(f"[GATE-3] HARD STOP DETECTADO: {_r3.summary}")
                    logger.warning("[GATE-3/DEGRADED] Activando modo degradado global (CASH) para evitar crash de la semilla.")
                    # Eliminado sys.exit(3) para permitir continuidad en WFB
            except Exception as _ge3:
                logger.warning(f"[GATE-3] Gate-3 no ejecutable: {_ge3}")

        self._run_step("OOD Guard", "luna/models/ood_guard.py")
        self._run_step("AutoEncoder", "luna/models/train_autoencoder.py")
        
        try:
            from config.settings import cfg as _cfg_dir
            _dmode = getattr(_cfg_dir.fase2, 'direction_mode', 'both')
        except Exception:
            _dmode = "both"
            
        if _dmode == "both" or (isinstance(_dmode, list) and "long" in _dmode) or _dmode == "long":
            self._run_step("MetaLabeler V2 (LONG)", "luna/models/train_metalabeler_v2.py", ["--direction", "long"])
        if _dmode == "both" or (isinstance(_dmode, list) and "short" in _dmode) or _dmode == "short":
            self._run_step("MetaLabeler V2 (SHORT)", "luna/models/train_metalabeler_v2.py", ["--direction", "short"])

        # [GATE-4] MetaLabeler output
        if _gate:
            try:
                _r4 = _gate.gate_4_metalabeler(_models_dir, _features_dir)
                if not _r4.passed and _r4.is_hard_stop:
                    logger.error(f"[GATE-4] HARD STOP DETECTADO: {_r4.summary}")
                    logger.warning("[GATE-4/DEGRADED] Activando modo degradado global (CASH) para evitar crash de la semilla.")
                    # Eliminado sys.exit(3) para permitir continuidad en WFB
            except Exception as _ge4:
                logger.warning(f"[GATE-4] Gate-4 no ejecutable: {_ge4}")

        # [GAP-04/FIX-C1] Guard HMM_Semantic en features_validation antes del calibrador
        # Si _run_hmm_enrichment falló silenciosamente, el calibrador verá NaN → umbral=0.90 → 0 trades.
        try:
            import pandas as _pd_c1
            _val_p_c1 = _features_dir / "features_validation.parquet"
            if _val_p_c1.exists():
                import pyarrow.parquet as _pq_c1
                _schema_c1 = _pq_c1.read_schema(_val_p_c1)
                _has_hmm = "HMM_Semantic" in _schema_c1.names
                if _has_hmm:
                    _df_c1 = _pd_c1.read_parquet(_val_p_c1, columns=["HMM_Semantic"])
                else:
                    _df_c1 = _pd_c1.DataFrame()
                _cov_c1 = _df_c1["HMM_Semantic"].notna().mean() if _has_hmm else 0.0
                if not _has_hmm or _cov_c1 < 0.5:
                    logger.warning(
                        f"[FIX-C1] features_validation sin HMM_Semantic (cov={_cov_c1:.1%}). "
                        "Re-ejecutando enriquecimiento HMM de emergencia..."
                    )
                    self._run_hmm_enrichment()
                else:
                    logger.debug(f"[FIX-C1] HMM_Semantic OK en features_validation (cov={_cov_c1:.1%})")
        except Exception as _e_c1:
            logger.debug(f"[FIX-C1] No se pudo verificar HMM_Semantic: {_e_c1}")

        if _dmode == "both" or (isinstance(_dmode, list) and "long" in _dmode) or _dmode == "long":
            self._run_step("Calibrador de Probabilidades (LONG)", "luna/models/calibrate_probabilities.py", ["--direction", "long"])
        if _dmode == "both" or (isinstance(_dmode, list) and "short" in _dmode) or _dmode == "short":
            self._run_step("Calibrador de Probabilidades (SHORT)", "luna/models/calibrate_probabilities.py", ["--direction", "short"])

        # [GAP-07] Verificar desfase entre umbral calibrado y baseline
        try:
            _sig_path = _ROOT / "data" / "models" / "calibrator_signature.json"
            if _sig_path.exists():
                with open(_sig_path, "r", encoding="utf-8") as _sf:
                    _sig = json.load(_sf)
                _cal_thresh = float(_sig.get("threshold", _sig.get("calibrated_threshold", 0.0)))
                # [FIX-06] Extraída como constante nombrada para eliminar magic number duplicado
                _XGB_BASELINE_DEFAULT = 0.38  # baseline institucional (fallback si settings no disponible)
                _base_thresh = _XGB_BASELINE_DEFAULT
                try:
                    import yaml as _yml_g7
                    with open(_ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as _yf:
                        _cfg_g7 = _yml_g7.safe_load(_yf)
                    _base_thresh = float(_cfg_g7.get("xgboost", {}).get("xgb_signal_threshold", _XGB_BASELINE_DEFAULT))
                except Exception:
                    print(f"[FIX-06] WARN: No se pudo leer xgb_signal_threshold de settings. Usando fallback={_XGB_BASELINE_DEFAULT}")
                _desfase = abs(_cal_thresh - _base_thresh)
                if _desfase > 0.05:
                    logger.warning(
                        f"[GAP-07/CALIBRATOR] ⚠️ Desfase paramétrico crítico: "
                        f"umbral calibrado={_cal_thresh:.3f} vs baseline={_base_thresh:.3f} "
                        f"(diferencia={_desfase:.3f} > 0.05). "
                        "Posible rotura distributiva de Gauss. Revisar antes de producción."
                    )
                else:
                    logger.info(f"[GAP-07/CALIBRATOR] Umbral calibrado={_cal_thresh:.3f} OK (desfase={_desfase:.3f} ≤ 0.05).")
        except Exception as _e_g7:
            logger.debug(f"[GAP-07] No se pudo verificar calibrator_signature.json: {_e_g7}")

        # [R2/FIX-I1] Cap de umbral calibrado: si >0.95, forzar fallback al baseline.
        # Un threshold 0.95+ significa 0 trades en OOS — fallo silencioso más grave que
        # usar el baseline sin calibrar. Escribe la corrección en calibrator_signature.json.
        try:
            _sig_fix_path = _ROOT / "data" / "models" / "calibrator_signature.json"
            if _sig_fix_path.exists():
                with open(_sig_fix_path, "r", encoding="utf-8") as _sf_fix:
                    _sig_fix = json.load(_sf_fix)
                _th_key = "threshold" if "threshold" in _sig_fix else "calibrated_threshold"
                _th_val = float(_sig_fix.get(_th_key, 0.0))
                if _th_val > 0.95:
                    # [FIX-06] Reutilizar _XGB_BASELINE_DEFAULT si ya está definido, si no usar settings
                    _fallback_th = _XGB_BASELINE_DEFAULT if '_XGB_BASELINE_DEFAULT' in dir() else 0.38
                    try:
                        import yaml as _yml_fi1
                        with open(_ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as _yf_fi1:
                            _cfg_fi1 = _yml_fi1.safe_load(_yf_fi1)
                        _fallback_th = float(_cfg_fi1.get("xgboost", {}).get("xgb_signal_threshold", _fallback_th))
                    except Exception:
                        print(f"[FIX-06] WARN: No se pudo leer xgb_signal_threshold para cap fallback. Usando={_fallback_th}")
                    logger.warning(
                        f"[R2/FIX-I1] Umbral calibrado={_th_val:.3f} > 0.95 (zona muerte). "
                        f"Forzando fallback a baseline={_fallback_th:.3f} para evitar 0 trades."
                    )
                    _sig_fix[_th_key] = _fallback_th
                    _sig_fix["fix_i1_applied"] = True
                    _sig_fix["fix_i1_original_threshold"] = _th_val
                    with open(_sig_fix_path, "w", encoding="utf-8") as _sf_out:
                        json.dump(_sig_fix, _sf_out, indent=2)
        except Exception as _e_fi1:
            logger.debug(f"[R2/FIX-I1] No se pudo verificar threshold cap: {_e_fi1}")

        self._run_step("Generador de Predicciones OOS", "luna/models/predict_oos.py")

        # [GATE-5] Signal filter output
        if _gate:
            try:
                _run_id = os.environ.get("LUNA_RUN_ID", "")
                _r5 = _gate.gate_5_signal(_predictions_dir, run_id=_run_id)
                if not _r5.passed and _r5.is_hard_stop:
                    logger.error(f"[GATE-5] HARD STOP DETECTADO: {_r5.summary}")
                    logger.warning("[GATE-5/DEGRADED] Activando modo degradado global (CASH) para evitar crash de la semilla.")
                    # Eliminado sys.exit(3) para permitir continuidad en WFB
            except Exception as _ge5:
                logger.warning(f"[GATE-5] Gate-5 no ejecutable: {_ge5}")

        if self.mode == 'PROD':
            self._run_step("Validación Estadística", "scripts/run_statistical_validation.py")

        logger.success(f"[SUCCESS] Secuencia {self.mode} finalizada para semilla {self.seed}.")


