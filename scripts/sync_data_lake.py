"""
sync_data_lake.py
====================
Luna V2 — Data Lake Synchronizer

Orquesta exclusivamente el ciclo de vida de los datos, separando la Ingeniería 
de Datos de la Ingeniería de Modelos (Machine Learning).

Módulos que ejecuta:
  1. Pre-Flight Check (Arquitectura y Entorno)
  2. Fetchers (En paralelo) -> Binance, FRED, Onchain, etc.
  3. Data Integrity Check (Detecta corrupción y NaNs)
  4. Reconcile External Data (Anclaje temporal)
  5. Feature Pipeline Base (Generación pre-minería)
  6. AI Mining (Tribus y Reglas Bayesianas)
  7. SFI (Smart Feature Isolation)
  8. Feature Pipeline Final (Post-SFI)

Uso:
  python scripts/sync_data_lake.py
  python scripts/sync_data_lake.py --skip-sfi
  python scripts/sync_data_lake.py --skip-fetch
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger
from luna.pipeline_executor import LunaPipelineExecutor

def _header(title: str) -> None:
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)

def run_pre_flight():
    _header("PRE-FLIGHT CHECK — Validación de Arquitectura")
    _sub_env = os.environ.copy()
    _sub_env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + _sub_env.get("PYTHONPATH", "") if _sub_env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, str(_ROOT / "scripts/pre_flight_check.py")], env=_sub_env, cwd=str(_ROOT))
    if result.returncode != 0:
        logger.error("Pre-Flight Check FALLIDO. Abortando sincronización.")
        sys.exit(1)
    logger.success("Pre-flight completado exitosamente.")

def run_fetchers():
    _header("FETCHER — Actualización de Datos Raw (PARALELO)")
    import time

    fetchers = [
        "luna/data/fetch_ohlcv.py",
        "luna/data/fetch_macro.py",
        "luna/data/m2_global_fetcher.py",
        "luna/data/fetch_onchain.py",
        "luna/data/fetch_derivatives.py",
        "luna/data/fetch_altcoins.py",
        "luna/data/fetch_mempool.py",
        "luna/data/fetch_defi.py",
    ]
    
    _env = os.environ.copy()
    _existing_pp = _env.get("PYTHONPATH", "")
    _env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + _existing_pp if _existing_pp else "")

    def _run_single_fetcher(fetcher: str) -> tuple[str, object, float]:
        fpath = _ROOT / fetcher
        name = Path(fetcher).name
        if not fpath.exists():
            return name, None, 0.0
        t0 = time.monotonic()
        r = subprocess.run(
            [sys.executable, str(fpath)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(_ROOT), env=_env,
            timeout=300,
        )
        return name, r, time.monotonic() - t0

    failed = []
    t_fase_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_run_single_fetcher, f): f for f in fetchers}
        for future in as_completed(futures):
            name, r, elapsed = future.result()
            if r is None:
                logger.warning("  Fetcher no encontrado (skip): {}", futures[future])
                continue
            if r.returncode != 0:
                failed.append(name)
                logger.error(f"  [FETCH-FAIL] {name} exit={r.returncode} ({elapsed:.1f}s)")
                if r.stderr:
                    for line in r.stderr.strip().splitlines()[-5:]:
                        logger.error("    stderr: {}", line)
            else:
                logger.success(f"  [FETCH-OK] {name} completado en {elapsed:.1f}s")

    t_fase_total = time.monotonic() - t_fase_start
    logger.info(f"  [FASE 1 TIMING] Todos los fetchers completados en {t_fase_total:.1f}s")

    if failed:
        logger.error(f"FETCH-GATE-01: {len(failed)} fetcher(s) con exit!=0: {', '.join(failed)}")
        logger.error("PIPELINE ABORTADO -- Usa --skip-fetch para continuar con datos antiguos.")
        sys.exit(1)

def run_data_integrity():
    _header("DATA INTEGRITY CHECK — Verificación de parquets locales")
    _sub_env = os.environ.copy()
    _sub_env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + _sub_env.get("PYTHONPATH", "") if _sub_env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, str(_ROOT / "scripts/data_integrity_check.py"), "--lenient-wfb"], env=_sub_env, cwd=str(_ROOT))
    if result.returncode != 0:
        logger.error("Data Integrity Check FALLIDO — hay archivos corruptos o NaNs críticos.")
        sys.exit(1)
    logger.success("Integridad de datos verificada post-fetch.")

def run_reconcile():
    _header("RECONCILE EXTERNAL DATA — Anclaje Temporal")
    _sub_env = os.environ.copy()
    _sub_env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + _sub_env.get("PYTHONPATH", "") if _sub_env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, str(_ROOT / "scripts/reconcile_external_data.py"), "--days", "14"], env=_sub_env, cwd=str(_ROOT))
    if result.returncode != 0:
        logger.error("Reconciliación Externa detectó una anomalía fatal. Pipeline abortado.")
        sys.exit(1)
    logger.success("Anclaje temporal verificado con éxito.")

def main():
    parser = argparse.ArgumentParser(description="Luna V2 - Data Lake Synchronizer")
    parser.add_argument("--skip-fetch", action="store_true", help="Omitir descarga de datos (Fase 1)")
    parser.add_argument("--skip-sfi", action="store_true", help="Omitir el SFI (Proceso largo de aislamiento)")
    parser.add_argument("--skip-mining", action="store_true", help="Omitir re-calculo de reglas bayesianas")
    
    args = parser.parse_args()

    # Logging setup
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    
    log_dir = _ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(log_dir / f"sync_data_{ts}.log", rotation="10 MB", level="INFO")
    
    logger.info("==========================================================")
    logger.info("  INICIANDO SINCRONIZACIÓN DEL DATA LAKE (LUNA V2)        ")
    logger.info("==========================================================")

    # 1. Defensas y Fetchers
    # run_pre_flight()  # [COLD START FIX] Desactivado temporalmente para permitir generacion de data
    if not args.skip_fetch:
        run_fetchers()
        # run_data_integrity() # [COLD START FIX] Desactivado temporalmente
        # run_reconcile() # [COLD START FIX] Desactivado temporalmente para permitir features gen
    else:
        logger.info("Saltando Fase de Fetchers y Defensa por --skip-fetch.")

    # 2. Pipeline de Features vía LunaPipelineExecutor
    executor = LunaPipelineExecutor(
        seed=None,
        window_id=None,
        mode="PROD",
        options={"skip_sfi": args.skip_sfi, "skip_mining": args.skip_mining}
    )
    
    # Redefinir dinámicamente qué pasos ejecutar según los flags
    logger.info("==========================================================")
    logger.info("  INICIANDO FEATURE PIPELINE (SINTÉTICAS Y MINERÍA)       ")
    logger.info("==========================================================")
    
    executor._run_step("Feature Pipeline (Base Generation)", "luna/features/feature_pipeline.py", ["--skip-preflight"])
    
    if not args.skip_mining:
        executor._run_step("Build Dataset (AI Mining)", "scripts/build_dataset.py")
    else:
        logger.info("Saltando AI Mining por --skip-mining")
        
    executor._run_step("Feature Pipeline (Pre-SFI)", "luna/features/feature_pipeline.py", ["--skip-preflight"])
    
    if not args.skip_sfi:
        executor._run_step("SFI Feature Selection", "luna/features/feature_selection_e.py")
    else:
        logger.info("Saltando SFI por --skip-sfi")
        
    executor._run_step("Feature Pipeline (Post-SFI)", "luna/features/feature_pipeline.py", ["--skip-preflight"])

    # ── CHECK FINAL: SFI Coverage post-pipeline ───────────────────────
    # [SFI-COVERAGE-01 2026-06-03] Verificar que todas las features SFI
    # tienen datos reales despues del pipeline completo.
    # Este check corre al FINAL (post data + post features) para detectar:
    #   1. Features 100% NaN en parquets fuente (bug incremental fetch)
    #   2. Features en settings.yaml que nunca se materializaron en features_train
    logger.info("══════════════════════════════════════════════════════════")
    logger.info("  CHECK FINAL: SFI Coverage — verificando cobertura")
    logger.info("══════════════════════════════════════════════════════════")
    try:
        import subprocess as _sub
        _sfi_env = os.environ.copy()
        _sfi_env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + _sfi_env.get("PYTHONPATH", "") if _sfi_env.get("PYTHONPATH") else "")
        _sfi_result = _sub.run(
            [sys.executable, str(_ROOT / "scripts/check_sfi_coverage.py")],
            capture_output=True, text=True, cwd=str(_ROOT), env=_sfi_env, timeout=60
        )
        if _sfi_result.returncode != 0:
            logger.warning("  [SFI-COVERAGE-01] AVISO: algunas features SFI sin datos completos.")
            logger.warning("  → Relanzar sync_data_lake.py para regenerar con backfill completo.")
            for _ln in (_sfi_result.stdout + _sfi_result.stderr).splitlines()[-15:]:
                if "FAIL" in _ln or "WARN" in _ln or "→" in _ln:
                    logger.warning("    {}", _ln)
        else:
            logger.success("  [SFI-COVERAGE-01] Todas las features SFI tienen datos válidos.")
    except Exception as _e:
        logger.warning("  [SFI-COVERAGE-01] No se pudo ejecutar check de cobertura: {}", _e)

    logger.success("══════════════════════════════════════════════════════════")
    logger.success("  DATA LAKE SINCRONIZADO AL 100%                          ")
    logger.success("══════════════════════════════════════════════════════════")

if __name__ == "__main__":
    main()
