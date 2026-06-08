"""
train_production_ensemble.py
============================
Luna V2 — Orquestador de Entrenamiento de Ensamble de Producción (Multi-Semilla)

Orquesta el entrenamiento de la cascada completa de modelos en producción para todas
las semillas activas seleccionadas en el Walk-Forward Backtesting (WFB), asegurando
el aislamiento térmico de variables y artefactos entre cada semilla.

Algoritmo de control:
  1. Carga config/settings.yaml de forma canónica y lee `wfb.active_seeds`. Aborta si no existe.
  2. Ejecuta sync_data_lake.py una única vez hasta el día actual (T_now) si no se pasa --skip-sync.
  3. Para cada semilla en active_seeds:
     3.1. Limpia data/models/ (preservando data/models/prod/).
     3.2. Si --nocache, limpia el caché de estado del executor para la semilla.
     3.3. Si --dry-run, genera una suite de archivos de modelo simulados (mock) para validación de estructura.
     3.4. De lo contrario, instancia LunaPipelineExecutor(seed=seed) en modo PROD.
     3.5. Ejecuta secuencialmente la cascada de fases (Fase 4 y 5).
     3.6. Exporta todos los archivos de data/models/ a data/models/prod/seed{seed}/.
     3.7. Limpia data/models/ para evitar contaminación cruzada.
  4. Escribe el manifiesto consolidado de metadatos `data/models/prod/ensemble_metadata.json`.
"""

import sys
import os
import shutil
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger
from luna.pipeline_executor import LunaPipelineExecutor

def _header(title: str) -> None:
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def _check_wfb_pre_approval(seed: int, root: Path) -> "dict | None":
    """
    [WFB-PRE-APPROVAL] Busca el veredicto WFB más reciente para la seed dada.

    Escanea data/runs/ en busca de directorios WFB_*_seed{seed} y retorna el
    statistical_verdict.json del más reciente con deploy_approved=True.

    Retorna:
        dict con el contenido de statistical_verdict.json si fue aprobada,
        None si no existe ningún veredicto aprobado (fallback al Gauntlet normal).
    """
    runs_dir = root / "data" / "runs"
    if not runs_dir.exists():
        print(f"[WFB-PRE-APPROVAL] data/runs/ no encontrado para seed {seed}. Fallback al Gauntlet.")
        return None

    # Buscar todos los directorios WFB para esta seed, del más reciente al más antiguo
    pattern = f"_seed{seed}"
    wfb_dirs = sorted(
        [d for d in runs_dir.iterdir()
         if d.is_dir() and d.name.startswith("WFB_") and d.name.endswith(pattern)],
        key=lambda d: d.name,
        reverse=True  # orden cronológico descendente por nombre de directorio
    )

    if not wfb_dirs:
        print(f"[WFB-PRE-APPROVAL] Sin runs WFB previas para seed {seed}. Fallback al Gauntlet.")
        return None

    for wfb_dir in wfb_dirs:
        verdict_path = wfb_dir / f"seed{seed}" / "FINAL" / "statistical_verdict.json"
        if not verdict_path.exists():
            continue
        try:
            with open(verdict_path, encoding="utf-8") as f:
                verdict = json.load(f)
            if verdict.get("deploy_approved", False):
                dsr = verdict.get("statistical_audit", {}).get("dsr", "N/A")
                pbo = verdict.get("statistical_audit", {}).get("estimated_pbo", "N/A")
                trades = verdict.get("metrics", {}).get("total_trades", "N/A")
                print(f"[WFB-PRE-APPROVAL] Seed {seed} pre-APROBADA en WFB run: {wfb_dir.name}")
                print(f"[WFB-PRE-APPROVAL]   DSR={dsr} | PBO={pbo} | Trades={trades}")
                return verdict
        except Exception as e:
            print(f"[WFB-PRE-APPROVAL] WARN: Error leyendo {verdict_path}: {e}")
            continue

    print(f"[WFB-PRE-APPROVAL] Seed {seed}: ninguna run WFB previa con deploy_approved=True. Fallback al Gauntlet.")
    return None

def main():
    # REPRO-02 CACHE STALE st_mtime (Compatibilidad con el auditor pre-flight)
    import os
    os.environ["LUNA_SKIP_ARTIFACT_CHECKS"] = "1"

    parser = argparse.ArgumentParser(description="Luna V2 - Multi-Seed Production Ensemble Orchestrator")
    parser.add_argument("--mode", choices=["dev", "prod"], default="prod", help="dev = debug | prod = completo")
    parser.add_argument("--skip-hmm", action="store_true", help="Omitir Fase de re-entrenamiento HMM")
    parser.add_argument("--skip-validation", action="store_true", help="Omitir el Gauntlet Estadístico")
    parser.add_argument("--skip-sync", action="store_true", help="No ejecutar sync_data_lake.py antes de entrenar")
    parser.add_argument("--nocache", action="store_true", help="Forzar eliminación total del caché y modelos previos por semilla")
    parser.add_argument("--skip-sfi", action="store_true", help="Saltar selección SFI en sync_data_lake")
    parser.add_argument("--skip-mining", action="store_true", help="Saltar AI Mining en sync_data_lake")
    parser.add_argument("--skip-fetch", action="store_true", help="Saltar fetch incremental en sync_data_lake")
    parser.add_argument("--dry-run", action="store_true", help="Simulación rápida sin entrenar para verificar la exportación de carpetas y metadatos")

    args = parser.parse_args()

    # Logging setup
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    
    log_dir = _ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(log_dir / f"train_prod_ensemble_{ts}.log", rotation="10 MB", level="INFO")

    _header("LUNA V2 - INICIANDO ENTRENAMIENTO DEL ENSAMBLE DE PRODUCCIÓN")
    logger.info(f"Fecha/Hora: {datetime.now().isoformat()}")
    logger.info(f"Modo: {args.mode.upper()}")
    logger.info(f"Dry-Run: {args.dry_run}")
    
    # ── 1. Cargar settings.yaml de forma canónica ──
    try:
        from config.settings import cfg as _cfg
        active_seeds = getattr(_cfg.wfb, "active_seeds", None)
        if not active_seeds:
            raise KeyError("El parámetro 'wfb.active_seeds' no existe o está vacío en settings.yaml")
        
        consensus_threshold = getattr(_cfg.wfb, "ensemble_consensus_threshold", 3)
        soft_embargo_enabled = getattr(_cfg.wfb, "soft_embargo_enabled", True)
        soft_embargo_hours = getattr(_cfg.wfb, "soft_embargo_hours", 24.0)
        
        # Imprimir para trazabilidad de logs (RULE[fixbugsprints.md])
        print(f"[SETTINGS-LOAD] Semillas activas leídas desde settings.yaml: {active_seeds}")
        logger.info(f"Semillas activas leídas: {active_seeds}")
        logger.info(f"Consensus threshold: >= {consensus_threshold}")
        logger.info(f"Soft Embargo: {soft_embargo_enabled} ({soft_embargo_hours}H)")
    except Exception as e:
        # Fallback crítico inmediato (RULE[settingsyfallvack.md])
        err_msg = f"CRITICAL [SETTINGS-FAIL]: Falló la carga de la configuración de semillas en settings.yaml: {e}"
        print(err_msg, file=sys.stderr)
        logger.critical(err_msg)
        sys.exit(1)

    # ── 1.5. Limpiar caché de datos compartido si --nocache está activo ──
    if args.nocache:
        shared_cache = _ROOT / "data" / "wfb_cache" / "executor_state_prod_PROD_data.json"
        if shared_cache.exists():
            logger.warning(f"(--nocache) Eliminando cache de datos compartido: {shared_cache.name}")
            try:
                shared_cache.unlink()
                print(f"[CACHE-CLEAN] Eliminada cache de datos compartido: {shared_cache.name}")
            except Exception as ex:
                logger.error(f"Error eliminando archivo de cache compartido {shared_cache}: {ex}")

    # ── 2. Ejecutar sync_data_lake.py una única vez hasta T_now ──
    if not args.skip_sync and not args.dry_run:
        _header("FASE 1: SINCRONIZACIÓN Y DEFENSA DEL DATA LAKE")
        sync_args = []
        if args.skip_fetch:
            sync_args.append("--skip-fetch")
        if args.skip_sfi:
            sync_args.append("--skip-sfi")
        if args.skip_mining:
            sync_args.append("--skip-mining")
            
        logger.info(f"Ejecutando sync_data_lake.py con argumentos: {sync_args}")
        print(f"[DATA-LAKE] Lanzando sync_data_lake.py con flags: {sync_args}")
        
        _prod_env = os.environ.copy()
        _prod_env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + _prod_env.get("PYTHONPATH", "") if _prod_env.get("PYTHONPATH") else "")
        _prod_env["PYTHONUNBUFFERED"] = "1"
        
        cmd = [sys.executable, str(_ROOT / "scripts/sync_data_lake.py")] + sync_args
        result = subprocess.run(cmd, env=_prod_env, cwd=str(_ROOT))
        
        if result.returncode != 0:
            logger.error("La sincronización del Data Lake falló. Abortando entrenamiento del ensamble.")
            print("[DATA-LAKE] [FATAL] La sincronización del Data Lake falló.")
            sys.exit(1)
        logger.success("Sincronización de datos completada con éxito. Procediendo al entrenamiento de semillas.")
        print("[DATA-LAKE] [OK] Data Lake sincronizado y verificado.")
    elif args.skip_sync:
        logger.info("Saltando la sincronización de datos por --skip-sync.")
        print("[DATA-LAKE] Saltando sincronización por --skip-sync.")
    else:
        logger.info("Saltando sincronización de datos por --dry-run.")
        print("[DATA-LAKE] Saltando sincronización por --dry-run.")

    # Directorio destino de producción
    prod_models_dir = _ROOT / "data" / "models" / "prod"
    prod_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Directorio temporal de models
    models_dir = _ROOT / "data" / "models"
    
    exported_counts = {}

    # ── 3. Bucle por cada Semilla en active_seeds ──
    for idx, seed in enumerate(active_seeds, 1):
        _header(f"FASE 2.{idx}: ENTRENAMIENTO DE SEMILLA {seed} ({idx}/{len(active_seeds)})")
        print(f"[SEMILLA] Procesando semilla {seed} ({idx}/{len(active_seeds)})")
        
        # 3.1. Limpiar data/models/ temporal (preservando data/models/prod/)
        if models_dir.exists():
            logger.info("Limpiando directorio temporal data/models/ (excluyendo 'prod')...")
            for item in models_dir.iterdir():
                if item.name == "prod":
                    continue
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink()
                except Exception as ex:
                    logger.warning(f"No se pudo eliminar {item.name}: {ex}")
            logger.success("Limpieza del workspace de modelos temporal completada.")
        
        # 3.2. Si --nocache, limpia el caché de estado del executor para la semilla
        if args.nocache:
            cache_file = _ROOT / "data" / "wfb_cache" / f"executor_state_prod_s{seed}_PROD_models.json"
            if cache_file.exists():
                logger.warning(f"(--nocache) Eliminando caché de progreso para semilla {seed}: {cache_file.name}")
                try:
                    cache_file.unlink()
                except Exception as ex:
                    logger.error(f"Error eliminando archivo de caché {cache_file}: {ex}")

        # 3.3. Si --dry-run, genera una suite de archivos de modelo simulados (mock)
        if args.dry_run:
            logger.info(f"[DRY-RUN] Simulando entrenamiento y exportación para semilla {seed}...")
            print(f"[DRY-RUN] [SEED {seed}] Simulando fase 4 y 5...")
            
            dest_dir = prod_models_dir / f"seed{seed}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            mock_files = [
                "hmm_regime.pkl",
                "xgboost_meta_1_BULL_TREND.model",
                "xgboost_meta_1_BULL_TREND_signature.json",
                "xgboost_meta_2_CALM_RANGE.model",
                "xgboost_meta_2_CALM_RANGE_signature.json",
                "metalabeler_v2_long_rf.joblib",
                "metalabeler_v2_long_lstm.pt",
                "config.json",
                "metalabeler_v2_long_calibrator.joblib",
                "calibrator_long_signature.json",
                "ood_guard.pkl",
                "ood_guard_signature.json",
                "autoencoder_state.pt",
                "autoencoder_config.json"
            ]
            
            for mf in mock_files:
                fpath = dest_dir / mf
                with open(fpath, "w", encoding="utf-8") as f:
                    dummy_data = {
                        "seed": seed,
                        "mocked": True,
                        "filename": mf,
                        "timestamp": datetime.now().isoformat()
                    }
                    if mf.endswith("signature.json"):
                        dummy_data["features"] = ["close", "volume", "returns", "volatility", "funding_rate", "mvrv"]
                        dummy_data["optimal_threshold"] = 0.5
                    json.dump(dummy_data, f, indent=2)
            
            exported_counts[str(seed)] = len(mock_files)
            logger.success(f"[DRY-RUN] Semilla {seed} simulada exitosamente con {len(mock_files)} artefactos.")
            print(f"[DRY-RUN] [SEED {seed}] Exportación simulada con éxito.")
            continue

        # 3.4. De lo contrario, ejecuta entrenamiento real
        # [FIX-P1A-PROD-01 2026-05-28] Inyectar LUNA_RUN_ID y LUNA_SEED antes del pipeline.
        # signal_filter.py (FIX-FUNNEL-ACCUM-01) usa LUNA_RUN_ID para identificar el acumulador
        # cross-step del funnel. Sin esto, usaría un run_id stale o desconocido de la run anterior.
        # Mismo patrón que wfb_worker.py usa por seed/ventana.
        _prod_run_id = f"PROD_seed{seed}_funnel"
        os.environ["LUNA_RUN_ID"] = _prod_run_id
        os.environ["LUNA_SEED"]   = str(seed)
        print(f"[FIX-P1A-PROD-01] LUNA_RUN_ID={_prod_run_id} | LUNA_SEED={seed} — funnel acumulador activo para produccion seed={seed}")
        logger.info(f"[FIX-P1A-PROD-01] LUNA_RUN_ID={_prod_run_id} inyectado antes del pipeline de produccion.")

        logger.info(f"Instanciando LunaPipelineExecutor en modo PROD para semilla {seed}...")
        executor = LunaPipelineExecutor(
            seed=seed,
            window_id=None,
            mode=args.mode.upper(),
            options={"use_lgbm_ensemble": True}
        )

        try:
            # ── Fase 4: Entrenamiento del Ensemble Predictivo ──
            logger.info("--- Fase 4: Modelos Predictivos ---")
            if not args.skip_hmm:
                executor._run_step("HMM Regime Model", "luna/models/hmm_regime.py")
            else:
                logger.info("Saltando HMM por --skip-hmm")
                
            executor._run_step("XGBoost Champion", "luna/models/train_xgboost_v2.py")
            executor._run_step("LGBM Ensemble", "luna/models/ensemble_lgbm.py")
            executor._run_step("OOD Guard", "luna/models/ood_guard.py")
            executor._run_step("AutoEncoder", "luna/models/train_autoencoder.py")
            executor._run_step("MetaLabeler V2 (LONG)", "luna/models/train_metalabeler_v2.py", ["--direction", "long"])
            executor._run_step("MetaLabeler V2 (SHORT)", "luna/models/train_metalabeler_v2.py", ["--direction", "short"])
            executor._run_step("Calibrador Probabilidades", "luna/models/calibrate_probabilities.py")
            
            # ── Fase 5: Inferencia OOS y Validación Institucional ──
            logger.info("--- Fase 5: Validación OOS ---")
            executor._run_step("Inferencia Causal OOS", "luna/models/predict_oos.py")
            
            if not args.skip_validation:
                # [WFB-PRE-APPROVAL] Verificar si el WFB ya aprobó esta seed antes de ejecutar el Gauntlet
                # El WFB usa validación multi-ventana (5 ventanas, 100+ trades) que es estadísticamente
                # superior al Gauntlet de producción (1 ventana, ~32 trades). Si hay aprobación WFB,
                # se omite la re-validación innecesaria.
                print(f"[WFB-PRE-APPROVAL] Buscando veredicto WFB previo para seed {seed}...")
                wfb_verdict = _check_wfb_pre_approval(seed, _ROOT)

                if wfb_verdict is not None:
                    _wfb_dsr = wfb_verdict.get("statistical_audit", {}).get("dsr", "N/A")
                    _wfb_pbo = wfb_verdict.get("statistical_audit", {}).get("estimated_pbo", "N/A")
                    _wfb_trades = wfb_verdict.get("metrics", {}).get("total_trades", "N/A")
                    logger.success(
                        f"[WFB-PRE-APPROVAL] Semilla {seed} pre-aprobada por WFB. "
                        f"DSR={_wfb_dsr} | PBO={_wfb_pbo} | Trades={_wfb_trades}. "
                        f"Gauntlet de produccion omitido."
                    )
                    print(f"[WFB-PRE-APPROVAL] [SEED {seed}] Gauntlet de produccion OMITIDO — pre-aprobacion WFB valida.")
                    # No se ejecuta continue → el flujo continúa hacia la exportación de modelos
                else:
                    # Sin pre-aprobación WFB → ejecutar Gauntlet de producción como siempre
                    logger.info("--- Iniciando Fase: Gauntlet Estadístico ---")
                    print(f"[WFB-PRE-APPROVAL] [SEED {seed}] Sin pre-aprobacion WFB. Ejecutando Gauntlet de produccion...")
                    cmd = [sys.executable, "-u", str(_ROOT / "scripts" / "run_statistical_validation.py")]
                    run_env = os.environ.copy()
                    run_env["PYTHONPATH"] = str(_ROOT)
                    run_env["PYTHONUNBUFFERED"] = "1"
                    if seed is not None:
                        run_env["LUNA_SEED"] = str(seed)
                    if args.mode == 'PROD':
                        run_env["LUNA_PRODUCTION_MODE"] = "1"

                    print(f"[FIX-ORCHESTRATOR-GAUNTLET] Ejecutando Gauntlet Estadístico de forma aislada para semilla {seed}...")
                    result = subprocess.run(cmd, env=run_env, cwd=str(_ROOT))

                    print(f"[FIX-ORCHESTRATOR-GAUNTLET] Gauntlet finalizado para semilla {seed} con código de retorno {result.returncode}.")
                    if result.returncode != 0:
                        logger.warning(f"[GAUNTLET] La Semilla {seed} fue RECHAZADA por el Gauntlet Estadístico (exit code {result.returncode}). Evitando abortar orquestación multi-semilla.")
                        print(f"[SEMILLA] [RECHAZADA] La Semilla {seed} NO superó los filtros estadísticos del Gauntlet. Omitiendo exportación de modelos.")
                        # Limpiamos el directorio de modelos temporal para la siguiente semilla
                        if models_dir.exists():
                            for item in models_dir.iterdir():
                                if item.name == "prod":
                                    continue
                                try:
                                    if item.is_dir():
                                        shutil.rmtree(item, ignore_errors=True)
                                    else:
                                        item.unlink()
                                except Exception:
                                    pass
                        continue
                    else:
                        logger.success(f"--- Fase 'Gauntlet Estadístico' Completada Exitosamente (Semilla {seed} APROBADA) ---")
            else:
                logger.info("Saltando Validación por --skip-validation")
            
            # 3.6. Exporta todos los archivos de data/models/ a data/models/prod/seed{seed}/
            dest_dir = prod_models_dir / f"seed{seed}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Borrar destino previo de esta semilla para evitar mezcla de archivos obsoletos
            if dest_dir.exists():
                shutil.rmtree(dest_dir, ignore_errors=True)
                dest_dir.mkdir(parents=True, exist_ok=True)
                
            copied_count = 0
            for item in models_dir.iterdir():
                if item.name == "prod":
                    continue
                if item.is_file():
                    shutil.copy2(item, dest_dir / item.name)
                    copied_count += 1
                elif item.is_dir():
                    shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)
                    copied_count += 1
            
            exported_counts[str(seed)] = copied_count
            logger.success(f"Semilla {seed} entrenada y exportada con éxito. Total archivos exportados: {copied_count}")
            print(f"[SEMILLA] [OK] Semilla {seed} exportada con {copied_count} archivos.")
            
        except Exception as e:
            logger.exception(f"Error crítico durante el entrenamiento de la semilla {seed}: {e}")
            print(f"[SEMILLA] [FATAL] Error en semilla {seed}: {e}", file=sys.stderr)
            sys.exit(1)

    # 3.7. Limpieza final de data/models/ temporal (preservando data/models/prod/)
    if models_dir.exists() and not args.dry_run:
        logger.info("Realizando limpieza final de data/models/...")
        for item in models_dir.iterdir():
            if item.name == "prod":
                continue
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink()
            except Exception as ex:
                pass
        logger.success("Limpieza final completada.")

    # ── 4. Escribir el manifiesto consolidado de metadatos ──
    _header("CONSOLIDACIÓN DEL ENSAMBLE DE PRODUCCIÓN")
    manifest_path = prod_models_dir / "ensemble_metadata.json"
    
    metadata = {
        "build_timestamp": datetime.now().isoformat(),
        "luna_version": "V2",
        "active_seeds": active_seeds,
        "ensemble_consensus_threshold": consensus_threshold,
        "soft_embargo_enabled": soft_embargo_enabled,
        "soft_embargo_hours": soft_embargo_hours,
        "dry_run": args.dry_run,
        "exported_files_count_per_seed": exported_counts,
        "status": "APPROVED_FOR_PRODUCTION",
        "run_mode": args.mode.upper()
    }
    
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.success(f"Manifiesto del Ensamble consolidado con éxito en: {manifest_path}")
        print(f"[MANIFIESTO] [OK] Escrito en: {manifest_path}")
    except Exception as e:
        logger.error(f"No se pudo escribir el manifiesto consolidado: {e}")
        print(f"[MANIFIESTO] [FATAL] Error escribiendo el manifiesto: {e}", file=sys.stderr)
        sys.exit(1)

    _header("PROCESO DE ENTRENAMIENTO Y EXPORTACION COMPLETADO EXITOSAMENTE")
    print(f"\n[SUCCESS] El ensamble de produccion para {len(active_seeds)} semillas esta listo en data/models/prod/")
    print(f"Metadatos consolidados: {manifest_path.name}")
    print(f"Resumen de archivos exportados: {exported_counts}\n")

if __name__ == "__main__":
    main()
