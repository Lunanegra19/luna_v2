"""
run_ai_mining.py — Orquestador de AI Mining Luna V1
====================================================
Ejecuta los 8 motores de AI Mining en secuencia obligatoria (SOP):

  1. BayesianCausalEngine → features_train_causal.parquet   (+Master_Causal_Signal)
  2. AdvancedEngine       → advanced_engine_results.csv      (SHAP + IsoForest + StructBreaks)
  3. MasterPatternEngine  → master_pattern_rules.csv         (Golden Rules por régimen HMM)
  4. DeepDiscoveryEngine  → deep_discovery_rules.csv         (RFE + Genetic Rules + DTW)
  5. ClusterPatternEngine → features_train_final.parquet     (K-Means Tribus + Estacionalidad)
  6. export_alpha_rules   → core/features/alpha_rules.py     (señales alpha Python nativo)

Flujo de datos (según manual 2.0_Data_Mining.md):
  features_train.parquet (base)
    → [1] bayesian → features_train_causal.parquet
    ← [NO aquí] integrate_mining_outputs() es parte de feature_pipeline.py
    → [2-5] advanced/master/deep/cluster  (leen el parquet más reciente disponible)
    → [6] export_alpha → alpha_rules.py

NOTA: Después de run_ai_mining.py es OBLIGATORIO ejecutar:
  python scripts/run_features_and_training.py --only-features  # integra kshape+causal en features_train
  python scripts/run_features_and_training.py --only-train      # re-entrena XGBoost

Argumentos:
  --mode dev|prod   (default: dev)
      dev  → aplica cutoff_date = settings.yaml:temporal_splits.train_end
      prod → cutoff_date = None (usa todo el histórico disponible)

  --engine ENGINE   (opcional)
      Ejecuta solo ese engine. Útil para debug y re-runs parciales.
      Valores válidos: bayesian | advanced | master | deep |
                       cluster | export (alias: export_alpha)

Uso típico:
  python scripts/run_ai_mining.py --mode dev
  python scripts/run_ai_mining.py --mode prod
  python scripts/run_ai_mining.py --mode dev --engine export
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_cutoff_date(mode: str):
    """
    Devuelve el cutoff_date según el modo:
      - dev  → train_end de settings.yaml (ej. "2023-12-31")
      - prod → None  (sin restricción temporal)
    """
    if mode == "prod":
        logger.info("AI Mining: modo PROD — sin cutoff_date (histórico completo)")
        return None

    # Modo dev: leer train_end de settings.yaml
    try:
        from config.settings import cfg
        train_end = cfg.temporal_splits.train_end
        logger.info(f"AI Mining: modo DEV — cutoff_date = {train_end}")
        return train_end
    except Exception as e:
        logger.warning(
            f"AI Mining: no se pudo leer train_end de settings.yaml ({e}). "
            "Usando fallback '2023-12-31'."
        )
        return "2023-12-31"


def _run_engine(name: str, engine_fn, results: list[dict]) -> bool:
    """
    Wrapper tolerante a fallos para ejecutar un engine.
    Registra duración y resultado en la lista `results`.
    Devuelve True si el engine terminó sin excepción.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  INICIANDO: {name.upper()}")
    logger.info("=" * 70)
    t0 = time.time()
    try:
        engine_fn()
        elapsed = round(time.time() - t0, 1)
        logger.success(f"  ✅ {name} completado en {elapsed}s")
        results.append({"engine": name, "status": "OK", "elapsed_s": elapsed})
        return True
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        logger.error(f"  ❌ {name} FALLÓ en {elapsed}s: {e}")
        results.append({"engine": name, "status": "FAIL", "elapsed_s": elapsed, "error": str(e)})
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Factories de engines
# ─────────────────────────────────────────────────────────────────────────────

def _make_engines(cutoff_date) -> list[tuple[str, callable]]:
    """
    Construye la lista ordenada de (nombre, callable) de los 6 engines activos.
    Cada callable es una función de cero argumentos que ejecuta engine.run().

    La importación se hace aquí (lazy) para evitar errores de import si una
    dependencia opcional (tslearn, dowhy, etc.) no está instalada — el error
    se capturará solo cuando ese engine se intente ejecutar.
    """
    engines: list[tuple[str, callable]] = []



    # 1. Bayesian Causal Engine
    def run_bayesian():
        from luna.ai_mining.bayesian_causal_engine import BayesianCausalEngine
        BayesianCausalEngine(cutoff_date=cutoff_date).run()

    engines.append(("bayesian", run_bayesian))

    # 2. Advanced Engine
    def run_advanced():
        import pandas as pd
        from luna.ai_mining.advanced_engine import AdvancedEngine
        cutoff_ts = pd.Timestamp(cutoff_date, tz="UTC") if cutoff_date else None
        AdvancedEngine(cutoff_date=cutoff_ts).run()

    engines.append(("advanced", run_advanced))

    # 3. Master Pattern Engine
    def run_master():
        import pandas as pd
        from luna.ai_mining.master_pattern_engine import MasterPatternEngine
        cutoff_ts = pd.Timestamp(cutoff_date, tz="UTC") if cutoff_date else None
        MasterPatternEngine(cutoff_date=cutoff_ts).run()

    engines.append(("master", run_master))

    # 4. Deep Discovery Engine
    def run_deep():
        import pandas as pd
        from luna.ai_mining.deep_discovery_engine import DeepDiscoveryEngine
        cutoff_ts = pd.Timestamp(cutoff_date, tz="UTC") if cutoff_date else None
        DeepDiscoveryEngine(cutoff_date=cutoff_ts).run()

    engines.append(("deep", run_deep))

    # 5. Cluster Pattern Engine
    def run_cluster():
        from luna.ai_mining.cluster_pattern_engine import ClusterPatternEngine
        ClusterPatternEngine(cutoff_date=cutoff_date).run()

    engines.append(("cluster", run_cluster))

    # 6. Export Alpha Rules → genera alpha_rules.py nativo
    def run_export_alpha():
        from luna.ai_mining.export_alpha_rules import main as export_main
        export_main()

    engines.append(("export_alpha", run_export_alpha))

    # 7. HMM Regime Model → genera HMM_Regime centralizado
    def run_hmm_regime():
        from luna.models.hmm_regime import HMMRegimeModel
        # Se ejecuta a nivel global o por ventana, usando la misma configuración de train_cutoff
        model = HMMRegimeModel()
        model.run()

    engines.append(("hmm_train", run_hmm_regime))

    return engines


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orquestador del pipeline AI Mining — Luna V1"
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        default="dev",
        help="dev: aplica cutoff_date=train_end | prod: datos completos",
    )
    parser.add_argument(
        "--engine",
        default=None,
        help=(
            "Ejecutar solo este engine. Valores válidos: "
            "bayesian | advanced | master | deep | "
            "cluster | export (alias: export_alpha)"
        ),
    )
    args = parser.parse_args()

    # Normalizar alias cortos definidos en el manual 2.0_Data_Mining.md
    _ENGINE_ALIASES = {"export": "export_alpha"}
    if args.engine:
        args.engine = args.engine.lower()
        if args.engine in _ENGINE_ALIASES:
            args.engine = _ENGINE_ALIASES[args.engine]

    # ── Log file propio del subproceso (trazabilidad por RUN_ID) ──────────────
    import os as _os
    from datetime import datetime as _dt
    _log_dir = PROJECT_ROOT / "logs"
    _log_dir.mkdir(exist_ok=True)
    _ts   = _dt.now().strftime("%Y%m%d_%H%M%S")
    _rid  = _os.environ.get("LUNA_RUN_ID", "")
    _lname = f"run_ai_mining_{_ts}_{_rid}.log" if _rid else f"run_ai_mining_{_ts}.log"
    logger.add(_log_dir / _lname, rotation="50 MB", level="DEBUG", encoding="utf-8")
    # ─────────────────────────────────────────────────────────────────────────

    t_total = time.time()
    logger.info("=" * 70)
    logger.info("  LUNA V1 — AI MINING ORCHESTRATOR")
    logger.info(f"  Modo: {args.mode.upper()}"
                + (f" | Engine: {args.engine}" if args.engine else " | Todos los engines"))
    logger.info("=" * 70)

    cutoff_date = _get_cutoff_date(args.mode)
    all_engines = _make_engines(cutoff_date)

    # Filtrar por --engine si se especificó
    if args.engine:
        valid_names = [name for name, _ in all_engines]
        if args.engine not in valid_names:
            logger.error(
                f"Engine '{args.engine}' no reconocido. "
                f"Válidos: {valid_names}"
            )
            return 1
        engines_to_run = [(n, fn) for n, fn in all_engines if n == args.engine]
    else:
        engines_to_run = all_engines

    _project_root = PROJECT_ROOT
    _critical_outputs = [
        _project_root / "data" / "features" / "features_train_causal.parquet",  # bayesian engine
        _project_root / "data" / "features" / "features_train_final.parquet",   # cluster engine
    ]
    _mtime_before = {p: p.stat().st_mtime if p.exists() else 0 for p in _critical_outputs}

    # Ejecutar engines en secuencia
    results: list[dict] = []
    for name, fn in engines_to_run:
        _run_engine(name, fn, results)

    # Resumen final
    elapsed_total = round(time.time() - t_total, 1)
    ok    = [r for r in results if r["status"] == "OK"]
    fails = [r for r in results if r["status"] == "FAIL"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("  RESUMEN AI MINING")
    logger.info("=" * 70)
    for r in results:
        icon = "✅" if r["status"] == "OK" else "❌"
        err  = f" → {r['error']}" if r.get("error") else ""
        logger.info(f"  {icon}  {r['engine']:<18} {r['elapsed_s']:>6.1f}s{err}")
    logger.info("-" * 70)
    logger.info(f"  Total: {len(ok)}/{len(results)} engines OK | "
                f"{len(fails)} FAIL | {elapsed_total}s total")
    logger.info("=" * 70)

    if fails:
        # P0-2-FIX (2026-03-30): Verificar que los archivos de salida críticos existen y
        # son recientes (modificados DESPUÉS de que comenzó este run). Si faltan o son stale,
        # retornar exit 1 para que el WFB orchestrator aborte en lugar de continuar
        # silenciosamente con parquets del ciclo anterior.
        _failed_engine_names = {r["engine"] for r in fails}
        _critical_engines_failed = _failed_engine_names & {"bayesian", "cluster"}
        _missing_critical = []
        _stale_critical = []
        if _critical_engines_failed:
            for _out_path in _critical_outputs:
                if not _out_path.exists():
                    _missing_critical.append(_out_path.name)
                elif _out_path.stat().st_mtime <= _mtime_before.get(_out_path, 0):
                    _stale_critical.append(_out_path.name)

        if _missing_critical or _stale_critical:
            logger.error(
                f"[P0-2-FIX] AI Mining CRÍTICO: engines {_critical_engines_failed} fallaron "
                f"y sus outputs son inválidos. "
                f"Ausentes: {_missing_critical} | Stale (sin actualizar): {_stale_critical}. "
                "El WFB NO puede continuar con datos del ciclo anterior — abortando."
            )
            return 1  # ← exit 1 para que WFB orchestrator aborte
        else:
            logger.warning(
                f"AI Mining completado con {len(fails)} error(es) en engines no-críticos: "
                f"{_failed_engine_names - _critical_engines_failed}. "
                "Los archivos de salida críticos están presentes y actualizados — pipeline puede continuar."
            )
            return 0

    logger.success("AI Mining completado exitosamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
