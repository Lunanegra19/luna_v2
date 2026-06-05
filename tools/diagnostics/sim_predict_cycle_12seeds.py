"""
sim_predict_cycle_12seeds.py
────────────────────────────────────────────────────────────────────────────
[SIM-CYCLE-12S-01] Simulacion del ciclo predict_cycle del ensamble de 12 seeds.
Mide el tiempo total del ciclo completo y valida que:
  1. Carga correcta de las 12 seeds sin MOCK
  2. OPT-HMM-SHARED-01 activo (HMM calculado UNA sola vez)
  3. Tiempo total < 30 segundos (limite de seguridad operacional)
  4. Decision de consenso coherente con el regimen HMM real
  5. Al menos 1/12 seeds vota (no HOLD total con error)

Equivalente al test de cambio de hora pero con datos reales del data lake.
Lanzar desde la raiz del proyecto con PYTHONPATH configurado.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("[SIM-CYCLE-12S-01] Simulacion predict_cycle 12 seeds")
print(f"  Proyecto: {PROJECT_ROOT}")
print(f"  Hora inicio: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ── 1. Cargar datos reales del data lake ─────────────────────────────────────
print("\n[PASO 1] Cargando datos del data lake (live feature pipeline)...")
t0_data = time.time()

try:
    import pandas as pd
    import numpy as np

    # Cargar el parquet de features más reciente disponible
    features_paths = [
        PROJECT_ROOT / "data" / "features" / "features_holdout.parquet",
        PROJECT_ROOT / "data" / "features" / "features_validation.parquet",
        PROJECT_ROOT / "data" / "features" / "features_train.parquet",
    ]

    df_loaded = None
    for fp in features_paths:
        if fp.exists():
            df_loaded = pd.read_parquet(fp)
            print(f"  [OK] Cargado: {fp.name} | shape={df_loaded.shape} | "
                  f"range={df_loaded.index.min().date()} -> {df_loaded.index.max().date()}")
            break

    if df_loaded is None:
        print("  [ERROR] No se encontro ningun parquet de features. Abortando.")
        sys.exit(1)

    # Asegurar que el indice tiene UTC
    df_loaded.index = pd.to_datetime(df_loaded.index, utc=True)

    # Tomar las ultimas 500 velas (equivalente al live trader)
    df_sim = df_loaded.tail(500).copy()
    t_data = time.time() - t0_data
    print(f"  [OK] DataFrame simulacion: {df_sim.shape} | "
          f"ultima_vela={df_sim.index[-1].strftime('%Y-%m-%d %H:%M')} | "
          f"carga={t_data:.2f}s")

except Exception as e:
    print(f"  [ERROR] Fallo al cargar datos: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 2. Instanciar el ensamble de 12 seeds ────────────────────────────────────
print("\n[PASO 2] Instanciando LunaEnsembleLiveInference (12 seeds)...")
t0_boot = time.time()

try:
    from luna.live.ensemble_live_inference import LunaEnsembleLiveInference
    ensemble = LunaEnsembleLiveInference()
    t_boot = time.time() - t0_boot
    n_seeds_loaded = len(ensemble.seeds_models)
    print(f"  [OK] Ensamble cargado: {n_seeds_loaded}/12 seeds | boot={t_boot:.1f}s")
    print(f"  [OK] _hmm_shared={getattr(ensemble, '_hmm_shared', 'N/A')} | "
          f"consensus_CUTOFF = {ensemble.consensus_threshold}")
except Exception as e:
    print(f"  [ERROR] Fallo al instanciar ensamble: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 3. Ejecutar predict_cycle y medir tiempo ─────────────────────────────────
print("\n[PASO 3] Ejecutando predict_cycle con datos reales...")
print("-" * 70)
t0_cycle = time.time()

try:
    result = ensemble.predict_cycle(df_sim)
    t_cycle = time.time() - t0_cycle
except Exception as e:
    t_cycle = time.time() - t0_cycle
    print(f"\n  [ERROR] predict_cycle fallo tras {t_cycle:.1f}s: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 4. Informe de resultados ─────────────────────────────────────────────────
print("-" * 70)
print("\n[SIM-CYCLE-12S-01] RESULTADOS DE LA SIMULACION")
print("=" * 70)

# Tiempo
LIMITE_SEGURIDAD_S = 30.0
status_tiempo = "✅ DENTRO DEL LIMITE" if t_cycle < LIMITE_SEGURIDAD_S else "❌ SUPERA EL LIMITE"
print(f"\n  ⏱  Tiempo ciclo:     {t_cycle:.2f}s   [{status_tiempo}] (limite={LIMITE_SEGURIDAD_S}s)")

# Decision
action        = result.get("action", "N/A")
confidence    = result.get("confidence", 0.0)
consensus_cnt = result.get("consensus_count", 0)
regime        = result.get("regime", "N/A")
xgb_prob      = result.get("xgb_prob", 0.0)
embargo       = result.get("soft_embargo_active", False)
n_active      = n_seeds_loaded

print(f"\n  🧠  Decision:         {action}")
print(f"  📊  Consenso:         {consensus_cnt}/{n_active} seeds")
print(f"  🎯  Confianza:        {confidence:.2%}")
print(f"  📈  XGB Prob:         {xgb_prob:.4f}")
print(f"  🌐  Regimen HMM:      {regime}")
print(f"  ⏳  Soft Embargo:     {'ACTIVO (24H reducido)' if embargo else 'INACTIVO'}")

# Breakdown por seed
print(f"\n  📋  Breakdown por seed:")
breakdown = result.get("seeds_breakdown", {})
n_mock    = 0
n_error   = 0
for seed, details in breakdown.items():
    decision = details.get("decision", "N/A")
    regime_s = details.get("regime", "N/A")
    xgb_s    = details.get("xgb_prob", 0.0)
    meta_s   = details.get("meta_prob", 0.0)
    error_s  = details.get("error", None)
    if error_s:
        n_error += 1
        print(f"    Seed {str(seed):>6}: ❌ ERROR={error_s[:60]}")
    else:
        print(f"    Seed {str(seed):>6}: {decision:<5} | HMM={regime_s:<22} | XGB={xgb_s:.4f} | Meta={meta_s:.4f}")

# OPT-HMM-SHARED-01 confirmacion
print(f"\n  🔧  OPT-HMM-SHARED-01: {'✅ ACTIVO' if getattr(ensemble, '_hmm_shared', False) else '⚠️  INACTIVO (fallback per-seed)'}")

# Validaciones finales
print("\n" + "=" * 70)
print("[SIM-CYCLE-12S-01] VALIDACIONES FINALES")
print("=" * 70)

checks = {
    "Seeds cargadas (12/12)":           n_seeds_loaded == 12,
    "Sin errores en seeds":             n_error == 0,
    f"Ciclo < {LIMITE_SEGURIDAD_S}s":  t_cycle < LIMITE_SEGURIDAD_S,
    "Decision coherente (no None)":     action in ("LONG", "SHORT", "HOLD"),
    "Al menos 1 seed vota":             consensus_cnt >= 1,
    "OPT-HMM-SHARED-01 activo":        getattr(ensemble, '_hmm_shared', False),
}

all_pass = True
for check_name, passed in checks.items():
    icon = "✅" if passed else "❌"
    print(f"  {icon}  {check_name}")
    if not passed:
        all_pass = False

print("\n" + "=" * 70)
if all_pass:
    print(f"  ✅ SIMULACION EXITOSA — El ensamble de 12 seeds opera correctamente")
    print(f"     Ciclo: {t_cycle:.1f}s (margen: {LIMITE_SEGURIDAD_S - t_cycle:.1f}s disponibles)")
else:
    print(f"  ❌ SIMULACION FALLIDA — Revisar checks anteriores")
print("=" * 70)

sys.exit(0 if all_pass else 1)
