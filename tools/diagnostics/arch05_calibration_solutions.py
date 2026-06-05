"""arch05_calibration_solutions.py
Analiza las 3 soluciones para el problema de calibracion con validation 100% BULL.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

df = pd.read_parquet(ROOT / 'data/features/features_train.parquet', columns=['close','HMM_Semantic'])
print("="*70)
print("ARCH-05: CALIBRACION EN MERCADO NO-ESTACIONARIO")
print("El problema: validation 100% BULL es consecuencia del mercado actual,")
print("NO de una mala eleccion de fechas. BTC cambio de regimen desde 2023.")
print("="*70)

# ── RAIZ DEL PROBLEMA ──────────────────────────────────────────────────────
print("\n[RAIZ] Distribucion HMM por año:")
dist = df.groupby([df.index.year, 'HMM_Semantic']).size().unstack(fill_value=0)
print(dist.to_string())

# ── SOLUCION 1: IS-Block Stratified ────────────────────────────────────────
print("\n" + "="*70)
print("SOLUCION 1: IS-Block Stratified Calibration")
print("Tomar las ultimas N barras de cada regimen IS y concatenarlas")
print("-"*70)
N = 1000
for regime in sorted(df['HMM_Semantic'].dropna().unique()):
    sub = df[df['HMM_Semantic'] == regime].tail(N)
    print(f"  {regime}: {len(sub)} barras | {sub.index.min().date()} -> {sub.index.max().date()}")
print("\n  PROBLEMA CRITICO: El modelo ya vio estos datos en train.")
print("  Los scores del modelo sobre IS son inflados -> calibrador sobreajustado.")
print("  VEREDICTO: DESCARTADA (contamina calibracion)")

# ── SOLUCION 2: CPCV OOB Scores ────────────────────────────────────────────
print("\n" + "="*70)
print("SOLUCION 2: CPCV OOB Scores para calibracion isotonica")
print("-"*70)
print("  El CPCV ya genera predicciones OOB honestas en cada fold.")
print("  Los OOB scores son los unicos scores donde el modelo no vio los datos.")
print("  Actualmente se usan solo para Optuna (seleccionar hiperparametros).")
print("  Fix: guardar y exponer los OOB scores del mejor trial Optuna")
print("       y usarlos para ajustar el calibrador isotonic.")
print()
print("  VENTAJAS:")
print("  - Los OOB scores cubren TODO el IS historico (todos los regimenes)")
print("  - No hay contaminacion: el modelo nunca vio esos datos")
print("  - Automaticamente multi-regimen (el IS tiene BULL+RANGE+BEAR 2017-2025)")
print("  - Es la unica fuente de calibracion verdaderamente OOS del IS")
print()
print("  DESVENTAJAS:")
print("  - Requiere cambios en train_xgboost_v2.py (guardar OOB scores de Optuna)")
print("  - Los OOB del CPCV tienen estructura temporal -> calibracion sobre distribucion IS")
print("    que difiere del OOS (pero NO 100% BULL - mucho mejor que ahora)")
print()
print("  IMPLEMENTABILIDAD: Media - requiere modificar Optuna callback en train_xgboost_v2.py")

# ── SOLUCION 3: Calibration IS holdout interno ─────────────────────────────
print("\n" + "="*70)
print("SOLUCION 3: Calibration holdout interno del IS (RECOMENDADA)")
print("-"*70)
print("  El IS actual: 2017-08 -> 2025-04")
print("  Propuesta: reservar los ultimos 12m del IS para calibracion del isotonic")
print("  El threshold sweep (Optuna) continua usando features_validation.parquet")
print()
cal_is = df[df.index >= '2024-04-01']
print(f"  Cal IS propuesto (abr 2024 -> abr 2025): {len(cal_is)} barras")
dist_cal = cal_is['HMM_Semantic'].value_counts()
for reg, n in dist_cal.items():
    print(f"    {reg}: {n} barras ({n/len(cal_is):.1%})")
print()
print("  Vs. validation actual (may-jun 2025):")
val = pd.read_parquet(ROOT / 'data/features/features_validation.parquet', columns=['close','HMM_Semantic'])
dist_val = val['HMM_Semantic'].value_counts()
for reg, n in dist_val.items():
    print(f"    {reg}: {n} barras ({n/len(val):.1%})")
print()
print("  MEJORA KL divergence vs IS:")
is_dist = df['HMM_Semantic'].value_counts(normalize=True)
cal_dist = cal_is['HMM_Semantic'].value_counts(normalize=True)
val_dist = val['HMM_Semantic'].value_counts(normalize=True)
all_regs = set(is_dist.index) | set(cal_dist.index) | set(val_dist.index)
kl_val = sum(is_dist.get(r,1e-6) * np.log(is_dist.get(r,1e-6) / val_dist.get(r,1e-6)) for r in all_regs)
kl_cal = sum(is_dist.get(r,1e-6) * np.log(is_dist.get(r,1e-6) / cal_dist.get(r,1e-6)) for r in all_regs)
print(f"  KL(IS||validation_actual): {kl_val:.3f}  <- actual (catastrofico)")
print(f"  KL(IS||cal_IS_propuesto):  {kl_cal:.3f}  <- solucion 3")
print()
print("  IMPLEMENTACION:")
print("  - El isotonic calibrator en train_xgboost_v2.py L2578+ usa features_validation.parquet")
print("  - NUEVO: usar los ultimos 12m del IS (subset de features_train.parquet) para isotonic")
print("  - El threshold sweep Optuna (_calibrate_threshold) MANTIENE features_validation.parquet")
print("  - El OOD guard MANTIENE features_validation.parquet")
print()
print("  RIESGO RESIDUAL:")
print("  - Los 12m del IS (2024-04 a 2025-04) tampoco son perfectos: BULL=84%")
print("  - PERO: el modelo los vio en train -> scores IS inflados -> NO ideal para isotonic")
print("  - La solucion correcta real ES la solucion 2 (CPCV OOB)")
print()
print("  VEREDICTO: MEJOR QUE ACTUAL pero no perfecto")

# ── SOLUCION 4 (real): Separar threshold sweep y calibracion isotonica ──────
print("\n" + "="*70)
print("SOLUCION 4 (CORRECTA FINAL): Separar las dos funciones del validation")
print("-"*70)
print("""
  INSIGHT CLAVE: el validation hace DOS cosas distintas con distintos requisitos:

  A) Threshold sweep (Optuna): necesita datos RECIENTES y VERDADERAMENTE OOS
     -> features_validation.parquet es CORRECTO para esto aunque sea 100% BULL
     -> El threshold del agente BULL se calibra en BULL OOS reciente = OK
     -> El threshold del agente RANGE se calibra con 0 datos RANGE = PROBLEMA

  B) Calibracion isotonica/Platt: necesita datos MULTI-REGIMEN y REPRESENTATIVOS
     -> features_validation.parquet 100% BULL es CATASTROFICO para esto
     -> El calibrador para RANGE ve 0 barras RANGE -> colapso

  FIX CORRECTO:
    - Threshold sweep (Optuna): sigue usando features_validation.parquet
    - Calibracion isotonica: usar CPCV OOB scores (solucion 2) o IS_CAL_BLOCK

  IMPLEMENTACION EN CODIGO:
    train_xgboost_v2.py L2574 (save_and_calibrate -> _do_isotonic_calibration):
      ANTES: usa features_validation.parquet
      DESPUES: usa CPCV OOB scores del mejor trial Optuna (ya disponibles en memoria)

  COMPLEJIDAD: Media - requiere exponer los OOB scores del callback Optuna
  IMPACTO: Alto - el isotonic calibrador pasa de 1441 barras BULL a ~67000 barras multi-regimen
""")
print("="*70)
print("[ARCH-05] PLAN DE IMPLEMENTACION RECOMENDADO")
print("  1. Exponer OOB scores del mejor trial Optuna (nueva variable en XGBoostAgent)")
print("  2. En _do_isotonic_calibration: si OOB scores disponibles, usarlos")
print("  3. Si OOB no disponibles: fallback a features_validation.parquet (actual)")
print("  4. Log explicito: [ARCH-05-FIX] Isotonic calibrator usando CPCV OOB")
