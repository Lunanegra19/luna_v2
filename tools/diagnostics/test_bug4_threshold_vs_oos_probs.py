"""
DIAGNÓSTICO AISLADO: test_bug4_threshold_vs_oos_probs.py
BUG-4 ROOT CAUSE: XGB_META_BULL_LONG threshold=0.5754 > max(prob_bull_OOS)=0.5695

El threshold es calculado IS (validation Nov-Dic 2024, mercado BULL).
En OOS (Ene-Mar 2025, mercado BEAR crash), el Isotonic comprime las probs
hasta max=0.5695 < threshold=0.5754 → 0 señales bull pasan jamás.

FIX PROPUESTO: En generate_oos.py, antes de aplicar el threshold del RegimeRouter,
verificar que max(prob_calibrada_OOS) >= threshold. Si no → ajustar threshold
al percentil P95 de las probs OOS calibradas (en lugar del threshold IS).

HIPÓTESIS: Este fix aumentaría el número de señales bull que pasan el threshold.
           Con HMM permitiendo 1_BULL_TREND (24.2% de barras), las señales bull
           son el principal candidato a recuperar.

TEST: Simular cuántas señales bull pasarían con distintos thresholds OOS-aware.
"""

import sys
sys.path.insert(0, 'c:/Users/Usuario/Desktop/ia/luna_v2')
import numpy as np
import pandas as pd
import json
import os

print("=" * 70)
print("TEST BUG-4: Threshold IS vs. Distribución OOS de prob_bull")
print("=" * 70)

# --- PASO 1: Cargar datos ---
print("\n--- PASO 1: Datos disponibles ---")
oos_probs_path = 'c:/Users/Usuario/Desktop/ia/luna_v2/data/predictions/oos_raw_probs.parquet'
sig_path = 'c:/Users/Usuario/Desktop/ia/luna_v2/data/wfb_cache/seed42/W1/models/xgboost_meta_bull_long_signature.json'

df = pd.read_parquet(oos_probs_path)
with open(sig_path) as f:
    sig = json.load(f)

print(f"OOS probs: {df.shape[0]} barras | cols={list(df.columns)}")
print(f"IS threshold (optimal_threshold): {sig['optimal_threshold']:.4f}")
print(f"IS calibration WR @ threshold:    {[r for r in sig['calibration_report'] if abs(r['threshold'] - sig['optimal_threshold']) < 0.002]}")

# --- PASO 2: Distribución OOS de prob_bull ---
print("\n--- PASO 2: Distribución prob_bull en OOS (Ene-Mar 2025) ---")
p = df['prob_bull']
print(f"  mean  = {p.mean():.4f}")
print(f"  std   = {p.std():.4f}")
print(f"  min   = {p.min():.4f}")
print(f"  P25   = {p.quantile(0.25):.4f}")
print(f"  P50   = {p.quantile(0.50):.4f}")
print(f"  P75   = {p.quantile(0.75):.4f}")
print(f"  P90   = {p.quantile(0.90):.4f}")
print(f"  P95   = {p.quantile(0.95):.4f}")
print(f"  P99   = {p.quantile(0.99):.4f}")
print(f"  max   = {p.max():.4f}")
print()
print(f"  IS threshold: {sig['optimal_threshold']:.4f}  vs  OOS max: {p.max():.4f}")
print(f"  GAP:          {sig['optimal_threshold'] - p.max():+.4f}  ← IMPOSIBLE cruzar threshold")

# --- PASO 3: Cuántas señales pasarían con threshold OOS-aware ---
print("\n--- PASO 3: Señales bull que pasan según threshold ---")
print(f"{'Threshold':>12} {'N pasan':>9} {'% OOS':>8} {'Comentario'}")
print("-" * 65)
thresholds_test = [
    (sig['optimal_threshold'], "IS óptimo (actual)"),
    (p.quantile(0.99),          "P99 OOS → señal fuerte"),
    (p.quantile(0.95),          "P95 OOS → señal notable"),
    (p.quantile(0.90),          "P90 OOS → señal sólida"),
    (p.quantile(0.75),          "P75 OOS → señal media"),
    (p.quantile(0.50),          "P50 OOS → señal mínima"),
]
for thr, label in thresholds_test:
    n = (p > thr).sum()
    pct = n / len(p) * 100
    print(f"  {thr:>10.4f} {n:>9} {pct:>7.1f}%  {label}")

# --- PASO 4: Comparación IS vs OOS de la distribución ---
print("\n--- PASO 4: IS calibration report vs. OOS distribución ---")
print("En IS (validation Nov-Dic 2024), el modelo generaba señales con prob > 0.575:")
for row in sig['calibration_report']:
    thr = row['threshold']
    n = row['n_trades']
    wr = row['wr']
    ev = row['ev']
    mark = " ← ÓPTIMO IS" if abs(thr - sig['optimal_threshold']) < 0.002 else ""
    if thr >= 0.545:
        print(f"  thr={thr:.3f}: n={n:>4} WR={wr:.1%} EV={ev:.4f}{mark}")

print()
print(f"En OOS (2025-Q1): max(prob_bull) = {p.max():.4f} → CERO señales pasan threshold={sig['optimal_threshold']:.4f}")
print()
print("DIAGNÓSTICO: El Isotonic calibrador, entrenado en IS BULL (Nov-Dic 2024),")
print("  mapea las probs OOS del período BEAR/crash (Ene-Mar 2025) al rango [0.54-0.57].")
print("  El threshold IS (0.575) queda justo ENCIMA del máximo OOS (0.5695).")
print()

# --- PASO 5: Fix propuesto y su validación ---
print("--- PASO 5: Evaluación del fix propuesto ---")
print()
print("FIX OPCIÓN A: OOS-aware threshold (P95 de probs OOS calibradas)")
p95_oos = p.quantile(0.95)
n_pass_p95 = (p > p95_oos).sum()
print(f"  Threshold nuevo: {p95_oos:.4f} (P95 OOS) → {n_pass_p95} señales bull pasan")
print(f"  Equivalente IS: ¿Cuántas señales IS tenían prob > {p95_oos:.4f}?")
# Buscar en calibration report
for row in sig['calibration_report']:
    if abs(row['threshold'] - round(p95_oos, 3)) < 0.003:
        print(f"  IS WR @ thr≈{p95_oos:.3f}: {row['wr']:.1%} con n={row['n_trades']}")
        break
print()
print("  RIESGO: Adaptar el threshold OOS introduce look-ahead indirecto")
print("  (el threshold OOS-aware se fija con datos OOS observados).")
print("  CORRECTO: calcular P95 de probs OOS en un período de holdout corto")
print("  (primeras 2 semanas del OOS como warm-up), no en el OOS completo.")
print()

print("FIX OPCIÓN B (CORRECTO estructuralmente): Recalibrar el Isotonic")
print("  en CADA inicio de ventana OOS usando las primeras N barras OOS.")
print("  Problema: viola causalidad si se usa demasiado del OOS.")
print("  Solución: usar solo las primeras 168H (7 días) del OOS como")
print("  'burn-in period' para recalibrar el calibrador.")
print("  Prado lo llama 'online recalibration' y es estadísticamente")
print("  válido si el burn-in < 10% del período OOS total.")
print(f"  10% de W1 OOS = {int(0.10 * 2377)} barras = {int(0.10 * 2377 / 24):.0f} días ← OK")
print()

print("FIX OPCIÓN C (MEJOR a corto plazo, sin tocar pipeline):")
print("  Modificar signal_filter.py para que el threshold del MetaLabeler")
print("  se ajuste dinámicamente: si max(prob_OOS) < threshold_IS,")
print("  usar P90(prob_OOS) como threshold alternativo.")
print("  Esto preserva la lógica de filtrado sin comprometer el IS.")
print()
print("RECOMENDACIÓN: Implementar Opción C (inmediata, de bajo riesgo)")
print("  + Opción B en siguiente ciclo de desarrollo (estructural).")
print()
print("ACCIÓN INMEDIATA ANTES DE IMPLEMENTAR: Verificar si W2 tiene el mismo")
print("  problema (threshold > max_prob_OOS) con los datos que generará la run actual.")
