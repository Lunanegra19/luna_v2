"""
DIAGNÓSTICO AISLADO: test_bug4_option_c_dynamic_threshold.py
BUG-4 ROOT CAUSE: Threshold IS > Max(Prob OOS) -> 0 señales.
Esto ocurre por covariate shift en las probabilidades generadas por el Isotonic
Regression entre el IS y el OOS.

FIX PROPUESTO (Opción C - Dynamic Fallback Threshold):
En `signal_filter.py`, si la probabilidad máxima observada en el batch OOS
es menor que el threshold calibrado en IS, el threshold IS es inalcanzable.
En este caso, debemos bajar el threshold dinámicamente al P90 (o un percentil alto)
de las probabilidades OOS para permitir que el 10% de las señales más fuertes
pasen, restaurando el embudo.

Este script simula este fallback con datos sintéticos basados en la realidad de W1.
"""

import numpy as np

print("=" * 70)
print("TEST BUG-4: Dynamic Threshold Fallback (Opción C)")
print("=" * 70)

# Datos observados en W1 (XGB_META_BULL_LONG)
threshold_is = 0.5754
max_prob_oos = 0.5695

# Simulamos 1000 probabilidades OOS comprimidas
np.random.seed(42)
oos_probs = np.random.normal(loc=0.52, scale=0.015, size=1000)
# Ajustamos para que max_prob_oos sea ~0.5695
oos_probs = np.clip(oos_probs, 0.0, max_prob_oos)

print(f"\n--- PASO 1: Diagnóstico de Colapso ---")
print(f"Threshold IS calibrado: {threshold_is:.4f}")
print(f"Max Probabilidad OOS observada: {np.max(oos_probs):.4f}")

# Lógica BUG-4 actual
n_signals_actual = np.sum(oos_probs >= threshold_is)
print(f"\nSeñales con lógica actual: {n_signals_actual} (BLOQUEO TOTAL)")

# Lógica Propuesta (Opción C)
print("\n--- PASO 2: Lógica Dynamic Fallback ---")
def apply_dynamic_threshold(probs, threshold_is, percentile=90):
    max_prob = np.max(probs)
    if max_prob < threshold_is:
        new_threshold = np.percentile(probs, percentile)
        print(f"⚠️ [BUG-4 FIX] Threshold IS ({threshold_is:.4f}) inalcanzable (Max={max_prob:.4f}).")
        print(f"   Bajando dinámicamente al P{percentile} OOS -> Nuevo Threshold: {new_threshold:.4f}")
        return new_threshold
    return threshold_is

# Simulamos la aplicación
new_threshold = apply_dynamic_threshold(oos_probs, threshold_is, percentile=90)
n_signals_new = np.sum(oos_probs >= new_threshold)

print(f"\nSeñales con lógica nueva (P90): {n_signals_new}")

print("\n--- PASO 3: Validación OOS-Aware ---")
print("Este fix requiere que la función `apply_metalabeler` en signal_filter.py")
print("tenga acceso a TODAS las probabilidades de la ventana OOS a la vez (lo cual es")
print("cierto, ya que recibe un batch en un DataFrame).")
print("La lógica exacta a inyectar será:")
print("  max_prob = df['prob_bull'].max()")
print("  if max_prob < threshold:")
print("      threshold = df['prob_bull'].quantile(0.90)")

print("\nACCIÓN: Implementar en signal_filter.py (línea ~970) tras la run actual.")
