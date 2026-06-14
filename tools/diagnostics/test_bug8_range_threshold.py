"""
DIAGNÓSTICO AISLADO: test_bug8_range_threshold.py
BUG-8: Agente range_long tiene Reliability Diagram invertido en W1.
       El Isotonic calibrator comprime probabilidades hacia [0.40-0.50].
       Señales con prob=0.6-0.8 tienen WR=97.3% — ¿cuántas pasan el threshold?

Hipótesis:
  - Threshold actual = 0.55 (meta_filter_threshold_range o meta_filter_threshold)
  - Reliability Diagram (post-isotonic, seed42 W1):
      [0.0-0.2] n=12  WR=0.0%
      [0.2-0.4] n=682 WR=44.0%  ← subestimado (modelo dice 30%, real 44%)
      [0.4-0.6] n=672 WR=65.8%  ← gran oportunidad perdida
      [0.6-0.8] n=75  WR=97.3%  ← prácticamente perfecto
  - GUARDIAN-01: Q0(bottom)=46.5% | Q4(top)=59.6% → spread=13.1pp positivo

Pregunta 1: ¿Con threshold=0.55 cuántas señales del bucket [0.4-0.6] (WR=65.8%) se capturan?
  → Si el bucket es 0.4-0.6, las señales con prob > 0.55 están en la mitad superior.
  → n_bucket=672, asumiendo distribución uniforme en [0.4-0.6]: n_over_055 ≈ 672*0.25 = 168

Pregunta 2: ¿Cuál es el threshold óptimo IS que maximiza WR sin overfitting?
  → Necesitamos curva WR vs threshold usando los datos Reliability del Isotonic.

Pregunta 3: ¿El fix de bajar threshold crea riesgo de look-ahead o contaminación IS→OOS?
  → El threshold se aplica en OOS, el calibrador entrena en IS/validation. OK.
"""

import sys
sys.path.insert(0, 'c:/Users/Usuario/Desktop/ia/luna_v2')
import numpy as np
import pandas as pd

print("=" * 65)
print("TEST BUG-8: Reliability Diagram agente range_long W1 seed42")
print("=" * 65)

# Datos exactos del log de train_xgboost W1 seed42:
# [CAL-DIAG-01] Post-Isotonic calibration reliability diagram:
reliability_data = [
    {'bin': '[0.0-0.2]', 'center': 0.10, 'n': 12,  'pred_prob': 0.198, 'wr_real': 0.000},
    {'bin': '[0.2-0.4]', 'center': 0.30, 'n': 682,  'pred_prob': 0.295, 'wr_real': 0.440},
    {'bin': '[0.4-0.6]', 'center': 0.50, 'n': 672,  'pred_prob': 0.504, 'wr_real': 0.658},
    {'bin': '[0.6-0.8]', 'center': 0.70, 'n': 75,   'pred_prob': 0.627, 'wr_real': 0.973},
]
df_rel = pd.DataFrame(reliability_data)

print("\n--- RELIABILITY DIAGRAM POST-ISOTONIC (seed42 W1, agente range) ---")
print(f"{'Bin':<12} {'N':>6} {'Pred prob':>10} {'WR real':>10} {'Delta WR-50%':>12} {'Edge neto':>12}")
print("-" * 65)
for _, row in df_rel.iterrows():
    delta = row['wr_real'] - 0.50
    edge_neto = row['wr_real'] * row['n']
    print(f"{row['bin']:<12} {row['n']:>6} {row['pred_prob']:>10.1%} {row['wr_real']:>10.1%} {delta:>+12.1%} {edge_neto:>12.0f}")

print()
print("--- ANÁLISIS DE THRESHOLD ---")
print()

# Simulación: ¿qué pasa con distintos thresholds?
# Suponer distribución uniforme de prob dentro de cada bucket
# Esto es una aproximación conservadora

thresholds = [0.40, 0.45, 0.50, 0.55, 0.58, 0.60]

print(f"{'Threshold':>10} {'N capturado':>12} {'WR esperado':>12} {'Trades ganados':>15} {'Cobertura':>10}")
print("-" * 65)

for thr in thresholds:
    n_captured = 0
    wins = 0
    for _, row in df_rel.iterrows():
        lo, hi = float(row['bin'][1:4]), float(row['bin'][5:8])
        if hi <= thr:
            continue  # todo el bucket debajo del threshold
        if lo >= thr:
            # todo el bucket por encima
            frac = 1.0
        else:
            # threshold corta el bucket
            frac = (hi - thr) / (hi - lo)
        n_in = row['n'] * frac
        n_captured += n_in
        wins += n_in * row['wr_real']

    wr_exp = wins / n_captured if n_captured > 0 else 0
    cobertura = n_captured / df_rel['n'].sum()
    label = " ← ACTUAL" if abs(thr - 0.55) < 0.001 else ""
    label2 = " ← PROPUESTO" if abs(thr - 0.45) < 0.001 else ""
    print(f"{thr:>10.2f} {n_captured:>12.0f} {wr_exp:>12.1%} {wins:>15.0f} {cobertura:>10.1%}{label}{label2}")

print()
print("--- CONCLUSIÓN ---")
print()
print("Threshold=0.55 captura principalmente bucket [0.4-0.6] mitad alta + [0.6-0.8]")
print("  → WR esperado: ~69% (mitad superior de 65.8% + 97.3%)")
print()
print("Threshold=0.45 captura [0.4-0.6] completo + [0.6-0.8]")
print("  → WR esperado: ~68% (65.8% * masa mayor + 97.3%)")
print("  → N capturado significativamente mayor (672 vs ~336)")
print()
print("⚠️  IMPORTANTE: El threshold=0.55 ya debería capturar las señales [0.6-0.8]")
print("    WR=97.3%. Si no las captura es porque el Isotonic comprimió las probs.")
print()
print("VERIFICACIÓN CLAVE: buscar en logs cuántas señales range tienen prob > 0.60")
print("  Si son pocas (<5), el Isotonic está comprimiendo todo hacia [0.40-0.55]")
print("  → El fix real es re-examinar la calibración Isotonic, no solo el threshold.")
print()

# Verificar la compresión del calibrador
print("--- ANÁLISIS DE COMPRESIÓN ISOTONIC ---")
n_total = df_rel['n'].sum()
n_above_06 = df_rel[df_rel['center'] >= 0.65]['n'].sum()
n_above_055 = df_rel[df_rel['center'] >= 0.50]['n'].sum()  # mitad de [0.4-0.6] + [0.6-0.8]
print(f"Total señales validation: {n_total}")
print(f"Con prob_calibrada ≥ 0.60: {n_above_06} ({n_above_06/n_total:.1%})")
print(f"Con prob_calibrada ≥ 0.55 (aprox): ~{int(n_above_06 + 672*0.25)} ({(n_above_06 + 672*0.25)/n_total:.1%})")
print()
print(f"El Isotonic comprime: {n_total - n_above_06}/{n_total} = {(n_total-n_above_06)/n_total:.1%} señales debajo de 0.60")
print()
print("HIPÓTESIS REVISADA BUG-8:")
print("  El problema NO es el threshold — es que el Isotonic calibra")
print("  comprimiendo las probs hacia [0.40-0.55], haciendo que señales")
print("  con WR=97.3% queden a 0.627 (justo encima de threshold=0.55).")
print("  Con n=75 en ese bucket, pasan pocas señales de alta calidad.")
print()
print("FIX CORRECTO (prioridad):")
print("  1. Verificar cuántas señales range pasan el threshold en OOS W1")
print("     (ya sabemos: de 1283 XGB raw, el funnel quedó en 0 con min_pf=1.0)")
print("     Con min_pf=0.0 (fix ya aplicado) hay más candidatos.")
print("  2. El threshold range en 0.55 es razonable dado el Reliability Diagram.")
print("  3. La acción más impactante: resolver discriminación pobre (std<0.02)")
print("     que hace que la mayoría de señales tenga prob ≈ 0.45-0.55 → bloqueo masivo.")
