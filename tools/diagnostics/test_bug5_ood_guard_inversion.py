"""
DIAGNÓSTICO AISLADO: test_bug5_ood_guard_inversion.py
BUG-5: El OOD Guard (IsolationForest) tiene hipótesis invertida en 2025-2026.
       Las barras que el modelo llama "anómalas" generan MÁS retorno que las "normales".
       Delta documentado en informe auditoría: +12.8pp

Hipótesis a verificar:
  - IF entrenado en IS (datos hasta 2024-10/01)
  - El mercado 2025 tiene distribución diferente → muchas barras son "anómalas"
  - Pero esas barras "anómalas" son precisamente las de mercado 2025 normal → buenas señales
  - El OOD Guard bloquea las mejores señales del período OOS

Objetivo del test:
  1. Cuantificar cuántas señales XGB bloquea el OOD Guard en W1/W2 (seed42)
  2. Verificar si hay evidencia de WR diferencial entre bloqueadas y no-bloqueadas
  3. Estimar el impacto en número de trades si se desactivara el OOD Guard
"""

import sys
sys.path.insert(0, 'c:/Users/Usuario/Desktop/ia/luna_v2')
import numpy as np
import pandas as pd
import os

print("=" * 65)
print("TEST BUG-5: OOD Guard Hipótesis Invertida")
print("=" * 65)

# --- Paso 1: Evidencia de logs de bloqueo OOD ---
print("\n--- PASO 1: Evidencia de bloqueo OOD en logs seed42 ---")
print()
print("  Datos del funnel seed42 W1 (generate_oos 12:19:56):")
print("    XGB:       1236 señales")
print("    OOD-block:    0 señales bloqueadas (0%)")
print("    [FILTROS] XGB=1236 | OOD-block=0")
print()
print("  Datos del funnel seed42 W2 (generate_oos 12:32:12):")
print("    XGB:       1283 señales")
print("    OOD-block:  322 señales bloqueadas (25.1%)")
print("    [FILTROS] XGB=1283 | OOD-block=322")
print()
print("  Datos del informe auditoría seed63678 (5 ventanas):")
print("    After XGBoost:  8,762 señales")
print("    After OOD:      8,239 señales → -523 bloqueadas (6%)")
print()

# --- Paso 2: Impacto cuantificado si se desactiva OOD ---
print("--- PASO 2: Impacto estimado de desactivar OOD Guard ---")
print()

# Datos W2 seed42
xgb_total_w2 = 1283
ood_block_w2 = 322
no_ood_w2 = xgb_total_w2 - ood_block_w2
print(f"  W2 seed42: {ood_block_w2} señales bloqueadas por OOD ({ood_block_w2/xgb_total_w2:.1%})")
print(f"  Sin OOD:   {xgb_total_w2} señales llegarían al siguiente filtro (+{ood_block_w2} = +{ood_block_w2/no_ood_w2:.1%})")
print()

# El funnel W2 final fue 0 trades (con min_pf=1.0 bug)
# Ahora W2 debería tener más trades con min_pf=0.0
# El OOD bloqueaba 322/1283=25% → si el MetaLabeler pasa el 22% de lo restante → 322*0.22=71 trades más
ood_recovered_est = int(ood_block_w2 * 0.22)
print(f"  Estimación: ~{ood_recovered_est} trades adicionales si se desactiva OOD en W2")
print()

# --- Paso 3: Cuantificación matemática del riesgo de desactivar OOD ---
print("--- PASO 3: Riesgo de desactivar OOD Guard ---")
print()
print("  PRO (desactivar):")
print("    - En W2: recuperaría ~71 trades potenciales")
print("    - Delta WR histórico: +12.8pp (barras 'anómalas' > 'normales')")
print("    - El IF entrenado en IS 2022-2024 está desfasado 12+ meses")
print()
print("  CONTRA (desactivar):")
print("    - Pérdida del mecanismo de protección anti-distribución extrema")
print("    - Sin OOD, el modelo opera en días de eventos de cola (crashes, etc.)")
print("    - El +12.8pp podría ser específico de ese período OOS (overfitting de diagnóstico)")
print()

# --- Paso 4: Test estadístico con datos disponibles ---
print("--- PASO 4: Verificación con datos disponibles ---")
print()

# Intentar cargar oos_trades.parquet para ver si tiene columna OOD
oos_path = 'c:/Users/Usuario/Desktop/ia/luna_v2/data/predictions/oos_trades.parquet'
if os.path.exists(oos_path):
    df_trades = pd.read_parquet(oos_path)
    print(f"  Loaded oos_trades.parquet: {df_trades.shape[0]} trades")
    print(f"  Columnas: {list(df_trades.columns)}")
    ood_cols = [c for c in df_trades.columns if 'ood' in c.lower() or 'anomal' in c.lower() or 'isolation' in c.lower()]
    print(f"  Columnas OOD: {ood_cols}")
    if ood_cols:
        for col in ood_cols:
            print(f"  {col}: {df_trades[col].value_counts().to_dict()}")
else:
    print("  oos_trades.parquet no encontrado (run en curso)")

# Intentar leer señal funnel para ver detalle OOD
funnel_path = 'c:/Users/Usuario/Desktop/ia/luna_v2/data/reports/signal_funnel.json'
if os.path.exists(funnel_path):
    import json
    with open(funnel_path) as f:
        funnel = json.load(f)
    print()
    print("  signal_funnel.json disponible:")
    for key, val in list(funnel.items())[:15]:
        print(f"    {key}: {val}")
else:
    print("  signal_funnel.json no encontrado")

print()
print("--- PASO 5: CONCLUSIÓN Y RECOMENDACIÓN ---")
print()
print("  EVIDENCIA DISPONIBLE:")
print("    W1: OOD-block=0 (OOD Guard no bloqueó nada en W1)")
print("    W2: OOD-block=322/1283 = 25.1% (bloqueó 1 de cada 4 señales XGB)")
print()
print("  DIAGNÓSTICO:")
print("    En W1 (2025-Q1, mercado BEAR), el OOD no bloquea nada → coherente")
print("    (mercado BEAR 2025 es similar al IS 2022-2024 que también tuvo BEARs)")
print("    En W2 (2025-Q2, mercado BULL recovery post-crash), el OOD bloquea 25%")
print("    → El IF entrena en IS con distribución mixta, pero el BULL recovery de")
print("      2025-Q2 tiene características técnicas distintas (new ATH momentum)")
print("      → El IF llama 'anómalas' a esas barras")
print()
print("  TEST PENDIENTE (requiere datos post-run):")
print("    1. Calcular WR de señales bloqueadas por OOD vs no bloqueadas")
print("       (necesita guardar en oos_raw_probs.parquet el score OOD de cada barra)")
print("    2. Comparar IS vs OOS distributions de las features OOD")
print()
print("  RECOMENDACIÓN:")
print("    FIX CORRECTO: Reentrenar IsolationForest en cada ventana WFB")
print("    usando los datos del train reciente (no el IF global histórico).")
print("    Esto es el fix estructural. El IF dinámico por ventana elimina el")
print("    covariate shift por diseño.")
print()
print("    FIX INMEDIATO (diagnóstico): Añadir oos_ood_score columna a")
print("    oos_raw_probs.parquet para poder cuantificar el delta WR OOD.")
print("    Requiere 1 línea en generate_oos.py.")
print()
print("    NO RECOMENDADO: Invertir el filtro OOD (OR desactivarlo completamente)")
print("    sin validación prospectiva. Demasiado riesgo de survivorship bias.")
