"""
DIAGNÓSTICO AISLADO: test_bug4_agent_discrimination.py
BUG-4: Los 3 agentes MetaLabeler tienen std de probabilidades < 0.02 en OOS.
       Esto significa que producen probabilidades quasi-constantes (no discriminan).

Evidencia de logs seed42:
  bear_long:  std=0.0043 < 0.02 | n=435 barras W2
  bull_long:  std=0.0118 < 0.02 | n=534 barras W2
  range_long: std=0.0049 < 0.02 | n=424 barras W2

Hipótesis para investigar:
  A) ¿El MetaLabelerV2 (LSTM+RollingStats) produce probs constantes por diseño?
     → Verificar la distribución de input features (RollingStats con ventana pequeña)
  B) ¿Es un problema de calibración post-LSTM?
     → El LSTM produce probs, luego el Isotonic calibra. Si el LSTM ya colapsa → Isotonic no puede corregir.
  C) ¿El input_dim=22 incluye features redundantes o constantes en períodos cortos?
     → Revisar qué son las 22 features de RollingStatsExtractor
  D) ¿El problema es que el horizonte OOS de W1/W2 (2025-Q1/Q2) tiene distribución
     muy diferente al IS de entrenamiento (hasta 2024-10/01)?
     → Covariate shift MetaLabeler (relacionado con BUG-5 OOD)

Objetivo: determinar cuál de estas causas es la principal para proponer el fix correcto.
"""

import sys
sys.path.insert(0, 'c:/Users/Usuario/Desktop/ia/luna_v2')
import numpy as np
import pandas as pd
import os

print("=" * 65)
print("TEST BUG-4: Discriminación pobre de agentes MetaLabeler")
print("=" * 65)

# 1. Verificar si el modelo MetaLabelerV2 existe y cuál es su arquitectura
print("\n--- PASO 1: Verificar arquitectura MetaLabelerV2 ---")
model_paths = [
    'data/models/metalabeler_bull_long.pkl',
    'data/models/metalabeler_bear_long.pkl',
    'data/models/metalabeler_range_long.pkl',
]
for mp in model_paths:
    full_path = os.path.join('c:/Users/Usuario/Desktop/ia/luna_v2', mp)
    if os.path.exists(full_path):
        size_kb = os.path.getsize(full_path) // 1024
        print(f"  {mp}: {size_kb}KB")
    else:
        print(f"  {mp}: NO EXISTE")

# 2. Cargar el modelo bull_long y examinar su arquitectura
print("\n--- PASO 2: Cargar MetaLabelerV2 bull_long y examinar arquitectura ---")
try:
    import pickle
    model_path = 'c:/Users/Usuario/Desktop/ia/luna_v2/data/models/metalabeler_bull_long.pkl'
    with open(model_path, 'rb') as f:
        meta_model = pickle.load(f)

    print(f"  Tipo: {type(meta_model).__name__}")
    # Examinar atributos clave
    for attr in ['input_dim', 'hidden_dim', 'extractor', 'lstm', 'fc', 'threshold_', 'calibrator_']:
        val = getattr(meta_model, attr, None)
        if val is not None:
            if hasattr(val, '__len__') and not isinstance(val, str):
                print(f"  {attr}: {type(val).__name__} (len={len(val)})")
            else:
                print(f"  {attr}: {val}")
    
    # Ver el extractor de features
    extractor = getattr(meta_model, 'extractor', None)
    if extractor is not None:
        print(f"  Extractor tipo: {type(extractor).__name__}")
        for attr2 in ['windows', 'feature_names', 'n_features']:
            val2 = getattr(extractor, attr2, None)
            if val2 is not None:
                print(f"    extractor.{attr2}: {val2}")

except Exception as e:
    print(f"  ERROR cargando modelo: {e}")

# 3. Análisis matemático: ¿qué std de probs es normal para un LSTM con threshold?
print("\n--- PASO 3: Análisis matemático de std esperado ---")
print()
print("  Con un LSTM que produce probs en [0,1] centradas en ~0.5:")
print("  - Si el modelo NO discrimina: std ~ 0.05-0.10 (variación aleatoria)")
print("  - Si el modelo discrimina bien: std ~ 0.15-0.25")
print("  - std < 0.02 = modelo COLAPSA → proba casi idéntica para todos los inputs")
print()
print("  Evidencia:")
print("    bear:  std=0.0043 → rango total ≈ 0.017 (de ~0.466 a ~0.483)")
print("    bull:  std=0.0118 → rango total ≈ 0.047 (de ~0.469 a ~0.516)")
print("    range: std=0.0049 → rango total ≈ 0.020 (de ~0.490 a ~0.510)")
print()
print("  Un LSTM que produce probs en rango de 2-5% NO puede discriminar trade")
print("  de calidad variable. El threshold a 0.55 bloqueará TODOS los trades.")
print()

# 4. Identificar causas posibles
print("--- PASO 4: Hipótesis de causa raíz ---")
print()
causes = [
    ("A", "Covariate shift IS→OOS",
     "El LSTM entrenó en datos hasta 2024-10/01. En 2025-Q1/Q2 las\n"
     "     distribuciones de RollingStats features son muy diferentes.\n"
     "     El LSTM produce scores en zona muerta (all inputs → similar embedding)."),
    ("B", "Ventana de RollingStats demasiado larga",
     "Si RollingStatsExtractor usa window=240 o similar, en 2025-Q1/Q2\n"
     "     las estadísticas rolling del IS (entrenamiento) vs OOS (producción)\n"
     "     están en rangos distintos → el LSTM no reconoce el input."),
    ("C", "LSTM hidden_dim=32 insuficiente",
     "Con solo 32 neuronas ocultas y 22 features de entrada, el LSTM\n"
     "     puede saturarse y producir outputs planos para distribuciones OOS."),
    ("D", "Isotonic calibrator comprime probabilidades",
     "El Isotonic se entrena en IS. Si los inputs OOS están fuera del\n"
     "     rango IS, el calibrador mapea todos hacia la mediana (0.45-0.55)."),
]

for letter, title, desc in causes:
    print(f"  {letter}. {title}")
    print(f"     {desc}")
    print()

# 5. Verificar cuál causa es más probable verificando los datos de los logs
print("--- PASO 5: Evidencia de logs para discriminar causas ---")
print()
print("  De logs seed42 W2 (OOS 2025-Q2):")
print("    RegimeRouter asignó: bear=435 barras, bull=534 barras, range=424 barras")
print("    Todos con std < 0.02 → SIMULTÁNEO en todos los agentes")
print()
print("  Esto descarta causas B y C (serían específicas de un agente).")
print("  La causa más probable es A+D combinadas:")
print("    → Covariate shift 2024→2025 hace que el LSTM produce outputs planos")
print("    → El Isotonic, entrenado en IS, no puede corregir distribuciones OOS nuevas")
print()
print("--- PASO 6: Fix propuesto y verificación requerida ---")
print()
print("  FIX PRIORITARIO (sin implementar aún):")
print("  1. Verificar la distribución de features RollingStats en IS vs OOS")
print("     → Cargar oos_raw_probs.parquet y comparar con is_features")
print("     → Si hay shift > 2 std en la mayoría de features → CONFIRMADO causa A")
print()
print("  2. Si causa A confirmada: el fix es re-escalar las features OOS")
print("     usando las estadísticas del IS más reciente (última ventana de train)")
print("     → El extractor debe normalizar con mean/std del IS, no global")
print()
print("  3. Fix alternativo más simple: reducir el threshold de cada agente")
print("     a la mediana de las probs OOS (≈0.47-0.49) en lugar de 0.55")
print("     → RISK: overfitting a la distribución de la ventana")
print()
print("  ACCIÓN INMEDIATA DISPONIBLE: verificar oos_raw_probs.parquet")
print("  para ver la distribución real de probs MetaLabeler en OOS")

# 6. Intentar cargar oos_raw_probs.parquet para verificar distribución real
print()
print("--- PASO 7: Verificar distribución real en oos_raw_probs.parquet ---")
try:
    oos_probs_path = 'c:/Users/Usuario/Desktop/ia/luna_v2/data/predictions/oos_raw_probs.parquet'
    if os.path.exists(oos_probs_path):
        df_probs = pd.read_parquet(oos_probs_path)
        print(f"  Loaded: {df_probs.shape[0]} filas | Columnas: {list(df_probs.columns)}")
        # Buscar columnas de probabilidad MetaLabeler
        meta_cols = [c for c in df_probs.columns if 'meta' in c.lower() or 'prob' in c.lower() or 'bull' in c.lower() or 'bear' in c.lower()]
        print(f"  Columnas relevantes: {meta_cols[:10]}")
        for col in meta_cols[:5]:
            s = df_probs[col].dropna()
            if len(s) > 0:
                print(f"    {col}: mean={s.mean():.4f} std={s.std():.4f} min={s.min():.4f} max={s.max():.4f}")
    else:
        print("  oos_raw_probs.parquet no encontrado (W1 de la nueva run aún no terminó)")
        print("  → Este test debe re-ejecutarse cuando W1 de seed42 termine")
except Exception as e:
    print(f"  ERROR: {e}")
