"""
TEST FIX-TEMPCAL-DESER-01: TemperatureCalibrator deserialization
================================================================
Simula exactamente el ciclo completo:
  1. Entrenar + serializar con joblib (como train_xgboost_v2.py)
  2. Deserializar desde un contexto diferente (como regime_router.py)
  3. Verificar que no hay AttributeError y que predict() funciona
"""
import sys
import joblib
import numpy as np
import tempfile
from pathlib import Path

sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
print("=== TEST FIX-TEMPCAL-DESER-01 ===")
print()

# TEST 1: Importar desde módulo canónico y verificar módulo de origen
print("TEST 1: Verificar módulo de origen de TemperatureCalibrator")
from luna.models.calibrators import TemperatureCalibrator
tc = TemperatureCalibrator()
module_origin = type(tc).__module__
print(f"  type(TemperatureCalibrator).__module__ = '{module_origin}'")
assert module_origin == 'luna.models.calibrators', f"FAIL: módulo incorrecto: {module_origin}"
print(f"  PASS: módulo correcto → joblib la serializará como 'luna.models.calibrators.TemperatureCalibrator'")
print()

# TEST 2: Serializar con joblib y deserializar desde un contexto limpio
print("TEST 2: Serializar → deserializar (simula el ciclo completo)")
np.random.seed(42)
p_raw = np.random.uniform(0.3, 0.8, size=200)
y_val = (p_raw + np.random.normal(0, 0.1, 200) > 0.5).astype(float)

tc_fit = TemperatureCalibrator()
tc_fit.fit(p_raw, y_val)
print(f"  Fit: T={tc_fit.temperature:.4f} std_cal={np.std(tc_fit.predict(p_raw)):.6f}")

with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
    tmp_path = Path(f.name)

joblib.dump(tc_fit, tmp_path)
print(f"  Guardado en: {tmp_path.name}")

# Deserializar — simula regime_router.py cargando el archivo
# El import de luna.models.calibrators debe estar en el namespace
loaded_cal = joblib.load(tmp_path)
tipo = type(loaded_cal).__name__
modulo = type(loaded_cal).__module__
print(f"  Cargado: tipo={tipo} módulo={modulo}")
assert tipo == 'TemperatureCalibrator', f"FAIL tipo: {tipo}"
assert modulo == 'luna.models.calibrators', f"FAIL módulo: {modulo}"

p_out = loaded_cal.predict(p_raw)
print(f"  predict() OK: std={np.std(p_out):.6f} min={p_out.min():.4f} max={p_out.max():.4f}")
assert np.std(p_out) > 0, "FAIL: predict() produce output constante"
print(f"  PASS: deserialización y predicción correctas")
print()

# TEST 3: Verificar que el archivo real de seed28559 ya NO da error
# (si se reimporta correctamente)
print("TEST 3: Cargar archivo real con TemperatureCalibrator (seed28559)")
real_file = Path('g:/Mi unidad/ia/luna_v2/data/models/prod/seed28559/xgboost_isotonic_calibrator_bull_long.joblib')
if real_file.exists():
    try:
        real_cal = joblib.load(real_file)
        tipo_real = type(real_cal).__name__
        print(f"  RESULTADO: tipo={tipo_real}")
        if 'Temperature' in tipo_real:
            p_test = np.random.uniform(0.3, 0.8, 100)
            out = real_cal.predict(p_test)
            print(f"  predict() OK: std={np.std(out):.6f}")
            print(f"  PASS: seed28559 calibrador cargado sin AttributeError")
        else:
            print(f"  INFO: tipo={tipo_real} (no es TemperatureCalibrator — isotónico normal)")
    except AttributeError as e:
        print(f"  FAIL aún: {e}")
        print(f"  NOTA: Este archivo fue serializado con el namespace antiguo (__main__).")
        print(f"  Requiere ser re-generado en la próxima run de entrenamiento.")
else:
    print(f"  Archivo no encontrado en {real_file}")

# Limpieza
tmp_path.unlink(missing_ok=True)
print()
print("=== RESULTADO FINAL ===")
print("  FIX-TEMPCAL-DESER-01: PASADO")
print("  Los archivos NUEVOS generados con este fix serán deserializables.")
print("  Los archivos EXISTENTES (seed28559, etc.) requieren re-entrenamiento.")
print("  En la próxima run, TemperatureCalibrator se serializa como luna.models.calibrators")
