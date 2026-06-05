"""
Test unitario para _validate_calibrator_cache.
Verifica que:
1. Detecta calibrador con std=0 (corrupto) y lo elimina
2. Acepta calibrador isotónico con varianza real
3. El import de _RFWithAdapter funciona correctamente
"""
import sys
sys.path.insert(0, r"g:\Mi unidad\ia\luna_v2")

import joblib, numpy as np, tempfile, os
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

print("=" * 60)
print("TEST: _validate_calibrator_cache")
print("=" * 60)

# 1. Test import de clases top-level
print("\n[1] Importando clases top-level del calibrador...")
try:
    from luna.models.calibrate_probabilities import _RFWithAdapter, _IdentityWrapper, _TSAdapter
    print("  OK: _RFWithAdapter, _IdentityWrapper, _TSAdapter importados")
except Exception as e:
    print(f"  FALLO: {e}")
    sys.exit(1)

# 2. Simular calibrador CORRUPTO (std=0)
print("\n[2] Simulando calibrador corrupto (std=0 en output)...")
_ir_bad = IsotonicRegression(out_of_bounds="clip")
_x = np.linspace(0.3, 0.7, 50)
_y = np.full(50, 0.594)  # etiquetas constantes → isotónico también constante
_ir_bad.fit(_x, _y)
_out_bad = _ir_bad.predict(_x)
print(f"  std del calibrador corrupto: {np.std(_out_bad):.8f}")
assert np.std(_out_bad) < 1e-4, "Test: el calibrador simulado debería tener std≈0"
print("  CORRECTO: el calibrador corrupto produce std=0")

# 3. Simular calibrador SANO (std>0)
print("\n[3] Simulando calibrador sano (varianza real)...")
_ir_ok = IsotonicRegression(out_of_bounds="clip")
_y_real = (_x > 0.5).astype(float)  # etiquetas binarias → salida variada
_ir_ok.fit(_x, _y_real)
_out_ok = _ir_ok.predict(_x)
print(f"  std del calibrador sano: {np.std(_out_ok):.4f}")
assert np.std(_out_ok) > 1e-4, "Test: el calibrador sano debería tener std>0"
print("  CORRECTO: el calibrador sano produce varianza real")

# 4. Simular hydrate + validate en directorio temporal
print("\n[4] Simulando hydrate + validate_calibrator_cache...")
with tempfile.TemporaryDirectory() as tmpdir:
    ws_dir = Path(tmpdir) / "models"
    cache_dir = Path(tmpdir) / "cache_models"
    ws_dir.mkdir(); cache_dir.mkdir()
    
    # Guardar calibrador corrupto
    _bad_path = ws_dir / "metalabeler_v2_long_calibrator.joblib"
    _bad_cache = cache_dir / "metalabeler_v2_long_calibrator.joblib"
    joblib.dump(_ir_bad, _bad_path)
    joblib.dump(_ir_bad, _bad_cache)
    
    # Importar la función
    sys.path.insert(0, r"g:\Mi unidad\ia\luna_v2")
    from luna.pipeline_executor import _validate_calibrator_cache
    from loguru import logger
    
    _validate_calibrator_cache(ws_dir, cache_dir, "W1", "42")
    
    # Verificar que eliminó el corrupto
    if not _bad_path.exists():
        print("  CORRECTO: calibrador corrupto eliminado del workspace")
    else:
        print("  FALLO: calibrador corrupto NO fue eliminado del workspace")
    if not _bad_cache.exists():
        print("  CORRECTO: calibrador corrupto eliminado de la caché")
    else:
        print("  FALLO: calibrador corrupto NO fue eliminado de la caché")
    
    # Guardar calibrador sano
    _ok_path = ws_dir / "metalabeler_v2_long_calibrator.joblib"
    _ok_cache = cache_dir / "metalabeler_v2_long_calibrator.joblib"
    joblib.dump(_ir_ok, _ok_path)
    joblib.dump(_ir_ok, _ok_cache)
    
    _validate_calibrator_cache(ws_dir, cache_dir, "W1", "42")
    
    if _ok_path.exists():
        print("  CORRECTO: calibrador sano NO fue eliminado del workspace")
    else:
        print("  FALLO: calibrador sano FUE eliminado incorrectamente")

print("\n" + "=" * 60)
print("TEST COMPLETADO")
print("=" * 60)
