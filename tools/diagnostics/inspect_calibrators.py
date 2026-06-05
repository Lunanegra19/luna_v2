import sys
sys.path.insert(0, '/root/luna_v2')
import joblib
import numpy as np

# Necesario para unpickle de calibradores luna
try:
    from luna.models.calibrate_probabilities import _RFWithAdapter, _IdentityWrapper, _TSAdapter
    import __main__
    __main__._RFWithAdapter = _RFWithAdapter
    __main__._IdentityWrapper = _IdentityWrapper
    __main__._TSAdapter = _TSAdapter
    print("[OK] _RFWithAdapter importado correctamente desde calibrate_probabilities")
except Exception as e:
    print(f"[WARN] No se pudo importar _RFWithAdapter: {e}")


# Inspect seed 1337 calibrator
cal_1337 = joblib.load('/root/luna_v2/data/models/prod/seed1337/metalabeler_v2_long_calibrator.joblib')
cal_99   = joblib.load('/root/luna_v2/data/models/prod/seed99/metalabeler_v2_long_calibrator.joblib')

print("=== Seed 1337 Calibrator ===")
print("Type:", type(cal_1337).__name__)
if hasattr(cal_1337, 'X_thresholds_'):
    print("Isotonic X_thresholds range:", cal_1337.X_thresholds_.min(), "->", cal_1337.X_thresholds_.max())
    print("Isotonic y_thresholds range:", cal_1337.y_thresholds_.min(), "->", cal_1337.y_thresholds_.max())
    print("Thresholds (first 10):", list(zip(cal_1337.X_thresholds_[:10].round(4), cal_1337.y_thresholds_[:10].round(4))))

test_vals = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
out_1337 = cal_1337.predict(test_vals)
print("Test input -> 1337 calibrator output:")
for x, y in zip(test_vals, out_1337):
    print(f"  {x:.2f} -> {y:.6f}")

print()
print("=== Seed 99 Calibrator ===")
print("Type:", type(cal_99).__name__)
out_99 = cal_99.predict(test_vals)
print("Test input -> 99 calibrator output:")
for x, y in zip(test_vals, out_99):
    print(f"  {x:.2f} -> {y:.6f}")
