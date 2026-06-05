import sys
import numpy as np
import pandas as pd
sys.path.append("g:/Mi unidad/ia/luna_v2")

from luna.models.train_xgboost_v2 import PlattCalibrator

def test_platt_calibrator_fallback():
    print("[LUNA-V2-CALIB-TEST] Starting PlattCalibrator unit tests...")
    
    # 1. Generate fake target y (binary) and raw model outputs p_raw
    np.random.seed(42)
    n_samples = 500  # < 1000 to trigger sigmoid/Platt fallback
    
    # Sigmoid function for generating realistic probabilities
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    x = np.random.normal(0, 1.5, n_samples)
    p_raw = sigmoid(x + np.random.normal(0, 0.5, n_samples))
    y = (x > 0).astype(int)
    
    # 2. Instantiate and fit PlattCalibrator
    calibrator = PlattCalibrator()
    calibrator.fit(p_raw, y)
    p_cal = calibrator.predict(p_raw)
    
    # 3. Assertions
    std_cal = np.std(p_cal)
    print(f"[LUNA-V2-CALIB-TEST] Calibrated probabilities std = {std_cal:.6f}")
    assert std_cal >= 0.010, f"Collapsed probability variance: std={std_cal:.6f} < 0.010"
    
    brier_raw = np.mean((p_raw - y) ** 2)
    brier_cal = np.mean((p_cal - y) ** 2)
    print(f"[LUNA-V2-CALIB-TEST] Brier Raw: {brier_raw:.6f} | Brier Calibrated: {brier_cal:.6f}")
    
    # Sigmoid calibration usually improves Brier score or stays close for valid probabilities
    assert brier_cal < 0.25, f"Brier score too high: {brier_cal:.6f}"
    
    # 4. Check boundaries and range
    assert p_cal.min() >= 0.0 and p_cal.max() <= 1.0, f"Calibrated probabilities out of bounds: [{p_cal.min()}, {p_cal.max()}]"
    
    print("[LUNA-V2-CALIB-TEST] All PlattCalibrator assertions PASSED successfully!")

if __name__ == "__main__":
    test_platt_calibrator_fallback()
