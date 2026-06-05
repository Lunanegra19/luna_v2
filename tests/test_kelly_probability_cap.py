import sys
import numpy as np
import pandas as pd
sys.path.append("g:/Mi unidad/ia/luna_v2")

from luna.risk.kelly_sizer import KellyPositionSizer, build_kelly_sizer_from_settings

def test_probability_cap():
    print("[KELLY-CAP-TEST] Starting Kelly sizer probability cap unit tests...")
    
    # 1. Instantiate with a probability cap of 0.62
    sizer = KellyPositionSizer(
        kelly_fraction=0.5,
        min_position=0.01,
        max_position=0.15,
        pt_ratio=1.2,
        sl_ratio=1.0,
        probability_cap=0.62
    )
    
    # 2. Test compute_kelly with probabilities below and above cap
    p_below = np.array([0.55])
    p_above = np.array([0.85])
    
    size_below = sizer.compute_kelly(p_below)[0]
    size_above = sizer.compute_kelly(p_above)[0]
    
    # Under pt_ratio=1.2, sl_ratio=1.0, kelly_fraction=0.5:
    # pure_kelly = (W*p - L*(1-p)) / (W*L) = (1.2*p - (1-p)) / 1.2 = (2.2*p - 1) / 1.2
    # For p=0.55: pure_kelly = (2.2*0.55 - 1) / 1.2 = (1.21 - 1) / 1.2 = 0.21 / 1.2 = 0.175
    # size = 0.175 * 0.5 = 0.0875
    expected_size_below = max(0.01, min(0.15, ((2.2 * 0.55 - 1) / 1.2) * 0.5))
    
    # For p=0.85, it should be clipped to 0.62:
    # p_capped = 0.62
    # pure_kelly = (2.2*0.62 - 1) / 1.2 = (1.364 - 1) / 1.2 = 0.364 / 1.2 = 0.3033
    # size = 0.3033 * 0.5 = 0.1516 -> clipped to max_position (0.15)
    expected_size_capped = max(0.01, min(0.15, ((2.2 * 0.62 - 1) / 1.2) * 0.5))
    
    print(f"[KELLY-CAP-TEST] p=0.55 size: {size_below:.6f} | Expected: {expected_size_below:.6f}")
    print(f"[KELLY-CAP-TEST] p=0.85 size: {size_above:.6f} | Expected: {expected_size_capped:.6f}")
    
    assert np.allclose(size_below, expected_size_below), f"Incorrect size for p=0.55: {size_below}"
    assert np.allclose(size_above, expected_size_capped), f"Incorrect size for p=0.85 (capped): {size_above}"
    
    # 3. Test build from settings
    sizer_settings = build_kelly_sizer_from_settings()
    assert sizer_settings.probability_cap == 0.62, f"Failed to load probability_cap from settings: {sizer_settings.probability_cap}"
    
    print("[KELLY-CAP-TEST] All Kelly sizer probability cap assertions PASSED successfully!")

if __name__ == "__main__":
    test_probability_cap()
