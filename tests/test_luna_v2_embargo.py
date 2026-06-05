import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append("g:/Mi unidad/ia/luna_v2")

from luna.models.signal_filter import SignalFilter

def test_luna_v2_volatility_decaying_embargo():
    print("[LUNA-V2-EMBARGO-TEST] Starting Volatility Decaying Embargo unit tests...")
    
    # Temporarily override settings for decay testing
    from config.settings import cfg as _cfg_test
    _orig_floor = _cfg_test.xgboost.embargo_hours
    _orig_sop = _cfg_test.sop.embargo_hours
    _cfg_test.xgboost.embargo_hours = 24.0
    _cfg_test.sop.embargo_hours = 168.0
    
    # 1. Initialize SignalFilter with dummy Path
    sf = SignalFilter(models_dir=Path("config")) # we can pass config as it exists
    
    # 2. Build mock dataset representing a crash spike followed by calm period
    # 600 hours (25 days)
    dates = pd.date_range("2026-01-01", periods=600, freq="h")
    
    # Generate prices with high volatility in the first half, decaying in the second half
    close = []
    high = []
    low = []
    
    current_price = 50000.0
    np.random.seed(42)
    
    for i in range(600):
        if i < 200:
            # Crash spike: extreme volatility
            swing = np.random.normal(0, 1000.0)
            h = current_price + abs(swing) + 500
            l = current_price - abs(swing) - 500
            c = current_price + swing
        else:
            # Volatility decay: extremely calm
            swing = np.random.normal(0, 50.0)
            h = current_price + abs(swing) + 20
            l = current_price - abs(swing) - 20
            c = current_price + swing
            
        close.append(c)
        high.append(h)
        low.append(l)
        current_price = c
        
    df = pd.DataFrame({
        "close": close,
        "high": high,
        "low": low,
        "HMM_Semantic": "3_BEAR_CRASH"  # Base embargo = 168.0H
    }, index=dates)
    
    # 3. Create a dense signal mask: candidate signals every 4 hours
    signal_mask = pd.Series(False, index=dates)
    signal_mask.iloc[::4] = True
    n_candidates = signal_mask.sum()
    print(f"[LUNA-V2-EMBARGO-TEST] Dense candidates count: {n_candidates}")
    assert n_candidates >= 20, "Should be normal density mode"
    
    # 4. Call apply_embargo
    selected_times = sf.apply_embargo(df, signal_mask)
    print(f"[LUNA-V2-EMBARGO-TEST] Selected signals count: {len(selected_times)}")
    
    # Let's inspect the gaps between consecutive selected signals
    time_gaps_hours = []
    for idx in range(1, len(selected_times)):
        gap = (selected_times[idx] - selected_times[idx-1]).total_seconds() / 3600.0
        time_gaps_hours.append(gap)
        
    print(f"[LUNA-V2-EMBARGO-TEST] Time gaps between signals: {time_gaps_hours}")
    
    # Assertions:
    # 1. Early gaps (high volatility phase) should be large (>= 48H)
    early_gaps = time_gaps_hours[:2]
    print(f"[LUNA-V2-EMBARGO-TEST] Early phase gaps: {early_gaps}")
    assert all(g >= 48.0 for g in early_gaps), "Early phase embargo should be large (>=48H)"
        
    # 2. Later gaps (calm phase) should decay to the safety floor of dynamic settings
    print(f"[LUNA-V2-EMBARGO-TEST] All gaps: {time_gaps_hours}")
    assert any(g <= 36.0 for g in time_gaps_hours), "Calm phase embargo should decay significantly"
    
    # [LUNA-V2-EMBARGO-TEST] Link test assertion to config settings instead of magic number
    _embargo_hours_floor = float(_cfg_test.xgboost.embargo_hours)
    print(f"[LUNA-V2-EMBARGO-TEST] Verifying all gaps are >= dynamic floor ({_embargo_hours_floor}H)...")
    assert all(g >= _embargo_hours_floor for g in time_gaps_hours), f"Embargo floor of {_embargo_hours_floor}H must be respected strictly"
        
    print("[LUNA-V2-EMBARGO-TEST] All Volatility Decaying Embargo assertions PASSED successfully!")
    
    # Restore original settings
    _cfg_test.xgboost.embargo_hours = _orig_floor
    _cfg_test.sop.embargo_hours = _orig_sop

if __name__ == "__main__":
    test_luna_v2_volatility_decaying_embargo()
