import pandas as pd
import datetime

# Mock settings
class MockTemporalSplits:
    train_end = datetime.date(2024, 10, 31)

class MockWFB:
    training_mode = 'rolling'
    rolling_window_years = 3

class MockCfg:
    wfb = MockWFB()
    temporal_splits = MockTemporalSplits()

_cfg_rw = MockCfg()

try:
    _t_mode = str(_cfg_rw.wfb.training_mode)
    if _t_mode == 'rolling':
        _rw_years = int(_cfg_rw.wfb.rolling_window_years)
        # Buggy line:
        _train_end_str = int(_cfg_rw.temporal_splits.train_end)
        print("Success:", _train_end_str)
except Exception as e:
    print(f"BUG REPRODUCIDO: {type(e).__name__}: {e}")

try:
    _t_mode = str(_cfg_rw.wfb.training_mode)
    if _t_mode == 'rolling':
        _rw_years = int(_cfg_rw.wfb.rolling_window_years)
        # Fixed line:
        _train_end_val = str(_cfg_rw.temporal_splits.train_end)
        if _train_end_val:
            _train_end_dt = pd.to_datetime(_train_end_val, utc=True)
            _rolling_start = _train_end_dt - pd.DateOffset(years=_rw_years)
            print("FIX EXITOSO. _rolling_start =", _rolling_start)
except Exception as e:
    print(f"Error en fix: {e}")
