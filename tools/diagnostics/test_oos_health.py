import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from scripts.oos_health_monitor import calculate_cusum, calculate_sharpe_degradation

def test_health_monitor():
    # Dummy data
    dates = pd.date_range("2026-01-01", periods=100, freq="1d")
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    
    # Test cusum
    res_cusum = calculate_cusum(returns, target=0.005, threshold=4.0)
    print("CUSUM Result:", res_cusum)
    assert "max_drift" in res_cusum
    assert "trigger" in res_cusum
    
    # Test sharpe
    res_sharpe = calculate_sharpe_degradation(returns, freq="W", critical_weeks=2)
    print("Sharpe Result:", res_sharpe)
    assert "recent_sharpe" in res_sharpe
    assert "trigger" in res_sharpe
    
    print("All tests passed.")

if __name__ == "__main__":
    test_health_monitor()
