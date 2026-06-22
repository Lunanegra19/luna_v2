import sys
import os
import json
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from dashboard.server import get_seed_metrics_from_verdict, get_wfb_seeds_summary, get_metric_strict

def test_strictness():
    print("--- Test 1: get_metric_strict missing key ---")
    try:
        get_metric_strict({"a": 1}, None, "b")
        print("FAIL: get_metric_strict did not raise KeyError for missing key.")
        return False
    except KeyError as e:
        print(f"PASS: KeyError correctly raised: {e}")

    print("\n--- Test 2: get_seed_metrics_from_verdict without verdicts ---")
    try:
        get_seed_metrics_from_verdict(999999) # Non-existent seed
        print("FAIL: get_seed_metrics_from_verdict did not raise RuntimeError for missing verdict.")
        return False
    except RuntimeError as e:
        print(f"PASS: RuntimeError correctly raised: {e}")

    print("\n--- Test 3: Dashboard fallback removal check ---")
    try:
        with open(PROJECT_ROOT / "dashboard" / "server.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "prod_seeds = [1337, 2025, 99]" in content:
                print("FAIL: Hardcoded prod_seeds fallback still exists.")
                return False
            if "KELLY_SWEEP = [" in content:
                print("FAIL: KELLY_SWEEP hardcoded array still exists.")
                return False
        print("PASS: Hardcoded defaults removed from server.py.")
    except Exception as e:
        print(f"FAIL: Could not read server.py: {e}")
        return False

    print("\n[SUCCESS] All No-Fallback Policy strictness tests passed.")
    return True

if __name__ == "__main__":
    success = test_strictness()
    sys.exit(0 if success else 1)
