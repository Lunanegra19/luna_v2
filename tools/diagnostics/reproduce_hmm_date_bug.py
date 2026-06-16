import sys
import yaml

with open('config/settings.yaml', 'r') as f:
    raw_cfg = yaml.safe_load(f)

class DictConfig:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, DictConfig(v))
            else:
                setattr(self, k, v)

cfg = DictConfig(raw_cfg)

print("=== REPRODUCING BUG ===")
try:
    _hmm_end = int(cfg.temporal_splits.hmm_train_end)
except Exception as e:
    print(f"BUG REPRODUCED: {type(e).__name__}: {e}")

print("\n=== VALIDATING FIX ===")
try:
    _hmm_end = str(cfg.temporal_splits.hmm_train_end)
    print(f"FIX SUCCESSFUL: _hmm_end = {_hmm_end} ({type(_hmm_end).__name__})")
except Exception as e:
    print(f"FIX FAILED: {e}")
