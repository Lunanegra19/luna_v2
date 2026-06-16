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

_cfg = DictConfig(raw_cfg)

print("=== REPRODUCING OFI BUG ===")
try:
    if int(_cfg.fase2) and bool(_cfg.fase2.use_ofi_features):
        print("OFI Enabled")
except Exception as e:
    print(f"BUG REPRODUCED: {type(e).__name__}: {e}")

print("\n=== VALIDATING OFI FIX ===")
try:
    if hasattr(_cfg, 'fase2') and bool(getattr(_cfg.fase2, 'use_ofi_features', False)):
        print("FIX SUCCESSFUL: OFI Enabled")
    else:
        print("FIX SUCCESSFUL: OFI Disabled")
except Exception as e:
    print(f"FIX FAILED: {e}")

print("\n=== REPRODUCING KALMAN BUG ===")
try:
    _kz_q = float(int(getattr(_cfg.features), 'kalman_q', 1e-4))
except Exception as e:
    print(f"BUG REPRODUCED: {type(e).__name__}: {e}")

print("\n=== VALIDATING KALMAN FIX ===")
try:
    _kz_q = float(getattr(_cfg.features, 'kalman_q', 1e-4))
    _kz_r = float(getattr(_cfg.features, 'kalman_r', 0.1))
    print(f"FIX SUCCESSFUL: kalman_q={_kz_q}, kalman_r={_kz_r}")
except Exception as e:
    print(f"FIX FAILED: {e}")

