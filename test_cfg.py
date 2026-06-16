import sys
import yaml
from pathlib import Path

# Load settings
with open('config/settings.yaml', 'r') as f:
    raw_cfg = yaml.safe_load(f)

# Mock config access like in luna
class DictConfig:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, DictConfig(v))
            else:
                setattr(self, k, v)

cfg = DictConfig(raw_cfg)

print(f"Type of hmm_extend_to_holdout: {type(getattr(cfg.hmm, 'hmm_extend_to_holdout', False))}")
print(f"Value: {getattr(cfg.hmm, 'hmm_extend_to_holdout', False)}")
