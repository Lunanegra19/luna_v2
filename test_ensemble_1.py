import sys
from unittest.mock import patch
import config.settings

original_cfg = config.settings.cfg
class MockWFB:
    def __init__(self, wfb):
        self._wfb = wfb
    def __getattr__(self, item):
        if item == "active_seeds":
            return [42, 100, 1337]
        if item == "ensemble_consensus_threshold":
            return 1
        return getattr(self._wfb, item)

class MockCfg:
    def __init__(self, cfg):
        self._cfg = cfg
        self.wfb = MockWFB(cfg.wfb)
    def __getattr__(self, item):
        return getattr(self._cfg, item)

config.settings.cfg = MockCfg(original_cfg)

import scripts.evaluate_ensemble_wfb as eval_wfb
try:
    eval_wfb.main()
except SystemExit:
    pass
