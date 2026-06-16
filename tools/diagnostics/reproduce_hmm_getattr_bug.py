import yaml

class MockDict(dict):
    def __getattr__(self, item):
        if item in self:
            if isinstance(self[item], dict):
                return MockDict(self[item])
            return self[item]
        raise AttributeError(f"Missing {item}")

with open('config/settings.yaml', 'r') as f:
    cfg = MockDict(yaml.safe_load(f))

print("=== REPRODUCING BUG ===")
try:
    _extend = bool(int(getattr(cfg.hmm), 'hmm_extend_to_holdout', False))
except Exception as e:
    print(f"BUG REPRODUCED: {type(e).__name__}: {e}")

print("\n=== VALIDATING FIX ===")
try:
    _extend = bool(int(getattr(cfg.hmm, 'hmm_extend_to_holdout', False)))
    print(f"FIX SUCCESSFUL: hmm_extend_to_holdout = {_extend}")
except Exception as e:
    print(f"FIX FAILED: {e}")
