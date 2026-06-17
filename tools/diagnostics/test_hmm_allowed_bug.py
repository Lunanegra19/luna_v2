class MockNamespace:
    pass

class MockMeta:
    hmm_allowed_regimes = ['1_BULL_TREND', '1_VOLATILE_BULL']
    
class MockCfg:
    metalabeler = MockMeta()

_cfg_hmm = MockCfg()

try:
    _hmm_cfg = int(_cfg_hmm.metalabeler)
    _hmm_allowed = int(_hmm_cfg.hmm_allowed_regimes)
except Exception as e:
    print("BUG REPRODUCIDO:", type(e).__name__, e)

try:
    _hmm_cfg = _cfg_hmm.metalabeler
    _hmm_allowed = _hmm_cfg.hmm_allowed_regimes
    print("FIX EXITOSO:", _hmm_allowed)
except Exception as e:
    print("ERROR en fix:", e)
