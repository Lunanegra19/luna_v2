import re

with open('luna/models/train_xgboost_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

def safe_replace(pattern, replacement):
    global content
    content, count = re.subn(pattern, replacement, content)
    return count

# 1. Calibration fallback
safe_replace(
    r'float\(_cfg_xgb\.xgboost\.calibration_fallback_method\)',
    "getattr(_cfg_xgb.xgboost, 'calibration_fallback_method', None)"
)

# 2. brier stat
safe_replace(
    r'int\(_cfg_brier\.stat\)',
    "getattr(_cfg_brier, 'stat', None)"
)

# 3. int(_cfg) wrappers
safe_replace(r'int\(_cfg[a-zA-Z0-9_]*\.features\)\.', 'getattr(_cfg.features, ')
safe_replace(r'int\(_cfg[a-zA-Z0-9_]*\.xgboost\)\.', 'getattr(_cfg.xgboost, ')
safe_replace(r'int\(_cfg[a-zA-Z0-9_]*\.xgboost\) and', 'hasattr(_cfg, "xgboost") and')
safe_replace(r'int\(_cfg\.xgboost\.regime_tbm_profiles\)', 'getattr(_cfg.xgboost, "regime_tbm_profiles", None)')

# 4. Double casts int(int(...))
safe_replace(r'int\(int\(([^)]+)\)\)', r'int(\1)')
safe_replace(r'float\(int\(([^)]+)\)\)', r'float(\1)')
safe_replace(r'float\(float\(([^)]+)\)\)', r'float(\1)')
safe_replace(r'bool\(int\(([^)]+)\)\)', r'bool(\1)')

with open('luna/models/train_xgboost_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
