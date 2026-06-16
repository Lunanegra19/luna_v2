import re

with open('luna/models/train_xgboost_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix int(Namespace)
content = re.sub(r'int\(_cfg[a-zA-Z0-9_]*\.features\)\.', 'getattr(_cfg.features, ', content)
content = re.sub(r'int\(_cfg[a-zA-Z0-9_]*\.xgboost\)\.', 'getattr(_cfg.xgboost, ', content)
content = re.sub(r'int\(_cfg[a-zA-Z0-9_]*\.xgboost\) and', 'hasattr(_cfg, "xgboost") and', content)

# Fix double casts
content = content.replace("int(int(", "int(")
content = content.replace("float(int(", "float(")
content = content.replace("float(float(", "float(")
content = content.replace("bool(int(", "bool(")

# Fix regime_tbm_profiles cast
content = content.replace("int(_cfg.xgboost.regime_tbm_profiles)", "getattr(_cfg.xgboost, 'regime_tbm_profiles', None)")

# Fix stat
content = content.replace("int(_cfg_brier.stat)", "getattr(_cfg_brier, 'stat', None)")

with open('luna/models/train_xgboost_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Double casts fixed in train_xgboost_v2.py")
