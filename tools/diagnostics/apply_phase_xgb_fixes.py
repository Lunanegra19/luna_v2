import re

# Fix phase_gates.py
with open('luna/validation/phase_gates.py', 'r', encoding='utf-8') as f:
    pg_content = f.read()

pg_content = pg_content.replace(
    "_stat = int(_cfg_pg.stat)",
    "_stat = _cfg_pg.stat"
)
pg_content = pg_content.replace(
    "_alpha_ratio = float(_cfg_pg.features.sfi_max_alpha_ratio)",
    "_alpha_ratio = float(_cfg_pg.features.sfi_max_alpha_ratio)"
)

with open('luna/validation/phase_gates.py', 'w', encoding='utf-8') as f:
    f.write(pg_content)

# Fix train_xgboost_v2.py
with open('luna/models/train_xgboost_v2.py', 'r', encoding='utf-8') as f:
    xgb_content = f.read()

xgb_content = xgb_content.replace(
    "if float(_cfg_xgb.xgboost.calibration_fallback_method) is None:",
    "if _cfg_xgb.xgboost.calibration_fallback_method is None:"
)

with open('luna/models/train_xgboost_v2.py', 'w', encoding='utf-8') as f:
    f.write(xgb_content)

print("Fixes applied to both files.")
