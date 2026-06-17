import re

with open('luna/features/feature_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix OFI bug: int(_cfg.fase2) -> hasattr(_cfg, 'fase2')
# And bool(_cfg.fase2.use_ofi_features) -> bool(_cfg.fase2.use_ofi_features)
content = content.replace(
    "if int(_cfg.fase2) and bool(_cfg.fase2.use_ofi_features):",
    "if hasattr(_cfg, 'fase2') and bool(_cfg.fase2.use_ofi_features):"
)

# Fix Kalman bugs
content = content.replace(
    "float(int(_cfg_kz.features).kalman_q)",
    "float(_cfg_kz.features.kalman_q)"
)
content = content.replace(
    "float(int(_cfg_kz.features).kalman_r)",
    "float(_cfg_kz.features.kalman_r)"
)

with open('luna/features/feature_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixes applied to feature_pipeline.py.")
