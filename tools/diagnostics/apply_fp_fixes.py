import re

with open('luna/features/feature_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix OFI bug: int(_cfg.fase2) -> hasattr(_cfg, 'fase2')
# And bool(_cfg.fase2.use_ofi_features) -> bool(getattr(_cfg.fase2, 'use_ofi_features', False))
content = content.replace(
    "if int(_cfg.fase2) and bool(_cfg.fase2.use_ofi_features):",
    "if hasattr(_cfg, 'fase2') and bool(getattr(_cfg.fase2, 'use_ofi_features', False)):"
)

# Fix Kalman bugs
content = content.replace(
    "float(int(getattr(_cfg_kz.features), 'kalman_q', 1e-4))",
    "float(getattr(_cfg_kz.features, 'kalman_q', 1e-4))"
)
content = content.replace(
    "float(int(getattr(_cfg_kz.features), 'kalman_r', 0.1))",
    "float(getattr(_cfg_kz.features, 'kalman_r', 0.1))"
)

with open('luna/features/feature_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixes applied to feature_pipeline.py.")
