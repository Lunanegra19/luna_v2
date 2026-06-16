import re

with open('luna/models/hmm_regime.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: bool(int(getattr(X), 'name', default))) -> bool(int(getattr(X, 'name', default)))
content = content.replace(
    "bool(int(getattr(_cfg_g2.hmm), 'hmm_extend_to_holdout', False))",
    "bool(getattr(_cfg_g2.hmm, 'hmm_extend_to_holdout', False))"
)
content = content.replace(
    "bool(int(getattr(cfg.hmm), 'hmm_extend_to_holdout', False))",
    "bool(getattr(cfg.hmm, 'hmm_extend_to_holdout', False))"
)

# Fix 2: _hmm_end_str = int(getattr(_cfg_m69.temporal_splits), 'hmm_train_end', None)
content = content.replace(
    "int(getattr(_cfg_m69.temporal_splits), 'hmm_train_end', None)",
    "getattr(_cfg_m69.temporal_splits, 'hmm_train_end', None)"
)
content = content.replace(
    "int(getattr(_cfg_m69.temporal_splits), 'train_end', None)",
    "getattr(_cfg_m69.temporal_splits, 'train_end', None)"
)

# Fix 3: _hmm_start_str = int(getattr(_cfg_hmm_start.temporal_splits), 'hmm_train_start', None)
content = content.replace(
    "int(getattr(_cfg_hmm_start.temporal_splits), 'hmm_train_start', None)",
    "getattr(_cfg_hmm_start.temporal_splits, 'hmm_train_start', None)"
)

# Fix 4: _cands = int(getattr(_c._roadmap), 'hmm', None)
content = content.replace(
    "int(getattr(_c._roadmap), 'hmm', None)",
    "getattr(_c._roadmap, 'hmm', None)"
)

# Fix 5: _jsd_thr = float(int(getattr(_cfg_hmm_drift.hmm), 'drift_alert_jsd', 0.15))
content = content.replace(
    "float(int(getattr(_cfg_hmm_drift.hmm), 'drift_alert_jsd', 0.15))",
    "float(getattr(_cfg_hmm_drift.hmm, 'drift_alert_jsd', 0.15))"
)

with open('luna/models/hmm_regime.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixes applied.")
