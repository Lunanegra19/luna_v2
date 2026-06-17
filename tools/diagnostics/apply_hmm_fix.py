import re

with open('luna/models/hmm_regime.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: bool(int(X).name)) -> bool(int(X.name))
content = content.replace(
    "bool(int(_cfg_g2.hmm).hmm_extend_to_holdout)",
    "bool(_cfg_g2.hmm.hmm_extend_to_holdout)"
)
content = content.replace(
    "bool(int(cfg.hmm).hmm_extend_to_holdout)",
    "bool(cfg.hmm.hmm_extend_to_holdout)"
)

# Fix 2: _hmm_end_str = int(_cfg_m69.temporal_splits).hmm_train_end
content = content.replace(
    "int(_cfg_m69.temporal_splits).hmm_train_end",
    "_cfg_m69.temporal_splits.hmm_train_end"
)
content = content.replace(
    "int(_cfg_m69.temporal_splits).train_end",
    "_cfg_m69.temporal_splits.train_end"
)

# Fix 3: _hmm_start_str = int(_cfg_hmm_start.temporal_splits).hmm_train_start
content = content.replace(
    "int(_cfg_hmm_start.temporal_splits).hmm_train_start",
    "_cfg_hmm_start.temporal_splits.hmm_train_start"
)

# Fix 4: _cands = int(_c._roadmap).hmm
content = content.replace(
    "int(_c._roadmap).hmm",
    "_c._roadmap.hmm"
)

# Fix 5: _jsd_thr = float(int(_cfg_hmm_drift.hmm).drift_alert_jsd)
content = content.replace(
    "float(int(_cfg_hmm_drift.hmm).drift_alert_jsd)",
    "float(_cfg_hmm_drift.hmm.drift_alert_jsd)"
)

with open('luna/models/hmm_regime.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixes applied.")
