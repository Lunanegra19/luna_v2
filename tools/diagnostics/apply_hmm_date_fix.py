import re

with open('luna/models/hmm_regime.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix int(cfg.temporal_splits.hmm_train_end)
content = content.replace(
    "_hmm_end = int(cfg.temporal_splits.hmm_train_end)",
    "_hmm_end = getattr(cfg.temporal_splits, 'hmm_train_end', None)"
)

# Fix int(cfg.temporal_splits.train_end)
content = content.replace(
    "train_cutoff = int(cfg.temporal_splits.train_end)",
    "train_cutoff = getattr(cfg.temporal_splits, 'train_end', None)"
)

with open('luna/models/hmm_regime.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fix applied to fit_global_for_analysis.")
