import os
import glob

files = glob.glob('scripts/pre_flight/*.py')
mappings = {
    'scripts/predict_oos.py': 'luna/models/predict_oos.py',
    'scripts/train_xgboost_v2.py': 'luna/models/train_xgboost_v2.py',
    'scripts/train_metalabeler_v2.py': 'luna/models/train_metalabeler_v2.py',
    'scripts/hmm_regime.py': 'luna/models/hmm_regime.py',
    'scripts/calibrate_probabilities.py': 'luna/models/calibrate_probabilities.py',
    'scripts/feature_pipeline.py': 'luna/features/feature_pipeline.py',
    'scripts/feature_selection_e.py': 'luna/features/feature_selection_e.py',
    'scripts/ood_guard.py': 'luna/models/ood_guard.py',
    'scripts/signal_filter.py': 'luna/models/signal_filter.py'
}

count = 0
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    original = content
    for old, new in mappings.items():
        content = content.replace(old, new)
    if content != original:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        count += 1
print(f'Fixed paths in {count} files')
