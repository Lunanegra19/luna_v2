with open('luna/models/train_xgboost_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the unmatched parenthesis
import re
content = re.sub(r'int\(_cfg_xgb\.xgboost\.optuna_trials\)\)', 'int(_cfg_xgb.xgboost.optuna_trials)', content)
content = re.sub(r'int\(_cfg_xgb\.sop\.purge_hours\)\)', 'int(_cfg_xgb.sop.purge_hours)', content)
content = re.sub(r'int\(_cfg_xgb\.sop\.embargo_hours\)\)', 'int(_cfg_xgb.sop.embargo_hours)', content)
content = re.sub(r'int\(_cfg_rw\.wfb\.rolling_window_years\)\)', 'int(_cfg_rw.wfb.rolling_window_years)', content)
content = re.sub(r'int\(_cfg_mvr\.xgboost\.vertical_barrier_hours\)\)', 'int(_cfg_mvr.xgboost.vertical_barrier_hours)', content)
content = re.sub(r'bool\(getattr\(_cfg_timing\.features,\s*timing_features_bypass_sfi\)\)\)', 'bool(getattr(_cfg_timing.features, "timing_features_bypass_sfi", True))', content)
content = re.sub(r'int\(_cfg\.xgboost\.event_sampling_hours\)\)', 'int(_cfg.xgboost.event_sampling_hours)', content)
content = re.sub(r'float\(_cfg\.xgboost\.tbm_min_return\)\)', 'float(_cfg.xgboost.tbm_min_return)', content)
content = re.sub(r'float\(_cfg\.xgboost\.pt_decay_fraction\)\)', 'float(_cfg.xgboost.pt_decay_fraction)', content)
content = re.sub(r'float\(_cfg_fl\.xgboost\.focal_loss_gamma\)\)', 'float(_cfg_fl.xgboost.focal_loss_gamma)', content)
content = re.sub(r'float\(_cfg_04a\.xgboost\.threshold_sweep_min\)\)', 'float(_cfg_04a.xgboost.threshold_sweep_min)', content)
content = re.sub(r'int\(_cfg_xgb\.xgboost\.optuna_seed\)\)', 'int(_cfg_xgb.xgboost.optuna_seed)', content)
content = re.sub(r'int\(_cfg_xgb\.xgboost\.holdout_calib_months\)\)', 'int(_cfg_xgb.xgboost.holdout_calib_months)', content)
content = re.sub(r'int\(_cfg_nest\.xgboost\.n_estimators_min_floor\)\)', 'int(_cfg_nest.xgboost.n_estimators_min_floor)', content)

# General replace for any remaining int(...)) created by my script
content = re.sub(r'(=\s*int\([^)]+\))\)', r'\1', content)
content = re.sub(r'(=\s*float\([^)]+\))\)', r'\1', content)
content = re.sub(r'(=\s*bool\([^)]+\))\)', r'\1', content)

with open('luna/models/train_xgboost_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
