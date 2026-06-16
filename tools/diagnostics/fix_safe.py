import re

with open('luna/models/train_xgboost_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix the crashing calibration_fallback_method
content = content.replace(
    'if float(_cfg_xgb.xgboost.calibration_fallback_method) is None:',
    'if getattr(_cfg_xgb.xgboost, "calibration_fallback_method", None) is None:'
)

# 2. Fix the crashing brier stat
content = content.replace(
    '_stat_brier = int(_cfg_brier.stat)',
    '_stat_brier = getattr(_cfg_brier, "stat", None)'
)

# 3. Fix the int(Namespace).prop by matching the prop
content = re.sub(
    r'int\((_cfg[a-zA-Z0-9_]*\.features)\)\.([a-zA-Z0-9_]+)',
    r'getattr(\1, "\2")',
    content
)
content = re.sub(
    r'int\((_cfg[a-zA-Z0-9_]*\.xgboost)\)\.([a-zA-Z0-9_]+)',
    r'getattr(\1, "\2")',
    content
)

# 4. Fix int(_cfg.xgboost) and bool(...)
content = content.replace(
    'int(_cfg.xgboost) and bool(_cfg.xgboost.dynamic_barrier)',
    'hasattr(_cfg, "xgboost") and bool(_cfg.xgboost.dynamic_barrier)'
)

# 5. Remove int(int(...))
content = content.replace('int(int(', 'int(')
content = content.replace('float(int(', 'float(')
content = content.replace('float(float(', 'float(')
content = content.replace('bool(int(', 'bool(')

# Since we replaced 'int(int(' with 'int(', we have unbalanced parentheses.
# Let's fix the specific ones we know:
content = content.replace('int(_cfg_xgb.xgboost.optuna_trials))', 'int(_cfg_xgb.xgboost.optuna_trials)')
content = content.replace('int(_cfg_xgb.sop.purge_hours))', 'int(_cfg_xgb.sop.purge_hours)')
content = content.replace('int(_cfg_xgb.sop.embargo_hours))', 'int(_cfg_xgb.sop.embargo_hours)')
content = content.replace('int(_cfg_rw.wfb.rolling_window_years))', 'int(_cfg_rw.wfb.rolling_window_years)')
content = content.replace('int(_cfg_mvr.xgboost.vertical_barrier_hours))', 'int(_cfg_mvr.xgboost.vertical_barrier_hours)')
content = content.replace('bool(getattr(_cfg_timing.features, "timing_features_bypass_sfi")))', 'bool(getattr(_cfg_timing.features, "timing_features_bypass_sfi"))')
content = content.replace('int(_cfg.xgboost.event_sampling_hours))', 'int(_cfg.xgboost.event_sampling_hours)')
content = content.replace('float(_cfg.xgboost.tbm_min_return))', 'float(_cfg.xgboost.tbm_min_return)')
content = content.replace('float(_cfg.xgboost.pt_decay_fraction))', 'float(_cfg.xgboost.pt_decay_fraction)')
content = content.replace('float(_cfg_fl.xgboost.focal_loss_gamma))', 'float(_cfg_fl.xgboost.focal_loss_gamma)')
content = content.replace('float(_cfg_04a.xgboost.threshold_sweep_min))', 'float(_cfg_04a.xgboost.threshold_sweep_min)')
content = content.replace('int(_cfg_xgb.xgboost.optuna_seed))', 'int(_cfg_xgb.xgboost.optuna_seed)')
content = content.replace('int(_cfg_xgb.xgboost.holdout_calib_months))', 'int(_cfg_xgb.xgboost.holdout_calib_months)')
content = content.replace('int(_cfg_nest.xgboost.n_estimators_min_floor))', 'int(_cfg_nest.xgboost.n_estimators_min_floor)')

with open('luna/models/train_xgboost_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
