import sys
sys.path.insert(0, r'G:\Mi unidad\ia\luna_v2')
from config.settings import cfg

m = cfg.metalabeler
print('[settings.yaml OK] Nuevos params MEJORA-MOMENTUM-01:')
print('  momentum_speed_window =', getattr(m, 'momentum_speed_window', 'NOT FOUND'))
print('  momentum_crash_speed_CUTOFF = ', getattr(m, 'momentum_crash_speed_threshold', 'NOT FOUND'))
print('  momentum_ordered_correction_CUTOFF = ', getattr(m, 'momentum_ordered_correction_threshold', 'NOT FOUND'))
print('  momentum_filter_CUTOFF = ', getattr(m, 'momentum_filter_threshold', 'NOT FOUND'))
print('  momentum_filter_threshold_upper =', getattr(m, 'momentum_filter_threshold_upper', 'NOT FOUND'))
print()
s = cfg.stat
print('[settings.yaml OK] Gates criticos:')
print('  xgb_auc_hard_stop =', getattr(s, 'xgb_auc_hard_stop', 'NOT FOUND'))
print('  xgb_brier_hard_stop =', getattr(s, 'xgb_brier_hard_stop', 'NOT FOUND'))
print('  xgb_proba_std_min =', getattr(s, 'xgb_proba_std_min', 'NOT FOUND'))
print()
g = cfg.gauntlet
print('[settings.yaml OK] Gauntlet:')
print('  max_pbo =', getattr(g, 'max_pbo', 'NOT FOUND'))
print('  min_dsr =', getattr(g, 'min_dsr', 'NOT FOUND'))
