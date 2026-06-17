"""arch19_20_verify_fix.py — Verifica fix de rolling_window_years y embargo"""
import yaml
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
with open(ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
wfb = cfg.wfb
sop = cfg.sop
xgb = cfg.xgboost
print("[ARCH-19/20 + ARCH-10] VERIFICACION FINAL DE FIXES:")
print(f"  rolling_window_years:      {wfb.get('rolling_window_years')}  (era 3, objetivo 5)")
print(f"  training_mode:             {wfb.get('training_mode')}")
print(f"  sop.embargo_hours:         {sop.get('embargo_hours')}  (era 72)")
print(f"  xgboost.embargo_hours:     {xgb.get('embargo_hours')}  (era 24)")
print(f"  xgboost.embargo_min_hours: {xgb.get('embargo_min_hours')} (era 48)")
print(f"  xgboost.dynamic_barrier:   {xgb.get('dynamic_barrier')}")
print()
ok_rolling = wfb.rolling_window_years == 5
ok_sop_embargo = sop.embargo_hours >= 96
ok_xgb_embargo = xgb.embargo_hours >= 96
ok_min_embargo = xgb.embargo_min_hours >= 96
ok_dynamic = xgb.get("dynamic_barrier") is not None
print(f"  rolling_window_years == 5:       {'OK' if ok_rolling else 'FALLO'}")
print(f"  sop.embargo_hours >= 96H:        {'OK' if ok_sop_embargo else 'FALLO'}")
print(f"  xgb.embargo_hours >= 96H:        {'OK' if ok_xgb_embargo else 'FALLO'}")
print(f"  xgb.embargo_min_hours >= 96H:    {'OK' if ok_min_embargo else 'FALLO'}")
print(f"  dynamic_barrier presente:         {'OK' if ok_dynamic else 'FALLO'}")
print()
all_ok = all([ok_rolling, ok_sop_embargo, ok_xgb_embargo, ok_min_embargo, ok_dynamic])
print(f"  YAML syntax: {'VALIDO' if cfg else 'ERROR'}")
print(f"  TODOS LOS FIXES: {'OK' if all_ok else 'ALGUNO FALLO'}")
