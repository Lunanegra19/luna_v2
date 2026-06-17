"""arch10_verify_fix.py — Verifica que el fix de ARCH-10 se aplicó correctamente"""
import yaml
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
with open(ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
sop = cfg.sop
xgb = cfg.xgboost
print("[ARCH-10-FIX] VERIFICACION POST-FIX:")
print(f"  sop.embargo_hours:         {sop.get('embargo_hours')}  (era 72)")
print(f"  sop.purge_hours:           {sop.get('purge_hours')}")
print(f"  xgboost.embargo_hours:     {xgb.get('embargo_hours')}  (era 24)")
print(f"  xgboost.embargo_min_hours: {xgb.get('embargo_min_hours')} (era 48)")
print(f"  xgboost.dynamic_barrier:   {xgb.get('dynamic_barrier')}  (restaurado)")
print(f"  xgboost.dynamic_horizon:   {xgb.get('dynamic_horizon_min_h')}  (restaurado)")
print()
sop_ok = sop.embargo_hours >= 96
xgb_ok = xgb.embargo_hours >= 96
dyn_ok = xgb.get("dynamic_barrier") is not None
print(f"  SOP R3 sop.embargo_hours >= 96H: {'OK' if sop_ok else 'FALLO'}")
print(f"  SOP R3 xgb.embargo_hours >= 96H: {'OK' if xgb_ok else 'FALLO'}")
print(f"  dynamic_barrier restaurado:       {'OK' if dyn_ok else 'FALLO'}")
print()
print("YAML syntax: VALIDO" if cfg else "YAML syntax: ERROR")
