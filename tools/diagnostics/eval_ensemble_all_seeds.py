"""
eval_ensemble_all_seeds.py
Wrapper de evaluate_ensemble_wfb.py que fuerza el uso de TODAS las seeds
encontradas en los parquets (override de active_seeds de settings.yaml).
Usado para análisis post-run "¿qué pasaría con las 20 seeds?".
"""
import sys, os, glob
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_ROOT = Path(r'g:\Mi unidad\ia\luna_v2')
sys.path.insert(0, str(_ROOT))

# Descubrir todas las seeds disponibles en parquets
WFB_DIR = _ROOT / 'data' / 'reports' / 'wfb'
parquets = glob.glob(str(WFB_DIR / 'oos_trades_W*_seed*.parquet'))
all_seeds = sorted(set(int(Path(f).stem.split('_seed')[1]) for f in parquets))
print(f"[EVAL-ALL] Seeds detectadas en parquets: {len(all_seeds)}")
print(f"[EVAL-ALL] Lista: {all_seeds}")

# Monkey-patch la configuración para inyectar all_seeds
from config.settings import cfg as _cfg
_cfg.wfb.active_seeds = all_seeds
print(f"[EVAL-ALL] Override active_seeds → {len(all_seeds)} seeds")

# Re-importar y ejecutar evaluate_ensemble_wfb
import importlib
import scripts.evaluate_ensemble_wfb as _eval_mod
_eval_mod.main()
