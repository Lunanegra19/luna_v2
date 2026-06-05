"""find_vbar_setting.py - localiza vertical_barrier_hours en el proyecto"""
import re, sys
sys.path.insert(0, '.')
from config.settings import cfg
from pathlib import Path

ROOT = Path('.')

# Encontrar en settings.yaml
with open('config/settings.yaml', 'r', encoding='utf-8') as f:
    raw_yaml = f.read()

print("=== Busqueda en settings.yaml ===")
pattern = re.compile(r'(?i)(vertical|barrier|vbar|tbm_|pt_mult|sl_mult)', re.I)
lines = raw_yaml.split('\n')
for i, line in enumerate(lines):
    if pattern.search(line):
        print(f"L{i+1}: {line}")

print()
print("=== Valores en cfg.xgboost ===")
for attr in dir(cfg.xgboost):
    if not attr.startswith('_'):
        val = getattr(cfg.xgboost, attr, None)
        if isinstance(val, (int, float, str)):
            if any(kw in attr.lower() for kw in ['barrier', 'vertical', 'vbar', 'tbm', 'pt_', 'sl_']):
                print(f"  cfg.xgboost.{attr} = {val}")
