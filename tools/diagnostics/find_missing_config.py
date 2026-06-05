import os
import re
import yaml

def get_yaml_keys(d, prefix=''):
    keys = set()
    if isinstance(d, dict):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(k)
            keys.update(get_yaml_keys(v, full_key))
    return keys

def find_getattr_keys():
    py_keys = set()
    with open(r'config\settings.yaml', 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    yaml_keys = get_yaml_keys(settings)

    for r, _, fs in os.walk('luna'):
        for f in fs:
            if f.endswith('.py'):
                path = os.path.join(r, f)
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    # Capture getattr(_cfg..., 'key', default)
                    matches = re.findall(r"getattr\s*\(\s*[^,]*_cfg[^,]*,\s*['\"]([^'\"]+)['\"]\s*,", content)
                    py_keys.update(matches)
                    # Capture get('key', default) on cfg dicts
                    matches2 = re.findall(r"cfg\.get\s*\(\s*['\"]([^'\"]+)['\"]", content)
                    py_keys.update(matches2)

    missing = sorted(list(py_keys - yaml_keys))
    print(f"Encontradas configuraciones faltantes:\n")
    for m in missing:
        if m not in ['sop', 'xgboost', 'fase2', 'features', 'data', 'temporal_splits']:
            print(f"- {m}")

find_getattr_keys()
