"""verify_crit03.py — verifica que FIX-CRIT-03 esta completo"""
import ast, sys

fp = 'luna/utils/ood_feature_guard.py'
content = open(fp, encoding='utf-8').read()

checks = [
    ('structural_features: list = []', 'CAMBIO-1 __init__'),
    ('FIX-CRIT-03', 'CAMBIO-2/3 presentes'),
    ('STRUCTURAL_EXEMPT', 'CAMBIO-3 _analyze_feature'),
    ('structural_features = list(getattr', 'CAMBIO-2 _load_from_settings'),
]

all_ok = True
for pattern, label in checks:
    ok = pattern in content
    status = "OK" if ok else "MISSING"
    print(f"  [{status}] {label}")
    if not ok:
        all_ok = False

try:
    ast.parse(content)
    print("SYNTAX: OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    all_ok = False

print("CRIT-03 STATUS:", "COMPLETO" if all_ok else "INCOMPLETO")
