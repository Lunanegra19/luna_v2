"""Investigar por qué el gate min_viable_train_samples no se activa con n=91"""
from pathlib import Path
import re

log = Path('C:/Users/Usuario/.gemini/antigravity-ide/brain/ad23283d-d02e-4616-9748-5d609f02bf06/.system_generated/tasks/task-1314.log').read_text(encoding='utf-8', errors='replace')
lines = log.split('\n')

# ¿Aparece el gate H-RANGE-01-FIX en el log?
gate_lines = [l for l in lines if 'min_viable' in l.lower() or 'H-RANGE-01' in l or 'BUG-RANGE-01' in l]
print(f'Líneas con min_viable_train_samples / H-RANGE-01 / BUG-RANGE-01: {len(gate_lines)}')
for l in gate_lines[:10]:
    print(f'  {l.strip()[:115]}')
print()

# Buscar el audit justo antes de un colapso con n<200
collapse_idxs = [i for i, l in enumerate(lines)
                 if 'POST-FIT IS' in l and 'bear' in l.lower() and 'std_IS=0.000000' in l]
print(f'Para los primeros 3 colapsos, contexto con AUDIT-REGIME-N previo:')
for ci in collapse_idxs[:3]:
    block = lines[max(0, ci-30):ci]
    audit  = [l for l in block if 'AUDIT-REGIME-N' in l and 'bear' in l.lower()]
    gated  = [l for l in block if 'viable' in l.lower() or 'BUG-RANGE' in l or 'SOP-R8' in l]
    num = collapse_idxs.index(ci) + 1
    print(f'\n  Colapso #{num}:')
    for l in audit[-2:]:
        print(f'    [AUDIT] {l.strip()[:110]}')
    for l in gated[-3:]:
        print(f'    [GATE]  {l.strip()[:110]}')
    n_m = re.search(r'n_train=(\d+)', ' '.join(audit))
    if n_m:
        n = int(n_m.group(1))
        activo = 'DEBERIA activarse pero NO lo hace' if n < 200 else 'no aplica'
        print(f'    n_train={n} | min_viable=200 | Gate: {activo}')

# ¿Cómo está implementado el gate?
print()
print('=== CÓDIGO DEL GATE EN train_xgboost_v2.py ===')
code_path = Path('g:/Mi unidad/ia/luna_v2/luna/models/train_xgboost_v2.py')
code = code_path.read_text(encoding='utf-8', errors='replace')
code_lines = code.split('\n')

# Buscar min_viable_train_samples en el código
for i, l in enumerate(code_lines):
    if 'min_viable_train_samples' in l or 'H-RANGE-01' in l or 'BUG-RANGE-01' in l:
        start = max(0, i-2)
        end   = min(len(code_lines), i+6)
        print(f'\n[L{i+1}]:')
        for j in range(start, end):
            print(f'  {j+1:4d}: {code_lines[j]}')
