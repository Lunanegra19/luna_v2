"""Reconstruir secuencia n_train → gate → std_IS para bear_long"""
from pathlib import Path
import re

log = Path('C:/Users/Usuario/.gemini/antigravity-ide/brain/ad23283d-d02e-4616-9748-5d609f02bf06/.system_generated/tasks/task-1314.log').read_text(encoding='utf-8', errors='replace')
lines = log.split('\n')

runs = []
curr_n = None
for i, l in enumerate(lines):
    if 'AUDIT-REGIME-N' in l and "Agente='bear'" in l:
        m = re.search(r'n_train=(\d+)', l)
        if m:
            curr_n = int(m.group(1))
    elif 'H-RANGE-01-FIX' in l and 'GATE PRE-FIT' in l and 'bear' in l:
        runs.append({'n_train': curr_n, 'gate': True, 'std_IS': None})
        curr_n = None
    elif 'POST-FIT IS' in l and 'bear' in l.lower():
        m = re.search(r'std_IS=(\S+)', l)
        if m and curr_n is not None:
            runs.append({'n_train': curr_n, 'gate': False, 'std_IS': float(m.group(1))})
            curr_n = None

print('n_train | gate_fired  | std_IS       | estado')
print('-'*60)
for r in runs[:45]:
    gate  = 'GATE-ABORT' if r.get('gate') else 'trained'
    std   = '{:.6f}'.format(r['std_IS']) if r.get('std_IS') is not None else 'N/A(aborted)'
    colap = '*** COLAPSO' if r.get('std_IS') == 0.0 else ''
    n_str = str(r.get('n_train', '?'))
    print('  {:>6} | {:10} | {:12} | {}'.format(n_str, gate, std, colap))

print()
collapses_n = [r['n_train'] for r in runs if r.get('std_IS') == 0.0 and r.get('n_train')]
ok_n        = [r['n_train'] for r in runs if r.get('std_IS', -1) > 0 and r.get('n_train')]
gated_n     = [r['n_train'] for r in runs if r.get('gate') and r.get('n_train')]

print('n_train que COLAPSARON (std_IS=0): {}'.format(sorted(collapses_n)))
print('  min={} max={}'.format(min(collapses_n) if collapses_n else 'N/A',
                                max(collapses_n) if collapses_n else 'N/A'))
print()
print('n_train que FUNCIONARON (std_IS>0): min={} max={}'.format(
    min(ok_n) if ok_n else '?', max(ok_n) if ok_n else '?'))
print()
print('n_train que fueron GATED (aborted): {}'.format(sorted(gated_n)))
print()
print('=== CONCLUSIÓN ===')
if collapses_n:
    mn = min(collapses_n)
    mx = max(collapses_n)
    print('Los colapsos ocurren con n_train en rango [{}, {}]'.format(mn, mx))
    if mn >= 200:
        print('TODOS los colapsos tienen n_train >= 200 -> el gate (200) NO los cubre')
        print('La causa raiz es el search space Optuna, NO el n_train bajo')
        print('Fix correcto: ajustar gamma_max y MCW en settings.yaml')
    else:
        print('Hay colapsos con n_train < 200 -> el gate debería haberlos bloqueado')
        print('Posible bug: el gate no se ejecuta para todos los agentes bear')
