"""
Auditoria completa del proyecto: busca valores hardcodeados criticos
que deberian venir de settings.yaml pero estan duplicados en codigo.

Categorias buscadas:
  1. .get("param", valor_numerico) - fallbacks en lectura de config
  2. getattr(obj, "param", valor_numerico) - fallbacks en atributos
  3. Constantes numericas criticas duplicadas en multiples archivos
  4. Bloques except con asignacion de fallback numerico
"""
import glob
import re
import os
import sys
from collections import defaultdict

from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent.parent)

# Directorios a excluir
EXCLUDE = ['__pycache__', '.git', r'data\\', r'logs\\', r'\\.venv', 'node_modules', os.path.join('tools', 'refactor')]  # [SCOPE-FIX 2026-06-24] refactor/migration one-off tools exentos del No-Fallback (no son pipeline produccion)

def should_exclude(path):
    return any(ex in path for ex in EXCLUDE)

# Recopilar archivos Python
py_files = [
    f for f in glob.glob(os.path.join(ROOT, '**', '*.py'), recursive=True)
    if not should_exclude(f)
]
yaml_files = [
    f for f in glob.glob(os.path.join(ROOT, '**', '*.yaml'), recursive=True)
    if not should_exclude(f)
]

print(f'[AUDIT] Archivos Python: {len(py_files)}')
print(f'[AUDIT] Archivos YAML:   {len(yaml_files)}')
print()

# Patrones criticos a buscar
patterns = {
    'get_default_numerico': re.compile(r'\.get\(["\'](\w+)["\'],\s*([\d.]+)\)'),
    'getattr_default_numerico': re.compile(r'getattr\([^,]+,\s*["\'](\w+)["\'],\s*([\d.e+]+)\)'),
    'fallback_except_asignacion': re.compile(r'self\.([A-Z_]+)\s*=\s*([\d.e+]+)'),
    'hardcoded_threshold': re.compile(r'(MIN_TRADES|MAX_PBO|MIN_DSR|PBO_N_BLOCKS|ALPHA_BINOMIAL|MAX_DRAWDOWN|EMBARGO|THRESHOLD|N_BLOCKS)\s*=\s*([\d.]+)'),
}

# Parametros conocidos que estan en settings.yaml
SETTINGS_PARAMS = {
    # Gauntlet y riesgo (CRITICO si fallback)
    'min_dsr', 'max_pbo', 'min_trades', 'alpha_binomial', 'max_drawdown',
    'pbo_n_blocks', 'total_return_cap', 'cusum_threshold', 'wfv_n_windows',
    'embargo_hours', 'purge_hours', 'mc_block_size_hours',
    'xgb_signal_threshold', 'meta_signal_threshold', 'kelly_fraction',
    'max_position', 'target_vol_annual', 'dd_kill_switch', 'dd_half_size',
    'momentum_filter_threshold', 'momentum_filter_threshold_upper',
    'ood_contamination', 'hmm_n_states',
    # [MEJORA-MATH-A 2026-06-18] Asymmetric TBM - parametros de mejora (no riesgo)
    'tbm_asymmetric', 'tbm_asymmetry_ratio_cap',
    # [MEJORA-MATH-B 2026-06-18] KNN Adaptativo SFI
    'sfi_knn_adaptive', 'sfi_mrmr_enabled',
    # [MEJORA-MATH-C 2026-06-18] Anchored AE Drift Loss
    'ae_anchored_kl_loss', 'ae_kl_lambda', 'ae_kl_drift_alarm_threshold',
}

# Parametros de riesgo critico: fallback produce RuntimeError (SOP R16)
CRITICAL_RISK_PARAMS = {
    'min_dsr', 'max_pbo', 'min_trades', 'alpha_binomial', 'max_drawdown',
    'pbo_n_blocks', 'embargo_hours', 'purge_hours', 'kelly_fraction',
    'dd_kill_switch', 'dd_half_size',
}


results = defaultdict(list)
hardcoded_constants = defaultdict(list)

for fpath in py_files:
    rel = os.path.relpath(fpath, ROOT)
    try:
        with open(fpath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        continue

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue

        # 1. get() con default numerico
        for m in patterns['get_default_numerico'].finditer(line):
            param_name = m.group(1).lower()
            default_val = m.group(2)
            if param_name in SETTINGS_PARAMS or 'n_blocks' in param_name or 'threshold' in param_name:
                results[rel].append({
                    'lineno': i,
                    'tipo': 'GET_DEFAULT',
                    'param': m.group(1),
                    'valor': default_val,
                    'linea': stripped[:120],
                    'riesgo': 'ALTO' if param_name in {'pbo_n_blocks','min_dsr','max_pbo','alpha_binomial'} else 'MEDIO'
                })

        # 2. getattr() con default numerico
        for m in patterns['getattr_default_numerico'].finditer(line):
            param_name = m.group(1).lower()
            default_val = m.group(2)
            # Solo reportar si: (a) es un param conocido de settings, o (b) el default != 0 (magic number potencial)
            if param_name in SETTINGS_PARAMS or float(default_val) != 0:
                # Severidad diferenciada: CRITICO para params de riesgo, MEDIO para params de mejora documentados
                if param_name in CRITICAL_RISK_PARAMS:
                    _sev = 'CRITICO'
                elif param_name in SETTINGS_PARAMS:
                    # Param conocido y documentado — solo informativo (MEDIO)
                    _sev = 'MEDIO'
                else:
                    # Default != 0 en param desconocido — potencial magic number
                    _sev = 'ALTO'
                results[rel].append({
                    'lineno': i,
                    'tipo': 'GETATTR_DEFAULT',
                    'param': m.group(1),
                    'valor': default_val,
                    'linea': stripped[:120],
                    'riesgo': _sev
                })

        # 3. Constantes criticas hardcodeadas
        for m in patterns['hardcoded_threshold'].finditer(line):
            const_name = m.group(1)
            const_val = m.group(2)
            hardcoded_constants[const_name].append({
                'file': rel,
                'lineno': i,
                'valor': const_val,
                'linea': stripped[:120]
            })

        # 4. Bloques except con asignacion en self (fallback silencioso)
        if 'except' in line.lower() and ('Exception' in line or 'Error' in line):
            # Mirar las 5 lineas siguientes para ver si hay asignaciones numericas
            for j in range(i, min(i+6, len(lines))):
                next_line = lines[j].strip()
                if re.search(r'self\.[A-Z_]+ *= *[\d.]+', next_line):
                    results[rel].append({
                        'lineno': j+1,
                        'tipo': 'FALLBACK_SILENCIOSO_EXCEPT',
                        'param': next_line[:60],
                        'valor': '(ver linea)',
                        'linea': next_line[:120],
                        'riesgo': 'CRITICO'
                    })


print('=' * 70)
print('HALLAZGOS POR ARCHIVO')
print('=' * 70)
total_findings = 0
criticos = []
altos = []
medios = []

for fpath, items in sorted(results.items()):
    for item in items:
        total_findings += 1
        entry = f"  {fpath}:L{item['lineno']} [{item['tipo']}] {item['param']}={item['valor']}"
        if item['riesgo'] == 'CRITICO':
            criticos.append(entry)
        elif item['riesgo'] == 'ALTO':
            altos.append(entry)
        else:
            medios.append(entry)

print(f'\n[CRITICO] {len(criticos)} hallazgos:')
for e in criticos:
    print(e)

print(f'\n[ALTO] {len(altos)} hallazgos:')
for e in altos:
    print(e)

print(f'\n[MEDIO] {len(medios)} hallazgos:')
for e in medios:
    print(e)

print()
print('=' * 70)
print('CONSTANTES DUPLICADAS EN MULTIPLES ARCHIVOS')
print('=' * 70)
has_inconsistencies = False
for const, occurrences in sorted(hardcoded_constants.items()):
    if len(occurrences) > 1:
        values = list(set(o['valor'] for o in occurrences))
        print(f'\n  {const}: {len(occurrences)} apariciones, valores={values}')
        for o in occurrences:
            print(f'    {o["file"]}:L{o["lineno"]} = {o["valor"]}')
        if len(values) > 1:
            print(f'    *** INCONSISTENCIA: valores distintos en diferentes archivos ***')
            has_inconsistencies = True

print(f'\n[TOTAL] {total_findings} hallazgos encontrados')

if has_inconsistencies:
    print("\n[AUDIT FAILED] Se encontraron inconsistencias en parametros fijos.")
    sys.exit(1)
