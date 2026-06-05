"""
fix_obs03_hmm_dtype.py
Aplica FIX-OBS-03: convierte HMM_Regime float64->int en split_and_save
antes de guardar el parquet, previniendo BUG-HMM-FILTER-01.
"""
import ast
import sys

fp = r'g:\Mi unidad\ia\luna_v2\luna\features\feature_pipeline.py'

with open(fp, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Encontrar el indice del loop (0-based)
loop_idx = None
for i, l in enumerate(lines):
    if 'for name, data in datasets.items():' in l and 1670 < i < 1700:
        loop_idx = i
        break

if loop_idx is None:
    print("ERROR: no se encontro el loop en el rango esperado")
    sys.exit(1)

# El bloque 'continue' esta en loop_idx+3, 'path = ...' en loop_idx+4
insert_pos = loop_idx + 4  # insertar ANTES de 'path = _FEATURES_DIR / ...'

# Verificar que la linea en insert_pos es la de path
target_line = lines[insert_pos]
print(f"Linea en insert_pos {insert_pos}: {repr(target_line)}")
assert 'path = _FEATURES_DIR' in target_line, f"Linea incorrecta: {target_line}"

fix_block = '''\
\n            # [FIX-OBS-03-HMM-DTYPE 2026-05-30] Convertir HMM_Regime float64->int ANTES de guardar.
            # BUG-HMM-FILTER-01: filtro semantico compara int keys del state_map vs float64 -> fallo silencioso.
            for _hmm_dtype_col in ["HMM_Regime"]:
                if _hmm_dtype_col in data.columns and str(data[_hmm_dtype_col].dtype) in ("float64", "float32"):
                    data = data.copy()
                    data[_hmm_dtype_col] = data[_hmm_dtype_col].fillna(-1).astype(int)
                    print(  # RULE[fixbugsprints.md]
                        f"[FIX-OBS-03-HMM-DTYPE] Split '{name}': "
                        f"{_hmm_dtype_col} float64->int (BUG-HMM-FILTER-01 prevenido)"
                    )
                    logger.info(
                        f"[FIX-OBS-03-HMM-DTYPE] Split '{name}': {_hmm_dtype_col} "
                        f"dtype float64->int -- evita BUG-HMM-FILTER-01"
                    )
\n'''

new_lines = lines[:insert_pos] + [fix_block] + lines[insert_pos:]

with open(fp, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"FIX-OBS-03 insertado en posicion {insert_pos} (linea {insert_pos+1})")

# Verificar sintaxis
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()
try:
    ast.parse(content)
    print("SYNTAX OK: feature_pipeline.py")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)
