"""
fix_ood_backslash.py  — elimina el caracter backslash espurio de la linea 196
"""
import ast, sys

fp = r'g:\Mi unidad\ia\luna_v2\luna\utils\ood_feature_guard.py'
with open(fp, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Linea 196 (0-indexed = 195)
bad_line = lines[195]
print(f"Linea 196 original: {repr(bad_line)}")

if bad_line.startswith('\\'):
    lines[195] = bad_line.lstrip('\\')
    print(f"Linea 196 corregida: {repr(lines[195])}")
else:
    print("No se encontro backslash espurio — abortando")
    sys.exit(0)

with open(fp, 'w', encoding='utf-8') as f:
    f.writelines(lines)

# Verificar sintaxis
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()
try:
    ast.parse(content)
    print("SYNTAX OK: ood_feature_guard.py")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)
