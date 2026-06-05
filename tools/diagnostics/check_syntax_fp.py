import ast
with open("/root/luna_v2/luna/features/feature_pipeline.py", "r") as f:
    src = f.read()
try:
    ast.parse(src)
    print("[SYNTAX-OK] feature_pipeline.py (VPS original) — sintaxis válida")
except SyntaxError as e:
    print(f"[SYNTAX-ERROR] {e}")
