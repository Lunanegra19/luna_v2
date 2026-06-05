import sys
sys.path.insert(0, "/root/luna_v2")
from dotenv import load_dotenv
load_dotenv()

print("[SYNTAX-CHECK] Importando ensemble_live_inference...")
try:
    from luna.live.ensemble_live_inference import LunaEnsembleLiveInference
    print("[SYNTAX-CHECK] OK - modulo importado sin errores de sintaxis")
except SyntaxError as e:
    print(f"[SYNTAX-CHECK] ERROR DE SINTAXIS: {e}")
    import sys; sys.exit(1)
except Exception as e:
    # Los ImportError de dependencias no son errores de sintaxis
    print(f"[SYNTAX-CHECK] Import exception (puede ser normal por dependencias): {type(e).__name__}: {e}")
    # Verificar solo sintaxis
    import ast
    with open("/root/luna_v2/luna/live/ensemble_live_inference.py", "r") as f:
        source = f.read()
    try:
        ast.parse(source)
        print("[SYNTAX-CHECK] Sintaxis Python OK (ast.parse exitoso)")
    except SyntaxError as se:
        print(f"[SYNTAX-CHECK] FALLO SINTAXIS: {se}")
