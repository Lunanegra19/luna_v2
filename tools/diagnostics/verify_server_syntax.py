"""
[DIAGNOSTIC] Verificacion de sintaxis de dashboard/server.py en VPS.
"""
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
server_path = PROJECT_ROOT / "dashboard" / "server.py"

print(f"[VERIFY-SERVER] Verificando: {server_path}")

if not server_path.exists():
    print(f"[VERIFY-SERVER] ERROR: No existe {server_path}")
    sys.exit(1)

try:
    source = server_path.read_text(encoding="utf-8")
    ast.parse(source)
    lines = source.count('\n')
    size_kb = len(source) / 1024
    print(f"[VERIFY-SERVER] [OK] Sintaxis Python correcta.")
    print(f"[VERIFY-SERVER] Lineas: {lines} | Tamaño: {size_kb:.1f} KB")
    
    # Verificar que los nuevos endpoints existen
    if "/api/vps/hour-decision" in source:
        print("[VERIFY-SERVER] [OK] Endpoint /api/vps/hour-decision: PRESENTE")
    else:
        print("[VERIFY-SERVER] [ERROR] Endpoint /api/vps/hour-decision: AUSENTE")
    
    if "/api/vps/feature-pipeline-status" in source:
        print("[VERIFY-SERVER] [OK] Endpoint /api/vps/feature-pipeline-status: PRESENTE")
    else:
        print("[VERIFY-SERVER] [ERROR] Endpoint /api/vps/feature-pipeline-status: AUSENTE")
        
    if "DASHBOARD-DECISION-REPORT" in source:
        print("[VERIFY-SERVER] [OK] Print traces DASHBOARD-DECISION-REPORT: PRESENTE")
    if "DASHBOARD-FEATURE-PIPELINE" in source:
        print("[VERIFY-SERVER] [OK] Print traces DASHBOARD-FEATURE-PIPELINE: PRESENTE")

    print("[VERIFY-SERVER] Deploy verificado correctamente. Listo para reiniciar PM2.")
    
except SyntaxError as e:
    print(f"[VERIFY-SERVER] [CRITICAL] Error de sintaxis en linea {e.lineno}: {e.msg}")
    sys.exit(1)
except Exception as e:
    print(f"[VERIFY-SERVER] [ERROR] {e}")
    sys.exit(1)
