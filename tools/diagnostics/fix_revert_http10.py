"""
[FIX-HTTP11-REVERT] Revierte protocol_version a "HTTP/1.0" en DashboardHTTPHandler.

ANÁLISIS DEL BUG:
- Con HTTP/1.1, `close_connection = False` (keep-alive) → Python espera más requests
- NGINX espera el body de la respuesta 302 sin Content-Length → DEADLOCK → login roto
- La solución es mantener HTTP/1.0 (close_connection=True siempre) 
  + Content-Length en _send_json() para respuestas grandes

POR QUÉ HTTP/1.0 FUNCIONA:
1. close_connection=True siempre → Python cierra tras cada response
2. NGINX detecta EOF → sabe que el body terminó
3. El _send_json() incluye Content-Length → NGINX lee exactamente el body JSON sin confundir el status line
4. NGINX proxy_http_version 1.1 sigue ayudando con buffers mayores
"""

SERVER_PATH = '/root/luna_v2/dashboard/server.py'

with open(SERVER_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

old_proto = '    # [FIX-HTTP11-CONTENT-LENGTH] HTTP/1.1 permite Content-Length explícito,\n    # evitando que NGINX interprete el raw status line como parte del body.\n    protocol_version = "HTTP/1.1"'

new_proto = '    # [FIX-HTTP10-STABLE] Mantenemos HTTP/1.0 para cerrar conexion tras cada response.\n    # close_connection=True (siempre) evita deadlocks de keep-alive con NGINX.\n    # Content-Length se añade en _send_json() para respuestas grandes (fix del bug JSON).\n    protocol_version = "HTTP/1.0"'

if old_proto in content:
    content = content.replace(old_proto, new_proto, 1)
    print('[FIX-HTTP11-REVERT] OK - protocol_version revertido a HTTP/1.0')
else:
    # Try finding and replacing
    idx = content.find('protocol_version = "HTTP/1.1"')
    if idx >= 0:
        print(f'[FIX-HTTP11-REVERT] Encontrado en posición {idx}')
        content = content.replace('protocol_version = "HTTP/1.1"', 'protocol_version = "HTTP/1.0"', 1)
        print('[FIX-HTTP11-REVERT] OK - protocol_version reemplazado directamente')
    else:
        print('[FIX-HTTP11-REVERT] ERROR - protocol_version no encontrado')
        exit(1)

with open(SERVER_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print('[FIX-HTTP11-REVERT] server.py guardado')

# Verificar
import subprocess, sys
result = subprocess.run(
    ['python3', '-c', f'import ast; ast.parse(open("{SERVER_PATH}").read()); print("SYNTAX OK")'],
    capture_output=True, text=True
)
print(f'[FIX-HTTP11-REVERT] Sintaxis: {result.stdout.strip() or result.stderr.strip()}')

# Verificar que HTTP/1.0 está
with open(SERVER_PATH, 'r', encoding='utf-8') as f:
    check = f.read()

if 'protocol_version = "HTTP/1.0"' in check:
    print('[FIX-HTTP11-REVERT] VERIFIED: protocol_version = "HTTP/1.0" en server.py ✓')
    print('[FIX-HTTP11-REVERT] Content-Length en _send_json() sigue activo ✓')
else:
    print('[FIX-HTTP11-REVERT] ERROR: protocol_version no actualizado')
    sys.exit(1)
