"""
[FIX-HTTP11-CONTENT-LENGTH] Fix crítico del protocolo HTTP del dashboard server.

CAUSA RAÍZ del bug "HTTP/1.0 2... is not valid JSON":
- Python usa HTTP/1.0 sin Content-Length
- NGINX no puede delimitar correctamente el body para respuestas grandes (>4KB)
- El status line de Python acaba en el body que ve el browser

FIX:
1. Añadir protocol_version = "HTTP/1.1" al DashboardHTTPHandler
2. Añadir método helper _send_json() con Content-Length automático
3. Reemplazar TODOS los bloques send_response+end_headers+wfile.write de APIs JSON
   con el nuevo helper
4. Actualizar NGINX config con proxy_http_version 1.1
"""

import re

SERVER_PATH = '/root/luna_v2/dashboard/server.py'

print("[FIX-HTTP11] Leyendo server.py...")
with open(SERVER_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# ============================================================
# FIX 1: Añadir protocol_version = "HTTP/1.1" a la clase
# y añadir helper _send_json()
# ============================================================
old_class_def = '''class DashboardHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Override directory to point to dashboard folder
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)'''

new_class_def = '''class DashboardHTTPHandler(http.server.SimpleHTTPRequestHandler):
    # [FIX-HTTP11-CONTENT-LENGTH] HTTP/1.1 permite Content-Length explícito,
    # evitando que NGINX interprete el status line como parte del body.
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        # Override directory to point to dashboard folder
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def _send_json(self, data, status=200, ensure_ascii=False):
        """[FIX-HTTP11-CONTENT-LENGTH] Helper que envía JSON con Content-Length correcto.
        
        Con HTTP/1.1, el Content-Length es OBLIGATORIO para que NGINX pueda
        delimitar el body correctamente. Sin él, el status line de Python acaba
        en el body del browser causando: SyntaxError: Unexpected token 'H'.
        """
        try:
            body = data if isinstance(data, bytes) else (
                data.encode('utf-8') if isinstance(data, str) else
                __import__('json').dumps(data, ensure_ascii=ensure_ascii, default=str).encode('utf-8')
            )
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            print(f"[FIX-HTTP11-CONTENT-LENGTH] JSON enviado: status={status} size={len(body)} bytes")
        except Exception as e:
            print(f"[FIX-HTTP11-CONTENT-LENGTH/ERROR] Error enviando JSON: {e}")'''

if old_class_def in content:
    content = content.replace(old_class_def, new_class_def, 1)
    print("[FIX-HTTP11] ✓ protocol_version + _send_json() helper añadido")
else:
    print("[FIX-HTTP11/ERROR] No se encontró el inicio de la clase DashboardHTTPHandler")
    exit(1)

# ============================================================
# FIX 2: Reemplazar todos los bloques de respuesta JSON
# del patrón:
#   self.send_response(XXX)
#   self.send_header('Content-Type', 'application/json...')
#   self.send_header('Access-Control-Allow-Origin', '*')
#   self.end_headers()
#   self.wfile.write(json.dumps(...).encode(...))
# con:
#   self._send_json(data, status=XXX)
# ============================================================

# Patrón regex para capturar bloques de respuesta JSON
pattern = re.compile(
    r'([ \t]*)self\.send_response\((\d+)\)\r?\n'
    r'\1[ \t]*self\.send_header\([\'"]Content-Type[\'"]\s*,\s*[\'"]application/json[^"\']*[\'"]\)\r?\n'
    r'\1[ \t]*self\.send_header\([\'"]Access-Control-Allow-Origin[\'"]\s*,\s*[\'"\*]+[\'"]\)\r?\n'
    r'\1[ \t]*self\.end_headers\(\)\r?\n'
    r'\1[ \t]*self\.wfile\.write\(json\.dumps\((.*?)\)\.encode\([^\)]*\)\)\r?\n',
    re.DOTALL
)

def replace_with_helper(m):
    indent = m.group(1)
    status_code = m.group(2)
    data_expr = m.group(3).strip()
    
    # Extrae ensure_ascii si está presente
    # json.dumps(data, ensure_ascii=False) -> data_expr includes the kwargs
    # We need to separate the data from kwargs
    replacement = f"{indent}self._send_json({data_expr}, status={status_code})\n"
    return replacement

new_content, count = re.subn(pattern, replace_with_helper, content)
print(f"[FIX-HTTP11] ✓ {count} bloques de respuesta JSON reemplazados con _send_json()")

if count > 0:
    content = new_content

# ============================================================
# FIX 3: También reemplazar el patrón con solo 1 send_header
# (solo Content-Type, sin Access-Control-Allow-Origin)
# ============================================================
pattern2 = re.compile(
    r'([ \t]*)self\.send_response\((\d+)\)\r?\n'
    r'\1[ \t]*self\.send_header\([\'"]Content-Type[\'"]\s*,\s*[\'"]application/json[^"\']*[\'"]\)\r?\n'
    r'(?!\1[ \t]*self\.send_header\([\'"]Access-Control)' # NO tiene Access-Control
    r'\1[ \t]*self\.end_headers\(\)\r?\n'
    r'\1[ \t]*self\.wfile\.write\(json\.dumps\((.*?)\)\.encode\([^\)]*\)\)\r?\n',
    re.DOTALL
)

new_content2, count2 = re.subn(pattern2, replace_with_helper, content)
print(f"[FIX-HTTP11] ✓ {count2} bloques adicionales reemplazados")
if count2 > 0:
    content = new_content2

# ============================================================
# Guardar el archivo
# ============================================================
with open(SERVER_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"[FIX-HTTP11] ✓ server.py guardado. Total cambios: {count + count2} endpoints")
print("[FIX-HTTP11] Siguiente: actualizar NGINX con proxy_http_version 1.1")
