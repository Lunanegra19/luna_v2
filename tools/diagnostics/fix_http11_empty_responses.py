"""
[FIX-HTTP11-EMPTY-RESPONSES] Añade Content-Length: 0 a todos los responses HTTP/1.1 
sin body (redirects 302, 404, etc.).

En HTTP/1.1, TODOS los responses deben tener Content-Length o Transfer-Encoding.
Sin Content-Length: 0 en redirects, NGINX/browser se queda esperando un body
que nunca llega, causando que el login se cuelgue.
"""

import re

SERVER_PATH = '/root/luna_v2/dashboard/server.py'

print("[FIX-HTTP11-EMPTY] Leyendo server.py...")
with open(SERVER_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

fixes = 0

# ============================================================
# FIX 1: send_response(302) + send_header('Location', ...) + end_headers()
# → Añadir Content-Length: 0 antes de end_headers()
# ============================================================

# Patrón: 302 + Location + [posiblemente Set-Cookie] + end_headers() SIN Content-Length
# Usamos una función para hacer el reemplazo seguro línea a línea

lines = content.split('\n')
new_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    new_lines.append(line)
    
    # Detecta send_response con código de redirect o error vacío
    stripped = line.strip()
    if stripped in ['self.send_response(302)', 'self.send_response(301)', 
                    'self.send_response(304)', 'self.send_response(404)',
                    'self.send_response(405)', 'self.send_response(403)']:
        
        indent = line[:len(line) - len(line.lstrip())]
        status_code = stripped.split('(')[1].rstrip(')')
        
        # Lookahead: busca el end_headers() en las próximas líneas
        # Para ver si ya tiene Content-Length y si hay wfile.write (body)
        j = i + 1
        has_content_length = False
        has_wfile_write = False
        end_headers_idx = None
        
        while j < min(i + 10, len(lines)):
            ls = lines[j].strip()
            if 'Content-Length' in ls:
                has_content_length = True
            if 'wfile.write' in ls:
                has_wfile_write = True
            if ls == 'self.end_headers()':
                end_headers_idx = j
                break
            if ls in ['return', 'pass', ''] or ls.startswith('def ') or ls.startswith('class '):
                if ls == '':
                    j += 1
                    continue
                break
            j += 1
        
        # Si hay end_headers sin Content-Length y sin body: inyectar Content-Length: 0
        if end_headers_idx is not None and not has_content_length and not has_wfile_write:
            # Insertar antes del end_headers: Content-Length: 0
            # Marcamos esta posición para la siguiente iteración
            print(f"[FIX-HTTP11-EMPTY] Línea {i+1}: send_response({status_code}) sin Content-Length → será corregido")
    
    i += 1

# Usamos regex para hacer el reemplazo de forma segura
# Patrón: send_response(3xx o 4xx) seguido de send_header(Location/etc.) 
# y end_headers() SIN Content-Length entre medio

# Fix 302 redirect + Location header (más común)
pattern_302_location = re.compile(
    r'(\s+)(self\.send_response\(30[0-9]\))\n'
    r'(\1\s+self\.send_header\(\'Location\'[^\n]+)\n'
    r'(\1\s+self\.end_headers\(\))',
    re.MULTILINE
)

def add_content_length_before_end(m):
    indent = m.group(1)
    send_resp = m.group(2)
    loc_header = m.group(3)
    end_headers = m.group(4)
    inner_indent = loc_header[:len(loc_header) - len(loc_header.lstrip())]
    return f"{indent}{send_resp}\n{loc_header}\n{inner_indent}self.send_header('Content-Length', '0')\n{end_headers}"

new_content, n1 = re.subn(pattern_302_location, add_content_length_before_end, content)
print(f"[FIX-HTTP11-EMPTY] {n1} redirects 302+Location corregidos")
content = new_content
fixes += n1

# Fix 302 redirect + Location + Set-Cookie headers
pattern_302_cookie = re.compile(
    r'(\s+)(self\.send_response\(30[0-9]\))\n'
    r'(\1\s+self\.send_header\(\'Location\'[^\n]+)\n'
    r'(\1\s+self\.send_header\(\'Set-Cookie\'[^\n]+)\n'
    r'(\1\s+self\.end_headers\(\))',
    re.MULTILINE
)

def add_cl_after_cookie(m):
    indent = m.group(1)
    send_resp = m.group(2)
    loc_header = m.group(3)
    cookie_header = m.group(4)
    end_headers = m.group(5)
    inner_indent = loc_header[:len(loc_header) - len(loc_header.lstrip())]
    return (f"{indent}{send_resp}\n{loc_header}\n{cookie_header}\n"
            f"{inner_indent}self.send_header('Content-Length', '0')\n{end_headers}")

new_content2, n2 = re.subn(pattern_302_cookie, add_cl_after_cookie, content)
print(f"[FIX-HTTP11-EMPTY] {n2} redirects 302+Location+Cookie corregidos")
content = new_content2
fixes += n2

# Fix 404 sin body
pattern_404 = re.compile(
    r'(\s+)(self\.send_response\(404\))\n'
    r'(\1\s+self\.end_headers\(\))',
    re.MULTILINE
)

def add_cl_404(m):
    indent = m.group(1)
    send_resp = m.group(2)
    end_headers = m.group(3)
    inner_indent = end_headers[:len(end_headers) - len(end_headers.lstrip())]
    return f"{indent}{send_resp}\n{inner_indent}self.send_header('Content-Length', '0')\n{end_headers}"

new_content3, n3 = re.subn(pattern_404, add_cl_404, content)
print(f"[FIX-HTTP11-EMPTY] {n3} respuestas 404 sin body corregidas")
content = new_content3
fixes += n3

# ============================================================
# Guardar
# ============================================================
with open(SERVER_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n[FIX-HTTP11-EMPTY] Total: {fixes} responses corregidos con Content-Length: 0")
print("[FIX-HTTP11-EMPTY] server.py guardado OK")

# Verificar sintaxis
import subprocess
result = subprocess.run(
    ['python3', '-c', f'import ast; ast.parse(open("{SERVER_PATH}").read()); print("SYNTAX OK")'],
    capture_output=True, text=True
)
print(f"[FIX-HTTP11-EMPTY] Verificacion sintaxis: {result.stdout.strip() or result.stderr.strip()}")
