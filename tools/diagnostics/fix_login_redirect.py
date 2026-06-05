"""
[FIX-HTTP11-LOGIN-REDIRECT] Fix específico para el redirect 302 de login con Set-Cookie multiline.
El cookie Set-Cookie span varias líneas, así que el regex general no lo capturó.
"""

SERVER_PATH = '/root/luna_v2/dashboard/server.py'

with open(SERVER_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Busca el bloque exacto del login redirect con Set-Cookie multiline
old_block = """            self.send_response(302)
            self.send_header('Location', '/')
            self.send_header('Set-Cookie',
                             f'luna_session={token}; Path=/; HttpOnly; Secure; SameSite=Strict; Expires={exp_http}')
            self.end_headers()"""

new_block = """            self.send_response(302)
            self.send_header('Location', '/')
            self.send_header('Set-Cookie',
                             f'luna_session={token}; Path=/; HttpOnly; Secure; SameSite=Strict; Expires={exp_http}')
            # [FIX-HTTP11-LOGIN-REDIRECT] Content-Length: 0 obligatorio en HTTP/1.1 para redirects sin body
            self.send_header('Content-Length', '0')
            self.end_headers()"""

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    print('[FIX-HTTP11-LOGIN-REDIRECT] OK - login redirect 302+Cookie corregido')
else:
    print('[FIX-HTTP11-LOGIN-REDIRECT] ERROR - bloque exacto no encontrado')
    # Buscar variante
    idx = content.find("self.send_response(302)\n            self.send_header('Location', '/')\n            self.send_header('Set-Cookie'")
    if idx >= 0:
        print('Encontrado variante en:', repr(content[idx:idx+300]))
    else:
        print('Nada encontrado con Set-Cookie')

with open(SERVER_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print('[FIX-HTTP11-LOGIN-REDIRECT] server.py guardado')

# Verificar sintaxis
import subprocess
result = subprocess.run(
    ['python3', '-c', f'import ast; ast.parse(open("{SERVER_PATH}").read()); print("SYNTAX OK")'],
    capture_output=True, text=True
)
print(f'[FIX-HTTP11-LOGIN-REDIRECT] Sintaxis: {result.stdout.strip() or result.stderr.strip()}')
