"""
[FIX-HTTP11-SIG] Corrige la firma de _send_json() para aceptar default=str parameter.
Evita TypeError cuando el handler llama _send_json(data, ensure_ascii=False, default=str, status=200)
"""

SERVER_PATH = '/root/luna_v2/dashboard/server.py'

with open(SERVER_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Busca la definición actual de _send_json
old_def_start = '    def _send_json(self, data, status=200, ensure_ascii=False):'
new_def_start = '    def _send_json(self, data, status=200, ensure_ascii=False, default=str):'

if old_def_start not in content:
    # Busca con todos los posibles formatos
    idx = content.find('def _send_json')
    if idx >= 0:
        print(f'[FIX-HTTP11-SIG] Found def at {idx}:')
        print(repr(content[idx:idx+100]))
    else:
        print('[FIX-HTTP11-SIG/ERROR] No se encuentra _send_json')
    exit(1)

# Reemplaza firma
content = content.replace(old_def_start, new_def_start, 1)
print(f'[FIX-HTTP11-SIG] OK - Firma actualizada: default=str añadido')

# También corrige __import__('json').dumps -> json.dumps dentro del helper
old_json_call = "__import__('json').dumps(data, ensure_ascii=ensure_ascii, default=str).encode('utf-8')"
new_json_call = "json.dumps(data, ensure_ascii=ensure_ascii, default=default).encode('utf-8')"

if old_json_call in content:
    content = content.replace(old_json_call, new_json_call, 1)
    print('[FIX-HTTP11-SIG] OK - json.dumps usa parametro default dinamico')
else:
    print('[FIX-HTTP11-SIG] WARN - old json.dumps call no encontrado, buscando variante...')
    idx2 = content.find('__import__')
    if idx2 >= 0:
        print(repr(content[idx2:idx2+100]))

with open(SERVER_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print('[FIX-HTTP11-SIG] server.py guardado.')

# Verificar
with open(SERVER_PATH, 'r', encoding='utf-8') as f:
    check = f.read()

if '_send_json(self, data, status=200, ensure_ascii=False, default=str)' in check:
    print('[FIX-HTTP11-SIG] VERIFIED OK - firma correcta en server.py')
else:
    print('[FIX-HTTP11-SIG] VERIFICATION FAILED')
