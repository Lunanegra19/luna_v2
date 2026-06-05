"""Fix RESIDUAL: pm2-action upfront headers - targeted single fix"""

path = '/root/luna_v2/dashboard/server.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# The pm2-action still has upfront headers after the try:
old = """        elif path == '/api/vps/pm2-action':
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                action = query_params.get('action', [None])[0]"""

new = """        elif path == '/api/vps/pm2-action':
            # [FIX-DOUBLE-HEADERS] Eliminados headers upfront - todo via _send_json
            try:
                action = query_params.get('action', [None])[0]"""

if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("[FIX-3a] pm2-action upfront headers eliminados correctamente")
else:
    print("[WARN] Patron no encontrado. Mostrando contexto alrededor de pm2-action...")
    idx = content.find("elif path == '/api/vps/pm2-action':")
    if idx >= 0:
        print(repr(content[idx:idx+400]))
