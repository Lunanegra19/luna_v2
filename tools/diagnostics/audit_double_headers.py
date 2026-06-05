"""
Detectar el patrón de double-headers en server.py:
Busca bloques donde se hace send_response(200) + end_headers() ANTES de llamar a _send_json()
lo que genera headers duplicados: el 2do set de headers va al body.

Patrón peligroso:
    self.send_response(200)
    self.send_header(...)
    self.end_headers()       <-- headers ya enviados
    ...
    self._send_json(...)     <-- vuelve a hacer send_response + end_headers → doble headers

"""

path = '/root/luna_v2/dashboard/server.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

issues = []
in_block = False
block_start = 0
block_lines = []
send_response_line = None
end_headers_line = None
found_end_headers = False

i = 0
while i < len(lines):
    line = lines[i].rstrip()
    stripped = line.strip()
    
    # Detectar if path == '/api/...'
    if ("if path == '/api/" in stripped or "elif path == '/api/" in stripped or 
        "if path == '/" in stripped or "elif path == '/" in stripped):
        in_block = True
        block_start = i + 1
        block_lines = [line]
        send_response_line = None
        end_headers_line = None
        found_end_headers = False
    
    if in_block:
        block_lines.append(line)
        
        if 'self.send_response(200)' in stripped and send_response_line is None:
            send_response_line = i + 1
        
        if 'self.end_headers()' in stripped and send_response_line is not None and not found_end_headers:
            end_headers_line = i + 1
            found_end_headers = True
        
        if found_end_headers and 'self._send_json(' in stripped:
            # DOBLE HEADERS DETECTADO
            path_line = None
            for bl in block_lines:
                if "path == '/" in bl:
                    path_line = bl.strip()
                    break
            issues.append({
                'path': path_line,
                'send_response_at': send_response_line,
                'end_headers_at': end_headers_line,
                'send_json_at': i + 1,
            })
            found_end_headers = False
            send_response_line = None
            end_headers_line = None
        
        # Fin del bloque si llegamos a otra ruta
        if i > block_start + 2 and ("if path == '/api/" in stripped or "elif path == '/api/" in stripped or 
                                       "if path == '/" in stripped or "elif path == '/" in stripped):
            in_block = True
            block_start = i + 1
            block_lines = [line]
            send_response_line = None
            end_headers_line = None
            found_end_headers = False
    
    i += 1

if issues:
    print(f"\n[DOUBLE-HEADERS AUDIT] {len(issues)} endpoints con patron DOBLE HEADERS detectado!\n")
    for iss in issues:
        print(f"  Endpoint: {iss['path']}")
        print(f"    send_response(200) en linea: {iss['send_response_at']}")
        print(f"    end_headers() en linea:      {iss['end_headers_at']}")
        print(f"    _send_json() en linea:       {iss['send_json_at']}")
        print()
else:
    print("[DOUBLE-HEADERS AUDIT] No se detectaron patrones de doble headers con metodo basico.")
    print("Usando metodo alternativo: buscar end_headers() seguido de _send_json() en ventana de 20 lineas...")
    
    # Metodo alternativo: sliding window
    for i, line in enumerate(lines):
        if 'self.end_headers()' in line:
            # Mirar las siguientes 20 lineas
            window = lines[i+1:i+21]
            for j, wline in enumerate(window):
                if 'self._send_json(' in wline:
                    # Verificar que hay un send_response antes del end_headers
                    pre_window = lines[max(0,i-15):i]
                    has_send_response = any('self.send_response(200)' in pl for pl in pre_window)
                    if has_send_response:
                        # Buscar el path de la ruta
                        path_context = ''
                        for k in range(max(0, i-30), i):
                            if "path == '/" in lines[k]:
                                path_context = lines[k].strip()
                        print(f"\n[BUG] Doble headers en linea {i+1} (end_headers) + linea {i+1+j+1} (_send_json)")
                        print(f"  Contexto de ruta: {path_context}")
                        print(f"  end_headers():  {line.rstrip()}")
                        print(f"  _send_json():   {wline.rstrip()}")
