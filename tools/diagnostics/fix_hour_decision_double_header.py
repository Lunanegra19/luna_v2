"""
[FIX-HOUR-DECISION-DOUBLE-HEADER] Elimina la doble emisión de headers HTTP en
el endpoint /api/vps/hour-decision.

BUG: El endpoint mandaba send_response(200)+end_headers() ANTES de la query DB,
y luego _send_json() mandaba OTRA VEZ headers como parte del body.
El browser veía "HTTP/1.0 200 OK\\r\\n..." como body → JSON error → STANDBY falso.

FIX: Eliminar los headers upfront. Todos los paths usan _send_json().
"""

SERVER_PATH = '/root/luna_v2/dashboard/server.py'

with open(SERVER_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# ============================================================
# FIX 1: Eliminar headers upfront + convertir el wfile.write standby/error
# ============================================================

old_block_top = """        elif path == '/api/vps/hour-decision':
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                start_utc = query_params.get('start_utc', [None])[0]
                end_utc = query_params.get('end_utc', [None])[0]
                local_date = query_params.get('local_date', [None])[0]
                local_hour = int(query_params.get('local_hour', [0])[0])

                print(f\"[DASHBOARD-HOUR-DECISION] Solicitud de decisión horaria: local_date={local_date} local_hour={local_hour} start_utc={start_utc} end_utc={end_utc}\")

                if not start_utc or not end_utc or not local_date:
                    self.wfile.write(json.dumps({\"status\": \"error\", \"message\": \"Missing parameters\"}, ensure_ascii=False).encode('utf-8'))
                    return"""

new_block_top = """        elif path == '/api/vps/hour-decision':
            try:
                # [FIX-HOUR-DECISION-DOUBLE-HEADER] Headers NO se envían upfront.
                # Todos los paths de respuesta usan _send_json() con Content-Length correcto.
                # Bug anterior: send_response+end_headers upfront + _send_json() duplicaba
                # los headers como body → browser recibía "HTTP/1.0 200 OK\\r\\n" como JSON → error.
                start_utc = query_params.get('start_utc', [None])[0]
                end_utc = query_params.get('end_utc', [None])[0]
                local_date = query_params.get('local_date', [None])[0]
                local_hour = int(query_params.get('local_hour', [0])[0])

                print(f\"[DASHBOARD-HOUR-DECISION] Solicitud de decisión horaria: local_date={local_date} local_hour={local_hour} start_utc={start_utc} end_utc={end_utc}\")

                if not start_utc or not end_utc or not local_date:
                    print(\"[DASHBOARD-HOUR-DECISION/ERROR] Parámetros faltantes en la petición\")
                    self._send_json({\"status\": \"error\", \"message\": \"Missing parameters\"}, status=400)
                    return"""

if old_block_top in content:
    content = content.replace(old_block_top, new_block_top, 1)
    print('[FIX-HOUR-DECISION-DOUBLE-HEADER] OK - headers upfront eliminados + error 400 via _send_json()')
else:
    print('[FIX-HOUR-DECISION-DOUBLE-HEADER] ERROR - bloque superior no encontrado, buscando variante...')
    idx = content.find("elif path == '/api/vps/hour-decision':")
    if idx >= 0:
        print(repr(content[idx:idx+500]))
    exit(1)

# ============================================================
# FIX 2: Convertir el wfile.write standby a _send_json()
# ============================================================

old_standby = """                if not row:
                    # No data found in the DB for this hour
                    print(f\"[DASHBOARD-HOUR-DECISION] No se encontró registro en DB para {local_date} {local_hour:02d}:00 hs. Retornando standby.\")
                    self.wfile.write(json.dumps({\"status\": \"standby\"}, ensure_ascii=False).encode('utf-8'))
                    return"""

new_standby = """                if not row:
                    # No data found in the DB for this hour
                    print(f\"[DASHBOARD-HOUR-DECISION] No se encontró registro en DB para {local_date} {local_hour:02d}:00 hs. Retornando standby.\")
                    # [FIX-HOUR-DECISION-DOUBLE-HEADER] Usar _send_json() con Content-Length
                    self._send_json({\"status\": \"standby\"}, status=200)
                    return"""

if old_standby in content:
    content = content.replace(old_standby, new_standby, 1)
    print('[FIX-HOUR-DECISION-DOUBLE-HEADER] OK - standby wfile.write → _send_json()')
else:
    print('[FIX-HOUR-DECISION-DOUBLE-HEADER] ERROR - bloque standby no encontrado')
    exit(1)

# ============================================================
# Guardar y verificar
# ============================================================
with open(SERVER_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print('[FIX-HOUR-DECISION-DOUBLE-HEADER] server.py guardado')

import subprocess
result = subprocess.run(
    ['python3', '-c', f'import ast; ast.parse(open("{SERVER_PATH}").read()); print("SYNTAX OK")'],
    capture_output=True, text=True
)
print(f'[FIX-HOUR-DECISION-DOUBLE-HEADER] Sintaxis: {result.stdout.strip() or result.stderr.strip()}')
