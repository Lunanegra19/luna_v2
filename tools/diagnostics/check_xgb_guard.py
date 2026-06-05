"""Check if [FIX-XGB-TRAZABILIDAD] appears in last cycle and why it's missing."""
content = open('/root/.pm2/logs/luna-v2-live-demo-out.log', 'r', errors='replace').read()
cycles = content.split('Iniciando Ciclo Operativo LUNA V2')
last = cycles[-1][:8000]
print(f'[FIX-XGB found in last 8000]: {chr(91)}FIX-XGB-TRAZABILIDAD{chr(93)} in last: ', '[FIX-XGB-TRAZABILIDAD]' in last)

# Buscar en el último ciclo expandiendo
last_full = cycles[-1]
print(f'Longitud total último ciclo: {len(last_full)} chars')
print(f'[FIX-XGB found in full]: ', '[FIX-XGB-TRAZABILIDAD]' in last_full)

# Mostrar los últimos 300 chars
print('\n--- Últimas 300 chars del último ciclo:')
print(last_full[-300:])

# ¿Es el último ciclo el más reciente de verdad?
import re
ts_pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')
timestamps = ts_pattern.findall(last_full)
if timestamps:
    print(f'\n--- Timestamps en último ciclo: first={timestamps[0]} last={timestamps[-1]}')
