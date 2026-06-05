"""
[FIX-DASHBOARD-ENDPOINTS] Corrige los paths de los endpoints en settings.yaml.
Los paths configurados apuntaban a rutas inexistentes:
  - /api/live-state → NO EXISTE en dashboard/server.py
  - /api/hour-decision → NO EXISTE (es /api/vps/hour-decision)
  - /api/status → ✅ EXISTE pero la respuesta no tiene campo 'status' en root
                    (tiene 'system', 'wfb', 'prod', 'vps', 'signal_funnel')

Fix: ajustar required_fields y paths a los endpoints reales del dashboard.
"""
import yaml
from pathlib import Path

SETTINGS_PATH = Path('/root/luna_v2/config/settings.yaml')

with open(SETTINGS_PATH, 'r') as f:
    settings = yaml.safe_load(f)

# El dashboard /api/status devuelve JSON con keys: system, wfb, prod, vps, signal_funnel
# No tiene campo 'status' en root. El healthcheck debe verificar 'system' o 'vps'.
# /api/vps/hour-decision: devuelve JSON con 'status' en root (success/standby/error)
# /api/live-state: no existe en el dashboard — eliminamos este check y ponemos uno válido

OLD_ENDPOINTS = settings['dashboard_healthcheck']['endpoints']
print(f"[FIX-DASHBOARD-ENDPOINTS] Endpoints actuales: {[e['path'] for e in OLD_ENDPOINTS]}")

NEW_ENDPOINTS = [
    {
        'path': '/api/status',
        'description': 'Dashboard status principal',
        # /api/status devuelve {system: {...}, wfb: {...}, ...} — no tiene 'status' en root
        # Usamos 'system' como required_field (siempre presente)
        'required_fields': ['system'],
        'allow_standby': False,
    },
    {
        'path': '/api/vps/hour-decision',
        'description': 'Decisión horaria VPS (hour-decision)',
        'required_fields': ['status'],
        'allow_standby': True,  # puede devolver standby si no hay datos de la hora
    },
    {
        'path': '/api/db/latency-history',
        'description': 'Historial de latencia DB',
        'required_fields': [],  # endpoint legacy sin campo status
        'allow_standby': True,
    },
]

settings['dashboard_healthcheck']['endpoints'] = NEW_ENDPOINTS

with open(SETTINGS_PATH, 'w') as f:
    yaml.dump(settings, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

print(f"[FIX-DASHBOARD-ENDPOINTS] ✅ Endpoints actualizados:")
for ep in NEW_ENDPOINTS:
    print(f"  {ep['path']} | required={ep['required_fields']} | standby={ep['allow_standby']}")
print("[FIX-DASHBOARD-ENDPOINTS] settings.yaml actualizado correctamente.")
