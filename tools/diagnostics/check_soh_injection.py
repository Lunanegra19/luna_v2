"""Fix: Inject run_sop_health_checks() before class DashboardHTTPHandler"""

path = '/root/luna_v2/dashboard/server.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

inject_marker = 'class DashboardHTTPHandler(http.server.SimpleHTTPRequestHandler):'

# Check if health checks function already exists
if 'run_sop_health_checks' in content:
    print("[SOH-BACKEND] Función run_sop_health_checks ya existe en server.py - solo verificando endpoint")
    if '/api/sop/health-checks' in content:
        print("[SOH-BACKEND] Endpoint /api/sop/health-checks ya existe")
    else:
        print("[WARN] Endpoint no existe - necesita re-inyección")
else:
    print("[SOH-BACKEND] Inyectando función run_sop_health_checks()...")

print(f"Marker encontrado: {inject_marker in content}")
print(f"health-checks endpoint: {'/api/sop/health-checks' in content}")
print(f"run_sop_health_checks fn: {'run_sop_health_checks' in content}")
