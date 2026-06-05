"""
[FIX-DASHBOARD-PORT] Corrige el puerto del healthcheck en settings.yaml:
5050 (incorrecto) → 8080 (puerto real donde escucha luna-dashboard).
Causa: settings.yaml en VPS tenía base_url apuntando al puerto incorrecto,
provocando 'Connection refused' en los 3 endpoints del healthcheck POST-CYCLE.
"""
p = '/root/luna_v2/config/settings.yaml'
with open(p, 'r') as f:
    c = f.read()

old = 'base_url: "http://localhost:5050"'
new = 'base_url: "http://localhost:8080"'

if old in c:
    c2 = c.replace(old, new)
    with open(p, 'w') as f:
        f.write(c2)
    print(f"[FIX-DASHBOARD-PORT] ✅ Puerto corregido: 5050 → 8080 en settings.yaml")
    for line in c2.splitlines():
        if 'base_url' in line:
            print(f"  Resultado: {line.strip()}")
elif 'base_url: "http://localhost:8080"' in c:
    print("[FIX-DASHBOARD-PORT] ✅ Puerto ya correcto (8080). Sin cambios.")
else:
    print(f"[FIX-DASHBOARD-PORT] ⚠️ base_url no encontrada con el valor esperado. Buscar manualmente.")
    for line in c.splitlines():
        if 'base_url' in line:
            print(f"  Encontrado: {line.strip()}")
