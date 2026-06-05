"""
Fix 3 bugs en run_sop_health_checks():
FIX-A: CHK-09 - mover s=line.strip() ANTES de los checks de in_soh_fn
FIX-B: CHK-06/07 - pasar env con PATH explícito al subprocess pm2
"""

path = '/root/luna_v2/dashboard/server.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# ─── FIX-A: CHK-09 - s definido ANTES de usarse ──────────────────────────
old_chk09_loop = '''        for i, line in enumerate(lines):
            if "def run_sop_health_checks" in s: in_soh_fn = True
            if in_soh_fn and "class DashboardHTTPHandler" in s: in_soh_fn = False
            if in_soh_fn: continue  # skip SOH fn body - contiene literales de codigo
            s = line.strip()
            if "def do_OPTIONS" in s: in_options = True'''

new_chk09_loop = '''        for i, line in enumerate(lines):
            s = line.strip()  # [FIX-CHK09-SCOPE] Definir s ANTES de cualquier uso
            if "def run_sop_health_checks" in s: in_soh_fn = True
            if in_soh_fn and "class DashboardHTTPHandler" in s: in_soh_fn = False
            if in_soh_fn: continue  # skip SOH fn body - contiene literales de codigo
            if "def do_OPTIONS" in s: in_options = True'''

if old_chk09_loop in content:
    content = content.replace(old_chk09_loop, new_chk09_loop, 1)
    print("[FIX-A] CHK-09: s=line.strip() movido ANTES de in_soh_fn checks")
else:
    print("[WARN-A] Patrón CHK-09 no encontrado exactamente")
    # Verificar si ya está corregido
    if 's = line.strip()  # [FIX-CHK09-SCOPE]' in content:
        print("[INFO-A] FIX-A ya aplicado previamente")

# ─── FIX-B: CHK-06/07 - pasar PATH explícito a pm2 ──────────────────────
# El server Python no hereda PATH con /usr/bin, pero pm2 está en /usr/bin/pm2
PM2_ENV = '_PM2_ENV = {"PATH": "/usr/bin:/usr/local/bin:/bin", "HOME": "/root"}'

old_chk06 = '''    def chk06():
        r = _sp.run(["pm2", "list"], capture_output=True, text=True, timeout=5)
        out = r.stdout
        if "luna-v2-live-demo" not in out: return "FAIL", "luna-v2-live-demo NO en PM2"
        chunk = out.split("luna-v2-live-demo")[1][:120]
        if "online" not in chunk: return "FAIL", "luna-v2-live-demo NO está online"
        return "PASS", "luna-v2-live-demo online en PM2"
    _chk("CHK-06", "PM2-TRADER", "Bot luna-v2-live-demo está online", chk06)

    def chk07():
        r = _sp.run(["pm2", "list"], capture_output=True, text=True, timeout=5)
        out = r.stdout
        if "luna-dashboard" not in out: return "FAIL", "luna-dashboard NO en PM2"
        chunk = out.split("luna-dashboard")[1][:120]
        if "online" not in chunk: return "FAIL", "luna-dashboard NO está online"
        return "PASS", "luna-dashboard online en PM2"
    _chk("CHK-07", "PM2-DASHBOARD", "Dashboard luna-dashboard está online", chk07)'''

new_chk06 = '''    # [FIX-B] PM2 path explícito — el server Python no hereda PATH con /usr/bin
    _PM2_ENV = {"PATH": "/usr/bin:/usr/local/bin:/bin", "HOME": "/root"}

    def chk06():
        r = _sp.run(["/usr/bin/pm2", "list"], capture_output=True, text=True, timeout=8, env=_PM2_ENV)
        out = r.stdout + r.stderr
        if r.returncode != 0 and not out.strip():
            return "FAIL", f"pm2 no responde (returncode={r.returncode})"
        if "luna-v2-live-demo" not in out:
            return "FAIL", f"luna-v2-live-demo NO en PM2 (pm2 salida: {len(out)} chars)"
        # Obtener estado del proceso
        chunk = out.split("luna-v2-live-demo")[1][:150] if "luna-v2-live-demo" in out else ""
        if "online" not in chunk:
            status_hint = chunk[:50].strip()
            return "FAIL", f"luna-v2-live-demo existe pero NO online: {status_hint}"
        # Extraer restarts si posible
        try:
            parts = [p.strip() for p in chunk.split("│") if p.strip()]
            restarts = parts[4] if len(parts) > 4 else "?"
        except Exception:
            restarts = "?"
        return "PASS", f"luna-v2-live-demo online en PM2 (restarts={restarts})"
    _chk("CHK-06", "PM2-TRADER", "Bot luna-v2-live-demo está online", chk06)

    def chk07():
        r = _sp.run(["/usr/bin/pm2", "list"], capture_output=True, text=True, timeout=8, env=_PM2_ENV)
        out = r.stdout + r.stderr
        if "luna-dashboard" not in out:
            return "FAIL", f"luna-dashboard NO en PM2 (pm2 salida: {len(out)} chars)"
        chunk = out.split("luna-dashboard")[1][:150] if "luna-dashboard" in out else ""
        if "online" not in chunk:
            return "FAIL", f"luna-dashboard existe pero NO online"
        return "PASS", "luna-dashboard online en PM2"
    _chk("CHK-07", "PM2-DASHBOARD", "Dashboard luna-dashboard está online", chk07)'''

if old_chk06 in content:
    content = content.replace(old_chk06, new_chk06, 1)
    print("[FIX-B] CHK-06/07: PATH explícito /usr/bin/pm2 aplicado")
else:
    print("[WARN-B] Patrón CHK-06/07 no encontrado exactamente")

# Guardar y verificar sintaxis
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

import subprocess
r = subprocess.run(['/root/miniconda3/envs/luna_env/bin/python', '-m', 'py_compile', path],
                  capture_output=True, text=True)
if r.returncode == 0:
    print("[FIXES] Sintaxis Python OK — server.py guardado")
else:
    print(f"[ERROR] Sintaxis: {r.stderr}")
