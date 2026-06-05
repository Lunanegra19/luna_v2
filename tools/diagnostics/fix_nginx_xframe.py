"""Fix nginx config: X-Frame-Options DENY → SAMEORIGIN and CSP unpkg.com"""

path = '/etc/nginx/sites-enabled/luna-dashboard'

# Try common paths
import os
for p in ['/etc/nginx/sites-enabled/luna-dashboard', '/etc/nginx/sites-enabled/default', '/etc/nginx/sites-enabled/luna']:
    if os.path.exists(p):
        path = p
        break

print(f"Editing nginx config: {path}")
with open(path, 'r') as f:
    content = f.read()

print(f"Original content has X-Frame-Options: {'X-Frame-Options' in content}")

# Fix 1: X-Frame-Options DENY -> SAMEORIGIN
old_xframe = 'add_header X-Frame-Options DENY;'
new_xframe = 'add_header X-Frame-Options SAMEORIGIN;'
if old_xframe in content:
    content = content.replace(old_xframe, new_xframe)
    print("[FIX-NGINX] X-Frame-Options DENY -> SAMEORIGIN")
else:
    print(f"[WARN] X-Frame-Options DENY not found as expected")
    # Search for it differently
    import re
    m = re.search(r'add_header X-Frame-Options.*?;', content)
    if m:
        print(f"  Found: {m.group()}")

# Fix 2: CSP - allow unpkg.com and jsdelivr.net for vis-network script
old_csp = 'add_header Content-Security-Policy "default-src \'self\'; script-src \'self\' \'unsafe-inline\'; style-src \'self\' \'unsafe-inline\'; img-src \'self\' data: blob:; font-src \'self\' data:;";'
new_csp = 'add_header Content-Security-Policy "default-src \'self\'; script-src \'self\' \'unsafe-inline\' https://unpkg.com https://cdn.jsdelivr.net; style-src \'self\' \'unsafe-inline\'; img-src \'self\' data: blob:; font-src \'self\' data:; connect-src \'self\';";'

if old_csp in content:
    content = content.replace(old_csp, new_csp)
    print("[FIX-NGINX] CSP actualizado para incluir unpkg.com y jsdelivr.net")
else:
    print(f"[WARN] CSP no encontrado exactamente. Intentando regex...")
    import re
    # More flexible replacement
    pattern = r'add_header Content-Security-Policy "([^"]+)";'
    match = re.search(pattern, content)
    if match:
        old_val = match.group(1)
        print(f"  CSP actual: {old_val}")
        # Update script-src to include unpkg.com
        new_val = old_val.replace(
            "script-src 'self' 'unsafe-inline'",
            "script-src 'self' 'unsafe-inline' https://unpkg.com https://cdn.jsdelivr.net"
        )
        if 'connect-src' not in new_val:
            new_val += " connect-src 'self';"
        new_csp_header = f'add_header Content-Security-Policy "{new_val}";'
        content = content.replace(match.group(), new_csp_header)
        print(f"[FIX-NGINX] CSP actualizado: {new_val}")

with open(path, 'w') as f:
    f.write(content)

print(f"[FIX-NGINX] Config guardado en {path}")
print("Verificando config nginx...")
import subprocess
result = subprocess.run(['nginx', '-t'], capture_output=True, text=True)
print(f"nginx -t stdout: {result.stdout}")
print(f"nginx -t stderr: {result.stderr}")
if result.returncode == 0:
    result2 = subprocess.run(['nginx', '-s', 'reload'], capture_output=True, text=True)
    print(f"nginx reload: {result2.stdout} {result2.stderr}")
    print("[FIX-NGINX] nginx recargado exitosamente")
else:
    print(f"[FIX-NGINX] ERROR en config nginx: {result.stderr}")
