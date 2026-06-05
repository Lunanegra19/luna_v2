"""
[FIX-NGINX-HTTP11] Actualiza la config de NGINX para usar HTTP/1.1 con el upstream Python.

SIN proxy_http_version 1.1, NGINX usa HTTP/1.0 al backend. Con HTTP/1.0 y sin
Content-Length, NGINX puede malinterpretar las respuestas grandes del servidor Python,
causando que el raw HTTP response (status line + headers) acabe en el body del browser.
"""

import subprocess

NGINX_CONF = '/etc/nginx/sites-enabled/luna-dashboard'

with open(NGINX_CONF, 'r', encoding='utf-8') as f:
    content = f.read()

old_location = '''    location / {
        proxy_pass         http://127.0.0.1:8080;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 60s;
    }'''

new_location = '''    location / {
        proxy_pass         http://127.0.0.1:8080;
        # [FIX-NGINX-HTTP11] HTTP/1.1 + Content-Length evitan que el raw status line
        # de Python acabe como body del browser ("HTTP/1.0 2..." is not valid JSON).
        proxy_http_version 1.1;
        proxy_set_header   Connection "";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 120s;
        # Buffer config para respuestas grandes (step logs > 4KB)
        proxy_buffer_size  16k;
        proxy_buffers      8 16k;
        proxy_busy_buffers_size 32k;
    }'''

if old_location in content:
    content = content.replace(old_location, new_location, 1)
    print('[FIX-NGINX-HTTP11] OK - location block actualizado')
else:
    print('[FIX-NGINX-HTTP11/ERROR] location block original no encontrado')
    print('Contenido actual:')
    idx = content.find('location /')
    if idx >= 0:
        print(repr(content[idx:idx+300]))
    exit(1)

with open(NGINX_CONF, 'w', encoding='utf-8') as f:
    f.write(content)
print('[FIX-NGINX-HTTP11] nginx.conf guardado')

# Test NGINX config
result = subprocess.run(['nginx', '-t'], capture_output=True, text=True)
if result.returncode == 0:
    print('[FIX-NGINX-HTTP11] nginx -t OK')
    # Reload
    result2 = subprocess.run(['nginx', '-s', 'reload'], capture_output=True, text=True)
    if result2.returncode == 0:
        print('[FIX-NGINX-HTTP11] nginx reload OK')
    else:
        print(f'[FIX-NGINX-HTTP11/ERROR] nginx reload: {result2.stderr}')
else:
    print(f'[FIX-NGINX-HTTP11/ERROR] nginx -t failed: {result.stderr}')
    exit(1)
