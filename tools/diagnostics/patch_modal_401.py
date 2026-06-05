"""Patch app.js fetch catch block to distinguish 401 from real standby."""
import re

path = '/root/luna_v2/dashboard/app.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Old: throw generic error for non-ok responses, then catch → renderStandbyState
old_fetch_block = """    fetch(apiUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })"""

new_fetch_block = """    // [FIX-MODAL-401] Distinguir 401 (sesión expirada) de error real de datos
    fetch(apiUrl)
        .then(response => {
            if (response.status === 401 || response.status === 403) {
                // Sesión expirada: redirigir a login con mensaje claro
                console.warn('[DECISION-MODAL-401] Sesión expirada o no autorizada. Redirigiendo a login.');
                document.getElementById('dec-modal-action').textContent = 'SESIÓN EXPIRADA';
                document.getElementById('dec-modal-action').style.color = '#f59e0b';
                document.getElementById('dec-modal-quorum').textContent = '---';
                document.getElementById('dec-modal-duration').textContent = '---';
                document.getElementById('dec-modal-regime').textContent = 'LOGIN REQUERIDO';
                document.getElementById('dec-modal-regime').style.color = '#f59e0b';
                const sc = document.getElementById('decision-modal-steps-container');
                if (sc) sc.innerHTML = '<div style="text-align:center;padding:40px;color:#f59e0b;font-family:monospace;font-size:12px;">⚠️ Sesión expirada. Por favor <a href=\"/login\" style=\"color:#3b82f6\">inicia sesión</a> de nuevo.</div>';
                throw new Error('session_expired');
            }
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })"""

if old_fetch_block in content:
    content = content.replace(old_fetch_block, new_fetch_block)
    print(f'[FIX-MODAL-401] Patch aplicado correctamente en fetch block')
else:
    print(f'[FIX-MODAL-401/WARN] No se encontro el bloque fetch exacto. Buscando alternativa...')
    # Try to patch just the !response.ok line
    old_simple = "            if (!response.ok) {\n                throw new Error(`HTTP error! status: ${response.status}`);\n            }"
    if content.count(old_simple) >= 1:
        # Only patch first occurrence (the decision modal one at ~line 5238)
        # Find it near 'DECISION-MODAL'
        idx = content.find('fetch(apiUrl)')
        if idx != -1:
            segment = content[idx:idx+500]
            if '!response.ok' in segment:
                new_segment = segment.replace(
                    "            if (!response.ok) {\n                throw new Error(`HTTP error! status: ${response.status}`);\n            }",
                    """            if (response.status === 401 || response.status === 403) {
                console.warn('[DECISION-MODAL-401] Sesion expirada. Redirigiendo a login.');
                document.getElementById('dec-modal-action').textContent = 'SESION EXPIRADA';
                document.getElementById('dec-modal-action').style.color = '#f59e0b';
                const sc = document.getElementById('decision-modal-steps-container');
                if (sc) sc.innerHTML = '<div style=\"text-align:center;padding:40px;color:#f59e0b;font-family:monospace;font-size:11px;\">⚠️ Sesión expirada. <a href=\"/login\" style=\"color:#3b82f6\">Inicia sesión de nuevo</a></div>';
                throw new Error('session_expired');
            }
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }""", 1)
                content = content[:idx] + new_segment + content[idx+500:]
                print('[FIX-MODAL-401] Patch alternativo aplicado.')
    else:
        print('[FIX-MODAL-401/ERROR] No se pudo aplicar el patch.')

# Also patch the catch block to NOT render standby on session_expired
old_catch = """        .catch(err => {
            console.error(\"[DECISION-MODAL-ERROR] Failed to fetch real decision logs:\", err);
            renderStandbyState();
        });"""

new_catch = """        .catch(err => {
            // [FIX-MODAL-401] No mostrar STANDBY si es sesión expirada (ya se maneja arriba)
            if (err.message !== 'session_expired') {
                console.error('[DECISION-MODAL-ERROR] Failed to fetch real decision logs:', err);
                renderStandbyState();
            }
        });"""

if old_catch in content:
    content = content.replace(old_catch, new_catch)
    print('[FIX-MODAL-401] Catch block patchado: 401 ya no llama renderStandbyState.')
else:
    print('[FIX-MODAL-401/WARN] Catch block no encontrado exactamente.')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('[FIX-MODAL-401] app.js guardado.')
