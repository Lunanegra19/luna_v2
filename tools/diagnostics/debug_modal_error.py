"""
[DEBUG-VISIBLE] Añade visibilidad del error exacto al catch block del modal de decisión.
Cuando hay un error JS en el success path, muestra el tipo de error y mensaje
directamente en el header del modal en lugar de simplemente renderStandbyState().
"""

path = '/root/luna_v2/dashboard/app.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the catch block
marker = "console.error('[DECISION-MODAL-ERROR] Failed to fetch real decision logs:', err);"
idx = content.find(marker)
if idx < 0:
    print('[DEBUG-VISIBLE/ERROR] Marker not found in app.js')
    exit(1)

# Insert visible debug BEFORE renderStandbyState
insert_after_marker = idx + len(marker)
# Find the renderStandbyState() call after the marker
rs_idx = content.find('renderStandbyState();', insert_after_marker)
if rs_idx < 0:
    print('[DEBUG-VISIBLE/ERROR] renderStandbyState not found after catch marker')
    exit(1)

debug_code = """
                // [DEBUG-VISIBLE-ERROR] Mostrar error exacto en modal header para diagnóstico
                try {
                    const _dbgAct = document.getElementById('dec-modal-action');
                    const _dbgReg = document.getElementById('dec-modal-regime');
                    if (_dbgAct) _dbgAct.textContent = 'ERR:' + (err.name || '?') + ':' + (err.message || 'unknown').substring(0, 50);
                    if (_dbgReg) { _dbgReg.textContent = (err.stack || 'no-stack').split('\\n').slice(0,2).join(' | ').substring(0, 100); _dbgReg.style.color = '#ef4444'; }
                } catch(_e) {}
                """

content = content[:rs_idx] + debug_code + content[rs_idx:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('[DEBUG-VISIBLE] Patch aplicado. El modal ahora mostrará el error exacto en el header.')
print(f'[DEBUG-VISIBLE] Insertado en posición {rs_idx}')
