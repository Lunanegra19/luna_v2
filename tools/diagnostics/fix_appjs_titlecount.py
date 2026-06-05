"""Add feat-title-count update to app.js after feat-stat-total update"""

app_js_path = '/root/luna_v2/dashboard/app.js'

with open(app_js_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_block = "        document.getElementById('feat-stat-total').textContent = summary.total;"
new_block = """        document.getElementById('feat-stat-total').textContent = summary.total;
        // Update dynamic title count
        const titleCount = document.getElementById('feat-title-count');
        if (titleCount) titleCount.textContent = summary.total;
        console.log('[FIX-DASHBOARD] feat-title-count actualizado dinamicamente:', summary.total);"""

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    with open(app_js_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("[FIX-DASHBOARD] app.js: feat-title-count dinamico añadido correctamente")
else:
    print("[FIX-DASHBOARD] WARN: bloque no encontrado en app.js")
