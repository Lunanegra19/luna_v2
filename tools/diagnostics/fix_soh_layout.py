"""
Fix el renderizador JS: reemplazar grid 3-col de cards mini por
una tabla compacta 5-col que muestra todos los checks claramente.
"""

js_path = '/root/luna_v2/dashboard/app.js'
with open(js_path, 'r', encoding='utf-8') as f:
    js = f.read()

# Reemplazar la función renderSohChecks completa
old_render = '''function renderSohChecks(data) {
    const gridEl = document.getElementById('soh-checks-grid');
    const loadingEl = document.getElementById('soh-loading');
    if (!gridEl) return;
    const checks = data.checks || [];
    // Grid de 3 columnas
    let html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">';
    for (const chk of checks) {
        const st = SOH_STATUS_STYLES[chk.status] || SOH_STATUS_STYLES.WARN;
        const catColor = SOH_CAT_COLORS[chk.cat] || '#94a3b8';
        html += `
        <div style="background:${st.bg};border:1px solid ${st.border};border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:8px;transition:all 0.2s;" onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='none'">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;color:${catColor};background:${catColor}18;padding:2px 7px;border-radius:4px;border:1px solid ${catColor}30;">${escapeHtml(chk.cat)}</span>
                <div style="display:flex;align-items:center;gap:6px;">
                    <span style="font-size:14px;">${st.icon}</span>
                    <span style="font-size:9px;font-weight:800;padding:2px 7px;border-radius:10px;background:${st.color}18;color:${st.color};border:1px solid ${st.color}35;">${st.label}</span>
                </div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;color:#94a3b8;">${escapeHtml(chk.id)}</div>
            <div style="font-size:12px;font-weight:600;color:#e2e8f0;line-height:1.3;">${escapeHtml(chk.name)}</div>
            <div style="font-size:10px;color:${st.color};line-height:1.4;font-family:'JetBrains Mono',monospace;word-break:break-word;opacity:0.9;">${escapeHtml(chk.details || '')}</div>
        </div>`;
    }
    html += '</div>';
    gridEl.innerHTML = html;
    gridEl.style.display = 'block';
    if (loadingEl) loadingEl.style.display = 'none';
    // Summary bar
    const summaryBar = document.getElementById('soh-summary-bar');
    if (summaryBar) summaryBar.style.display = 'flex';
    const s = data.summary;
    const p = document.getElementById('soh-count-pass'); if (p) p.textContent = s.pass || 0;
    const w = document.getElementById('soh-count-warn'); if (w) w.textContent = s.warn || 0;
    const f = document.getElementById('soh-count-fail'); if (f) f.textContent = s.fail || 0;
}'''

new_render = '''function renderSohChecks(data) {
    const gridEl = document.getElementById('soh-checks-grid');
    const loadingEl = document.getElementById('soh-loading');
    if (!gridEl) return;
    const checks = data.checks || [];

    // Tabla compacta con 15 filas — mucho más legible que un grid de mini-cards
    let html = `<table style="width:100%;border-collapse:collapse;">
        <thead>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
                <th style="text-align:left;padding:8px 10px;font-size:10px;font-weight:700;color:#64748b;font-family:'JetBrains Mono',monospace;width:80px;">ID</th>
                <th style="text-align:left;padding:8px 10px;font-size:10px;font-weight:700;color:#64748b;width:130px;">CATEGORÍA</th>
                <th style="text-align:left;padding:8px 10px;font-size:10px;font-weight:700;color:#64748b;">CHECK</th>
                <th style="text-align:left;padding:8px 10px;font-size:10px;font-weight:700;color:#64748b;min-width:320px;">RESULTADO</th>
                <th style="text-align:center;padding:8px 10px;font-size:10px;font-weight:700;color:#64748b;width:70px;">STATUS</th>
            </tr>
        </thead>
        <tbody>`;

    for (const chk of checks) {
        const st = SOH_STATUS_STYLES[chk.status] || SOH_STATUS_STYLES.WARN;
        const catColor = SOH_CAT_COLORS[chk.cat] || '#94a3b8';
        const rowBg = chk.status === 'FAIL' ? 'rgba(239,68,68,0.04)' : chk.status === 'WARN' ? 'rgba(245,158,11,0.03)' : 'transparent';
        html += `
        <tr style="border-bottom:1px solid rgba(255,255,255,0.04);background:${rowBg};transition:background 0.15s;" onmouseover="this.style.background='rgba(255,255,255,0.03)'" onmouseout="this.style.background='${rowBg}'">
            <td style="padding:10px 10px;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;color:#475569;">${escapeHtml(chk.id)}</td>
            <td style="padding:10px 10px;">
                <span style="font-size:9px;font-weight:700;padding:2px 8px;border-radius:4px;background:${catColor}15;color:${catColor};border:1px solid ${catColor}25;white-space:nowrap;">${escapeHtml(chk.cat)}</span>
            </td>
            <td style="padding:10px 10px;font-size:12px;color:#cbd5e1;font-weight:500;line-height:1.35;">${escapeHtml(chk.name)}</td>
            <td style="padding:10px 10px;font-size:11px;font-family:'JetBrains Mono',monospace;color:${st.color};line-height:1.4;word-break:break-word;">${escapeHtml(chk.details || '—')}</td>
            <td style="padding:10px 10px;text-align:center;">
                <span style="font-size:16px;" title="${chk.status}">${st.icon}</span>
                <div style="font-size:8px;font-weight:800;color:${st.color};margin-top:2px;">${st.label}</div>
            </td>
        </tr>`;
    }
    html += '</tbody></table>';

    gridEl.innerHTML = html;
    gridEl.style.display = 'block';
    if (loadingEl) loadingEl.style.display = 'none';
    // Summary bar
    const summaryBar = document.getElementById('soh-summary-bar');
    if (summaryBar) summaryBar.style.display = 'flex';
    const s = data.summary;
    const p = document.getElementById('soh-count-pass'); if (p) p.textContent = s.pass || 0;
    const w = document.getElementById('soh-count-warn'); if (w) w.textContent = s.warn || 0;
    const f = document.getElementById('soh-count-fail'); if (f) f.textContent = s.fail || 0;
}'''

if old_render in js:
    js = js.replace(old_render, new_render, 1)
    print("[SOH-JS] renderSohChecks reemplazado con tabla compacta")
else:
    print("[WARN] Función renderSohChecks no encontrada exactamente")

with open(js_path, 'w', encoding='utf-8') as f:
    f.write(js)
print(f"[SOH-JS] app.js actualizado — {len(js)//1024}KB")
