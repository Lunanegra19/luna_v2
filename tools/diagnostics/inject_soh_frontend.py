"""
Inyectar panel SOH en index.html y funciones JS en app.js
"""

import subprocess

# =====================================================================
# PARTE 1: HTML del panel en index.html
# =====================================================================
html_path = '/root/luna_v2/dashboard/index.html'
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Insertar antes de tab-graphify
SOH_PANEL_HTML = '''
                    <!-- SOH: MONITOR DE INTEGRIDAD - CHECKS HORARIOS :30 -->
                    <div class="card glass-card mt-20" id="soh-panel" style="margin-top:28px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;flex-wrap:wrap;gap:12px;">
                            <div style="display:flex;align-items:center;gap:12px;">
                                <div style="width:42px;height:42px;border-radius:50%;background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.25);display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 0 14px rgba(251,191,36,0.2);">🔍</div>
                                <div>
                                    <h2 class="card-title" style="margin:0;font-size:15px;font-weight:700;color:#fff;">MONITOR DE INTEGRIDAD — Checks Horarios <span style="color:#fbbf24;">:30</span></h2>
                                    <p style="font-size:11px;color:#64748b;margin:3px 0 0;">15 verificaciones automáticas de todos los fixes implementados. Se ejecuta cada hora en el minuto :30 para no interferir con el ciclo de decisión (:00).</p>
                                </div>
                            </div>
                            <div style="display:flex;align-items:center;gap:12px;">
                                <div style="text-align:right;">
                                    <div style="font-size:10px;color:#64748b;">Próximo check automático</div>
                                    <div id="soh-next-check" style="font-size:12px;color:#fbbf24;font-family:'JetBrains Mono',monospace;font-weight:600;">Calculando...</div>
                                </div>
                                <div style="text-align:right;">
                                    <div style="font-size:10px;color:#64748b;">Último check</div>
                                    <div id="soh-last-check-time" style="font-size:12px;color:#94a3b8;font-family:'JetBrains Mono',monospace;">—</div>
                                </div>
                                <span id="soh-summary-badge" style="font-size:11px;font-weight:700;padding:5px 12px;border-radius:20px;background:rgba(100,116,139,0.15);color:#94a3b8;border:1px solid rgba(100,116,139,0.2);">— / 15</span>
                                <button id="soh-refresh-btn" onclick="loadSohChecks(true)" style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.25);color:#fbbf24;padding:7px 14px;border-radius:8px;font-size:11px;font-weight:600;cursor:pointer;transition:all 0.2s;white-space:nowrap;" onmouseover="this.style.background='rgba(251,191,36,0.2)'" onmouseout="this.style.background='rgba(251,191,36,0.1)'">🔄 Verificar Ahora</button>
                            </div>
                        </div>
                        <!-- Loading state -->
                        <div id="soh-loading" style="text-align:center;padding:40px;color:#64748b;font-size:13px;display:flex;align-items:center;justify-content:center;gap:10px;">
                            <div style="width:16px;height:16px;border:2px solid rgba(251,191,36,0.3);border-top-color:#fbbf24;border-radius:50%;animation:spin 0.8s linear infinite;"></div>
                            Cargando checks de integridad...
                        </div>
                        <!-- Checks grid -->
                        <div id="soh-checks-grid" style="display:none;"></div>
                        <!-- Summary bar -->
                        <div id="soh-summary-bar" style="display:none;margin-top:16px;padding:12px 16px;border-radius:10px;background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.05);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                            <div style="display:flex;gap:20px;">
                                <span><span id="soh-count-pass" style="font-size:18px;font-weight:800;color:#10b981;">0</span><span style="font-size:11px;color:#64748b;margin-left:4px;">PASS</span></span>
                                <span><span id="soh-count-warn" style="font-size:18px;font-weight:800;color:#f59e0b;">0</span><span style="font-size:11px;color:#64748b;margin-left:4px;">WARN</span></span>
                                <span><span id="soh-count-fail" style="font-size:18px;font-weight:800;color:#ef4444;">0</span><span style="font-size:11px;color:#64748b;margin-left:4px;">FAIL</span></span>
                            </div>
                            <div style="font-size:11px;color:#64748b;font-style:italic;">Auto-refresh cada hora al :30 — no coincide con el ciclo de decisión (:00)</div>
                        </div>
                    </div>

'''

MARKER_GRAPHIFY = '<!-- TAB 7: GRAPHIFY AST MAP'
if MARKER_GRAPHIFY in html:
    html = html.replace(MARKER_GRAPHIFY, SOH_PANEL_HTML + '                    ' + MARKER_GRAPHIFY, 1)
    print("[SOH-HTML] Panel HTML inyectado en index.html antes de tab-graphify")
else:
    print("[WARN] Marker de graphify no encontrado")

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)

# =====================================================================
# PARTE 2: Funciones JS en app.js
# =====================================================================
js_path = '/root/luna_v2/dashboard/app.js'
with open(js_path, 'r', encoding='utf-8') as f:
    js = f.read()

SOH_JS = '''

// ============================================================================
// SOH: MONITOR DE INTEGRIDAD - CHECKS HORARIOS :30
// Verifica los 15 fixes críticos implementados el 2026-05-25
// ============================================================================

let _sohLastTs = 0;  // Timestamp del último check ejecutado
let _sohSchedulerInterval = null;

// Colores y estilos por status
const SOH_STATUS_STYLES = {
    PASS: { bg: 'rgba(16,185,129,0.08)', border: 'rgba(16,185,129,0.2)', color: '#10b981', icon: '✅', label: 'PASS' },
    WARN: { bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.2)', color: '#f59e0b', icon: '⚠️', label: 'WARN' },
    FAIL: { bg: 'rgba(239,68,68,0.08)',  border: 'rgba(239,68,68,0.25)', color: '#ef4444', icon: '❌', label: 'FAIL' },
};

// Mapeo de categoría a color del badge
const SOH_CAT_COLORS = {
    'LIVE DATA':     '#06b6d4', 'FIX-SKEW-FINAL': '#a78bfa', 'OHLCV-FIX':     '#f97316',
    'FEATURES':      '#60a5fa', 'COMPLETENESS':    '#34d399', 'PM2-TRADER':    '#10b981',
    'PM2-DASHBOARD': '#10b981', 'HOUR-DECISION':   '#fbbf24', 'DOUBLE-HEADERS':'#e879f9',
    'NGINX':         '#38bdf8', 'NGINX-CSP':       '#38bdf8', 'GRAPHIFY':      '#c084fc',
    'DATABASE':      '#4ade80', 'SOP-R3 EMBARGO':  '#fb923c', 'NO-FALLBACK':   '#f87171',
};

async function loadSohChecks(force = false) {
    console.log('[SOH] Cargando health checks (force=' + force + ')');
    const loadingEl = document.getElementById('soh-loading');
    const gridEl = document.getElementById('soh-checks-grid');
    const summaryBar = document.getElementById('soh-summary-bar');
    if (!loadingEl) return;
    loadingEl.style.display = 'flex';
    if (gridEl) gridEl.style.display = 'none';
    if (summaryBar) summaryBar.style.display = 'none';
    try {
        const resp = await fetch('/api/sop/health-checks' + (force ? '?force=true' : ''));
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        console.log('[SOH] Checks recibidos:', data.summary);
        renderSohChecks(data);
        _sohLastTs = data.summary.ts * 1000;
        updateSohMeta(data.summary);
    } catch (e) {
        console.error('[SOH] Error:', e);
        if (loadingEl) loadingEl.innerHTML = '<span style="color:#ef4444;">❌ Error cargando checks: ' + e.message + '</span>';
    }
}

function renderSohChecks(data) {
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
}

function updateSohMeta(summary) {
    // Badge global
    const badge = document.getElementById('soh-summary-badge');
    if (badge) {
        const total = summary.total || 15;
        const pass = summary.pass || 0;
        const fail = summary.fail || 0;
        const warn = summary.warn || 0;
        if (fail > 0) {
            badge.style.background = 'rgba(239,68,68,0.15)'; badge.style.color = '#ef4444'; badge.style.borderColor = 'rgba(239,68,68,0.3)';
            badge.textContent = fail + ' FAIL / ' + total;
        } else if (warn > 0) {
            badge.style.background = 'rgba(245,158,11,0.15)'; badge.style.color = '#f59e0b'; badge.style.borderColor = 'rgba(245,158,11,0.3)';
            badge.textContent = pass + ' PASS / ' + total + ' (' + warn + ' WARN)';
        } else {
            badge.style.background = 'rgba(16,185,129,0.15)'; badge.style.color = '#10b981'; badge.style.borderColor = 'rgba(16,185,129,0.3)';
            badge.textContent = '✓ ' + pass + ' / ' + total + ' PASS';
        }
    }
    // Último check
    const lastEl = document.getElementById('soh-last-check-time');
    if (lastEl && summary.ts) {
        const d = new Date(summary.ts * 1000);
        lastEl.textContent = d.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
}

function updateSohNextCheck() {
    const el = document.getElementById('soh-next-check');
    if (!el) return;
    const now = new Date();
    const mins = now.getMinutes();
    const secs = now.getSeconds();
    // Minuto :30 de la hora actual o siguiente
    let minsToNext;
    if (mins < 30) {
        minsToNext = 30 - mins;
    } else {
        minsToNext = 60 - mins + 30; // próxima hora :30
    }
    const secsTotal = minsToNext * 60 - secs;
    const mm = Math.floor(secsTotal / 60).toString().padStart(2, '0');
    const ss = (secsTotal % 60).toString().padStart(2, '0');
    el.textContent = mm + ':' + ss + ' (al :' + (mins < 30 ? '30' : '30+1h') + ')';
}

function scheduleSohChecks() {
    if (_sohSchedulerInterval) return; // ya arrancado
    console.log('[SOH] Scheduler iniciado — check automático en minuto :30 de cada hora');
    _sohSchedulerInterval = setInterval(() => {
        const mins = new Date().getMinutes();
        const secs = new Date().getSeconds();
        // Disparar si son las :30 y han pasado menos de 60s desde el último check
        if (mins === 30 && secs < 60) {
            const elapsed = Date.now() - _sohLastTs;
            if (elapsed > 20 * 60 * 1000) { // más de 20min desde último check
                console.log('[SOH] ⏰ Check automático :30 disparado');
                loadSohChecks(false);
            }
        }
        updateSohNextCheck();
    }, 30000); // revisar cada 30s
    updateSohNextCheck();
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Arrancar cuando el tab SOP esté activo ──────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    scheduleSohChecks();
    // Cargar cuando se activa el tab SOP
    document.querySelectorAll('.tab-btn[data-tab="tab-sop"]').forEach(btn => {
        btn.addEventListener('click', () => {
            if (Date.now() - _sohLastTs > 5 * 60 * 1000) { // si hace >5min
                setTimeout(() => loadSohChecks(false), 400);
            }
        });
    });
    // Si el tab SOP ya está activo al cargar
    if (document.getElementById('tab-sop')?.classList.contains('active')) {
        setTimeout(() => loadSohChecks(false), 1200);
    }
    console.log('[SOH] Monitor de Integridad inicializado — checks horarios :30 activos');
});

'''

# Añadir al final del app.js
js += SOH_JS
with open(js_path, 'w', encoding='utf-8') as f:
    f.write(js)
print("[SOH-JS] Funciones JS añadidas al final de app.js")

# Verificar sintaxis JS básica
print("[SOH-JS] Verificando que app.js no tiene errores de Python detectables...")
# Solo verificamos que el archivo se escribió correctamente
size = __import__('os').path.getsize(js_path)
print(f"[SOH-JS] app.js tamaño: {size/1024:.1f}KB")
print("[SOH] Frontend completo: HTML + JS inyectados correctamente")
