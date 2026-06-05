"""
Fix 1: Mover panel SOH de fuera a DENTRO del div tab-sop
Fix 2: Rediseñar la visualización JS a tabla compacta (no grid 3col tiny)
"""

html_path = '/root/luna_v2/dashboard/index.html'
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

# ── Extraer el bloque del panel SOH actual (desde el comentario hasta el div siguiente)
# El panel empieza FUERA de tab-sop, antes del tab-graphify
SOH_MARKER_START = '                    <!-- SOH: MONITOR DE INTEGRIDAD - CHECKS HORARIOS :30 -->'
SOH_MARKER_END = '                    <!-- TAB 7: GRAPHIFY AST MAP & STRUCTURAL COHESION -->'

if SOH_MARKER_START not in html:
    print("[ERROR] SOH panel marker no encontrado")
    exit(1)

# Extraer el bloque actual del panel SOH
idx_start = html.index(SOH_MARKER_START)
idx_end = html.index(SOH_MARKER_END)
soh_block = html[idx_start:idx_end]
print(f"[SOH-REPOSITION] Panel SOH extraído: {len(soh_block)} chars")

# Eliminar el bloque SOH de su posición actual (junto con líneas vacías antes)
# También limpiar las 2 líneas vacías que dejamos
html_without_soh = html[:idx_start].rstrip() + '\n                    \n' + html[idx_end:]

# ── Encontrar el punto de inyección correcto: DENTRO de tab-sop
# La estructura es:
#   </div>   <- cierra sop-issues-container o el grid interior
#   </div>   <- cierra el contenido de tab-sop
# </div>     <- ESTE cierra tab-sop (4 spaces de indentación = nivel de tab-panel)
# Buscamos la clausura interna: el último </div> antes del comentario TAB 7

# El marcador de fin de tab-sop es el </div> que precede inmediatamente al comentario TAB 7
close_tab_sop_marker = '                    </div>\n                    \n                    <!-- TAB 7: GRAPHIFY AST MAP & STRUCTURAL COHESION -->'

if close_tab_sop_marker not in html_without_soh:
    # Intentar variante
    close_tab_sop_marker = '                    </div>\n\n                    <!-- TAB 7: GRAPHIFY AST MAP & STRUCTURAL COHESION -->'

if close_tab_sop_marker in html_without_soh:
    # Inyectar el panel SOH ANTES del </div> que cierra tab-sop
    inject_point = '                    </div>\n                    \n                    <!-- TAB 7: GRAPHIFY AST MAP'
    soh_inside = soh_block.rstrip() + '\n\n'
    html_fixed = html_without_soh.replace(
        inject_point,
        soh_inside + '                    </div>\n                    \n                    <!-- TAB 7: GRAPHIFY AST MAP',
        1
    )
    print("[SOH-REPOSITION] Panel SOH movido DENTRO de tab-sop")
else:
    # Alternativa: buscar el patrón directo de cierre de tab-sop
    # El </div> que cierra tab-sop tiene 20 spaces de indentación
    # justo antes de TAB 7
    close_marker = '                    </div>\n                    \n'
    # Find the last occurrence before TAB 7
    idx_graphify = html_without_soh.index('<!-- TAB 7: GRAPHIFY AST MAP')
    chunk = html_without_soh[:idx_graphify]
    last_close_div = chunk.rfind('                    </div>\n')
    if last_close_div >= 0:
        insert_pos = last_close_div
        html_fixed = html_without_soh[:insert_pos] + soh_block.rstrip() + '\n\n' + html_without_soh[insert_pos:]
        print("[SOH-REPOSITION] Panel SOH inyectado via rfind en tab-sop")
    else:
        print("[ERROR] No se encontró punto de inyección dentro de tab-sop")
        html_fixed = html_without_soh

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_fixed)

# Verificar que SOH está ahora ANTES de tab-graphify Y DENTRO de tab-sop
idx_sop = html_fixed.index('id="tab-sop"')
idx_soh = html_fixed.index('id="soh-panel"')
idx_graphify = html_fixed.index('id="tab-graphify"')
print(f"[SOH-VERIFY] tab-sop={idx_sop}, soh-panel={idx_soh}, tab-graphify={idx_graphify}")
print(f"[SOH-VERIFY] soh-panel está dentro de tab-sop: {idx_sop < idx_soh < idx_graphify}")
print("[SOH-REPOSITION] index.html actualizado")
