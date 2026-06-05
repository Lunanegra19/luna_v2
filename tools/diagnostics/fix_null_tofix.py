"""
[FIX-NULL-TOFIX] Fix null.toFixed() TypeError in decision modal metrics.

Bug: data.clock_drift_minutes and data.execution_latency_sec come from a LEFT JOIN
to operational_audit_logs. If no matching row, they are JSON null.
In JS: null !== undefined → TRUE (strict check fails for null!)
So null.toFixed(1) → TypeError → catch() → renderStandbyState() → STANDBY falso.

Fix: use != null (loose equality) which catches both null and undefined.
"""

path = '/root/luna_v2/dashboard/app.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Bug 1: clock_drift_minutes null check (strict !== undefined fails for null)
old1 = "driftEl.textContent = data.clock_drift_minutes !== undefined ? `${data.clock_drift_minutes.toFixed(1)} min` : '\u2014';"
new1 = "// [FIX-NULL-TOFIX] null !== undefined = true → null.toFixed() → TypeError. Usar != null\n                    driftEl.textContent = data.clock_drift_minutes != null ? `${data.clock_drift_minutes.toFixed(1)} min` : '\u2014';"

if old1 in content:
    content = content.replace(old1, new1, 1)
    print('[FIX-NULL-TOFIX] Bug 1 (clock_drift_minutes) corregido.')
else:
    # Try with — character alternatives
    old1b = "driftEl.textContent = data.clock_drift_minutes !== undefined ? `${data.clock_drift_minutes.toFixed(1)} min` : '--';"
    if old1b in content:
        content = content.replace(old1b, new1.replace('\u2014', '--'), 1)
        print('[FIX-NULL-TOFIX] Bug 1 (clock_drift_minutes) corregido (alt).')
    else:
        # Search by fragment
        import re
        pattern = r"driftEl\.textContent = data\.clock_drift_minutes !== undefined \?"
        match = re.search(pattern, content)
        if match:
            # Replace the full line
            line_start = content.rfind('\n', 0, match.start()) + 1
            line_end = content.find('\n', match.end())
            old_line = content[line_start:line_end]
            new_line = old_line.replace('!== undefined', '!= null')
            content = content[:line_start] + "                    // [FIX-NULL-TOFIX] Bug 1: null.toFixed fix\n" + new_line + content[line_end:]
            print(f'[FIX-NULL-TOFIX] Bug 1 corregido via regex.')
        else:
            print('[FIX-NULL-TOFIX/WARN] Bug 1 no encontrado.')

# Bug 2: execution_latency_sec null check
old2 = "latEl.textContent = data.execution_latency_sec !== undefined ? `${data.execution_latency_sec.toFixed(1)}s` : '\u2014';"
new2 = "// [FIX-NULL-TOFIX] Bug 2: execution_latency_sec null check\n                    latEl.textContent = data.execution_latency_sec != null ? `${data.execution_latency_sec.toFixed(1)}s` : '\u2014';"

if old2 in content:
    content = content.replace(old2, new2, 1)
    print('[FIX-NULL-TOFIX] Bug 2 (execution_latency_sec toFixed) corregido.')
else:
    import re
    pattern2 = r"latEl\.textContent = data\.execution_latency_sec !== undefined \?"
    match2 = re.search(pattern2, content)
    if match2:
        line_start = content.rfind('\n', 0, match2.start()) + 1
        line_end = content.find('\n', match2.end())
        old_line = content[line_start:line_end]
        new_line = old_line.replace('!== undefined', '!= null')
        content = content[:line_start] + "                    // [FIX-NULL-TOFIX] Bug 2: null.toFixed fix\n" + new_line + content[line_end:]
        print('[FIX-NULL-TOFIX] Bug 2 corregido via regex.')
    else:
        print('[FIX-NULL-TOFIX/WARN] Bug 2 no encontrado.')

# Bug 3: execution_latency_sec < 60 with null (null < 60 = true in JS, but we want safe)
old3 = "latEl.style.color = data.execution_latency_sec < 60 ? '#10b981' : '#f59e0b';"
new3 = "// [FIX-NULL-TOFIX] Bug 3: null < 60 = true in JS (unexpected), guard with null check\n                    latEl.style.color = (data.execution_latency_sec != null && data.execution_latency_sec < 60) ? '#10b981' : '#f59e0b';"
if old3 in content:
    content = content.replace(old3, new3, 1)
    print('[FIX-NULL-TOFIX] Bug 3 (latency color null) corregido.')
else:
    print('[FIX-NULL-TOFIX/INFO] Bug 3 ya corregido o no encontrado.')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('[FIX-NULL-TOFIX] app.js guardado con exito.')
