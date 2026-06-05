"""Fix hardcoded 498 in index.html and update feature pool title dynamically"""

path = '/root/luna_v2/dashboard/index.html'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_title = '🧠 FEATURE POOL & TELEMETRÍA (498 VARIABLES CANÓNICAS)'
new_title = '🧠 FEATURE POOL & TELEMETRÍA (<span id="feat-title-count">512</span> VARIABLES CANÓNICAS)'

if old_title in content:
    content = content.replace(old_title, new_title)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[FIX-DASHBOARD] feat-title-count: reemplazado '498' con span dinamico en index.html")
else:
    print(f"[FIX-DASHBOARD] WARN: texto no encontrado. Buscando variantes...")
    # Try to find what's in the file
    import re
    matches = [m.start() for m in re.finditer('498', content)]
    for pos in matches:
        print(f"  498 en pos {pos}: {content[max(0,pos-30):pos+50]!r}")
    feat_pool_matches = [m.start() for m in re.finditer('FEATURE POOL', content)]
    for pos in feat_pool_matches:
        print(f"  FEATURE POOL en pos {pos}: {content[pos:pos+80]!r}")
