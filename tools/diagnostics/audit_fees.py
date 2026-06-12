import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import re

print("=== [AUDIT] Verificacion de centralizacion de fees (No-Fallback) ===")

src_dir = Path('luna')
files_to_check = list(src_dir.rglob('*.py'))

hardcodes_found = 0
correct_reads_found = 0

fee_regex = re.compile(r'0\.0015|0\.0025|0\.002|0\.001')

for fpath in files_to_check:
    text = fpath.read_text('utf-8', errors='ignore')
    
    # Check for correct config reads
    if 'cfg.sop.cost_pct' in text or '_cfg_sfi.sop.cost_pct' in text or '_cfg_xgb.sop.cost_pct' in text or '_cfg_meta.sop.cost_pct' in text or '_cfg_cost.sop.cost_pct' in text or '_cfg_meta_cal.sop.cost_pct' in text or '_cfg.sop.cost_pct' in text:
        correct_reads_found += 1
        print(f"[OK] {fpath.name} lee cost_pct de settings")

    # Look for suspicious hardcodes in actual logic (ignoring comments)
    lines = text.split('\n')
    for i, line in enumerate(lines, 1):
        clean_line = line.split('#')[0] # remove comments
        if 'cost' in clean_line.lower() or 'fee' in clean_line.lower():
            if fee_regex.search(clean_line):
                # Ignore test or diagnostic files if any sneaked in
                if 'test_' not in fpath.name and 'diagnostics' not in str(fpath):
                    print(f"[WARN] Posible hardcode en {fpath.name}:{i} -> {clean_line.strip()}")
                    hardcodes_found += 1

print("\n=== RESUMEN ===")
print(f"Archivos que leen correctamente cfg.sop.cost_pct: {correct_reads_found}")
print(f"Posibles hardcodes detectados: {hardcodes_found}")

if hardcodes_found > 0:
    print("\n[RESULTADO] FAIL: Se encontraron hardcodes en el codigo de produccion.")
    sys.exit(1)
else:
    print("\n[RESULTADO] PASS: El pipeline principal usa fees centralizados.")
