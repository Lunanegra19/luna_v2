"""
restore_seed99_settings.py — Restaura seed99 en active_seeds del VPS
[FIX-VPS-P1-V4b 2026-05-30]
"""
import re, yaml, shutil
from pathlib import Path

SETTINGS = Path("/root/luna_v2/config/settings.yaml")
bak = Path("/root/luna_v2/config/settings.yaml.bak_restore_seed99")
shutil.copy2(SETTINGS, bak)

with open(SETTINGS, "r") as f:
    txt = f.read()

new_block = (
    "  active_seeds:\n"
    "  # [FIX-VPS-P1-V4b 2026-05-30] seed99/1337/2025 XGBoost binarios verificados reales\n"
    "  - 99\n"
    "  - 1337\n"
    "  - 2025\n"
)

pattern = r"  active_seeds:\n(?:(?:  #[^\n]*)?\n|  - \d+\n)+"
m = re.search(pattern, txt)
if m:
    txt2 = txt[:m.start()] + new_block + txt[m.end():]
    with open(SETTINGS, "w") as f:
        f.write(txt2)
    with open(SETTINGS, "r") as f:
        v = yaml.safe_load(f)
    seeds = v["wfb"]["active_seeds"]
    print("active_seeds FINAL: " + str(seeds))
    print("seed99 restaurado: " + str(99 in seeds))
else:
    print("WARN: regex no encontro bloque active_seeds")
    print("Bloque actual:")
    for i, line in enumerate(txt.split("\n")):
        if "active_seeds" in line or "- 99" in line or "- 1337" in line:
            print("  " + str(i) + ": " + line)
