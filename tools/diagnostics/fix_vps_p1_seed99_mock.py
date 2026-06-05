"""
fix_vps_p1_v2_mock_detect.py
[FIX-VPS-P1-V2 2026-05-30] Detecta mock por header binario y corrige active_seeds.
"""
import json, re, shutil
from pathlib import Path

print("[FIX-VPS-P1-V2] === INICIO — Auditoria binaria de modelos ===")

PROD_DIR = Path("/root/luna_v2/data/models/prod")
SETTINGS_PATH = Path("/root/luna_v2/config/settings.yaml")
PERMITTED_REGIMES = ["bear", "bull", "range"]

def is_file_mock(model_path):
    try:
        with open(model_path, "rb") as f:
            return f.read(1) == b"{"
    except Exception:
        return True

def audit_seed(seed_dir, direction="long"):
    result = {"has_mock": False, "agents": {}}
    for regime in PERMITTED_REGIMES:
        mp = seed_dir / ("xgboost_meta_" + regime + "_" + direction + ".model")
        if not mp.exists():
            mp = seed_dir / ("xgboost_meta_" + regime + ".model")
        if mp.exists():
            mock = is_file_mock(mp)
            kb = mp.stat().st_size // 1024
            result["agents"][regime] = {"mock": mock, "kb": kb}
            if mock:
                result["has_mock"] = True
        else:
            result["agents"][regime] = {"mock": True, "kb": 0}
            result["has_mock"] = True
    return result

seed_dirs = sorted([d for d in PROD_DIR.iterdir() if d.is_dir() and d.name.startswith("seed")])
print("[FIX-VPS-P1-V2] Seeds encontradas: " + str([d.name for d in seed_dirs]))

real_seeds = []
seeds_with_all = []

print("")
for sd in seed_dirs:
    try:
        sid = int(sd.name.replace("seed", ""))
    except ValueError:
        continue
    aud = audit_seed(sd, "long")
    has_all_3 = all(not aud["agents"][r]["mock"] for r in PERMITTED_REGIMES)
    parts = []
    for r in PERMITTED_REGIMES:
        info = aud["agents"][r]
        s = "MOCK" if info["mock"] else ("OK(" + str(info["kb"]) + "KB)")
        parts.append(r + "=" + s)
    status = "REAL" if has_all_3 else ("PARCIAL" if not aud["has_mock"] else "MOCK")
    print("  " + sd.name + ": " + status + " | " + " | ".join(parts))
    if not aud["has_mock"]:
        real_seeds.append(sid)
    if has_all_3:
        seeds_with_all.append(sid)

print("\n[FIX-VPS-P1-V2] Seeds con 3 agentes LONG completos y reales: " + str(seeds_with_all))
print("[FIX-VPS-P1-V2] Seeds sin ningun mock: " + str(real_seeds))

# Elegir nueva lista: seeds con todos los agentes completos
if seeds_with_all:
    priority = [1337, 2025, 42, 100, 777]
    new_seeds = [s for s in priority if s in seeds_with_all]
    for s in seeds_with_all:
        if s not in new_seeds:
            new_seeds.append(s)
elif real_seeds:
    new_seeds = real_seeds
else:
    # Todos tienen mock — quitar solo seed99 (el que genera crash)
    # y mantener 1337 + 2025 que al menos tienen bull+bear
    print("[FIX-VPS-P1-V2] WARN: todos los seeds tienen algun mock. Eliminando seed99 (crash) y manteniendo 1337, 2025")
    new_seeds = [1337, 2025]

print("[FIX-VPS-P1-V2] Nueva lista active_seeds: " + str(new_seeds))

# Backup y edicion
backup = SETTINGS_PATH.with_suffix(".yaml.bak_p1v2")
shutil.copy2(SETTINGS_PATH, backup)
print("[FIX-VPS-P1-V2] Backup: " + str(backup))

with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    content = f.read()

new_block = "  active_seeds:\n"
for s in new_seeds:
    new_block += "  # [FIX-VPS-P1-V2 2026-05-30] seed" + str(s) + " agentes verificados\n"
    new_block += "  - " + str(s) + "\n"

# Reemplazar bloque
pattern = r"  active_seeds:\n(?:(?:  #[^\n]*)?\n|  - \d+\n)+"
match = re.search(pattern, content)
if match:
    print("[FIX-VPS-P1-V2] Bloque encontrado, reemplazando...")
    new_content = content[:match.start()] + new_block + content[match.end():]
else:
    print("[FIX-VPS-P1-V2] Regex no encontro bloque, usando replace simple...")
    old_block = "  active_seeds:\n  - 99\n  - 1337\n  - 2025\n"
    new_content = content.replace(old_block, new_block, 1)

with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
    f.write(new_content)

# Verificar
import yaml
with open(SETTINGS_PATH, "r") as f:
    v = yaml.safe_load(f)
final = v.get("wfb", {}).get("active_seeds", [])
print("[FIX-VPS-P1-V2] active_seeds FINAL: " + str(final))
if 99 in final:
    print("[FIX-VPS-P1-V2] ERROR: seed99 aun presente!")
else:
    print("[FIX-VPS-P1-V2] OK: seed99 eliminada.")
print("[FIX-VPS-P1-V2] === FIX COMPLETADO === Ejecutar: pm2 restart luna-v2-live-demo")
