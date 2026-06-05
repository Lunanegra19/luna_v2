"""
fix_vps_p1_v3_mock_detection.py
[FIX-VPS-P1-V3 2026-05-30] Fix definitivo del bug Mock Detection.

CAUSA RAIZ:
  XGBoost binario empieza con bytes 0x7b 0x4c ('{L') en formato msgpack.
  El codigo de deteccion mock usaba open(path, 'r') modo TEXTO y leia '{' como primer char.
  Esto marcaba TODOS los modelos XGBoost reales como Mock → Security Guard crasheaba.

FIX:
  Cambiar la deteccion a modo binario (rb).
  Un mock JSON real tiene bytes: 7b 22 ({"...) o 7b 0a ({\\n) o 7b 20 ({ ...).
  Un XGBoost binario tiene bytes: 7b 4c 00 00 ({L\\x00\\x00).
  Solo es mock si header[1] != 0x4c (no es XGBoost msgpack).
"""
import re
import shutil
from pathlib import Path

print("[FIX-VPS-P1-V3] === INICIO Fix Mock Detection Bug ===")

FILES_TO_FIX = [
    Path("/root/luna_v2/luna/live/ensemble_live_inference.py"),
    Path("/root/luna_v2/luna/models/regime_router.py"),
]

# La deteccion correcta en modo binario
OLD_PATTERN_TEXT = """                        is_mock = False
                        try:
                            with open(model_path, 'r', encoding='utf-8') as _fm:
                                if _fm.read(100).strip().startswith('{'):
                                    is_mock = True
                        except Exception:
                            pass"""

NEW_PATTERN_TEXT = """                        is_mock = False
                        try:
                            # [FIX-VPS-P1-V3 2026-05-30] Deteccion Mock en modo BINARIO.
                            # XGBoost real: header 0x7b 0x4c 0x00 0x00 (msgpack '{L\\x00\\x00')
                            # Mock JSON:    header 0x7b 0x22 o 0x7b 0x20 ('{"' o '{ ')
                            with open(model_path, 'rb') as _fm:
                                _hdr = _fm.read(4)
                            # Es mock si empieza con '{' pero NO con '{L' (XGBoost msgpack)
                            is_mock = (_hdr[:1] == b'{') and (_hdr[1:2] != b'\\x4c')
                            print(f"[MOCK-DETECT-FIX] {model_path.name}: header={_hdr[:4].hex()} | is_mock={is_mock}")
                        except Exception:
                            pass"""

# Fix en ensemble_live_inference.py
elf_path = FILES_TO_FIX[0]
with open(elf_path, "r", encoding="utf-8") as f:
    content = f.read()

# Backup
bak = elf_path.with_suffix(".py.bak_p1v3")
shutil.copy2(elf_path, bak)
print(f"[FIX-VPS-P1-V3] Backup: {bak}")

# Contar ocurrencias del patron en ensemble_live_inference
old_in_elf = """                    is_mock = False
                        try:
                            with open(model_path, 'r', encoding='utf-8') as _fm:
                                if _fm.read(100).strip().startswith('{'):
                                    is_mock = True
                        except Exception:
                            pass"""

# Buscar todos los bloques de deteccion is_mock con open texto
pattern = r"([ ]+)is_mock = False\n\s+try:\n\s+with open\(model_path, 'r', encoding='utf-8'\) as _fm:\n\s+if _fm\.read\(100\)\.strip\(\)\.startswith\('\{'\):\n\s+is_mock = True\n\s+except Exception:\n\s+pass"

matches = list(re.finditer(pattern, content))
print(f"[FIX-VPS-P1-V3] Encontrados {len(matches)} bloques de deteccion mock en {elf_path.name}")

if matches:
    # Reemplazar todos con la version binaria (preservando la indentacion)
    def make_replacement(indent):
        return (indent + "is_mock = False\n" +
                indent + "try:\n" +
                indent + "    # [FIX-VPS-P1-V3 2026-05-30] Deteccion Mock en modo BINARIO.\n" +
                indent + "    # XGBoost real: header 0x7b 0x4c 0x00 0x00 (msgpack)\n" +
                indent + "    # Mock JSON: header 0x7b 0x22 o 0x7b 0x20\n" +
                indent + "    with open(model_path, 'rb') as _fm:\n" +
                indent + "        _hdr = _fm.read(4)\n" +
                indent + "    is_mock = (_hdr[:1] == b'{') and (_hdr[1:2] != b'\\x4c')\n" +
                indent + "    print('[MOCK-DETECT-FIX] ' + str(model_path.name) + ': header=' + _hdr[:4].hex() + ' | is_mock=' + str(is_mock))\n" +
                indent + "except Exception:\n" +
                indent + "    pass")

    new_content = content
    for m in reversed(matches):
        indent = m.group(1)
        replacement = make_replacement(indent)
        new_content = new_content[:m.start()] + replacement + new_content[m.end():]
    
    with open(elf_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[FIX-VPS-P1-V3] {elf_path.name}: {len(matches)} bloques corregidos.")
else:
    print(f"[FIX-VPS-P1-V3] WARN: No se encontraron bloques con regex en {elf_path.name}. Verificar manualmente.")
    # Mostrar las lineas relevantes
    for i, line in enumerate(content.split('\n'), 1):
        if 'is_mock' in line or "startswith('{')" in line:
            print(f"  L{i}: {line}")

# Fix en regime_router.py
rr_path = FILES_TO_FIX[1]
with open(rr_path, "r", encoding="utf-8") as f:
    rr_content = f.read()

bak_rr = rr_path.with_suffix(".py.bak_p1v3")
shutil.copy2(rr_path, bak_rr)
print(f"[FIX-VPS-P1-V3] Backup: {bak_rr}")

# Pattern para regime_router (indentacion puede diferir)
rr_matches = list(re.finditer(pattern, rr_content))
print(f"[FIX-VPS-P1-V3] Encontrados {len(rr_matches)} bloques en {rr_path.name}")

if rr_matches:
    new_rr_content = rr_content
    for m in reversed(rr_matches):
        indent = m.group(1)
        replacement = make_replacement(indent)
        new_rr_content = new_rr_content[:m.start()] + replacement + new_rr_content[m.end():]
    with open(rr_path, "w", encoding="utf-8") as f:
        f.write(new_rr_content)
    print(f"[FIX-VPS-P1-V3] {rr_path.name}: {len(rr_matches)} bloques corregidos.")
else:
    # Buscar el patron alternativo en regime_router (puede tener indentacion distinta)
    alt_pattern = r"(_start\.startswith\('\{'\))"
    alt_matches = list(re.finditer(alt_pattern, rr_content))
    print(f"[FIX-VPS-P1-V3] Bloques alternativos en {rr_path.name}: {len(alt_matches)}")
    for m in alt_matches:
        print(f"  Linea aprox: ...{rr_content[max(0,m.start()-100):m.end()+100]}...")

print("\n[FIX-VPS-P1-V3] Verificando resultado final...")
# Verificar que no quedan detecciones texto
with open(elf_path, "r") as f:
    elf_final = f.read()
remaining = elf_final.count("with open(model_path, 'r', encoding='utf-8') as _fm:")
print(f"[FIX-VPS-P1-V3] {elf_path.name} — detecciones texto restantes: {remaining}")

with open(rr_path, "r") as f:
    rr_final = f.read()
remaining_rr = rr_final.count("with open(model_path, 'r', encoding='utf-8')")
print(f"[FIX-VPS-P1-V3] {rr_path.name} — detecciones texto restantes: {remaining_rr}")

# Tambien corregir active_seeds: ya fue corregido en v2 (seed99 eliminada)
# Verificar estado actual
import yaml
with open(Path("/root/luna_v2/config/settings.yaml"), "r") as f:
    cfg = yaml.safe_load(f)
seeds = cfg.get("wfb", {}).get("active_seeds", [])
print(f"[FIX-VPS-P1-V3] active_seeds actual: {seeds}")

print("\n[FIX-VPS-P1-V3] === FIX COMPLETADO ===")
print("[FIX-VPS-P1-V3] Ejecutar: pm2 restart luna-v2-live-demo")
