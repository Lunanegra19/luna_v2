"""
fix_regime_router_baseline.py
[FIX-VPS-P1-V4b 2026-05-30] Corrige baseline mock detection en regime_router.py
Cambia apertura modo texto 'r' a binario 'rb' para detectar correctamente XGBoost vs JSON mock.
"""
import shutil
from pathlib import Path

RR = Path("/root/luna_v2/luna/models/regime_router.py")
print("[FIX-VPS-P1-V4b] Leyendo " + str(RR))

bak = Path("/root/luna_v2/luna/models/regime_router.py.bak_p1v4b")
shutil.copy2(RR, bak)
print("[FIX-VPS-P1-V4b] Backup: " + str(bak))

with open(RR, "r", encoding="utf-8") as f:
    lines = f.readlines()

print("[FIX-VPS-P1-V4b] Total lineas: " + str(len(lines)))

# Encontrar y reemplazar el bloque problematico (lineas 109-112 aprox)
new_lines = []
i = 0
fixed = 0
while i < len(lines):
    line = lines[i]
    # Detectar inicio del bloque problematico
    if "is_bm_mock = False" in line and i+1 < len(lines) and "with open(_baseline_path, 'r'" in lines[i+1]:
        indent = line[:len(line) - len(line.lstrip())]
        print("[FIX-VPS-P1-V4b] Bloque encontrado en linea " + str(i+1))
        # Reemplazar las 4 lineas del bloque
        new_lines.append(indent + "is_bm_mock = False\n")
        new_lines.append(indent + "# [FIX-VPS-P1-V4b 2026-05-30] Deteccion binaria: XGBoost real={7b4c}, Mock JSON={7b22}\n")
        new_lines.append(indent + "try:\n")
        new_lines.append(indent + "    with open(_baseline_path, 'rb') as _f_bm:\n")
        new_lines.append(indent + "        _bm_hdr = _f_bm.read(4)\n")
        new_lines.append(indent + "    is_bm_mock = (_bm_hdr[:1] == b'{') and (_bm_hdr[1:2] != b'\\x4c')\n")
        new_lines.append(indent + "    print('[MOCK-BM-FIX] ' + str(_baseline_path.name) + ' header=' + _bm_hdr.hex() + ' mock=' + str(is_bm_mock))\n")
        new_lines.append(indent + "except Exception:\n")
        new_lines.append(indent + "    pass\n")
        fixed += 1
        # Saltar las 4 lineas originales del bloque
        i += 4  # is_bm_mock=False, with open, if startswith, is_bm_mock=True
    else:
        new_lines.append(line)
        i += 1

print("[FIX-VPS-P1-V4b] Bloques corregidos: " + str(fixed))

with open(RR, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

# Verificar
with open(RR, "r") as f:
    final = f.read()
remaining = final.count("with open(_baseline_path, 'r'")
print("[FIX-VPS-P1-V4b] Detecciones texto restantes: " + str(remaining))
print("[FIX-VPS-P1-V4b] OK")
