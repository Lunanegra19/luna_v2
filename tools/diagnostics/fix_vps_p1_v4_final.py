"""
fix_vps_p1_v4_final.py
[FIX-VPS-P1-V4 2026-05-30] Fix final consolidado:
1. Corrige is_bm_mock (baseline model) en regime_router.py - mismo bug binario
2. Restaura seed99 en active_seeds (sus modelos XGBoost son reales binarios)
3. Verifica que todos los paths de deteccion mock esten corregidos

CAUSA RAIZ CONFIRMADA:
  regime_router.py L111: open(baseline_path, 'r') -> XGBoost binario {L\\x00 -> startswith('{') = True
  Esto marca el baseline como Mock -> _baseline_model = MockXGBClassifier()
  Luego Security Guard ve isinstance(router._baseline_model, MockXGBClassifier) -> RuntimeError
  -> 107 reinicios del bot.

NOTE: El bloque L140-148 en regime_router.py esta BIEN (usa json.loads() que falla en binario -> except -> is_mock=False).
"""
import re, shutil, yaml
from pathlib import Path

print("[FIX-VPS-P1-V4] === INICIO Fix Final Consolidado ===")

RR_PATH = Path("/root/luna_v2/luna/models/regime_router.py")
SETTINGS = Path("/root/luna_v2/config/settings.yaml")

# 1. Fix regime_router.py baseline mock detection
bak = RR_PATH.with_suffix(".py.bak_p1v4")
shutil.copy2(RR_PATH, bak)
print("[FIX-VPS-P1-V4] Backup: " + str(bak))

with open(RR_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# El bloque problematico del baseline:
OLD_BM = (
    "                is_bm_mock = False\n"
    "                with open(_baseline_path, 'r', encoding='utf-8') as _f_bm:\n"
    "                    if _f_bm.read(100).strip().startswith('{'):\n"
    "                        is_bm_mock = True\n"
    "                if is_bm_mock:\n"
)

NEW_BM = (
    "                is_bm_mock = False\n"
    "                # [FIX-VPS-P1-V4 2026-05-30] Deteccion mock en modo BINARIO.\n"
    "                # XGBoost real: header 0x7b 0x4c ('{L'). Mock JSON: 0x7b 0x22 ('{\\"').\n"
    "                try:\n"
    "                    with open(_baseline_path, 'rb') as _f_bm:\n"
    "                        _bm_hdr = _f_bm.read(4)\n"
    "                    is_bm_mock = (_bm_hdr[:1] == b'{') and (_bm_hdr[1:2] != b'\\x4c')\n"
    "                    print('[MOCK-DETECT-FIX-BM] ' + _baseline_path.name + ': header=' + _bm_hdr.hex() + ' | is_mock=' + str(is_bm_mock))\n"
    "                except Exception:\n"
    "                    pass\n"
    "                if is_bm_mock:\n"
)

if OLD_BM in content:
    new_content = content.replace(OLD_BM, NEW_BM, 1)
    with open(RR_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("[FIX-VPS-P1-V4] regime_router.py: baseline mock detection corregido.")
else:
    print("[FIX-VPS-P1-V4] WARN: No se encontro bloque baseline en regime_router.py.")
    # Verificar si ya fue corregido en un fix anterior
    if "FIX-VPS-P1-V4" in content:
        print("[FIX-VPS-P1-V4] Ya corregido previamente.")
    else:
        # Buscar variaciones
        lines_found = [l for l in content.split('\n') if 'is_bm_mock' in l]
        print("[FIX-VPS-P1-V4] Lineas con is_bm_mock: " + str(lines_found[:5]))

# 2. Verificar ensemble_live_inference.py (corregido en P1-V3)
ELF = Path("/root/luna_v2/luna/live/ensemble_live_inference.py")
with open(ELF, "r") as f:
    elf_content = f.read()
txt_remaining = elf_content.count("open(model_path, 'r', encoding='utf-8') as _fm")
print("[FIX-VPS-P1-V4] ensemble_live_inference.py — detecciones texto restantes: " + str(txt_remaining))

# 3. Restaurar seed99 en active_seeds (sus modelos son reales XGBoost binarios)
with open(SETTINGS, "r", encoding="utf-8") as f:
    cfg_txt = f.read()

cfg_yaml = yaml.safe_load(cfg_txt)
current = cfg_yaml.get("wfb", {}).get("active_seeds", [])
print("[FIX-VPS-P1-V4] active_seeds actual: " + str(current))

# Verificar seed99 tiene modelos reales
PROD = Path("/root/luna_v2/data/models/prod")
seed99_real = True
for regime in ["bear", "bull", "range"]:
    mp = PROD / "seed99" / ("xgboost_meta_" + regime + "_long.model")
    if mp.exists():
        with open(mp, "rb") as f:
            hdr = f.read(2)
        is_real = (hdr[:1] == b"{") and (hdr[1:2] == b"\x4c")
        print("  seed99/" + mp.name + ": real=" + str(is_real) + " (" + hdr.hex() + ")")
        if not is_real:
            seed99_real = False
    else:
        print("  seed99/" + mp.name + ": FALTA")
        seed99_real = False

if seed99_real:
    # Restaurar seed99 en active_seeds
    new_seeds = [99, 1337, 2025]
    print("[FIX-VPS-P1-V4] seed99 modelos verificados como reales. Restaurando en active_seeds: " + str(new_seeds))
    
    new_block = (
        "  active_seeds:\n"
        "  # [FIX-VPS-P1-V4 2026-05-30] seed99, seed1337, seed2025 - binarios XGBoost verificados\n"
        "  - 99\n"
        "  - 1337\n"
        "  - 2025\n"
    )
    pattern = r"  active_seeds:\n(?:(?:  #[^\n]*)?\n|  - \d+\n)+"
    match = re.search(pattern, cfg_txt)
    if match:
        cfg_txt = cfg_txt[:match.start()] + new_block + cfg_txt[match.end():]
        with open(SETTINGS, "w", encoding="utf-8") as f:
            f.write(cfg_txt)
        print("[FIX-VPS-P1-V4] active_seeds restaurado con seed99.")
    else:
        print("[FIX-VPS-P1-V4] WARN: regex no encontro bloque active_seeds.")
else:
    print("[FIX-VPS-P1-V4] WARN: seed99 NO tiene todos los modelos reales. Mantener sin seed99.")

# 4. Verificacion final
with open(SETTINGS, "r") as f:
    final_cfg = yaml.safe_load(f)
final_seeds = final_cfg.get("wfb", {}).get("active_seeds", [])
print("\n[FIX-VPS-P1-V4] active_seeds FINAL: " + str(final_seeds))

print("\n[FIX-VPS-P1-V4] === FIX COMPLETADO ===")
print("[FIX-VPS-P1-V4] ACCION: pm2 restart luna-v2-live-demo")
