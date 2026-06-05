"""patch_arch2123.py - ARCH-21/23-FIX-A: macro stable boost en SFI combined score"""
import sys
sys.path.insert(0, '.')
from pathlib import Path

src_path = Path("luna/features/feature_selection_e.py")
raw = src_path.read_bytes()

# Marcador unico para el punto de insercion
TARGET_MARKER = b"combined = {c: max(icir_norm[c], dsr_norm[c]) for c in members}"
idx = raw.find(TARGET_MARKER)
if idx < 0:
    print("ERROR: marcador combined score no encontrado")
    sys.exit(1)

line_start = raw.rfind(b"\n", 0, idx) + 1
line_end   = raw.find(b"\n", idx) + 1

OLD_LINE = raw[line_start:line_end]
print(f"Linea a reemplazar ({len(OLD_LINE)} bytes):")
print(repr(OLD_LINE))
print("---")

# Nuevo bloque con boost macro-estables
NEW_BLOCK  = b"            combined = {c: max(icir_norm[c], dsr_norm[c]) for c in members}\r\n"
NEW_BLOCK += b"            # [ARCH-21/23-FIX-A 2026-06-02] Macro-stable boost en SFI cluster ranking.\r\n"
NEW_BLOCK += b"            # Features macro de largo plazo (M2, Fed_Liq, CPI, yield_curve, MOVE, Puell)\r\n"
NEW_BLOCK += b"            # pueden tener ICIR ligeramente menor en ventanas especificas pero son\r\n"
NEW_BLOCK += b"            # MAS estables cross-window que features tecnicas. El boost garantiza\r\n"
NEW_BLOCK += b"            # que el SFI las favorezca en el ranking frente a features inestables.\r\n"
NEW_BLOCK += b"            # El boost es ADITIVO (no multiplicativo) para preservar el ordinal relativo.\r\n"
NEW_BLOCK += b"            try:\r\n"
NEW_BLOCK += b"                from config.settings import cfg as _cfg_macro\r\n"
NEW_BLOCK += b"                _macro_boost = float(getattr(_cfg_macro.features, 'sfi_macro_stable_boost', 0.15))\r\n"
NEW_BLOCK += b"                _macro_feats = set(getattr(_cfg_macro.features, 'sfi_macro_stable_features', []) or [])\r\n"
NEW_BLOCK += b"            except Exception:\r\n"
NEW_BLOCK += b"                _macro_boost = 0.15\r\n"
NEW_BLOCK += b"                _macro_feats = set()\r\n"
NEW_BLOCK += b"            if _macro_feats:\r\n"
NEW_BLOCK += b"                _boosted = []\r\n"
NEW_BLOCK += b"                for feat_c, score_c in combined.items():\r\n"
NEW_BLOCK += b"                    # Aplicar boost solo si la feature esta en la whitelist macro\r\n"
NEW_BLOCK += b"                    _base_name = feat_c.split('_milag')[0] if '_milag' in feat_c else feat_c\r\n"
NEW_BLOCK += b"                    _in_macro  = feat_c in _macro_feats or _base_name in _macro_feats\r\n"
NEW_BLOCK += b"                    combined[feat_c] = min(1.0, score_c + (_macro_boost if _in_macro else 0.0))\r\n"
NEW_BLOCK += b"                    if _in_macro and combined[feat_c] > score_c:\r\n"
NEW_BLOCK += b"                        _boosted.append(feat_c)\r\n"
NEW_BLOCK += b"                if _boosted:\r\n"
NEW_BLOCK += b"                    print(  # RULE[fixbugsprints.md]\r\n"
NEW_BLOCK += b"                        f'[ARCH-21/23-FIX-A] SFI macro-stable boost +{_macro_boost:.2f} aplicado a '\r\n"
NEW_BLOCK += b"                        f'{len(_boosted)} features: {_boosted}'\r\n"
NEW_BLOCK += b"                    )\r\n"

new_raw = raw[:line_start] + NEW_BLOCK + raw[line_end:]
src_path.write_bytes(new_raw)
print(f"\n[ARCH-21/23-FIX-A] Aplicado ({len(raw)} -> {len(new_raw)} bytes)")

import ast
try:
    ast.parse(new_raw.decode("utf-8", "replace"))
    print("[OK] Syntax valida")
except SyntaxError as e:
    print(f"[ERROR] L{e.lineno}: {e.msg} -- ROLLBACK")
    src_path.write_bytes(raw)
    print("[ROLLBACK] original restaurado")
