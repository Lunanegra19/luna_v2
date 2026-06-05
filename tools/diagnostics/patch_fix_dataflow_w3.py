"""patch_fix_dataflow_w3.py — FIX-DATAFLOW-INT-01: corrige warning falso en DATAFLOW-EXPORT-FP-01"""
import sys, ast
sys.path.insert(0, '.')
from pathlib import Path

src = Path("luna/features/feature_pipeline.py")
raw = src.read_bytes()

# Identificar el bloque a reemplazar buscando bytes literales
TARGET = b'                    if str(_dtype) in ("float64", "float32", "int64", "int32"):\r\n'
idx = raw.find(TARGET)
if idx < 0:
    print("ERROR: bloque target no encontrado (busqueda exacta)")
    # Intentar busqueda mas laxa
    idx2 = raw.find(b'"int64", "int32"')
    if idx2 >= 0:
        line_start = raw.rfind(b'\n', 0, idx2) + 1
        line_end   = raw.find(b'\n', idx2) + 1
        print(f"Encontrado en offset {idx2}: {raw[line_start:line_end]!r}")
    sys.exit(1)

# Encontrar el bloque completo (hasta cerrar el bloque del if, 3 lineas)
block_end_marker = b'                        )\r\n'
block_end = raw.find(block_end_marker, idx)
if block_end < 0:
    print("ERROR: fin del bloque no encontrado")
    sys.exit(1)
block_end += len(block_end_marker)

OLD_BLOCK = raw[idx:block_end]
print(f"Bloque a reemplazar ({len(OLD_BLOCK)} bytes):")
print(repr(OLD_BLOCK[:200]))
print("---")

NEW_BLOCK  = b'                    # [FIX-DATAFLOW-INT-01 2026-06-02] Warning solo si dtype es FLOAT.\r\n'
NEW_BLOCK += b'                    # ANTES: alertaba si dtype in (float64, float32, int64, int32)\r\n'
NEW_BLOCK += b'                    # BUG: int64 ES el dtype correcto post-FIX-OBS-03-HMM-DTYPE.\r\n'
NEW_BLOCK += b'                    # El BUG-HMM-FILTER-01 ocurre solo con float64 (float 1.0 != int 1).\r\n'
NEW_BLOCK += b'                    # AHORA: warning solo si float, print-OK si ya es int (correcto).\r\n'
NEW_BLOCK += b'                    if str(_dtype) in ("float64", "float32"):\r\n'
NEW_BLOCK += b'                        logger.warning(\r\n'
NEW_BLOCK += b'                            f"  [DATAFLOW-EXPORT-FP-01] ALERTA: {_hc} en holdout es FLOAT ({_dtype}). "\r\n'
NEW_BLOCK += b'                            f"Causara BUG-HMM-FILTER-01 (filtro regime ==1 fallara). "\r\n'
NEW_BLOCK += b'                            f"Verificar que FIX-OBS-03-HMM-DTYPE esta activo."\r\n'
NEW_BLOCK += b'                        )\r\n'
NEW_BLOCK += b'                        print(  # RULE[fixbugsprints.md]\r\n'
NEW_BLOCK += b'                            f"[FIX-DATAFLOW-INT-01] ALERTA: {_hc} dtype={_dtype} (float) -> BUG-HMM-FILTER-01 riesgo"\r\n'
NEW_BLOCK += b'                        )\r\n'
NEW_BLOCK += b'                    elif str(_dtype) in ("int64", "int32", "int16"):\r\n'
NEW_BLOCK += b'                        print(  # RULE[fixbugsprints.md]\r\n'
NEW_BLOCK += b'                            f"[FIX-DATAFLOW-INT-01] {_hc} dtype={_dtype} (int) -> OK, FIX-OBS-03 activo"\r\n'
NEW_BLOCK += b'                        )\r\n'

new_raw = raw[:idx] + NEW_BLOCK + raw[block_end:]
src.write_bytes(new_raw)
print(f"\n[FIX-DATAFLOW-INT-01] Aplicado ({len(raw)} -> {len(new_raw)} bytes)")

try:
    ast.parse(new_raw.decode('utf-8', 'replace'))
    print("[OK] Syntax valida")
except SyntaxError as e:
    print(f"[ERROR] L{e.lineno}: {e.msg} -- ROLLBACK")
    src.write_bytes(raw)
    print("[ROLLBACK] original restaurado")
    sys.exit(1)
