"""patch_fix_obs03b.py — FIX-OBS-03-B: actualizar datasets[name] tras conversion dtype HMM_Regime"""
import sys, ast
sys.path.insert(0, '.')
from pathlib import Path

src = Path("luna/features/feature_pipeline.py")
txt = src.read_text(encoding='utf-8', errors='replace')

# Buscar el final del bloque de conversion FIX-OBS-03-HMM-DTYPE
# La linea a encontrar es la que cierra el bloque de la conversion (logger.info)
TARGET = (
    "                    logger.info(\n"
    "                        f\"[FIX-OBS-03-HMM-DTYPE] Split '{name}': {_hmm_dtype_col} \"\n"
    "                        f\"dtype float64->int -- evita BUG-HMM-FILTER-01\"\n"
    "                    )\n"
)

idx = txt.find(TARGET)
if idx < 0:
    # Intentar con \r\n
    TARGET2 = TARGET.replace('\n', '\r\n')
    idx = txt.find(TARGET2)
    if idx >= 0:
        TARGET = TARGET2
        print("Usando CRLF")

if idx < 0:
    # Buscar mas flexible
    idx = txt.find("dtype float64->int -- evita BUG-HMM-FILTER-01")
    if idx < 0:
        idx = txt.find("evita BUG-HMM-FILTER-01")
    if idx >= 0:
        # Encontrar el fin de este bloque (cierre del logger.info)
        end_of_logger = txt.find("                    )\n", idx)
        if end_of_logger < 0:
            end_of_logger = txt.find("                    )\r\n", idx)
        if end_of_logger >= 0:
            TARGET = txt[idx:end_of_logger + len("                    )\n")]
            print(f"Marcador flexible: {repr(TARGET[-80:])}")
    if idx < 0:
        print("ERROR: bloque FIX-OBS-03-HMM-DTYPE no encontrado")
        sys.exit(1)

idx_target = txt.find(TARGET)
print(f"Bloque encontrado en offset {idx_target}")
print(f"Tail: {repr(TARGET[-100:])}")

ADDITION = (
    "                    # [FIX-OBS-03-B 2026-06-02] Actualizar datasets dict tras la conversion.\n"
    "                    # BUG: data = data.copy() crea referencia local; datasets.get('holdout')\n"
    "                    # en DATAFLOW-EXPORT-FP-01 leia el objeto original (float64) -> falsa ALERTA.\n"
    "                    datasets[name] = data\n"
    "                    print(  # RULE[fixbugsprints.md]\n"
    "                        f\"[FIX-OBS-03-B] datasets['{name}'] actualizado dtype int64\"\n"
    "                        f\" -> DATAFLOW check leera objeto correcto (no falsa ALERTA)\"\n"
    "                    )\n"
)

new_txt = txt[:idx_target + len(TARGET)] + ADDITION + txt[idx_target + len(TARGET):]

src.write_text(new_txt, encoding='utf-8')
print(f"\n[FIX-OBS-03-B] Aplicado ({len(txt)} -> {len(new_txt)} chars)")

try:
    ast.parse(new_txt)
    print("[OK] Syntax valida")
except SyntaxError as e:
    print(f"[ERROR] L{e.lineno}: {e.msg} -- ROLLBACK")
    src.write_text(txt, encoding='utf-8')
    sys.exit(1)

assert 'FIX-OBS-03-B' in new_txt
assert "datasets[name] = data" in new_txt
print("[OK] FIX-OBS-03-B y datasets[name] = data presentes")
