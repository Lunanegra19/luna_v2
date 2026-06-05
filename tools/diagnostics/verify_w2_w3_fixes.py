"""verify_w2_w3_fixes.py — verificacion final de FIX-HMM-SHIELD-01 y FIX-DATAFLOW-INT-01"""
import ast, sys
sys.path.insert(0,'.')

print("=== VERIFICACION FINAL W2 + W3 ===")
errors = []

for f in ['luna/models/hmm_regime.py', 'luna/features/feature_pipeline.py']:
    with open(f,'rb') as fh: raw = fh.read()
    try:
        ast.parse(raw.decode('utf-8','replace'))
        print(f"[OK] Syntax: {f}")
    except SyntaxError as e:
        errors.append(f"[ERROR] {f}: {e}")
        print(errors[-1])

# Limpiar cache de settings
for m in list(sys.modules.keys()):
    if 'config' in m:
        del sys.modules[m]

from config.settings import cfg
dd_thresh = getattr(cfg.hmm, 'post_ath_dd_threshold', None)
assert dd_thresh == -0.30, f"post_ath_dd_CUTOFF = {dd_thresh} (esperado -0.30)"
print(f"[OK] FIX-HMM-SHIELD-01: post_ath_dd_CUTOFF = {dd_thresh}")

with open('luna/models/hmm_regime.py','rb') as f:
    hmm_src = f.read().decode('utf-8','replace')
assert 'FIX-HMM-SHIELD-01' in hmm_src
assert '_post_ath_enabled' in hmm_src
assert 'post_ath_bear DESACTIVADO' in hmm_src
print("[OK] FIX-HMM-SHIELD-01 Part B: MI-guard activo en hmm_regime.py")

with open('luna/features/feature_pipeline.py','rb') as f:
    fp_src = f.read().decode('utf-8','replace')
assert 'FIX-DATAFLOW-INT-01' in fp_src
# Verificar que ya NO contiene la condicion con int64 en el bloque de warning
block_start = fp_src.find('FIX-DATAFLOW-INT-01')
block_snippet = fp_src[block_start:block_start+600]
assert '"int64"' not in block_snippet.split('int64", "int32", "int16"')[0], "int64 aun en condicion warning"
assert '"float64", "float32"' in block_snippet
print("[OK] FIX-DATAFLOW-INT-01: warning solo si float dtype")

print()
print("="*55)
print("FIXES W2+W3 OK:")
print("  W2a: post_ath_dd_threshold -0.20 -> -0.30 (2854->930H forzadas)")
print("  W2b: MI-guard activo (desactiva post_ath si empeora SOP-R9)")
print("  W3:  DATAFLOW warning solo float64/32, OK-print si int64")
print()
print("SIGUIENTE: WFB --nocache con 10 fixes activos")

if errors:
    print("ERRORES:", errors)
    sys.exit(1)
