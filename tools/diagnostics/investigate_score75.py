"""
Investiga en profundidad:
1. La causa raiz del pkl -> 123 en el audit (via subprocess con python correcto)
2. Por qué Guards 3/5/6 y [Consensus/RESULT] están ausentes
"""
import sys, joblib, traceback, subprocess
from pathlib import Path
sys.path.insert(0, "/root/luna_v2")

PYTHON = "/root/miniconda3/envs/luna_env/bin/python"

print("=" * 70)
print("INVESTIGACION F — El audit reporta pkl→123. ¿Por qué?")
print("=" * 70)
pkl_path = "/root/luna_v2/data/models/prod/seed99/ood_guard.pkl"

# Simular EXACTAMENTE lo que hace el loop del audit:
# for pkl in pkl_files: obj = joblib.load(pkl)
import os
pkl_files = list(Path("/root/luna_v2/data/models/prod").rglob("*.pkl"))
print(f"  Total .pkl: {len(pkl_files)}")
for pkl in pkl_files:
    try:
        obj = joblib.load(pkl)
        if isinstance(obj, (int, float, str, bytes)) and not hasattr(obj, '__module__'):
            print(f"  ❌ PRIMITIVO: {pkl.name} → {repr(obj)[:50]}")
        else:
            print(f"  ✅ {pkl.name}: {type(obj).__name__}")
    except Exception as e:
        print(f"  ❌ EXCEPCION: {pkl.name} → {repr(e)[:80]}")

# Probar subprocess con python correcto
print("\n  Subprocess con python correcto:")
result = subprocess.run(
    [PYTHON, "-c", f"import joblib; obj=joblib.load('{pkl_path}'); print(type(obj).__name__)"],
    capture_output=True, text=True, cwd="/root/luna_v2"
)
print(f"  returncode: {result.returncode}")
print(f"  stdout: {result.stdout.strip()[:200]}")
print(f"  stderr: {result.stderr.strip()[:200]}")

# 123 es el codigo de retorno de SIGPIPE o de una señal del OS. ¿Qué pasa si joblib usa loky?
print("\n  Probando joblib con backend loky:")
result2 = subprocess.run(
    [PYTHON, "-c", """
import joblib
from pathlib import Path
import os
pth = Path('/root/luna_v2/data/models/prod/seed99/ood_guard.pkl')
print(f'exists={pth.exists()} size={pth.stat().st_size}')
obj = joblib.load(pth)
print(f'type={type(obj).__name__}')
print('SUCCESS')
"""],
    capture_output=True, text=True, cwd="/root/luna_v2", timeout=30
)
print(f"  returncode: {result2.returncode}")
print(f"  stdout: {result2.stdout.strip()[:300]}")
print(f"  stderr: {result2.stderr.strip()[:300]}")

print("\n" + "=" * 70)
print("INVESTIGACION H — Guards 3/5/6 y markers ausentes")
print("=" * 70)

pm2_log = Path("/root/.pm2/logs/luna-v2-live-demo-out.log")
with open(pm2_log, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Buscar cuántos ciclos hay y los marcadores de cada uno
all_guards = [
    "[Auditor] Guard 1", "[Auditor] Guard 2", "[Auditor] Guard 3",
    "[Auditor] Guard 4", "[Auditor] Guard 5", "[Auditor] Guard 6",
    "[Consensus/RESULT]", "[FIX-XGB-TRAZABILIDAD]", "[SIZER]",
    "[SIZER-KELLY]", "[FIX-SKEW-01]", "[FIX-SKEW-02]", "[FIX-SKEW-03]",
    "[FIX-O4-MISSING-MILAGS]", "[AUDITOR]",
]

# Extraer todos los ciclos (no solo el último)
cycles = content.split("Iniciando Ciclo Operativo LUNA V2")
print(f"  Total ciclos en log: {len(cycles)-1}")

# Analizar los últimos 3 ciclos
for ci, cyc_content in enumerate(cycles[-4:-1], start=1):
    print(f"\n  --- Ciclo -{(4-ci)} ---")
    for g in all_guards:
        found = g in cyc_content
        print(f"    {'✅' if found else '❌'} {g}")

# Último ciclo completo
print(f"\n  === ULTIMO CICLO COMPLETO ===")
last = cycles[-1] if len(cycles) > 1 else ""
# Mostrar todo el texto
print(last[:6000])
