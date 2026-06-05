"""
Investiga exactamente qué pkl genera KeyError(123) y por qué.
123 = dimensión del bottleneck del AutoEncoder (492->246->123->32)
"""
import sys, joblib, traceback
from pathlib import Path
sys.path.insert(0, "/root/luna_v2")

ROOT = Path("/root/luna_v2")
pkl_files = list((ROOT / "data" / "models" / "prod").rglob("*.pkl"))
print(f"Total .pkl: {len(pkl_files)}")

for pkl in sorted(pkl_files):
    print(f"\n{'='*60}")
    print(f"  {pkl.relative_to(ROOT)}")
    print(f"  Tamaño: {pkl.stat().st_size:,} bytes")
    try:
        obj = joblib.load(pkl)
        t = type(obj).__name__
        m = getattr(type(obj), '__module__', 'unknown')
        print(f"  ✅ OK: {m}.{t}")
        if isinstance(obj, dict):
            print(f"     keys (muestra): {list(obj.keys())[:8]}")
        # Verificar si tiene clave int 123 que pueda confundir el audit
        if isinstance(obj, dict) and 123 in obj:
            print(f"  ⚠️  CONTIENE CLAVE int 123: {type(obj[123])}")
    except KeyError as e:
        print(f"  ❌ KeyError: {e!r}")
        traceback.print_exc()
        # ¿Es el dict del HMM model que tiene clave int 123?
        try:
            # Intentar carga con pickle directamente
            import pickle
            with open(pkl, 'rb') as f:
                raw = pickle.load(f)
            print(f"  pickle.load: {type(raw).__name__}")
            if isinstance(raw, dict):
                print(f"  keys: {list(raw.keys())[:10]}")
        except Exception as e2:
            print(f"  pickle.load también falla: {e2}")
    except Exception as e:
        print(f"  ❌ ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()

print("\n\nVERIFICACION: ¿El AE tiene dim 123 (392->246->123->32)?")
# Buscar en el código del AE
import subprocess
r = subprocess.run(
    ["grep", "-rn", "123", "/root/luna_v2/luna/live/", "--include=*.py"],
    capture_output=True, text=True
)
for line in r.stdout.splitlines()[:20]:
    if "autoencoder" in line.lower() or "ae" in line.lower() or "arch" in line.lower() or "123" in line:
        print(f"  {line}")
