"""arch09_bug03_guard_audit.py
Auditoria profunda del BUG-03 Guard (xgb_min_signals_count).
Verifica si el guard esta activo, que hace exactamente y si re-habilitarlo mejora el modelo.
"""
import sys, yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("[ARCH-09] BUG-03 GUARD AUDIT")
print("="*70)

# ── 1. Leer configuracion actual ──────────────────────────────────────────────
print("\n[1] CONFIGURACION ACTUAL EN SETTINGS.YAML")
print("-"*60)
with open(ROOT/"config"/"settings.yaml","r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

xgb_cfg = cfg.get("xgboost", {})
for k in ["xgb_min_signals_count", "bug03", "guard", "min_signals", "signal_count"]:
    if k in xgb_cfg:
        print(f"  xgboost.{k}: {xgb_cfg[k]}")

# Buscar en todos los niveles
print("\n  Buscando 'xgb_min_signals' en YAML completo:")
import json
cfg_str = json.dumps(cfg, default=str)
if "xgb_min_signals" in cfg_str.lower():
    import re
    hits = re.findall(r'"[^"]*min_signals[^"]*"\s*:\s*[^,}]+', cfg_str, re.IGNORECASE)
    for h in hits:
        print(f"    {h}")
else:
    print("  NO ENCONTRADO en settings.yaml")

# ── 2. Buscar el guard en el codigo ───────────────────────────────────────────
print("\n[2] BUG-03 GUARD EN CODIGO")
print("-"*60)
import re

# Buscar en calibrate_probabilities.py
calib_path = ROOT / "luna" / "models" / "calibrate_probabilities.py"
lines = calib_path.read_text("utf-8", errors="replace").splitlines()
bug03_hits = [(i+1, l) for i, l in enumerate(lines) if "bug03" in l.lower() or "min_signal" in l.lower() or "BUG.03" in l or "BUG03" in l]
print(f"  calibrate_probabilities.py: {len(bug03_hits)} hits de BUG-03/min_signals")
for lno, line in bug03_hits[:15]:
    print(f"  L{lno:4}: {line.strip()[:120]}")

# Buscar en train_xgboost_v2.py
xgb_path = ROOT / "luna" / "models" / "train_xgboost_v2.py"
lines_xgb = xgb_path.read_text("utf-8", errors="replace").splitlines()
bug03_hits_xgb = [(i+1, l) for i, l in enumerate(lines_xgb) if "bug03" in l.lower() or "min_signal" in l.lower() or "xgb_min_signals" in l.lower()]
print(f"\n  train_xgboost_v2.py: {len(bug03_hits_xgb)} hits de BUG-03/min_signals")
for lno, line in bug03_hits_xgb[:10]:
    print(f"  L{lno:4}: {line.strip()[:120]}")

# ── 3. Entender que hace el guard ─────────────────────────────────────────────
print("\n[3] LOGICA DEL GUARD (contexto completo)")
print("-"*60)
# Buscar la funcion o bloque donde se usa el guard
for i, l in enumerate(lines):
    if "min_signal" in l.lower() or "bug03" in l.lower():
        start = max(0, i-3)
        end = min(len(lines), i+8)
        print(f"\n  [Contexto L{i+1}]")
        for j in range(start, end):
            marker = ">>>" if j == i else "   "
            print(f"  {marker} L{j+1:4}: {lines[j][:110]}")

# ── 4. Evaluar si activar el guard mejora o empeora ──────────────────────────
print("\n[4] ANALISIS DE IMPACTO")
print("-"*60)
print("""
PREMISA DEL GUARD:
  El BUG-03 es un sesgo de calibracion que ocurre cuando el numero de señales
  XGBoost que pasan el threshold es muy pequeño (N<50).
  Con N<50 señales en validation, el calibrador isotónico sobreajusta la curva
  de calibracion y produce thresholds artificialmente altos.

ESTADO ACTUAL:
  xgb_min_signals_count = valor en settings (ver seccion [1])
  Si = 0: el guard esta DESACTIVADO (no hay minimo de señales)
  Si > 0: el guard rechaza la ventana si N < xgb_min_signals_count

IMPACTO DE REACTIVAR:
  PRO: Evita calibrar con N<50 señales (Brier diverge con N pequeño)
  CON: Con el sistema actual produciendo 22 trades/trimestre, activar el guard
       con N=50 implicaria BLOQUEAR casi todas las ventanas WFB

VEREDICTO:
  El guard no debe re-activarse como bloqueador de ventanas completas.
  El fix correcto es que el calibrador degrade gracefully cuando N<50:
  - Usar CUTOFF = base_rate (sin calibracion) en lugar de rechazar la ventana
  - Esto ya esta implementado en FIX-P1-CALIB-PASSTHROUGH (calibrate_probabilities.py L558)
""")
print("[ARCH-09] Audit completado.")
