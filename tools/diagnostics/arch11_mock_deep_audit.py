"""
arch11_mock_deep_audit.py
============================
Auditoria profunda del sistema MockXGBClassifier:
1. Cuantos archivos .model en produccion son JSON mocks?
2. El baseline model es mock o real?
3. El ensemble_live_inference.py tiene el mismo problema?
4. Que pasa en runtime cuando el agente no existe en disco?
5. Cual es el flujo de codigo exacto para cada caso de error

USO: python tools/diagnostics/arch11_mock_deep_audit.py
"""
import sys, json, struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("[ARCH-11] MOCK XGB DEEP AUDIT")
print("="*70)

# ── 1. Buscar todos los .model en el proyecto ─────────────────────────────────
print("\n[1] ESCANEO DE ARCHIVOS .model EN EL PROYECTO")
print("-"*60)
model_files = []
for d in [ROOT / "data", ROOT / "luna"]:
    if d.exists():
        model_files.extend(d.rglob("*.model"))

REAL_MAGIC_BYTES = b'\x42\x58\x47'  # XGBoost native format
JSON_MAGIC = b'{'

real_count = 0
mock_count = 0
empty_count = 0

for mf in model_files:
    try:
        size = mf.stat().st_size
        with open(mf, 'rb') as f:
            header = f.read(8)
        
        if size == 0:
            status = "EMPTY"
            empty_count += 1
        elif header[:1] == JSON_MAGIC:
            # Es JSON - verificar si tiene mocked:true
            try:
                content = mf.read_text(encoding='utf-8', errors='replace')[:500]
                data = json.loads(content.strip())
                is_mocked = data.get("mocked", False)
                status = f"JSON-MOCK mocked={is_mocked}"
                mock_count += 1
            except:
                status = "JSON-UNKNOWN"
                mock_count += 1
        else:
            status = "XGB-REAL"
            real_count += 1
        
        rel = mf.relative_to(ROOT)
        print(f"  {status:25} | {size:>8,}B | {rel}")
    except Exception as e:
        print(f"  ERROR: {mf.name}: {e}")

print(f"\n  RESUMEN: {real_count} reales | {mock_count} mocks/JSON | {empty_count} vacíos")

# ── 2. Analizar el flujo de _load_models cuando modelo NO existe ──────────────
print("\n[2] FLUJO CUANDO EL ARCHIVO .model NO EXISTE EN DISCO")
print("-"*60)
print("  L188: if model_path.exists() and sig_path.exists():")
print("  L264:   logger.warning(Faltan artefactos para agente...)")
print("  L312: if agent_name not in self.models:")  
print("  L314:   if self._baseline_model is not None and self._baseline_features:")
print("  L344:     -> BASELINE FALLBACK (usa modelo global, no Mock)")
print("  L357:   else:")
print("  L358:     -> prob=0.0 (barras bloqueadas)")
print()
print("  CONCLUSION: cuando el archivo NO existe, NO se usa Mock.")
print("  El Mock solo se activa cuando el archivo EXISTE y su header es JSON.")

# ── 3. Verificar ensemble_live_inference.py ────────────────────────────────────
print("\n[3] ensemble_live_inference.py — uso de MockXGBClassifier")
print("-"*60)
live_path = ROOT / "luna" / "models" / "ensemble_live_inference.py"
if live_path.exists():
    content = live_path.read_text(encoding='utf-8', errors='replace')
    lines = content.splitlines()
    mock_hits = [(i+1, l) for i, l in enumerate(lines) if 'mock' in l.lower() or 'Mock' in l]
    print(f"  Hits de Mock en ensemble_live_inference.py: {len(mock_hits)}")
    for lno, line in mock_hits:
        print(f"  L{lno:4}: {line.strip()[:120]}")
    
    # Buscar donde se carga el modelo
    load_hits = [(i+1, l) for i, l in enumerate(lines) if 'load_model' in l or 'joblib.load' in l or 'XGBClassifier' in l]
    print(f"\n  Carga de modelos en live inference:")
    for lno, line in load_hits[:5]:
        print(f"  L{lno:4}: {line.strip()[:120]}")

# ── 4. Analizar si el Mock produce trades con WR=100% ─────────────────────────
print("\n[4] ANALISIS: ¿PUEDE EL MOCK PRODUCIR WR=100% CON N PEQUEÑO?")
print("-"*60)
print("  MockXGBClassifier.predict_proba siempre devuelve:")
print("  res[:, 0] = 0.4  (prob clase 0)")
print("  res[:, 1] = 0.6  (prob clase 1)")
print()
print("  Con CUTOFF = 0.62 (agente RANGE actual):")
print("  0.60 < 0.62 → 0 trades generados (silencioso, no error)")
print()
print("  Con CUTOFF = 0.58 (si se baja):")
print("  0.60 > 0.58 → 100% de barras generan trades con prob=0.6")
print()
print("  MECANISMO DEL WR=100%:")
print("  - Mock activo + threshold bajo → señal en TODAS las barras RANGE")
print("  - MetaLabeler filtra algunas")
print("  - Las que sobreviven son las que tienen vertical_barrier corta")
print("  - Barras con cierre en t+N horas donde precio subió → WR=100% con N pequeño")
print()
print("  Para confirmar si los 22 trades fueron mock, necesito ver xgb_prob_raw en los logs.")

# ── 5. Propuesta de fix ────────────────────────────────────────────────────────
print("\n[5] PROPUESTA DE FIX PROFESIONAL")
print("-"*60)
print("""
SITUACION ACTUAL:
  - MockXGBClassifier existe en regime_router.py L45-54
  - Se usa cuando el archivo .model existe pero es JSON {"mocked": true}
  - Esto es un mecanismo INTENCIONAL de placeholder para seeds sin entrenar

PROBLEMA:
  1. El mock devuelve prob=0.6 SIEMPRE → si threshold < 0.6, genera señales falsas
  2. No hay RuntimeError → fallo silencioso
  3. El mock tiene set_params() vacío → no rompe el pipeline
  
FIX CORRECTO (dos niveles):
  
  Nivel 1 — Guard en predict_proba:
    Si se llama a MockXGBClassifier.predict_proba:
    → logger.CRITICAL + print visible
    → Lanzar RuntimeError con mensaje claro
    → NUNCA retornar silenciosamente prob=0.6
    
  Nivel 2 — Guard en _load_models:
    Si is_mock == True:
    → No cargar el Mock
    → Usar baseline como fallback (ya existe en L314)
    → O dejar el agente ausente → prob=0.0 (bloqueado)
    
  RAZON: prob=0.0 (bloqueado) es SIEMPRE MEJOR que prob=0.6 (mock aleatorio)
  porque bloquea las señales en lugar de generar señales falsas con WR impredecible.
""")

print("[ARCH-11] Audit completado.")
