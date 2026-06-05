"""Fase 4 — Causa raíz H-A: Bear_long colapso. Localizar modelos y entender el mecanismo."""
import sys, json, re
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
from pathlib import Path
import numpy as np

SEP = '─'*72

# ─── 1. Localizar archivos de modelos bear ────────────────────────────────
print(SEP)
print('1. LOCALIZACIÓN DE MODELOS BEAR_LONG')
print(SEP)

search_dirs = [
    Path('g:/Mi unidad/ia/luna_v2/data/models'),
    Path('g:/Mi unidad/ia/luna_v2/data/runs'),
    Path('g:/Mi unidad/ia/luna_v2/data/wfb_models'),
]
for d in search_dirs:
    if not d.exists():
        print(f'  {d}: NO EXISTE')
        continue
    bear_files = sorted(d.rglob('*bear*'))
    print(f'\n{d.name}/: {len(bear_files)} archivos bear')
    for f in bear_files[:10]:
        try:
            size = f.stat().st_size
            print(f'  {str(f.relative_to(d)):60s} {size:>10,} bytes')
        except:
            print(f'  {f}')

# ─── 2. Leer un modelo bear_long real para inspeccionar ──────────────────
print()
print(SEP)
print('2. INSPECCIÓN INTERNA DEL MODELO BEAR_LONG')
print(SEP)

# Buscar en las runs de hoy
runs_dir = Path('g:/Mi unidad/ia/luna_v2/data/runs')
bear_models_found = []
if runs_dir.exists():
    for f in sorted(runs_dir.rglob('*.joblib')):
        if 'bear' in f.name.lower():
            bear_models_found.append(f)
    print(f'Archivos .joblib con bear: {len(bear_models_found)}')
    for f in bear_models_found[:5]:
        print(f'  {f}  ({f.stat().st_size:,} bytes)')

# Intentar cargar uno
if bear_models_found:
    import joblib
    try:
        model = joblib.load(bear_models_found[0])
        print(f'\nTipo de modelo: {type(model)}')
        if hasattr(model, 'n_estimators'):
            print(f'  n_estimators: {model.n_estimators}')
        if hasattr(model, 'n_features_in_'):
            print(f'  n_features_in: {model.n_features_in_}')
        # Intentar predecir con datos sintéticos para ver si colapsa
        import numpy as np
        n_feat = getattr(model, 'n_features_in_', 50)
        X_test = np.random.normal(0, 1, (100, n_feat))
        try:
            probs = model.predict_proba(X_test)[:, 1]
            print(f'\nPredict_proba sobre X sintético (N=100):')
            print(f'  std={probs.std():.6f} | mean={probs.mean():.4f} | '
                  f'min={probs.min():.4f} | max={probs.max():.4f}')
            if probs.std() < 1e-6:
                print(f'  *** COLAPSO CONFIRMADO: std=0 en datos sintéticos ***')
            else:
                print(f'  Modelo responde con variación normal en datos sintéticos')
        except Exception as e:
            print(f'  Error en predict_proba: {e}')
    except Exception as e:
        print(f'Error cargando modelo: {e}')
else:
    print('No se encontraron archivos .joblib bear en data/runs/')
    # Buscar en otros lugares
    for p in [Path('g:/Mi unidad/ia/luna_v2/data'),
              Path('g:/Mi unidad/ia/luna_v2/luna')]:
        if p.exists():
            jl = list(p.rglob('*bear*.joblib'))
            if jl:
                print(f'  En {p.name}: {[str(f.name) for f in jl[:5]]}')

# ─── 3. Leer el código del RegimeRouter para entender la predicción bear ─
print()
print(SEP)
print('3. ANÁLISIS DEL LOG — ¿EN QUÉ PASO OCURRE EL COLAPSO?')
print(SEP)

log_path = Path('C:/Users/Usuario/.gemini/antigravity-ide/brain/ad23283d-d02e-4616-9748-5d609f02bf06/.system_generated/tasks/task-1314.log')
if log_path.exists():
    log = log_path.read_text(encoding='utf-8', errors='replace')
    # Extraer contexto alrededor de cada colapso
    lines = log.split('\n')
    collapse_indices = [i for i, l in enumerate(lines) if 'COLAPSO TOTAL' in l and 'bear_long' in l]
    print(f'Total eventos COLAPSO TOTAL bear_long: {len(collapse_indices)}')
    print()

    # Mostrar contexto [-5, +5] del primer y segundo colapso
    for idx in collapse_indices[:3]:
        start = max(0, idx - 5)
        end   = min(len(lines), idx + 6)
        print(f'--- Evento colapso #{collapse_indices.index(idx)+1} (línea ~{idx}) ---')
        for i in range(start, end):
            marker = '>>>' if i == idx else '   '
            print(f'{marker} {lines[i].strip()[:110]}')
        print()

    # ¿Qué paso del pipeline llama a bear_long?
    bear_call_lines = [l for l in lines if 'bear' in l.lower() and
                       any(k in l for k in ['predict', 'route', 'agent', 'modelo', 'inferencia'])]
    print(f'\nLíneas del log con bear + predict/route/agent (primeras 10):')
    for l in bear_call_lines[:10]:
        print(f'  {l.strip()[:110]}')

# ─── 4. Inspeccionar RegimeRouter — cómo se carga y predice bear ─────────
print()
print(SEP)
print('4. CÓDIGO FUENTE: regime_router.py — Mecanismo de predicción bear')
print(SEP)

router_path = Path('g:/Mi unidad/ia/luna_v2/luna/models/regime_router.py')
if router_path.exists():
    code = router_path.read_text(encoding='utf-8', errors='replace')
    # Buscar la sección de bear_long / std_prob
    lines = code.split('\n')
    # Encontrar línea del std_prob check
    for i, l in enumerate(lines):
        if 'std_prob' in l or 'COLAPSO' in l:
            start = max(0, i-3)
            end   = min(len(lines), i+8)
            print(f'[L{i+1}] Contexto del check std_prob:')
            for j in range(start, end):
                print(f'  {j+1:4d}: {lines[j]}')
            print()
            break

    # Buscar cómo se inicializa el agente bear_long
    for i, l in enumerate(lines):
        if 'bear_long' in l and ('init' in l.lower() or 'load' in l.lower() or 'model' in l.lower()):
            start = max(0, i-2)
            end   = min(len(lines), i+5)
            print(f'[L{i+1}] Inicialización bear_long:')
            for j in range(start, end):
                print(f'  {j+1:4d}: {lines[j]}')
            print()
            if i > 50:
                break

print()
print(SEP)
print('FASE 4 H-A — Inspección de causa raíz completada')
print(SEP)
