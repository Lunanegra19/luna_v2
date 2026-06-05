"""
arch06_to_11_code_inspection.py
=================================
Diagnostico batch de ARCH-06 al ARCH-11 mediante inspeccion de codigo.
Todos son verificables sin ejecucion de modelos — solo lectura de archivos.

ARCH-06: HMM routing semánticamente incoherente cross-window
ARCH-07: timing_features bypass SFI silencioso
ARCH-08: ETH lag=0 sin validacion empirica
ARCH-09: BUG-03 Guard — threshold reduction viola logica calibrador
ARCH-10: Inconsistencia embargo 24H vs 72H vs SOP 96H
ARCH-11: MockXGBClassifier en produccion sin fallo explicito

USO: python tools/diagnostics/arch06_to_11_code_inspection.py
"""
import sys, yaml, json, re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

LUNA = ROOT / "luna"
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "config"

def read_file_lines(path, start=None, end=None):
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if start and end:
            return lines[start-1:end]
        return lines
    except:
        return []

def grep_file(path, pattern, context=2):
    results = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                results.append((i+1, line.strip()))
    except:
        pass
    return results

print("="*70)
print("[ARCH-06 al ARCH-11] DIAGNOSTICO BATCH DE CODIGO")
print("="*70)

# ── ARCH-06: HMM routing incoherente cross-window ─────────────────────────────
print("\n[ARCH-06] HMM ROUTING CROSS-WINDOW: consistencia del state_map")
print("-"*60)
hmm_path = LUNA / "models" / "hmm_regime.py"
results = grep_file(hmm_path, r"state_map|_analyze_and_map|predict_regime_series|warm_start")
print(f"  Archivo: {hmm_path.name} ({'EXISTE' if hmm_path.exists() else 'NO EXISTE'})")
for lno, line in results[:15]:
    print(f"  L{lno:4}: {line[:100]}")

# Verificar si hay warm_start en el HMM
warm_start_hits = grep_file(hmm_path, r"warm_start|means_prior|previous_model|wfb_cache")
print(f"\n  WarmStart / previo model hits: {len(warm_start_hits)}")
if warm_start_hits:
    for lno, line in warm_start_hits[:5]:
        print(f"    L{lno}: {line[:100]}")

# Verificar en wfb_worker
worker_path = SCRIPTS / "wfb_worker.py"
warm_worker = grep_file(worker_path, r"hmm.*warm|warm.*hmm|predict_regime|HMM_Semantic.*window")
print(f"\n  wfb_worker.py HMM warm hints: {len(warm_worker)}")
for lno, line in warm_worker[:5]:
    print(f"    L{lno}: {line[:100]}")

# ── ARCH-07: timing_features bypass SFI ───────────────────────────────────────
print("\n[ARCH-07] TIMING_FEATURES BYPASS SFI")
print("-"*60)
sfi_path = LUNA / "features" / "feature_selection_e.py"
timing_hits = grep_file(sfi_path, r"timing|calendar|PASSTHROUGH|pass.through|bypass|excluded.*timing")
print(f"  Hits 'timing/calendar/PASSTHROUGH' en feature_selection_e.py: {len(timing_hits)}")
for lno, line in timing_hits[:10]:
    print(f"  L{lno:4}: {line[:120]}")

# Buscar definicion de PASSTHROUGH_FEATURES
pass_hits = grep_file(sfi_path, r"PASSTHROUGH_FEATURES\s*=")
for lno, line in pass_hits[:5]:
    print(f"\n  PASSTHROUGH_FEATURES definicion L{lno}: {line[:150]}")

# ── ARCH-08: ETH lag=0 ────────────────────────────────────────────────────────
print("\n[ARCH-08] ETH LAG=0 SIN VALIDACION EMPIRICA")
print("-"*60)
fp_path = LUNA / "features" / "feature_pipeline.py"
eth_hits = grep_file(fp_path, r"ETH.*lag|lag.*ETH|eth_lag|lag.*=.*0|milag0")
print(f"  Hits ETH lag en feature_pipeline.py: {len(eth_hits)}")
for lno, line in eth_hits[:10]:
    print(f"  L{lno:4}: {line[:120]}")

# Buscar en feature_selection_e.py tambien
eth_sfi = grep_file(sfi_path, r"ETH.*lag|lag.*ETH|milag0h|milag1h")
print(f"  Hits ETH en feature_selection_e.py: {len(eth_sfi)}")
for lno, line in eth_sfi[:5]:
    print(f"  L{lno:4}: {line[:120]}")

# Verificar el lag real descubierto por SFI
feat_dir = ROOT / "data" / "features"
sf_path = feat_dir / "selected_features.json"
if sf_path.exists():
    sf = json.loads(sf_path.read_text(encoding="utf-8"))
    eth_feats = [f for f in sf.get("selected_features", []) + sf.get("pass_through_features", []) if "ETH" in f.upper()]
    print(f"\n  ETH features seleccionadas por SFI: {eth_feats}")
    if eth_feats:
        lags = [re.search(r"milag(\d+)", f) for f in eth_feats]
        lag_values = [int(m.group(1)) for m in lags if m]
        print(f"  Lags reales ETH en produccion: {lag_values}")
        if any(l == 0 for l in lag_values):
            print("  *** ARCH-08 CONFIRMADO: ETH tiene lag=0 en features de produccion ***")

# ── ARCH-09: BUG-03 GUARD threshold reduction ─────────────────────────────────
print("\n[ARCH-09] BUG-03 GUARD: THRESHOLD REDUCTION")
print("-"*60)
signal_filter_paths = list(LUNA.glob("**/signal_filter.py")) + list(LUNA.glob("**/predict_oos.py"))
for sfpath in signal_filter_paths[:2]:
    print(f"  Archivo: {sfpath.name}")
    guard_hits = grep_file(sfpath, r"BUG.03|guard.*threshold|threshold.*reduce|min_threshold|floor.*thresh")
    for lno, line in guard_hits[:8]:
        print(f"  L{lno:4}: {line[:120]}")

# Verificar en train_xgboost_v2.py
xgb_path = LUNA / "models" / "train_xgboost_v2.py"
guard_xgb = grep_file(xgb_path, r"BUG.03|guard.*thresh|threshold_floor|signal_rescue")
print(f"\n  BUG-03 en train_xgboost_v2.py: {len(guard_xgb)} hits")
for lno, line in guard_xgb[:5]:
    print(f"  L{lno:4}: {line[:120]}")

# ── ARCH-10: Embargo inconsistency ────────────────────────────────────────────
print("\n[ARCH-10] EMBARGO INCONSISTENCIA 24H/72H/96H vs SOP 96H")
print("-"*60)
with open(CONFIG / "settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

sop_cfg = cfg.get("sop", {})
xgb_cfg = cfg.get("xgboost", {})
hmm_cfg = cfg.get("hmm", {})
wfb_cfg = cfg.get("wfb", {})

print(f"  sop.embargo_hours          : {sop_cfg.get('embargo_hours', 'NO')}")
print(f"  xgboost.embargo_hours      : {xgb_cfg.get('embargo_hours', 'NO')}")
print(f"  xgboost.sfi_embargo_h      : {xgb_cfg.get('sfi_embargo_h', 'NO')}")
print(f"  hmm.oos_window_hours       : {hmm_cfg.get('oos_window_hours', 'NO')}")
print(f"  wfb.embargo_hours          : {wfb_cfg.get('embargo_hours', 'NO')}")
# Buscar en el archivo directamente
embargo_hits = grep_file(CONFIG / "settings.yaml", r"embargo")
print(f"\n  Todas las instancias de 'embargo' en settings.yaml:")
for lno, line in embargo_hits[:15]:
    print(f"  L{lno:4}: {line[:120]}")

# SFI embargo
sfi_embargo = grep_file(sfi_path, r"sfi_embargo|embargo_h\b|purge_h\b|embargo.*hours")
print(f"\n  SFI embargo referencias en feature_selection_e.py:")
for lno, line in sfi_embargo[:5]:
    print(f"  L{lno:4}: {line[:120]}")

# ── ARCH-11: MockXGBClassifier ────────────────────────────────────────────────
print("\n[ARCH-11] MOCKXGBCLASSIFIER EN PRODUCCION")
print("-"*60)
# Buscar en todos los archivos Python
mock_hits = []
for py_file in LUNA.rglob("*.py"):
    hits = grep_file(py_file, r"MockXGB|MockClassifier|mock.*classifier|class Mock")
    if hits:
        mock_hits.append((py_file, hits))

for fpath, hits in mock_hits:
    print(f"\n  {fpath.relative_to(ROOT)}:")
    for lno, line in hits[:5]:
        print(f"  L{lno:4}: {line[:120]}")

# Verificar en scripts
for py_file in SCRIPTS.rglob("*.py"):
    hits = grep_file(py_file, r"MockXGB|MockClassifier")
    if hits:
        print(f"\n  scripts/{py_file.name}:")
        for lno, line in hits[:5]:
            print(f"    L{lno}: {line[:120]}")

if not mock_hits:
    print("  [INFO] No se encontraron referencias a MockXGBClassifier en luna/")
    print("  H3 de ARCH-01 ya habia sido descartada — las señales son reales (std=0.14)")

print("\n"+"="*70)
print("RESUMEN ARCH-06 al ARCH-11")
print("="*70)
print("""
  ARCH-06: Requiere analisis adicional del HMM state_map cross-window
  ARCH-07: Verificar si PASSTHROUGH_FEATURES incluye timing sin SFI filter
  ARCH-08: Verificar lag real de ETH en selected_features.json
  ARCH-09: Verificar si BUG-03 GUARD existe en signal pipeline
  ARCH-10: Verificar inconsistencia de embargo en settings.yaml
  ARCH-11: Verificar si MockXGBClassifier existe en codebase de produccion
""")
print("[ARCH-06-11] Diagnostico completado.")
