"""
tools/diagnostics/audit_momentum_filter.py
Diagnostica el filtro de momentum como cuello de botella del 99.6% de reduccion.
Busca en logs, codigo y signatures el punto exacto de mayor reduccion.
"""
import json, re
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path("g:/Mi unidad/ia/luna_v2")
LOGS = ROOT / "logs"
PREDS = ROOT / "data" / "predictions"

print("=== AUDIT: FILTRO MOMENTUM - DIAGNOSTICO DEL CUELLO DE BOTELLA ===")
print()

# 1. Buscar en signal_filter.py la logica de momentum
print("--- [A] Codigo del filtro momentum en signal_filter.py ---")
sf_src = (ROOT / "luna/models/signal_filter.py").read_text(encoding="utf-8")
# Buscar seccion momentum
idx_mom = sf_src.find("momentum")
if idx_mom > 0:
    ctx = sf_src[max(0, idx_mom-200):idx_mom+800]
    for l in ctx.split("\n")[:30]:
        print(f"  {l}")
print()

# 2. Buscar signal_funnel.json para ver valores reales del embudo
print("--- [B] signal_funnel.json - estructura y valores reales ---")
funnel_candidates = list(PREDS.glob("signal_funnel*.json")) + list((ROOT/"data"/"reports").glob("signal_funnel*.json"))
for fp in funnel_candidates[:3]:
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
        print(f"  {fp.name}:")
        for k, v in d.items():
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"  {fp.name}: error {e}")
print()

# 3. Buscar en los logs los valores del embudo por ventana
print("--- [C] Embudo de senales en logs WFB (valores por ventana) ---")
log_files = sorted(LOGS.glob("wfb_worker*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:3]
funnel_keywords = ["n_initial", "after_xgb", "after_lgbm", "after_meta", "after_momentum",
                   "after_embargo", "n_signals", "filter_funnel", "FUNNEL", "funnel"]
for lf in log_files:
    print(f"  {lf.name}:")
    lines = lf.read_bytes().decode("utf-8", errors="replace").split("\n")
    count = 0
    for l in lines:
        if any(kw.lower() in l.lower() for kw in funnel_keywords):
            print(f"    {l.strip()[:120]}")
            count += 1
            if count > 20:
                break
    if count == 0:
        print("    (no se encontraron lineas de embudo en este log)")
print()

# 4. Buscar en el codigo predict_oos.py la cadena de filtros
print("--- [D] Cadena de filtros en predict_oos.py ---")
predict_src = (ROOT / "luna/models/predict_oos.py").read_text(encoding="utf-8")
# Buscar todas las lineas que reducen señales
filter_patterns = ["momentum", "embargo", "apply_model_threshold",
                   "filter_signals", "n_signals", "after_", "FUNNEL"]
print("  Menciones de filtros en predict_oos.py:")
for i, line in enumerate(predict_src.split("\n"), 1):
    if any(pat.lower() in line.lower() for pat in filter_patterns):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and len(stripped) > 5:
            print(f"    L{i}: {stripped[:100]}")

print()

# 5. Analizar si el momentum filter esta en signal_filter o en predict_oos
print("--- [E] Donde esta implementado el filtro momentum? ---")
has_mom_sf = "momentum" in sf_src.lower()
has_mom_po = "momentum" in predict_src.lower()
print(f"  signal_filter.py tiene 'momentum': {has_mom_sf}")
print(f"  predict_oos.py tiene 'momentum'  : {has_mom_po}")

# Contar ocurrencias
mom_sf = sf_src.lower().count("momentum")
mom_po = predict_src.lower().count("momentum")
print(f"  Ocurrencias en signal_filter.py  : {mom_sf}")
print(f"  Ocurrencias en predict_oos.py    : {mom_po}")

# 6. Buscar en las firmas XGB los n_signals de cada ventana
print()
print("--- [F] Funnel de señales en wfb_cache (signatures por ventana) ---")
wfb_cache = ROOT / "data" / "wfb_cache"
for seed_dir in sorted(wfb_cache.glob("seed*"))[:2]:
    print(f"  {seed_dir.name}:")
    for w in ["W1", "W2", "W3", "W4", "W5"]:
        # Buscar funnel en gate_G5
        g5 = seed_dir / w / "reports" / f"gate_G5_{w}_seed*.json"
        g5_files = list((seed_dir / w / "reports").glob(f"gate_G5_{w}*.json")) if (seed_dir/w/"reports").exists() else []
        if g5_files:
            d = json.loads(g5_files[0].read_text(encoding="utf-8"))
            metrics = d.get("metrics", {})
            print(f"    {w} G5: {metrics}")
        # Buscar signal_funnel.json en la ventana
        sf_json = seed_dir / w / "predictions" / "signal_funnel.json"
        if sf_json.exists():
            d = json.loads(sf_json.read_text(encoding="utf-8"))
            print(f"    {w} funnel: {d}")
