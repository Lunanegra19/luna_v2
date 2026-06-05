import json, re
from pathlib import Path

ROOT = Path("g:/Mi unidad/ia/luna_v2")
PREDS = ROOT / "data" / "predictions"
MODELS = ROOT / "data" / "models"

# ─── 1. Causa EV negativo: avg_win vs avg_loss en TBM ───────────────────────
print("=== CAUSA EV NEGATIVO: avg_win vs avg_loss en TBM ===")
print("Si WR=58% pero EV<0, avg_loss >> avg_win (asimetria negativa del TBM)")
print()
for agent in ["bull", "range"]:
    sp = MODELS / f"xgboost_meta_{agent}_long_signature.json"
    sig = json.loads(sp.read_text(encoding="utf-8"))
    rep = sig.get("calibration_report", [])
    if not rep:
        continue
    best = max(rep, key=lambda x: x.get("ev", -999))
    wr      = best.get("wr", best.get("win_rate", 0))
    avg_win = best.get("avg_win", 0)
    avg_los = best.get("avg_loss", 0)
    ev      = best.get("ev", 0)
    thr     = best.get("threshold", best.get("t", "?"))
    print(f"{agent.upper()} (mejor thr={thr}):")
    print(f"  WR={wr:.1%}  avg_win={avg_win:.5f}  avg_loss={avg_los:.5f}")
    if avg_win > 0:
        ratio = avg_los / avg_win
        wr_breakeven = avg_los / (avg_win + avg_los)
        print(f"  Loss/Win ratio    = {ratio:.2f}x")
        print(f"  WR breakeven (EV=0) = {wr_breakeven:.1%}  (modelo tiene {wr:.1%} < {wr_breakeven:.1%})")
        print(f"  Deficit WR        = {wr_breakeven - wr:.1%} de WR adicional necesaria")
    print(f"  EV confirmado     = {ev:.5f}")
    print()

# ─── 2. Cadena de filtros: signal_funnel.json actual ───────────────────────
print("=== signal_funnel.json ACTUAL - cadena causal completa ===")
sf_path = PREDS.parent / "reports" / "signal_funnel.json"
if sf_path.exists():
    d = json.loads(sf_path.read_text(encoding="utf-8"))
    total = d.get("raw_oos_bars", 1)
    prev = total
    items = [(k, v) for k, v in d.items()]
    for k, v in items:
        if isinstance(v, int) and k != "filter_fallback_level":
            pct = v / total * 100
            delta = v - prev if k != "raw_oos_bars" else 0
            delta_str = f"(perdidos: {abs(delta)})" if delta < 0 else ""
            print(f"  {k:30s}: {v:6d}  ({pct:5.1f}%)  {delta_str}")
            prev = v
        else:
            print(f"  {k:30s}: {v}")

# ─── 3. Cadena de filtros en predict_oos.py ─────────────────────────────────
print()
print("=== CADENA DE FILTROS en predict_oos.py ===")
src = (ROOT / "luna/models/predict_oos.py").read_text(encoding="utf-8")
funnel_keys = ["after_xgb", "after_lgbm", "after_ood", "after_hmm",
               "after_meta", "after_cash", "after_momentum", "after_embargo",
               "n_raw", "raw_oos", "funnel"]
lines_found = []
for i, line in enumerate(src.split("\n"), 1):
    if any(kw in line for kw in funnel_keys):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and len(stripped) > 5:
            lines_found.append((i, stripped[:110]))
for i, l in lines_found[:30]:
    print(f"  L{i:4d}: {l}")

# ─── 4. Que hace apply_momentum exactamente ──────────────────────────────────
print()
print("=== IMPLEMENTACION apply_momentum en signal_filter.py ===")
sf_src = (ROOT / "luna/models/signal_filter.py").read_text(encoding="utf-8")
# Buscar el metodo apply_momentum
idx_start = sf_src.find("def apply_momentum")
if idx_start > 0:
    idx_end = sf_src.find("\n    def ", idx_start + 10)
    method = sf_src[idx_start:idx_end if idx_end > 0 else idx_start+3000]
    for l in method.split("\n")[:60]:
        print(f"  {l}")
else:
    print("  'def apply_momentum' no encontrado en signal_filter.py")
    # Buscar cualquier mencion de momentum en el filtro
    for i, l in enumerate(sf_src.split("\n"), 1):
        if "momentum" in l.lower() and "filter" in l.lower():
            print(f"  L{i}: {l.strip()[:100]}")
