"""
scan_magic_numbers.py
======================
[AUDIT-MAGIC-01] Escanea TODOS los ficheros Python del proyecto buscando
literales numéricos que parezcan parámetros mágicos (thresholds, limits, multipliers).

Genera una tabla CSV con: fichero, línea, valor, categoría, contexto.
"""
import pathlib, re, sys, csv

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tools" / "diagnostics" / "magic_numbers_audit.csv"

print(f"[AUDIT-MAGIC-01] Escaneando proyecto en: {ROOT}")

# Valores tan genéricos que no son magic numbers
SKIP_VALUES = {0, 1, -1, 2, 3, 10, 100}

# Módulos a escanear
SCAN_DIRS = [
    ROOT / "luna",
    ROOT / "scripts",
    ROOT / "config",
    ROOT / "luna" / "risk",
    ROOT / "luna" / "sizing",
    ROOT / "luna" / "labeling",
    ROOT / "luna" / "monitoring",
    ROOT / "luna" / "validation",
    ROOT / "luna" / "ai_mining",
    ROOT / "luna" / "features",
    ROOT / "luna" / "models",
    ROOT / "luna" / "losses",
    ROOT / "luna" / "calibration",
]

# Categorías por tipo de valor
def categorize(val, ctx):
    ctx_low = ctx.lower()
    if any(k in ctx_low for k in ["embargo", "purge", "horizon", "barrier", "hours"]):
        return "TEMPORAL"
    if any(k in ctx_low for k in ["threshold", "thr", "prob", "min_prob", "signal"]):
        return "THRESHOLD"
    if any(k in ctx_low for k in ["kelly", "position", "fraction", "max_pos", "sizing"]):
        return "RISK_SIZING"
    if any(k in ctx_low for k in ["cost", "fee", "slippage", "pct"]):
        return "COST"
    if any(k in ctx_low for k in ["dsr", "sharpe", "sr_", "min_dsr", "validation"]):
        return "VALIDATION"
    if any(k in ctx_low for k in ["split", "val_", "train_", "80", "0.8"]):
        return "SPLIT_RATIO"
    if any(k in ctx_low for k in ["n_group", "n_trial", "n_fold", "cpcv", "optuna"]):
        return "CV_PARAMS"
    if any(k in ctx_low for k in ["hmm", "regime", "state", "bear", "bull", "range"]):
        return "REGIME"
    if any(k in ctx_low for k in ["lr", "dropout", "hidden", "lstm", "epoch", "batch"]):
        return "NN_ARCH"
    if any(k in ctx_low for k in ["pt_", "sl_", "profit", "stop_loss", "multiplier"]):
        return "TBM_BARRIERS"
    if any(k in ctx_low for k in ["alpha", "genetic", "dtw", "mining"]):
        return "AI_MINING"
    return "OTHER"

rows = []

def scan_file(path):
    try:
        src = path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return []

    found = []
    lines = src.splitlines()
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        # Ignorar líneas de comentario puro
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # Ignorar imports
        if stripped.startswith('import ') or stripped.startswith('from '):
            continue

        # PATRON 1: asignaciones de float tipo 0.XX
        for m in re.finditer(r'(?<!["\'\w])(\d*\.\d+)(?!["\'\w])', line):
            val_str = m.group(1)
            try:
                val = float(val_str)
            except:
                continue
            if val in SKIP_VALUES or val == 0.0:
                continue
            # Ignorar si está en un string
            before = line[:m.start()]
            if before.count('"') % 2 == 1 or before.count("'") % 2 == 1:
                continue
            found.append((lineno, val_str, stripped[:150]))

        # PATRON 2: enteros especiales (no 0,1,2,-1)
        for m in re.finditer(r'(?<!\w)([3-9][0-9]{1,3})(?!\w)', line):
            val_str = m.group(1)
            try:
                val = int(val_str)
            except:
                continue
            if val in SKIP_VALUES:
                continue
            # Solo en contexto de asignación o comparación
            ctx_before = line[:m.start()].rstrip()
            if not any(op in ctx_before[-3:] for op in ['= ', '>=', '<=', '> ', '< ', '(', ',']):
                continue
            before = line[:m.start()]
            if before.count('"') % 2 == 1 or before.count("'") % 2 == 1:
                continue
            found.append((lineno, val_str, stripped[:150]))

    return found

seen_locs = set()
for d in sorted(set(SCAN_DIRS)):
    if not d.exists():
        continue
    for py_file in sorted(d.rglob("*.py")):
        if "__pycache__" in str(py_file) or "_archive" in str(py_file):
            continue
        rel = str(py_file.relative_to(ROOT)).replace("\\", "/")
        items = scan_file(py_file)
        for lineno, val_str, ctx in items:
            loc_key = f"{rel}:{lineno}:{val_str}"
            if loc_key in seen_locs:
                continue
            seen_locs.add(loc_key)
            cat = categorize(float(val_str) if '.' in val_str else int(val_str), ctx)
            rows.append({
                "fichero": rel,
                "linea": lineno,
                "valor": val_str,
                "categoria": cat,
                "contexto": ctx
            })

# Ordenar por categoría y fichero
rows.sort(key=lambda r: (r["categoria"], r["fichero"], r["linea"]))

# Escribir CSV
with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["fichero", "linea", "valor", "categoria", "contexto"])
    w.writeheader()
    w.writerows(rows)

print(f"[AUDIT-MAGIC-01] Total literales encontrados: {len(rows)}")
print(f"[AUDIT-MAGIC-01] Resultado en: {OUTPUT}")

# Imprimir resumen por categoria
from collections import Counter
cats = Counter(r["categoria"] for r in rows)
print("\n=== RESUMEN POR CATEGORÍA ===")
for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat:<20} {n:>4} ocurrencias")

# Imprimir las categorías más críticas completas
CRITICAL_CATS = ["THRESHOLD", "TEMPORAL", "RISK_SIZING", "SPLIT_RATIO", "REGIME"]
print("\n=== DETALLE CATEGORÍAS CRÍTICAS ===")
for r in rows:
    if r["categoria"] in CRITICAL_CATS:
        print(f"  [{r['categoria']}] {r['fichero']}:{r['linea']} | val={r['valor']} | {r['contexto'][:100]}")
