"""
audit_v2.py — Auditoría institucional automatizada de Luna V2
Busca: bare except, division insegura, leakage de datos, tz-naive, errores matemáticos.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(r"G:/Mi unidad/ia/luna_v2")
files = list(ROOT.glob("scripts/*.py")) + list(ROOT.glob("luna/**/*.py"))
files = [f for f in files if "__pycache__" not in str(f)]

issues = []

def add(category, fpath, lineno, msg):
    issues.append({"cat": category, "file": fpath.name, "line": lineno, "msg": msg})


# ─── PASS 1: AST ANALYSIS ────────────────────────────────────────────────────
for fpath in files:
    try:
        src = fpath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except SyntaxError as e:
        add("SYNTAX_ERROR", fpath, e.lineno, str(e))
        continue

    for node in ast.walk(tree):
        # DIV BY ZERO literal
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                add("DIV_BY_ZERO", fpath, node.lineno, "División literal por 0")

        # Bare except que puede silenciar SystemExit/KeyboardInterrupt
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            add("BARE_EXCEPT", fpath, node.lineno, "bare except: puede silenciar SystemExit/KeyboardInterrupt")

        # == None en lugar de is None (anti-patrón, a veces matemáticamente peligroso en pandas)
        if isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant) and comp.value is None:
                    add("EQ_NONE", fpath, node.lineno, "Usar 'is None' en lugar de '== None' (puede fallar con arrays pandas)")

        # Operación aritmética sobre resultado de len() sin guardia
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Call):
                if isinstance(node.right.func, ast.Name) and node.right.func.id == "len":
                    add("UNSAFE_DIV_LEN", fpath, node.lineno, "División por len() sin guardia de cero — puede ZeroDivisionError si DataFrame vacío")


# ─── PASS 2: LINE-BY-LINE TEXT ANALYSIS ───────────────────────────────────────
LEAKAGE_PATTERNS = [
    (".shift(-",            "FUTURE_SHIFT",    "shift negativo puede introducir futuro"),
    ("features_holdout",    "HOLDOUT_LEAKAGE", "features_holdout referenciado fuera de predict/validate"),
    ("datetime.now()",      "TZ_NAIVE_NOW",    "datetime.now() sin tz — inconsistente con UTC-aware index"),
    ("pd.Timestamp.now()",  "TZ_NAIVE_NOW",    "pd.Timestamp.now() sin tz=utc"),
    (".dropna(inplace=True)","INPLACE_DROPNA",  "inplace=True en dropna puede causar SettingWithCopyWarning"),
    ("reset_index(inplace=","INPLACE_RESET",   "reset_index inplace sobre slice — posible SettingWithCopyWarning"),
    ("fillna(inplace=",     "INPLACE_FILLNA",  "fillna inplace en slice — posible SettingWithCopyWarning"),
    ("merge(df",            "UNNAMED_MERGE",   "merge sin validate= puede crear duplicados silenciosamente"),
    ("sort_values(inplace=","INPLACE_SORT",    "sort_values inplace — puede romper alineación de índice"),
]

for fpath in files:
    try:
        src = fpath.read_text(encoding="utf-8", errors="replace")
    except:
        continue
    lines = src.splitlines()
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        for pattern, cat, msg in LEAKAGE_PATTERNS:
            if pattern in s:
                # Filtrar falsos positivos conocidos
                if cat == "HOLDOUT_LEAKAGE" and any(x in fpath.name for x in ["wfb_worker", "predict_oos", "pipeline_exec"]):
                    continue  # uso correcto en estos módulos
                if cat == "FUTURE_SHIFT" and "target" in s.lower():
                    continue  # target engineering intencional
                if cat == "TZ_NAIVE_NOW" and ("utc=True" in s or "timezone.utc" in s or "UTC" in s):
                    continue
                add(cat, fpath, i, f"{msg}: {s[:80]}")


# ─── PASS 3: MATH PATTERN ANALYSIS ───────────────────────────────────────────
MATH_PATTERNS = [
    # Sharpe incorrecto: no anualizar o dividir por std de retornos
    ("sharpe",      "SHARPE_CALC", "Verificar que Sharpe anualiza correctamente (√252 para diario, √8760 para horario)"),
    ("np.log(0",    "LOG_ZERO",    "log(0) = -inf"),
    ("math.log(0",  "LOG_ZERO",    "log(0) = -inf"),
    # Kelly sin cap
    ("kelly",       "KELLY_NOCAP", "Verificar que Kelly fraction está cappada a max_fraction"),
    # Brier score invertido
    ("brier",       "BRIER_SIGN",  "Brier score: menor = mejor. Verificar que no hay comparación invertida"),
    # Correlación de Pearson con NaN
    (".corr()",     "CORR_NAN",    "corr() sin dropna previo puede retornar NaN que luego se usa en comparaciones"),
    # std() con ddof incorrecto
    (".std(ddof=0", "STD_DDOF0",   "std(ddof=0) es población (biased), ¿es intencional?"),
    # log retornos vs retornos aritméticos mezclados
    ("np.log(",     "LOG_RETURNS", "verificar coherencia: log retorno vs aritmético"),
]

for fpath in files:
    try:
        src_lower = fpath.read_text(encoding="utf-8", errors="replace").lower()
    except:
        continue
    lines_orig = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
    lines_lower = src_lower.splitlines()
    for i, (lo, orig) in enumerate(zip(lines_lower, lines_orig), 1):
        s = lo.strip()
        if not s or s.startswith("#"):
            continue
        for pattern, cat, msg in MATH_PATTERNS:
            if pattern.lower() in s:
                add(cat, fpath, i, f"{msg}: {orig.strip()[:80]}")


# ─── OUTPUT ─────────────────────────────────────────────────────────────────
from collections import Counter
cats = Counter(x["cat"] for x in issues)

print("=" * 70)
print("LUNA V2 - AUDITORÍA INSTITUCIONAL AUTOMATIZADA")
print(f"Archivos analizados: {len(files)}")
print(f"Issues encontrados: {len(issues)}")
print("=" * 70)
print("\nRESUMEN POR CATEGORÍA:")
for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat:<25} {cnt:>4}")

print("\nDETALLE (primeras 15 por categoría crítica):")
CRITICAL_CATS = ["SYNTAX_ERROR", "DIV_BY_ZERO", "BARE_EXCEPT", "FUTURE_SHIFT",
                 "HOLDOUT_LEAKAGE", "TZ_NAIVE_NOW", "LOG_ZERO", "UNSAFE_DIV_LEN"]
for cat in CRITICAL_CATS:
    group = [x for x in issues if x["cat"] == cat]
    if group:
        print(f"\n── {cat} ({len(group)} ocurrencias) ──")
        for x in group[:15]:
            print(f"  {x['file']}:{x['line']} — {x['msg']}")
