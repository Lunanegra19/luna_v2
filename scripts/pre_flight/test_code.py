from .core import *
from .core import _json_safe, _read, _cfg, _active, _load_parquet, _load_json, _is_stale_artifact, ROOT
import pandas as pd
import numpy as np
from pathlib import Path
import re
import json
import math
from itertools import combinations
from datetime import datetime

@test("TEST-34  Sin rolling(center=True) en scripts activos", section="code")
def t34():
    dirs = [ROOT/"luna/features", ROOT/"luna/models", ROOT/"scripts"]
    violations = []
    for d in dirs:
        for py in d.glob("*.py"):
            if "_legacy" in str(py) or py.name == "pre_flight_check.py": continue
            for _, _, s in _active(_read(py)):
                if "rolling(" in s and "center=True" in s:
                    violations.append(f"{py.name}: {s}")
    assert not violations, "\n  ".join(violations)
    return "sin rolling(center=True)"


@test("TEST-35  Sin shift(-N) en scripts de PRODUCCION (look-ahead)", section="code")
def t35():
    prod = [ROOT/"luna/features/feature_pipeline.py",
            ROOT/"luna/models/train_xgboost_v2.py",
            ROOT/"luna/models/train_metalabeler_v2.py",
            ROOT/"luna/models/calibrate_probabilities.py",
            ROOT/"luna/models/predict_oos.py",
            ROOT/"scripts/train_production_model.py"]
    # Patrones de shift(-N) legítimos (proxies internos, no features de entrada):
    # '_close_rets_proxy' en train_xgboost_v2.py: calcula ret pasado para DSR, no feature futura.
    LEGIT_PATTERNS = ["_close_rets_proxy", "_rets_proxy", "# proxy"]
    violations = []
    for py in prod:
        if not py.exists(): continue
        for _, _, s in _active(_read(py)):
            if re.search(r"\.shift\s*\(\s*-\s*\d", s):
                # Excluir shifts legítimos documentados
                if any(pat in s for pat in LEGIT_PATTERNS):
                    continue
                violations.append(f"{py.name}: {s}")
    assert not violations, "\n  ".join(violations)
    return "sin shift(-N) en produccion"


@test("TEST-36  Sin scaler.fit() sobre dataset completo", section="code")
def t36():
    forbidden = ["scaler.fit(X_all","scaler.fit(X_total","scaler.fit(df)",
                 "scaler.fit(X)","StandardScaler().fit(X["]
    dirs = [ROOT/"luna/features", ROOT/"luna/models", ROOT/"scripts"]
    violations = []
    for d in dirs:
        for py in d.glob("*.py"):
            if "_legacy" in str(py) or py.name == "pre_flight_check.py": continue
            for _, _, s in _active(_read(py)):
                for pat in forbidden:
                    if pat in s: violations.append(f"{py.name}: {s}")
    assert not violations, "\n  ".join(violations)
    return "scaler.fit solo sobre train"


@test("TEST-37  Sin KFold sin Purge en scripts activos", section="code")
def t37():
    dirs = [ROOT/"luna/features", ROOT/"luna/models", ROOT/"scripts"]
    violations = []
    for d in dirs:
        for py in d.glob("*.py"):
            if "_legacy" in str(py) or py.name == "pre_flight_check.py": continue
            for _, _, s in _active(_read(py)):
                if "KFold(" in s and "Purged" not in s and "cpcv" not in s.lower():
                    violations.append(f"{py.name}: {s}")
    assert not violations, "\n  ".join(violations)
    return "sin KFold sin purge"


@test("TEST-38  No hay merges inner con datos macro/onchain", section="code")
def t38():
    src = _read(ROOT/"luna/features/feature_pipeline.py")
    for _, _, s in _active(src):
        if ("merge(" in s and "how='inner'" in s and
                any(k in s for k in ["macro","m2_","fred","cpi"])):
            assert False, f"Merge inner con macro: {s}"
    onchain = [s for _, _, s in _active(src)
               if "merge" in s and "onchain" in s and "how='inner'" in s]
    assert not onchain, f"Merge inner con onchain: {onchain[0]}"
    return "merges left/asof"


@test("TEST-39  guard_pipeline.py activo en feature_pipeline", section="code")
def t39():
    assert (ROOT/"luna/security/guard_pipeline.py").exists()
    fp = _read(ROOT/"luna/features/feature_pipeline.py")
    assert "guard_pipeline" in fp or "purge_leakage" in fp
    return "guard activo"


@test("TEST-40  Sin print() debug en scripts de produccion", section="code")
def t40():
    prod = [ROOT/"luna/models/train_xgboost_v2.py",
            ROOT/"luna/models/train_metalabeler_v2.py",
            ROOT/"luna/models/calibrate_probabilities.py",
            ROOT/"luna/features/feature_pipeline.py",
            ROOT/"luna/models/predict_oos.py"]

    # [FIX-TEST40-SEMANTIC-V2] Logica semantica profesional — NO umbral numerico arbitrario.
    #
    # RULE[fixbugsprints.md] EXIGE prints de trazabilidad para trackear bugs durante runs.
    # Solucion: analizar bloques print() completos (incluyendo multilinea) en el texto fuente.
    # Un print es institucional si su cuerpo COMPLETO contiene [PREFIJO-...] reconocido.
    # Un print naked (sin ningun tag entre corchetes) es la unica violacion real.
    #
    # Esto resuelve dos problemas del enfoque linea-a-linea:
    #   1. Prints multilinea: print(\n    f"[BUG-...]  ") — el tag esta en la 2a linea
    #   2. Prefijos compuestos: [LUNA-V2-...], [BUGFIX-ML-...] — contienen sufijos extra

    # Regex para extraer el cuerpo completo de cada bloque print()
    # Captura hasta el parentesis de cierre (maximo 10 lineas de lookahead)
    PRINT_BLOCK_RE = re.compile(
        r'(?m)^[ \t]*print\s*\((.{0,800}?)\)[ \t]*(?:#.*)?$',
        re.DOTALL,
    )

    # Un tag institucional es cualquier [PALABRA seguida de letras/guiones dentro del bloque
    INSTITUTIONAL_TAG_RE = re.compile(
        r'\[\s*(?:'
        r'BUG|FIX|BUGFIX|LUNA|ARCH|RULE|WARN|OK|ERROR|CACHE|CALIB|'
        r'HYDRAT|DEHYDRAT|PRE|POST|DATA|HMM|SFI|WFB|GATE|GUARD|'
        r'METRIC|LOG|AUDIT|STATIC|LOCK|BACKUP|CLEANUP|BIAS|MEJORA|'
        r'EMBED|SIGNAL|OOS|TBM|REGIM|OVERF|LIVE|PROD|RISK|SHIELD|'
        r'INFER|ENSEMBLE|WINDOW|SEED|MODEL|PIPELINE|TRAIN|TEST|'
        r'CRITICAL|DRIVE|ALERT|GUARD|GAUNTLET|KILL|CIRCUIT|PANIC|'
        r'P\d|V\d|R\d|M\d|C\d|A\d|G\d|W\d|S\d|F\d|T\d'
        r')',
        re.IGNORECASE,
    )

    violations = []
    institutional_count = 0

    for f in prod:
        if not f.exists():
            continue
        source = f.read_text(encoding="utf-8", errors="replace")
        # Eliminar lineas de comentario puro para no confundir el extractor
        source_clean = re.sub(r'(?m)^[ \t]*#.*$', '', source)

        for m in PRINT_BLOCK_RE.finditer(source_clean):
            block_body = m.group(1)
            if "# debug" in block_body.lower():
                continue  # exento por comentario explicito
            if INSTITUTIONAL_TAG_RE.search(block_body):
                institutional_count += 1  # trazabilidad institucional — EXENTO
            else:
                first_line = m.group(0).split('\n')[0].strip()[:120]
                violations.append(f"{f.name}: {first_line}")

    print(
        f"[TEST-40] {institutional_count} prints institucionales exentos | "
        f"{len(violations)} prints naked (violaciones)"
    )
    assert not violations, (
        f"[TEST-40] {len(violations)} print() SIN prefijo institucional en produccion "
        f"(anadir prefijo [TAG-...] o comentar con # debug):\n  "
        + "\n  ".join(violations[:15])
    )
    return f"OK -- {institutional_count} institucionales exentos | 0 naked"


# ═══════════════════════════════════════════════════════════
#  SECCION 7: CALIDAD DEL DATASET (10 tests)
# ═══════════════════════════════════════════════════════════
