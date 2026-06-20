"""
scripts/pre_flight_check.py — Luna V1 Pre-Flight Check System v3.1
==================================================================
Ejecutar ANTES de cualquier run. Si algún test falla el pipeline NO debe arrancarse.

Uso:
    python scripts/pre_flight_check.py
    python scripts/pre_flight_check.py --fail-fast
    python scripts/pre_flight_check.py --verbose
    python scripts/pre_flight_check.py --section sop
    python scripts/pre_flight_check.py --section data

Secciones:
    legacy       TESTS 01-10  : Integridad baseline (v1.0)
    sop          TESTS 11-17  : SOP Iron Rules (R1-R10)
    temporal     TESTS 18-22  : Splits temporales y causalidad
    architecture TESTS 23-28  : Arquitectura y orden de entrenamiento
    artifacts    TESTS 29-33  : Artefactos en disco (modelos, parquets)
    code         TESTS 34-40  : Patrones anti-leakage en codigo
    data         TESTS 41-50  : Calidad del dataset
    math         TESTS 51-57  : Formulas matematicas criticas
    consistency  TESTS 58-63  : Coherencia entre artefactos
    env          TESTS 64-67  : Entorno y dependencias

"Si el codigo no pasa los tests, no esta listo."
Cada bug corregido debe tener su test aqui. Si no tiene test, no esta corregido.
"""

import sys
import re
import json
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from datetime import datetime

# Fix cp1252: consola Windows no soporta caracteres unicode (→, ←, ✅, etc.) en cp1252.
# Forzar UTF-8 para que los nombres de test se impriman correctamente.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────
#  Colores ANSI
# ─────────────────────────────────────────────────────────
class C:
    OK   = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    END  = "\033[0m"
    BOLD = "\033[1m"
    DIM  = "\033[2m"

_tests_registry = {}

def test(name: str, section: str = "misc"):
    def decorator(fn):
        fn._test_name = name
        fn._section = section
        _tests_registry.setdefault(section, []).append(fn)
        return fn
    return decorator


def run_all(fail_fast=False, verbose=False, section_filter=None):
    section_order = [
        "legacy", "sop", "temporal", "architecture",
        "artifacts", "code", "data", "math", "consistency", "env", "v5_bugs", "r14_fixes",
        "data_lake",  # [FIX-PREFLIGHT-DATALAKE-01 2026-06-18] Data Lake Integrity
    ]
    # Normalizar section_filter: puede ser str ("env"), str coma-separado ("env,v5_bugs"),
    # set, list o None. Convertir siempre a set para comparación O(1).
    if section_filter is not None:
        if isinstance(section_filter, str):
            section_filter = {s.strip() for s in section_filter.split(",") if s.strip()}
        else:
            section_filter = set(section_filter)
    all_fns = []
    for sec in section_order:
        if sec in _tests_registry:
            if section_filter is None or sec in section_filter:
                all_fns.extend(_tests_registry[sec])
    for sec, fns in _tests_registry.items():
        if sec not in section_order:
            if section_filter is None or sec in section_filter:
                all_fns.extend(fns)

    passed = failed = warned = 0
    current_section = None
    section_labels = {
        "legacy":       "SECCION 1: Integridad Baseline (v1.0)",
        "sop":          "SECCION 2: SOP Iron Rules",
        "temporal":     "SECCION 3: Splits Temporales y Causalidad",
        "architecture": "SECCION 4: Arquitectura y Orden de Entrenamiento",
        "artifacts":    "SECCION 5: Artefactos en Disco",
        "code":         "SECCION 6: Patrones Anti-Leakage en Codigo",
        "data":         "SECCION 7: Calidad del Dataset",
        "math":         "SECCION 8: Formulas Matematicas Criticas",
        "consistency":  "SECCION 9: Coherencia entre Artefactos",
        "env":          "SECCION 10: Entorno y Dependencias",
        "v5_bugs":      "SECCION 11: V5 Bug Regressions (Run 12)",
        "r14_fixes":    "SECCION 12: Run 14 Fixes (LOG-BUG-01/02, SFI-02, MOD-02)",
        "data_lake":    "SECCION 13: Data Lake Integrity [FIX-PREFLIGHT-DATALAKE-01]",
    }

    print(f"\n{C.BOLD}[PRE-FLIGHT] Luna V1 v3.3 -- {len(all_fns)} tests{C.END}\n")
    print("=" * 70)

    for fn in all_fns:
        sec = fn._section
        if sec != current_section:
            current_section = sec
            print(f"\n{C.BOLD}{C.DIM}{section_labels.get(sec, sec.upper())}{C.END}")
            print("-" * 70)
        try:
            detail = fn()
            # Soporte WARN nativo: si el retorno empieza con "WARN", mostrar en amarillo
            # pero NO contar como FAIL. El pipeline puede continuar.
            if detail and str(detail).startswith("WARN"):
                print(f"  {C.WARN}WARN{C.END}  {fn._test_name}")
                print(f"        {C.WARN}-> {detail}{C.END}")
                warned += 1
            else:
                suffix = f"  {C.DIM}{detail}{C.END}" if detail and verbose else ""
                print(f"  {C.OK}PASS{C.END}  {fn._test_name}{suffix}")
                passed += 1
        except AssertionError as e:
            print(f"  {C.FAIL}FAIL{C.END}  {fn._test_name}")
            print(f"        {C.FAIL}-> {e}{C.END}")
            failed += 1
            if fail_fast:
                break
        except Exception as e:
            print(f"  {C.WARN}ERR {C.END}  {fn._test_name}")
            print(f"        {C.WARN}-> {type(e).__name__}: {e}{C.END}")
            failed += 1
            if fail_fast:
                break

    print("\n" + "=" * 70)
    color = C.OK if failed == 0 else C.FAIL
    status = "ALL SYSTEMS GO" if failed == 0 else f"PIPELINE BLOCKED — {failed} fallo(s)"
    print(f"\n  {C.BOLD}{color}{status}{C.END}")
    warn_suffix = f" | {C.WARN}{warned} WARN{C.END}" if warned > 0 else ""
    print(f"  {passed}/{passed+failed+warned} tests pasados{warn_suffix}\n")
    return failed == 0


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────
def _read(path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

def _is_stale_artifact(json_path: Path, parquet_path: Path, tolerance_s: int = 120) -> bool:
    """
    Devuelve True cuando los artefactos son de runs distintos (residuo mid-run).
    Dos criterios:
    1. TIMESTAMP: si parquet es MAS RECIENTE que json por > tolerance_s => FASE 3A corrio
       sin SFI => JSON es del run anterior.
    2. CONTENIDO: si las features del JSON no estan en el parquet => inconsistencia
       de contenido = artefactos de fases distintas del pipeline (mid-run siempre).
    """
    import os, json as _j
    if not json_path.exists() or not parquet_path.exists():
        return False
    # Criterio 1: timestamps
    mtime_json    = os.path.getmtime(json_path)
    mtime_parquet = os.path.getmtime(parquet_path)
    diff = mtime_parquet - mtime_json
    if diff > tolerance_s or abs(diff) > 4 * 3600:
        return True
    # Criterio 2: inconsistencia de contenido (features en JSON no en parquet)
    try:
        import pandas as _pd
        sel = _j.loads(json_path.read_text(encoding="utf-8"))
        feats = sel.get("selected_features", sel.get("features", []))
        if not feats:
            return False
        cols = set(_pd.read_parquet(parquet_path, columns=[]).columns)
        missing = [f for f in feats if f not in cols]
        # Si mas del 20% de las features del JSON no estan en el parquet => mid-run
        return len(missing) / len(feats) > 0.20
    except Exception:
        return False

def _active(src: str):
    """Lineas ejecutables: sin comentarios ni docstrings."""
    lines, in_doc = [], False
    for i, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        if s.startswith('"""') or s.startswith("'''"):
            in_doc = not in_doc
        if in_doc or s.startswith("#"):
            continue
        lines.append((i, line, s))
    return lines

def _cfg():
    from config.settings import cfg
    return cfg

def _load_parquet(name: str) -> pd.DataFrame:
    path = ROOT / "data" / "features" / name
    assert path.exists(), f"{name} no encontrado: {path}"
    return pd.read_parquet(path)

def _load_json(rel_path: str) -> dict:
    path = ROOT / rel_path
    assert path.exists(), f"{rel_path} no encontrado"
    return json.loads(path.read_text())


# ═══════════════════════════════════════════════════════════
#  SECCION 1: INTEGRIDAD BASELINE (v1.0 — 10 tests)
# ═══════════════════════════════════════════════════════════

def _json_safe(path) -> dict:
    p = Path(path)
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return {}
