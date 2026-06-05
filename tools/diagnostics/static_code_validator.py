"""
tools/diagnostics/static_code_validator.py
============================================
Validador pre-run de Luna V2. Dos capas sin ejecución de datos:

  CAPA 1 — Análisis AST (código fuente):
    [DICT-KEY]    self.results["key"] leída pero nunca asignada → KeyError.
    [LOCAL-DICT]  dict literal con clave no definida → KeyError.
    [IMPORT]      Paquete crítico no instalado en el entorno.
    [SYNTAX]      Error de sintaxis Python en archivo crítico.

  CAPA 2 — Estado del entorno y artefactos (sin cargar datos):
    [LOCK]        .wfb_lock huérfano con PID muerto → bloqueo de run.
    [SCHEMA]      selected_features.json referencia features ausentes en parquet.
    [NUMERIC]     NaN/Inf en columnas clave (muestreo 500 filas).
    [DRIFT]       Schema del parquet cambió >10% respecto al run anterior.
    [RESOURCE]    RAM/VRAM insuficiente para el pipeline.

Uso:
    python tools/diagnostics/static_code_validator.py
    python tools/diagnostics/static_code_validator.py --no-env   # solo AST
    python tools/diagnostics/static_code_validator.py --fail-fast

Integración con el orquestador:
    Llamado automáticamente por run_wfb_orchestrator.py antes de
    cualquier run. Si detecta ERRORs, aborta el pipeline.
"""

import ast
import sys
import time
import argparse
import importlib.util
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ─── ANSI ────────────────────────────────────────────────────────────────────
class C:
    OK   = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    INFO = "\033[94m"
    END  = "\033[0m"
    BOLD = "\033[1m"
    DIM  = "\033[2m"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Dataclasses ─────────────────────────────────────────────────────────────
@dataclass
class Issue:
    severity: str          # "ERROR" | "WARN"
    check_id: str          # e.g. "DICT-KEY"
    file: str
    line: int
    message: str

@dataclass
class ValidationResult:
    issues: List[Issue] = field(default_factory=list)
    files_checked: int = 0
    elapsed_ms: float = 0.0

    @property
    def errors(self):
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self):
        return [i for i in self.issues if i.severity == "WARN"]

    def ok(self):
        return len(self.errors) == 0


# =============================================================================
# CHECK 1: self.results dict key consistency
# =============================================================================
class DictKeyConsistencyVisitor(ast.NodeVisitor):
    """
    Detecta accesos self.<attr>["key"] con clave literal donde la misma
    clave nunca fue asignada via self.<attr>["key"] = ... en la misma clase.

    Ejemplo de bug detectado:
        self.results["n_input"]      ← solo lectura → KeyError
        self.results["n_after_B"] = x  ← solo escritura → ok
    """

    TARGET_ATTRS = {"results", "scores", "_state", "metrics"}  # atributos de instancia a vigilar

    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Issue] = []
        self._class_stack: List[str] = []
        self._method_stack: List[str] = []

        # Por clase: {cls: {attr: {key: {"writes": [...], "reads": [...]}}}}
        self._class_dict_usage: Dict[str, Dict[str, Dict[str, Dict[str, list]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: {"writes": [], "reads": []}))
        )

    # ── Tracking de contexto ─────────────────────────────────────────────
    def visit_ClassDef(self, node):
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node):
        self._method_stack.append(node.name)
        self.generic_visit(node)
        self._method_stack.pop()
    visit_AsyncFunctionDef = visit_FunctionDef

    # ── Detectar escrituras: self.attr["key"] = valor ────────────────────
    def visit_Assign(self, node):
        for target in node.targets:
            self._check_subscript_write(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._check_subscript_write(node.target)
        self.generic_visit(node)

    def _check_subscript_write(self, node):
        """self.results["key"] = ... → registro de escritura"""
        if not isinstance(node, ast.Subscript):
            return
        if not isinstance(node.value, ast.Attribute):
            return
        if not isinstance(node.value.value, ast.Name):
            return
        if node.value.value.id != "self":
            return
        attr = node.value.attr
        if attr not in self.TARGET_ATTRS:
            return
        key = self._extract_str_key(node.slice)
        if key is None:
            return
        cls = self._class_stack[-1] if self._class_stack else "__module__"
        self._class_dict_usage[cls][attr][key]["writes"].append(node.lineno)

    # ── Detectar lecturas: self.attr["key"] (no asignación) ──────────────
    def visit_Subscript(self, node):
        # Solo nos interesan lecturas (no el lado izquierdo de asignaciones,
        # que ya capturamos en visit_Assign)
        if not isinstance(node.value, ast.Attribute):
            self.generic_visit(node)
            return
        if not isinstance(node.value.value, ast.Name):
            self.generic_visit(node)
            return
        if node.value.value.id != "self":
            self.generic_visit(node)
            return

        attr = node.value.attr
        if attr not in self.TARGET_ATTRS:
            self.generic_visit(node)
            return

        # Ignorar si ya está siendo analizado como target de asignación
        key = self._extract_str_key(node.slice)
        if key is None:
            self.generic_visit(node)
            return

        cls = self._class_stack[-1] if self._class_stack else "__module__"
        self._class_dict_usage[cls][attr][key]["reads"].append(node.lineno)
        self.generic_visit(node)

    @staticmethod
    def _extract_str_key(slice_node) -> Optional[str]:
        """Extrae la clave si es un literal string."""
        # Python 3.9+: slice es directamente el nodo
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        # Python 3.8: slice puede ser ast.Index
        if isinstance(slice_node, ast.Index):
            if isinstance(slice_node.value, ast.Constant) and isinstance(slice_node.value.value, str):
                return slice_node.value.value
        return None

    def get_issues(self) -> List[Issue]:
        """
        Genera issues para claves que tienen lecturas pero NINGUNA escritura en la clase.
        """
        issues = []
        for cls, attrs in self._class_dict_usage.items():
            for attr, keys in attrs.items():
                for key, access in keys.items():
                    writes = access.get("writes", [])
                    reads  = access.get("reads",  [])
                    if reads and not writes:
                        for lineno in reads:
                            issues.append(Issue(
                                severity="ERROR",
                                check_id="DICT-KEY",
                                file=self.filename,
                                line=lineno,
                                message=(
                                    f"self.{attr}[\"{key}\"] leída en {cls} "
                                    f"pero NUNCA asignada en esta clase → KeyError garantizado."
                                )
                            ))
        return issues


# =============================================================================
# CHECK 2: imports críticos
# =============================================================================
CRITICAL_IMPORTS = [
    ("xgboost",        "XGBoost — necesario para SFI"),
    ("lightgbm",       "LightGBM — necesario para ensemble"),
    ("optuna",         "Optuna — necesario para HPO"),
    ("hmmlearn",       "hmmlearn — necesario para HMM"),
    ("statsmodels",    "statsmodels — necesario para ADF/FracDiff"),
    ("sklearn",        "scikit-learn — base del pipeline"),
    ("pandas",         "pandas"),
    ("numpy",          "numpy"),
    ("pyarrow",        "pyarrow — lectura/escritura parquet"),
    ("loguru",         "loguru — logging"),
]

def check_imports() -> List[Issue]:
    issues = []
    for pkg, desc in CRITICAL_IMPORTS:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            issues.append(Issue(
                severity="ERROR",
                check_id="IMPORT",
                file="<entorno>",
                line=0,
                message=f"Paquete '{pkg}' no instalado: {desc}"
            ))
    return issues


# =============================================================================
# CHECK 3: accesos dict["key"] en dicts locales conocidos
# =============================================================================
class LocalDictAccessVisitor(ast.NodeVisitor):
    """
    Detecta patrones como:
        out = {"a": 1, "b": 2}
        ...
        out["c"]   ← 'c' nunca fue insertada → KeyError

    Solo opera sobre dicts locales cuyos literales son visibles en el mismo scope.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Issue] = []
        # {varname: {key: lineno}} por función
        self._scope_stack: List[Dict[str, Set[str]]] = []

    def visit_FunctionDef(self, node):
        self._scope_stack.append({})
        self.generic_visit(node)
        self._scope_stack.pop()
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        """Captura: var = {"key1": ..., "key2": ...}"""
        if not self._scope_stack:
            self.generic_visit(node)
            return
        scope = self._scope_stack[-1]
        if (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Dict)):
            varname = node.targets[0].id
            known_keys = set()
            for k in node.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    known_keys.add(k.value)
            scope[varname] = known_keys
        self.generic_visit(node)

    def visit_Subscript(self, node):
        """Detecta: var["key"] donde 'key' no está en las claves del dict literal."""
        if not self._scope_stack:
            self.generic_visit(node)
            return
        scope = self._scope_stack[-1]
        if not isinstance(node.value, ast.Name):
            self.generic_visit(node)
            return
        varname = node.value.id
        if varname not in scope:
            self.generic_visit(node)
            return
        key = None
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            key = node.slice.value
        elif isinstance(node.slice, ast.Index):
            if isinstance(node.slice.value, ast.Constant) and isinstance(node.slice.value.value, str):
                key = node.slice.value.value
        if key is not None and key not in scope[varname]:
            self.issues.append(Issue(
                severity="WARN",
                check_id="LOCAL-DICT",
                file=self.filename,
                line=node.lineno,
                message=(
                    f"'{varname}[\"{key}\"]' accede a clave no definida en el dict literal. "
                    f"Claves conocidas: {sorted(scope[varname])}"
                )
            ))
        self.generic_visit(node)


# =============================================================================
# CAPA 2: Checks de entorno, artefactos y recursos
# =============================================================================

def check_stale_lock() -> List[Issue]:
    """
    [LOCK] Detecta .wfb_lock con PID muerto → run bloqueada para siempre.
    Bug exacto de la noche del 2026-05-15.
    """
    issues = []
    lock_path = ROOT / ".wfb_lock"
    if not lock_path.exists():
        return []
    try:
        import psutil
        raw = lock_path.read_text(encoding="utf-8").strip()
        # El lock puede contener solo PID o JSON con {"pid": N}
        if raw.startswith("{"):
            import json as _j
            pid = int(_j.loads(raw).get("pid", 0))
        else:
            pid = int(raw)
        if pid > 0 and not psutil.pid_exists(pid):
            issues.append(Issue(
                severity="ERROR",
                check_id="LOCK",
                file=".wfb_lock",
                line=0,
                message=(
                    f"Lock huérfano detectado: PID {pid} ya no existe. "
                    f"Elimina '.wfb_lock' antes de continuar → rm .wfb_lock"
                )
            ))
        elif pid > 0:
            issues.append(Issue(
                severity="WARN",
                check_id="LOCK",
                file=".wfb_lock",
                line=0,
                message=f"Lock activo: PID {pid} existe. Hay una run en curso."
            ))
    except Exception as e:
        issues.append(Issue(
            severity="WARN",
            check_id="LOCK",
            file=".wfb_lock",
            line=0,
            message=f"No se pudo leer .wfb_lock: {e}"
        ))
    return issues


def check_artifact_schema_consistency() -> List[Issue]:
    """
    [SCHEMA] Verifica que selected_features.json solo referencia columnas
    que existen en features_train.parquet. Lee solo el schema (metadatos),
    sin cargar ninguna fila de datos. ~30-50ms.
    """
    issues = []
    json_path    = ROOT / "data" / "features" / "selected_features.json"
    parquet_path = ROOT / "data" / "features" / "features_train.parquet"

    if not json_path.exists() or not parquet_path.exists():
        return []  # Artefactos aún no generados — no es error en primera run

    try:
        import json as _j
        import pyarrow.parquet as _pq

        selected_data = _j.loads(json_path.read_text(encoding="utf-8"))
        selected = selected_data.get("selected_features", []) + selected_data.get("pass_through_features", [])
        if not selected:
            return []

        schema_cols = set(_pq.read_schema(parquet_path).names)
        missing = [f for f in selected if f not in schema_cols]

        if missing:
            issues.append(Issue(
                severity="ERROR",
                check_id="SCHEMA",
                file="data/features/selected_features.json",
                line=0,
                message=(
                    f"{len(missing)} feature(s) en selected_features.json "
                    f"ausentes en features_train.parquet → KeyError garantizado en entrenamiento. "
                    f"Primeras: {missing[:5]}"
                )
            ))
        else:
            pass  # OK silencioso
    except Exception as e:
        issues.append(Issue(
            severity="WARN",
            check_id="SCHEMA",
            file="data/features/selected_features.json",
            line=0,
            message=f"No se pudo verificar consistencia de schema: {e}"
        ))
    return issues


def check_numeric_spot() -> List[Issue]:
    """
    [NUMERIC] Carga 500 filas de las columnas más críticas y verifica
    que no haya Inf/-Inf ni NaN totales. ~100-200ms.
    """
    issues = []
    parquet_path = ROOT / "data" / "features" / "features_train.parquet"
    if not parquet_path.exists():
        return []

    CRITICAL_COLS = ["close", "Target_TBM_Bin", "HMM_Regime"]

    try:
        import pandas as _pd
        import numpy as _np
        import pyarrow.parquet as _pq

        # Leer solo las columnas críticas que existan
        schema_cols = set(_pq.read_schema(parquet_path).names)
        cols_to_check = [c for c in CRITICAL_COLS if c in schema_cols]
        if not cols_to_check:
            return []

        # Leer las últimas 500 filas (las más recientes son las más propensas a issues)
        df_sample = _pd.read_parquet(parquet_path, columns=cols_to_check).tail(500)

        for col in cols_to_check:
            s = df_sample[col]
            # Inf/-Inf → ERROR: producen NaN silencioso en operaciones downstream
            n_inf = _np.isinf(s.replace([None], _np.nan).dropna()).sum()
            if n_inf > 0:
                issues.append(Issue(
                    severity="ERROR",
                    check_id="NUMERIC",
                    file="data/features/features_train.parquet",
                    line=0,
                    message=f"Columna '{col}': {n_inf} valores Inf/−Inf en las últimas 500 filas."
                ))
            # NaN total en columna crítica → ERROR
            nan_pct = s.isna().mean()
            if nan_pct > 0.95:
                if col == "HMM_Regime":
                    issues.append(Issue(
                        severity="WARN",
                        check_id="NUMERIC",
                        file="data/features/features_train.parquet",
                        line=0,
                        message=f"Columna '{col}': {nan_pct:.0%} NaN en las últimas 500 filas — [PERMITIDO POR DISEÑO HMM]"
                    ))
                else:
                    issues.append(Issue(
                        severity="ERROR",
                        check_id="NUMERIC",
                        file="data/features/features_train.parquet",
                        line=0,
                        message=f"Columna '{col}': {nan_pct:.0%} NaN en las últimas 500 filas — columna prácticamente vacía."
                    ))
    except Exception as e:
        issues.append(Issue(
            severity="WARN",
            check_id="NUMERIC",
            file="data/features/features_train.parquet",
            line=0,
            message=f"No se pudo ejecutar spot-check numérico: {e}"
        ))
    return issues


def check_schema_drift() -> List[Issue]:
    """
    [DRIFT] Compara el número de columnas del parquet actual con el
    fingerprint del run anterior (_fp_history.json). Si cambió >10%,
    algo estructural cambió y debe revisarse. ~30ms.
    """
    issues = []
    fp_path      = ROOT / "data" / "features" / "_fp_history.json"
    parquet_path = ROOT / "data" / "features" / "features_train.parquet"

    if not fp_path.exists() or not parquet_path.exists():
        return []  # Primera run — sin historial

    try:
        import json as _j
        import pyarrow.parquet as _pq

        prev = _j.loads(fp_path.read_text(encoding="utf-8"))
        prev_fp = prev.get("features_train_fp", "")
        if not prev_fp:
            return []

        # El fingerprint tiene formato "Nrows|fecha|fecha|Mcols"
        try:
            prev_cols = int(prev_fp.split("|")[-1].replace("cols", ""))
        except (ValueError, IndexError):
            return []

        curr_cols = len(_pq.read_schema(parquet_path).names)
        drift_pct = abs(curr_cols - prev_cols) / max(prev_cols, 1)

        if drift_pct > 0.10:
            issues.append(Issue(
                severity="WARN",
                check_id="DRIFT",
                file="data/features/features_train.parquet",
                line=0,
                message=(
                    f"Schema drift del {drift_pct:.0%}: "
                    f"run anterior={prev_cols} cols, actual={curr_cols} cols. "
                    f"Verifica que el Feature Pipeline no haya cambiado estructuralmente."
                )
            ))
    except Exception as e:
        issues.append(Issue(
            severity="WARN",
            check_id="DRIFT",
            file="data/features/_fp_history.json",
            line=0,
            message=f"No se pudo verificar schema drift: {e}"
        ))
    return issues


def check_resources() -> List[Issue]:
    """
    [RESOURCE] Verifica RAM libre y VRAM disponible antes de arrancar.
    Mínimos conservadores: 8GB RAM, 3GB VRAM (XGBoost GPU mode).
    """
    issues = []
    RAM_MIN_GB  = 5.0
    VRAM_MIN_GB = 3.0

    # RAM
    try:
        import psutil
        ram_free_gb = psutil.virtual_memory().available / 1e9
        print(f"[VALIDATOR-FIX-RAM] Free RAM available: {ram_free_gb:.2f} GB (threshold: {RAM_MIN_GB} GB)")
        if ram_free_gb < RAM_MIN_GB:
            issues.append(Issue(
                severity="WARN",
                check_id="RESOURCE",
                file="<sistema>",
                line=0,
                message=(
                    f"RAM libre insuficiente: {ram_free_gb:.1f}GB < {RAM_MIN_GB}GB mínimo recomendado. "
                    f"El pipeline puede experimentar OOM durante XGBoost/MetaLabeler."
                )
            ))
    except Exception as e:
        issues.append(Issue(severity="WARN", check_id="RESOURCE", file="<sistema>", line=0,
                            message=f"No se pudo verificar RAM: {e}"))

    # VRAM
    try:
        import torch
        if torch.cuda.is_available():
            vram_free_gb = torch.cuda.mem_get_info()[0] / 1e9
            if vram_free_gb < VRAM_MIN_GB:
                issues.append(Issue(
                    severity="WARN",
                    check_id="RESOURCE",
                    file="<GPU>",
                    line=0,
                    message=(
                        f"VRAM libre: {vram_free_gb:.1f}GB < {VRAM_MIN_GB}GB recomendado. "
                        f"XGBoost puede caer a CPU mode (más lento)."
                    )
                ))
    except Exception:
        pass  # torch no disponible o sin GPU — no es error bloqueante

    return issues

# =============================================================================
# ARCHIVOS CRÍTICOS A ANALIZAR
# =============================================================================
CRITICAL_FILES = [
    ROOT / "luna" / "features" / "feature_selection_e.py",
    ROOT / "luna" / "pipeline_executor.py",
    ROOT / "scripts" / "wfb_worker.py",
    ROOT / "luna" / "models" / "train_xgboost_v2.py",
    ROOT / "luna" / "models" / "train_metalabeler_v2.py",
    ROOT / "luna" / "models" / "ensemble_lgbm.py",
    ROOT / "luna" / "models" / "calibrate_probabilities.py",
]


# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================
def run_static_validation(
    paths: Optional[List[Path]] = None,
    fail_fast: bool = False,
    verbose: bool = False,
    skip_env: bool = False,
) -> ValidationResult:
    """
    skip_env=True: solo análisis AST (útil en CI sin datos reales).
    """
    t0 = time.perf_counter()
    result = ValidationResult()

    targets = paths or CRITICAL_FILES

    # ── CAPA 1: imports del entorno ──────────────────────────────────────────
    result.issues.extend(check_imports())

    # ── CAPA 1: análisis AST por archivo ─────────────────────────────────────
    for path in targets:
        if not path.exists():
            result.issues.append(Issue(
                severity="WARN",
                check_id="MISSING-FILE",
                file=str(path.relative_to(ROOT)),
                line=0,
                message="Archivo no encontrado — se saltó el análisis."
            ))
            continue

        result.files_checked += 1
        rel = str(path.relative_to(ROOT))

        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            result.issues.append(Issue(
                severity="ERROR",
                check_id="SYNTAX",
                file=rel,
                line=e.lineno or 0,
                message=f"Error de sintaxis: {e.msg}"
            ))
            if fail_fast:
                break
            continue

        dict_visitor = DictKeyConsistencyVisitor(rel)
        dict_visitor.visit(tree)
        result.issues.extend(dict_visitor.get_issues())

        local_visitor = LocalDictAccessVisitor(rel)
        local_visitor.visit(tree)
        result.issues.extend(local_visitor.issues)

        if fail_fast and result.errors:
            break

    # ── CAPA 2: entorno, artefactos y recursos ───────────────────────────────
    if not skip_env:
        print("[STATIC-VALIDATOR] Verificando entorno y artefactos...", flush=True)
        result.issues.extend(check_stale_lock())
        result.issues.extend(check_artifact_schema_consistency())
        result.issues.extend(check_numeric_spot())
        result.issues.extend(check_schema_drift())
        result.issues.extend(check_resources())

    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    return result


# =============================================================================
# REPORT
# =============================================================================
def print_report(result: ValidationResult) -> None:
    print(f"\n{C.BOLD}[STATIC-VALIDATOR] Luna V2 — {result.files_checked} archivos analizados{C.END}")
    print("=" * 70)

    if not result.issues:
        print(f"\n  {C.OK}{C.BOLD}✅ Sin issues detectados{C.END}")
    else:
        # Agrupar por archivo
        by_file: Dict[str, List[Issue]] = defaultdict(list)
        for issue in result.issues:
            by_file[issue.file].append(issue)

        for filepath, issues in by_file.items():
            print(f"\n  {C.BOLD}{filepath}{C.END}")
            for iss in sorted(issues, key=lambda x: x.line):
                color = C.FAIL if iss.severity == "ERROR" else C.WARN
                label = f"{color}[{iss.severity}]{C.END}"
                print(f"    {label} L{iss.line:4d}  [{iss.check_id}]  {iss.message}")

    # Resumen
    errors  = len(result.errors)
    warnings = len(result.warnings)
    print("\n" + "=" * 70)
    if errors == 0:
        status = f"{C.OK}{C.BOLD}ALL CLEAR — Pipeline puede arrancar{C.END}"
    else:
        status = f"{C.FAIL}{C.BOLD}BLOQUEADO — {errors} error(es) críticos detectados{C.END}"

    print(f"\n  {status}")
    warn_str = f" | {C.WARN}{warnings} WARN{C.END}" if warnings else ""
    print(f"  {errors} ERROR(S) / {warnings} WARN(S){warn_str}")
    print(f"  Tiempo: {result.elapsed_ms:.0f}ms\n")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Luna V2 — Static Code + Environment Validator"
    )
    parser.add_argument(
        "--path", nargs="*",
        help="Archivos específicos a analizar (por defecto: todos los críticos)"
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Detener al primer error"
    )
    parser.add_argument(
        "--no-env", action="store_true",
        help="Solo análisis AST, sin checks de entorno/artefactos"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.path] if args.path else None
    result = run_static_validation(
        paths=paths,
        fail_fast=args.fail_fast,
        verbose=args.verbose,
        skip_env=args.no_env,
    )
    print_report(result)
    sys.exit(0 if result.ok() else 1)
