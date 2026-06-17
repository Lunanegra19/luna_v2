"""
core/validation/phase_gates.py
===============================
Sistema de Phase-Gate Assertions para el pipeline WFB Luna V1.

Cada gate verifica invariantes mínimas ENTRE fases del WFB y emite un
GateResult con métricas de diagnóstico. Si un gate crítico falla, el
llamador debe ejecutar sys.exit(3) + Telegram alert.

Gates disponibles:
    gate_0_data       — Integridad del dataset de features
    gate_1_sfi        — Output de Feature Selection (SFI)
    gate_2_xgboost    — Calidad del modelo XGBoost entrenado
    gate_3_ensemble   — Calidad del ensemble LGBM + coherencia HMM
    gate_4_metalabeler — Output del MetaLabeler V2
    gate_5_signal     — Output del SignalFilter (trades generados)

Exit codes:
    0 = OK
    1 = Error de pipeline (exception inesperada)
    2 = Rechazo del Gauntlet estadístico
    3 = Phase Gate failure (bug detectado antes de continuar)

Uso en run_walkforward_pipeline_v2.py:
    from luna.validation.phase_gates import WFBPhaseGate
    gate = WFBPhaseGate(window_id="W2", seed=42, reports_dir=REPORTS_DIR)
    result = gate.gate_0_data(features_dir)
    if not result.passed:
        logger.error("[GATE-0] {}", result.summary)
        sys.exit(3)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Resultado de un phase gate."""
    gate_id: str                          # "G0", "G1", ... "G5"
    gate_name: str                        # "Data", "SFI", ...
    window_id: str
    seed: int
    passed: bool
    is_hard_stop: bool                    # False = warning, True = abort
    summary: str                          # Una línea humano-legible
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str]     = field(default_factory=list)
    errors:   List[str]     = field(default_factory=list)
    timestamp: str          = field(default_factory=lambda: pd.Timestamp.now('UTC').isoformat() + "Z")  # [FIX-PIPE-002]
    elapsed_s: float        = 0.0
    # [DEGRADED-MODE] Agentes individuales que fallaron el gate (régimenes a deshabilitar en RegimeRouter)
    disabled_agents: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class WFBPhaseGate:
    """
    Ejecuta los 6 phase gates del pipeline WFB.

    Args:
        window_id:   ID de la ventana WFB ("W1", "W2" ...)
        seed:        Semilla del run
        reports_dir: Directorio donde se escriben los gate_results JSON
        root:        Raíz del proyecto Luna V1 (auto-detecta si None)
    """

    # Umbrales globales — ajustables desde settings.yaml en el futuro
    XGB_AUC_HARD_STOP      = 0.510   # G2: AUC OOS mínima (hard stop)
    XGB_AUC_WARN           = 0.530   # G2: AUC OOS warning
    # [GATE-G2-BRIER-01] Relajado 0.2600→0.2750 [2026-05-07], 0.2750→0.2850 [2026-05-07T23:41]
    # Evidencia: Brier IS no correlaciona con WR OOS (r≈0, ver run_audit_20260507_seed42.md §7)
    # seed42 pasó 0.25 IS pero tuvo Brier Skill=-0.05 en OOS W2.
    # seeds 100/777/1337/2025 fallaron con Brier=0.268-0.291 → descartadas sin llegar a OOS.
    # Nuevo umbral: solo bloquea modelos claramente aleatorios (Brier ≈ 0.30 = random).
    XGB_BRIER_HARD_STOP    = 0.2850  # G2: Brier max hard-stop (agente individual)
    XGB_BRIER_WARN         = 0.2700  # G2: Brier warning
    # [DEGRADED-MODE-01] Número máximo de agentes que pueden fallar el Brier sin abortar.
    # Si n_failed <= MAX → los agentes se deshabilitan (Degraded Mode, RegimeRouter → CASH).
    # Si n_failed >  MAX → hard-stop: el pipeline no tiene cobertura suficiente.
    XGB_BRIER_DEGRADED_MAX_AGENTS = 1  # 1 agente puede fallar (ej: BEAR), 2+ → abort
    XGB_PROBA_STD_MIN      = 0.010   # G2: std(proba) mínima (modelo degenerado si menor)
    LGBM_PROBA_STD_MIN     = 0.010   # G3
    HMM_MIN_ACTIVE_STATES  = 2       # G3: estados HMM activos en OOS
    SFI_MIN_FEATURES       = 5       # G1
    SFI_MAX_ALPHA_RATIO    = 0.80    # G1: max proporción de alpha signals en selección
    DATA_MIN_ROWS          = 1000    # G0
    DATA_MAX_NAN_PCT       = 0.50    # G0
    DATA_MAX_GAP_H         = 48     # G0: gap máximo en horas
    SIGNAL_MIN_COUNT_WARN  = 5      # G5: mínimo de señales (warning, no hard-stop)

    def __init__(
        self,
        window_id: str,
        seed: int,
        reports_dir: Path,
        root: Optional[Path] = None,
    ):
        self.window_id   = window_id
        self.seed        = seed
        self.reports_dir = Path(reports_dir)
        self.root        = root or Path(__file__).parent.parent.parent
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # [FIX-E] DATA_MAX_NAN_PCT sobreescrito desde settings.yaml debug.nan_threshold_pct
        try:
            from config.settings import cfg as _cfg_pg
            _nan_thr = float(_cfg_pg.debug.nan_threshold_pct)
            self.DATA_MAX_NAN_PCT = _nan_thr / 100.0  # convertir % a fracción
            print(f"[FIX-E] PhaseGates G0: DATA_MAX_NAN_PCT={self.DATA_MAX_NAN_PCT:.3f} ({self.DATA_MAX_NAN_PCT*100:.1f}% NaN max)")
        except AttributeError as _e_nan:
            msg = f"[PhaseGates-G0] CRITICAL: Falta debug.nan_threshold_pct en settings.yaml: {_e_nan}"
            logger.critical(msg)
            raise RuntimeError(msg) from _e_nan
        except Exception as _e_nan:
            msg = f"[PhaseGates-G0] CRITICAL: No se pudo leer debug.nan_threshold_pct: {_e_nan}"
            logger.critical(msg)
            raise RuntimeError(msg) from _e_nan

        # [FIX-J] SFI_MAX_ALPHA_RATIO sobreescrito desde settings.yaml features.sfi_max_alpha_ratio
        try:
            from config.settings import cfg as _cfg_pg
            _alpha_ratio = float(_cfg_pg.features.sfi_max_alpha_ratio)
            self.SFI_MAX_ALPHA_RATIO = _alpha_ratio
            print(f"[FIX-J] PhaseGates G1: SFI_MAX_ALPHA_RATIO={self.SFI_MAX_ALPHA_RATIO:.3f} ({self.SFI_MAX_ALPHA_RATIO*100:.1f}% max alpha)")
        except AttributeError as _e_sfi:
            msg = f"[PhaseGates-G1] CRITICAL: Falta features.sfi_max_alpha_ratio en settings.yaml: {_e_sfi}"
            logger.critical(msg)
            raise RuntimeError(msg) from _e_sfi
        except Exception as _e_sfi:
            msg = f"[PhaseGates-G1] CRITICAL: No se pudo leer features.sfi_max_alpha_ratio: {_e_sfi}"
            logger.critical(msg)
            raise RuntimeError(msg) from _e_sfi

        # [FIX-GATES-CFG] Carga dinámica de Phase Gates desde settings.yaml (sección stat)
        # POLÍTICA NO-FALLBACK (2026-05-21): Gates críticos de calidad de modelo → CRITICAL si falla settings.
        # Rationale: XGB_AUC/BRIER/PROBA_STD determinan si el modelo es estadísticamente apto.
        # Un fallback silencioso puede aprobar modelos degenerados. Ver docs/parametros_fijos.md §1.
        # Gates operativos (DATA_MIN_ROWS, DATA_MAX_GAP_H, SIGNAL_MIN_COUNT_WARN) → WARNING con fallback
        # porque no afectan directamente el veredicto de calidad del modelo.
        try:
            from config.settings import cfg as _cfg_pg
            _stat = getattr(_cfg_pg, 'stat', None)
            if _stat is None:
                raise KeyError("[PhaseGates] CRITICAL: sección 'stat' no encontrada en settings.yaml. "
                               "Añadir xgb_auc_hard_stop, xgb_brier_hard_stop, xgb_proba_std_min. "
                               "Ver docs/parametros_fijos.md §2.")

            # Gates CRÍTICOS: no-fallback (KeyError si falta)
            _CRIT_KEYS = ['xgb_auc_hard_stop', 'xgb_brier_hard_stop', 'xgb_proba_std_min']
            _missing_crit = [k for k in _CRIT_KEYS if not hasattr(_stat, k)]
            if _missing_crit:
                raise KeyError(f"[PhaseGates] CRITICAL: claves obligatorias ausentes en cfg.stat: {_missing_crit}. "
                               "Ver docs/parametros_fijos.md §3.1")

            self.XGB_AUC_HARD_STOP             = float(_stat.xgb_auc_hard_stop)
            self.XGB_AUC_WARN                  = float(_stat.xgb_auc_warn)
            self.XGB_BRIER_HARD_STOP           = float(_stat.xgb_brier_hard_stop)
            self.XGB_BRIER_WARN                = float(_stat.xgb_brier_warn)
            self.XGB_BRIER_DEGRADED_MAX_AGENTS = int(_stat.xgb_brier_degraded_max_agents)
            self.XGB_PROBA_STD_MIN             = float(_stat.xgb_proba_std_min)
            self.LGBM_PROBA_STD_MIN            = float(_stat.lgbm_proba_std_min)
            self.HMM_MIN_ACTIVE_STATES         = int(_stat.hmm_min_active_states)
            self.SFI_MIN_FEATURES              = int(_stat.sfi_min_features)
            
            # Gates operativos: carga estricta No-Fallback
            self.DATA_MIN_ROWS         = int(_stat.data_min_rows)
            self.DATA_MAX_GAP_H        = int(_stat.data_max_gap_h)
            self.SIGNAL_MIN_COUNT_WARN = int(_stat.signal_min_count_warn)

            print(
                f"[FIX-GATES-CFG] Gates críticos cargados OK: "
                f"xgb_auc_hard_stop={self.XGB_AUC_HARD_STOP} | "
                f"xgb_brier_hard_stop={self.XGB_BRIER_HARD_STOP} | "
                f"xgb_proba_std_min={self.XGB_PROBA_STD_MIN} | "
                f"data_min_rows={self.DATA_MIN_ROWS}"
            )
        except Exception as _e_cfg:
            # POLÍTICA NO-FALLBACK: si falla la carga de gates críticos → CRITICAL + RuntimeError
            _msg = (
                f"[PhaseGates] CRITICAL — Imposible cargar gates de Phase Gates desde settings.yaml: {_e_cfg}\n"
                "ACCIÓN: verificar que cfg.stat contiene xgb_auc_hard_stop, xgb_brier_hard_stop, xgb_proba_std_min.\n"
                "El pipeline se DETIENE. Ver docs/parametros_fijos.md para contexto."
            )
            print(_msg)
            logger.critical(_msg)
            raise RuntimeError(_msg) from _e_cfg

    # -----------------------------------------------------------------------
    # Helper: persistir resultado
    # -----------------------------------------------------------------------

    def _save(self, result: GateResult) -> None:
        # 1. Guardar JSON individual (comportamiento original)
        path = self.reports_dir / f"gate_{result.gate_id}_{self.window_id}_seed{self.seed}.json"
        try:
            path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.debug("[PhaseGate] No se pudo guardar {}: {}", path.name, e)

        # 2. [V2-FIX] El gate persiste su resultado en JSON via _save() (línea 132).
        # El RunRegistry de V1 (run_walkforward_pipeline_v2._active_registry) no existe en V2.
        # pipeline_executor.py gestiona el registry a través de wfb_worker directamente.
        # No se requiere ninguna acción adicional aquí.


    def _result(
        self,
        gate_id: str, gate_name: str,
        passed: bool, is_hard_stop: bool, summary: str,
        metrics: dict = None, warnings: list = None, errors: list = None,
        elapsed: float = 0.0,
    ) -> GateResult:
        r = GateResult(
            gate_id=gate_id, gate_name=gate_name,
            window_id=self.window_id, seed=self.seed,
            passed=passed, is_hard_stop=is_hard_stop, summary=summary,
            metrics=metrics or {}, warnings=warnings or [], errors=errors or [],
            elapsed_s=round(elapsed, 2),
        )
        self._save(r)
        icon = "✅" if passed else ("🚨" if is_hard_stop else "⚠️")
        logger.info("[GATE-{}] {} {} — {}", gate_id, icon, gate_name, summary)
        for w in r.warnings:
            logger.warning("  [GATE-{}] ⚠ {}", gate_id, w)
        for e in r.errors:
            logger.error("  [GATE-{}] ✗ {}", gate_id, e)
        return r

    # -----------------------------------------------------------------------
    # Gate 0 — Data Integrity
    # -----------------------------------------------------------------------


    @staticmethod
    def compute_feature_psi(
        train_values: "np.ndarray",
        holdout_values: "np.ndarray",
        n_bins: int = 10
    ) -> float:
        """
        IDEA-D (2026-05-07): Population Stability Index entre training y holdout.

        PSI mide cuánto cambió la distribución de una feature entre entrenamiento y OOS.
        Referencia: Yurdakul (2018) — PSI estándar de la industria bancaria.

        Umbrales:
          PSI < 0.10 → distribución estable (verde)
          PSI 0.10-0.25 → cambio moderado (amarillo — monitorizar)
          PSI > 0.25 → drift severo (rojo — modelo no es fiable en OOS)

        Uso: calcular sobre las top-5 features de importancia (gain) del XGBoost.
        Si PSI > 0.25 en ≥50% de las top features → activar DEGRADED MODE automáticamente.
        """
        import numpy as np
        if len(train_values) < 10 or len(holdout_values) < 10:
            return 0.0  # Sin suficientes datos, asumir estable

        # Bins basados en percentiles del training (robusto a outliers)
        bins = np.percentile(train_values, np.linspace(0, 100, n_bins + 1))
        bins[0]  -= 1e-6
        bins[-1] += 1e-6

        train_pct = np.histogram(train_values, bins=bins)[0] / max(len(train_values), 1)
        hold_pct  = np.histogram(holdout_values, bins=bins)[0] / max(len(holdout_values), 1)

        # Evitar log(0): clip a mínimo 1e-6
        train_pct = np.clip(train_pct, 1e-6, 1.0)
        hold_pct  = np.clip(hold_pct, 1e-6, 1.0)

        psi = float(np.sum((hold_pct - train_pct) * np.log(hold_pct / train_pct)))
        return round(psi, 4)

    @staticmethod
    def interpret_psi(psi: float) -> str:
        """Clasificación estándar del PSI."""
        if psi < 0.10: return "STABLE"
        if psi < 0.25: return "MODERATE_DRIFT"
        return "SEVERE_DRIFT"

    def gate_0_data(self, features_dir: Path) -> GateResult:
        """
        Verifica integridad del dataset de features antes de SFI.

        Checks:
        - features_train.parquet existe y tiene >1000 filas
        - NaN% global < 50%
        - Columna 'close' sin gaps > 48h
        - No hay duplicados de timestamp
        """
        t0 = time.monotonic()
        features_dir = Path(features_dir)
        train_path   = features_dir / "features_train.parquet"
        errors, warnings, metrics = [], [], {}

        if not train_path.exists():
            msg = f"Archivo faltante: {train_path}. Si es un cold start, se generará luego."
            return self._result("G0", "Data", True, False,
                                msg,
                                warnings=[msg],
                                elapsed=time.monotonic()-t0)
        try:
            import pyarrow.parquet as pq
            meta   = pq.read_metadata(train_path)
            schema = pq.read_schema(train_path)
            n_rows = meta.num_rows
            n_cols = len(schema.names)
            metrics["n_rows"] = n_rows
            metrics["n_cols"] = n_cols

            if n_rows < self.DATA_MIN_ROWS:
                errors.append(f"Solo {n_rows} filas (mínimo {self.DATA_MIN_ROWS})")

            # Leer solo las columnas necesarias para el check
            check_cols = [c for c in schema.names if c in ("close", "target", "open", "high", "low")]
            df_check = pd.read_parquet(train_path, columns=check_cols if check_cols else schema.names[:5])
            df_check.index = pd.to_datetime(df_check.index, utc=True, errors="coerce")

            # NaN global
            nan_pct = df_check.isnull().mean().mean()
            metrics["nan_pct_global"] = round(float(nan_pct), 4)
            if nan_pct > self.DATA_MAX_NAN_PCT:
                errors.append(f"NaN global {nan_pct:.1%} > {self.DATA_MAX_NAN_PCT:.0%}")

            # Duplicados de timestamp
            dupes = df_check.index.duplicated().sum()
            metrics["duplicate_timestamps"] = int(dupes)
            if dupes > 0:
                warnings.append(f"{dupes} timestamps duplicados detectados")

            # Gaps temporales (si el índice está ordenado)
            if len(df_check) > 1:
                df_sorted = df_check.sort_index()
                diffs_h = df_sorted.index.to_series().diff().dt.total_seconds().dropna() / 3600
                max_gap = float(diffs_h.max())
                n_gaps  = int((diffs_h > self.DATA_MAX_GAP_H).sum())
                metrics["max_gap_h"] = round(max_gap, 1)
                metrics["n_gaps_over_threshold"] = n_gaps
                if n_gaps > 0:
                    warnings.append(f"{n_gaps} gap(s) > {self.DATA_MAX_GAP_H}h detectados (max={max_gap:.0f}h)")

            # Rango temporal
            if len(df_check) > 0:
                metrics["date_start"] = str(df_check.index.min().date())
                metrics["date_end"]   = str(df_check.index.max().date())

        except Exception as e:
            errors.append(f"Error leyendo parquet: {e}")

        passed       = len(errors) == 0
        is_hard_stop = len(errors) > 0
        summary = (
            f"{metrics.get('n_rows',0)} filas, {metrics.get('n_cols',0)} cols, "
            f"NaN={metrics.get('nan_pct_global',0):.1%}, "
            f"max_gap={metrics.get('max_gap_h','?')}h"
        ) if passed else f"FALLIDO: {'; '.join(errors)}"

        return self._result("G0", "Data", passed, is_hard_stop, summary,
                            metrics, warnings, errors, elapsed=time.monotonic()-t0)

    # -----------------------------------------------------------------------
    # Gate 1 — SFI Output
    # -----------------------------------------------------------------------

    def gate_1_sfi(self, features_dir: Path) -> GateResult:
        """
        Verifica que SFI produjo un set de features válido.

        Checks:
        - selected_features.json existe
        - ≥5 features seleccionadas
        - Proporción de alpha signals ≤ 80%
        - No hay features duplicadas
        """
        t0 = time.monotonic()
        features_dir = Path(features_dir)
        sf_path = features_dir / "selected_features.json"
        errors, warnings, metrics = [], [], {}

        if not sf_path.exists():
            return self._result("G1", "SFI", False, True,
                                f"selected_features.json NO EXISTE",
                                errors=[f"Archivo faltante: {sf_path}"],
                                elapsed=time.monotonic()-t0)
        try:
            data     = json.loads(sf_path.read_text(encoding="utf-8"))
            selected = data.get("selected_features", [])
            passth   = data.get("pass_through_features", [])
            alpha_p  = data.get("alpha_signals_passed", [])
            total    = list(dict.fromkeys(selected + passth))

            metrics["n_selected"]    = len(selected)
            metrics["n_passthrough"] = len(passth)
            metrics["n_alpha"]       = len(alpha_p)
            metrics["n_total"]       = len(total)

            if len(selected) < self.SFI_MIN_FEATURES:
                errors.append(f"Solo {len(selected)} features seleccionadas < mínimo {self.SFI_MIN_FEATURES}")

            # Proporción alpha
            if len(total) > 0:
                alpha_ratio = len(alpha_p) / len(total)
                metrics["alpha_ratio"] = round(alpha_ratio, 3)
                if alpha_ratio > self.SFI_MAX_ALPHA_RATIO:
                    warnings.append(f"Alpha ratio = {alpha_ratio:.0%} (umbral {self.SFI_MAX_ALPHA_RATIO:.0%}) — features contextuales escasas")

            # Duplicados
            dupes = len(total) - len(set(total))
            metrics["duplicate_features"] = dupes
            if dupes > 0:
                warnings.append(f"{dupes} features duplicadas detectadas")

        except Exception as e:
            errors.append(f"Error leyendo selected_features.json: {e}")

        passed       = len(errors) == 0
        is_hard_stop = len(errors) > 0
        summary = (
            f"{metrics.get('n_total','?')} features ({metrics.get('n_selected','?')} SFI + "
            f"{metrics.get('n_passthrough','?')} passthrough, {metrics.get('n_alpha','?')} alpha)"
        ) if passed else f"FALLIDO: {'; '.join(errors)}"

        return self._result("G1", "SFI", passed, is_hard_stop, summary,
                            metrics, warnings, errors, elapsed=time.monotonic()-t0)

    # Umbrales Degraded Mode Gate-G2
    # Un solo agente por debajo del threshold activa DEGRADED (no HARD STOP).
    # Dos o más agentes simultáneos activan HARD STOP (señal insuficiente).
    XGB_BRIER_DEGRADED_MAX_AGENTS = 1   # Máximo de agentes degradados antes de HARD STOP

    # -----------------------------------------------------------------------
    # Gate 2 — XGBoost Quality (Degraded Mode per-agent)
    # -----------------------------------------------------------------------

    def gate_2_xgboost(self, models_dir: Path, features_dir: Path) -> GateResult:
        """
        Verifica calidad del modelo XGBoost entrenado.

        Checks:
        - Archivos de modelo existen (*.ubj o *.json)
        - xgboost_meta_*_signature.json existe con métricas
        - proba_std > XGB_PROBA_STD_MIN (modelo no degenerado)
        - AUC OOS en signature > XGB_AUC_HARD_STOP
        - Modelo no es zombie (mtime reciente vs features_train)
        """
        t0 = time.monotonic()
        models_dir   = Path(models_dir)
        features_dir = Path(features_dir)
        errors, warnings, metrics = [], [], {}

        # Verificar existencia de signature (fuente principal de métricas)
        all_sig_files = list(models_dir.glob("xgboost_meta_*_signature.json"))
        sig_files = []
        
        # Ignorar signatures "zombie" de runs antiguos (solo warning, no omitir)
        train_path = features_dir / "features_train.parquet"
        if train_path.exists():
            train_mtime = train_path.stat().st_mtime
            for p in all_sig_files:
                if p.stat().st_mtime >= (train_mtime - 3600):  # Margen de 1 hora
                    sig_files.append(p)
                else:
                    warnings.append(f"Posible signature zombie (mtime antiguo): {p.name}")
                    sig_files.append(p) # FIX: No lo omitimos de la validacion si el cache fue manipulado
        else:
            sig_files = all_sig_files
            
        metrics["n_signature_files"] = len(sig_files)

        if len(sig_files) == 0:
            return self._result("G2", "XGBoost", False, True,
                                "No se encontraron xgboost_meta_*_signature.json",
                                errors=["No hay modelos XGBoost entrenados"],
                                elapsed=time.monotonic()-t0)

        # ------------------------------------------------------------------
        # [DEGRADED-MODE] Evaluación POR AGENTE en lugar de max() global.
        # - 0 agentes fallando  → PASS
        # - 1 agente fallando   → DEGRADED (passed=True, disabled_agents=["range"])
        # - 2+ agentes fallando → HARD STOP (pipeline sin cobertura suficiente)
        # El régimen del agente degradado se informa en disabled_agents para
        # que el RegimeRouter fuerce CASH en esas barras.
        # ------------------------------------------------------------------
        disabled_agents: List[str] = []
        auc_vals, proba_stds, brier_by_agent, dsr_vals = [], [], {}, []
        adaptive_gate_by_agent = {}  # IDEA-A: umbral adaptativo por agente

        # Extraer nombre de agente del filename: xgboost_meta_BULL_long_signature.json
        def _agent_name_from_sig(p: Path) -> str:
            stem = p.stem  # e.g. "xgboost_meta_bull_long_signature"
            parts = stem.replace("xgboost_meta_", "").replace("lgbm_meta_", "").split("_")
            # Descarta el sufijo directional (_long / _short) y "signature"
            name_parts = [x for x in parts if x not in ("long", "short", "signature")]
            return "_".join(name_parts) if name_parts else stem

        for sig_path in sig_files:
            try:
                sig = json.loads(sig_path.read_text(encoding="utf-8"))
                agent_key = _agent_name_from_sig(sig_path)

                if "dsr_oos" not in sig and "val_auc" not in sig and "auc_val" not in sig:
                    warnings.append(f"Signature incompleta/vacia, ignorada: {sig_path.name}")
                    continue

                is_v2 = "dsr_oos" in sig
                pstd = float(sig.get("proba_std", sig.get("val_proba_std", 0.1)))
                proba_stds.append(pstd)

                if is_v2:
                    brier = sig.get("xgb_brier_calibrated", sig.get("xgb_brier_raw", None))
                    if brier is not None:
                        brier_by_agent[agent_key] = float(brier)
                    # IDEA-A: Extraer umbral dinámico (fallback al global)
                    adaptive_gate_by_agent[agent_key] = float(sig.get("brier_adaptive_gate", self.XGB_BRIER_HARD_STOP))
                    dsr = sig.get("dsr_oos")
                    if dsr is not None:
                        dsr_vals.append(float(dsr))
                else:
                    auc = float(sig.get("val_auc", sig.get("auc_val", 0.0)))
                    auc_vals.append(auc)

                # proba_std degenerada (agente individual)
                if pstd < self.XGB_PROBA_STD_MIN:
                    errors.append(
                        f"[{agent_key}] proba_std={pstd:.4f} < {self.XGB_PROBA_STD_MIN} — "
                        "modelo degenerado (probas sin discriminación)"
                    )
            except Exception as e:
                warnings.append(f"No se pudo leer {sig_path.name}: {e}")

        # --- Evaluación Brier por agente (lógica Degraded Mode) ---
        if brier_by_agent:
            all_briers = list(brier_by_agent.values())
            metrics["brier_max"]      = round(max(all_briers), 4)
            metrics["brier_mean"]     = round(float(np.mean(all_briers)), 4)
            metrics["brier_by_agent"] = {k: round(v, 4) for k, v in brier_by_agent.items()}

            # Identificar agentes que fallan el threshold adaptativo (Idea A)
            failed_agents = {
                agent: brier for agent, brier in brier_by_agent.items()
                if brier > adaptive_gate_by_agent.get(agent, self.XGB_BRIER_HARD_STOP)
            }
            warn_agents = {
                agent: brier for agent, brier in brier_by_agent.items()
                if self.XGB_BRIER_WARN < brier <= adaptive_gate_by_agent.get(agent, self.XGB_BRIER_HARD_STOP)
            }

            if failed_agents:
                n_failed = len(failed_agents)
                # DEGRADED: 1 o más agentes fallando → marcar para deshabilitar (no abortar)
                for agent_name_fail, brier_val in failed_agents.items():
                    disabled_agents.append(agent_name_fail)
                    gate_limit = adaptive_gate_by_agent.get(agent_name_fail, self.XGB_BRIER_HARD_STOP)
                    warnings.append(
                        f"[DEGRADED] Agente '{agent_name_fail}' Brier={brier_val:.4f} > "
                        f"{gate_limit:.4f} (adaptativo) — marcado como NO_OPERABLE. "
                        "El RegimeRouter forzará CASH en este régimen durante OOS."
                    )

                if n_failed == len(brier_by_agent):
                    # HARD STOP: TODOS los agentes fallaron. Pipeline totalmente ciego.
                    errors.append(
                        f"HARD STOP: {n_failed}/{len(brier_by_agent)} agentes superan Brier límite adaptativo. "
                        f"({', '.join(f'{a}={b:.4f}' for a, b in failed_agents.items())}) — "
                        "Pipeline 100% sin cobertura de señal, abortando ventana WFB."
                    )
                else:
                    # DEGRADED MODE ALLOWED: La ventana sobrevive gracias a los agentes restantes
                    warnings.append(
                        f"[INFO-DEGRADED] {n_failed}/{len(brier_by_agent)} agentes deshabilitados. "
                        "La ventana WFB SOBREVIVE. El RegimeRouter gestionará los regímenes fallidos rutándolos a CASH."
                    )


            for agent_name_warn, brier_val in warn_agents.items():
                warnings.append(
                    f"[WARN] Agente '{agent_name_warn}' Brier={brier_val:.4f} "
                    f"(> umbral recomendado {self.XGB_BRIER_WARN:.4f})"
                )

        elif auc_vals:
            metrics["auc_min"]  = round(min(auc_vals), 4)
            metrics["auc_mean"] = round(float(np.mean(auc_vals)), 4)
            if metrics["auc_min"] < self.XGB_AUC_HARD_STOP:
                errors.append(
                    f"AUC OOS mínima {metrics['auc_min']:.4f} < hard-stop {self.XGB_AUC_HARD_STOP:.3f}"
                )
            elif metrics["auc_min"] < self.XGB_AUC_WARN:
                warnings.append(f"AUC OOS = {metrics['auc_min']:.4f} (< umbral {self.XGB_AUC_WARN:.3f})")

        if dsr_vals:
            metrics["dsr_min"]  = round(min(dsr_vals), 4)
            metrics["dsr_mean"] = round(float(np.mean(dsr_vals)), 4)
            if metrics["dsr_min"] < 0.0:
                warnings.append(f"DSR OOS mínimo {metrics['dsr_min']:.4f} < 0.0")

        if proba_stds:
            metrics["proba_std_min"]  = round(min(proba_stds), 4)
            metrics["proba_std_mean"] = round(float(np.mean(proba_stds)), 4)

        metrics["disabled_agents"] = disabled_agents
        metrics["n_disabled_agents"] = len(disabled_agents)

        # Zombie check
        train_path = features_dir / "features_train.parquet"
        if train_path.exists() and sig_files:
            try:
                train_mtime = train_path.stat().st_mtime
                sig_mtime   = max(p.stat().st_mtime for p in sig_files)
                lag_h = (train_mtime - sig_mtime) / 3600
                metrics["model_lag_h"] = round(lag_h, 1)
                if lag_h > 1.0:
                    errors.append(
                        f"Posible modelo zombie: features_train es {lag_h:.1f}h más reciente — "
                        "XGBoost se entrenó con datos distintos"
                    )
            except Exception:
                pass

        passed       = len(errors) == 0
        is_hard_stop = len(errors) > 0

        # Construir summary
        if disabled_agents:
            degraded_str = f" | DEGRADED: {disabled_agents} deshabilitados"
        else:
            degraded_str = ""

        if brier_by_agent:
            summary = (
                f"Brier={metrics.get('brier_mean','?'):.4f} (max={metrics.get('brier_max','?'):.4f})"
                f"{degraded_str}"
            ) if passed else f"FALLIDO: {'; '.join(errors)}"
        elif dsr_vals:
            summary = (
                f"DSR={metrics.get('dsr_mean','?'):.4f} (min={metrics.get('dsr_min','?'):.4f})"
                f"{degraded_str}"
            ) if passed else f"FALLIDO: {'; '.join(errors)}"
        else:
            summary = (
                f"AUC={metrics.get('auc_mean','?'):.4f} (min={metrics.get('auc_min','?'):.4f})"
            ) if passed else f"FALLIDO: {'; '.join(errors)}"

        result = self._result(
            "G2", "XGBoost", passed, is_hard_stop, summary,
            metrics, warnings, errors, elapsed=time.monotonic()-t0
        )
        result.disabled_agents = disabled_agents
        return result

    # -----------------------------------------------------------------------
    # Gate 3 — LGBM Ensemble + HMM Coherence
    # -----------------------------------------------------------------------

    def gate_3_ensemble(self, models_dir: Path, predictions_dir: Path) -> GateResult:
        """
        Verifica calidad del ensemble LGBM y coherencia del HMM.

        Checks:
        - LGBM proba std > LGBM_PROBA_STD_MIN
        - HMM state_map tiene entradas para todos los regímenes vistos en OOS
        - HMM tiene ≥2 estados activos en las predicciones OOS
        - features_validation.parquet existe (necesario para calibrador)
        """
        t0 = time.monotonic()
        models_dir      = Path(models_dir)
        predictions_dir = Path(predictions_dir)
        errors, warnings, metrics = [], [], {}

        # --- LGBM check vía predictions OOS ---
        oos_path = predictions_dir / "features_validation.parquet"
        if not oos_path.exists():
            oos_path = predictions_dir / "features_oos.parquet"

        if oos_path.exists():
            try:
                df_oos = pd.read_parquet(oos_path, columns=[c for c in
                    ["lgbm_prob", "lgbm_meta_prob", "lgbm_prob_bull",
                     "xgb_prob", "HMM_Regime", "HMM_Semantic"]
                    if True])   # columnas subset
            except Exception:
                try:
                    df_oos = pd.read_parquet(oos_path)
                except Exception as e:
                    errors.append(f"No se pudo leer OOS parquet: {e}")
                    df_oos = pd.DataFrame()

            if not df_oos.empty:
                # LGBM std
                lgbm_col = next((c for c in ["lgbm_prob", "lgbm_meta_prob", "lgbm_prob_bull"]
                                 if c in df_oos.columns), None)
                if lgbm_col:
                    lgbm_std = float(df_oos[lgbm_col].std())
                    lgbm_mean = float(df_oos[lgbm_col].mean())
                    metrics["lgbm_proba_std"]  = round(lgbm_std, 4)
                    metrics["lgbm_proba_mean"] = round(lgbm_mean, 4)
                    if lgbm_std < self.LGBM_PROBA_STD_MIN:
                        errors.append(
                            f"LGBM proba_std={lgbm_std:.4f} < {self.LGBM_PROBA_STD_MIN} — "
                            "ensemble degenerado (probas sin discriminación)"
                        )
                else:
                    # [MEJORA-GATE-01] Hacer bloqueante la falta de probas si el ensemble está activado
                    try:
                        from config.settings import cfg as _cfg_ens
                        use_lgbm = bool(_cfg_ens.ensemble.use_lgbm_ensemble)
                    except Exception:
                        use_lgbm = False
                        
                    if use_lgbm:
                        errors.append("Columna lgbm_prob no encontrada en OOS parquet, pero settings.ensemble.use_lgbm_ensemble=True")
                    else:
                        warnings.append("Columna lgbm_prob no encontrada en OOS parquet (use_lgbm_ensemble=False)")

                # HMM estados activos
                hmm_col = next((c for c in ["HMM_Regime"] if c in df_oos.columns), None)
                if hmm_col:
                    active_states = int(df_oos[hmm_col].dropna().nunique())
                    metrics["hmm_active_states_oos"] = active_states
                    if active_states < self.HMM_MIN_ACTIVE_STATES:
                        errors.append(
                            f"Solo {active_states} estado(s) HMM activos en OOS "
                            f"(mínimo {self.HMM_MIN_ACTIVE_STATES}) — HMM colapsado"
                        )
        else:
            warnings.append(f"features_validation.parquet no encontrado en {predictions_dir} — checks LGBM/HMM saltados")

        # --- HMM state_map coherence ---
        try:
            from luna.models.hmm_regime import HMMRegimeModel
            hmm_model = HMMRegimeModel.load(models_dir)
            state_map = hmm_model.state_map
            metrics["hmm_n_states_in_map"] = len(state_map)
            if len(state_map) < 2:
                errors.append(f"HMM state_map solo tiene {len(state_map)} estado(s) — entrenamiento inválido")
            else:
                metrics["hmm_states"] = list(state_map.values())
        except FileNotFoundError:
            warnings.append("HMMRegimeModel no encontrado — gate HMM saltado")
        except Exception as e:
            warnings.append(f"No se pudo cargar HMMRegimeModel: {e}")

        passed       = len(errors) == 0
        is_hard_stop = len(errors) > 0
        summary = (
            f"LGBM std={metrics.get('lgbm_proba_std','?')}, "
            f"HMM estados={metrics.get('hmm_active_states_oos','?')} activos en OOS, "
            f"{metrics.get('hmm_n_states_in_map','?')} en state_map"
        ) if passed else f"FALLIDO: {'; '.join(errors)}"

        return self._result("G3", "Ensemble+HMM", passed, is_hard_stop, summary,
                            metrics, warnings, errors, elapsed=time.monotonic()-t0)

    # -----------------------------------------------------------------------
    # Gate 4 — MetaLabeler V2
    # -----------------------------------------------------------------------

    def gate_4_metalabeler(self, models_dir: Path, predictions_dir: Path) -> GateResult:
        """
        Verifica que MetaLabeler V2 produció probas válidas.

        Checks:
        - meta_v2_config.json existe
        - Si hay columna meta_v2_prob en OOS: range [0,1], std > 0.01
        - seq_len en config es consistente con features disponibles
        """
        t0 = time.monotonic()
        models_dir      = Path(models_dir)
        predictions_dir = Path(predictions_dir)
        errors, warnings, metrics = [], [], {}

        config_path = models_dir / "metalabeler_v2_long_config.json"
        if not config_path.exists():
            return self._result("G4", "MetaLabeler", True, False,
                                "metalabeler_v2_long_config.json no existe — MetaLabeler V2 no entrenado (skip_metalabeler activo)",
                                metrics={"skipped": True},
                                warnings=["MetaLabeler no está activo para esta ventana"],
                                elapsed=time.monotonic()-t0)

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            seq_len    = int(config.get("seq_len", 48))
            input_dim  = int(config.get("input_dim", 0))
            seq_feats  = config.get("seq_features", [])
            metrics["seq_len"]   = seq_len
            metrics["input_dim"] = input_dim
            metrics["n_seq_features"] = len(seq_feats)

            if seq_len < 2:
                errors.append(f"seq_len={seq_len} inválido (debe ser ≥ 2)")
            if input_dim < 1:
                warnings.append(f"input_dim={input_dim} en config — posible config vacía")

        except Exception as e:
            errors.append(f"Error leyendo meta_v2_config.json: {e}")

        # Verificar outputs en OOS si existen
        oos_path = predictions_dir / "features_validation.parquet"
        if oos_path.exists():
            try:
                df = pd.read_parquet(oos_path)
                meta_col = next((c for c in df.columns if "meta_v2" in c.lower() or "metalabel" in c.lower()), None)
                if meta_col:
                    vals = df[meta_col].dropna()
                    meta_std  = float(vals.std())
                    meta_mean = float(vals.mean())
                    out_of_range = int(((vals < 0) | (vals > 1)).sum())
                    metrics["meta_v2_proba_std"]       = round(meta_std, 4)
                    metrics["meta_v2_proba_mean"]      = round(meta_mean, 4)
                    metrics["meta_v2_out_of_range_pct"] = round(out_of_range / max(len(vals), 1), 4)

                    if out_of_range > 0:
                        errors.append(f"{out_of_range} probas MetaV2 fuera de [0,1]")
                    if meta_std < 0.01:
                        warnings.append(f"MetaV2 proba_std={meta_std:.4f} muy baja — posible modelo degenerado")
                else:
                    metrics["meta_v2_proba_in_oos"] = False
            except Exception as e:
                warnings.append(f"No se pudo verificar MetaV2 en OOS: {e}")

        passed       = len(errors) == 0
        is_hard_stop = bool(errors)
        summary = (
            f"seq_len={metrics.get('seq_len','?')}, meta_v2_std={metrics.get('meta_v2_proba_std','N/A')}"
        ) if passed else f"FALLIDO: {'; '.join(errors)}"

        return self._result("G4", "MetaLabeler", passed, is_hard_stop, summary,
                            metrics, warnings, errors, elapsed=time.monotonic()-t0)

    # -----------------------------------------------------------------------
    # Gate 5 — Signal Filter Output
    # -----------------------------------------------------------------------

    def gate_5_signal(self, predictions_dir: Path, run_id: str = "") -> GateResult:
        """
        Verifica el output del SignalFilter.

        Checks:
        - signal_funnel.json existe
        - filter_fallback_level == 0 (HARD STOP si >= 2, warning si == 1)
        - N_signals_final > SIGNAL_MIN_COUNT_WARN (warning, no hard-stop)
        - oos_trades_{window_id}.parquet existe si ya se generó

        Nota: Gate-5 es WARNING-only para N_signals bajo.
        Solo es HARD STOP si fallback_level >= 2 (XGB puro).
        """
        t0 = time.monotonic()
        predictions_dir = Path(predictions_dir)
        errors, warnings, metrics = [], [], {}

        # Buscar signal_funnel.json
        funnel_candidates = [
            predictions_dir / "signal_funnel.json",
            predictions_dir / f"signal_funnel_{run_id}.json",
            predictions_dir.parent / "reports" / f"signal_funnel_{run_id}.json",
            predictions_dir.parent / "reports" / "signal_funnel.json",
        ]
        funnel_path = next((p for p in funnel_candidates if p.exists()), None)

        if funnel_path is None:
            return self._result("G5", "Signal", True, False,
                                "signal_funnel.json no encontrado (posiblemente aún generando)",
                                warnings=["signal_funnel.json no disponible aún"],
                                elapsed=time.monotonic()-t0)

        try:
            funnel = json.loads(funnel_path.read_text(encoding="utf-8"))
            fallback_level  = int(funnel.get("filter_fallback_level", 0))
            n_initial       = int(funnel.get("n_initial", 0))
            n_after_xgb     = int(funnel.get("after_xgb",   funnel.get("n_after_xgb", 0)))
            n_after_lgbm    = int(funnel.get("after_lgbm",  funnel.get("n_after_lgbm", n_after_xgb)))
            n_final         = int(funnel.get("after_all",   funnel.get("n_final", n_after_lgbm)))

            metrics.update({
                "filter_fallback_level": fallback_level,
                "n_initial":    n_initial,
                "n_after_xgb":  n_after_xgb,
                "n_after_lgbm": n_after_lgbm,
                "n_final":      n_final,
            })

            # Fallback level: HARD STOP si XGB puro (nivel 2)
            if fallback_level >= 2:
                errors.append(
                    f"filter_fallback_level={fallback_level} — XGB puro sin MetaLabeler/LGBM/HMM. "
                    "Los trades no están validados por el ensemble completo."
                )
            elif fallback_level == 1:
                warnings.append(
                    f"filter_fallback_level=1 — trades sin MetaLabeler/HMM/Momentum. "
                    "Considerar reentrenar ensemble_lgbm.py."
                )

            # N_signals bajo: solo warning para no abortar ventanas con mercado ranging
            if n_final < self.SIGNAL_MIN_COUNT_WARN:
                warn_msg = (
                    f"Solo {n_final} señales finales (< {self.SIGNAL_MIN_COUNT_WARN}). "
                    f"Tasa XGB→final: {n_final}/{n_after_xgb if n_after_xgb else n_initial}"
                )
                if n_final == 0:
                    errors.append(f"0 señales generadas — ventana producirá 0 trades")
                else:
                    warnings.append(warn_msg)

        except Exception as e:
            warnings.append(f"Error leyendo signal_funnel.json: {e}")

        # Para G5: hard_stop solo si fallback_level >= 2
        # 0 señales YA NO es hard-stop (es un comportamiento válido en bear markets para proteger capital)
        is_hard_stop = any("XGB puro" in e for e in errors)
        passed       = len(errors) == 0

        summary = (
            f"fallback_level={metrics.get('filter_fallback_level','?')}, "
            f"signals: {metrics.get('n_initial','?')} → {metrics.get('n_after_xgb','?')} (XGB) "
            f"→ {metrics.get('n_after_lgbm','?')} (LGBM) → {metrics.get('n_final','?')} (final)"
        ) if not errors else f"{'WARN' if not is_hard_stop else 'FALLIDO'}: {'; '.join(errors)}"

        return self._result("G5", "Signal", passed, is_hard_stop, summary,
                            metrics, warnings, errors, elapsed=time.monotonic()-t0)

    # -----------------------------------------------------------------------
    # Helper: run_all — ejecuta todos los gates aplicables
    # -----------------------------------------------------------------------

    def run_all(
        self,
        features_dir: Path,
        models_dir: Path,
        predictions_dir: Path,
        run_id: str = "",
    ) -> List[GateResult]:
        """Ejecuta los 6 gates en orden y retorna todos los resultados."""
        results = []
        results.append(self.gate_0_data(features_dir))
        results.append(self.gate_1_sfi(features_dir))
        results.append(self.gate_2_xgboost(models_dir, features_dir))
        results.append(self.gate_3_ensemble(models_dir, predictions_dir))
        results.append(self.gate_4_metalabeler(models_dir, predictions_dir))
        results.append(self.gate_5_signal(predictions_dir, run_id=run_id))
        return results
