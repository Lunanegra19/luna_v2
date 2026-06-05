"""
core/validation/run_registry.py
================================
Registro institucional centralizado para runs WFB de Luna V1.

Implementa un ledger JSONL append-only (una línea = un evento) que actúa
como única fuente de verdad para cada run. Permite:
  - Consultar qué ocurrió en qué orden y cuánto tiempo tomó cada fase
  - Comparar métricas entre runs (AUC, LGBM std, señales, etc.)
  - Exportar reportes HTML con gráficos Chart.js sin dependencias extra
  - Diagnóstico post-mortem en segundos en lugar de buscar en 10 archivos

Formato de evento (una línea JSON en ledger_{run_id}.jsonl):
    {
      "ts":       "2026-04-23T17:15:22Z",    # timestamp UTC ISO-8601
      "run_id":   "WFB_20260423_142215",     # ID único del run
      "window":   "W2",                      # ventana WFB
      "seed":     42,                        # semilla
      "phase":    "xgboost_train",           # fase del pipeline
      "event":    "phase_end",               # tipo de evento
      "status":   "ok",                      # ok | warn | error | skip
      "elapsed_s": 847.3,                    # duración en segundos
      "metrics":  {...},                     # métricas clave de la fase
      "warnings": [...],                     # warnings no-fatales
      "errors":   [...]                      # errores
    }

Tipos de evento:
    phase_start   — inicio de una fase (subprocess iniciado)
    phase_end     — fin de una fase con métricas y timing
    gate_result   — resultado de un WFBPhaseGate
    error         — error crítico no recuperado
    signal_summary — resumen del embudo de señales
    gauntlet      — resultado del Gauntlet estadístico

Uso:
    # En run_walkforward_pipeline_v2.py
    from luna.validation.run_registry import RunRegistry
    registry = RunRegistry(run_id="WFB_20260423", window="W2", seed=42, root=_ROOT)
    registry.log_phase_end("xgboost_train", metrics={"auc_val": 0.534}, elapsed=847.3)

    # Post-mortem / inspect_run.py
    registry = RunRegistry.load(run_id="WFB_20260423", root=_ROOT)
    df = registry.to_dataframe()
    summary = registry.phase_summary()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import datetime

# ---------------------------------------------------------------------------
# Mapa de scripts → fase + signature glob
# ---------------------------------------------------------------------------

_PHASE_MAP: Dict[str, str] = {
    "feature_pipeline.py":           "feature_pipeline",
    "run_ai_mining.py":               "ai_mining",
    "feature_selection_e.py":         "sfi",
    "data_integrity_check.py":        "data_integrity",
    "train_xgboost_v2.py":            "xgboost_train",
    "ensemble_lgbm.py":               "lgbm_train",
    "hmm_regime.py":                  "hmm_train",
    "ood_guard.py":                   "ood_guard",
    "train_metalabeler_v2.py":        "metalabeler_train",
    "calibrate_probabilities.py":     "calibrate",
    "generate_oos_predictions_v2.py": "generate_predictions",  # nombre V1 (retrocompat)
    "predict_oos.py":                 "generate_predictions",  # [V2-FIX] nombre V2
    "run_statistical_validation.py":  "gauntlet",
}

_SIGNATURE_MAP: Dict[str, Dict] = {
    "xgboost_train": {
        "glob": "xgboost_meta_*_signature.json",
        "keys": ["val_auc", "auc_val", "proba_std", "val_proba_std", "n_features", "n_estimators"],
    },
    "lgbm_train": {
        "glob": "lgbm_meta_*_signature.json",
        "keys": ["dsr_oos", "optimal_threshold", "cost_discounted", "features"],
    },
    "hmm_train": {
        "glob": "hmm_*_signature.json",
        "keys": ["n_states", "state_map", "jsd_drift", "log_likelihood"],
    },
    "metalabeler_train": {
        "glob": "metalabeler_*_signature.json",
        "keys": ["val_loss", "seq_len", "input_dim", "val_f1", "rf_n_estimators"],
    },
    "ood_guard": {
        "glob": "ood_guard_signature.json",
        "keys": ["n_features", "contamination", "anomaly_score_threshold"],
    },
    "sfi": {
        "glob": "selected_features.json",  # en features/, no models/
        "keys": ["selected_features", "pass_through_features", "alpha_signals_passed"],
        "in_features_dir": True,
    },
}


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class RunRegistry:
    """
    Registro institucional de eventos de un run WFB.

    El ledger se almacena en:
        {root}/data/wfb_outputs/ledger_{run_id}.jsonl

    Cada instancia corresponde a un (run_id, window, seed) específico,
    pero el ledger es compartido para todo el run (todas las ventanas).
    Contexto (window, seed) se inyecta automáticamente en cada evento.
    """

    LEDGER_DIR = "data/wfb_outputs"

    def __init__(
        self,
        run_id: str,
        window: str = "",
        seed: Union[int, str] = 0,
        root: Optional[Path] = None,
    ):
        self.run_id  = run_id
        self.window  = window
        self.seed    = int(seed) if seed else 0
        self.root    = Path(root) if root else Path(__file__).parent.parent.parent

        ledger_dir = self.root / self.LEDGER_DIR
        ledger_dir.mkdir(parents=True, exist_ok=True)
        self._ledger_path = ledger_dir / f"ledger_{run_id}.jsonl"
        self._models_dir  = self.root / "data" / "models"
        self._feats_dir   = self.root / "data" / "features"

    # -----------------------------------------------------------------------
    # API pública de escritura
    # -----------------------------------------------------------------------

    def log_phase_start(self, phase: str, script: str = "", args: list = None) -> None:
        """Registra el inicio de una fase (subprocess lanzado)."""
        self._append({
            "phase":   phase,
            "event":   "phase_start",
            "status":  "running",
            "script":  script,
            "args":    args or [],
            "metrics": {},
        })

    def log_phase_end(
        self,
        phase: str,
        elapsed: float = 0.0,
        status: str = "ok",
        returncode: int = 0,
        metrics: Dict[str, Any] = None,
        warnings: List[str] = None,
        errors: List[str] = None,
        skipped: bool = False,
    ) -> None:
        """
        Registra el fin de una fase con métricas y timing.
        Además de las métricas pasadas explícitamente, intenta recolectar
        las del *_signature.json correspondiente (pasivo, no falla).
        """
        collected = self._collect_metrics(phase) if not skipped else {}
        merged = {**collected, **(metrics or {})}

        self._append({
            "phase":     phase,
            "event":     "phase_end",
            "status":    "skip" if skipped else status,
            "elapsed_s": round(elapsed, 1),
            "returncode": returncode,
            "metrics":   merged,
            "warnings":  warnings or [],
            "errors":    errors or [],
        })

    def log_gate(self, gate_result) -> None:
        """Registra el resultado de un WFBPhaseGate."""
        try:
            d = gate_result.to_dict() if hasattr(gate_result, "to_dict") else dict(gate_result)
        except Exception:
            return
        self._append({
            "phase":     f"gate_{d.get('gate_id','?').lower()}",
            "event":     "gate_result",
            "status":    "pass" if d.get("passed") else ("hard_stop" if d.get("is_hard_stop") else "warn"),
            "elapsed_s": d.get("elapsed_s", 0),
            "gate_id":   d.get("gate_id", "?"),
            "gate_name": d.get("gate_name", "?"),
            "metrics":   d.get("metrics", {}),
            "warnings":  d.get("warnings", []),
            "errors":    d.get("errors", []),
        })

    def log_signal_summary(self, funnel: Dict[str, Any]) -> None:
        """Registra el resumen del embudo de señales (signal_funnel.json)."""
        self._append({
            "phase":   "signal_filter",
            "event":   "signal_summary",
            "status":  "warn" if funnel.get("filter_fallback_level", 0) > 0 else "ok",
            "metrics": {
                "n_initial":            funnel.get("n_initial", 0),
                "n_after_xgb":          funnel.get("after_xgb", 0),
                "n_after_lgbm":         funnel.get("after_lgbm", 0),
                "n_final":              funnel.get("after_all", funnel.get("n_final", 0)),
                "filter_fallback_level": funnel.get("filter_fallback_level", 0),
            },
            "warnings": [f"fallback_level={funnel.get('filter_fallback_level')}"] if funnel.get("filter_fallback_level", 0) > 0 else [],
            "errors":   [],
        })

    def log_gauntlet(self, verdict: Dict[str, Any]) -> None:
        """Registra el veredicto del Gauntlet estadístico."""
        approved = verdict.get("deploy_approved", False)
        m = verdict.get("metrics", {})
        self._append({
            "phase":   "gauntlet",
            "event":   "gauntlet",
            "status":  "pass" if approved else "reject",
            "metrics": {
                "deploy_approved":   approved,
                "sharpe":            m.get("sharpe_ratio", m.get("sharpe")),
                "max_drawdown":      m.get("max_drawdown"),
                "win_rate":          m.get("win_rate"),
                "n_trades":          m.get("n_trades"),
                "dsr":               m.get("dsr"),
                "rejection_reason":  verdict.get("rejection_reason", ""),
                "fallback_level":    verdict.get("signal_filter_fallback_level", 0),
            },
            "warnings": [],
            "errors":   [verdict.get("rejection_reason", "")] if not approved else [],
        })

    def log_error(self, phase: str, error_msg: str, exc: Exception = None) -> None:
        """Registra un error crítico."""
        self._append({
            "phase":   phase,
            "event":   "error",
            "status":  "error",
            "metrics": {},
            "warnings": [],
            "errors":   [error_msg, str(exc) if exc else ""],
        })

    # -----------------------------------------------------------------------
    # API pública de lectura
    # -----------------------------------------------------------------------

    def load_events(self) -> List[Dict]:
        """Carga todos los eventos del ledger como lista de dicts."""
        if not self._ledger_path.exists():
            return []
        events = []
        for line in self._ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return events

    def to_dataframe(self, window: str = "", phase: str = ""):
        """
        Retorna todos los eventos como pandas DataFrame.
        Filtra opcionalmente por window o phase.
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas requerido para to_dataframe()")
        events = self.load_events()
        if window:
            events = [e for e in events if e.get("window") == window]
        if phase:
            events = [e for e in events if e.get("phase") == phase]
        if not events:
            return None
        df = pd.json_normalize(events, sep="_")
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        return df

    def phase_summary(self) -> Dict[str, Dict]:
        """
        Resumen por fase: duración total, status, métricas clave.
        Retorna dict: {window: {phase: {elapsed, status, metrics_key...}}}
        """
        events = self.load_events()
        summary: Dict[str, Dict] = {}
        for e in events:
            if e.get("event") not in ("phase_end", "gate_result", "signal_summary", "gauntlet"):
                continue
            w = e.get("window", "?")
            p = e.get("phase", "?")
            summary.setdefault(w, {})[p] = {
                "status":    e.get("status", "?"),
                "elapsed_s": e.get("elapsed_s", 0),
                "metrics":   e.get("metrics", {}),
                "warnings":  e.get("warnings", []),
                "errors":    e.get("errors", []),
                "ts":        e.get("ts", ""),
            }
        return summary

    def query(self, event: str = "", phase: str = "", status: str = "", window: str = "") -> List[Dict]:
        """Filtra eventos por tipo, fase, status o ventana."""
        events = self.load_events()
        if event:
            events = [e for e in events if e.get("event") == event]
        if phase:
            events = [e for e in events if e.get("phase") == phase]
        if status:
            events = [e for e in events if e.get("status") == status]
        if window:
            events = [e for e in events if e.get("window") == window]
        return events

    def get_metric_series(self, phase: str, metric: str) -> List[Dict]:
        """Extrae la serie temporal de una métrica específica por ventana."""
        results = []
        for e in self.load_events():
            if e.get("phase") == phase and e.get("event") == "phase_end":
                val = e.get("metrics", {}).get(metric)
                if val is not None:
                    results.append({
                        "window": e.get("window"),
                        "seed":   e.get("seed"),
                        "ts":     e.get("ts"),
                        "value":  val,
                    })
        return results

    # -----------------------------------------------------------------------
    # Comparación entre runs
    # -----------------------------------------------------------------------

    @classmethod
    def load(cls, run_id: str, root: Optional[Path] = None) -> "RunRegistry":
        """Carga un registry existente (solo lectura útil para comparación)."""
        r = cls(run_id=run_id, root=root)
        return r

    def compare(self, other: "RunRegistry", phases: List[str] = None) -> Dict:
        """
        Compara métricas clave entre este run y otro.
        Retorna dict con diferencias por ventana y fase.
        """
        _COMPARE_METRICS = {
            "xgboost_train":   ["auc_val", "val_auc", "proba_std"],
            "lgbm_train":      ["lgbm_proba_std", "dsr_oos"],
            "hmm_train":       ["n_states", "jsd_drift"],
            "metalabeler_train": ["val_loss", "val_f1"],
            "signal_filter":   ["n_final", "filter_fallback_level"],
            "gauntlet":        ["sharpe", "max_drawdown", "win_rate", "n_trades"],
        }
        target_phases = phases or list(_COMPARE_METRICS.keys())
        self_summary  = self.phase_summary()
        other_summary = other.phase_summary()
        all_windows   = sorted(set(list(self_summary.keys()) + list(other_summary.keys())))

        comparison = {}
        for window in all_windows:
            comparison[window] = {}
            for phase in target_phases:
                s_data = self_summary.get(window, {}).get(phase, {})
                o_data = other_summary.get(window, {}).get(phase, {})
                for metric_key in _COMPARE_METRICS.get(phase, []):
                    s_val = s_data.get("metrics", {}).get(metric_key)
                    o_val = o_data.get("metrics", {}).get(metric_key)
                    if s_val is None and o_val is None:
                        continue
                    delta = None
                    if isinstance(s_val, (int, float)) and isinstance(o_val, (int, float)):
                        delta = round(s_val - o_val, 4)
                    key = f"{phase}.{metric_key}"
                    comparison[window][key] = {
                        "self":  s_val,
                        "other": o_val,
                        "delta": delta,
                    }
        return comparison

    # -----------------------------------------------------------------------
    # Export HTML
    # -----------------------------------------------------------------------

    def export_html(self, output_path: Optional[Path] = None) -> Path:
        """
        Genera un reporte HTML con tablas y gráficos Chart.js.
        Si output_path es None, guarda en data/reports/run_report_{run_id}.html
        """
        if output_path is None:
            rep_dir = self.root / "data" / "reports"
            rep_dir.mkdir(parents=True, exist_ok=True)
            output_path = rep_dir / f"run_report_{self.run_id}.html"

        events   = self.load_events()
        summary  = self.phase_summary()
        all_wins = sorted(summary.keys())

        # Extraer series para gráficos
        auc_series: Dict[str, list] = {}
        signals_series: Dict[str, list] = {}
        timing_by_phase: Dict[str, float] = {}

        for e in events:
            w = e.get("window", "?")
            if e.get("phase") == "xgboost_train" and e.get("event") == "phase_end":
                auc = e.get("metrics", {}).get("auc_val") or e.get("metrics", {}).get("val_auc")
                if auc:
                    auc_series.setdefault(w, []).append(float(auc))
            if e.get("phase") == "signal_filter" and e.get("event") == "signal_summary":
                n = e.get("metrics", {}).get("n_final", 0)
                signals_series.setdefault(w, []).append(int(n))
            if e.get("event") == "phase_end" and e.get("elapsed_s"):
                p = e.get("phase", "?")
                timing_by_phase[p] = timing_by_phase.get(p, 0) + float(e.get("elapsed_s", 0))

        # Promediar series
        auc_avg     = {w: round(sum(v)/len(v), 4) for w, v in auc_series.items() if v}
        signals_avg = {w: round(sum(v)/len(v))    for w, v in signals_series.items() if v}

        # Tabla de resumen por ventana
        table_rows = ""
        for w in all_wins:
            for phase, data in sorted(summary.get(w, {}).items()):
                status  = data.get("status", "?")
                elapsed = data.get("elapsed_s", 0)
                errors  = "; ".join(data.get("errors", []))[:80]
                warnings= "; ".join(data.get("warnings", []))[:80]
                status_icon = "✅" if status == "ok" or status == "pass" else ("⚠️" if status == "warn" else ("❌" if "error" in status or "hard" in status else "⏭️"))
                row_class = "" if status in ("ok", "pass", "skip") else ("warn-row" if status == "warn" else "fail-row")
                table_rows += f"""
                <tr class="{row_class}">
                    <td>{w}</td><td>{phase}</td>
                    <td>{status_icon} {status}</td>
                    <td>{elapsed:.0f}s</td>
                    <td class="small">{errors}</td>
                    <td class="small">{warnings}</td>
                </tr>"""

        # Datos para Chart.js
        chart_labels  = json.dumps(all_wins)
        chart_auc     = json.dumps([auc_avg.get(w) for w in all_wins])
        chart_signals = json.dumps([signals_avg.get(w) for w in all_wins])
        timing_labels = json.dumps(list(timing_by_phase.keys()))
        timing_data   = json.dumps([round(v, 1) for v in timing_by_phase.values()])
        gen_time      = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Luna WFB Report — {self.run_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e2e8f0; margin: 0; padding: 24px; }}
  h1 {{ color: #63b3ed; font-size: 1.5rem; border-bottom: 1px solid #2d3748; padding-bottom: 12px; }}
  h2 {{ color: #90cdf4; font-size: 1.1rem; margin-top: 32px; }}
  .meta {{ color: #718096; font-size: 0.85rem; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }}
  .chart-box {{ background: #1a202c; border-radius: 8px; padding: 16px; }}
  .chart-box canvas {{ max-height: 220px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #2d3748; color: #90cdf4; padding: 8px 12px; text-align: left; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #2d3748; }}
  tr:hover td {{ background: #2d3748; }}
  .warn-row td {{ background: #2d3a1a; }}
  .fail-row td {{ background: #2d1a1a; }}
  .small {{ font-size: 0.78rem; color: #a0aec0; max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }}
  .badge-ok {{ background: #2d6a4f; color: #b7e4c7; }}
  .badge-warn {{ background: #5c4a00; color: #ffd166; }}
  .badge-fail {{ background: #6b1a1a; color: #ffb3b3; }}
</style>
</head>
<body>
<h1>🔬 Luna WFB Run Report</h1>
<div class="meta">
  Run ID: <strong>{self.run_id}</strong> &nbsp;|&nbsp;
  Ventanas: <strong>{len(all_wins)}</strong> &nbsp;|&nbsp;
  Generado: {gen_time}
</div>

<div class="grid">
  <div class="chart-box">
    <h2>XGBoost AUC por Ventana</h2>
    <canvas id="aucChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>Señales Finales por Ventana</h2>
    <canvas id="signalsChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>Tiempo Total por Fase (s)</h2>
    <canvas id="timingChart"></canvas>
  </div>
</div>

<h2>Detalle por Fase y Ventana</h2>
<table>
  <thead><tr><th>Window</th><th>Fase</th><th>Status</th><th>Tiempo</th><th>Errores</th><th>Warnings</th></tr></thead>
  <tbody>{table_rows}</tbody>
</table>

<script>
const chartDefaults = {{
  plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#a0aec0' }}, grid: {{ color: '#2d3748' }} }},
    y: {{ ticks: {{ color: '#a0aec0' }}, grid: {{ color: '#2d3748' }} }}
  }}
}};

new Chart(document.getElementById('aucChart'), {{
  type: 'line',
  data: {{ labels: {chart_labels}, datasets: [{{
    label: 'AUC OOS', data: {chart_auc},
    borderColor: '#63b3ed', backgroundColor: 'rgba(99,179,237,0.1)',
    tension: 0.3, fill: true, pointRadius: 5
  }}] }},
  options: {{ ...chartDefaults, plugins: {{ ...chartDefaults.plugins,
    annotation: {{ annotations: {{ line1: {{ type: 'line', yMin: 0.51, yMax: 0.51,
      borderColor: '#fc8181', borderWidth: 1, borderDash: [4,4] }} }} }} }} }}
}});

new Chart(document.getElementById('signalsChart'), {{
  type: 'bar',
  data: {{ labels: {chart_labels}, datasets: [{{
    label: 'Señales', data: {chart_signals},
    backgroundColor: 'rgba(72,187,120,0.7)', borderColor: '#48bb78', borderWidth: 1
  }}] }},
  options: chartDefaults
}});

new Chart(document.getElementById('timingChart'), {{
  type: 'bar',
  data: {{ labels: {timing_labels}, datasets: [{{
    label: 'Segundos', data: {timing_data},
    backgroundColor: 'rgba(246,173,85,0.7)', borderColor: '#f6ad55', borderWidth: 1
  }}] }},
  options: {{ ...chartDefaults, indexAxis: 'y' }}
}});
</script>
</body>
</html>"""

        output_path.write_text(html, encoding="utf-8")
        return output_path

    # -----------------------------------------------------------------------
    # Interno: colector pasivo de *_signature.json
    # -----------------------------------------------------------------------

    def _collect_metrics(self, phase: str) -> Dict[str, Any]:
        """
        Lee métricas del *_signature.json correspondiente a la fase.
        Completamente pasivo: no falla si el archivo no existe.
        """
        sig_cfg = _SIGNATURE_MAP.get(phase)
        if not sig_cfg:
            return {}

        search_dir = self._feats_dir if sig_cfg.get("in_features_dir") else self._models_dir
        try:
            matches = list(search_dir.glob(sig_cfg["glob"]))
            if not matches:
                return {}
            # Si hay varios, tomar el más reciente
            sig_path = max(matches, key=lambda p: p.stat().st_mtime)
            raw = json.loads(sig_path.read_text(encoding="utf-8"))
            # Extraer solo las claves relevantes (flatten un nivel si es dict anidado)
            result = {}
            for key in sig_cfg["keys"]:
                val = raw.get(key)
                if val is not None:
                    if isinstance(val, (int, float, str, bool)):
                        result[key] = val
                    elif isinstance(val, list):
                        result[key] = len(val)  # solo el count
                    elif isinstance(val, dict):
                        result[f"{key}_keys"] = list(val.keys())[:5]
            return result
        except Exception:
            return {}

    # -----------------------------------------------------------------------
    # Interno: append al ledger JSONL
    # -----------------------------------------------------------------------

    def _append(self, payload: Dict) -> None:
        """Append atómico de un evento al ledger JSONL."""
        event = {
            "ts":      datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_id":  self.run_id,
            "window":  self.window,
            "seed":    self.seed,
            **payload,
        }
        try:
            line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
            with open(self._ledger_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # Fail-safe total: el pipeline no falla por un error de logging
            pass

    # -----------------------------------------------------------------------
    # Factory: script name → fase
    # -----------------------------------------------------------------------

    @staticmethod
    def script_to_phase(script_path: str) -> str:
        """Convierte un path de script a un nombre de fase canónico."""
        name = Path(script_path).name
        return _PHASE_MAP.get(name, name.replace(".py", ""))

    # -----------------------------------------------------------------------
    # Búsqueda de ledgers existentes
    # -----------------------------------------------------------------------

    @classmethod
    def list_runs(cls, root: Optional[Path] = None) -> List[Dict]:
        """Lista todos los runs disponibles con metadatos básicos."""
        root = Path(root) if root else Path(__file__).parent.parent.parent
        ledger_dir = root / cls.LEDGER_DIR
        runs = []
        for p in sorted(ledger_dir.glob("ledger_*.jsonl"), reverse=True):
            run_id = p.stem.replace("ledger_", "")
            lines  = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
            size   = p.stat().st_size
            mtime  = datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat()
            runs.append({"run_id": run_id, "n_events": lines, "size_kb": round(size/1024,1), "last_modified": mtime})
        return runs
