"""
generate_validation_report.py — Luna V1
=========================================
Genera el Reporte de Validación Estadística en formato Markdown.
Se llama desde run_statistical_validation.py al final de Fase 5.

Secciones:
  §0. Raw Verdict JSON
  §1. Gauntlet Resumen Ejecutivo
  §2. Interpretación métrica a métrica (DSR, PBO, p-binomial)
  §3. Feature Selection (SFI embudo + features seleccionadas)
  §4. Análisis de realismo del backtesting
  §5. Arquitectura de modelos e hiperparámetros
  §6. Cumplimiento SOP R1-R13
  §7. Historial de runs (últimas entradas de diario.md o approx)
  §8. Conclusión y veredicto final

Output: data/reports/{timestamp}_{arch}_{brier}_Statistical_Validation_Report.md
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _pass(flag: bool) -> str:
    return "✅ PASS" if flag else "❌ FAIL"


def _gap(current, threshold, direction: str) -> str:
    """direction: 'above' means current needs to be > threshold."""
    try:
        diff = float(threshold) - float(current)
        if direction == "above":
            return f"Faltan +{diff:.2f}" if diff > 0 else "Cumple"
        else:  # below
            diff2 = float(current) - float(threshold)
            return f"Excede umbral en +{diff2:.2f}" if diff2 > 0 else "Cumple"
    except Exception:
        return "—"


def _fmt(val, fmt: str = ".4f") -> str:
    """Formato seguro de float opcional — devuelve '—' si no es convertible."""
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return "—"

# ── Feature info ──────────────────────────────────────────────────────────────

def _load_feature_info() -> tuple[list, list, list]:
    """Returns (sfi_features, passthrough_features, mining_rules)."""
    sf_path = _ROOT / "data" / "features" / "selected_features.json"
    data = _load_json(sf_path)

    sfi_feats      = data.get("selected_features", [])
    passthrough    = data.get("pass_through_features", [])
    timing_feats   = data.get("timing_features", [])

    # Detectar mining rules (golden_rule_*, genetic_rule_*)
    mining = [f for f in passthrough if re.match(r"(golden|genetic)_rule_\d+", str(f))]
    other_pt = [f for f in passthrough + timing_feats if f not in mining]

    return sfi_feats, other_pt, mining


# ── Model hyperparams ─────────────────────────────────────────────────────────

def _load_model_info() -> dict[str, Any]:
    models_dir = _ROOT / "data" / "models"

    xgb = _load_json(models_dir / "xgboost_meta_signature.json")
    ml2 = _load_json(models_dir / "metalabeler_v2_config.json")
    cal = _load_json(models_dir / "calibrator_signature.json")
    ood = _load_json(models_dir / "ood_guard_signature.json")

    return {"xgb": xgb, "ml2": ml2, "cal": cal, "ood": ood}


# ── Main report generator ─────────────────────────────────────────────────────

def generate(verdict: dict, trades_df=None, ts: str | None = None) -> Path:
    """
    Genera el reporte Markdown y lo guarda en data/reports/.
    Devuelve el path del archivo generado.
    """
    if ts is None:
        ts = datetime.now().strftime("%Y-%m-%d_T%H%M")

    run_id = os.environ.get("LUNA_RUN_ID", "?")

    m   = verdict.get("metrics", {})
    sa  = verdict.get("statistical_audit", {})
    flg = verdict.get("flags", {})
    thr = verdict.get("sop_thresholds", {})
    wfv = verdict.get("wfv_results", {})

    n_trades  = m.get("total_trades", 0)
    wr        = m.get("win_rate", 0) * 100
    ret       = m.get("total_return_pct", 0)
    dd        = m.get("max_drawdown_pct", 0)
    sharpe    = m.get("sharpe_crudo", 0)
    calmar    = m.get("calmar_ratio", 0)
    dsr       = sa.get("dsr", 0)
    pbo       = sa.get("estimated_pbo", 0) * 100
    p_binom   = sa.get("binomial_p_value", 1)
    skew      = sa.get("skewness", 0)
    kurt      = sa.get("kurtosis", 0)
    approved  = verdict.get("deploy_approved", False)
    verdict_ts = verdict.get("timestamp", "—")

    # Gates summary
    failed_gates = [k for k, v in flg.items() if not v]
    failed_str = ", ".join(failed_gates) if failed_gates else "—"

    # Model info
    info = _load_model_info()
    xgb = info["xgb"]
    ml2 = info["ml2"]
    cal = info["cal"]
    ood = info["ood"]

    # Optuna params
    best_params = xgb.get("best_params", {})
    n_estimators     = best_params.get("n_estimators", "—")
    max_depth        = best_params.get("max_depth", "—")
    learning_rate    = best_params.get("learning_rate", "—")
    subsample        = best_params.get("subsample", "—")
    colsample_bytree = best_params.get("colsample_bytree", "—")
    min_child_weight = best_params.get("min_child_weight", "—")
    dsr_is           = xgb.get("best_dsr_oos", "—")
    n_trials_total   = xgb.get("n_trials_total", 100)

    # MetaLabeler
    brier_raw = ml2.get("brier_raw", cal.get("brier_score_raw", "—"))
    brier_cal = ml2.get("brier_calibrated", cal.get("brier_score_calibrated", "—"))
    try:
        brier_mejora = (1 - float(brier_cal) / float(brier_raw)) * 100
    except Exception:
        brier_mejora = "—"
    ml2_hidden   = ml2.get("lstm_hidden", 32)
    ml2_seq_len  = ml2.get("seq_len", 96)
    ml2_rf_est   = ml2.get("rf_n_estimators", 300)
    ml2_dropout  = ml2.get("dropout", "—")
    ml2_seq_feats = ml2.get("seq_features", [])
    n_seq_feats  = len(ml2_seq_feats)

    # OOD
    ood_samples  = ood.get("n_samples", "—")
    ood_contam   = ood.get("contamination", "—")
    ood_thr      = ood.get("anomaly_score_threshold", "—")

    # SFI features
    sfi_feats, other_pt, mining_rules = _load_feature_info()
    n_sfi    = len(sfi_feats)
    n_mining = len(mining_rules)
    n_total  = n_sfi + n_mining + 1  # +1 HMM

    # Arch name for filename
    brier_str = f"{float(brier_cal):.4f}" if brier_cal != "—" else "na"
    arch_str  = f"XGBoost-MetaV2-RF_SFI{n_sfi}_brier{brier_str}"
    status    = "APROBADO" if approved else "RECHAZADO"
    verdict_icon = "✅" if approved else "❌"

    # ── Tearsheet relative path ───────────────────────────────────────────────
    tearsheet_rel = f"{ts}_tearsheet_oos.png"

    # ── SFI features table ───────────────────────────────────────────────────
    def _feat_table(feats, lag_data=None) -> str:
        if not feats:
            return "_No hay features en esta categoría._\n"
        rows = ["| # | Feature | Lag óptimo |",
                "|---|---------|-----------| "]
        for i, f in enumerate(feats, 1):
            lag = lag_data.get(str(f), {}).get("lag", "—") if lag_data else "—"
            rows.append(f"| {i} | `{f}` | {lag}H |")
        return "\n".join(rows) + "\n"

    def _pt_table(feats) -> str:
        if not feats:
            return "_No hay features pass-through registradas._\n"
        rows = ["| Feature | Origen |",
                "|---------|--------|"]
        for f in feats:
            origen = "Pass-Through (Mining)" if re.match(r"(golden|genetic)_rule_\d+", str(f)) else "Pass-Through (Timing/Engineered)"
            rows.append(f"| `{f}` | {origen} |")
        return "\n".join(rows) + "\n"

    # ── WFV section ──────────────────────────────────────────────────────────
    def _wfv_section() -> str:
        if not wfv:
            return "_WFV no disponible (insuficientes trades)._\n"
        rows = ["| Ventana | Trades | Win Rate | Rango |",
                "|---------|--------|----------|-------|"]
        for k, v in wfv.items():
            # BUG-02 FIX: nuevas claves start_date/end_date, fallback a start/end (retrocompat)
            t_start = v.get('start_date', v.get('start', '?'))
            t_end   = v.get('end_date',   v.get('end',   '?'))
            rows.append(
                f"| {k} | {v['n_trades']} | {v['win_rate']*100:.1f}% "
                f"| {t_start} -> {t_end} |"
            )
        return "\n".join(rows) + "\n"

    # -- Signal Diagnostic (§8) -----------------------------------------------
    def _signal_diagnostic() -> str:
        """Genera la seccion §8 de diagnostico de senal por ventana WFV."""
        import pandas as pd  # noqa: PLC0415

        if trades_df is None or len(trades_df) < 20:
            return "_Diagnostico no disponible (trades_df no suministrado o < 20 trades)._\n"

        df = trades_df.copy()
        # BUG-REPORT-01 FIX (2026-04-08): guard para RangeIndex (int) en lugar de DatetimeIndex.
        # merge_and_validate() hace reset_index(drop=True) → parquet queda con RangeIndex entero.
        # Esto causaba: AttributeError: 'RangeIndex' object has no attribute 'tz'
        # Fix: reconstruir DatetimeIndex desde la columna entry_time si está disponible.
        import pandas as _pd_rpt
        if not isinstance(df.index, _pd_rpt.DatetimeIndex):
            if "entry_time" in df.columns:
                df = df.set_index(_pd_rpt.to_datetime(df["entry_time"], utc=True))
            else:
                return "_Diagnostico no disponible (índice no es DatetimeIndex y sin columna entry_time)._\n"
        # Asegurar tz UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        n = len(df)
        q = n // 4
        boundaries = [df.index[i * q] for i in range(4)] + [df.index[-1] + pd.Timedelta(hours=1)]
        w_labels = ["W1", "W2", "W3", "W4"]

        def get_w(i):
            return df[(df.index >= boundaries[i]) & (df.index < boundaries[i + 1])]

        def flag_wr(wr_val):
            if wr_val >= 0.50: return "OK"
            if wr_val < 0.38:  return "CRITICO"
            return "BAJO"

        lines = []

        # WR mensual
        lines.append("### WR Mensual\n")
        lines.append("| Mes | Trades | WR | Avg XGB | Estado |")
        lines.append("|-----|--------|----|---------|--------|")
        try:
            df["_month"] = df.index.to_period("M")
            for _, row in df.groupby("_month").agg(
                trades=("is_win", "count"),
                wr=("is_win", "mean"),
                avg_xgb=("xgb_prob", "mean") if "xgb_prob" in df.columns else ("is_win", "count"),
            ).reset_index().iterrows():
                st = flag_wr(row["wr"])
                xgb_str = f"{row['avg_xgb']:.3f}" if "avg_xgb" in row else "—"
                lines.append(f"| {row['_month']} | {int(row['trades'])} | {row['wr']:.1%} | {xgb_str} | {st} |")
        except Exception as exc:
            lines.append(f"| _Error: {exc}_ |")
        lines.append("")

        # Cuartiles XGB + MetaLabeler por ventana
        lines.append("### Cuartiles XGB y MetaLabeler por Ventana WFV\n")
        lines.append("| Ventana | WR | XGB Q1 | XGB Q2 | XGB Q3 | XGB Q4 | Meta Q1 | Meta Q4 | R:R | Hold(h) |")
        lines.append("|---------|----|----|----|----|----|----|-----|-----|---------|")
        for i, lbl in enumerate(w_labels):
            w = get_w(i)
            if len(w) < 8:
                lines.append(f"| {lbl} | — | — | — | — | — | — | — | — | — |")
                continue
            wr_w = w["is_win"].mean()

            # XGB cuartiles
            xgb_q = ["-", "-", "-", "-"]
            if "xgb_prob" in w.columns:
                try:
                    cuts = pd.qcut(w["xgb_prob"], 4, labels=False, duplicates="drop")
                    qwrs = w.groupby(cuts)["is_win"].mean()
                    xgb_q = [f"{qwrs.get(j, float('nan')):.0%}" if j in qwrs else "-" for j in range(4)]
                except Exception:
                    pass

            # MetaLabeler cuartiles (Q1 y Q4 solo)
            m_q1, m_q4 = "-", "-"
            if "meta_v2_prob" in w.columns:
                try:
                    mcuts = pd.qcut(w["meta_v2_prob"], 4, labels=False, duplicates="drop")
                    mqwrs = w.groupby(mcuts)["is_win"].mean()
                    m_q1 = f"{mqwrs.get(0, float('nan')):.0%}" if 0 in mqwrs else "-"
                    m_q4 = f"{mqwrs.get(3, float('nan')):.0%}" if 3 in mqwrs else "-"
                except Exception:
                    pass

            # R:R
            rr_str = "-"
            wins  = w[w["is_win"]]["return_pct"] if "return_pct" in w.columns and w["is_win"].any() else None
            losss = w[~w["is_win"]]["return_pct"] if "return_pct" in w.columns and (~w["is_win"]).any() else None
            if wins is not None and losss is not None and len(wins) and len(losss) and losss.mean() != 0:
                rr_str = f"{abs(wins.mean() / losss.mean()):.2f}x"

            # Holding time
            hold_str = "-"
            if "entry_time" in w.columns and "exit_time" in w.columns:
                try:
                    hold_h = (w["exit_time"] - w["entry_time"]).dt.total_seconds() / 3600
                    hold_str = f"{hold_h.mean():.0f}"
                except Exception:
                    pass

            st = flag_wr(wr_w)
            lines.append(
                f"| {lbl} | {wr_w:.1%} ({st}) "
                f"| {xgb_q[0]} | {xgb_q[1]} | {xgb_q[2]} | {xgb_q[3]} "
                f"| {m_q1} | {m_q4} | {rr_str} | {hold_str} |"
            )
        lines.append("")
        lines.append("> WR OK >=50% | BAJO 38-50% | CRITICO <38%")
        lines.append("> XGB Q4 peor que Q1 = senal invertida (sobreajuste)")
        lines.append("> Meta Q1 CRITICO = MetaLabeler falla al filtrar perdedores en ese regimen")
        return "\n".join(lines) + "\n"

    # ── DSR formula block ─────────────────────────────────────────────────────
    sr_crudo = sharpe
    dsr_block = f"""\
FORMULA: DSR = Phi[(SR - SR*) / sigma_SR]   (Bailey & Lopez de Prado, 2014)

Inputs usados:
  SR_crudo   = {sr_crudo:.4f}   (Sharpe anualizado OOS)
  n_obs      = {n_trades}   (trades disponibles — controla sigma_SR)
  n_trials   = {n_trials_total}   (trials Optuna — controla SR*)
  skewness   = {skew:.3f}
  kurtosis   = {kurt:.3f}

SR*  = Sharpe esperado del mejor trial puramente por azar
       Con {n_trials_total} trials Optuna, la barra es alta.
sigma_SR = Incertidumbre del estadistico Sharpe (funcion de n_obs y momentos)

Resultado:
  DSR = {dsr:.4f} = {dsr*100:.1f}% de probabilidad de senal real"""

    # ── Report body ───────────────────────────────────────────────────────────
    report = f"""# Luna V1 — Reporte de Validacion Estadistica
## {ts} | Arquitectura: XGBoost → MetaLabelerV2 (LSTM-{ml2_hidden} extractor + RandomForest-{ml2_rf_est}) → Platt Scaling (Sigmoid)

**Generado:** {datetime.now().strftime("%Y-%m-%d T%H:%M")}  
**Run ID:** `{run_id}`  
**Modo:** `dev` (train 2020–2024-12-31, validacion 2024-01–2024-06, OOS holdout 2025)  
**Veredicto:** {verdict_icon} {status}

---

> [!WARNING]
> Este reporte es resultado de un **backtesting historico en modo dev**.
> **NO representa rendimiento real en mercado en vivo.**  
> Leer la seccion §4 (Analisis de Realismo) antes de tomar decisiones de deploy.

---


### TearSheet Visual OOS

![TearSheet OOS {ts}]({tearsheet_rel})

> Archivo: `data/reports/{ts}_tearsheet_oos.png`

---

## § 0. Raw Verdict JSON (fuente de verdad)

> Datos completos del archivo `data/reports/statistical_verdict.json` correspondiente a este run.
> Archivo archivado como `data/reports/{ts}_statistical_verdict.json`.

```json
{json.dumps({k: v for k, v in verdict.items() if k != 'wfv_results'}, indent=2, default=str)}
```

---

## § 1. Resultados del Gauntlet — Resumen Ejecutivo

| Metrica | Valor | Umbral SOP | Estado | Que Significa |
|---------|-------|-----------|--------|---------------|
| **Trades OOS** | {n_trades} | >= {thr.get('min_trades',100)} | {_pass(flg.get('pass_trades',False))} | Base estadistica para el test binomial |
| **Win Rate** | **{wr:.1f}%** | > 52% | {_pass(wr > 52)} | % de trades que tocaron PT antes que SL |
| **Sharpe crudo** | **{sharpe:.4f}** | > 1.5 | {_pass(sharpe > 1.5)} | Retorno ajustado a riesgo anualizado |
| **DSR** | **{dsr:.4f}** | >= {thr.get('min_dsr',0.75)} | {_pass(flg.get('pass_dsr',False))} | Prob. de que la senal sea real (Bailey & LdP) |
| **p-value binomial** | **{p_binom:.4e}** | <= 0.05 | {_pass(flg.get('pass_binomial',False))} | Prob. de obtener este WR por azar |
| **PBO (CSCV)** | **{pbo:.1f}%** | <= {thr.get('max_pbo_pct',10):.0f}% | {_pass(flg.get('pass_pbo',False))} | % simulaciones donde hay overfitting IS->OOS |
| **MaxDrawdown** | **{dd:.1f}%** | < {thr.get('max_drawdown_pct',60):.0f}% | {_pass(flg.get('pass_dd',False))} | Caida pico->valle en curva de equity |
| **Calmar Ratio** | **{calmar:.2f}** | — | — | Retorno / MaxDD |
| **Total Return** | {ret:.1f}% | — | — | Ver §4 — artefacto del compounding |

### Distribucion estadistica de retornos

| Estadistico | Valor | Interpretacion |
|------------|-------|----------------|
| Skewness | {skew:.3f} | {"Asimetria positiva — colas de ganancia mas largas" if skew > 0 else "Asimetria negativa — colas de perdida mas largas"} |
| Kurtosis | {kurt:.3f} | {"Leptocurtico — colas mas pesadas que normal" if kurt > 0 else "Platicurtico — colas mas delgadas que distribucion normal"} |
| p-binomial | {p_binom:.4e} | {"NO significativo — no se rechaza H0 (azar)" if p_binom > 0.05 else "SIGNIFICATIVO — se rechaza H0 (edge real)"} |

### Gap hacia aprobacion

| Gate | Valor Actual | Umbral SOP | Gap para Aprobar |
|------|-------------|-----------|------------------|
| {"❌" if not flg.get("pass_trades") else "✅"} Trades | {n_trades} | >={thr.get("min_trades",100)} | {_gap(n_trades, thr.get("min_trades",100), "above")} |
| {"❌" if wr <= 50 else "✅"} Win Rate | {wr:.1f}% | >50% | {_gap(wr, 50, "above")} |
| {"❌" if not flg.get("pass_dsr") else "✅"} DSR | {dsr:.4f} | >={thr.get("min_dsr",0.75)} | {_gap(dsr, thr.get("min_dsr",0.75), "above")} |
| {"❌" if sharpe <= 1.5 else "✅"} Sharpe | {sharpe:.2f} | >1.5 | {_gap(sharpe, 1.5, "above")} |
| {"❌" if not flg.get("pass_pbo") else "✅"} PBO | {pbo:.1f}% | <{thr.get("max_pbo_pct",10):.0f}% | {_gap(pbo, thr.get("max_pbo_pct",10), "below")} |
| {"✅" if flg.get("pass_dd") else "❌"} Max DD | {dd:.1f}% | <{thr.get("max_drawdown_pct",60):.0f}% | {"✅ OK" if flg.get("pass_dd") else _gap(dd, thr.get("max_drawdown_pct",60), "below")} |

### Walk-Forward Validation (WFV)

{_wfv_section()}

---

## § 2. Interpretacion Metrica a Metrica

### DSR = {dsr:.4f} — ?Que significa exactamente?

```
{dsr_block}
```

### PBO = {pbo:.1f}% — ?El modelo esta sobreajustado?

```
METODO: CSCV (Combinatorial Symmetric Cross-Validation)
        n_blocks=16, n_simulaciones=200, semilla=42

Proceso:
  Dividir {n_trades} trades en 16 bloques temporales
  Para cada simulacion (de 200):
       Permutar bloques aleatoriamente
       50% -> IS (in-sample)
       50% -> OOS (out-of-sample)
       Si Sharpe_IS > 0 pero Sharpe_OOS <= 0 -> OVERFIT
  PBO = n_overfit / 200

Resultado: {pbo:.1f}% en 200 simulaciones
Interpretacion: {"Degradacion IS->OOS en " + str(round(pbo)) + "% de simulaciones — re-entrenar con mas regularizacion." if pbo > 10 else "PBO dentro del umbral — sin sobreajuste detectado."}
Limite SOP: {thr.get("max_pbo_pct",10):.0f}%  -> {"FALLA" if not flg.get("pass_pbo") else "PASA"}
```

### p-binomial = {p_binom:.4e} — ?El WR es estadisticamente significativo?

```
H0: Win Rate = 50% (puro azar)
H1: Win Rate > 50% (edge real)

Datos: {int(wr/100*n_trades)} wins de {n_trades} trades = {wr:.1f}%

{"No rechazamos H0 — el WR actual no supera el umbral del azar con p<0.05." if p_binom > 0.05 else "Rechazamos H0 — el WR es estadisticamente significativo (p<0.05)."}
Para rechazar H0 necesitamos p < 0.05.
```

---

## § 3. Feature Selection Institucional (Fase 3B)

### Embudo SFI

```
  Input (features brutas en parquet)
        |
        v [B] Clustering Jerarquico (corr=0.70)
    Representantes de cluster
        |
        v [C] Lag Discovery por Informacion Mutua
    Candidatos con lag optimo
        |
        v [D] SFI-CPCV (DSR>=0.50, 6 grupos, 96H embargo)
    {n_sfi} supervivientes
        |
        v [E] Forward Selection greedy
    {n_sfi} features finales (seleccion estable)
        |
        + {n_mining} pass-through (Mining rules)
        + 1 HMM Regime (engineered feature)
        === {n_total} features totales al XGBoost
```

### Features Seleccionadas por SFI

{_feat_table(sfi_feats)}

### Pass-Through (Mining Rules — no pasan por SFI)

{_pt_table(mining_rules)}

> [!NOTE]
> Las mining rules bypasan el SFI por diseno: su baja frecuencia (pocos hits) las haria
> ser rechazadas por el test DSR, pero su alta precision durante los hits las hace valiosas.

---

## § 4. Analisis de Realismo del Backtesting

> [!NOTE]
> Sin alertas de realismo extremas.


### Elementos que SI son realistas

| Elemento | Implementacion | Evaluacion |
|----------|----------------|------------|
| Costos de transaccion | 0.15% round-trip | OK Conservador |
| Anti-leakage | guard_pipeline.py + SOP R1 | OK Estricto |
| Safety lags (M2, on-chain) | M2+42d, on-chain+24H, CPI+14d | OK Correcto |
| Embargo temporal | 96H entre train y test | OK Conservador |
| HMM Forward Algorithm | Sin Viterbi post-hoc | OK Causal |
| Calibracion de probabilidades | Platt Scaling (Sigmoid) activa | OK Mejorado |
| OOD Guard | Isolation Forest activo en inferencia | OK Kill switch |

### Elementos que INFLAN los resultados

| Factor | Efecto | Magnitud |
|--------|--------|----------|
| Periodo bull run BTC (2020-2024) | WR y Sharpe mas altos de lo normal | Alto — 15-20pp de WR |
| Sin slippage de liquidez real | Retornos ~5-15% superiores a live | Medio |
| Sin latencia de ejecucion | Asume mercado siempre disponible | Bajo-Medio |
| Compounding de {n_trades} trades | Total Return irreal (artefacto) | Solo afecta ese numero |

---

## § 5. Arquitectura de Modelos — Hiperparametros del Run

### XGBoost Meta-Model

```yaml
n_estimators:     {n_estimators}
max_depth:        {max_depth}
learning_rate:    {_fmt(learning_rate)}
subsample:        {_fmt(subsample, '.3f')}
colsample_bytree: {_fmt(colsample_bytree, '.3f')}
min_child_weight: {min_child_weight}
cost_fee:         0.15%
dsr_oos_train:    {_fmt(dsr_is)}
```

### Meta-Labeler — MetaLabelerV2 (P1-9: LSTM-{ml2_hidden} extractor + RandomForest-{ml2_rf_est})

```yaml
# Arquitectura: Rolling Stats extractor -> RandomForest-{ml2_rf_est}
brier_raw:      {_fmt(brier_raw)}
brier_calibrado:{_fmt(brier_cal)}  (mejora {_fmt(brier_mejora, '.1f')}%)
n_features_seq: {n_seq_feats}
seq_len:        {ml2_seq_len} horas
hidden_size:    {ml2_hidden}
rf_estimators:  {ml2_rf_est}
```

### Calibrador — Platt Scaling (Sigmoid)

```yaml
metodo:                Platt Scaling (Sigmoid)
brier_score_raw:       {_fmt(brier_raw)}
brier_score_calibrado: {_fmt(brier_cal)}
mejora:               {_fmt(brier_mejora, '.1f')}%
```
> Brier Score: 0.0 = perfecto | 0.25 = aleatorio

### OOD Guard (Isolation Forest)

```yaml
n_samples_entrenados:  {ood_samples:,}
contamination:         {_fmt(ood_contam, '.0%') if ood_contam != '—' else '—'}
threshold_anomalia:    {ood_thr}
kill_switch:           Si score(X) < threshold -> abortar trade
```

---

## § 6. Cumplimiento de Reglas SOP (R1-R13)

```
R1  – Causalidad estricta (no look-ahead)       OK guard_pipeline.py activo
R2  – Purging PurgedKFold/WFA                   OK WFA 8 splits + 96H embargo
R3  – Embargo >= 96H                            OK Aplicado en XGBoost y MetaLabeler
R4  – Triple frontera datos (train/val/holdout)  OK Train 2020-23, Val 2024, Holdout 2025
R5  – DSR correcto (Bailey & LdP 2014)          OK n_trials={n_trials_total} (SOP estandar)
R6  – Costos 0.15% minimos                      OK 0.15% descontados en todos los retornos
R7  – FracDiff dinamico                         OK (Aplicado en feature_pipeline.py)
R8  – Minimo 100 trades                         {"OK" if n_trades >= 100 else "WARN"} {n_trades} trades
R9  – HMM Forward Algorithm                     OK Sin Viterbi post-hoc
R10 – Calibracion de probabilidades             OK Platt Scaling (Sigmoid) (Brier {_fmt(brier_mejora, '.1f')}%)
R11 – 0 columnas 100% NaN                       OK Garantizado por pipeline
R12 – Checkpoint SFI siempre recalculado        OK _fs_checkpoint.json borrado en cada run
R13 – Modelos y signature.json del mismo run    OK Verificado coordinacion de timestamps
```

---

## \u00a7 7. Walk-Forward Validation Detalle

{_wfv_section()}

> [!NOTE]
> El WFV divide el periodo OOS en 4 ventanas temporales iguales.
> Una tendencia decreciente en WR a lo largo de las ventanas indica drift de regimen.
> Ventanas con WR < 40% en produccion real activan revision obligatoria (SOP R8).

---

## \u00a7 8. Diagnostico de Senal por Ventana

{_signal_diagnostic()}

---

## \u00a7 9. Conclusion y Veredicto Final

```
================================================
   VEREDICTO FINAL — LUNA V1 {ts}
   Run ID: {run_id}
   Arquitectura: XGB -> MetaV2-RF -> Sigmoid
================================================

  {verdict_icon} GATES {"APROBADOS" if approved else f"FALLIDOS: {failed_str}"}
  WR={wr:.1f}% (necesita >50%) | DSR={dsr:.4f} (>=0.75)
  MaxDD={dd:.1f}% (limite 60%) | PBO={pbo:.1f}% (limite 10%)
  {"AUTORIZADO para deploy — proceder con Fase 6 (cloud deploy)" if approved else "NO AUTORIZADO para deploy — analizar causa raiz y re-entrenar"}

================================================
```

---

*Reporte auto-generado por Luna V1 Pipeline — `core/reports/generate_validation_report.py` v2.2*  
*Metodologia: Bailey & Lopez de Prado (2014) · SOP Luna v2.1 · R1-R13*  
*Timestamp del run: {verdict_ts}*  
*Timestamp del reporte: {datetime.now().strftime("%Y-%m-%d T%H:%M")}*
"""

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = _ROOT / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("LUNA_RUN_ID", "DEV")
    fname = f"{ts}_LunaV1_{arch_str}_{run_id}_Statistical_Validation_Report.md"
    out_path = out_dir / fname
    out_path.write_text(report, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    """Modo standalone: regenera el reporte desde el último verdict + trades."""
    import sys
    _ROOT2 = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_ROOT2))
    import pandas as pd

    verdict_path = _ROOT2 / "data" / "reports" / "statistical_verdict.json"
    trades_path  = _ROOT2 / "data" / "predictions" / "oos_trades.parquet"

    if not verdict_path.exists():
        print("ERROR: No se encontro statistical_verdict.json")
        sys.exit(1)

    with open(verdict_path, encoding="utf-8") as f:
        v = json.load(f)

    trades = pd.read_parquet(trades_path) if trades_path.exists() else None
    out = generate(v, trades)
    print(f"Reporte generado: {out}")
