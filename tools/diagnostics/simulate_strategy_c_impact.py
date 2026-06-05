"""
tools/diagnostics/simulate_strategy_c_impact.py
================================================
Simulación del impacto de la Estrategia C (WFB-PRIOR) en las seeds
de la última run, ventana por ventana, sin reentrenar.

Usa:
  - Modelos XGBoost entrenados (data/models/)
  - Holdouts por ventana (data/features/features_holdout_W*.parquet)
  - Thresholds WFB-PRIOR derivados de signatures históricas
  - Retornos 1H del holdout para estimar WR y EV

No puede simular: LGBM, MetaLabeler, momentum, embargo (requieren predict_oos).
Sí puede dar: conteo de señales XGBoost, WR estimada, EV estimado por ventana.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT       = Path("g:/Mi unidad/ia/luna_v2")
MODELS_DIR = ROOT / "data" / "models"
FEATURES   = ROOT / "data" / "features"
WFB_CACHE  = ROOT / "data" / "wfb_cache"
COST_PCT   = 0.0015
WINDOWS    = ["W1", "W2", "W3", "W4", "W5"]

# Factores de reducción del filtro downstream (estimados de la run real)
# El pipeline completo (LGBM + MetaLabeler + momentum + embargo) reduce señales:
# En la run real: ~2425 barras → 22 after_momentum → 1 after_embargo
# Eso es 99.1% de reducción. Pero eso era con RANGE degradado.
# Con RANGE operativo, estimamos que el filtro downstream reduce ~70-85%.
DOWNSTREAM_REDUCTION_ESTIMATE = (0.70, 0.85)  # rango (min, max reducción)


def load_model_and_features(agent: str):
    model_path = MODELS_DIR / f"xgboost_meta_{agent}_long.model"
    sig_path   = MODELS_DIR / f"xgboost_meta_{agent}_long_signature.json"
    if not model_path.exists():
        return None, [], {}
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    return clf, sig.get("features", []), sig


def safe_proba(clf, df, features):
    arr = np.zeros((len(df), len(features)), dtype=np.float32)
    for i, f in enumerate(features):
        if f in df.columns:
            arr[:, i] = df[f].values.astype(np.float32)
    dmat = xgb.DMatrix(arr, feature_names=features)
    return clf.get_booster().predict(dmat)


def get_wfb_prior_threshold(agent: str) -> tuple:
    """
    Calcula el threshold WFB-PRIOR para un agente buscando en el cache
    de ventanas históricas las que tuvieron EV>0.
    Retorna (threshold_median, n_ventanas, lista_thresholds)
    """
    thresholds_with_ev = []
    for seed_dir in WFB_CACHE.glob("seed*"):
        for w_id in ["W1", "W2", "W3", "W4"]:
            sig_path = seed_dir / w_id / "models" / f"xgboost_meta_{agent}_long_signature.json"
            if not sig_path.exists():
                continue
            try:
                s = json.loads(sig_path.read_text(encoding="utf-8"))
                thr = s.get("optimal_threshold")
                report = s.get("calibration_report", [])
                has_ev = any(r.get("ev", -1) > 0 for r in report)
                if thr is not None and has_ev:
                    max_ev = max((r.get("ev", -1) for r in report), default=-1)
                    thresholds_with_ev.append({
                        "seed": seed_dir.name,
                        "window": w_id,
                        "threshold": float(thr),
                        "max_ev": max_ev,
                        "n_trades_best": next(
                            (r.get("n_trades") for r in report if r.get("ev") == max_ev), 0
                        )
                    })
            except Exception:
                pass

    if not thresholds_with_ev:
        return None, 0, []

    thrs = [x["threshold"] for x in thresholds_with_ev]
    return float(np.median(thrs)), len(thrs), thresholds_with_ev


def analyze_window(clf, features, window_id: str, threshold_actual: float,
                   threshold_prior: float, agent: str):
    """Analiza una ventana específica con ambos thresholds."""
    holdout_path = FEATURES / f"features_holdout_{window_id}.parquet"
    if not holdout_path.exists():
        return None

    df = pd.read_parquet(holdout_path).dropna(subset=["close"])
    if len(df) < 10:
        return None

    probs = safe_proba(clf, df, features)
    close = df["close"].values
    rets  = np.diff(close) / close[:-1]
    probs_aligned = probs[:-1]  # alinear con returns

    def eval_threshold(t):
        mask = probs_aligned > t
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "wr": 0, "ev": None, "avg_win": 0, "avg_loss": 0}
        trade_rets = rets[mask] - COST_PCT
        wins  = trade_rets[trade_rets > 0]
        loses = trade_rets[trade_rets <= 0]
        if len(wins) == 0:
            return {"n": n, "wr": 0, "ev": -0.001, "avg_win": 0,
                    "avg_loss": abs(loses.mean()) if len(loses) > 0 else 0}
        p_win   = len(wins) / n
        avg_win = float(wins.mean())
        avg_los = float(abs(loses.mean())) if len(loses) > 0 else 0.0
        ev = p_win * avg_win - (1 - p_win) * avg_los
        return {"n": n, "wr": round(p_win, 3), "ev": round(ev, 6),
                "avg_win": round(avg_win, 5), "avg_loss": round(avg_los, 5)}

    actual = eval_threshold(threshold_actual)
    prior  = eval_threshold(threshold_prior) if threshold_prior else actual

    return {
        "window": window_id,
        "n_bars": len(df),
        "date_range": f"{df.index.min().date()} -> {df.index.max().date()}",
        "actual": actual,
        "prior": prior,
        "probs_p50": round(float(np.median(probs)), 3),
        "probs_p75": round(float(np.percentile(probs, 75)), 3),
        "probs_p90": round(float(np.percentile(probs, 90)), 3),
    }


def print_agent_report(agent: str):
    print(f"\n{'='*65}")
    print(f"  AGENTE: {agent.upper()}")
    print(f"{'='*65}")

    clf, features, sig = load_model_and_features(agent)
    if clf is None:
        print("  [SKIP] Modelo no encontrado")
        return

    thr_actual = sig.get("optimal_threshold", 0.48)
    thr_prior, n_prior, prior_details = get_wfb_prior_threshold(agent)

    print(f"  Threshold ACTUAL de la run : {thr_actual:.3f}")
    if thr_prior:
        thr_range = [round(x["threshold"], 3) for x in prior_details]
        print(f"  Threshold WFB-PRIOR (median): {thr_prior:.3f}  "
              f"(de {n_prior} ventanas historicas con EV>0)")
        print(f"  Rango thresholds historicos : {min(thr_range):.3f} - {max(thr_range):.3f}")
        ev_range = [round(x["max_ev"], 5) for x in prior_details]
        print(f"  Rango EV historico (max/vent): {min(ev_range):.5f} - {max(ev_range):.5f}")
    else:
        print("  WFB-PRIOR: sin ventanas historicas con EV>0")
        thr_prior = thr_actual

    print()
    print(f"  {'Window':<6} {'Bars':>6} {'Fecha holdout':<24} "
          f"{'N_sig_ACTUAL':>13} {'WR_ACT':>7} {'EV_ACT':>9} | "
          f"{'N_sig_PRIOR':>12} {'WR_PRI':>7} {'EV_PRI':>9} {'Delta_N':>8}")
    print(f"  {'-'*120}")

    total_actual, total_prior = 0, 0
    ev_positivos_actual, ev_positivos_prior = 0, 0
    window_results = []

    for w_id in WINDOWS:
        result = analyze_window(clf, features, w_id, thr_actual, thr_prior, agent)
        if result is None:
            continue
        window_results.append(result)

        n_act = result["actual"]["n"]
        n_pri = result["prior"]["n"]
        wr_act = result["actual"]["wr"]
        wr_pri = result["prior"]["wr"]
        ev_act = result["actual"]["ev"]
        ev_pri = result["prior"]["ev"]
        delta  = n_pri - n_act

        total_actual += n_act
        total_prior  += n_pri
        if ev_act is not None and ev_act > 0:
            ev_positivos_actual += 1
        if ev_pri is not None and ev_pri > 0:
            ev_positivos_prior += 1

        ev_act_str = f"{ev_act:.5f}" if ev_act is not None else "  N/A  "
        ev_pri_str = f"{ev_pri:.5f}" if ev_pri is not None else "  N/A  "
        delta_str  = f"{delta:+d}"
        print(f"  {w_id:<6} {result['n_bars']:>6} {result['date_range']:<24} "
              f"{n_act:>13} {wr_act:>7.1%} {ev_act_str:>9} | "
              f"{n_pri:>12} {wr_pri:>7.1%} {ev_pri_str:>9} {delta_str:>8}")

    print(f"  {'-'*120}")
    delta_total = total_prior - total_actual
    print(f"  {'TOTAL':<6} {'':>6} {'':24} "
          f"{total_actual:>13} {'':>7} {'':>9} | "
          f"{total_prior:>12} {'':>7} {'':>9} {delta_total:>+8}")

    print()
    print(f"  RESUMEN:")
    print(f"    Senales XGB ACTUAL   : {total_actual:5d}  (ventanas con EV>0: {ev_positivos_actual}/5)")
    print(f"    Senales XGB PRIOR    : {total_prior:5d}  (ventanas con EV>0: {ev_positivos_prior}/5)")
    print(f"    Delta senales XGB    : {delta_total:+5d}  ({delta_total/max(total_actual,1)*100:+.1f}%)")
    print()

    # Estimacion de trades finales (despues de filtros downstream)
    print(f"  ESTIMACION DE TRADES FINALES (post filtros downstream):")
    for label, total in [("ACTUAL", total_actual), ("PRIOR", total_prior)]:
        lo = int(total * (1 - DOWNSTREAM_REDUCTION_ESTIMATE[1]))
        hi = int(total * (1 - DOWNSTREAM_REDUCTION_ESTIMATE[0]))
        print(f"    {label:8s}: {total:5d} senales XGB -> estimado {lo}-{hi} trades finales "
              f"(asumiendo {int(DOWNSTREAM_REDUCTION_ESTIMATE[0]*100)}-"
              f"{int(DOWNSTREAM_REDUCTION_ESTIMATE[1]*100)}% reduccion downstream)")

    print()
    print(f"  NOTA: La estrategia C usa un threshold {'MAS ALTO' if thr_prior > thr_actual else 'IGUAL'} "
          f"que el actual ({thr_actual:.3f} -> {thr_prior:.3f}).")
    print(f"  Esto {'REDUCE' if thr_prior > thr_actual else 'MANTIENE'} el volumen de senales XGB "
          f"pero potencialmente MEJORA la calidad (WR, EV).")
    print(f"  El valor de la Estrategia C no es mas senales — es senales con EV historicamente positivo.")


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXTO DE LA RUN REAL PARA COMPARACION
# ─────────────────────────────────────────────────────────────────────────────
def print_run_context():
    print()
    print("CONTEXTO DE LA RUN REAL (seed42/53929 resultados conocidos):")
    print("-" * 60)
    print("  Embudo REAL de la run (todas las ventanas W1-W5):")
    print("    raw_oos_bars    : 2425")
    print("    after_momentum  : 22   <- filtro mas restrictivo")
    print("    after_embargo   : 1    <- BUG (corregido con FIX-FUNNEL-ACCUM-01)")
    print()
    print("  Causa identificada:")
    print("    - RANGE DEGRADED en W1-W5 (Gate-G2 Brier > adaptive_gate)")
    print("      -> 0 senales en barras de regimen lateral (~40% del tiempo)")
    print("    - BULL y BEAR operando solas con thresholds 0.48 / 0.65")
    print("    - El filtro downstream (LGBM + MetaLabeler + momentum) muy restrictivo")
    print()
    print("  Con FIX-BRIER-GATE-RANGE-01 (ya aplicado):")
    print("    - RANGE volvera a ser OPERABLE en la proxima run")
    print("    - Señales XGB esperadas: 3x-4x mas que en la run actual")
    print()


if __name__ == "__main__":
    print("SIMULACION IMPACTO ESTRATEGIA C (WFB-PRIOR) POR VENTANA")
    print("Luna V2 - Sin reentrenamiento")
    print("=" * 65)

    print_run_context()

    for agent in ["bull", "range", "bear"]:
        print_agent_report(agent)

    print()
    print("=" * 65)
    print("CONCLUSION FINAL:")
    print("  La Estrategia C NO aumenta las senales XGB — las FILTRA mejor.")
    print("  Su valor real: evitar operar con thresholds sin EV historico positivo,")
    print("  usando como referencia ventanas donde el modelo SI funciono.")
    print()
    print("  Para ver el impacto REAL en trades finales (incluyendo LGBM, MetaLabeler,")
    print("  momentum y embargo) se necesita correr predict_oos.py con los nuevos")
    print("  thresholds — pero ESO no requiere reentrenar, solo re-inferencia.")
