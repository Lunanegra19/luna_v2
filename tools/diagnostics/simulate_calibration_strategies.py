"""
tools/diagnostics/simulate_calibration_strategies.py
=====================================================
Simulador retrospectivo de estrategias de calibración de threshold.
NO re-entrena ningún modelo. Usa los modelos XGBoost ya entrenados
y los parquets de datos disponibles para simular qué threshold
habría elegido cada estrategia y cuántas señales OOS habría generado.

Estrategias simuladas:
  A) ACTUAL    : EV sweep solo en features_validation.parquet (comportamiento real)
  B) IS-TAIL   : Si validation falla, usar el último 20% de features_train
                 con penalización 0.5 en el EV mínimo requerido
  C) WFB-PRIOR : Usar mediana de thresholds de ventanas anteriores como fallback
  D) MULTI-VAL : Sweep sobre TODAS las ventanas de validation disponibles (W1-W5)
                 y tomar la mediana del threshold óptimo
  E) REGIME-IS : Filtrar el IS por régimen HMM y calibrar solo en esas barras
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────────────────────────────────
ROOT        = Path("g:/Mi unidad/ia/luna_v2")
MODELS_DIR  = ROOT / "data" / "models"
FEATURES    = ROOT / "data" / "features"
COST_PCT    = 0.0015  # round-trip cost

# ── Parámetros del sweep (mismos que _calibrate_threshold) ────────────────────
T_MIN     = 0.45
T_MAX     = 0.72
T_STEP    = 0.005

# Importar configuración institucional
try:
    from luna.core.config import settings as _cfg
except ImportError:
    import yaml
    _cfg_path = ROOT / "config" / "settings.yaml"
    class _MockCfg:
        pass
    _cfg = _MockCfg()
    with open(_cfg_path, "r", encoding="utf-8") as _f:
        _yaml_data = yaml.safe_load(_f)
        _cfg.stat = _yaml_data.get("stat", {})

_stat = getattr(_cfg, "stat", {}) if hasattr(_cfg, "stat") else getattr(_cfg, "gauntlet", {})
if isinstance(_stat, dict):
    MIN_TRADES = int(_stat.get("min_trades", 30))
else:
    MIN_TRADES = int(getattr(_stat, "min_trades", 30))

# ── Ventanas disponibles ──────────────────────────────────────────────────────
WINDOWS = ["W1", "W2", "W3", "W4", "W5"]

# ── Carga de recursos ──────────────────────────────────────────────────────────
def load_model(agent: str):
    """Carga un modelo XGBoost ya entrenado."""
    model_path = MODELS_DIR / f"xgboost_meta_{agent}_long.model"
    sig_path   = MODELS_DIR / f"xgboost_meta_{agent}_long_signature.json"
    if not model_path.exists():
        print(f"  [WARN] Modelo {agent} no encontrado: {model_path.name}")
        return None, None
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    sig = json.loads(sig_path.read_text(encoding="utf-8")) if sig_path.exists() else {}
    features = sig.get("features", [])
    return clf, features


def safe_proba(clf, df, features):
    """Genera probabilidades pasando feature_names explícitos al DMatrix."""
    arr = np.zeros((len(df), len(features)), dtype=np.float32)
    for i, f in enumerate(features):
        if f in df.columns:
            arr[:, i] = df[f].values.astype(np.float32)
    # feature_names debe coincidir con los del booster (guardados en el modelo)
    dmat = xgb.DMatrix(arr, feature_names=features)
    return clf.get_booster().predict(dmat)



def ev_sweep(probs, rets, t_min=T_MIN, t_max=T_MAX, t_step=T_STEP,
             min_trades=MIN_TRADES, ev_floor=0.0):
    """
    Barre thresholds buscando el máximo EV.
    ev_floor: EV mínimo aceptable (0.0 = estricto, -epsilon = permisivo)
    Retorna (best_threshold, best_ev, n_encontrados)
    """
    best_t, best_ev, best_n = t_min, -np.inf, 0
    thresholds = np.arange(t_min, t_max + t_step / 2, t_step)
    for t in thresholds:
        mask = probs > t
        n = int(mask.sum())
        if n < min_trades:
            continue
        trade_rets = rets[mask] - COST_PCT
        wins  = trade_rets[trade_rets > 0]
        loses = trade_rets[trade_rets <= 0]
        if len(wins) == 0:
            continue
        p_win   = len(wins) / n
        avg_win = wins.mean()
        avg_los = abs(loses.mean()) if len(loses) > 0 else 0.0
        ev = p_win * avg_win - (1 - p_win) * avg_los
        if ev > ev_floor and ev > best_ev:
            best_ev, best_t, best_n = ev, float(t), n
    return best_t, best_ev, best_n


def get_tbm_rets(df):
    """Retorno simple 1h como proxy (igual que el fallback del calibrador)."""
    close = df["close"].values
    rets  = np.diff(close) / close[:-1]
    return rets


def load_parquet_safe(path, features=None):
    """Carga un parquet con manejo de errores."""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path).dropna(subset=["close"])
        if features:
            for f in [ff for ff in features if ff not in df.columns]:
                df[f] = 0.0
        return df
    except Exception as e:
        print(f"  [ERROR] No se pudo leer {path.name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def simulate_agent(agent: str):
    print(f"\n{'='*65}")
    print(f"  AGENTE: {agent.upper()}")
    print(f"{'='*65}")

    clf, features = load_model(agent)
    if clf is None:
        return

    sig = json.loads((MODELS_DIR / f"xgboost_meta_{agent}_long_signature.json")
                     .read_text(encoding="utf-8"))
    threshold_actual = sig.get("optimal_threshold", 0.48)
    cal_source_actual = sig.get("cal_source", "?")

    print(f"  Threshold REAL de la run: {threshold_actual:.3f}  (fuente: {cal_source_actual})")
    print(f"  calibration_report: {len(sig.get('calibration_report', []))} entradas, "
          f"EV>0: {sum(1 for r in sig.get('calibration_report', []) if r.get('ev', -1) > 0)}")

    results = {}

    # ── Estrategia A: ACTUAL (validation W5 = última ventana) ──────────────────
    df_val = load_parquet_safe(FEATURES / "features_validation.parquet", features)
    if df_val is not None and len(df_val) >= MIN_TRADES:
        probs_val = safe_proba(clf, df_val, features)
        rets_val  = get_tbm_rets(df_val)
        probs_val_aligned = probs_val[:-1]
        t_a, ev_a, n_a = ev_sweep(probs_val_aligned, rets_val)
        results["A_ACTUAL"] = {"thr": t_a if ev_a > -np.inf else threshold_actual,
                                "ev": ev_a, "n_cal": n_a,
                                "fuente": f"validation ({len(df_val)} bars)"}
    else:
        results["A_ACTUAL"] = {"thr": threshold_actual, "ev": -np.inf, "n_cal": 0,
                                "fuente": "fallback (validation vacia)"}

    # ── Estrategia B: IS-TAIL 20% con penalización ────────────────────────────
    df_train = load_parquet_safe(FEATURES / "features_train.parquet", features)
    if df_train is not None:
        tail_start = int(len(df_train) * 0.80)
        df_tail = df_train.iloc[tail_start:].copy()
        print(f"\n  [B] IS-TAIL: {len(df_tail)} filas | "
              f"{df_tail.index.min().date()} -> {df_tail.index.max().date()}")
        if len(df_tail) >= MIN_TRADES:
            probs_tail = safe_proba(clf, df_tail, features)
            rets_tail  = get_tbm_rets(df_tail)
            # Penalización: EV mínimo = 0.0 (igual), pero aplicamos factor 0.7 al EV
            # para compensar el sesgo IS (el modelo vio estos datos en training)
            t_b, ev_b, n_b = ev_sweep(probs_tail[:-1], rets_tail, ev_floor=0.0)
            ev_b_penalizado = ev_b * 0.7 if ev_b > -np.inf else -np.inf
            results["B_IS_TAIL"] = {
                "thr": t_b if ev_b > -np.inf else threshold_actual,
                "ev": ev_b_penalizado, "n_cal": n_b,
                "fuente": f"IS-tail-20% penalizado 0.7x ({len(df_tail)} bars)"
            }
    else:
        results["B_IS_TAIL"] = {"thr": threshold_actual, "ev": -np.inf, "n_cal": 0,
                                  "fuente": "IS no disponible"}

    # ── Estrategia C: WFB-PRIOR (mediana de thresholds de ventanas anteriores) ──
    wfb_thresholds = []
    wfb_cache = ROOT / "data" / "wfb_cache"
    for seed_subdir in wfb_cache.glob("seed*"):
        for w_id in ["W1", "W2", "W3", "W4"]:
            sig_path = seed_subdir / w_id / "models" / f"xgboost_meta_{agent}_long_signature.json"
            if sig_path.exists():
                try:
                    s = json.loads(sig_path.read_text(encoding="utf-8"))
                    thr = s.get("optimal_threshold")
                    ev_check = s.get("calibration_report", [])
                    has_positive_ev = any(r.get("ev", -1) > 0 for r in ev_check)
                    if thr is not None and has_positive_ev:
                        wfb_thresholds.append(float(thr))
                except Exception:
                    pass

    if wfb_thresholds:
        t_c = float(np.median(wfb_thresholds))
        print(f"\n  [C] WFB-PRIOR: {len(wfb_thresholds)} ventanas con EV>0 encontradas | "
              f"thresholds: {[round(x, 3) for x in wfb_thresholds]}")
        results["C_WFB_PRIOR"] = {
            "thr": t_c, "ev": None, "n_cal": len(wfb_thresholds),
            "fuente": f"mediana WFB ({len(wfb_thresholds)} ventanas con EV>0)"
        }
    else:
        print(f"\n  [C] WFB-PRIOR: sin ventanas previas con EV>0 en cache")
        results["C_WFB_PRIOR"] = {"thr": threshold_actual, "ev": -np.inf, "n_cal": 0,
                                    "fuente": "sin prior disponible"}

    # ── Estrategia D: MULTI-VALIDATION (mediana sobre W1-W5) ────────────────────
    multi_thresholds = []
    for w_id in WINDOWS:
        p = FEATURES / f"features_validation_{w_id}.parquet"
        df_w = load_parquet_safe(p, features)
        if df_w is None or len(df_w) < MIN_TRADES:
            continue
        pr_w = safe_proba(clf, df_w, features)
        re_w = get_tbm_rets(df_w)
        t_w, ev_w, n_w = ev_sweep(pr_w[:-1], re_w)
        ev_str_w = f"{ev_w:.5f}" if ev_w > -np.inf else "N/A"
        label = "EV>0" if ev_w > -np.inf else "EV<=0"
        print(f"\n  [D] {w_id} validation ({len(df_w)} bars): "
              f"thr={t_w:.3f}  ev={ev_str_w}  {label}")

        if ev_w > -np.inf:
            multi_thresholds.append(t_w)

    if multi_thresholds:
        t_d = float(np.median(multi_thresholds))
        results["D_MULTI_VAL"] = {
            "thr": t_d, "ev": None, "n_cal": len(multi_thresholds),
            "fuente": f"mediana multi-validation ({len(multi_thresholds)}/{len(WINDOWS)} ventanas con EV>0)"
        }
    else:
        print(f"\n  [D] MULTI-VAL: ninguna ventana produjo EV>0")
        results["D_MULTI_VAL"] = {"thr": threshold_actual, "ev": -np.inf, "n_cal": 0,
                                    "fuente": "ninguna validation con EV>0"}

    # ── Estrategia E: REGIME-IS (IS filtrado por régimen HMM del agente) ──────────
    hmm_labels_path = FEATURES / "hmm_regime_labels.parquet"
    regime_map = {"bull": ["0_BULL_QUIET", "1_BULL_MOMENTUM", "2_BULL_EUPHORIA"],
                   "range": ["5_RANGE_COMPRESS", "6_RANGE_FLAT"],
                   "bear": ["3_CALM_BEAR", "3_BEAR_CRASH", "4_BEAR_FORCED"]}

    if df_train is not None and hmm_labels_path.exists():
        try:
            df_hmm = pd.read_parquet(hmm_labels_path)
            # Intentar merge en el IS
            if "HMM_Semantic" in df_train.columns:
                df_regime_is = df_train.copy()
            elif "HMM_Semantic" in df_hmm.columns:
                df_regime_is = df_train.join(df_hmm[["HMM_Semantic"]], how="left")
            else:
                df_regime_is = None

            if df_regime_is is not None and "HMM_Semantic" in df_regime_is.columns:
                regimes_for_agent = regime_map.get(agent, [])
                # Mostrar regimenes disponibles para debug
                unique_hmm = df_regime_is["HMM_Semantic"].dropna().unique()
                print(f"\n  [E] HMM regimenes en IS: {list(unique_hmm)[:8]}")
                df_filtered = df_regime_is[
                    df_regime_is["HMM_Semantic"].isin(regimes_for_agent)
                ].copy()
                print(f"\n  [E] REGIME-IS: {len(df_filtered)} barras IS con régimen {agent} "
                      f"(de {len(df_train)} totales)")
                if len(df_filtered) >= MIN_TRADES * 2:
                    # Usar el último 30% de esas barras para calibrar
                    tail_e = int(len(df_filtered) * 0.70)
                    df_tail_e = df_filtered.iloc[tail_e:].copy()
                    pr_e = safe_proba(clf, df_tail_e, features)
                    re_e = get_tbm_rets(df_tail_e)
                    t_e, ev_e, n_e = ev_sweep(pr_e[:-1], re_e, ev_floor=0.0)
                    ev_e_pen = ev_e * 0.65 if ev_e > -np.inf else -np.inf
                    results["E_REGIME_IS"] = {
                        "thr": t_e if ev_e > -np.inf else threshold_actual,
                        "ev": ev_e_pen, "n_cal": n_e,
                        "fuente": f"IS filtrado por régimen {agent} tail-30% pen 0.65x"
                    }
                else:
                    results["E_REGIME_IS"] = {"thr": threshold_actual, "ev": -np.inf,
                                               "n_cal": 0, "fuente": "regime-IS insuficiente"}
            else:
                results["E_REGIME_IS"] = {"thr": threshold_actual, "ev": -np.inf,
                                           "n_cal": 0, "fuente": "HMM labels no disponibles en IS"}
        except Exception as ex:
            print(f"  [E] Error: {ex}")
            results["E_REGIME_IS"] = {"thr": threshold_actual, "ev": -np.inf,
                                       "n_cal": 0, "fuente": f"error: {ex}"}
    else:
        results["E_REGIME_IS"] = {"thr": threshold_actual, "ev": -np.inf,
                                    "n_cal": 0, "fuente": "IS o HMM no disponible"}

    # -----------------------------------------------------------------------------
    # Impacto en señales OOS
    # -----------------------------------------------------------------------------
    # Usar features_holdout.parquet (el OOS real de la última ventana)
    print(f"\n  {'-'*60}")
    print(f"  IMPACTO EN SENALES OOS (holdout real)")
    print(f"  {'-'*60}")

    df_holdout = load_parquet_safe(FEATURES / "features_holdout.parquet", features)
    if df_holdout is not None:
        probs_oos = safe_proba(clf, df_holdout, features)
        print(f"  Barras OOS totales: {len(df_holdout)}")
        print(f"  Distribución probs: P25={np.percentile(probs_oos,25):.3f} "
              f"P50={np.percentile(probs_oos,50):.3f} "
              f"P75={np.percentile(probs_oos,75):.3f} "
              f"P90={np.percentile(probs_oos,90):.3f} "
              f"P95={np.percentile(probs_oos,95):.3f}")
        print()

        header = f"{'Estrategia':<20} {'Thr':>6} {'Senales OOS':>12} {'%Activ':>8} {'EV_cal':>10}  Fuente"
        print(f"  {header}")
        print(f"  {'-'*95}")

        for strat_name, info in results.items():
            t = info["thr"]
            n_oos = int((probs_oos > t).sum())
            pct   = n_oos / len(df_holdout) * 100
            ev_str = f"{info['ev']:.5f}" if info["ev"] is not None and info["ev"] > -np.inf else "N/A"
            fuente = info["fuente"][:40]
            print(f"  {strat_name:<20} {t:>6.3f} {n_oos:>12} {pct:>11.1f}% {ev_str:>10} {fuente}")
    else:
        print("  [WARN] holdout no disponible para simulación OOS")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SIMULADOR RETROSPECTIVO DE ESTRATEGIAS DE CALIBRACIÓN")
    print("Luna V2 — Sin re-entrenamiento — Usando modelos de la última run")
    print("=" * 65)

    for agent in ["bull", "range", "bear"]:
        simulate_agent(agent)

    print("\n\nCONCLUSIONES:")
    print("  - Estrategia A (ACTUAL): baseline de referencia")
    print("  - Estrategia B (IS-TAIL): viable si EV_penalizado > 0")
    print("  - Estrategia C (WFB-PRIOR): requiere runs previas con EV>0")
    print("  - Estrategia D (MULTI-VAL): robusta si hay ventanas con EV>0")
    print("  - Estrategia E (REGIME-IS): más selectiva, menos sesgo IS")
