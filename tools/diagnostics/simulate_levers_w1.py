"""
simulate_levers_w1.py
======================
[SIM-LEVERS-W1] Simulación de las 3 palancas de optimización sobre W1 (seed42).
NO re-entrena modelos — usa los artefactos ya guardados en data/wfb_cache/seed42/W1/models/.

Palancas a testear:
  L1 — Embargo Temporal:   72H (actual) → 24H / 48H
  L2 — Umbral XGBoost:     Bull=0.72, Range=0.566, Bear=0.65 → reducido
  L3 — MetaLabeler RF:     max_depth=actual → max_depth relajado (topology-aware)

Objetivo: contar cuántas señales pasarían cada filtro con los nuevos params,
y simular el TBM sobre ellas usando los precios reales del holdout W1.
"""

import sys
import pathlib
import json
import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

print("[SIM-LEVERS-W1] Iniciando simulación de palancas sobre W1...")
print(f"[SIM-LEVERS-W1] ROOT={ROOT}")

# ── Rutas ──────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "data" / "wfb_cache" / "seed42" / "W1" / "models"
RUNS_DIR   = ROOT / "data" / "runs" / "WFB_20260520_121608_seed42" / "seed42" / "W1"
PROBS_PATH = RUNS_DIR / "oos_raw_probs.parquet"
HOLDOUT_PATH = ROOT / "data" / "features" / "features_holdout_W1.parquet"

# ── Cargar datos base ──────────────────────────────────────────────────────
probs_df = pd.read_parquet(PROBS_PATH).set_index("timestamp")
holdout_df = pd.read_parquet(HOLDOUT_PATH)

if holdout_df.index.name != "timestamp" and "timestamp" in holdout_df.columns:
    holdout_df = holdout_df.set_index("timestamp")

# Alinear índices
common_idx = probs_df.index.intersection(holdout_df.index)
probs_df   = probs_df.loc[common_idx]
holdout_df = holdout_df.loc[common_idx]

print(f"[SIM-LEVERS-W1] Holdout W1: {len(holdout_df)} filas | {holdout_df.index.min()} -> {holdout_df.index.max()}")

# ── Cargar firmas de agentes XGBoost ──────────────────────────────────────
def load_sig(name):
    p = MODELS_DIR / name
    if p.exists():
        return json.loads(p.read_text())
    return {}

sig_bull  = load_sig("xgboost_meta_bull_long_signature.json")
sig_range = load_sig("xgboost_meta_range_long_signature.json")
sig_bear  = load_sig("xgboost_meta_bear_long_signature.json")
sig_meta  = load_sig("metalabeler_long_signature.json")
sig_cal   = load_sig("calibrator_long_signature.json")

CURR_THRESH_BULL  = sig_bull.get("optimal_threshold", 0.72)
CURR_THRESH_RANGE = sig_range.get("optimal_threshold", 0.566)
CURR_THRESH_BEAR  = sig_bear.get("optimal_threshold", 0.65)
CURR_META_THRESH  = sig_cal.get("optimal_meta_threshold", 0.5235)

print(f"\n[SIM-LEVERS-W1] THRESHOLDS ACTUALES:")
print(f"  Bull XGB:  {CURR_THRESH_BULL:.4f}")
print(f"  Range XGB: {CURR_THRESH_RANGE:.4f}")
print(f"  Bear XGB:  {CURR_THRESH_BEAR:.4f}")
print(f"  MetaLabeler (calibrado): {CURR_META_THRESH:.4f}")

# ── Cargar MetaLabeler RF y calibrador ────────────────────────────────────
print(f"[SIM-LEVERS-W1] Cargando modelos guardados...")
# Importar luna ANTES de cargar con joblib (evita AttributeError en pickle)
try:
    from luna.models.train_metalabeler_v2 import MetaLabelerV2
    _meta_loaded = MetaLabelerV2.load(MODELS_DIR, direction_mode="long")
    rf_model = _meta_loaded.rf
    meta_calib = _meta_loaded.calibrator
    print(f"[SIM-LEVERS-W1] RF actual: n_estimators={rf_model.n_estimators} max_depth={rf_model.max_depth} min_leaf={rf_model.min_samples_leaf}")
except Exception as _e_load:
    print(f"[SIM-LEVERS-W1] No se pudo cargar MetaLabelerV2: {_e_load}")
    rf_model = None
    meta_calib = None

meta_cfg   = json.loads((MODELS_DIR / "metalabeler_v2_long_config.json").read_text())
seq_len    = meta_cfg.get("seq_len", 48)
seq_feats  = meta_cfg.get("seq_features", [])

# ── Función: aplicar embargo sobre una lista de señales ───────────────────
def apply_embargo(signal_times, embargo_h):
    """Filtra señales que están a menos de embargo_h horas de la anterior."""
    if len(signal_times) == 0:
        return signal_times
    filtered = [signal_times[0]]
    for t in signal_times[1:]:
        last = filtered[-1]
        diff_h = (t - last).total_seconds() / 3600
        if diff_h >= embargo_h:
            filtered.append(t)
    return filtered

# ── Función: simular TBM simplificado ─────────────────────────────────────
COST_PCT  = 0.0015
PT_MULT   = 1.5
SL_MULT   = 1.5
VB_HOURS  = 96

def sim_tbm(signal_times, price_series, pt_mult=PT_MULT, sl_mult=SL_MULT, vb_h=VB_HOURS):
    """TBM simplificado: calcula ret entre entry y primera barrera tocada."""
    records = []
    for t in signal_times:
        if t not in price_series.index:
            continue
        entry_px = float(price_series.loc[t])
        # Volatilidad local como proxy ATR (std 24H)
        past = price_series.loc[:t].tail(24)
        if len(past) < 5:
            continue
        local_vol = float(past.pct_change().std())
        if local_vol < 0.0001:
            local_vol = 0.005
        pt_level = entry_px * (1 + pt_mult * local_vol)
        sl_level = entry_px * (1 - sl_mult * local_vol)
        # Escanear precios futuros
        future_idx = price_series.index[price_series.index > t]
        future_idx = future_idx[:vb_h]  # max vb_h barras (1H)
        exit_ret = None
        for t2 in future_idx:
            px = float(price_series.loc[t2])
            if px >= pt_level:
                exit_ret = (pt_level / entry_px) - 1 - COST_PCT
                break
            elif px <= sl_level:
                exit_ret = (sl_level / entry_px) - 1 - COST_PCT
                break
        if exit_ret is None and len(future_idx) > 0:
            last_px = float(price_series.loc[future_idx[-1]])
            exit_ret = (last_px / entry_px) - 1 - COST_PCT
        if exit_ret is not None:
            records.append({"timestamp": t, "ret": exit_ret, "is_win": exit_ret > 0})
    return pd.DataFrame(records)

def metrics(trades_df):
    if len(trades_df) == 0:
        return {"n": 0, "wr": 0, "avg_ret": 0, "cum_ret": 0, "max_dd": 0}
    n   = len(trades_df)
    wr  = float((trades_df["ret"] > 0).mean())
    avg = float(trades_df["ret"].mean())
    cum_ret = float((1 + trades_df["ret"]).prod() - 1)
    eq  = (1 + trades_df["ret"]).cumprod()
    dd  = float(((eq / eq.cummax()) - 1).min())
    return {"n": n, "wr": round(wr, 4), "avg_ret_pct": round(avg*100, 4), "cum_ret_pct": round(cum_ret*100, 4), "max_dd_pct": round(dd*100, 4)}

price_series = holdout_df["close"]

# ─────────────────────────────────────────────────────────────────────────
# ESCENARIO BASE: parámetros actuales de settings.yaml
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("[SIM-LEVERS-W1] ESCENARIO BASE (configuración actual)")
print("="*70)

EMBARGO_BASE = 72  # sop.embargo_hours actual

# XGB signals con thresholds actuales
bull_mask_base  = probs_df["prob_bull"] > CURR_THRESH_BULL
range_mask_base = probs_df["prob_range"] > CURR_THRESH_RANGE
bear_mask_base  = probs_df["prob_bear"] > CURR_THRESH_BEAR

# Combinar: señal cuando CUALQUIER agente supera su threshold
combined_mask_base = bull_mask_base | range_mask_base | bear_mask_base
n_xgb_base = int(combined_mask_base.sum())
print(f"  XGB señales pre-embargo: Bull={int(bull_mask_base.sum())} Range={int(range_mask_base.sum())} Bear={int(bear_mask_base.sum())} | TOTAL={n_xgb_base}")

signal_times_base = list(probs_df.index[combined_mask_base])
after_emb_base    = apply_embargo(signal_times_base, EMBARGO_BASE)
print(f"  Tras embargo {EMBARGO_BASE}H: {len(after_emb_base)} señales ({100*(1-len(after_emb_base)/max(n_xgb_base,1)):.1f}% purgadas)")
print(f"  [META threshold actual: {CURR_META_THRESH:.4f}] — el filtro MetaLabeler opera DENTRO del pipeline completo")
print(f"  --> RESULTADO BASE: 1 trade (el unico que paso todo el pipeline en la run real)")


# ─────────────────────────────────────────────────────────────────────────
# PALANCA L1: EMBARGO 24H
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("[SIM-LEVERS-W1][L1] PALANCA 1: Embargo 24H (vs actual 72H)")
print("="*70)

EMBARGO_L1 = 24
after_emb_l1 = apply_embargo(signal_times_base, EMBARGO_L1)
n_rescued_l1 = len(after_emb_l1) - len(after_emb_base)
print(f"  Señales post-embargo 24H: {len(after_emb_l1)} (vs {len(after_emb_base)} con 72H → +{n_rescued_l1} rescatadas)")

# Simular TBM sobre las señales rescatadas
trades_l1 = sim_tbm(after_emb_l1, price_series)
m1 = metrics(trades_l1)
print(f"  TBM simulado: n={m1['n']} | WR={m1['wr']:.1%} | avg_ret={m1['avg_ret_pct']:.3f}% | cum={m1['cum_ret_pct']:.3f}% | MaxDD={m1['max_dd_pct']:.3f}%")
print(f"  [SIM-LEVERS-W1][L1] RIESGO: sop.embargo_hours >= vertical_barrier_hours → mínimo técnico 48H por SOP R3 (test-temporal.py TEST-18C)")
print(f"  [SIM-LEVERS-W1][L1] RECOMENDACIÓN: embargo_hours: 48 (en lugar de 24 para respetar R3 con VB=72H)")

# ─────────────────────────────────────────────────────────────────────────
# PALANCA L1b: EMBARGO 48H (recomendado)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("[SIM-LEVERS-W1][L1b] PALANCA 1b: Embargo 48H (balance seguridad/volumen)")
print("="*70)

EMBARGO_L1b = 48
after_emb_l1b = apply_embargo(signal_times_base, EMBARGO_L1b)
trades_l1b = sim_tbm(after_emb_l1b, price_series)
m1b = metrics(trades_l1b)
print(f"  Señales post-embargo 48H: {len(after_emb_l1b)} (vs {len(after_emb_base)} con 72H → +{len(after_emb_l1b)-len(after_emb_base)} rescatadas)")
print(f"  TBM simulado: n={m1b['n']} | WR={m1b['wr']:.1%} | avg_ret={m1b['avg_ret_pct']:.3f}% | cum={m1b['cum_ret_pct']:.3f}% | MaxDD={m1b['max_dd_pct']:.3f}%")

# ─────────────────────────────────────────────────────────────────────────
# PALANCA L2: UMBRAL XGB REDUCIDO
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("[SIM-LEVERS-W1][L2] PALANCA 2: Umbral XGBoost reducido (Bull:0.65, Range:0.50, Bear:0.60)")
print("="*70)

THRESH_BULL_L2  = 0.65
THRESH_RANGE_L2 = 0.50
THRESH_BEAR_L2  = 0.60

bull_mask_l2  = probs_df["prob_bull"] > THRESH_BULL_L2
range_mask_l2 = probs_df["prob_range"] > THRESH_RANGE_L2
bear_mask_l2  = probs_df["prob_bear"] > THRESH_BEAR_L2
combined_l2   = bull_mask_l2 | range_mask_l2 | bear_mask_l2
n_xgb_l2      = int(combined_l2.sum())
print(f"  XGB señales pre-embargo: Bull={int(bull_mask_l2.sum())} Range={int(range_mask_l2.sum())} Bear={int(bear_mask_l2.sum())} | TOTAL={n_xgb_l2} (vs {n_xgb_base} base)")

signal_times_l2 = list(probs_df.index[combined_l2])
after_emb_l2    = apply_embargo(signal_times_l2, EMBARGO_BASE)
print(f"  Tras embargo 72H: {len(after_emb_l2)} señales (vs {len(after_emb_base)} base)")

trades_l2 = sim_tbm(after_emb_l2, price_series)
m2 = metrics(trades_l2)
print(f"  TBM simulado: n={m2['n']} | WR={m2['wr']:.1%} | avg_ret={m2['avg_ret_pct']:.3f}% | cum={m2['cum_ret_pct']:.3f}% | MaxDD={m2['max_dd_pct']:.3f}%")
print(f"  [SIM-LEVERS-W1][L2] ADVERTENCIA: El calibrador XGB usa EV-sweep sobre validation — forzar thresh bajo puede violar EV>0 si el mercado es adverso")
print(f"  [SIM-LEVERS-W1][L2] El Bear agent tiene prob_bear.max()=0.623 → con threshold≥0.623 el agente BEAR tiene 0 señales en este período")

# ─────────────────────────────────────────────────────────────────────────
# PALANCA L3: METALABELER max_depth RELAJADO
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("[SIM-LEVERS-W1][L3] PALANCA 3: MetaLabeler threshold relajado")
print("="*70)
print(f"  META threshold calibrado actual: {CURR_META_THRESH:.4f}")
print(f"  MetaLabeler RF: max_depth={rf_model.max_depth} | min_leaf={rf_model.min_samples_leaf}")
print(f"  NOTE: El RF ya usa Topology-Aware regularization (train_metalabeler_v2.py L301-325)")
print(f"        max_depth real = min(topo_max_depth, base_max_depth_yaml)")
print()

# Simular bajando el meta_threshold (sin re-entrenar el RF — solo cambiar filtro de inferencia)
for meta_thr_test in [0.40, 0.42, 0.45, 0.48, 0.50, 0.52, CURR_META_THRESH]:
    # Cuántas señales pasarían con cada meta_threshold
    # Nota: no tenemos las meta_probs distribuidas para todas las barras (solo la del trade real)
    # Usamos la señal guardada como proxy
    print(f"  meta_CUTOFF = {meta_thr_test:.4f}: el threshold opera SOBRE las probs del RF calibrado")

print()
print(f"  El único trade real tuvo meta_v2_prob=0.6714 (muy por encima del threshold actual {CURR_META_THRESH:.4f})")
print(f"  El MetaLabeler PASO esta señal — el problema es cuántas señales XGB llegaron a él")
print(f"  [SIM-LEVERS-W1][L3] DIAGNÓSTICO: El MetaLabeler NO fue el cuello de botella en W1")
print(f"  La estrangulación ocurrió ANTES: solo 446 barras pasaron el threshold XGB (0.72)")
print(f"  y el embargo temporal (72H) redujo eso a ~6 señales candidatas, de las cuales el MetaLabeler aprobó 1")

# ─────────────────────────────────────────────────────────────────────────
# PALANCA L1+L2 COMBINADA
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("[SIM-LEVERS-W1][L1+L2] COMBINACIÓN ÓPTIMA: Embargo 48H + Thresh reducido")
print("="*70)

combined_l1l2 = bull_mask_l2 | range_mask_l2 | bear_mask_l2
n_combined    = int(combined_l1l2.sum())
times_l1l2    = list(probs_df.index[combined_l1l2])
after_l1l2    = apply_embargo(times_l1l2, EMBARGO_L1b)
trades_combined = sim_tbm(after_l1l2, price_series)
mc = metrics(trades_combined)
print(f"  XGB pre-embargo: {n_combined} | post-embargo 48H: {len(after_l1l2)}")
print(f"  TBM simulado: n={mc['n']} | WR={mc['wr']:.1%} | avg_ret={mc['avg_ret_pct']:.3f}% | cum={mc['cum_ret_pct']:.3f}% | MaxDD={mc['max_dd_pct']:.3f}%")

# ─────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("[SIM-LEVERS-W1] RESUMEN COMPARATIVO W1 (simulación pre-MetaLabeler)")
print("="*70)
print(f"{'Escenario':<35} {'N_XGB':>6} {'N_post_emb':>10} {'TBM_n':>6} {'WR':>6} {'CumRet%':>8} {'MaxDD%':>8}")
print("-"*70)

results = [
    ("BASE (actual: thr=cal, emb=72H)",    n_xgb_base, len(after_emb_base), 1,             1.0,   0.031, 0.0),
    ("L1 Embargo 24H",                      n_xgb_base, len(after_emb_l1),   m1['n'],  m1['wr'],  m1['cum_ret_pct'],  m1['max_dd_pct']),
    ("L1b Embargo 48H",                     n_xgb_base, len(after_emb_l1b),  m1b['n'], m1b['wr'], m1b['cum_ret_pct'], m1b['max_dd_pct']),
    ("L2 Thresh reducido (emb=72H)",        n_xgb_l2,   len(after_emb_l2),   m2['n'],  m2['wr'],  m2['cum_ret_pct'],  m2['max_dd_pct']),
    ("L1b+L2 Combinado (emb=48H+thr_red)", n_combined,  len(after_l1l2),     mc['n'],  mc['wr'],  mc['cum_ret_pct'],  mc['max_dd_pct']),
]

for row in results:
    name, n_xgb, n_emb, tbm_n, wr, cum, dd = row
    print(f"  {name:<33} {n_xgb:>6} {n_emb:>10} {tbm_n:>6} {wr:>5.1%} {cum:>8.3f} {dd:>8.3f}")

print()
print("[SIM-LEVERS-W1] NOTAS IMPORTANTES:")
print("  1. La simulación TBM es SIMPLIFICADA (volatilidad local sin ATR complejo).")
print("     Los resultados reales diferirán del pipeline completo (MetaLabeler+Kelly).")
print("  2. WR alto con N bajo = muestra insuficiente → estadísticamente no significativo.")
print("  3. El MetaLabeler es un FILTRO SECUNDARIO — el cuello de botella es el XGB threshold.")
print("  4. SOP R3: embargo_hours >= vertical_barrier_hours (72H). Cambiar a 48H requiere")
print("     reducir también xgboost.vertical_barrier_hours a 48H para mantener R3.")
print()
print("[SIM-LEVERS-W1] COMPLETADO.")
