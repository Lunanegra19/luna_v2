# -*- coding: utf-8 -*-
"""
Investigacion profunda de como arreglar XGBoost descalibrado.
Tests:
  1. Brier Score actual por regimen (cuantificar la descalibracion)
  2. Simulacion de sample_weights por regimen
  3. Simulacion de optuna_metric = brier (en lugar de dsr)
  4. Simulacion de regularizacion mas fuerte (max_depth, min_child_weight)
  5. Impacto de cada intervencion en seeds positivas y Sharpe
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as sc

reports_dir = Path("data/reports/wfb")
seeds = [int(f.stem.split("seed")[1]) for f in reports_dir.glob("oos_trades_W5_seed*.parquet")]
COMM = 0.0015
VOLATILE = ["1_VOLATILE_BULL", "2_VOLATILE_RANGE"]

all_trades = []
for seed in seeds:
    files = sorted(reports_dir.glob(f"oos_trades_W*_seed{seed}.parquet"))
    if len(files) == 0:
        continue
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df["seed"] = seed
        all_trades.append(df)
df_all = pd.concat(all_trades, ignore_index=True)


def portfolio_metrics(df):
    if len(df) == 0:
        return {"n": 0, "wr": 0, "pb": 0, "pn": 0, "sh": 0, "mdd": 0}
    n = len(df)
    wr = (df["return_raw"] > 0).mean() * 100
    pb = df["return_raw"].sum() * 100
    pn = pb - COMM * n * 100
    ret_k = (df["return_raw"] - COMM) * df["kelly_fraction_used"].fillna(0.043)
    sh = ret_k.mean() / ret_k.std() * (252 * 24) ** 0.5 if ret_k.std() > 0 else 0
    eq = (1 + ret_k).cumprod()
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    seeds_pos = sum(
        1 for s in df["seed"].unique()
        if len(df[df["seed"] == s]) > 0
        and df[df["seed"] == s]["return_raw"].sum() * 100
        - COMM * len(df[df["seed"] == s]) * 100 > 0
    )
    return {"n": n, "wr": wr, "pb": pb, "pn": pn, "sh": sh, "mdd": mdd, "seeds_pos": seeds_pos}


# ────────────────────────────────────────────────────────────────────
print("=" * 75)
print("  TEST 1: BRIER SCORE ACTUAL POR REGIMEN (Cuantificar descalibracion)")
print("  Brier perfecto = 0. Brier aleatorio = 0.25. Brier sobreconfiado > 0.25")
print("=" * 75)
print()
print(f"{'Regimen':<25} {'n':>5} {'xgb_mean':>9} {'WR_real':>9} {'Brier_xgb':>11} {'Brier_meta':>12} {'Gap_cal':>9}")
print("-" * 85)

brier_by_regime = {}
for regime in df_all["hmm_regime"].value_counts().index:
    df_r = df_all[df_all["hmm_regime"] == regime].copy()
    y_true = (df_r["return_raw"] > 0).astype(float).values
    xgb_p = df_r["xgb_prob_cal"].values
    meta_p = df_r["meta_v2_prob"].values

    brier_xgb = float(np.mean((xgb_p - y_true) ** 2))
    brier_meta = float(np.mean((meta_p - y_true) ** 2))
    brier_random = float(np.mean((np.full_like(y_true, 0.5) - y_true) ** 2))
    wr = y_true.mean()
    gap = xgb_p.mean() - wr

    brier_by_regime[regime] = brier_xgb
    print(
        f"{regime:<25} {len(df_r):>5} {xgb_p.mean():>9.3f} {wr*100:>8.1f}% "
        f"{brier_xgb:>11.4f} {brier_meta:>12.4f} {gap:>9.3f}"
    )

print()
print(f"  Brier_random = 0.2500 (predictor que siempre dice 50%)")
print(f"  Brier_xgb > 0.25 en TODOS los regimenes = xgb PEOR que random")

# ────────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("  TEST 2: SIMULACION SAMPLE WEIGHTS - Efecto teorico")
print("  Idea: penalizar trades VOLATILE en el loss de XGBoost durante train")
print("  Efecto esperado: modelo aprende a desconfiar de sus predicciones VOLATILE")
print("  Simulacion: reducir el impacto de los trades VOLATILE en el portfolio")
print("  (equivale a lo que sample_weight hace en training)")
print("=" * 75)
print()
# Simulacion de SW via post-hoc: reducir Kelly en VOLATILE segun peso
print(f"{'SW_volatile':>12} {'SW_ok':>8} {'Trades':>7} {'Seeds+':>7} {'PnL_neto':>10} {'Sharpe':>8} {'MaxDD':>8}")
print("-" * 70)
for sw_vol in [0.0, 0.25, 0.50, 0.75, 1.0]:
    df_sim = df_all.copy()
    is_vol = df_sim["hmm_regime"].isin(VOLATILE)
    df_sim.loc[is_vol, "kelly_fraction_used"] *= sw_vol
    # Solo trades con kelly > 0 (los que realmente se operan)
    df_active = df_sim[df_sim["kelly_fraction_used"] > 0.001]
    m = portfolio_metrics(df_active)
    print(
        f"{sw_vol:>12.2f} {1.0:>8.2f} {m['n']:>7} {str(m.get('seeds_pos','?'))+'/12':>7} "
        f"{m['pn']:>9.2f}% {m['sh']:>8.3f} {m['mdd']:>7.1f}%"
    )

# ────────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("  TEST 3: SIMULACION REGULARIZACION FUERTE EN XGB")
print("  Idea: min_child_weight alto = XGB necesita mas datos por hoja")
print("       max_depth bajo = arboles mas simples, menos overfitting")
print("  Efecto simulado: eliminar trades donde xgb_prob > 0.80 (zona de overfitting)")
print("  Los datos muestran que prob>0.80 tiene WR=42-53%, peor que prob=0.70")
print("=" * 75)
print()
print("Recordatorio del reliability diagram:")
print("  prob (0.76-0.80): WR=48.5%  <- PEOR que prob=0.68")
print("  prob (0.80-0.84): WR=42.9%  <- El modelo mas confiado, peor resultado")
print("  prob (0.84-0.93): WR=53%    <- Recupera algo")
print()

# Simular cap de probabilidad: ignorar trades donde xgb_prob > cap
print(f"{'xgb_cap':>9} {'Trades':>7} {'Seeds+':>7} {'PnL_neto':>10} {'avg_ret':>8} {'WR':>7}")
print("-" * 60)
for xgb_cap in [0.78, 0.76, 0.74, 0.72, 0.70, "none"]:
    if xgb_cap == "none":
        df_t = df_all[df_all["meta_v2_prob"] >= 0.705]
    else:
        df_t = df_all[
            (df_all["meta_v2_prob"] >= 0.705) & (df_all["xgb_prob_cal"] <= xgb_cap)
        ]
    n = len(df_t)
    if n == 0:
        continue
    pn = df_t["return_raw"].sum() * 100 - COMM * n * 100
    avg = df_t["return_raw"].mean() * 100
    wr = (df_t["return_raw"] > 0).mean() * 100
    pos = sum(
        1 for s in seeds
        if len(df_t[df_t["seed"] == s]) > 0
        and df_t[df_t["seed"] == s]["return_raw"].sum() * 100
        - COMM * len(df_t[df_t["seed"] == s]) * 100 > 0
    )
    label = str(xgb_cap) if xgb_cap != "none" else "sin_cap"
    print(f"{label:>9} {n:>7} {str(pos)+'/12':>7} {pn:>9.2f}% {avg:>7.4f}% {wr:>6.1f}%")

# ────────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("  TEST 4: COMBINACION OPTIMA - meta_v2_prob >= 0.705 + xgb_cap")
print("  Buscar si el cap de xgb_prob_cal anade valor sobre el threshold meta solo")
print("=" * 75)
print()

# Baseline con solo meta threshold
df_meta_only = df_all[df_all["meta_v2_prob"] >= 0.705]
print(f"Solo meta_v2_prob>=0.705: n={len(df_meta_only)} | PnL_neto={df_meta_only['return_raw'].sum()*100 - COMM*len(df_meta_only)*100:.2f}%")
print()

# Verificar correlacion entre xgb_prob alto y malos trades
df_hi_xgb = df_all[df_all["xgb_prob_cal"] > 0.78]
df_lo_xgb = df_all[df_all["xgb_prob_cal"] <= 0.78]
t, p = sc.ttest_ind(df_hi_xgb["return_raw"], df_lo_xgb["return_raw"], equal_var=False)
print(f"xgb_prob>0.78 vs <=0.78: avg_hi={df_hi_xgb['return_raw'].mean()*100:.4f}% avg_lo={df_lo_xgb['return_raw'].mean()*100:.4f}%")
print(f"Test t: t={t:.3f} p={p:.4f} -> {'SIGNIFICATIVO' if p<0.05 else 'NO significativo'}")
print()
print("CONCLUSION:")
if p < 0.05:
    print("  xgb_prob alto SI predice peores retornos -- cap de xgb_prob TIENE VALOR")
else:
    print("  xgb_prob alto NO predice peores retornos -- cap de xgb_prob NO APORTA")
    print("  El xgb_prob es basicamente ruido aleatorio -- mejor ignorarlo en el gate")

# ────────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("  TEST 5: DIAGNOSTICO DE LA CAUSA RAIZ DEL XGB DESCALIBRADO")
print("=" * 75)
print()
print("El XGB usa binary:logistic con Brier Score + DSR como metrica de Optuna.")
print("Con Focal Loss activo, el objetivo custom devuelve gradientes de Focal Loss")
print("pero post-fit restaura binary:logistic para predict_proba.")
print()
print("Hipotesis sobre la causa de la descalibracion:")
print()
print("1. FOCAL LOSS + restauracion post-fit:")
print("   Los arboles se construyen con gradientes de Focal Loss (penaliza ejemplos faciles).")
print("   La restauracion a binary:logistic post-fit cambia la interpretacion de los")
print("   raw_margins de los arboles. El output de predict_proba ya no es una probabilidad")
print("   calibrada sino un raw_margin reinterpretado via sigmoid.")
print("   -> Esto explicaria por que xgb_prob no tiene poder predictivo")
print()
print("2. OPTUNA optimiza DSR (Sharpe), no calibracion:")
print("   Maximizar DSR en CPCV selecciona el modelo con MAYOR Sharpe en IS,")
print("   no el mejor calibrado. Un modelo sobreentrenado puede tener DSR alto en IS")
print("   pero mala calibracion en OOS.")
print()
print("3. SFI feature selection sesgada:")
print("   SFI (Single Feature Importance) puede seleccionar features que correlacionan")
print("   con el movimiento futuro en IS pero que son ilusiones estadisticas.")
print()

# Ver el numero de features que entran al XGBoost
sig_files = list(Path("data/wfb_cache").glob("seed*/W5/models/xgboost_meta_bull_long_signature.json"))
if sig_files:
    import json
    sig = json.loads(sig_files[0].read_text())
    n_features = len(sig.get("features", []))
    opt_thr = sig.get("optimal_threshold", "N/A")
    best_params = sig.get("params", {})
    print(f"Firma XGBoost W5 (seed {sig_files[0].parts[-4]}):")
    print(f"  n_features = {n_features}")
    print(f"  optimal_CUTOFF = {opt_thr}")
    print(f"  max_depth = {best_params.get('max_depth', 'N/A')}")
    print(f"  n_estimators = {best_params.get('n_estimators', 'N/A')}")
    print(f"  min_child_weight = {best_params.get('min_child_weight', 'N/A')}")
    print(f"  reg_alpha = {best_params.get('reg_alpha', 'N/A')}")
    print(f"  reg_lambda = {best_params.get('reg_lambda', 'N/A')}")
    print(f"  scale_pos_weight = {best_params.get('scale_pos_weight', 'N/A')}")

# ────────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("  RESUMEN: RANKING DE SOLUCIONES PARA XGBOOST")
print("=" * 75)
print()
soluciones = [
    ("1. [YA HECHO] Subir meta_v2_prob threshold a 0.705",
     "Inmediato, sin reentrenar. 9/12 seeds positivas en simulacion.",
     "IMPLEMENTADO"),
    ("2. Cambiar optuna_metric a 'brier' en lugar de 'dsr'",
     "Optuna buscaria el modelo mejor calibrado, no el de mayor Sharpe IS.",
     "EN SETTINGS.YAML: optuna_metric: brier"),
    ("3. Desactivar Focal Loss o usarlo correctamente",
     "El Focal Loss + restauracion post-fit rompe la calibracion.",
     "EN SETTINGS.YAML: use_focal_loss: false"),
    ("4. Sample weights: SW=0 en VOLATILE durante training",
     "Modelo no aprende de trades que luego no operamos.",
     "EN SETTINGS.YAML: regime_sample_weights (nueva feature)"),
    ("5. Subir min_child_weight en xgboost optuna_search_space",
     "Arboles con mas datos por hoja = menos overfitting.",
     "min_child_weight_min: 30, min_child_weight_max: 100"),
]
for title, desc, impl in soluciones:
    print(f"  {title}")
    print(f"    {desc}")
    print(f"    Implementacion: {impl}")
    print()
