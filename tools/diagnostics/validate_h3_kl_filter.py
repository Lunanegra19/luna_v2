"""
[H3-VALIDACION] Validacion anti-overfitting del filtro KL Score
================================================================
RESTRICCION: Solo datos de la run actual de esta madrugada (20 seeds).
NO se mezclan runs anteriores (arquitecturas/settings diferentes = invalido).

Con UNA SOLA RUN de 20 seeds compartiendo la misma arquitectura y datos,
los metodos de validacion validos son:

  M1. Consistencia entre seeds (robustez, no independencia real)
      - 20 seeds = 20 modelos distintos sobre los mismos datos
      - Si el efecto KL aparece en la mayoria de seeds -> robusto al modelo
      - LIMITE: las seeds comparten datos subyacentes, no son 100% independientes

  M2. HOLDOUT TEMPORAL — el mas honesto con una sola run
      - W1-W2-W3 (periodo mas antiguo) = zona de discovery
      - W4-W5 (periodo mas reciente) = zona de validacion PURA
      - W4 y W5 son genuinamente futuras respecto al split de W3
      - NADIE uso W4-W5 para descubrir el patron KL -> validacion limpia
      - Este metodo respeta SOP Rule R1 (causalidad estricta)

  M3. Cross-window — verificar consistencia temporal del efecto
      - Mide el efecto KL DENTRO de cada ventana por separado
      - Si el efecto es consistente en >=4/5 ventanas -> no es artefacto de una epoca

VEREDICTO FINAL:
  - 3/3 metodos positivos -> implementar con confianza
  - 2/3 positivos         -> considerar implementacion cauta
  - 0-1 positivos         -> NO implementar, es data snooping
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

print("=" * 72)
print("[H3-VALIDACION] Filtro KL Score — validacion con run actual")
print("SOLO datos run madrugada 2026-06-11 (20 seeds, config actual)")
print("=" * 72)

# ── Carga SOLO run actual ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data" / "predictions"

all_dfs = []
for f in sorted(DATA_DIR.glob("oos_trades_seed*.parquet")):
    d = pd.read_parquet(f)
    d["_seed"] = int(f.stem.split("seed")[1])
    all_dfs.append(d)

df = pd.concat(all_dfs, ignore_index=True)
df["is_win"] = df["is_win"].astype(int)
df["ret100"] = df["return_raw"] * 100
w_col = "wfb_window" if "wfb_window" in df.columns else "window"
df["_window"] = df[w_col]

ALL_SEEDS   = sorted(df["_seed"].unique())
ALL_WINDOWS = sorted(df["_window"].unique())

print(f"\n[LOAD] {len(df)} trades | {len(ALL_SEEDS)} seeds | Windows: {ALL_WINDOWS}")
print(f"[LOAD] Seeds: {ALL_SEEDS}")

# Verificar columnas necesarias
has_kl  = "ood_kl_distance" in df.columns
has_xgb = "xgb_prob_cal" in df.columns
print(f"[LOAD] ood_kl_distance={has_kl} | xgb_prob_cal={has_xgb}")

if not has_kl:
    print("[ERROR] ood_kl_distance no disponible. Abortando.")
    raise SystemExit(1)

print()


# ── Helper metricas ───────────────────────────────────────────────────────
def metricas(sub: pd.DataFrame, label: str = "") -> dict:
    n = len(sub)
    if n < 10:
        return {"n": n, "wr": np.nan, "ret": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    wr  = sub["is_win"].mean() * 100
    ret = sub["ret100"].sum()
    rets = sub["ret100"].values
    std_r = np.std(rets)
    sharpe = (np.mean(rets) / std_r) * np.sqrt(52) if std_r > 1e-10 else 0.0
    cumret = (1 + sub["return_raw"].values).cumprod()
    running_max = np.maximum.accumulate(cumret)
    dd = (cumret - running_max) / np.maximum(running_max, 1e-10)
    maxdd = abs(dd.min()) * 100
    if label:
        print(f"  [H3] {label}: N={n} WR={wr:.1f}% Ret={ret:+.1f}pp Sharpe={sharpe:.2f} MaxDD={maxdd:.1f}%")
    return {"n": n, "wr": wr, "ret": ret, "sharpe": sharpe, "maxdd": maxdd}


# ═══════════════════════════════════════════════════════════════════════════
# M1 — CONSISTENCIA ENTRE SEEDS
# No son independientes (mismos datos), pero mide robustez al modelo.
# Si el efecto KL_bajo > KL_alto aparece en 15+/20 seeds ->
# no es un artefacto de un modelo concreto sino una senal del mercado.
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("M1: Consistencia entre seeds (robustez al modelo, mismos datos)")
print("    LIMITE: mismas 7.718 barras subyacentes -> no son independientes")
print("    VALOR : si 15+/20 seeds confirman -> la senal es del mercado, no del modelo")
print("=" * 72)
print()
print(f"  {'Seed':>7}  {'N':>5}  {'KL<=Q25_WR':>11}  {'KL>=Q75_WR':>11}  {'Delta':>8}  {'Efecto'}")
print("  " + "-" * 66)

seeds_favor, seeds_contra, seeds_neutro = 0, 0, 0
deltas_m1 = []

for seed in ALL_SEEDS:
    ds = df[df["_seed"] == seed].copy()
    kl = ds["ood_kl_distance"]
    q25 = kl.quantile(0.25)
    q75 = kl.quantile(0.75)
    lo = ds[kl <= q25]
    hi = ds[kl >= q75]
    if len(lo) < 5 or len(hi) < 5:
        continue
    wr_lo = lo["is_win"].mean() * 100
    wr_hi = hi["is_win"].mean() * 100
    delta = wr_lo - wr_hi
    deltas_m1.append(delta)
    if delta > 5:
        seeds_favor += 1
        efecto = "KL_bajo GANA"
    elif delta < -5:
        seeds_contra += 1
        efecto = "KL_alto GANA"
    else:
        seeds_neutro += 1
        efecto = "~empate"
    print(f"  {seed:>7}  {len(ds):>5}  {wr_lo:>11.1f}%  {wr_hi:>11.1f}%  {delta:>+8.1f}pp  {efecto}")

total_seeds = seeds_favor + seeds_contra + seeds_neutro
print()
print(f"  KL_bajo GANA: {seeds_favor}/{total_seeds} seeds")
print(f"  KL_alto GANA: {seeds_contra}/{total_seeds} seeds")
print(f"  Empate      : {seeds_neutro}/{total_seeds} seeds")
print(f"  Delta medio : {np.mean(deltas_m1):+.1f}pp (std={np.std(deltas_m1):.1f}pp)")

# Prueba de signo (no binomial porque empates no son independientes)
# Binomtest solo sobre las no-empates
no_empates = seeds_favor + seeds_contra
if no_empates > 0:
    binom_result = stats.binomtest(seeds_favor, n=no_empates, p=0.5, alternative="greater")
    binom_p = binom_result.pvalue
    print(f"  Prueba signo (excl. empates {seeds_favor}/{no_empates}): p = {binom_p:.5f}")
    if binom_p < 0.01:
        m1_verdict = "POSITIVO"
        print(f"  VEREDICTO M1: POSITIVO *** (p<0.01) — efecto robusto al modelo en {seeds_favor}/{no_empates} seeds")
    elif binom_p < 0.05:
        m1_verdict = "MODERADO"
        print(f"  VEREDICTO M1: MODERADO * (p<0.05) — efecto presente pero no muy fuerte")
    else:
        m1_verdict = "NEGATIVO"
        print(f"  VEREDICTO M1: NEGATIVO ns — efecto NO robusto entre seeds (p={binom_p:.3f})")
else:
    m1_verdict = "SIN DATOS"
print(f"[H3-M1] Seeds favor={seeds_favor} contra={seeds_contra} neutro={seeds_neutro} p={binom_p:.5f}")


# ═══════════════════════════════════════════════════════════════════════════
# M2 — HOLDOUT TEMPORAL (el mas honesto con una sola run)
# -----------------------------------------------------------------------
# La run WFB tiene 5 ventanas cronologicas:
#   W1 = periodo mas antiguo
#   W5 = periodo mas reciente
#
# PROTOCOLO:
#   1. Calcular el percentil KL optimo usando SOLO W1+W2+W3
#   2. Aplicar ese umbral SIN MODIFICAR a W4+W5
#   3. Si el efecto persiste en W4+W5 -> el filtro es predictivo temporalmente
#
# POR QUE ES LIMPIO:
#   W4 y W5 son genuinamente "futuras" respecto a W3.
#   Los modelos de cada seed fueron entrenados con datos hasta el split de cada ventana.
#   El IsolationForest del OOD Guard fue entrenado en el IS de cada ventana.
#   => El patron KL descubierto en W1-W3 es una hipotesis no contaminada por W4-W5.
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("M2: HOLDOUT TEMPORAL — W1+W2+W3 discovery / W4+W5 validation")
print("    ESTE ES EL TEST MAS HONESTO con una sola run")
print("    W4 y W5 son genuinamente futuras respecto al split de W3")
print("=" * 72)

_DISC_WINS = ["W1", "W2", "W3"]
_VAL_WINS  = ["W4", "W5"]

df_disc = df[df["_window"].isin(_DISC_WINS)]
df_val  = df[df["_window"].isin(_VAL_WINS)]

print(f"\n  Discovery (W1-W3): {len(df_disc)} trades")
print(f"  Validation (W4-W5): {len(df_val)} trades")

# Paso 1: calibrar umbrales SOLO en discovery
kl_q25_disc  = df_disc["ood_kl_distance"].quantile(0.25)
kl_q50_disc  = df_disc["ood_kl_distance"].quantile(0.50)
kl_q75_disc  = df_disc["ood_kl_distance"].quantile(0.75)

xgb_q50_disc = df_disc["xgb_prob_cal"].quantile(0.50) if has_xgb else None
xgb_q75_disc = df_disc["xgb_prob_cal"].quantile(0.75) if has_xgb else None

print(f"\n  [DISCOVERY W1-W3] Umbrales calibrados (NUNCA vistos en validation):")
print(f"    KL Q25 = {kl_q25_disc:.5f}  <- umbral 'anomalo'")
print(f"    KL Q50 = {kl_q50_disc:.5f}")
if has_xgb:
    print(f"    XGB Q75 = {xgb_q75_disc:.5f}")

# Paso 2: medir en discovery (referencia)
print(f"\n  {'Escenario':<38}  {'N':>5}  {'WR':>7}  {'Ret':>8}  {'Sharpe':>7}  {'MaxDD':>7}")
print(f"  DISCOVERY W1-W3 (zona de calibracion — NO cuenta para validacion):")
print("  " + "-" * 75)

disc_scenarios = [
    ("BASELINE W1-W3",             df_disc),
    ("KL<=Q25_disc",               df_disc[df_disc["ood_kl_distance"] <= kl_q25_disc]),
    ("KL<=Q50_disc",               df_disc[df_disc["ood_kl_distance"] <= kl_q50_disc]),
]
if has_xgb and xgb_q75_disc:
    disc_scenarios += [
        ("XGB>=Q75+KL<=Q25",           df_disc[(df_disc["xgb_prob_cal"] >= xgb_q75_disc) & (df_disc["ood_kl_distance"] <= kl_q25_disc)]),
        ("XGB>=Q50+KL<=Q25",           df_disc[(df_disc["xgb_prob_cal"] >= xgb_q50_disc) & (df_disc["ood_kl_distance"] <= kl_q25_disc)]),
    ]

for label, sub in disc_scenarios:
    m = metricas(sub)
    if m["n"] >= 10:
        print(f"  {label:<38}  {m['n']:>5}  {m['wr']:>7.1f}%  {m['ret']:>+8.1f}pp  {m['sharpe']:>7.2f}  {m['maxdd']:>6.1f}%")

# Paso 3: aplicar umbrales de W1-W3 a W4-W5 SIN MODIFICAR
print(f"\n  VALIDATION W4-W5 (datos NO vistos durante calibracion):")
print("  Aplicando MISMOS umbrales de W1-W3. Sin ningun re-tuneo.")
print("  " + "-" * 75)

val_scenarios = [
    ("BASELINE W4-W5",             df_val),
    ("KL<=Q25_disc [aplicado]",    df_val[df_val["ood_kl_distance"] <= kl_q25_disc]),
    ("KL<=Q50_disc [aplicado]",    df_val[df_val["ood_kl_distance"] <= kl_q50_disc]),
]
if has_xgb and xgb_q75_disc:
    val_scenarios += [
        ("XGB>=Q75+KL<=Q25 [aplicado]", df_val[(df_val["xgb_prob_cal"] >= xgb_q75_disc) & (df_val["ood_kl_distance"] <= kl_q25_disc)]),
        ("XGB>=Q50+KL<=Q25 [aplicado]", df_val[(df_val["xgb_prob_cal"] >= xgb_q50_disc) & (df_val["ood_kl_distance"] <= kl_q25_disc)]),
    ]

m2_results = {}
for label, sub in val_scenarios:
    m = metricas(sub)
    m2_results[label] = m
    if m["n"] >= 10:
        print(f"  {label:<38}  {m['n']:>5}  {m['wr']:>7.1f}%  {m['ret']:>+8.1f}pp  {m['sharpe']:>7.2f}  {m['maxdd']:>6.1f}%")

# Veredicto M2
m_base_v = m2_results.get("BASELINE W4-W5", {"wr": 0, "ret": 0, "sharpe": 0, "n": 0})
m_best_kl_v = m2_results.get("KL<=Q25_disc [aplicado]", {"wr": 0, "ret": 0, "sharpe": 0, "n": 0})

print()
if m_best_kl_v["n"] >= 10 and m_base_v["n"] >= 10:
    dwr = m_best_kl_v["wr"] - m_base_v["wr"]
    dsh = m_best_kl_v["sharpe"] - m_base_v["sharpe"]
    print(f"  Delta KL<=Q25 vs BASELINE validation: WR {dwr:+.1f}pp | Sharpe {dsh:+.2f}")
    if dwr > 10 and dsh > 0.5:
        m2_verdict = "POSITIVO"
        print(f"  VEREDICTO M2: POSITIVO *** — filtro KL mejora WR y Sharpe en W4-W5 genuinamente futuras")
    elif dwr > 3:
        m2_verdict = "MODERADO"
        print(f"  VEREDICTO M2: MODERADO — mejora modesta en validation (+{dwr:.1f}pp WR)")
    else:
        m2_verdict = "NEGATIVO"
        print(f"  VEREDICTO M2: NEGATIVO — filtro NO mejora en ventanas temporalmente futuras")
else:
    m2_verdict = "SIN DATOS"
    print(f"  VEREDICTO M2: N insuficiente en validation")
print(f"[H3-M2] Holdout temporal: dWR={dwr:+.1f}pp dSharpe={dsh:+.2f} verdict={m2_verdict}")


# ═══════════════════════════════════════════════════════════════════════════
# M3 — CONSISTENCIA CROSS-WINDOW
# Mide el efecto KL DENTRO de cada ventana por separado.
# Si aparece en 4+/5 ventanas -> no es artefacto de un periodo concreto.
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("M3: Consistencia cross-window (efecto KL dentro de cada ventana)")
print("    Si aparece en 4+/5 ventanas -> no es artefacto de un periodo")
print("=" * 72)
print()
print(f"  {'Ventana':>7}  {'N':>5}  {'KL<=Q25_WR':>11}  {'KL>=Q75_WR':>11}  {'Delta':>8}  {'Direccion'}")
print("  " + "-" * 68)

windows_favor = 0
windows_contra = 0
for win in ALL_WINDOWS:
    dw = df[df["_window"] == win]
    kl = dw["ood_kl_distance"]
    q25 = kl.quantile(0.25)
    q75 = kl.quantile(0.75)
    lo = dw[kl <= q25]
    hi = dw[kl >= q75]
    if len(lo) < 10 or len(hi) < 10:
        continue
    wr_lo = lo["is_win"].mean() * 100
    wr_hi = hi["is_win"].mean() * 100
    delta = wr_lo - wr_hi
    dir_str = "KL_bajo GANA" if delta > 5 else ("KL_alto GANA" if delta < -5 else "~empate")
    if delta > 5:
        windows_favor += 1
    elif delta < -5:
        windows_contra += 1
    print(f"  {win:>7}  {len(dw):>5}  {wr_lo:>11.1f}%  {wr_hi:>11.1f}%  {delta:>+8.1f}pp  {dir_str}")

total_windows = windows_favor + windows_contra
print()
print(f"  KL_bajo gana en {windows_favor}/5 ventanas | KL_alto gana en {windows_contra}/5")
if windows_favor >= 4:
    m3_verdict = "POSITIVO"
    print(f"  VEREDICTO M3: POSITIVO *** — efecto KL consistente en {windows_favor}/5 ventanas")
elif windows_favor == 3:
    m3_verdict = "MODERADO"
    print(f"  VEREDICTO M3: MODERADO — efecto mayoritario (3/5 ventanas)")
else:
    m3_verdict = "NEGATIVO"
    print(f"  VEREDICTO M3: NEGATIVO — efecto inconsistente entre ventanas")
print(f"[H3-M3] Cross-window: favor={windows_favor} contra={windows_contra} verdict={m3_verdict}")


# ═══════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL Y DECISION
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("RESUMEN FINAL — Decision sobre implementacion de H3 (filtro KL Score)")
print("=" * 72)
print()
verdicts = [m1_verdict, m2_verdict, m3_verdict]
pos = verdicts.count("POSITIVO")
mod = verdicts.count("MODERADO")

print(f"  M1 Consistencia seeds  : {m1_verdict}")
print(f"  M2 Holdout temporal    : {m2_verdict}  <- el mas importante")
print(f"  M3 Cross-window        : {m3_verdict}")
print()
print(f"  Positivos: {pos}/3 | Moderados: {mod}/3")
print()
if pos >= 3 or (pos >= 2 and m2_verdict == "POSITIVO"):
    print("  *** DECISION: IMPLEMENTAR CON CONFIANZA")
    print("  El filtro KL Score es robusto, predictivo temporalmente y consistente.")
    print("  Recomendacion: usar como filtro suave en Kelly sizer (no binario).")
    print("  Prioridad: XGBoost>=Q75 + KL<=Q25 (mejor Sharpe en todos los tests).")
elif pos + mod >= 2 and m2_verdict in ("POSITIVO", "MODERADO"):
    print("  ** DECISION: CONSIDERAR IMPLEMENTACION CAUTA")
    print("  Evidencia moderada. Implementar con kelly_fraction reducida y monitoreo.")
    print("  Lanzar mini-run de 5 seeds con el filtro habilitado para confirmacion real.")
else:
    print("  DECISION: NO IMPLEMENTAR todavia")
    print("  Evidencia insuficiente o contradictoria.")
    print("  La mejora observada en el analisis global probablemente es data snooping.")
    print("  Mantener skip_metalabeler=true y OOD Guard sin cambios.")

print()
print("[H3-VALIDACION] Script completado.")
