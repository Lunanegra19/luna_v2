"""
[INVESTIGACION-PASO3] Simulacion de skip_metalabeler sobre datos OOS existentes
================================================================================
NO modifica el pipeline ni lanza ninguna run.
Toma los 7.718 trades reales y simula que el MetaLabeler no filtrara nada,
calculando el WR/Sharpe/MaxDD/trades que habriamos obtenido.

Tambien simula escenarios intermedios:
- Threshold mas bajo (0.60, 0.65, 0.70) vs el actual (rolling_percentile >= 0.85)
- Skip total (todas las senales XGBoost pasan)
- Skip solo en W3 (la ventana donde esta invertido)

Metodologia: los parquets de oos_trades ya contienen meta_v2_prob y el resultado
final. Podemos re-simular cualquier umbral del MetaLabeler filtrando esos trades.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

print("=" * 72)
print("[INVESTIGACION-PASO3] SIMULACION skip_metalabeler vs baseline")
print("=" * 72)

# ── Carga de datos ────────────────────────────────────────────────────────
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

print(f"\n[LOAD] {len(df)} trades | {df['_seed'].nunique()} seeds | "
      f"Ventanas: {sorted(df['_window'].unique())}")
print(f"[LOAD] Columnas clave: meta_v2_prob={('meta_v2_prob' in df.columns)} | "
      f"xgb_prob_cal={('xgb_prob_cal' in df.columns)} | "
      f"ood_kl_distance={('ood_kl_distance' in df.columns)}")


# ── Metricas resumen ──────────────────────────────────────────────────────
def metricas(subset: pd.DataFrame, label: str) -> dict:
    """Calcula WR, Sharpe anualizado, MaxDD y Calmar sobre un subconjunto."""
    n = len(subset)
    if n == 0:
        return {"label": label, "n": 0, "wr": np.nan, "ret_total": np.nan,
                "sharpe": np.nan, "maxdd": np.nan, "calmar": np.nan}
    wr = subset["is_win"].mean() * 100
    ret_total = subset["ret100"].sum()
    # Calmar y Sharpe sobre retornos de trade individuales
    rets = subset["ret100"].values
    mean_r = np.mean(rets)
    std_r  = np.std(rets) if len(rets) > 1 else 1e-10
    # Sharpe anualizado (asumiendo ~1 trade/semana en media = 52/año)
    trades_per_year = 52  # conservador
    sharpe = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 1e-10 else 0.0
    # MaxDD acumulado
    cumret = (1 + subset["return_raw"]).cumprod()
    running_max = cumret.cummax()
    dd = (cumret - running_max) / running_max
    maxdd = abs(dd.min()) * 100
    calmar = (ret_total / max(maxdd, 0.01)) if maxdd > 0 else 0.0
    return {"label": label, "n": n, "wr": wr, "ret_total": ret_total,
            "sharpe": sharpe, "maxdd": maxdd, "calmar": calmar}


def fmt_row(m: dict) -> str:
    if m["n"] == 0:
        return f"  {m['label']:<45} {'N/A':>6}   {'—':>7}   {'—':>7}   {'—':>7}   {'—':>7}"
    return (f"  {m['label']:<45} {m['n']:>6}   {m['wr']:>6.1f}%   "
            f"{m['ret_total']:>+7.1f}pp   {m['sharpe']:>7.2f}   {m['maxdd']:>6.1f}%")


print("\n" + "=" * 72)
print("ESCENARIO 1: MetaLabeler como gate ACTIVO (baseline run actual)")
print("=" * 72)
print("  Nota: En la run actual skip_metalabeler=true, el MetaLabeler CALCULA")
print("  pero NO filtra (las seeds con prob<=umbral igual pasan).")
print("  Pero meta_v2_prob esta disponible para simular filtros retroactivos.")
print()

# Baseline = todos los trades que llegaron al parquet (skip=true ya activo)
m_base = metricas(df, "BASELINE (run actual, skip=true)")
print(f"  {'Escenario':<45} {'N':>6}   {'WR':>7}   {'Ret%':>7}   {'Sharpe':>7}   {'MaxDD':>7}")
print("  " + "-" * 75)
print(fmt_row(m_base))


print("\n" + "=" * 72)
print("ESCENARIO 2: Simulacion de MetaLabeler como GATE ACTIVO")
print("  Filtro retroactivo: solo se aprueban trades con meta_v2_prob >= umbral")
print("=" * 72)
print()

if "meta_v2_prob" in df.columns:
    umbrales = [0.58, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75]
    print(f"  {'Escenario':<45} {'N':>6}   {'WR':>7}   {'Ret%':>7}   {'Sharpe':>7}   {'MaxDD':>7}")
    print("  " + "-" * 75)
    print(fmt_row(m_base))
    for u in umbrales:
        sub = df[df["meta_v2_prob"] >= u]
        m = metricas(sub, f"MetaLabeler gate >= {u:.2f} ({len(sub)/len(df)*100:.0f}% pasan)")
        print(fmt_row(m))
    print()
    print("  [DIAGNOSTICO] Si Ret_total cae con TODOS los umbrales → MetaLabeler esta INVERTIDO globalmente.")
    print("  Si Ret_total sube con un umbral → hay algun nivel donde el MetaLabeler aporta.")
else:
    print("  [ERROR] meta_v2_prob no en parquet. No se puede simular.")


print("\n" + "=" * 72)
print("ESCENARIO 3: Analisis POR VENTANA — donde perjudica y donde ayuda")
print("=" * 72)
print()

if "meta_v2_prob" in df.columns:
    print(f"  {'Escenario':<45} {'N':>6}   {'WR':>7}   {'Ret%':>7}   {'Sharpe':>7}   {'MaxDD':>7}")
    print("  " + "-" * 75)
    for win in sorted(df["_window"].unique()):
        dw = df[df["_window"] == win]
        m_all = metricas(dw, f"W={win} BASELINE (todos los trades)")
        m_filt = metricas(
            dw[dw["meta_v2_prob"] >= 0.65],
            f"W={win} MetaLabeler >= 0.65 ({len(dw[dw['meta_v2_prob']>=0.65])/max(len(dw),1)*100:.0f}% pasan)"
        )
        print(fmt_row(m_all))
        print(fmt_row(m_filt))
        diff_ret = m_filt["ret_total"] - m_all["ret_total"]
        diff_wr  = m_filt["wr"] - m_all["wr"]
        verdict = "PERJUDICA" if diff_ret < 0 else "MEJORA"
        print(f"  {'  DELTA: ' + verdict:<45} {'':>6}   {diff_wr:>+6.1f}pp   {diff_ret:>+7.1f}pp")
        print("  " + "-" * 75)


print("\n" + "=" * 72)
print("ESCENARIO 4: Skip MetaLabeler SOLO en W3 (ventana conflictiva)")
print("  Simulacion: en W3 se ignora el umbral MetaLabeler, en el resto se mantiene")
print("=" * 72)
print()

if "meta_v2_prob" in df.columns:
    UMBRAL = 0.65
    sub_hybrid = pd.concat([
        df[df["_window"] != "W3"],                          # W1/W2/W4/W5: sin filtro
        df[(df["_window"] == "W3")]                         # W3: sin filtro (skip)
    ])
    sub_hybrid_filtered = pd.concat([
        df[df["_window"] != "W3"],                          # W1/W2/W4/W5: sin filtro
        df[(df["_window"] == "W3") & (df["meta_v2_prob"] >= UMBRAL)]  # W3: con filtro
    ])
    m_hybrid = metricas(sub_hybrid, f"Hybrid: W3 sin MetaLabeler, resto sin filtro")
    m_hybrid_f = metricas(sub_hybrid_filtered, f"Hybrid: W3 con MetaLabeler >= {UMBRAL}")
    print(f"  {'Escenario':<45} {'N':>6}   {'WR':>7}   {'Ret%':>7}   {'Sharpe':>7}   {'MaxDD':>7}")
    print("  " + "-" * 75)
    print(fmt_row(m_base))
    print(fmt_row(m_hybrid))
    print(fmt_row(m_hybrid_f))


print("\n" + "=" * 72)
print("ESCENARIO 5: OOD Guard inverso (usar KL bajo = anomalo para ENTRAR)")
print("  Hipotesis de mejora: filtrar SOLO trades con KL bajo (anomalo en 2025-26)")
print("=" * 72)
print()

if "ood_kl_distance" in df.columns:
    kl_med = df["ood_kl_distance"].median()
    kl_q25 = df["ood_kl_distance"].quantile(0.25)
    kl_q75 = df["ood_kl_distance"].quantile(0.75)
    print(f"  Distribucion KL: min={df['ood_kl_distance'].min():.4f} | "
          f"Q25={kl_q25:.4f} | med={kl_med:.4f} | Q75={kl_q75:.4f} | "
          f"max={df['ood_kl_distance'].max():.4f}")
    print()
    scenarios_ood = [
        ("Solo KL<=Q25 (mas anomalo, 25% trades)", df[df["ood_kl_distance"] <= kl_q25]),
        ("Solo KL<=med (50% trades con menor KL)", df[df["ood_kl_distance"] <= kl_med]),
        ("Solo KL>=Q75 (mas normal, 25% trades)", df[df["ood_kl_distance"] >= kl_q75]),
        ("Solo KL>=med (50% trades con mayor KL)", df[df["ood_kl_distance"] >= kl_med]),
        ("BASELINE (todos)", df),
    ]
    print(f"  {'Escenario':<47} {'N':>6}   {'WR':>7}   {'Ret%':>7}   {'Sharpe':>7}   {'MaxDD':>7}")
    print("  " + "-" * 75)
    for label, sub in scenarios_ood:
        print(fmt_row(metricas(sub, label)))


print("\n" + "=" * 72)
print("ESCENARIO 6: Combinacion optima — XGBoost PURO (sin MetaLabeler, sin OOD gate)")
print("  Esto mide el edge REAL del XGBoost sin contaminar con filtros que perjudican")
print("=" * 72)
print()

# Simular XGBoost puro: solo se usa xgb_prob_cal
if "xgb_prob_cal" in df.columns:
    xgb_med = df["xgb_prob_cal"].median()
    xgb_q75 = df["xgb_prob_cal"].quantile(0.75)
    xgb_q85 = df["xgb_prob_cal"].quantile(0.85)
    print(f"  Distribucion xgb_prob_cal: med={xgb_med:.4f} | Q75={xgb_q75:.4f} | Q85={xgb_q85:.4f}")
    print()
    scenarios_xgb = [
        ("BASELINE (todos, skip=true actual)", df),
        (f"Solo XGBoost >= Q50 ({xgb_med:.3f}), 50% trades", df[df["xgb_prob_cal"] >= xgb_med]),
        (f"Solo XGBoost >= Q75 ({xgb_q75:.3f}), 25% trades", df[df["xgb_prob_cal"] >= xgb_q75]),
        (f"Solo XGBoost >= Q85 ({xgb_q85:.3f}), 15% trades", df[df["xgb_prob_cal"] >= xgb_q85]),
        # Combinacion: XGBoost + KL bajo (anomalo)
        (f"XGBoost>=Q75 + KL<=med (25%+50%)", 
         df[(df["xgb_prob_cal"] >= xgb_q75) & (df["ood_kl_distance"] <= kl_med)] if "ood_kl_distance" in df.columns else df),
        (f"XGBoost>=Q50 + KL<=Q25 (mejores anomalos)", 
         df[(df["xgb_prob_cal"] >= xgb_med) & (df["ood_kl_distance"] <= kl_q25)] if "ood_kl_distance" in df.columns else df),
    ]
    print(f"  {'Escenario':<50} {'N':>5}   {'WR':>7}   {'Ret%':>7}   {'Sharpe':>7}   {'MaxDD':>7}")
    print("  " + "-" * 78)
    for label, sub in scenarios_xgb:
        print(fmt_row(metricas(sub, label)))


print("\n" + "=" * 72)
print("ESCENARIO 7: Seeds aprobadas vs rechazadas — reproduccion del sesgo del comite")
print("=" * 72)
print()

# Verificar qué seeds son las "aprobadas" en la run
APROBADAS_CONOCIDAS = {38581, 777, 39395}  # de la auditoría anterior
print(f"  Seeds analizadas en profundidad: {sorted(APROBADAS_CONOCIDAS)}")
print()
print(f"  {'Escenario':<45} {'N':>6}   {'WR':>7}   {'Ret%':>7}   {'Sharpe':>7}   {'MaxDD':>7}")
print("  " + "-" * 75)
print(fmt_row(metricas(df, "BASELINE (todos)")))
for seed in sorted(df["_seed"].unique()):
    sub = df[df["_seed"] == seed]
    r_meta, _ = stats.spearmanr(sub["meta_v2_prob"], sub["return_raw"]) if "meta_v2_prob" in sub.columns else (np.nan, np.nan)
    label = f"Seed {seed} [r_meta={r_meta:+.3f}]"
    print(fmt_row(metricas(sub, label)))


print("\n" + "=" * 72)
print("CONCLUSION CUANTITATIVA — Veredicto final por escenario")
print("=" * 72)
print()
if "meta_v2_prob" in df.columns:
    m_skip = metricas(df, "Skip total (baseline actual)")
    m_gate65 = metricas(df[df["meta_v2_prob"] >= 0.65], "MetaLabeler gate >= 0.65")
    m_xgb75 = metricas(df[df["xgb_prob_cal"] >= xgb_q75] if "xgb_prob_cal" in df.columns else df, "XGBoost puro Q75")

    print(f"  {'Veredicto':<45} {'N':>6}   {'WR':>7}   {'Ret%':>7}")
    print("  " + "-" * 65)
    print(f"  {'1. Skip total (situacion actual)':<45} {m_skip['n']:>6}   {m_skip['wr']:>6.1f}%   {m_skip['ret_total']:>+7.1f}pp")
    print(f"  {'2. MetaLabeler gate >= 0.65':<45} {m_gate65['n']:>6}   {m_gate65['wr']:>6.1f}%   {m_gate65['ret_total']:>+7.1f}pp  <- PERJUDICA si < baseline")
    print(f"  {'3. XGBoost puro Q75':<45} {m_xgb75['n']:>6}   {m_xgb75['wr']:>6.1f}%   {m_xgb75['ret_total']:>+7.1f}pp")
    print()
    print("  CONCLUSION FINAL:")
    if m_gate65["ret_total"] < m_skip["ret_total"]:
        print("  ✅ CONFIRMADO: MetaLabeler como gate PERJUDICA vs skip total.")
        print(f"     Diferencia: {m_gate65['ret_total'] - m_skip['ret_total']:+.1f}pp en Ret_total")
        print("     RECOMENDACION: mantener skip_metalabeler=true en la proxima run.")
    else:
        print("  ⚠️  SORPRESA: MetaLabeler gate MEJORA vs skip total en este escenario.")
        print("  Revisar los datos — puede haber un efecto de seleccion de trades.")
