"""
AUDITORIA PROFUNDA — Causa Raiz de la Inversion de Señal
=========================================================
Objetivo: Determinar POR QUE el MetaLabeler y el OOD Guard tienen correlaciones invertidas.

Hipotesis a testar:
A) La inversion es un artefacto de warm-up LSTM (primeras barras de cada ventana)
B) La inversion es especifica a ciertos regimenes HMM
C) La inversion es especifica a ciertas semillas (model-specific, no sistematica)
D) Hay colinealidad inversa entre xgb_prob_cal y meta_v2_prob
E) El MetaLabeler predice correctamente en ALGUNAS ventanas (drift temporal)
F) El OOD Guard tiene señal correcta en ventanas de tendencia y señal invertida en rangos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 200)

reports_dir = Path("data/reports/wfb")
parquets = sorted(reports_dir.glob("oos_trades_W*_seed*.parquet"))

all_trades = []
for p in parquets:
    df = pd.read_parquet(p)
    parts = p.stem.split("_")
    df["window"] = parts[2]
    df["seed"] = int(parts[3].replace("seed", ""))
    all_trades.append(df)

df_all = pd.concat(all_trades, ignore_index=True)
df_all["is_win"] = (df_all["return_raw"] > 0).astype(int)

SEP = "=" * 90

# Semillas aprobadas vs rechazadas (del reporte nocturno)
APPROVED_SEEDS = {100, 777, 1337, 39395, 70519, 61865, 44793}
REJECTED_SEEDS = {83942, 38581, 72186, 58668, 58373, 34596}
PARTIAL_SEEDS  = {74734, 83925, 12239, 36655, 50830, 42, 2025}

df_all["seed_status"] = df_all["seed"].apply(
    lambda s: "APROBADA" if s in APPROVED_SEEDS else ("RECHAZADA" if s in REJECTED_SEEDS else "PARCIAL")
)

print(SEP)
print("  AUDITORIA PROFUNDA: INVERSION DE SENAL EN META_V2 Y OOD_GUARD")
print(SEP)
print(f"\nTotal trades: {len(df_all)}")
print(f"Seeds APROBADAS: {sorted(APPROVED_SEEDS & set(df_all['seed'].unique()))}")
print(f"Seeds RECHAZADAS: {sorted(REJECTED_SEEDS & set(df_all['seed'].unique()))}")
print(f"Seeds PARCIALES: {sorted(PARTIAL_SEEDS & set(df_all['seed'].unique()))}")


# =============================================================================
print(f"\n{SEP}")
print("  BLOQUE A: Inversion por ventana temporal (posible drift o warm-up)")
print(f"  Hipotesis: LSTM warm-up contamina W1 (primeras 48 barras -> seq_len=48)")
print(SEP)

print(f"\n{'Ventana':<8} {'n':>6} {'WR_total':>9} {'Spearman_meta':>14} {'p_meta':>10} {'Spearman_ood':>14} {'p_ood':>10}")
print("-" * 80)
for w in ["W1", "W2", "W3", "W4", "W5"]:
    dw = df_all[df_all["window"] == w].copy()
    if len(dw) < 20:
        continue
    wr = dw["is_win"].mean() * 100
    r_meta, p_meta = stats.spearmanr(dw["meta_v2_prob"], dw["return_raw"])
    r_ood,  p_ood  = stats.spearmanr(dw["ood_kl_distance"], dw["return_raw"])
    flag_m = "<- INVERTIDO!" if r_meta < -0.05 else ("OK" if r_meta > 0.05 else "neutral")
    flag_o = "<- INVERTIDO!" if r_ood  < -0.05 else ("OK" if r_ood  > 0.05 else "neutral")
    print(f"{w:<8} {len(dw):>6} {wr:>8.1f}% {r_meta:>14.4f} {p_meta:>10.4f} {r_ood:>14.4f} {p_ood:>10.4f}   {flag_m} | OOD: {flag_o}")


# =============================================================================
print(f"\n{SEP}")
print("  BLOQUE B: Inversion por regimen HMM (posible desajuste de calibracion por regimen)")
print(SEP)

print(f"\n{'Regimen':<30} {'n':>6} {'WR':>7} {'Spear_meta':>12} {'p_meta':>9} {'Spear_ood':>12} {'p_ood':>9}")
print("-" * 90)
for regime in sorted(df_all["hmm_regime"].dropna().unique()):
    dr = df_all[df_all["hmm_regime"] == regime]
    if len(dr) < 20:
        continue
    wr = dr["is_win"].mean() * 100
    r_m, p_m = stats.spearmanr(dr["meta_v2_prob"], dr["return_raw"])
    r_o, p_o = stats.spearmanr(dr["ood_kl_distance"], dr["return_raw"])
    flag_m = "<<INVERTIDO" if r_m < -0.05 else ("ok" if r_m > 0.05 else "~neutral")
    flag_o = "<<INVERTIDO" if r_o < -0.05 else ("ok" if r_o > 0.05 else "~neutral")
    print(f"{str(regime):<30} {len(dr):>6} {wr:>6.1f}% {r_m:>12.4f} {p_m:>9.4f} {r_o:>12.4f} {p_o:>9.4f}  META:{flag_m} OOD:{flag_o}")


# =============================================================================
print(f"\n{SEP}")
print("  BLOQUE C: Inversion por semilla (sistematica vs seed-especifica)")
print(SEP)

print(f"\n{'Seed':<8} {'Status':<10} {'n':>6} {'WR':>7} {'Spear_meta':>12} {'Spear_ood':>12} {'Colineal_xgb_meta':>18}")
print("-" * 85)
for seed in sorted(df_all["seed"].unique()):
    ds = df_all[df_all["seed"] == seed]
    if len(ds) < 20:
        continue
    status = ds["seed_status"].iloc[0]
    wr = ds["is_win"].mean() * 100
    r_m, _ = stats.spearmanr(ds["meta_v2_prob"], ds["return_raw"])
    r_o, _ = stats.spearmanr(ds["ood_kl_distance"], ds["return_raw"])
    # Colinealidad entre xgb_prob_cal y meta_v2_prob
    r_col, _ = stats.spearmanr(ds["xgb_prob_cal"], ds["meta_v2_prob"])
    flag = "<<INV" if r_m < -0.05 else ("ok" if r_m > 0.05 else "~")
    print(f"{seed:<8} {status:<10} {len(ds):>6} {wr:>6.1f}% {r_m:>12.4f} {r_o:>12.4f} {r_col:>18.4f}  META:{flag}")


# =============================================================================
print(f"\n{SEP}")
print("  BLOQUE D: Colinealidad XGBoost vs MetaLabeler (posible señal redundante/inversa)")
print(f"  Si xgb_prob_cal y meta_v2_prob tienen Spearman NEGATIVA -> INVERSION SISTEMATICA")
print(SEP)

r_col_global, p_col = stats.spearmanr(df_all["xgb_prob_cal"], df_all["meta_v2_prob"])
print(f"\n  Spearman global (xgb_prob_cal vs meta_v2_prob): r={r_col_global:.4f} | p={p_col:.4f}")
if r_col_global < -0.1:
    print("  CRITICO: Correlacion NEGATIVA entre XGBoost y MetaLabeler.")
    print("  El MetaLabeler SUBE su confianza cuando XGBoost BAJA la suya.")
    print("  Esto puede ser por diseño (MetaLabeler calibra sobre errores de XGBoost)")
    print("  o por un bug de alineamiento temporal.")
elif r_col_global > 0.3:
    print("  Los modelos estan correlacionados positivamente (señal redundante).")
else:
    print("  Correlacion baja/neutra (modelos independientes, normal).")

# Cuartiles de xgb_prob_cal: ¿meta_v2_prob sube o baja con xgb?
print(f"\n  Cuartiles xgb_prob_cal: meta_v2_prob y WR correspondientes:")
df_all["xgb_q"] = pd.qcut(df_all["xgb_prob_cal"], 4, labels=["Q1(bajo)", "Q2", "Q3", "Q4(alto)"], duplicates="drop")
cross = df_all.groupby("xgb_q", observed=True).agg(
    n=("is_win","count"),
    wr=("is_win","mean"),
    xgb_mean=("xgb_prob_cal","mean"),
    meta_mean=("meta_v2_prob","mean"),
    ood_mean=("ood_kl_distance","mean"),
).reset_index()
print(f"\n  {'XGB_Cuartil':<12} {'n':>6} {'WR':>7} {'XGB_prob':>10} {'Meta_prob':>10} {'OOD_kl':>10}")
print(f"  {'-'*60}")
for _, r in cross.iterrows():
    print(f"  {str(r['xgb_q']):<12} {int(r['n']):>6} {r['wr']*100:>6.1f}% {r['xgb_mean']:>10.4f} {r['meta_mean']:>10.4f} {r['ood_mean']:>10.4f}")


# =============================================================================
print(f"\n{SEP}")
print("  BLOQUE E: Analisis WFV temporal — ¿Drift del MetaLabeler sobre el tiempo?")
print(f"  El MetaLabeler se entrena en el training set. En OOS puede driftar.")
print(SEP)

# Analisis por ventana con breakdown por semilla aprobada/rechazada
print(f"\n  WR y Spearman META por Ventana x Estado de Semilla:")
print(f"\n  {'Ventana':<8} {'Status':<10} {'n':>6} {'WR':>7} {'meta_mean':>10} {'Spear_meta':>12} {'p_meta':>9}")
print(f"  {'-'*70}")
for w in ["W1","W2","W3","W4","W5"]:
    for st in ["APROBADA","RECHAZADA","PARCIAL"]:
        dws = df_all[(df_all["window"]==w) & (df_all["seed_status"]==st)]
        if len(dws) < 10:
            continue
        wr = dws["is_win"].mean() * 100
        meta_m = dws["meta_v2_prob"].mean()
        r_m, p_m = stats.spearmanr(dws["meta_v2_prob"], dws["return_raw"])
        flag = "<<INV" if r_m < -0.05 else ("ok" if r_m > 0.05 else "~")
        print(f"  {w:<8} {st:<10} {len(dws):>6} {wr:>6.1f}% {meta_m:>10.4f} {r_m:>12.4f} {p_m:>9.4f}  {flag}")


# =============================================================================
print(f"\n{SEP}")
print("  BLOQUE F: OOD Guard — ¿Por que barras anomalas ganan mas?")
print(f"  KL bajo = mas anomalo. Si Q1 gana mas, el IsolationForest esta invertido")
print(f"  o el mercado de alta volatilidad (anomalo para el train) es rentable en OOS.")
print(SEP)

# Distribucion de OOD por ventana y estado de semilla
print(f"\n  OOD_kl_distance por ventana — Media, Q1-WR, Q4-WR:")
print(f"\n  {'Ventana':<8} {'n':>6} {'OOD_mean':>10} {'OOD_std':>10} {'WR_Q1(bajo)':>13} {'WR_Q4(alto)':>13}")
print(f"  {'-'*65}")
for w in ["W1","W2","W3","W4","W5"]:
    dw = df_all[df_all["window"]==w].copy()
    if len(dw) < 20:
        continue
    # Cuartiles OOD dentro de la ventana
    try:
        dw["ood_q_local"] = pd.qcut(dw["ood_kl_distance"], 4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        wr_q1 = dw[dw["ood_q_local"]=="Q1"]["is_win"].mean() * 100
        wr_q4 = dw[dw["ood_q_local"]=="Q4"]["is_win"].mean() * 100
    except:
        wr_q1 = wr_q4 = float("nan")
    print(f"  {w:<8} {len(dw):>6} {dw['ood_kl_distance'].mean():>10.5f} {dw['ood_kl_distance'].std():>10.5f} {wr_q1:>12.1f}% {wr_q4:>12.1f}%")

# ¿En que regimenes el OOD bajo = alta rentabilidad?
print(f"\n  OOD_kl_distance bajo (anomalias) por regimen HMM — WR Q1 vs WR Q4:")
print(f"\n  {'Regimen':<30} {'n':>6} {'WR_Q1(anomalo)':>16} {'WR_Q4(normal)':>15} {'Delta':>8}")
print(f"  {'-'*80}")
for regime in sorted(df_all["hmm_regime"].dropna().unique()):
    dr = df_all[df_all["hmm_regime"]==regime].copy()
    if len(dr) < 40:
        continue
    try:
        dr["ood_q_r"] = pd.qcut(dr["ood_kl_distance"], 4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        wr_q1 = dr[dr["ood_q_r"]=="Q1"]["is_win"].mean() * 100
        wr_q4 = dr[dr["ood_q_r"]=="Q4"]["is_win"].mean() * 100
        delta = wr_q1 - wr_q4
        flag = "<<< FUERTE INVERSION" if delta > 20 else ("< inversion" if delta > 5 else ("normal" if delta < -5 else "~plano"))
        print(f"  {str(regime):<30} {len(dr):>6} {wr_q1:>15.1f}% {wr_q4:>14.1f}% {delta:>+7.1f}pp  {flag}")
    except:
        pass


# =============================================================================
print(f"\n{SEP}")
print("  RESUMEN FINAL Y DIAGNOSTICO")
print(SEP)

# Calculamos estadisticos globales clave
r_meta_global, _ = stats.spearmanr(df_all["meta_v2_prob"], df_all["return_raw"])
r_ood_global, _  = stats.spearmanr(df_all["ood_kl_distance"], df_all["return_raw"])
r_xgb_global, _  = stats.spearmanr(df_all["xgb_prob_cal"], df_all["return_raw"])
r_col_g, _       = stats.spearmanr(df_all["xgb_prob_cal"], df_all["meta_v2_prob"])

print(f"""
  Spearman xgb_prob_cal  vs return_raw: r={r_xgb_global:+.4f}  {'OK (señal correcta)' if r_xgb_global>0 else 'INVERTIDO'}
  Spearman meta_v2_prob  vs return_raw: r={r_meta_global:+.4f}  {'OK (señal correcta)' if r_meta_global>0 else 'INVERTIDO - BUG CANDIDATO'}
  Spearman ood_kl_dist   vs return_raw: r={r_ood_global:+.4f}  {'OK (señal correcta)' if r_ood_global>0 else 'INVERTIDO - KL bajo = mejor trade'}
  Spearman xgb_prob_cal  vs meta_v2:    r={r_col_g:+.4f}  {'COLINEAL positivo' if r_col_g>0.3 else ('INVERSO - posible bug alineamiento' if r_col_g<-0.1 else 'independiente (normal)')}

  INTERPRETACION:
  - El XGBoost SI tiene señal valida (si r_xgb > 0)
  - El MetaLabeler tiene correlacion NEGATIVA -> actua como ruido/inversor, no como filtro
  - El OOD Guard (KL bajo = anomalo = mas rentable) sugiere que las mejores oportunidades
    estan en condiciones NO vistas en training (breakouts, movimientos excepcionales)
    El modelo entrenado en "normalidad" censura las situaciones mas rentables de OOS.

  CAUSA RAIZ MAS PROBABLE (requiere confirmar con ingenieria):
  1. MetaLabeler: Posible inversion de etiquetas en metalabeling (1=win pero el modelo
     aprendio a predecir la probabilidad de perder, o el threshold de calibracion esta
     invirtiendo la salida). Verificar luna/models/train_metalabeler_v2.py -> etiqueta.
  2. OOD Guard: El IsolationForest entrena con datos del training set (mercado 2022-2024).
     El OOS 2025-2026 es un periodo de movimientos excepcionales (ETF rally, halvings).
     Las barras 'anomalas' son las de mayor movimiento -> mayor edge capturado por XGBoost.
     El OOD Guard esta penalizando las mejores oportunidades porque el training no las vio.
""")
