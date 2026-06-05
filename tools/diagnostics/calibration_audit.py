# -*- coding: utf-8 -*-
"""Audit de calibracion completo por regimen HMM y por buckets de probabilidad."""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

reports_dir = Path("data/reports/wfb")
seeds = [int(f.stem.split("seed")[1]) for f in reports_dir.glob("oos_trades_W5_seed*.parquet")]
COMM = 0.0015

all_trades = []
for seed in seeds:
    files = sorted(reports_dir.glob(f"oos_trades_W*_seed{seed}.parquet"))
    if len(files) < 5:
        continue
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["seed"] = seed
    all_trades.append(df)
df_all = pd.concat(all_trades, ignore_index=True)

print("=" * 110)
print("  AUDIT DE CALIBRACION: xgb_prob_cal vs WR_real POR REGIMEN")
print("  Descalibracion = modelo predice mas wins de los que ocurren")
print("=" * 110)
print()
print(f"{'Regimen':<25} {'n':>5} {'xgb_prob':>10} {'meta_prob':>10} {'WR_real':>9} {'Error_xgb':>11} {'Error_meta':>12} {'avg_ret':>9} {'ESTADO'}")
print("-" * 110)

regimenes = df_all["hmm_regime"].value_counts().index.tolist()
for regime in regimenes:
    df_r = df_all[df_all["hmm_regime"] == regime]
    n = len(df_r)
    xgb_p = df_r["xgb_prob_cal"].mean()
    meta_p = df_r["meta_v2_prob"].mean()
    wr = (df_r["return_raw"] > 0).mean()
    avg = df_r["return_raw"].mean() * 100
    err_xgb = xgb_p - wr
    err_meta = meta_p - wr

    if err_xgb > 0.05:
        estado = "SOBRECONFIADO <<< PROBLEMA"
    elif err_xgb > 0.02:
        estado = "Leve sobreconfianza"
    elif err_xgb < -0.05:
        estado = "INFRACONFIADO"
    else:
        estado = "OK calibrado"

    print(f"{regime:<25} {n:>5} {xgb_p:>10.3f} {meta_p:>10.3f} {wr*100:>8.1f}% {err_xgb:>11.3f} {err_meta:>12.3f} {avg:>8.3f}%  {estado}")

print()
print("=" * 80)
print("  RELIABILITY DIAGRAM (buckets de xgb_prob_cal)")
print("  Modelo perfecto: prob_mean == WR_real en cada bucket")
print("=" * 80)
print()
df_all["prob_bucket"] = pd.cut(df_all["xgb_prob_cal"], bins=10)
reliability = df_all.groupby("prob_bucket", observed=True).agg(
    n=("return_raw", "count"),
    prob_mean=("xgb_prob_cal", "mean"),
    wr_real=("return_raw", lambda x: (x > 0).mean()),
    avg_ret=("return_raw", "mean"),
).reset_index()
reliability["calib_error"] = reliability["prob_mean"] - reliability["wr_real"]

print(f"{'Rango prob':>20} {'n':>5} {'prob_mean':>10} {'WR_real':>9} {'Error':>8} {'avg_ret':>9} {'ESTADO'}")
print("-" * 80)
for _, r in reliability.iterrows():
    if r["n"] < 5:
        continue
    if r["calib_error"] > 0.05:
        flag = "SOBRECONFIADO"
    elif r["calib_error"] < -0.05:
        flag = "INFRACONFIADO"
    else:
        flag = "OK"
    print(f"{str(r['prob_bucket']):>20} {int(r['n']):>5} {r['prob_mean']:>10.3f} {r['wr_real']*100:>8.1f}% {r['calib_error']:>8.3f} {r['avg_ret']*100:>8.3f}%  {flag}")

print()
print("=" * 80)
print("  CALIBRACION CRUZADA: REGIMEN x BUCKET DE PROBABILIDAD")
print("  Detectar si la sobreconfianza es especifica a ciertos regimenes")
print("  EN ciertos rangos de probabilidad")
print("=" * 80)
print()
df_all["prob_bin"] = pd.cut(df_all["xgb_prob_cal"], bins=[0.5, 0.62, 0.70, 0.80, 1.0],
                             labels=["bajo(0.50-0.62)", "medio(0.62-0.70)", "alto(0.70-0.80)", "muy_alto(0.80+)"])
cross = df_all.groupby(["hmm_regime", "prob_bin"], observed=True).agg(
    n=("return_raw", "count"),
    xgb_p=("xgb_prob_cal", "mean"),
    wr=("return_raw", lambda x: (x > 0).mean()),
    avg=("return_raw", "mean"),
).reset_index()
cross["err"] = cross["xgb_p"] - cross["wr"]

print(f"{'Regimen':<25} {'prob_bin':<20} {'n':>4} {'xgb_p':>7} {'WR':>7} {'Error':>7} {'avg_ret':>8}  ESTADO")
print("-" * 95)
for _, r in cross[cross["n"] >= 5].iterrows():
    if r["err"] > 0.08:
        flag = "<<< MUY SOBRECONFIADO"
    elif r["err"] > 0.04:
        flag = "< sobreconfiado"
    elif r["err"] < -0.05:
        flag = "> infraconfiado"
    else:
        flag = ""
    print(f"{r['hmm_regime']:<25} {str(r['prob_bin']):<20} {int(r['n']):>4} {r['xgb_p']:>7.3f} {r['wr']*100:>6.1f}% {r['err']:>7.3f} {r['avg']*100:>7.3f}%  {flag}")

print()
print("=" * 80)
print("  ANALISIS OOD (Out-of-Distribution): KL divergence vs calibracion")
print("  Si OOD alto correlaciona con mala calibracion, el HMM senaliza correctamente")
print("=" * 80)
print()
if "ood_kl_distance" in df_all.columns:
    df_all["ood_bucket"] = pd.cut(df_all["ood_kl_distance"], bins=4, labels=["bajo", "medio", "alto", "muy_alto"])
    ood_stats = df_all.groupby(["hmm_regime", "ood_bucket"], observed=True).agg(
        n=("return_raw", "count"),
        wr=("return_raw", lambda x: (x > 0).mean()),
        avg=("return_raw", "mean"),
        kl_mean=("ood_kl_distance", "mean"),
    ).reset_index()
    print(f"{'Regimen':<25} {'OOD_nivel':<12} {'n':>4} {'KL_mean':>9} {'WR':>8} {'avg_ret':>9}")
    print("-" * 75)
    for _, r in ood_stats[ood_stats["n"] >= 5].iterrows():
        print(f"{r['hmm_regime']:<25} {str(r['ood_bucket']):<12} {int(r['n']):>4} {r['kl_mean']:>9.4f} {r['wr']*100:>7.1f}% {r['avg']*100:>8.3f}%")
    
    corr = df_all[["ood_kl_distance", "return_raw"]].corr().iloc[0, 1]
    print()
    print(f"Correlacion OOD_KL <-> return_raw: {corr:.4f}")
    if corr < -0.05:
        print("  --> OOD alto correlaciona con peor rendimiento (el HMM es informativo)")
    else:
        print("  --> OOD no correlaciona con rendimiento (el HMM no anade informacion directa)")
