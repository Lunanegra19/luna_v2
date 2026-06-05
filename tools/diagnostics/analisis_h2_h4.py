# -*- coding: utf-8 -*-
import pandas as pd
import json
from pathlib import Path
BASE = Path("g:/Mi unidad/ia/luna_v2")

# H2
print("H2: DISTRIBUCION IS (2018-2024)")
hmm_paths = list(BASE.glob("**/hmm_regime_labels.parquet"))
if hmm_paths:
    hmm = pd.read_parquet(hmm_paths[0])
    hmm.index = pd.to_datetime(hmm.index, utc=True, errors="coerce")
    col = "HMM_Semantic" if "HMM_Semantic" in hmm.columns else "HMM_Regime"
    dist = hmm[col].value_counts()
    total = len(hmm)
    for reg, n in dist.items():
        print(f"  {str(reg):<30}: {n:>6} ({n/total*100:>5.1f}%)")
    bull  = sum(n for r,n in dist.items() if "BULL" in str(r))
    rng   = sum(n for r,n in dist.items() if "RANGE" in str(r))
    bear  = sum(n for r,n in dist.items() if "BEAR" in str(r))
    print(f"  --- BULL total: {bull/total*100:.1f}% | RANGE: {rng/total*100:.1f}% | BEAR: {bear/total*100:.1f}%")
print()

# H3: BTC en W4 (Oct-Dic 2025)
print("H3: BTC PRECIO EN W4 OOS (OCT-DIC 2025)")
# Primero intentar holdout
holdout_paths = list(BASE.glob("**/features_holdout.parquet"))
feat_paths = list(BASE.glob("**/features/features_train.parquet"))
if not feat_paths:
    feat_paths = list(BASE.glob("**/features_train.parquet"))

source_df = None
for paths, label in [(holdout_paths, "holdout"), (feat_paths, "train")]:
    if paths:
        try:
            df = pd.read_parquet(paths[0], columns=["close"])
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            source_df = df
            print(f"  Fuente: {label} | Rango: {df.index.min().date()} -> {df.index.max().date()}")
            break
        except Exception as e:
            print(f"  Error con {label}: {e}")

if source_df is not None:
    w4 = source_df[(source_df.index >= "2025-10-01") & (source_df.index <= "2025-12-31")]
    if len(w4) > 24:
        c0 = float(w4["close"].iloc[0])
        cf = float(w4["close"].iloc[-1])
        mx = float(w4["close"].max())
        mn = float(w4["close"].min())
        ret = (cf/c0 - 1) * 100
        dd  = (mn/mx - 1) * 100
        vol = float(w4["close"].pct_change().std() * 100)
        print(f"  BTC inicio Oct 2025: {c0:,.0f}")
        print(f"  BTC fin Dic 2025:    {cf:,.0f}")
        print(f"  Retorno total:       {ret:+.1f}%")
        print(f"  ATH en periodo:      {mx:,.0f}")
        print(f"  Minimo en periodo:   {mn:,.0f}")
        print(f"  Drawdown maximo:     {dd:.1f}%")
        print(f"  Volatilidad horaria: {vol:.4f}%")
        for m, nm in [(10,"Oct"),(11,"Nov"),(12,"Dic")]:
            mm = w4[w4.index.month == m]
            if len(mm) > 0:
                c_ini = float(mm["close"].iloc[0])
                c_fin = float(mm["close"].iloc[-1])
                r_m = (c_fin/c_ini - 1) * 100
                print(f"  {nm}: {r_m:+.1f}% | {c_ini:,.0f} -> {c_fin:,.0f}")
    else:
        print(f"  W4 OOS sin datos (filas={len(w4)}). Max dato disponible: {source_df.index.max().date()}")
print()

# H4: Trades OOS ya generados
print("H4: TRADES OOS YA GENERADOS (ENSEMBLE PARCIAL)")
trades_files = sorted(BASE.glob("**/oos_trades_W*.parquet"))
print(f"Archivos encontrados: {len(trades_files)}")
total = 0
for tf in trades_files:
    try:
        t = pd.read_parquet(tf)
        print(f"  {tf.name}: {len(t)} trades")
        total += len(t)
    except Exception as e:
        print(f"  {tf.name}: ERROR {e}")
print(f"TOTAL trades hasta ahora: {total}")
if len(trades_files) > 0:
    media = total / len(trades_files)
    est = media * 48
    print(f"Media por combo seed/ventana: {media:.2f}")
    print(f"Estimacion ensemble 12 seeds x 4 ventanas: ~{est:.0f} trades")
else:
    print("Sin datos de trades aun")
