"""
[DUAL-BOT-SEED42-W1W2W3] Analisis trade-a-trade seed42
Long (20-Jun run) vs Short (22-Jun run) en W1, W2, W3
"""
import pandas as pd
import numpy as np
import os

LONG_BASE  = "data/runs/WFB_20260620_095224_seed42/seed42"
SHORT_BASE = "data/reports/wfb"

windows = ["W1", "W2", "W3"]

records_long  = []
records_short = []

for w in windows:
    # LONG
    lp = f"{LONG_BASE}/{w}/oos_trades.parquet"
    if os.path.exists(lp):
        df = pd.read_parquet(lp)
        df["window"] = w
        records_long.append(df)

    # SHORT
    sp = f"{SHORT_BASE}/oos_trades_{w}_seed42.parquet"
    if os.path.exists(sp):
        df = pd.read_parquet(sp)
        df["window"] = w
        records_short.append(df)

df_long  = pd.concat(records_long,  ignore_index=False) if records_long  else pd.DataFrame()
df_short = pd.concat(records_short, ignore_index=False) if records_short else pd.DataFrame()

# Normalizar timestamps
for df, name in [(df_long, "LONG"), (df_short, "SHORT")]:
    if len(df) == 0:
        print(f"[{name}] Sin datos")
        continue
    ts_col = next((c for c in ["entry_time","timestamp","ts"] if c in df.columns), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df_long["ts"] = pd.to_datetime(df_long["entry_time"], utc=True, errors="coerce") if "entry_time" in df_long.columns else df_long.index

df_short.index = pd.to_datetime(df_short.index, utc=True, errors="coerce")
df_short["ts"] = df_short.index

# ─── CABECERA ────────────────────────────────────────────────────────────────
print("=" * 75)
print(" DUAL-BOT SEED42 — Analisis W1, W2, W3 (Long vs Short)")
print("=" * 75)

holdouts = {"W1": "Jul-2025", "W2": "Ago-2025", "W3": "Sep-2025"}

for w in windows:
    sl = df_long[df_long.window  == w] if len(df_long)  else pd.DataFrame()
    ss = df_short[df_short.window == w] if len(df_short) else pd.DataFrame()

    print(f"\n{'='*75}")
    print(f"  VENTANA {w} — Holdout {holdouts.get(w,'?')}")
    print(f"{'='*75}")

    # --- LONG ---
    print(f"\n  [LONG seed42 {w}] {len(sl)} trades")
    if len(sl) > 0:
        r = "return_raw"
        wr = sl[r].gt(0).mean()
        mean_r = sl[r].mean()
        tot_r  = sl[r].sum()
        max_dd = (sl[r].cumsum() - sl[r].cumsum().cummax()).min()
        print(f"  WR={wr:.1%} | MeanRet={mean_r:.4f}% | TotalRet={tot_r:.4f}% | MaxDD={max_dd:.4f}%")
        print(f"  {'Fecha Entry':<26} {'Exit':<26} {'Ret%':>8} {'Win':>5} {'XGB':>7} {'Meta':>7} {'Regime'}")
        print(f"  {'-'*85}")
        for _, row in sl.iterrows():
            win = "Y" if row[r] > 0 else "N"
            regime = str(row.get("HMM_Semantic", row.get("hmm_regime", "??")))[:15]
            entry = str(row.get("entry_time", "?"))[:19]
            exit_  = str(row.get("exit_time",  "?"))[:19]
            xgb = row.get("xgb_prob_cal", row.get("xgb_prob", 0))
            meta = row.get("meta_v2_prob", 0)
            print(f"  {entry:<26} {exit_:<26} {row[r]:>8.4f} {win:>5} {xgb:>7.3f} {meta:>7.3f} {regime}")

    # --- SHORT ---
    print(f"\n  [SHORT seed42 {w}] {len(ss)} trades")
    if len(ss) > 0:
        r = "return_raw"
        wr = ss[r].gt(0).mean()
        mean_r = ss[r].mean()
        tot_r  = ss[r].sum()
        max_dd = (ss[r].cumsum() - ss[r].cumsum().cummax()).min()
        print(f"  WR={wr:.1%} | MeanRet={mean_r:.4f}% | TotalRet={tot_r:.4f}% | MaxDD={max_dd:.4f}%")
        print(f"  {'Fecha Entry':<26} {'XGB':>7} {'Meta':>7} {'Ret%':>8} {'Win':>5}")
        print(f"  {'-'*60}")
        for ts, row in ss.iterrows():
            win = "Y" if row[r] > 0 else "N"
            xgb  = row.get("xgb_prob_cal", row.get("xgb_prob", 0))
            meta = row.get("meta_v2_prob", 0)
            print(f"  {str(ts)[:19]:<26} {xgb:>7.3f} {meta:>7.3f} {row[r]:>8.4f} {win:>5}")

    # --- CONFLICTOS ---
    if len(sl) > 0 and len(ss) > 0:
        long_days  = set(sl["ts"].dt.normalize().dt.strftime("%Y-%m-%d"))
        short_days = set(ss["ts"].dt.normalize().dt.strftime("%Y-%m-%d"))
        ov = sorted(long_days & short_days)
        print(f"\n  [CONFLICTOS {w}] Dias solapados: {len(ov)}")
        if ov:
            for day in ov:
                ld = sl[sl["ts"].dt.strftime("%Y-%m-%d") == day]
                sd = ss[ss["ts"].dt.strftime("%Y-%m-%d") == day]
                lr = ld["return_raw"].values[0] if len(ld) else 0
                sr = sd["return_raw"].values[0] if len(sd) else 0
                lx = ld["xgb_prob_cal"].values[0] if len(ld) and "xgb_prob_cal" in ld.columns else 0
                sx = sd["xgb_prob_cal"].values[0] if len(sd) and "xgb_prob_cal" in sd.columns else 0
                winner = "LONG " if lx >= sx else "SHORT"
                print(f"    {day}: Long xgb={lx:.3f} ret={lr:+.4f}% | Short xgb={sx:.3f} ret={sr:+.4f}% --> GANA {winner}")
        else:
            print(f"    CERO conflictos en {w}")

# ─── RESUMEN GLOBAL W1+W2+W3 ─────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"  RESUMEN GLOBAL seed42 — W1+W2+W3 combinados")
print(f"{'='*75}")

r = "return_raw"
for label, df in [("LONG ", df_long), ("SHORT", df_short)]:
    if len(df) == 0: continue
    sub = df[df.window.isin(["W1","W2","W3"])]
    wr   = sub[r].gt(0).mean()
    mean_r = sub[r].mean()
    tot_r  = sub[r].sum()
    max_dd = (sub[r].cumsum() - sub[r].cumsum().cummax()).min()
    print(f"\n  {label}: {len(sub)} trades | WR={wr:.1%} | MeanRet={mean_r:.4f}% | Total={tot_r:.4f}% | MaxDD={max_dd:.4f}%")

# PORTAFOLIO COMBINADO (Alpha Arbitrage)
sl3 = df_long[df_long.window.isin(["W1","W2","W3"])]  if len(df_long)  else pd.DataFrame()
ss3 = df_short[df_short.window.isin(["W1","W2","W3"])] if len(df_short) else pd.DataFrame()

if len(sl3) > 0 and len(ss3) > 0:
    # Detectar solapados
    long_days  = set(sl3["ts"].dt.normalize().dt.strftime("%Y-%m-%d"))
    short_days = set(ss3["ts"].dt.normalize().dt.strftime("%Y-%m-%d"))
    overlap_days = long_days & short_days

    # En dias sin conflicto: sumar ambos. En conflicto: tomar el ganador
    ret_arb = []
    for day in sorted(long_days | short_days):
        ld = sl3[sl3["ts"].dt.strftime("%Y-%m-%d") == day]
        sd = ss3[ss3["ts"].dt.strftime("%Y-%m-%d") == day]
        if day in overlap_days:
            # Arbitrage: opera el de mayor XGB
            lx = ld["xgb_prob_cal"].values[0] if len(ld) else 0
            sx = sd["xgb_prob_cal"].values[0] if len(sd) else 0
            trade = ld if lx >= sx else sd
            ret_arb.append(trade[r].values[0])
        elif len(ld) > 0:
            ret_arb.extend(ld[r].tolist())
        elif len(sd) > 0:
            ret_arb.extend(sd[r].tolist())

    ret_arb = pd.Series(ret_arb)
    wr_arb  = (ret_arb > 0).mean()
    tot_arb = ret_arb.sum()
    dd_arb  = (ret_arb.cumsum() - ret_arb.cumsum().cummax()).min()
    n_arb   = len(ret_arb)

    print(f"\n  ALPHA ARBITRAGE: {n_arb} trades | WR={wr_arb:.1%} | Total={tot_arb:.4f}% | MaxDD={dd_arb:.4f}%")
    print(f"\n  MEJORA vs Long solo:")
    tot_long = sl3[r].sum()
    wr_long  = sl3[r].gt(0).mean()
    print(f"    Long solo:   {len(sl3)} trades | WR={wr_long:.1%} | Total={tot_long:.4f}%")
    print(f"    Combinado:   {n_arb} trades | WR={wr_arb:.1%} | Total={tot_arb:.4f}%")
    delta_ret = tot_arb - tot_long
    delta_n   = n_arb - len(sl3)
    print(f"    Delta ret:   {delta_ret:+.4f}%  |  Delta trades: {delta_n:+d}")
