import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

WFB = Path("data/reports/wfb")
dfs = []
for f in sorted(WFB.glob("oos_trades_W*_seed*.parquet")):
    parts = f.stem.split("_")
    df = pd.read_parquet(f)
    df["_w"]    = parts[2]
    df["_seed"] = parts[3].replace("seed", "")
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined["entry_dt"] = pd.to_datetime(combined["entry_time"], utc=True, errors="coerce")
combined["hour_utc"] = combined["entry_dt"].dt.hour
combined["dow"]      = combined["entry_dt"].dt.dayofweek
baseline_wr = combined["is_win"].mean()

print("HORA 7H-13H — robustez por ventana:")
header = f"  {'Ventana':>8} {'N_in':>6} {'WR_in':>8} {'N_out':>6} {'WR_out':>8} {'Delta':>8} {'p':>8}"
print(header)
for w in ["W2","W3","W4","W5"]:
    sub = combined[combined["_w"]==w]
    ins = sub[(sub["hour_utc"]>=7) & (sub["hour_utc"]<=13)]
    out = sub[~((sub["hour_utc"]>=7) & (sub["hour_utc"]<=13))]
    if len(ins)<5 or len(out)<5: continue
    wi, wo = ins["is_win"].mean(), out["is_win"].mean()
    ct = stats.chi2_contingency([
        [int(ins["is_win"].sum()), len(ins)-int(ins["is_win"].sum())],
        [int(out["is_win"].sum()), len(out)-int(out["is_win"].sum())]
    ])
    p = ct[1]
    f = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "~" if p<0.15 else ""
    print(f"  {w:>8} {len(ins):>6} {wi:>8.4f} {len(out):>6} {wo:>8.4f} {wi-wo:>+8.4f} {p:>8.4f} {f}")

print()
print("SIMULACION metricas — gate hora 7H-13H solamente:")
filt1 = combined[(combined["hour_utc"]>=7) & (combined["hour_utc"]<=13)].sort_values("entry_dt")
ret1 = filt1["return_raw"].fillna(0)
eq1  = (1+ret1).cumprod()
dd1  = ((eq1-eq1.cummax())/eq1.cummax()).min()
tot1 = eq1.iloc[-1]-1
sh1  = (ret1.mean()/ret1.std()*np.sqrt(716)) if ret1.std()>0 else 0
cal1 = abs(tot1/dd1) if dd1!=0 else 0
print(f"  N={len(filt1)} WR={filt1['is_win'].mean():.4f} ret={tot1*100:+.1f}% MaxDD={dd1*100:.1f}% Sharpe={sh1:.2f} Calmar={cal1:.2f}")

print()
print("SIMULACION metricas — gate COMBINADO (excl.Lunes + 7H-13H):")
filt2 = combined[(combined["dow"]!=0) & (combined["hour_utc"]>=7) & (combined["hour_utc"]<=13)].sort_values("entry_dt")
ret2 = filt2["return_raw"].fillna(0)
eq2  = (1+ret2).cumprod()
dd2  = ((eq2-eq2.cummax())/eq2.cummax()).min()
tot2 = eq2.iloc[-1]-1
sh2  = (ret2.mean()/ret2.std()*np.sqrt(716)) if ret2.std()>0 else 0
cal2 = abs(tot2/dd2) if dd2!=0 else 0
print(f"  N={len(filt2)} WR={filt2['is_win'].mean():.4f} ret={tot2*100:+.1f}% MaxDD={dd2*100:.1f}% Sharpe={sh2:.2f} Calmar={cal2:.2f}")

print()
print("Lunes por ventana:")
header2 = f"  {'Ventana':>8} {'N_Lun':>7} {'WR_Lun':>8} {'N_Rst':>7} {'WR_Rst':>8} {'Delta':>8} {'p':>8}"
print(header2)
for w in ["W2","W3","W4","W5"]:
    sub = combined[combined["_w"]==w]
    lun = sub[sub["dow"]==0]
    rst = sub[sub["dow"]!=0]
    if len(lun)<5: continue
    wl, wr = lun["is_win"].mean(), rst["is_win"].mean()
    ct = stats.chi2_contingency([
        [int(lun["is_win"].sum()), len(lun)-int(lun["is_win"].sum())],
        [int(rst["is_win"].sum()), len(rst)-int(rst["is_win"].sum())]
    ])
    p = ct[1]
    f = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "~" if p<0.15 else ""
    print(f"  {w:>8} {len(lun):>7} {wl:>8.4f} {len(rst):>7} {wr:>8.4f} {wl-wr:>+8.4f} {p:>8.4f} {f}")
