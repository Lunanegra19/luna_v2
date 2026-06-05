"""arch01_return_raw_analysis.py
Analiza return_raw (= ret_bruto = retorno TBM del precio - coste, sin Kelly).
Este es el verdadero retorno del trade antes del position sizing.
return_pct = return_raw * kelly_fraction (0.035) -> escala a 3.5% del capital.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("[ARCH-01] return_raw — RETORNO BRUTO REAL DEL TBM (pre-Kelly)")
print("="*70)

cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
runs_dir = ROOT / "data" / "runs"
all_dfs = []
for f in runs_dir.rglob("oos_trades.parquet"):
    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
    if mtime >= cutoff:
        try:
            df = pd.read_parquet(f)
            df["_run_id"] = f.parts[-4]
            df["_window"] = f.parts[-2]
            all_dfs.append(df)
        except Exception:
            pass

df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
COST_PCT = 0.0015

r_raw = df_all["return_raw"].dropna()  # ya descontado el coste, sin kelly
r_pct = df_all["return_pct"].dropna()  # = return_raw * kelly_fraction

print(f"\n  Trades totales: {len(df_all)}")

# Verificar la relacion matematica
print(f"\n[1] VERIFICACION MATEMATICA return_pct = return_raw * kelly")
print("-"*60)
df_valid = df_all[["return_raw", "return_pct", "tribe_mult"]].dropna()
df_valid["computed_ret_pct"] = df_valid["return_raw"] * df_valid["tribe_mult"]
df_valid["diff"] = (df_valid["computed_ret_pct"] - df_valid["return_pct"]).abs()
print(f"  Max diferencia computed vs actual: {df_valid['diff'].max():.8f}")
print(f"  --> return_pct = return_raw * tribe_mult: {'CONFIRMADO' if df_valid['diff'].max() < 1e-6 else 'NO COINCIDE'}")

print(f"\n[2] ESTADISTICAS DE return_raw (retorno bruto precio - coste)")
print("-"*60)
print(f"  N trades:  {len(r_raw)}")
print(f"  Media:     {r_raw.mean()*100:.4f}%  <- retorno bruto por trade")
print(f"  Mediana:   {r_raw.median()*100:.4f}%")
print(f"  Std:       {r_raw.std()*100:.4f}%")
print(f"  Min:       {r_raw.min()*100:.4f}%")
print(f"  Max:       {r_raw.max()*100:.4f}%")

wr = (r_raw > 0).mean()  # is_win = r_raw > 0 (ya descontado coste)
avg_win  = r_raw[r_raw > 0].mean() if (r_raw > 0).any() else 0.0
avg_loss = abs(r_raw[r_raw < 0].mean()) if (r_raw < 0).any() else 0.0
ev = wr * avg_win - (1 - wr) * avg_loss

print(f"\n  Win Rate:      {wr*100:.2f}%")
print(f"  Avg Win:       {avg_win*100:.4f}%")
print(f"  Avg Loss:      {avg_loss*100:.4f}%")
print(f"  Ratio W/L:     {avg_win/avg_loss:.2f}x" if avg_loss > 0 else "  Ratio W/L:     INF")
print(f"  EV (ya neto):  {ev*100:.4f}%  <- EV verdadero por trade")

if ev > 0:
    print(f"  ✅ El sistema tiene EV POSITIVO de {ev*100:.4f}% por trade")
    print(f"     return_pct={r_pct.mean()*100:.4f}% era Kelly-scaled (kelly~3.5%)")
    print(f"     En terminos absolutos: {ev*100/0.035:.2f}% / trade sin Kelly")
else:
    print(f"  ⚠️ EV NEGATIVO: {ev*100:.4f}% por trade")

print(f"\n[3] DISTRIBUCION return_raw (ya incluye coste descontado)")
print("-"*60)
bins = [-np.inf, -0.02, -0.01, -0.005, -0.002, 0, 0.002, 0.005, 0.01, 0.02, np.inf]
labels = ["<-2%", "-2/-1%", "-1/-0.5%", "-0.5/-0.2%", "-0.2/0%",
          "0/+0.2%", "+0.2/+0.5%", "+0.5/+1%", "+1/+2%", ">+2%"]
hist = pd.cut(r_raw, bins=bins, labels=labels).value_counts().sort_index()
for label, count in hist.items():
    bar = "█" * min(35, int(count / max(hist) * 35)) if max(hist) > 0 else ""
    ev_marker = " ← coste=0.15% ya descontado" if label == "0/+0.2%" else ""
    print(f"  {label:15} | {count:3d} | {bar}{ev_marker}")

print(f"\n[4] POR AGENTE (hmm_regime)")
print("-"*60)
if "hmm_regime" in df_all.columns:
    for regime, grp in df_all.groupby("hmm_regime"):
        r_ag = grp["return_raw"].dropna()
        if len(r_ag) < 1:
            continue
        ev_ag = (r_ag > 0).mean() * r_ag[r_ag>0].mean() - (r_ag <= 0).mean() * abs(r_ag[r_ag<=0].mean()) if len(r_ag[r_ag<=0]) > 0 else r_ag.mean()
        print(f"  {str(regime):25} | N={len(r_ag):3d} | "
              f"WR={((r_ag>0).mean()*100):.1f}% | "
              f"EV={r_ag.mean()*100:.4f}%"
              f"{'  ✅' if r_ag.mean() > 0 else '  ⚠️'}")

print(f"\n[5] CORRELACION prob vs return_raw")
print("-"*60)
if "xgb_prob_cal" in df_all.columns:
    df_corr = df_all[["xgb_prob_cal","return_raw"]].dropna()
    corr = df_corr.corr().iloc[0,1]
    print(f"  xgb_prob_cal vs return_raw: {corr:.4f}")
    df_corr["prob_q"] = pd.qcut(df_corr["xgb_prob_cal"], q=4, labels=["Q1_bajo","Q2","Q3","Q4_alto"])
    for q, grp in df_corr.groupby("prob_q", observed=True):
        print(f"  {q}: prob_mean={grp['xgb_prob_cal'].mean():.3f} | "
              f"EV={grp['return_raw'].mean()*100:.4f}%"
              f"{'  ✅' if grp['return_raw'].mean() > 0 else '  ⚠️'}")

print("\n[ARCH-01] Analisis return_raw completado.")
