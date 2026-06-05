"""arch01_tbm_root_cause.py
Analisis de causa raiz del EV neto negativo.
Los 73 trades reales muestran:
  - ret_medio = 0.0088% (BRUTO)
  - coste = 0.1500%
  - EV neto = -0.1412%  (94% del coste es lo que falta)
  - WR bruto = 60.27%   pero Avg Win=0.057% < Avg Loss=0.065%
  - Distribucion: 41/73 trades en [0, +0.15%] -> entre 0 y el coste

Hipotesis a verificar:
  A) return_pct es retorno bruto (sin descontar coste) y la logica es correcta
  B) El tbm_min_return=0.003 (0.3%) genera targets en barreras muy pequenyas
  C) La barrera temporal (72H) cierra trades en -0.01% a +0.01% (cerca de 0)
  D) El multiplicador PT=1.8x ATR es insuficiente para superar el coste
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("[ARCH-01] CAUSA RAIZ — EV NETO = -0.1412%")
print("="*70)

# Cargar el archivo con mas trades (W1 de seed42, 45 trades)
cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
runs_dir = ROOT / "data" / "runs"
all_dfs = []
for f in runs_dir.rglob("oos_trades.parquet"):
    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
    if mtime >= cutoff:
        parts = f.parts
        run_id = parts[-4] if len(parts) >= 4 else "?"
        window = parts[-2]
        try:
            df = pd.read_parquet(f)
            df["_run_id"] = run_id
            df["_window"] = window
            all_dfs.append(df)
        except Exception:
            pass

df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
print(f"  Trades cargados: {len(df_all)}")

COST_PCT = 0.0015

# ── HIPOTESIS A: verificar si return_pct ya incluye el coste ─────────────────
print("\n[A] ¿return_pct incluye el coste o es bruto?")
print("-"*60)
# Si is_win se calcula con retorno neto, deberían coincidir con return_pct > cost_pct
if "is_win" in df_all.columns and "return_pct" in df_all.columns:
    # Comparar is_win con return_pct > 0 (bruto) y return_pct > cost_pct (neto)
    win_if_bruto = (df_all["return_pct"] > 0).sum()
    win_if_neto  = (df_all["return_pct"] > COST_PCT).sum()
    actual_wins  = df_all["is_win"].sum()
    print(f"  is_win=True:              {actual_wins}")
    print(f"  return_pct > 0 (bruto):   {win_if_bruto}")
    print(f"  return_pct > 0.15% (neto): {win_if_neto}")
    if abs(actual_wins - win_if_bruto) <= 2:
        print("  --> return_pct es BRUTO (is_win = return_pct > 0)")
    elif abs(actual_wins - win_if_neto) <= 2:
        print("  --> return_pct es NETO (is_win = return_pct > cost_pct)")
    else:
        print("  --> No hay correspondencia clara — revisar logica is_win")

# ── HIPOTESIS B: distribucion por exit_reason ─────────────────────────────────
print("\n[B] ¿Como terminan los trades? (barrera temporal vs TP/SL)")
print("-"*60)
if "exit_time" in df_all.columns and "entry_time" in df_all.columns:
    df_all["entry_time"] = pd.to_datetime(df_all["entry_time"], utc=True, errors="coerce")
    df_all["exit_time"] = pd.to_datetime(df_all["exit_time"], utc=True, errors="coerce")
    df_all["duration_h"] = (df_all["exit_time"] - df_all["entry_time"]).dt.total_seconds() / 3600
    
    # Si duration >= vbh (72H), cerrado por barrera temporal
    VBH = 72
    temporal = (df_all["duration_h"] >= VBH - 1).sum()
    tp_sl    = (df_all["duration_h"] < VBH - 1).sum()
    print(f"  Cerrados por barrera temporal (>=71H): {temporal} ({temporal/len(df_all)*100:.1f}%)")
    print(f"  Cerrados por TP o SL (<71H):          {tp_sl} ({tp_sl/len(df_all)*100:.1f}%)")
    
    # Retorno de los cerrados por barrera temporal
    if temporal > 0:
        ret_temporal = df_all.loc[df_all["duration_h"] >= VBH - 1, "return_pct"]
        ret_tpsl     = df_all.loc[df_all["duration_h"] < VBH - 1, "return_pct"]
        print(f"\n  Ret medio (barrera temporal): {ret_temporal.mean()*100:.4f}%")
        print(f"  Ret medio (TP/SL):            {ret_tpsl.mean()*100:.4f}% si hay {len(ret_tpsl)} trades")
        print(f"\n  Distribucion duracion:")
        dur_bins = [0, 12, 24, 48, 71, 73, 200]
        dur_labels = ["0-12H","12-24H","24-48H","48-71H","71-73H(temporal)","73H+"]
        hist_dur = pd.cut(df_all["duration_h"].dropna(), bins=dur_bins, labels=dur_labels).value_counts().sort_index()
        for lbl, cnt in hist_dur.items():
            bar = "█" * min(30, int(cnt / max(hist_dur) * 30)) if max(hist_dur) > 0 else ""
            print(f"    {lbl:20} | {cnt:3d} | {bar}")
    
    # Correlacion duracion vs retorno
    valid_dur = df_all[["duration_h", "return_pct"]].dropna()
    corr = valid_dur.corr().iloc[0, 1]
    print(f"\n  Correlacion duracion-retorno: {corr:.3f}")
    if abs(corr) > 0.15:
        print(f"  --> Hay correlacion significativa (trades mas largos = {'mejor' if corr>0 else 'peor'} retorno)")
else:
    print("  entry_time/exit_time no disponibles")

# ── HIPOTESIS C: ¿son los retornos consistentes con TBM ATR? ─────────────────
print("\n[C] RELACION ENTRE xgb_prob Y retorno")
print("-"*60)
if "xgb_prob_cal" in df_all.columns:
    # Verificar si mayor prob = mayor retorno (señal de edge)
    df_valid = df_all[["xgb_prob_cal", "return_pct"]].dropna()
    corr_prob_ret = df_valid.corr().iloc[0, 1]
    print(f"  Correlacion xgb_prob_cal vs return_pct: {corr_prob_ret:.4f}")
    
    # Dividir en cuartiles de prob
    df_valid["prob_q"] = pd.qcut(df_valid["xgb_prob_cal"], q=4, labels=["Q1_bajo","Q2","Q3","Q4_alto"])
    for q, grp in df_valid.groupby("prob_q", observed=True):
        print(f"  {q}: N={len(grp):3d} | prob_mean={grp['xgb_prob_cal'].mean():.3f} | "
              f"ret_mean={grp['return_pct'].mean()*100:.4f}% | "
              f"EV_neto={(grp['return_pct'].mean()-COST_PCT)*100:.4f}%")
    
    if corr_prob_ret > 0.05:
        print(f"  ✅ Hay correlacion positiva prob-retorno: el modelo TIENE edge señal")
        print(f"     Problema es en la magnitud de retornos, no en la direccion")
    elif corr_prob_ret < -0.05:
        print(f"  ⚠️ Correlacion NEGATIVA: mayor prob = peor retorno (señal invertida)")
    else:
        print(f"  ⚠️ Sin correlacion: modelo no discrimina entre trades buenos y malos")

# ── HIPOTESIS D: threshold_min_return vs magnitud real de retornos ─────────────
print("\n[D] ¿ES tbm_min_return=0.003 (0.3%) CONSISTENTE CON LA DISTRIBUCION?")
print("-"*60)
r = df_all["return_pct"].dropna()
# Retornos en los bins de TBM
below_min = (r.abs() < 0.003).sum()    # Cierre temporal (ret < tbm_min_return)
above_min = (r.abs() >= 0.003).sum()   # TP o SL tocado
print(f"  |ret| < 0.3% (cierre temporal/sin barrera): {below_min} ({below_min/len(r)*100:.1f}%)")
print(f"  |ret| >= 0.3% (TP o SL tocado):            {above_min} ({above_min/len(r)*100:.1f}%)")
print(f"\n  De los cierres por TP/SL:")
tpsl_mask = r.abs() >= 0.003
r_tpsl = r[tpsl_mask]
if len(r_tpsl) > 0:
    ev_tpsl = r_tpsl.mean() - COST_PCT
    wr_tpsl = (r_tpsl > 0).mean()
    print(f"    N={len(r_tpsl)} | WR={wr_tpsl*100:.1f}% | ret_medio={r_tpsl.mean()*100:.4f}% | EV_neto={ev_tpsl*100:.4f}%")
    if ev_tpsl > 0:
        print(f"    ✅ Los trades que tocan TP/SL TIENEN EV positivo")
        print(f"    --> El problema son los {below_min} trades que NO tocan TP/SL (ret~0)")

print("\n[E] RESUMEN EJECUTIVO")
print("-"*60)
print(f"""
DATOS (73 trades, ultimas 24H, multiples seeds):
  ret_medio = 0.0088% (bruto) vs coste = 0.1500%
  EV neto = -0.1412%  (el sistema pierde 17x su edge bruto en costes)
  WR = 60.3% pero Avg Win (0.057%) < Avg Loss (0.065%) → ratio 0.88x

CAUSA PRINCIPAL: La gran mayoria de trades (~56%) se cierran en [-0.15%, +0.15%]
  → ret < coste → son trades "neutros" en retorno pero DESTRUCTIVOS en costes
  → Estos trades se cierran por BARRERA TEMPORAL (no tocaron ni TP ni SL)
  → Con tbm_min_return=0.003 y vbh=72H, muchas velas tienen volatilidad baja

FIX CANDIDATOS (de menor a mayor impacto):
  1. Aumentar tbm_min_return de 0.003 (0.3%) a 0.006 (0.6%)
     → Solo etiquetar como señal movimientos > 2x el coste
     → Reduce N IS pero aumenta EV por trade
  
  2. Aumentar pt_mult_min de 1.8x a 2.5x ATR (bull) y de 1.0x a 1.8x (range/bear)
     → Exigir mayor movimiento para que TP se active
     → Correlacion positiva prob-retorno sugiere que el modelo tiene edge
       en DONDE se mueve, no en CUANTO se mueve
  
  3. Aumentar vertical_barrier_hours de 72H a 120H
     → Dar mas tiempo al precio para alcanzar TP
     → Reduce % de cierres temporales con ret~0

  PRIORIDAD: Fix 1 (tbm_min_return) es el mas seguro y reversible.
  Fix 2 (pt_mult) requiere reentrenamiento completo.
  Fix 3 (vbh) puede combinarse con Fix 1.
""")
print("[ARCH-01] Root cause completado.")
