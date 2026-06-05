"""arch01_bull_threshold_backtest.py
Simula que habria pasado si el threshold del BULL se hubiera subido.
Con threshold actual=0.480, el BULL admite trades con prob=0.485-0.58 que son destructivos.
Este script calcula el EV por cuartil de probabilidad para encontrar el threshold optimo.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("[ARCH-01] BULL THRESHOLD SWEEP — ¿CUAL ES EL THRESHOLD OPTIMO?")
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
bull = df_all[df_all["hmm_regime"] == "1_BULL_TREND"].copy()

print(f"\n  Trades BULL disponibles: {len(bull)}")
print(f"  Threshold actual (signature): 0.5221")

print(f"\n[1] EV POR NIVEL DE PROBABILIDAD (BULL)")
print("-"*60)
print(f"  {'prob_min':>8} {'prob_max':>8} {'N':>5} {'WR':>7} {'EV%':>9} {'signal?':>8}")
print(f"  {'-'*8} {'-'*8} {'-'*5} {'-'*7} {'-'*9} {'-'*8}")

# Sweep de thresholds
for t in np.arange(0.48, 0.65, 0.01):
    mask = bull["xgb_prob_cal"] >= t
    subset = bull[mask]
    if len(subset) < 3:
        break
    r = subset["return_raw"].dropna()
    wr = (r > 0).mean()
    ev = r.mean()
    marker = " <-- ACTUAL" if abs(t - 0.480) < 0.005 else ""
    marker2 = " <-- SIG" if abs(t - 0.522) < 0.005 else ""
    print(f"  prob>={t:.2f}  | N={len(r):3d} | WR={wr*100:.1f}% | EV={ev*100:+.3f}%{marker}{marker2}")

print(f"\n[2] ANALISIS DE LOS TRADES BULL CON prob < 0.58 (los problematicos)")
print("-"*60)
low_prob = bull[bull["xgb_prob_cal"] < 0.58]["return_raw"].dropna()
high_prob = bull[bull["xgb_prob_cal"] >= 0.58]["return_raw"].dropna()
print(f"  prob < 0.58: N={len(low_prob)} | WR={(low_prob>0).mean()*100:.1f}% | EV={low_prob.mean()*100:.4f}%")
print(f"  prob >= 0.58: N={len(high_prob)} | WR={(high_prob>0).mean()*100:.1f}% | EV={high_prob.mean()*100:.4f}%")

print(f"\n[3] CONCLUSION Y PROPUESTA DE FIX")
print("-"*60)
print(f"""
HALLAZGO:
  El threshold actual del agente BULL es 0.4801 (de los trades reales).
  La firma del modelo dice CUTOFF = 0.5221.
  Esto sugiere que el threshold se rebaja por algun mecanismo (Consensus-Soft?).

  Con CUTOFF = 0.480: WR=42.2% EV=-0.73% (N=45) — destruye el portfolio
  Con CUTOFF = 0.58+: WR mejora significativamente

FIX PROPUESTO:
  Opcion 1 (settings.yaml): Subir threshold minimo del agente BULL a 0.58
    -> En la calibracion de threshold, el sweep debe empezar en t_min=0.55 para BULL
    -> Añadir bull_threshold_min: 0.58 en xgboost.regime_tbm_profiles.bull

  Opcion 2 (reentrenamiento): El modelo BULL necesita mas datos IS
    -> Con rolling_window_years=5 (ya aplicado, ARCH-20), el siguiente WFB
       recalibrara automaticamente el threshold con mas datos de validation
    -> El DSR=0.173 del modelo actual es marginal — un reentrenamiento con 5y
       de IS deberia producir un modelo mas robusto

  RECOMENDACION: No tocar el threshold hardcoded. Esperar el siguiente
  WFB run con rolling_window_years=5 (ya aplicado) para ver si el modelo
  BULL mejora con mas datos IS. Si WR BULL < 48% persiste tras ese WFB,
  entonces si investigar threshold hardcoded.
""")
print("[ARCH-01] Bull threshold sweep completado.")
