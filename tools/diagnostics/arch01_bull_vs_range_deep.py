"""arch01_bull_vs_range_deep.py
El agente BULL tiene WR=42.2% EV=-0.73% con 45 trades OOS.
El agente RANGE tiene WR=89.3% EV=+1.43% con 28 trades OOS.
Este analisis investiga:
1. Que probabilidades XGBoost tiene el agente BULL (threshold demasiado bajo?)
2. Si el BULL opera con el modelo correcto o con el baseline fallback
3. Si la asimetria BULL vs RANGE es estructural o casual (pocos datos OOS)
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("[ARCH-01] BULL vs RANGE — DIAGNOSTICO ASIMETRIA DE EV")
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

# Separar BULL y RANGE
bull = df_all[df_all["hmm_regime"] == "1_BULL_TREND"].copy()
rng  = df_all[df_all["hmm_regime"] == "2_VOLATILE_RANGE"].copy()

print(f"\n  BULL:  {len(bull)} trades | RANGE: {len(rng)} trades")

# ── 1. Probabilidades XGBoost por agente ─────────────────────────────────────
print("\n[1] PROBABILIDADES XGBoost EN BULL vs RANGE")
print("-"*60)
for name, df in [("BULL", bull), ("RANGE", rng)]:
    if "xgb_prob_cal" not in df.columns:
        continue
    p = df["xgb_prob_cal"].dropna()
    thresh = df["signal_threshold"].dropna()
    print(f"\n  {name} ({len(p)} trades):")
    print(f"    xgb_prob_cal: media={p.mean():.4f} | min={p.min():.4f} | max={p.max():.4f}")
    print(f"    signal_threshold: media={thresh.mean():.4f} | min={thresh.min():.4f} | max={thresh.max():.4f}")
    print(f"    Margen prob sobre threshold: {(p - thresh).mean():.4f} (media)")
    print(f"    threshold_was_lowered: {df['threshold_was_lowered'].sum()} de {len(df)} trades")

# ── 2. Distribucion de retornos BULL vs RANGE ─────────────────────────────────
print("\n[2] DISTRIBUCION return_raw — BULL vs RANGE")
print("-"*60)
for name, df in [("BULL", bull), ("RANGE", rng)]:
    r = df["return_raw"].dropna()
    if len(r) == 0:
        continue
    wr = (r > 0).mean()
    avg_win  = r[r > 0].mean() if (r > 0).any() else 0.0
    avg_loss = abs(r[r < 0].mean()) if (r < 0).any() else 0.0
    ev = r.mean()
    print(f"\n  {name}: WR={wr*100:.1f}% | EV={ev*100:.4f}% | AvgWin={avg_win*100:.2f}% | AvgLoss={avg_loss*100:.2f}%")
    
    # Percentiles
    print(f"    P10={r.quantile(0.1)*100:.2f}% | P25={r.quantile(0.25)*100:.2f}% | "
          f"P50={r.quantile(0.5)*100:.2f}% | P75={r.quantile(0.75)*100:.2f}% | P90={r.quantile(0.9)*100:.2f}%")
    
    # Grandes perdidas
    big_losses = r[r < -0.02]
    if len(big_losses) > 0:
        print(f"    Trades > -2%: {len(big_losses)} ({len(big_losses)/len(r)*100:.1f}%)")
        print(f"    Mayor perdida: {r.min()*100:.2f}%")

# ── 3. El agente BULL podria estar usando el baseline/fallback? ───────────────
print("\n[3] DIAGNOSTICO: ¿BULL usa baseline o modelo especializado?")
print("-"*60)
# Si el BULL usa el baseline, sus probabilidades serian las del modelo global
# xgboost_meta.model, no del especializado xgboost_meta_bull_long.model
# Una señal: el margen prob-threshold es muy pequeno en BULL (prob ~ threshold)
if "xgb_prob_cal" in bull.columns and "signal_threshold" in bull.columns:
    bull_margin = (bull["xgb_prob_cal"] - bull["signal_threshold"]).dropna()
    rng_margin  = (rng["xgb_prob_cal"] - rng["signal_threshold"]).dropna()
    print(f"  Margen prob-threshold:")
    print(f"    BULL:  media={bull_margin.mean():.4f} | std={bull_margin.std():.4f}")
    print(f"    RANGE: media={rng_margin.mean():.4f} | std={rng_margin.std():.4f}")
    if bull_margin.mean() < 0.01:
        print(f"  ⚠️ BULL: margen muy bajo ({bull_margin.mean():.4f}) — "
              f"señales estan en el limite del threshold")
    
# ── 4. Verificar firma del agente BULL activo ─────────────────────────────────
print("\n[4] FIRMA DEL AGENTE BULL ACTIVO")
print("-"*60)
import json
sigs = list((ROOT/"data"/"models").glob("*bull*signature*.json"))
for sig_path in sigs:
    try:
        sig = json.loads(sig_path.read_text("utf-8"))
        print(f"\n  {sig_path.name}:")
        print(f"    threshold: {sig.get('optimal_threshold')}")
        print(f"    dsr_oos: {sig.get('dsr_oos')}")
        print(f"    target_base_rate: {sig.get('target_base_rate')}")
        print(f"    brier: {sig.get('xgb_brier_raw')}")
        print(f"    n_features: {len(sig.get('features',[]))}")
    except Exception as e:
        print(f"  ERROR {sig_path.name}: {e}")

# ── 5. Conclusion y recomendacion ─────────────────────────────────────────────
print("\n[5] CONCLUSION")
print("-"*60)
bull_wr = (bull["return_raw"] > 0).mean() if len(bull) > 0 else 0
bull_ev = bull["return_raw"].mean() if len(bull) > 0 else 0
print(f"""
SITUACION:
  BULL:  WR={bull_wr*100:.1f}% | EV={bull_ev*100:.4f}% | N=45
  RANGE: WR=89.3% | EV=+1.43% | N=28

El agente RANGE tiene WR=89.3% — esto es extraordinariamente alto.
Con 28 trades y WR=89%, la probabilidad binomial de que sea ruido:
  P(X>=25 de 28 | p=0.5) ≈ muy baja, pero p=0.5 es el baseline
  Con modelo que da prob=0.62 (CUTOFF = 0.62), WR esperado > 62%
  WR=89% podria ser parcialmente ruido estadistico con N=28

El agente BULL con WR=42% es claramente deficiente.
Un modelo random daría 50%, este da 42% — PEOR QUE AZAR.
Posibles causas:
  1. El modelo bull usa threshold demasiado bajo (señales de baja calidad)
  2. El agente bull esta operando en un regimen adverso (BTC bull en OOS 2025?)
  3. El baseline FIX-NEW-03 esta siendo activado (modelo bull sin modelo entrenado?)
  4. Las features del bull son data snooping (buenas IS, malas OOS)
""")
print("[ARCH-01] Diagnostico BULL vs RANGE completado.")
