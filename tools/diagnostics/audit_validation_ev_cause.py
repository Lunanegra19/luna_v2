"""
tools/diagnostics/audit_validation_ev_cause.py
Investiga la causa real del EV negativo en los periodos de validacion reales
(2024-Q4 a 2025-Q4) usando los parquets de validacion disponibles.
"""
import pandas as pd, numpy as np, json
from pathlib import Path

ROOT = Path("g:/Mi unidad/ia/luna_v2")
FEATURES = ROOT / "data" / "features"
MODELS = ROOT / "data" / "models"

print("=== AUDIT: CAUSA DEL EV NEGATIVO EN VALIDACION ===")
print()

# Leer calibration_report completo de cada agente
for agent in ["bull", "range"]:
    sig = json.loads((MODELS / f"xgboost_meta_{agent}_long_signature.json").read_text(encoding="utf-8"))
    rep = sig.get("calibration_report", [])
    if not rep:
        continue
    print(f"--- {agent.upper()} calibration_report ({len(rep)} entradas) ---")
    # Mostrar todas las entradas para ver el sweep completo
    best = max(rep, key=lambda x: x.get("ev", -999))
    worst = min(rep, key=lambda x: x.get("ev", 999))
    print(f"  Mejor threshold: thr={best.get('threshold', best.get('t'))} "
          f"ev={best.get('ev'):.5f} wr={best.get('win_rate', best.get('wr', '?'))} "
          f"n={best.get('n_trades', best.get('n', '?'))}")
    print(f"  Peor threshold : thr={worst.get('threshold', worst.get('t'))} "
          f"ev={worst.get('ev'):.5f}")
    print(f"  Primer entry   : {rep[0]}")
    print(f"  Estructura keys: {list(rep[0].keys())}")
    print()

# Analizar mercado BTC en los periodos de validacion reales
print("=== ANALISIS DE MERCADO EN PERIODOS DE VALIDACION ===")
for w in ["W1", "W2", "W3", "W4", "W5"]:
    p = FEATURES / f"features_validation_{w}.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p, columns=["close"])
    # Estadisticas de retorno
    rets = df["close"].pct_change().dropna()
    trend = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
    vol   = rets.std() * np.sqrt(24*365)  # vol anualizada hourly
    sharpe_dir = trend / (rets.std() * np.sqrt(len(rets))) if rets.std() > 0 else 0
    # % de barras con retorno positivo (bullishness)
    bull_pct = (rets > 0).mean()
    # Autocorrelacion lag-1 (tendencia vs reversion)
    autocorr = rets.autocorr(1)
    print(f"  {w} ({df.index.min().date()} -> {df.index.max().date()}):")
    print(f"    Return total    : {trend:+.2%}")
    print(f"    Vol anualizada  : {vol:.1%}")
    print(f"    % barras alcistas: {bull_pct:.1%}")
    print(f"    Autocorr lag-1  : {autocorr:.3f} ({'momentum' if autocorr > 0 else 'reversal'})")
    print(f"    Interpretacion  : {'BULL fuerte' if trend > 0.05 else 'BEAR' if trend < -0.05 else 'LATERAL/VOLATIL'}")
    print()

# Comparar regimen HMM en validacion vs holdout
print("=== REGIMENES HMM EN VALIDACION VS HOLDOUT ===")
for w in ["W1", "W3", "W5"]:
    for split, suffix in [("validation", f"_validation_{w}"), ("holdout", f"_holdout_{w}")]:
        p = FEATURES / f"features{suffix}.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            if "HMM_Semantic" in df.columns:
                vc = df["HMM_Semantic"].value_counts(normalize=True)
                top = vc.head(3)
                print(f"  {w} {split}: " + " | ".join(f"{r}={v:.0%}" for r, v in top.items()))
        except Exception as e:
            print(f"  {w} {split}: error {e}")
