"""
[MEJORA-TRADES-01] analyze_embargo_impact.py
============================================
Diagnóstico del cuello de botella del Embargo en el embudo de señales.

El log del run cancelado mostró:
  XGBoost:  2166 señales
  Embargo:    24 señales  ← 99% eliminación

Este script analiza TODOS los parquets OOS disponibles y calcula:
1. Trades por ventana y seed — ¿cuántas ventanas tienen < 32 trades?
2. Distribución temporal de señales — ¿cómo afecta el embargo (72-168H) al ratio de supervivencia?
3. Simulación de LOW_DENSITY_THRESHOLD: si subimos de 20 a 50/100, ¿cuántas ventanas extra alcanzan >= 32 trades?

Uso:
    python tools/diagnostics/analyze_embargo_impact.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Fix encoding Windows cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
RUNS_DIR = PROJECT_ROOT / "data" / "runs"

print("=" * 70)
print("[MEJORA-TRADES-01] Diagnóstico de impacto del Embargo en trades OOS")
print("=" * 70)

# ── 1. Cargar todos los parquets OOS disponibles ──────────────────────────────
records = []
for parquet_path in sorted(RUNS_DIR.rglob("oos_trades*.parquet")):
    parts = parquet_path.parts
    # Extraer seed y ventana del path
    try:
        run_id = next((p for p in parts if p.startswith("WFB_")), "?")
        seed_str = next((p for p in parts if p.startswith("seed")), "?")
        window = next((p for p in parts if p.startswith("W") and len(p) == 2), "?")
        seed_num = int(seed_str.replace("seed", "")) if seed_str != "?" else -1

        df = pd.read_parquet(parquet_path)
        n_trades = len(df)
        win_rate = float(df["is_win"].mean()) if "is_win" in df.columns and n_trades > 0 else float("nan")

        records.append({
            "run_id":   run_id,
            "seed":     seed_num,
            "window":   window,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "path":     str(parquet_path),
        })
    except Exception as e:
        print(f"  [WARN] Error leyendo {parquet_path.name}: {e}")

df_all = pd.DataFrame(records)
print(f"\n[1] Parquets cargados: {len(df_all)} archivos | Seeds: {df_all['seed'].nunique()} | Ventanas: {df_all['window'].nunique()}")
print()

# ── 2. Distribución de trades por ventana ────────────────────────────────────
print("[2] Distribución de n_trades por parquet:")
bins = [0, 10, 20, 32, 50, 100, 9999]
labels = ["0-10", "11-20", "21-32 (INSUF)", "33-50 (OK)", "51-100", ">100"]
df_all["bracket"] = pd.cut(df_all["n_trades"], bins=bins, labels=labels)
print(df_all["bracket"].value_counts().sort_index().to_string())
print()

n_insuf = (df_all["n_trades"] < 32).sum()
n_total = len(df_all)
print(f"  Ventanas con < 32 trades (INSUFICIENTES): {n_insuf}/{n_total} ({n_insuf/n_total*100:.1f}%)")
print(f"  Ventanas con >= 32 trades (OK):           {n_total - n_insuf}/{n_total}")
print()

# ── 3. Por ventana (W1-W5): ¿qué ventana tiene más problemas? ────────────────
print("[3] Trades promedio por ventana WFB:")
pivot = df_all.groupby("window")["n_trades"].agg(["mean", "min", "max", "count"])
pivot.columns = ["media", "min", "max", "n_runs"]
pivot["pct_insuf"] = df_all.groupby("window").apply(lambda x: (x["n_trades"] < 32).mean()).values * 100
print(pivot.round(1).to_string())
print()

# ── 4. Análisis de seed2025 específicamente (MEJORA-WR-01 preparación) ───────
seed2025 = df_all[df_all["seed"] == 2025].sort_values("window")
if not seed2025.empty:
    print("[4] seed2025 — tabla completa:")
    print(seed2025[["run_id", "window", "n_trades", "win_rate"]].to_string(index=False))
    print()

# ── 5. Simulación LOW_DENSITY_THRESHOLD ──────────────────────────────────────
print("[5] Impacto de LOW_DENSITY_THRESHOLD (FIX-EMBARGO-01):")
print("   Con el flag activo, el embargo se reduce a 48H cuando n_candidatos < THRESHOLD.")
print("   Asumiendo ~2166 candidatos típicos post-MetaLabeler (del log real):")
print()

# Con 2166 candidatos nunca cae por debajo de ningún threshold razonable.
# El verdadero problema es la DENSIDAD de candidatos POR VENTANA, no el total.
# Cada ventana tiene ~90 días = 2160 horas. Si el 87% de esas horas tiene señal XGB...

# Cálculo teórico: con embargo H, cuántos trades máximo en una ventana de T días
print(f"  {'Embargo (H)':>12} | {'Ventana (días)':>14} | {'Max trades teórico':>18} | {'Obs'}")
print("  " + "-" * 62)
for embargo_h in [48, 72, 96, 120, 144, 168]:
    for window_days in [90, 60]:
        max_trades = int(window_days * 24 / embargo_h)
        obs = "<- bajo densidad" if embargo_h == 48 else ("<- regimen Bear" if embargo_h == 168 else "")
        print(f"  {embargo_h:>12}H | {window_days:>14} | {max_trades:>18} | {obs}")
print()

print("[5] CONCLUSION:")
print("   Con embargo=168H (regimen Bear) en ventana de 90 dias: MAX 12 trades.")
print("   Con embargo=72H (regimen Bull) en ventana de 90 dias: MAX 30 trades.")
print("   El embargo determina el TECHO de trades, independientemente del MetaLabeler.")
print()
print("   LOW_DENSITY_CUTOFF = 20 rara vez se activa porque hay ~2000+ candidatos.")
print("   El problema real: el REGIMEN HMM determina el embargo (72H Bull vs 168H Bear).")
print()
print("   PROPUESTA A: embargo=96H para TODOS los regimenes (max 22 trades/90d — insuficiente).")
print("   PROPUESTA B: aumentar ventanas WFB de 90 -> 120 dias:")
print("     - Con embargo=72H: MAX 40 trades -- suficiente para CSCV.")
print("     - Con embargo=168H: MAX 17 trades -- sigue fallando en Bear.")
print("   PROPUESTA C (recomendada): B + reducir embargo Bull a 48H = ~60 trades/ventana.")
print()
print("   ACCION INMEDIATA: revisar configuracion de duracion de ventanas WFB en settings.yaml.")
print()
print("[MEJORA-TRADES-01] Diagnóstico completado.")
