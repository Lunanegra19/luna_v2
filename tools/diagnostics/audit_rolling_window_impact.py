"""
audit_rolling_window_impact.py
===============================
Audita el impacto del rolling_window_years en las barras IS disponibles
por régimen HMM para cada ventana WFB.

Compara:
  - rolling 3 años (actual)
  - rolling 5 años (propuesto)
  - expanding (full IS)

Calcula cuántas barras de cada régimen tiene cada agente disponible
para entrenamiento bajo cada configuración.

RULE[fixbugsprints.md]: prints de trazabilidad en cada paso.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

print("[AUDIT-ROLLING] Iniciando auditoría de rolling window IS/OOS...")
print(f"[AUDIT-ROLLING] Proyecto root: {ROOT}")

# ── 1. Cargar features_train.parquet ──────────────────────────────────────────
features_path = ROOT / "data" / "features" / "features_train.parquet"
hmm_path      = ROOT / "data" / "features" / "hmm_regime_labels.parquet"

if not features_path.exists():
    print(f"[AUDIT-ROLLING] ERROR: features_train.parquet no encontrado en {features_path}")
    sys.exit(1)

print(f"[AUDIT-ROLLING] Cargando features_train.parquet...")
df = pd.read_parquet(features_path, columns=["close"])
print(f"[AUDIT-ROLLING] Shape: {df.shape} | Fechas: {df.index.min().date()} -> {df.index.max().date()}")

# ── 2. Cargar HMM labels ───────────────────────────────────────────────────────
if hmm_path.exists():
    print(f"[AUDIT-ROLLING] Cargando hmm_regime_labels.parquet...")
    df_hmm = pd.read_parquet(hmm_path)
    # Join
    for col in [c for c in df_hmm.columns if c in df.columns]:
        df = df.drop(columns=[col])
    df = df.join(df_hmm, how="left")
    print(f"[AUDIT-ROLLING] HMM integrado. Columnas: {list(df_hmm.columns)}")
else:
    print(f"[AUDIT-ROLLING] WARN: hmm_regime_labels.parquet no encontrado. Usando HMM_Semantic sintético.")
    df["HMM_Semantic"] = "UNKNOWN"

# Asegurar timezone UTC
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")

# ── 3. Definición de ventanas WFB (de settings.yaml) ─────────────────────────
windows = [
    {"id": "W1", "train_end": "2024-10-31"},
    {"id": "W2", "train_end": "2025-01-31"},
    {"id": "W3", "train_end": "2025-04-30"},
    {"id": "W4", "train_end": "2025-07-31"},
    {"id": "W5", "train_end": "2025-10-31"},
]

# ── 4. Mapeo de agentes → regímenes ───────────────────────────────────────────
agent_regimes = {
    "bull":       ["1_BULL_TREND", "1_VOLATILE_BULL", "1_BULL_GRIND", "1_BULL_TREND_WEAK",
                   "1_BULL_TREND_B", "1_VOLATILE_BULL_B"],
    "range":      ["2_CALM_RANGE", "2_VOLATILE_RANGE", "2_CALM_RANGE_B", "2_VOLATILE_RANGE_B"],
    "calm_bear":  ["3_CALM_BEAR", "3_CALM_BEAR_B", "3_CALM_BEAR_C", "3_CALM_BEAR_D"],
    "bear":       ["3_BEAR_CRASH", "3_BEAR_CRASH_B", "4_BEAR_FORCED"],
}

# ── 5. Configuraciones a comparar ─────────────────────────────────────────────
configs = {
    "rolling_3y":  3,
    "rolling_5y":  5,
    "expanding":   None,  # None = usar todo
}

# ── 6. Análisis por ventana y configuración ───────────────────────────────────
print("\n" + "="*90)
print("ANÁLISIS DE BARRAS IS POR VENTANA, AGENTE Y CONFIGURACIÓN")
print("="*90)

results = []

for w in windows:
    w_id      = w["id"]
    train_end = pd.Timestamp(w["train_end"], tz="UTC")
    print(f"\n[AUDIT-ROLLING] ── Ventana {w_id} (train_end={train_end.date()}) ──")

    for cfg_name, years in configs.items():
        if years is None:
            # Expanding: desde el inicio
            train_start = df.index.min()
        else:
            train_start = train_end - pd.DateOffset(years=years)

        df_is = df[(df.index >= train_start) & (df.index <= train_end)].copy()
        total_bars = len(df_is)

        print(f"  [{cfg_name}] IS: {train_start.date()} → {train_end.date()} | Total barras: {total_bars:,}")

        for agent, regimes in agent_regimes.items():
            if "HMM_Semantic" in df_is.columns:
                n_bars = df_is["HMM_Semantic"].isin(regimes).sum()
                pct    = n_bars / total_bars * 100 if total_bars > 0 else 0
            else:
                n_bars = 0
                pct    = 0

            # Clasificación de suficiencia
            if n_bars >= 2000:
                flag = "✅ OK"
            elif n_bars >= 500:
                flag = "⚠️  MARGINAL"
            elif n_bars >= 100:
                flag = "🔴 INSUFICIENTE"
            else:
                flag = "💀 CRÍTICO"

            print(f"    Agente '{agent}': {n_bars:>6,} barras ({pct:5.1f}%) {flag}")

            results.append({
                "window":   w_id,
                "config":   cfg_name,
                "agent":    agent,
                "n_bars":   n_bars,
                "pct":      round(pct, 1),
                "total_is": total_bars,
                "train_start": str(train_start.date()),
                "train_end":   str(train_end.date()),
            })

# ── 7. Resumen comparativo por agente ─────────────────────────────────────────
print("\n" + "="*90)
print("RESUMEN COMPARATIVO — GANANCIA DE BARRAS: rolling_3y → rolling_5y → expanding")
print("="*90)

df_res = pd.DataFrame(results)

for agent in agent_regimes.keys():
    print(f"\n  Agente: {agent.upper()}")
    pivot = df_res[df_res["agent"] == agent].pivot_table(
        index="window", columns="config", values="n_bars", aggfunc="first"
    )
    # Reordenar columnas
    cols_order = [c for c in ["rolling_3y", "rolling_5y", "expanding"] if c in pivot.columns]
    pivot = pivot[cols_order]

    # Calcular ganancia
    if "rolling_3y" in pivot.columns and "rolling_5y" in pivot.columns:
        pivot["gain_5y_vs_3y"] = pivot["rolling_5y"] - pivot["rolling_3y"]
    if "rolling_3y" in pivot.columns and "expanding" in pivot.columns:
        pivot["gain_exp_vs_3y"] = pivot["expanding"] - pivot["rolling_3y"]

    print(pivot.to_string())

# ── 8. Análisis de BEAR_CRASH específico (el más crítico) ────────────────────
print("\n" + "="*90)
print("ANÁLISIS CRÍTICO: BEAR_CRASH — Barras IS por año disponibles")
print("="*90)

if "HMM_Semantic" in df.columns:
    bear_df = df[df["HMM_Semantic"].isin(["3_BEAR_CRASH", "3_BEAR_CRASH_B", "4_BEAR_FORCED"])]
    print(f"[AUDIT-ROLLING] Total barras BEAR en IS global: {len(bear_df):,}")
    print(f"[AUDIT-ROLLING] Distribución temporal de BEAR_CRASH:")
    bear_by_year = bear_df.groupby(bear_df.index.year).size()
    for yr, cnt in bear_by_year.items():
        flag = "⬅️ ELIMINADO por rolling 3y en W1" if yr < 2022 else ""
        flag2 = "⬅️ ELIMINADO por rolling 3y en W2+" if yr < 2022 else ""
        print(f"    {yr}: {cnt:>5,} barras {flag}")
else:
    print("[AUDIT-ROLLING] HMM_Semantic no disponible — no se puede desglosar por régimen.")

# ── 9. Comparación IS total barras ────────────────────────────────────────────
print("\n" + "="*90)
print("TOTAL BARRAS IS POR VENTANA Y CONFIGURACIÓN")
print("="*90)

total_pivot = df_res[["window", "config", "total_is"]].drop_duplicates().pivot_table(
    index="window", columns="config", values="total_is", aggfunc="first"
)
cols_order = [c for c in ["rolling_3y", "rolling_5y", "expanding"] if c in total_pivot.columns]
total_pivot = total_pivot[cols_order]
if "rolling_3y" in total_pivot.columns and "expanding" in total_pivot.columns:
    total_pivot["pct_descartado"] = ((total_pivot["expanding"] - total_pivot["rolling_3y"]) / total_pivot["expanding"] * 100).round(1).astype(str) + "%"
print(total_pivot.to_string())

print("\n" + "="*90)
print("[AUDIT-ROLLING] Auditoría completada.")
print("="*90)
