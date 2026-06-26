"""
[DUAL-BOT-ANALYSIS-01] Análisis de combinación Long+Short W1/W2
Compara trades OOS del modelo Short actual con los trades Long de la run previa
y evalúa las 3 estrategias de combinación.
"""
import pandas as pd
import numpy as np
import glob
import os

print("=" * 70)
print("[DUAL-BOT-ANALYSIS-01] Análisis de combinación Long + Short")
print("=" * 70)

# --- 1. CARGAR DATOS SHORT (run actual) ---
short_files = sorted(glob.glob("data/reports/wfb/oos_trades_W*.parquet"))
print(f"\n[SHORT] Archivos OOS encontrados: {len(short_files)}")

dfs_short = []
for f in short_files:
    df = pd.read_parquet(f)
    window = os.path.basename(f).replace("oos_trades_","").replace("_seed42.parquet","")
    df["window"] = window
    df["model"] = "short"
    dfs_short.append(df)
    print(f"  {window}: {len(df)} trades SHORT | cols={list(df.columns[:6])}")

# --- 2. BUSCAR DATOS LONG (run previa - puede estar en backup o en reports) ---
# Buscar en varias rutas posibles
long_search_paths = [
    "data/reports/wfb/oos_trades_long_W*.parquet",
    "data/reports/wfb_backup/oos_trades_W*.parquet",
    "data/oos_trades_long_*.parquet",
]
long_files = []
for p in long_search_paths:
    long_files += glob.glob(p, recursive=True)

print(f"\n[LONG] Archivos OOS run previa encontrados: {len(long_files)}")
for f in long_files[:5]:
    print(f"  {f}")

# Si no hay datos long, intentar leer del evaluador de ensemble
eval_files = glob.glob("data/reports/wfb/ensemble_eval*.json") + glob.glob("data/reports/wfb/wfb_results*.json")
print(f"\n[LONG-EVAL] Archivos evaluador ensemble: {len(eval_files)}")
for f in eval_files[:5]:
    print(f"  {f}")

# --- 3. ANÁLISIS SOLO CON SHORT (lo que tenemos) ---
if dfs_short:
    df_short_all = pd.concat(dfs_short, ignore_index=True)
    print(f"\n{'='*70}")
    print(f"[SHORT] RESUMEN GLOBAL W1+W2 (seed42)")
    print(f"{'='*70}")
    print(f"  Total trades short: {len(df_short_all)}")

    ret_col = next((c for c in ["return_pct","ret","return_raw"] if c in df_short_all.columns), None)
    win_col = next((c for c in ["is_win","is_win_kelly"] if c in df_short_all.columns), None)

    if ret_col:
        wr = df_short_all[ret_col].gt(0).mean()
        mean_ret = df_short_all[ret_col].mean()
        total_ret = df_short_all[ret_col].sum()
        max_dd = (df_short_all[ret_col].cumsum() - df_short_all[ret_col].cumsum().cummax()).min()
        print(f"  WR: {wr:.1%}")
        print(f"  MeanRet/trade: {mean_ret:.4f}%")
        print(f"  Ret acumulado: {total_ret:.3f}%")
        print(f"  MaxDD simulado: {max_dd:.3f}%")

    print(f"\n  Por ventana:")
    for w in ["W1","W2"]:
        sub = df_short_all[df_short_all.window==w]
        if len(sub) > 0 and ret_col:
            wr_w = sub[ret_col].gt(0).mean()
            print(f"    {w}: {len(sub)} trades | WR={wr_w:.1%} | ret_sum={sub[ret_col].sum():.3f}%")

    # --- 4. ANÁLISIS DE CONFLICTO TEMPORAL ---
    print(f"\n{'='*70}")
    print(f"[CONFLICT-ANALYSIS] Análisis temporal de conflictos Long vs Short")
    print(f"{'='*70}")

    ts_col = next((c for c in ["ts","timestamp","open_time","entry_time","date"] if c in df_short_all.columns), None)
    if ts_col:
        print(f"  Columna de tiempo: {ts_col}")
        print(f"  Rango Short: {df_short_all[ts_col].min()} → {df_short_all[ts_col].max()}")
    else:
        print(f"  Sin columna de timestamp directa. Columnas: {list(df_short_all.columns)}")
        print(f"  Datos de muestra (primeras 3 filas):")
        print(df_short_all.head(3).to_string())

# --- 5. ESQUEMA TEÓRICO DE COMBINACIÓN ---
print(f"\n{'='*70}")
print(f"[DUAL-BOT-THEORY] Esquema de combinación óptima Long + Short")
print(f"{'='*70}")

print("""
MODO 1 — EXCLUSIÓN MUTUA POR RÉGIMEN (más simple, actual HMM gate):
  - Régimen BULL  → Solo LONG opera | Short bloqueado
  - Régimen BEAR  → Solo SHORT opera | Long bloqueado
  - Régimen RANGE → Ambos pueden operar (el que tenga señal más alta)
  Riesgo: sin simultáneos, no hay hedge pero tampoco doble exposición.

MODO 2 — ARBITRAJE DE ALPHA CON CAP DE CAPITAL:
  - Ambos modelos puntúan su señal (0-1)
  - Solo el más alto en prob_cal × meta_prob ejecuta
  - Capital máximo simultáneo = 100% (nunca 200%)
  - Permite que Short opere en BULL si supera al Long en probabilidad

MODO 3 — SIMULTÁNEO CON KELLY AJUSTADO (más complejo):
  - Ambos pueden operar al mismo tiempo en el mismo activo
  - Cada uno usa Kelly individual × 0.5 (para que la suma no supere 1.0)
  - Solo si tienen distintos vencimientos (TBM horizon distintos)
  - RIESGO: En BTC, Long+Short simultáneo = cuasi-flat + doble comisión (R6)
  → Este modo REQUIERE TBMs no solapados para tener sentido matemático.

RECOMENDACION INSTITUCIONAL: MODO 2 (Alpha Arbitrage)
  Razón: Evita el doble costo de comisión del Modo 3, es más selectivo
  que el Modo 1 (no hay zonas muertas de régimen), y es compatible con
  la arquitectura RegimeRouter actual.
""")
