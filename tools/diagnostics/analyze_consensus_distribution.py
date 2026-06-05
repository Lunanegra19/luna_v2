"""
analyze_consensus_distribution.py
===================================
Analiza la distribucion de votos por bucket temporal en el WFB ensemble.
Para cada trade de consenso >= 3, cuenta cuantas seeds lo tenian activo.
Simula thresholds 3..8 y reporta trades resultantes + metricas clave.

Uso:
    python tools/diagnostics/analyze_consensus_distribution.py
"""
import os
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"
import io as _io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

print("[CONSENSUS-DIST] Iniciando análisis de distribución de consenso WFB")

WFB_REPORTS = _ROOT / "data" / "reports" / "wfb"

# Seeds del ensemble aprobado
ACTIVE_SEEDS = [42, 100, 777, 1337, 2025, 29611, 85199, 43812, 28559, 76576, 62815, 60075]
WINDOWS = [1, 2, 3, 4, 5]
BUCKET_SIZE = "2h"  # agrupacion temporal del consensus engine (pandas 2.x: lowercase)

print(f"[CONSENSUS-DIST] Seeds activas: {ACTIVE_SEEDS}")
print(f"[CONSENSUS-DIST] Ventanas WFB: W1..W5")
print(f"[CONSENSUS-DIST] Bucket temporal: {BUCKET_SIZE}")

# ── 1. Cargar todos los trades OOS por seed/ventana ──
all_trades = []
missing = []

for seed in ACTIVE_SEEDS:
    for w in WINDOWS:
        parquet_path = WFB_REPORTS / f"oos_trades_W{w}_seed{seed}.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                df["seed"] = seed
                df["window"] = w
                all_trades.append(df)
                print(f"[LOAD] W{w}_seed{seed}: {len(df)} trades cargados")
            except Exception as e:
                print(f"[WARN] Error leyendo W{w}_seed{seed}: {e}")
        else:
            flag_path = WFB_REPORTS / f"oos_trades_W{w}_seed{seed}_EMPTY.flag"
            if flag_path.exists():
                print(f"[EMPTY] W{w}_seed{seed}: sin trades (flag EMPTY)")
            else:
                missing.append(f"W{w}_seed{seed}")
                print(f"[MISSING] W{w}_seed{seed}: archivo no encontrado")

if not all_trades:
    print("[ERROR] No se encontraron trades en ninguna ventana. Abortando.")
    sys.exit(1)

df_all = pd.concat(all_trades, ignore_index=True)
print(f"\n[CONSENSUS-DIST] Total filas cargadas: {len(df_all)}")
print(f"[CONSENSUS-DIST] Columnas disponibles: {list(df_all.columns)}")

# ── 2. Normalizar timestamp ──
# Buscar columna de timestamp
ts_col = None
for col in ["entry_time", "entry_dt", "timestamp", "date", "open_time", "entry"]:
    if col in df_all.columns:
        ts_col = col
        break

if ts_col is None:
    print(f"[ERROR] No se encontró columna de timestamp. Columnas: {list(df_all.columns)}")
    print("[INFO] Intentando con primera columna datetime...")
    for col in df_all.columns:
        if pd.api.types.is_datetime64_any_dtype(df_all[col]):
            ts_col = col
            break

if ts_col is None:
    print("[ERROR] No hay columna datetime. Abortando.")
    sys.exit(1)

print(f"[CONSENSUS-DIST] Usando columna timestamp: '{ts_col}'")

df_all[ts_col] = pd.to_datetime(df_all[ts_col], utc=True, errors="coerce")
df_all = df_all.dropna(subset=[ts_col])

# Filtrar al período holdout común: 2025-01-16 → 2026-01-19 (período del Gauntlet Ensemble)
df_holdout = df_all[
    (df_all[ts_col] >= "2025-01-16") &
    (df_all[ts_col] <= "2026-01-19")
].copy()
print(f"[CONSENSUS-DIST] Trades en período holdout (2025-01-16→2026-01-19): {len(df_holdout)}")

# ── 3. Agrupar en buckets temporales de 2H y contar votos por seed ──
df_holdout["bucket"] = df_holdout[ts_col].dt.floor(BUCKET_SIZE)

# Por cada bucket, contar cuántas seeds tienen al menos 1 trade
bucket_votes = (
    df_holdout.groupby("bucket")["seed"]
    .nunique()
    .rename("n_seeds_voting")
    .reset_index()
)

print(f"\n[CONSENSUS-DIST] Buckets temporales únicos con ≥1 trade: {len(bucket_votes)}")

# ── 4. Distribución de votos ──
print("\n" + "="*60)
print("DISTRIBUCIÓN DE VOTOS POR BUCKET (cuántas seeds acuerdan)")
print("="*60)

vote_dist = bucket_votes["n_seeds_voting"].value_counts().sort_index()
for n_seeds, count in vote_dist.items():
    pct = count / len(bucket_votes) * 100
    bar = "█" * int(pct / 2)
    print(f"  {n_seeds:2d} seeds acuerdan: {count:4d} buckets ({pct:5.1f}%) {bar}")

# ── 5. Simular distintos thresholds ──
print("\n" + "="*60)
print("SIMULACIÓN POR THRESHOLD DE CONSENSO")
print("="*60)

# Obtener retorno por bucket (promedio de todas las seeds que votaron)
ret_col = None
for col in ["ret", "return", "pnl", "trade_return", "net_return", "r"]:
    if col in df_holdout.columns:
        ret_col = col
        break

if ret_col:
    print(f"[INFO] Columna de retorno encontrada: '{ret_col}'")
    bucket_metrics = df_holdout.groupby("bucket").agg(
        n_seeds=("seed", "nunique"),
        avg_return=(ret_col, "mean"),
        n_trades_total=(ret_col, "count")
    ).reset_index()
else:
    print("[WARN] Sin columna de retorno — solo conteo de trades")
    bucket_metrics = df_holdout.groupby("bucket").agg(
        n_seeds=("seed", "nunique")
    ).reset_index()

print(f"\n{'Threshold':>10} | {'Buckets':>8} | {'Seeds_pct':>10} | {'WR_estimada':>12} | {'Trades_unicos':>14}")
print("-" * 70)

for thr in range(2, min(13, len(ACTIVE_SEEDS) + 1)):
    filtered = bucket_metrics[bucket_metrics["n_seeds"] >= thr]
    n_buckets = len(filtered)
    seeds_pct = thr / len(ACTIVE_SEEDS) * 100

    if ret_col and n_buckets > 0:
        wr = (filtered["avg_return"] > 0).mean() * 100
        wr_str = f"{wr:11.1f}%"
    else:
        wr_str = f"{'N/A':>11}"

    print(f"  >= {thr:2d}/12  | {n_buckets:8d} | {seeds_pct:9.1f}% | {wr_str} | {n_buckets:>14}")

# ── 6. Resumen ejecutivo ──
print("\n" + "="*60)
print("RESUMEN EJECUTIVO — RECOMENDACIÓN THRESHOLD")
print("="*60)

for thr in [3, 4, 5, 6]:
    n = len(bucket_metrics[bucket_metrics["n_seeds"] >= thr])
    annual_rate = n  # approx 1 año de datos
    sop_r8_ok = "✅ SOP R8" if n >= 30 else "❌ < 30 trades"
    print(f"  Threshold {thr}: {n:3d} trades/año  {sop_r8_ok}")

# ── 7. Histograma de distribución acumulada ──
print("\n" + "="*60)
print("DISTRIBUCIÓN ACUMULADA (trades con >= N seeds)")
print("="*60)
total_buckets = len(bucket_votes)
for n in range(1, min(13, len(ACTIVE_SEEDS) + 1)):
    count = len(bucket_votes[bucket_votes["n_seeds_voting"] >= n])
    pct = count / total_buckets * 100 if total_buckets > 0 else 0
    bar = "█" * int(pct / 3)
    print(f"  >= {n:2d} seeds: {count:4d} buckets ({pct:5.1f}%) {bar}")

print("\n[CONSENSUS-DIST] Análisis completado.")
