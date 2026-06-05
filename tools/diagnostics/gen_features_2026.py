"""
gen_features_2026.py — Genera features_2026.parquet usando el pipeline canónico
para el periodo 2026-01-01 → hoy, usando los raw parquets que el live bot mantiene
actualizados en tiempo real.

Pasos:
  1. Verifica el rango temporal de los raw parquets disponibles
  2. Carga DataCollector (modo histórico — sin re-fetch de APIs)
  3. Corre FeaturePipeline sobre el dataset completo
  4. Filtra barras 2026 y guarda features_2026.parquet
  5. Imprime stats del resultado
"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUT_PATH     = FEATURES_DIR / "features_2026_replay.parquet"

print("=" * 70)
print("[GEN-FEATURES-2026] Inicio generación de features holdout 2026")
print("=" * 70)

# ── 1. Verificar raw data disponible ─────────────────────────────────────────
print("\n[GEN-FEATURES-2026] Rango temporal raw parquets:")
for p in sorted(RAW_DIR.glob("*/*raw.parquet")):
    try:
        df_tmp = pd.read_parquet(p)
        df_tmp.index = pd.to_datetime(df_tmp.index, utc=True)
        print(f"  {p.parent.name:<15s} {df_tmp.index.min().date()} → {df_tmp.index.max().date()} ({len(df_tmp)} rows)")
    except Exception as e:
        print(f"  {p.name}: ERROR {e}")

# ── 2. Cargar DataCollector en modo histórico (usa raw parquets en disco) ────
print("\n[GEN-FEATURES-2026] Cargando DataCollector (modo historical — sin API calls)...")
t0 = time.monotonic()

from luna.data.data_collector import DataCollector
collector = DataCollector()
# Modo historical: solo carga parquets en disco, sin descargar nada
# En VPS el bot ya tiene los datos raw actualizados
collector.build(mode="historical")
raw_data = collector.load_all()

print(f"[GEN-FEATURES-2026] DataCollector cargado en {time.monotonic()-t0:.1f}s")
print(f"[GEN-FEATURES-2026] Categorías disponibles: {list(raw_data.keys())}")

if "ohlcv_1h" in raw_data:
    ohlcv = raw_data["ohlcv_1h"]
    ohlcv.index = pd.to_datetime(ohlcv.index, utc=True)
    print(f"[GEN-FEATURES-2026] OHLCV range: {ohlcv.index.min().date()} → {ohlcv.index.max().date()}")
    rows_2026 = (ohlcv.index >= "2026-01-01").sum()
    rows_2025 = ((ohlcv.index >= "2025-01-01") & (ohlcv.index < "2026-01-01")).sum()
    print(f"[GEN-FEATURES-2026] Barras 2025: {rows_2025} | Barras 2026: {rows_2026}")

# ── 3. Correr FeaturePipeline ─────────────────────────────────────────────────
print("\n[GEN-FEATURES-2026] Ejecutando FeaturePipeline...")
t1 = time.monotonic()

try:
    from luna.features.feature_pipeline import FeaturePipeline
    pipeline = FeaturePipeline(raw_data)
    df_features = pipeline.build()
    print(f"[GEN-FEATURES-2026] FeaturePipeline completado en {time.monotonic()-t1:.1f}s")
except Exception as e:
    print(f"[GEN-FEATURES-2026] ERROR en FeaturePipeline: {e}")
    # Intentar carga directa del features_live que el bot actualiza cada ciclo
    print("[GEN-FEATURES-2026] Intentando usar features_live.parquet del bot...")
    live_path = FEATURES_DIR / "features_live.parquet"
    if live_path.exists():
        df_features = pd.read_parquet(live_path)
        print(f"[GEN-FEATURES-2026] features_live.parquet cargado: {df_features.shape}")
    else:
        print("[GEN-FEATURES-2026] FATAL: No se pueden generar features. Abortando.")
        sys.exit(1)

df_features.index = pd.to_datetime(df_features.index, utc=True)
print(f"[GEN-FEATURES-2026] Features totales: {df_features.shape} | {df_features.index.min().date()} → {df_features.index.max().date()}")

# ── 4. Filtrar 2026 ───────────────────────────────────────────────────────────
df_2026 = df_features[df_features.index >= "2026-01-01"].copy()
n_2026 = len(df_2026)
print(f"\n[GEN-FEATURES-2026] Barras 2026 (Jan 1 → hoy): {n_2026}")

if n_2026 == 0:
    # Si no hay 2026, usar desde 2025-06-01 como proxy del periodo reciente
    df_2026 = df_features[df_features.index >= "2025-06-01"].copy()
    n_2026 = len(df_2026)
    print(f"[GEN-FEATURES-2026] Sin datos 2026 — usando desde 2025-06-01: {n_2026} barras")

if n_2026 == 0:
    print("[GEN-FEATURES-2026] ERROR: No hay datos suficientemente recientes. Abortando.")
    sys.exit(1)

# ── 5. Guardar y confirmar ────────────────────────────────────────────────────
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
df_2026.to_parquet(OUT_PATH)
print(f"[GEN-FEATURES-2026] ✅ Guardado: {OUT_PATH} ({n_2026} barras × {df_2026.shape[1]} cols)")

# Mostrar columnas HMM disponibles
hmm_cols = [c for c in df_2026.columns if "HMM" in c or "Regime" in c or "regime" in c]
print(f"[GEN-FEATURES-2026] Columnas HMM/Regime: {hmm_cols}")

print(f"\n[GEN-FEATURES-2026] Total elapsed: {time.monotonic()-t0:.1f}s")
print("[GEN-FEATURES-2026] Listo para correr oos_replay_2026.py con features_2026_replay.parquet")
