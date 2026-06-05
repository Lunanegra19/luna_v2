"""
oos_replay_2026.py — Simula cuántos trades habría generado el ensemble de 12 seeds
en lo que va de 2026 (Jan 1 → hoy), usando los modelos de producción en disco.

Estrategia: carga los features de producción completos, aplica el ensemble
predict-route para cada bar 2026 y cuenta transiciones de estado.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "data" / "models" / "prod"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

print("=" * 70)
print("[OOS-REPLAY-2026] Inicio simulación trades 2026 con ensemble 12 seeds")
print("=" * 70)

# ── 1. Cargar features canónicas ──────────────────────────────────────────────
# [FIX-OOS-REPLAY] features_live.parquet ya tiene 3598 barras de 2026 (actualizado en vivo)
# features_holdout_W5 tiene 2377 barras de 2026 (Jan-Apr) como alternativa
features_path = None
for candidate in [
    FEATURES_DIR / "features_live.parquet",           # tiene hasta hoy, 3598 barras 2026
    FEATURES_DIR / "features_holdout_W5.parquet",     # Jan-Apr 2026, 2377 barras
    FEATURES_DIR / "features_holdout_W4.parquet",     # Oct25-Jan26, 217 barras 2026
    FEATURES_DIR / "features_holdout.parquet",
    FEATURES_DIR / "features_canonical.parquet",
    FEATURES_DIR / "features_prod.parquet",
]:
    if candidate.exists():
        features_path = candidate
        break

if features_path is None:
    for p in FEATURES_DIR.glob("*.parquet"):
        features_path = p
        break

if features_path is None:
    print("[OOS-REPLAY-2026] ERROR: No se encontró parquet de features.")
    sys.exit(1)


print(f"[OOS-REPLAY-2026] Cargando features desde: {features_path.name}")
df_all = pd.read_parquet(features_path)
df_all.index = pd.to_datetime(df_all.index, utc=True)
print(f"[OOS-REPLAY-2026] Dataset completo: {df_all.shape} | {df_all.index.min().date()} → {df_all.index.max().date()}")

# ── 2. Filtrar 2026 ───────────────────────────────────────────────────────────
df_2026 = df_all[df_all.index >= "2026-01-01"].copy()
print(f"[OOS-REPLAY-2026] Barras 2026 (Jan 1 → hoy): {len(df_2026)} horas")

if len(df_2026) == 0:
    # Intentar desde 2025 como holdout
    df_2026 = df_all[df_all.index >= "2025-01-01"].copy()
    print(f"[OOS-REPLAY-2026] Usando holdout 2025+: {len(df_2026)} barras")

# ── 3. Cargar ensemble 12 seeds ───────────────────────────────────────────────
SEEDS = [42, 100, 777, 1337, 2025, 29611, 85199, 43812, 28559, 76576, 62815, 60075]
from luna.models.regime_router import RegimeRouter

seed_routers = {}
for seed in SEEDS:
    seed_dir = MODELS_DIR / f"seed{seed}"
    if seed_dir.exists():
        try:
            router = RegimeRouter(models_dir=seed_dir, agent_type="xgboost", direction="long")
            seed_routers[seed] = router
            print(f"[OOS-REPLAY-2026] ✅ seed{seed} router cargado. Modelos: {list(router.models.keys())}")
        except Exception as e:
            print(f"[OOS-REPLAY-2026] ⚠️ seed{seed} error: {e}")

print(f"\n[OOS-REPLAY-2026] Seeds cargadas: {len(seed_routers)}/12")

# ── 4. HMM compartido ─────────────────────────────────────────────────────────
hmm_pkl = MODELS_DIR / "hmm_regime.pkl"
if not hmm_pkl.exists():
    # Buscar en primera seed
    hmm_pkl = MODELS_DIR / "seed42" / "hmm_regime.pkl"

hmm_semantic = None
if hmm_pkl.exists():
    try:
        import joblib
        hmm_model = joblib.load(hmm_pkl)
        print(f"[OOS-REPLAY-2026] HMM cargado desde {hmm_pkl.name}")
    except Exception as e:
        print(f"[OOS-REPLAY-2026] Error cargando HMM: {e}")
        hmm_model = None
else:
    hmm_model = None
    print("[OOS-REPLAY-2026] HMM no encontrado — usando HMM_Semantic del parquet si existe")

# Usar HMM_Semantic si ya está en el parquet, si no mapear desde HMM_Regime
if "HMM_Semantic" in df_2026.columns:
    print(f"[OOS-REPLAY-2026] HMM_Semantic encontrado en parquet. Distribución:")
    print(df_2026["HMM_Semantic"].value_counts().to_string())
    hmm_precomputed = True
elif "HMM_Regime" in df_2026.columns:
    # [FIX-OOS-REPLAY] Mapear HMM_Regime (int) → HMM_Semantic usando mapping canónico
    REGIME_MAP = {
        1: "1_BULL_TREND",
        2: "2_CALM_RANGE",
        3: "3_BEAR_CRASH",
        4: "4_BEAR_FORCED",
        0: "2_CALM_RANGE",  # fallback
    }
    df_2026 = df_2026.copy()
    df_2026["HMM_Semantic"] = df_2026["HMM_Regime"].map(REGIME_MAP).fillna("2_CALM_RANGE")
    print(f"[OOS-REPLAY-2026] HMM_Semantic generado desde HMM_Regime. Distribución:")
    print(df_2026["HMM_Semantic"].value_counts().to_string())
    hmm_precomputed = True
else:
    hmm_precomputed = False
    print("[OOS-REPLAY-2026] HMM_Semantic NO disponible en parquet — se necesita inferencia HMM")


# ── 5. Inferencia bar-by-bar para 2026 ───────────────────────────────────────
CONSENSUS_CUTOFF = 3  # >= 3/12 seeds votan LONG → señal
decisions = []

print(f"\n[OOS-REPLAY-2026] Inferencia con {len(seed_routers)} seeds sobre {len(df_2026)} barras...")
print(f"[OOS-REPLAY-2026] Consensus threshold: {CONSENSUS_THRESHOLD}/12")

# Ruta rápida: si HMM_Semantic ya está precalculado en el parquet,
# podemos procesar todo de una vez (mucho más rápido que bar-by-bar)
if hmm_precomputed and len(seed_routers) > 0:
    all_seed_probs = []
    
    for seed, router in seed_routers.items():
        try:
            result = router.route_and_predict(df_2026)
            probs = result["raw"].values
            all_seed_probs.append(probs)
            print(f"[OOS-REPLAY-2026] seed{seed}: mean_prob={np.nanmean(probs):.4f} | nonzero={np.sum(probs > 0)}")
        except Exception as e:
            print(f"[OOS-REPLAY-2026] seed{seed} ERROR: {e}")
    
    if all_seed_probs:
        # Stack: shape (n_seeds, n_bars)
        prob_matrix = np.array(all_seed_probs)  # (n_seeds, n_bars)
        
        # Cada seed vota LONG si su prob > 0.5
        vote_matrix = (prob_matrix > 0.5).astype(int)
        vote_counts = vote_matrix.sum(axis=0)  # (n_bars,) — cuántas seeds votan LONG
        
        # Decisión de consenso
        ensemble_decisions = np.where(vote_counts >= CONSENSUS_THRESHOLD, "LONG", "HOLD")
        
        decisions_series = pd.Series(ensemble_decisions, index=df_2026.index)
        
        print(f"\n[OOS-REPLAY-2026] Distribución de decisiones 2026:")
        print(decisions_series.value_counts().to_string())
        
        # ── 6. Contar trades (transiciones de estado) ─────────────────────────
        # LONG entrada = transición HOLD→LONG
        # LONG salida  = transición LONG→HOLD
        prev = decisions_series.shift(1).fillna("HOLD")
        entries = ((prev == "HOLD") & (decisions_series == "LONG")).sum()
        exits   = ((prev == "LONG") & (decisions_series == "HOLD")).sum()
        
        # Posición actual al final
        current_pos = decisions_series.iloc[-1]
        
        # Duración media de posiciones
        position_active = (decisions_series == "LONG").astype(int)
        
        # Calcular duración de cada posición
        in_position = False
        pos_durations = []
        pos_start = None
        pos_timestamps = []
        
        for ts, dec in decisions_series.items():
            if dec == "LONG" and not in_position:
                in_position = True
                pos_start = ts
            elif dec == "HOLD" and in_position:
                in_position = False
                pos_durations.append((ts - pos_start).total_seconds() / 3600)
                pos_timestamps.append(pos_start)
        if in_position:
            pos_durations.append((decisions_series.index[-1] - pos_start).total_seconds() / 3600)
            pos_timestamps.append(pos_start)
        
        print("\n" + "=" * 70)
        print("[OOS-REPLAY-2026] RESULTADO FINAL — TRADES 2026")
        print("=" * 70)
        
        total_bars_2026 = len(df_2026)
        long_bars = (decisions_series == "LONG").sum()
        hold_bars = (decisions_series == "HOLD").sum()
        
        print(f"Periodo analizado:     {df_2026.index.min().date()} → {df_2026.index.max().date()}")
        print(f"Total barras H1:       {total_bars_2026}")
        print(f"Barras en LONG:        {long_bars} ({long_bars/total_bars_2026*100:.1f}%)")
        print(f"Barras en HOLD:        {hold_bars} ({hold_bars/total_bars_2026*100:.1f}%)")
        print(f"")
        print(f"ENTRADAS (LONG):       {entries}")
        print(f"SALIDAS (→HOLD):       {exits}")
        print(f"Trades completos:      {min(entries, exits)}")
        print(f"Posicion actual:       {current_pos}")
        print(f"")
        if pos_durations:
            print(f"Duración media pos:    {np.mean(pos_durations):.1f}h")
            print(f"Duración max pos:      {max(pos_durations):.1f}h")
            print(f"Duración min pos:      {min(pos_durations):.1f}h")
        print(f"")
        print(f"Frecuencia de trades:  {entries / (total_bars_2026/720):.1f} entradas/mes")
        
        # Detalle por mes
        print(f"\nDetalle mensual:")
        decisions_series_copy = decisions_series.copy()
        decisions_series_copy = decisions_series_copy.to_frame("decision")
        decisions_series_copy["entry"] = ((decisions_series_copy["decision"].shift(1).fillna("HOLD") == "HOLD") & 
                                           (decisions_series_copy["decision"] == "LONG"))
        monthly = decisions_series_copy.resample("ME").agg(
            bars=("decision", "count"),
            long_bars=("decision", lambda x: (x=="LONG").sum()),
            entries=("entry", "sum")
        )
        for month, row in monthly.iterrows():
            print(f"  {month.strftime('%Y-%m')}: {int(row['entries'])} entradas | {int(row['long_bars'])}h en LONG / {int(row['bars'])}h totales ({row['long_bars']/row['bars']*100:.0f}%)")
        
        # Vote count distribution
        print(f"\nDistribución de votos de consenso (0-12 seeds votando LONG):")
        vote_series = pd.Series(vote_counts, index=df_2026.index)
        for v in range(0, 13):
            cnt = (vote_series == v).sum()
            bar = "█" * int(cnt/max(vote_series.value_counts())*30)
            print(f"  {v:2d} seeds: {cnt:5d} barras {bar}")
        
else:
    print("[OOS-REPLAY-2026] ⚠️ HMM_Semantic no disponible en parquet — necesita pipeline completo")
    print("[OOS-REPLAY-2026] Listar columnas disponibles:")
    hmm_cols = [c for c in df_2026.columns if "hmm" in c.lower() or "regime" in c.lower() or "HMM" in c]
    print(f"  HMM-related cols: {hmm_cols}")
    print(f"  Total cols: {len(df_2026.columns)}")
    print(f"  Sample cols: {list(df_2026.columns[:20])}")

print("\n[OOS-REPLAY-2026] Simulación completada.")
