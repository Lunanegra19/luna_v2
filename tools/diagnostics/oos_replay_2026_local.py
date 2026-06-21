"""
oos_replay_2026_local.py — Re-infiere el HMM de producción sobre datos 2026 (local)
y ejecuta el replay del ensemble de 12 seeds con el régimen correcto.

Ejecutar en local porque tiene acceso a los modelos prod + features_live.parquet.
"""
import sys, os, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "data" / "models" / "prod"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUT_JSON     = PROJECT_ROOT / "data" / "reports" / "oos_replay_2026_result.json"

print("=" * 70)
print("[OOS-REPLAY-2026-LOCAL] Re-inferencia HMM prod + replay 12 seeds")
print("=" * 70)

# ── 1. Cargar features_live ──────────────────────────────────────────────────
features_path = FEATURES_DIR / "features_live.parquet"
if not features_path.exists():
    print(f"[ERROR] No existe {features_path}. Ejecuta sync_data_lake.py primero.")
    sys.exit(1)

print(f"[OOS-REPLAY-2026-LOCAL] Cargando {features_path.name}...")
t0 = time.monotonic()
df_all = pd.read_parquet(features_path)
df_all.index = pd.to_datetime(df_all.index, utc=True)
print(f"[OOS-REPLAY-2026-LOCAL] Dataset: {df_all.shape} | {df_all.index.min().date()} -> {df_all.index.max().date()}")

# ── 2. Filtrar 2026 ──────────────────────────────────────────────────────────
df_2026 = df_all[df_all.index >= "2026-01-01"].copy()
n_2026 = len(df_2026)
print(f"[OOS-REPLAY-2026-LOCAL] Barras 2026: {n_2026} ({df_2026.index.min().date()} -> {df_2026.index.max().date()})")

if n_2026 == 0:
    print("[ERROR] No hay datos 2026 en features_live.parquet")
    sys.exit(1)

# ── 3. Re-inferir HMM de producción sobre 2026 ──────────────────────────────
# Necesitamos el contexto histórico completo para el HMM (ventana deslizante)
# Usamos los últimos 6 meses de contexto + 2026

CTX_START = "2025-07-01"
df_ctx = df_all[df_all.index >= CTX_START].copy()
print(f"\n[OOS-REPLAY-2026-LOCAL] Contexto HMM: {df_ctx.shape[0]} barras desde {CTX_START}")

# Buscar HMM pkl de producción
hmm_candidates = [
    PROJECT_ROOT / "data" / "models" / "hmm_regime.pkl",  # ruta correcta local
    MODELS_DIR / "hmm_regime.pkl",
    MODELS_DIR / "seed42" / "hmm_regime.pkl",
]
hmm_pkl = None
for c in hmm_candidates:
    if c.exists():
        hmm_pkl = c
        break

if hmm_pkl is None:
    print("[OOS-REPLAY-2026-LOCAL] ERROR: No se encontro hmm_regime.pkl en produccion")
    sys.exit(1)

print(f"[OOS-REPLAY-2026-LOCAL] HMM cargando desde: {hmm_pkl}")
hmm_bundle = joblib.load(hmm_pkl)

# [FIX-HMM-FORMAT] Nuestro pkl es un dict custom con model/scaler/state_map/features
if isinstance(hmm_bundle, dict):
    hmm_model  = hmm_bundle["model"]
    hmm_scaler = hmm_bundle["scaler"]
    state_map  = hmm_bundle["state_map"]         # {int_state: "SEMANTIC_NAME"}
    hmm_feats  = hmm_bundle.get("features", [])
    print(f"[OOS-REPLAY-2026-LOCAL] HMM custom format | n_components={hmm_model.n_components} | state_map={state_map}")
    print(f"[OOS-REPLAY-2026-LOCAL] HMM features: {hmm_feats}")
else:
    # Bare GaussianHMM — no debería ocurrir pero protegemos
    hmm_model  = hmm_bundle
    hmm_scaler = None
    state_map  = {}
    hmm_feats  = ["close", "volume", "VIX", "SP500"]
    print(f"[OOS-REPLAY-2026-LOCAL] HMM bare format | n_components={hmm_model.n_components}")

# ── 4. Preparar features HMM (exactamente las del bundle) ────────────────────
available_hmm = [f for f in hmm_feats if f in df_ctx.columns]
missing_hmm   = [f for f in hmm_feats if f not in df_ctx.columns]
if missing_hmm:
    print(f"[OOS-REPLAY-2026-LOCAL] WARNING: HMM features faltantes: {missing_hmm} — usando las disponibles")
if not available_hmm:
    # Fallback a features básicas
    available_hmm = [f for f in ["close", "volume", "VIX", "SP500"] if f in df_ctx.columns]

print(f"[OOS-REPLAY-2026-LOCAL] HMM features usadas: {available_hmm}")

X_ctx = df_ctx[available_hmm].ffill().fillna(0).values

# Escalar con el scaler canónico del bundle (preserva la calibración del entrenamiento)
if hmm_scaler is not None:
    try:
        X_ctx_scaled = hmm_scaler.transform(X_ctx)
        print("[OOS-REPLAY-2026-LOCAL] Scaler del bundle aplicado OK")
    except Exception as e:
        print(f"[OOS-REPLAY-2026-LOCAL] Scaler error: {e} — usando fit_transform de emergencia")
        from sklearn.preprocessing import StandardScaler
        scaler_tmp = StandardScaler()
        X_ctx_scaled = scaler_tmp.fit_transform(X_ctx)
else:
    from sklearn.preprocessing import StandardScaler
    scaler_tmp = StandardScaler()
    X_ctx_scaled = scaler_tmp.fit_transform(X_ctx)

# Forward-predict con el HMM de producción
print("[OOS-REPLAY-2026-LOCAL] Ejecutando HMM predict sobre contexto completo...")
hmm_states = hmm_model.predict(X_ctx_scaled)
print(f"[OOS-REPLAY-2026-LOCAL] HMM predict OK. Estados unicos: {np.unique(hmm_states)}")

df_ctx = df_ctx.copy()
df_ctx["HMM_State_Prod"] = hmm_states

# ── 5. Mapear estados HMM con el state_map canónico del bundle ────────────────
# El state_map ya viene calibrado: {2: '1_BULL_TREND', 0: '3_BEAR_CRASH', ...}
# Necesitamos mapear a los 4 regímenes simplificados que usa el RegimeRouter

# Normalizar state_map a los 4 regímenes canónicos del RegimeRouter
CANONICAL_MAP = {}
for state_int, semantic in state_map.items():
    # Agrupar variantes: BULL_TREND_B → BULL_TREND, BEAR_CRASH_B → BEAR_CRASH
    if "BULL" in semantic:
        CANONICAL_MAP[state_int] = "1_BULL_TREND"
    elif "BEAR_FORCED" in semantic:
        CANONICAL_MAP[state_int] = "4_BEAR_FORCED"
    elif "BEAR" in semantic:
        CANONICAL_MAP[state_int] = "3_BEAR_CRASH"
    else:
        CANONICAL_MAP[state_int] = "2_CALM_RANGE"

print(f"\n[OOS-REPLAY-2026-LOCAL] Mapping canonico estado->regimen:")
for s, r in CANONICAL_MAP.items():
    cnt = int((hmm_states == s).sum())
    print(f"  Estado {s} ({state_map.get(s,'?')}) -> {r}: {cnt} barras")

df_ctx["HMM_Semantic_Prod"] = df_ctx["HMM_State_Prod"].map(CANONICAL_MAP).fillna("2_CALM_RANGE")

# Volatilidad por régimen para validación
if "close" in df_ctx.columns:
    df_ctx["ret_tmp"] = df_ctx["close"].pct_change()
    print(f"\n[OOS-REPLAY-2026-LOCAL] Volatilidad por regimen (validacion):")
    for reg in df_ctx["HMM_Semantic_Prod"].unique():
        mask = df_ctx["HMM_Semantic_Prod"] == reg
        vol = df_ctx.loc[mask, "ret_tmp"].std() * np.sqrt(8760)
        print(f"  {reg}: vol={vol:.4f} | {mask.sum()} barras")
    df_ctx.drop(columns=["ret_tmp"], inplace=True)

# Filtrar solo 2026 con el nuevo HMM_Semantic_Prod
df_2026_hmm = df_ctx[df_ctx.index >= "2026-01-01"].copy()
df_2026_hmm["HMM_Semantic"] = df_2026_hmm["HMM_Semantic_Prod"]

print(f"\n[OOS-REPLAY-2026-LOCAL] Distribucion HMM_Semantic (prod) en 2026:")
print(df_2026_hmm["HMM_Semantic"].value_counts().to_string())


# ── 6. Cargar ensemble 12 seeds ──────────────────────────────────────────────
SEEDS = [42, 100, 777, 1337, 2025, 29611, 85199, 43812, 28559, 76576, 62815, 60075]
from luna.models.regime_router import RegimeRouter

seed_routers = {}
for seed in SEEDS:
    seed_dir = MODELS_DIR / f"seed{seed}"
    if seed_dir.exists():
        try:
            router = RegimeRouter(models_dir=seed_dir, agent_type="xgboost", direction="long")
            seed_routers[seed] = router
        except Exception as e:
            print(f"[OOS-REPLAY-2026-LOCAL] seed{seed} error: {e}")

print(f"\n[OOS-REPLAY-2026-LOCAL] Seeds cargadas: {len(seed_routers)}/12")

# ── 7. Inferencia ensemble sobre 2026 con HMM prod correcto ─────────────────
CONSENSUS_CUTOFF = 3
all_seed_probs = []

for seed, router in seed_routers.items():
    try:
        result = router.route_and_predict(df_2026_hmm)
        # Usar probabilidades calibradas (SOP R10)
        probs = result["calibrated"].fillna(result["raw"]).values
        all_seed_probs.append(probs)
        bear_count = (df_2026_hmm["HMM_Semantic"] == "3_BEAR_CRASH").sum()
        zero_count = (probs == 0.0).sum()
        print(f"[OOS-REPLAY-2026-LOCAL] seed{seed}: mean_prob={np.nanmean(probs):.4f} | "
              f"nonzero={np.sum(probs>0)} | bear_bars={bear_count} | zero_prob={zero_count}")
    except Exception as e:
        print(f"[OOS-REPLAY-2026-LOCAL] seed{seed} ERROR: {e}")

# ── 8. Calcular decisiones de consenso ───────────────────────────────────────
if not all_seed_probs:
    print("[ERROR] No hay seeds con resultados válidos")
    sys.exit(1)

prob_matrix = np.array(all_seed_probs)  # (n_seeds, n_bars)
vote_matrix = (prob_matrix > 0.5).astype(int)
vote_counts  = vote_matrix.sum(axis=0)   # (n_bars,)

ensemble_decisions = np.where(vote_counts >= CONSENSUS_CUTOFF, "LONG", "HOLD")
decisions_series = pd.Series(ensemble_decisions, index=df_2026_hmm.index)

print(f"\n[OOS-REPLAY-2026-LOCAL] Distribución de decisiones 2026 (HMM prod):")
print(decisions_series.value_counts().to_string())

# ── 9. Calcular trades y métricas ────────────────────────────────────────────
prev = decisions_series.shift(1).fillna("HOLD")
entries_mask = ((prev == "HOLD") & (decisions_series == "LONG"))
exits_mask   = ((prev == "LONG") & (decisions_series == "HOLD"))

entries = entries_mask.sum()
exits   = exits_mask.sum()

# Reconstruir posiciones para métricas
trade_list = []
in_position = False
pos_start = None
pos_entry_price = None

for ts, dec in decisions_series.items():
    close_px = df_2026_hmm.loc[ts, "close"] if ts in df_2026_hmm.index else None
    if close_px is None:
        continue

    if dec == "LONG" and not in_position:
        in_position = True
        pos_start = ts
        pos_entry_price = close_px
    elif dec == "HOLD" and in_position:
        in_position = False
        ret = (close_px - pos_entry_price) / pos_entry_price
        dur_h = (ts - pos_start).total_seconds() / 3600
        trade_list.append({
            "type": "SIMULATED_2026",
            "entry_date": pos_start.strftime("%Y-%m-%d %H:%M"),
            "exit_date": ts.strftime("%Y-%m-%d %H:%M"),
            "entry_price": round(pos_entry_price, 2),
            "exit_price": round(close_px, 2),
            "return_pct": round(ret * 100, 4),
            "duration_h": round(dur_h, 1),
            "regime": df_2026_hmm.loc[pos_start, "HMM_Semantic"] if pos_start in df_2026_hmm.index else "N/A"
        })

if in_position and pos_start is not None:
    last_ts = df_2026_hmm.index[-1]
    last_px = df_2026_hmm["close"].iloc[-1]
    ret = (last_px - pos_entry_price) / pos_entry_price
    dur_h = (last_ts - pos_start).total_seconds() / 3600
    trade_list.append({
        "type": "SIMULATED_2026_OPEN",
        "entry_date": pos_start.strftime("%Y-%m-%d %H:%M"),
        "exit_date": "ABIERTA",
        "entry_price": round(pos_entry_price, 2),
        "exit_price": round(last_px, 2),
        "return_pct": round(ret * 100, 4),
        "duration_h": round(dur_h, 1),
        "regime": df_2026_hmm.loc[pos_start, "HMM_Semantic"] if pos_start in df_2026_hmm.index else "N/A"
    })

# Métricas
rets = [t["return_pct"] for t in trade_list]
wins = sum(1 for r in rets if r > 0)
win_rate = wins / len(rets) * 100 if rets else 0
max_dd = min(rets) if rets else 0
sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0

# ── 10. Output final ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[OOS-REPLAY-2026-LOCAL] RESULTADO FINAL — TRADES 2026 (HMM PROD)")
print("=" * 70)
print(f"Periodo: {df_2026_hmm.index.min().date()} -> {df_2026_hmm.index.max().date()}")
print(f"Total barras H1: {len(df_2026_hmm)}")
print(f"Barras LONG: {(decisions_series=='LONG').sum()} ({(decisions_series=='LONG').mean()*100:.1f}%)")
print(f"Barras HOLD: {(decisions_series=='HOLD').sum()} ({(decisions_series=='HOLD').mean()*100:.1f}%)")
print(f"\nEntradas (LONG): {entries}")
print(f"Salidas (->HOLD): {exits}")
print(f"Trades completos: {exits}")
print(f"Posición actual: {decisions_series.iloc[-1]}")
print(f"\nWin Rate: {win_rate:.1f}%")
print(f"Max DD trade: {max_dd:.2f}%")
print(f"Sharpe aprox: {sharpe:.3f}")

print(f"\nDetalle de trades 2026:")
for i, t in enumerate(trade_list):
    tag = "[WIN]" if t["return_pct"] > 0 else "[LOSS]"
    open_tag = " [ABIERTA]" if t["type"] == "SIMULATED_2026_OPEN" else ""
    print(f"  T{i+1}{open_tag} {tag} {t['entry_date']} -> {t['exit_date']} | "
          f"Entrada ${t['entry_price']:,.0f} Salida ${t['exit_price']:,.0f} | "
          f"Ret={t['return_pct']:+.2f}% | Dur={t['duration_h']:.0f}h | Regimen={t['regime']}")

# Detalle mensual
print(f"\nDetalle mensual:")
decisions_df = decisions_series.to_frame("decision")
decisions_df["entry"] = entries_mask
monthly = decisions_df.resample("ME").agg(
    bars=("decision","count"),
    long_bars=("decision", lambda x: (x=="LONG").sum()),
    entries=("entry","sum")
)
for month, row in monthly.iterrows():
    pct = row["long_bars"]/row["bars"]*100 if row["bars"]>0 else 0
    print(f"  {month.strftime('%Y-%m')}: {int(row['entries'])} entradas | "
          f"{int(row['long_bars'])}h LONG / {int(row['bars'])}h ({pct:.0f}%)")

# ── 11. Guardar resultado JSON para el dashboard ─────────────────────────────
import json
result = {
    "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    "period": f"{df_2026_hmm.index.min().date()} -> {df_2026_hmm.index.max().date()}",
    "total_bars": len(df_2026_hmm),
    "long_bars": int((decisions_series=="LONG").sum()),
    "hold_bars": int((decisions_series=="HOLD").sum()),
    "entries": int(entries),
    "exits": int(exits),
    "current_position": decisions_series.iloc[-1],
    "win_rate": round(win_rate, 2),
    "max_dd_pct": round(max_dd, 2),
    "sharpe": round(float(sharpe), 3),
    "trades": trade_list,
    "hmm_distribution": df_2026_hmm["HMM_Semantic"].value_counts().to_dict(),
    "vote_distribution": {str(v): int((pd.Series(vote_counts)==v).sum()) for v in range(0, len(SEEDS)+1)},
    "seeds_used": len(seed_routers),
    "consensus_threshold": CONSENSUS_CUTOFF,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
print(f"\n[OOS-REPLAY-2026-LOCAL] Mapping canonico estado->regimen:")
for s, r in CANONICAL_MAP.items():
    cnt = int((hmm_states == s).sum())
    print(f"  Estado {s} ({state_map.get(s,'?')}) -> {r}: {cnt} barras")

df_ctx["HMM_Semantic_Prod"] = df_ctx["HMM_State_Prod"].map(CANONICAL_MAP).fillna("2_CALM_RANGE")

# Volatilidad por régimen para validación
if "close" in df_ctx.columns:
    df_ctx["ret_tmp"] = df_ctx["close"].pct_change()
    print(f"\n[OOS-REPLAY-2026-LOCAL] Volatilidad por regimen (validacion):")
    for reg in df_ctx["HMM_Semantic_Prod"].unique():
        mask = df_ctx["HMM_Semantic_Prod"] == reg
        vol = df_ctx.loc[mask, "ret_tmp"].std() * np.sqrt(8760)
        print(f"  {reg}: vol={vol:.4f} | {mask.sum()} barras")
    df_ctx.drop(columns=["ret_tmp"], inplace=True)

# Filtrar solo 2026 con el nuevo HMM_Semantic_Prod
df_2026_hmm = df_ctx[df_ctx.index >= "2026-01-01"].copy()
df_2026_hmm["HMM_Semantic"] = df_2026_hmm["HMM_Semantic_Prod"]

print(f"\n[OOS-REPLAY-2026-LOCAL] Distribucion HMM_Semantic (prod) en 2026:")
print(df_2026_hmm["HMM_Semantic"].value_counts().to_string())


# ── 6. Cargar ensemble dinámicamente desde settings.yaml ──────────────────
import yaml
with open(PROJECT_ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
SEEDS = cfg["wfb"]["active_seeds"]
CONSENSUS_CUTOFF = cfg["wfb"]["ensemble_consensus_threshold"]

from luna.models.regime_router import RegimeRouter

seed_routers = {}
for seed in SEEDS:
    seed_dir = MODELS_DIR / f"seed{seed}"
    if seed_dir.exists():
        try:
            router = RegimeRouter(models_dir=seed_dir, agent_type="xgboost", direction="long")
            seed_routers[seed] = router
        except Exception as e:
            print(f"[OOS-REPLAY-2026-LOCAL] seed{seed} error: {e}")

print(f"\n[OOS-REPLAY-2026-LOCAL] Seeds cargadas: {len(seed_routers)}/{len(SEEDS)}")

# ── 7. Inferencia ensemble sobre 2026 con HMM prod correcto ─────────────────
all_seed_probs = []

for seed, router in seed_routers.items():
    try:
        result = router.route_and_predict(df_2026_hmm)
        # Usar probabilidades calibradas (SOP R10)
        probs = result["calibrated"].fillna(result["raw"]).values
        all_seed_probs.append(probs)
        bear_count = (df_2026_hmm["HMM_Semantic"] == "3_BEAR_CRASH").sum()
        zero_count = (probs == 0.0).sum()
        print(f"[OOS-REPLAY-2026-LOCAL] seed{seed}: mean_prob={np.nanmean(probs):.4f} | "
              f"nonzero={np.sum(probs>0)} | bear_bars={bear_count} | zero_prob={zero_count}")
    except Exception as e:
        print(f"[OOS-REPLAY-2026-LOCAL] seed{seed} ERROR: {e}")

# ── 8. Calcular decisiones de consenso ───────────────────────────────────────
if not all_seed_probs:
    print("[ERROR] No hay seeds con resultados válidos")
    sys.exit(1)

prob_matrix = np.array(all_seed_probs)  # (n_seeds, n_bars)
vote_matrix = (prob_matrix > 0.5).astype(int)
vote_counts  = vote_matrix.sum(axis=0)   # (n_bars,)

ensemble_decisions = np.where(vote_counts >= CONSENSUS_CUTOFF, "LONG", "HOLD")
decisions_series = pd.Series(ensemble_decisions, index=df_2026_hmm.index)

print(f"\n[OOS-REPLAY-2026-LOCAL] Distribución de decisiones 2026 (HMM prod):")
print(decisions_series.value_counts().to_string())

# ── 9. Calcular trades y métricas ────────────────────────────────────────────
prev = decisions_series.shift(1).fillna("HOLD")
entries_mask = ((prev == "HOLD") & (decisions_series == "LONG"))
exits_mask   = ((prev == "LONG") & (decisions_series == "HOLD"))

entries = entries_mask.sum()
exits   = exits_mask.sum()

# Reconstruir posiciones para métricas
trade_list = []
in_position = False
pos_start = None
pos_entry_price = None

for ts, dec in decisions_series.items():
    close_px = df_2026_hmm.loc[ts, "close"] if ts in df_2026_hmm.index else None
    if close_px is None:
        continue

    if dec == "LONG" and not in_position:
        in_position = True
        pos_start = ts
        pos_entry_price = close_px
    elif dec == "HOLD" and in_position:
        in_position = False
        ret = (close_px - pos_entry_price) / pos_entry_price
        dur_h = (ts - pos_start).total_seconds() / 3600
        trade_list.append({
            "type": "SIMULATED_2026",
            "entry_date": pos_start.strftime("%Y-%m-%d %H:%M"),
            "exit_date": ts.strftime("%Y-%m-%d %H:%M"),
            "entry_price": round(pos_entry_price, 2),
            "exit_price": round(close_px, 2),
            "return_pct": round(ret * 100, 4),
            "duration_h": round(dur_h, 1),
            "regime": df_2026_hmm.loc[pos_start, "HMM_Semantic"] if pos_start in df_2026_hmm.index else "N/A"
        })

if in_position and pos_start is not None:
    last_ts = df_2026_hmm.index[-1]
    last_px = df_2026_hmm["close"].iloc[-1]
    ret = (last_px - pos_entry_price) / pos_entry_price
    dur_h = (last_ts - pos_start).total_seconds() / 3600
    trade_list.append({
        "type": "SIMULATED_2026_OPEN",
        "entry_date": pos_start.strftime("%Y-%m-%d %H:%M"),
        "exit_date": "ABIERTA",
        "entry_price": round(pos_entry_price, 2),
        "exit_price": round(last_px, 2),
        "return_pct": round(ret * 100, 4),
        "duration_h": round(dur_h, 1),
        "regime": df_2026_hmm.loc[pos_start, "HMM_Semantic"] if pos_start in df_2026_hmm.index else "N/A"
    })

# Métricas
rets = [t["return_pct"] for t in trade_list]
wins = sum(1 for r in rets if r > 0)
win_rate = wins / len(rets) * 100 if rets else 0
max_dd = min(rets) if rets else 0
sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0

# ── 10. Output final ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[OOS-REPLAY-2026-LOCAL] RESULTADO FINAL — TRADES 2026 (HMM PROD)")
print("=" * 70)
print(f"Periodo: {df_2026_hmm.index.min().date()} -> {df_2026_hmm.index.max().date()}")
print(f"Total barras H1: {len(df_2026_hmm)}")
print(f"Barras LONG: {(decisions_series=='LONG').sum()} ({(decisions_series=='LONG').mean()*100:.1f}%)")
print(f"Barras HOLD: {(decisions_series=='HOLD').sum()} ({(decisions_series=='HOLD').mean()*100:.1f}%)")
print(f"\nEntradas (LONG): {entries}")
print(f"Salidas (->HOLD): {exits}")
print(f"Trades completos: {exits}")
print(f"Posición actual: {decisions_series.iloc[-1]}")
print(f"\nWin Rate: {win_rate:.1f}%")
print(f"Max DD trade: {max_dd:.2f}%")
print(f"Sharpe aprox: {sharpe:.3f}")

print(f"\nDetalle de trades 2026:")
for i, t in enumerate(trade_list):
    tag = "[WIN]" if t["return_pct"] > 0 else "[LOSS]"
    open_tag = " [ABIERTA]" if t["type"] == "SIMULATED_2026_OPEN" else ""
    print(f"  T{i+1}{open_tag} {tag} {t['entry_date']} -> {t['exit_date']} | "
          f"Entrada ${t['entry_price']:,.0f} Salida ${t['exit_price']:,.0f} | "
          f"Ret={t['return_pct']:+.2f}% | Dur={t['duration_h']:.0f}h | Regimen={t['regime']}")

# Detalle mensual
print(f"\nDetalle mensual:")
decisions_df = decisions_series.to_frame("decision")
decisions_df["entry"] = entries_mask
monthly = decisions_df.resample("ME").agg(
    bars=("decision","count"),
    long_bars=("decision", lambda x: (x=="LONG").sum()),
    entries=("entry","sum")
)
for month, row in monthly.iterrows():
    pct = row["long_bars"]/row["bars"]*100 if row["bars"]>0 else 0
    print(f"  {month.strftime('%Y-%m')}: {int(row['entries'])} entradas | "
          f"{int(row['long_bars'])}h LONG / {int(row['bars'])}h ({pct:.0f}%)")

# ── 11. Guardar resultado JSON para el dashboard ─────────────────────────────
import json
import datetime
result = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "seeds_used": len(SEEDS),
    "consensus_threshold": CONSENSUS_CUTOFF,
    "period": f"{df_2026_hmm.index.min().date()} -> {df_2026_hmm.index.max().date()}",
    "total_bars": len(df_2026_hmm),
    "long_bars": int((decisions_series=="LONG").sum()),
    "hold_bars": int((decisions_series=="HOLD").sum()),
    "entries": int(entries),
    "exits": int(exits),
    "current_position": decisions_series.iloc[-1],
    "win_rate": round(win_rate, 2),
    "max_dd_pct": round(max_dd, 2),
    "sharpe": round(float(sharpe), 3),
    "trades": trade_list,
    "hmm_distribution": df_2026_hmm["HMM_Semantic"].value_counts().to_dict(),
    "vote_distribution": {str(v): int((pd.Series(vote_counts)==v).sum()) for v in range(0, len(SEEDS)+1)},
    "seeds_used": len(seed_routers),
    "consensus_threshold": CONSENSUS_CUTOFF,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)

print(f"\n[OOS-REPLAY-2026-LOCAL] OK JSON guardado: {OUT_JSON}")

# [DASHBOARD INJECTION]
metadata_path = PROJECT_ROOT / "data" / "models" / "prod" / "ensemble_metadata.json"
if metadata_path.exists():
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        meta["oos_metrics"] = result
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        print(f"[DASHBOARD-INJECTION] OK: oos_metrics completos inyectados en {metadata_path.name}")
    except Exception as e:
        print(f"[DASHBOARD-INJECTION] ERROR al inyectar oos_metrics: {e}")
else:
    print(f"[DASHBOARD-INJECTION] WARN: {metadata_path.name} no encontrado. No se inyectaron métricas.")

print(f"[OOS-REPLAY-2026-LOCAL] Tiempo total: {time.monotonic()-t0:.1f}s")
