"""
arch18_calm_bear_deep_investigation.py
========================================
Investigacion profunda de ARCH-18: 3_CALM_BEAR aparece en OOS trades pero
tiene 0 barras en hmm_regime_labels.parquet.

Preguntas a responder:
1. ¿Como llegan barras 3_CALM_BEAR al router si el parquet dice 0?
2. ¿Donde se genera el label 3_CALM_BEAR en OOS runtime?
3. ¿El modelo XGBoost del agente calm_bear fue entrenado con algun dato?
4. ¿Hay xgboost models de calm_bear en el cache WFB?

USO: python tools/diagnostics/arch18_calm_bear_deep_investigation.py
"""
import sys, re, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("[ARCH-18] CALM_BEAR DEEP INVESTIGATION")
print("="*70)

# ── 1. Verificar el hmm_regime_labels.parquet ────────────────────────────────
print("\n[1] HMM_REGIME_LABELS.PARQUET — distribucion de etiquetas")
print("-"*60)
hmm_path = ROOT / "data" / "features" / "hmm_regime_labels.parquet"
if hmm_path.exists():
    df_hmm = pd.read_parquet(hmm_path)
    print(f"  Filas: {len(df_hmm):,}")
    if "HMM_Semantic" in df_hmm.columns:
        counts = df_hmm["HMM_Semantic"].value_counts()
        for label, n in counts.items():
            print(f"    {label:<30}: {n:>6,} barras")
        calm_bear_n = counts.get("3_CALM_BEAR", 0)
        print(f"\n  3_CALM_BEAR en parquet IS: {calm_bear_n} barras")
    if "HMM_Regime" in df_hmm.columns:
        regimes = df_hmm["HMM_Regime"].value_counts()
        print(f"\n  Estados HMM numericos: {dict(regimes)}")
else:
    print("  ERROR: hmm_regime_labels.parquet no encontrado")

# ── 2. Verificar los trade logs OOS ─────────────────────────────────────────
print("\n[2] TRADE LOGS OOS — 3_CALM_BEAR en trades reales")
print("-"*60)
oos_logs = list((ROOT / "data").rglob("oos_trades*.parquet")) + \
           list((ROOT / "data").rglob("trade_log*.parquet")) + \
           list((ROOT / "data").rglob("*trades*.csv"))
calm_bear_trades = 0
for log in oos_logs[:10]:
    try:
        df = pd.read_parquet(log) if log.suffix == ".parquet" else pd.read_csv(log)
        if "regime" in df.columns or "HMM_Semantic" in df.columns:
            col = "regime" if "regime" in df.columns else "HMM_Semantic"
            cb = (df[col].astype(str).str.contains("CALM_BEAR|calm_bear", case=False)).sum()
            if cb > 0:
                print(f"  {log.name}: {cb} trades CALM_BEAR")
                calm_bear_trades += cb
    except:
        pass

if calm_bear_trades == 0:
    print("  No se encontraron trade logs con CALM_BEAR.")

# ── 3. Verificar WFB cache — modelos del agente calm_bear ───────────────────
print("\n[3] WFB CACHE — modelos XGBoost del agente calm_bear")
print("-"*60)
cache_dir = ROOT / "data" / "wfb_cache"
calm_bear_models = []
if cache_dir.exists():
    for model_path in cache_dir.rglob("*.model"):
        if "calm" in model_path.name.lower() or "calm" in str(model_path.parent).lower():
            size = model_path.stat().st_size
            calm_bear_models.append((model_path, size))
    
    for mp, sz in calm_bear_models[:10]:
        rel = mp.relative_to(ROOT)
        print(f"  {rel} ({sz:,} bytes)")
    
    if not calm_bear_models:
        print("  No se encontraron modelos calm_bear en wfb_cache/")
        # Mostrar que seeds/windows hay
        seeds = [d.name for d in cache_dir.iterdir() if d.is_dir()]
        print(f"  Seeds en cache: {seeds[:5]}")
        if seeds:
            w_dirs = [d.name for d in (cache_dir/seeds[0]).iterdir() if d.is_dir()]
            print(f"  Windows en {seeds[0]}: {w_dirs}")
            if w_dirs:
                model_files = list((cache_dir/seeds[0]/w_dirs[-1]/"models").glob("*.model"))
                print(f"  Modelos en {seeds[0]}/{w_dirs[-1]}/models/:")
                for mf in model_files[:10]:
                    print(f"    {mf.name} ({mf.stat().st_size:,} bytes)")

# ── 4. Investigar donde se genera la etiqueta 3_CALM_BEAR en runtime ────────
print("\n[4] CODIGO — donde se genera 3_CALM_BEAR en runtime OOS")
print("-"*60)
# Buscar en hmm_regime.py el mapeo semantico
hmm_code = ROOT / "luna" / "models" / "hmm_regime.py"
if hmm_code.exists():
    lines = hmm_code.read_text(encoding="utf-8", errors="replace").splitlines()
    calm_hits = [(i+1, l) for i, l in enumerate(lines) if "calm" in l.lower() or "CALM" in l]
    print(f"  Hits 'calm' en hmm_regime.py: {len(calm_hits)}")
    for lno, line in calm_hits[:15]:
        print(f"    L{lno:4}: {line.strip()[:120]}")

# ── 5. Verificar settings.yaml — regime_mapping ──────────────────────────────
print("\n[5] SETTINGS — fase2.regime_mapping (fuente de los nombres de regimen)")
print("-"*60)
import yaml
with open(ROOT/"config"/"settings.yaml","r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

fase2 = cfg.get("fase2", cfg.get("routing", {}))
regime_map = fase2.get("regime_mapping", {})
print(f"  Seccion fase2 encontrada: {'SI' if fase2 else 'NO'}")
print(f"  Regime mapping keys: {list(regime_map.keys())}")
for key, val in regime_map.items():
    print(f"    {key}: {val}")

# ── 6. Verificar features_train.parquet para CALM_BEAR ───────────────────────
print("\n[6] FEATURES_TRAIN.PARQUET — existencia de CALM_BEAR en IS")
print("-"*60)
train_path = ROOT / "data" / "features" / "features_train.parquet"
if train_path.exists():
    df_tr = pd.read_parquet(train_path, columns=["HMM_Semantic"] if "HMM_Semantic" in pd.read_parquet(train_path, columns=[]).columns else None)
    if df_tr is not None and "HMM_Semantic" in df_tr.columns:
        counts_tr = df_tr["HMM_Semantic"].value_counts()
        calm_tr = counts_tr.get("3_CALM_BEAR", 0)
        print(f"  3_CALM_BEAR en features_train: {calm_tr} filas")
        print(f"  Todos los regimenes en features_train:")
        for lab, n in counts_tr.items():
            print(f"    {lab:<30}: {n:>6,}")

print("\n[ARCH-18] Deep investigation completada.")
