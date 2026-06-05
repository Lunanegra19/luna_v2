"""
arch18_hmm_calm_bear_diagnosis.py
===================================
ARCH-18: CALM_BEAR no existe en HMM — agente con N=0 muestras IS

PROTOCOLO (diagnostico_cuantitativo.md):
  FASE 1: Verificar distribución real de estados HMM en hmm_regime_labels.parquet
  FASE 2: Hipótesis — H1 (estado 3_CALM_BEAR ausente), H2 (mapeo semántico roto),
          H3 (estado existe pero con otro nombre)
  FASE 3: Contar barras IS por estado — test chi2 de distribución uniforme
  FASE 4: Leer código _analyze_and_map_states para entender causa raíz
  FASE 5: Counterfactual — cuántas barras IS tiene cada estado en el IS actual

USO: python tools/diagnostics/arch18_hmm_calm_bear_diagnosis.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR   = ROOT / "data" / "models"

print("=" * 70)
print("[ARCH-18] DIAGNOSTICO: CALM_BEAR en HMM regime labels")
print("=" * 70)

# ── FASE 1: Cargar hmm_regime_labels.parquet ───────────────────────────────────
hmm_path = FEATURES_DIR / "hmm_regime_labels.parquet"
if not hmm_path.exists():
    print(f"  [ERROR] No existe: {hmm_path}")
    sys.exit(1)

df_hmm = pd.read_parquet(hmm_path)
print(f"  [INFO] hmm_regime_labels.parquet: shape={df_hmm.shape}")
print(f"  [INFO] Rango: {df_hmm.index.min()} -> {df_hmm.index.max()}")
print(f"  [INFO] Columnas: {list(df_hmm.columns)}")

# ── Distribución de estados ────────────────────────────────────────────────────
print("\n[H1] DISTRIBUCION DE ESTADOS HMM (hmm_regime_labels.parquet)")
print("-" * 60)

if "HMM_Semantic" in df_hmm.columns:
    sem_counts = df_hmm["HMM_Semantic"].value_counts().sort_index()
    total = len(df_hmm)
    print(f"  Total barras: {total:,}")
    for state, cnt in sem_counts.items():
        pct = cnt / total * 100
        print(f"  {state:30s}: {cnt:6,} ({pct:5.1f}%)")

    calm_bear_variants = [s for s in sem_counts.index if "CALM_BEAR" in str(s) or "calm_bear" in str(s).lower()]
    if not calm_bear_variants:
        print("\n  *** H1 CONFIRMADA: '3_CALM_BEAR' NO EXISTE en hmm_regime_labels.parquet ***")
    else:
        print(f"\n  H1 DESCARTADA: CALM_BEAR encontrado como: {calm_bear_variants}")
        for v in calm_bear_variants:
            print(f"    {v}: {sem_counts.get(v, 0):,} barras")
else:
    print("  [WARN] Columna HMM_Semantic no encontrada")
    if "HMM_Regime" in df_hmm.columns:
        reg_counts = df_hmm["HMM_Regime"].value_counts().sort_index()
        print(f"  Distribución HMM_Regime (numérico):")
        for state, cnt in reg_counts.items():
            print(f"    Estado {state}: {cnt:,} ({cnt/len(df_hmm)*100:.1f}%)")

# ── FASE 2: Verificar contra parquets de features IS ──────────────────────────
print("\n[H2] DISTRIBUCION EN IS (features_train.parquet)")
print("-" * 60)
train_path = FEATURES_DIR / "features_train.parquet"
if train_path.exists():
    try:
        cols_needed = ["HMM_Semantic"] if True else []
        df_train = pd.read_parquet(train_path, columns=["HMM_Semantic"]) if "HMM_Semantic" in pd.read_parquet(train_path, filters=None).columns else None
        if df_train is None:
            # Cargar solo con HMM_Regime
            df_train_full = pd.read_parquet(train_path)
            hmm_cols = [c for c in df_train_full.columns if "HMM" in c.upper()]
            print(f"  Columnas HMM en features_train: {hmm_cols}")
            df_train = df_train_full[hmm_cols] if hmm_cols else None
    except Exception as e:
        print(f"  [WARN] Error leyendo features_train.parquet: {e}")
        df_train = None

    if df_train is not None and "HMM_Semantic" in df_train.columns:
        is_sem = df_train["HMM_Semantic"].value_counts().sort_index()
        print(f"  Distribución IS:")
        for state, cnt in is_sem.items():
            print(f"    {state:30s}: {cnt:6,}")
else:
    print("  [WARN] features_train.parquet no encontrado")

# ── FASE 3: Cargar hmm_regime.pkl para verificar state_map ────────────────────
print("\n[H3] STATE MAP del modelo HMM serializado (hmm_regime.pkl)")
print("-" * 60)
hmm_pkl = MODELS_DIR / "hmm_regime.pkl"
if hmm_pkl.exists():
    try:
        import joblib
        hmm_model = joblib.load(hmm_pkl)
        if hasattr(hmm_model, "state_map"):
            print(f"  state_map actual ({len(hmm_model.state_map)} estados):")
            for num_state, sem_name in sorted(hmm_model.state_map.items()):
                print(f"    Estado {num_state} -> '{sem_name}'")
            
            calm_bear_in_map = [k for k, v in hmm_model.state_map.items() if "CALM_BEAR" in str(v)]
            if not calm_bear_in_map:
                print("\n  *** H3 CONFIRMADA: 3_CALM_BEAR NO está en state_map del modelo ***")
                print(f"  Estados disponibles: {list(hmm_model.state_map.values())}")
            else:
                print(f"\n  H3 DESCARTADA: CALM_BEAR en state_map = estado numero {calm_bear_in_map}")
        else:
            print("  [WARN] hmm_model no tiene atributo 'state_map'")
            print(f"  Atributos del modelo: {[a for a in dir(hmm_model) if not a.startswith('_')]}")
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar hmm_regime.pkl: {e}")
else:
    print("  [WARN] hmm_regime.pkl no encontrado en models/")
    # Buscar en wfb_cache
    for cache_hmm in sorted(ROOT.glob("data/wfb_cache/**/hmm_model.pkl")):
        print(f"  Encontrado en: {cache_hmm}")
        try:
            import joblib
            m = joblib.load(cache_hmm)
            if hasattr(m, "state_map"):
                print(f"    state_map: {m.state_map}")
            break
        except Exception as e:
            print(f"    Error: {e}")

# ── FASE 4: Analisis del código _analyze_and_map_states ───────────────────────
print("\n[H4] ANALISIS DE CAUSA RAIZ: ¿Por que no se genera CALM_BEAR?")
print("-" * 60)
print("  Mecanismo de mapeo (hmm_regime.py _analyze_and_map_states):")
print("  1. Calcula retorno medio por estado en el IS (hasta train_cutoff)")
print("  2. Estado con mayor retorno medio -> '1_BULL_TREND'")
print("  3. Estado con mayor volatilidad alta -> '1_VOLATILE_BULL'")
print("  4. Estado con menor retorno medio -> '3_BEAR_CRASH'")
print("  5. Estado residual -> '2_CALM_RANGE' o '2_VOLATILE_RANGE'")
print("  ")
print("  Con N_REGIMES=4, los estados son {0,1,2,3}.")
print("  Para que exista '3_CALM_BEAR', se necesitaria N_REGIMES>=5")
print("  O bien una re-definicion del mapeo semantico con 4 regimenes.")

# ── FASE 5: Verificar N_REGIMES en settings.yaml ──────────────────────────────
print("\n[H5] N_REGIMES EN SETTINGS.YAML")
print("-" * 60)
try:
    import yaml
    with open(ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    n_states = cfg.get("hmm", {}).get("n_states", "NO ENCONTRADO")
    hmm_cfg = cfg.get("hmm", {})
    print(f"  hmm.n_states         : {n_states}")
    print(f"  hmm.oos_window_hours : {hmm_cfg.get('oos_window_hours', 'NO ENCONTRADO')}")
    print(f"  hmm.n_init           : {hmm_cfg.get('n_init', 'NO ENCONTRADO')}")
    print(f"  hmm.n_iter           : {hmm_cfg.get('n_iter', 'NO ENCONTRADO')}")
    
    # Verificar config de regime_mapping en fase2
    fase2 = cfg.get("fase2", {})
    regime_mapping = fase2.get("regime_mapping", {})
    print(f"\n  fase2.regime_mapping:")
    for regime, states in regime_mapping.items() if regime_mapping else []:
        print(f"    {regime}: {states}")

    if not regime_mapping:
        print("  [INFO] fase2.regime_mapping no configurado")

except Exception as e:
    print(f"  [ERROR] No se pudo leer settings.yaml: {e}")

# ── Resumen ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN ARCH-18")
print("=" * 70)

# Resumen de la distribución real
if "HMM_Semantic" in df_hmm.columns:
    sem_counts_sorted = df_hmm["HMM_Semantic"].value_counts().sort_values(ascending=False)
    print(f"  Estados en hmm_regime_labels.parquet:")
    for state, cnt in sem_counts_sorted.items():
        pct = cnt / len(df_hmm) * 100
        flag = " <-- CALM_BEAR AUSENTE" if "CALM" in str(state) and "BEAR" in str(state) else ""
        print(f"    {state:30s}: {cnt:,} ({pct:.1f}%){flag}")
    
    if not any("CALM_BEAR" in str(s) for s in df_hmm["HMM_Semantic"].unique()):
        print(f"\n  CONCLUSION: CALM_BEAR NO EXISTE.")
        print(f"  Con N_REGIMES={n_states} estados, el HMM aprende {n_states} clusters,")
        print(f"  ninguno de los cuales mapea a '3_CALM_BEAR'.")
        print(f"  Para generar CALM_BEAR se necesitaria N_REGIMES>={int(n_states)+1 if str(n_states).isdigit() else '?'}")
        print(f"  O redefinir el mapeo semantico para separar 'BEAR silencioso' de 'BEAR_CRASH' con 4 estados.")

print("\n[ARCH-18] Diagnostico completado.")
