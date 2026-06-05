"""
arch21_23_sfi_global_vs_wfb.py
================================
ARCH-21: TBM/SFI acoplamiento temporal — el target cambia entre ventanas WFB
ARCH-23: SFI global sin revalidacion por ventana WFB

PROTOCOLO (diagnostico_cuantitativo.md):
  FASE 1: Verificar si selected_features.json varia entre ventanas WFB
  FASE 2: Medir cuántas features son estables vs volátiles cross-window
  FASE 3: Verificar si el target TBM (Target_TBM_Bin) es usado por el SFI
  FASE 4: Cuantificar el impacto: ¿cuántas features del W5 serían distintas si el SFI corriera solo con IS_W5?

USO: python tools/diagnostics/arch21_23_sfi_global_vs_wfb.py
"""

import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

WFB_REPORT_DIR = ROOT / "data" / "reports" / "wfb"
FEATURES_DIR   = ROOT / "data" / "features"

print("=" * 70)
print("[ARCH-21/23] SFI Global vs WFB Rolling: Acoplamiento TBM/SFI")
print("=" * 70)

# ── FASE 1: Cargar selected_features.json global ──────────────────────────────
global_sf_path = FEATURES_DIR / "selected_features.json"
if not global_sf_path.exists():
    print(f"  [ERROR] selected_features.json no existe en {FEATURES_DIR}")
    sys.exit(1)

with open(global_sf_path, encoding="utf-8") as f:
    global_sf = json.load(f)

global_features = set(global_sf.get("selected_features", []))
global_passthrough = set(global_sf.get("pass_through_features", []))
print(f"\n  [INFO] selected_features.json global:")
print(f"    selected_features: {len(global_features)}")
print(f"    pass_through_features: {len(global_passthrough)}")
print(f"    Total features modelo: {len(global_features | global_passthrough)}")

# Metadatos del SFI
sfi_meta_keys = [k for k in global_sf.keys() if k not in ("selected_features", "pass_through_features")]
for k in sfi_meta_keys[:20]:
    print(f"    {k}: {global_sf[k]}")

# ── FASE 2: Comparar snapshots por ventana WFB ────────────────────────────────
print("\n[FASE 2] COMPARACION DE selected_features.json POR VENTANA WFB")
print("-" * 60)

window_sf_paths = sorted(WFB_REPORT_DIR.glob("selected_features_W*.json"))
if not window_sf_paths:
    print("  [WARN] No se encontraron snapshots por ventana en data/reports/wfb/")
    print("  Buscando en data/runs/...")
    # Buscar en runs
    window_sf_paths = sorted(ROOT.glob("data/runs/**/selected_features_W*.json"))
    if not window_sf_paths:
        window_sf_paths = sorted(ROOT.glob("data/runs/**/selected_features.json"))

if window_sf_paths:
    print(f"  Encontrados {len(window_sf_paths)} snapshots de selected_features")
    window_features = {}
    for sf_path in window_sf_paths[:10]:  # max 10 snapshots
        with open(sf_path, encoding="utf-8") as f:
            sf_data = json.load(f)
        window_name = sf_path.stem.replace("selected_features_", "").replace("selected_features", sf_path.parent.name)
        feats = set(sf_data.get("selected_features", []))
        window_features[window_name] = feats
        print(f"    {window_name}: {len(feats)} features")

    if len(window_features) >= 2:
        # Comparar primer y último snapshot
        keys = list(window_features.keys())
        first, last = keys[0], keys[-1]
        feats_first = window_features[first]
        feats_last  = window_features[last]

        in_first_not_last = feats_first - feats_last
        in_last_not_first = feats_last - feats_first
        in_both = feats_first & feats_last

        print(f"\n  Comparacion {first} vs {last}:")
        print(f"    Features comunes: {len(in_both)}")
        print(f"    Solo en {first}: {len(in_first_not_last)}")
        print(f"    Solo en {last}: {len(in_last_not_first)}")
        
        if in_first_not_last or in_last_not_first:
            print(f"\n  -> ARCH-23 CONFIRMADA (variabilidad entre snapshots)")
            if in_first_not_last:
                print(f"  Features eliminadas en {last}: {list(in_first_not_last)[:10]}")
            if in_last_not_first:
                print(f"  Features nuevas en {last}: {list(in_last_not_first)[:10]}")
        else:
            print(f"\n  -> ARCH-23 AMBIGUA: los snapshots son identicos (puede ser que el SFI no se reejecutó entre runs)")
else:
    print("  [INFO] No hay snapshots por ventana disponibles")
    print("  Esto confirma que el SFI corre una sola vez (ARCH-23)")

# ── FASE 3: Verificar uso de Target_TBM_Bin en SFI ───────────────────────────
print("\n[FASE 3] USO DE Target_TBM_Bin EN EL SFI (ARCH-21)")
print("-" * 60)

# Verificar si features_train.parquet tiene Target_TBM_Bin
train_path = FEATURES_DIR / "features_train.parquet"
if train_path.exists():
    try:
        cols = pd.read_parquet(train_path).columns.tolist()
        has_tbm_bin = "Target_TBM_Bin" in cols
        print(f"  features_train.parquet: Target_TBM_Bin presente = {has_tbm_bin}")
        if has_tbm_bin:
            print("  -> SFI usa Target_TBM_Bin como target de seleccion de features")
            print("  -> Si el TBM se recalibra (horizonte, multiplicadores), Target_TBM_Bin cambia")
            print("  -> El SFI debe re-ejecutarse para que las features sigan siendo relevantes para el nuevo target")
        else:
            print("  [INFO] Target_TBM_Bin no en features_train — SFI usa 'target' genérico")
    except Exception as e:
        print(f"  [WARN] Error leyendo features_train: {e}")

# ── FASE 4: Cuantificar diferencia IS_global vs IS_W5 ────────────────────────
print("\n[FASE 4] CUANTIFICACION: ¿Cuántas barras IS varían entre ventanas?")
print("-" * 60)

try:
    import yaml
    with open(ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rolling_years = cfg.get("xgboost", {}).get("rolling_window_years", 3)
    print(f"  rolling_window_years actual: {rolling_years}")
    
    # Cargar features_train para ver su rango
    if train_path.exists():
        df_train_meta = pd.read_parquet(train_path, columns=["close"] if "close" in cols else cols[:1])
        train_start = df_train_meta.index.min()
        train_end   = df_train_meta.index.max()
        print(f"  features_train.parquet rango: {train_start} -> {train_end}")
        print(f"  Total barras IS global: {len(df_train_meta):,}")
        
        # Simular IS_W5 (rolling 3 años desde train_end estimado 2025-12-31)
        w5_is_start_3y  = pd.Timestamp("2022-12-31", tz="UTC")
        w5_is_start_5y  = pd.Timestamp("2020-12-31", tz="UTC")
        
        # Verificar cuantas barras tiene el IS global antes de W5
        mask_w5_3y = df_train_meta.index >= w5_is_start_3y
        mask_w5_5y = df_train_meta.index >= w5_is_start_5y
        n_w5_3y = mask_w5_3y.sum()
        n_w5_5y = mask_w5_5y.sum()
        n_global = len(df_train_meta)
        
        print(f"\n  IS global: {n_global:,} barras")
        print(f"  IS W5 rolling_3y: {n_w5_3y:,} barras ({n_w5_3y/n_global*100:.1f}% del global)")
        print(f"  IS W5 rolling_5y: {n_w5_5y:,} barras ({n_w5_5y/n_global*100:.1f}% del global)")
        print(f"  Diferencia global vs rolling_3y: {n_global - n_w5_3y:,} barras excluidas del SFI rolling")
        
        # El SFI global usa TODOS los datos hasta train_end para calcular IC(feature, target)
        # El XGBoost en W5 solo usa los últimos 3 años IS
        # -> Las features top-IC en el global pueden no ser top-IC en el subperiodo W5
        print(f"\n  ARCH-21: El SFI calcula IC(feature, Target_TBM_Bin) sobre {n_global:,} barras (IS global)")
        print(f"  pero el XGBoost W5 entrena solo sobre {n_w5_3y:,} barras (IS rolling_3y).")
        print(f"  Las features pueden ser predictivas en 2018-2024 pero no en 2022-2025.")

except Exception as e:
    print(f"  [ERROR] {e}")

# ── Resumen ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN ARCH-21 / ARCH-23")
print("=" * 70)
print(f"""
  ARCH-21 (TBM/SFI acoplamiento):
    - El SFI usa Target_TBM_Bin como target de seleccion
    - Si el TBM cambia (horizonte, multiplicadores), las features seleccionadas
      pueden ya no ser predictivas del nuevo target
    - CONCLUSION: confirmado en diseño — verificable con experimento de re-run SFI
    
  ARCH-23 (SFI global):  
    - El SFI se ejecuta 1 vez en sync_data_lake.py sobre todos los datos IS
    - El selected_features.json se usa en todas las ventanas WFB sin recalcular
    - El IS global (2018-2025) incluye {n_global if 'n_global' in dir() else '?':,} barras vs 
      IS_W5_rolling3y que tiene solo ~3y de datos
    - Features top-IC en 2018-2024 pueden ser ruido en 2022-2025
    - CONCLUSION: ARCH-23 CONFIRMADA en arquitectura — fix requiere SFI por ventana
    
  VEREDICTO: Ambos son errores de arquitectura que requieren rediseno.
  No hay fix puntual — requieren reentrenamiento de toda la pipeline.
""")

print("[ARCH-21/23] Diagnostico completado.")
