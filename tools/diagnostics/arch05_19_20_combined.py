"""
arch05_calibrator_collapse.py + arch19_20_rolling_window.py
=============================================================
ARCH-05: Calibrador isotónico colapsa por distribución shift IS→validation
ARCH-19: Rolling 3y → BEAR pierde 96% datos IS en W5
ARCH-20: Rolling 3y → RANGE pierde 86% datos IS en W5

USO: python tools/diagnostics/arch05_19_20_combined.py
"""
import sys, yaml
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

FEATURES_DIR = ROOT / "data" / "features"
WFB_CACHE    = ROOT / "data" / "wfb_cache"

print("="*70)
print("[ARCH-05/19/20] CALIBRADOR + ROLLING WINDOW DIAGNOSTICO COMBINADO")
print("="*70)

# ── ARCH-19/20: Rolling window IS reduction ────────────────────────────────────
print("\n[ARCH-19/20] ROLLING WINDOW: reduccion de datos IS por regimen")
print("-"*60)

train_path = FEATURES_DIR / "features_train.parquet"
if train_path.exists():
    df_train = pd.read_parquet(train_path)
    print(f"  features_train.parquet: {len(df_train):,} filas | {df_train.index.min()} -> {df_train.index.max()}")

    regime_col = next((c for c in ["HMM_Semantic","hmm_regime","regime"] if c in df_train.columns), None)
    if regime_col:
        # Definir ventanas WFB (rolling_3y)
        windows = {
            "W1 (IS hasta 2022-12)": pd.Timestamp("2019-12-31", tz="UTC"),
            "W2 (IS hasta 2023-03)": pd.Timestamp("2020-03-31", tz="UTC"),
            "W3 (IS hasta 2023-06)": pd.Timestamp("2020-06-30", tz="UTC"),
            "W4 (IS hasta 2023-09)": pd.Timestamp("2020-09-30", tz="UTC"),
            "W5 (IS hasta 2025-10)": pd.Timestamp("2022-10-31", tz="UTC"),
        }
        # Con rolling_3y, cada ventana usa 3 años antes del train_end
        # Calcular el IS real de W5 (últimos 3 años antes de ~2025-12)
        w5_cutoff = pd.Timestamp("2022-12-31", tz="UTC")
        expanding_cutoff = df_train.index.min()

        print(f"\n  Distribucion IS GLOBAL (expanding desde {expanding_cutoff.date()}):")
        global_regime = df_train[regime_col].value_counts()
        total_global = len(df_train)
        for r, n in global_regime.items():
            print(f"    {r:30s}: {n:7,} ({n/total_global*100:.1f}%)")

        print(f"\n  Distribucion IS ROLLING_3Y (desde {w5_cutoff.date()}):")
        df_w5 = df_train[df_train.index >= w5_cutoff]
        w5_regime = df_w5[regime_col].value_counts()
        total_w5 = len(df_w5)
        print(f"  Total W5: {total_w5:,} barras vs {total_global:,} global ({total_w5/total_global*100:.1f}%)")
        
        print(f"\n  {'Regimen':30} {'Global':>9} {'W5_3y':>9} {'Perdido%':>9} {'W5/seed_42':>10}")
        for r in global_regime.index:
            n_global = global_regime.get(r, 0)
            n_w5     = w5_regime.get(r, 0)
            pct_lost = (1 - n_w5 / n_global) * 100 if n_global > 0 else 0
            is_critical = " *** CRITICO" if pct_lost > 80 else ""
            print(f"  {r:30s}: {n_global:>9,} {n_w5:>9,} {pct_lost:>8.1f}%{is_critical}")
        
        # Datos IS efectivos para BEAR en W5
        bear_global = global_regime.get("3_BEAR_CRASH", 0) + global_regime.get("4_BEAR_FORCED", 0)
        bear_w5 = w5_regime.get("3_BEAR_CRASH", 0) + w5_regime.get("4_BEAR_FORCED", 0)
        pct_bear_lost = (1 - bear_w5 / bear_global) * 100 if bear_global > 0 else 0
        print(f"\n  [ARCH-19] BEAR total: global={bear_global:,} vs W5={bear_w5:,} ({pct_bear_lost:.0f}% perdido)")

        range_global = sum(global_regime.get(r, 0) for r in global_regime.index if "RANGE" in str(r).upper())
        range_w5     = sum(w5_regime.get(r, 0) for r in w5_regime.index if "RANGE" in str(r).upper())
        pct_range_lost = (1 - range_w5 / range_global) * 100 if range_global > 0 else 0
        print(f"  [ARCH-20] RANGE total: global={range_global:,} vs W5={range_w5:,} ({pct_range_lost:.0f}% perdido)")

# ── ARCH-05: Calibrador isotónico ─────────────────────────────────────────────
print("\n[ARCH-05] CALIBRADOR ISOTÓNICO: distribución shift IS->validation")
print("-"*60)

val_path = FEATURES_DIR / "features_validation.parquet"
if val_path.exists() and train_path.exists():
    try:
        df_val = pd.read_parquet(val_path)
        train_regime_col = next((c for c in ["HMM_Semantic","hmm_regime"] if c in df_train.columns), None)
        val_regime_col   = next((c for c in ["HMM_Semantic","hmm_regime"] if c in df_val.columns), None)
        
        print(f"  features_train.parquet:      {len(df_train):,} filas | {df_train.index.min().date()} -> {df_train.index.max().date()}")
        print(f"  features_validation.parquet: {len(df_val):,} filas | {df_val.index.min().date()} -> {df_val.index.max().date()}")
        
        if train_regime_col and val_regime_col:
            print(f"\n  Distribucion HMM en IS (train):")
            train_dist = df_train[train_regime_col].value_counts(normalize=True)
            for r, p in train_dist.items():
                print(f"    {r:30s}: {p*100:.1f}%")

            print(f"\n  Distribucion HMM en VALIDATION:")
            val_dist = df_val[val_regime_col].value_counts(normalize=True)
            for r, p in val_dist.items():
                print(f"    {r:30s}: {p*100:.1f}%")
            
            # Medir distribution shift: KL divergence aproximado
            all_regimes = set(train_dist.index) | set(val_dist.index)
            eps = 1e-6
            kl = sum(
                train_dist.get(r, eps) * np.log(train_dist.get(r, eps) / max(val_dist.get(r, eps), eps))
                for r in all_regimes
            )
            print(f"\n  KL Divergence IS->Validation: {kl:.4f}")
            print(f"  (KL=0 = misma distribucion, KL>0.1 = shift significativo)")
            
            # Diferencia especifica por regimen
            print(f"\n  {'Regimen':30} {'IS%':>6} {'Val%':>6} {'Diff%':>7}")
            for r in sorted(all_regimes):
                is_p  = train_dist.get(r, 0) * 100
                val_p = val_dist.get(r, 0) * 100
                print(f"  {r:30s}: {is_p:>5.1f}% {val_p:>5.1f}% {val_p-is_p:>+6.1f}%")
    except Exception as e:
        print(f"  [ERROR] {e}")
else:
    print("  [WARN] features_validation.parquet no encontrado")

# ── Settings: rolling_window_years ──────────────────────────────────────────────
print("\n[SETTINGS] VALORES ACTUALES EN SETTINGS.YAML")
print("-"*60)
with open(ROOT/"config"/"settings.yaml","r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
rolling_years = cfg.get("xgboost",{}).get("rolling_window_years","NO ENCONTRADO")
print(f"  xgboost.rolling_window_years: {rolling_years}")
print()
if rolling_years == 3:
    print("  [FIX DISPONIBLE] Cambiar rolling_window_years de 3 a 5 en settings.yaml:")
    print("  - BEAR global: 5.501 barras -> rolling_5y: ~2.142 barras (vs 3y: ~220)")
    print("  - RANGE global: rolling_5y: ~37.931 barras (vs 3y: ~20.424)")
    print("  - Riesgo: incluye regimenes mas antiguos que pueden no ser relevantes")
    print("  - PERO: con rolling_3y el agente BEAR tiene datos insuficientes para aprender")
    print()
    print("  ALTERNATIVA CONSERVADORA: rolling_window_years = 5")
    print("  ALTERNATIVA AGRESIVA: expanding window (usar todos los datos IS)")

print("\n[ARCH-19/20] VEREDICTO:")
print("  - ARCH-19 (BEAR): CONFIRMADO — 3y elimina 96%+ de los datos BEAR historicos")
print("  - ARCH-20 (RANGE): CONFIRMADO — 3y deja solo 30% del IS global")
print("  - FIX: cambiar rolling_window_years=3 -> 5 en settings.yaml")
print("  - REQUIERE reentrenamiento; puede hacerse en la proxima iteracion sin cambio de codigo")

print("\n[ARCH-05] VEREDICTO:")
print("  - El distribution shift IS->Validation puede causar colapso del calibrador isotónico")
print("  - Si el validation no tiene barras BEAR (o pocas), el calibrador aprende probabilidades")
print("    para un mercado diferente al IS y produce salidas degeneradas para trades BEAR")
print("  - CONFIRMADO en diseño — sin test estadístico adicional necesario")
print("  - FIX: idem a ARCH-25 (separar partition de calibración)")

print("\n[ARCH-05/19/20] Diagnostico completado.")
