#!/usr/bin/env python
"""
inspect_forced_features.py
==========================
Verifica la calidad de las features nuevas/forzadas antes de que el SFI las evalúe.

Features auditadas:
  - DXY-HMM-01: DXY_HMM_cond, DXY_HMM_bull_neg, DXY_HMM_interact
  - EXCHANGE-FLOW-01: Exchange_NetFlow, Exchange_NetFlow_7dEMA, Exchange_NetFlow_z30d,
                      Exchange_NetFlow_Accum30d, Exchange_Outflow_Signal
  - LTH-SUPPLY-01: NonEx_Supply, Exchange_Supply_Pct, LTH_Supply_Change_30d,
                   LTH_Accum_Signal, NonEx_Supply_z90d

Para cada feature se reporta:
  ✅ Presente con datos buenos
  ⚠️  Presente pero con problemas (NaN, constante, rango sospechoso)
  ❌  Ausente del parquet — el SFI nunca la verá
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "features"

print("=" * 72)
print("[FORCED-FEATURES-AUDIT] Calidad de features nuevas/forzadas")
print("=" * 72)

# ── Definicion de features por grupo ─────────────────────────────────────────
GROUPS = {
    "DXY-HMM-01 (features condicionales DXY×HMM)": [
        "DXY_HMM_cond",
        "DXY_HMM_bull_neg",
        "DXY_HMM_interact",
    ],
    "EXCHANGE-FLOW-01 (Exchange Net Flows onchain)": [
        "Exchange_NetFlow",
        "Exchange_NetFlow_7dEMA",
        "Exchange_NetFlow_z30d",
        "Exchange_NetFlow_Accum30d",
        "Exchange_Outflow_Signal",
    ],
    "LTH-SUPPLY-01 (LTH Supply / Non-Exchange Supply)": [
        "NonEx_Supply",
        "Exchange_Supply_Pct",
        "LTH_Supply_Change_30d",
        "LTH_Accum_Signal",
        "NonEx_Supply_z90d",
    ],
}

# ── Settings: cuotas y listas esperadas ──────────────────────────────────────
try:
    from config.settings import cfg
    sfi_macro    = list(getattr(cfg.features, "sfi_macro_features",   []))
    sfi_onchain  = list(getattr(cfg.features, "sfi_onchain_features", []))
    sfi_calendar = list(getattr(cfg.features, "sfi_calendar_features", []))
    sfi_boost    = list(getattr(cfg.features, "sfi_macro_stable_features", []))
    macro_slots  = int(getattr(cfg.features, "sfi_macro_min_slots",   3))
    onchain_slots= int(getattr(cfg.features, "sfi_onchain_min_slots", 1))
    cal_slots    = int(getattr(cfg.features, "sfi_calendar_min_slots",1))
    print(f"[SFI-BALANCE-01] Cuotas: macro={macro_slots} onchain={onchain_slots} calendar={cal_slots}")
    print(f"[SFI] macro_features={len(sfi_macro)} | onchain_features={len(sfi_onchain)} | boost={len(sfi_boost)}")
except Exception as e:
    sfi_macro = sfi_onchain = sfi_calendar = sfi_boost = []
    macro_slots = onchain_slots = cal_slots = 0
    print(f"[WARN] No se pudo leer settings: {e}")

print()

# ── Cargar parquets ───────────────────────────────────────────────────────────
PARQUETS = {
    "features_train":      DATA / "features_train.parquet",
    "features_validation": DATA / "features_validation.parquet",
    "features_holdout":    DATA / "features_holdout.parquet",
}

dfs = {}
for name, path in PARQUETS.items():
    if path.exists():
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index, utc=True)
            dfs[name] = df
            print(f"[OK]  {name}: {df.shape[0]:,} filas x {df.shape[1]} cols | {df.index.min().date()} → {df.index.max().date()}")
        except Exception as e:
            print(f"[ERR] {name}: {e}")
    else:
        print(f"[--]  {name}: NO EXISTE")

# Preferir features_train para el análisis principal
main_df = dfs.get("features_train", dfs.get("features_validation", None))
if main_df is None:
    print("\n[FATAL] No hay parquets disponibles para analizar.")
    sys.exit(1)

print()

# ── Función de análisis por feature ──────────────────────────────────────────
def analyze_feature(df: pd.DataFrame, feat: str, parquet_name: str) -> dict:
    if feat not in df.columns:
        return {"status": "AUSENTE", "nan_pct": None, "std": None, "min": None, "max": None, "n_obs": 0}

    col = df[feat]
    n_total = len(col)
    n_nan   = col.isna().sum()
    nan_pct = n_nan / n_total * 100
    valid   = col.dropna()
    n_valid = len(valid)

    if n_valid == 0:
        return {"status": "TODO_NAN", "nan_pct": 100.0, "std": 0, "min": None, "max": None, "n_obs": 0}

    std_val = float(valid.std())
    min_val = float(valid.min())
    max_val = float(valid.max())
    last_val= float(valid.iloc[-1])

    # Detectar últimos 30d con datos
    cutoff_30d = df.index.max() - pd.Timedelta(days=30)
    last_30d   = col[df.index >= cutoff_30d]
    nan_30d_pct = last_30d.isna().mean() * 100 if len(last_30d) > 0 else 100.0

    # Clasificar status
    if nan_pct >= 99:
        status = "TODO_NAN"
    elif nan_pct >= 80:
        status = "CRITICO"  # >80% NaN — SFI lo rechazará por nan_threshold
    elif nan_pct >= 50:
        status = "DEGRADADO"  # SFI puede rechazarlo
    elif std_val < 1e-8:
        status = "CONSTANTE"  # varianza cero — no hay señal
    elif nan_30d_pct >= 90:
        status = "API_MUERTA"  # datos históricos OK pero sin actualización reciente
    elif nan_pct < 20 and std_val > 1e-6:
        status = "OK"
    else:
        status = "WARN"

    return {
        "status": status,
        "nan_pct": nan_pct,
        "nan_30d_pct": nan_30d_pct,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "n_obs": n_valid,
        "last": last_val,
    }

# ── SFI elegibilidad ──────────────────────────────────────────────────────────
def sfi_eligibility(feat: str) -> str:
    """Indica si la feature está registrada como macro, onchain, boost, etc."""
    parts = []
    if feat in sfi_macro:    parts.append("📋macro-list")
    if feat in sfi_onchain:  parts.append("📋onchain-list")
    if feat in sfi_calendar: parts.append("📋calendar-list")
    if feat in sfi_boost:    parts.append("⭐boost")
    return " ".join(parts) if parts else "⚠️ NO en ninguna lista SFI"

# ── Analizar cada grupo ───────────────────────────────────────────────────────
STATUS_ICON = {
    "OK":        "✅",
    "WARN":      "⚠️ ",
    "DEGRADADO": "🟡",
    "CRITICO":   "🔴",
    "TODO_NAN":  "❌",
    "CONSTANTE": "🔴",
    "API_MUERTA":"🟠",
    "AUSENTE":   "❌",
}

total_ok = 0
total_warn = 0
total_fail = 0
problemas = []

for group_name, features in GROUPS.items():
    print(f"\n── {group_name} {'─'*(50-len(group_name))}")
    for feat in features:
        info = analyze_feature(main_df, feat, "features_train")
        icon = STATUS_ICON.get(info["status"], "?")
        sfi_tag = sfi_eligibility(feat)

        if info["status"] == "AUSENTE":
            # Verificar si está en otros parquets
            found_in = [name for name, df in dfs.items() if feat in df.columns]
            if found_in:
                print(f"  {icon} {feat:<40} AUSENTE en train, presente en: {found_in}")
                total_warn += 1
                problemas.append((feat, "AUSENTE_EN_TRAIN", group_name))
            else:
                print(f"  {icon} {feat:<40} AUSENTE en TODOS los parquets")
                total_fail += 1
                problemas.append((feat, "AUSENTE_TOTAL", group_name))
            print(f"     SFI: {sfi_tag}")

        elif info["status"] in ("TODO_NAN", "CONSTANTE", "CRITICO"):
            print(f"  {icon} {feat:<40} NaN={info['nan_pct']:.1f}% | std={info['std']:.2e} | N={info['n_obs']:,}")
            total_fail += 1
            problemas.append((feat, info["status"], group_name))
            print(f"     SFI: {sfi_tag}")

        elif info["status"] in ("DEGRADADO", "WARN", "API_MUERTA"):
            print(f"  {icon} {feat:<40} NaN={info['nan_pct']:.1f}% | NaN_30d={info.get('nan_30d_pct',0):.0f}% | std={info['std']:.4f}")
            total_warn += 1
            problemas.append((feat, info["status"], group_name))
            print(f"     SFI: {sfi_tag}")

        else:  # OK
            print(f"  {icon} {feat:<40} NaN={info['nan_pct']:.1f}% | std={info['std']:.4f} | "
                  f"rango=[{info['min']:.3f}, {info['max']:.3f}] | last={info['last']:.4f} | N={info['n_obs']:,}")
            print(f"     SFI: {sfi_tag}")
            total_ok += 1

# ── Verificar presencia en features_holdout (OOS real) ───────────────────────
if "features_holdout" in dfs:
    holdout = dfs["features_holdout"]
    print("\n── Presencia en features_holdout (OOS 2025) ──────────────────────────")
    all_forced = [f for feats in GROUPS.values() for f in feats]
    for feat in all_forced:
        if feat not in holdout.columns:
            print(f"  ❌ {feat:<40} AUSENTE en holdout — el modelo NO puede usar esta feature OOS")
            problemas.append((feat, "AUSENTE_HOLDOUT", "OOS"))
        else:
            col = holdout[feat].dropna()
            nan_pct = (holdout[feat].isna().mean() * 100)
            if nan_pct > 80:
                print(f"  🔴 {feat:<40} NaN holdout={nan_pct:.0f}% — señal inusable en OOS")
                problemas.append((feat, "HOLDOUT_NAN_CRITICO", "OOS"))
            elif nan_pct > 30:
                print(f"  🟡 {feat:<40} NaN holdout={nan_pct:.0f}% | N={len(col):,}")
            else:
                print(f"  ✅ {feat:<40} NaN holdout={nan_pct:.0f}% | N={len(col):,} | std={col.std():.4f}")

# ── Chequeo de cuotas SFI: ¿hay suficientes features por categoría? ───────────
print("\n── Cobertura de cuotas SFI ───────────────────────────────────────────────")
def count_viable_in_list(feat_list: list, df: pd.DataFrame) -> tuple:
    """Retorna (n_viables, viables, no_viables)"""
    viables = []
    no_viables = []
    for feat in feat_list:
        if feat not in df.columns:
            no_viables.append((feat, "AUSENTE"))
            continue
        col = df[feat].dropna()
        nan_pct = df[feat].isna().mean() * 100
        if nan_pct > 85 or len(col) < 100 or col.std() < 1e-8:
            no_viables.append((feat, f"NaN={nan_pct:.0f}%"))
        else:
            viables.append(feat)
    return len(viables), viables, no_viables

n_macro_viable, macro_v, macro_nv = count_viable_in_list(sfi_macro, main_df)
n_onchain_viable, onchain_v, onchain_nv = count_viable_in_list(sfi_onchain, main_df)

macro_ok  = "✅" if n_macro_viable  >= macro_slots  else "❌"
onchain_ok= "✅" if n_onchain_viable >= onchain_slots else "❌"

print(f"  {macro_ok} Macro: {n_macro_viable}/{len(sfi_macro)} viables (cuota mínima={macro_slots})")
if macro_nv:
    for feat, reason in macro_nv[:5]:
        print(f"       ⚠️  {feat}: {reason}")

print(f"  {onchain_ok} Onchain: {n_onchain_viable}/{len(sfi_onchain)} viables (cuota mínima={onchain_slots})")
if onchain_nv:
    for feat, reason in onchain_nv[:8]:
        print(f"       ⚠️  {feat}: {reason}")

# ── Resumen final ─────────────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"[RESUMEN FORCED-FEATURES-AUDIT]")
print(f"  ✅ OK: {total_ok}")
print(f"  ⚠️  WARN: {total_warn}")
print(f"  ❌ FAIL: {total_fail}")
total_features = sum(len(v) for v in GROUPS.values())
print(f"  Total auditadas: {total_features}")
print()

if total_fail > 0:
    print("[ACCIÓN REQUERIDA] Features con problemas críticos:")
    for feat, reason, group in problemas:
        if reason not in ("DEGRADADO", "WARN"):
            print(f"  → {feat} ({group}): {reason}")
elif total_warn > 0:
    print("[AVISO] Features con calidad reducida (el SFI decidirá si las incluye):")
    for feat, reason, group in problemas:
        print(f"  → {feat} ({group}): {reason}")
else:
    print("✅ Todas las features forzadas tienen calidad suficiente para que el SFI las evalúe.")

print("=" * 72)
