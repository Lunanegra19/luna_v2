"""
arch_structural_reanalysis.py
Diagnóstico profundo para ARCH-01, ARCH-04, ARCH-21/23, ARCH-25
Solo usa seeds con datos de las últimas 24h.
"""
import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

print("=" * 70)
print("ARCH STRUCTURAL REANALYSIS — solo seeds < 24h")
fmt = "%Y-%m-%d %H:%M UTC"
print(f"Cutoff: {cutoff.strftime(fmt)}")
print("=" * 70)

# ─── CARGAR SEEDS RECIENTES ───────────────────────────────────────────────
oos_dir = ROOT / "data" / "reports" / "wfb"
files = sorted(oos_dir.glob("oos_trades_W*.parquet"), key=lambda f: f.stat().st_mtime, reverse=True)

recent_files = []
old_files = []
for f in files:
    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
    age_h = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
    if mtime >= cutoff:
        recent_files.append((f, age_h))
    else:
        old_files.append((f, age_h))

print(f"\nSeeds < 24h: {len(recent_files)} | Seeds descartadas: {len(old_files)}")
for f, age_h in recent_files:
    print(f"  {f.name}  ({age_h:.1f}h ago)")

if not recent_files:
    print("\nERROR: No hay seeds con datos < 24h. Ampliar criterio.")
    sys.exit(1)

# Consolidar todos los trades recientes
dfs = []
for f, age_h in recent_files:
    try:
        df_tmp = pd.read_parquet(f)
        df_tmp["_source_file"] = f.name
        dfs.append(df_tmp)
    except Exception as e:
        print(f"  WARN: no se pudo leer {f.name}: {e}")

if not dfs:
    print("ERROR: No se pudo leer ningún parquet reciente")
    sys.exit(1)

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nTotal trades (seeds < 24h): {len(df_all)} en {len(dfs)} archivos")

# ═══════════════════════════════════════════════════════════════════════════
# ARCH-01 — TBM: Análisis Profundo del Target
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ARCH-01: ANÁLISIS TBM — ¿Qué tipo de target produce edge real?")
print("=" * 70)

# 1a. Distribución de retornos por tipo de cierre
ret_col = None
for c in ["ret", "return", "pct_return", "ret_pct", "trade_return"]:
    if c in df_all.columns:
        ret_col = c
        break

regime_col = None
for c in ["hmm_regime", "regime", "HMM_Regime", "HMM_Semantic", "hmm_semantic"]:
    if c in df_all.columns:
        regime_col = c
        break

print(f"\nColumnas disponibles: {list(df_all.columns)[:20]}")
print(f"Columna de retorno detectada: {ret_col}")
print(f"Columna de régimen detectada: {regime_col}")

if ret_col:
    df_all["ret_pct"] = df_all[ret_col] * 100 if df_all[ret_col].abs().max() < 10 else df_all[ret_col]
    cost_pct = 0.15

    print(f"\n--- Estadísticas de retorno bruto (N={len(df_all)}) ---")
    print(f"  Media:    {df_all['ret_pct'].mean():.4f}%")
    print(f"  Mediana:  {df_all['ret_pct'].median():.4f}%")
    print(f"  Std:      {df_all['ret_pct'].std():.4f}%")
    print(f"  P5:       {df_all['ret_pct'].quantile(0.05):.4f}%")
    print(f"  P25:      {df_all['ret_pct'].quantile(0.25):.4f}%")
    print(f"  P75:      {df_all['ret_pct'].quantile(0.75):.4f}%")
    print(f"  P95:      {df_all['ret_pct'].quantile(0.95):.4f}%")

    wins = df_all[df_all["ret_pct"] > 0]
    losses = df_all[df_all["ret_pct"] < 0]
    print(f"\n  WR:       {len(wins)/len(df_all):.1%}")
    print(f"  AvgWin:   +{wins['ret_pct'].mean():.4f}%")
    print(f"  AvgLoss:  {losses['ret_pct'].mean():.4f}%")
    pl_ratio = wins['ret_pct'].mean() / abs(losses['ret_pct'].mean()) if len(losses) > 0 else float('inf')
    print(f"  P/L Ratio: {pl_ratio:.3f}")
    ev_bruto = df_all['ret_pct'].mean()
    ev_neto  = ev_bruto - cost_pct
    print(f"\n  EV bruto:  {ev_bruto:+.4f}%")
    print(f"  Coste SOP: -{cost_pct:.4f}%")
    print(f"  EV NETO:   {ev_neto:+.4f}%  {'✗ NEGATIVO' if ev_neto < 0 else '✓ POSITIVO'}")

    # 1b. ¿Qué pasaría con un target forward 48H?
    print("\n--- Simulación: ¿Qué retorno sería necesario para EV neto > 0? ---")
    needed_avg = cost_pct  # para que EV neto = 0
    print(f"  Se necesita AvgReturn > {needed_avg:.3f}% para cubrir costes")
    print(f"  Retorno actual: {ev_bruto:.4f}% ({ev_bruto/needed_avg:.1f}x menor que el mínimo)")
    print(f"  Conclusión: necesitamos trades con retorno medio ~{needed_avg*2:.2f}% para EV positivo con buffer")

    # 1c. Por régimen
    if regime_col:
        print(f"\n--- EV neto por régimen (col: {regime_col}) ---")
        for reg, grp in df_all.groupby(regime_col):
            n = len(grp)
            if n < 5:
                continue
            wr = (grp["ret_pct"] > 0).mean()
            ev_n = grp["ret_pct"].mean() - cost_pct
            print(f"  {str(reg)[:30]:30s}  N={n:4d}  WR={wr:.1%}  EV_neto={ev_n:+.4f}%")

# ═══════════════════════════════════════════════════════════════════════════
# ARCH-04 — Optuna Metric Mismatch + Threshold Deployment
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ARCH-04: OPTUNA METRIC — Brier vs DSR, threshold 0.5 vs 0.62")
print("=" * 70)

# Leer la métrica actual de settings
try:
    from config.settings import cfg
    opt_metric = cfg.xgboost.optuna_metric
    sweep_min = cfg.xgboost.threshold_sweep_min
    sweep_max = cfg.xgboost.threshold_sweep_max
    sweep_step = cfg.xgboost.threshold_sweep_step
    min_density = cfg.xgboost.threshold_min_density_pct
    min_trades = cfg.xgboost.threshold_min_trades
    print(f"\n  optuna_metric:          {opt_metric}")
    print(f"  threshold_sweep_min:    {sweep_min}")
    print(f"  threshold_sweep_max:    {sweep_max}")
    print(f"  threshold_sweep_step:   {sweep_step}")
    print(f"  threshold_min_density:  {min_density}")
    print(f"  threshold_min_trades:   {min_trades}")
except Exception as e:
    print(f"  ERROR leyendo settings: {e}")

prob_col = None
for c in ["xgb_prob_cal", "xgb_prob", "prob_cal", "signal_prob"]:
    if c in df_all.columns:
        prob_col = c
        break

if prob_col and ret_col:
    print(f"\n--- Distribución de probabilidades (col: {prob_col}) ---")
    probs = df_all[prob_col].dropna()
    print(f"  N señales:  {len(probs)}")
    print(f"  Media prob: {probs.mean():.4f}")
    print(f"  Std prob:   {probs.std():.4f}")
    for thr in [0.45, 0.48, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70]:
        mask = df_all[prob_col] >= thr
        n_t = mask.sum()
        if n_t > 0:
            ev_n = df_all.loc[mask, "ret_pct"].mean() - cost_pct if ret_col else float('nan')
            wr = (df_all.loc[mask, "ret_pct"] > 0).mean() if ret_col else float('nan')
            print(f"  thr={thr:.2f}  N={n_t:4d}  WR={wr:.1%}  EV_neto={ev_n:+.5f}%")

# ═══════════════════════════════════════════════════════════════════════════
# ARCH-21/23 — SFI Global: Estabilidad de Features Cross-Window
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ARCH-21/23: SFI GLOBAL — Estabilidad de features entre ventanas WFB")
print("=" * 70)

# Buscar selected_features por ventana
sel_path = ROOT / "data" / "features" / "selected_features.json"
if sel_path.exists():
    with open(sel_path) as f:
        sel_data = json.load(f)
    global_feats = set(sel_data.get("selected_features", []))
    print(f"\n  Features seleccionadas (SFI global): {len(global_feats)}")
    print(f"  {sorted(global_feats)}")
else:
    print("  WARN: selected_features.json no encontrado")
    global_feats = set()

# Buscar selected_features por ventana (si existen)
wfb_sel_files = list((ROOT / "data" / "features").glob("selected_features_W*.json"))
if wfb_sel_files:
    print(f"\n  selected_features por ventana encontrados: {len(wfb_sel_files)}")
    all_window_feats = {}
    for wf in sorted(wfb_sel_files):
        with open(wf) as f:
            wd = json.load(f)
        all_window_feats[wf.stem] = set(wd.get("selected_features", []))

    # Calcular estabilidad
    if len(all_window_feats) >= 2:
        windows = list(all_window_feats.keys())
        all_feat_union = set().union(*all_window_feats.values())
        consistency = {}
        for feat in all_feat_union:
            n_windows_with = sum(1 for w in all_window_feats.values() if feat in w)
            consistency[feat] = n_windows_with / len(all_window_feats)

        stable = {f: c for f, c in consistency.items() if c >= 0.8}
        unstable = {f: c for f, c in consistency.items() if c < 0.5}
        print(f"\n  Features estables (>= 80% ventanas):  {len(stable)}")
        for feat, pct in sorted(stable.items(), key=lambda x: -x[1]):
            print(f"    {feat:40s} {pct:.0%}")
        print(f"\n  Features inestables (< 50% ventanas): {len(unstable)}")
        for feat, pct in sorted(unstable.items(), key=lambda x: x[1])[:10]:
            print(f"    {feat:40s} {pct:.0%}")
else:
    print("\n  No hay selected_features_W*.json — SFI se ejecuta globalmente (no por ventana)")
    print("  CONFIRMADO: ARCH-21/23 presente — SFI no re-ejecuta por ventana WFB")

    # Analizar variación del IS por ventana desde los parquets de features
    print("\n  --- Análisis de variación IS por ventana (features_train.parquet) ---")
    feats_train = ROOT / "data" / "features" / "features_train.parquet"
    if feats_train.exists():
        try:
            from config.settings import cfg as _cfg
            windows = _cfg.wfb.windows if hasattr(_cfg.wfb, 'windows') else []
            rwy = _cfg.wfb.rolling_window_years

            df_ft = pd.read_parquet(feats_train, columns=list(global_feats)[:10] if global_feats else None)
            print(f"\n  features_train.parquet: {len(df_ft)} barras ({df_ft.index.min().year}-{df_ft.index.max().year})")

            if windows and global_feats:
                print("\n  Correlación entre features y target VARÍA por ventana:")
                target_col = None
                df_full = pd.read_parquet(feats_train)
                for c in ["target", "bin", "label"]:
                    if c in df_full.columns:
                        target_col = c
                        break
                if target_col:
                    feat_list = sorted(global_feats)[:8]
                    corr_by_window = {}
                    for w in windows[:5]:
                        w_end = pd.Timestamp(str(w.get("train_end", "2025-01-01")), tz="UTC")
                        w_start = w_end - pd.DateOffset(years=rwy)
                        mask = (df_full.index >= w_start) & (df_full.index <= w_end)
                        df_w = df_full.loc[mask]
                        w_corrs = {}
                        for feat in feat_list:
                            if feat in df_w.columns:
                                c_val = df_w[feat].corr(df_w[target_col])
                                w_corrs[feat] = round(c_val, 4)
                        corr_by_window[w.get("id", "??")] = w_corrs
                        print(f"    {w.get('id'):4s} (IS: {w_start.date()}-{w_end.date()}): {w_corrs}")

                    # Calcular varianza de correlación por feature
                    print("\n  Varianza de correlación (alto = inestable cross-window):")
                    for feat in feat_list:
                        vals = [corr_by_window[w][feat] for w in corr_by_window if feat in corr_by_window[w]]
                        if vals:
                            print(f"    {feat:35s} std={np.std(vals):.4f}  vals={[round(v,3) for v in vals]}")
        except Exception as e:
            print(f"  WARN al analizar varianza por ventana: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# ARCH-25 — Validation Reutilizado
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ARCH-25: VALIDATION REUTILIZADO — ¿Cuántos usos del mismo set?")
print("=" * 70)

val_parquet = ROOT / "data" / "features" / "features_validation.parquet"
if val_parquet.exists():
    df_val = pd.read_parquet(val_parquet)
    n_val = len(df_val)
    print(f"\n  features_validation.parquet: {n_val} barras")
    print(f"  Rango: {df_val.index.min().date()} → {df_val.index.max().date()}")

    # ¿Cuántos meses tiene?
    months = (df_val.index.max() - df_val.index.min()).days / 30.4
    print(f"  Duración: {months:.1f} meses")
    print(f"\n  USOS ACTUALES del mismo validation set:")
    print(f"  1. OOD Guard (ood_guard): detecta drift en validation")
    print(f"  2. Threshold sweep (_calibrate_threshold): sweep 0.45-0.72")
    print(f"  3. Calibrador isotónico (_do_isotonic_calibration): [ARCH-05-D] ya movido a IS propio")
    print(f"\n  USOS RESTANTES que compiten por el mismo set:")
    print(f"  → OOD Guard + Threshold sweep usan las mismas {n_val} barras como set 'OOS'")
    print(f"  → Si dividimos en 2 bloques: {n_val//2} barras/bloque ({months/2:.1f} meses)")
    print(f"\n  ¿Es suficiente N para split?")
    min_block = 300
    print(f"  Mínimo recomendado por bloque: {min_block} barras")
    print(f"  {'✓ SUFICIENTE' if n_val//2 >= min_block else '✗ INSUFICIENTE'}: {n_val//2} barras/bloque")

    # Análisis de régimen en validation
    if "HMM_Semantic" in df_val.columns:
        print(f"\n  Distribución régimen en validation (ARCH-05 contexto):")
        for reg, cnt in df_val["HMM_Semantic"].value_counts().items():
            print(f"    {str(reg):30s} {cnt:5d} ({cnt/n_val:.1%})")

    # ¿Cuántos trades recientes vinieron de threshold sweep sobre este val?
    if prob_col and ret_col:
        # Simular split: primer 50% para OOD/threshold, último 50% para holdout
        split_idx = n_val // 2
        print(f"\n  Propuesta de split:")
        print(f"  val_A (OOD Guard + isotónico IS): barras 1-{split_idx} ({df_val.index.min().date()} → {df_val.iloc[split_idx].name.date() if hasattr(df_val.iloc[split_idx], 'name') else '??'})")
        print(f"  val_B (threshold sweep): barras {split_idx}-{n_val} ← OOS verdaderamente independiente")
else:
    print("  WARN: features_validation.parquet no encontrado")

print("\n" + "=" * 70)
print("ANÁLISIS COMPLETO")
print("=" * 70)
