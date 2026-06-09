"""
arch_deep_diagnosis.py — ARCH-01/04/21/25 análisis profundo
Usa features_train.parquet (IS actual) + seeds del día
"""
import sys, warnings
sys.path.insert(0, str(__import__('pathlib').Path('.').resolve()))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path('g:/Mi unidad/ia/luna_v2')

# ════════════════════════════════════════════════════════════════════════
# ARCH-01: TBM Real vs Forward Return
# ════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("ARCH-01: TBM — TARGET ACTUAL vs ALTERNATIVAS")
print("=" * 65)

feats_path = ROOT / "data" / "features" / "features_train.parquet"
df = pd.read_parquet(feats_path)

from config.settings import cfg
pt_m   = float(getattr(cfg.xgboost, "pt_mult_min", 1.8))
sl_m   = float(getattr(cfg.xgboost, "sl_mult_min", 1.5))
vbar   = int(getattr(cfg.xgboost, "vertical_barrier_hours", 72))
tbm_mr = float(getattr(cfg.xgboost, "tbm_min_return", 0.003))
cost   = 0.15

print(f"\nConfig TBM actual: pt={pt_m}x, sl={sl_m}x, vbar={vbar}H, min_return={tbm_mr:.4f}")

# 1a. Base rate del target IS actual
if "target" in df.columns:
    t = df["target"].dropna()
    br_total = (t == 1).mean()
    print(f"\nTarget IS: N={len(t)} | base_rate={br_total:.3f} ({br_total:.1%} wins)")
    print("\nBase rate por año:")
    for yr in range(2019, 2026):
        sub = df.dropna(subset=["target"])
        sub = sub[sub.index.year == yr]
        if len(sub) < 500:
            continue
        br = (sub["target"] == 1).mean()
        print(f"  {yr}: N={len(sub):5d}  BR={br:.3f}  ({br:.1%})")

# 1b. Simulación de retorno forward puro (sin TBM)
print("\nSimulación retorno forward puro (IS global 2017-2025):")
print(f"{'Horizonte':>10s} {'WR%':>7s} {'AvgWin%':>10s} {'AvgLoss%':>11s} {'P/L':>6s} {'EV_bruto%':>11s} {'EV_neto%':>10s}")
for h in [24, 48, 72, 96]:
    fwd = df["close"].pct_change(h).shift(-h) * 100
    fwd = fwd.dropna()
    wins   = fwd[fwd > cost]
    losses = fwd[fwd < -cost]
    wr     = (fwd > 0).mean()
    avg_w  = wins.mean() if len(wins) else 0.0
    avg_l  = losses.mean() if len(losses) else 0.0
    pl     = avg_w / abs(avg_l) if avg_l != 0 else 0.0
    ev_b   = fwd.mean()
    ev_n   = ev_b - cost
    print(f"  fwd_{h:2d}H: {wr:>6.1%} {avg_w:>10.3f}% {avg_l:>10.3f}% {pl:>6.2f} {ev_b:>10.4f}% {ev_n:>10.4f}%")

# 1c. ATR por año  
print("\nVolatilidad realizada por año (determina tamaño de barrera TBM):")
print(f"{'Año':>5s} {'HVol%/h':>9s} {'DailyVol%':>11s} {'PT(1.8x)%':>11s} {'SL(1.5x)%':>11s} {'EV_fwd72h_neto':>16s}")
for yr in range(2020, 2026):
    sub = df[df.index.year == yr]
    if len(sub) < 1000:
        continue
    hvol  = sub["close"].pct_change().std() * 100
    dvol  = hvol * (24 ** 0.5)
    pt_sz = pt_m * dvol
    sl_sz = sl_m * dvol
    fwd72 = sub["close"].pct_change(72).shift(-72) * 100
    ev_n  = fwd72.mean() - cost
    print(f"  {yr}: {hvol:>8.3f}% {dvol:>10.3f}% {pt_sz:>10.3f}% {sl_sz:>10.3f}% {ev_n:>15.4f}%")

# ════════════════════════════════════════════════════════════════════════
# ARCH-04: Optuna Metric + Threshold Mismatch
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ARCH-04: OPTUNA METRIC — Brier vs DSR, threshold 0.5 vs deploy")
print("=" * 65)

opt_metric   = getattr(cfg.xgboost, "optuna_metric", "unknown")
sw_min       = getattr(cfg.xgboost, "threshold_sweep_min", 0.45)
sw_max       = getattr(cfg.xgboost, "threshold_sweep_max", 0.72)
xgb_signal   = getattr(cfg.xgboost, "xgb_signal_threshold", 0.48)
xgb_min_sig  = getattr(cfg.xgboost, "xgb_min_signals_threshold", 0.45)
min_density  = getattr(cfg.xgboost, "threshold_min_density_pct", 0.15)
min_trades   = getattr(cfg.xgboost, "threshold_min_trades", 20)

print(f"\nSettings actuales:")
print(f"  optuna_metric:           {opt_metric}")
print(f"  threshold_sweep_min/max: {sw_min} → {sw_max}")
print(f"  xgb_signal_threshold:    {xgb_signal}")
print(f"  xgb_min_signals_thresh:  {xgb_min_sig}")
print(f"  threshold_min_density:   {min_density}")
print(f"  threshold_min_trades:    {min_trades}")

print(f"\nProblema de alineación:")
print(f"  Threshold en entrenamiento Optuna: 0.5 (hardcodeado en objective)")
print(f"  Threshold en deployment: {sw_min}-{sw_max} sweep → selecciona {xgb_signal:.2f} típico")
print(f"  Gap: Optuna nunca evaluó el modelo con CUTOFF = {xgb_signal:.2f}")

# Cuantificar el impacto del gap de threshold
val_path = ROOT / "data" / "features" / "features_validation.parquet"
if val_path.exists():
    df_v = pd.read_parquet(val_path)
    print(f"\nValidation set: N={len(df_v)} ({df_v.index.min().date()} → {df_v.index.max().date()})")

    # Simular sweep de threshold sobre distribution de probabilidades (proxy)
    # El calibrador isotónico output sería la distribución del modelo  
    # Usamos xgb_prob de las seeds del día como proxy real
    cutoff = datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc)
    dfs_today = []
    for f in (ROOT / "data" / "runs").rglob("oos_trades.parquet"):
        if datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc) >= cutoff:
            try:
                dfs_today.append(pd.read_parquet(f))
            except Exception:
                pass
    for f in (ROOT / "data" / "predictions").glob("oos_trades*.parquet"):
        if datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc) >= cutoff:
            try:
                dfs_today.append(pd.read_parquet(f))
            except Exception:
                pass

    if dfs_today:
        df_td = pd.concat(dfs_today, ignore_index=True).drop_duplicates()
        print(f"\nTrades del día (N={len(df_td)}):")
        if "xgb_prob_cal" in df_td.columns:
            p = df_td["xgb_prob_cal"].dropna()
            print(f"  xgb_prob_cal: mean={p.mean():.4f} std={p.std():.4f} min={p.min():.4f} max={p.max():.4f}")
        if "return_pct" in df_td.columns and "xgb_prob_cal" in df_td.columns:
            print(f"\n  Simulación sweep threshold sobre trades del día:")
            for thr in [0.45, 0.48, 0.50, 0.55, 0.60]:
                mask = df_td["xgb_prob_cal"] >= thr
                n_t = mask.sum()
                if n_t > 0:
                    ev = df_td.loc[mask, "return_pct"].mean()
                    wr = (df_td.loc[mask, "return_pct"] > 0).mean()
                    print(f"    thr={thr:.2f}: N={n_t:3d}  WR={wr:.1%}  EV_neto={ev:+.4f}%")

# ════════════════════════════════════════════════════════════════════════
# ARCH-21/23: SFI Global — varianza de correlación IS→target por ventana
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ARCH-21/23: SFI GLOBAL — estabilidad de features cross-window")
print("=" * 65)

sel_path = ROOT / "data" / "features" / "selected_features.json"
if sel_path.exists():
    with open(sel_path) as f:
        sel_data = json.load(f)
    global_feats = sel_data.get("selected_features", [])
    print(f"\nFeatures SFI global ({len(global_feats)}): {global_feats}")

    # Calcular correlacion con target por ventana rolling 5y
    if "target" in df.columns:
        df_full = df.dropna(subset=["target"])
        feat_list = [x for x in global_feats if x in df_full.columns][:12]
        windows_cfg = cfg.wfb.windows if hasattr(cfg.wfb, "windows") else []
        rwy = getattr(cfg.wfb, "rolling_window_years", 5)

        print(f"\nCorrelación IS con target por ventana (rolling_window_years={rwy}y):")
        header = f"{'Feature':35s}"
        for w in windows_cfg[:5]:
            header += f"  {w.get('id','??'):>6s}"
        header += "  StdVar"
        print(header)

        window_corrs = {}
        for feat in feat_list:
            row = f"  {feat[:33]:33s}"
            corrs = []
            for w in windows_cfg[:5]:
                w_end   = pd.Timestamp(str(w.get("train_end", "2025-01-01")), tz="UTC")
                w_start = w_end - pd.DateOffset(years=rwy)
                mask    = (df_full.index >= w_start) & (df_full.index <= w_end)
                df_w    = df_full.loc[mask]
                if feat in df_w.columns and len(df_w) > 100:
                    c = df_w[feat].corr(df_w["target"])
                    corrs.append(c)
                    row += f"  {c:>+6.3f}"
                else:
                    corrs.append(np.nan)
                    row += f"  {'NA':>6s}"
            std_c = np.nanstd(corrs)
            row += f"  {std_c:>6.3f}"
            print(row)
            window_corrs[feat] = corrs

        # Resumen de inestabilidad
        std_by_feat = {f: np.nanstd(v) for f, v in window_corrs.items()}
        stable   = [f for f, s in std_by_feat.items() if s < 0.03]
        unstable = [f for f, s in std_by_feat.items() if s >= 0.05]
        print(f"\n  Features estables (std<0.03): {len(stable)} — {stable}")
        print(f"  Features inestables (std>=0.05): {len(unstable)} — {unstable}")
        pct_unstable = len(unstable) / len(feat_list) if feat_list else 0
        print(f"  % de features inestables: {pct_unstable:.1%}")

# ════════════════════════════════════════════════════════════════════════
# ARCH-25: Validation — split en 2 bloques
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ARCH-25: VALIDATION REUTILIZADO — análisis de split")
print("=" * 65)

if val_path.exists():
    n_val = len(df_v)
    half  = n_val // 2
    df_v1 = df_v.iloc[:half]
    df_v2 = df_v.iloc[half:]
    print(f"\nValidation completo: N={n_val} ({df_v.index.min().date()} → {df_v.index.max().date()})")
    print(f"\nPropuesta de split funcional:")
    print(f"  val_A (OOD Guard):         N={len(df_v1)} ({df_v1.index.min().date()} → {df_v1.index.max().date()})")
    print(f"  val_B (threshold sweep):   N={len(df_v2)} ({df_v2.index.min().date()} → {df_v2.index.max().date()})")
    print(f"\n  [ARCH-05-D ya movió el isotónico al IS propio — solo quedan 2 usos]")
    print(f"  val_A N={len(df_v1)} >= 300: {'OK' if len(df_v1) >= 300 else 'INSUFICIENTE'}")
    print(f"  val_B N={len(df_v2)} >= 300: {'OK' if len(df_v2) >= 300 else 'INSUFICIENTE'}")

    if "HMM_Semantic" in df_v.columns:
        print(f"\n  Régimen en val_A:")
        for r, c in df_v1["HMM_Semantic"].value_counts().items():
            print(f"    {str(r):30s} {c} ({c/len(df_v1):.1%})")
        print(f"  Régimen en val_B:")
        for r, c in df_v2["HMM_Semantic"].value_counts().items():
            print(f"    {str(r):30s} {c} ({c/len(df_v2):.1%})")

print("\n" + "=" * 65)
print("DIAGNOSTICO COMPLETO")
print("=" * 65)
