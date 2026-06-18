# -*- coding: utf-8 -*-
"""
[SWEEP-THRESHOLD-01] Barrido de umbral prob_bull sobre OOS W1/W2/W3 seed-42
Cruza raw_probs con oos_trades para obtener outcomes reales por barras.
Genera tabla: percentil 0-100 -> n_trades, WR, Sharpe, MaxDD, Calmar

Run: python tools/diagnostics/sweep_threshold_windows.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "data" / "reports" / "wfb"
COST_RT = 0.0025   # 0.25% round-trip (R6 SOP V11.0)
BARS_PER_YEAR = 8760

WINDOWS = ["W1", "W2", "W3"]

print("[SWEEP-THRESHOLD-01] Barrido de umbrales prob_bull 0-100 sobre W1/W2/W3 seed-42")
print(f"[SWEEP-THRESHOLD-01] Costo RT: {COST_RT*100:.2f}%  |  Bars/year: {BARS_PER_YEAR}")
print("=" * 95)

# ---------------------------------------------------------------------------
# Inspeccion de columnas disponibles
# ---------------------------------------------------------------------------
for w in WINDOWS:
    p_prob  = REPORTS / f"oos_raw_probs_{w}_seed42.parquet"
    p_trade = REPORTS / f"oos_trades_{w}_seed42.parquet"
    if p_prob.exists():
        df = pd.read_parquet(p_prob)
        print(f"[SWEEP-THRESHOLD-01] {w} raw_probs  : {len(df)} filas | cols: {list(df.columns)}")
    if p_trade.exists():
        df = pd.read_parquet(p_trade)
        print(f"[SWEEP-THRESHOLD-01] {w} oos_trades : {len(df)} filas | cols: {list(df.columns)}")
print()

# ---------------------------------------------------------------------------
def compute_sweep(w_name: str) -> pd.DataFrame | None:
    p_prob  = REPORTS / f"oos_raw_probs_{w_name}_seed42.parquet"
    p_trade = REPORTS / f"oos_trades_{w_name}_seed42.parquet"

    if not p_prob.exists():
        print(f"[SWEEP-THRESHOLD-01] {w_name}: raw_probs NO encontrado, skip.")
        return None

    df_probs = pd.read_parquet(p_prob)

    # Columna de prob bull
    prob_col = None
    for c in ["prob_bull", "xgb_prob_final", "prob_final", "bull_long_prob",
              "xgb_prob_cal", "xgb_prob_raw"]:
        if c in df_probs.columns:
            prob_col = c
            break
    if prob_col is None:
        num = df_probs.select_dtypes(include=[np.number]).columns.tolist()
        if num:
            prob_col = num[0]
        else:
            print(f"[SWEEP-THRESHOLD-01] {w_name}: sin columna de prob, skip.")
            return None

    print(f"[SWEEP-THRESHOLD-01] {w_name} -> prob_col='{prob_col}' | "
          f"min={df_probs[prob_col].min():.4f} med={df_probs[prob_col].median():.4f} "
          f"max={df_probs[prob_col].max():.4f}")

    # Columna de retorno / outcome
    ret_col = None
    win_col = None
    df_trades = None

    if p_trade.exists():
        df_trades = pd.read_parquet(p_trade)
        for c in ["ret_pct", "ret", "pnl_pct", "return_pct", "pnl"]:
            if c in df_trades.columns:
                ret_col = c
                break
        for c in ["win", "outcome", "label", "y_true", "target", "is_win"]:
            if c in df_trades.columns:
                win_col = c
                break
        print(f"[SWEEP-THRESHOLD-01] {w_name} trades cols: "
              f"ret_col='{ret_col}' win_col='{win_col}' | {list(df_trades.columns)}")

    # Sweep
    percentiles = list(range(0, 101, 5))
    rows = []
    for pct in percentiles:
        thr = float(np.percentile(df_probs[prob_col].dropna(), pct))
        mask = df_probs[prob_col] >= thr
        n_signals = int(mask.sum())

        # Si tenemos trades reales, cruzar por indice/fecha
        if df_trades is not None and len(df_trades) > 0:
            # Intentar cruce por indice temporal si el indice de probs es datetime
            if hasattr(df_probs.index, 'dtype') and str(df_probs.index.dtype).startswith('datetime'):
                selected_idx = df_probs.index[mask]
                # Trades que tienen su entrada en el subset seleccionado
                entry_col = None
                for c in ["entry_time", "open_time", "entry_bar", "timestamp"]:
                    if c in df_trades.columns:
                        entry_col = c
                        break
                if entry_col:
                    df_sub = df_trades[df_trades[entry_col].isin(selected_idx)]
                else:
                    df_sub = df_trades  # fallback: todos los trades
            else:
                df_sub = df_trades  # sin cruce temporal: usar todos los trades como proxy

            n_trades = len(df_sub)

            if n_trades == 0:
                rows.append({"pct": pct, "thr": round(thr, 4), "n_signals": n_signals,
                             "n_trades": 0, "wr": np.nan, "sharpe": np.nan,
                             "maxdd": np.nan, "calmar": np.nan, "mean_ret": np.nan})
                continue

            # Calcular retornos
            if ret_col and ret_col in df_sub.columns:
                rets = df_sub[ret_col].values.astype(float) - COST_RT
            elif win_col and win_col in df_sub.columns:
                wins = df_sub[win_col].values.astype(float)
                # Asumir perfil TBM tipico: +1.5% win / -1.0% loss
                rets = np.where(wins == 1, 0.015 - COST_RT, -0.010 - COST_RT)
            else:
                rets = np.array([])

        else:
            # Sin trades file: solo contamos señales, sin metricas de rendimiento
            n_trades = n_signals
            rets = np.array([])

        if len(rets) == 0:
            rows.append({"pct": pct, "thr": round(thr, 4), "n_signals": n_signals,
                         "n_trades": n_trades, "wr": np.nan, "sharpe": np.nan,
                         "maxdd": np.nan, "calmar": np.nan, "mean_ret": np.nan})
            continue

        wr = float(np.mean(rets > 0))
        mean_ret = float(np.mean(rets))
        std_ret  = float(np.std(rets))
        sharpe   = (mean_ret / std_ret * np.sqrt(BARS_PER_YEAR / max(len(rets), 1))
                    ) if std_ret > 1e-10 else np.nan
        equity   = np.cumprod(1 + rets)
        runmax   = np.maximum.accumulate(equity)
        dd       = (equity - runmax) / runmax
        maxdd    = float(np.min(dd))
        ann_ret  = (1 + np.sum(rets)) ** (BARS_PER_YEAR / max(len(rets), 1)) - 1
        calmar   = ann_ret / abs(maxdd) if abs(maxdd) > 1e-10 else np.nan

        rows.append({
            "pct":      pct,
            "thr":      round(thr, 4),
            "n_signals": n_signals,
            "n_trades": len(rets),
            "wr":       round(wr * 100, 1),
            "mean_ret": round(mean_ret * 100, 3),
            "sharpe":   round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
            "maxdd":    round(maxdd * 100, 2),
            "calmar":   round(calmar, 3) if not np.isnan(calmar) else np.nan,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ejecutar y mostrar resultados
# ---------------------------------------------------------------------------
all_results = {}
for w in WINDOWS:
    df_res = compute_sweep(w)
    if df_res is None:
        continue
    all_results[w] = df_res
    print()
    print(f"{'=' * 95}")
    print(f"  {w} seed-42 -- Sweep prob_bull | Costo {COST_RT*100:.2f}% RT")
    print(f"{'=' * 95}")
    print(f"  {'%ile':>5} | {'Umbral':>7} | {'Senales':>8} | {'Trades':>7} | "
          f"{'WR%':>6} | {'MeanRet%':>9} | {'Sharpe':>7} | {'MaxDD%':>7} | {'Calmar':>7}")
    print(f"  {'-'*93}")
    for _, r in df_res.iterrows():
        wr_s  = f"{r['wr']:.1f}%" if not np.isnan(r['wr']) else "  N/A"
        mr_s  = f"{r['mean_ret']:.3f}%" if not np.isnan(r['mean_ret']) else "     N/A"
        sh_s  = f"{r['sharpe']:.3f}" if not np.isnan(r['sharpe']) else "   N/A"
        dd_s  = f"{r['maxdd']:.2f}%" if not np.isnan(r['maxdd']) else "   N/A"
        ca_s  = f"{r['calmar']:.3f}" if not np.isnan(r['calmar']) else "   N/A"
        flag  = " << R8>=30" if r['n_trades'] >= 30 else ("" if r['n_trades'] < 5 else " << min5")
        print(f"  {r['pct']:>5.0f}% | {r['thr']:>7.4f} | {int(r['n_signals']):>8} | {int(r['n_trades']):>7} | "
              f"{wr_s:>6} | {mr_s:>9} | {sh_s:>7} | {dd_s:>7} | {ca_s:>7}{flag}")

# ---------------------------------------------------------------------------
# Resumen: mejor umbral por metrica
# ---------------------------------------------------------------------------
print()
print("=" * 95)
print("  RESUMEN: Optimo por metrica (n_trades >= 5)")
print("=" * 95)
for w, df_res in all_results.items():
    valid = df_res[df_res["n_trades"] >= 5].copy()
    if valid.empty:
        print(f"  {w}: sin filas con n>=5 trades")
        continue
    print(f"\n  {w}:")
    for metric, label in [("wr", "Mejor WR"), ("sharpe", "Mejor Sharpe"), ("calmar", "Mejor Calmar")]:
        col = valid[metric].dropna()
        if col.empty:
            print(f"    {label}: N/A (sin datos de rendimiento)")
            continue
        best = valid.loc[col.idxmax()]
        print(f"    {label:15s} -> pct={best['pct']:.0f}% umbral={best['thr']:.4f} "
              f"n={int(best['n_trades'])} WR={best['wr']:.1f}% Sharpe={best['sharpe']} "
              f"MaxDD={best['maxdd']:.2f}% Calmar={best['calmar']}")

print()
print("[SWEEP-THRESHOLD-01] Completado.")
print("[SWEEP-THRESHOLD-01] AVISO: n_trades pequeno. Solo informativo. NO calibrar params sobre OOS.")
