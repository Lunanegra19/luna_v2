"""
arch01_tbm_ev_diagnosis.py
==========================
ARCH-01: TBM retornos submínimos — ¿Es el EV neto de costos realmente negativo?

PROTOCOLO (diagnostico_cuantitativo.md):
  FASE 1: Cargar todos los trades OOS de la última run disponible
  FASE 2: Hipótesis H1 (EV_neto < 0), H2 (WR>50% pero AvgLoss cancela AvgWin), H3 (MockXGB prob=0.6)
  FASE 3: Tests estadísticos (binom_test, ttest, distribución retornos)
  FASE 4: Causa raíz si H confirmada
  FASE 5: Counterfactual — qué tbm_min_return resolvería el EV

USO: python tools/diagnostics/arch01_tbm_ev_diagnosis.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import json

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

COST_PCT = 0.0015  # 0.15% round-trip SOP R6
PREDICTIONS_DIR = ROOT / "data" / "predictions"
RUNS_DIR = ROOT / "data" / "runs"

print("=" * 70)
print("[ARCH-01] DIAGNÓSTICO: TBM Expected Value neto de costos")
print("=" * 70)

# ── FASE 1: Cargar datos ───────────────────────────────────────────────────────

def load_all_oos_trades() -> pd.DataFrame:
    """Carga todos los trades OOS disponibles — predictions/ y runs/ más reciente."""
    all_dfs = []

    # 1a. data/predictions/ — trades acumulados multi-seed
    for f in sorted(PREDICTIONS_DIR.glob("oos_trades_seed*.parquet")):
        try:
            df = pd.read_parquet(f)
            df["seed"] = int(f.stem.replace("oos_trades_seed", ""))
            df["source"] = "predictions"
            all_dfs.append(df)
        except Exception as e:
            print(f"  [WARN] No se pudo leer {f.name}: {e}")

    # 1b. data/runs/ — última run WFB
    run_dirs = sorted(RUNS_DIR.glob("WFB_20260602_*"))
    if run_dirs:
        latest_run = run_dirs[-1]
        print(f"  [INFO] Última run WFB: {latest_run.name}")
        for f in sorted(latest_run.rglob("oos_trades.parquet")):
            try:
                df = pd.read_parquet(f)
                parts = f.parts
                seed_part = [p for p in parts if p.startswith("seed")]
                window_part = [p for p in parts if p.startswith("W") and len(p) == 2]
                df["seed"] = int(seed_part[0].replace("seed", "")) if seed_part else -1
                df["window"] = window_part[0] if window_part else "?"
                df["source"] = "latest_run"
                all_dfs.append(df)
            except Exception as e:
                print(f"  [WARN] No se pudo leer {f}: {e}")

    if not all_dfs:
        raise RuntimeError("No se encontraron archivos oos_trades. Verificar directorio.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"  [INFO] Total registros cargados: {len(df_all):,} | Seeds: {df_all['seed'].nunique()}")
    return df_all


df = load_all_oos_trades()

# Verificar columnas disponibles
print(f"\n  [INFO] Columnas disponibles: {list(df.columns)}")

# Normalizar columna de retorno
ret_col = None
for candidate in ["return_pct", "return_raw", "ret", "ret_pct", "net_return", "return"]:
    if candidate in df.columns:
        ret_col = candidate
        break

if ret_col is None:
    print("  [ERROR] No se encontró columna de retorno. Columnas:", list(df.columns))
    sys.exit(1)

print(f"  [INFO] Columna de retorno: '{ret_col}'")

# Normalizar en porcentaje decimal (0.01 = 1%)
sample_val = df[ret_col].abs().median()
if sample_val > 1.0:
    print(f"  [INFO] Retornos en porcentaje bruto (mediana={sample_val:.3f}%) → normalizando /100")
    df["ret_decimal"] = df[ret_col] / 100.0
else:
    df["ret_decimal"] = df[ret_col]

# Columna de retorno neto
df["ret_net"] = df["ret_decimal"] - COST_PCT

# Columna de régimen HMM
regime_col = None
for candidate in ["hmm_semantic", "HMM_Semantic", "regime", "agent", "hmm_regime"]:
    if candidate in df.columns:
        regime_col = candidate
        break

if regime_col:
    print(f"  [INFO] Columna régimen: '{regime_col}'")
    df["regime"] = df[regime_col].astype(str)
else:
    print("  [WARN] No se encontró columna de régimen — análisis global únicamente")
    df["regime"] = "ALL"

# ── FASE 2+3: Tests estadísticos ───────────────────────────────────────────────

print("\n" + "=" * 70)
print("FASE 2+3 — HIPÓTESIS Y TESTS ESTADÍSTICOS")
print("=" * 70)

# ── H0: Estadísticas globales ──────────────────────────────────────────────────
print("\n[H0] ESTADÍSTICAS BASE GLOBALES")
print("-" * 50)

n_total = len(df)
wr_global = (df["ret_decimal"] > 0).mean()
avg_win = df[df["ret_decimal"] > 0]["ret_decimal"].mean()
avg_loss = df[df["ret_decimal"] <= 0]["ret_decimal"].mean()
ev_bruto = df["ret_decimal"].mean()
ev_neto = df["ret_net"].mean()

print(f"  N total trades       : {n_total:,}")
print(f"  Win Rate (bruto)     : {wr_global:.1%}")
print(f"  Avg Win (bruto)      : {avg_win * 100:+.4f}%")
print(f"  Avg Loss (bruto)     : {avg_loss * 100:+.4f}%")
print(f"  P/L Ratio            : {abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'):.3f}")
print(f"  EV bruto/trade       : {ev_bruto * 100:+.4f}%")
print(f"  Costo SOP R6         : {COST_PCT * 100:.3f}%")
print(f"  EV NETO/trade        : {ev_neto * 100:+.4f}%")

# ── H1: EV neto < 0 ────────────────────────────────────────────────────────────
print("\n[H1] ¿EV NETO < 0? (test t de una muestra vs 0)")
t_stat, p_val = stats.ttest_1samp(df["ret_net"].dropna(), 0.0)
print(f"  t={t_stat:.4f}  p={p_val:.6f}")
if ev_neto < 0 and p_val < 0.05:
    print("  → H1 CONFIRMADA: EV neto es significativamente NEGATIVO")
elif ev_neto < 0 and p_val >= 0.05:
    print("  → H1 SUGERIDA pero no significativa (p>=0.05) — posible por N pequeño")
else:
    print("  → H1 DESCARTADA: EV neto NO es negativo (EV>0)")

# Binom test: ¿WR > 50% significativo?
from scipy.stats import binomtest
n_wins = int((df["ret_decimal"] > 0).sum())
binom_result = binomtest(n_wins, n_total, 0.5, alternative='greater')
print(f"\n  Binomial test WR>50%: p={binom_result.pvalue:.6f} (wins={n_wins}/{n_total})")
if binom_result.pvalue < 0.05:
    print("  → WR significativamente > 50%")
else:
    print("  → WR NO significativamente distinto de 50%")

# ── H2: Análisis por régimen ────────────────────────────────────────────────────
print("\n[H2] EV POR RÉGIMEN — ¿Qué régimen destruye EV?")
print("-" * 70)

regime_stats = df.groupby("regime").apply(lambda g: pd.Series({
    "N": len(g),
    "WR_%": round((g["ret_decimal"] > 0).mean() * 100, 1),
    "AvgWin_%": round(g[g["ret_decimal"] > 0]["ret_decimal"].mean() * 100, 4) if (g["ret_decimal"] > 0).any() else 0,
    "AvgLoss_%": round(g[g["ret_decimal"] <= 0]["ret_decimal"].mean() * 100, 4) if (g["ret_decimal"] <= 0).any() else 0,
    "EV_bruto_%": round(g["ret_decimal"].mean() * 100, 4),
    "EV_neto_%": round((g["ret_decimal"].mean() - COST_PCT) * 100, 4),
    "EV_neg": (g["ret_decimal"].mean() - COST_PCT) < 0,
})).reset_index()

print(regime_stats.to_string(index=False))

# ── H3: Distribución de xgb_prob — ¿hay prob=0.6 constante (MockXGB)? ─────────
print("\n[H3] ¿Señales con prob~0.60 constante? (hipótesis MockXGBClassifier)")
prob_col = None
for candidate in ["xgb_prob_cal", "xgb_prob", "prob", "signal_prob"]:
    if candidate in df.columns:
        prob_col = candidate
        break

if prob_col:
    probs = df[prob_col].dropna()
    print(f"  Columna prob: '{prob_col}' | N={len(probs)}")
    print(f"  mean={probs.mean():.4f} std={probs.std():.6f} min={probs.min():.4f} max={probs.max():.4f}")
    print(f"  Percentiles: p5={probs.quantile(0.05):.4f} p25={probs.quantile(0.25):.4f} "
          f"p50={probs.quantile(0.50):.4f} p75={probs.quantile(0.75):.4f} p95={probs.quantile(0.95):.4f}")

    # ¿Cuántos tienen prob en [0.58, 0.62]?
    mock_mask = (probs >= 0.58) & (probs <= 0.62)
    n_mock = mock_mask.sum()
    print(f"  Trades con prob∈[0.58,0.62] (posible Mock): {n_mock} ({n_mock/len(probs):.1%})")

    if probs.std() < 1e-4:
        print("  → H3 CONFIRMADA: std~0 indica MockXGBClassifier activo")
    elif n_mock / len(probs) > 0.80:
        print("  → H3 PROBABLE: >80% de señales en rango Mock [0.58, 0.62]")
    else:
        print("  → H3 DESCARTADA: distribución de probabilidades tiene varianza real")
else:
    print("  [WARN] No se encontró columna de probabilidad — H3 no evaluable")

# ── H4: Distribución de duración de trades ─────────────────────────────────────
print("\n[H4] DISTRIBUCIÓN DE DURACIÓN DE TRADES (horas)")
dur_col = None
for candidate in ["duration_h", "duration_hours", "hold_hours", "hours_held"]:
    if candidate in df.columns:
        dur_col = candidate
        break

if dur_col:
    durs = df[dur_col].dropna()
    print(f"  mean={durs.mean():.1f}h  median={durs.median():.1f}h  p10={durs.quantile(0.1):.1f}h  p90={durs.quantile(0.9):.1f}h")
    print(f"  Trades con duración < 24h: {(durs < 24).sum()} ({(durs < 24).mean():.1%})")
    print(f"  Trades con duración < 6h:  {(durs < 6).sum()} ({(durs < 6).mean():.1%})")

    # Correlación duración→retorno
    r, p = stats.spearmanr(durs, df.loc[durs.index, "ret_decimal"].dropna())
    print(f"  Spearman(duración, retorno): r={r:+.4f} p={p:.4f} → {'SIGNIFICATIVO' if p < 0.05 else 'ruido'}")
else:
    print("  [WARN] No se encontró columna de duración")

# ── H5: Counterfactual — ¿qué min_return resolvería el EV? ─────────────────────
print("\n[H5] COUNTERFACTUAL — min_return óptimo para EV > 0")
print("-" * 50)

# Lee el min_return actual de settings
try:
    import yaml
    with open(ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)
    tbm_min_return = float(cfg_raw.get("xgboost", {}).get("tbm_min_return", 0.003))
    pt_mult = float(cfg_raw.get("xgboost", {}).get("pt_mult_min", 1.5))
    sl_mult = float(cfg_raw.get("xgboost", {}).get("sl_mult_min", 0.8))
    print(f"  settings.yaml: tbm_min_return={tbm_min_return:.4f} ({tbm_min_return*100:.2f}%)")
    print(f"  settings.yaml: PT mult={pt_mult}x  SL mult={sl_mult}x")
except Exception as e:
    print(f"  [WARN] No se pudo leer settings.yaml: {e}")
    tbm_min_return = 0.003

print(f"\n  Retorno mínimo actual: {tbm_min_return*100:.3f}%")
print(f"  Costo round-trip SOP: {COST_PCT*100:.3f}%")
print(f"  EV break-even mínimo por trade: {COST_PCT*100:.3f}% bruto")

# Sweep: si filtramos solo trades con retorno > X, ¿cuántos quedan y cuál es el EV?
print("\n  Sweep de umbral mínimo de retorno bruto:")
print(f"  {'Umbral':>8} {'N_restante':>12} {'WR%':>8} {'EV_neto%':>10} {'Viable':>8}")
for min_ret in [0.001, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020]:
    sub = df[df["ret_decimal"].abs() >= min_ret]  # solo trades que superaron la barrera
    if len(sub) < 5:
        continue
    wr_s = (sub["ret_decimal"] > 0).mean()
    ev_s = sub["ret_decimal"].mean() - COST_PCT
    viable = "✓" if ev_s > 0 else "✗"
    print(f"  {min_ret*100:>7.3f}%  {len(sub):>12,}  {wr_s*100:>7.1f}%  {ev_s*100:>+9.4f}%  {viable:>8}")

# ── Resumen Final ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN ARCH-01")
print("=" * 70)
print(f"  N total trades: {n_total:,} | Seeds: {df['seed'].nunique()}")
print(f"  EV bruto: {ev_bruto*100:+.4f}% | EV neto: {ev_neto*100:+.4f}% | Costo: {COST_PCT*100:.3f}%")
print(f"  WR global: {wr_global:.1%}")
print(f"  P/L Ratio: {abs(avg_win/avg_loss) if avg_loss != 0 else float('inf'):.3f}")
print(f"  EV_neto {'NEGATIVO ❌' if ev_neto < 0 else 'POSITIVO ✓'}")

# Guardar CSV resumen
out_path = ROOT / "tools" / "diagnostics" / "arch01_results.csv"
regime_stats.to_csv(out_path, index=False)
print(f"\n  Resultados guardados en: {out_path}")
print("[ARCH-01] Diagnóstico completado.")
