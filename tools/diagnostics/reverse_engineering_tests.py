"""
tools/diagnostics/reverse_engineering_tests.py
================================================
Suite de tests de ingenieria inversa sobre los datos de la ultima run.
Objetivo: validar/refutar las 7 hipotesis del analisis profundo
SIN necesidad de relanzar el WFB.

Tests ejecutados:
  T1 - Bias 07h UTC: es artefacto TBM o sesgo de feature?
  T2 - Solo LONG: por que el agente bear nunca gana?
  T3 - Sensibilidad Consensus: curva trades vs threshold (1..5)
  T4 - Sensibilidad Embargo: curva trades vs embargo_hours
  T5 - Noviembre 2025: que ocurrio en el mercado? (precio BTC)
  T6 - Barreras TBM: cuanto miden en porcentaje?
  T7 - W1 EMPTY sistematico: DSR vs ventana historica

[REVERSE-ENG-TESTS 2026-05-30]
"""
import sys, re, json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT    = Path(r"g:\Mi unidad\ia\luna_v2")
wfb_dir = ROOT / "data" / "reports" / "wfb"
sys.path.insert(0, str(ROOT))

LAST_RUN_SEEDS = [42, 100, 777, 1337, 2025, 12751, 28020, 30915, 34324, 43610, 77542]

# Cargar todos los trades de la ultima run
def load_all_trades(seeds=None):
    seeds = seeds or LAST_RUN_SEEDS
    dfs = []
    for seed in seeds:
        for f in sorted(wfb_dir.glob(f"oos_trades_W*_seed{seed}.parquet")):
            try:
                df = pd.read_parquet(f)
                if df.empty:
                    continue
                win = re.search(r"(W\d)", f.stem).group(1)
                df["window"] = win
                df["seed"]   = seed
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                df.index = pd.to_datetime(df.index, utc=True)
                dfs.append(df)
            except Exception as e:
                print(f"  [WARN] {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).sort_index()

df_all = load_all_trades()
print(f"[REVERSE-ENG-TESTS] Trades cargados: {len(df_all)} de {len(LAST_RUN_SEEDS)} seeds")
print()

# ═══════════════════════════════════════════════════════════════════
# T1 — BIAS 07h UTC
# Hipotesis: el TBM cierra barreras a las 07h porque el max_holding
# es un multiplo fijo que alinea con esa hora. No es look-ahead.
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("T1 — BIAS 07h UTC: es artefacto TBM o sesgo real?")
print("=" * 70)

df_all["hour"] = df_all.index.hour
df_all["dow"]  = df_all.index.dayofweek  # 0=Lun

by_hour = df_all.groupby("hour").agg(
    n=("return_pct","count"),
    wr=("is_win","mean"),
    ret_mean=("return_pct","mean")
)
print("Distribucion por hora UTC:")
for h, row in by_hour.iterrows():
    bar = "#" * min(int(row["n"]), 40)
    print(f"  {int(h):02d}h: {int(row['n']):4d} trades | WR={row['wr']:.1%} | ret={row['ret_mean']*100:.4f}% {bar}")

# Comprobar: los trades de 07h, a que hora ENTRARON (si hay columna entry_time)
print()
trades_07h = df_all[df_all["hour"] == 7]
print(f"Trades a las 07h: {len(trades_07h)}")
if "entry_time" in df_all.columns:
    entry_hours = pd.to_datetime(trades_07h["entry_time"], utc=True).dt.hour.value_counts().sort_index()
    print("Horas de ENTRADA de trades que cierran a las 07h:")
    print(entry_hours)
else:
    # La columna de cierre es el INDEX. Checkear si hay duration o holding_hours
    cols = [c for c in df_all.columns if "dur" in c.lower() or "hold" in c.lower() or "exit" in c.lower() or "entry" in c.lower()]
    print(f"Columnas disponibles para analizar timing: {cols}")
    print("Muestra de trades a las 07h:")
    print(trades_07h[["seed","window","direction","is_win","return_pct"] + cols[:3]].head(10).to_string())

print()

# ═══════════════════════════════════════════════════════════════════
# T2 — SOLO LONG: analisis de la distribucion de señales
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("T2 — SOLO LONG: por que el agente bear nunca gana?")
print("=" * 70)

print("Direcciones en todos los trades:")
if "direction" in df_all.columns:
    dirs = df_all["direction"].value_counts()
    print(dirs)
else:
    print("  [WARN] Columna 'direction' no encontrada")

print()

# Analizar raw_probs para ver si el agente bull domina
raw_probs_files = list(wfb_dir.glob("oos_raw_probs_W*_seed*.parquet"))
if raw_probs_files:
    print(f"Archivos oos_raw_probs disponibles: {len(raw_probs_files)}")
    # Cargar uno para ver columnas
    try:
        df_probs = pd.read_parquet(raw_probs_files[0])
        print(f"Columnas en raw_probs: {df_probs.columns.tolist()}")
        prob_cols = [c for c in df_probs.columns if "prob" in c.lower() or "bull" in c.lower() or "bear" in c.lower()]
        if prob_cols:
            print()
            print("Estadisticas de probabilidades por agente:")
            print(df_probs[prob_cols].describe().round(4))
            # Cuantas veces bull > bear
            if "prob_bull" in df_probs.columns and "prob_bear" in df_probs.columns:
                bull_dom = (df_probs["prob_bull"] > df_probs["prob_bear"]).sum()
                bear_dom = (df_probs["prob_bear"] > df_probs["prob_bull"]).sum()
                total_r  = len(df_probs)
                print(f"\nBull > Bear: {bull_dom} ({bull_dom/total_r:.1%}) veces")
                print(f"Bear > Bull: {bear_dom} ({bear_dom/total_r:.1%}) veces")
    except Exception as e:
        print(f"  [WARN] Error leyendo raw_probs: {e}")
else:
    print("No hay archivos oos_raw_probs (solo se guardan para algunas seeds/ventanas)")

print()

# ═══════════════════════════════════════════════════════════════════
# T3 — SENSIBILIDAD CONSENSUS: cuantos trades a cada threshold
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("T3 — SENSIBILIDAD CONSENSUS: trades por threshold (sin rerun)")
print("=" * 70)

df_all["consensus_bucket"] = df_all.index.floor("2h")
bucket_unique_seeds = (
    df_all.groupby("consensus_bucket")["seed"]
    .nunique()
    .rename("consensus_count")
)
df_all["consensus_count"] = df_all["consensus_bucket"].map(bucket_unique_seeds)

print("Distribucion de consenso (seeds unicas por bucket 2H):")
dist = bucket_unique_seeds.value_counts().sort_index(ascending=False)
total_buckets = len(bucket_unique_seeds)
for n_s, n_b in dist.items():
    print(f"  {n_s:2d} seeds -> {n_b:4d} buckets ({n_b/total_buckets:.1%})")

print()
print("Curva trades-disponibles vs threshold (antes de embargo):")
print(f"{'Threshold':>10} | {'Filas':>6} | {'Buckets':>7} | {'WR approx':>10}")
print("-" * 45)
for thr in range(1, 9):
    subset = df_all[df_all["consensus_count"] >= thr]
    n_buckets = subset["consensus_bucket"].nunique()
    wr = subset["is_win"].mean() if len(subset) > 0 else 0.0
    print(f"  >= {thr:2d}     | {len(subset):6d} | {n_buckets:7d} | {wr:.1%}")

print()

# ═══════════════════════════════════════════════════════════════════
# T4 — SENSIBILIDAD EMBARGO: simulacion con distintas horas
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("T4 — SENSIBILIDAD EMBARGO: trades supervivientes por embargo_hours")
print("=" * 70)

# Usar consenso >= 2 (razonable) y variar el embargo
df_c2 = df_all[df_all["consensus_count"] >= 2].copy()
by_bucket_c2 = (
    df_c2.groupby("consensus_bucket")
    .agg(return_pct=("return_pct","mean"), is_win=("is_win","max"))
    .sort_index()
)

def simulate_embargo(df_portfolio, embargo_hours):
    selected = []
    last_time = None
    for ts, row in df_portfolio.iterrows():
        if last_time is None:
            selected.append(ts)
            last_time = ts
        else:
            delta_h = (ts - last_time).total_seconds() / 3600.0
            if delta_h >= embargo_hours:
                selected.append(ts)
                last_time = ts
    return df_portfolio.loc[selected]

print(f"Base: consensus >= 2 -> {len(by_bucket_c2)} buckets de consenso")
print()
print(f"{'Embargo (H)':>12} | {'Trades':>6} | {'WR':>7} | {'Sharpe aprox':>13} | {'Ret medio':>10}")
print("-" * 60)
for emb_h in [0, 12, 24, 36, 48, 72, 96, 144, 168]:
    if emb_h == 0:
        df_emb = by_bucket_c2.copy()
    else:
        df_emb = simulate_embargo(by_bucket_c2, emb_h)
    n = len(df_emb)
    wr  = df_emb["is_win"].mean() if n > 0 else 0.0
    ret = df_emb["return_pct"].mean() * 100 if n > 0 else 0.0
    std = df_emb["return_pct"].std() if n > 1 else float("nan")
    sh  = (df_emb["return_pct"].mean() / std * (n**0.5)) if (n > 1 and std > 1e-10) else float("nan")
    sh_str = f"{sh:.3f}" if not (isinstance(sh, float) and np.isnan(sh)) else "N/A"
    print(f"  {emb_h:>8}H   | {n:6d} | {wr:7.1%} | {sh_str:>13} | {ret:>10.4f}%")

print()

# ═══════════════════════════════════════════════════════════════════
# T5 — NOVIEMBRE 2025: que paso?
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("T5 — NOVIEMBRE 2025: analisis del periodo WR=0%")
print("=" * 70)

nov_trades = df_all[df_all.index.month == 11].copy()
print(f"Trades en noviembre 2025: {len(nov_trades)}")
if len(nov_trades) > 0:
    print(f"WR: {nov_trades['is_win'].mean():.1%}")
    print(f"Retorno medio: {nov_trades['return_pct'].mean()*100:.4f}%")
    print(f"Min retorno: {nov_trades['return_pct'].min()*100:.4f}%")
    print(f"Max retorno: {nov_trades['return_pct'].max()*100:.4f}%")
    if "hmm_regime" in nov_trades.columns:
        print(f"Regimenes HMM en Nov: {nov_trades['hmm_regime'].value_counts().to_dict()}")
    print()
    print("Detalle de trades de noviembre:")
    cols_show = ["seed","window","direction","is_win","return_pct","hmm_regime"] if "hmm_regime" in df_all.columns else ["seed","window","direction","is_win","return_pct"]
    print(nov_trades[cols_show].to_string())

print()

# ═══════════════════════════════════════════════════════════════════
# T6 — BARRERAS TBM: cuanto miden?
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("T6 — BARRERAS TBM: magnitud de retornos WIN vs LOSS")
print("=" * 70)

wins  = df_all[df_all["is_win"] == True]["return_pct"]  * 100
losses = df_all[df_all["is_win"] == False]["return_pct"] * 100

print(f"WINS  ({len(wins):3d} trades): media={wins.mean():.4f}% | max={wins.max():.4f}% | min={wins.min():.4f}%")
print(f"LOSSES({len(losses):3d} trades): media={losses.mean():.4f}% | min={losses.min():.4f}% | max={losses.max():.4f}%")
print(f"Ratio win/loss medio: {abs(wins.mean()/losses.mean()):.3f}x" if losses.mean() != 0 else "N/A")
print()

# Distribucion de retornos por percentil
print("Percentiles de retorno (todos los trades):")
for p in [5, 10, 25, 50, 75, 90, 95]:
    v = np.percentile(df_all["return_pct"]*100, p)
    print(f"  P{p:2d}: {v:.4f}%")

print()

# ═══════════════════════════════════════════════════════════════════
# T7 — W1 EMPTY SISTEMATICO: DSR vs periodo historico
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("T7 — W1 EMPTY SISTEMATICO: DSR por ventana en gate_G2")
print("=" * 70)

dsr_by_window = defaultdict(list)
for seed in LAST_RUN_SEEDS:
    for win in ["W1","W2","W3","W4","W5"]:
        gfile = wfb_dir / f"gate_G2_{win}_seed{seed}.json"
        if gfile.exists():
            try:
                data = json.load(open(gfile))
                dsr_mean = data.get("metrics", {}).get("dsr_mean", None)
                brier_mean = data.get("metrics", {}).get("brier_mean", None)
                if dsr_mean is not None:
                    dsr_by_window[win].append(dsr_mean)
            except Exception:
                pass

print(f"{'Ventana':8} | {'N seeds':8} | {'DSR mean avg':12} | {'DSR range':20}")
print("-" * 55)
for win in ["W1","W2","W3","W4","W5"]:
    vals = dsr_by_window[win]
    if vals:
        print(f"  {win}    | {len(vals):8d} | {np.mean(vals):12.4f} | [{min(vals):.4f}, {max(vals):.4f}]")
    else:
        print(f"  {win}    | {'0':8} | {'N/A':12} | N/A")

print()
print("[REVERSE-ENG-TESTS] Analisis completado.")
