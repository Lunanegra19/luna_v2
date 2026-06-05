import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint
import datetime

BASE = Path(r"g:\Mi unidad\ia\luna_v2")
RUNS = BASE / "data" / "runs"

print("=" * 65)
print("  H-RANGE-EDGE — El edge WR=100% (N=23) en VOLATILE_RANGE es real?")
print("=" * 65)
print()
print("  Protocolo:")
print("  1. Cargar todos los trades OOS RANGE de la run nocturna")
print("  2. binom_test H0: WR <= 0.50 (random)")
print("  3. IC95 Wilson del WR real")
print("  4. EV, avg_win, avg_loss, Sharpe estimado")
print("  5. Power analysis: con que N real llega p<0.05?")
print()

cutoff = datetime.datetime(2026, 6, 1, 22, 0, 0).timestamp()
overnight_runs = [d for d in sorted(RUNS.iterdir())
                  if d.is_dir() and d.stat().st_mtime >= cutoff]

all_trades = []
for run in overnight_runs:
    seed_label = run.name.split("seed")[-1] if "seed" in run.name else None
    if seed_label is None:
        continue
    seed_sub = run / seed_label
    if not seed_sub.exists():
        subs = [d for d in run.iterdir() if d.is_dir()]
        seed_sub = subs[0] if subs else None
    if seed_sub is None:
        continue
    for w in ["W1", "W2", "W3", "W4", "W5"]:
        pq = seed_sub / w / "oos_trades.parquet"
        if pq.exists():
            try:
                df = pd.read_parquet(pq)
                df["seed"] = seed_label
                df["window"] = w
                all_trades.append(df)
            except Exception as e:
                print(f"  [WARN] {pq}: {e}")

print(f"Runs nocturnas analizadas: {len(overnight_runs)}")
print(f"Parquets oos_trades cargados: {len(all_trades)}")

if not all_trades:
    print("  [ERROR] No se encontraron trades OOS en runs nocturnas")
    exit(1)

df_all = pd.concat(all_trades, ignore_index=True)
print(f"Total trades cargados: {len(df_all)}")
print()

# Filtrar por regimen RANGE
if "hmm_regime" in df_all.columns:
    df_range = df_all[df_all["hmm_regime"].str.contains("RANGE", na=False)]
    df_bull = df_all[df_all["hmm_regime"].str.contains("BULL", na=False)]
    df_bear = df_all[df_all["hmm_regime"].str.contains("BEAR", na=False)]
else:
    print("  [WARN] Columna hmm_regime no encontrada")
    df_range = df_all
    df_bull = pd.DataFrame()
    df_bear = pd.DataFrame()

print("Distribucion por regimen:")
print(f"  VOLATILE_RANGE: {len(df_range)} trades")
print(f"  BULL:           {len(df_bull)} trades")
print(f"  BEAR:           {len(df_bear)} trades")
print()

# Analisis completo de RANGE
print("=" * 65)
print("  ANALISIS RANGE — Estadisticas base")
print("=" * 65)

n = len(df_range)
if n == 0:
    print("  [ERROR] 0 trades RANGE encontrados")
    exit(1)

n_wins = int(df_range["is_win"].sum())
wr = df_range["is_win"].mean()
r = df_range["return_pct"]
wins = r[df_range["is_win"] == True]
losses = r[df_range["is_win"] == False]

print(f"  N trades RANGE:  {n}")
print(f"  Wins:            {n_wins}")
print(f"  WR:              {wr*100:.1f}%")
print(f"  EV por trade:    {r.mean()*100:+.4f}%")
print(f"  Std retorno:     {r.std()*100:.4f}%")
if len(wins) > 0:
    print(f"  avg_win:         {wins.mean()*100:+.4f}%  (N={len(wins)})")
if len(losses) > 0:
    print(f"  avg_loss:        {losses.mean()*100:+.4f}%  (N={len(losses)})")
    pl_ratio = abs(wins.mean() / losses.mean()) if len(wins) > 0 else float("inf")
    print(f"  P/L ratio:       {pl_ratio:.2f}")
else:
    print(f"  avg_loss:        N/A (0 losses)")
    pl_ratio = float("inf")

print()

# Percentiles de retorno
print("Distribución de retornos (return_pct):")
for p in [10, 25, 50, 75, 90]:
    print(f"  p{p}:  {r.quantile(p/100)*100:+.4f}%")
print()

# Detalle por seed y ventana
print("Trades por seed/ventana:")
if "seed" in df_range.columns and "window" in df_range.columns:
    summary = df_range.groupby(["seed", "window"]).agg(
        n=("is_win", "count"),
        wr=("is_win", "mean"),
        ev=("return_pct", "mean")
    ).reset_index()
    for _, row in summary.iterrows():
        print(f"  seed{row['seed']} {row['window']}: N={row['n']}  WR={row['wr']*100:.0f}%  EV={row['ev']*100:+.4f}%")
print()

# === TEST ESTADISTICO ===
print("=" * 65)
print("  FASE 3 — Test estadistico binom_test (one-sided)")
print("=" * 65)
print(f"  H0: WR_RANGE <= 0.50 (random, sin edge)")
print(f"  H1: WR_RANGE >  0.50 (edge real)")
print()

result = binomtest(n_wins, n, 0.5, alternative="greater")
p_val = result.pvalue
ic_low, ic_high = proportion_confint(n_wins, n, alpha=0.05, method="wilson")

print(f"  binom_test: wins={n_wins}, n={n}, p_H0=0.5, alt='greater'")
print(f"  p-value:    {p_val:.4f}")
print(f"  IC95 Wilson:[{ic_low*100:.1f}%, {ic_high*100:.1f}%]")
print()

if p_val < 0.05:
    print("  >>> RESULTADO: p < 0.05 -> H-RANGE-EDGE CONFIRMADA estadisticamente")
else:
    print(f"  >>> RESULTADO: p = {p_val:.4f} > 0.05 -> NO concluyente con N={n}")
    print(f"  >>> ESTADO: EXPLORATORIA — WR={wr*100:.0f}% es prometedor pero N insuficiente")
print()

# === POWER ANALYSIS ===
print("=" * 65)
print("  FASE 5 — Power analysis: con que N llega p<0.05?")
print("=" * 65)

from scipy.stats.distributions import binom as binom_dist

print("  Asumiendo WR_real conservador = IC95 lower bound (Wilson)")
print(f"  WR_real asumida: {ic_low*100:.1f}%")
print()
print(f"  {'N':>6}  {'k_min':>6}  {'Power':>8}  {'Runs necesarias (~1 trade/seed/run)':>36}")
print("  " + "-" * 65)

wr_assumed = ic_low  # conservador: IC95 lower
for n_test in [23, 30, 40, 50, 75, 100, 150]:
    # Encontrar k_min (cuantos wins para p<0.05)
    k_min = next(k for k in range(n_test, -1, -1) if binom_dist.sf(k-1, n_test, 0.5) < 0.05)
    # Probabilidad de detectar el edge con WR_real
    power = 1 - binom_dist.cdf(k_min - 1, n_test, wr_assumed)
    # Runs adicionales necesarias (tenemos N=23, tasa 23/29-seeds/run)
    runs_needed = max(0, (n_test - n) / (n / 29)) if n > 0 else 999
    print(f"  {n_test:>6}  {k_min:>6}  {power*100:>7.1f}%  {runs_needed:>5.1f} runs adicionales")

print()
print("  También probando WR_real = 0.70 y 0.75 (escenarios optimistas):")
print(f"  {'WR_real':>8}  {'N=30 power':>12}  {'N=50 power':>12}  {'N=100 power':>12}")
print("  " + "-" * 50)
for wr_test in [ic_low, 0.65, 0.70, 0.75, 0.80]:
    powers = []
    for n_test in [30, 50, 100]:
        k_min = next(k for k in range(n_test, -1, -1) if binom_dist.sf(k-1, n_test, 0.5) < 0.05)
        power = 1 - binom_dist.cdf(k_min - 1, n_test, wr_test)
        powers.append(power)
    print(f"  {wr_test*100:>7.0f}%  {powers[0]*100:>10.1f}%  {powers[1]*100:>10.1f}%  {powers[2]*100:>10.1f}%")

print()
# === RESUMEN FINAL ===
print("=" * 65)
print("  RESULTADO FINAL H-RANGE-EDGE")
print("=" * 65)
print(f"  N = {n}  |  WR = {wr*100:.0f}%  |  p-value = {p_val:.4f}")
print(f"  IC95 WR: [{ic_low*100:.1f}%, {ic_high*100:.1f}%]")
print(f"  EV por trade: {r.mean()*100:+.4f}%")
print()
if p_val < 0.05:
    print("  >>> H-RANGE-EDGE: CONFIRMADA (p < 0.05)")
else:
    print("  >>> H-RANGE-EDGE: EXPLORATORIA (p > 0.05, N insuficiente)")
    print(f"  >>> El IC95 lower bound ({ic_low*100:.1f}%) > 50% indica edge probable")
    print(f"  >>> Se necesitan ~{max(0, 30 - n)} trades mas para primer test concluyente (N=30)")
    print(f"  >>> Se necesitan ~{max(0, 50 - n)} trades para power >80% (asumiendo WR={wr*100:.0f}%)")
print()
print("[FIX-DIAG-H-RANGE-01] Test H-RANGE-EDGE completado sobre todos los trades OOS de la run nocturna.")
