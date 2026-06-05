"""
test_H_SAFE_benchmarks.py
==========================
H-SAFE-1: Null model benchmark (percentil de nuestro WR vs aleatorio)
H-SAFE-2: Bootstrap confidence interval del WR
H-SAFE-3: Calmar, Sharpe y metricas V2 completas del ensemble
H-SAFE-4: Test binomial formal del edge
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

np.random.seed(42)
WFB = Path("data/reports/wfb")
dfs = []
for f in sorted(WFB.glob("oos_trades_W*_seed*.parquet")):
    parts = f.stem.split("_")
    df = pd.read_parquet(f)
    df["_w"]    = parts[2]
    df["_seed"] = parts[3].replace("seed", "")
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined["entry_dt"] = pd.to_datetime(combined["entry_time"], utc=True, errors="coerce")
combined = combined.sort_values("entry_dt").reset_index(drop=True)

N = len(combined)
WR = combined["is_win"].mean()
wins = int(combined["is_win"].sum())

print("=" * 65)
print("TEST H-SAFE — Benchmarks Estadisticos")
print("=" * 65)
print(f"\nDataset: N={N} trades | WR={WR:.4f} ({WR*100:.2f}%) | Wins={wins}")

# ── H-SAFE-1: Null model benchmark ───────────────────────────────
print("\n─" * 65)
print("H-SAFE-1: Benchmark vs modelo aleatorio (Monte Carlo)")
print("─" * 65)
N_SIM = 50_000
sim_wrs = np.random.binomial(N, 0.50, size=N_SIM) / N
pct_rank = (sim_wrs < WR).mean() * 100
print(f"  WR del sistema: {WR:.4f} ({WR*100:.2f}%)")
print(f"  Distribución aleatoria (50% WR, N={N}): media={np.mean(sim_wrs):.4f} std={np.std(sim_wrs):.4f}")
print(f"  Percentil de nuestro WR: {pct_rank:.1f}%")
print(f"  Rango 95% de WRs aleatorios: [{np.percentile(sim_wrs,2.5):.4f}, {np.percentile(sim_wrs,97.5):.4f}]")
if pct_rank >= 95:
    print(f"  → EDGE CONFIRMADO: mejor que el 95% de modelos aleatorios")
elif pct_rank >= 80:
    print(f"  → Edge MODERADO: mejor que el 80% de modelos aleatorios")
else:
    print(f"  → Sin edge: el sistema no supera significativamente al azar")

# ── H-SAFE-2: Bootstrap CI del WR ─────────────────────────────────
print("\n─" * 65)
print("H-SAFE-2: Bootstrap IC 95% del Win Rate")
print("─" * 65)
N_BOOT = 10_000
is_win_arr = combined["is_win"].values
boot_wrs = np.array([
    np.random.choice(is_win_arr, size=N, replace=True).mean()
    for _ in range(N_BOOT)
])
ci_lo = np.percentile(boot_wrs, 2.5)
ci_hi = np.percentile(boot_wrs, 97.5)
print(f"  WR observado:     {WR:.4f}")
print(f"  IC 95% Bootstrap: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Amplitud IC:      {ci_hi-ci_lo:.4f} ({(ci_hi-ci_lo)*100:.2f}pp)")
if ci_lo > 0.50:
    print(f"  → IC EXCLUYE 50%: edge estadisticamente significativo (p<0.05)")
else:
    print(f"  → IC INCLUYE 50%: edge NO estadisticamente significativo")

# Test binomial formal
from scipy.stats import binomtest
result = binomtest(wins, N, 0.50, alternative="greater")
print(f"\n  Test binomial (H0: WR=50%, H1: WR>50%):")
print(f"  p-value = {result.pvalue:.4f}  {'→ SIGNIFICATIVO' if result.pvalue < 0.05 else '→ No significativo'}")

# ── H-SAFE-3: Métricas V2 completas ──────────────────────────────
print("\n─" * 65)
print("H-SAFE-3: Metricas V2 completas del ensemble")
print("─" * 65)

# Retorno acumulado
ret = combined["return_raw"].fillna(0)
equity = (1 + ret).cumprod()
total_ret = equity.iloc[-1] - 1

# Drawdown
running_max = equity.cummax()
drawdown = (equity - running_max) / running_max
max_dd = drawdown.min()

# Sharpe (asumiendo retornos por trade, ~1 trade/hora en promedio)
# Frecuencia media en horas
if combined["entry_dt"].notna().sum() > 1:
    total_hours = (combined["entry_dt"].max() - combined["entry_dt"].min()).total_seconds() / 3600
    trades_per_year = N / (total_hours / 8760)
    annual_factor = np.sqrt(trades_per_year)
else:
    annual_factor = np.sqrt(365)

sharpe = (ret.mean() / ret.std()) * annual_factor if ret.std() > 0 else 0
calmar = abs(total_ret / max_dd) if max_dd != 0 else float("inf")

print(f"  Retorno total acumulado:    {total_ret*100:+.2f}%")
print(f"  Retorno total nominal:      {ret.sum()*100:+.2f}%")
print(f"  Maximo Drawdown:            {max_dd*100:.2f}%")
print(f"  Sharpe anualizado:          {sharpe:.3f}")
print(f"  Calmar ratio:               {calmar:.3f}")
print(f"  Win Rate:                   {WR*100:.2f}%")
print(f"  Total trades:               {N}")
print(f"  Trades/anio estimados:      {trades_per_year:.0f}")

# Kelly half: f* = (WR - (1-WR)) / (media_win/media_loss)
wins_ret  = combined[combined["is_win"] == 1]["return_raw"].mean()
losses_ret = abs(combined[combined["is_win"] == 0]["return_raw"].mean())
if losses_ret > 0:
    kelly_full = (WR - (1 - WR)) / (wins_ret / losses_ret) if wins_ret > 0 else 0
    kelly_half = kelly_full / 2
    print(f"  Kelly completo:             {kelly_full:.4f} ({kelly_full*100:.2f}%)")
    print(f"  Half-Kelly (14.17%):        {min(kelly_half, 0.1417):.4f} ({min(kelly_half,0.1417)*100:.2f}%)")

# Leverage analysis
for lev in [10, 20]:
    lev_ret = total_ret * lev
    lev_dd  = max_dd * lev
    print(f"  Leverage x{lev}: ret={lev_ret*100:+.1f}% maxDD={lev_dd*100:.1f}%")

# ── H-SAFE-4: Rendimiento por ventana V2 ──────────────────────────
print("\n─" * 65)
print("H-SAFE-4: Metricas V2 por ventana Walk-Forward")
print("─" * 65)
print(f"  {'Ventana':>8} {'N':>5} {'WR':>7} {'Ret%':>8} {'MaxDD%':>8} {'Sharpe':>8} {'Calmar':>8}")
for w in ["W1","W2","W3","W4","W5"]:
    sub = combined[combined["_w"] == w].sort_values("entry_dt")
    if len(sub) < 5:
        continue
    r = sub["return_raw"].fillna(0)
    eq = (1 + r).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    total = eq.iloc[-1] - 1
    sh = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    cal = abs(total / dd) if dd != 0 else 0
    print(f"  {w:>8} {len(sub):>5} {sub['is_win'].mean():>7.3f} {total*100:>+8.2f} {dd*100:>8.2f} {sh:>8.3f} {cal:>8.3f}")

# ── VEREDICTO FINAL H-SAFE ─────────────────────────────────────────
print("\n" + "=" * 65)
print("VEREDICTO H-SAFE")
print("=" * 65)
print(f"  H-SAFE-1: Percentil {pct_rank:.0f}% vs aleatorio  {'→ EDGE REAL' if pct_rank >= 90 else '→ Edge debil'}")
print(f"  H-SAFE-2: IC=[{ci_lo:.3f},{ci_hi:.3f}]  {'→ Excluye 50% (OK)' if ci_lo>0.50 else '→ Incluye 50% (debil)'}")
print(f"  H-SAFE-3: Calmar={calmar:.2f} Sharpe={sharpe:.2f}")
print(f"  H-SAFE-4: ver tabla por ventana arriba")
