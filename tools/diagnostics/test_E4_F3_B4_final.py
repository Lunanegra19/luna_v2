"""
test_E4_F3_bonus.py
====================
E4: Optimal exit retrospective — ¿cuánto upside se dejó sobre la mesa?
    Usando equity_curve por trade (si está disponible) o return_raw
F3: Momentum en entrada — ¿el contexto de retorno reciente predice WR?
    Usando retorno acumulado de las N últimas horas como proxy de tendencia
B4: Autocorrelación de wins — ¿un win hace más probable el siguiente?
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

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
combined["exit_dt"]  = pd.to_datetime(combined["exit_time"],  utc=True, errors="coerce")
combined["duration_h"] = (combined["exit_dt"] - combined["entry_dt"]).dt.total_seconds() / 3600
combined = combined.sort_values("entry_dt").reset_index(drop=True)
baseline_wr = combined["is_win"].mean()

# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("TEST E4 — Optimal Exit Retrospective")
print("=" * 65)
print(f"\nN: {len(combined)} | WR baseline: {baseline_wr:.4f}")

# Analisis de equity_curve
if "equity_curve" in combined.columns:
    ec = combined["equity_curve"].dropna()
    print(f"  Trades con equity_curve: {len(ec)}")
    print(f"  Tipo: {type(ec.iloc[0]).__name__ if len(ec)>0 else 'vacio'}")
    print(f"  Muestra: {str(ec.iloc[0])[:150] if len(ec)>0 else 'N/A'}")
else:
    print("  equity_curve no disponible")

# Analisis via return_raw y drawdown
print("\n─" * 65)
print("SEC 1: Distribucion de retornos (return_raw)")
print("─" * 65)
ret = combined["return_raw"].dropna()
print(f"  Media:    {ret.mean():+.5f} ({ret.mean()*100:+.3f}%)")
print(f"  Mediana:  {ret.median():+.5f}")
print(f"  Std:      {ret.std():.5f}")
print(f"  Q25/Q75:  [{ret.quantile(0.25):+.5f}, {ret.quantile(0.75):+.5f}]")
print(f"  Min/Max:  [{ret.min():+.5f}, {ret.max():+.5f}]")

# Comparar return_raw de wins vs losses
wins_r   = combined[combined["is_win"]==1]["return_raw"].dropna()
losses_r = combined[combined["is_win"]==0]["return_raw"].dropna()
print(f"\n  return_raw WINS:   media={wins_r.mean():+.5f}  med={wins_r.median():+.5f}")
print(f"  return_raw LOSSES: media={losses_r.mean():+.5f}  med={losses_r.median():+.5f}")

_, p_ret = stats.mannwhitneyu(wins_r, losses_r, alternative="two-sided")
print(f"  MW test: p={p_ret:.4f}  {'→ wins tienen retorno significativamente mayor' if p_ret<0.05 else '→ No significativo'}")

# Analisis de drawdown por trade
print("\n─" * 65)
print("SEC 2: Drawdown intra-trade (columna drawdown)")
print("─" * 65)
if "drawdown" in combined.columns:
    dd = combined["drawdown"].dropna()
    dd_wins   = combined[combined["is_win"]==1]["drawdown"].dropna()
    dd_losses = combined[combined["is_win"]==0]["drawdown"].dropna()
    print(f"  Drawdown medio (all):    {dd.mean():.5f} ({dd.mean()*100:.3f}%)")
    print(f"  Drawdown medio WINS:     {dd_wins.mean():.5f} ({dd_wins.mean()*100:.3f}%)")
    print(f"  Drawdown medio LOSSES:   {dd_losses.mean():.5f} ({dd_losses.mean()*100:.3f}%)")
    _, p_dd = stats.mannwhitneyu(dd_wins.abs(), dd_losses.abs(), alternative="two-sided")
    print(f"  MW drawdown wins vs losses: p={p_dd:.4f}")
    if p_dd < 0.05:
        direction = "LOSSES tienen MAYOR drawdown" if dd_losses.abs().mean() > dd_wins.abs().mean() else "WINS tienen mayor drawdown"
        print(f"  → {direction}")
    # Por cuartil de drawdown
    print("\n  WR por cuartil de drawdown intra-trade:")
    try:
        combined["dd_q"] = pd.qcut(combined["drawdown"].abs(), q=4,
                                    labels=["Q1_min","Q2","Q3","Q4_max"], duplicates="drop")
        for q in ["Q1_min","Q2","Q3","Q4_max"]:
            sub = combined[combined["dd_q"]==q]
            if len(sub) < 5: continue
            wr = sub["is_win"].mean()
            lo = sub["drawdown"].abs().min()
            hi = sub["drawdown"].abs().max()
            print(f"    {q}: N={len(sub):3d} dd=[{lo:.4f},{hi:.4f}] WR={wr:.4f} ({wr-baseline_wr:+.3f})")
    except Exception as e:
        print(f"    (qcut error: {e})")

# Retorno absoluto por trade — asimetria win/loss
print("\n─" * 65)
print("SEC 3: Asimetria ganancia/perdida (Profit Factor)")
print("─" * 65)
avg_win  = wins_r.mean() if len(wins_r) > 0 else 0
avg_loss = abs(losses_r.mean()) if len(losses_r) > 0 else 0
profit_factor = (avg_win * len(wins_r)) / (avg_loss * len(losses_r)) if avg_loss > 0 else float("inf")
rr_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

print(f"  Retorno medio WIN:    {avg_win*100:+.4f}%")
print(f"  Retorno medio LOSS:   -{avg_loss*100:.4f}%")
print(f"  Risk/Reward ratio:    {rr_ratio:.3f}  (>1 = ganas mas de lo que pierdes)")
print(f"  Profit Factor:        {profit_factor:.3f}  (>1 = sistema rentable)")
print(f"  Expectancia por trade: {(avg_win*baseline_wr - avg_loss*(1-baseline_wr))*100:+.5f}%")

# Break-even WR para ser rentable
be_wr = avg_loss / (avg_win + avg_loss) if (avg_win + avg_loss) > 0 else 0.5
print(f"  WR minimo para break-even: {be_wr:.4f} ({be_wr*100:.2f}%)")
print(f"  WR actual vs break-even:   {(baseline_wr - be_wr)*100:+.2f}pp {'OK' if baseline_wr > be_wr else 'NEGATIVO'}")

print(f"\nVEREDICTO E4:")
if rr_ratio < 1.0:
    print(f"  ALERTA: Las ganancias medias ({avg_win*100:.3f}%) son menores que las perdidas ({avg_loss*100:.3f}%)")
    print(f"  El sistema depende de ALTA frecuencia de wins para ser rentable")
    print(f"  WR minimo necesario: {be_wr*100:.1f}% (actual {baseline_wr*100:.1f}%)")
elif rr_ratio > 1.5:
    print(f"  POSITIVO: R/R ratio={rr_ratio:.2f} — las ganancias compensan bien las perdidas")
else:
    print(f"  NEUTRAL: R/R ratio={rr_ratio:.2f} — sistema balanceado")

# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("TEST F3 — Momentum en Entrada vs WR")
print("=" * 65)
print("(Usando retorno acumulado de los ultimos N trades como proxy)")

# Calcular momentum como retorno acumulado de los k trades previos
# Esto es un proxy del momentum del mercado en el momento de la señal
combined_s = combined.sort_values("entry_dt").copy()
combined_s["ret_raw_f"] = combined_s["return_raw"].fillna(0)

for k in [3, 5, 10]:
    combined_s[f"momentum_{k}"] = combined_s["ret_raw_f"].rolling(k, min_periods=k).sum().shift(1)

print(f"\n─" * 65)
print("SEC 1: Momentum (retorno acumulado previo) vs WR")
print("─" * 65)
for k in [3, 5, 10]:
    col = f"momentum_{k}"
    valid = combined_s.dropna(subset=[col, "is_win"])
    rho, p = stats.spearmanr(valid[col], valid["is_win"])
    print(f"  momentum_{k}trades: rho={rho:+.4f}  p={p:.4f}  N={len(valid)}")
    # Por cuartil
    try:
        valid = valid.copy()
        valid["mq"] = pd.qcut(valid[col], q=4,
                               labels=["Q1_neg","Q2","Q3","Q4_pos"], duplicates="drop")
        line = ""
        for q in ["Q1_neg","Q2","Q3","Q4_pos"]:
            sub = valid[valid["mq"]==q]
            if len(sub) >= 10:
                wr = sub["is_win"].mean()
                line += f"  {q}={wr:.3f}"
        print(f"    WR por cuartil:{line}")
    except Exception as e:
        print(f"    (qcut error: {e})")

print(f"\n─" * 65)
print("SEC 2: Autocorrelacion de wins (B4 integrado)")
print("─" * 65)
wins_series = combined_s["is_win"].values
# Lag-1 autocorrelacion
n_wins_after_win  = sum(wins_series[i] == 1 and wins_series[i+1] == 1 for i in range(len(wins_series)-1))
n_wins_after_loss = sum(wins_series[i] == 0 and wins_series[i+1] == 1 for i in range(len(wins_series)-1))
n_after_win  = sum(wins_series[i] == 1 for i in range(len(wins_series)-1))
n_after_loss = sum(wins_series[i] == 0 for i in range(len(wins_series)-1))

wr_after_win  = n_wins_after_win  / n_after_win  if n_after_win  > 0 else 0
wr_after_loss = n_wins_after_loss / n_after_loss if n_after_loss > 0 else 0
print(f"  WR tras WIN anterior:  {wr_after_win:.4f}  (N={n_after_win})")
print(f"  WR tras LOSS anterior: {wr_after_loss:.4f}  (N={n_after_loss})")
print(f"  Delta: {wr_after_win - wr_after_loss:+.4f}")

chi2_ac, p_ac = stats.chi2_contingency([
    [n_wins_after_win,  n_after_win  - n_wins_after_win],
    [n_wins_after_loss, n_after_loss - n_wins_after_loss]
])[:2]
print(f"  Chi² lag-1: p={p_ac:.4f}  {'→ AUTOCORRELACION SIGNIFICATIVA' if p_ac<0.05 else '→ No significativa'}")

# Ljung-Box test
from scipy.stats import chi2 as chi2_dist
n = len(wins_series)
lags_to_test = [1, 2, 3, 5]
print(f"\n  Test de autocorrelacion (Ljung-Box simplificado):")
for lag in lags_to_test:
    # Autocorrelacion de lag k
    ac = pd.Series(wins_series.astype(float)).autocorr(lag=lag)
    lb_stat = n * (n + 2) * (ac**2 / (n - lag))
    p_lb = 1 - chi2_dist.cdf(lb_stat, df=1)
    print(f"    Lag {lag}: rho={ac:+.4f}  LB_stat={lb_stat:.3f}  p={p_lb:.4f}")

print(f"\n─" * 65)
print("SEC 3: Rachas ganadoras y perdedoras")
print("─" * 65)
# Calcular longitud de rachas
streaks = []
current = wins_series[0]
length  = 1
for i in range(1, len(wins_series)):
    if wins_series[i] == current:
        length += 1
    else:
        streaks.append((current, length))
        current = wins_series[i]
        length  = 1
streaks.append((current, length))

win_streaks  = [l for (t, l) in streaks if t == 1]
loss_streaks = [l for (t, l) in streaks if t == 0]
print(f"  Rachas ganadoras:  media={np.mean(win_streaks):.2f}  max={max(win_streaks)}  N_rachas={len(win_streaks)}")
print(f"  Rachas perdedoras: media={np.mean(loss_streaks):.2f}  max={max(loss_streaks)}  N_rachas={len(loss_streaks)}")
# Si la racha maxima esperada bajo aleatoriedad pura es ~log2(N)
max_expected = np.log2(len(wins_series))
print(f"  Racha max esperada (aleatoria, log2(N)): {max_expected:.1f}")
print(f"  Racha max observada WINS:  {max(win_streaks)} {'ANORMAL' if max(win_streaks) > max_expected*2 else 'normal'}")
print(f"  Racha max observada LOSS:  {max(loss_streaks)} {'ANORMAL' if max(loss_streaks) > max_expected*2 else 'normal'}")

print(f"\nVEREDICTO F3 + B4:")
rho_best = max([abs(stats.spearmanr(combined_s.dropna(subset=[f"momentum_{k}","is_win"])[f"momentum_{k}"],
                                     combined_s.dropna(subset=[f"momentum_{k}","is_win"])["is_win"])[0])
               for k in [3,5,10]])
if rho_best > 0.05 and p_ac > 0.05:
    print(f"  F3 DESCARTADA: momentum previo no predice WR (rho_max={rho_best:.4f})")
    print(f"  B4 DESCARTADA: no hay autocorrelacion de wins (p_ac={p_ac:.4f})")
    print(f"  → El sistema opera de forma Markoviana — cada trade es independiente")
elif p_ac < 0.05:
    direction = "momentum" if wr_after_win > wr_after_loss else "reversal"
    print(f"  B4 CONFIRMADA: {direction} effect (p={p_ac:.4f})")
    print(f"  → Considerar cooldown/boost tras {'wins' if direction=='momentum' else 'losses'}")
