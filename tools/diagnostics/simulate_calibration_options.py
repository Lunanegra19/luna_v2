"""
simulate_calibration_options.py
================================
Simula el efecto de 7 estrategias de calibración sobre los 3767 trades OOS reales.
NO modifica ningún código de producción. Usa datos ya existentes para comparar.

Las métricas clave son:
  1. Spearman(prob_corr, is_win)     — discriminación ordinal
  2. Reliability (ECE)               — calibración probabilística
  3. Kelly-weighted return simulado  — lo más importante: ¿cuánto gana el sistema?
     Fracción Kelly = (p_est * b - (1-p_est)) / b  donde b = ratio ganancia/pérdida
     Capped en Half-Kelly = 14.17%
"""
import sys, numpy as np, pandas as pd, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.stats import spearmanr
from scipy.special import logit, expit   # expit = sigmoid

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\predictions')
SEP  = "=" * 76
DASH = "-" * 76

# ── CARGA ─────────────────────────────────────────────────────────────────
dfs = []
for f in sorted(DATA.glob('oos_trades_seed*.parquet')):
    d = pd.read_parquet(f)
    d['_seed'] = int(f.stem.split('seed')[1])
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df['ret100'] = df['return_raw'] * 100
df['_win']   = df['is_win'].astype(float)
print(f"[LOAD] {len(df)} trades | {df['_seed'].nunique()} seeds | W1-W5")

raw = df['xgb_prob'].values
cal = df['xgb_prob_cal'].values
win = df['_win'].values
ret = df['return_raw'].values   # fracción, no porcentaje

# WR base real
base_wr = win.mean() * 100
# Ratio b (ganancia/pérdida) estimado de los datos
wins_ret   = ret[win == 1]
losses_ret = np.abs(ret[win == 0])
b_ratio    = wins_ret.mean() / losses_ret.mean() if losses_ret.mean() > 0 else 1.0
print(f"[BASE] WR={base_wr:.1f}% | b_ratio={b_ratio:.3f} | N={len(df)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────

def kelly_fraction(p_est, b=b_ratio, cap=0.1417):
    """Half-Kelly fraction, capped at 14.17% (politica institucional)."""
    k = (p_est * b - (1 - p_est)) / b
    k = np.clip(k, 0.0, cap)  # nunca negativo ni sobre cap
    return k

def simulate_returns(p_est, actual_ret, cap=0.1417):
    """Simula retorno total usando Kelly fraction basada en p_est."""
    fracs = kelly_fraction(p_est, b=b_ratio, cap=cap)
    # retorno ponderado por fracción: trade_ret_kelly = frac * actual_ret
    kelly_rets = fracs * actual_ret
    # Equity curve multiplicativa
    equity = np.cumprod(1 + kelly_rets)
    total_return_pct = (equity[-1] - 1) * 100
    avg_frac = fracs.mean() * 100
    return total_return_pct, avg_frac, fracs

def ece(p_est, actual_wins, n_bins=10):
    """Expected Calibration Error — menor es mejor."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (p_est >= bins[i]) & (p_est < bins[i+1])
        if mask.sum() < 5: continue
        avg_p = p_est[mask].mean()
        avg_w = actual_wins[mask].mean()
        ece_val += mask.sum() / len(p_est) * abs(avg_p - avg_w)
    return ece_val

def spearman_per_seed(p_est_arr, df_full):
    """Spearman per-seed (honesto, sin pseudoreplicación)."""
    df_full = df_full.copy()
    df_full['_pest'] = p_est_arr
    rhos = []
    for _, g in df_full.groupby('_seed'):
        if g['_pest'].std() < 1e-6 or g['_win'].std() < 1e-6:
            continue
        r, _ = spearmanr(g['_pest'], g['_win'])
        rhos.append(r)
    return np.nanmean(rhos) if rhos else float('nan'), len(rhos)

def evaluate(name, p_est, df_full, actual_ret, actual_win, verbose=False):
    """Evalúa una estrategia de calibración. Devuelve dict de métricas."""
    p_est = np.clip(p_est, 0.01, 0.99)
    rho_g, _ = spearmanr(p_est, actual_win)
    rho_s, n_seeds = spearman_per_seed(p_est, df_full)
    ece_val = ece(p_est, actual_win)
    total_ret, avg_frac, fracs = simulate_returns(p_est, actual_ret)
    # Sharpe simulado (muy aproximado)
    kelly_rets_arr = fracs * actual_ret
    sharpe = kelly_rets_arr.mean() / (kelly_rets_arr.std() + 1e-10) * np.sqrt(8760)
    if verbose:
        print(f"  prob range: [{p_est.min():.3f}, {p_est.max():.3f}]  mean={p_est.mean():.3f}")
    return {
        'name':        name,
        'rho_global':  rho_g,
        'rho_per_seed': rho_s,
        'n_seeds':     n_seeds,
        'ece':         ece_val,
        'kelly_ret%':  total_ret,
        'avg_frac%':   avg_frac,
        'sharpe_approx': sharpe,
    }

results = []

# ─────────────────────────────────────────────────────────────────────────
# OPCIONES (7 estrategias + 2 baselines)
# ─────────────────────────────────────────────────────────────────────────
print(SEP)
print("SIMULANDO 9 ESTRATEGIAS DE CALIBRACION...")
print(SEP)

# ── BASELINE 0: prob_raw (sin calibrador) ─────────────────────────────
r = evaluate("0.RAW (sin calibrador)", raw, df, ret, win)
results.append(r)
print(f"OK 0. RAW")

# ── BASELINE 1: prob_cal actual (calibrador IS, status quo) ─────────────
r = evaluate("1.CAL_IS (status quo)", cal, df, ret, win)
results.append(r)
print(f"OK 1. CAL_IS (status quo)")

# ── OPCION A: Half-Kelly estático (prob fija 0.55 para todos) ───────────
# Equivale a usar una probabilidad constante de WR real + margen pequeño
# La fracción Kelly sería constante = kelly(0.55) para todos
p_static = np.full(len(df), base_wr / 100 + 0.01)  # WR real + 1pp
r = evaluate("A.STATIC (kelly con WR real)", p_static, df, ret, win)
results.append(r)
print(f"OK A. STATIC")

# ── OPCION B: Re-calibración OOS rolling (simulada con W(n-1)) ─────────
# Simulamos calibrando el calibrador isotónico con los trades OOS de cada
# ventana anterior y aplicándolo a la ventana siguiente.
from sklearn.isotonic import IsotonicRegression
p_oos_rolling = cal.copy()
windows_ord = sorted(df['wfb_window'].unique())
for i, win_name in enumerate(windows_ord[1:], 1):
    prev_win = windows_ord[i - 1]
    mask_prev = df['wfb_window'] == prev_win
    mask_curr = df['wfb_window'] == win_name
    if mask_prev.sum() < 20:
        continue
    X_fit = raw[mask_prev]
    y_fit = win[mask_prev]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(X_fit, y_fit)
    p_oos_rolling[mask_curr] = iso.predict(raw[mask_curr])
r = evaluate("B.OOS_ROLLING (re-cal con W(n-1))", p_oos_rolling, df, ret, win)
results.append(r)
print(f"OK B. OOS_ROLLING")

# ── OPCION C: Quantile Ranking → fracción Kelly proporcional ─────────────
# Convierte prob_cal a su quantile dentro de la ventana [0,1] y usa ese
# quantile como intensidad del sizing (prob_scaled = WR_real + quantile * rango)
p_quantile = cal.copy()
for w_name in windows_ord:
    mask = df['wfb_window'] == w_name
    if mask.sum() < 5: continue
    p_w = cal[mask]
    wr_w = win[mask].mean()
    # Quantile relativo dentro de la ventana
    ranks = pd.Series(p_w).rank(pct=True).values
    # Mapear ranks → [WR_base - 5pp, WR_base + 5pp] (rango modesto)
    p_quantile[mask] = wr_w - 0.05 + ranks * 0.10
r = evaluate("C.QUANTILE_RANK (ranking relativo)", p_quantile, df, ret, win)
results.append(r)
print(f"OK C. QUANTILE_RANK")

# ── OPCION D: Temperature Scaling ──────────────────────────────────────
# Busca el T óptimo minimizando ECE en los datos OOS disponibles
# prob_corrected = sigmoid(logit(prob_cal) / T)
from scipy.optimize import minimize_scalar
def ece_for_T(T):
    logit_cal = logit(np.clip(cal, 0.01, 0.99))
    p_t = expit(logit_cal / T)
    return ece(p_t, win)

res_T = minimize_scalar(ece_for_T, bounds=(0.1, 5.0), method='bounded')
T_opt = res_T.x
logit_cal = logit(np.clip(cal, 0.01, 0.99))
p_temp = expit(logit_cal / T_opt)
r = evaluate(f"D.TEMP_SCALING (T={T_opt:.2f})", p_temp, df, ret, win)
results.append(r)
print(f"OK D. TEMPERATURE_SCALING (T_opt={T_opt:.3f})")

# ── OPCION E: Platt Scaling post-hoc (LR sobre prob_cal → WR) ──────────
# Fit logistic regression: WR ~ prob_cal (usando todos los datos OOS disponibles)
# En producción: se fittea con W(n-1) trades (similar a B pero más simple)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0)
lr.fit(cal.reshape(-1, 1), win)
p_platt = lr.predict_proba(cal.reshape(-1, 1))[:, 1]
r = evaluate("E.PLATT_POSTHOC (LR sobre prob_cal)", p_platt, df, ret, win)
results.append(r)
print(f"OK E. PLATT_POSTHOC")

# ── OPCION F: Clamp superior — truncar prob > 0.70 a WR real ─────────────
# Idea simple: si el calibrador no discrimina por encima de 0.70, no lo usemos
# para subir el sizing. Por encima de 0.70 → usar WR real como estimacion.
p_clamp = cal.copy()
threshold_clamp = 0.70
wr_high = win[cal > threshold_clamp].mean() if (cal > threshold_clamp).any() else base_wr / 100
p_clamp[cal > threshold_clamp] = wr_high
r = evaluate(f"F.CLAMP_HIGH (prob>0.70→WR real)", p_clamp, df, ret, win)
results.append(r)
print(f"OK F. CLAMP_HIGH (WR real en zona alta={wr_high:.3f})")

# ── OPCION G: Bayesian Shrinkage (shrink hacia WR base) ──────────────────
# En lugar de usar prob_cal directamente, hacer shrink hacia la WR base
# p_shrunk = alpha * prob_cal + (1-alpha) * WR_base
# Alpha óptimo minimizando ECE
wr_base = base_wr / 100
def ece_for_alpha(alpha):
    p_sh = alpha * np.clip(cal, 0.01, 0.99) + (1 - alpha) * wr_base
    return ece(p_sh, win)
res_a = minimize_scalar(ece_for_alpha, bounds=(0.0, 1.0), method='bounded')
alpha_opt = res_a.x
p_shrunk = alpha_opt * np.clip(cal, 0.01, 0.99) + (1 - alpha_opt) * wr_base
r = evaluate(f"G.BAYES_SHRINK (alpha={alpha_opt:.2f})", p_shrunk, df, ret, win)
results.append(r)
print(f"OK G. BAYES_SHRINK (alpha_opt={alpha_opt:.3f})")

# ─────────────────────────────────────────────────────────────────────────
# RESULTADOS — TABLA COMPARATIVA
# ─────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("TABLA COMPARATIVA — 9 ESTRATEGIAS")
print(SEP)
print()
print(f"{'Estrategia':<42} {'rho_seed':>9} {'ECE':>7} {'Kelly_Ret%':>11} {'AvgFrac%':>9} {'Sharpe':>8}")
print(DASH)

rdf = pd.DataFrame(results)
# Ordenar por Kelly_Ret% (lo que más importa)
rdf_sorted = rdf.sort_values('kelly_ret%', ascending=False)

best_kelly = rdf_sorted['kelly_ret%'].max()
for _, row in rdf_sorted.iterrows():
    marker = " ← MEJOR" if abs(row['kelly_ret%'] - best_kelly) < 0.1 else ""
    print(f"  {row['name']:<42} {row['rho_per_seed']:>+9.4f} {row['ece']:>7.4f} {row['kelly_ret%']:>+11.2f}% {row['avg_frac%']:>9.3f}% {row['sharpe_approx']:>+8.3f}{marker}")

print()
print(SEP)
print("ANALISIS POR VENTANA — efecto de cada opcion en cada window")
print(SEP)
print()

options_to_show = [
    ("1.CAL_IS",        cal),
    ("B.OOS_ROLLING",   p_oos_rolling),
    ("D.TEMP_SCAL",     p_temp),
    ("F.CLAMP_HIGH",    p_clamp),
    ("G.SHRINK",        p_shrunk),
]
header_w = f"  {'Opcion':<18}"
for w in windows_ord:
    header_w += f" {w:>10}"
print(header_w + f" {'TOTAL':>10}")
print("  " + DASH)

for opt_name, p_arr in options_to_show:
    row_str = f"  {opt_name:<18}"
    total_ret_all = 0
    for w_name in windows_ord:
        mask = df['wfb_window'] == w_name
        if mask.sum() < 5:
            row_str += f" {'N/A':>10}"
            continue
        tr, _, _ = simulate_returns(p_arr[mask], ret[mask])
        total_ret_all += tr
        row_str += f" {tr:>+10.2f}%"
    row_str += f" {total_ret_all:>+10.2f}%"
    print(row_str)

print()
print(SEP)
print("DIAGNOSTICO FINAL — ¿Qué estrategia implementar?")
print(SEP)
print()
rdf_sorted2 = rdf.sort_values('kelly_ret%', ascending=False)
winner = rdf_sorted2.iloc[0]['name']
runner_up = rdf_sorted2.iloc[1]['name']
print(f"  GANADOR (Kelly Return):     {winner}")
print(f"  SEGUNDO (Kelly Return):     {runner_up}")
print()
status_quo = rdf[rdf['name'].str.startswith('1.CAL')].iloc[0]
print(f"  Status quo (CAL_IS) Kelly Return: {status_quo['kelly_ret%']:+.2f}%")
print()
for _, row in rdf_sorted2.iterrows():
    delta = row['kelly_ret%'] - status_quo['kelly_ret%']
    print(f"  {row['name']:<42} delta vs status quo: {delta:>+8.2f}%")
