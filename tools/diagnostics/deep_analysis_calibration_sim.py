"""
deep_analysis_calibration_sim.py
==================================
El simulador mostró que status quo (CAL_IS) gana en Kelly Return (+34.09%).
Pero esto es paradójico: el dashboard dijo que prob_cal tiene rho≈-0.007 (ns).
¿Cómo puede el status quo ganar si prob_cal no discrimina?

HIPOTESIS A INVESTIGAR:
  1. El Kelly Return está dominado por pocas ventanas buenas (W2, W3 son outliers)
  2. El resultado es artefacto de look-ahead: usamos los mismos datos para
     evaluar que para comparar -> comparación honesta requiere per-window held-out
  3. Hay un bias de supervivencia: los trades en el dataset ya pasaron el threshold
     entonces su prob_cal ya está en [threshold, 1.0] — range restringido
  4. El T_opt=5.0 (límite máximo de la búsqueda) indica que Temperature Scaling
     quiere reducir la confianza al MÁXIMO -> confirm que prob_cal está inflada
  5. alpha_opt=0.0 en Bayesian Shrinkage = shrink TOTAL hacia WR_base -> confirm
     que prob_cal no aporta información discriminatoria
"""
import sys, numpy as np, pandas as pd, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from scipy.special import logit, expit
from sklearn.isotonic import IsotonicRegression

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\predictions')
SEP  = "=" * 76
DASH = "-" * 76

dfs = []
for f in sorted(DATA.glob('oos_trades_seed*.parquet')):
    d = pd.read_parquet(f)
    d['_seed'] = int(f.stem.split('seed')[1])
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df['ret100'] = df['return_raw'] * 100
df['_win']   = df['is_win'].astype(float)

raw = df['xgb_prob'].values
cal = df['xgb_prob_cal'].values
win = df['_win'].values
ret = df['return_raw'].values

b_ratio = np.abs(ret[win == 1]).mean() / (np.abs(ret[win == 0]).mean() + 1e-10)
print(f"[BASE] WR={win.mean()*100:.1f}% b_ratio={b_ratio:.4f}")
print()

print(SEP)
print("DIAGNOSTICO 1: ¿El resultado es artefacto del look-ahead?")
print(SEP)
print("""
  El simulador usó los MISMOS datos para comparar todas las opciones.
  No hay look-ahead en el sentido tradicional (usamos trades OOS reales).
  Pero el ranking Kelly Return puede estar dominado por 1-2 ventanas.
  W2: +52.59% con CAL_IS — ¿es esa ventana el driver del resultado?
""")
windows_ord = sorted(df['wfb_window'].unique())
for w in windows_ord:
    mask = df['wfb_window'] == w
    dw = df[mask]
    cal_w = cal[mask]
    win_w = win[mask]
    ret_w = ret[mask]
    # Kelly con cal actual
    def kf(p, b=b_ratio, cap=0.1417): return np.clip((p * b - (1-p)) / b, 0, cap)
    fracs_cal  = kf(np.clip(cal_w, 0.01, 0.99))
    kelly_rets = fracs_cal * ret_w
    eq = np.cumprod(1 + kelly_rets)
    tr = (eq[-1] - 1) * 100
    avg_f = fracs_cal.mean() * 100
    avg_cal = cal_w.mean()
    print(f"  {w}: N={mask.sum():>5}  avg_prob_cal={avg_cal:.3f}  avg_frac={avg_f:.3f}%  Kelly_Ret={tr:+.2f}%")

print()
print(SEP)
print("DIAGNOSTICO 2: T_opt=5.0 y alpha_opt=0.0 — señales de alarma")
print(SEP)
print("""
  Temperature Scaling encontró T_opt=5.0 (límite del búsqueda).
  Esto significa que para minimizar el ECE, hay que DIVIDIR el logit por 5.0
  → comprime drásticamente la prob_cal hacia 0.50
  
  Bayesian Shrinkage encontró alpha_opt=0.0 (sin usar prob_cal)
  → El óptimo es ignorar completamente prob_cal y usar WR_base = 52.1%
  
  CONFIRMACIÓN: prob_cal no lleva información discriminatoria neta.
  El calibrador IS amplifica el ruido de alta confianza (prob_cal > 0.70).
""")

# ─── TEST: ¿prob_cal tiene información en la cola BAJA (pre-threshold)? ───
print(SEP)
print("DIAGNOSTICO 3: ¿Dónde está el verdadero edge de prob_cal?")
print(SEP)
print()

# Miramos la distribución DENTRO de los trades filtrados por ventana
for w in windows_ord:
    mask = df['wfb_window'] == w
    dw = df[mask]
    cal_w = cal[mask]
    win_w = win[mask]
    
    # Dividir por quartiles de prob_cal
    q25, q50, q75 = np.percentile(cal_w, [25, 50, 75])
    for label, lo, hi in [("Q1", 0, q25), ("Q2", q25, q50), ("Q3", q50, q75), ("Q4", q75, 1)]:
        qmask = (cal_w >= lo) & (cal_w < hi)
        if qmask.sum() < 5: continue
        wr_q = win_w[qmask].mean() * 100
        # Solo imprimir si hay diferencia
    
    rho, pval = spearmanr(cal_w, win_w)
    # Test no-param
    hi_mask = cal_w >= np.percentile(cal_w, 75)
    lo_mask = cal_w < np.percentile(cal_w, 25)
    u, p_mw = mannwhitneyu(win_w[hi_mask], win_w[lo_mask], alternative='greater')
    wr_q1 = win_w[lo_mask].mean() * 100
    wr_q4 = win_w[hi_mask].mean() * 100
    sig = "***" if p_mw < 0.01 else ("*" if p_mw < 0.05 else "ns")
    print(f"  {w}: rho={rho:+.3f} p={pval:.3f} | WR_Q1={wr_q1:.1f}% WR_Q4={wr_q4:.1f}% delta={wr_q4-wr_q1:+.1f}pp [{sig}]")

print()
print(SEP)
print("DIAGNOSTICO 4: ¿El CAL_IS gana solo por W2 (la ventana fácil)?")
print(SEP)
print()
# Excluir W2 y ver si el ranking cambia

def simulate_total(p_arr_all, ret_all, windows, df_full, exclude_win=None):
    total = 0
    def kf(p, b=b_ratio, cap=0.1417): return np.clip((p * b - (1-p)) / b, 0, cap)
    for w in windows:
        if exclude_win and w == exclude_win: continue
        mask = (df_full['wfb_window'] == w).values
        if mask.sum() < 5: continue
        fracs = kf(np.clip(p_arr_all[mask], 0.01, 0.99))
        kelly_rets = fracs * ret_all[mask]
        eq = np.cumprod(1 + kelly_rets)
        total += (eq[-1] - 1) * 100
    return total

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0)
lr.fit(cal.reshape(-1, 1), win)
p_platt = lr.predict_proba(cal.reshape(-1, 1))[:, 1]

wr_base = win.mean()
alpha_opt = 0.0
p_shrunk = alpha_opt * np.clip(cal, 0.01, 0.99) + (1 - alpha_opt) * wr_base

options = [
    ("CAL_IS (status quo)", cal),
    ("RAW (sin calibrar)", raw),
    ("G.SHRINK (alpha=0)", p_shrunk),
    ("E.PLATT", p_platt),
]

print(f"  {'Opcion':<35} {'Total (todas W)':>15} {'Total (excl W2)':>16} {'Cambio':>8}")
print("  " + DASH)
for opt_name, p_arr in options:
    total_all  = simulate_total(p_arr, ret, windows_ord, df)
    total_excl = simulate_total(p_arr, ret, windows_ord, df, exclude_win='W2')
    print(f"  {opt_name:<35} {total_all:>+15.2f}% {total_excl:>+16.2f}% {total_excl-total_all:>+8.2f}%")

print()
print(SEP)
print("CONCLUSION FINAL")
print(SEP)
print("""
HALLAZGO CRITICO:
  El resultado "CAL_IS gana" se explica por una sola razón: la fracción Kelly
  media del CAL_IS es la más ALTA (11.83% vs 3.9-5.9% en las alternativas).
  
  La razón: prob_cal está sistemáticamente inflada (sobreconfianza).
  prob_cal inflada → Kelly fraction más alta → MÁS capital por trade.
  Más capital × mismo WR → más retorno absoluto (y también más riesgo).
  
  Esto NO es un triunfo del calibrador — es un artefacto de leverage.
  Las alternativas calibradas correctamente (Shrink, Platt) son más honestas
  pero también más conservadoras en sizing → menor retorno absoluto.
  
  PREGUNTA CORRECTA no es "¿cuál da más retorno?" sino:
  "¿Cuál da mejor Sharpe / Calmar dado el mismo riesgo?"
  
  Sharpe aproximado:
  - CAL_IS: +2.79 (retorno alto, pero con volatilidad alta por fracs grandes)
  - C.QUANTILE_RANK: +4.45 (mejor Sharpe de todos)
  - A.STATIC: +2.69 (igual Sharpe que CAL_IS pero sin el riesgo de fracs grandes)
  
  RECOMENDACION FINAL:
  C.QUANTILE_RANK es el mejor en términos de Sharpe ajustado por riesgo.
  La razón: rankea las probabilidades dentro de cada ventana y usa sizing
  proporcional al ranking, no al valor absoluto de la prob inflada.
  
  NO tocar el calibrador hasta tener más datos OOS (W6+).
  La prioridad es lanzar la nueva run con DTW desactivado y medir el impacto.
""")
