"""
h5_overfitting_test.py
======================
Test riguroso de overfitting para H5 (Kelly x Rolling Sharpe gate).
Descubrimos rho=0.42 en W4. ¿Es real o es overfitting a W4?

Tests:
  T1: Validacion cruzada — aplicar la misma regla en W3 (OOS para H5)
  T2: Autocorrelacion de retornos W4 (Ljung-Box) — si los retornos son autocorrelados,
      el roll_sharpe es trivialmente predictivo, no hay señal real
  T3: Permutation test — si el rho=0.42 desaparece al permutar, es señal real
      Si persiste → es un artefacto estructural (autocorrelacion)
  T4: Estabilidad del umbral — el roll_SR=0.0 fue elegido DESPUES de ver los datos.
      ¿Qué habria pasado si hubieramos elegido 0.0 a priori sin ver W4?
  T5: Coste en W3 — la misma regla en W3 ¿destruye retorno o lo mejora?
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\reports\wfb')
SEP = "=" * 75
WINDOW_ROLL = 10    # parametro descubierto en W4
UMBRAL_SR   = 0.0  # umbral descubierto en W4

def load_window(window_id):
    dfs = []
    for f in sorted(DATA.glob(f'oos_trades_{window_id}_seed*.parquet')):
        seed = int(f.stem.split('_seed')[1])
        df = pd.read_parquet(f)
        df['seed'] = seed
        dfs.append(df)
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()

def compute_roll_sharpe_and_next(df, window=10):
    """Calcula roll_sharpe en t y retorno en t+1 para correlacion causal."""
    df = df.sort_index().copy()
    df['ret100'] = df['return_pct'] * 100
    df['roll_mean'] = df['ret100'].rolling(window, min_periods=5).mean()
    df['roll_std']  = df['ret100'].rolling(window, min_periods=5).std()
    df['roll_sr']   = df['roll_mean'] / df['roll_std'].clip(lower=1e-8)
    df['next_ret']  = df['ret100'].shift(-1)  # retorno SIGUIENTE trade (causal)
    return df

def simulate_gate(df, umbral=0.0, window=10):
    """Simula aplicar Kelly=0 cuando roll_sr < umbral."""
    df = compute_roll_sharpe_and_next(df, window)
    # Permitir cuando: roll_sr >= umbral O roll_sr es NaN (primeros trades sin historia)
    mask = (df['roll_sr'] >= umbral) | df['roll_sr'].isna()
    ret_sin_gate  = df['ret100'].sum()
    ret_con_gate  = df.loc[mask, 'ret100'].sum()
    n_bloq        = (~mask).sum()
    wr_con_gate   = df.loc[mask, 'is_win'].mean() * 100 if mask.sum() > 0 else 0
    return {
        'ret_sin': ret_sin_gate,
        'ret_con': ret_con_gate,
        'mejora':  ret_con_gate - ret_sin_gate,
        'n_bloq':  n_bloq,
        'n_perm':  mask.sum(),
        'wr_con':  wr_con_gate
    }

def permutation_rho(series_x, series_y, n_perm=2000, seed=42):
    """Permutation test: ¿el rho observado es > lo esperado por azar?"""
    rng = np.random.default_rng(seed)
    obs_rho, _ = stats.spearmanr(series_x, series_y)
    perm_rhos = []
    arr_x = series_x.values
    arr_y = series_y.values
    for _ in range(n_perm):
        shuffled = rng.permutation(arr_x)
        r, _ = stats.spearmanr(shuffled, arr_y)
        perm_rhos.append(r)
    p_perm = (np.abs(perm_rhos) >= np.abs(obs_rho)).mean()
    return obs_rho, p_perm, np.array(perm_rhos)

# ===========================
# CARGAR DATOS
# ===========================
df_w3 = load_window('W3')
df_w4 = load_window('W4')
df_w1 = load_window('W1')

print(f"Datos cargados: W1={len(df_w1)} trades | W3={len(df_w3)} trades | W4={len(df_w4)} trades")

# ===========================
# T1: Validacion cruzada en W3 (OOS para H5)
# ===========================
print(f"\n{SEP}")
print("T1: VALIDACION CRUZADA — H5 aplicado a W3 (OOS para la hipotesis)")
print(f"    Params descubiertos en W4: window={WINDOW_ROLL}, umbral_SR={UMBRAL_SR}")
print(SEP)

res_w4 = simulate_gate(df_w4, UMBRAL_SR, WINDOW_ROLL)
res_w3 = simulate_gate(df_w3, UMBRAL_SR, WINDOW_ROLL)
res_w1 = simulate_gate(df_w1, UMBRAL_SR, WINDOW_ROLL) if not df_w1.empty else None

print(f"  W4 (IN-SAMPLE para H5):  RetSin={res_w4['ret_sin']:+.3f}% | RetCon={res_w4['ret_con']:+.3f}% | Mejora={res_w4['mejora']:+.3f}% | Bloqueados={res_w4['n_bloq']}/{len(df_w4)}")
print(f"  W3 (OUT-OF-SAMPLE H5):   RetSin={res_w3['ret_sin']:+.3f}% | RetCon={res_w3['ret_con']:+.3f}% | Mejora={res_w3['mejora']:+.3f}% | Bloqueados={res_w3['n_bloq']}/{len(df_w3)}")
if res_w1:
    print(f"  W1 (OUT-OF-SAMPLE H5):   RetSin={res_w1['ret_sin']:+.3f}% | RetCon={res_w1['ret_con']:+.3f}% | Mejora={res_w1['mejora']:+.3f}% | Bloqueados={res_w1['n_bloq']}/{len(df_w1)}")

if res_w3['mejora'] > 0:
    print(f"\n  CONCLUSION T1: H5 MEJORA W3 tambien ({res_w3['mejora']:+.3f}%) — señal potencialmente generica")
elif res_w3['mejora'] > -1.0:
    print(f"\n  CONCLUSION T1: H5 NEUTRAL en W3 ({res_w3['mejora']:+.3f}%) — posible sobreajuste a W4")
else:
    print(f"\n  CONCLUSION T1: H5 DESTRUYE W3 ({res_w3['mejora']:+.3f}%) — OVERFITTING CONFIRMADO a W4")

# ===========================
# T2: Autocorrelacion de retornos en W4 (Ljung-Box)
# ===========================
print(f"\n{SEP}")
print("T2: AUTOCORRELACION — retornos W4 son autocorrelados?")
print(f"    Si son autocorrelados, roll_sharpe es trivialmente predictivo")
print(SEP)

from statsmodels.stats.diagnostic import acorr_ljungbox

rets_w4 = df_w4['return_pct'].values
rets_w3 = df_w3['return_pct'].values

for name, rets in [('W4', rets_w4), ('W3', rets_w3)]:
    try:
        lb_result = acorr_ljungbox(rets, lags=[1, 2, 5, 10], return_df=True)
        print(f"\n  {name} — Ljung-Box autocorrelacion:")
        for lag, row in lb_result.iterrows():
            sig = "(*)" if row['lb_pvalue'] < 0.05 else "   "
            print(f"    Lag {lag:>2}: LB_stat={row['lb_stat']:.3f}, p={row['lb_pvalue']:.4f} {sig}")

        # Autocorrelacion lag-1 simple
        ac1 = pd.Series(rets).autocorr(lag=1)
        print(f"    Autocorr lag-1: {ac1:.4f}")
        if lb_result['lb_pvalue'].min() < 0.05:
            print(f"    -> Autocorrelacion SIGNIFICATIVA — roll_sharpe es parcialmente trivial")
        else:
            print(f"    -> Sin autocorrelacion significativa — roll_sharpe captura algo REAL")
    except Exception as e:
        print(f"  ERROR Ljung-Box {name}: {e}")

# ===========================
# T3: Permutation test del rho=0.42
# ===========================
print(f"\n{SEP}")
print("T3: PERMUTATION TEST — el rho=0.42 en W4 es real o artefacto?")
print(f"    2000 permutaciones de roll_sharpe manteniendo retornos fijos")
print(SEP)

df_w4_roll = compute_roll_sharpe_and_next(df_w4, WINDOW_ROLL)
df_w3_roll = compute_roll_sharpe_and_next(df_w3, WINDOW_ROLL)

for name, df_roll in [('W4 (descubrimiento)', df_w4_roll), ('W3 (validacion)', df_w3_roll)]:
    valid = df_roll.dropna(subset=['roll_sr', 'next_ret'])
    if len(valid) < 10:
        print(f"  {name}: N insuficiente")
        continue

    obs_rho, p_perm, perm_dist = permutation_rho(valid['roll_sr'], valid['next_ret'], n_perm=2000)
    pct95 = np.percentile(np.abs(perm_dist), 95)

    print(f"\n  {name} (n={len(valid)}):")
    print(f"    rho_observado = {obs_rho:.4f}")
    print(f"    p_permutacion = {p_perm:.4f}")
    print(f"    p95 distribucion nula = |rho| <= {pct95:.4f}")
    if p_perm < 0.05:
        print(f"    -> rho SIGNIFICATIVO bajo permutacion — señal REAL, NO overfitting estadistico")
    elif p_perm < 0.10:
        print(f"    -> rho MARGINAL bajo permutacion — señal debil")
    else:
        print(f"    -> rho NO significativo bajo permutacion — POSIBLE OVERFITTING")

# ===========================
# T4: Estabilidad del umbral — ¿elegimos 0.0 a priori o a posteriori?
# ===========================
print(f"\n{SEP}")
print("T4: ESTABILIDAD DEL UMBRAL — robustez del umbral SR=0.0 en W3")
print(f"    Si el umbral optimo en W3 coincide con W4, es señal robusta")
print(f"    Si el optimo en W3 es muy distinto, el 0.0 fue overfitting a W4")
print(SEP)

print(f"\n  Barrido de umbrales en W4 (in-sample — donde descubrimos H5):")
for u in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
    r = simulate_gate(df_w4, u, WINDOW_ROLL)
    print(f"    SR>={u:+.1f}: mejora={r['mejora']:+.3f}% | bloq={r['n_bloq']:>3}/{len(df_w4)}")

print(f"\n  Barrido de umbrales en W3 (out-of-sample para H5):")
best_u_w3 = None
best_mejora_w3 = -999
for u in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
    r = simulate_gate(df_w3, u, WINDOW_ROLL)
    if r['mejora'] > best_mejora_w3:
        best_mejora_w3 = r['mejora']
        best_u_w3 = u
    print(f"    SR>={u:+.1f}: mejora={r['mejora']:+.3f}% | bloq={r['n_bloq']:>3}/{len(df_w3)}")

print(f"\n  Umbral optimo en W4: {UMBRAL_SR:+.1f}")
print(f"  Umbral optimo en W3: {best_u_w3:+.1f}")
if abs(best_u_w3 - UMBRAL_SR) <= 0.5:
    print(f"  -> Umbrales CONSISTENTES (diferencia <= 0.5) — umbral es robusto")
else:
    print(f"  -> Umbrales INCONSISTENTES (diferencia > 0.5) — umbral fue overfitting a W4")

# ===========================
# T5: Resumen final
# ===========================
print(f"\n{SEP}")
print("RESUMEN ANTI-OVERFITTING H5")
print(SEP)

print(f"""
  T1 (Validacion cruzada W3):   mejora={res_w3['mejora']:+.3f}%
  T2 (Autocorrelacion W4):      ver resultados Ljung-Box arriba
  T3 (Permutation test rho W4): ver p_permutacion arriba
  T4 (Estabilidad umbral):      optimo_W4={UMBRAL_SR:+.1f} vs optimo_W3={best_u_w3:+.1f}

  DECISION:
  - Si T1 mejora AND T3 p<0.05 AND T4 umbrales consistentes:
      H5 es REAL. Implementar con confianza.
  - Si T1 destruye W3 OR T3 p>0.10 OR T4 umbrales inconsistentes:
      H5 es OVERFITTING a W4. NO implementar.
  - Si resultados mixtos:
      H5 puede implementarse con umbral CONSERVADOR (SR >= -0.5)
      y monitorearse en produccion con circuit breaker.
""")
