"""
investigation_xgb_dtw.py
========================
Investigación profunda de dos hallazgos críticos del Component Dashboard:
  1. XGBoost prob_cal no discrimina (posible calibración rota)
  2. Alpha DTW Signal perjudica (WR=47% vs 56% sin trigger)
"""
import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from scipy import stats

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\predictions')
SEP  = "=" * 72
DASH = "-" * 72

# ── CARGA ─────────────────────────────────────────────────────────────────
dfs = []
for f in sorted(DATA.glob('oos_trades_seed*.parquet')):
    d = pd.read_parquet(f)
    d['_seed'] = int(f.stem.split('seed')[1])
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df['ret100']  = df['return_raw'] * 100
df['_window'] = df['wfb_window']
print(f"[LOAD] {len(df)} trades | {df['_seed'].nunique()} seeds | W1-W5")
print()

# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACION 1: FALLO DE CALIBRACION XGBOOST
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print("INVESTIGACION 1: FALLO DE CALIBRACION XGBOOST")
print(SEP)

# 1A. ¿xgb_prob_cal == xgb_prob_raw? El bug documentado en predict_oos.py
print("\n[1A] Diagnostico: ¿calibracion activa o silenciada?")
diff = (df['xgb_prob_cal'] - df['xgb_prob']).abs()
pct_equal = (diff < 1e-6).mean() * 100
pct_near   = (diff < 0.01).mean() * 100
print(f"  xgb_prob_cal == xgb_prob_raw en {pct_equal:.1f}% de trades (threshold 1e-6)")
print(f"  xgb_prob_cal ≈  xgb_prob_raw en {pct_near:.1f}% de trades (|diff| < 0.01)")
print(f"  diff: mean={diff.mean():.4f} | std={diff.std():.4f} | max={diff.max():.4f}")

if pct_equal > 90:
    print("  ⚠️  CRITICO: El calibrador NO ESTA APLICANDO ninguna transformacion")
    print("  -> Bug documentado: xgb_prob_cal == xgb_prob_raw en TODOS los trades")
    print("  -> Causa probable: calibrador no cargado o colapsado a constante")
elif pct_near > 80:
    print("  ⚠️  ADVERTENCIA: Calibracion casi nula (diferencias muy pequenas)")
else:
    print("  OK: Calibrador aplica transformacion real")

# 1B. Distribucion de prob_cal por ventana
print("\n[1B] Distribucion xgb_prob y xgb_prob_cal por ventana:")
print(f"  {'Win':>5} {'prob_raw_mean':>14} {'prob_cal_mean':>14} {'diff_mean':>10} {'diff_std':>9} {'pct_igual':>9}")
print("  " + DASH)
for win in sorted(df['_window'].unique()):
    dw = df[df['_window'] == win]
    rm = dw['xgb_prob'].mean()
    cm = dw['xgb_prob_cal'].mean()
    dm = (dw['xgb_prob_cal'] - dw['xgb_prob']).abs().mean()
    ds = (dw['xgb_prob_cal'] - dw['xgb_prob']).abs().std()
    pe = ((dw['xgb_prob_cal'] - dw['xgb_prob']).abs() < 1e-6).mean() * 100
    print(f"  {win:>5} {rm:>14.4f} {cm:>14.4f} {dm:>10.4f} {ds:>9.4f} {pe:>9.1f}%")

# 1C. Si la calibración es nula, ¿xgb_prob_raw discrimina mejor?
print("\n[1C] ¿Discrimina el xgb_prob RAW (sin calibrar)?")
from scipy.stats import spearmanr
rho_raw, p_raw = spearmanr(df['xgb_prob'], df['is_win'].astype(float))
rho_cal, p_cal = spearmanr(df['xgb_prob_cal'], df['is_win'].astype(float))
print(f"  Spearman(xgb_prob_raw, is_win): rho={rho_raw:+.4f} p={p_raw:.4f}")
print(f"  Spearman(xgb_prob_cal, is_win): rho={rho_cal:+.4f} p={p_cal:.4f}")

# 1D. Reliability diagram manual (prob_cal vs WR real por decil)
print("\n[1D] Reliability Diagram — prob_cal vs WR real (calibracion ideal = diagonal):")
print(f"  {'Decil':>8} {'prob_cal_rng':>18} {'N':>5} {'WR_real%':>10} {'Diferencia':>12} {'Estado'}")
print("  " + DASH)
valid = df[['xgb_prob_cal', 'is_win']].dropna()
valid['decil'] = pd.qcut(valid['xgb_prob_cal'], q=10, labels=False, duplicates='drop')
reliability_ok = True
for d_val in sorted(valid['decil'].dropna().unique()):
    grp = valid[valid['decil'] == d_val]
    prob_mid = grp['xgb_prob_cal'].mean()
    wr_real  = grp['is_win'].mean() * 100
    diff     = wr_real - prob_mid * 100
    estado   = "OK" if abs(diff) < 5 else ("SOBRE_CONF" if diff < -5 else "INFRA_CONF")
    if estado != "OK": reliability_ok = False
    prob_rng = f"[{grp['xgb_prob_cal'].min():.3f},{grp['xgb_prob_cal'].max():.3f}]"
    print(f"  {int(d_val):>8} {prob_rng:>18} {len(grp):>5} {wr_real:>10.1f}% {diff:>+12.1f}pp  {estado}")

if not reliability_ok:
    print("\n  ⚠️  Calibracion imperfecta — la prob_cal no es una probabilidad bien calibrada")
else:
    print("\n  OK: Calibracion razonablemente buena (diferencias < 5pp)")

# 1E. Análisis por seed: ¿hay seeds con calibración buena y otras rotas?
print("\n[1E] ¿Hay seeds con calibrador funcionando y otras sin él?")
seed_cal_status = []
for seed, grp in df.groupby('_seed'):
    d_abs = (grp['xgb_prob_cal'] - grp['xgb_prob']).abs()
    pct_eq = (d_abs < 1e-6).mean() * 100
    seed_cal_status.append({'seed': seed, 'pct_igual': pct_eq, 'diff_mean': d_abs.mean()})
sc = pd.DataFrame(seed_cal_status)
rotas = sc[sc['pct_igual'] > 90]
parciales = sc[(sc['pct_igual'] > 50) & (sc['pct_igual'] <= 90)]
ok = sc[sc['pct_igual'] <= 50]
print(f"  Seeds con calibrador ROTO (>90% igual): {len(rotas)}/{len(sc)}")
print(f"  Seeds con calibrador PARCIAL (50-90% igual): {len(parciales)}/{len(sc)}")
print(f"  Seeds con calibrador OK (<50% igual): {len(ok)}/{len(sc)}")
if len(rotas) > 0:
    print(f"  Seeds rotas: {sorted(rotas['seed'].tolist())[:10]}...")

# 1F. ¿Tienen las seeds con calibrador OK mejor WR?
if len(ok) > 0 and len(rotas) > 0:
    ok_seeds   = set(ok['seed'])
    rota_seeds = set(rotas['seed'])
    wr_ok   = df[df['_seed'].isin(ok_seeds)]['is_win'].mean() * 100
    wr_rota = df[df['_seed'].isin(rota_seeds)]['is_win'].mean() * 100
    print(f"\n  WR seeds calibrador OK:    {wr_ok:.1f}%  (N_seeds={len(ok_seeds)})")
    print(f"  WR seeds calibrador ROTO:  {wr_rota:.1f}%  (N_seeds={len(rota_seeds)})")
    delta = wr_ok - wr_rota
    verdict = "✅ Calibrador mejora WR" if delta > 2 else ("❌ Calibrador no mejora WR" if delta < -2 else "⚠️  Sin diferencia")
    print(f"  Delta: {delta:+.1f}pp -> {verdict}")

# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGACION 2: ALPHA DTW SIGNAL — CAUSA DEL EFECTO NEGATIVO
# ═══════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("INVESTIGACION 2: ALPHA DTW SIGNAL — POR QUE PERJUDICA?")
print(SEP)

# Primero: reproducir la definicion del trigger tal como lo define predict_oos.py:
# alpha_trigger incluye 'alpha_dtw_signal' si el valor de alpha_dtw_signal > 0 en ese trade
# Fuente: predict_oos.py:1584

# 2A. El bug fundamental: DTW_BULL_PROB = 1.0 SIEMPRE
print("\n[2A] Causa raiz en alpha_rules.py:")
print("  DTW_BULL_PROB: float = 1.0  <- HARDCODEADO, NUNCA CAMBIA")
print("  dtw_direction = +1.0 SIEMPRE (porque 1.0 >= 0.5)")
print("  alpha_dtw_signal = tanh(mom_24h * 20)")
print("  -> El DTW signal es PURO MOMENTUM A 24H, sin ninguna comparacion de patrones reales")
print("  -> No hay Dynamic Time Warping real aqui — es un alias del momentum")
print()
print("  El Alpha Trigger se activa si alpha_dtw_signal > 0 en ese timestamp:")
print("  -> DTW_trigger activo <==> mom_24h > 0 <==> precio subio en las ultimas 24h")
print("  -> Esto es simplemente 'comprar cuando el precio sube' — momentum chaser")
print("  -> En OOS, si el precio lleva 24h subiendo, a menudo ya esta sobreextendido")

# 2B. Verificar: ¿los trades con DTW tienen mayor momentum_24h?
print("\n[2B] Perfil de trades DTW vs no-DTW:")
df['has_dtw'] = df['alpha_trigger'].fillna('').str.contains('alpha_dtw_signal')

for label, mask in [("CON DTW", df['has_dtw']), ("SIN DTW", ~df['has_dtw'])]:
    grp = df[mask]
    wr   = grp['is_win'].mean() * 100
    rm   = grp['ret100'].mean()
    rt   = grp['ret100'].sum()
    n    = len(grp)
    print(f"  {label}: N={n:>5} WR={wr:.1f}% RetMed={rm:+.4f}% RetTot={rt:+.3f}%")

# 2C. DTW por ventana y regimen — para ver si es confounding o efecto real
print("\n[2C] DTW vs no-DTW por ventana (efecto persistente o de ventana?):")
print(f"  {'Win':>5} {'WR_DTW':>9} {'WR_noDTW':>10} {'Delta':>7} {'N_DTW':>7} {'N_noDTW':>9} {'Veredicto'}")
print("  " + DASH)
for win in sorted(df['_window'].unique()):
    dw = df[df['_window'] == win]
    dtw   = dw[dw['has_dtw']];   ndtw  = dw[~dw['has_dtw']]
    if len(dtw) < 5 or len(ndtw) < 5: continue
    wr_d  = dtw['is_win'].mean() * 100;  wr_n  = ndtw['is_win'].mean() * 100
    delta = wr_d - wr_n
    v = "DTW PEOR" if delta < -3 else ("DTW MEJOR" if delta > 3 else "~igual")
    print(f"  {win:>5} {wr_d:>9.1f}% {wr_n:>10.1f}% {delta:>+7.1f}pp {len(dtw):>7} {len(ndtw):>9}   {v}")

# 2D. DTW por regimen HMM — ¿DTW activa en peores regimenes?
print("\n[2D] Distribucion de DTW por regimen HMM (¿activa mas en bear?):")
print(f"  {'Regimen':<35} {'%_DTW_activo':>13} {'WR_DTW':>8} {'WR_noDTW':>10} {'Delta':>7}")
print("  " + DASH)
for regime, grp in df.groupby('hmm_regime'):
    n_dtw = grp['has_dtw'].sum()
    n_tot = len(grp)
    pct_dtw = n_dtw / n_tot * 100
    dtw_grp  = grp[grp['has_dtw']]
    ndtw_grp = grp[~grp['has_dtw']]
    wr_d = dtw_grp['is_win'].mean() * 100 if len(dtw_grp) > 5 else float('nan')
    wr_n = ndtw_grp['is_win'].mean() * 100 if len(ndtw_grp) > 5 else float('nan')
    delta = wr_d - wr_n if not (np.isnan(wr_d) or np.isnan(wr_n)) else float('nan')
    d_str = f"{delta:+.1f}pp" if not np.isnan(delta) else "N/A"
    print(f"  {str(regime):<35} {pct_dtw:>12.1f}% {wr_d:>8.1f}% {wr_n:>10.1f}% {d_str:>7}")

# 2E. Distribucion de retornos DTW vs no-DTW
print("\n[2E] Distribucion de retornos (DTW vs no-DTW):")
dtw_ret  = df[df['has_dtw']]['ret100']
ndtw_ret = df[~df['has_dtw']]['ret100']
for label, series in [("DTW",  dtw_ret), ("noDTW", ndtw_ret)]:
    q25, q50, q75 = series.quantile([0.25, 0.5, 0.75])
    print(f"  {label}: mean={series.mean():+.4f}% | median={q50:+.4f}% | Q25={q25:+.4f}% | Q75={q75:+.4f}% | std={series.std():.4f}%")

# Mann-Whitney test
u_stat, p_mw = stats.mannwhitneyu(dtw_ret.dropna(), ndtw_ret.dropna(), alternative='less')
print(f"  Mann-Whitney (DTW < noDTW): U={u_stat:.0f} p={p_mw:.4f} {'*** SIGNIFICATIVO' if p_mw < 0.05 else 'ns'}")

# 2F. El diagnostico: ¿vale la pena el DTW o debe desactivarse?
print("\n[2F] DIAGNOSTICO FINAL sobre Alpha DTW Signal:")
print("""
  CAUSA RAIZ IDENTIFICADA:
  - DTW_BULL_PROB = 1.0 HARDCODEADO en alpha_rules.py (linea 260)
  - Nunca se recalcula con el algoritmo real de DTW (comparacion de patrones)
  - dtw_direction = +1 SIEMPRE (because 1.0 >= 0.5)
  - alpha_dtw_signal = tanh(mom_24h * 20) = momentum puro a 24H

  EFECTO:
  - El 'DTW trigger' equivale a 'el precio subio en las ultimas 24H'
  - Esto es un momentum chaser que compra cuando ya subio
  - En OOS, comprar sobreextension tiene WR < 50% en la mayoria de ventanas

  OPCIONES:
  1. [RAPIDO] Desactivar DTW trigger: use_dtw_signal: false en settings
     -> Los trades sin trigger tienen WR=56% vs 47.5% con DTW
     -> Recuperariamos ~1717 trades con mejor calidad

  2. [CORRECTO] Reimplementar DTW real con comparacion de patrones historicos
     -> DTW_BULL_PROB debe calcularse dinamicamente por ventana
     -> Requiere run_weekly_mining.py que genere la prob real
     -> Semanalmente, no hardcoded

  RECOMENDACION INMEDIATA: Opcion 1 (desactivar). Hacerlo ANTES de la proxima run.
""")
