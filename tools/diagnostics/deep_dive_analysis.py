"""
deep_dive_analysis.py
=====================
Análisis de máxima profundidad sobre los 545 trades de las 20 seeds.
Cubre dimensiones nunca analizadas:
  A. Holding time y exit type (PT/SL/VB)
  B. Consenso de señal vs calidad del trade
  C. Multi-model agreement (XGB+Meta+LGBM)
  D. SHAP drivers dominantes en wins vs losses
  E. Tiempo de entrada (hora/día) y patrones temporales
  F. Calibración real del Kelly fraction
  G. Distribución de retornos: fat tails, CVaR, skewness
  H. Bootstrap confidence intervals del portfolio
  I. Cross-seed correlation (diversificación real)
  J. Filter fallback y signal threshold vs calidad
  K. Alpha trigger efectividad
  L. Consistencia IS→OOS por seed (¿cuáles seeds transferen mejor?)
  M. Análisis de rachas (streaks) estadístico
  N. Sensibilidad a costos de transacción
  O. Consensus threshold sweep: ¿a qué nivel de consenso mejora la calidad?
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter, defaultdict

DATA  = Path(r'g:\Mi unidad\ia\luna_v2\data\reports\wfb')
SEP   = "=" * 75
SEP2  = "-" * 75

def load_all():
    dfs = []
    for f in sorted(DATA.glob('oos_trades_W*_seed*.parquet')):
        stem = f.stem; wid = stem.split('_')[2]; seed = int(stem.split('_seed')[1])
        df = pd.read_parquet(f)
        if 'timestamp' in df.columns: df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index, utc=True)
        df['_seed'] = seed; df['_window'] = wid
        dfs.append(df)
    return pd.concat(dfs).sort_index()

df = load_all()
n  = len(df)
print(f"\n{SEP}")
print(f"DEEP DIVE — {n} trades | 20 seeds | W1/W3/W4")
print(SEP)

# ─── Preprocessing ───────────────────────────────────────────────────────────
df['holding_h'] = (df['exit_time'] - df.index).dt.total_seconds() / 3600
df['hour_entry'] = df.index.hour
df['dow_entry']  = df.index.dayofweek  # 0=Mon
df['ret100']     = df['return_raw'] * 100
df['ret_kel100'] = df['return_pct'] * 100

# Exit type from holding time vs max_barrier heuristic
df['exit_type'] = 'VB'  # default
# PT: holding < 0.5 * median_holding AND is_win=True
# SL: holding < 0.5 * median_holding AND is_win=False
med_h = df['holding_h'].median()
df.loc[(df['holding_h'] < med_h * 0.7) & (df['is_win']),  'exit_type'] = 'PT'
df.loc[(df['holding_h'] < med_h * 0.7) & (~df['is_win']), 'exit_type'] = 'SL'

# ─────────────────────────────────────────────────────────────────────────────
# A. HOLDING TIME Y EXIT TYPE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'A. HOLDING TIME Y EXIT TYPE':^75}")
print(SEP2)
print(f"  Holding medio global: {df['holding_h'].mean():.1f}h | mediana: {df['holding_h'].median():.1f}h")
for win in ['W1','W3','W4']:
    dw = df[df['_window']==win]
    if len(dw)==0: continue
    print(f"  {win}: holding_med={dw['holding_h'].median():.1f}h | "
          f"PT={( dw['exit_type']=='PT').mean()*100:.0f}% "
          f"SL={(dw['exit_type']=='SL').mean()*100:.0f}% "
          f"VB={(dw['exit_type']=='VB').mean()*100:.0f}%")

# ¿Holding time predice retorno?
valid_h = df[['holding_h','ret100']].dropna()
rho_h, p_h = stats.spearmanr(valid_h['holding_h'], valid_h['ret100'])
print(f"\n  Spearman(holding_h, ret_bruto): rho={rho_h:.4f}, p={p_h:.4f} "
      f"-> {'PREDICTIVO' if p_h<0.10 else 'no predictivo'}")

# ─────────────────────────────────────────────────────────────────────────────
# B. CONSENSUS SEED COUNT vs CALIDAD (usando wfb_window column)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'B. CONSENSO (seeds por bucket) vs CALIDAD':^75}")
print(SEP2)
df['bucket'] = df.index.floor('2h')
bucket_seeds = df.groupby('bucket')['_seed'].nunique()
df['consensus'] = df['bucket'].map(bucket_seeds)

print(f"  {'Consenso':>10} {'N_trades':>9} {'WR%':>7} {'RetMed%':>9} {'RetBest%':>9}")
print("  " + "-" * 50)
for c_min, c_max in [(1,2),(3,4),(5,6),(7,9),(10,14),(15,20)]:
    mask = (df['consensus'] >= c_min) & (df['consensus'] <= c_max)
    d = df[mask]
    if len(d) == 0: continue
    print(f"  {c_min:>3}-{c_max:<3} seeds: {len(d):>6} {d['is_win'].mean()*100:>7.1f}% "
          f"{d['ret100'].mean():>9.4f}% {d['ret100'].quantile(0.75):>9.4f}%")

# Test: ¿más consenso = mejor retorno?
rho_c, p_c = stats.spearmanr(df['consensus'], df['ret100'])
print(f"\n  Spearman(consensus, ret_bruto): rho={rho_c:.4f}, p={p_c:.4f} "
      f"-> {'PREDICTIVO' if p_c<0.10 else 'no predictivo'}")

# ─────────────────────────────────────────────────────────────────────────────
# C. MULTI-MODEL AGREEMENT: cuándo XGB, Meta y LGBM coinciden
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'C. MULTI-MODEL AGREEMENT':^75}")
print(SEP2)
# Cargar de df que ya tiene _seed y _window (df ya tiene los datos)
df2 = df.copy()
# Eliminar duplicados de indice si los hay
df2 = df2[~df2.index.duplicated(keep='first')]

# Agreement score: [0, 1, 2, 3] cuántos modelos dan prob > 0.55
df2['xgb_agree']  = (df2['xgb_prob_cal'] > 0.55).astype(int)
df2['meta_agree'] = (df2['meta_v2_prob'].fillna(0) > 0.55).astype(int) if 'meta_v2_prob' in df2.columns else 0
# lgbm missing for most seeds
df2['n_models_agree'] = df2['xgb_agree'] + df2['meta_agree']

print(f"  {'Acuerdo modelos':>16} {'N':>6} {'WR%':>7} {'RetMed%':>9}")
print("  " + "-" * 45)
for n_agree in [0, 1, 2]:
    d_a = df2[df2['n_models_agree'] == n_agree]
    if len(d_a) == 0: continue
    print(f"  {n_agree} modelos (>{0.55}):   {len(d_a):>6} {d_a['is_win'].mean()*100:>7.1f}% {d_a['ret100'].mean():>9.4f}%")

# XGB solo — prob_cal granularidad
print(f"\n  XGB prob_cal vs retorno (quantiles):")
for q_lo, q_hi in [(0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.0)]:
    lo = df2['xgb_prob_cal'].quantile(q_lo); hi = df2['xgb_prob_cal'].quantile(q_hi)
    d_q = df2[(df2['xgb_prob_cal'] >= lo) & (df2['xgb_prob_cal'] < hi)]
    if len(d_q) == 0: continue
    print(f"  prob_cal [{lo:.3f},{hi:.3f}): n={len(d_q):>4} WR={d_q['is_win'].mean()*100:.1f}% "
          f"RetMed={d_q['ret100'].mean():.4f}%")

# ─────────────────────────────────────────────────────────────────────────────
# D. SHAP DRIVERS dominantes en wins vs losses
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'D. SHAP DRIVERS — features que discriminan wins vs losses':^75}")
print(SEP2)
df2['shap1'] = df2['shap_drivers'].str.split('|').str[0].str.strip().str.split('(').str[0].str.strip()
shap_wins   = df2[df2['is_win']]['shap1'].value_counts().head(10)
shap_losses = df2[~df2['is_win']]['shap1'].value_counts().head(10)
print(f"  Top SHAP feature en WINS:")
for feat, cnt in shap_wins.items():
    pct = cnt/df2['is_win'].sum()*100
    print(f"    {feat:<45} {cnt:>4} ({pct:.1f}%)")
print(f"\n  Top SHAP feature en LOSSES:")
for feat, cnt in shap_losses.items():
    pct = cnt/( ~df2['is_win']).sum()*100
    print(f"    {feat:<45} {cnt:>4} ({pct:.1f}%)")

# Features que SOLO aparecen en wins o solo en losses
only_wins   = set(shap_wins.index) - set(shap_losses.index)
only_losses = set(shap_losses.index) - set(shap_wins.index)
if only_wins:   print(f"\n  Features EXCLUSIVAS de wins:   {only_wins}")
if only_losses: print(f"  Features EXCLUSIVAS de losses: {only_losses}")

# ─────────────────────────────────────────────────────────────────────────────
# E. PATRONES TEMPORALES (hora/día)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'E. PATRONES TEMPORALES':^75}")
print(SEP2)
print(f"  Win Rate por hora de entrada (UTC):")
hr_wr = df2.groupby('hour_entry').agg(n=('is_win','count'), wr=('is_win','mean'), ret=('ret100','mean'))
print(f"  {'Hora':>5} {'N':>4} {'WR%':>7} {'RetMed%':>9}")
for hr, row in hr_wr.iterrows():
    if row['n'] < 3: continue
    flag = " <--" if row['wr'] < 0.40 or row['ret'] < -0.05 else ""
    print(f"  {int(hr):>5}h {int(row['n']):>4} {row['wr']*100:>7.1f}% {row['ret']:>9.4f}%{flag}")

print(f"\n  Win Rate por día de semana:")
days = ['Lun','Mar','Mie','Jue','Vie','Sab','Dom']
dow_wr = df2.groupby('dow_entry').agg(n=('is_win','count'), wr=('is_win','mean'), ret=('ret100','mean'))
for dow, row in dow_wr.iterrows():
    if row['n'] < 3: continue
    flag = " <--" if row['wr'] < 0.40 or row['ret'] < -0.05 else ""
    print(f"  {days[int(dow)]}: n={int(row['n']):<3} WR={row['wr']*100:.1f}% RetMed={row['ret']:.4f}%{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# F. CALIBRACIÓN DEL KELLY FRACTION
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'F. CALIBRACION KELLY FRACTION':^75}")
print(SEP2)
print(f"  {'Kelly rango':>14} {'N':>4} {'WR%':>7} {'E[ret_bruto]%':>14} {'ret_kelly%':>11}")
kel = df2['kelly_fraction_used']
for lo, hi in [(0,0.05),(0.05,0.10),(0.10,0.15),(0.15,0.20),(0.20,0.30),(0.30,1.0)]:
    d_k = df2[(kel >= lo) & (kel < hi)]
    if len(d_k) == 0: continue
    print(f"  [{lo:.2f},{hi:.2f}): {len(d_k):>4} {d_k['is_win'].mean()*100:>7.1f}% "
          f"{d_k['ret100'].mean():>14.4f}% {d_k['ret_kel100'].mean():>11.4f}%")

# Optimal Kelly check: win_rate * avg_win / avg_loss
wins   = df2[df2['is_win']]['ret100']
losses = df2[~df2['is_win']]['ret100']
p_win = df2['is_win'].mean()
if len(losses) > 0 and losses.mean() < 0:
    k_optimal = p_win - (1 - p_win) / (-losses.mean() / wins.mean())
    k_used_med = df2['kelly_fraction_used'].median()
    print(f"\n  Kelly óptimo teórico (formula completa): {k_optimal:.4f} = {k_optimal*100:.2f}%")
    print(f"  Kelly mediano usado en trades:            {k_used_med:.4f} = {k_used_med*100:.2f}%")
    ratio = k_used_med / k_optimal if k_optimal > 0 else float('nan')
    print(f"  Ratio usado/óptimo:                       {ratio:.2f}x "
          f"({'sub-optimal: muy conservador' if ratio < 0.4 else 'AGRESIVO' if ratio > 0.8 else 'razonable'})")

# ─────────────────────────────────────────────────────────────────────────────
# G. DISTRIBUCIÓN DE RETORNOS: fat tails, CVaR, skewness
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'G. DISTRIBUCION DE RETORNOS — fat tails y tail risk':^75}")
print(SEP2)
for win in ['GLOBAL','W3','W4']:
    if win == 'GLOBAL': d_r = df2['ret100']
    else: d_r = df2[df2['_window']==win]['ret100']
    if len(d_r) < 5: continue
    skew  = float(stats.skew(d_r))
    kurt  = float(stats.kurtosis(d_r))  # excess kurtosis (0=normal)
    cvar5 = d_r.quantile(0.05)          # 5th percentile = CVaR95 threshold
    var5  = d_r[d_r <= cvar5].mean()    # Expected Shortfall 95%
    print(f"  {win:<8}: mean={d_r.mean():.4f}% std={d_r.std():.4f}% "
          f"skew={skew:.2f} kurt={kurt:.2f} CVaR95={var5:.4f}%")
    if abs(kurt) > 3:
        print(f"           -> COLAS GRUESAS (kurtosis={kurt:.1f} >> 3): cola izquierda peligrosa")

# Normalidad test
_, p_normal = stats.normaltest(df2['ret100'])
print(f"\n  Test normalidad (D'Agostino): p={p_normal:.6f} "
      f"-> {'NO normal (fat tails confirmados)' if p_normal < 0.05 else 'compatible con normal'}")

# ─────────────────────────────────────────────────────────────────────────────
# H. BOOTSTRAP CONFIDENCE INTERVALS del portfolio ensemble
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'H. BOOTSTRAP CONFIDENCE INTERVALS (n=2000)':^75}")
print(SEP2)
# Reconstruir portfolio ensemble
df2['bucket'] = df2.index.floor('2h')
bkt_seeds = df2.groupby('bucket')['_seed'].nunique()
df2['consensus'] = df2['bucket'].map(bkt_seeds)
df_cons = df2[df2['consensus'] >= 3].copy()
agg_cols = {'return_pct': 'mean', 'is_win': 'max', 'consensus': 'first', '_window': 'first'}
df_port = df_cons.groupby('bucket').agg(agg_cols).sort_index()

# Embargo 96H
selected, last_t = [], None
for ts in df_port.index:
    if last_t is None or (ts - last_t).total_seconds()/3600 >= 96:
        selected.append(ts); last_t = ts
df_port = df_port.loc[selected]
rets = df_port['return_pct'].values
if len(rets) < 2:
    print(f"  Portfolio vacio o insuficiente con embargo 96H, usando embargo 48H")
    sel2, lt2 = [], None
    for ts in df_port_full.index if 'df_port_full' in dir() else df_port.index:
        if lt2 is None or (ts-lt2).total_seconds()/3600 >= 48: sel2.append(ts); lt2=ts
    df_port = df_port_full.loc[sel2] if 'df_port_full' in dir() else df_port
    rets = df_port['return_pct'].values

n_days = max((df_port.index.max()-df_port.index.min()).days, 1) if len(df_port) > 1 else 365

rng = np.random.default_rng(42)
n_boot = 2000
sharpes, wrs, tots = [], [], []
for _ in range(n_boot):
    sample = rng.choice(rets, size=max(len(rets),2), replace=True)
    ann_factor = np.sqrt(len(rets) * 365.25 / max(n_days, 1))
    sr = sample.mean() / (sample.std() + 1e-10) * ann_factor
    sharpes.append(sr); wrs.append((sample > 0).mean()); tots.append(sample.sum()*100)

print(f"  Portfolio ensemble: {len(df_port)} trades")
print(f"  Sharpe anual:  {np.mean(sharpes):.4f}  CI95=[{np.percentile(sharpes,2.5):.4f}, {np.percentile(sharpes,97.5):.4f}]")
print(f"  Win Rate:      {np.mean(wrs)*100:.2f}%  CI95=[{np.percentile(wrs,2.5)*100:.2f}%, {np.percentile(wrs,97.5)*100:.2f}%]")
print(f"  Ret Total:     {np.mean(tots):.4f}%  CI95=[{np.percentile(tots,2.5):.4f}%, {np.percentile(tots,97.5):.4f}%]")
p_positive = (np.array(tots) > 0).mean()
print(f"  P(ret > 0):    {p_positive*100:.1f}% de los bootstraps dan retorno positivo")

# ─────────────────────────────────────────────────────────────────────────────
# I. CROSS-SEED CORRELATION (diversificacion real)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'I. CROSS-SEED CORRELATION — diversificacion efectiva':^75}")
print(SEP2)
# Pivot: cada columna = una seed, cada fila = bucket temporal, valor = return_pct medio
df2['bucket2'] = df2.index.floor('2h')
pivot = df2.groupby(['bucket2','_seed'])['return_pct'].mean().unstack(level='_seed')
corr_mat = pivot.corr()
upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
mean_corr = upper.stack().mean()
max_corr  = upper.stack().max()
min_corr  = upper.stack().min()
print(f"  Correlacion cross-seed media:  {mean_corr:.4f}")
print(f"  Correlacion maxima:            {max_corr:.4f}")
print(f"  Correlacion minima:            {min_corr:.4f}")
eff_n = 1 / (1 + (len(corr_mat)-1) * mean_corr) if mean_corr > -1/(len(corr_mat)-1) else len(corr_mat)
print(f"  N efectivo de seeds (Kish):    {eff_n:.1f} (de {len(corr_mat)} seeds)")
print(f"  -> {'Buena diversificacion' if eff_n > len(corr_mat)*0.5 else 'Seeds altamente correladas'}")

# Pairs con mayor correlacion
pairs_high = upper.stack().sort_values(ascending=False).head(5)
print(f"\n  Pares de seeds mas correladas:")
for (s1,s2), c in pairs_high.items():
    print(f"    seed {int(s1)} x seed {int(s2)}: rho={c:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# J. FILTER FALLBACK Y SIGNAL THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'J. FILTER FALLBACK Y SIGNAL THRESHOLD':^75}")
print(SEP2)
fb_counts = df2['filter_fallback_level'].value_counts().sort_index()
print(f"  Distribucion de filter_fallback_level:")
for lvl, cnt in fb_counts.items():
    d_fb = df2[df2['filter_fallback_level']==lvl]
    print(f"    Level {lvl}: {cnt:>4} trades | WR={d_fb['is_win'].mean()*100:.1f}% | RetMed={d_fb['ret100'].mean():.4f}%")

lowered = df2['threshold_was_lowered'].sum()
print(f"\n  Trades con threshold lowered: {lowered} ({lowered/n*100:.1f}%)")
if lowered > 0:
    d_low  = df2[df2['threshold_was_lowered']]
    d_high = df2[~df2['threshold_was_lowered']]
    print(f"    Threshold normal: WR={d_high['is_win'].mean()*100:.1f}% RetMed={d_high['ret100'].mean():.4f}%")
    print(f"    Threshold lowered: WR={d_low['is_win'].mean()*100:.1f}% RetMed={d_low['ret100'].mean():.4f}%")

# Signal threshold vs return
rho_thr, p_thr = stats.spearmanr(df2['signal_threshold'], df2['ret100'])
print(f"\n  Spearman(signal_threshold, ret_bruto): rho={rho_thr:.4f}, p={p_thr:.4f} "
      f"-> {'PREDICTIVO' if p_thr<0.10 else 'no predictivo'}")

# ─────────────────────────────────────────────────────────────────────────────
# K. ALPHA TRIGGER EFECTIVIDAD
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'K. ALPHA TRIGGER EFECTIVIDAD':^75}")
print(SEP2)
triggers = ['alpha_golden_score','alpha_genetic_score','alpha_dtw_signal']
for trig in triggers:
    has = df2['alpha_trigger'].str.contains(trig, na=False)
    if has.sum() < 5: continue
    d_has = df2[has]; d_not = df2[~has]
    print(f"  {trig}:")
    print(f"    CON trigger: n={len(d_has):>4} WR={d_has['is_win'].mean()*100:.1f}% RetMed={d_has['ret100'].mean():.4f}%")
    print(f"    SIN trigger: n={len(d_not):>4} WR={d_not['is_win'].mean()*100:.1f}% RetMed={d_not['ret100'].mean():.4f}%")

# ─────────────────────────────────────────────────────────────────────────────
# L. CONSISTENCIA IS→OOS POR SEED
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'L. CONSISTENCIA POR SEED — cuales transfieren mejor':^75}")
print(SEP2)
print(f"  {'Seed':>8} {'W3_WR%':>8} {'W4_WR%':>8} {'W3_ret%':>9} {'W4_ret%':>9} {'Delta_WR':>9} {'Estable?'}")
print("  " + "-" * 68)
seed_stats = []
for seed in sorted(df2['_seed'].unique()):
    d3 = df2[(df2['_seed']==seed) & (df2['_window']=='W3')]
    d4 = df2[(df2['_seed']==seed) & (df2['_window']=='W4')]
    if len(d3) == 0 and len(d4) == 0: continue
    wr3 = d3['is_win'].mean()*100 if len(d3)>0 else float('nan')
    wr4 = d4['is_win'].mean()*100 if len(d4)>0 else float('nan')
    r3  = d3['ret100'].sum() if len(d3)>0 else float('nan')
    r4  = d4['ret100'].sum() if len(d4)>0 else float('nan')
    dwr = wr4 - wr3 if not (pd.isna(wr3) or pd.isna(wr4)) else float('nan')
    stable = 'ESTABLE' if not pd.isna(dwr) and abs(dwr) < 20 else ('DEGRADA' if not pd.isna(dwr) and dwr < -20 else '?')
    seed_stats.append({'seed': seed, 'wr3': wr3, 'wr4': wr4, 'r3': r3, 'r4': r4, 'dwr': dwr, 'stable': stable})
    wr3s = f"{wr3:.1f}%" if not pd.isna(wr3) else "  N/A "
    wr4s = f"{wr4:.1f}%" if not pd.isna(wr4) else "  N/A "
    r3s  = f"{r3:.4f}%" if not pd.isna(r3) else "    N/A  "
    r4s  = f"{r4:.4f}%" if not pd.isna(r4) else "    N/A  "
    dwrs = f"{dwr:+.1f}pp" if not pd.isna(dwr) else "   N/A"
    print(f"  {int(seed):>8} {wr3s:>8} {wr4s:>8} {r3s:>9} {r4s:>9} {dwrs:>9}  {stable}")

df_ss = pd.DataFrame(seed_stats).dropna(subset=['dwr'])
if len(df_ss) > 2:
    best_seeds = df_ss.nlargest(3, 'dwr')['seed'].tolist()
    worst_seeds = df_ss.nsmallest(3, 'dwr')['seed'].tolist()
    print(f"\n  Seeds mas ESTABLES (menor degradacion W3->W4): {[int(s) for s in best_seeds]}")
    print(f"  Seeds mas INESTABLES (mayor degradacion):      {[int(s) for s in worst_seeds]}")

# ─────────────────────────────────────────────────────────────────────────────
# M. ANALISIS DE RACHAS (STREAKS)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'M. ANALISIS DE RACHAS — estadistica de streaks':^75}")
print(SEP2)
# Por cada seed+window, calcular max losing streak y max winning streak
max_loss_streaks = []; max_win_streaks = []
for seed in df2['_seed'].unique():
    for win in ['W3','W4']:
        d_sw = df2[(df2['_seed']==seed)&(df2['_window']==win)].sort_index()
        if len(d_sw) < 3: continue
        wins_seq = d_sw['is_win'].astype(int).tolist()
        cur_loss = cur_win = max_loss = max_win = 0
        for w in wins_seq:
            if w == 0: cur_loss += 1; cur_win = 0
            else:      cur_win  += 1; cur_loss = 0
            max_loss = max(max_loss, cur_loss)
            max_win  = max(max_win, cur_win)
        max_loss_streaks.append(max_loss); max_win_streaks.append(max_win)

print(f"  Max racha perdedora por seed+ventana: media={np.mean(max_loss_streaks):.1f} | max={max(max_loss_streaks)} | p90={np.percentile(max_loss_streaks,90):.0f}")
print(f"  Max racha ganadora  por seed+ventana: media={np.mean(max_win_streaks):.1f}  | max={max(max_win_streaks)}  | p90={np.percentile(max_win_streaks,90):.0f}")

# ¿Recuperación tras rachas: el trade DESPUES de una racha de 3+ perdidas?
print(f"\n  Recuperacion tras rachas de perdidas consecutivas:")
recovery_stats = defaultdict(list)
for seed in df2['_seed'].unique():
    for win in ['W3','W4']:
        d_sw = df2[(df2['_seed']==seed)&(df2['_window']==win)].sort_index()
        if len(d_sw) < 4: continue
        wins_seq = d_sw['is_win'].astype(int).tolist()
        rets_seq  = d_sw['ret100'].tolist()
        cur_loss = 0
        for i, w in enumerate(wins_seq[:-1]):
            if w == 0: cur_loss += 1
            else:      cur_loss = 0
            if cur_loss >= 2 and i+1 < len(rets_seq):
                recovery_stats[min(cur_loss,5)].append(rets_seq[i+1])
for n_losses, rets_rec in sorted(recovery_stats.items()):
    if len(rets_rec) < 3: continue
    wr_rec = sum(r > 0 for r in rets_rec) / len(rets_rec) * 100
    ret_rec = np.mean(rets_rec)
    print(f"  Tras {n_losses}+ perdidas consecutivas: n={len(rets_rec):>3} siguiente_WR={wr_rec:.1f}% RetMed={ret_rec:.4f}%")

# ─────────────────────────────────────────────────────────────────────────────
# N. SENSIBILIDAD A COSTOS DE TRANSACCION
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'N. SENSIBILIDAD A COSTOS DE TRANSACCION':^75}")
print(SEP2)
print(f"  {'Cost RT':>10} {'WR%':>7} {'RetMed%':>10} {'RetTot% W3':>12} {'RetTot% W4':>12}")
for cost in [0.0005, 0.001, 0.0015, 0.002, 0.003]:
    # Recalcular ret con nuevo costo
    ret_recalc  = (df2['return_raw'] + 0.0015 - cost)  # deshacer 0.0015 y aplicar nuevo
    wr_recalc   = (ret_recalc > 0).mean() * 100
    tot_recalc  = ret_recalc.mean() * 100
    tot_w3 = ret_recalc[df2['_window']=='W3'].sum() * 100
    tot_w4 = ret_recalc[df2['_window']=='W4'].sum() * 100
    flag = " <- ACTUAL" if abs(cost - 0.0015) < 0.0001 else ""
    print(f"  {cost*100:.2f}%:     {wr_recalc:>7.2f}% {tot_recalc:>10.4f}% {tot_w3:>12.4f}% {tot_w4:>12.4f}%{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# O. CONSENSUS THRESHOLD SWEEP
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'O. CONSENSUS THRESHOLD SWEEP — umbral optimo sin lookback':^75}")
print(SEP2)
print(f"  (evaluado sobre portfolio ensemble con embargo 96H)")
print(f"  {'MinSeeds':>9} {'Trades':>7} {'WR%':>7} {'Sharpe':>8} {'RetTot%':>9} {'MaxDD%':>8}")
print("  " + "-" * 60)
for min_seeds in [2, 3, 4, 5, 6, 8, 10]:
    df_c = df2[df2['consensus'] >= min_seeds].copy()
    if len(df_c) == 0: continue
    agg = {'return_pct':'mean', 'is_win':'max', '_window':'first'}
    dfp = df_c.groupby('bucket').agg(agg).sort_index()
    sel, lt = [], None
    for ts in dfp.index:
        if lt is None or (ts-lt).total_seconds()/3600 >= 96: sel.append(ts); lt=ts
    dfp = dfp.loc[sel]
    if len(dfp) < 2: continue
    r = dfp['return_pct']
    sr = r.mean()/r.std()*np.sqrt(len(r)/(len(r)/365.25)) if r.std()>0 else 0
    eq = (1+r).cumprod(); mdd = float((eq-eq.cummax()).min()*100)
    flag = " <-- ACTUAL" if min_seeds == 3 else ""
    print(f"  {min_seeds:>9} {len(dfp):>7} {dfp['is_win'].mean()*100:>7.1f}% {sr:>8.4f} {r.sum()*100:>9.4f}% {mdd:>8.2f}%{flag}")

print(f"\n{SEP}")
print("FIN ANALISIS PROFUNDO — resumen ejecutivo al final")
print(SEP)
print("""
  HALLAZGOS ACCIONABLES:
  1. Holding time vs retorno: ver Spearman arriba
  2. Consenso: ver si consenso alto (>10) domina retorno positivamente
  3. SHAP: features dominantes en wins revelan qué aprende realmente el modelo
  4. Patrones temporales: horas/días con WR < 40% son candidatas a excluir
  5. Kelly calibration: ratio usado/optimo muestra si el sizing es correcto
  6. Fat tails: kurtosis > 3 implica que el CVaR real es mayor al estimado
  7. Bootstrap CI: si P(ret>0) < 50% => no hay ventaja estadística
  8. Cross-seed correlation: si N_eff < 5, el ensemble no diversifica bien
  9. Filter fallback: si fallback_level>0 genera peores trades -> cerrar esa puerta
  10. Threshold sweep: si consenso>5 mejora calidad -> subir umbral
""")
