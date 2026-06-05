"""
=============================================================================
FASES 2 + 3 — FORMULACIÓN DE HIPÓTESIS Y TESTS ESTADÍSTICOS
Protocolo diagnostico_cuantitativo.md
=============================================================================
Las hipótesis se formulan sobre lo que los datos de FASE 1 revelaron.
Cada test reporta p-value. Umbral: p < 0.05 para CONFIRMAR.
=============================================================================
"""
import sys, json, warnings
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

wfb_dir = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')

all_dfs = []
for f in sorted(wfb_dir.glob('oos_trades_W*_seed*.parquet')):
    try:
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue
        parts = f.stem.split('_')
        df['seed']   = int(next(p.replace('seed','') for p in parts if p.startswith('seed')))
        df['window'] = next(p for p in parts if p.startswith('W'))
        all_dfs.append(df)
    except Exception as e:
        print(f'[ERROR] {f.name}: {e}')

df_all = pd.concat(all_dfs, ignore_index=True)
SEP = '─'*72

# ═══════════════════════════════════════════════════════════════════════════
# H-NEW-1: CORRELACIÓN INVERSA Δcal → RETORNO
# Hipótesis: mayor Δcal (inflar probs) → menor retorno por trade
# Consecuencia medible: Spearman(Δcal, return_pct) < 0 con p < 0.05
# ═══════════════════════════════════════════════════════════════════════════
print('═'*72)
print('H-NEW-1: ¿El Δcal correlaciona negativamente con el retorno?')
print('  Formulación: seeds/ventanas con mayor Δcal tienen EV más negativo')
print('  Consecuencia: Spearman(Δcal, return_pct) < 0, p < 0.05')
print('═'*72)

df_all['delta_cal'] = df_all['xgb_prob_cal'] - df_all['xgb_prob']

# Test nivel trade (968 obs)
r_trade, p_trade = stats.spearmanr(df_all['delta_cal'].dropna(),
                                    df_all.loc[df_all['delta_cal'].notna(), 'return_pct'])
print(f'\nTest nivel TRADE (N={len(df_all)}):')
print(f'  Spearman r={r_trade:+.4f} p={p_trade:.4f} → '
      f'{"CONFIRMADA" if p_trade < 0.05 and r_trade < 0 else "DESCARTADA"}')

# Test nivel seed×ventana (agrupado — más limpio, menos autocorrelación)
sv_stats = df_all.groupby(['seed','window']).apply(lambda g: pd.Series({
    'n':          len(g),
    'delta_cal':  float(g['delta_cal'].mean()),
    'ev':         float(g['return_pct'].mean()),
    'wr':         float(g['is_win'].mean()),
    'ret_std':    float(g['return_pct'].std()),
})).reset_index()

sv_stats_min30 = sv_stats[sv_stats['n'] >= 10]  # mínimo 10 trades para evitar N pequeño
print(f'\nTest nivel SEED×VENTANA (N>={10}, n_grupos={len(sv_stats_min30)}):')
if len(sv_stats_min30) >= 5:
    r_sv, p_sv = stats.spearmanr(sv_stats_min30['delta_cal'], sv_stats_min30['ev'])
    print(f'  Spearman(Δcal, EV): r={r_sv:+.4f} p={p_sv:.4f} → '
          f'{"CONFIRMADA" if p_sv < 0.05 and r_sv < 0 else "DESCARTADA (p>0.05 o r>0)"}')
    r_sv2, p_sv2 = stats.spearmanr(sv_stats_min30['delta_cal'], sv_stats_min30['wr'])
    print(f'  Spearman(Δcal, WR): r={r_sv2:+.4f} p={p_sv2:.4f} → '
          f'{"CONFIRMADA" if p_sv2 < 0.05 and r_sv2 < 0 else "DESCARTADA"}')
else:
    print('  N insuficiente para test agrupado')

# Cuartiles de Δcal y EV para exploración
print(f'\nDesglose por cuartil Δcal (nivel trade):')
df_all['q_dcal'] = pd.qcut(df_all['delta_cal'], 4, labels=['Q1_low','Q2','Q3','Q4_high'])
for q, g in df_all.groupby('q_dcal', observed=True):
    ev_q  = float(g['return_pct'].mean())
    wr_q  = float(g['is_win'].mean())
    n_q   = len(g)
    dc_m  = float(g['delta_cal'].mean())
    print(f'  {q} (Δcal≈{dc_m:+.3f}): N={n_q} WR={wr_q*100:.1f}% EV={ev_q*100:+.5f}%')

# Trades "recuperados" (cal>=thr, raw<thr) vs "originales" (raw>=thr)
thr = df_all['signal_threshold']
recovered = (df_all['xgb_prob_cal'] >= thr) & (df_all['xgb_prob'] < thr)
original  = df_all['xgb_prob'] >= thr

g_rec = df_all[recovered]
g_ori = df_all[original]
print(f'\nComparación RECUPERADOS vs ORIGINALES por calibración:')
print(f'  RECUPERADOS (raw<thr, cal>=thr): N={len(g_rec)} '
      f'WR={float(g_rec["is_win"].mean())*100:.1f}% '
      f'EV={float(g_rec["return_pct"].mean())*100:+.5f}%')
print(f'  ORIGINALES  (raw>=thr):          N={len(g_ori)} '
      f'WR={float(g_ori["is_win"].mean())*100:.1f}% '
      f'EV={float(g_ori["return_pct"].mean())*100:+.5f}%')
if len(g_rec) >= 20 and len(g_ori) >= 20:
    t_rv, p_rv = stats.ttest_ind(g_rec['return_pct'].dropna(), g_ori['return_pct'].dropna())
    print(f'  t-test EV(rec vs ori): t={t_rv:.3f} p={p_rv:.4f} → '
          f'{"diferencia SIGNIFICATIVA" if p_rv < 0.05 else "NO significativa"}')
    ks_stat, ks_p = stats.ks_2samp(g_rec['return_pct'].dropna(), g_ori['return_pct'].dropna())
    print(f'  KS-test distribuciones: KS={ks_stat:.4f} p={ks_p:.4f} → '
          f'{"DISTINTAS" if ks_p < 0.05 else "no distinguibles"}')

print()

# ═══════════════════════════════════════════════════════════════════════════
# H-NEW-2: BUG DE CLONES — seeds producen modelos idénticos
# Hipótesis: las seeds {789, 42975, 44085, 36457} generan distribuciones
#            de retorno estadísticamente indistinguibles en W1
# Consecuencia: KS-test entre pares no rechaza H0 (p > 0.05 = SON iguales)
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('H-NEW-2: ¿Las seeds sospechosas son clones estadísticos?')
print('  Consecuencia: KS-test p > 0.05 entre pares (no se puede rechazar igualdad)')
print(SEP)

clone_seeds = [789, 42975, 44085, 36457]
w1 = df_all[df_all['window'] == 'W1']

print(f'\nEstadísticas individuales en W1:')
clone_dfs = {}
for s in clone_seeds:
    g = w1[w1['seed'] == s]
    clone_dfs[s] = g
    ret = g['return_pct'].dropna()
    cal = g['xgb_prob_cal'].dropna()
    print(f'  seed{s}: N={len(g)} WR={float((ret>0).mean())*100:.1f}% '
          f'mean={ret.mean()*100:+.5f}% std={ret.std()*100:.5f}% '
          f'cal_mean={cal.mean():.5f}')

print(f'\nKS-test entre pares de clones (return_pct W1):')
from itertools import combinations
for (s1, s2) in combinations(clone_seeds, 2):
    r1 = clone_dfs[s1]['return_pct'].dropna()
    r2 = clone_dfs[s2]['return_pct'].dropna()
    if len(r1) < 5 or len(r2) < 5:
        print(f'  seed{s1} vs seed{s2}: N insuficiente')
        continue
    ks, p = stats.ks_2samp(r1, r2)
    clone_verdict = 'CLONES (p>0.05, no distinguibles)' if p > 0.05 else f'DISTINTOS (p={p:.4f})'
    print(f'  seed{s1} vs seed{s2}: KS={ks:.5f} p={p:.4f} → {clone_verdict}')

print(f'\nKS-test: sospechosos vs seed control (seed123, N=51 también):')
seed123_w1 = w1[w1['seed'] == 123]['return_pct'].dropna()
for s in clone_seeds:
    r = clone_dfs[s]['return_pct'].dropna()
    ks, p = stats.ks_2samp(r, seed123_w1)
    print(f'  seed{s} vs seed123: KS={ks:.5f} p={p:.4f} → '
          f'{"INDISTINGUIBLES" if p > 0.05 else "distinguibles"}')

print(f'\nTest de identidad en xgb_prob_cal (¿probabilidades idénticas?):')
for (s1, s2) in combinations(clone_seeds, 2):
    c1 = clone_dfs[s1]['xgb_prob_cal'].dropna().values
    c2 = clone_dfs[s2]['xgb_prob_cal'].dropna().values
    min_n = min(len(c1), len(c2))
    if min_n < 5:
        continue
    ks, p = stats.ks_2samp(c1, c2)
    print(f'  seed{s1} vs seed{s2} [cal]: KS={ks:.5f} p={p:.4f}')

print()

# ═══════════════════════════════════════════════════════════════════════════
# H-A: BEAR_LONG COLAPSO — ¿Es estructural o esporádico?
# Hipótesis: el modelo bear_long produce std_prob=0 en OOS porque los datos
#            OOS del agente bear en W1 están casi vacíos (régimen ausente en IS)
# Consecuencia: inspeccionar los executor_state (modelos) para ver n_samples bear
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('H-A: ¿El colapso bear_long es estructural (pocos samples IS)?')
print('  Consecuencia: executor_state de todas las seeds muestran n_samples_bear << n_samples_bull')
print(SEP)

cache_dir = Path('g:/Mi unidad/ia/luna_v2/data/wfb_cache')
bear_samples = []
bull_samples = []

for f in sorted(cache_dir.glob('executor_state_wfb_s*_W1_models.json')):
    try:
        with open(f) as fp:
            state = json.load(fp)
        seed_id = f.stem.split('_s')[1].split('_')[0]
        # Buscar info de entrenamiento en el state
        for key, val in state.items():
            if isinstance(val, dict):
                if 'bear' in key.lower() and 'n_samples' in str(val):
                    bear_samples.append({'seed': seed_id, 'key': key, 'val': str(val)[:80]})
                if 'bull' in key.lower() and 'n_samples' in str(val):
                    bull_samples.append({'seed': seed_id, 'key': key, 'val': str(val)[:80]})
    except Exception as e:
        pass

if bear_samples:
    print(f'Executor states con info bear: {len(bear_samples)}')
    for s in bear_samples[:5]:
        print(f'  {s}')
else:
    print('No se encontró n_samples en executor_state — inspeccionando estructura...')
    # Mostrar estructura de un estado para entender el schema
    sample_files = list(cache_dir.glob('executor_state_wfb_s42_W1_models.json'))
    if sample_files:
        with open(sample_files[0]) as fp:
            state = json.load(fp)
        top_keys = list(state.keys())[:15]
        print(f'  Keys de executor_state: {top_keys}')
        # Buscar cualquier referencia a bear
        for k, v in state.items():
            if 'bear' in str(k).lower() or 'bear' in str(v)[:50].lower():
                print(f'  [bear key] {k}: {str(v)[:100]}')

# Contar colapsos desde los logs
log_path = Path('C:/Users/Usuario/.gemini/antigravity-ide/brain/ad23283d-d02e-4616-9748-5d609f02bf06/.system_generated/tasks/task-1314.log')
import re
if log_path.exists():
    log = log_path.read_text(encoding='utf-8', errors='replace')
    collapses = re.findall(r'COLAPSO TOTAL.*?bear_long.*?std_prob=(\S+).*?min=max=(\S+).*?n_rows=(\S+)', log)
    print(f'\nEventos de colapso en logs (N={len(collapses)}):')
    if collapses:
        stds  = [float(c[0]) for c in collapses]
        probs = [float(c[1]) for c in collapses]
        nrows = [int(c[2].rstrip('.')) for c in collapses]
        from collections import Counter
        prob_cnt = Counter(probs)
        print(f'  std_prob: siempre={set(stds)} (debería ser 0)')
        print(f'  min=max prob values: {prob_cnt.most_common(5)}')
        print(f'  n_rows: mean={np.mean(nrows):.0f} std={np.std(nrows):.0f} '
              f'min={min(nrows)} max={max(nrows)}')
        # ¿Cuántas seeds distintas tienen colapso?
        collapse_windows = re.findall(r'SEMILLA: (\d+)', log)
        # Filter to only windows with collapse
        seeds_w_collapse = []
        lines = log.split('\n')
        curr_seed = None
        for line in lines:
            m = re.search(r'SEMILLA: (\d+)', line)
            if m:
                curr_seed = m.group(1)
            if 'COLAPSO TOTAL' in line and 'bear_long' in line and curr_seed:
                seeds_w_collapse.append(curr_seed)
        print(f'  Seeds con colapso identificadas en logs: {len(set(collapse_windows))} → {sorted(set(collapse_windows), key=int)[:10]}')
print()

# ═══════════════════════════════════════════════════════════════════════════
# H-D: ANOMALÍA HORA 11 UTC
# Hipótesis: las entradas a las 11 UTC tienen WR=8.8% — ¿es estadísticamente
#            significativo o solo ruido de N=57?
# Test: binom_test (¿WR=8.8% es significativamente < 50%?)
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('H-D: ¿La anomalía WR=8.8% en H11 UTC es estadísticamente real?')
print('  Test: binomial (p < 0.05 → anomalía real, no ruido)')
print(SEP)

df_all['entry_hour'] = pd.to_datetime(df_all['entry_time'], utc=True, errors='coerce').dt.hour

print('\nWR por hora UTC (todas las ventanas):')
hour_stats = []
for h, g in df_all.groupby('entry_hour'):
    n    = len(g)
    wins = int(g['is_win'].sum())
    wr   = wins / n
    # Binomial test: H0 = WR = 0.5
    binom_p = stats.binomtest(wins, n, 0.5, alternative='two-sided').pvalue
    ev   = float(g['return_pct'].mean())
    hour_stats.append((h, n, wr, binom_p, ev))
    sig  = '*** p<0.001' if binom_p < 0.001 else ('** p<0.01' if binom_p < 0.01 else ('* p<0.05' if binom_p < 0.05 else ''))
    print(f'  H{int(h):02d}: N={n:>3} WR={wr*100:>5.1f}% EV={ev*100:>+7.4f}% | binom p={binom_p:.4f} {sig}')

# Trades H11 vs resto
h11 = df_all[df_all['entry_hour'] == 11]
resto = df_all[df_all['entry_hour'] != 11]
print(f'\nH11 vs RESTO:')
print(f'  H11:  N={len(h11)} WR={float(h11["is_win"].mean())*100:.1f}% '
      f'EV={float(h11["return_pct"].mean())*100:+.5f}%')
print(f'  RESTO: N={len(resto)} WR={float(resto["is_win"].mean())*100:.1f}% '
      f'EV={float(resto["return_pct"].mean())*100:+.5f}%')

if len(h11) >= 10:
    t_h, p_h = stats.ttest_ind(h11['return_pct'].dropna(), resto['return_pct'].dropna())
    ks_h, p_kh = stats.ks_2samp(h11['return_pct'].dropna(), resto['return_pct'].dropna())
    print(f'  t-test EV: t={t_h:.3f} p={p_h:.4f} → {"DIFERENCIA REAL" if p_h < 0.05 else "no sig."}')
    print(f'  KS-test distribución: KS={ks_h:.4f} p={p_kh:.4f} → {"DISTINTAS" if p_kh < 0.05 else "indistinguibles"}')

# ¿H11 se concentra en W1?
print(f'\nH11 por ventana:')
for w, g in h11.groupby('window'):
    print(f'  {w}: N={len(g)} WR={float(g["is_win"].mean())*100:.1f}%')

print()

# ═══════════════════════════════════════════════════════════════════════════
# H-E: ¿OOD predice pérdida? (Spearman OOD → return_pct)
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('H-E: ¿OOD (KL distance) predice pérdida? Spearman(OOD, return)')
print(SEP)

valid = df_all[['ood_kl_distance','return_pct','is_win']].dropna()
r_ood, p_ood = stats.spearmanr(valid['ood_kl_distance'], valid['return_pct'])
print(f'\nSpearman(OOD, return): r={r_ood:+.4f} p={p_ood:.4f} → '
      f'{"CONFIRMADA" if p_ood < 0.05 else "DESCARTADA"}')

# Sweep por percentil OOD
print('\nWR y EV por percentil OOD:')
df_all['q_ood'] = pd.qcut(df_all['ood_kl_distance'], 4,
                            labels=['p0-25','p25-50','p50-75','p75-100'])
for q, g in df_all.groupby('q_ood', observed=True):
    ood_m = float(g['ood_kl_distance'].mean())
    wr_q  = float(g['is_win'].mean())
    ev_q  = float(g['return_pct'].mean())
    print(f'  {q} (OOD≈{ood_m:.3f}): N={len(g)} WR={wr_q*100:.1f}% EV={ev_q*100:+.5f}%')

# ═══════════════════════════════════════════════════════════════════════════
# H-F: SWEEP DE THRESHOLD EN xgb_prob_cal
# ─ ¿Existe un threshold mayor que mejora EV sin pérdida crítica de N?
# Regla: sweep IS-only sobre datos OOS es informativo pero no decisivo
# ═══════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('H-F: SWEEP DE THRESHOLD xgb_prob_cal → EV (EXPLORATORIO, no IS/OOS)')
print('  AVISO: este sweep es informativo, no se usa como fix directo (riesgo overfitting OOS)')
print(SEP)
print()

for thresh in np.arange(0.50, 0.80, 0.02):
    sub = df_all[df_all['xgb_prob_cal'] >= thresh]
    if len(sub) < 20:
        break
    wr_t = float((sub['return_pct'] > 0).mean())
    ev_t = float(sub['return_pct'].mean())
    n_t  = len(sub)
    binom_p = stats.binomtest(int(sub['is_win'].sum()), n_t, 0.5, alternative='two-sided').pvalue
    print(f'  thresh>={thresh:.2f}: N={n_t:>4} WR={wr_t*100:>5.1f}% '
          f'EV={ev_t*100:>+7.4f}% | binom p={binom_p:.4f} '
          f'{"*sig*" if binom_p < 0.05 else ""}')

print()
print('═'*72)
print('FASES 2+3 COMPLETADAS — Resultados listos para FASE 4 (causa raíz)')
print('═'*72)
