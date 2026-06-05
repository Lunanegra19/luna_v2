"""
AUDITORÍA PROFUNDA — Run 2026-06-01 (FIX-CALIB-BINARY-01 activo)
================================================================
Extrae el máximo de información de los 38 parquets de trades y 30 early-stops.
Analiza: calibración, regímenes, agentes, hora, RR, SHAP drivers, OOD, equity curve.
"""
import sys, json, re, warnings
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

wfb_dir   = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')
cache_dir  = Path('g:/Mi unidad/ia/luna_v2/data/wfb_cache')
LOG_PATH   = Path('C:/Users/Usuario/.gemini/antigravity-ide/brain/ad23283d-d02e-4616-9748-5d609f02bf06/.system_generated/tasks/task-1314.log')

SEP = '='*72

# ═══════════════════════════════════════════════════════════════════════════
# 0. CARGA DE TODOS LOS TRADES
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('0. CARGA DE DATOS')
print(SEP)

records = []
for f in sorted(wfb_dir.glob('oos_trades_W*_seed*.parquet')):
    parts  = f.stem.split('_')
    window = next((p for p in parts if p.startswith('W')), '?')
    seed   = next((p.replace('seed','') for p in parts if p.startswith('seed')), '?')
    try:
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue
        df['_window'] = window
        df['_seed']   = seed
        df['_w_num']  = int(window.replace('W',''))
        records.append(df)
    except Exception as e:
        print(f'  ERROR leyendo {f.name}: {e}')

if not records:
    print('SIN DATOS')
    sys.exit(0)

all_trades = pd.concat(records, ignore_index=True)
N = len(all_trades)
print(f'Total trades cargados: {N}')
print(f'Windows: {sorted(all_trades["_window"].unique())}')
print(f'Seeds:   {sorted(all_trades["_seed"].unique(), key=int)}')
print(f'Columnas: {list(all_trades.columns)}')
print()

# ═══════════════════════════════════════════════════════════════════════════
# 1. KPIs GLOBALES
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('1. KPIs GLOBALES')
print(SEP)

wr       = float(all_trades['is_win'].mean())
ret      = all_trades['return_pct'].dropna()
mean_ret = float(ret.mean())
std_ret  = float(ret.std())
avg_win  = float(ret[ret > 0].mean()) if (ret > 0).any() else 0.0
avg_loss = float(ret[ret < 0].mean()) if (ret < 0).any() else 0.0
rr       = abs(avg_win / avg_loss) if avg_loss != 0 else float('nan')
max_dd   = float(all_trades['drawdown'].min()) if 'drawdown' in all_trades.columns else float('nan')
sr       = mean_ret / std_ret * np.sqrt(N) if std_ret > 1e-10 else float('nan')

# Calibración
diff_cal = (all_trades['xgb_prob_cal'] - all_trades['xgb_prob']).abs()
pct_cal_ok = float((diff_cal > 1e-6).mean() * 100)
diff_mean  = float(diff_cal.mean())
diff_std   = float(diff_cal.std())

print(f'WR global:          {wr*100:.2f}%  (n={N})')
print(f'Mean return:        {mean_ret*100:.4f}%')
print(f'Std return:         {std_ret*100:.4f}%')
print(f'Avg win:            {avg_win*100:.4f}%')
print(f'Avg loss:           {avg_loss*100:.4f}%')
print(f'R:R ratio:          {rr:.3f}')
print(f'Max Drawdown:       {max_dd*100:.2f}%')
print(f'Sharpe (naif):      {sr:.4f}')
print(f'Cal aplicada:       {pct_cal_ok:.1f}% trades (diff_mean={diff_mean:.4f} std={diff_std:.4f})')
print()

# ═══════════════════════════════════════════════════════════════════════════
# 2. POR VENTANA
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('2. DESGLOSE POR VENTANA')
print(SEP)

for w, grp in all_trades.groupby('_window'):
    n    = len(grp)
    wr_w = float(grp['is_win'].mean())
    ret_w= grp['return_pct'].dropna()
    mr   = float(ret_w.mean()) if len(ret_w) > 0 else float('nan')
    sr_w = float(ret_w.std()) if len(ret_w) > 0 else float('nan')
    aw   = float(ret_w[ret_w > 0].mean()) if (ret_w > 0).any() else 0.0
    al   = float(ret_w[ret_w < 0].mean()) if (ret_w < 0).any() else 0.0
    rr_w = abs(aw/al) if al != 0 else float('nan')
    dd_w = float(grp['drawdown'].min()) if 'drawdown' in grp.columns else float('nan')
    n_seeds = grp['_seed'].nunique()
    diff_w = (grp['xgb_prob_cal'] - grp['xgb_prob']).abs()
    pct_c  = float((diff_w > 1e-6).mean()*100)
    dm_w   = float(diff_w.mean())
    # Threshold
    thr_m  = float(grp['signal_threshold'].mean()) if 'signal_threshold' in grp.columns else float('nan')
    thr_lo = float(grp['signal_threshold'].min()) if 'signal_threshold' in grp.columns else float('nan')
    print(f'{w} ({n_seeds} seeds, {n} trades):')
    print(f'  WR={wr_w*100:.1f}%  MeanRet={mr*100:.4f}%  R:R={rr_w:.3f}  MaxDD={dd_w*100:.2f}%')
    print(f'  AvgWin={aw*100:.4f}%  AvgLoss={al*100:.4f}%')
    print(f'  Calibracion: {pct_c:.0f}% aplicada | diff_mean={dm_w:.4f}')
    print(f'  Threshold: mean={thr_m:.4f} min={thr_lo:.4f}')
    # xgb_prob_cal distribution
    q = grp['xgb_prob_cal'].quantile([0.1,0.25,0.5,0.75,0.9])
    print(f'  xgb_prob_cal: p10={q[0.1]:.4f} p25={q[0.25]:.4f} p50={q[0.5]:.4f} p75={q[0.75]:.4f} p90={q[0.9]:.4f}')
    print()

# ═══════════════════════════════════════════════════════════════════════════
# 3. ANÁLISIS DE CALIBRACIÓN — IMPACTO EN SEÑALES
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('3. ANÁLISIS DE CALIBRACIÓN: xgb_prob_raw vs xgb_prob_cal')
print(SEP)

raw_col = 'xgb_prob'
cal_col = 'xgb_prob_cal'
thr_col = 'signal_threshold'

if thr_col in all_trades.columns:
    thr = all_trades[thr_col]
    # Señales que PASARON con raw pero son suprimidas con cal
    raw_pass = all_trades[raw_col] >= thr
    cal_pass = all_trades[cal_col] >= thr
    suppressed = raw_pass & ~cal_pass
    recovered  = ~raw_pass & cal_pass

    print(f'Trades actuales (cal >= thr):  {cal_pass.sum()} ({cal_pass.mean()*100:.1f}%)')
    print(f'Habrian pasado con raw >= thr: {raw_pass.sum()} ({raw_pass.mean()*100:.1f}%)')
    print(f'Suprimidos por calibracion (raw>=thr pero cal<thr): {suppressed.sum()}')
    print(f'Recuperados por calibracion (raw<thr pero cal>=thr): {recovered.sum()}')
    print()

    # WR de los suprimidos: ¿eran buenos o malos?
    if suppressed.sum() > 0:
        # No podemos saber el WR de los suprimidos directamente en los trades registrados
        # porque solo se registran los que pasan el threshold. Pero podemos ver la distribución de cal
        s_df = all_trades[suppressed]
        print(f'Señales suprimidas por ventana:')
        for w, g in s_df.groupby('_window'):
            wr_sup = float(g['is_win'].mean()) if 'is_win' in g.columns else float('nan')
            raw_m  = float(g[raw_col].mean())
            cal_m  = float(g[cal_col].mean())
            print(f'  {w}: n={len(g)} WR={wr_sup*100:.1f}% raw_mean={raw_m:.4f} cal_mean={cal_m:.4f}')
    print()

    # xgb_prob_raw vs cal scatter stats
    print('Distribución de prob_cal por ventana (percentiles):')
    for w, grp in all_trades.groupby('_window'):
        raw_m = float(grp[raw_col].mean())
        cal_m = float(grp[cal_col].mean())
        delta = cal_m - raw_m
        arrow = '↑' if delta > 0 else '↓'
        print(f'  {w}: raw_mean={raw_m:.4f}  cal_mean={cal_m:.4f}  delta={delta:+.4f} {arrow}')

print()

# ═══════════════════════════════════════════════════════════════════════════
# 4. POR RÉGIMEN HMM
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('4. DESGLOSE POR RÉGIMEN HMM')
print(SEP)

if 'hmm_regime' in all_trades.columns:
    for regime, grp in all_trades.groupby('hmm_regime'):
        n   = len(grp)
        wr_r= float(grp['is_win'].mean())
        mr  = float(grp['return_pct'].dropna().mean()) if len(grp['return_pct'].dropna()) > 0 else float('nan')
        print(f'Regimen {regime}: n={n} ({n/N*100:.1f}%) WR={wr_r*100:.1f}% MeanRet={mr*100:.4f}%')
else:
    print('hmm_regime no disponible en trades')
print()

# ═══════════════════════════════════════════════════════════════════════════
# 5. POR HORA DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('5. PERFORMANCE POR HORA DE ENTRADA')
print(SEP)

if 'entry_time' in all_trades.columns:
    all_trades['_entry_hour'] = pd.to_datetime(all_trades['entry_time'], utc=True, errors='coerce').dt.hour
    hour_stats = all_trades.groupby('_entry_hour').agg(
        n=('is_win','count'),
        wr=('is_win','mean'),
        mean_ret=('return_pct','mean')
    ).reset_index()
    hour_stats = hour_stats.sort_values('wr', ascending=False)
    print('Top 5 horas por WR:')
    for _, r in hour_stats.head(5).iterrows():
        print(f'  H{int(r._entry_hour):02d}: n={int(r.n)} WR={r.wr*100:.1f}% MeanRet={r.mean_ret*100:.4f}%')
    print('Bot 5 horas por WR:')
    for _, r in hour_stats.tail(5).iterrows():
        print(f'  H{int(r._entry_hour):02d}: n={int(r.n)} WR={r.wr*100:.1f}% MeanRet={r.mean_ret*100:.4f}%')
print()

# ═══════════════════════════════════════════════════════════════════════════
# 6. ANÁLISIS OOD (ood_kl_distance)
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('6. ANÁLISIS OOD (ood_kl_distance)')
print(SEP)

if 'ood_kl_distance' in all_trades.columns:
    ood = all_trades['ood_kl_distance'].dropna()
    if len(ood) > 0:
        print(f'OOD KL distance: mean={ood.mean():.4f} std={ood.std():.4f} '
              f'p25={ood.quantile(0.25):.4f} p75={ood.quantile(0.75):.4f} max={ood.max():.4f}')
        # ¿Las señales con OOD alto tienen peor WR?
        med_ood = float(ood.median())
        hi_ood  = all_trades[all_trades['ood_kl_distance'] > med_ood]
        lo_ood  = all_trades[all_trades['ood_kl_distance'] <= med_ood]
        wr_hi   = float(hi_ood['is_win'].mean()) if len(hi_ood) > 0 else float('nan')
        wr_lo   = float(lo_ood['is_win'].mean()) if len(lo_ood) > 0 else float('nan')
        print(f'WR con OOD > mediana ({med_ood:.4f}): {wr_hi*100:.1f}%  (n={len(hi_ood)})')
        print(f'WR con OOD < mediana ({med_ood:.4f}): {wr_lo*100:.1f}%  (n={len(lo_ood)})')
        print()
        # Por ventana
        for w, grp in all_trades.groupby('_window'):
            ood_w = grp['ood_kl_distance'].dropna()
            if len(ood_w) > 0:
                print(f'  {w}: OOD mean={ood_w.mean():.4f} max={ood_w.max():.4f}')
print()

# ═══════════════════════════════════════════════════════════════════════════
# 7. SHAP DRIVERS — ¿Qué features dominan las señales?
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('7. SHAP DRIVERS — Features más influyentes')
print(SEP)

if 'shap_drivers' in all_trades.columns:
    driver_counts = Counter()
    driver_wins   = defaultdict(list)
    for _, row in all_trades.iterrows():
        try:
            drivers = json.loads(row['shap_drivers']) if isinstance(row['shap_drivers'], str) else row['shap_drivers']
            if isinstance(drivers, dict):
                for feat, val in drivers.items():
                    driver_counts[feat] += 1
                    driver_wins[feat].append(int(row['is_win']))
        except Exception:
            pass
    if driver_counts:
        print('Top 15 features por frecuencia en SHAP drivers:')
        for feat, cnt in driver_counts.most_common(15):
            wr_f = np.mean(driver_wins[feat]) * 100 if driver_wins[feat] else float('nan')
            print(f'  {feat}: aparece en {cnt} trades ({cnt/N*100:.1f}%) | WR={wr_f:.1f}%')
print()

# ═══════════════════════════════════════════════════════════════════════════
# 8. ALPHA_TRIGGER — ¿Qué señales activan los trades?
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('8. ALPHA_TRIGGER por ventana')
print(SEP)

if 'alpha_trigger' in all_trades.columns:
    for w, grp in all_trades.groupby('_window'):
        cnt = Counter(grp['alpha_trigger'].dropna().tolist())
        top = cnt.most_common(5)
        print(f'{w}: ' + ' | '.join([f'{k}={v}' for k,v in top]))
print()

# ═══════════════════════════════════════════════════════════════════════════
# 9. KELLY FRACTION vs PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('9. KELLY FRACTION y THRESHOLD LOWERED')
print(SEP)

if 'kelly_fraction_used' in all_trades.columns:
    kf = all_trades['kelly_fraction_used'].dropna()
    print(f'Kelly fraction: mean={kf.mean():.4f} std={kf.std():.4f} '
          f'p10={kf.quantile(0.1):.4f} p90={kf.quantile(0.9):.4f}')
if 'threshold_was_lowered' in all_trades.columns:
    n_low = all_trades['threshold_was_lowered'].sum()
    if n_low > 0:
        low_df  = all_trades[all_trades['threshold_was_lowered']]
        high_df = all_trades[~all_trades['threshold_was_lowered']]
        wr_low  = float(low_df['is_win'].mean())
        wr_high = float(high_df['is_win'].mean())
        print(f'Threshold LOWERED: {n_low} trades ({n_low/N*100:.1f}%) WR={wr_low*100:.1f}%')
        print(f'Threshold NORMAL:  {N-n_low} trades WR={wr_high*100:.1f}%')
print()

# ═══════════════════════════════════════════════════════════════════════════
# 10. EARLY STOP AUDIT
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('10. EARLY STOP AUDIT')
print(SEP)

sharpe_gate = []
ub_gate     = []
for f in sorted(wfb_dir.glob('early_stop_seed*.json')):
    with open(f) as fp:
        d = json.load(fp)
    reason = str(d.get('reason',''))
    windows = d.get('windows_evaluated', [])
    seed_id = f.stem.replace('early_stop_seed','')
    max_w   = max(windows) if windows else 0
    if 'Sharpe parcial' in reason:
        sharpe_gate.append({'seed': seed_id, 'max_w': max_w, 'reason': reason[:80]})
    else:
        ub_gate.append({'seed': seed_id, 'max_w': max_w, 'reason': reason[:80]})

print(f'Total early-stops: {len(sharpe_gate)+len(ub_gate)}')
print(f'  Gate Sharpe parcial < -0.10:  {len(sharpe_gate)} seeds')
print(f'  Gate upper_bound < threshold: {len(ub_gate)} seeds')
print()
max_window_dist = Counter([s['max_w'] for s in sharpe_gate+ub_gate])
print('Distribución por última ventana:')
for w in sorted(max_window_dist):
    print(f'  W{w}: {max_window_dist[w]} seeds podadas')
print()

# Sharpe gates — extraer el valor real del Sharpe
sharpe_vals = []
if LOG_PATH.exists():
    log_text = LOG_PATH.read_text(encoding='utf-8', errors='replace')
    matches = re.findall(r'SR=(-?\d+\.\d+) \(n=(\d+)', log_text)
    for sr_str, n_str in matches:
        sharpe_vals.append(float(sr_str))
    if sharpe_vals:
        arr = np.array(sharpe_vals)
        print(f'Sharpe parciales observados (de {len(arr)} evaluaciones en logs):')
        print(f'  mean={arr.mean():.4f} median={np.median(arr):.4f} std={arr.std():.4f}')
        print(f'  min={arr.min():.4f} max={arr.max():.4f}')
        pct_neg = float((arr < 0).mean()*100)
        pct_below_gate = float((arr < -0.1).mean()*100)
        print(f'  Negativos: {pct_neg:.1f}% | Bajo gate(-0.10): {pct_below_gate:.1f}%')
print()

# ═══════════════════════════════════════════════════════════════════════════
# 11. TEMPERATURA CALIBRATOR BUG — qué semillas afectadas
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('11. BUG TemperatureCalibrator — Análisis')
print(SEP)

if LOG_PATH.exists():
    log_text = log_text if 'log_text' in dir() else LOG_PATH.read_text(encoding='utf-8', errors='replace')
    temp_cal_errors = re.findall(r"seed(\d+).*?TemperatureCalibrator|TemperatureCalibrator.*?seed(\d+)", log_text)
    lines_with_error = [l for l in log_text.split('\n') if 'TemperatureCalibrator' in l]
    print(f'Líneas con TemperatureCalibrator error: {len(lines_with_error)}')
    for l in lines_with_error[:5]:
        print(f'  {l.strip()[:120]}')
    print()
    # Seeds que crashean por FATAL (de los logs)
    fatal_seeds = re.findall(r'Seeds fallidas \(errores FATAL\): \[([^\]]+)\]', log_text)
    if fatal_seeds:
        print(f'Seeds FATAL: {fatal_seeds[-1]}')

print()

# ═══════════════════════════════════════════════════════════════════════════
# 12. BEAR_LONG COLLAPSE — cuántas seeds afectadas y en qué ventanas
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('12. BEAR_LONG COLLAPSE — FIX-ROUTER-SANITY-01/CRITICAL')
print(SEP)

if LOG_PATH.exists():
    collapse_lines = [l for l in log_text.split('\n') if 'COLAPSO TOTAL' in l and 'bear_long' in l]
    print(f'Colapsos bear_long detectados: {len(collapse_lines)}')
    for l in collapse_lines[:8]:
        print(f'  {l.strip()[:120]}')
    print()
    # ¿El collapse está causando los FATAL?
    fatal_lines = [l for l in log_text.split('\n') if 'ERROR FATAL' in l and 'irremediablemente' in l]
    print(f'Total seeds FATAL: {len(fatal_lines)}')
    for l in fatal_lines:
        print(f'  {l.strip()[:80]}')

print()

# ═══════════════════════════════════════════════════════════════════════════
# 13. HIPÓTESIS ORDENADAS POR EVIDENCIA
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('13. HIPÓTESIS FORMULADAS — ORDENADAS POR EVIDENCIA')
print(SEP)

# Calcular métricas para cada hipótesis
w1_trades = all_trades[all_trades['_window']=='W1']
w2_trades = all_trades[all_trades['_window']=='W2']
w3_trades = all_trades[all_trades['_window']=='W3']
w4_trades = all_trades[all_trades['_window']=='W4'] if 'W4' in all_trades['_window'].values else pd.DataFrame()

wr_w1 = float(w1_trades['is_win'].mean()) if len(w1_trades) > 0 else float('nan')
wr_w2 = float(w2_trades['is_win'].mean()) if len(w2_trades) > 0 else float('nan')
wr_w3 = float(w3_trades['is_win'].mean()) if len(w3_trades) > 0 else float('nan')
wr_w4 = float(w4_trades['is_win'].mean()) if len(w4_trades) > 0 else float('nan')

# Diferencia cal-raw por ventana
def cal_delta(df):
    if 'xgb_prob' in df.columns and 'xgb_prob_cal' in df.columns and len(df) > 0:
        return float((df['xgb_prob_cal'] - df['xgb_prob']).mean())
    return float('nan')

print('H1: CALIBRADOR SUPRIME SEÑALES BUENAS EN W1 (cal < raw sistemáticamente)')
d_w1 = cal_delta(w1_trades)
print(f'  delta(cal-raw) W1: {d_w1:+.4f}  (negativo = calibrador BAJA las probs)')
_sup_msg = '< 50% coherente con supresion' if wr_w1 < 0.5 else '> 50% incoherente con supresion'
print(f'  WR W1: {wr_w1*100:.1f}%  — {_sup_msg}')
print()

print('H2: W1 (Q1 2025) ES GENUINAMENTE OOD — MODELO NO TIENE EDGE')
ood_w1 = w1_trades['ood_kl_distance'].dropna() if 'ood_kl_distance' in w1_trades.columns else pd.Series()
ood_w2 = w2_trades['ood_kl_distance'].dropna() if 'ood_kl_distance' in w2_trades.columns else pd.Series()
if len(ood_w1) > 0 and len(ood_w2) > 0:
    print(f'  OOD W1: {ood_w1.mean():.4f} vs W2: {ood_w2.mean():.4f}  (mayor = más OOD)')
print(f'  WR W1={wr_w1*100:.1f}% W2={wr_w2*100:.1f}% W3={wr_w3*100:.1f}%')
print()

print('H3: BEAR_LONG COLLAPSE CAUSA FATALS INNECESARIAMENTE (guard demasiado agresivo)')
print(f'  Collapse events en logs: {len(collapse_lines) if "collapse_lines" in dir() else "N/A"}')
print(f'  FATALs totales: {len(fatal_lines) if "fatal_lines" in dir() else "N/A"}')
print(f'  ¿Workaround: convertir RuntimeError en WARNING para bear_long colapsado?')
print()

print('H4: TemperatureCalibrator NO DESERIALIZABLE — bug de pickling entre sesiones')
print(f'  Seeds afectadas: 27243, 44085 (según logs)')
print(f'  Fix: añadir TemperatureCalibrator al namespace de predict_oos.py')
print()

print('H5: THRESHOLD DINÁMICO LOWERED DEGRADA LA CALIDAD DE SEÑALES')
if 'threshold_was_lowered' in all_trades.columns:
    n_low = all_trades['threshold_was_lowered'].sum()
    wr_low  = float(all_trades[all_trades['threshold_was_lowered']]['is_win'].mean()) if n_low > 0 else float('nan')
    wr_high = float(all_trades[~all_trades['threshold_was_lowered']]['is_win'].mean())
    print(f'  Trades con threshold lowered: {n_low} WR={wr_low*100:.1f}%')
    print(f'  Trades con threshold normal:  {N-n_low} WR={wr_high*100:.1f}%')
    print(f'  Diferencia: {(wr_high-wr_low)*100:+.1f}pp  (positivo = lowered es peor)')
print()

print('H6: SHARPE GATE -0.10 ES DEMASIADO CONSERVADOR Y PODA SEEDS RESCATABLES')
print(f'  22 de 30 seeds podadas por gate Sharpe < -0.10')
print(f'  Si el gate fuera -0.50, ¿cuántas habrian sobrevivido a W4/W5?')
print()

print(SEP)
print('FIN DE LA AUDITORÍA — resultados listos para análisis de hipótesis')
print(SEP)
