"""
Validación cruzada del gradiente threshold->WR en CALM_BEAR
"""
import pathlib, pandas as pd, numpy as np
from scipy import stats

runs = pathlib.Path('g:/Mi unidad/ia/luna_v2/data/runs')
all_t = sorted(runs.rglob('*/W*/oos_trades.parquet'), key=lambda p: p.stat().st_mtime, reverse=True)
recent = [f for f in all_t if '20260601' in f.parts[-4]]

rows = []
for fp in recent:
    try:
        df = pd.read_parquet(fp)
        df['_window'] = fp.parts[-2]
        rows.append(df)
    except:
        pass
df_all = pd.concat(rows, ignore_index=True)
bear_df = df_all[df_all['hmm_regime'].astype(str).str.contains('CALM_BEAR|3_CALM', na=False)].copy()

print('=== VALIDACION CRUZADA: gradiente threshold-WR por VENTANA ===')
print('(Cada ventana es un OOS independiente — evita sesgo in-sample)')
print()
for w in ['W1', 'W2', 'W3', 'W4']:
    sub = bear_df[bear_df['_window'] == w]
    if len(sub) < 3:
        print(f'  {w}: N={len(sub)} insuficiente')
        continue
    for thr in [0.40, 0.55, 0.61, 0.64, 0.67]:
        sT = sub[sub['xgb_prob_cal'] >= thr]
        wr = (sT['return_pct'] > 0).mean() if len(sT) > 0 else float('nan')
        print(f'  {w} @ thr={thr:.2f}: N={len(sT):>4} WR={wr:.1%}')
    print()

print()
print('=== SPEARMAN xgb_prob_cal -> return en CALM_BEAR (por ventana) ===')
for w in ['W1', 'W2', 'W3', 'W4']:
    sub = bear_df[bear_df['_window'] == w].dropna(subset=['xgb_prob_cal', 'return_pct'])
    if len(sub) < 10:
        print(f'  {w}: N={len(sub)} insuficiente para Spearman')
        continue
    r, p = stats.spearmanr(sub['xgb_prob_cal'], sub['return_pct'])
    sig = 'SIG' if p < 0.05 else 'ruido'
    print(f'  {w}: N={len(sub)} r={r:+.4f} p={p:.4f} -> {sig}')

print()
print('=== BINOM TEST: si se filtrara thr>=0.67 en CALM_BEAR ===')
n67   = 104
wins67 = 65
res  = stats.binomtest(wins67, n67, p=0.50, alternative='greater')
sig  = 'SIGNIFICATIVO' if res.pvalue < 0.05 else 'no sig'
ic_lo = wins67/n67 - 1.96*(wins67/n67*(1-wins67/n67)/n67)**0.5
ic_hi = wins67/n67 + 1.96*(wins67/n67*(1-wins67/n67)/n67)**0.5
print(f'  N={n67} wins={wins67} WR={wins67/n67:.1%} p={res.pvalue:.5f} -> {sig}')
print(f'  IC95 WR: [{ic_lo:.1%}, {ic_hi:.1%}]')

print()
print('=== ADVERTENCIA: es esto in-sample sweep? ===')
print('  SI: las 104 trades se seleccionaron mirando xgb_prob_cal en el mismo OOS')
print('  Para ser valido, el gradiente debe ser consistente across ventanas:')
# Calcula spearman global
valid = bear_df.dropna(subset=['xgb_prob_cal', 'return_pct'])
r, p = stats.spearmanr(valid['xgb_prob_cal'], valid['return_pct'])
print(f'  Spearman global CALM_BEAR: r={r:+.4f} p={p:.4f}')
if p < 0.05:
    print('  -> gradiente ES real (r significativo en datos agregados)')
    print('  -> Subir threshold de CALM_BEAR tiene base estadistica')
else:
    print('  -> gradiente no es significativo -> sweep seria in-sample bias')

print()
print('=== ALPHA_DTW_SIGNAL: correlacion con retorno ===')
if 'alpha_trigger' in df_all.columns:
    dtw = df_all[df_all['alpha_trigger'] == 'alpha_dtw_signal']
    nodtw = df_all[df_all['alpha_trigger'] != 'alpha_dtw_signal']
    print(f'  CON DTW: N={len(dtw)} WR={(dtw.return_pct>0).mean():.1%} '
          f'EV={dtw.return_pct.mean()*100:+.6f}%')
    print(f'  SIN DTW: N={len(nodtw)} WR={(nodtw.return_pct>0).mean():.1%} '
          f'EV={nodtw.return_pct.mean()*100:+.6f}%')
    t, p = stats.ttest_ind(dtw['return_pct'], nodtw['return_pct'])
    print(f'  t-test DTW vs no-DTW: t={t:.3f} p={p:.4f} -> {"SIG DIFERENCIA" if p < 0.05 else "sin diferencia"}')
    for regime in ['1_BULL_TREND', '3_CALM_BEAR']:
        rs = df_all[df_all['hmm_regime'] == regime]
        rd = rs[rs['alpha_trigger'] == 'alpha_dtw_signal']
        rn = rs[rs['alpha_trigger'] != 'alpha_dtw_signal']
        if len(rd) > 5 and len(rn) > 5:
            print(f'  {regime}: CON_DTW WR={(rd.return_pct>0).mean():.1%} N={len(rd)} | '
                  f'SIN_DTW WR={(rn.return_pct>0).mean():.1%} N={len(rn)}')

print()
print('=== RANGE W3 ONLY: inspeccion HMM labels ===')
range_df = df_all[df_all['hmm_regime'].astype(str).str.contains('RANGE', na=False)]
print(f'  RANGE trades por ventana: {range_df["_window"].value_counts().to_dict()}')
print(f'  RANGE xgb_prob_cal stats: min={range_df.xgb_prob_cal.min():.3f} '
      f'mean={range_df.xgb_prob_cal.mean():.3f} max={range_df.xgb_prob_cal.max():.3f}')
print(f'  RANGE all in W3 -> modelo RANGE entrenado con IS hasta 2024-06:')
print(f'  -> Aprendio los rebotes en RANGE de 2019-2024, aplica SOLO en 2024H2')
print(f'  -> En W1 (2023H2) y W2 (2024H1), el modelo RANGE falla al clasificar OOS')
