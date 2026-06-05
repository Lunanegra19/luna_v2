"""
Test estadístico alpha_dtw_signal efecto en CALM_BEAR
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

bear  = df_all[df_all['hmm_regime'].astype(str).str.contains('CALM_BEAR|3_CALM', na=False)]
dtw   = bear[bear['alpha_trigger'] == 'alpha_dtw_signal']
nodtw = bear[bear['alpha_trigger'] != 'alpha_dtw_signal']

print('=== TEST ESTADISTICO: alpha_dtw vs no-dtw en CALM_BEAR ===')
wr_dtw   = (dtw['return_pct'] > 0).mean()
wr_nodtw = (nodtw['return_pct'] > 0).mean()
print(f'CON DTW:  N={len(dtw):>4} wins={(dtw.return_pct>0).sum():>3} WR={wr_dtw:.1%} EV={dtw.return_pct.mean()*100:+.5f}%')
print(f'SIN DTW:  N={len(nodtw):>4} wins={(nodtw.return_pct>0).sum():>3} WR={wr_nodtw:.1%} EV={nodtw.return_pct.mean()*100:+.5f}%')
print()

# H_DTW: WR de DTW en CALM_BEAR < 50% (perdedor sistematico)
n_dtw_wins = int((dtw['return_pct'] > 0).sum())
res_dtw    = stats.binomtest(n_dtw_wins, len(dtw), p=0.50, alternative='less')
verdict_dtw = 'CONFIRMADO' if res_dtw.pvalue < 0.05 else 'descartado'
print(f'binom_test DTW-WR < 50%:    p={res_dtw.pvalue:.5f} -> {verdict_dtw}')

# H_NODTW: WR no-DTW > 50% (ganador con edge real)
n_nodtw_wins = int((nodtw['return_pct'] > 0).sum())
res_nodtw    = stats.binomtest(n_nodtw_wins, len(nodtw), p=0.50, alternative='greater')
ic_lo = n_nodtw_wins/len(nodtw) - 1.96*(n_nodtw_wins/len(nodtw)*(1-n_nodtw_wins/len(nodtw))/len(nodtw))**0.5
ic_hi = n_nodtw_wins/len(nodtw) + 1.96*(n_nodtw_wins/len(nodtw)*(1-n_nodtw_wins/len(nodtw))/len(nodtw))**0.5
verdict_nodtw = 'CONFIRMADO' if res_nodtw.pvalue < 0.05 else 'descartado'
print(f'binom_test no-DTW-WR > 50%: p={res_nodtw.pvalue:.6f} -> {verdict_nodtw}')
print(f'  IC95 WR no-DTW: [{ic_lo:.1%}, {ic_hi:.1%}]')

# t-test entre grupos (solo CALM_BEAR)
t, p_diff = stats.ttest_ind(dtw['return_pct'].dropna(), nodtw['return_pct'].dropna())
verdict_t = 'DIFERENCIA SIG.' if p_diff < 0.05 else 'no sig'
print(f't-test DTW vs no-DTW (CALM_BEAR only): t={t:.3f} p={p_diff:.5f} -> {verdict_t}')

print()
print('=== KS TEST: distribucion retornos DTW vs no-DTW ===')
ks, p_ks = stats.ks_2samp(dtw['return_pct'].dropna(), nodtw['return_pct'].dropna())
verdict_ks = 'DISTRIBUCIONES DISTINTAS' if p_ks < 0.05 else 'misma distribucion'
print(f'KS: stat={ks:.4f} p={p_ks:.5f} -> {verdict_ks}')

print()
print('=== CONTRAFACTUAL: eliminar DTW en CALM_BEAR ===')
ev_actual  = bear['return_pct'].sum() * 100
ev_nodtw   = nodtw['return_pct'].sum() * 100
ev_dtw_bad = dtw['return_pct'].sum() * 100
print(f'Actual (todos):        N={len(bear):>4} WR={(bear.return_pct>0).mean():.1%} total_EV={ev_actual:+.3f}%')
print(f'Solo no-DTW:           N={len(nodtw):>4} WR={wr_nodtw:.1%}  total_EV={ev_nodtw:+.3f}%')
print(f'Retorno trades DTW:    {ev_dtw_bad:+.3f}%  (esta cantidad se recuperaria eliminando DTW CALM_BEAR)')

print()
print('=== LOCALIZACION DE alpha_dtw EN EL CODIGO ===')
luna_path = pathlib.Path('g:/Mi unidad/ia/luna_v2/luna')
found = {}
for pat in ['alpha_dtw', 'dtw_signal', 'alpha_trigger']:
    for f in sorted(luna_path.rglob('*.py')):
        try:
            lines = f.read_text(encoding='utf-8', errors='ignore').split('\n')
            hits = [(i+1, l.strip()) for i, l in enumerate(lines) if pat in l and not l.strip().startswith('#')]
            if hits:
                found.setdefault(f.name, []).extend(hits[:3])
        except:
            pass

for fname, hits in list(found.items())[:8]:
    print(f'  {fname}:')
    for lineno, content in hits[:2]:
        print(f'    L{lineno}: {content[:80]}')
