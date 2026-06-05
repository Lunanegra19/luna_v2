"""
FASE 2 + FASE 3 — Hipótesis y Tests Estadísticos
Siguiendo diagnostico_cuantitativo.md
Targets: P1 (bull_gate_min_dsr) y P2 (RANGE threshold)
"""
import pathlib, pandas as pd, numpy as np, json
from scipy import stats

runs = pathlib.Path('g:/Mi unidad/ia/luna_v2/data/runs')

# ─── Cargar datos base ────────────────────────────────────────────────────
all_trades_raw = sorted(runs.rglob('*/W*/oos_trades.parquet'),
    key=lambda p: p.stat().st_mtime, reverse=True)
recent = [f for f in all_trades_raw if '20260601' in f.parts[-4]]
rows = []
for fp in recent:
    try:
        df = pd.read_parquet(fp)
        if len(df) < 1:
            continue
        df['_window'] = fp.parts[-2]
        df['_run']    = fp.parts[-4]
        df['_seed']   = fp.parts[-3]
        rows.append(df)
    except:
        pass
df_all = pd.concat(rows, ignore_index=True)
hmm_col = 'hmm_regime'

bull_df  = df_all[df_all[hmm_col].astype(str).str.contains('BULL', na=False)].copy()
bear_df  = df_all[df_all[hmm_col].astype(str).str.contains('BEAR|CALM', na=False)].copy()
range_df = df_all[df_all[hmm_col].astype(str).str.contains('RANGE', na=False)].copy()

SEP = '=' * 68

print(SEP)
print('HIPÓTESIS FORMULADAS Y TESTS ESTADÍSTICOS (FASE 2+3)')
print(SEP)

# ════════════════════════════════════════════════════════════════════════
# H1: "El EV de BULL es significativamente negativo (no es ruido)"
# Consecuencia: si H1 es cierta, WR=41.8% es significativamente < 50%
# Test: binom_test (prueba de WR=50%)
# ════════════════════════════════════════════════════════════════════════
print('\nH1: EV de BULL es significativamente negativo — WR significativamente < 50%')
print(f'    N={len(bull_df)} trades BULL | WR=41.8%')
n_bull  = len(bull_df)
n_wins  = (bull_df['return_pct'] > 0).sum()
res_h1  = stats.binomtest(n_wins, n_bull, p=0.50, alternative='less')
p_h1    = res_h1.pvalue
verdict = 'CONFIRMADA' if p_h1 < 0.05 else 'DESCARTADA'
print(f'    binom_test: n_wins={n_wins}/{n_bull}, p={p_h1:.6f} → H1 {verdict}')
# t-test adicional: EV vs 0
t, p_ev = stats.ttest_1samp(bull_df['return_pct'], 0)
print(f'    t-test EV vs 0: t={t:.3f}, p={p_ev:.6f} → EV {"SIG.NEG." if p_ev < 0.05 and t < 0 else "NO SIG."}')

# ════════════════════════════════════════════════════════════════════════
# H2: "Mayor xgb_prob_cal en BULL se correlaciona con mejor WR"
# Consecuencia: si H2 es falsa, el modelo BULL no discrimina calidad
# Test: Spearman entre xgb_prob_cal y return_pct
# ════════════════════════════════════════════════════════════════════════
print('\nH2: xgb_prob_cal en BULL correlaciona con return_pct (el modelo discrimina)')
if 'xgb_prob_cal' in bull_df.columns:
    valid = bull_df[['xgb_prob_cal', 'return_pct']].dropna()
    r, p_h2 = stats.spearmanr(valid['xgb_prob_cal'], valid['return_pct'])
    verdict2 = 'CONFIRMADA' if p_h2 < 0.05 else 'DESCARTADA'
    print(f'    Spearman: r={r:+.4f}, p={p_h2:.6f} → H2 {verdict2}')
    # Sweep threshold para ver si mayor prob_cal = mejor WR
    print('    Sweep xgb_prob_cal threshold en BULL (contrafactual P1):')
    print(f'    {"thresh":>8} {"N":>6} {"WR":>7} {"EV/trade":>10} {"acum_ret":>10}')
    for thr in np.arange(0.48, 0.80, 0.04):
        sub = bull_df[bull_df['xgb_prob_cal'] >= thr]
        if len(sub) < 10:
            continue
        wr  = (sub['return_pct'] > 0).mean()
        ev  = sub['return_pct'].mean() * 100
        ret = sub['return_pct'].sum() * 100
        print(f'    {thr:>8.2f} {len(sub):>6} {wr:>7.1%} {ev:>+10.5f}% {ret:>+10.3f}%')

# ════════════════════════════════════════════════════════════════════════
# H3: "CALM_BEAR EV es significativamente positivo (WR > 50%)"
# Consecuencia: si H3 es cierta, el agente tiene edge real estadístico
# Test: binom_test(WR > 50%)
# ════════════════════════════════════════════════════════════════════════
print('\nH3: CALM_BEAR WR es significativamente > 50% (edge real)')
n_bear  = len(bear_df)
n_bear_wins = (bear_df['return_pct'] > 0).sum()
res_h3  = stats.binomtest(n_bear_wins, n_bear, p=0.50, alternative='greater')
p_h3    = res_h3.pvalue
verdict3 = 'CONFIRMADA' if p_h3 < 0.05 else 'DESCARTADA'
print(f'    N={n_bear} trades BEAR | n_wins={n_bear_wins} | WR={n_bear_wins/n_bear:.1%}')
print(f'    binom_test (WR > 50%): p={p_h3:.6f} → H3 {verdict3}')
# IC95 de WR
from math import sqrt
ic95_lo = n_bear_wins/n_bear - 1.96*sqrt(n_bear_wins/n_bear*(1-n_bear_wins/n_bear)/n_bear)
ic95_hi = n_bear_wins/n_bear + 1.96*sqrt(n_bear_wins/n_bear*(1-n_bear_wins/n_bear)/n_bear)
print(f'    IC95 WR: [{ic95_lo:.1%}, {ic95_hi:.1%}]')

# ════════════════════════════════════════════════════════════════════════
# H4: "RANGE WR=100% (N=19) es significativamente > 50% o es ruido"
# Test: binom_test(N=19, wins=19)
# ════════════════════════════════════════════════════════════════════════
print('\nH4: RANGE WR=100% (N=19) es estadísticamente significativo')
n_range  = len(range_df)
n_range_wins = (range_df['return_pct'] > 0).sum()
if n_range > 0:
    res_h4  = stats.binomtest(n_range_wins, n_range, p=0.50, alternative='greater')
    p_h4    = res_h4.pvalue
    verdict4 = f'CONFIRMADA (p={p_h4:.5f})' if p_h4 < 0.05 else f'DESCARTADA (p={p_h4:.5f})'
    print(f'    N={n_range} trades RANGE | n_wins={n_range_wins} | WR={n_range_wins/n_range:.1%}')
    print(f'    binom_test: p={p_h4:.6f} → H4 {verdict4}')
    # Con N=19 y WR=100%, IC95 incluye 80%-100%
    ic_lo = n_range_wins/n_range - 1.96*sqrt(n_range_wins/n_range*(1-n_range_wins/n_range+1e-9)/n_range)
    print(f'    N insuficiente para IC95 estándar — resultado exploratorio')

# ════════════════════════════════════════════════════════════════════════
# H5: "Bajar threshold RANGE de 0.62 a 0.54 mantendría WR > 50%"
# Test: sweep threshold sobre prob_range en barras donde regime=RANGE
# ════════════════════════════════════════════════════════════════════════
print('\nH5: Bajar threshold RANGE de 0.62→0.54 mantiene WR > 50% (sweep counterfactual)')
if 'xgb_prob_cal' in range_df.columns and len(range_df) > 0:
    print('    Sweep xgb_prob_cal threshold en RANGE (N disponible):')
    print(f'    {"thresh":>8} {"N":>5} {"WR":>7} {"EV/trade":>10}')
    for thr in np.arange(0.30, 0.70, 0.04):
        sub = range_df[range_df['xgb_prob_cal'] >= thr]
        if len(sub) < 3:
            print(f'    {thr:>8.2f} {len(sub):>5} (insuficiente)')
            continue
        wr = (sub['return_pct'] > 0).mean()
        ev = sub['return_pct'].mean() * 100
        print(f'    {thr:>8.2f} {len(sub):>5} {wr:>7.1%} {ev:>+10.5f}%')
    print()
    print('    NOTA: N=19 total en RANGE — sweep exploratorio únicamente.')
    print('    Se necesita N>=30 para conclusiones (SOP Error #5).')

# ════════════════════════════════════════════════════════════════════════
# H6: "BULL con DSR > 0.10 tendría mejor WR que BULL con DSR < 0.10"
# Test: leer DSR de firmas y correlacionar con WR OOS de esa run/ventana
# ════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('H6: BULL con DSR_CPCV > 0.10 tiene mejor WR OOS que DSR < 0.10')
print('    (Valida el nuevo threshold propuesto para el gate)')
print(SEP)

# Cargar firmas bull de todas las runs disponibles (no solo 01/06)
bull_dsr_wr = []
for fp in sorted(runs.rglob('*/models/xgboost_meta_bull_long_signature.json'),
                 key=lambda p: p.stat().st_mtime, reverse=True)[:200]:
    try:
        with open(fp) as f:
            sig = json.load(f)
        dsr = sig.get('dsr_cpcv_best', sig.get('dsr_oos', None))
        if dsr is None:
            continue
        # Buscar oos_trades correspondientes a esta run
        run_dir = fp.parent.parent  # models/ es 1 nivel adentro del run
        oos_files = list(run_dir.glob('*/oos_trades.parquet'))
        if not oos_files:
            # Intentar estructura seed/W/
            oos_files = list(run_dir.parent.glob('*/*/oos_trades.parquet'))
        for ofs in oos_files:
            try:
                df = pd.read_parquet(ofs)
                bull_sub = df[df['hmm_regime'].astype(str).str.contains('BULL', na=False)]
                if len(bull_sub) < 5:
                    continue
                wr_oos = (bull_sub['return_pct'] > 0).mean()
                bull_dsr_wr.append({'dsr': float(dsr), 'wr_oos': wr_oos, 'n': len(bull_sub)})
            except:
                pass
    except:
        pass

if len(bull_dsr_wr) >= 10:
    df_dsr = pd.DataFrame(bull_dsr_wr)
    r, p = stats.spearmanr(df_dsr['dsr'], df_dsr['wr_oos'])
    print(f'  N pares (DSR, WR_OOS): {len(df_dsr)}')
    print(f'  Spearman DSR_IS → WR_OOS: r={r:+.4f}, p={p:.4f} → '
          f'{"SIGNIFICATIVO" if p < 0.05 else "no significativo"}')
    # Dividir por threshold propuesto 0.10
    low  = df_dsr[df_dsr['dsr'] < 0.10]
    high = df_dsr[df_dsr['dsr'] >= 0.10]
    if len(low) > 3 and len(high) > 3:
        t, p_t = stats.ttest_ind(low['wr_oos'], high['wr_oos'])
        print(f'  DSR<0.10: n={len(low)} avg_WR_OOS={low["wr_oos"].mean():.1%}')
        print(f'  DSR≥0.10: n={len(high)} avg_WR_OOS={high["wr_oos"].mean():.1%}')
        print(f'  t-test: t={t:.3f}, p={p_t:.4f} → diferencia '
              f'{"SIGNIFICATIVA" if p_t < 0.05 else "NO significativa"}')
else:
    print(f'  Solo {len(bull_dsr_wr)} pares encontrados — insuficiente para H6')
    print('  Buscando firmas en wfb_cache/archive...')
    # Buscar en archive
    archive = pathlib.Path('g:/Mi unidad/ia/luna_v2/data/archive')
    bull_dsr_wr2 = []
    for fp in sorted(archive.rglob('*bull*long*signature*.json'))[:100]:
        try:
            with open(fp) as f:
                sig = json.load(f)
            dsr = sig.get('dsr_cpcv_best', sig.get('dsr_oos', None))
            if dsr is not None:
                bull_dsr_wr2.append(float(dsr))
        except:
            pass
    if bull_dsr_wr2:
        dsrs = np.array(bull_dsr_wr2)
        print(f'  DSR BULL en archivo: N={len(dsrs)} min={dsrs.min():+.4f} '
              f'mean={dsrs.mean():+.4f} max={dsrs.max():+.4f}')
        pct_above_010 = (dsrs >= 0.10).mean() * 100
        pct_above_000 = (dsrs >= 0.00).mean() * 100
        print(f'  BULL DSR >= 0.00: {pct_above_000:.0f}% de firmas (gate actual pasa)')
        print(f'  BULL DSR >= 0.10: {pct_above_010:.0f}% de firmas (gate propuesto)')
        print(f'  El gate 0.10 bloquearía {100-pct_above_010:.0f}% de las firmas BULL')
