"""
Resultados por seed y ventana — run del 02/06/2026
"""
import pathlib, pandas as pd, numpy as np
from scipy import stats

runs = pathlib.Path('g:/Mi unidad/ia/luna_v2/data/runs')
today = sorted(runs.glob('WFB_20260602_*'), key=lambda p: p.name)

rows = []
all_dfs = []
for run_dir in today:
    seed_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not seed_dirs:
        continue
    seed_name = seed_dirs[0].name
    for win_dir in sorted(d for d in seed_dirs[0].iterdir() if d.is_dir()):
        pq = win_dir / 'oos_trades.parquet'
        if not pq.exists():
            continue
        try:
            df = pd.read_parquet(pq)
            if len(df) == 0:
                continue
            df['_seed']   = seed_name
            df['_window'] = win_dir.name
            all_dfs.append(df)
            hmm_col = 'hmm_regime'
            n      = len(df)
            wr     = (df['return_pct'] > 0).mean()
            ev     = df['return_pct'].mean() * 100
            bull_n = df[df[hmm_col].astype(str).str.contains('BULL', na=False)].shape[0]
            bear_n = df[df[hmm_col].astype(str).str.contains('CALM_BEAR|3_CALM', na=False)].shape[0]
            rng_n  = df[df[hmm_col].astype(str).str.contains('RANGE', na=False)].shape[0]
            rows.append({'seed': seed_name, 'window': win_dir.name,
                         'N': n, 'WR': wr, 'EV': ev,
                         'BULL': bull_n, 'CB': bear_n, 'RNG': rng_n})
        except:
            pass

if not rows:
    print('Sin datos todavia — la run sigue procesando')
else:
    df_res = pd.DataFrame(rows).sort_values(['seed', 'window'])
    print(f'Ventanas con datos: {len(df_res)} | Seeds distintas: {df_res.seed.nunique()}')
    print()
    hdr = f"{'SEED':<14} {'WIN':<5} {'N':>5} {'WR':>7} {'EV/trade':>10} {'BULL':>6} {'CB':>5} {'RNG':>5}"
    print(hdr)
    print('-' * 58)
    prev_seed = None
    for _, r in df_res.iterrows():
        if prev_seed and prev_seed != r['seed']:
            print()
        prev_seed = r['seed']
        bull_mark = '0 OK' if r['BULL'] == 0 else f"!{int(r['BULL'])}"
        print(f"{r['seed']:<14} {r['window']:<5} {int(r['N']):>5} {r['WR']:>7.1%} "
              f"{r['EV']:>+10.5f}% {bull_mark:>6} {int(r['CB']):>5} {int(r['RNG']):>5}")
    print()
    print('-' * 58)
    total_n  = int(df_res['N'].sum())
    total_wr = (df_res['N'] * df_res['WR']).sum() / total_n
    total_ev = (df_res['N'] * df_res['EV']).sum() / total_n
    print(f"{'TOTAL':<14} {'':<5} {total_n:>5} {total_wr:>7.1%} {total_ev:>+10.5f}%  "
          f"BULL={int(df_res['BULL'].sum())} CB={int(df_res['CB'].sum())} RNG={int(df_res['RNG'].sum())}")
    print()

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        cb = df_all[df_all['hmm_regime'].astype(str).str.contains('CALM_BEAR|3_CALM', na=False)]
        print('=== CALM_BEAR ACUMULADO (esta run) ===')
        wr_cb = (cb['return_pct'] > 0).mean()
        ev_cb = cb['return_pct'].mean() * 100
        print(f'  N={len(cb)} | WR={wr_cb:.1%} | EV={ev_cb:+.5f}%')
        if len(cb) >= 20:
            wins = int((cb['return_pct'] > 0).sum())
            res  = stats.binomtest(wins, len(cb), p=0.50, alternative='greater')
            ic_lo = wr_cb - 1.96*(wr_cb*(1-wr_cb)/len(cb))**0.5
            ic_hi = wr_cb + 1.96*(wr_cb*(1-wr_cb)/len(cb))**0.5
            print(f'  binom_test (WR>50%): p={res.pvalue:.5f} | IC95=[{ic_lo:.1%}, {ic_hi:.1%}]')
        if 'alpha_trigger' in cb.columns:
            dtw   = cb[cb['alpha_trigger'] == 'alpha_dtw_signal']
            nodtw = cb[cb['alpha_trigger'] != 'alpha_dtw_signal']
            print(f'  CON DTW: N={len(dtw)} WR={(dtw["return_pct"]>0).mean():.1%}')
            print(f'  SIN DTW: N={len(nodtw)} WR={(nodtw["return_pct"]>0).mean():.1%}')

        # Por ventana para CALM_BEAR
        print()
        print('=== CALM_BEAR por VENTANA (nueva run) ===')
        for w in ['W1', 'W2', 'W3', 'W4', 'W5']:
            sub = cb[cb['_window'] == w]
            if len(sub) == 0:
                continue
            wr_w = (sub['return_pct'] > 0).mean()
            print(f'  {w}: N={len(sub)} WR={wr_w:.1%} EV={sub["return_pct"].mean()*100:+.5f}%')
