"""
Audit script para analizar la última run de seeds.
Genera estadísticas comparativas entre seeds y diagnóstico de filtros.
"""
import pandas as pd
import json
import glob
import os

report_dir = r'G:\Mi unidad\ia\luna_v2\data\reports'
verdicts = sorted(glob.glob(os.path.join(report_dir, '2026-05-2*_T*_FINAL_statistical_verdict.json')))

print('[AUDIT] === ANÁLISIS DEL FILTRO DE SEÑALES W5 ===')
print()

for v in verdicts:
    with open(v) as f:
        d = json.load(f)
    sp = d.get('signal_pipeline', {})
    run_id = d.get('run_id', '')
    status = sp.get('status', 'active')
    seed = run_id.split('_seed')[1].split('_')[0] if '_seed' in run_id else '?'

    if status == 'zero_signals':
        print(f'SEED {seed} | {run_id}')
        print(f'  Status: ZERO SIGNALS en W5')
        print(f'  after_xgb: {sp.get("after_xgb", "N/A")}')
        print(f'  after_meta: {sp.get("after_meta", "N/A")}')
        print(f'  after_hmm: {sp.get("after_hmm", "N/A")}')
        print(f'  after_embargo: {sp.get("after_embargo", "N/A")}')
        print(f'  Reason: {sp.get("reason", "N/A")}')
        print()
    else:
        print(f'SEED {seed} | {run_id} [ACTIVE]')
        raw = sp.get('raw_oos_bars', 0)
        after_xgb = sp.get('after_xgb', 0)
        after_hmm = sp.get('after_hmm', 0)
        after_meta = sp.get('after_meta', 0)
        after_mom = sp.get('after_momentum', 0)
        after_emb = sp.get('after_embargo', 0)
        if raw > 0:
            pct_xgb = round(after_xgb / raw * 100, 1)
            pct_mom = round(after_mom / after_xgb * 100, 1) if after_xgb else 0
            pct_emb = round(after_emb / after_mom * 100, 1) if after_mom else 0
            print(f'  Raw->XGB: {raw}->{after_xgb} ({pct_xgb}%)')
            print(f'  XGB->Meta: {after_xgb}->{after_meta}')
            print(f'  Meta->Momentum: {after_meta}->{after_mom} ({pct_mom}% pasan momentum)')
            print(f'  Momentum->Embargo: {after_mom}->{after_emb} ({pct_emb}% pasan embargo)')
        print()

# ===================================================================
print()
print('[AUDIT] === TABLA COMPARATIVA TODAS LAS SEEDS ===')
rows = []
for v in verdicts:
    with open(v) as f:
        d = json.load(f)
    run_id = d.get('run_id', '')
    approved = d.get('deploy_approved', False)
    m = d.get('metrics', {})
    sa = d.get('statistical_audit', {})
    sp = d.get('signal_pipeline', {})
    flags = d.get('flags', {})
    wfv = d.get('wfv_results', {})

    seed = run_id.split('_seed')[1].split('_')[0] if '_seed' in run_id else 'N/A'
    timestamp = d.get('timestamp', '')

    # W5 trades
    w5 = wfv.get('W5', {})
    w5_trades = w5.get('n_trades', 0)
    w5_wr = w5.get('win_rate', 0)

    row = {
        'run_id': run_id,
        'seed': seed,
        'approved': approved,
        'trades': m.get('total_trades', 0),
        'win_rate': round(m.get('win_rate', 0) * 100, 1),
        'ret_pct': round(m.get('total_return_pct', 0) * 100, 2),
        'max_dd': round(m.get('max_drawdown_pct', 0) * 100, 1),
        'sharpe': round(m.get('sharpe_crudo', 0), 3),
        'calmar': round(m.get('calmar_ratio', 0), 1),
        'dsr': round(sa.get('dsr', 0), 4),
        'pbo': round(sa.get('estimated_pbo', 0) * 100, 1),
        'skew': round(sa.get('skewness', 0), 3),
        'kurt': round(sa.get('kurtosis', 0), 3),
        'pass_dsr': flags.get('pass_dsr', False),
        'pass_pbo': flags.get('pass_pbo', False),
        'pass_dd': flags.get('pass_dd', False),
        'w5_trades': w5_trades,
        'w5_wr': round(w5_wr * 100, 1),
        'w5_status': sp.get('status', 'active')
    }
    rows.append(row)

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# Approved only
print()
print('[AUDIT] === SEEDS APROBADAS ===')
approved_df = df[df['approved'] == True]
if len(approved_df) > 0:
    print(approved_df.to_string(index=False))
else:
    print('  Ninguna seed aprobada.')

# Summary stats
print()
print('[AUDIT] === RESUMEN ESTADÍSTICO ===')
print(f"  Seeds totales analizadas: {len(df)}")
print(f"  Seeds aprobadas: {df['approved'].sum()}")
print(f"  Win Rate promedio: {df['win_rate'].mean():.1f}%")
print(f"  Sharpe promedio: {df['sharpe'].mean():.3f}")
print(f"  DSR promedio: {df['dsr'].mean():.4f}")
print(f"  PBO promedio: {df['pbo'].mean():.1f}%")
print(f"  Max DD promedio: {df['max_dd'].mean():.1f}%")
print(f"  Seeds con W5 zero signals: {(df['w5_status']=='zero_signals').sum()}")
print(f"  Seeds con pass_dsr: {df['pass_dsr'].sum()}")
print(f"  Seeds con pass_pbo: {df['pass_pbo'].sum()}")

# PBO analysis
print()
print('[AUDIT] === ANÁLISIS PBO ===')
print(f"  PBO < 22% (umbral SOP): {(df['pbo'] < 22).sum()} seeds")
print(f"  PBO entre 22-40%: {((df['pbo'] >= 22) & (df['pbo'] < 40)).sum()} seeds")
print(f"  PBO > 40%: {(df['pbo'] >= 40).sum()} seeds")
print(f"  PBO = 50% (max teórico): {(df['pbo'] == 50).sum()} seeds  <- SEÑAL DE ALARMA")
