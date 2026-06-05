"""
AUDITORIA ESTRUCTURAL PROFUNDA — Luna V2 — 2026-06-01
Estructura correcta: runs/{RUN}/{SEED}/{WINDOW}/oos_trades.parquet
"""
import pathlib, pandas as pd, numpy as np, json

runs = pathlib.Path('g:/Mi unidad/ia/luna_v2/data/runs')

all_trades = sorted(runs.rglob('*/W*/oos_trades.parquet'),
    key=lambda p: p.stat().st_mtime, reverse=True)
recent = [f for f in all_trades if '20260601' in f.parts[-4]]
print(f'Total oos_trades.parquet en runs del 01/06: {len(recent)}')

window_data, agent_data, run_totals = {}, {}, {}
all_w, all_l = [], []

for fp in recent:
    try:
        df = pd.read_parquet(fp)
        if len(df) < 2: continue
        v   = df['return_pct'].values
        w   = fp.parts[-2]
        run = fp.parts[-4]
        wr  = (v > 0).sum() / len(v)
        ret = v.sum() * 100
        window_data.setdefault(w, []).append({'wr': wr, 'ret': ret, 'n': len(v)})
        run_totals[run] = run_totals.get(run, 0) + len(v)
        all_w.extend(v[v > 0].tolist())
        all_l.extend(v[v < 0].tolist())
        hmm_col = next((c for c in ['hmm_regime_label', 'HMM_Semantic', 'hmm_semantic', 'hmm_regime']
                        if c in df.columns), None)
        if hmm_col:
            for regime, grp in df.groupby(hmm_col):
                k  = str(regime)[:28]
                rv = grp['return_pct'].values
                s  = agent_data.setdefault(k, {'n': 0, 'wins': 0, 'ret': 0.0})
                s['n']    += len(rv)
                s['wins'] += (rv > 0).sum()
                s['ret']  += rv.sum() * 100
    except Exception as e:
        pass

SEP = '=' * 68

print()
print(SEP)
print('1. RENDIMIENTO POR VENTANA (todas las seeds del 01/06)')
print(SEP)
for w in ['W1', 'W2', 'W3', 'W4', 'W5']:
    d = window_data.get(w, [])
    if not d: continue
    avg_wr  = np.mean([x['wr'] for x in d])
    avg_ret = np.mean([x['ret'] for x in d])
    avg_n   = np.mean([x['n'] for x in d])
    pct_pos = sum(1 for x in d if x['wr'] > 0.50) / len(d) * 100
    verdict = 'EDGE' if avg_wr > 0.52 else ('BORDE' if avg_wr > 0.48 else 'SIN-EDGE')
    print(f"  {w}: seeds={len(d)} WR={avg_wr:.1%} ret={avg_ret:+.3f}% "
          f"avg_trades={avg_n:.0f} WR>50%={pct_pos:.0f}%  [{verdict}]")

print()
print(SEP)
print('2. RENDIMIENTO POR REGIMEN (todos los trades juntos)')
print(SEP)
for k, s in sorted(agent_data.items(), key=lambda x: -x[1]['n']):
    if s['n'] < 5: continue
    wr = s['wins'] / s['n']
    verdict = 'EDGE' if wr > 0.52 else ('BORDE' if wr > 0.48 else 'SIN-EDGE')
    print(f"  {k:<28} N={s['n']:>5} WR={wr:.1%} ret={s['ret']:+.3f}%  [{verdict}]")

print()
print(SEP)
print('3. SUFICIENCIA ESTADISTICA y BARRIERS R:R')
print(SEP)
tots = list(run_totals.values())
if tots:
    print(f"  Total trades: {sum(tots)} en {len(tots)} runs parciales")
    print(f"  Runs >100 trades: {sum(1 for t in tots if t > 100)}/{len(tots)}")
    print(f"  Runs  <30 trades: {sum(1 for t in tots if t < 30)}/{len(tots)}")
if all_w and all_l:
    avg_win  = np.mean(all_w) * 100
    avg_loss = np.mean(all_l) * 100
    rr       = abs(avg_win / avg_loss)
    wr_g     = len(all_w) / (len(all_w) + len(all_l))
    be_wr    = 1 / (1 + rr)
    deficit  = wr_g - be_wr
    print(f"  avg_win={avg_win:+.4f}%  avg_loss={avg_loss:+.4f}%")
    print(f"  R:R={rr:.3f} | WR_breakeven={be_wr:.1%} | WR_real={wr_g:.1%}")
    print(f"  Deficit WR vs breakeven: {deficit:+.1%}")
    if wr_g < be_wr:
        print("  --> CRITICO: WR insuficiente para el R:R. Pierde con cualquier N.")
    else:
        print("  --> OK: WR suficiente para el R:R actual.")

print()
print(SEP)
print('4. DSR de modelos XGBoost — ¿señal real o ruido IS?')
print(SEP)
dsr_by_agent = {}
sig_paths = sorted(runs.rglob('*/models/xgboost_meta_*long*signature.json'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
for fp in sig_paths:
    if '20260601' not in str(fp): continue
    try:
        with open(fp) as f:
            sig = json.load(f)
        # extraer nombre agente del nombre de fichero
        stem   = fp.stem  # xgboost_meta_bull_long_signature
        parts  = stem.replace('xgboost_meta_', '').replace('_signature', '').split('_long')
        agent  = parts[0] if parts else stem
        dsr    = float(sig.get('dsr_cpcv_best', sig.get('dsr_oos', -99)))
        brier  = sig.get('xgb_brier_raw', None)
        base   = sig.get('target_base_rate', None)
        dsr_by_agent.setdefault(agent, []).append({'dsr': dsr, 'brier': brier, 'base': base})
    except:
        pass

for agent, vals in sorted(dsr_by_agent.items()):
    dsrs  = [v['dsr'] for v in vals]
    avg   = np.mean(dsrs)
    ppos  = sum(1 for d in dsrs if d > 0) / len(dsrs) * 100
    briers = [v['brier'] for v in vals if v['brier'] is not None]
    bases  = [v['base'] for v in vals if v['base'] is not None]
    verdict = 'SEÑAL' if avg > 0.3 else ('MARGINAL' if avg > 0 else 'RUIDO')
    brier_str = f"brier={np.mean(briers):.3f}" if briers else ""
    base_str  = f"base_rate={np.mean(bases):.3f}" if bases else ""
    print(f"  {agent:<20} n={len(dsrs)} avg_DSR={avg:+.4f} DSR>0={ppos:.0f}%  "
          f"{brier_str} {base_str}  [{verdict}]")

print()
print(SEP)
print('5. ESTABILIDAD DE FEATURES (SFI — señal causal o espuria?)')
print(SEP)
feat_counts = {}
n_sf = 0
for fp in runs.rglob('*/W*/selected_features.json'):
    if '20260601' not in str(fp): continue
    try:
        with open(fp) as f:
            feats = json.load(f)
        if isinstance(feats, list):
            n_sf += 1
            for feat in feats:
                feat_counts[feat] = feat_counts.get(feat, 0) + 1
    except:
        pass

if n_sf > 0:
    sorted_feats = sorted(feat_counts.items(), key=lambda x: -x[1])
    stable   = sum(1 for _, c in sorted_feats if c / n_sf > 0.6)
    unstable = sum(1 for _, c in sorted_feats if c == 1)
    print(f"  selected_features.json leídos: {n_sf}")
    print(f"  Features estables (>60% runs): {stable} <- señal robusta")
    print(f"  Features inestables (1 run):   {unstable} <- posible ruido")
    print(f"  Ratio estabilidad: {stable}/{len(feat_counts)} ({stable/max(len(feat_counts),1)*100:.0f}%)")
    print(f"\n  TOP-20 features más estables:")
    for feat, cnt in sorted_feats[:20]:
        bar = '#' * int(cnt / n_sf * 20)
        print(f"    {feat:<40} {cnt:>3}/{n_sf} {bar}")

print()
print(SEP)
print('6. VEREDICTO ESTRUCTURAL')
print(SEP)
