import pathlib, json, numpy as np

runs = pathlib.Path('g:/Mi unidad/ia/luna_v2/data/runs')

range_sigs = []
for fp in sorted(runs.rglob('*/models/xgboost_meta_range_long_signature.json'),
                 key=lambda p: p.stat().st_mtime, reverse=True):
    if '20260601' not in str(fp):
        continue
    try:
        with open(fp) as f:
            sig = json.load(f)
        range_sigs.append({
            'run':       fp.parts[-4],
            'dsr':       sig.get('dsr_cpcv_best', sig.get('dsr_oos', None)),
            'thr':       sig.get('optimal_threshold', None),
            'brier':     sig.get('xgb_brier_raw', None),
            'base_rate': sig.get('target_base_rate', None),
        })
    except:
        pass

print(f'Firmas XGBoost RANGE encontradas del 01/06: {len(range_sigs)}')
if range_sigs:
    dsrs   = [s['dsr'] for s in range_sigs if s['dsr'] is not None]
    thrs   = [s['thr'] for s in range_sigs if s['thr'] is not None]
    briers = [s['brier'] for s in range_sigs if s['brier'] is not None]
    bases  = [s['base_rate'] for s in range_sigs if s['base_rate'] is not None]
    print(f'  avg_DSR_CPCV:     {np.mean(dsrs):+.4f}   (DSR>0: {sum(1 for d in dsrs if d>0)}/{len(dsrs)})')
    print(f'  avg_threshold:    {np.mean(thrs):.4f}   <- si > 0.60 muy pocas seniales en OOS')
    print(f'  avg_brier:        {np.mean(briers):.4f}')
    print(f'  avg_base_rate_IS: {np.mean(bases):.4f}')
    print()
    print('Detalle por run (primeras 10):')
    for s in range_sigs[:10]:
        thr_v = f"{s['thr']:.3f}" if s['thr'] is not None else 'N/A'
        dsr_v = f"{s['dsr']:+.4f}" if s['dsr'] is not None else 'N/A'
        bri_v = f"{s['brier']:.3f}" if s['brier'] is not None else 'N/A'
        bas_v = f"{s['base_rate']:.3f}" if s['base_rate'] is not None else 'N/A'
        print(f'  thr={thr_v} DSR={dsr_v} brier={bri_v} base_rate={bas_v}  [{s["run"][-20:]}]')
    print()
    avg_thr = np.mean(thrs) if thrs else 0
    avg_dsr = np.mean(dsrs) if dsrs else 0
    if avg_dsr < 0:
        print(f'DIAGNOSTICO: DSR medio = {avg_dsr:+.4f} < 0 -> RANGE model SIN SEÑAL IS')
        print('  El modelo range no encuentra patron predictivo en IS.')
        print('  Es CORRECTO que genere pocos trades: el sistema lo sabe y pone threshold alto.')
        print('  NO es un threshold demasiado agresivo — es ausencia de señal.')
    elif avg_thr > 0.60:
        print(f'DIAGNOSTICO: threshold medio = {avg_thr:.3f} > 0.60 -> MUY POCOS trades')
        print('  El modelo tiene DSR positivo pero el threshold calibrado es demasiado alto.')
        print('  Optuna encontra señal pero muy selectiva.')
