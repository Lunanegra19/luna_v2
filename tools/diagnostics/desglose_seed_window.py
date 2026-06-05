"""Desglose completo por ventana y por seed."""
import sys, json, warnings
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from pathlib import Path

wfb_dir = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')

# Carga todos los parquets con metadata
records = {}
for f in sorted(wfb_dir.glob('oos_trades_W*_seed*.parquet')):
    parts  = f.stem.split('_')
    window = next((p for p in parts if p.startswith('W')), '?')
    seed   = next((p.replace('seed','') for p in parts if p.startswith('seed')), '?')
    try:
        df = pd.read_parquet(f)
        if len(df) > 0:
            records[(window, seed)] = df
    except Exception as e:
        records[(window, seed)] = None

# Early stop logs
early_stops = {}
for f in sorted(wfb_dir.glob('early_stop_seed*.json')):
    s = f.stem.replace('early_stop_seed','')
    with open(f) as fp:
        early_stops[s] = json.load(fp)

# Todas las seeds únicas
all_seeds = sorted(set(s for (w, s) in records.keys()), key=int)
all_windows = ['W1','W2','W3','W4','W5']

def stats(df):
    if df is None or len(df) == 0:
        return None
    n    = len(df)
    wr   = float(df['is_win'].mean()) if 'is_win' in df.columns else float('nan')
    ret  = df['return_pct'].dropna() if 'return_pct' in df.columns else pd.Series(dtype=float)
    mr   = float(ret.mean())   if len(ret) > 0 else float('nan')
    std  = float(ret.std())    if len(ret) > 1 else float('nan')
    aw   = float(ret[ret > 0].mean()) if (ret > 0).any() else float('nan')
    al   = float(ret[ret < 0].mean()) if (ret < 0).any() else float('nan')
    rr   = abs(aw/al)          if (not np.isnan(aw) and not np.isnan(al) and al != 0) else float('nan')
    dd   = float(df['drawdown'].min()) if 'drawdown' in df.columns else float('nan')
    sr   = mr / std * np.sqrt(n) if (std and std > 1e-10) else float('nan')
    # calibracion
    if 'xgb_prob' in df.columns and 'xgb_prob_cal' in df.columns:
        diff = (df['xgb_prob_cal'] - df['xgb_prob']).abs()
        cal_ok = float((diff > 1e-6).mean() * 100)
        cal_delta = float((df['xgb_prob_cal'] - df['xgb_prob']).mean())
    else:
        cal_ok = float('nan')
        cal_delta = float('nan')
    # ood
    ood = float(df['ood_kl_distance'].mean()) if 'ood_kl_distance' in df.columns else float('nan')
    return dict(n=n, wr=wr, mr=mr, std=std, aw=aw, al=al, rr=rr, dd=dd, sr=sr,
                cal_ok=cal_ok, cal_delta=cal_delta, ood=ood)

SEP = '─'*90

# ════════════════════════════════════════════════════════════════════════════
print('═'*90)
print('DESGLOSE POR SEED (cada seed: qué ventanas completó y sus KPIs)')
print('═'*90)
print()

for seed in all_seeds:
    es = early_stops.get(seed, {})
    es_windows = es.get('windows_evaluated', [])
    es_reason  = str(es.get('reason', 'NO EARLY-STOP'))[:70]
    seed_windows = [w for (w, s) in records.keys() if s == seed]

    # Estado global de esta seed
    if es:
        if 'Sharpe' in es_reason:
            status = f'PODADA (Sharpe gate) tras W{max(es_windows)}'
        else:
            status = f'PODADA (UB gate) tras W{max(es_windows)}'
    elif not seed_windows:
        status = 'FATAL (no hay trades)'
    elif max(seed_windows, key=lambda x: int(x[1:])) == 'W5':
        status = '✅ COMPLETADA'
    else:
        max_w = max(int(w[1:]) for w in seed_windows)
        status = f'FATAL (máx W{max_w})'

    print(f'SEED {seed:>6}  [{status}]')
    if es:
        print(f'  Early-stop: {es_reason}')

    for w in all_windows:
        df = records.get((w, seed))
        s  = stats(df)
        if s is None:
            if w in seed_windows or int(w[1:]) <= (max(int(x[1:]) for x in seed_windows) if seed_windows else 0):
                print(f'  {w}: SIN TRADES')
            continue
        # Render
        wr_bar  = '█' * int(s['wr'] * 20)
        cal_str = f"cal={s['cal_ok']:.0f}% Δ={s['cal_delta']:+.3f}" if not np.isnan(s['cal_ok']) else ''
        ood_str = f"OOD={s['ood']:.3f}" if not np.isnan(s['ood']) else ''
        rr_str  = f"{s['rr']:.3f}" if not np.isnan(s['rr']) else 'N/A'
        dd_str  = f"{s['dd']*100:.2f}%" if not np.isnan(s['dd']) else 'N/A'
        sr_str  = f"{s['sr']:.3f}" if not np.isnan(s['sr']) else 'N/A'
        aw_str  = f"+{s['aw']*100:.3f}%" if not np.isnan(s['aw']) else 'N/A'
        al_str  = f"{s['al']*100:.3f}%" if not np.isnan(s['al']) else 'N/A'
        wr_pct  = s['wr']*100
        wr_sym  = '✅' if wr_pct >= 52 else ('⚠️' if wr_pct >= 45 else '❌')
        print(f'  {w}: n={s["n"]:>3}  WR={wr_pct:>5.1f}% {wr_sym}  MeanRet={s["mr"]*100:>+7.4f}%  '
              f'R:R={rr_str}  SR={sr_str}  MaxDD={dd_str}')
        print(f'       AvgWin={aw_str}  AvgLoss={al_str}  {cal_str}  {ood_str}')
    print()

# ════════════════════════════════════════════════════════════════════════════
print('═'*90)
print('DESGLOSE POR VENTANA (todas las seeds que llegaron a esa ventana)')
print('═'*90)
print()

for w in all_windows:
    w_seeds = [(s, records[(w,s)]) for (ww,s) in records.keys() if ww == w]
    if not w_seeds:
        print(f'{w}: SIN DATOS')
        continue

    print(f'{SEP}')
    print(f'VENTANA {w}  ({len(w_seeds)} seeds con trades)')
    print(f'{SEP}')
    print(f'{"SEED":>8} | {"N":>4} | {"WR%":>6} | {"MeanRet%":>9} | {"R:R":>5} | {"Sharpe":>7} | {"MaxDD%":>7} | {"Cal%":>5} | {"Δcal":>6} | {"OOD":>6}')
    print(f'{"─"*8}-+-{"─"*4}-+-{"─"*6}-+-{"─"*9}-+-{"─"*5}-+-{"─"*7}-+-{"─"*7}-+-{"─"*5}-+-{"─"*6}-+-{"─"*6}')

    seed_stats = []
    for (seed, df) in sorted(w_seeds, key=lambda x: int(x[0])):
        s = stats(df)
        if s:
            seed_stats.append((seed, s))
            wr_s  = f"{s['wr']*100:.1f}"
            mr_s  = f"{s['mr']*100:+.4f}"
            rr_s  = f"{s['rr']:.3f}" if not np.isnan(s['rr']) else ' N/A'
            sr_s  = f"{s['sr']:.3f}" if not np.isnan(s['sr']) else '   N/A'
            dd_s  = f"{s['dd']*100:.2f}" if not np.isnan(s['dd']) else '   N/A'
            cal_s = f"{s['cal_ok']:.0f}" if not np.isnan(s['cal_ok']) else 'N/A'
            dc_s  = f"{s['cal_delta']:+.3f}" if not np.isnan(s['cal_delta']) else '  N/A'
            od_s  = f"{s['ood']:.3f}" if not np.isnan(s['ood']) else '  N/A'
            wr_sym = '✅' if s['wr'] >= 0.52 else ('⚠️' if s['wr'] >= 0.45 else '❌')
            print(f'{seed:>8} | {s["n"]:>4} | {wr_s:>5}{wr_sym}| {mr_s:>9} | {rr_s:>5} | {sr_s:>7} | {dd_s:>7} | {cal_s:>5} | {dc_s:>6} | {od_s:>6}')

    # Totales de ventana
    if seed_stats:
        all_df = pd.concat([records[(w,s)] for s,_ in seed_stats], ignore_index=True)
        tot = stats(all_df)
        print(f'{"─"*8}-+-{"─"*4}-+-{"─"*6}-+-{"─"*9}-+-{"─"*5}-+-{"─"*7}-+-{"─"*7}-+-{"─"*5}-+-{"─"*6}-+-{"─"*6}')
        wr_t = f"{tot['wr']*100:.1f}"
        mr_t = f"{tot['mr']*100:+.4f}"
        rr_t = f"{tot['rr']:.3f}" if not np.isnan(tot['rr']) else ' N/A'
        sr_t = f"{tot['sr']:.3f}" if not np.isnan(tot['sr']) else '   N/A'
        dd_t = f"{tot['dd']*100:.2f}" if not np.isnan(tot['dd']) else '   N/A'
        print(f'{"TOTAL":>8} | {tot["n"]:>4} | {wr_t:>6} | {mr_t:>9} | {rr_t:>5} | {sr_t:>7} | {dd_t:>7} |')
    print()

# ════════════════════════════════════════════════════════════════════════════
print('═'*90)
print('SEEDS QUE NO LLEGARON A GENERAR TRADES (solo FATAL, sin parquets)')
print('═'*90)
seeds_with_trades = set(s for (w,s) in records.keys())
seeds_all_attempted = set(early_stops.keys()) | seeds_with_trades
seeds_no_trades = [s for s in sorted(seeds_all_attempted, key=int) if s not in seeds_with_trades]
if seeds_no_trades:
    for s in seeds_no_trades:
        es = early_stops.get(s, {})
        print(f'  seed{s}: {str(es.get("reason","FATAL sin early-stop"))[:80]}')
else:
    print('  (todas las seeds generaron al menos 1 trade)')
print()
