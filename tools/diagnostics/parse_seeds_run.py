import pandas as pd, os, glob, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
DATA_DIR = r'g:\Mi unidad\ia\luna_v2\data\reports\wfb'
trade_files = sorted(glob.glob(os.path.join(DATA_DIR, 'oos_trades_W*_seed*.parquet')))
print(f"Archivos encontrados: {len(trade_files)}")

results = {}
for f in trade_files:
    fname = os.path.basename(f)
    parts = fname.replace('.parquet','').split('_')
    window = parts[2]
    seed = parts[3].replace('seed','')
    try:
        df = pd.read_parquet(f)
        ret_col = next((c for c in ['return_raw','ret','return','pnl','pnl_pct','return_pct'] if c in df.columns), None)
        n = len(df)
        if n > 0 and ret_col:
            wr = float((df[ret_col]>0).mean())*100
            rm = float(df[ret_col].mean())*100
            rt = float(df[ret_col].sum())*100
        else:
            wr=rm=rt=0.0
        results[f'{window}_{seed}'] = {'s':seed,'w':window,'n':n,'wr':round(wr,2),'rm':round(rm,5),'rt':round(rt,4)}
    except Exception as e:
        print(f"ERROR {fname}: {e}")

# Por ventana
print()
print("DETALLE POR VENTANA:")
print(f"{'Ventana':<6} {'Seed':>7} {'Trades':>7} {'WR%':>7} {'RetMed%':>10} {'RetTotal%':>11}")
print("-"*60)
for k in sorted(results.keys()):
    v = results[k]
    print(f"  {v['w']:<5} {v['s']:>7} {v['n']:>7} {v['wr']:>7.2f} {v['rm']:>10.5f} {v['rt']:>11.4f}")

# Por seed
by_seed = defaultdict(lambda: {'n':0,'ws':[],'wr':[],'rt':[],'rt_by_w':{}})
for k,v in results.items():
    s=v['s']
    by_seed[s]['n']+=v['n']
    by_seed[s]['ws'].append(v['w'])
    by_seed[s]['rt_by_w'][v['w']]=v['rt']
    if v['n']>0:
        by_seed[s]['wr'].append(v['wr'])
        by_seed[s]['rt'].append(v['rt'])

print()
print("RESUMEN POR SEED:")
print(f"{'Seed':>8} {'Trades':>7} {'AvgWR%':>8} {'RetTot%':>10} {'Status':<12} {'Ventanas':<15} {'Desglose RetTotal%'}")
print("-"*120)
for s in sorted(by_seed.keys(), key=lambda x: int(x) if x.isdigit() else 0):
    d = by_seed[s]
    aw = sum(d['wr'])/len(d['wr']) if d['wr'] else 0
    tr = sum(d['rt'])
    ws = ' '.join(sorted(d['ws']))
    status = "PASS>=30" if d['n']>=30 else "FAIL<30 "
    desglose = '  '.join([f"{w}={d['rt_by_w'].get(w,0.0):+.3f}%" for w in sorted(d['ws'])])
    print(f"  {s:>8} {d['n']:>7} {aw:>8.2f} {tr:>10.4f}   {status:<12} {ws:<15} {desglose}")

print()
print("SEEDS APROBADAS (>=30 trades):")
aprobadas = [s for s in by_seed if by_seed[s]['n']>=30]
if aprobadas:
    for s in aprobadas:
        d = by_seed[s]
        aw = sum(d['wr'])/len(d['wr']) if d['wr'] else 0
        print(f"  seed={s}: trades={d['n']}, AvgWR={aw:.2f}%")
else:
    print("  NINGUNA")

print(f"\nTotal seeds procesadas: {len(by_seed)}")
print(f"Total archivos parquet: {len(trade_files)}")
