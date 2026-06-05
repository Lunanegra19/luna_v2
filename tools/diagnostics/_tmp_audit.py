import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\predictions')
files = sorted(DATA.glob('oos_trades_seed*.parquet'))
print(f"Archivos: {len(files)} seeds")
dfs = []
for f in files:
    d = pd.read_parquet(f)
    d['_seed'] = int(f.stem.split('seed')[1])
    dfs.append(d)
df = pd.concat(dfs)
print(f"Total trades: {len(df)}")
n_seeds = df['_seed'].nunique()
print(f"Seeds: {n_seeds}")
print()
if 'wfb_window' in df.columns:
    print("wfb_window distribucion:")
    print(df['wfb_window'].value_counts().to_string())
print()
inconsistencias = ((df['is_win']==1) & (df['return_raw']<0) | (df['is_win']==0) & (df['return_raw']>0)).sum()
print(f"is_win vs return_raw inconsistencias: {inconsistencias}")
print(f"Columnas: {list(df.columns)}")
# Fechas
if 'entry_time' in df.columns:
    et = pd.to_datetime(df['entry_time'], utc=True, errors='coerce')
    print(f"\nFechas entry_time: {et.min()} -> {et.max()}")
