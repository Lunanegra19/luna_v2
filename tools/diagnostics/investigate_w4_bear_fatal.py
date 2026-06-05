"""
INVESTIGACIÓN W4 FATAL — bear_long n_train sistémico
======================================================
Hipótesis: W4 cubre período predominantemente bull (2024).
El HMM clasifica casi 0 barras como bear en el período IS de W4.
→ bear_long se entrena con n≈1 → std=0 → FATAL.

Fases de investigación:
1. ¿Qué período temporal cubre W4?
2. ¿Cuántas barras bear hay en el período IS de W4?
3. ¿Cuántas barras bear habría en OOS de W4? (lo que perderíamos)
4. ¿Es el mismo patrón para todas las seeds o depende del seed?
5. ¿Existe código para manejar n_train bajo (min_viable_train_samples)?
"""
import sys
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

SEP = '─'*68
cfg = yaml.safe_load(Path('g:/Mi unidad/ia/luna_v2/config/settings.yaml').read_text(encoding='utf-8'))

print(SEP)
print('FASE 1: ¿Qué período temporal cubre cada ventana WFB?')
print(SEP)

# Buscar parquets de features/wfb_cache para ver los períodos
# Primero intentamos con los snapshots de la run actual
wfb_cache = Path('g:/Mi unidad/ia/luna_v2/data/wfb_cache')
found_windows = {}
for seed_dir in sorted(wfb_cache.iterdir())[:3]:  # primeras 3 seeds
    if not seed_dir.is_dir(): continue
    seed = seed_dir.name
    for w_dir in sorted(seed_dir.iterdir()):
        if not w_dir.name.startswith('W'): continue
        w = w_dir.name
        # Buscar features parquet
        for pattern in ['features_holdout*.parquet', 'features_validation*.parquet', 'features_train*.parquet']:
            files = list(w_dir.rglob(pattern))
            if files:
                try:
                    df = pd.read_parquet(files[0], columns=['datetime'] if 'datetime' in pd.read_parquet(files[0], columns=[]).columns else None)
                    # Intentar leer el índice temporal
                    df2 = pd.read_parquet(files[0])
                    if hasattr(df2.index, 'min'):
                        t_min = df2.index.min()
                        t_max = df2.index.max()
                    elif 'datetime' in df2.columns:
                        t_min = pd.to_datetime(df2['datetime']).min()
                        t_max = pd.to_datetime(df2['datetime']).max()
                    else:
                        continue
                    key = f'{seed}/{w}/{pattern.split("*")[0]}'
                    found_windows[key] = (t_min, t_max, len(df2))
                    break
                except Exception as e:
                    pass

for key, (t_min, t_max, n) in sorted(found_windows.items())[:15]:
    print(f'  {key}: {str(t_min)[:10]} → {str(t_max)[:10]} (n={n})')

print()
print(SEP)
print('FASE 2: HMM — distribución de regímenes en datos IS por período')
print(SEP)

# Cargar el HMM principal y ver cuántas barras bear tiene cada ventana IS
hmm_pkl = Path('g:/Mi unidad/ia/luna_v2/data/models/hmm_regime.pkl')
raw_parquet = Path('g:/Mi unidad/ia/luna_v2/data/raw/ohlcv/btc_usdt_1h.parquet')

# Ver si hay un feature dataset completo que tenga HMM_Semantic
feature_files = list(Path('g:/Mi unidad/ia/luna_v2/data').rglob('features_full*.parquet'))[:3]
feature_files += list(Path('g:/Mi unidad/ia/luna_v2/data').rglob('features_base*.parquet'))[:2]

for ff in feature_files[:3]:
    try:
        df = pd.read_parquet(ff)
        if 'HMM_Semantic' in df.columns:
            print(f'  {ff.name}: {len(df)} filas con HMM_Semantic')
            regime_counts = df['HMM_Semantic'].value_counts()
            print(f'  Regímenes: {regime_counts.to_dict()}')
            break
    except Exception as e:
        pass

print()
print(SEP)
print('FASE 3: ¿Cuántas barras bear_crash hay en cada sub-período anual?')
print(SEP)

# Usar los wfb_cache que tenemos para seed29291 (la más avanzada)
seed_dir = wfb_cache / 'seed29291'
if seed_dir.exists():
    for w_dir in sorted(seed_dir.iterdir()):
        if not w_dir.name.startswith('W'): continue
        w = w_dir.name
        # Buscar features con HMM_Semantic
        for ff in w_dir.rglob('*.parquet'):
            try:
                df = pd.read_parquet(ff)
                if 'HMM_Semantic' not in df.columns: continue
                bear_cols = [c for c in df['HMM_Semantic'].unique() if 'BEAR' in str(c).upper() or 'bear' in str(c).lower()]
                n_bear = df['HMM_Semantic'].isin(bear_cols).sum() if bear_cols else 0
                n_total = len(df)
                t_min = str(df.index.min())[:10] if hasattr(df.index, 'min') else '?'
                t_max = str(df.index.max())[:10] if hasattr(df.index, 'max') else '?'
                print(f'  seed29291/{w}/{ff.name[:30]}: {t_min}→{t_max} | bear={n_bear}/{n_total} ({n_bear/n_total*100:.1f}%)')
                print(f'    Regímenes: {df["HMM_Semantic"].value_counts().to_dict()}')
                break
            except Exception as e:
                pass
