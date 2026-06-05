"""
INVESTIGACIÓN W4 FATAL — Análisis de regímenes HMM por ventana
===============================================================
"""
import sys
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
from pathlib import Path
import pandas as pd
import numpy as np

SEP = '─'*68
cache = Path('g:/Mi unidad/ia/luna_v2/data/wfb_cache/W1/features')

print(SEP)
print('FASE 1: Períodos temporales y regímenes HMM por ventana WFB')
print(SEP)

BEAR_STATES = ['3_BEAR_CRASH', '3_BEAR_CRASH_B', '4_BEAR_FORCED']

for w in ['W1', 'W2', 'W3', 'W4', 'W5']:
    # features_validation = datos IS de esa ventana (el período de entrenamiento)
    val_file = cache / f'features_validation_{w}.parquet'
    hld_file = cache / f'features_holdout_{w}.parquet'
    
    for label, fpath in [('IS (train)', val_file), ('OOS (holdout)', hld_file)]:
        if not fpath.exists():
            continue
        try:
            df = pd.read_parquet(fpath)
            # Período temporal
            if hasattr(df.index, 'min') and pd.api.types.is_datetime64_any_dtype(df.index):
                t_min = str(df.index.min())[:10]
                t_max = str(df.index.max())[:10]
            else:
                t_min = t_max = '?'
            
            if 'HMM_Semantic' not in df.columns:
                print(f'  {w}/{label}: {t_min}→{t_max} | n={len(df)} | SIN HMM_Semantic')
                continue
            
            n_total = len(df)
            regimes = df['HMM_Semantic'].value_counts()
            n_bear = df['HMM_Semantic'].isin(BEAR_STATES).sum()
            n_bull = df['HMM_Semantic'].str.contains('BULL|VOLATILE', case=False, na=False).sum()
            
            print(f'  {w}/{label}: {t_min}→{t_max} | n={n_total}')
            print(f'    bear={n_bear} ({n_bear/n_total*100:.1f}%) | bull/volatile={n_bull} ({n_bull/n_total*100:.1f}%)')
            print(f'    regimes: {regimes.to_dict()}')
        except Exception as e:
            print(f'  {w}/{label}: ERROR {e}')
    print()

print(SEP)
print('FASE 2: ¿Qué período IS cubre W4 exactamente?')
print(SEP)
val_w4 = cache / 'features_validation_W4.parquet'
if val_w4.exists():
    df4 = pd.read_parquet(val_w4)
    idx = df4.index
    print(f'  W4 IS: {str(idx.min())[:16]} → {str(idx.max())[:16]}')
    print(f'  n_total_IS: {len(df4)}')
    if 'HMM_Semantic' in df4.columns:
        for state in BEAR_STATES:
            n = (df4['HMM_Semantic'] == state).sum()
            print(f'  {state}: {n} barras IS ({n/len(df4)*100:.2f}%)')

print()
print(SEP)
print('FASE 3: ¿Cuántos trades OOS bear habría en W4?')
print('(Lo que PERDEMOS si hacemos SKIP de bear_long en W4)')
print(SEP)
hld_w4 = cache / 'features_holdout_W4.parquet'
if hld_w4.exists():
    df4h = pd.read_parquet(hld_w4)
    print(f'  W4 OOS: {str(df4h.index.min())[:16]} → {str(df4h.index.max())[:16]}')
    print(f'  n_total_OOS: {len(df4h)}')
    if 'HMM_Semantic' in df4h.columns:
        for state in BEAR_STATES:
            n = (df4h['HMM_Semantic'] == state).sum()
            print(f'  {state}: {n} barras OOS ({n/len(df4h)*100:.2f}%)')
        n_bear_oos = df4h['HMM_Semantic'].isin(BEAR_STATES).sum()
        print(f'  TOTAL barras bear en OOS W4: {n_bear_oos} ({n_bear_oos/len(df4h)*100:.1f}%)')
        print(f'  → SKIP bear_long costaría máx ~{n_bear_oos} señales potenciales en W4')

print()
print(SEP)
print('FASE 4: ¿Dónde está el parámetro min_viable_train_samples en el código?')
print(SEP)
import subprocess
result = subprocess.run(
    ['grep', '-rn', 'min_viable_train', 'luna/', '--include=*.py'],
    capture_output=True, text=True, cwd='g:/Mi unidad/ia/luna_v2'
)
if result.stdout:
    print(result.stdout[:2000])
else:
    print('  No encontrado "min_viable_train" — buscar alternativas...')
    result2 = subprocess.run(
        ['grep', '-rn', 'n_train.*<\|min_train\|min_samples', 'luna/models/', '--include=*.py'],
        capture_output=True, text=True, cwd='g:/Mi unidad/ia/luna_v2', shell=True
    )
    print(result2.stdout[:2000] if result2.stdout else '  No encontrado')

print()
print(SEP)
print('FASE 5: ¿Cómo se podría implementar SKIP sin violación SOP?')
print(SEP)
print('Opción A — SKIP en train_xgboost_v2.py:')
print('  Si n_train < MIN_BEAR_SAMPLES → guardar MockModel (pred constante=0.0)')
print('  → regime_router ve prob=0.0 → no genera señal → 0 trades bear')
print('  → NO lanza RuntimeError → W4/W5 continúan')
print()
print('Opción B — SKIP en regime_router.py:')
print('  Si n_rows < MIN_BEAR_ROWS_OOS → skipear agente (return prob=0.0 silencioso)')
print('  → WARNING en log, no FATAL')
print()
print('Riesgo overfitting: NULO — SKIP no aprende nada, solo suprime señales en ese régimen')
print('Riesgo SOP: BAJO — No genera resultados espurios, simplemente no opera en ese régimen')
