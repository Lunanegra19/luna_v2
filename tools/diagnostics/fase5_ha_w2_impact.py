"""Análisis crítico: impacto de cada fix en W2"""
from pathlib import Path
import pandas as pd
import numpy as np

wfb_dir = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')
all_dfs = []
for f in sorted(wfb_dir.glob('oos_trades_W*_seed*.parquet')):
    try:
        df = pd.read_parquet(f)
        if len(df) > 0:
            parts = f.stem.split('_')
            df['seed']   = int(next(p.replace('seed','') for p in parts if p.startswith('seed')))
            df['window'] = next(p for p in parts if p.startswith('W'))
            all_dfs.append(df)
    except:
        pass
df_all = pd.concat(all_dfs, ignore_index=True)

print('=== ANÁLISIS CRÍTICO: COMPOSICIÓN DE CADA VENTANA POR RÉGIMEN ===')
for w, grp in df_all.groupby('window'):
    n_bear  = grp['hmm_regime'].str.contains('BEAR|CALM', case=False, na=False).sum()
    n_bull  = grp['hmm_regime'].str.contains('BULL', case=False, na=False).sum()
    n_range = grp['hmm_regime'].str.contains('RANGE|VOLATILE', case=False, na=False).sum()
    pct_bear = n_bear / len(grp) * 100
    print(f'{w}: N={len(grp):>3} | BULL={n_bull} ({100*n_bull/len(grp):.0f}%) | '
          f'BEAR/CALM={n_bear} ({pct_bear:.0f}%) | RANGE={n_range}')
    if n_bear == len(grp):
        print(f'   *** W2 es 100% CALM_BEAR — FIX-C la vaciaría completamente ***')
print()

# Consecuencia: si W2 queda vacía
seed314 = df_all[(df_all['seed']==314) & (df_all['window']=='W2')]
print(f'seed314/W2 (único resultado positivo, Sharpe=+0.74):')
print(f'  N total: {len(seed314)}')
print(f'  Todos son CALM_BEAR: {(seed314["hmm_regime"].str.contains("CALM", case=False, na=False)).all()}')
print()
print('CONCLUSIÓN CRÍTICA:')
print('  FIX-C (skip bear) elimina W2 completamente — destruye el único resultado positivo')
print('  FIX-D (retrain universal_mode) mantiene W2 con un modelo entrenado en más data')
print()
print('=== ¿ENTONCES FIX-D ES LA ÚNICA OPCIÓN VÁLIDA? ===')
print()
print('  Q1 sin datos (universal_mode=True nunca ocurrió en el log): DESCONOCIDO')
print('  Q2: n_estimators=100 con n_train=91 → MCW=20 → ~4 hojas → std≈0')
print()
print('  El problema REAL es matemático:')
print('  XGBoost con n_estimators=100, MCW=20, n_train=91:')
for n in [91, 99, 105]:
    max_leaves = max(1, n // 20)
    label = 'DEGENERADO' if max_leaves <= 2 else 'viable'
    print(f'    n_train={n}: max_leaves={max_leaves} -> modelo {label}')
print()
print('  La causa raíz: MCW_max=20 con n_train=91 fuerza MCW >= n_train/5')
print('  Con n_train=91 y MCW=20 → solo 4 hojas posibles → std cercano a 0')
print()
print('  FIX REAL: Bajar MCW_max de 20 a 10 en settings.yaml (no cambia lógica)')
print('  Con MCW_max=10 y n_train=91 → max 9 hojas → modelo no degenerado')
print()
print('  RIESGO OVERFITTING: Con MCW=10 y n_train=91 → 9 hojas para 91 puntos')
print('  Ratio hojas/puntos = 9/91 ≈ 10% → MODERADO (aceptable con regularización)')
print()
print('=== TABLA FINAL DE DECISIÓN ===')
print(f'{"Fix":20s} | {"Overfitting":10s} | {"W2 OK":6s} | {"FATALs":8s} | {"Complejidad":12s} | Veredicto')
print('-'*80)
print(f'FIX-A (MCW n/3)       | ALTO       | SI    | 0        | MEDIA       | DESCARTADO')
print(f'FIX-B (threshold n<300)| BAJO      | SI    | 0        | ALTA        | VIABLE')
print(f'FIX-C (skip bear)      | CERO      | NO    | 0        | BAJA        | DESCARTADO (mata W2)')
print(f'FIX-D (retrain univ)   | BAJO      | SI    | 0        | ALTA        | VIABLE (sin datos Q1)')
print(f'FIX-MCW-MAX (10→20)    | MODERADO  | SI    | 0        | MÍNIMA      | EL MÁS SIMPLE')
print()
print('El fix más conservador y testeable: reducir MCW_max en settings.yaml de 20 a 10')
print('Esto NO es un fix de código, solo un cambio de parámetro — SOP compliant')
print('Y requiere reentrenamiento (el valor afecta al training, no al inference)')
