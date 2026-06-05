"""
H-NEW-2: Bug de Clones — Identificación de causa raíz
=======================================================
Seeds 789/42975/44085/36457 producen exactamente N=51, WR=43.1%, MeanRet=-0.019% en W1.
KS-test p=1.0 confirma distribuciones idénticas.

Investigar QUÉ parte del pipeline no usa la seed correctamente.
Candidatos: TBM labels, SFI feature selection, XGBoost Optuna seed, HMM labels.
"""
import sys
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib

SEP = '─'*68

CLONE_SEEDS = [789, 42975, 44085, 36457]

print(SEP)
print('H-NEW-2 FASE 1: Verificar identidad exacta de los trades (nivel fila)')
print(SEP)

wfb_dir = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')
dfs_clone = {}
for seed in CLONE_SEEDS:
    f = wfb_dir / f'oos_trades_W1_seed{seed}.parquet'
    if f.exists():
        df = pd.read_parquet(f)
        dfs_clone[seed] = df
        print(f'  seed{seed}/W1: N={len(df)} WR={float(df["is_win"].mean())*100:.1f}% '
              f'EV={float(df["return_pct"].mean())*100:+.5f}%')

if len(dfs_clone) >= 2:
    seeds = list(dfs_clone.keys())
    ref   = dfs_clone[seeds[0]]
    print()
    print(f'  Comparación columna por columna (ref=seed{seeds[0]}):')
    compare_cols = ['entry_time','return_pct','is_win','xgb_prob','xgb_prob_cal']
    for col in compare_cols:
        if col not in ref.columns: continue
        for s in seeds[1:]:
            df2 = dfs_clone[s]
            if col not in df2.columns: continue
            if ref[col].dtype == object or ref[col].dtype.name == 'datetime64[ns, UTC]':
                identical = (ref[col].values == df2[col].values).all()
            else:
                identical = np.allclose(ref[col].values, df2[col].values, equal_nan=True)
            print(f'    {col:25s} | seed{s}: {"IDÉNTICO" if identical else "DISTINTO"}')
    print()

print(SEP)
print('H-NEW-2 FASE 2: Comparar modelos IS guardados (features usadas)')
print(SEP)

models_base = Path('g:/Mi unidad/ia/luna_v2/data/models/prod')
for seed in CLONE_SEEDS:
    seed_dir = models_base / f'seed{seed}'
    if not seed_dir.exists():
        print(f'  seed{seed}: directorio no encontrado')
        continue
    files = list(seed_dir.glob('*.json')) + list(seed_dir.glob('*.joblib'))
    print(f'  seed{seed}: {len(files)} archivos en models/prod/')
    # Hash de los modelos
    for f in sorted(files)[:5]:
        try:
            h = hashlib.md5(f.read_bytes()).hexdigest()[:10]
            print(f'    {f.name:55s} md5={h}')
        except: pass

print()
print(SEP)
print('H-NEW-2 FASE 3: Comparar features IS (¿mismas features seleccionadas?)')
print(SEP)

# Las features seleccionadas están en el parquet de features_train
features_dir = Path('g:/Mi unidad/ia/luna_v2/data')
for seed in CLONE_SEEDS:
    # Buscar feature metadata o selected_features
    for pattern in [f'*seed{seed}*features*', f'*{seed}*sfi*', f'*{seed}*selected*']:
        found = list(features_dir.rglob(pattern))
        if found:
            print(f'  seed{seed}: {[f.name for f in found[:3]]}')
            break
    else:
        print(f'  seed{seed}: no se encontró features IS en data/')

print()
print(SEP)
print('H-NEW-2 FASE 4: Revisar cómo el orquestador WFB pasa la seed al pipeline')
print(SEP)

# Buscar en el código cómo se usa la seed
orchestrator = Path('g:/Mi unidad/ia/luna_v2/scripts/run_wfb_orchestrator.py')
if orchestrator.exists():
    code = orchestrator.read_text(encoding='utf-8', errors='replace')
    code_lines = code.split('\n')
    seed_lines = [(i+1, l) for i, l in enumerate(code_lines)
                  if 'seed' in l.lower() and ('random' in l.lower() or 'np.random' in l.lower()
                      or 'set_seed' in l.lower() or 'SEED' in l or 'optuna' in l.lower())]
    print(f'  Líneas de orquestador relacionadas con seed+random:')
    for ln, l in seed_lines[:15]:
        print(f'    L{ln:4d}: {l.rstrip()[:110]}')

print()
print(SEP)
print('H-NEW-2 FASE 5: ¿El SFI usa la seed para selección de features?')
print(SEP)

sfi_path = Path('g:/Mi unidad/ia/luna_v2/luna/features/feature_selection_e.py')
if sfi_path.exists():
    code_sfi = sfi_path.read_text(encoding='utf-8', errors='replace')
    sfi_lines = code_sfi.split('\n')
    seed_sfi = [(i+1, l) for i, l in enumerate(sfi_lines)
                if 'seed' in l.lower() and ('random' in l.lower() or 'np.random' in l.lower()
                    or 'random_state' in l.lower() or 'RandomState' in l)]
    print(f'  SFI líneas con seed+random (primeras 15):')
    for ln, l in seed_sfi[:15]:
        print(f'    L{ln:4d}: {l.rstrip()[:110]}')

print()
print(SEP)
print('H-NEW-2 CONCLUSIÓN (provisional):')
print(SEP)
