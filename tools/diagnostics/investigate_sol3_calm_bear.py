"""
INVESTIGACION SOLUCION 3 — CALM_BEAR como régimen desperdiciado
Cuantifica barras disponibles, distribución de regímenes y potencial edge
"""
import sys, pathlib, pandas as pd, numpy as np
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')

data = pathlib.Path('g:/Mi unidad/ia/luna_v2/data')
feat = data / 'features'
SEP  = '═' * 70

# ── 1. Distribución HMM Semantic en IS canónico ───────────────────────────
print(SEP)
print('1. DISTRIBUCIÓN DE REGÍMENES HMM EN IS')
print(SEP)
for fname in ['hmm_regime_labels.parquet', 'features_train.parquet', 'features_validation.parquet']:
    fp = feat / fname
    if not fp.exists(): continue
    df = pd.read_parquet(fp, columns=['HMM_Semantic'])
    print(f'\n  {fname} ({len(df)} filas):')
    vc = df['HMM_Semantic'].value_counts()
    total = len(df)
    for regime, n in vc.items():
        pct = n / total * 100
        is_calm = 'CALM_BEAR' in str(regime)
        tag = ' ◄ OBJETIVO' if is_calm else ''
        print(f'    {regime:<30} {n:>6} barras  ({pct:>5.1f}%){tag}')
    break

# ── 2. Impacto específico en ventanas W2/W4 (donde CALM_BEAR domina en OOS) ─
print()
print(SEP)
print('2. ANÁLISIS DE VENTANAS CON RÉGIMEN CALM_BEAR EN OOS')
print(SEP)

# Ver los OOS trades de W2 (el mejor window) — ¿cuántos son CALM_BEAR?
runs = data / 'runs'
best_w2_files = sorted(runs.rglob('*/W2/oos_trades.parquet'), key=lambda p: p.stat().st_mtime, reverse=True)
for fp in best_w2_files[:3]:
    try:
        df = pd.read_parquet(fp)
        if len(df) == 0: continue
        seed = fp.parts[-3]
        n = len(df)
        if 'hmm_regime' in df.columns:
            vc = df['hmm_regime'].value_counts()
            calm = df['hmm_regime'].str.contains('CALM_BEAR', na=False).sum()
        elif 'HMM_Semantic' in df.columns:
            vc = df['HMM_Semantic'].value_counts()
            calm = df['HMM_Semantic'].str.contains('CALM_BEAR', na=False).sum()
        else:
            print(f'  {seed}/W2: sin columna HMM. Cols: {list(df.columns[:8])}')
            continue
        v = df['return_pct'].values
        wr = (v > 0).sum() / n * 100
        ret = v.sum() * 100
        print(f'\n  {seed}/W2: N={n} | WR={wr:.1f}% | ret={ret:+.3f}%')
        print(f'    CALM_BEAR trades: {calm}/{n} ({calm/n*100:.0f}%)')
        print(f'    Distribución HMM: {dict(vc)}')
    except Exception as e:
        print(f'  ERROR: {e}')

# ── 3. ¿Cuántas barras CALM_BEAR hay en el periodo W4 IS? ─────────────────
print()
print(SEP)
print('3. BARRAS CALM_BEAR DISPONIBLES EN CADA VENTANA WFB')
print(SEP)

# Buscar los snapshots de features por ventana
wfb_feat = data / 'wfb_cache'
if wfb_feat.exists():
    for seed_dir in sorted(wfb_feat.iterdir())[:2]:
        if not seed_dir.is_dir(): continue
        for w in ['W1','W2','W3','W4','W5']:
            wdir = seed_dir / w
            for fname in ['features_train.parquet', 'features_is.parquet']:
                fp = wdir / fname
                if not fp.exists(): continue
                try:
                    df = pd.read_parquet(fp, columns=['HMM_Semantic'])
                    n_calm = df['HMM_Semantic'].str.contains('CALM_BEAR', na=False).sum()
                    n_bull = df['HMM_Semantic'].str.contains('BULL', na=False).sum()
                    n_tot  = len(df)
                    print(f'  {seed_dir.name}/{w}: {n_tot} total | CALM_BEAR={n_calm} ({n_calm/n_tot*100:.1f}%) | BULL={n_bull} ({n_bull/n_tot*100:.1f}%)')
                except Exception as e:
                    print(f'  ERROR {seed_dir.name}/{w}: {e}')
                break

# ── 4. El problema real: bear_long trained on CALM_BEAR+BEAR_CRASH juntos ─
print()
print(SEP)
print('4. DIAGNÓSTICO: ¿POR QUÉ CALM_BEAR NO GENERA SEÑALES?')
print(SEP)

# Ver la distribución IS de los 3 sub-regímenes dentro del agente bear
hmm_fp = feat / 'hmm_regime_labels.parquet'
if hmm_fp.exists():
    df = pd.read_parquet(hmm_fp, columns=['HMM_Semantic'])
    total = len(df)
    print('  Composición del agente bear (3_CALM_BEAR + 3_BEAR_CRASH + 4_BEAR_FORCED):')
    bear_regimes = [c for c in df['HMM_Semantic'].unique() if any(
        x in str(c) for x in ['CALM_BEAR','BEAR_CRASH','BEAR_FORCED'])]
    n_bear_total = df['HMM_Semantic'].isin(bear_regimes).sum()
    for r in sorted(bear_regimes):
        n = (df['HMM_Semantic'] == r).sum()
        pct_bear = n / n_bear_total * 100 if n_bear_total > 0 else 0
        pct_tot  = n / total * 100
        print(f'    {r:<30} {n:>6} ({pct_bear:>5.1f}% del agente bear, {pct_tot:>4.1f}% del total IS)')
    print(f'\n  TOTAL barras agente bear: {n_bear_total} ({n_bear_total/total*100:.1f}% del IS)')
    n_calm_total = df['HMM_Semantic'].str.contains('CALM_BEAR', na=False).sum()
    print(f'  Del cual CALM_BEAR solo:  {n_calm_total} ({n_calm_total/n_bear_total*100:.1f}% del agente bear)')
    print(f'\n  → El modelo bear entrena mezclando CALM ({n_calm_total} barras, lento)')
    print(f'    con CRASH ({n_bear_total - n_calm_total} barras, extremo). ')
    print(f'    → En OOS CALM_BEAR el modelo espera volatilidad de CRASH → threshold no se alcanza.')

print()
print(SEP)
print('5. SOLUCION 3 — PROPUESTA: AGENTE calm_bear DEDICADO')
print(SEP)
print('''  Cambio: separar el agente bear en dos agentes especializados
  
  AGENTE bear_crash (renombrado del actual bear):
    regimes: [3_BEAR_CRASH, 3_BEAR_CRASH_B, 3_BEAR_CRASH_C, 4_BEAR_FORCED]
    direction: long (rebotes en crash) o short según se defina
    
  AGENTE calm_bear (NUEVO):
    regimes: [3_CALM_BEAR, 3_CALM_BEAR_B, 3_CALM_BEAR_C, 3_CALM_BEAR_D]
    direction: long (counter-trend en bajada suave)
    Ventaja: 1310 barras limpias de IS sin contaminación de CRASH
    
  Impacto en settings.yaml:
    fase2.regime_mapping:
      bear_crash: [3_BEAR_CRASH, 3_BEAR_CRASH_B, 3_BEAR_CRASH_C, 4_BEAR_FORCED]
      calm_bear:  [3_CALM_BEAR, 3_CALM_BEAR_B, 3_CALM_BEAR_C, 3_CALM_BEAR_D]
      bull:       [1_BULL_TREND, ...]      # sin cambios
      range:      [2_VOLATILE_RANGE, ...]  # sin cambios
      
  Cascada de cambios necesarios:
    1. settings.yaml → regime_mapping
    2. regime_router.py → añadir caso calm_bear en el mapeo de agentes
    3. train_xgboost_v2.py → MultiAgentXGBoostTrainer auto-detecta desde settings (OK)
    4. predict_oos.py → mapeo de agente → TBM profile (línea ~339)
    5. signal_filter.py → _bear_regimes split en dos grupos
''')
