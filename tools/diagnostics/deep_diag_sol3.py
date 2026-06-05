"""
INVESTIGACION PROFUNDA — ¿Es la Solución 3 la correcta?
Diagnóstico multicapa: ¿el problema es el modelo, el threshold, los datos, o el filtro?
"""
import sys, pathlib, pandas as pd, numpy as np, json
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')

data = pathlib.Path('g:/Mi unidad/ia/luna_v2/data')
runs = data / 'runs'
SEP = '═' * 68

# ── Encontrar el mejor run con datos de W2 ─────────────────────────────────
w2_probs_files = sorted(runs.rglob('*/W2/oos_raw_probs.parquet'),
                        key=lambda p: p.stat().st_mtime, reverse=True)
w2_trades_files = sorted(runs.rglob('*/W2/oos_trades.parquet'),
                         key=lambda p: p.stat().st_mtime, reverse=True)

print(SEP)
print('ARCHIVOS DISPONIBLES W2:')
for f in w2_probs_files[:5]:
    print(f'  RAW_PROBS: {f.parent.parent.name}/{f.parent.name}')
for f in w2_trades_files[:5]:
    print(f'  TRADES:    {f.parent.parent.name}/{f.parent.name}')

# ── DIAGNÓSTICO 1: Distribución de probabilidades en CALM_BEAR OOS ────────
print()
print(SEP)
print('DIAGNÓSTICO 1 — Prob. del modelo bear_long en barras CALM_BEAR OOS')
print('(Si el modelo genera probs bajas → problema es el modelo; si altas pero threshold alto → problema es el threshold)')
print(SEP)

for probs_fp in w2_probs_files[:3]:
    try:
        df_p = pd.read_parquet(probs_fp)
        seed = probs_fp.parts[-4]
        print(f'\n  {seed}/W2 — {len(df_p)} barras OOS')
        print(f'  Columnas: {list(df_p.columns[:12])}')

        # Filtrar CALM_BEAR
        hmm_col = next((c for c in ['hmm_regime', 'HMM_Semantic', 'hmm_semantic'] if c in df_p.columns), None)
        if hmm_col:
            calm_mask = df_p[hmm_col].str.contains('CALM_BEAR', na=False)
            n_calm    = calm_mask.sum()
            n_total   = len(df_p)
            print(f'  CALM_BEAR bars en OOS: {n_calm}/{n_total} ({n_calm/n_total*100:.1f}%)')

            # Prob del agente bear
            prob_col = next((c for c in ['prob_bear_long', 'bear_long', 'bear_prob', 'xgb_prob_bear_long'] if c in df_p.columns), None)
            if prob_col:
                probs_calm = df_p.loc[calm_mask, prob_col].dropna()
                print(f'\n  Distribución prob_bear_long en barras CALM_BEAR:')
                print(f'    min={probs_calm.min():.4f} | p25={probs_calm.quantile(0.25):.4f} | '
                      f'median={probs_calm.median():.4f} | p75={probs_calm.quantile(0.75):.4f} | max={probs_calm.max():.4f}')
                # ¿Cuántas barras superan distintos thresholds?
                for thr in [0.51, 0.55, 0.58, 0.60, 0.65, 0.70]:
                    n_above = (probs_calm > thr).sum()
                    print(f'    prob > {thr:.2f}: {n_above} barras ({n_above/len(probs_calm)*100:.1f}%)')
            else:
                print(f'  Columnas de probabilidad: {[c for c in df_p.columns if "prob" in c.lower() or "bear" in c.lower()]}')
        else:
            print(f'  Sin columna HMM. Cols: {list(df_p.columns)}')
    except Exception as e:
        print(f'  ERROR: {e}')

# ── DIAGNÓSTICO 2: ¿Cuál es el threshold real del modelo bear? ────────────
print()
print(SEP)
print('DIAGNÓSTICO 2 — Thresholds reales del agente bear_long (desde firmas JSON)')
print(SEP)

models_dir = data / 'models'
for pattern in ['xgboost_meta_bear_long_signature.json', 'xgboost_meta_calm_bear_long_signature.json']:
    for fp in models_dir.glob(pattern):
        try:
            with open(fp) as f:
                sig = json.load(f)
            print(f'\n  {fp.name}:')
            print(f'    optimal_threshold    : {sig.get("optimal_threshold", "N/A"):.4f}')
            print(f'    dsr_oos              : {sig.get("dsr_oos", "N/A"):.4f}')
            print(f'    n_features           : {len(sig.get("features", []))}')
            thr_per_regime = sig.get('optimal_threshold_per_regime', {})
            if thr_per_regime:
                print(f'    threshold_per_regime : {thr_per_regime}')
        except Exception as e:
            print(f'  ERROR {fp.name}: {e}')

# Buscar en los runs de la sesión actual también
for sig_fp in sorted(runs.rglob('*/models/xgboost_meta_bear_long_signature.json'),
                     key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
    try:
        with open(sig_fp) as f:
            sig = json.load(f)
        run_name = sig_fp.parts[-4]
        print(f'\n  RUN {run_name}: bear_long signature')
        print(f'    optimal_threshold    : {sig.get("optimal_threshold", "N/A")}')
        tpr = sig.get('optimal_threshold_per_regime', {})
        print(f'    threshold_per_regime : {tpr}')
    except Exception as e:
        pass

# ── DIAGNÓSTICO 3: ¿Cuántas barras CALM_BEAR tiene W2 en OOS? ────────────
print()
print(SEP)
print('DIAGNÓSTICO 3 — Densidad del régimen CALM_BEAR en OOS por ventana')
print(SEP)

for probs_fp in sorted(runs.rglob('*/oos_raw_probs.parquet'),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:8]:
    try:
        df_p = pd.read_parquet(probs_fp, columns=['HMM_Semantic'] if 'HMM_Semantic' in pd.read_parquet(probs_fp).columns else None)
        if df_p is None or len(df_p) == 0: continue
        hmm_col = next((c for c in ['hmm_regime', 'HMM_Semantic'] if c in df_p.columns), None)
        if not hmm_col: continue
        calm = df_p[hmm_col].str.contains('CALM_BEAR', na=False).sum()
        bull = df_p[hmm_col].str.contains('BULL', na=False).sum()
        total = len(df_p)
        parts = probs_fp.parts
        label = f'{parts[-4]}/{parts[-2]}'
        print(f'  {label:<40} CALM={calm:>5} ({calm/total*100:>4.1f}%) BULL={bull:>5} ({bull/total*100:>4.1f}%) total={total}')
    except Exception:
        pass

# ── DIAGNÓSTICO 4: bull_long — ¿por qué pierde en W1/W3? ─────────────────
print()
print(SEP)
print('DIAGNÓSTICO 4 — bull_long: análisis de pérdidas en W1/W3')
print(SEP)

for trades_fp in sorted(runs.rglob('*/oos_trades.parquet'),
                        key=lambda p: p.stat().st_mtime, reverse=True):
    ventana = trades_fp.parts[-2]
    if ventana not in ('W1', 'W3'): continue
    try:
        df = pd.read_parquet(trades_fp)
        if len(df) < 5: continue
        v = df['return_pct'].values
        wr = (v > 0).sum() / len(v) * 100
        run_name = trades_fp.parts[-4]
        # Mostrar solo los peores
        if wr < 45:
            threshold_col = next((c for c in ['signal_threshold', 'xgb_prob_cal', 'xgb_prob'] if c in df.columns), None)
            thr_val = df[threshold_col].mean() if threshold_col else None
            thr_str = f'avg_thr={thr_val:.3f}' if thr_val else ''
            print(f'  {run_name}/{ventana}: N={len(df)} WR={wr:.1f}% ret={v.sum()*100:+.3f}% {thr_str}')
    except Exception:
        pass

# ── DIAGNÓSTICO 5: señales bloqueadas por signal_filter ──────────────────
print()
print(SEP)
print('DIAGNÓSTICO 5 — ¿Signal filter bloquea señales CALM_BEAR?')
print('(Revisar filtro de veto en signal_filter.py)')
print(SEP)

sf_veto_lines = []
sf_path = pathlib.Path('g:/Mi unidad/ia/luna_v2/luna/models/signal_filter.py')
if sf_path.exists():
    with open(sf_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'CALM_BEAR' in line or ('bear' in line.lower() and 'veto' in line.lower()):
            sf_veto_lines.append((i+1, line.rstrip()))
    for ln, content in sf_veto_lines[:15]:
        print(f'  L{ln}: {content.strip()}')

# ── CONCLUSIÓN ─────────────────────────────────────────────────────────────
print()
print(SEP)
print('EVIDENCIA PARA DECIDIR SOLUCIÓN ÓPTIMA:')
print(SEP)
print('''  Q1: ¿Prob. bear en CALM_BEAR OOS es alta pero threshold la bloquea?
       → Si sí: solución = bajar threshold para calm_bear (sin separar agente)
       → Si no: solución = separar agente (entrenar sin contaminación CRASH)
  
  Q2: ¿CALM_BEAR tiene pocas barras en OOS? 
       → Si <100 barras OOS CALM_BEAR → sample size explica los 3-12 trades
  
  Q3: ¿bull_long tiene WR<45% en BULL régimen?
       → Si sí: el problema principal es bull_long, no bear_long
       → El edge positivo está en calm_bear pero bull_long destruye el ensemble
''')
