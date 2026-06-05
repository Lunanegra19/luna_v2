"""
VERIFICACIÓN PROFUNDA: ¿FIX-D + FIX-C es la mejor opción?
Se investigan 3 preguntas críticas antes de decidir:
  Q1: ¿universal_mode genera modelos útiles o igual de degenerados?
  Q2: ¿La causa raíz es solo MCW, o también gamma/reg interactúan?
  Q3: ¿El agente bear_long tiene valor informacional en este OOS?
"""
import sys, json, re
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

SEP = '═'*72
log = Path('C:/Users/Usuario/.gemini/antigravity-ide/brain/ad23283d-d02e-4616-9748-5d609f02bf06/.system_generated/tasks/task-1314.log').read_text(encoding='utf-8', errors='replace')
lines_log = log.split('\n')

# ═══════════════════════════════════════════════════════════════════════════
# Q1: ¿Cuándo bear usa universal_mode (n_bear=0), produce std_IS>0?
#     Evidencia en el propio log: buscar runs con universal_mode=True
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('Q1: ¿universal_mode produce modelos no-degenerados?')
print('    Busco en logs: POST-FIT IS donde universal_mode=True')
print(SEP)

# Reconstruir: para cada entrenamiento, ¿cuál fue el std_IS Y el modo?
# El pattern es: AUDIT-REGIME-N (universal_mode=X) seguido de POST-FIT IS (std=Y)
universa_stds = []
curr_universal = None
for i, l in enumerate(lines_log):
    if 'AUDIT-REGIME-N' in l and 'bear' in l.lower():
        if 'universal_mode=True' in l:
            curr_universal = True
        elif 'universal_mode=False' in l:
            curr_universal = False
    if 'POST-FIT IS' in l and 'bear' in l.lower() and curr_universal is not None:
        m = re.search(r'std_IS=(\S+)', l)
        if m:
            universa_stds.append({'universal': curr_universal, 'std_IS': float(m.group(1))})
            curr_universal = None

print(f'Total events con modo conocido: {len(universa_stds)}')
for mode in [True, False]:
    subset = [x for x in universa_stds if x['universal'] == mode]
    if subset:
        stds = [x['std_IS'] for x in subset]
        zeros = sum(1 for s in stds if s < 1e-6)
        print(f'  universal_mode={mode}: N={len(subset)} | '
              f'STD=0: {zeros} ({zeros/len(subset)*100:.0f}%) | '
              f'mean_std={np.mean(stds):.4f} max_std={max(stds):.4f}')
print()

# ═══════════════════════════════════════════════════════════════════════════
# Q2: Causa raíz exacta — ¿gamma o MCW?
#     Cuando std_IS=0, ¿qué hiperparámetros eligió Optuna?
# ═══════════════════════════════════════════════════════════════════════════
print(SEP)
print('Q2: ¿Qué hiperparámetros elige Optuna cuando bear colapsa (std_IS=0)?')
print(SEP)

# Extraer params de Optuna para bear — buscar best_params cerca de POST-FIT
import yaml
settings = yaml.safe_load(open('g:/Mi unidad/ia/luna_v2/config/settings.yaml'))
optuna_ss = settings.get('xgboost', {}).get('optuna_search_space', {})
print(f'Search space actual:')
for k in ['min_child_weight_min','min_child_weight_max','gamma_min','gamma_max',
          'reg_alpha_min','reg_alpha_max','reg_lambda_min','reg_lambda_max']:
    print(f'  {k}: {optuna_ss.get(k, "N/A")}')
print()

# Buscar los hyperparámetros del trial ganador para bear en el log
bear_params_lines = [l for l in lines_log if ('best_params' in l.lower() or 'best params' in l.lower() or 'Params:' in l)
                     and 'bear' in l.lower()]
print(f'Líneas con best_params + bear: {len(bear_params_lines)}')
for l in bear_params_lines[:5]:
    print(f'  {l.strip()[:120]}')

# Buscar en torno a cada colapso
collapse_idxs = [i for i, l in enumerate(lines_log)
                  if 'POST-FIT IS' in l and 'bear' in l.lower() and 'std_IS=0.000000' in l]
print(f'\nContexto [-20, 0] de las primeras 3 colapsos — buscando hiperparámetros:')
for ci in collapse_idxs[:3]:
    block = lines_log[max(0, ci-20):ci+1]
    param_lines = [l for l in block if any(k in l for k in
                   ['min_child_weight','gamma','reg_alpha','reg_lambda','n_estimators',
                    'learning_rate','max_depth','Optuna','trial','best'])]
    print(f'\n  Colapso #{collapse_idxs.index(ci)+1}:')
    for l in param_lines:
        print(f'    {l.strip()[:115]}')

# ═══════════════════════════════════════════════════════════════════════════
# Q3: ¿El agente bear_long tiene valor informacional REAL?
#     Test: ¿los trades cuando bear predice alta prob tienen mejor WR
#     que cuando bear predice baja prob?
# ═══════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('Q3: ¿bear_long tiene poder predictivo? (mutual information test)')
print(SEP)

wfb_dir = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')
all_dfs = []
for f in sorted(wfb_dir.glob('oos_trades_W*_seed*.parquet')):
    try:
        df = pd.read_parquet(f)
        if len(df) > 0:
            df['window'] = next(p for p in f.stem.split('_') if p.startswith('W'))
            all_dfs.append(df)
    except:
        pass
df_all = pd.concat(all_dfs, ignore_index=True)

# ¿Existe una columna con la prob del agente bear_long?
bear_cols = [c for c in df_all.columns if 'bear' in c.lower()]
print(f'Columnas relacionadas con bear: {bear_cols}')
print()

# Analizar distribución de prob_cal en trades de régimen CALM_BEAR
bear_trades = df_all[df_all['hmm_regime'].str.contains('BEAR|CALM', case=False, na=False)].copy()
bull_trades  = df_all[df_all['hmm_regime'].str.contains('BULL', case=False, na=False)].copy()
print(f'Trades CALM_BEAR: N={len(bear_trades)}')
print(f'Trades BULL_TREND: N={len(bull_trades)}')

if len(bear_trades) > 5:
    # ¿Se distinguen las probs del modelo en trades bear vs bull?
    for col in ['xgb_prob_cal', 'xgb_prob', 'meta_v2_prob']:
        if col in df_all.columns:
            b_probs = bear_trades[col].dropna()
            B_probs = bull_trades[col].dropna()
            if len(b_probs) >= 5:
                ks, p_ks = stats.ks_2samp(b_probs, B_probs)
                r, p_r   = stats.spearmanr(bear_trades[col].dropna(),
                                            bear_trades.loc[bear_trades[col].notna(), 'return_pct'])
                print(f'\n  {col}:')
                print(f'    bear_mean={b_probs.mean():.4f} bull_mean={B_probs.mean():.4f}')
                print(f'    KS(bear,bull): KS={ks:.4f} p={p_ks:.4f} → '
                      f'{"DISTINTAS" if p_ks<0.05 else "indistinguibles"}')
                print(f'    Spearman(prob, return) en trades BEAR: r={r:+.4f} p={p_r:.4f} → '
                      f'{"señal real" if p_r<0.05 else "RUIDO"}')

# ¿La permanencia en régimen CALM_BEAR tiene duración suficiente?
# SOP R9 exige duración > 120H para evitar micro-régimen
# Cuántas barras consecutivas se etiquetan como CALM_BEAR en OOS
print(f'\nDuración del régimen CALM_BEAR en los trades:')
if 'entry_time' in df_all.columns:
    bear_sorted = bear_trades.sort_values('entry_time')
    if len(bear_sorted) > 1:
        gaps = pd.to_datetime(bear_sorted['entry_time'], utc=True, errors='coerce').diff().dt.total_seconds() / 3600
        print(f'  N trades bear: {len(bear_sorted)}')
        print(f'  Gap medio entre bear trades: {gaps.mean():.1f}H')
        print(f'  ¿Están concentrados (gap<2H): {(gaps<2).sum()} bloques consecutivos')

print()
print(SEP)
print('Q4: ANÁLISIS DIMENSIONAL — ¿Eliminar bear_long es viable?')
print(SEP)

# ¿Qué ocurre con los trades actuales de W2 (la única ventana rentable)?
w2 = df_all[df_all['window'] == 'W2']
w2_bear = w2[w2['hmm_regime'].str.contains('BEAR|CALM', case=False, na=False)]
w2_bull  = w2[~w2['hmm_regime'].str.contains('BEAR|CALM', case=False, na=False)]
print(f'\nW2 (única ventana con R:R=1.006):')
print(f'  Trades BULL:     N={len(w2_bull)} WR={float(w2_bull["is_win"].mean())*100:.1f}%')
print(f'  Trades BEAR/CALM: N={len(w2_bear)} WR={float(w2_bear["is_win"].mean())*100:.1f}%' if len(w2_bear)>0 else '  Trades BEAR/CALM: N=0')

# Seed314/W2 (mejor resultado): ¿tiene trades bear?
seed314_w2 = df_all[(df_all['window']=='W2')]
print(f'\nTodos los trades W2 por régimen:')
for r, g in w2.groupby('hmm_regime'):
    print(f'  {r}: N={len(g)} WR={float(g["is_win"].mean())*100:.1f}% EV={float(g["return_pct"].mean())*100:+.5f}%')

print()
print(SEP)
print('SÍNTESIS ANTES DE DECIDIR EL FIX:')
print(SEP)
