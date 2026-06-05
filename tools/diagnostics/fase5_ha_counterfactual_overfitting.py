"""
FASE 5 — COUNTERFACTUAL ANÁLISIS DE OVERFITTING
Hipótesis H-A: Fix MCW constraint en Optuna para bear_long
=============================================================
Pregunta: ¿Un fix que reduce min_child_weight introduce overfitting?
Método: Análisis teórico + inspección del espacio de búsqueda actual
=============================================================
"""
import sys, json, re
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
from pathlib import Path
import numpy as np

SEP = '─'*72
log_path = Path('C:/Users/Usuario/.gemini/antigravity-ide/brain/ad23283d-d02e-4616-9748-5d609f02bf06/.system_generated/tasks/task-1314.log')
log = log_path.read_text(encoding='utf-8', errors='replace')
lines = log.split('\n')

# ─── 1. ¿Qué hiper-parámetros elige Optuna para bear con n_train pequeño? ─
print(SEP)
print('1. HIPERPARÁMETROS OPTUNA BEAR — ¿qué valores elige con n_train=91-120?')
print(SEP)

# Buscar logs de Optuna para el agente bear
optuna_bear = [l for l in lines if 'bear' in l.lower() and
               ('min_child_weight' in l or 'reg_alpha' in l or 'optuna' in l.lower()
                or 'best_params' in l or 'trial' in l.lower())]
print(f'Líneas Optuna x bear: {len(optuna_bear)}')
for l in optuna_bear[:15]:
    print(f'  {l.strip()[:115]}')
print()

# Buscar el search space de Optuna en settings.yaml
settings_path = Path('g:/Mi unidad/ia/luna_v2/config/settings.yaml')
import yaml
with open(settings_path) as f:
    cfg = yaml.safe_load(f)

print(SEP)
print('2. SEARCH SPACE OPTUNA ACTUAL (settings.yaml)')
print(SEP)

xgb_cfg = cfg.get('xgboost', {})
optuna_cfg = xgb_cfg.get('optuna_search_space', xgb_cfg.get('optuna', {}))
print(f'Configuración XGBoost Optuna:')
for k, v in optuna_cfg.items():
    print(f'  {k}: {v}')
print()

# Buscar específicamente min_child_weight
mcw = optuna_cfg.get('min_child_weight', None)
print(f'min_child_weight bounds: {mcw}')
print()

# ─── 2. Cálculo de riesgo de overfitting ────────────────────────────────────
print(SEP)
print('3. ANÁLISIS TEÓRICO: ¿MCW constraint causa overfitting?')
print(SEP)

n_train_bear_values = [91, 93, 97, 99, 102, 103, 105, 267, 282, 733, 742, 751]
print('Para cada n_train_bear típico, MCW máximo viable sin overfitting:')
print()
print(f'  {"n_train":>8} | {"MCW_max(n/3)":>12} | {"MCW_max(n/5)":>12} | {"n_leaves(n/MCW)":>15} | {"Riesgo":<20}')
print(f'  {"─"*8}-+-{"─"*12}-+-{"─"*12}-+-{"─"*15}-+-{"─"*20}')
for n in n_train_bear_values:
    mcw_n3 = n // 3
    mcw_n5 = n // 5
    leaves_n3 = n // max(mcw_n3, 1)
    riesgo = 'ALTO (n<200)' if n < 200 else ('MEDIO' if n < 400 else 'BAJO')
    print(f'  {n:>8} | {mcw_n3:>12} | {mcw_n5:>12} | {leaves_n3:>15} | {riesgo:<20}')

print()
print('PROBLEMA IDENTIFICADO:')
print('  Con n_train=91 y MCW=30 → max 3 hojas → modelo casi constante')
print('  Con n_train=91 y MCW=5  → max 18 hojas → SOBREAJUSTE a 91 samples')
print('  El fix MCW=n/3 NO resuelve el colapso, solo lo suaviza marginalmente')
print()

# ─── 3. Los 3 fixes alternativos — análisis de overfitting para cada uno ──
print(SEP)
print('4. COMPARATIVA DE FIXES ALTERNATIVOS — Riesgo de overfitting')
print(SEP)
print()
print('FIX-A: Limitar MCW ≤ n_train/3 en Optuna')
print('  Efecto: Optuna puede elegir MCW más bajo → model con más hojas')
print('  Overfitting: ALTO con n_train=91 (solo ~3 hojas de todos modos)')
print('  Cambios training: SÍ (modifica search space IS) → IS/OOS mismatch POSIBLE')
print('  Veredicto: DESCARTADO — riesgo real sin garantía de fix')
print()

print('FIX-B: Activar universal_mode cuando n_bear < umbral (ej: 300)')
print('  Efecto: bear entrena con 700-816 samples IS (todos los regímenes)')
print('  Overfitting: BAJO — más data = menos overfitting')
print('  Pero: modelo bear aprende en datos BULL/RANGE, no solo BEAR')
print('  IS/OOS mismatch: NO — mismo umbral se aplica tanto en IS como OOS')
print('  Veredicto: VIABLE, pero el modelo pierde especialización de régimen')
print()

print('FIX-C: Cuando std_IS=0 → degradar a WARNING (no FATAL) y SKIP bear')
print('  Efecto: El agente bear_long no hace predicciones en esa ventana')
print('  Las 1310 barras bear en OOS no reciben señal → 0 trades bear')
print('  Overfitting: CERO — no se toca el training')
print('  IS/OOS: NINGUNO — solo cambia el manejo del error en runtime')
print('  Veredicto: MÁS SEGURO — elimina los FATAL sin cambiar la señal')
print()

print('FIX-D: Detectar std_IS=0 en POST-FIT y reentrenar con universal_mode')
print('  Efecto: si el modelo colapsa en IS → reentrenar automáticamente con todos los datos')
print('  Overfitting: BAJO — segunda iteración con más data')
print('  IS/OOS: NINGUNO — la detección y corrección ocurren en IS')
print('  Veredicto: MEJOR SOLUCIÓN — corrige el problema en el origen')
print()

# ─── 4. Test estadístico: ¿skip bear cambia mucho el EV? ─────────────────
print(SEP)
print('5. IMPACTO CUANTITATIVO: ¿Cuántos trades bear hay actualmente?')
print(SEP)

import pandas as pd
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
df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

if len(df_all) > 0 and 'hmm_regime' in df_all.columns:
    for regime, grp in df_all.groupby('hmm_regime'):
        n  = len(grp)
        wr = float(grp['is_win'].mean())
        ev = float(grp['return_pct'].mean())
        print(f'  {regime}: N={n} WR={wr*100:.1f}% EV={ev*100:+.5f}%')

    bear_trades = df_all[df_all['hmm_regime'].str.contains('BEAR|CALM', case=False, na=False)]
    all_trades  = len(df_all)
    print(f'\n  Trades en régimen BEAR/CALM: {len(bear_trades)} ({len(bear_trades)/all_trades*100:.1f}% del total)')
    print(f'  Si FIX-C elimina bear: se pierden {len(bear_trades)} trades')
    print(f'  WR bear trades: {float(bear_trades["is_win"].mean())*100:.1f}% '
          f'EV={float(bear_trades["return_pct"].mean())*100:+.5f}%')
    print()

    # ¿El skip de bear mejora el EV global?
    without_bear = df_all[~df_all['hmm_regime'].str.contains('BEAR|CALM', case=False, na=False)]
    with_bear    = df_all
    ev_with    = float(with_bear['return_pct'].mean())
    ev_without = float(without_bear['return_pct'].mean())
    print(f'  EV global CON bear:    {ev_with*100:+.5f}%')
    print(f'  EV global SIN bear:    {ev_without*100:+.5f}%')
    print(f'  Delta:                 {(ev_without-ev_with)*100:+.5f}% por trade')
    from scipy import stats
    t, p = stats.ttest_ind(bear_trades['return_pct'].dropna(),
                            without_bear['return_pct'].dropna())
    print(f'  t-test bear vs no-bear: t={t:.3f} p={p:.4f} → '
          f'{"diferencia SIGNIFICATIVA" if p < 0.05 else "no significativa"}')

print()
print(SEP)
print('CONCLUSIÓN FASE 5:')
print('  FIX-A (MCW constraint):   DESCARTADO — riesgo de overfitting real')
print('  FIX-B (universal_mode):   VIABLE pero pierde especialización')
print('  FIX-C (skip bear → warn): MÁS SEGURO, sin riesgo de overfitting')
print('  FIX-D (reentrenar auto):  MEJOR solución — corrige en origen')
print()
print('  RECOMENDACIÓN: implementar FIX-D como primaria + FIX-C como fallback')
print('  FIX-D: POST-FIT std_IS=0 → reentrenar con universal_mode automáticamente')
print('  FIX-C: si reentrenamiento también colapsa → WARNING + skip (no FATAL)')
print(SEP)
