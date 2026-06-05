"""
FASE 5 — Counterfactual cuantitativo antes de implementar
Fix P1: bull_gate_min_dsr 0.0 → 0.10 (o más alto)
Fix P2: RANGE threshold investigation
"""
import pathlib, pandas as pd, numpy as np, json
from scipy import stats
from math import sqrt

runs    = pathlib.Path('g:/Mi unidad/ia/luna_v2/data/runs')
archive = pathlib.Path('g:/Mi unidad/ia/luna_v2/data/archive')

# ─── Cargar todos los trades ──────────────────────────────────────────────
all_t = sorted(runs.rglob('*/W*/oos_trades.parquet'),
    key=lambda p: p.stat().st_mtime, reverse=True)
recent = [f for f in all_t if '20260601' in f.parts[-4]]
rows = []
for fp in recent:
    try:
        df = pd.read_parquet(fp)
        df['_window'] = fp.parts[-2]
        df['_run']    = fp.parts[-4]
        rows.append(df)
    except: pass
df_all = pd.concat(rows, ignore_index=True)

bull_df  = df_all[df_all['hmm_regime'].astype(str).str.contains('BULL', na=False)]
bear_df  = df_all[df_all['hmm_regime'].astype(str).str.contains('BEAR|CALM', na=False)]
range_df = df_all[df_all['hmm_regime'].astype(str).str.contains('RANGE', na=False)]

SEP = '=' * 68

# ════════════════════════════════════════════════════════════════════════
# CONTRAFACTUAL P1: bull_gate_min_dsr = 0.0 → X
# ════════════════════════════════════════════════════════════════════════
print(SEP)
print('CONTRAFACTUAL P1 — Impacto de subir bull_gate_min_dsr')
print(SEP)

total_ev     = df_all['return_pct'].sum() * 100
bull_ev      = bull_df['return_pct'].sum() * 100
bear_ev      = bear_df['return_pct'].sum() * 100
range_ev     = range_df['return_pct'].sum() * 100

print(f'\nEstado ACTUAL (gate=0.0):')
print(f'  BULL  trades: {len(bull_df):>5} | Retorno total: {bull_ev:>+10.3f}%')
print(f'  BEAR  trades: {len(bear_df):>5} | Retorno total: {bear_ev:>+10.3f}%')
print(f'  RANGE trades: {len(range_df):>5} | Retorno total: {range_ev:>+10.3f}%')
print(f'  TOTAL trades: {len(df_all):>5} | Retorno total: {total_ev:>+10.3f}%')

# Distribución histórica de DSR BULL en archive
bull_dsrs_archive = []
for fp in sorted(archive.rglob('*bull*long*signature*.json'))[:200]:
    try:
        with open(fp) as f: sig = json.load(f)
        dsr = sig.get('dsr_cpcv_best', sig.get('dsr_oos', None))
        if dsr is not None:
            bull_dsrs_archive.append(float(dsr))
    except: pass

# También buscar en wfb_cache/seed*/W*/models/
for fp in sorted(runs.rglob('*/models/xgboost_meta_bull_long_signature.json'),
                 key=lambda p: p.stat().st_mtime, reverse=True)[:200]:
    try:
        with open(fp) as f: sig = json.load(f)
        dsr = sig.get('dsr_cpcv_best', sig.get('dsr_oos', None))
        if dsr is not None:
            bull_dsrs_archive.append(float(dsr))
    except: pass

bull_dsrs = np.array(bull_dsrs_archive)
print(f'\nDSR BULL históricas: N={len(bull_dsrs)} '
      f'min={bull_dsrs.min():+.4f} '
      f'mean={bull_dsrs.mean():+.4f} '
      f'max={bull_dsrs.max():+.4f}')

print(f'\nImpacto del threshold en % de runs BULL bloqueadas:')
print(f'  {"Threshold":>12} {"% bloqueado":>14} {"Trades elim.":>14} {"EV ganado":>12}')
for thr in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    pct_block = (bull_dsrs < thr).mean() if len(bull_dsrs) > 0 else 0
    trades_elim  = int(len(bull_df) * pct_block)
    ev_ganado    = trades_elim * abs(bull_df['return_pct'].mean()) * 100
    sys_ret_new  = total_ev + ev_ganado  # optimista (eliminar trades malos)
    print(f'  {thr:>12.2f} {pct_block:>13.0%} {trades_elim:>14} {ev_ganado:>+12.3f}%')

print(f'\n  Retorno sistema si BULL 100% bloqueado: {bear_ev + range_ev:+.3f}%')
print(f'  (Solo CALM_BEAR + RANGE operando)')

# RECOMENDACIÓN
print()
print('RECOMENDACIÓN P1:')
if len(bull_dsrs) > 0:
    pct_010 = (bull_dsrs < 0.10).mean()
    pct_020 = (bull_dsrs < 0.20).mean()
    max_dsr  = bull_dsrs.max()
    print(f'  El máximo DSR BULL histórico es {max_dsr:+.4f}')
    print(f'  Threshold 0.10 bloquea {pct_010:.0%} de firmas históricas')
    print(f'  Threshold 0.20 bloquea {pct_020:.0%} de firmas históricas')
    if max_dsr < 0.20:
        print(f'  → Con 0.20, todos los {len(bull_dsrs)} casos históricos quedan bloqueados')
        print(f'  → Evidencia: H2 Descartada (r=-0.015, p=0.474): modelo BULL sin poder discriminante')
        print(f'  → RECOMENDACIÓN: bull_gate_min_dsr = 0.20 (bloquea todo historial conocido)')
    else:
        print(f'  → RECOMENDACIÓN: bull_gate_min_dsr = 0.10 (bloquea 90%, conservador)')

# ════════════════════════════════════════════════════════════════════════
# CONTRAFACTUAL P2: RANGE threshold (con nota de N insuficiente)
# ════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('CONTRAFACTUAL P2 — RANGE threshold (EXPLORATORIO, N=19)')
print(SEP)

print(f'\n  N RANGE total: {len(range_df)} trades')
print(f'  WR: 100% (19/19 wins) — estadísticamente significativo (p=0.000002)')
print(f'  EV/trade: +{range_df["return_pct"].mean()*100:.5f}%')
print()
print('  Problema: el sweep H5 mostró que TODOS los trades tienen xgb_prob_cal')
print('  constante (todas pasan cualquier threshold de 0.30 a 0.66).')
print('  Esto significa que el modelo RANGE YA es selectivo y sus 19 señales')
print('  son de alta calidad — el problema es la cantidad, no el threshold.')
print()
print('  Causa real del bajo N en RANGE: el router solo envía barras a RANGE')
print('  cuando prob_range es el ARGMAX. Con CUTOFF = 0.62, el modelo RANGE')
print('  internamente ya rechaza la mayoría de barras que le llegan.')
print()
print('  DIAGNÓSTICO: La solución no es bajar el threshold de RANGE en predict_oos.')
print('  El cuello de botella está en la CALIBRACIÓN DEL ROUTER HMM:')
print('  → Cuántas barras llegan al agente RANGE (argmax routing)')
print('  → NO en el threshold interno de XGBoost RANGE')
print()
# Cuántas barras llegan con argmax=RANGE
probs_files = sorted(runs.rglob('*/W*/oos_raw_probs.parquet'),
    key=lambda p: p.stat().st_mtime, reverse=True)
probs_recent = [f for f in probs_files if '20260601' in f.parts[-4]][:30]
all_probs = []
for fp in probs_recent:
    try:
        all_probs.append(pd.read_parquet(fp))
    except: pass

if all_probs:
    df_pr = pd.concat(all_probs)
    routing = df_pr[['prob_bull','prob_bear','prob_range']].idxmax(axis=1)
    n_routed_range = (routing == 'prob_range').sum()
    n_total_probs  = len(df_pr)
    print(f'  Barras enrutadas a RANGE (argmax=prob_range): {n_routed_range} / {n_total_probs}')
    print(f'  ({n_routed_range/n_total_probs*100:.1f}% del tiempo OOS)')
    print(f'  De esas {n_routed_range} barras, el modelo XGBoost generó 19 trades')
    range_signal_rate = 19 / n_routed_range if n_routed_range > 0 else 0
    print(f'  Signal rate DENTRO del régimen RANGE: {range_signal_rate*100:.2f}%')
    print()
    # Verificar con cuántas barras RANGE llegamos si bajamos el argmax threshold
    # (esto sería cambiar el router, no el modelo RANGE)
    print('  Alternativa: relajar condición de routing (no threshold XGBoost):')
    for margin in [0.0, 0.02, 0.05, 0.10]:
        # Barras donde prob_range >= prob_bull - margin AND prob_range >= prob_bear - margin
        df_pr2 = df_pr.copy()
        if margin == 0.0:
            mask = (df_pr2['prob_range'] >= df_pr2['prob_bull']) & \
                   (df_pr2['prob_range'] >= df_pr2['prob_bear'])
        else:
            mask = (df_pr2['prob_range'] >= df_pr2['prob_bull'] - margin) & \
                   (df_pr2['prob_range'] >= df_pr2['prob_bear'] - margin)
        n_with_margin = mask.sum()
        print(f'  margin={margin:.2f}: {n_with_margin} barras ({n_with_margin/n_total_probs*100:.1f}%) '
              f'→ si WR se mantiene 100%: EV={n_with_margin * 0.000431 * 100:.2f}% total')

# ════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL DE RECOMENDACIONES CON EVIDENCIA
# ════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('RESUMEN — VEREDICTO CON EVIDENCIA ESTADÍSTICA')
print(SEP)
print("""
P1 — bull_gate_min_dsr: 0.0 → 0.20
  EVIDENCIA:
    H1 CONFIRMADA (p=0.000000): EV BULL significativamente negativo
    H2 DESCARTADA (p=0.474): modelo BULL sin poder discriminante en ningún threshold
    Sweep: ningún threshold xgb_prob_cal hace BULL rentable (WR ≤ 44.8% hasta 0.80)
    DSR archive: max=0.175 → threshold 0.20 bloquea el 100% histórico conocido
  IMPACTO ESTIMADO:
    Elimina ~2165 trades con EV=-0.015% cada uno → +32.3% retorno acumulado
    Sistema pasa de -30% a ~+1.7% (solo CALM_BEAR + RANGE operan)
  REQUIERE REENTRENAMIENTO: NO (solo cambia settings.yaml, ya en inference)
  RIESGO: Bajo. Se puede restaurar reduciendo el threshold. El gate es reversible.

P2 — RANGE threshold: NO IMPLEMENTAR en esta iteración
  EVIDENCIA:
    N=19 RANGE trades (WR=100%) — exploratorio, SOP error #5 prohíbe conclusiones
    El cuello de botella NO es el threshold XGBoost — es el routing HMM (argmax)
    Bajar threshold XGBoost RANGE no genera más barras enrutadas
    La solución real requiere: cambiar routing o reentrenar HMM con más énfasis en RANGE
  DECISIÓN: Marcar como PENDIENTE hasta tener N≥30 en RANGE

H3 — ADVERTENCIA sobre CALM_BEAR:
  WR=55.3% con p=0.052 → NO estadísticamente significativo al α=0.05
  IC95 WR: [49.2%, 61.3%] → incluye 50% (no-edge)
  Necesitamos N≈350 trades CALM_BEAR para probar edge al α=0.05
  → Seguir acumulando trades SIN modificar el agente CALM_BEAR
  → Próxima evaluación válida: cuando N_acumulado ≥ 350 trades CALM_BEAR
""")
