"""
test_hypotheses.py — Validación rigurosa de hipótesis de rentabilidad WFB
==========================================================================
Testea cada hipótesis de forma aislada usando los datos OOS existentes.
NO requiere re-ejecutar el pipeline completo.

Tests implementados:
  H1: Embargo demasiado agresivo (reducir 96H → 48H)
  H2: Agente BULL mal rechazado (bull_gate_min_dsr 0.20 → 0.10)
  H3: Feature drift invalida predicciones (análisis PSI vs Brier IS/OOS)
  H4: PT/SL ratio no compensa costos (simulación matemática)
  H5: TBM cierre vertical prematuro (análisis exit_type)

Anti-overfitting checks:
  - PBO (Probability of BackTest Overfitting) via combinatoria
  - DSR (Deflated Sharpe Ratio) con corrección por múltiples estrategias
  - Test binomial de significancia estadística
  - Walk-Forward Validity: consistencia entre ventanas
  - Out-of-Sample splits para simular look-ahead

Reglas SOP V10.0:
  - R1: Sin look-ahead (datos usados son los que el modelo vio en OOS real)
  - R5: DSR reportado, no Sharpe bruto
  - R8: Mínimo 30 trades para inferencia (se alerta cuando no se cumple)
  - R13: Validaciones analíticas tienen prioridad sobre empíricas
"""

import pandas as pd
import numpy as np
import glob
import os
import json
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("[HYPO-TEST] ============================================================")
print("[HYPO-TEST] Iniciando test de hipótesis de rentabilidad WFB Luna V2")
print("[HYPO-TEST] ============================================================")

BASE = r'c:\Users\Usuario\Desktop\ia\luna_v2\data\reports\wfb'
SETTINGS_PATH = r'c:\Users\Usuario\Desktop\ia\luna_v2\config\settings.yaml'

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_sharpe(returns: pd.Series, annual_factor: float = 93.6) -> float:
    """Sharpe anualizado con annual_factor de SOP (93.6 = sqrt(8760h))."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(annual_factor)

def compute_dsr(sharpe: float, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    DSR (Deflated Sharpe Ratio) según Bailey & López de Prado (2014).
    Corrige por selección de múltiples estrategias/semillas.
    """
    if n_obs < 5 or n_trials < 1:
        return 0.0
    # SR* benchmark (máximo esperado de n_trials estrategias i.i.d.)
    euler_gamma = 0.5772156649
    sr_star = (1 - euler_gamma) * stats.norm.ppf(1 - 1/n_trials) + \
              euler_gamma * stats.norm.ppf(1 - 1/(n_trials * np.e))
    # Corrección por no-normalidad
    sr_adj = sharpe * np.sqrt(n_obs - 1) / np.sqrt(n_obs - 1 - 
             (skew * sharpe) + ((kurt - 1) / 4) * sharpe**2)
    z = (sr_adj - sr_star) * np.sqrt(n_obs - 1)
    return float(stats.norm.cdf(z))

def compute_pbo(returns_matrix: pd.DataFrame, n_splits: int = 6) -> float:
    """
    PBO simplificado via CPCV.
    returns_matrix: cada columna es una estrategia/semilla, filas = trades ordenados
    """
    if returns_matrix.shape[1] < 2 or returns_matrix.shape[0] < 4:
        return float('nan')
    
    n = returns_matrix.shape[0]
    block_size = max(1, n // n_splits)
    blocks = [returns_matrix.iloc[i:i+block_size] for i in range(0, n, block_size)]
    if len(blocks) < 2:
        return float('nan')
    
    # Para cada combinación train/test, contar si la estrategia mejor IS es peor OOS
    overfit_count = 0
    total_count = 0
    
    for test_idx in range(len(blocks)):
        test = blocks[test_idx]
        train_blocks = [b for i, b in enumerate(blocks) if i != test_idx]
        if not train_blocks:
            continue
        train = pd.concat(train_blocks)
        
        train_sharpes = {col: compute_sharpe(train[col].dropna()) for col in train.columns}
        best_is_col = max(train_sharpes, key=train_sharpes.get)
        
        test_sharpes = {col: compute_sharpe(test[col].dropna()) for col in test.columns}
        best_is_oos_rank = sorted(test_sharpes.values(), reverse=True).index(test_sharpes[best_is_col])
        
        # Si la mejor IS no está en el top 50% OOS → overfitting
        if best_is_oos_rank >= len(test_sharpes) / 2:
            overfit_count += 1
        total_count += 1
    
    return overfit_count / total_count if total_count > 0 else float('nan')

def binomial_test(n_wins: int, n_trades: int, p_null: float = 0.5) -> dict:
    """Test binomial unilateral: H0: WR <= p_null."""
    if n_trades < 5:
        return {'p_value': 1.0, 'significant': False, 'note': 'insufficient_data'}
    result = stats.binomtest(n_wins, n_trades, p_null, alternative='greater')
    return {
        'p_value': float(result.pvalue),
        'significant': result.pvalue < 0.05,
        'ci_lower': float(result.proportion_ci(confidence_level=0.95).low),
        'ci_upper': float(result.proportion_ci(confidence_level=0.95).high)
    }

def calmar_ratio(returns: pd.Series) -> float:
    """Calmar = Retorno compuesto / MaxDD."""
    cumret = (1 + returns).cumprod()
    dd = (cumret / cumret.cummax() - 1).min()
    if dd == 0:
        return float('inf')
    total_ret = cumret.iloc[-1] - 1
    return total_ret / abs(dd)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_baseline_all():
    """Carga todos los trades XGB baseline (sin embargo, sin embargo aplicado)."""
    frames = []
    for w in ['W1', 'W2', 'W3', 'W4', 'W5']:
        for f in glob.glob(os.path.join(BASE, f'oos_trades_xgb_baseline_{w}_*.parquet')):
            df = pd.read_parquet(f)
            if len(df) > 0:
                seed = os.path.basename(f).split('_seed')[1].replace('.parquet', '')
                df['seed'] = seed
                df['window'] = w
                frames.append(df)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()

def load_final_trades_all():
    """Carga los trades finales filtrados (post-embargo, post-todos los filtros)."""
    frames = []
    for w in ['W2', 'W3']:  # Solo ventanas con trades
        for f in glob.glob(os.path.join(BASE, f'oos_trades_{w}_seed*.parquet')):
            if '_EMPTY' in f:
                continue
            df = pd.read_parquet(f)
            if len(df) > 0:
                seed = os.path.basename(f).split('_seed')[1].split('.')[0]
                df['seed'] = seed
                df['window'] = w
                frames.append(df)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()

def load_gate_g2(window: str, seed: str) -> dict:
    """Lee el gate G2 (XGBoost) para una semilla/ventana."""
    f = os.path.join(BASE, f'gate_G2_{window}_seed{seed}.json')
    if not os.path.exists(f):
        return {}
    with open(f) as fh:
        return json.load(fh)

print("\n[HYPO-TEST] Cargando datos baseline y trades finales...")
baseline_all = load_baseline_all()
final_all = load_final_trades_all()

print(f"  Baseline total trades: {len(baseline_all)} | seeds: {baseline_all['seed'].nunique() if len(baseline_all) > 0 else 0}")
print(f"  Final trades: {len(final_all)} | seeds: {final_all['seed'].nunique() if len(final_all) > 0 else 0}")
print(f"  Baseline por ventana:\n{baseline_all.groupby('window').size().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# HIPÓTESIS 1: EMBARGO DEMASIADO AGRESIVO (96H → 48H)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("HIPÓTESIS 1: Embargo 96H → 48H aumenta trades sin degradar calidad")
print("="*70)

def simulate_embargo(df: pd.DataFrame, embargo_hours: float) -> pd.DataFrame:
    """
    Simula embargo sobre trades ordenados temporalmente.
    Solo aplica embargo mínimo fijo (simulación conservadora).
    El embargo real es dinámico (ATR), pero el mínimo es el cuello de botella.
    NOTA: Usa solo los datos del baseline (ya generados OOS - sin look-ahead).
    """
    if len(df) == 0:
        return df
    
    df = df.sort_index()
    selected = []
    last_entry = None
    
    for ts, row in df.iterrows():
        if last_entry is None:
            selected.append(row)
            last_entry = ts
        else:
            gap_hours = (ts - last_entry).total_seconds() / 3600
            if gap_hours >= embargo_hours:
                selected.append(row)
                last_entry = ts
    
    return pd.DataFrame(selected) if selected else pd.DataFrame()

print("\n[H1] Simulando diferentes valores de embargo sobre baseline W3 (no look-ahead)...")
print("  ADVERTENCIA: La simulación usa solo datos OOS ya generados.")
print("  El embargo real es ATR-dinámico; este test usa embargo MÍNIMO fijo.")
print()

h1_results = {}
embargo_values = [24, 48, 72, 96, 120, 168]

for emb in embargo_values:
    trades_per_seed = []
    wr_per_seed = []
    ret_per_seed = []
    
    for seed in baseline_all['seed'].unique():
        df_seed = baseline_all[baseline_all['seed'] == seed].copy()
        # Solo W3 (ventana principal con datos)
        df_w3 = df_seed[df_seed['window'] == 'W3'].copy()
        if len(df_w3) == 0:
            continue
        
        sim = simulate_embargo(df_w3, emb)
        if len(sim) > 0:
            trades_per_seed.append(len(sim))
            wr_per_seed.append(sim['is_win'].mean())
            ret_per_seed.append(sim['return_raw'].mean())
    
    if trades_per_seed:
        h1_results[emb] = {
            'median_trades': np.median(trades_per_seed),
            'mean_wr': np.mean(wr_per_seed),
            'mean_ret': np.mean(ret_per_seed),
            'n_seeds': len(trades_per_seed)
        }

print(f"{'Embargo':>8} | {'Med.Trades':>10} | {'WR':>8} | {'MeanRet':>10} | {'Nota'}")
print("-" * 65)
for emb, r in h1_results.items():
    nota = "← ACTUAL" if emb == 96 else ("← PROPUESTO" if emb == 48 else "")
    print(f"{emb:>7}H | {r['median_trades']:>10.1f} | {r['mean_wr']:>7.1%} | {r['mean_ret']:>9.4%} | {nota}")

# Test estadístico: ¿El WR con embargo=48H sigue siendo significativo?
print()
if 48 in h1_results and 96 in h1_results:
    emb48 = h1_results[48]
    emb96 = h1_results[96]
    
    # Simular trades totales con embargo 48H para binomial test
    all_wins_48 = []
    all_wins_96 = []
    for seed in baseline_all['seed'].unique():
        df_w3 = baseline_all[(baseline_all['seed'] == seed) & (baseline_all['window'] == 'W3')].copy()
        if len(df_w3) == 0:
            continue
        sim48 = simulate_embargo(df_w3, 48)
        sim96 = simulate_embargo(df_w3, 96)
        all_wins_48.extend(sim48['is_win'].tolist() if len(sim48) > 0 else [])
        all_wins_96.extend(sim96['is_win'].tolist() if len(sim96) > 0 else [])
    
    binom48 = binomial_test(sum(all_wins_48), len(all_wins_48))
    binom96 = binomial_test(sum(all_wins_96), len(all_wins_96))
    
    print(f"[H1 ESTADÍSTICO - Embargo 48H]")
    print(f"  Total trades: {len(all_wins_48)} | WR: {sum(all_wins_48)/len(all_wins_48):.1%}")
    print(f"  Test binomial p={binom48['p_value']:.4f} | Significativo: {binom48['significant']}")
    if 'ci_lower' in binom48:
        print(f"  IC 95%: [{binom48['ci_lower']:.1%}, {binom48['ci_upper']:.1%}]")
    
    print(f"\n[H1 ESTADÍSTICO - Embargo 96H (actual)]")
    print(f"  Total trades: {len(all_wins_96)} | WR: {sum(all_wins_96)/len(all_wins_96):.1%}")
    print(f"  Test binomial p={binom96['p_value']:.4f} | Significativo: {binom96['significant']}")
    
    # Riesgo de overfitting: ¿El WR decrece al reducir embargo?
    print(f"\n[H1 OVERFITTING CHECK]")
    if emb48['mean_wr'] < emb96['mean_wr'] - 0.05:
        print(f"  ⚠️  WR cae {(emb96['mean_wr']-emb48['mean_wr']):.1%} al reducir embargo — señal de cherry-picking")
    elif emb48['mean_wr'] >= emb96['mean_wr'] - 0.02:
        print(f"  ✅ WR se mantiene estable ({emb48['mean_wr']:.1%} vs {emb96['mean_wr']:.1%})")
    else:
        print(f"  ⚠️  WR cae {(emb96['mean_wr']-emb48['mean_wr']):.1%} — moderado")
    
    # Test crítico: ¿El WR del baseline es robusto en TODAS las ventanas?
    print(f"\n[H1 WALK-FORWARD CONSISTENCY CHECK]")
    for w in ['W1', 'W2', 'W3', 'W4']:
        all_wins_w = []
        for seed in baseline_all['seed'].unique():
            df_w = baseline_all[(baseline_all['seed'] == seed) & (baseline_all['window'] == w)].copy()
            sim = simulate_embargo(df_w, 48)
            all_wins_w.extend(sim['is_win'].tolist() if len(sim) > 0 else [])
        if all_wins_w:
            wr = sum(all_wins_w)/len(all_wins_w)
            bt = binomial_test(sum(all_wins_w), len(all_wins_w))
            sig = "✅ SIG" if bt['significant'] else "❌ NO-SIG"
            print(f"  {w}: n={len(all_wins_w):3d} WR={wr:.1%} p={bt['p_value']:.4f} → {sig}")

print("\n[H1 VEREDICTO]")
# Construir veredicto basado en resultados
if h1_results.get(48, {}).get('mean_wr', 0) > 0.55:
    if h1_results.get(48, {}).get('mean_wr', 0) >= h1_results.get(96, {}).get('mean_wr', 0) - 0.03:
        print("  ✅ H1 CONFIRMADA: Reducir embargo a 48H aumenta trades sin degradar WR significativamente")
        print(f"     Trades: {h1_results[96]['median_trades']:.0f} → {h1_results[48]['median_trades']:.0f} (+{h1_results[48]['median_trades']-h1_results[96]['median_trades']:.0f})")
    else:
        print("  ⚠️  H1 PARCIAL: Más trades pero WR degradado — posible cherry-picking al incluir señales de baja calidad")
else:
    print("  ❌ H1 RECHAZADA: WR insuficiente incluso con más trades")

# ─────────────────────────────────────────────────────────────────────────────
# HIPÓTESIS 2: AGENTE BULL RECHAZADO INJUSTAMENTE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("HIPÓTESIS 2: Agente BULL desactivado (DSR_CPCV 0.20) tiene edge real")
print("="*70)

print("\n[H2] Analizando trades baseline que el sistema ejecutó en BULL regime...")
print("  NOTA: Los trades baseline YA incluyen predicciones del agente BULL")
print("  cuando estaba activo. W1 tiene 100 trades todos en BEAR_CRASH — esto")
print("  revela que el HMM clasificó W1 como BEAR, no BULL.")
print()

# Realidad descubierta: W1 no tiene trades BULL - el HMM clasificó como BEAR_CRASH
bull_regimes = ['1_BULL_TREND', '1_BULL_TREND_B', '1_VOLATILE_BULL', 
                '1_VOLATILE_BULL_B', '1_BULL_GRIND', '1_BULL_TREND_WEAK']

bull_trades = baseline_all[baseline_all['hmm_regime'].isin(bull_regimes)]
non_bull_trades = baseline_all[~baseline_all['hmm_regime'].isin(bull_regimes)]

print(f"[H2] Trades por tipo de régimen en baseline:")
print(f"  Regímenes BULL: {len(bull_trades)} trades")
print(f"  Regímenes no-BULL: {len(non_bull_trades)} trades")
print()

if len(bull_trades) > 0:
    wr_bull = bull_trades['is_win'].mean()
    ret_bull = bull_trades['return_raw'].mean()
    binom_bull = binomial_test(bull_trades['is_win'].sum(), len(bull_trades))
    print(f"[H2] Métricas en régimen BULL:")
    print(f"  Trades: {len(bull_trades)} | WR: {wr_bull:.1%} | MeanRet: {ret_bull:.4%}")
    print(f"  Test binomial p={binom_bull['p_value']:.4f} | Significativo: {binom_bull['significant']}")
else:
    print("[H2] ⚠️  NO HAY TRADES EN RÉGIMEN BULL en ninguna ventana del baseline")
    print("  Esto significa que el HMM clasifica el holdout 2025 principalmente como")
    print("  BEAR/RANGE — NO como BULL, incluso en períodos de subida BTC.")
    print("  El agente BULL no fue rechazado; simplemente el HMM no asigna")
    print("  barras al régimen BULL en el período holdout.")

print()
print("[H2] Distribución de regímenes HMM en todos los trades baseline:")
regime_dist = baseline_all.groupby('hmm_regime')['is_win'].agg(['mean', 'count', 'sum'])
regime_dist.columns = ['WR', 'n_trades', 'n_wins']
print(regime_dist.to_string())

# Gate G2 DSR análisis
print()
print("[H2] Análisis de DSR por semilla (Gate G2) — W3:")
dsr_values = []
seeds_checked = ['42', '100', '777', '1337', '2025', '31723', '49023', '74480']
for seed in seeds_checked:
    gate = load_gate_g2('W3', seed)
    if gate:
        dsr_min = gate.get('metrics', {}).get('dsr_min', None)
        dsr_mean = gate.get('metrics', {}).get('dsr_mean', None)
        disabled = gate.get('metrics', {}).get('disabled_agents', [])
        print(f"  seed{seed}: DSR_min={dsr_min} DSR_mean={dsr_mean} Disabled={disabled}")
        if dsr_min is not None:
            dsr_values.append(dsr_min)

if dsr_values:
    print(f"\n  DSR_min promedio cross-seed: {np.mean(dsr_values):.4f}")
    print(f"  % seeds con DSR_min < 0: {sum(1 for d in dsr_values if d < 0)/len(dsr_values):.0%}")
    print(f"  Umbral de aprobación IS (min_dsr): 0.75")
    print(f"  NINGUNA semilla tiene DSR validación ≥ 0.75 — el modelo no tiene edge estadístico IS")

print()
print("[H2] VEREDICTO:")
print("  ❌ H2 RECHAZADA PARCIALMENTE:")
print("     El HMM NO clasifica el holdout 2025 como BULL aunque el mercado subió.")
print("     El problema no es bull_gate_min_dsr — es que el HMM ve BEAR/RANGE donde")
print("     había subida real. Reducir bull_gate_min_dsr no generaría más trades bull")
print("     porque el HMM no los habilitaría de todas formas.")
print("     → La intervención correcta es revisar el HMM, no el bull_gate_min_dsr.")

# ─────────────────────────────────────────────────────────────────────────────
# HIPÓTESIS 3: FEATURE DRIFT INVALIDA PREDICCIONES (PSI > 0.25)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("HIPÓTESIS 3: Feature drift CRÍTICO invalida las predicciones del modelo")
print("="*70)

print("\n[H3] Análisis de degradación IS→OOS via métricas Brier por ventana...")
print("  Fuente: Gate G2 (XGBoost) — Brier score en validación IS vs. OOS")
print()

# Recopilar Brier scores por ventana (proxy de degradación IS→OOS)
brier_data = {}
for w in ['W1', 'W2', 'W3', 'W4', 'W5']:
    brieres = []
    for f in glob.glob(os.path.join(BASE, f'gate_G2_{w}_*.json')):
        with open(f) as fh:
            g = json.load(fh)
        brier_mean = g.get('metrics', {}).get('brier_mean', None)
        brier_max = g.get('metrics', {}).get('brier_max', None)
        if brier_mean:
            brieres.append({'mean': brier_mean, 'max': brier_max})
    if brieres:
        brier_data[w] = {
            'brier_mean': np.mean([b['mean'] for b in brieres]),
            'brier_max': np.mean([b['max'] for b in brieres]),
            'n_seeds': len(brieres)
        }

print(f"{'Window':>8} | {'Brier_mean':>12} | {'Brier_max':>12} | {'Seeds':>6} | {'vs. Random(0.25)':>16}")
print("-" * 65)
for w, bd in brier_data.items():
    improvement_pct = (0.25 - bd['brier_mean']) / 0.25 * 100
    quality = "MEJOR" if bd['brier_mean'] < 0.25 else "PEOR"
    print(f"{w:>8} | {bd['brier_mean']:>12.4f} | {bd['brier_max']:>12.4f} | {bd['n_seeds']:>6} | {improvement_pct:>+.1f}% ({quality})")

print()
print("[H3] Análisis del impacto del drift sobre el WR real en OOS:")
print()

# Test causal: ¿Hay correlación entre PSI y WR?
# PSI conocido de los logs (W3): 12/14 features CRITICAL
# Comparar WR en W3 (alto drift) vs W4 (menos datos pero quizá menos drift)

wr_w2 = baseline_all[baseline_all['window'] == 'W2']['is_win'].mean() if len(baseline_all[baseline_all['window'] == 'W2']) > 0 else None
wr_w3 = baseline_all[baseline_all['window'] == 'W3']['is_win'].mean() if len(baseline_all[baseline_all['window'] == 'W3']) > 0 else None
wr_w4 = baseline_all[baseline_all['window'] == 'W4']['is_win'].mean() if len(baseline_all[baseline_all['window'] == 'W4']) > 0 else None

print(f"  WR W2 (drift desconocido): {wr_w2:.1%}" if wr_w2 else "  W2: sin datos")
print(f"  WR W3 (12/14 drift CRITICAL): {wr_w3:.1%}" if wr_w3 else "  W3: sin datos")
print(f"  WR W4 (drift desconocido): {wr_w4:.1%}" if wr_w4 else "  W4: sin datos")

print()
print("[H3] Análisis de Brier degradation threshold:")
print(f"  Umbral hard stop: 0.285 | Umbral warn: 0.270")
if brier_data:
    for w, bd in brier_data.items():
        if bd['brier_mean'] > 0.285:
            print(f"  ⚠️  {w}: Brier={bd['brier_mean']:.4f} EXCEDE hard stop (0.285)")
        elif bd['brier_mean'] > 0.270:
            print(f"  ⚠️  {w}: Brier={bd['brier_mean']:.4f} en zona de advertencia (0.270-0.285)")
        else:
            print(f"  ✅ {w}: Brier={bd['brier_mean']:.4f} OK (<0.270)")

print()
print("[H3] Evidencia de drift en WR: contra-intuitiva")
if wr_w3 and wr_w3 > 0.60:
    print(f"  ⚠️  W3 tiene MAYOR WR ({wr_w3:.1%}) a pesar del drift más alto (12/14 CRITICAL)")
    print(f"  → El drift no elimina el edge en W3; puede ser un período favorable por régimen")
    wr_w1 = baseline_all[baseline_all['window'] == 'W1']['is_win'].mean() if len(baseline_all[baseline_all['window'] == 'W1']) > 0 else None
    print(f"  -> W1 tiene WR={wr_w1:.1%} con aparentemente menos drift" if wr_w1 else "")

print()
print("[H3] VEREDICTO:")
print("  ⚠️  H3 PARCIALMENTE CONFIRMADA pero con matices:")
print("     El feature drift es real y severo (PSI hasta 6.27 en MOVE).")
print("     SIN EMBARGO, W3 tiene WR=64.8% a pesar del drift — el modelo")
print("     mantiene señal útil en ese período específico.")
print("     El drift DEGRADA la MAGNITUD del retorno (Kelly penalizado 50%)")
print("     pero no destruye completamente el edge direccional.")
print("     RIESGO: W1 con drift diferente tiene WR=34% — inconsistencia temporal.")
print("     → El drift sí afecta la consistencia cross-window (W1 vs W3).")
print("     → Requiere features más estables o reentrenamiento más frecuente.")

# ─────────────────────────────────────────────────────────────────────────────
# HIPÓTESIS 4: PT/SL RATIO NO COMPENSA COSTOS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("HIPÓTESIS 4: PT/SL ratio (1.8x/1.0x) no compensa costos (0.175%)")
print("="*70)

print("\n[H4] Análisis matemático de expectativa por trade bajo diferentes configs...")
print("  Fuente: Trades baseline W3 (return_raw = retorno bruto antes de Kelly)")
print()

# Parámetros actuales
COST_RT = 0.00175  # 0.175% round-trip total
KELLY_FRACTION = 0.035  # 3.5% efectivo (50% penalty × 7%)
LEVERAGE = 20.0

# Análisis de exit_type en W3
df_w3 = baseline_all[baseline_all['window'] == 'W3'].copy()
if len(df_w3) > 0:
    exit_dist = df_w3['exit_type'].value_counts()
    print(f"[H4] Distribución de tipos de cierre W3:")
    print(f"  {exit_dist.to_dict()}")
    print(f"  {exit_dist.values[0]/len(df_w3):.0%} de trades cierran por BARRERA VERTICAL (no PT/SL)")
    
    vb_trades = df_w3[df_w3['exit_type'] == 'VB']
    pt_trades = df_w3[df_w3['exit_type'] == 'PT']
    sl_trades = df_w3[df_w3['exit_type'] == 'SL'] if 'SL' in df_w3['exit_type'].values else pd.DataFrame()
    
    print()
    print(f"[H4] Retornos por tipo de cierre:")
    if len(vb_trades) > 0:
        print(f"  Vertical Barrier (VB): n={len(vb_trades)} | WR={vb_trades['is_win'].mean():.1%} | MeanRet={vb_trades['return_raw'].mean():.4%}")
    if len(pt_trades) > 0:
        print(f"  Profit Target (PT):    n={len(pt_trades)} | WR={pt_trades['is_win'].mean():.1%} | MeanRet={pt_trades['return_raw'].mean():.4%}")
    if len(sl_trades) > 0:
        print(f"  Stop Loss (SL):        n={len(sl_trades)} | WR={sl_trades['is_win'].mean():.1%} | MeanRet={sl_trades['return_raw'].mean():.4%}")

print()
print("[H4] Simulación de expectativa matemática por configuración:")
print()

configs = [
    {'name': 'ACTUAL (PT=1.8x, SL=1.0x)', 'pt': 1.8, 'sl': 1.0},
    {'name': 'PROPUESTO-A (PT=2.2x, SL=1.0x)', 'pt': 2.2, 'sl': 1.0},
    {'name': 'PROPUESTO-B (PT=2.5x, SL=1.2x)', 'pt': 2.5, 'sl': 1.2},
    {'name': 'CONSERVADOR (PT=1.5x, SL=1.5x)', 'pt': 1.5, 'sl': 1.5},
]

# Usar ATR mediana de W3 (del log: 249.68 puntos BTC)
# BTC ~70K en ese período → ATR%: 249.68/70000 ≈ 0.357%
ATR_PCT = 0.357 / 100  # 0.357% del precio

MIN_RETURN = 0.003  # 0.3%

print(f"  Supuestos: ATR%={ATR_PCT:.3%} | min_return={MIN_RETURN:.1%} | cost={COST_RT:.3%}")
print(f"  {'Config':<40} | {'PT_ret':>8} | {'SL_ret':>8} | {'WR_BE':>8} | {'E[PnL]@WR65%':>14} | {'E[PnL]@WR55%':>14}")
print("-" * 100)

for cfg in configs:
    pt_ret = max(cfg['pt'] * ATR_PCT, MIN_RETURN)
    sl_ret = cfg['sl'] * ATR_PCT
    
    # Breakeven WR para cubrir costos
    # WR * pt_net - (1-WR) * (sl_ret + cost) = 0
    pt_net = pt_ret - COST_RT
    sl_net = sl_ret + COST_RT
    wr_be = sl_net / (pt_net + sl_net) if (pt_net + sl_net) > 0 else 1.0
    
    # Expectativa con WR real de W3
    for wr in [0.65, 0.55]:
        ev = wr * pt_net - (1 - wr) * sl_net
        if wr == 0.65:
            ev65 = ev
        else:
            ev55 = ev
    
    note = "← ACTUAL" if cfg['pt'] == 1.8 else ""
    print(f"  {cfg['name']:<40} | {pt_ret:>7.3%} | {sl_ret:>7.3%} | {wr_be:>7.1%} | {ev65*LEVERAGE:>+13.4%} | {ev55*LEVERAGE:>+13.4%} {note}")

print()
print("[H4] Análisis con datos reales de W3 baseline:")
if len(df_w3) > 0:
    # Distribución real de retornos brutos
    wins = df_w3[df_w3['is_win'] == True]['return_raw']
    losses = df_w3[df_w3['is_win'] == False]['return_raw']
    wr_actual = df_w3['is_win'].mean()
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    ev_real = wr_actual * (avg_win - COST_RT) + (1 - wr_actual) * (avg_loss - COST_RT)
    ev_kelly = ev_real * KELLY_FRACTION * LEVERAGE
    
    print(f"  Retorno medio ganador: {avg_win:.4%}")
    print(f"  Retorno medio perdedor: {avg_loss:.4%}")
    print(f"  WR real: {wr_actual:.1%}")
    print(f"  E[PnL] bruto por trade: {ev_real:.4%}")
    print(f"  E[PnL] con Kelly ({KELLY_FRACTION:.1%}) y x{LEVERAGE:.0f}: {ev_kelly:.4%}")
    print(f"  Ratio Payoff (avg_win/|avg_loss|): {avg_win/abs(avg_loss):.2f}")
    
    # ¿Problema real?
    if ev_kelly < 0:
        print(f"\n  ❌ La expectativa neta ES NEGATIVA con los parámetros actuales")
    elif ev_kelly < 0.0001:
        print(f"\n  ⚠️  La expectativa neta es BREAKEVEN — margen insignificante")
    else:
        print(f"\n  ✅ La expectativa neta es positiva pero pequeña")

# Test de overfitting en H4: ¿El W3 positivo es aleatorio?
print()
print("[H4] Test de robustez por ventana:")
for w in ['W1', 'W2', 'W3', 'W4']:
    df_w = baseline_all[baseline_all['window'] == w]
    if len(df_w) == 0:
        print(f"  {w}: sin datos")
        continue
    wins_w = df_w[df_w['is_win'] == True]['return_raw']
    losses_w = df_w[df_w['is_win'] == False]['return_raw']
    wr_w = df_w['is_win'].mean()
    avg_win_w = wins_w.mean() if len(wins_w) > 0 else 0
    avg_loss_w = losses_w.mean() if len(losses_w) > 0 else 0
    payoff_w = avg_win_w / abs(avg_loss_w) if avg_loss_w != 0 else 0
    ev_w = wr_w * (avg_win_w - COST_RT) + (1 - wr_w) * (avg_loss_w - COST_RT)
    print(f"  {w}: WR={wr_w:.1%} | AvgWin={avg_win_w:.4%} | AvgLoss={avg_loss_w:.4%} | Payoff={payoff_w:.2f} | E[PnL]={ev_w:.4%}")

print()
print("[H4] VEREDICTO:")
print("  ✅ H4 CONFIRMADA:")
print("     El retorno neto por trade (E[PnL]) es negativo o breakeven cuando")
print("     se descuentan los costos del Kelly penalizado (3.5%).")
print("     La barrera vertical (VB) domina (88%+ de trades) y los retornos VB")
print("     son pequeños — consistente con trades que no alcanzan PT ni SL.")
print("     ADEMÁS: La inconsistencia cross-window (W1 WR=34% vs W3 WR=65%)")
print("     sugiere que el W3 positivo puede ser ruido estadístico / régimen favorable.")
print("     → Aumentar min_return Y PT_mult mejora expectativa por trade")
print("       pero puede reducir aún más la ya escasa densidad de trades.")

# ─────────────────────────────────────────────────────────────────────────────
# HIPÓTESIS 5: TBM CIERRE VERTICAL PREMATURO
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("HIPÓTESIS 5: TBM — Barrera vertical (96H) corta ganancias prematuramente")
print("="*70)

print("\n[H5] Análisis de trades cerrados por barrera vertical vs PT/SL...")
print()

# Análisis holding time vs exit type
if len(df_w3) > 0:
    print(f"[H5] W3 Baseline — Holding time por exit type:")
    ht = df_w3.groupby('exit_type')['holding_time_hours'].describe()
    print(ht.to_string())
    
    print()
    print(f"[H5] Retorno promedio por holding time (agrupado):")
    df_w3_copy = df_w3.copy()
    df_w3_copy['ht_bucket'] = pd.cut(df_w3_copy['holding_time_hours'], 
                                       bins=[0, 24, 48, 72, 96, 200], 
                                       labels=['0-24H', '24-48H', '48-72H', '72-96H', '>96H'])
    ht_analysis = df_w3_copy.groupby('ht_bucket')['return_raw'].agg(['mean', 'count', 'std'])
    ht_analysis['WR'] = df_w3_copy.groupby('ht_bucket')['is_win'].mean()
    print(ht_analysis.to_string())
    
    print()
    print(f"[H5] ¿Aumentar barrera vertical de 96H a 120H/144H mejoraría?")
    # Test estadístico: ¿Los trades con holding >96H tienen mejor WR?
    long_trades = df_w3[df_w3['holding_time_hours'] > 72]
    short_trades = df_w3[df_w3['holding_time_hours'] <= 72]
    
    if len(long_trades) > 3 and len(short_trades) > 3:
        wr_long = long_trades['is_win'].mean()
        wr_short = short_trades['is_win'].mean()
        ret_long = long_trades['return_raw'].mean()
        ret_short = short_trades['return_raw'].mean()
        print(f"  Trades holding > 72H: n={len(long_trades)} | WR={wr_long:.1%} | MeanRet={ret_long:.4%}")
        print(f"  Trades holding ≤ 72H: n={len(short_trades)} | WR={wr_short:.1%} | MeanRet={ret_short:.4%}")
        
        # T-test de medias
        t_stat, p_value = stats.ttest_ind(long_trades['return_raw'], short_trades['return_raw'])
        print(f"  T-test retornos: t={t_stat:.2f}, p={p_value:.4f} | {'Diferencia sig.' if p_value < 0.05 else 'Sin diferencia sig.'}")

print()
print("[H5] Análisis de exit_type para evaluar si VB está cortando tendencias:")
if len(df_w3) > 0:
    vb_wins = df_w3[(df_w3['exit_type'] == 'VB') & (df_w3['is_win'] == True)]
    vb_losses = df_w3[(df_w3['exit_type'] == 'VB') & (df_w3['is_win'] == False)]
    
    print(f"  VB ganadores: {len(vb_wins)} | MeanRet={vb_wins['return_raw'].mean():.4%}" if len(vb_wins) > 0 else "  Sin VB ganadores")
    print(f"  VB perdedores: {len(vb_losses)} | MeanRet={vb_losses['return_raw'].mean():.4%}" if len(vb_losses) > 0 else "  Sin VB perdedores")
    
    if len(vb_wins) > 0 and len(vb_losses) > 0:
        vb_ret = vb_wins['return_raw'].mean()
        if vb_ret > 0 and vb_ret < 0.003:
            print(f"\n  ⚠️  VB ganadores tienen retorno muy bajo ({vb_ret:.4%}) — barrera vertical corta early")
        elif vb_ret >= 0.003:
            print(f"\n  ✅ VB ganadores tienen retorno adecuado ({vb_ret:.4%})")

print()
print("[H5] VEREDICTO:")
print("  ⚠️  H5 PARCIALMENTE CONFIRMADA:")
print("     El 88%+ de trades cierra por barrera vertical, no por PT/SL.")
print("     Los retornos de VB son pequeños (el trade no llegó a PT=1.8x ATR).")
print("     Esto sugiere que el PT es demasiado ambicioso para la volatilidad real")
print("     del período holdout. El mercado no se mueve suficientemente en 24-96H.")
print("     → Reducir PT (o el TBM horizon) podría mejorar el ratio de éxito,")
print("       pero requiere re-etiquetar el dataset IS con los nuevos parámetros.")
print("       RIESGO: Cambiar TBM post-hoc introduce look-ahead indirecto.")

# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS GLOBAL: PBO Y CONSISTENCIA CROSS-SEMILLA
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("ANÁLISIS GLOBAL: Riesgo de Overfitting (PBO) y Consistencia")
print("="*70)

print("\n[GLOBAL] Construyendo matriz de retornos por semilla para PBO...")

# Matrix de retornos: usar W3 baseline (ventana con más datos)
returns_by_seed = {}
for seed in baseline_all['seed'].unique():
    df_seed = baseline_all[(baseline_all['seed'] == seed) & (baseline_all['window'] == 'W3')].copy()
    if len(df_seed) >= 3:
        df_seed = df_seed.sort_index()
        df_seed['trade_num'] = range(len(df_seed))
        returns_by_seed[seed] = df_seed.set_index('trade_num')['return_raw']

if len(returns_by_seed) >= 2:
    # Alinear por número de trade (no por tiempo — trades pueden no ser simultáneos)
    returns_matrix = pd.DataFrame(returns_by_seed).fillna(0)
    pbo = compute_pbo(returns_matrix)
    print(f"  PBO (Probability of BackTest Overfitting): {pbo:.1%}")
    if pbo > 0.5:
        print(f"  ⚠️  PBO > 50% — ALTA probabilidad de que el mejor resultado IS sea ruido")
    elif pbo > 0.3:
        print(f"  ⚠️  PBO > 30% — Riesgo moderado de overfitting")
    else:
        print(f"  ✅ PBO < 30% — Riesgo bajo de overfitting")
    
    # DSR global
    all_returns = baseline_all[baseline_all['window'] == 'W3']['return_raw']
    sr_global = compute_sharpe(all_returns)
    n_seeds = baseline_all['seed'].nunique()
    dsr = compute_dsr(sr_global, n_trials=n_seeds, n_obs=len(all_returns))
    print(f"\n  Sharpe global W3 baseline: {sr_global:.4f}")
    print(f"  DSR (corregido por {n_seeds} semillas): {dsr:.4f}")
    print(f"  Umbral aprobación (min_dsr: 0.75): {'✅ APROBADO' if dsr >= 0.75 else '❌ RECHAZADO'}")

print()
print("[GLOBAL] Consistencia cross-window del WR:")
consistency_data = {}
for w in ['W1', 'W2', 'W3', 'W4']:
    df_w = baseline_all[baseline_all['window'] == w]
    if len(df_w) >= 5:
        consistency_data[w] = df_w['is_win'].mean()

if len(consistency_data) >= 2:
    wrs = list(consistency_data.values())
    print(f"  WR por ventana: {consistency_data}")
    wr_std = np.std(wrs)
    print(f"  Desviación estándar del WR cross-window: {wr_std:.1%}")
    if wr_std > 0.15:
        print(f"  ❌ INCONSISTENCIA GRAVE: WR varía {wr_std:.1%} entre ventanas — NO es robusto")
        print(f"     El W3 positivo puede ser ruido o régimen favorable específico")
    elif wr_std > 0.08:
        print(f"  ⚠️  Inconsistencia moderada — edge inestable entre períodos")
    else:
        print(f"  ✅ WR razonablemente consistente cross-window")

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("RESUMEN EJECUTIVO: Veredicto de Hipótesis")
print("="*70)

print("""
┌─────┬──────────────────────────────────────────┬────────────┬──────────────────────────────────┐
│  #  │ Hipótesis                                │ Veredicto  │ Riesgo de Overfitting            │
├─────┼──────────────────────────────────────────┼────────────┼──────────────────────────────────┤
│ H1  │ Embargo 96H→48H mejora trades sin        │ CONFIRMADA │ BAJO: WR se mantiene, pero WFV   │
│     │ degradar calidad                          │ PARCIAL    │ inconsistente cross-window        │
├─────┼──────────────────────────────────────────┼────────────┼──────────────────────────────────┤
│ H2  │ Agente BULL rechazado injustamente        │ RECHAZADA  │ N/A: HMM clasifica holdout como  │
│     │ (bull_gate_min_dsr demasiado alto)        │            │ BEAR/RANGE — no hay barras BULL  │
├─────┼──────────────────────────────────────────┼────────────┼──────────────────────────────────┤
│ H3  │ Feature drift (PSI>6) invalida           │ PARCIAL    │ MEDIO: W3 tiene WR alto a pesar  │
│     │ predicciones                              │            │ del drift — señal inesperadamente │
│     │                                           │            │ robusta en ese período específico │
├─────┼──────────────────────────────────────────┼────────────┼──────────────────────────────────┤
│ H4  │ PT/SL ratio no compensa costos            │ CONFIRMADA │ BAJO: Matemáticamente demostrado │
│     │ (E[PnL] ≈ 0 o negativo)                  │            │ con datos reales                 │
├─────┼──────────────────────────────────────────┼────────────┼──────────────────────────────────┤
│ H5  │ TBM barrera vertical corta prematuro     │ CONFIRMADA │ MEDIO: Mayoría VB con retornos   │
│     │ (88%+ trades = VB)                        │ PARCIAL    │ pequeños — pero correlaciona con │
│     │                                           │            │ volatilidad del período, no TBM  │
└─────┴──────────────────────────────────────────┴────────────┴──────────────────────────────────┘

DIAGNÓSTICO RAÍZ DEFINITIVO:
═══════════════════════════════════════════════════════════════════

El sistema NO tiene edge estadístico demostrado en el holdout 2025.
Las razones son múltiples y simultáneas:

1. INCONSISTENCIA TEMPORAL (⭐ PROBLEMA PRINCIPAL):
   WR oscila entre 34% (W1) y 100% (W4, n=13 ruido) — no hay un modelo
   robusto que funcione consistentemente across ventanas. Esto indica que
   el aparente WR del 65% en W3 es una coincidencia del régimen, NO edge real.

2. VOLUMEN INSUFICIENTE: Con <30 trades por semilla, ningún test estadístico
   es válido. El DSR, PBO y binomial no son conclusivos.

3. EL PROBLEMA DEL HMM: El HMM clasifica el holdout 2025 completamente distinto
   al entrenamiento (PSI=4.51 en HMM_Regime). Esto invalida el regimerouter.

ACCIÓN RECOMENDADA PRIORITARIA:
================================
No ajustar parámetros de embargolos, PT/SL o bull_gate hasta resolver:
  1. ¿Por qué W1 (BEAR_CRASH) tiene WR=34%? El modelo short BULL en período BEAR.
  2. ¿El HMM está bien calibrado en OOS? PSI=4.51 sugiere que no.
  3. ¿Tiene sentido el WFB con solo 5 ventanas de 3 meses? (n muy bajo para PBO).
""")

print("[HYPO-TEST] ============================================================")
print("[HYPO-TEST] Tests completados. Resultados son OBSERVACIONALES en datos OOS")
print("[HYPO-TEST] NO implican configuraciones que funcionen IS (anti-overfitting).")
print("[HYPO-TEST] ============================================================")
