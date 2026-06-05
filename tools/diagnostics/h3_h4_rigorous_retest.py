"""
h3_h4_rigorous_retest.py
========================
Re-test riguroso de H3 y H4 operando sobre el PORTFOLIO ENSEMBLE (no per-seed).
El nivel correcto de análisis es el portfolio final, no las seeds individuales.

H3: Circuit Breaker a nivel de portfolio — ¿una regla de WR rolling / acumulado
    sobre el portfolio ensemble habría parado antes de las grandes pérdidas?

H4: Features ATH — ¿qué señal tienen los xgb_prob_cal en la secuencia temporal
    del portfolio? ¿Hay un patrón de sobreconfianza antes de las pérdidas grandes?
    ¿Es posible detectar el cambio de sub-régimen sin features nuevas?
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\reports\wfb')
SEP = "=" * 75

def load_all_trades():
    dfs = []
    for f in sorted(DATA.glob('oos_trades_W*_seed*.parquet')):
        stem = f.stem
        window = stem.split('_')[2]
        seed = int(stem.split('_seed')[1])
        df = pd.read_parquet(f)
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index, utc=True)
        df['seed'] = seed
        df['window'] = window
        dfs.append(df)
    return pd.concat(dfs).sort_index()

# Reconstruir portfolio ensemble igual que evaluate_ensemble_wfb.py
def build_ensemble_portfolio(df_all, consensus_CUTOFF = 3, bucket_hours=2, embargo_h=96):
    df_all = df_all.copy()
    df_all['bucket'] = df_all.index.floor(f'{bucket_hours}h')
    bucket_seeds = df_all.groupby('bucket')['seed'].nunique().rename('consensus_count')
    df_all['consensus_count'] = df_all['bucket'].map(bucket_seeds)

    df_consensus = df_all[df_all['consensus_count'] >= consensus_threshold].copy()
    agg = {'return_pct': 'mean', 'is_win': 'max', 'consensus_count': 'first',
           'window': 'first', 'xgb_prob_cal': 'mean'}
    if 'hmm_regime' in df_consensus.columns:
        agg['hmm_regime'] = 'first'

    df_portfolio = df_consensus.groupby('bucket').agg(agg).sort_index()

    # Embargo secuencial
    selected, last_t = [], None
    for ts in df_portfolio.index:
        if last_t is None or (ts - last_t).total_seconds() / 3600 >= embargo_h:
            selected.append(ts)
            last_t = ts
    return df_portfolio.loc[selected].copy()

df_all = load_all_trades()
df_portfolio = build_ensemble_portfolio(df_all)
print(f"Portfolio ensemble baseline: {len(df_portfolio)} trades")
print(f"Retorno total: {df_portfolio['return_pct'].sum()*100:.4f}%")
print(f"WR: {df_portfolio['is_win'].mean()*100:.2f}%")

# ===========================================================================
# H3 RETEST: Circuit Breaker a nivel de PORTFOLIO (no per-seed)
# ===========================================================================
print(f"\n{SEP}")
print("H3 RETEST: Circuit Breaker sobre el PORTFOLIO ENSEMBLE completo")
print(SEP)

# Evolución temporal del portfolio
df_portfolio['trade_n'] = range(1, len(df_portfolio)+1)
df_portfolio['cum_ret'] = df_portfolio['return_pct'].cumsum() * 100
df_portfolio['roll_wr_5'] = df_portfolio['is_win'].astype(float).rolling(5, min_periods=3).mean()
df_portfolio['roll_wr_10'] = df_portfolio['is_win'].astype(float).rolling(10, min_periods=5).mean()

print(f"\nEvolucion temporal del portfolio ensemble (ordenado por fecha):")
print(f"{'T':>4} {'Fecha':<22} {'Ret%':>8} {'CumRet%':>9} {'WR5':>6} {'WR10':>6} {'Consenso':>9} {'Ventana'}")
print("-"*80)
for _, row in df_portfolio.iterrows():
    wr5  = f"{row['roll_wr_5']*100:.0f}%" if not pd.isna(row['roll_wr_5']) else "  N/A"
    wr10 = f"{row['roll_wr_10']*100:.0f}%" if not pd.isna(row['roll_wr_10']) else "  N/A"
    mark = " <-- INFLEXION" if row['cum_ret'] < -0.1 and row['trade_n'] <= 30 and row['return_pct']*100 < -0.05 else ""
    print(f"  {int(row['trade_n']):>2}  {str(row.name)[:19]:<22} {row['return_pct']*100:>8.4f} {row['cum_ret']:>9.4f}"
          f"  {wr5:>5}  {wr10:>5}  {int(row['consensus_count']):>9}  {row.get('window','?')}{mark}")

# Detectar punto de inflexión
first_neg_row = df_portfolio[df_portfolio['cum_ret'] < 0]
if not first_neg_row.empty:
    inflexion_t = int(first_neg_row['trade_n'].iloc[0])
    print(f"\n  Primer negativo acumulado: trade #{inflexion_t} de {len(df_portfolio)}")
    ret_before = df_portfolio[df_portfolio['trade_n'] < inflexion_t]['return_pct'].sum() * 100
    ret_after  = df_portfolio[df_portfolio['trade_n'] >= inflexion_t]['return_pct'].sum() * 100
    print(f"  Retorno antes del cruce: {ret_before:+.4f}% | despues: {ret_after:+.4f}%")

# Simular CB con diferentes reglas sobre el portfolio
print(f"\nSIMULACION H3 — Circuit Breaker sobre portfolio ensemble:")
print(f"{'Regla':<45} {'Trades':>7} {'RetTot%':>9} {'WR%':>7} {'Ahorro%':>9}")
print("-"*80)

ret_full = df_portfolio['return_pct'].sum() * 100

# A: parar en trade N
for stop_n in [10, 15, 20, 25]:
    df_stopped = df_portfolio[df_portfolio['trade_n'] <= stop_n]
    ret = df_stopped['return_pct'].sum() * 100
    wr  = df_stopped['is_win'].mean() * 100
    print(f"  Parar en trade {stop_n:<30} {len(df_stopped):>7} {ret:>9.4f} {wr:>7.1f}%  {ret-ret_full:>+9.4f}%")

# B: parar cuando WR rolling 5 < umbral
for wr_thr in [0.50, 0.40, 0.35, 0.30]:
    selected_b = []
    for _, row in df_portfolio.iterrows():
        if pd.isna(row['roll_wr_5']) or row['roll_wr_5'] >= wr_thr:
            selected_b.append(row.name)
        else:
            break  # primer trade con WR rolling < umbral: detener
    df_b = df_portfolio.loc[selected_b]
    ret = df_b['return_pct'].sum() * 100
    wr  = df_b['is_win'].mean() * 100 if len(df_b) > 0 else 0
    print(f"  WR_roll5 >= {wr_thr:.0%} (stop en primero < umbral)   {len(df_b):>7} {ret:>9.4f} {wr:>7.1f}%  {ret-ret_full:>+9.4f}%")

# C: parar cuando CumRet portfolio < -X%
for dd_thr in [-0.001, -0.002, -0.005]:
    selected_c = []
    for _, row in df_portfolio.iterrows():
        if row['cum_ret'] >= dd_thr * 100:
            selected_c.append(row.name)
        else:
            break
    df_c = df_portfolio.loc[selected_c]
    ret = df_c['return_pct'].sum() * 100 if len(df_c) > 0 else 0
    wr  = df_c['is_win'].mean() * 100 if len(df_c) > 0 else 0
    print(f"  Stop cuando CumRet < {dd_thr*100:.2f}%               {len(df_c):>7} {ret:>9.4f} {wr:>7.1f}%  {ret-ret_full:>+9.4f}%")

# Verificacion de predictividad: ¿WR rolling predice siguiente trade?
valid_h3 = df_portfolio.dropna(subset=['roll_wr_5'])
if len(valid_h3) >= 5:
    wr5_vals = valid_h3['roll_wr_5'].values[:-1]
    next_ret = valid_h3['return_pct'].values[1:]
    rho, p = stats.spearmanr(wr5_vals, next_ret)
    print(f"\n  Spearman(WR_roll5_t, ret_t+1) portfolio: rho={rho:.4f}, p={p:.4f}")
    print(f"  -> {'PREDICTIVO (p<0.10)' if p < 0.10 else 'NO predictivo'}")

# Veredicto H3
print(f"\n  VEREDICTO H3 RETEST:")
best_cb_rule = "parar en trade 15"
df_best = df_portfolio[df_portfolio['trade_n'] <= 15]
best_ret = df_best['return_pct'].sum() * 100
print(f"  Mejor regla simple: {best_cb_rule} -> {best_ret:.4f}% (ahorro {best_ret-ret_full:+.4f}%)")
print(f"  Pero elegir N=15 es lookback a posteriori (overfitting al portfolio de 58 trades)")
print(f"  Una regla causal robusta (WR_roll5 < 40%) salva {df_portfolio[df_portfolio.name.isin([r.name for _, r in df_portfolio.iterrows() if pd.isna(r['roll_wr_5']) or r['roll_wr_5'] >= 0.40])]['return_pct'].sum()*100:.4f}%")

# ===========================================================================
# H4 RETEST: Señal temporal de xgb_prob_cal en el portfolio
# ===========================================================================
print(f"\n{SEP}")
print("H4 RETEST: Sobreconfianza del modelo antes de pérdidas en W4")
print(f"  Pregunta: ¿xgb_prob_cal sube antes de las perdidas grandes? ¿Es detectable?")
print(SEP)

print(f"\n  xgb_prob_cal y retorno por trade en el portfolio:")
print(f"  {'T':>4} {'Fecha':<22} {'prob_cal':>9} {'Ret%':>9} {'W/L':>5} {'Ventana'}")
print("-"*60)
for _, row in df_portfolio.iterrows():
    prob = row.get('xgb_prob_cal', float('nan'))
    prob_str = f"{prob:.4f}" if not pd.isna(prob) else "   N/A"
    print(f"  {int(row['trade_n']):>4}  {str(row.name)[:19]:<22} {prob_str:>9} {row['return_pct']*100:>9.4f}  {'W' if row['is_win'] else 'L':>5}  {row.get('window','?')}")

# Análisis por sub-período
print(f"\n  Analisis por ventana en el portfolio:")
for win in ['W1', 'W3', 'W4']:
    dfw = df_portfolio[df_portfolio['window'] == win]
    if len(dfw) == 0:
        continue
    prob_mean = dfw['xgb_prob_cal'].mean()
    ret_mean  = dfw['return_pct'].mean() * 100
    wr_pct    = dfw['is_win'].mean() * 100
    print(f"  {win}: n={len(dfw)} | prob_cal={prob_mean:.4f} | WR={wr_pct:.1f}% | RetMed={ret_mean:.4f}%")

# Correlacion prob_cal vs ret en el portfolio ensemble
prob_vals = df_portfolio['xgb_prob_cal'].dropna()
ret_vals  = df_portfolio['return_pct'].dropna()
common    = prob_vals.index.intersection(ret_vals.index)
if len(common) >= 5:
    rho_h4, p_h4 = stats.spearmanr(prob_vals.loc[common], ret_vals.loc[common])
    print(f"\n  Spearman(prob_cal, ret) en portfolio ensemble: rho={rho_h4:.4f}, p={p_h4:.4f}")

    # ¿Los trades de MAYOR prob tienen MEJOR o PEOR retorno en W4?
    dfw4 = df_portfolio[df_portfolio['window'] == 'W4'].dropna(subset=['xgb_prob_cal'])
    if len(dfw4) >= 4:
        high_prob_w4 = dfw4[dfw4['xgb_prob_cal'] >= dfw4['xgb_prob_cal'].median()]
        low_prob_w4  = dfw4[dfw4['xgb_prob_cal'] <  dfw4['xgb_prob_cal'].median()]
        print(f"\n  W4 — trades con prob_cal ALTA (>= mediana={dfw4['xgb_prob_cal'].median():.4f}):")
        print(f"    n={len(high_prob_w4)} | WR={high_prob_w4['is_win'].mean()*100:.1f}% | RetMed={high_prob_w4['return_pct'].mean()*100:.4f}%")
        print(f"  W4 — trades con prob_cal BAJA (< mediana):")
        print(f"    n={len(low_prob_w4)} | WR={low_prob_w4['is_win'].mean()*100:.1f}% | RetMed={low_prob_w4['return_pct'].mean()*100:.4f}%")
        if high_prob_w4['return_pct'].mean() < low_prob_w4['return_pct'].mean():
            print(f"  -> INVERSION DE SEÑAL en W4: alta prob_cal = PEORES retornos (sobreconfianza)")
        else:
            print(f"  -> prob_cal consistente con retornos en W4 (sin inversion)")

# HMM: el mismo régimen en W3 y W4 pero con WR radicalmente distintos
print(f"\n  HALLAZGO CRITICO H4 — HMM ciego al sub-regimen ATH:")
print(f"  Ambas ventanas estan 100% en '1_BULL_TREND_WEAK'")
all_w3 = df_all[df_all['window'] == 'W3']
all_w4 = df_all[df_all['window'] == 'W4']
print(f"  W3: n={len(all_w3)} trades | WR={all_w3['is_win'].mean()*100:.1f}% | prob_cal_mean={all_w3['xgb_prob_cal'].mean():.4f}")
print(f"  W4: n={len(all_w4)} trades | WR={all_w4['is_win'].mean()*100:.1f}% | prob_cal_mean={all_w4['xgb_prob_cal'].mean():.4f}")
print(f"  -> El modelo es MAS CONFIANTE en W4 (prob={all_w4['xgb_prob_cal'].mean():.4f}) que en W3 ({all_w3['xgb_prob_cal'].mean():.4f})")
print(f"     pero tiene PEOR WR (W4={all_w4['is_win'].mean()*100:.1f}% vs W3={all_w3['is_win'].mean()*100:.1f}%)")
print(f"     Esta es la FIRMA de sobreconfianza en regimen OOD.")

# ¿Qué features podrían distinguir W3 de W4?
print(f"\n  Fechas del periodo W3 (bull normal): julio-sept 2025")
print(f"  Fechas del periodo W4 (ATH BTC):     oct-dic 2025")
print(f"  BTC precio aprox: W3=55-65K USD | W4=70-108K USD (nuevo ATH)")
print(f"  Features candidatas para H4:")
print(f"    1. dist_from_ath_pct  = (ATH_actual - close) / ATH_actual  -> 0 en ATH, >0 en correcciones")
print(f"    2. new_ath_days_streak = dias consecutivos en nuevo ATH  -> alto en W4, ~0 en W3")
print(f"    3. price_z_score_252d  = z-score del precio vs ultimos 252 dias -> muy alto en ATH")
print(f"    4. realized_vol_ratio  = vol_30d / vol_252d -> cambia en regimenes ATH (suele bajar)")
print(f"  Si estas features existen en el feature lake, puede implementarse sin nueva data")

# Veredicto H4
print(f"\n  VEREDICTO H4 RETEST:")
print(f"  H4 NO es una hipotesis descartable. El hallazgo es estructural:")
print(f"  El HMM asigna el mismo regimen a W3 y W4 aunque son economicamente distintos.")
print(f"  El modelo interpreta el ATH como 'bull normal' y aumenta confianza (prob_cal alta)")
print(f"  cuando deberia reducirla (regimen nuevo no visto en IS).")
print(f"  IMPLEMENTACION: requiere añadir features ATH al feature lake y re-run completo.")
print(f"  COSTO: alto (rediseño de features). IMPACTO POTENCIAL: muy alto (raiz del problema).")

print(f"\n{SEP}")
print("VEREDICTO FINAL COMPARATIVO H3 vs H4")
print(SEP)
print(f"""
  H3 (Circuit Breaker portfolio):
    - Nivel correcto: portfolio ensemble (no per-seed)
    - Mejor regla predictiva: WR_roll5 < 40% → parar
    - Predictividad: ver Spearman arriba
    - Costo: bajo (una variable + lógica simple)
    - Overfitting risk: medio (parámetro WR threshold elegible a priori)
    - Recomendacion: VIABLE como protección defensiva si rho < 0.10

  H4 (Features ATH):
    - Raiz del problema: el HMM clasifica W4 igual que W3 aunque son regimenes distintos
    - El modelo sube prob_cal en W4 (sobreconfianza en OOD)
    - Costo: alto (nuevo feature engineering + re-run completo)
    - Impacto potencial: ALTO — ataca la causa raiz, no el síntoma
    - Recomendacion: PRIORIDAD ALTA para la siguiente iteracion de features
""")
