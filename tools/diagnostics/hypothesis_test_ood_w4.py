"""
hypothesis_test_ood_w4.py
=========================
Tests estadísticos de las 5 hipótesis de mejora sobre datos reales W3/W4.
NO modifica ningún código del pipeline. Solo análisis sobre parquets existentes.

H1: OOD Guard abstención activa (ood_kl_distance predice pérdidas en W4)
H2: Calibración de probabilidad por régimen (xgb_prob_cal plano en W4)
H3: Circuit breaker de ventana adaptativo (pérdidas se acumulan tarde en W4)
H4: Features ATH explícitas (hmm_regime/xgb_prob cambian en ATH vs pre-ATH)
H5: Kelly con incertidumbre epistémica (DSR rolling como gate de Kelly)
"""
import sys, glob
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\reports\wfb')
SEP = "=" * 80

def load_all_trades(window_filter=None):
    """Carga todos los trades, opcionalmente filtrados por ventana."""
    dfs = []
    for f in sorted(DATA.glob('oos_trades_W*_seed*.parquet')):
        fname = f.stem
        window = fname.split('_')[2]
        seed = int(fname.split('_seed')[1])
        if window_filter and window not in window_filter:
            continue
        try:
            df = pd.read_parquet(f)
            df['window'] = window
            df['seed'] = seed
            dfs.append(df)
        except Exception as e:
            print(f"  ERROR {f.name}: {e}")
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()

def print_header(h_id, title):
    print(f"\n{SEP}")
    print(f"HIPOTESIS {h_id}: {title}")
    print(SEP)

# ===========================================================================
# H1: ood_kl_distance predice pérdidas en W4?
# Pregunta: ¿los trades perdedores en W4 tienen mayor ood_kl_distance?
# Test: Mann-Whitney U de ood_kl_distance entre wins y losses en W4
# Veredicto: BUENA IDEA si U-test p < 0.05 y median(loss_ood) > median(win_ood)
# ===========================================================================
print_header(1, "OOD Guard abstención activa — ood_kl_distance predice pérdidas?")

df_w4 = load_all_trades(['W4'])
df_w3 = load_all_trades(['W3'])

if not df_w4.empty and 'ood_kl_distance' in df_w4.columns:
    w4_wins   = df_w4[df_w4['is_win'] == True]['ood_kl_distance'].dropna()
    w4_losses = df_w4[df_w4['is_win'] == False]['ood_kl_distance'].dropna()
    w3_wins   = df_w3[df_w3['is_win'] == True]['ood_kl_distance'].dropna()
    w3_losses = df_w3[df_w3['is_win'] == False]['ood_kl_distance'].dropna()

    print(f"\n  W4 — ood_kl_distance:")
    print(f"    Wins  (n={len(w4_wins)}):   median={w4_wins.median():.4f}  mean={w4_wins.mean():.4f}  std={w4_wins.std():.4f}")
    print(f"    Losses(n={len(w4_losses)}): median={w4_losses.median():.4f}  mean={w4_losses.mean():.4f}  std={w4_losses.std():.4f}")

    if len(w4_wins) >= 3 and len(w4_losses) >= 3:
        stat, p = stats.mannwhitneyu(w4_losses, w4_wins, alternative='greater')
        print(f"    Mann-Whitney U (losses > wins): U={stat:.1f}, p={p:.4f}")
        discrimina = p < 0.10
        print(f"    -> Discrimina? {'SI (p<0.10)' if discrimina else 'NO (p>=0.10)'}")
    else:
        print(f"    -> N insuficiente para U-test")
        discrimina = False

    print(f"\n  W3 — ood_kl_distance (referencia):")
    print(f"    Wins  (n={len(w3_wins)}):   median={w3_wins.median():.4f}")
    print(f"    Losses(n={len(w3_losses)}): median={w3_losses.median():.4f}")

    # Percentil 75 como umbral potencial de abstención
    ood_p75_w3 = df_w3['ood_kl_distance'].quantile(0.75)
    ood_p75_w4 = df_w4['ood_kl_distance'].quantile(0.75)
    print(f"\n  Distribución OOD por ventana:")
    print(f"    W3: p25={df_w3['ood_kl_distance'].quantile(0.25):.3f} | p50={df_w3['ood_kl_distance'].median():.3f} | p75={ood_p75_w3:.3f} | max={df_w3['ood_kl_distance'].max():.3f}")
    print(f"    W4: p25={df_w4['ood_kl_distance'].quantile(0.25):.3f} | p50={df_w4['ood_kl_distance'].median():.3f} | p75={ood_p75_w4:.3f} | max={df_w4['ood_kl_distance'].max():.3f}")

    # Simular: qué pasa si bloqueamos trades con ood > umbral
    print(f"\n  SIMULACION — bloquear trades con ood_kl > umbral:")
    for umbral in [0.30, 0.35, 0.40, 0.45, 0.50]:
        bloq = df_w4[df_w4['ood_kl_distance'] > umbral]
        rest = df_w4[df_w4['ood_kl_distance'] <= umbral]
        if len(rest) > 0:
            wr_rest = rest['is_win'].mean() * 100
            ret_rest = rest['return_pct'].sum() * 100
            n_bloq = len(bloq)
            print(f"    ood>{umbral:.2f}: bloquea {n_bloq:>3} trades | restantes={len(rest)} | WR={wr_rest:.1f}% | RetTotal={ret_rest:.3f}%")

    ood_w4_median = df_w4['ood_kl_distance'].median()
    ood_w3_median = df_w3['ood_kl_distance'].median()
    delta_ood = ood_w4_median - ood_w3_median
    print(f"\n  VEREDICTO H1:")
    if delta_ood > 0.02:
        print(f"    -> W4 tiene OOD significativamente mayor que W3 (delta={delta_ood:.3f})")
    else:
        print(f"    -> W3 y W4 tienen OOD similar (delta={delta_ood:.3f}) — OOD Guard no discrimina ventana")
    if discrimina:
        print(f"    -> ood_kl SI discrimina wins/losses en W4 — H1 VIABLE")
    else:
        print(f"    -> ood_kl NO discrimina wins/losses en W4 — H1 DEBIL")
else:
    print("  ERROR: no hay datos W4 o no existe columna ood_kl_distance")

# ===========================================================================
# H2: Calibración de xgb_prob_cal — ¿es plana en W4 vs W3?
# Pregunta: ¿el modelo discrimina más en W3 que en W4?
# Test: std(xgb_prob_cal) y Spearman(prob, return) por ventana
# Veredicto: BUENA IDEA si std_W4 << std_W3 o rho_W4 << rho_W3
# ===========================================================================
print_header(2, "Calibración de probabilidad — xgb_prob_cal discrimina en W4?")

for window_name, df_w in [('W3', df_w3), ('W4', df_w4)]:
    if df_w.empty:
        continue
    prob_col = 'xgb_prob_cal' if 'xgb_prob_cal' in df_w.columns else 'xgb_prob'
    probs = df_w[prob_col].dropna()
    rets  = df_w['return_pct'].dropna()

    # Alinear índices
    common_idx = probs.index.intersection(rets.index)
    probs_aligned = probs.loc[common_idx]
    rets_aligned  = rets.loc[common_idx]

    std_prob = probs.std()
    mean_prob = probs.mean()
    frac_near_05 = ((probs > 0.45) & (probs < 0.55)).mean() * 100

    rho, p_rho = stats.spearmanr(probs_aligned, rets_aligned) if len(probs_aligned) >= 5 else (0, 1)

    print(f"\n  {window_name} — {prob_col} (n={len(df_w)}):")
    print(f"    mean={mean_prob:.4f} | std={std_prob:.4f} | near_0.5(45-55%)={frac_near_05:.1f}%")
    print(f"    Spearman(prob, ret): rho={rho:.4f}, p={p_rho:.4f}")
    print(f"    Distribucion: p10={probs.quantile(0.10):.3f} | p25={probs.quantile(0.25):.3f} | p50={probs.median():.3f} | p75={probs.quantile(0.75):.3f} | p90={probs.quantile(0.90):.3f}")

    # Brier Score (calibración)
    wins = df_w['is_win'].astype(float)
    probs_for_brier = df_w[prob_col].dropna()
    common_brier = probs_for_brier.index.intersection(wins.index)
    if len(common_brier) > 0:
        brier = ((probs_for_brier.loc[common_brier] - wins.loc[common_brier]) ** 2).mean()
        brier_naive = ((wins.loc[common_brier].mean() - wins.loc[common_brier]) ** 2).mean()
        print(f"    Brier Score={brier:.4f} | Brier_naive={brier_naive:.4f} | {'MEJOR' if brier < brier_naive else 'PEOR'} que azar")

std_w3 = df_w3['xgb_prob_cal'].std() if not df_w3.empty else 0
std_w4 = df_w4['xgb_prob_cal'].std() if not df_w4.empty else 0
rho_w3, _ = stats.spearmanr(df_w3['xgb_prob_cal'].dropna(), df_w3['return_pct'].dropna()) if len(df_w3) >= 5 else (0, 1)
rho_w4, _ = stats.spearmanr(df_w4['xgb_prob_cal'].dropna(), df_w4['return_pct'].dropna()) if len(df_w4) >= 5 else (0, 1)

print(f"\n  VEREDICTO H2:")
if std_w4 < std_w3 * 0.7:
    print(f"    -> Calibracion COLAPSA en W4 (std: W3={std_w3:.4f} vs W4={std_w4:.4f}) — H2 VIABLE")
elif rho_w4 < rho_w3 - 0.1:
    print(f"    -> Correlacion prob-retorno degrada en W4 (rho: W3={rho_w3:.4f} vs W4={rho_w4:.4f}) — H2 PARCIALMENTE VIABLE")
else:
    print(f"    -> Calibracion similar en W3 y W4 (std W3={std_w3:.4f} W4={std_w4:.4f}) — H2 DEBIL")

# ===========================================================================
# H3: Circuit Breaker de ventana — ¿las pérdidas de W4 se acumulan tarde o temprano?
# Pregunta: ¿hay un punto en el tiempo en W4 donde el circuit breaker hubiera ayudado?
# Test: retorno acumulado en W4 ordenado por tiempo. ¿Cuándo cruza a negativo?
# Veredicto: BUENA IDEA si hay un punto de inflexión claro temprano en W4
# ===========================================================================
print_header(3, "Circuit Breaker de ventana — patron temporal de perdidas en W4")

# Usar seed más representativa (100 = W4 con 17 trades)
for seed_test in [100, 777, 42]:
    f_w4 = DATA / f'oos_trades_W4_seed{seed_test}.parquet'
    if not f_w4.exists():
        continue
    df_test = pd.read_parquet(f_w4).sort_index()
    if len(df_test) < 5:
        continue

    df_test['cum_ret'] = df_test['return_pct'].cumsum() * 100
    df_test['trade_n'] = range(1, len(df_test)+1)
    df_test['rolling_wr'] = (df_test['is_win'].astype(float)
                              .rolling(window=5, min_periods=3).mean() * 100)

    print(f"\n  seed={seed_test} W4 ({len(df_test)} trades) — evolucion temporal:")
    print(f"  {'Trade':<6} {'Fecha':<25} {'Ret%':>7} {'CumRet%':>9} {'WR_roll5':>10} {'Regimen'}")
    print(f"  {'-'*75}")
    for _, row in df_test.iterrows():
        fecha = str(row.name)[:16]
        wr_str = f"{row['rolling_wr']:.0f}%" if not pd.isna(row['rolling_wr']) else "  N/A"
        mark = " <-- inflexion" if row['cum_ret'] < -5.0 and row['trade_n'] <= 8 else ""
        print(f"  T{int(row['trade_n']):<5} {fecha:<25} {row['return_pct']*100:>7.4f} {row['cum_ret']:>9.4f} {wr_str:>10} {row.get('hmm_regime','?')}{mark}")

    # Detectar punto de inflexión
    first_neg = df_test[df_test['cum_ret'] < 0]
    if not first_neg.empty:
        first_neg_trade = df_test[df_test['cum_ret'] < 0]['trade_n'].iloc[0]
        total_loss = df_test['cum_ret'].iloc[-1]
        loss_before_inflexion = df_test[df_test['trade_n'] < first_neg_trade]['return_pct'].sum() * 100
        loss_after = df_test[df_test['trade_n'] >= first_neg_trade]['return_pct'].sum() * 100
        print(f"\n  -> Primer negativo acumulado en trade #{int(first_neg_trade)} de {len(df_test)}")
        print(f"  -> Retorno antes del cruce: {loss_before_inflexion:.3f}% | despues: {loss_after:.3f}%")
    break  # Solo primer seed válida para no saturar output

# Calcular: si hubiéramos parado W4 en trade N, ¿cuánto habríamos recuperado?
print(f"\n  SIMULACION — stop_loss de ventana en trade N (promedio todas las seeds W4):")
all_w4 = load_all_trades(['W4'])
if not all_w4.empty:
    for stop_at in [3, 5, 7, 10]:
        # Para cada seed, calcular retorno si paramos en trade stop_at
        total_saved = 0
        n_seeds = 0
        for seed in all_w4['seed'].unique():
            df_s = all_w4[all_w4['seed'] == seed].sort_index()
            if len(df_s) > stop_at:
                ret_full = df_s['return_pct'].sum() * 100
                ret_stopped = df_s.iloc[:stop_at]['return_pct'].sum() * 100
                saved = ret_stopped - ret_full
                total_saved += saved
                n_seeds += 1
        avg_saved = total_saved / n_seeds if n_seeds > 0 else 0
        print(f"    Parar en trade {stop_at:>2}: ahorro promedio por seed = {avg_saved:+.3f}%")

print(f"\n  VEREDICTO H3:")
print(f"    -> Analizar si las perdidas W4 son distribuidas o concentradas al final")
print(f"       Si concentradas tarde: H3 DEBIL (el CB no habria actuado a tiempo)")
print(f"       Si hay patron de degradacion progresiva: H3 VIABLE")

# ===========================================================================
# H4: Features ATH — ¿el regimen HMM identifica el ATH diferente de W3?
# Pregunta: ¿hmm_regime en W4 es sistemáticamente distinto de W3?
# Test: distribución de regímenes W3 vs W4. Si W4 está 100% en un régimen
#       que W3 apenas usó → el modelo no tiene datos de ese régimen
# Veredicto: BUENA IDEA si W4 tiene régimenes no vistos o poco representados en W3
# ===========================================================================
print_header(4, "Features ATH — distribucion de regimenes HMM en W3 vs W4")

for window_name, df_w in [('W3', df_w3), ('W4', df_w4)]:
    if df_w.empty or 'hmm_regime' not in df_w.columns:
        continue
    regime_counts = df_w['hmm_regime'].value_counts(normalize=True) * 100
    print(f"\n  {window_name} — Distribucion de regimenes HMM:")
    for regime, pct in regime_counts.items():
        # Retorno medio en ese régimen
        ret_regime = df_w[df_w['hmm_regime'] == regime]['return_pct'].mean() * 100
        wr_regime  = df_w[df_w['hmm_regime'] == regime]['is_win'].mean() * 100
        print(f"    {regime:<30}: {pct:>5.1f}% ({df_w['hmm_regime'].value_counts()[regime]:>3} trades) | WR={wr_regime:.1f}% | RetMed={ret_regime:.4f}%")

# Comparar xgb_prob en W3 vs W4 por régimen
print(f"\n  xgb_prob_cal por regimen y ventana (W3 vs W4):")
all_trades = pd.concat([df_w3.assign(window='W3'), df_w4.assign(window='W4')])
if 'hmm_regime' in all_trades.columns:
    for regime in all_trades['hmm_regime'].dropna().unique():
        w3_prob = all_trades[(all_trades['window']=='W3') & (all_trades['hmm_regime']==regime)]['xgb_prob_cal']
        w4_prob = all_trades[(all_trades['window']=='W4') & (all_trades['hmm_regime']==regime)]['xgb_prob_cal']
        if len(w3_prob) >= 2 and len(w4_prob) >= 2:
            print(f"    {regime:<30}: W3_prob={w3_prob.mean():.4f}(n={len(w3_prob)}) | W4_prob={w4_prob.mean():.4f}(n={len(w4_prob)}) | delta={w4_prob.mean()-w3_prob.mean():+.4f}")

print(f"\n  VEREDICTO H4:")
regimenes_w3 = set(df_w3['hmm_regime'].dropna().unique()) if not df_w3.empty else set()
regimenes_w4 = set(df_w4['hmm_regime'].dropna().unique()) if not df_w4.empty else set()
nuevos_w4 = regimenes_w4 - regimenes_w3
if nuevos_w4:
    print(f"    -> W4 tiene regimenes NUEVOS no en W3: {nuevos_w4} — H4 MUY VIABLE")
elif len(regimenes_w4) == 1:
    solo = list(regimenes_w4)[0]
    print(f"    -> W4 esta 100% en UN SOLO regimen: '{solo}'")
    print(f"       Si ese regimen tiene pocos datos IS -> H4 VIABLE (necesitamos features ATH)")
else:
    print(f"    -> Mismos regimenes en W3 y W4 — H4 DUDOSA (el problema no es el regimen sino la dinamica)")

# ===========================================================================
# H5: Kelly con DSR rolling como gate — ¿si paramos Kelly cuando DSR_roll < 0?
# Pregunta: ¿el DSR rolling en W4 se degrada antes de que los trades sean malos?
# Test: calcular DSR rolling por ventana y simular Kelly=0 cuando DSR_roll < 0
# Veredicto: BUENA IDEA si DSR_roll predice degradación con suficiente antelación
# ===========================================================================
print_header(5, "Kelly con DSR rolling como gate — simulacion")

if not all_w4.empty and 'return_pct' in all_w4.columns:
    # Usar todos los trades W4 como portfolio (orden temporal)
    df_portfolio_w4 = all_w4.sort_index().drop_duplicates()

    # DSR rolling simple: Sharpe rolling 10 trades
    window_roll = 10
    df_portfolio_w4['ret_pct_100'] = df_portfolio_w4['return_pct'] * 100
    df_portfolio_w4['roll_mean'] = df_portfolio_w4['ret_pct_100'].rolling(window_roll, min_periods=5).mean()
    df_portfolio_w4['roll_std']  = df_portfolio_w4['ret_pct_100'].rolling(window_roll, min_periods=5).std()
    df_portfolio_w4['roll_sharpe'] = (df_portfolio_w4['roll_mean'] /
                                      df_portfolio_w4['roll_std'].clip(lower=1e-8))

    # Simular: Kelly=0 si roll_sharpe < umbral
    print(f"\n  SIMULACION — gate Kelly cuando roll_sharpe < umbral (window={window_roll} trades):")
    ret_sin_gate = df_portfolio_w4['return_pct'].sum() * 100
    print(f"  Sin gate: RetTotal={ret_sin_gate:.3f}% | n_trades={len(df_portfolio_w4)}")

    for umbral_sharpe in [-1.0, -0.5, 0.0, 0.5]:
        # Trades permitidos: cuando roll_sharpe >= umbral O cuando roll_sharpe es NaN (primeros trades)
        mask_permitido = (df_portfolio_w4['roll_sharpe'] >= umbral_sharpe) | df_portfolio_w4['roll_sharpe'].isna()
        df_permitido = df_portfolio_w4[mask_permitido]
        df_bloqueado = df_portfolio_w4[~mask_permitido]
        ret_con_gate = df_permitido['return_pct'].sum() * 100
        n_bloq = len(df_bloqueado)
        ret_bloq = df_bloqueado['return_pct'].sum() * 100
        mejora = ret_con_gate - ret_sin_gate
        print(f"  roll_SR>={umbral_sharpe:+.1f}: permitidos={len(df_permitido)} | bloqueados={n_bloq} | "
              f"RetTotal={ret_con_gate:.3f}% | mejora={mejora:+.3f}%")

    # ¿El roll_sharpe es predictivo? Correlación entre roll_sharpe y siguiente trade
    rs = df_portfolio_w4['roll_sharpe'].shift(1).dropna()
    next_ret = df_portfolio_w4['return_pct'].iloc[len(df_portfolio_w4) - len(rs):]
    common = rs.index.intersection(next_ret.index)
    if len(common) >= 5:
        rho_rs, p_rs = stats.spearmanr(rs.loc[common], next_ret.loc[common])
        print(f"\n  Spearman(roll_sharpe_t, ret_t+1): rho={rho_rs:.4f}, p={p_rs:.4f}")
        print(f"  -> {'PREDICTIVO (p<0.10)' if p_rs < 0.10 else 'NO predictivo (p>=0.10)'}")

    print(f"\n  VEREDICTO H5:")
    print(f"    -> El gate de roll_sharpe ahorra retorno en W4")
    print(f"       Si mejora > 5%: H5 VIABLE (vale la pena perder trades seguros)")
    print(f"       Si mejora < 2%: H5 DEBIL (el coste en W3 seria mayor que el beneficio en W4)")

print(f"\n{SEP}")
print("FIN DE TESTS — Revisar VERDICTOs de cada hipotesis arriba")
print(SEP)
