"""
component_value_dashboard.py
============================
Dashboard de atribucion de valor por componente del pipeline Luna V2.

FUENTE: data/predictions/oos_trades_seed*.parquet
  - Run de las ultimas ~21h con la arquitectura actual
  - 78 seeds | 3767 trades | W1-W5 con columna wfb_window correcta
  - 0 inconsistencias is_win vs return_raw (verificado)

METODOLOGIA:
  Un componente APORTA EDGE si:
    - Sus metricas de alta confianza -> WR significativamente mayor
    - Spearman(metrica, is_win) > 0 con p < 0.05 POR SEED (no global)
    - Delta WR (Q4 vs Q1) > 5pp de forma consistente entre ventanas

  Un componente NO APORTA si:
    - WR alto ≈ WR bajo (discriminacion nula)
    - El efecto desaparece al controlar por ventana (confounding)
    - Spearman promedio por seed es no significativo
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats

# [FIX-CVD-PATH-01 2026-06-11] Ruta dinamica: funciona en G: (produccion) y C: (local dev)
_THIS_DIR  = Path(__file__).resolve().parent
_ROOT_AUTO = _THIS_DIR.parent.parent  # tools/diagnostics -> tools -> root
_FALLBACK_G = Path(r'g:\Mi unidad\ia\luna_v2\data\predictions')
_DATA_AUTO  = _ROOT_AUTO / 'data' / 'predictions'
if _DATA_AUTO.exists() and len(list(_DATA_AUTO.glob('oos_trades_seed*.parquet'))) > 0:
    DATA = _DATA_AUTO
    print(f"[FIX-CVD-PATH-01] Ruta local auto-detectada: {DATA}")
elif _FALLBACK_G.exists() and len(list(_FALLBACK_G.glob('oos_trades_seed*.parquet'))) > 0:
    DATA = _FALLBACK_G
    print(f"[FIX-CVD-PATH-01] Ruta G: produccion: {DATA}")
else:
    DATA = Path('data') / 'predictions'
    print(f"[FIX-CVD-PATH-01] WARN: fallback ruta relativa CWD: {DATA.resolve()}")
SEP   = "=" * 72
DASH  = "-" * 72

# ─── CARGA ────────────────────────────────────────────────────────────────
print("[LOAD] Cargando datos de la run de las ultimas ~21h...")
dfs = []
for f in sorted(DATA.glob('oos_trades_seed*.parquet')):
    d = pd.read_parquet(f)
    if d.index.name == 'entry_time' or 'entry_time' not in d.columns:
        d = d.reset_index(names='entry_time') if d.index.name else d.reset_index()
    d['_seed'] = int(f.stem.split('seed')[1])
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True)
df['ret100'] = df['return_raw'] * 100
# Usar wfb_window como _window canonico
df['_window'] = df['wfb_window']

print(f"[LOAD] OK: {len(df)} trades | {df['_seed'].nunique()} seeds | Ventanas: {sorted(df['_window'].unique())}")
print(f"[LOAD] Inconsistencias is_win/return_raw: {((df['is_win']==1)&(df['return_raw']<0)|(df['is_win']==0)&(df['return_raw']>0)).sum()}")
print()

# ─── HELPERS ──────────────────────────────────────────────────────────────
def spearman_global(x, y, label):
    """Spearman global (ADVERTENCIA: sobreestima significancia con N-seeds)."""
    mask = x.notna() & y.notna()
    if mask.sum() < 20:
        return f"  {label}: N insuficiente ({mask.sum()})"
    rho, p = stats.spearmanr(x[mask], y[mask])
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    direction = "POSITIVO" if rho > 0.05 else "NEGATIVO" if rho < -0.05 else "NULO"
    return f"  {label}: rho={rho:+.4f} p={p:.4f} {stars} [{direction}]  (N={mask.sum()})"

def spearman_per_seed(metrica_col, target_col, label, df_in=None):
    """Spearman promedio POR SEED. Mas honesto estadisticamente."""
    d = df_in if df_in is not None else df
    rhos = []
    for seed, grp in d.groupby('_seed'):
        valid = grp[[metrica_col, target_col]].dropna()
        if len(valid) >= 8:
            r, _ = stats.spearmanr(valid[metrica_col], valid[target_col].astype(float))
            rhos.append(r)
    if not rhos:
        return f"  {label} (per-seed): sin seeds con N>=8"
    rho_avg = np.mean(rhos)
    rho_std = np.std(rhos)
    t_stat  = rho_avg / (rho_std / np.sqrt(len(rhos))) if rho_std > 0 else 0
    p_val   = 2 * stats.t.sf(abs(t_stat), df=len(rhos)-1)
    stars   = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    direction = "POSITIVO" if rho_avg > 0.05 else "NEGATIVO" if rho_avg < -0.05 else "NULO"
    return f"  {label} (per-seed, N_seeds={len(rhos)}): rho_avg={rho_avg:+.4f} ± {rho_std:.4f} {stars} [{direction}]"

def quartile_by_window(col, label):
    """Q1 vs Q4 por cada ventana — evita confounding de ventana."""
    print(f"  {label} — por ventana (control de confounding):")
    print(f"  {'Ventana':>8} {'Q1_WR%':>8} {'Q4_WR%':>8} {'Delta':>7} {'N_Q1':>6} {'N_Q4':>6} {'Veredicto'}")
    print("  " + DASH)
    deltas = []
    for win in sorted(df['_window'].unique()):
        dw = df[df['_window'] == win]
        valid = dw[col].dropna()
        if len(valid) < 30: continue
        q25 = valid.quantile(0.25); q75 = valid.quantile(0.75)
        lo = dw[dw[col] <= q25]; hi = dw[dw[col] >= q75]
        wr_lo = lo['is_win'].mean() * 100
        wr_hi = hi['is_win'].mean() * 100
        delta = wr_hi - wr_lo
        deltas.append(delta)
        v = "✅" if delta > 5 else ("❌" if delta < -5 else "~")
        print(f"  {win:>8} {wr_lo:>8.1f}% {wr_hi:>8.1f}% {delta:>+7.1f}pp {len(lo):>6} {len(hi):>6}  {v}")
    if deltas:
        avg_delta = np.mean(deltas)
        verdict = "✅ APORTA EDGE" if avg_delta > 5 else ("❌ PERJUDICA" if avg_delta < -5 else "⚠️  NEUTRAL")
        print(f"  {'MEDIA':>8} {' ':>8} {' ':>8} {avg_delta:>+7.1f}pp  → {verdict}")
    return np.mean(deltas) if deltas else None

def group_by_window(groupby_col, label):
    """WR por categoria (HMM regime, alpha trigger) controlado por ventana."""
    print(f"  {label} — WR por categoria (global y por ventana):")
    # Global
    grp = df.groupby(groupby_col).agg(
        N=('is_win','count'),
        WR=('is_win', lambda x: x.mean()*100),
        RetMed=('ret100','mean'),
        RetTot=('ret100','sum')
    ).sort_values('WR', ascending=False)
    print(f"\n  {'Categoria':<38} {'N':>5} {'WR_global%':>11} {'RetMed%':>9} {'RetTot%':>9}")
    print("  " + DASH)
    for cat, row in grp.iterrows():
        marker = " ← MEJOR" if row['WR']==grp['WR'].max() else (" ← PEOR" if row['WR']==grp['WR'].min() else "")
        print(f"  {str(cat):<38} {int(row['N']):>5} {row['WR']:>11.1f}% {row['RetMed']:>9.4f}% {row['RetTot']:>9.2f}%{marker}")
    wr_range = grp['WR'].max() - grp['WR'].min()

    # Por ventana — para ver si el efecto persiste o es un artefacto
    print(f"\n  Por ventana (check de consistencia del efecto):")
    win_data = {}
    for win in sorted(df['_window'].unique()):
        dw = df[df['_window']==win]
        gw = dw.groupby(groupby_col)['is_win'].agg(['count','mean'])
        gw['mean'] *= 100
        gw.columns = ['N','WR']
        win_data[win] = gw

    cats = grp.index.tolist()
    header = f"  {'Categoria':<30}" + "".join(f" {w:>10}" for w in sorted(df['_window'].unique()))
    print(header)
    print("  " + DASH)
    for cat in cats:
        row_str = f"  {str(cat):<30}"
        for win in sorted(df['_window'].unique()):
            if cat in win_data[win].index and win_data[win].loc[cat,'N'] >= 5:
                row_str += f" {win_data[win].loc[cat,'WR']:>9.1f}%"
            else:
                row_str += f" {'--':>10}"
        print(row_str)
    return wr_range

# [CVD-MEJORA-01 2026-06-11] Helper de metricas financieras completas
# Integra la logica de sim_skip_metalabeler.py para mostrar Sharpe/MaxDD/Calmar
# en vez de solo Delta WR (que no captura el riesgo real del escenario)
def metricas_fin(subset: pd.DataFrame, label: str = "") -> dict:
    """Calcula WR, Ret total, Sharpe, MaxDD y Calmar sobre un subconjunto de trades."""
    n = len(subset)
    if n < 5:
        return {"n": n, "wr": np.nan, "ret": np.nan, "sharpe": np.nan, "maxdd": np.nan, "calmar": np.nan}
    wr  = subset["is_win"].mean() * 100
    ret = subset["ret100"].sum()
    rets = subset["ret100"].values
    mean_r, std_r = np.mean(rets), np.std(rets)
    # Sharpe anualizado conservador (52 trades/año = ~1/semana)
    sharpe = (mean_r / std_r) * np.sqrt(52) if std_r > 1e-10 else 0.0
    # MaxDD acumulado sobre secuencia de trades
    cumret = (1 + subset["return_raw"].values).cumprod()
    running_max = np.maximum.accumulate(cumret)
    dd = (cumret - running_max) / np.maximum(running_max, 1e-10)
    maxdd = abs(dd.min()) * 100
    calmar = (ret / maxdd) if maxdd > 0.01 else 0.0
    if label:
        print(f"  [CVD-MEJORA-01] {label}: N={n} WR={wr:.1f}% Ret={ret:+.1f}pp Sharpe={sharpe:.2f} MaxDD={maxdd:.1f}%")
    return {"n": n, "wr": wr, "ret": ret, "sharpe": sharpe, "maxdd": maxdd, "calmar": calmar}


results = {}
results_fin = {}  # [CVD-MEJORA-01] metricas financieras por componente

# ─── BASELINE ─────────────────────────────────────────────────────────────
print(SEP)
print("BASELINE — Performance por ventana")
print(SEP)
# [CVD-MEJORA-01] Ahora incluye Sharpe y MaxDD por ventana ademas de WR/Ret
print(f"  {'Ventana':>6} {'N':>4}  {'WR':>7}  {'RetMed':>8}  {'RetTot':>8}  {'Sharpe':>7}  {'MaxDD':>7}")
print("  " + DASH)
for win in ['W1', 'W2', 'W3', 'W4', 'W5', 'GLOBAL']:
    dw = df if win == 'GLOBAL' else df[df['_window'] == win]
    if len(dw) == 0:
        continue
    m = metricas_fin(dw)
    rm = dw['ret100'].mean()
    print(f"  {win:>6} {m['n']:>4}  {m['wr']:>7.1f}%  {rm:>+8.4f}%  {m['ret']:>+8.1f}pp  {m['sharpe']:>7.2f}  {m['maxdd']:>6.1f}%")
print(f"[CVD-MEJORA-01] Baseline con metricas financieras completas cargado.")

# ─── 1. XGBOOST PROB_CAL ──────────────────────────────────────────────────
print()
print(SEP)
print("COMPONENTE 1: XGBoost Probabilidad Calibrada (xgb_prob_cal)")
print("Hipotesis: mayor prob_cal -> mayor confianza del modelo -> mejor trade")
print(SEP)
if 'xgb_prob_cal' in df.columns:
    print(spearman_global(df['xgb_prob_cal'], df['is_win'].astype(float), "Spearman global (inflado x seeds)"))
    print(spearman_per_seed('xgb_prob_cal', 'is_win', "Spearman honesto"))
    print()
    d1 = quartile_by_window('xgb_prob_cal', "XGBoost prob_cal")
    results['XGBoost_prob_cal'] = d1

# ─── 2. METALABELER V2 ────────────────────────────────────────────────────
print()
print(SEP)
print("COMPONENTE 2: MetaLabeler V2 (meta_v2_prob) — calculado, no usado como gate")
print("skip_metalabeler=true: se calcula pero no filtra. Rango [0.5, 0.79]")
print(SEP)
if 'meta_v2_prob' in df.columns:
    nv = df['meta_v2_prob'].notna().sum()
    print(f"  Valores disponibles: {nv}/{len(df)} | Range [{df['meta_v2_prob'].min():.4f},{df['meta_v2_prob'].max():.4f}]")
    print(spearman_global(df['meta_v2_prob'], df['is_win'].astype(float), "Spearman global"))
    print(spearman_per_seed('meta_v2_prob', 'is_win', "Spearman honesto"))
    print()
    d1 = quartile_by_window('meta_v2_prob', "MetaLabeler V2 prob")
    print()
    # [CVD-MEJORA-02 2026-06-11] Simulacion con metricas financieras completas
    # Integra sim_skip_metalabeler.py: Ret_total + Sharpe + MaxDD por umbral
    # ANTES: solo Delta WR -> no capturaba el impacto real en el capital
    _m_base = metricas_fin(df)
    print("  [CVD-MEJORA-02] Simulacion MetaLabeler como gate — metricas financieras completas:")
    print(f"  {'Umbral':>8} {'N_pass':>7} {'%pass':>6}  {'WR':>7}  {'Ret_tot':>8}  {'Sharpe':>7}  {'MaxDD':>7}  {'Veredicto'}")
    print("  " + DASH)
    print(f"  {'SKIP':>8} {_m_base['n']:>7} {'100%':>6}  {_m_base['wr']:>7.1f}%  {_m_base['ret']:>+8.1f}pp  {_m_base['sharpe']:>7.2f}  {_m_base['maxdd']:>6.1f}%  <- BASELINE ACTUAL")
    for thr in [0.58, 0.62, 0.65, 0.68, 0.70, 0.72]:
        p = df[df['meta_v2_prob'] >= thr]
        if len(p) < 5: continue
        mp = metricas_fin(p)
        pct = len(p) / len(df) * 100
        delta_ret = mp['ret'] - _m_base['ret']
        v = "MEJORA" if delta_ret > 50 else ("PERJUDICA" if delta_ret < -50 else "~neutral")
        print(f"  {thr:>8.2f} {mp['n']:>7} {pct:>5.0f}%  {mp['wr']:>7.1f}%  {mp['ret']:>+8.1f}pp  {mp['sharpe']:>7.2f}  {mp['maxdd']:>6.1f}%  {v} (dRet={delta_ret:+.0f}pp)")
    print()
    # [CVD-MEJORA-02] Impacto por ventana con gate
    _GATE_THR = 0.65
    print(f"  [CVD-MEJORA-02] Impacto por ventana con gate >= {_GATE_THR} vs baseline:")
    print(f"  {'Vent':>5}  {'N_base':>6}  {'WR_base':>8}  {'N_gate':>6}  {'WR_gate':>8}  {'DeltaWR':>8}  {'DeltaRet':>9}  {'Veredicto'}")
    print("  " + DASH)
    for win in sorted(df['_window'].unique()):
        dw = df[df['_window'] == win]
        dw_g = dw[dw['meta_v2_prob'] >= _GATE_THR]
        if len(dw) < 5: continue
        mw = metricas_fin(dw)
        mg = metricas_fin(dw_g) if len(dw_g) >= 5 else {"n": 0, "wr": 0, "ret": 0}
        dwr = mg['wr'] - mw['wr']
        drt = mg['ret'] - mw['ret']
        vv = "MEJORA" if drt > 20 else ("PERJUDICA" if drt < -20 else "~neutro")
        print(f"  {win:>5}  {mw['n']:>6}  {mw['wr']:>8.1f}%  {mg['n']:>6}  {mg['wr']:>8.1f}%  {dwr:>+8.1f}pp  {drt:>+9.1f}pp  {vv}")
    print(f"[CVD-MEJORA-02] MetaLabeler sim integrada (fuente: sim_skip_metalabeler.py).")
    results['MetaLabeler_V2'] = d1
    results_fin['MetaLabeler_V2'] = _m_base

# ─── 3. HMM REGIME ────────────────────────────────────────────────────────
print()
print(SEP)
print("COMPONENTE 3: HMM Regime Filter — discrimina entre regimenes de mercado")
print(SEP)
if 'hmm_regime' in df.columns:
    wr_range = group_by_window('hmm_regime', "HMM Regime")
    verdict = "✅ APORTA EDGE" if wr_range > 10 else ("⚠️  MARGINAL" if wr_range > 5 else "❌ NO DISCRIMINA")
    print(f"\n  Rango WR global entre regimenes: {wr_range:.1f}pp -> {verdict}")
    results['HMM_Regime'] = wr_range

# ─── 4. OOD GUARD ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("COMPONENTE 4: OOD Guard (ood_kl_distance)")
print("[FIX-CVD-OOD-01 2026-06-11] Hipotesis diseño: KL alto=normal -> gana mas (Delta Q4-Q1 deberia ser negativo)")
print("  Si Delta < -5pp: hipotesis CORRECTA | Si Delta > +5pp: HIPOTESIS INVERTIDA (covariate shift)")
print(SEP)
if 'ood_kl_distance' in df.columns:
    nv = df['ood_kl_distance'].notna().sum()
    print(f"  Valores: {nv} | Range [{df['ood_kl_distance'].min():.4f},{df['ood_kl_distance'].max():.4f}]")
    print(f"  Media: {df['ood_kl_distance'].mean():.4f} | Std: {df['ood_kl_distance'].std():.4f}")
    _n_kl_neg = (df['ood_kl_distance'] < 0).sum()
    print(f"  Trades con KL<0 (Kelly penalty activo): {_n_kl_neg} ({_n_kl_neg/max(1,nv)*100:.1f}%) - penalizacion casi inactiva si <2%")
    print(spearman_global(df['ood_kl_distance'], df['is_win'].astype(float), "Spearman global (KL vs is_win)"))
    print(spearman_per_seed('ood_kl_distance', 'is_win', "Spearman honesto"))
    print()
    # [FIX-CVD-OOD-01 2026-06-11] Usar ood_kl_distance DIRECTAMENTE sin inversion artificial.
    # Bug anterior: df['_ood_inv'] = -df['ood_kl_distance'] causaba que Delta positivo
    # (anomalos ganan mas) se interpretara como 'OOD Guard aporta edge', ocultando
    # el covariate shift temporal del IsolationForest (entrenado 2022-2024, OOS 2025-2026).
    # Q4 = KL alto = barras 'normales' segun IsolationForest
    # Q1 = KL bajo = barras 'anomalas' segun IsolationForest
    # Delta = WR(Q4_normal) - WR(Q1_anomalo) -> negativo si hipotesis correcta
    print("[FIX-CVD-OOD-01] Analizando ood_kl_distance DIRECTO (Q4=KL_alto=normal, Q1=KL_bajo=anomalo):")
    d1 = quartile_by_window('ood_kl_distance', "OOD Guard [Q4=normal/KLalto, Q1=anomalo/KLbajo]")
    if d1 is not None:
        if d1 < -5:
            print(f"  [FIX-CVD-OOD-01] DIAGNOSTICO OOD: Delta={d1:+.1f}pp -> HIPOTESIS CORRECTA (normales ganan mas).")
        elif d1 > 5:
            print(f"  [FIX-CVD-OOD-01] DIAGNOSTICO OOD: Delta={d1:+.1f}pp -> HIPOTESIS INVERTIDA.")
            print(f"  Las barras ANOMALAS (KL bajo) ganan MAS que las normales (KL alto).")
            print(f"  Causa probable: covariate shift IsolationForest 2022-2024 -> OOS 2025-2026.")
            print(f"  El IsolationForest llama 'anomalas' a las mejores oportunidades del periodo OOS.")
        else:
            print(f"  [FIX-CVD-OOD-01] DIAGNOSTICO OOD: Delta={d1:+.1f}pp -> SIN DISCRIMINACION CLARA.")
    results['OOD_Guard'] = d1

# ─── 5. ALPHA TRIGGER ─────────────────────────────────────────────────────
print()
print(SEP)
print("COMPONENTE 5: Alpha Trigger — tipo de señal que activa el trade")
print("ADVERTENCIA: Sin control de ventana el analisis esta confounded")
print(SEP)
if 'alpha_trigger' in df.columns:
    wr_range = group_by_window('alpha_trigger', "Alpha Trigger")
    verdict = "✅ DIFERENCIA REAL" if wr_range > 15 else ("⚠️  POSIBLE CONFOUNDING" if wr_range > 5 else "❌ NO DISCRIMINA")
    print(f"\n  Rango WR global entre triggers: {wr_range:.1f}pp -> {verdict}")
    results['Alpha_Trigger'] = wr_range

# ─── 6. SIGNAL THRESHOLD ──────────────────────────────────────────────────
print()
print(SEP)
print("COMPONENTE 6: Signal Threshold")
print(SEP)
if 'signal_threshold' in df.columns:
    print(spearman_global(df['signal_threshold'], df['is_win'].astype(float), "Spearman global"))
    print(spearman_per_seed('signal_threshold', 'is_win', "Spearman honesto"))
    print()
    d1 = quartile_by_window('signal_threshold', "Signal Threshold")
    results['Signal_Threshold'] = d1

# ─── 7. LGBM ──────────────────────────────────────────────────────────────
print()
print(SEP)
print("COMPONENTE 7: LightGBM (lgbm_prob) — modelos entrenados pero prob=NaN en trades")
print(SEP)
nv = df['lgbm_prob'].notna().sum() if 'lgbm_prob' in df.columns else 0
print(f"  Valores lgbm_prob disponibles: {nv}/{len(df)}")
if nv > 20:
    print(spearman_per_seed('lgbm_prob', 'is_win', "Spearman honesto LGBM"))
    quartile_by_window('lgbm_prob', "LGBM prob")
else:
    print("  ❌ LGBM no registra probabilidades en los trades OOS.")
    print("  Los .model existen en data/models/prod/ pero formato incompatible para re-score.")
    print("  Para evaluar: activar use_lgbm_ensemble=true en settings y re-run.")
    results['LGBM'] = None

# ─── 8. KELLY FRACTION ────────────────────────────────────────────────────
print()
print(SEP)
print("COMPONENTE 8: Kelly Fraction (kelly_fraction_used)")
print(SEP)
if 'kelly_fraction_used' in df.columns:
    print(spearman_global(df['kelly_fraction_used'], df['is_win'].astype(float), "Spearman global"))
    print(spearman_per_seed('kelly_fraction_used', 'is_win', "Spearman honesto"))
    print()
    quartile_by_window('kelly_fraction_used', "Kelly Fraction")

# ─── 9. COMBINACION XGBOOST + OOD_KL (NUEVO) ──────────────────────────────
# [CVD-MEJORA-03 2026-06-11] Combinacion optima hallada en investigacion H3
# Fuente: audit_ood_guard_pkl.py + sim_skip_metalabeler.py
# El KL score (ood_kl_distance) es un predictor INVERSO: KL bajo -> mejor trade
# Combinar XGBoost alto + KL bajo produce la mejor relacion Sharpe/MaxDD
print()
print(SEP)
print("COMPONENTE 9: Combinacion XGBoost + OOD_KL Score [CVD-MEJORA-03]")
print("Hipotesis H3: KL bajo (anomalo para IF) + XGBoost alto = mejor combinacion")
print("Fuente: investigacion hipotesis OOD 2026-06-11 sobre 7.718 trades reales")
print(SEP)
if 'xgb_prob_cal' in df.columns and 'ood_kl_distance' in df.columns:
    _xq50 = df['xgb_prob_cal'].quantile(0.50)
    _xq75 = df['xgb_prob_cal'].quantile(0.75)
    _xq85 = df['xgb_prob_cal'].quantile(0.85)
    _kq25 = df['ood_kl_distance'].quantile(0.25)
    _kq50 = df['ood_kl_distance'].quantile(0.50)
    _kq75 = df['ood_kl_distance'].quantile(0.75)
    print(f"  XGBoost cuantiles: Q50={_xq50:.4f} Q75={_xq75:.4f} Q85={_xq85:.4f}")
    print(f"  KL cuantiles     : Q25={_kq25:.4f} Q50={_kq50:.4f} Q75={_kq75:.4f}")
    print(f"  KL<0 (anomalias genuinas): {(df['ood_kl_distance']<0).sum()} trades ({(df['ood_kl_distance']<0).mean()*100:.1f}%)")
    print()
    _scenarios = [
        ("BASELINE (todos)",              df),
        (f"Solo XGBoost>=Q50 ({_xq50:.3f})",    df[df['xgb_prob_cal'] >= _xq50]),
        (f"Solo XGBoost>=Q75 ({_xq75:.3f})",    df[df['xgb_prob_cal'] >= _xq75]),
        (f"Solo XGBoost>=Q85 ({_xq85:.3f})",    df[df['xgb_prob_cal'] >= _xq85]),
        (f"Solo KL<=Q25 (anomalos)",             df[df['ood_kl_distance'] <= _kq25]),
        (f"Solo KL<=Q50 (mitad anomala)",        df[df['ood_kl_distance'] <= _kq50]),
        (f"XGBoost>=Q75 + KL<=Q50",              df[(df['xgb_prob_cal'] >= _xq75) & (df['ood_kl_distance'] <= _kq50)]),
        (f"XGBoost>=Q75 + KL<=Q25 OPTIMO",       df[(df['xgb_prob_cal'] >= _xq75) & (df['ood_kl_distance'] <= _kq25)]),
        (f"XGBoost>=Q50 + KL<=Q25 SNIPER",       df[(df['xgb_prob_cal'] >= _xq50) & (df['ood_kl_distance'] <= _kq25)]),
    ]
    print(f"  [CVD-MEJORA-03] Escenarios XGBoost+KL — metricas financieras completas:")
    print(f"  {'Escenario':<38} {'N':>5}  {'WR':>7}  {'Ret_tot':>8}  {'Sharpe':>7}  {'MaxDD':>7}  {'Calmar':>7}")
    print("  " + DASH)
    _best_sharpe = -999
    _best_label  = ""
    for label, sub in _scenarios:
        m = metricas_fin(sub)
        if m['n'] < 10:
            print(f"  {label:<38} {'N<10':>5}")
            continue
        flag = ""
        if m['sharpe'] > _best_sharpe and label != "BASELINE (todos)":
            _best_sharpe = m['sharpe']
            _best_label  = label
            flag = " <- MEJOR SHARPE"
        print(f"  {label:<38} {m['n']:>5}  {m['wr']:>7.1f}%  {m['ret']:>+8.1f}pp  {m['sharpe']:>7.2f}  {m['maxdd']:>6.1f}%  {m['calmar']:>7.2f}{flag}")
    print()
    print(f"  [CVD-MEJORA-03] Mejor combinacion hallada: '{_best_label}' (Sharpe={_best_sharpe:.2f})")
    print(f"  ADVERTENCIA: resultado retroactivo sobre datos observados.")
    print(f"  Necesita validacion causal (PurgedKFold OOS) antes de implementar en produccion.")
    print(f"[CVD-MEJORA-03] Combinacion XGBoost+KL integrada. Fuente: audit_ood_guard_pkl.py.")
else:
    print("  xgb_prob_cal u ood_kl_distance no disponibles en parquets.")

# ─── 10. HOLDING TIME Y SALIDAS (NUEVO) ───────────────────────────────────
print()
print(SEP)
print("COMPONENTE 10: Holding Time y Fricción de Salida (TBM)")
print("Hipótesis: Analizar si el 'Time-in-Market' o el Time Barrier erosionan el Alpha")
print(SEP)
if 'entry_time' in df.columns and 'exit_time' in df.columns:
    # Convertir a datetime si no lo están
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    df['exit_time'] = pd.to_datetime(df['exit_time'], utc=True)
    df['holding_time_hours'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600.0

    print(spearman_global(df['holding_time_hours'], df['is_win'].astype(float), "Spearman global (Holding Time vs Win)"))
    print(spearman_per_seed('holding_time_hours', 'is_win', "Spearman honesto (Holding Time vs Win)"))
    print()

    # Agrupar por cuartiles de Holding Time (Duración)
    print("  Distribución por duración del trade (Cuartiles):")
    ht_q25 = df['holding_time_hours'].quantile(0.25)
    ht_q50 = df['holding_time_hours'].quantile(0.50)
    ht_q75 = df['holding_time_hours'].quantile(0.75)
    
    # Etiquetar para agrupar
    conditions = [
        (df['holding_time_hours'] <= ht_q25),
        (df['holding_time_hours'] > ht_q25) & (df['holding_time_hours'] <= ht_q50),
        (df['holding_time_hours'] > ht_q50) & (df['holding_time_hours'] <= ht_q75),
        (df['holding_time_hours'] > ht_q75)
    ]
    labels = [f"1. Rápido (<= {ht_q25:.1f}h)", f"2. Medio-Rápido ({ht_q25:.1f}h - {ht_q50:.1f}h)", 
              f"3. Medio-Lento ({ht_q50:.1f}h - {ht_q75:.1f}h)", f"4. Lento (> {ht_q75:.1f}h)"]
    df['holding_category'] = np.select(conditions, labels, default="Unknown")
    
    wr_range_ht = group_by_window('holding_category', "Holding Time")
    
    # Inferencia de Time Barrier
    # Si un trade dura > 90h, probablemente fue cortado por Time Barrier en vez de Take Profit/Stop Loss
    tb_trades = df[df['holding_time_hours'] >= 90]
    ptsl_trades = df[df['holding_time_hours'] < 90]
    
    print("\n  [TBM EXIT EDGE] Take-Profit/Stop-Loss vs Time Barrier:")
    m_ptsl = metricas_fin(ptsl_trades)
    m_tb = metricas_fin(tb_trades)
    print(f"  {'Categoría':<30} {'N':>6}  {'WR':>7}  {'RetTot':>8}  {'Sharpe':>7}  {'MaxDD':>7}")
    print("  " + DASH)
    print(f"  {'Salida Dinámica (PT/SL)':<30} {m_ptsl['n']:>6}  {m_ptsl['wr']:>7.1f}%  {m_ptsl['ret']:>+8.1f}pp  {m_ptsl['sharpe']:>7.2f}  {m_ptsl['maxdd']:>6.1f}%")
    print(f"  {'Salida Lenta (Time Barrier)':<30} {m_tb['n']:>6}  {m_tb['wr']:>7.1f}%  {m_tb['ret']:>+8.1f}pp  {m_tb['sharpe']:>7.2f}  {m_tb['maxdd']:>6.1f}%")

    # Invertimos el Delta porque Q4 = Lentos. Si Delta es negativo, los lentos pierden más -> Hipótesis correcta
    print("\n[FIX-CVD-HT-01] Analizando Holding Time DIRECTO (Q4=Lento, Q1=Rápido):")
    d_ht = quartile_by_window('holding_time_hours', "Holding Time (Q4=Lento, Q1=Rápido)")
    results['Holding_Time_Friction'] = d_ht
else:
    print("  entry_time o exit_time no disponibles en el parquet.")

# ─── 11. DIRECTIONAL BIAS ─────────────────────────────────────────────────
print()
print(SEP)
print("COMPONENTE 11: Sesgo Direccional (Directional Bias)")
print("Hipótesis: Analizar si el modelo sufre asimetría y pierde Edge operando en corto (Short)")
print(SEP)
if 'direction' in df.columns:
    wr_range_dir = group_by_window('direction', "Direction (Long vs Short)")
    
    _longs = df[df['direction'].str.lower() == 'long']
    _shorts = df[df['direction'].str.lower() == 'short']
    
    m_long = metricas_fin(_longs)
    m_short = metricas_fin(_shorts)
    
    print("\n  [DIRECTIONAL EDGE] Long vs Short:")
    print(f"  {'Categoría':<30} {'N':>6}  {'WR':>7}  {'RetTot':>8}  {'Sharpe':>7}  {'MaxDD':>7}")
    print("  " + DASH)
    print(f"  {'Long':<30} {m_long['n']:>6}  {m_long['wr']:>7.1f}%  {m_long['ret']:>+8.1f}pp  {m_long['sharpe']:>7.2f}  {m_long['maxdd']:>6.1f}%")
    print(f"  {'Short':<30} {m_short['n']:>6}  {m_short['wr']:>7.1f}%  {m_short['ret']:>+8.1f}pp  {m_short['sharpe']:>7.2f}  {m_short['maxdd']:>6.1f}%")

    if m_long['n'] > 5 and m_short['n'] > 5:
        results['Directional_Symmetry'] = abs(m_long['wr'] - m_short['wr'])
    else:
        results['Directional_Symmetry'] = None
else:
    print("  direction no disponible en parquets.")

# ─── 12. STARVATION FALLBACK ──────────────────────────────────────────────
print()
print(SEP)
print("COMPONENTE 12: Degradación por Inanición (Starvation Fallback)")
print("Hipótesis: Forzar trades bajando los umbrales de seguridad destruye el Alpha")
print(SEP)
if 'threshold_was_lowered' in df.columns:
    wr_range_lowered = group_by_window('threshold_was_lowered', "Threshold Lowered")
    
    _pure = df[df['threshold_was_lowered'] == False]
    _forced = df[df['threshold_was_lowered'] == True]
    
    m_pure = metricas_fin(_pure)
    m_forced = metricas_fin(_forced)
    
    print("\n  [STARVATION EDGE] Umbral Original vs Degradado:")
    print(f"  {'Categoría':<30} {'N':>6}  {'WR':>7}  {'RetTot':>8}  {'Sharpe':>7}  {'MaxDD':>7}")
    print("  " + DASH)
    print(f"  {'Puro (Original)':<30} {m_pure['n']:>6}  {m_pure['wr']:>7.1f}%  {m_pure['ret']:>+8.1f}pp  {m_pure['sharpe']:>7.2f}  {m_pure['maxdd']:>6.1f}%")
    print(f"  {'Forzado (Degradado)':<30} {m_forced['n']:>6}  {m_forced['wr']:>7.1f}%  {m_forced['ret']:>+8.1f}pp  {m_forced['sharpe']:>7.2f}  {m_forced['maxdd']:>6.1f}%")

    if m_pure['n'] > 5 and m_forced['n'] > 5:
        delta_lowered = m_pure['wr'] - m_forced['wr']
        results['Starvation_Degradation'] = delta_lowered
    else:
        results['Starvation_Degradation'] = None
else:
    print("  threshold_was_lowered no disponible en parquets.")


print()
print(SEP)
print("RESUMEN EJECUTIVO")
print(SEP)
print("""
  FIABILIDAD: Los Spearman 'per-seed' son estadisticamente honestos.
  El Spearman 'global' esta inflado por pseudo-replicacion (78 seeds, mismos datos).
  El analisis por ventana evita el confounding window->componente.
""")
# [CVD-MEJORA-01] Baseline financiero global para referencia rapida
_mb = metricas_fin(df)
print(f"  BASELINE GLOBAL: N={_mb['n']} WR={_mb['wr']:.1f}% Ret={_mb['ret']:+.1f}pp Sharpe={_mb['sharpe']:.2f} MaxDD={_mb['maxdd']:.1f}%")
print()
print(f"  {'Componente':<30} {'Delta_WR_avg':>13} {'Status'}")
print("  " + DASH)

# [FIX-CVD-OOD-01] OOD_Guard necesita interpretacion inversa:
# Delta positivo = anomalos ganan mas = hipotesis INVERTIDA (covariate shift)
# Delta negativo = normales ganan mas = hipotesis CORRECTA
_OOD_INVERTED_COMPONENTS = {'OOD_Guard'}  # componentes donde + = problema

def fmt_result(name, val):
    if val is None:
        print(f"  {name:<30} {'N/A':>13}  ❓ SIN DATOS")
    elif name in _OOD_INVERTED_COMPONENTS:
        # [FIX-CVD-OOD-01] Interpretacion especial para OOD Guard
        if val < -5:
            print(f"  {name:<30} {val:>+13.1f}pp  ✅ HIPOTESIS CORRECTA (normales ganan mas)")
        elif val > 5:
            print(f"  {name:<30} {val:>+13.1f}pp  ⚠️  HIPOTESIS INVERTIDA — covariate shift (anomalos ganan)")
        else:
            print(f"  {name:<30} {val:>+13.1f}pp  ~ SIN DISCRIMINACION CLARA")
    elif name == 'Directional_Symmetry':
        if val > 15:
            print(f"  {name:<30} {val:>+13.1f}pp  ❌ ASIMETRÍA CRÍTICA (Desbalance Long/Short)")
        elif val > 5:
            print(f"  {name:<30} {val:>+13.1f}pp  ⚠️  SESGO MODERADO")
        else:
            print(f"  {name:<30} {val:>+13.1f}pp  ✅ MODELO SIMÉTRICO")
    elif name == 'Starvation_Degradation':
        if val > 5:
            print(f"  {name:<30} {val:>+13.1f}pp  ❌ DESTRUYE EDGE (Trades forzados pierden)")
        elif val < -5:
            print(f"  {name:<30} {val:>+13.1f}pp  ✅ APORTA EDGE (Fallback es útil)")
        else:
            print(f"  {name:<30} {val:>+13.1f}pp  ⚠️  NEUTRAL")
    elif name == 'Holding_Time_Friction':
        if val < -5:
            print(f"  {name:<30} {val:>+13.1f}pp  ❌ PERJUDICA (Time-in-Market destruye Edge)")
        elif val > 5:
            print(f"  {name:<30} {val:>+13.1f}pp  ✅ APORTA EDGE (Beneficia mantener)")
        else:
            print(f"  {name:<30} {val:>+13.1f}pp  ⚠️  NEUTRAL")
    elif val > 5:
        print(f"  {name:<30} {val:>+13.1f}pp  ✅ APORTA EDGE")
    elif val < -5:
        print(f"  {name:<30} {val:>+13.1f}pp  ❌ PERJUDICA")
    else:
        print(f"  {name:<30} {val:>+13.1f}pp  ⚠️  NEUTRAL/MARGINAL")

for comp, val in results.items():
    fmt_result(comp, val)

print()
print("  MEJORAS INTEGRADAS EN ESTE REPORTE [CVD-MEJORA 2026-06-11]:")
print("  [CVD-MEJORA-01] metricas_fin(): Sharpe/MaxDD/Calmar disponibles en BASELINE y secciones")
print("  [CVD-MEJORA-02] MetaLabeler sim con Ret_total + Sharpe + impacto por ventana")
print("  [CVD-MEJORA-03] Componente 9: combinacion XGBoost+KL Score (Hipotesis H3)")
print("[CVD-MEJORA] Dashboard CVD-01 completado. Mejoras: MEJORA-01 MEJORA-02 MEJORA-03")
