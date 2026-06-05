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

DATA  = Path(r'g:\Mi unidad\ia\luna_v2\data\predictions')
SEP   = "=" * 72
DASH  = "-" * 72

# ─── CARGA ────────────────────────────────────────────────────────────────
print("[LOAD] Cargando datos de la run de las ultimas ~21h...")
dfs = []
for f in sorted(DATA.glob('oos_trades_seed*.parquet')):
    d = pd.read_parquet(f)
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

results = {}

# ─── BASELINE ─────────────────────────────────────────────────────────────
print(SEP)
print("BASELINE — Performance por ventana")
print(SEP)
for win in ['W1','W2','W3','W4','W5','GLOBAL']:
    dw = df if win=='GLOBAL' else df[df['_window']==win]
    if len(dw)==0: continue
    wr = dw['is_win'].mean()*100
    rm = dw['ret100'].mean()
    rt = dw['ret100'].sum()
    n  = len(dw)
    print(f"  {win:6}: N={n:>4} WR={wr:.1f}% RetMed={rm:+.4f}% RetTot={rt:+.3f}%")

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
    # Simulacion de umbral con control de ventana
    print("  Simulacion de activar MetaLabeler como gate (umbral >= X):")
    print(f"  {'Umbral':>8} {'N_pass':>8} {'%pass':>6} {'WR_pass':>9} {'WR_bloq':>9} {'Delta':>7} {'Veredicto'}")
    print("  " + DASH)
    for thr in [0.58, 0.62, 0.65, 0.68, 0.70]:
        p = df[df['meta_v2_prob'] >= thr]
        b = df[df['meta_v2_prob'] < thr]
        if len(p) < 5 or len(b) < 5: continue
        wr_p = p['is_win'].mean()*100; wr_b = b['is_win'].mean()*100
        delta = wr_p - wr_b
        pct   = len(p)/len(df)*100
        v = "✅ ACTIVA" if delta > 3 else ("❌ PERJUDICA" if delta < -3 else "~neutral")
        print(f"  {thr:>8.2f} {len(p):>8} {pct:>6.0f}% {wr_p:>9.1f}% {wr_b:>9.1f}% {delta:>+7.1f}pp {v}")
    results['MetaLabeler_V2'] = d1

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

# ─── 4. OOD GUARD ─────────────────────────────────────────────────────────
print()
print(SEP)
print("COMPONENTE 4: OOD Guard (ood_kl_distance)")
print("Hipotesis: mayor KL -> mas OOD -> peor resultado")
print(SEP)
if 'ood_kl_distance' in df.columns:
    nv = df['ood_kl_distance'].notna().sum()
    print(f"  Valores: {nv} | Range [{df['ood_kl_distance'].min():.4f},{df['ood_kl_distance'].max():.4f}]")
    print(spearman_global(df['ood_kl_distance'], df['is_win'].astype(float), "Spearman global (KL vs is_win)"))
    print(spearman_per_seed('ood_kl_distance', 'is_win', "Spearman honesto"))
    # Invertido: Q1 bajo KL = mas in-distribution
    df['_ood_inv'] = -df['ood_kl_distance']
    print()
    d1 = quartile_by_window('_ood_inv', "OOD Guard [in-dist arriba, OOD abajo]")
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

# ─── RESUMEN EJECUTIVO ────────────────────────────────────────────────────
print()
print(SEP)
print("RESUMEN EJECUTIVO")
print(SEP)
print("""
  FIABILIDAD: Los Spearman 'per-seed' son estadisticamente honestos.
  El Spearman 'global' esta inflado por pseudo-replicacion (78 seeds, mismos datos).
  El analisis por ventana evita el confounding window->componente.
""")
print(f"  {'Componente':<30} {'Delta_WR_avg':>13} {'Status'}")
print("  " + DASH)

def fmt_result(name, val):
    if val is None:
        print(f"  {name:<30} {'N/A':>13}  ❓ SIN DATOS")
    elif val > 5:
        print(f"  {name:<30} {val:>+13.1f}pp  ✅ APORTA EDGE")
    elif val < -5:
        print(f"  {name:<30} {val:>+13.1f}pp  ❌ PERJUDICA")
    else:
        print(f"  {name:<30} {val:>+13.1f}pp  ⚠️  NEUTRAL/MARGINAL")

for comp, val in results.items():
    fmt_result(comp, val)
