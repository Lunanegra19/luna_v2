"""audit_dashboard_integrity.py — verifica si el dashboard tiene errores de logica o datos."""
import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from scipy import stats

DATA_PRED = Path(r'g:\Mi unidad\ia\luna_v2\data\predictions')
DATA_WFB  = Path(r'g:\Mi unidad\ia\luna_v2\data\reports\wfb')
SEP = "=" * 70

# ── 1. CARGAR DATOS IGUAL QUE EL DASHBOARD ──────────────────────────────
print(SEP)
print("AUDIT 1: INTEGRIDAD DE LA CARGA DE DATOS")
print(SEP)

pred_files = sorted(DATA_PRED.glob('oos_trades_seed*.parquet'))
wfb_files  = sorted(DATA_WFB.glob('oos_trades_W*_seed*.parquet'))
print(f"Archivos predictions: {len(pred_files)}")
print(f"Archivos WFB (run actual): {len(wfb_files)}")

dfs_pred, dfs_wfb = [], []
for f in pred_files:
    seed = int(f.stem.split('seed')[1])
    d = pd.read_parquet(f)
    if 'timestamp' in d.columns: d = d.set_index('timestamp')
    d.index = pd.to_datetime(d.index, utc=True)
    d['_seed'] = seed; d['_window'] = 'PREV'
    dfs_pred.append(d)

for f in wfb_files:
    stem = f.stem
    wid  = stem.split('_')[2]
    seed = int(stem.split('_seed')[1])
    d = pd.read_parquet(f)
    if 'timestamp' in d.columns: d = d.set_index('timestamp')
    d.index = pd.to_datetime(d.index, utc=True)
    d['_seed'] = seed; d['_window'] = wid
    dfs_wfb.append(d)

df_pred = pd.concat(dfs_pred) if dfs_pred else pd.DataFrame()
df_wfb  = pd.concat(dfs_wfb)  if dfs_wfb  else pd.DataFrame()
df_all  = pd.concat([df_pred, df_wfb]).sort_index() if len(dfs_pred) or len(dfs_wfb) else pd.DataFrame()
df_all['ret100'] = df_all['return_raw'] * 100

print(f"\nPredictions: {len(df_pred)} trades | Seeds: {df_pred['_seed'].nunique() if len(df_pred) else 0}")
print(f"WFB actual:  {len(df_wfb)} trades | Seeds: {df_wfb['_seed'].nunique() if len(df_wfb) else 0}")
print(f"TOTAL combinado: {len(df_all)} trades | Seeds reportadas: {df_all['_seed'].nunique()}")

# ── 2. BUSCAR DUPLICADOS ─────────────────────────────────────────────────
print()
print(SEP)
print("AUDIT 2: DETECCION DE TRADES DUPLICADOS")
print(SEP)

# ¿Seed 42 está en ambas fuentes?
s42_pred = df_pred[df_pred['_seed'] == 42] if len(df_pred) else pd.DataFrame()
s42_wfb  = df_wfb[df_wfb['_seed'] == 42]  if len(df_wfb)  else pd.DataFrame()
print(f"Seed 42 en predictions: {len(s42_pred)} trades")
print(f"Seed 42 en WFB actual:  {len(s42_wfb)} trades")

if len(s42_pred) > 0 and len(s42_wfb) > 0:
    overlap = s42_pred.index.intersection(s42_wfb.index)
    print(f"Solapamiento de timestamps seed42: {len(overlap)}")
    if len(overlap) > 0:
        print("  ⚠️  CRÍTICO: Los mismos trades de seed42 se cuentan DOS VECES")
    else:
        print("  OK: Sin timestamps duplicados (trades distintos de runs distintas)")

# Duplicados generales en el dataset completo
dup_count = df_all.index.duplicated().sum()
print(f"\nTimestamps duplicados en dataset completo: {dup_count}")
if dup_count > 0:
    print(f"  ⚠️  {dup_count} filas con mismo timestamp — posible doble conteo")
else:
    print("  OK: Sin duplicados por timestamp")

# ── 3. VERIFICAR COLUMNA wfb_window ─────────────────────────────────────
print()
print(SEP)
print("AUDIT 3: INFORMACION DE VENTANA WFB EN PREDICTIONS")
print(SEP)

if 'wfb_window' in df_pred.columns:
    dist = df_pred['wfb_window'].value_counts()
    print("Columna wfb_window EXISTE en predictions:")
    print(dist.to_string())
    print()
    print("  PROBLEMA: El dashboard asigna _window='PREV' a TODOS estos trades.")
    print("  Perdemos la info de ventana. dw3 y dw4 solo tienen WFB-actual (seed42).")
    print("  -> Alpha Trigger y HMM analisis sobre 'PREV' mezclan W1+W2+W3+W4+W5")
else:
    print("  wfb_window NO existe en predictions — window asignado como PREV es la unica opcion")

# ── 4. CONFOUNDING: WINDOW vs COMPONENTES ───────────────────────────────
print()
print(SEP)
print("AUDIT 4: CONFOUNDING — Alpha Trigger vs Ventana WFB")
print(SEP)
print("HIPOTESIS: El efecto 'DTW hurts' puede ser un efecto de ventana,")
print("no del trigger en si. Si DTW se activa mas en W4 (malo) que W3 (bueno),")
print("la correlacion negativa es un artefacto de datos, no causalidad.")
print()

wfb_col = 'wfb_window' if 'wfb_window' in df_pred.columns else '_window'
if wfb_col in df_pred.columns and 'alpha_trigger' in df_pred.columns:
    cross = pd.crosstab(
        df_pred[wfb_col].fillna('unknown'),
        df_pred['alpha_trigger'].fillna('empty'),
        normalize='index'
    ) * 100
    print("Distribucion de alpha_trigger por ventana (% de trades):")
    print(cross.round(1).to_string())
    print()
    # WR por ventana+trigger
    print("WR por ventana × trigger (las primeras combinaciones):")
    grp = df_pred.groupby([wfb_col, 'alpha_trigger'])['is_win'].agg(['count','mean'])
    grp['mean'] = grp['mean'] * 100
    grp.columns = ['N', 'WR%']
    print(grp[grp['N'] >= 10].sort_values('WR%', ascending=False).head(15).to_string())
else:
    print(f"  Sin columna {wfb_col} o alpha_trigger en predictions")

# ── 5. INDEPENDENCIA ESTADISTICA ────────────────────────────────────────
print()
print(SEP)
print("AUDIT 5: PROBLEMA DE PSEUDO-REPLICACION (SEEDS NO SON INDEPENDIENTES)")
print(SEP)
print("Las 20+ seeds usan LOS MISMOS datos de mercado subyacentes.")
print("El Spearman calculado sobre 3787 trades de 78 seeds sobreestima")
print("la significancia estadistica (p-values demasiado optimistas).")
print()
if len(df_pred) > 0 and 'xgb_prob_cal' in df_pred.columns:
    # Comparar rho global vs rho por seed (deberia ser similar si no hay pseudo-replicacion)
    rho_global, p_global = stats.spearmanr(
        df_pred['xgb_prob_cal'].dropna(),
        df_pred.loc[df_pred['xgb_prob_cal'].notna(), 'is_win'].astype(float)
    )
    print(f"Spearman global (N=3787, pseudo-replicado): rho={rho_global:+.4f} p={p_global:.6f}")

    # Calcular rho por seed y promediar (mas honesto)
    rhos_per_seed = []
    for seed, grp in df_pred.groupby('_seed'):
        valid = grp[['xgb_prob_cal','is_win']].dropna()
        if len(valid) >= 10:
            r, _ = stats.spearmanr(valid['xgb_prob_cal'], valid['is_win'].astype(float))
            rhos_per_seed.append(r)
    if rhos_per_seed:
        rho_avg = np.mean(rhos_per_seed)
        rho_std = np.std(rhos_per_seed)
        print(f"Spearman promedio POR SEED (N={len(rhos_per_seed)} seeds): rho={rho_avg:+.4f} ± {rho_std:.4f}")
        t_stat = rho_avg / (rho_std / np.sqrt(len(rhos_per_seed)))
        p_ttest = 2 * stats.t.sf(abs(t_stat), df=len(rhos_per_seed)-1)
        print(f"  t-test entre seeds: t={t_stat:.3f} p={p_ttest:.4f}")
        verdict = "REAL" if p_ttest < 0.05 else "ARTEFACTO ESTADISTICO"
        print(f"  Veredicto: {verdict}")

# ── 6. VERIFICAR LOGICA is_win ───────────────────────────────────────────
print()
print(SEP)
print("AUDIT 6: LOGICA DE is_win vs return_raw")
print(SEP)
if 'is_win' in df_pred.columns and 'return_raw' in df_pred.columns:
    # is_win debe ser True cuando return_raw > 0
    inconsistencies = df_pred[
        ((df_pred['is_win'] == 1) & (df_pred['return_raw'] < 0)) |
        ((df_pred['is_win'] == 0) & (df_pred['return_raw'] > 0))
    ]
    pct = len(inconsistencies) / len(df_pred) * 100
    print(f"Trades donde is_win != signo(return_raw): {len(inconsistencies)} ({pct:.1f}%)")
    if len(inconsistencies) > 0:
        print(f"  Ejemplos:")
        print(inconsistencies[['is_win','return_raw','return_pct']].head(5).to_string())
        if pct > 5:
            print("  ⚠️  CRITICO: is_win no refleja el resultado real del trade")
        else:
            print("  Probable definicion alternativa de is_win (Kelly-adjusted o net-of-costs)")
    else:
        print("  OK: is_win perfectamente alineado con signo(return_raw)")

# ── 7. RESUMEN FINAL ─────────────────────────────────────────────────────
print()
print(SEP)
print("RESUMEN DE FIABILIDAD DEL DASHBOARD")
print(SEP)
