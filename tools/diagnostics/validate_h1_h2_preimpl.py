"""
TEST ESTADISTICO PRE-IMPLEMENTACION
====================================
Valida matematicamente las hipotesis H1 y H2 antes de cualquier cambio de codigo.

H1: ¿ood_kl_distance discrimina wins de losses? (valida si el OOD Guard tiene señal predictiva)
H2: ¿Cual es la tasa real de bloqueo del MetaLabeler y cuántas señales bloqueadas habrian ganado?

RESULTADO: Si H1 es positivo → el OOD Guard tiene valor real, cambiar contamination tiene sentido.
           Si H2 muestra alto false-negative rate → relajar rolling_percentile recupera edge real.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

print("=" * 80)
print("  VALIDACION ESTADISTICA PRE-IMPLEMENTACION: H1 y H2")
print("  Fecha: run nocturna 10-11/06/2026")
print("=" * 80)

reports_dir = Path("data/reports/wfb")
parquets = sorted(reports_dir.glob("oos_trades_W*_seed*.parquet"))
print(f"\n[INFO] Parquets encontrados: {len(parquets)}")

all_trades = []
for p in parquets:
    try:
        df = pd.read_parquet(p)
        # Extraer ventana y semilla del nombre
        stem = p.stem  # oos_trades_W3_seed100
        parts = stem.split("_")
        df["window"] = parts[2]   # W3
        df["seed"]   = int(parts[3].replace("seed", ""))
        all_trades.append(df)
    except Exception as e:
        print(f"  [WARN] No se pudo leer {p.name}: {e}")

if not all_trades:
    print("[ERROR CRITICO] No hay datos para analizar.")
    exit(1)

df_all = pd.concat(all_trades, ignore_index=True)
df_all["is_win"] = (df_all["return_raw"] > 0).astype(int)

print(f"\n[INFO] Total trades cargados: {len(df_all)}")
print(f"[INFO] Seeds: {sorted(df_all['seed'].unique())}")
print(f"[INFO] Ventanas: {sorted(df_all['window'].unique())}")
print(f"[INFO] Columnas disponibles: {list(df_all.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  H1: OOD Guard — ¿ood_kl_distance discrimina wins de losses?")
print("  Test: Mann-Whitney U (unidireccional). Hipótesis: wins tienen KL > losses.")
print("  Razon: KL_distance = score IsolationForest. Mayor score = más 'normal'.")
print("=" * 80)

if "ood_kl_distance" not in df_all.columns or df_all["ood_kl_distance"].isna().all():
    print("\n[ADVERTENCIA] ood_kl_distance no disponible en los datos de esta run.")
    print("  Posible causa: contamination=0.005 hace que todos los scores sean muy negativos")
    print("  y el campo existe pero puede ser constante/NaN.")
    h1_result = "INDETERMINATE"
else:
    ood_notna = df_all["ood_kl_distance"].notna()
    df_ood = df_all[ood_notna]
    print(f"\n[INFO] Trades con ood_kl_distance válido: {len(df_ood)} / {len(df_all)}")
    print(f"  ood_kl_distance — mean={df_ood['ood_kl_distance'].mean():.6f}  std={df_ood['ood_kl_distance'].std():.6f}")
    print(f"  min={df_ood['ood_kl_distance'].min():.6f}  max={df_ood['ood_kl_distance'].max():.6f}")
    
    ood_wins   = df_ood[df_ood["is_win"] == 1]["ood_kl_distance"].values
    ood_losses = df_ood[df_ood["is_win"] == 0]["ood_kl_distance"].values
    
    print(f"\n  Wins  (n={len(ood_wins)})  — KL mean={ood_wins.mean():.6f}  median={np.median(ood_wins):.6f}")
    print(f"  Losses(n={len(ood_losses)}) — KL mean={ood_losses.mean():.6f}  median={np.median(ood_losses):.6f}")
    
    if len(ood_wins) > 5 and len(ood_losses) > 5:
        # Mann-Whitney U: ¿KL de wins es sistemáticamente mayor que KL de losses?
        # Test bidireccional: losses tienen KL más alto que wins?
        u_stat_fwd, p_val_fwd = stats.mannwhitneyu(ood_wins, ood_losses, alternative="greater")
        u_stat_inv, p_val_inv = stats.mannwhitneyu(ood_losses, ood_wins, alternative="greater")
        # Spearman: correlacion entre KL y retorno
        r_sp, p_sp = stats.spearmanr(df_ood["ood_kl_distance"], df_ood["return_raw"])
        
        print(f"\n  Mann-Whitney U (wins > losses en KL): U={u_stat_fwd:.1f} | p={p_val_fwd:.4f}")
        print(f"  Mann-Whitney U (losses > wins en KL): U={u_stat_inv:.1f} | p={p_val_inv:.4f}")
        print(f"  Spearman (KL_distance vs return_raw): r={r_sp:.4f} | p={p_sp:.4f}")
        p_val = p_val_inv  # Realmente testamos si losses tienen KL > wins
        
        if p_val < 0.05 and r_sp > 0:
            verdict_h1 = "CONFIRMADA ✅"
            detail_h1 = f"KL discrimina wins (p={p_val:.4f} < 0.05, Spearman r={r_sp:.4f}). El OOD tiene señal real."
        elif p_val < 0.10:
            verdict_h1 = "TENDENCIA ⚠️"
            detail_h1 = f"Tendencia débil (p={p_val:.4f}). El OOD tiene señal marginal."
        else:
            verdict_h1 = "NO CONFIRMADA ❌"
            detail_h1 = f"KL NO discrimina wins de losses (p={p_val:.4f} ≥ 0.05). El OOD no tiene señal predictiva."
        
        print(f"\n  [VEREDICTO H1] {verdict_h1}")
        print(f"  {detail_h1}")
        
        # Análisis por cuartiles de KL
        df_ood["kl_quartile"] = pd.qcut(df_ood["ood_kl_distance"], 4, labels=["Q1(bajo)", "Q2", "Q3", "Q4(alto)"], duplicates="drop")
        print(f"\n  Win Rate por cuartil de ood_kl_distance:")
        print(f"  {'Cuartil':<12} {'n':>5} {'WR':>8} {'KL_mean':>10} {'ret_mean':>10}")
        print(f"  {'-'*50}")
        kl_wr = df_ood.groupby("kl_quartile", observed=True).agg(
            n=("is_win", "count"),
            wr=("is_win", "mean"),
            kl_mean=("ood_kl_distance", "mean"),
            ret=("return_raw", "mean")
        ).reset_index()
        for _, row in kl_wr.iterrows():
            print(f"  {str(row['kl_quartile']):<12} {int(row['n']):>5} {row['wr']*100:>7.1f}% {row['kl_mean']:>10.5f} {row['ret']*100:>9.4f}%")
        
        h1_result = verdict_h1
    else:
        print("  [WARN] N insuficiente para test estadístico.")
        h1_result = "INDETERMINATE"

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  H2: MetaLabeler V2 — Tasa de falsos negativos (over-censorship)")
print("  Test: ¿Señales bloqueadas por MetaLabeler habrían tenido WR > 50%?")
print("  Proxy: trades con meta_v2_prob disponible pero por debajo del threshold efectivo.")
print("=" * 80)

if "meta_v2_prob" not in df_all.columns or df_all["meta_v2_prob"].isna().all():
    print("\n[ADVERTENCIA] meta_v2_prob no disponible en los datos.")
    h2_result = "INDETERMINATE"
else:
    meta_notna = df_all["meta_v2_prob"].notna()
    df_meta = df_all[meta_notna]
    print(f"\n[INFO] Trades con meta_v2_prob válido: {len(df_meta)} / {len(df_all)}")
    print(f"  meta_v2_prob — mean={df_meta['meta_v2_prob'].mean():.4f}  std={df_meta['meta_v2_prob'].std():.4f}")
    print(f"  min={df_meta['meta_v2_prob'].min():.4f}  max={df_meta['meta_v2_prob'].max():.4f}")
    
    # Distribución de probabilidades del MetaLabeler
    print(f"\n  Distribución de meta_v2_prob (percentiles):")
    for pct in [10, 25, 50, 60, 70, 75, 80, 85, 90, 95]:
        val = np.percentile(df_meta["meta_v2_prob"].dropna(), pct)
        print(f"    p{pct:2d} = {val:.4f}")
    
    # Simular qué habría pasado si bajamos el percentile de 0.85 a 0.60
    # Necesitamos el threshold efectivo histórico: en rolling, usamos el percentil de las probs pasadas
    # Aproximación: usar percentil global de la distribución de meta_v2_prob como proxy
    thresh_85 = np.percentile(df_meta["meta_v2_prob"].dropna(), 85)
    thresh_70 = np.percentile(df_meta["meta_v2_prob"].dropna(), 70)
    thresh_60 = np.percentile(df_meta["meta_v2_prob"].dropna(), 60)
    thresh_50 = np.percentile(df_meta["meta_v2_prob"].dropna(), 50)
    
    print(f"\n  Threshold efectivo (percentil global de la distribución observada):")
    print(f"    p85 (actual Sniper-Mode Range) = {thresh_85:.4f}")
    print(f"    p70 (actual Sniper-Mode Bull)  = {thresh_70:.4f}")
    print(f"    p60 (propuesta moderada)        = {thresh_60:.4f}")
    print(f"    p50 (propuesta conservadora)    = {thresh_50:.4f}")
    
    # Para cada threshold, calcular cuántas señales pasan y qué WR tendrían
    print(f"\n  Simulacion de impacto por threshold (PROXY - NO causal):")
    print(f"  {'Threshold':<15} {'Señales OK':>11} {'%_bloqueadas':>13} {'WR_OK':>8} {'WR_bloq':>9} {'ΔWR_recup':>10}")
    print(f"  {'-'*70}")
    
    for label, thresh in [("p85 (actual)", thresh_85), ("p70", thresh_70), ("p60", thresh_60), ("p50", thresh_50)]:
        mask_ok   = df_meta["meta_v2_prob"] >= thresh
        mask_bloq = df_meta["meta_v2_prob"] <  thresh
        n_ok   = mask_ok.sum()
        n_bloq = mask_bloq.sum()
        pct_bloq = n_bloq / len(df_meta) * 100
        wr_ok   = df_meta[mask_ok]["is_win"].mean() * 100 if n_ok > 0 else 0.0
        wr_bloq = df_meta[mask_bloq]["is_win"].mean() * 100 if n_bloq > 0 else 0.0
        delta_wr = wr_bloq - 50.0  # ¿Las bloqueadas superan el azar?
        
        print(f"  {label:<15} {n_ok:>11} {pct_bloq:>12.1f}% {wr_ok:>7.1f}% {wr_bloq:>8.1f}% {delta_wr:>+10.1f}pp")
    
    # Test: ¿Las señales bloqueadas (meta_v2_prob < p85) habrían ganado más del 50%?
    bloqueadas_wr = df_meta[df_meta["meta_v2_prob"] < thresh_85]["is_win"]
    if len(bloqueadas_wr) > 10:
        # Binomial test: ¿WR > 0.5?
        n_wins_bloq = int(bloqueadas_wr.sum())
        n_bloq_total = len(bloqueadas_wr)
        binom_p = stats.binom_test(n_wins_bloq, n_bloq_total, 0.5, alternative="greater")
        wr_bloq_real = n_wins_bloq / n_bloq_total
        
        print(f"\n  Test Binomial (señales bloqueadas por p85: ¿WR > 50%):")
        print(f"    n_bloqueadas={n_bloq_total}, wins={n_wins_bloq}, WR={wr_bloq_real*100:.1f}%")
        print(f"    p-valor Binomial = {binom_p:.4f}")
        
        if binom_p < 0.05 and wr_bloq_real > 0.50:
            verdict_h2 = "CONFIRMADA ✅"
            detail_h2 = f"Las señales bloqueadas habrían ganado a WR={wr_bloq_real*100:.1f}% > 50% (p={binom_p:.4f}). El MetaLabeler tiene over-censorship."
        elif wr_bloq_real > 0.50:
            verdict_h2 = "TENDENCIA ⚠️"
            detail_h2 = f"WR bloqueadas={wr_bloq_real*100:.1f}% > 50% pero no significativo (p={binom_p:.4f})."
        else:
            verdict_h2 = "NO CONFIRMADA ❌"
            detail_h2 = f"WR bloqueadas={wr_bloq_real*100:.1f}% ≤ 50%. El MetaLabeler FILTRA correctamente."
        
        print(f"\n  [VEREDICTO H2] {verdict_h2}")
        print(f"  {detail_h2}")
        h2_result = verdict_h2
    else:
        h2_result = "INDETERMINATE"
    
    # Spearman entre meta_v2_prob y retorno real
    r_meta, p_meta = stats.spearmanr(df_meta["meta_v2_prob"], df_meta["return_raw"])
    print(f"\n  Spearman (meta_v2_prob ↔ return_raw): r={r_meta:.4f} | p={p_meta:.4f}")
    if r_meta > 0.05 and p_meta < 0.05:
        print("  --> MetaLabeler tiene señal predictiva real (probs más altas = mejores retornos)")
    elif r_meta > 0:
        print("  --> MetaLabeler tiene señal predictiva débil/marginal")
    else:
        print("  --> MetaLabeler NO tiene señal predictiva (probs altas no predicen mejores retornos)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  RESUMEN FINAL — VEREDICTO PRE-IMPLEMENTACION")
print("=" * 80)
print(f"\n  H1 (OOD Guard contamination=0.005 → subir a 0.03): {h1_result}")
print(f"  H2 (MetaLabeler rolling_percentile=0.85 → bajar a 0.60): {h2_result}")
print(f"\n  Recomendacion:")
if "CONFIRMADA" in str(h1_result) and "CONFIRMADA" in str(h2_result):
    print("  Ambas hipotesis confirmadas. SAFE TO PROCEED con fixes en settings.yaml.")
elif "NO CONFIRMADA" in str(h1_result):
    print("  H1 no confirmada: cambiar contamination no garantiza mejora. REVISAR antes de proceder.")
elif "NO CONFIRMADA" in str(h2_result):
    print("  H2 no confirmada: el MetaLabeler filtra correctamente. NO relajar rolling_percentile.")
else:
    print("  Resultados mixtos o indeterminados. Analizar output detallado antes de decidir.")
