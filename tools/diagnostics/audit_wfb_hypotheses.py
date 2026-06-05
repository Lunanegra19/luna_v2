"""
audit_wfb_hypotheses.py
=======================
Auditoria institucional de las 4 hipotesis de mejora identificadas en la run WFB activa.
Cada hipotesis se CONFIRMA o DESCARTA con tests estadisticos sobre los datos reales.

H1: 1_BULL_TREND_WEAK (42% trades, WR=48.5%) — gate de regimen mejoraria performance?
H2: 3_BEAR_CRASH (36 trades, WR=38.9%) — el fix P1-BEAR-CRASH-01 deberia haber zeroeado estos?
H3: meta_v2_prob barely discrimina (delta=0.007) — tiene valor real o es ruido?
H4: Consenso multi-seed — entry_time disponible para calcular overlap?
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import json

WFB_DIR = Path("data/reports/wfb")
SEP = "=" * 72
SEP2 = "-" * 72

# ─── CARGA GLOBAL ────────────────────────────────────────────────────────────
print(SEP)
print("  AUDITORIA INSTITUCIONAL WFB — Confirmacion/Descarte de Hipotesis")
print(SEP)

all_dfs = []
meta = []
for f in sorted(WFB_DIR.glob("oos_trades_W*_seed*.parquet")):
    parts = f.stem.split("_")
    window = parts[2]
    seed   = parts[3].replace("seed","")
    df = pd.read_parquet(f)
    df["_window"] = window
    df["_seed"]   = seed
    df["_wn"]     = int(window[1:])
    all_dfs.append(df)
    meta.append({"seed": seed, "window": window, "n": len(df), "file": f.name})

combined = pd.concat(all_dfs, ignore_index=True)
N_TOTAL  = len(combined)
WR_BASELINE = combined["is_win"].mean()

print(f"\nDataset: {N_TOTAL} trades | {combined['_seed'].nunique()} seeds | WR_baseline={WR_BASELINE:.4f}")
print(f"Columnas: {list(combined.columns)}\n")

# ─── H1: BULL_TREND_WEAK GATE ────────────────────────────────────────────────
print(SEP)
print("H1: Gate 1_BULL_TREND_WEAK — mejora WR si filtramos este regimen?")
print(SEP)

hmm_col = "hmm_regime"
btw_mask = combined[hmm_col].astype(str).str.contains("BULL_TREND_WEAK", na=False)
btw      = combined[btw_mask]
non_btw  = combined[~btw_mask]

n_btw      = len(btw)
wr_btw     = btw["is_win"].mean()
n_non_btw  = len(non_btw)
wr_non_btw = non_btw["is_win"].mean()

print(f"\n  1_BULL_TREND_WEAK: N={n_btw} ({n_btw/N_TOTAL*100:.1f}%) | WR={wr_btw:.4f}")
print(f"  Resto regimenes:   N={n_non_btw} ({n_non_btw/N_TOTAL*100:.1f}%) | WR={wr_non_btw:.4f}")

# Test binomial: WR_BTW significativamente < 0.50?
binom_btw = stats.binomtest(int(btw["is_win"].sum()), n_btw, 0.50, alternative="less")
print(f"\n  TEST BINOMIAL (H0: WR_BTW >= 0.50, H1: WR_BTW < 0.50):")
print(f"    wins={int(btw['is_win'].sum())} / {n_btw} | p-value={binom_btw.pvalue:.4f}")
if binom_btw.pvalue < 0.10:
    print(f"    => CONFIRMADO (p<0.10): WR_BTW significativamente < 50%")
else:
    print(f"    => NO CONFIRMADO (p={binom_btw.pvalue:.4f} >= 0.10): WR_BTW no difiere significativamente de 50%")

# Test chi2: WR_BTW != WR_non_BTW?
contingency = np.array([
    [int(btw["is_win"].sum()), n_btw - int(btw["is_win"].sum())],
    [int(non_btw["is_win"].sum()), n_non_btw - int(non_btw["is_win"].sum())]
])
chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
print(f"\n  TEST CHI2 (WR_BTW vs WR_no_BTW independientes?):")
print(f"    chi2={chi2:.4f} | p={p_chi2:.4f} | dof={dof}")
if p_chi2 < 0.05:
    print(f"    => CONFIRMADO (p<0.05): diferencia estadisticamente significativa")
    print(f"    => DECISION: Filtrar 1_BULL_TREND_WEAK eleva WR de {WR_BASELINE:.1%} a {wr_non_btw:.1%} (+{(wr_non_btw-WR_BASELINE)*100:.1f}pp)")
else:
    print(f"    => NO CONFIRMADO (p={p_chi2:.4f}): diferencia puede ser ruido")

# Impacto en N: cuantos trades perderiamos?
print(f"\n  IMPACTO DEL GATE:")
print(f"    Trades eliminados: {n_btw} ({n_btw/N_TOTAL*100:.1f}% del total)")
print(f"    Trades restantes:  {n_non_btw}")
print(f"    WR resultante:     {wr_non_btw:.4f} ({(wr_non_btw-WR_BASELINE)*100:+.2f}pp vs baseline)")

# Por ventana
print(f"\n  BTW por ventana:")
for w in ["W1","W2","W3","W4","W5"]:
    sub = btw[btw["_window"] == w]
    if len(sub) > 0:
        print(f"    {w}: N={len(sub)} | WR={sub['is_win'].mean():.3f}")

# ─── H2: BEAR_CRASH GATE ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("H2: 3_BEAR_CRASH — El fix P1-BEAR-CRASH-01 deberia haber bloqueado estos trades?")
print(SEP)

crash_mask = combined[hmm_col].astype(str).str.contains("BEAR_CRASH", na=False)
crash      = combined[crash_mask]
non_crash  = combined[~crash_mask]

n_crash    = len(crash)
wr_crash   = crash["is_win"].mean() if n_crash > 0 else float("nan")

print(f"\n  3_BEAR_CRASH encontrados: N={n_crash} ({n_crash/N_TOTAL*100:.1f}%)")
if n_crash > 0:
    print(f"  WR en BEAR_CRASH: {wr_crash:.4f} ({wr_crash*100:.1f}%)")
    print(f"  AvgRet en BEAR_CRASH: {crash['return_pct'].mean()*100:.4f}%")

    # Por seed y ventana
    print(f"\n  BEAR_CRASH por seed/ventana:")
    for (s, w), grp in crash.groupby(["_seed","_window"]):
        print(f"    seed={s} | {w} | N={len(grp)} | WR={grp['is_win'].mean():.3f} | AvgRet={grp['return_pct'].mean()*100:.4f}%")

    # Test: retorno medio en BEAR_CRASH significativamente negativo?
    tstat, pval_t = stats.ttest_1samp(crash["return_pct"], 0.0)
    print(f"\n  TEST T (H0: AvgRet_CRASH = 0, H1: < 0):")
    pval_one = pval_t / 2 if tstat < 0 else 1 - pval_t / 2
    print(f"    t={tstat:.4f} | p_one_sided={pval_one:.4f}")
    if pval_one < 0.10:
        print(f"    => CONFIRMADO: retornos en BEAR_CRASH son negativos significativamente")
    else:
        print(f"    => NO CONFIRMADO: retornos en BEAR_CRASH no son negativos de forma significativa (N pequeño?)")

    # Verificar si kelly_fraction_used fue reducida en BEAR_CRASH
    if "kelly_fraction_used" in crash.columns:
        kf_crash  = crash["kelly_fraction_used"].mean()
        kf_normal = non_crash["kelly_fraction_used"].mean()
        print(f"\n  Kelly en BEAR_CRASH vs normal:")
        print(f"    BEAR_CRASH kelly_fraction_used: {kf_crash:.4f}")
        print(f"    Normal     kelly_fraction_used: {kf_normal:.4f}")
        if kf_crash < kf_normal * 0.5:
            print(f"    => FIX ACTIVO: Kelly en crash reducido a {kf_crash/kf_normal*100:.0f}% del normal")
        elif kf_crash < 0.001:
            print(f"    => FIX ACTIVO: Kelly en crash es ~0 (bloqueado correctamente)")
        else:
            print(f"    => FIX POSIBLEMENTE INACTIVO: Kelly en crash ({kf_crash:.4f}) similar al normal ({kf_normal:.4f})")

    # Verificar filter_fallback_level si existe
    if "filter_fallback_level" in crash.columns:
        print(f"\n  filter_fallback_level en BEAR_CRASH:")
        print(crash["filter_fallback_level"].value_counts().to_string())

    # Verificar threshold_was_lowered
    if "threshold_was_lowered" in crash.columns:
        pct_lowered = crash["threshold_was_lowered"].mean()
        print(f"\n  threshold_was_lowered en BEAR_CRASH: {pct_lowered*100:.1f}%")

    # VEREDICTO
    print(f"\n  VEREDICTO H2:")
    if n_crash > 0 and wr_crash < 0.45:
        print(f"    => CONFIRMADO BUG: {n_crash} trades en BEAR_CRASH con WR={wr_crash:.1%}")
        print(f"       El fix P1-BEAR-CRASH-01 NO los elimino (deberia ser WR=N/A, 0 trades)")
        print(f"       Impacto: eliminando estos trades, WR sube de {WR_BASELINE:.1%} a {non_crash['is_win'].mean():.1%}")
    elif n_crash == 0:
        print(f"    => FIX CORRECTO: 0 trades en BEAR_CRASH. Fix P1-BEAR-CRASH-01 funciona.")
    else:
        print(f"    => AMBIGUO: {n_crash} trades en crash con WR={wr_crash:.1%} (no suficientemente bajo para ser bug claro)")

# ─── H3: META_V2_PROB DISCRIMINACION ─────────────────────────────────────────
print(f"\n{SEP}")
print("H3: meta_v2_prob — discrimina realmente entre wins y losses?")
print(SEP)

if "meta_v2_prob" in combined.columns:
    wins   = combined[combined["is_win"]==1]["meta_v2_prob"].dropna()
    losses = combined[combined["is_win"]==0]["meta_v2_prob"].dropna()

    print(f"\n  meta_v2_prob estadisticas:")
    print(f"    WINS   — mean={wins.mean():.4f} | std={wins.std():.4f} | median={wins.median():.4f}")
    print(f"    LOSSES — mean={losses.mean():.4f} | std={losses.std():.4f} | median={losses.median():.4f}")
    print(f"    Delta  = {wins.mean()-losses.mean():+.4f}")

    # Mann-Whitney U (no asume normalidad)
    u_stat, p_mw = stats.mannwhitneyu(wins, losses, alternative="greater")
    auc_mw = u_stat / (len(wins) * len(losses))  # AUC aproximado
    print(f"\n  TEST MANN-WHITNEY U (H0: meta_v2_prob igual en wins/losses):")
    print(f"    U={u_stat:.0f} | p={p_mw:.4f} | AUC_approx={auc_mw:.4f}")
    if p_mw < 0.05:
        print(f"    => CONFIRMADO: meta_v2_prob discrimina significativamente (AUC={auc_mw:.4f})")
    elif p_mw < 0.10:
        print(f"    => TENDENCIA (p<0.10): discriminacion marginal (AUC={auc_mw:.4f})")
    else:
        print(f"    => NO CONFIRMADO: meta_v2_prob NO discrimina (p={p_mw:.4f}, AUC={auc_mw:.4f})")

    # Quantile analysis: top vs bottom prob
    q75 = combined["meta_v2_prob"].quantile(0.75)
    q25 = combined["meta_v2_prob"].quantile(0.25)
    high_prob = combined[combined["meta_v2_prob"] >= q75]
    low_prob  = combined[combined["meta_v2_prob"] <= q25]
    print(f"\n  Analisis por cuartil:")
    print(f"    Q75+ (prob>={q75:.4f}): N={len(high_prob)} | WR={high_prob['is_win'].mean():.4f}")
    print(f"    Q25- (prob<={q25:.4f}): N={len(low_prob)}  | WR={low_prob['is_win'].mean():.4f}")
    print(f"    Delta WR Q75 vs Q25: {(high_prob['is_win'].mean()-low_prob['is_win'].mean())*100:+.2f}pp")

    # Correlacion de Spearman (no lineal)
    sp_corr, sp_pval = stats.spearmanr(combined["meta_v2_prob"], combined["is_win"])
    print(f"\n  Correlacion Spearman (meta_v2_prob vs is_win):")
    print(f"    rho={sp_corr:.4f} | p={sp_pval:.4f}")

    # Comparar con xgb_prob_cal
    if "xgb_prob_cal" in combined.columns:
        xgb_wins   = combined[combined["is_win"]==1]["xgb_prob_cal"].dropna()
        xgb_losses = combined[combined["is_win"]==0]["xgb_prob_cal"].dropna()
        u_xgb, p_xgb = stats.mannwhitneyu(xgb_wins, xgb_losses, alternative="greater")
        auc_xgb = u_xgb / (len(xgb_wins) * len(xgb_losses))
        sp_xgb, sp_xgb_p = stats.spearmanr(combined["xgb_prob_cal"].dropna(),
                                             combined.loc[combined["xgb_prob_cal"].notna(), "is_win"])
        print(f"\n  COMPARACION XGB_PROB_CAL (referencia):")
        print(f"    AUC Mann-Whitney: {auc_xgb:.4f} (p={p_xgb:.4f})")
        print(f"    Spearman: rho={sp_xgb:.4f} (p={sp_xgb_p:.4f})")
        print(f"\n  RANKING DISCRIMINACION:")
        print(f"    xgb_prob_cal  AUC={auc_xgb:.4f}")
        print(f"    meta_v2_prob  AUC={auc_mw:.4f}")
        if auc_xgb > auc_mw:
            print(f"    => XGB base supera a meta_v2 (diff={auc_xgb-auc_mw:+.4f})")
            print(f"    => VEREDICTO H3: meta_v2_prob NO añade valor discriminativo sobre XGB base")
        else:
            print(f"    => meta_v2 supera a XGB base (diff={auc_mw-auc_xgb:+.4f})")
            print(f"    => VEREDICTO H3: meta_v2_prob SI tiene valor incremental")

    # Test de calibracion: si prob=0.6 deberia ganar ~60%?
    print(f"\n  CALIBRACION de meta_v2_prob (realidad vs promesa):")
    bins = [(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,1.0)]
    for lo, hi in bins:
        sub = combined[(combined["meta_v2_prob"]>=lo) & (combined["meta_v2_prob"]<hi)]
        if len(sub) >= 5:
            actual_wr = sub["is_win"].mean()
            mid = (lo+hi)/2
            calib_err = actual_wr - mid
            print(f"    prob [{lo:.1f},{hi:.1f}): N={len(sub):>4} | WR_real={actual_wr:.3f} | prometido~{mid:.2f} | err={calib_err:+.3f}")
else:
    print("  [meta_v2_prob no encontrada en las columnas]")

# ─── H4: TIMESTAMPS Y CONSENSO MULTI-SEED ───────────────────────────────────
print(f"\n{SEP}")
print("H4: Timestamps disponibles para consenso multi-seed?")
print(SEP)

ts_candidates = ["entry_time","timestamp","ts","exit_time","date","time","index"]
ts_found = [c for c in ts_candidates if c in combined.columns]
print(f"\n  Columnas timestamp candidatas encontradas: {ts_found}")

for col in ts_found:
    sample = combined[col].dropna().head(3)
    print(f"  {col}: {list(sample.values)}")

if ts_found:
    ts_col = ts_found[0]
    combined["_ts"] = pd.to_datetime(combined[ts_col], utc=True, errors="coerce")
    n_valid_ts = combined["_ts"].notna().sum()
    print(f"\n  Timestamps validos en '{ts_col}': {n_valid_ts}/{N_TOTAL}")

    if n_valid_ts > 0:
        combined["_ts_bucket"] = combined["_ts"].dt.floor("2h")
        consensus = combined.groupby(["_window","_ts_bucket"]).agg(
            n_seeds=("_seed","nunique"),
            wr=("is_win", lambda x: x.mean()),
            n=("is_win","count")
        ).reset_index()

        print(f"\n  Distribucion de consenso (seeds coincidentes en misma hora):")
        for k, v in consensus["n_seeds"].value_counts().sort_index(ascending=False).items():
            sub_c = consensus[consensus["n_seeds"]==k]
            wr_c  = (sub_c["wr"] * sub_c["n"]).sum() / sub_c["n"].sum()
            print(f"    {k} seeds coincidentes: {v} buckets | WR={wr_c:.3f}")

        # Consenso >= 3 vs baseline
        for min_s in [4, 3, 2]:
            sub_c = consensus[consensus["n_seeds"] >= min_s]
            if len(sub_c) > 0:
                wr_c = (sub_c["wr"] * sub_c["n"]).sum() / sub_c["n"].sum()
                n_c  = sub_c["n"].sum()
                print(f"\n  Consenso >= {min_s} seeds: {len(sub_c)} buckets | N_trades={int(n_c)} | WR={wr_c:.4f} vs baseline={WR_BASELINE:.4f} | diff={wr_c-WR_BASELINE:+.4f}")
                # Test chi2
                wins_c   = int(round(wr_c * n_c))
                losses_c = int(n_c) - wins_c
                wins_b   = int(WR_BASELINE * N_TOTAL)
                losses_b = N_TOTAL - wins_b
                if wins_c > 0 and losses_c > 0:
                    ct = np.array([[wins_c, losses_c],[wins_b, losses_b]])
                    _, p_c, _, _ = stats.chi2_contingency(ct)
                    print(f"    Chi2 p={p_c:.4f} -> {'SIGNIFICATIVO' if p_c < 0.10 else 'NO significativo'}")

        print(f"\n  => VEREDICTO H4: Timestamps {'disponibles' if n_valid_ts > 100 else 'escasos'} — consenso {'calculable' if n_valid_ts > 100 else 'no calculable con fiabilidad'}")
else:
    print(f"\n  Timestamps NO encontrados en los trades parquet.")
    print(f"  Columnas disponibles: {list(combined.columns)}")
    print(f"  => VEREDICTO H4: Para calcular consenso se necesita añadir entry_time/timestamp a oos_trades_*.parquet")
    # Ver si index tiene timestamps
    sample_df = pd.read_parquet(list(WFB_DIR.glob("oos_trades_W2_seed42.parquet"))[0])
    print(f"\n  Inspeccion del INDEX del parquet:")
    print(f"    dtype={sample_df.index.dtype}")
    print(f"    sample={list(sample_df.index[:3])}")
    if hasattr(sample_df.index, "tz"):
        print(f"    timezone={sample_df.index.tz}")
        print(f"  => INDEX es timestamp! Usar index como referencia temporal.")

# ─── H5: ANALISIS ADICIONAL — OOD_KL_DISTANCE ────────────────────────────────
print(f"\n{SEP}")
print("H5 (BONUS): ood_kl_distance como gate de calidad — discrimina wins?")
print(SEP)

if "ood_kl_distance" in combined.columns:
    kl_wins   = combined[combined["is_win"]==1]["ood_kl_distance"].dropna()
    kl_losses = combined[combined["is_win"]==0]["ood_kl_distance"].dropna()
    u_kl, p_kl = stats.mannwhitneyu(kl_losses, kl_wins, alternative="greater")  # losses tienen mayor KL?
    auc_kl = u_kl / (len(kl_wins) * len(kl_losses))

    print(f"\n  ood_kl_distance:")
    print(f"    WINS   mean={kl_wins.mean():.4f} | std={kl_wins.std():.4f}")
    print(f"    LOSSES mean={kl_losses.mean():.4f} | std={kl_losses.std():.4f}")
    print(f"    Mann-Whitney (losses > wins): p={p_kl:.4f} | AUC={auc_kl:.4f}")

    # Analisis de cuartiles
    q_kl = combined["ood_kl_distance"].quantile(0.75)
    high_kl = combined[combined["ood_kl_distance"] >= q_kl]
    low_kl  = combined[combined["ood_kl_distance"] <  q_kl]
    print(f"\n  Trades con alta distrib shift (KL >= p75={q_kl:.4f}): N={len(high_kl)} | WR={high_kl['is_win'].mean():.4f}")
    print(f"  Trades con baja  distrib shift (KL <  p75={q_kl:.4f}): N={len(low_kl)}  | WR={low_kl['is_win'].mean():.4f}")
    if p_kl < 0.10:
        print(f"  => GATE POTENCIAL: filtrar trades con KL alto mejora WR")
    else:
        print(f"  => ood_kl_distance no es gate util (no discrimina)")

# ─── H6: TRIBE_MULT COMO GATE ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("H6 (BONUS): tribe_mult — multiplica la señal base, mejora con tribe alto?")
print(SEP)

if "tribe_mult" in combined.columns:
    tm_wins   = combined[combined["is_win"]==1]["tribe_mult"].dropna()
    tm_losses = combined[combined["is_win"]==0]["tribe_mult"].dropna()
    u_tm, p_tm = stats.mannwhitneyu(tm_wins, tm_losses, alternative="greater")
    auc_tm = u_tm / (len(tm_wins) * len(tm_losses))
    print(f"\n  tribe_mult — WINS mean={tm_wins.mean():.4f} | LOSSES mean={tm_losses.mean():.4f}")
    print(f"  Mann-Whitney: p={p_tm:.4f} | AUC={auc_tm:.4f}")

    # Cuartiles
    q_tm = combined["tribe_mult"].quantile(0.75)
    high_tm = combined[combined["tribe_mult"] >= q_tm]
    low_tm  = combined[combined["tribe_mult"] <  combined["tribe_mult"].quantile(0.25)]
    print(f"  tribe_mult >= p75 ({q_tm:.4f}): N={len(high_tm)} | WR={high_tm['is_win'].mean():.4f}")
    print(f"  tribe_mult <= p25:              N={len(low_tm)}  | WR={low_tm['is_win'].mean():.4f}")

# ─── RESUMEN EJECUTIVO ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  RESUMEN EJECUTIVO — Veredictos institucionales")
print(SEP)

print(f"""
  H1 [BULL_TREND_WEAK gate]:
     {n_btw} trades ({n_btw/N_TOTAL*100:.0f}%) en regimen debil | WR={wr_btw:.1%}
     Test chi2: p={p_chi2:.4f} | Mejora si se filtra: +{(wr_non_btw-WR_BASELINE)*100:.1f}pp
     => {'CONFIRMADO — implementar gate de regimen' if p_chi2 < 0.05 else 'NO CONFIRMADO — diferencia no significativa'}

  H2 [BEAR_CRASH fix activo?]:
     {n_crash} trades en BEAR_CRASH | WR={wr_crash:.1%}
     => {'BUG CONFIRMADO — fix no esta eliminando estos trades' if n_crash > 0 and wr_crash < 0.45 else 'FIX OK o ambiguo'}

  H3 [meta_v2_prob discrimination]:
     Ver AUC Mann-Whitney arriba vs xgb_prob_cal
     => Revisar output de H3 para veredicto

  H4 [Consenso timestamps]:
     Timestamps {'disponibles via index' if ts_found else 'NO disponibles'} en parquets de trades

""")
