"""deep_dive_H_to_O.py — secciones H a O del analisis profundo"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\reports\wfb')
SEP = "=" * 70

def load_all():
    dfs = []
    for f in sorted(DATA.glob('oos_trades_W*_seed*.parquet')):
        stem = f.stem; wid = stem.split('_')[2]; seed = int(stem.split('_seed')[1])
        df = pd.read_parquet(f)
        if 'timestamp' in df.columns: df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index, utc=True)
        df['_seed'] = seed; df['_window'] = wid
        dfs.append(df)
    return pd.concat(dfs).sort_index()

df = load_all()
df['ret100'] = df['return_raw'] * 100
df['bucket'] = df.index.floor('2h')
bkt = df.groupby('bucket')['_seed'].nunique()
df['consensus'] = df['bucket'].map(bkt)

# ─── H: Bootstrap CI ─────────────────────────────────────────────────────────
print(SEP)
print("H. BOOTSTRAP CONFIDENCE INTERVALS (3000 muestras)")
print(SEP)
rng = np.random.default_rng(42)
n_boot = 3000

for label, mask in [("GLOBAL", slice(None)), ("W3", df['_window']=='W3'), ("W4", df['_window']=='W4')]:
    if label == "GLOBAL":
        rets = df['ret100'].values / 100
    else:
        rets = df.loc[mask, 'ret100'].values / 100
    if len(rets) < 5:
        continue
    shr, wrs, tots = [], [], []
    for _ in range(n_boot):
        s = rng.choice(rets, size=len(rets), replace=True)
        sd = s.std()
        sr = (s.mean() / (sd + 1e-10)) * np.sqrt(len(s)) if sd > 1e-10 else 0.0
        shr.append(sr); wrs.append((s > 0).mean()); tots.append(s.sum() * 100)
    p_pos = (np.array(tots) > 0).mean() * 100
    print(f"  {label:6}: Sharpe={np.mean(shr):.4f} CI95=[{np.percentile(shr,2.5):.4f},{np.percentile(shr,97.5):.4f}]"
          f"  WR={np.mean(wrs)*100:.2f}%  P(ret>0)={p_pos:.1f}%")
    if p_pos < 50:
        print(f"         -> ALERTA: menos del 50% de bootstraps son positivos!")

# ─── I: Cross-seed correlation ────────────────────────────────────────────────
print()
print(SEP)
print("I. CROSS-SEED CORRELATION")
print(SEP)
pivot = df.groupby(['bucket', '_seed'])['return_raw'].mean().unstack('_seed')
corr_mat = pivot.corr()
upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
vals = upper.stack()
mc = vals.mean(); maxc = vals.max(); minc = vals.min()
eff_n = 1 / (1 + (len(corr_mat) - 1) * mc) if mc > -1 / (len(corr_mat) - 1) else float(len(corr_mat))
print(f"  Corr media={mc:.4f} | max={maxc:.4f} | min={minc:.4f}")
print(f"  N_efectivo Kish={eff_n:.1f} de {len(corr_mat)} seeds")
div_label = "BUENA diversificacion" if eff_n > len(corr_mat) * 0.4 else "Seeds muy correladas — el ensemble sobreestima la diversificacion"
print(f"  -> {div_label}")
print(f"  Top 5 pares mas correlados:")
for (s1, s2), c in vals.sort_values(ascending=False).head(5).items():
    print(f"    seed {int(s1)} x seed {int(s2)}: rho={c:.4f}")

# ─── J: Filter fallback ──────────────────────────────────────────────────────
print()
print(SEP)
print("J. FILTER FALLBACK Y SIGNAL THRESHOLD")
print(SEP)
for lvl, cnt in df['filter_fallback_level'].value_counts().sort_index().items():
    dfl = df[df['filter_fallback_level'] == lvl]
    print(f"  Level {lvl}: {cnt:>4} trades WR={dfl['is_win'].mean()*100:.1f}% RetMed={dfl['ret100'].mean():.4f}%")
lowered = df['threshold_was_lowered']
print(f"  Threshold lowered: {lowered.sum()} trades ({lowered.mean()*100:.1f}%)")
if lowered.sum() > 0:
    print(f"    Normal:  WR={df[~lowered]['is_win'].mean()*100:.1f}% RetMed={df[~lowered]['ret100'].mean():.4f}%")
    print(f"    Lowered: WR={df[lowered]['is_win'].mean()*100:.1f}% RetMed={df[lowered]['ret100'].mean():.4f}%")
rho_t, p_t = stats.spearmanr(df['signal_threshold'], df['ret100'])
print(f"  Spearman(signal_threshold, ret): rho={rho_t:.4f}, p={p_t:.4f} -> {'PREDICTIVO' if p_t<0.10 else 'no predictivo'}")

# ─── K: Alpha triggers ───────────────────────────────────────────────────────
print()
print(SEP)
print("K. ALPHA TRIGGER EFECTIVIDAD")
print(SEP)
for trig in ['alpha_golden_score', 'alpha_genetic_score', 'alpha_dtw_signal']:
    has = df['alpha_trigger'].str.contains(trig, na=False)
    if has.sum() < 3:
        continue
    dh = df[has]; dn = df[~has]
    print(f"  {trig}:")
    print(f"    CON: n={len(dh):>4} WR={dh['is_win'].mean()*100:.1f}% RetMed={dh['ret100'].mean():.4f}%")
    print(f"    SIN: n={len(dn):>4} WR={dn['is_win'].mean()*100:.1f}% RetMed={dn['ret100'].mean():.4f}%")

# ─── L: Seed stability W3->W4 ────────────────────────────────────────────────
print()
print(SEP)
print("L. DEGRADACION POR SEED W3->W4")
print(SEP)
print(f"  {'Seed':>8} {'W3_WR':>7} {'W4_WR':>7} {'DeltaWR':>8} {'W3_ret%':>9} {'W4_ret%':>9}")
print("  " + "-" * 58)
rows = []
for seed in sorted(df['_seed'].unique()):
    d3 = df[(df['_seed'] == seed) & (df['_window'] == 'W3')]
    d4 = df[(df['_seed'] == seed) & (df['_window'] == 'W4')]
    wr3 = d3['is_win'].mean()*100 if len(d3) > 0 else float('nan')
    wr4 = d4['is_win'].mean()*100 if len(d4) > 0 else float('nan')
    r3  = d3['ret100'].sum() if len(d3) > 0 else float('nan')
    r4  = d4['ret100'].sum() if len(d4) > 0 else float('nan')
    dwr = wr4 - wr3 if not (pd.isna(wr3) or pd.isna(wr4)) else float('nan')
    rows.append({'seed': seed, 'wr3': wr3, 'wr4': wr4, 'dwr': dwr, 'r3': r3, 'r4': r4})
    wr3s = f"{wr3:.1f}%" if not pd.isna(wr3) else " N/A "
    wr4s = f"{wr4:.1f}%" if not pd.isna(wr4) else " N/A "
    dwrs = f"{dwr:+.1f}pp" if not pd.isna(dwr) else "  N/A"
    r3s  = f"{r3:.3f}%" if not pd.isna(r3) else "  N/A  "
    r4s  = f"{r4:.3f}%" if not pd.isna(r4) else "  N/A  "
    print(f"  {int(seed):>8} {wr3s:>7} {wr4s:>7} {dwrs:>8} {r3s:>9} {r4s:>9}")

df_ss = pd.DataFrame(rows).dropna(subset=['dwr'])
if len(df_ss) > 0:
    best = df_ss.nlargest(1, 'dwr').iloc[0]
    worst = df_ss.nsmallest(1, 'dwr').iloc[0]
    print(f"\n  Mejor transfer: seed={int(best['seed'])} dWR={best['dwr']:+.1f}pp")
    print(f"  Peor  transfer: seed={int(worst['seed'])} dWR={worst['dwr']:+.1f}pp")
    print(f"  dWR media={df_ss['dwr'].mean():+.1f}pp std={df_ss['dwr'].std():.1f}pp")

# ─── M: Rachas ───────────────────────────────────────────────────────────────
print()
print(SEP)
print("M. RACHAS Y RECUPERACION TRAS PERDIDAS")
print(SEP)
max_l, max_w, rec = [], [], defaultdict(list)
for seed in df['_seed'].unique():
    for win in ['W3', 'W4']:
        d_sw = df[(df['_seed'] == seed) & (df['_window'] == win)].sort_index()
        if len(d_sw) < 3:
            continue
        seq = d_sw['is_win'].astype(int).tolist()
        rts = d_sw['ret100'].tolist()
        cl = cw = ml = mw = 0
        for i, w in enumerate(seq):
            if w == 0: cl += 1; cw = 0
            else:      cw += 1; cl = 0
            ml = max(ml, cl); mw = max(mw, cw)
            if cl >= 2 and i + 1 < len(rts):
                rec[min(cl, 5)].append(rts[i + 1])
        max_l.append(ml); max_w.append(mw)

print(f"  Max racha perdedora: media={np.mean(max_l):.1f} p90={np.percentile(max_l,90):.0f} max={max(max_l)}")
print(f"  Max racha ganadora:  media={np.mean(max_w):.1f} p90={np.percentile(max_w,90):.0f} max={max(max_w)}")
print(f"  Recuperacion tras N perdidas consecutivas:")
for nl, rs in sorted(rec.items()):
    if len(rs) < 3: continue
    wr_r = sum(r > 0 for r in rs) / len(rs) * 100
    print(f"    Tras {nl}+: n={len(rs):>3} WR_sig={wr_r:.1f}% RetMed={np.mean(rs):.4f}%")

# ─── N: Sensibilidad costos ──────────────────────────────────────────────────
print()
print(SEP)
print("N. SENSIBILIDAD A COSTOS DE TRANSACCION")
print(SEP)
print(f"  {'Cost RT':>8} {'WR%':>6} {'RetMed%':>9} {'RetW3%':>10} {'RetW4%':>10}")
for cost_pct in [0.05, 0.10, 0.15, 0.20, 0.30]:
    cost = cost_pct / 100
    adj = df['return_raw'] + 0.0015 - cost
    wr  = (adj > 0).mean() * 100
    tot = adj.mean() * 100
    w3  = adj[df['_window'] == 'W3'].sum() * 100
    w4  = adj[df['_window'] == 'W4'].sum() * 100
    flag = " <- ACTUAL" if abs(cost_pct - 0.15) < 0.01 else ""
    print(f"  {cost_pct:.2f}%:  {wr:>6.2f}% {tot:>9.4f}% {w3:>10.4f}% {w4:>10.4f}%{flag}")

# ─── O: Consensus sweep ──────────────────────────────────────────────────────
print()
print(SEP)
print("O. CONSENSUS THRESHOLD SWEEP (embargo 96H)")
print(SEP)
print(f"  {'MinSeeds':>9} {'Trades':>7} {'WR%':>7} {'RetTot%':>9} {'MaxDD%':>8}")
for ms in [2, 3, 4, 5, 6, 8, 10]:
    dfc = df[df['consensus'] >= ms].copy()
    if len(dfc) == 0: continue
    agg = {'return_pct': 'mean', 'is_win': 'max', '_window': 'first'}
    dfp = dfc.groupby('bucket').agg(agg).sort_index()
    sel, lt = [], None
    for ts in dfp.index:
        if lt is None or (ts - lt).total_seconds() / 3600 >= 96:
            sel.append(ts); lt = ts
    dfp = dfp.loc[sel]
    if len(dfp) < 1: continue
    r = dfp['return_pct']
    eq = (1 + r).cumprod()
    mdd = float((eq - eq.cummax()).min() * 100)
    flag = " <- ACTUAL" if ms == 3 else ""
    print(f"  {ms:>9} {len(dfp):>7} {dfp['is_win'].mean()*100:>7.1f}% {r.sum()*100:>9.4f}% {mdd:>8.2f}%{flag}")

print()
print(SEP)
print("FIN ANALISIS PROFUNDO H-O")
print(SEP)
