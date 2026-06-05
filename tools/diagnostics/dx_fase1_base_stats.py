"""
FASE 1 — Estadísticas Base (diagnostico_cuantitativo.md)
Targets: P1 (bull_gate_min_dsr) y P2 (RANGE threshold)
"""
import pathlib, pandas as pd, numpy as np, json
from scipy import stats

runs = pathlib.Path('g:/Mi unidad/ia/luna_v2/data/runs')
SEP  = '=' * 68

# ─── Cargar TODOS los oos_trades del 01/06 ───────────────────────────────
all_trades_raw = sorted(runs.rglob('*/W*/oos_trades.parquet'),
    key=lambda p: p.stat().st_mtime, reverse=True)
recent = [f for f in all_trades_raw if '20260601' in f.parts[-4]]

rows = []
for fp in recent:
    try:
        df = pd.read_parquet(fp)
        if len(df) < 1:
            continue
        df['_window'] = fp.parts[-2]
        df['_run']    = fp.parts[-4]
        df['_seed']   = fp.parts[-3]
        rows.append(df)
    except:
        pass

df_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
print(f"Total trades cargados: {len(df_all)}")
print(f"Columnas disponibles:  {list(df_all.columns[:20])}")
print()

# ─── Identificar columna de régimen ──────────────────────────────────────
hmm_col = next((c for c in ['hmm_regime_label', 'HMM_Semantic', 'hmm_regime']
                if c in df_all.columns), None)
prob_cols = [c for c in df_all.columns if 'prob' in c.lower() or 'xgb' in c.lower()]
print(f"Columna régimen: {hmm_col}")
print(f"Columnas prob:   {prob_cols}")
print()

# ─── ESTADÍSTICAS BASE por régimen ───────────────────────────────────────
print(SEP)
print('ESTADÍSTICAS BASE POR RÉGIMEN (FASE 1)')
print(SEP)

def regime_stats(sub, label):
    if len(sub) < 2:
        return
    v    = sub['return_pct'].values
    wins = v[v > 0]
    loss = v[v < 0]
    wr   = len(wins) / len(v)
    avg_w = wins.mean() * 100 if len(wins) > 0 else 0
    avg_l = loss.mean() * 100 if len(loss) > 0 else 0
    rr    = abs(avg_w / avg_l) if avg_l != 0 else 0
    ev    = v.mean() * 100
    p50   = np.percentile(v * 100, 50)
    p10   = np.percentile(v * 100, 10)
    p90   = np.percentile(v * 100, 90)
    print(f"\n  [{label}] N={len(v)}")
    print(f"    WR={wr:.1%} | avg_win={avg_w:+.4f}% | avg_loss={avg_l:+.4f}% | R:R={rr:.3f}")
    print(f"    EV/trade={ev:+.5f}% | P10={p10:+.4f}% | P50={p50:+.4f}% | P90={p90:+.4f}%")
    return {'label': label, 'n': len(v), 'wr': wr, 'avg_win': avg_w,
            'avg_loss': avg_l, 'rr': rr, 'ev': ev}

results = {}
if hmm_col:
    for regime in df_all[hmm_col].unique():
        sub = df_all[df_all[hmm_col] == regime]
        s = regime_stats(sub, str(regime)[:30])
        if s:
            results[str(regime)] = s
else:
    regime_stats(df_all, 'GLOBAL (sin HMM col)')

print()
print(SEP)
print('ESTADÍSTICAS BASE POR VENTANA')
print(SEP)
for w in ['W1', 'W2', 'W3', 'W4', 'W5']:
    sub = df_all[df_all['_window'] == w]
    if len(sub) < 2:
        continue
    v   = sub['return_pct'].values
    wr  = (v > 0).mean()
    ev  = v.mean() * 100
    std = v.std() * 100
    n_seeds = sub['_seed'].nunique()
    print(f"  {w}: N={len(v)} seeds={n_seeds} WR={wr:.1%} EV={ev:+.5f}% std={std:.4f}%")

print()
print(SEP)
print('DISTRIBUCIÓN DE PROBABILIDADES OOS (si disponible)')
print(SEP)
# Cargar oos_raw_probs para ver distribución real de prob en OOS
probs_files = sorted(runs.rglob('*/W*/oos_raw_probs.parquet'),
    key=lambda p: p.stat().st_mtime, reverse=True)
probs_recent = [f for f in probs_files if '20260601' in f.parts[-4]][:30]

all_probs = []
for fp in probs_recent:
    try:
        df = pd.read_parquet(fp)
        df['_window'] = fp.parts[-2]
        all_probs.append(df)
    except:
        pass

if all_probs:
    df_probs = pd.concat(all_probs, ignore_index=True)
    print(f"Barras OOS con prob data: {len(df_probs)} ({len(probs_recent)} archivos)")
    for col in ['prob_bull', 'prob_bear', 'prob_range']:
        if col not in df_probs.columns:
            continue
        p = df_probs[col]
        # Distribución por percentil
        print(f"\n  {col}: mean={p.mean():.4f} std={p.std():.4f}")
        print(f"    P25={p.quantile(0.25):.4f} P50={p.quantile(0.50):.4f} "
              f"P75={p.quantile(0.75):.4f} P90={p.quantile(0.90):.4f} P95={p.quantile(0.95):.4f}")
        # Fracción sobre distintos thresholds
        for thr in [0.48, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70]:
            frac = (p >= thr).mean()
            print(f"    % barras con {col} >= {thr:.2f}: {frac*100:.2f}% → "
                  f"~{frac*len(df_probs):.0f} barras potenciales")
