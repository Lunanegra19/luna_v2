"""
Simulación: ¿Qué pasaría si incluimos TODAS las seeds en el ensemble vs solo las aprobadas?
"""
import pandas as pd
import numpy as np

PRED_DIR = "data/predictions"

# ─────────────────────────────────────────
# 1. Cargar datos de TODAS las seeds
# ─────────────────────────────────────────
all_seeds = {
    42:   pd.read_parquet(f"{PRED_DIR}/oos_trades_seed42.parquet"),
    1337: pd.read_parquet(f"{PRED_DIR}/oos_trades_seed1337.parquet"),
    2025: pd.read_parquet(f"{PRED_DIR}/oos_trades_seed2025.parquet"),
    100:  pd.read_parquet(f"{PRED_DIR}/oos_trades_seed100.parquet"),
    777:  pd.read_parquet(f"{PRED_DIR}/oos_trades_seed777.parquet"),
}

APPROVED = [42, 1337, 2025]
REJECTED  = [100, 777]

def build_ensemble(seeds, embargo_h=48):
    """
    Combina trades de múltiples seeds:
    - Agrupa por timestamp (consensus count)
    - Aplica embargo cronológico
    Retorna el portafolio final y sus métricas.
    """
    frames = []
    for s, df in seeds.items():
        df2 = df.copy()
        df2 = df2.reset_index()
        # normalizar columna de tiempo
        time_col = [c for c in ['entry_time', 'index', 'timestamp'] if c in df2.columns]
        if not time_col:
            continue
        df2 = df2.rename(columns={time_col[0]: 'entry_time'})
        df2['entry_time'] = pd.to_datetime(df2['entry_time'])
        df2['seed'] = s
        frames.append(df2)
    
    if not frames:
        return pd.DataFrame(), {}
    
    df_all = pd.concat(frames, ignore_index=True).sort_values('entry_time')
    
    # Columna is_win
    if 'is_win' in df_all.columns:
        df_all['is_win'] = df_all['is_win'].astype(bool)
    elif 'return_pct' in df_all.columns:
        df_all['is_win'] = df_all['return_pct'] > 0

    # Agregar por timestamp
    agg = df_all.groupby('entry_time').agg(
        consensus=('seed', 'nunique'),
        return_pct=('return_pct', 'first'),
        is_win=('is_win', 'first'),
    ).reset_index().set_index('entry_time').sort_index()
    
    # Embargo secuencial
    selected = []
    last_time = None
    for ts, row in agg.iterrows():
        if last_time is None or (ts - last_time).total_seconds() / 3600 >= embargo_h:
            selected.append(ts)
            last_time = ts
    
    df_final = agg.loc[selected]
    
    if len(df_final) == 0:
        return df_final, {}
    
    wr = df_final['is_win'].mean() * 100
    n = len(df_final)
    ret = df_final['return_pct']
    sharpe = (ret.mean() / ret.std() * np.sqrt(8760)) if ret.std() > 0 else 0
    equity = (1 + ret).cumprod()
    maxdd = ((equity - equity.expanding().max()) / equity.expanding().max()).min() * 100

    metrics = {
        'n_trades': n,
        'win_rate': wr,
        'sharpe': sharpe,
        'max_dd': maxdd,
        'unique_timestamps': len(agg),
        'consensus_2plus': (agg['consensus'] >= 2).sum(),
        'consensus_3plus': (agg['consensus'] >= 3).sum(),
    }
    return df_final, metrics

def print_m(label, m):
    ok = "✅ OK" if m['win_rate'] >= 50 and m['n_trades'] >= 30 else "❌ REJECT"
    print(f"""
[{label}] {ok}
  Trades post-embargo   : {m['n_trades']}
  Win Rate              : {m['win_rate']:.2f}%
  Sharpe Anualizado     : {m['sharpe']:.4f}
  Max Drawdown          : {m['max_dd']:.4f}%
  Timestamps únicos     : {m['unique_timestamps']}
  Con >=2 seeds (consenso): {m['consensus_2plus']}
  Con >=3 seeds (consenso): {m['consensus_3plus']}""")

# ─────────────────────────────────────────
# TEST A: Solo seeds APROBADAS (baseline)
# ─────────────────────────────────────────
print("=" * 60)
print("COMPARATIVA: APROBADAS vs TODAS LAS SEEDS")
print("=" * 60)

for emb in [48, 72, 96]:
    seeds_approved = {k: v for k, v in all_seeds.items() if k in APPROVED}
    _, m_a = build_ensemble(seeds_approved, embargo_h=emb)
    print_m(f"SOLO APROBADAS  | Embargo {emb}H", m_a)

    seeds_all = all_seeds
    _, m_all = build_ensemble(seeds_all, embargo_h=emb)
    print_m(f"TODAS (5 seeds) | Embargo {emb}H", m_all)
    print("-"*60)

# ─────────────────────────────────────────
# TEST B: ¿Por qué las rechazadas contaminan?
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("ANÁLISIS DE CONTAMINACIÓN DE SEEDS RECHAZADAS")
print("=" * 60)
for sid in REJECTED:
    df = all_seeds[sid].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if 'is_win' in df.columns:
        wins = df['is_win'].astype(bool)
    elif 'return_pct' in df.columns:
        wins = df['return_pct'] > 0
    ret = df['return_pct']
    sharpe = (ret.mean() / ret.std() * np.sqrt(8760)) if ret.std() > 0 else 0
    print(f"""
  Seed {sid}: WR={wins.mean()*100:.1f}% | Sharpe={sharpe:.3f}
  Ret/Trade medio = {ret.mean()*100:.5f}%
  Avg Win  = {ret[ret>0].mean()*100:.5f}%  | Avg Loss = {ret[ret<0].mean()*100:.5f}%
  Wins: {(ret>0).sum()} | Losses: {(ret<0).sum()} | Zeros: {(ret==0).sum()}""")
