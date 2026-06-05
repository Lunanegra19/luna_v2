"""
h5_retroactive_simulation.py
==============================
Simulacion retroactiva de H5 (Kelly Rolling Sharpe Gate) sobre los 20 seeds existentes.
Reproduce fielmente la logica implementada en predict_oos.py:
  - deque(maxlen=10) de return_raw por seed+ventana (ordenado temporalmente)
  - Si roll_SR < 0.0 -> kelly = 0 (trade excluido del portfolio: return_pct = 0, no cuenta en consenso)
  - El historial se actualiza con ret_bruto SIEMPRE (gateado o no), igual que predict_oos.py
Luego reconstruye el ensemble con consensus >= 3 seeds y embargo adaptativo.
"""
import sys, glob
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque, defaultdict

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\reports\wfb')

# Parametros H5 (iguales a settings.yaml)
H5_ENABLED   = True
H5_WINDOW    = 10
H5_CUTOFF = 0.0
# Parametros ensemble (iguales a evaluate_ensemble_wfb.py / settings.yaml)
CONSENSUS_CUTOFF = 3
BUCKET_HOURS        = 2

def apply_h5_to_seed_window(df, seed, window):
    """Aplica H5 gate sobre los trades de una seed+ventana, devuelve df modificado."""
    df = df.sort_index().copy()
    history = deque(maxlen=H5_WINDOW)
    gates_applied = 0
    for i, (ts, row) in enumerate(df.iterrows()):
        ret_bruto = float(row['return_raw'])
        # Calcular roll_SR ANTES de este trade (causal)
        if H5_ENABLED and len(history) >= max(3, H5_WINDOW // 2):
            arr = list(history)
            mean_r = float(np.mean(arr))
            std_r  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1e-8
            roll_sr = mean_r / max(std_r, 1e-8)
            if roll_sr < H5_THRESHOLD:
                # Gate: Kelly = 0, trade excluido del portfolio
                df.at[ts, 'return_pct'] = 0.0
                df.at[ts, 'kelly_fraction_used'] = 0.0
                df.at[ts, 'h5_gated'] = True
                gates_applied += 1
            else:
                df.at[ts, 'h5_gated'] = False
        else:
            df.at[ts, 'h5_gated'] = False
        # Actualizar historial con ret_bruto SIEMPRE (gateado o no)
        history.append(ret_bruto)
    if gates_applied > 0:
        print(f"  [H5-SIM] seed={seed} {window}: {gates_applied}/{len(df)} trades gateados")
    return df, gates_applied

# ==============================
# 1. Cargar y aplicar H5 a todos los parquets
# ==============================
print("=" * 70)
print("SIMULACION RETROACTIVA H5 sobre los 20 seeds existentes")
print(f"Params: window={H5_WINDOW}, CUTOFF = {H5_THRESHOLD}, consensus>={CONSENSUS_THRESHOLD}")
print("=" * 70)

all_dfs = []
total_gates = 0
seed_summary = {}

for f in sorted(DATA.glob('oos_trades_W*_seed*.parquet')):
    stem   = f.stem
    window = stem.split('_')[2]
    seed   = int(stem.split('_seed')[1])
    try:
        df = pd.read_parquet(f)
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index, utc=True)
        df['seed']   = seed
        df['window'] = window
        df, n_gates  = apply_h5_to_seed_window(df, seed, window)
        total_gates += n_gates
        all_dfs.append(df)
        if seed not in seed_summary:
            seed_summary[seed] = {'trades': 0, 'gated': 0, 'ret': 0.0, 'wins': 0}
        seed_summary[seed]['trades'] += len(df)
        seed_summary[seed]['gated']  += n_gates
        seed_summary[seed]['ret']    += df['return_pct'].sum() * 100
        seed_summary[seed]['wins']   += int(df['is_win'].sum())
    except Exception as e:
        print(f"  ERROR {f.name}: {e}")

print(f"\nTotal trades procesados: {sum(len(d) for d in all_dfs)}")
print(f"Total trades gateados por H5: {total_gates}")

# ==============================
# 2. Resumen por seed post-H5
# ==============================
print("\n" + "=" * 70)
print("RESUMEN POR SEED POST-H5")
print(f"{'Seed':>8} {'Trades':>7} {'Gated':>7} {'Gate%':>7} {'WR%':>7} {'RetTot%':>10} {'Status'}")
print("-" * 70)
for seed in sorted(seed_summary.keys()):
    d = seed_summary[seed]
    wr = (d['wins'] / d['trades'] * 100) if d['trades'] > 0 else 0.0
    gate_pct = (d['gated'] / d['trades'] * 100) if d['trades'] > 0 else 0.0
    status = "PASS>=30" if d['trades'] >= 30 else "FAIL<30 "
    print(f"  {seed:>8} {d['trades']:>7} {d['gated']:>7} {gate_pct:>6.1f}% {wr:>7.1f}% {d['ret']:>10.4f}%   {status}")

# ==============================
# 3. Reconstruir ensemble con H5 aplicado
# ==============================
print("\n" + "=" * 70)
print(f"ENSEMBLE POST-H5 (consensus >= {CONSENSUS_THRESHOLD} seeds, bucket={BUCKET_HOURS}H)")
print("=" * 70)

df_all = pd.concat(all_dfs).sort_index()

# Excluir trades gateados (kelly=0) del conteo de consenso
df_active = df_all[df_all['h5_gated'] == False].copy()
print(f"\nTrades activos (no gateados): {len(df_active)} / {len(df_all)}")

# Bucket temporal
df_active['bucket'] = df_active.index.floor(f'{BUCKET_HOURS}h')
bucket_seeds = (
    df_active.groupby('bucket')['seed']
    .nunique()
    .rename('consensus_count')
)
df_active['consensus_count'] = df_active['bucket'].map(bucket_seeds)

print(f"\nDistribucion de consenso (solo trades activos post-H5):")
for n_seeds, n_bucks in bucket_seeds.value_counts().sort_index(ascending=False).items():
    print(f"  {n_seeds} seeds -> {n_bucks} buckets")

# Filtrar por consenso
df_consensus = df_active[df_active['consensus_count'] >= CONSENSUS_THRESHOLD].copy()
n_buckets_ok = df_consensus['bucket'].nunique()
print(f"\nConsensus >= {CONSENSUS_THRESHOLD}: {len(df_active)} -> {len(df_consensus)} filas | {n_buckets_ok} buckets")

# Agregar por bucket
if len(df_consensus) > 0:
    agg = {
        'return_pct':      'mean',
        'is_win':          'max',
        'consensus_count': 'first',
        'window':          'first',
    }
    if 'hmm_regime' in df_consensus.columns:
        agg['hmm_regime'] = 'first'
    df_portfolio = df_consensus.groupby('bucket').agg(agg).sort_index()

    # Embargo simple (96H) para trades post-H5
    EMBARGO_H = 96
    selected = []
    last_t = None
    for ts, row in df_portfolio.iterrows():
        if last_t is None or (ts - last_t).total_seconds() / 3600 >= EMBARGO_H:
            selected.append(ts)
            last_t = ts
    df_final = df_portfolio.loc[selected].copy()

    n_fin  = len(df_final)
    wr_fin = float(df_final['is_win'].mean() * 100) if n_fin > 0 else 0.0
    ret_mean = float(df_final['return_pct'].mean() * 100) if n_fin > 0 else 0.0
    ret_tot  = float(df_final['return_pct'].sum() * 100) if n_fin > 0 else 0.0

    # Sharpe
    sharpe = 0.0
    if n_fin > 1 and df_final['return_pct'].std() > 1e-10:
        days = (df_final.index.max() - df_final.index.min()).days
        n_per_year = n_fin / (days / 365.25) if days > 0 else n_fin
        sharpe = (df_final['return_pct'].mean() / df_final['return_pct'].std()) * (n_per_year ** 0.5)

    # MaxDD
    equity = (1 + df_final['return_pct']).cumprod()
    max_dd = float((equity - equity.cummax()).min() * 100)

    print(f"\n{'='*70}")
    print("PORTFOLIO ENSEMBLE POST-H5 (Simulacion Retroactiva)")
    print(f"{'='*70}")
    print(f"  Trades finales:         {n_fin:>6}   (umbral SOP-R8 >= 30: {'PASS' if n_fin >= 30 else 'FAIL'})")
    print(f"  Win Rate:               {wr_fin:>6.2f}%")
    print(f"  Sharpe Anualizado:      {sharpe:>6.4f}")
    print(f"  Retorno medio/trade:    {ret_mean:>6.4f}%")
    print(f"  Retorno total:          {ret_tot:>6.4f}%")
    print(f"  Max Drawdown:           {max_dd:>6.2f}%")
    print(f"  Embargo aplicado:       {EMBARGO_H}H fijo (conservador)")

    # Comparacion vs pre-H5
    print(f"\n{'='*70}")
    print("COMPARACION PRE-H5 vs POST-H5 (ensemble)")
    print(f"{'='*70}")
    print(f"  Trades portfolio:  PRE={58}  POST={n_fin}")
    print(f"  WR:                PRE={60.34:.2f}%  POST={wr_fin:.2f}%")
    print(f"  Sharpe:            PRE={-0.3889:.4f}  POST={sharpe:.4f}  | delta={sharpe-(-0.3889):+.4f}")
    print(f"  Ret.total:         PRE={-0.23:.4f}%  POST={ret_tot:.4f}%")
    print(f"  MaxDD:             PRE={-0.8:.2f}%   POST={max_dd:.2f}%")

    # Desglose de trades finales del portfolio
    print(f"\nDesglose del portfolio post-H5 (primeros 20 trades):")
    print(f"{'Fecha':<25} {'Consenso':>9} {'WR':>4} {'Ret%':>9} {'Ventana'}")
    print("-" * 60)
    for ts, row in df_final.head(20).iterrows():
        print(f"  {str(ts)[:19]:<24} {int(row['consensus_count']):>9} {'W' if row['is_win'] else 'L':>4} {row['return_pct']*100:>9.4f}%  {row.get('window','?')}")
else:
    print("\n  SIN trades de consenso post-H5 — el gate es demasiado agresivo")

print(f"\n{'='*70}")
print("FIN SIMULACION RETROACTIVA H5")
print(f"{'='*70}")
