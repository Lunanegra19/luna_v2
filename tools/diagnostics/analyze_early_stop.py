"""Análisis del early-stop: por qué las seeds mueren en W1-W3."""
import sys, json, numpy as np, pandas as pd, yaml
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
from pathlib import Path

wfb_dir = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')
ROOT    = Path('g:/Mi unidad/ia/luna_v2')

print('='*70)
print('ANALISIS EARLY-STOP: ¿Por qué las seeds mueren en W1-W3?')
print('='*70)
print()

# 1. Dynamic benchmark
bench_path = wfb_dir / 'dynamic_benchmark.json'
if bench_path.exists():
    with open(bench_path) as f:
        bench = json.load(f)
    print('dynamic_benchmark.json:')
    for k, v in bench.items():
        print(f'  {k}: {v}')
else:
    print('dynamic_benchmark.json: NO EXISTE (baseline=50.0)')
    bench = {'champion_score': 50.0}
print()

champion_score = bench.champion_score

# 2. Early-stop logs
early_stop_files = sorted(wfb_dir.glob('early_stop_seed*.json'))
print(f'Seeds con early-stop en la ultima run: {len(early_stop_files)}')
for f in early_stop_files:
    with open(f) as fp:
        data = json.load(fp)
    print(f'  {f.name}: ventanas evaluadas={data.windows_evaluated} | razon={str(data.reason)[:80]}')
print()

# 3. Ventanas completadas por seed
print('Ventanas completadas por seed:')
seeds_found = set()
for f in wfb_dir.glob('oos_trades_W*_seed*.parquet'):
    parts = f.stem.split('_')
    for p in parts:
        if p.startswith('seed'):
            seeds_found.add(int(p.replace('seed', '')))

for seed in sorted(seeds_found):
    windows = []
    for f in sorted(wfb_dir.glob(f'oos_trades_W*_seed{seed}.parquet')):
        parts = f.stem.split('_')
        w = int([p.replace('W', '') for p in parts if p.startswith('W')][0])
        try:
            df = pd.read_parquet(f)
            wr = float(df['is_win'].mean()) if 'is_win' in df.columns and len(df) > 0 else float('nan')
            windows.append((w, len(df), wr))
        except Exception:
            windows.append((w, 0, float('nan')))
    max_w = max(w for w, n, wr in windows) if windows else 0
    win_str = '  |  '.join([f'W{w}: {n}t WR={wr*100:.0f}%' for w, n, wr in windows])
    print(f'  seed{seed}: max_W={max_w}  [{win_str}]')
print()

# 4. Configuracion de ventanas
with open(ROOT / 'config' / 'settings.yaml') as f:
    cfg = yaml.safe_load(f)
windows_cfg    = cfg.wfb.windows
prune_thr      = cfg.wfb.prune_threshold
_N_WINDOWS_CFG = len(windows_cfg)

print(f'settings.yaml:')
print(f'  wfb.windows: {_N_WINDOWS_CFG} ventanas definidas')
print(f'  wfb.prune_threshold: {prune_thr}')
print()
if windows_cfg:
    for i, w in enumerate(windows_cfg, 1):
        is_s  = w.is_start
        is_e  = w.is_end
        oos_s = w.oos_start
        oos_e = w.oos_end
        print(f'  W{i}: IS=[{is_s}→{is_e}] OOS=[{oos_s}→{oos_e}]')
print()

# 5. Simulacion del early-stop para cada seed tras W3
print('='*70)
print('SIMULACION EARLY-STOP: ¿A qué score corta el sistema tras W3?')
print('='*70)
print()

_WR_OPT  = 0.70
_RR_OPT  = 2.0
_RET_OPT = 0.01

for seed in sorted(seeds_found):
    windows = []
    dfs = []
    for f in sorted(wfb_dir.glob(f'oos_trades_W*_seed{seed}.parquet')):
        parts = f.stem.split('_')
        w = int([p.replace('W', '') for p in parts if p.startswith('W')][0])
        try:
            df = pd.read_parquet(f)
            if len(df) > 0:
                windows.append(w)
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        continue

    combined = pd.concat(dfs, ignore_index=False)
    n = len(combined)

    wr_per_w = {}
    for df, w in zip(dfs, windows):
        wr_per_w[w] = float(df['is_win'].mean()) if 'is_win' in df.columns and len(df) > 0 else 0.5

    wr_seen  = float(combined['is_win'].mean()) if 'is_win' in combined.columns else 0.5
    wr_min_s = min(wr_per_w.values()) if wr_per_w else wr_seen
    wr_rng_s = max(wr_per_w.values()) - min(wr_per_w.values()) if len(wr_per_w) > 1 else 0.0

    r = combined['return_pct'].dropna() if 'return_pct' in combined.columns else pd.Series(dtype=float)
    mean_ret = float(r.mean()) if len(r) > 0 else 0.0
    avg_win  = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    avg_loss = float(abs(r[r < 0].mean())) if (r < 0).any() else 1e-9
    rr = min(avg_win / avg_loss, 10.0)

    s_wr_global = float(np.clip((wr_seen  - 0.40) / (0.75 - 0.40) * 100, 0, 100))
    s_wr_min    = float(np.clip((wr_min_s - 0.35) / (0.70 - 0.35) * 100, 0, 100))
    s_stability = float(np.clip((1 - wr_rng_s / 0.50) * 100, 0, 100))
    s_rr        = float(np.clip((rr - 0.5) / (2.0 - 0.5) * 100, 0, 100))
    s_ret       = float(np.clip((mean_ret - 0.0) / (0.01 - 0.0) * 100, 0, 100))
    score_seen  = 0.20 * s_wr_global + 0.30 * s_wr_min + 0.25 * s_stability + 0.10 * s_rr + 0.15 * s_ret

    n_remaining = max(0, _N_WINDOWS_CFG - len(windows))

    s_wr_global_opt = float(np.clip((_WR_OPT - 0.40) / (0.75 - 0.40) * 100, 0, 100))
    s_wr_min_opt    = max(s_wr_min, float(np.clip((_WR_OPT - 0.35) / (0.70 - 0.35) * 100, 0, 100)))
    s_stability_opt = 100.0
    s_rr_opt        = float(np.clip((_RR_OPT - 0.5) / (2.0 - 0.5) * 100, 0, 100))
    s_ret_opt       = float(np.clip((_RET_OPT - 0.0) / (0.01 - 0.0) * 100, 0, 100))
    score_remaining = 0.20 * s_wr_global_opt + 0.30 * s_wr_min_opt + 0.25 * s_stability_opt + 0.10 * s_rr_opt + 0.15 * s_ret_opt

    w_seen = len(windows) / _N_WINDOWS_CFG
    w_rem  = n_remaining / _N_WINDOWS_CFG
    ub = score_seen * w_seen + score_remaining * w_rem

    thr = champion_score * prune_thr
    would_prune = ub < thr and champion_score > 0

    print(f'seed{seed} tras W{sorted(windows)}:')
    print(f'  score_seen={score_seen:.1f} | UB={ub:.1f} | CUTOFF = {thr:.1f} | '
          f'PRUNED={would_prune}')
    print(f'  WR_global={wr_seen*100:.1f}% WR_min={wr_min_s*100:.1f}% WR_range={wr_rng_s*100:.1f}%pp '
          f'RR={rr:.2f} mean_ret={mean_ret*100:.3f}%')
    print(f'  Componentes: s_wr_global={s_wr_global:.1f} s_wr_min={s_wr_min:.1f} '
          f's_stability={s_stability:.1f} s_rr={s_rr:.1f} s_ret={s_ret:.1f}')
    print()
