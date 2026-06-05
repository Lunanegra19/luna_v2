"""
Investigacion profunda MEJORA-WR-01 + MEJORA-TRADES-01
=======================================================
Analiza en profundidad:
1. Por que seed2025 tiene WR=40% vs seed1337 WR=51%
2. Distribucion de probabilidades XGB/MetaLabeler por regimen
3. La relacion entre threshold y WR/n_trades (trade-off)
"""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PATHS = {
    '2025_W2': r'G:\Mi unidad\ia\luna_v2\data\runs\WFB_20260521_012823_seed2025\seed2025\W2\oos_trades.parquet',
    '2025_W3': r'G:\Mi unidad\ia\luna_v2\data\runs\WFB_20260521_012823_seed2025\seed2025\W3\oos_trades.parquet',
    '2025_W4': r'G:\Mi unidad\ia\luna_v2\data\runs\WFB_20260521_012823_seed2025\seed2025\W4\oos_trades.parquet',
    '2025_W5': r'G:\Mi unidad\ia\luna_v2\data\runs\WFB_20260521_012823_seed2025\seed2025\W5\oos_trades.parquet',
    '1337_W2': r'G:\Mi unidad\ia\luna_v2\data\runs\WFB_20260521_033115_seed1337\seed1337\W2\oos_trades.parquet',
    '1337_W3': r'G:\Mi unidad\ia\luna_v2\data\runs\WFB_20260521_033115_seed1337\seed1337\W3\oos_trades.parquet',
    '1337_W4': r'G:\Mi unidad\ia\luna_v2\data\runs\WFB_20260521_033115_seed1337\seed1337\W4\oos_trades.parquet',
}

# Tambien la run sfi16 aprobada
SFI16_PATHS = {
    '1337_sfi16_W2': r'G:\Mi unidad\ia\luna_v2\data\runs\WFB_20260521_033115_seed1337\seed1337\W2\oos_trades.parquet',
    '1337_sfi16_W3': r'G:\Mi unidad\ia\luna_v2\data\runs\WFB_20260521_033115_seed1337\seed1337\W3\oos_trades.parquet',
    '1337_sfi16_W4': r'G:\Mi unidad\ia\luna_v2\data\runs\WFB_20260521_033115_seed1337\seed1337\W4\oos_trades.parquet',
}

frames = {}
for k, p in PATHS.items():
    try:
        frames[k] = pd.read_parquet(p)
        frames[k]['_key'] = k
    except Exception as e:
        print(f"WARN: no se pudo leer {k}: {e}")

s2025 = pd.concat([v for k, v in frames.items() if k.startswith('2025')], ignore_index=True)
s1337 = pd.concat([v for k, v in frames.items() if k.startswith('1337')], ignore_index=True)

# ===== PARTE 1: Analisis comparativo global =====
print("=" * 70)
print("[MEJORA-WR-01] Analisis comparativo seed2025 vs seed1337")
print("=" * 70)
print()
for label, df in [('seed2025', s2025), ('seed1337', s1337)]:
    if df.empty:
        continue
    wr = df['is_win'].mean() * 100
    ret = (np.prod(1 + df['return_pct']) - 1) * 100
    n = len(df)
    xp = df['xgb_prob_cal'].mean()
    mp = df['meta_v2_prob'].mean()
    st = df['signal_threshold'].mean()
    sharpe = df['return_pct'].mean() / (df['return_pct'].std() + 1e-10) * np.sqrt(n)
    print(f"  {label}: n={n} | WR={wr:.1f}% | ret={ret:+.2f}% | Sharpe={sharpe:.3f}")
    print(f"    xgb_prob_cal (media trades seleccionados): {xp:.4f}")
    print(f"    meta_v2_prob (media trades seleccionados): {mp:.4f}")
    print(f"    signal_threshold (media): {st:.4f}")
    print()

# ===== PARTE 2: Por regimen HMM =====
print("=" * 70)
print("[MEJORA-WR-01] WR, return y probabilidades por regimen HMM")
print("=" * 70)
print()

for label, df in [('seed2025', s2025), ('seed1337', s1337)]:
    if df.empty or 'hmm_regime' not in df.columns:
        print(f"  {label}: sin columna hmm_regime")
        continue
    print(f"  {label}:")
    print(f"  {'Regimen':<28} | {'N':>4} | {'WR%':>6} | {'mean_ret%':>9} | {'xgb_p':>6} | {'meta_p':>6} | {'thr':>6}")
    print("  " + "-" * 80)
    for reg, g in df.groupby('hmm_regime'):
        wr = g['is_win'].mean() * 100
        mr = g['return_pct'].mean() * 100
        xp = g['xgb_prob_cal'].mean()
        mp = g['meta_v2_prob'].mean()
        st = g['signal_threshold'].mean()
        flag = " <<< WR bajo" if wr < 45 else (" (+)" if wr > 55 else "")
        print(f"  {str(reg):<28} | {len(g):>4} | {wr:>6.1f} | {mr:>9.3f} | {xp:>6.3f} | {mp:>6.3f} | {st:>6.3f}{flag}")
    print()

# ===== PARTE 3: Ventana a ventana =====
print("=" * 70)
print("[MEJORA-WR-01] Detalle por ventana")
print("=" * 70)
print()
print(f"  {'Key':<12} | {'N':>3} | {'WR%':>6} | {'ret%':>7} | {'xgb_p':>6} | {'meta_p':>6} | {'thr':>6} | {'regime_distribucion'}")
print("  " + "-" * 90)
for k, df in sorted(frames.items()):
    wr = df['is_win'].mean() * 100
    ret = (np.prod(1 + df['return_pct']) - 1) * 100
    n = len(df)
    xp = df['xgb_prob_cal'].mean()
    mp = df['meta_v2_prob'].mean()
    st = df['signal_threshold'].mean()
    if 'hmm_regime' in df.columns:
        regimes = df['hmm_regime'].value_counts().to_dict()
        reg_str = " | ".join(f"{r[:12]}={cnt}" for r, cnt in list(regimes.items())[:3])
    else:
        reg_str = "N/A"
    print(f"  {k:<12} | {n:>3} | {wr:>6.1f} | {ret:>+7.2f} | {xp:>6.3f} | {mp:>6.3f} | {st:>6.3f} | {reg_str}")

print()

# ===== PARTE 4: Analisis de probabilidades - separar ganadores vs perdedores =====
print("=" * 70)
print("[MEJORA-TRADES-01] Distribucion de probabilidades en trades seleccionados")
print("Hipotesis: si bajamos el threshold, entramos trades con menor prob -> peor WR")
print("=" * 70)
print()

for label, df in [('seed2025', s2025), ('seed1337', s1337)]:
    if df.empty:
        continue
    wins = df[df['is_win'] == True]
    losses = df[df['is_win'] == False]
    print(f"  {label}:")
    print(f"    xgb_prob_cal  | Ganadores: {wins['xgb_prob_cal'].mean():.4f} | Perdedores: {losses['xgb_prob_cal'].mean():.4f} | GAP: {(wins['xgb_prob_cal'].mean() - losses['xgb_prob_cal'].mean()):.4f}")
    print(f"    meta_v2_prob  | Ganadores: {wins['meta_v2_prob'].mean():.4f} | Perdedores: {losses['meta_v2_prob'].mean():.4f} | GAP: {(wins['meta_v2_prob'].mean() - losses['meta_v2_prob'].mean()):.4f}")
    print()
    # Distribucion de xgb_prob_cal en deciles
    print(f"    Deciles xgb_prob_cal (trades seleccionados):")
    deciles = df['xgb_prob_cal'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    for q, v in deciles.items():
        print(f"      P{int(q*100):>2}: {v:.4f}")
    print()
