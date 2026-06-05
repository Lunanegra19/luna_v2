"""
H-G: Kelly Dinámico — Análisis de viabilidad
=============================================
Hipótesis: Kelly constante 3.5% no adapta el tamaño a la confianza de la señal.
Un Kelly dinámico basado en prob_cal mejoraría el PnL ajustado al riesgo.

Test: simular retornos con Kelly dinámico vs constante usando los datos de la run.
"""
import sys
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

SEP = '─'*68

wfb_dir = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')
all_dfs = []
for f in sorted(wfb_dir.glob('oos_trades_W*_seed*.parquet')):
    try:
        df = pd.read_parquet(f)
        if len(df) == 0: continue
        parts = f.stem.split('_')
        df['seed']   = int(next(p.replace('seed','') for p in parts if p.startswith('seed')))
        df['window'] = next(p for p in parts if p.startswith('W'))
        all_dfs.append(df)
    except: pass
df_all = pd.concat(all_dfs, ignore_index=True)

print(SEP)
print('H-G FASE 1: ¿Tiene prob_cal varianza suficiente para Kelly dinámico?')
print(SEP)
for col in ['xgb_prob_cal', 'xgb_prob', 'meta_v2_prob']:
    if col in df_all.columns:
        vals = df_all[col].dropna()
        print(f'  {col}: mean={vals.mean():.4f} std={vals.std():.4f} '
              f'min={vals.min():.4f} max={vals.max():.4f} '
              f'p10={vals.quantile(0.1):.4f} p90={vals.quantile(0.9):.4f}')
print()

print(SEP)
print('H-G FASE 2: ¿prob_cal predice el retorno individual? (señal informacional)')
print(SEP)
if 'xgb_prob_cal' in df_all.columns:
    r, p = stats.spearmanr(df_all['xgb_prob_cal'].dropna(),
                            df_all.loc[df_all['xgb_prob_cal'].notna(), 'return_pct'])
    print(f'  Spearman(prob_cal, return): r={r:+.4f} p={p:.4f}')
    print(f'  Veredicto: {"SEÑAL REAL — Kelly dinámico viable" if p < 0.05 else "RUIDO — Kelly dinámico no aportaría edge"}')
    print()

    # Cuantiles de prob_cal vs WR
    print('  Quintiles de prob_cal → WR observado:')
    df_tmp = df_all[['xgb_prob_cal','is_win','return_pct']].dropna()
    df_tmp['quintile'] = pd.qcut(df_tmp['xgb_prob_cal'], 5, labels=False)
    for q, grp in df_tmp.groupby('quintile'):
        wr = float(grp['is_win'].mean())
        ev = float(grp['return_pct'].mean())
        plim = grp['xgb_prob_cal'].quantile([0,1]).values
        print(f'    Q{q+1} [{plim[0]:.3f}-{plim[1]:.3f}]: N={len(grp)} WR={wr*100:.1f}% EV={ev*100:+.5f}%')
    print()

print(SEP)
print('H-G FASE 3: Simulación Kelly Dinámico vs Constante')
print(SEP)

KELLY_CONST = 0.035   # actual
MAX_KELLY   = 0.07    # cap institucional (Half-Kelly → 14.17% / 2)

if 'xgb_prob_cal' in df_all.columns:
    df_sim = df_all[['xgb_prob_cal', 'return_pct', 'is_win', 'window', 'seed']].dropna().copy()

    # Kelly constante
    df_sim['ret_kelly_const']  = df_sim['return_pct'] * KELLY_CONST

    # Kelly dinámico: f(p) = min(MAX_KELLY, max(0, 2*p - 1))
    # Fórmula Kelly: f* = p - (1-p)/b donde b=1 (binario simétrico)
    # Simplificada: f* = 2p - 1 (clipa en 0 si p < 0.5)
    df_sim['kelly_dyn'] = np.clip(2 * df_sim['xgb_prob_cal'] - 1, 0, MAX_KELLY)
    df_sim['ret_kelly_dyn'] = df_sim['return_pct'] * df_sim['kelly_dyn']

    # Half-Kelly dinámico (más conservador)
    df_sim['kelly_half_dyn'] = df_sim['kelly_dyn'] * 0.5
    df_sim['ret_kelly_half_dyn'] = df_sim['return_pct'] * df_sim['kelly_half_dyn']

    print(f'  N trades simulados: {len(df_sim)}')
    print()
    print(f'  {"Estrategia":30s} | {"EV_trade":>10} | {"Sharpe":>8} | {"MaxDD":>8} | {"Calmar":>8}')
    print(f'  {"-"*30}-+-{"-"*10}-+-{"-"*8}-+-{"-"*8}-+-{"-"*8}')

    def sim_stats(ret_col):
        rets = df_sim[ret_col]
        ev   = float(rets.mean())
        sh   = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
        cum  = (1 + rets).cumprod()
        roll_max = cum.cummax()
        dd   = float(((cum - roll_max) / roll_max).min())
        calmar = -ev * 252 / dd if dd < 0 else 0
        return ev, sh, dd, calmar

    for name, col in [
        ('Kelly Constante 3.5%',   'ret_kelly_const'),
        ('Kelly Dinámico (Full)',   'ret_kelly_dyn'),
        ('Kelly Dinámico (Half)',   'ret_kelly_half_dyn'),
    ]:
        ev, sh, dd, calmar = sim_stats(col)
        print(f'  {name:30s} | {ev*100:>+9.5f}% | {sh:>+7.3f} | {dd*100:>+7.3f}% | {calmar:>+7.3f}')

    print()
    # Distribución de kelly_dyn
    print(f'  Kelly dinámico distribution:')
    print(f'    mean={df_sim["kelly_dyn"].mean():.4f} std={df_sim["kelly_dyn"].std():.4f}')
    print(f'    trades con kelly=0 (prob<0.5): {(df_sim["kelly_dyn"]==0).sum()} '
          f'({(df_sim["kelly_dyn"]==0).mean()*100:.1f}%)')
    print(f'    → Kelly dinámico suprime trades de baja confianza automáticamente')

print()
print(SEP)
print('H-G CONCLUSIÓN:')
print(SEP)
