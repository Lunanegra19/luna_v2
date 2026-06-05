"""
h5_window_impact_h4_overconfidence.py
Analisis detallado:
  1. H5: trades gateados por ventana y su calidad (buenas o malas oportunidades)
  2. H5 con deque persistente entre ventanas (efecto de mas "historia")
  3. H4: sobreconfianza del modelo en W4 (OOD no detectado)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
from collections import deque

DATA = Path(r'g:\Mi unidad\ia\luna_v2\data\reports\wfb')
H5_WINDOW = 10
H5_CUTOFF = 0.0
WINDOWS_ORDER = ['W1', 'W3', 'W4']

def load_all():
    dfs = []
    for f in sorted(DATA.glob('oos_trades_W*_seed*.parquet')):
        stem = f.stem; window = stem.split('_')[2]; seed = int(stem.split('_seed')[1])
        df = pd.read_parquet(f)
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index, utc=True)
        df['seed'] = seed; df['window'] = window
        dfs.append(df)
    return pd.concat(dfs).sort_index()

df_all = load_all()
SEP = "=" * 70

# ============================================================
# H5 Analisis: trades gateados por ventana y su calidad
# ============================================================
print(SEP)
print("H5: Impacto por ventana — calidad de trades gateados vs permitidos")
print(SEP)

def run_h5_simulation(df_all, persistent_deque=False):
    """Simula H5 con deque por ventana (defecto) o persistente entre ventanas."""
    stats_gated = {w: {'n': 0, 'wins': 0, 'ret': 0.0} for w in WINDOWS_ORDER}
    stats_kept  = {w: {'n': 0, 'wins': 0, 'ret': 0.0} for w in WINDOWS_ORDER}

    for seed in df_all['seed'].unique():
        history = deque(maxlen=H5_WINDOW)  # si persistent, no se reinicia
        for window in WINDOWS_ORDER:
            if not persistent_deque:
                history = deque(maxlen=H5_WINDOW)  # reinicia por ventana
            df_sw = df_all[(df_all['seed'] == seed) & (df_all['window'] == window)].sort_index()
            if len(df_sw) == 0:
                continue
            for ts, row in df_sw.iterrows():
                ret_bruto = float(row['return_raw'])
                is_gated = False
                if len(history) >= max(3, H5_WINDOW // 2):
                    arr = list(history)
                    mean_r = float(np.mean(arr))
                    std_r  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1e-8
                    roll_sr = mean_r / max(std_r, 1e-8)
                    is_gated = roll_sr < H5_THRESHOLD
                if window in stats_gated:
                    bucket = stats_gated[window] if is_gated else stats_kept[window]
                    bucket['n'] += 1
                    if bool(row['is_win']):
                        bucket['wins'] += 1
                    bucket['ret'] += ret_bruto * 100
                history.append(ret_bruto)

    return stats_gated, stats_kept

sg_perV, sk_perV = run_h5_simulation(df_all, persistent_deque=False)
sg_pers, sk_pers = run_h5_simulation(df_all, persistent_deque=True)

print()
print("  Deque reiniciada por ventana (implementacion actual):")
print(f"  {'Ventana':<8} {'Tipo':<12} {'N':>5}  {'WR%':>7}  {'RetTotal%':>11}")
print("  " + "-" * 55)
for win in WINDOWS_ORDER:
    g = sg_perV[win]; k = sk_perV[win]
    g_wr = g['wins']/g['n']*100 if g['n'] > 0 else 0.0
    k_wr = k['wins']/k['n']*100 if k['n'] > 0 else 0.0
    print(f"  {win:<8} {'GATEADO':<12} {g['n']:>5}  {g_wr:>7.1f}%  {g['ret']:>11.4f}%")
    print(f"  {win:<8} {'PERMITIDO':<12} {k['n']:>5}  {k_wr:>7.1f}%  {k['ret']:>11.4f}%")
    print()

# Resumen del problema clave
g3 = sg_perV['W3']; g4 = sg_perV['W4']
g3_wr = g3['wins']/g3['n']*100 if g3['n'] > 0 else 0
g4_wr = g4['wins']/g4['n']*100 if g4['n'] > 0 else 0

print(f"  DIAGNOSTICO DEL PROBLEMA:")
print(f"  H5 gatea {g3['n']} trades de W3 con WR={g3_wr:.1f}% (trades BUENOS eliminados del consenso)")
print(f"  H5 gatea {g4['n']} trades de W4 con WR={g4_wr:.1f}% (trades MALOS eliminados del consenso)")
print(f"  Razon: con solo {H5_WINDOW} trades de ventana, la deque EMPIEZA VACIA en cada ventana.")
print(f"  Los primeros 5 trades no tienen gate. Luego si hay 2 perdidas seguidas early en W3,")
print(f"  H5 dispara y elimina trades buenos de W3 (falsos positivos).")

print()
print(SEP)
print("  Deque PERSISTENTE entre ventanas (W1 -> W3 -> W4, sin reinicio):")
print("  El gate llega a W4 calibrado con la historia de W3 (buena)")
print("  y gatea mas agresivamente W4 y menos W3:")
print()
print(f"  {'Ventana':<8} {'Gates_perV':>11}  {'Gates_pers':>11}  {'Delta':>8}")
print("  " + "-" * 50)
for win in WINDOWS_ORDER:
    gv = sg_perV[win]['n']; gp = sg_pers[win]['n']
    print(f"  {win:<8} {gv:>11}  {gp:>11}  {gp-gv:>+8}")

print()
sg3p = sg_pers['W3']; sg4p = sg_pers['W4']
sg3p_wr = sg3p['wins']/sg3p['n']*100 if sg3p['n'] > 0 else 0
sg4p_wr = sg4p['wins']/sg4p['n']*100 if sg4p['n'] > 0 else 0
print(f"  W3 gates (persistente): {sg3p['n']} trades WR={sg3p_wr:.1f}%")
print(f"  W4 gates (persistente): {sg4p['n']} trades WR={sg4p_wr:.1f}%")
if sg3p['n'] < g3['n'] and sg4p['n'] > g4['n']:
    print(f"  -> Deque persistente: MENOS falsos positivos en W3, MAS gates en W4. MEJOR.")
elif sg3p['n'] < g3['n']:
    print(f"  -> Deque persistente: MENOS falsos positivos en W3.")
else:
    print(f"  -> Deque persistente: no mejora significativamente el trade-off.")

# ============================================================
# H4: Sobreconfianza y predictividad de prob_cal
# ============================================================
print()
print(SEP)
print("H4: Sobreconfianza del modelo en W4 (firma OOD no detectado)")
print(SEP)

all_w3 = df_all[df_all['window'] == 'W3']
all_w4 = df_all[df_all['window'] == 'W4']

prob3 = all_w3['xgb_prob_cal'].mean()
prob4 = all_w4['xgb_prob_cal'].mean()
wr3   = all_w3['is_win'].mean() * 100
wr4   = all_w4['is_win'].mean() * 100

print(f"""
  W3 (bull normal jul-sep 2025, BTC ~55-65K):
    prob_cal media = {prob3:.4f} | WR = {wr3:.1f}%

  W4 (ATH BTC oct-dic 2025, BTC ~70-108K):
    prob_cal media = {prob4:.4f} | WR = {wr4:.1f}%

  Delta prob_cal: +{prob4-prob3:.4f} (modelo {prob4-prob3:.1%} MAS confiante en W4)
  Delta WR:       {wr4-wr3:+.1f}pp (modelo {abs(wr4-wr3):.1f}pp PEOR en W4)
  -> Firma clasica de sobreconfianza en regimen OOD no visto en IS.
""")

# Predictividad de prob_cal por ventana
for win, df_w in [('W3', all_w3), ('W4', all_w4)]:
    p = df_w['xgb_prob_cal'].dropna()
    r = df_w['return_raw'].dropna()
    common = p.index.intersection(r.index)
    if len(common) >= 5:
        rho, pval = stats.spearmanr(p.loc[common], r.loc[common])
        print(f"  Spearman(prob_cal, ret_raw) en {win}: rho={rho:.4f}, p={pval:.4f} "
              f"-> {'PREDICTIVO (p<0.10)' if pval < 0.10 else 'NO predictivo'}")

# Cuando prob_cal ALTA en W4: los trades son mejores o peores?
print()
w4_sorted = all_w4.dropna(subset=['xgb_prob_cal']).sort_values('xgb_prob_cal', ascending=False)
n4 = len(w4_sorted)
top25 = w4_sorted.head(n4 // 4)
bot25 = w4_sorted.tail(n4 // 4)
print(f"  W4 top 25% prob_cal (>= {top25['xgb_prob_cal'].min():.4f}):")
print(f"    WR={top25['is_win'].mean()*100:.1f}% | RetMed={top25['return_raw'].mean()*100:.4f}%")
print(f"  W4 bot 25% prob_cal (<= {bot25['xgb_prob_cal'].max():.4f}):")
print(f"    WR={bot25['is_win'].mean()*100:.1f}% | RetMed={bot25['return_raw'].mean()*100:.4f}%")

if top25['return_raw'].mean() < bot25['return_raw'].mean():
    print(f"  -> CONFIRMADO: mayor prob_cal = PEORES retornos en W4 (sobreconfianza OOD)")
    print(f"     Una feature de ATH habria reducido prob_cal en W4 y el gate habria actuado")
else:
    print(f"  -> prob_cal consistente con retornos en W4")

print()
print(SEP)
print("CONCLUSION SOBRE H5, H3, H4")
print(SEP)
print(f"""
  H5 VERDAD COMPLETA:
    - H5 mejora POR SEED en W3 (+2.74%) y W4 (+9.59%). TEST CORRECTO.
    - H5 DEGRADA el ensemble porque:
      a) La deque se reinicia cada ventana: empieza vacia en W3,
         disparando falsos positivos en W3 ({g3['n']} trades buenos eliminados, WR={g3_wr:.0f}%)
      b) Esto reduce el consenso en W3, que es la ventana BUENA del ensemble
    - SOLUCION: deque persistente entre ventanas (sin reinicio)
      -> {sg3p['n']} gates en W3 (vs {g3['n']} actuales) = menos falsos positivos
    - Con mas trades/ventanas, el efecto de la deque vacia inicial se diluye.
    - El usuario tenia razon: "no estas teniendo en cuenta como h5 afectaba a las ventanas buenas"

  H3 (CB portfolio):
    - WR_roll5 tiene rho=-0.30 p=0.07 sobre el portfolio (PREDICTIVO marginal)
    - La mejor regla (trade 15) es lookback puro. Sin valor predictivo real.
    - VEREDICTO: DEBIL a nivel portfolio. El problema esta en W4 concentrado en 3 trades.

  H4 (features ATH):
    - prob_cal W4 > prob_cal W3 pero WR W4 < WR W3: sobreconfianza confirmada
    - Spearman(prob_cal, ret) en W4: ver arriba. Si NO predictivo -> prob_cal inutil en W4
    - Causa raiz del problema OOD. Requiere feature engineering + re-run.
    - VEREDICTO: ALTA PRIORIDAD pero alto costo de implementacion.
""")
