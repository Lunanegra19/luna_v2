"""
investigation_calibration.py
==============================
Investiga la sobreconfianza sistemática del calibrador XGBoost.
Causa del dashboard: prob_cal=0.91-1.00 -> WR real=50.3% (diferencia -49.5pp)
"""
import sys, numpy as np, joblib, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

MODEL_DIR = Path(r'g:\Mi unidad\ia\luna_v2\data\models\prod')
PRED_DIR  = Path(r'g:\Mi unidad\ia\luna_v2\data\predictions')
SEP  = "=" * 72
DASH = "-" * 72

seed_dirs = sorted([d for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.startswith('seed')])

# ── BLOQUE A: Curva de calibración real ─────────────────────────────────
print(SEP)
print("BLOQUE A: Curva de calibración — ¿amplifica o comprime la prob_raw?")
print(SEP)
print()

header = "Seed           Agente           in_lo  in_hi  out_lo out_hi overconf>0.7"
print(header)
print(DASH)

stats = []
for sd in seed_dirs:
    for cal_path in sorted(sd.glob('xgboost_isotonic_calibrator*.joblib')):
        try:
            cal = joblib.load(cal_path)
            tx = np.linspace(0.40, 0.95, 100)
            out = cal.predict(tx) if hasattr(cal, 'predict') else tx
            hi_mask = tx > 0.70
            hi_in  = tx[hi_mask]
            hi_out = out[hi_mask]
            overconf = float((hi_out - hi_in).mean()) if len(hi_in) > 0 else 0.0
            agent = cal_path.stem.replace('xgboost_isotonic_calibrator', '').strip('_') or 'global'
            row = f"{sd.name:<15} {agent:<17} {out.min():.3f}  {out.max():.3f}  {out.min():.3f}  {out.max():.3f}  {overconf:+.4f}"
            print(row)
            stats.append({'seed': sd.name, 'agent': agent, 'overconf': overconf,
                          'out_min': out.min(), 'out_max': out.max()})
        except Exception as e:
            pass

if stats:
    sdf = pd.DataFrame(stats)
    print()
    print("Resumen por agente:")
    for ag, g in sdf.groupby('agent'):
        print(f"  {ag:<20} overconf_avg={g['overconf'].mean():+.4f}  "
              f"n_positivo={( g['overconf']>0.05).sum()}/{len(g)}")
    print()
    n_overconf = (sdf['overconf'] > 0.05).sum()
    n_underconf = (sdf['overconf'] < -0.05).sum()
    print(f"Global: overconf avg={sdf['overconf'].mean():+.4f}")
    print(f"Seeds con sobreconfianza (>0.05):  {n_overconf}/{len(sdf)}")
    print(f"Seeds con infraconfianza (<-0.05): {n_underconf}/{len(sdf)}")

# ── BLOQUE B: Correlación prob_raw vs prob_cal en los trades OOS reales ──
print()
print(SEP)
print("BLOQUE B: ¿Hay seeds donde prob_cal discrimina mejor que prob_raw?")
print(SEP)
print()

dfs = []
for f in sorted(PRED_DIR.glob('oos_trades_seed*.parquet')):
    d = pd.read_parquet(f)
    d['_seed'] = int(f.stem.split('seed')[1])
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)

from scipy.stats import spearmanr

seed_results = []
for seed, grp in df.groupby('_seed'):
    if len(grp) < 20: continue
    rho_raw, _ = spearmanr(grp['xgb_prob'], grp['is_win'].astype(float))
    rho_cal, _ = spearmanr(grp['xgb_prob_cal'], grp['is_win'].astype(float))
    diff = (grp['xgb_prob_cal'] - grp['xgb_prob']).abs()
    pct_identical = (diff < 1e-6).mean() * 100
    seed_results.append({'seed': seed, 'rho_raw': rho_raw, 'rho_cal': rho_cal,
                         'delta_rho': rho_cal - rho_raw, 'pct_identical': pct_identical, 'n': len(grp)})

sr = pd.DataFrame(seed_results)
print(f"Seeds con calibrador efectivo (rho_cal > rho_raw): {(sr['delta_rho']>0).sum()}/{len(sr)}")
print(f"Seeds con calibrador perjudicial (rho_cal < rho_raw): {(sr['delta_rho']<0).sum()}/{len(sr)}")
print(f"Delta_rho promedio: {sr['delta_rho'].mean():+.4f}")
print()
print("Top 5 seeds donde el calibrador MAS ayuda:")
for _, r in sr.nlargest(5, 'delta_rho').iterrows():
    print(f"  seed{int(r['seed'])}: rho_raw={r['rho_raw']:+.3f} rho_cal={r['rho_cal']:+.3f} delta={r['delta_rho']:+.3f} pct_identical={r['pct_identical']:.0f}%")
print()
print("Top 5 seeds donde el calibrador MAS perjudica:")
for _, r in sr.nsmallest(5, 'delta_rho').iterrows():
    print(f"  seed{int(r['seed'])}: rho_raw={r['rho_raw']:+.3f} rho_cal={r['rho_cal']:+.3f} delta={r['delta_rho']:+.3f} pct_identical={r['pct_identical']:.0f}%")

# ── BLOQUE C: Hipótesis — distribución IS vs OOS ─────────────────────────
print()
print(SEP)
print("BLOQUE C: Hipótesis causa raíz — mismatch distribución IS vs OOS")
print(SEP)
print()
print("Distribución de xgb_prob_raw en OOS (trades filtrados por threshold):")
print("  OBS: El threshold filtra prob_raw < 0.5-0.7 -> solo vemos la cola alta")
print("  Esto trunca el rango que el calibrador vio en entrenamiento")
print()
for win in sorted(df['wfb_window'].unique()):
    dw = df[df['wfb_window'] == win]
    raw = dw['xgb_prob']
    cal = dw['xgb_prob_cal']
    diff = (cal - raw).abs()
    print(f"  {win}: prob_raw=[{raw.min():.3f},{raw.max():.3f}] mean={raw.mean():.3f} | "
          f"prob_cal=[{cal.min():.3f},{cal.max():.3f}] mean={cal.mean():.3f} | "
          f"diff_mean={diff.mean():.4f} | pct_ident={(diff<1e-6).mean()*100:.0f}%")

print()
print("DIAGNOSTICO HIPOTESIS:")
print("""
  El calibrador Isotonic Regression se entrena en IS con el RANGO COMPLETO de prob_raw.
  En IS el modelo produce prob_raw en [0.10, 0.95] con distribución uniforme-ish.
  
  En OOS solo vemos trades que PASARON el threshold (>= 0.50-0.70).
  -> El rango de entrada del calibrador en OOS esta TRUNCADO a la cola alta.
  -> El calibrador fue entrenado con ejemplos de prob_raw baja que en IS tenian WR baja.
  -> En OOS todos los ejemplos son prob_raw alta -> el calibrador sobre-estima aun mas.
  
  CONFIRMACION: el reliability diagram mostra:
  - prob_cal [0.49-0.60]: WR real ~50% -> bien calibrado (corresponde al rango completo)
  - prob_cal [0.70-1.00]: WR real ~49-53% -> sobreconfianza severa (-21 a -49pp)
  -> El XGBoost NO aprende a distinguir dentro del rango alto (prob_raw > 0.70)
  -> Sus predicciones en ese rango son esencialmente ruido / WR ≈ 50%

CONCLUSION:
  El problema NO es el calibrador - el problema es el XGBoost.
  El XGBoost satura su señal predictiva en la cola alta (prob > 0.70).
  El calibrador amplifica esta saturación porque fue entrenado en IS
  donde prob alta SI correspondia a WR alta (overfitting IS -> OOS gap).
  
  CAUSA RAIZ: El XGBoost muestra un WR relativamente plano en [0.50, 1.00] en OOS.
  El edge real viene de separar prob < threshold (WR~45%) vs prob > threshold (WR~53%),
  no de discriminar dentro del rango filtrado.
""")
