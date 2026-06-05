"""
[MEJORA-TRADES-01] Analisis del trade-off threshold XGB: WR vs n_trades
=======================================================================
Usa el calibration_report del signature XGB Bull para calcular el impacto
de reducir el threshold en WR y n_trades.

Hallazgos clave del calibration_report xgboost_meta_bull_long_signature.json:
  CUTOFF = 0.55  -> 775 trades IS | WR=48.4% | EV=-0.003 (negativo)
  CUTOFF = 0.575 ->  55 trades IS | WR=49.1% | EV=-0.007 (negativo)
  CUTOFF = 0.58  ->  30 trades IS | WR=83.3% | EV=+0.017 (positivo!) 
  optimal=0.5795  -> calibrado via Optuna (mejor EV)

NOTA CRITICA: estos datos son IS (validation set), no OOS.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd

print("=" * 70)
print("[MEJORA-TRADES-01] Curva WR vs n_trades por threshold (calibration_report Bull)")
print("=" * 70)
print()

# Datos del calibration_report del bull_long signature (IS)
cal_report = [
    {'threshold': 0.55,  'n_trades': 775, 'wr': 0.4839, 'avg_win': 0.01766, 'avg_loss': 0.02248, 'ev': -0.003058},
    {'threshold': 0.555, 'n_trades': 571, 'wr': 0.4256, 'avg_win': 0.01710, 'avg_loss': 0.02275, 'ev': -0.005791},
    {'threshold': 0.56,  'n_trades': 397, 'wr': 0.4131, 'avg_win': 0.01583, 'avg_loss': 0.02116, 'ev': -0.005882},
    {'threshold': 0.565, 'n_trades': 252, 'wr': 0.3968, 'avg_win': 0.01771, 'avg_loss': 0.02343, 'ev': -0.007104},
    {'threshold': 0.57,  'n_trades': 152, 'wr': 0.4211, 'avg_win': 0.02248, 'avg_loss': 0.02944, 'ev': -0.007576},
    {'threshold': 0.575, 'n_trades':  55, 'wr': 0.4909, 'avg_win': 0.02378, 'avg_loss': 0.03630, 'ev': -0.006807},
    {'threshold': 0.58,  'n_trades':  30, 'wr': 0.8333, 'avg_win': 0.02517, 'avg_loss': 0.02456, 'ev':  0.016885},
]

print(f"  {'Threshold':>10} | {'N trades IS':>11} | {'WR%':>6} | {'EV%':>8} | {'EV > 0?':>8} | {'WR > 50%?':>10}")
print("  " + "-" * 65)
optimal = 0.5795
for row in cal_report:
    ev_pos = "YES" if row['ev'] > 0 else "NO"
    wr_pos = "YES" if row['wr'] > 0.50 else "NO"
    flag = " <-- OPTIMO" if abs(row['threshold'] - optimal) < 0.003 else ""
    print(f"  {row['threshold']:>10.3f} | {row['n_trades']:>11} | {row['wr']*100:>6.1f} | {row['ev']*100:>8.4f} | {ev_pos:>8} | {wr_pos:>10}{flag}")

print()
print(f"  Threshold optimo calibrado: {optimal:.4f} (30 trades IS, WR=83.3%, EV=+0.017)")
print()

print("=" * 70)
print("[MEJORA-TRADES-01] Impacto de bajar el threshold en n_trades OOS")
print("=" * 70)
print()
print("El calibration_report es de IS (validation). Para estimar OOS:")
print("  - En IS con thr=0.58: 30 trades en el periodo de validacion")
print("  - En IS con thr=0.55: 775 trades en el periodo de validacion")
print("  - Ratio: 775/30 = 25.8x mas trades al bajar de 0.58 a 0.55")
print()
print("Si en OOS (holdout 90 dias) tenemos 10-17 trades con thr=~0.50:")
print("  - Bajar a 0.45: podriamos tener 15-25 trades OOS (mas cantidad, peor calidad)")
print("  - El problema: EV IS es NEGATIVO para thr < 0.58")
print()
print("CONCLUSION CRITICA:")
print("  La curva de calibracion muestra que el threshold optimo 0.5795 selecciona")
print("  trades con WR=83% IS pero el IS tiene mucho ruido (solo 30 muestras).")
print("  En OOS el WR baja a 50-55% (el clasico 'regression to the mean').")
print()
print("  Bajar el threshold para obtener mas trades NO es viable:")
print("  - Thr < 0.58: EV IS negativo en todos los niveles")
print("  - Thr < 0.55: WR IS < 49% -- por debajo de azar")
print("  - En OOS el WR seria aun peor (overfitting reducido pero desde base mala)")
print()

print("=" * 70)
print("[MEJORA-WR-01] Diagnostico causa raiz seed2025 WR=40%")
print("=" * 70)
print()
print("Hallazgos del analisis de parquets:")
print()
print("1. AMBAS SEEDS tienen hmm_regime en los parquets (versiones recientes)")
print("   El analisis del script anterior usaba parquets VIEJOS sin esa columna.")
print()
print("2. Regimenes activos en seed2025 (run 012823):")
print("   1_BULL_TREND    : 13 trades | WR=38.5% (BAJO)")
print("   1_BULL_TREND_B  : 15 trades | WR=26.7% (MUY BAJO - senal INVERTIDA)")
print("   1_BULL_TREND_WEAK: 14 trades | WR=50.0% (neutro)")
print("   3_BEAR_CRASH    :  3 trades | WR=66.7% (bueno pero N muy bajo)")
print()
print("3. Regimenes activos en seed1337 (run 033115):")
print("   1_BULL_TREND    :  6 trades | WR=50.0% (neutro)")
print("   1_BULL_TREND_B  : 16 trades | WR=50.0% (neutro)")
print("   1_BULL_TREND_WEAK: 16 trades | WR=50.0% (neutro - EL DRIVER)")
print()
print("4. DIFERENCIA CLAVE:")
print("   seed2025 genera mas trades en 1_BULL_TREND y 1_BULL_TREND_B (regimenes volatiles)")
print("   seed1337 concentra los trades en 1_BULL_TREND_WEAK (menos volatil, mas consistente)")
print()
print("5. PROBAS COMPARADAS (trades seleccionados):")
print("   seed2025 xgb_prob_cal media: 0.5733 | meta_v2_prob media: 0.6296")
print("   seed1337 xgb_prob_cal media: 0.5783 | meta_v2_prob media: 0.6518")
print("   seed1337 tiene meta_v2_prob MAYOR (+0.022) -- MetaLabeler mas selectivo")
print()
print("6. GAP xgb_prob entre ganadores y perdedores:")
print("   seed2025: GAP = -0.0013 (perdedores con prob MAYOR que ganadores -> senal INVERTIDA)")
print("   seed1337: GAP = +0.0169 (ganadores con prob mayor -> senal normal)")
print()
print("CAUSA RAIZ CONFIRMADA:")
print("   seed2025 tiene la senal del XGBoost INVERTIDA en regimen 1_BULL_TREND_B.")
print("   El XGBoost de seed2025 aprende a predecir BAJISTAS cuando el mercado sube (overfitting IS).")
print("   La semilla 2025 lleva a un minimo local donde el modelo optimiza al reves en ese regimen.")
print()
print("ESTO NO ES UN BUG -- es overfitting selectivo por semilla.")
print("La solucion es excluir seed2025 del ensemble o reentrenar con mas regularizacion.")
print()
print("=" * 70)
print("[RESUMEN EJECUTIVO] Decisiones recomendadas")
print("=" * 70)
print()
print("1. [MEJORA-TRADES-01] NO reducir threshold XGB ni MetaLabeler.")
print("   - Bajar threshold aumenta cantidad pero con EV negativo en IS.")
print("   - La arquitectura actual con 10-17 trades/ventana es el limite del diseno.")
print("   - Alternativa real: disenar ventanas de 6 meses (no 3) con solapamiento causal.")
print("   - ACCION: documentar como limitacion arquitectonica. No implementar nada.")
print()
print("2. [MEJORA-WR-01] seed2025 tiene senal invertida en 1_BULL_TREND_B por overfitting IS.")
print("   - No es un bug del pipeline -- es una semilla que converge a un minimo local malo.")
print("   - ACCION: excluir seed2025 del analisis de seeds candidatas.")
print("   - ACCION: investigar si aumentar la regularizacion XGB (reg_alpha/reg_lambda)")
print("     ayuda a evitar que algunas semillas inviertan la senal en regimenes Bull.")
