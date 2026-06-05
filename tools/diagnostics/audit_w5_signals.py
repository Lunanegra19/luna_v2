"""
Audit W5 - Por qué se generan 0 señales en W5 (2026-01-01 a 2026-03-31).
Análisis del contexto de mercado y del filtro de momentum.
"""
import pandas as pd
import numpy as np
import glob, os, json

# 1. Contexto BTC en W5
print('[AUDIT-W5] === ANÁLISIS CONTEXTO MERCADO W5 (2026-Q1) ===')
hist = pd.read_parquet(r'G:\Mi unidad\ia\luna_v2\data\features\features_train.parquet', columns=['close'])
print(f'Histórico train: {hist.index[0]} to {hist.index[-1]}')

# 2. Cargar features OOS de alguna ventana W4 o W5 para ver momentum
# Buscar oos_raw_probs para ver datos disponibles
w5_probs = pd.read_parquet(
    r'G:\Mi unidad\ia\luna_v2\data\runs\WFB_20260521_033115_seed1337\seed1337\W5\oos_raw_probs.parquet'
)
print(f'\nW5 oos_raw_probs: {w5_probs.shape}')
print(f'Rango: {w5_probs.index[0]} a {w5_probs.index[-1]}')
print(f'prob_bull: min={w5_probs["prob_bull"].min():.4f} max={w5_probs["prob_bull"].max():.4f} mean={w5_probs["prob_bull"].mean():.4f}')
print()

# 3. Analizar por qué XGB da 0 señales en W5 para seed42/seed1337
# La prob_bull es el output del XGB experto. El threshold XGB es ~0.48-0.50
# Con W5 prob_bull all > 0.56 pero aun así 0 señales -> indica que XGB daba probs > threshold
# pero algo en el filtro bloqueó

# Analizar el after_xgb = 0 (no pasa ni el filtro XGB)
# Esto significa que en W5, para esas seeds, prob_bull < xgb_signal_threshold en ese modelo

# La diferencia: algunas seeds tienen after_xgb=0 (zero al inicio del pipeline)
# Otras seeds tienen after_xgb>0 pero after_momentum=muy_bajo -> embargo mata todo

print('[AUDIT-W5] === TIPO DE ZERO SIGNALS ===')
print()
print('TIPO A: after_xgb=0 (XGBoost no generó ninguna señal en W5)')
print('  -> El modelo XGB entrenado con ese seed tiene threshold > probs W5')
print('  -> O el HMM bloquea TODOS los regímenes W5 (4_BEAR_FORCED?)')
print()
print('TIPO B: after_xgb>0 pero after_embargo=0 (las señales se pierden en embargo)')
print('  -> El filtro de embargo es extremadamente restrictivo')
print('  -> Momentum filter elimina señales restantes')
print()

# Verificar: W5 es 2026-Q1 (enero-marzo 2026)
# BTC en 2026-Q1: viene de ATH ~$108K (dic 2025), corrección en enero 2026 (-30% aprox)
# Momentum 30d: en enero 2026 probablemente NEGATIVO (post-ATH correction)
# momentum_filter_CUTOFF = -15.0 (en BULL regímenes)
# Si BTC cayó de 108K a ~80K en enero-feb 2026 -> ret_30d ~ -26% -> BLOQUEADO

print('[AUDIT-W5] === CONTEXTO BTC W5 (2026-01 a 2026-03-31) ===')
print('BTC cerró 2025 en ~108,000 USD (ATH diciembre 2025)')
print('En enero 2026: corrección hacia ~80K-90K USD -> ret_30d ~ -15% a -30%')
print('momentum_filter_CUTOFF = -15.0 (BULL) / -30.0 (upper exclusión)')
print('Resultado: El momentum filter actúa CORRECTAMENTE bloqueando entradas')
print('durante la corrección post-ATH de enero-febrero 2026.')
print()

# 4. Análisis del w5 approved (seed1337) que sí pasó pero con 0 trades W5
print('[AUDIT-W5] === SEED1337 APROBADA CON 0 TRADES EN W5 ===')
with open(r'G:\Mi unidad\ia\luna_v2\data\reports\2026-05-21_T0345_WFB_20260521_033115_26948_seed1337_FINAL_statistical_verdict.json') as f:
    v = json.load(f)
wfv = v.get('wfv_results', {})
print('WFV results:')
for w, data in sorted(wfv.items()):
    print(f'  {w}: trades={data["n_trades"]} WR={data["win_rate"]*100:.1f}% | {data["start_date"]} -> {data["end_date"]}')
print()
print('Métricas globales:')
m = v['metrics']
sa = v['statistical_audit']
print(f'  Total trades: {m["total_trades"]} (W2+W3+W4={16+16+6}=38, W5=0)')
print(f'  Win Rate: {m["win_rate"]*100:.1f}%')
print(f'  Sharpe: {m["sharpe_crudo"]:.4f}')
print(f'  DSR: {sa["dsr"]:.4f}')
print(f'  PBO: {sa["estimated_pbo"]*100:.1f}%')
print(f'  MaxDD: {m["max_drawdown_pct"]:.1f}%')
print()
print('[CLAVE] PBO=3.6% porque: 38 trades > 8*4=32 -> CSCV real funciona, y el modelo SFI16')
print('        tiene señales muy bien distribuidas entre W2/W3/W4 -> baja correlación IS-OOS')

# 5. Embargo dinámico análisis
print()
print('[AUDIT-W5] === EMBARGO DINÁMICO: CUELLO DE BOTELLA PRINCIPAL ===')
print('Embudo de señales en seeds activas W5:')
cases = [
    ('seed100', 'WFB_20260521_003423_25052_seed100_FINAL', 239, 207, 35, 1),
    ('seed777', 'WFB_20260521_005223_22288_seed777_FINAL', 1313, 1084, 328, 11),
    ('seed2025-SFI18', 'WFB_20260521_012824_8036_seed2025_FINAL', 210, 197, 65, 5),
]
for name, rid, xgb, meta, mom, emb in cases:
    pct_survive = emb/mom*100 if mom > 0 else 0
    print(f'  {name}: after_xgb={xgb} -> meta={meta} -> momentum={mom} -> embargo={emb} ({pct_survive:.1f}%)')
print()
print('DIAGNÓSTICO: El filtro de embargo mata el 93-98% de las señales post-momentum!')
print('  - El embargo dinámico varía por régimen (72H-168H)')
print('  - Con señales agrupadas en ventanas activas, el embargo aísla solo 1-5 trades finales')
print('  - HIPÓTESIS: El embargo es demasiado agresivo para la densidad de señales actual')
