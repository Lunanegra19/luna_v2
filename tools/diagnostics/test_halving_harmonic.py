"""
test_halving_harmonic.py — Test funcional de HALVING-HARMONIC-01
"""
import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from luna.features.calendar_features import CalendarFeatures

print("=== TEST HALVING-HARMONIC-01 ===")
print()

# Test 1: Rango historico completo (2020-2026)
idx = pd.date_range("2020-01-01", "2026-06-03", freq="1D", tz="UTC")
df = pd.DataFrame({"close": 50000.0}, index=idx)
cf = CalendarFeatures()
result = cf.transform(df)

new_feats = ["cal_halving_cycle_sin", "cal_halving_cycle_cos", "cal_days_to_next_halving"]
existing  = ["cal_days_since_halving", "cal_halving_cycle_pct"]
all_feats = existing + new_feats

print("Features generadas:")
for f in all_feats:
    if f in result.columns:
        mn = result[f].min()
        mx = result[f].max()
        nu = result[f].isnull().sum()
        print(f"  {f}: min={mn:.3f} max={mx:.3f} nulls={nu}")
    else:
        print(f"  {f}: FALTA EN RESULTADO")

print()

# Test 2: Continuidad en el rollover del halving 2024-04-20
pre  = result.loc["2024-04-19"]
post = result.loc["2024-04-21"]
pct_jump = abs(post["cal_halving_cycle_pct"] - pre["cal_halving_cycle_pct"])
sin_jump = abs(post["cal_halving_cycle_sin"] - pre["cal_halving_cycle_sin"])
cos_jump = abs(post["cal_halving_cycle_cos"] - pre["cal_halving_cycle_cos"])
print("Test continuidad en halving 2024-04-20:")
print(f"  pct antes={pre['cal_halving_cycle_pct']:.4f}  despues={post['cal_halving_cycle_pct']:.4f}  salto={pct_jump:.4f}  (esperado GRANDE)")
print(f"  sin antes={pre['cal_halving_cycle_sin']:.4f}  despues={post['cal_halving_cycle_sin']:.4f}  salto={sin_jump:.4f}  (esperado pequeno)")
print(f"  cos antes={pre['cal_halving_cycle_cos']:.4f}  despues={post['cal_halving_cycle_cos']:.4f}  salto={cos_jump:.4f}  (esperado pequeno)")
ok_sin = sin_jump < 0.10
ok_cos = cos_jump < 0.10
print(f"  Continuidad sin/cos: {'OK' if (ok_sin and ok_cos) else 'FALLO'}")

print()

# Test 3: Valores actuales (posicion en el ciclo)
today = result.iloc[-1]
print("Valores actuales (2026-06-03 aprox):")
for f in all_feats:
    print(f"  {f} = {today[f]:.4f}")
print()
print(f"  Interpretacion: dia {today['cal_days_since_halving']:.0f} del ciclo (halving 2024-04-20)")
print(f"  {today['cal_halving_cycle_pct']*100:.1f}% del ciclo completado")
print(f"  Faltan {today['cal_days_to_next_halving']:.0f} dias para el proximo halving (~2028-03-15)")

print()
print("=== TEST COMPLETO: OK ===" if all(f in result.columns for f in all_feats) else "=== TEST FALLIDO ===")
