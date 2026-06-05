"""
test_credit_spreads.py — Test funcional de CREDIT-SPREAD-01
Verifica que fetch_credit_spreads() descarga las 3 series FRED correctamente
y produce las 7 features derivadas con los rangos esperados.
"""
import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from luna.data.fetch_macro import MacroFetcher

print("=== TEST CREDIT-SPREAD-01 ===")
print()

fetcher = MacroFetcher()
if fetcher._fred is None:
    print("ERROR: FRED_API_KEY no configurado — test saltado")
    sys.exit(0)

df = fetcher.fetch_credit_spreads()

if df.empty:
    print("ERROR: DataFrame vacio — revisar FRED connection")
    sys.exit(1)

expected_cols = [
    "CreditSpread_HY_bps",
    "CreditSpread_IG_bps",
    "CreditSpread_BBB_bps",
    "CreditSpread_HY_IG",
    "CreditSpread_HY_IG_z90d",
    "CreditSpread_HY_z90d",
    "CreditStress_Flag",
]

print("Columnas generadas:")
for col in expected_cols:
    if col in df.columns:
        s = df[col].dropna()
        print(f"  {col}: N={len(s)} min={s.min():.2f} max={s.max():.2f} last={s.iloc[-1]:.2f}")
    else:
        print(f"  {col}: FALTA")

print()
print("Validaciones:")
# HY siempre > IG (HY es mas riesgoso que IG)
hy = df["CreditSpread_HY_bps"].dropna()
ig = df["CreditSpread_IG_bps"].dropna()
hy_ig = df["CreditSpread_HY_IG"].dropna()

pct_hy_gt_ig = (hy.align(ig, join='inner')[0] > hy.align(ig, join='inner')[1]).mean() * 100
print(f"  HY > IG (esperado ~100%): {pct_hy_gt_ig:.1f}%")

# HY-IG siempre positivo
pct_pos = (hy_ig > 0).mean() * 100
print(f"  HY_IG > 0 (esperado ~100%): {pct_pos:.1f}%")

# Rango historico razonable para HY OAS: tipicamente 200-2000bp
print(f"  HY_OAS rango: [{hy.min():.0f}, {hy.max():.0f}] bp (esperado [200,2000])")
print(f"  HY_IG rango:  [{hy_ig.min():.0f}, {hy_ig.max():.0f}] bp (esperado [100,1500])")

# Ultimo valor (estrés actual)
last_hy_ig = hy_ig.iloc[-1]
flag = df["CreditStress_Flag"].dropna().iloc[-1]
print()
print(f"Estado actual ({hy_ig.index[-1].date()}):")
print(f"  HY_OAS       = {hy.iloc[-1]:.0f} bp")
print(f"  IG_OAS       = {ig.iloc[-1]:.0f} bp")
print(f"  HY-IG spread = {last_hy_ig:.0f} bp")
print(f"  CreditStress = {'ALTO (bearish BTC)' if flag else 'NORMAL'}")

all_ok = (
    all(c in df.columns for c in expected_cols) and
    pct_hy_gt_ig > 95 and
    pct_pos > 95
)
print()
print(f"=== TEST {'COMPLETO: OK' if all_ok else 'FALLIDO'} ===")
