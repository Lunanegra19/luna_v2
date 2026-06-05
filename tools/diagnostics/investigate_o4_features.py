"""
[INVESTIGACION-O4] Investiga las 4 features ausentes en el pipeline live:
- DXY_z90d_milag96h
- Whale_Proxy_Volume_USD_milag500h
- Stablecoins_Delta_30d_milag12h
- CPI_YoY_kz_milag48h

Y las features base que las originan en el parquet.
"""
import pandas as pd
import sys

PARQUET = "/root/luna_v2/data/features/features_live.parquet"
TARGET_FEATURES = [
    "DXY_z90d_milag96h",
    "Whale_Proxy_Volume_USD_milag500h",
    "Stablecoins_Delta_30d_milag12h",
    "CPI_YoY_kz_milag48h",
]
BASE_FEATURES = [
    "DXY_z90d", "DXY", "DXY_Slope30d_z90d",
    "Whale_Proxy_Volume_USD", "Whale_Vol_30d_MA",
    "Stablecoins_Delta_30d", "Stablecoins_30d_MA",
    "CPI_YoY_kz", "CPI_YoY",
]

print("=" * 70)
print("[INVESTIGACION-O4] Features ausentes en pipeline live")
print("=" * 70)

df = pd.read_parquet(PARQUET)
print(f"\nTotal columnas en features_live.parquet: {len(df.columns)}")
print(f"Filas: {len(df)}")
try:
    print(f"Fecha máxima: {df.index.max()}")
except:
    pass

print("\n--- PRESENCIA DE LAS 4 FEATURES AUSENTES ---")
for feat in TARGET_FEATURES:
    present = feat in df.columns
    if present:
        last = df[feat].dropna().iloc[-1] if not df[feat].dropna().empty else None
        nan_pct = df[feat].isna().mean() * 100
        print(f"  {'✅' if present else '❌'} {feat} | last={last:.6f} | NaN={nan_pct:.1f}%")
    else:
        print(f"  ❌ {feat} — NO PRESENTE en el parquet live")

print("\n--- PRESENCIA DE LAS FEATURES BASE (sin milag) ---")
for feat in BASE_FEATURES:
    present = feat in df.columns
    if present:
        last = df[feat].dropna().iloc[-1] if not df[feat].dropna().empty else None
        nan_pct = df[feat].isna().mean() * 100
        print(f"  ✅ {feat} | last={last:.6f} | NaN={nan_pct:.1f}%")
    else:
        print(f"  ❌ {feat} — NO PRESENTE")

print("\n--- BÚSQUEDA PARCIAL (columnas que contienen keywords) ---")
keywords = ["DXY", "Whale", "Stablecoin", "CPI"]
for kw in keywords:
    matches = [c for c in df.columns if kw.lower() in c.lower()]
    print(f"  '{kw}': {matches[:10]}")

print("\n--- VERIFICACIÓN: ¿SE PUEDE CONSTRUIR 'DXY_z90d_milag96h' DESDE EL PARQUET? ---")
# DXY_z90d_milag96h = DXY_z90d shifteado 96 barras (horas)
if "DXY_z90d" in df.columns:
    candidate = df["DXY_z90d"].shift(96)
    last_val = candidate.dropna().iloc[-1] if not candidate.dropna().empty else None
    print(f"  DXY_z90d existe → milag96h = shift(96) → último valor: {last_val}")
else:
    print("  DXY_z90d NO existe → no se puede construir")

# CPI_YoY_kz_milag48h = CPI_YoY_kz shifteado 48 barras
if "CPI_YoY_kz" in df.columns:
    candidate = df["CPI_YoY_kz"].shift(48)
    last_val = candidate.dropna().iloc[-1] if not candidate.dropna().empty else None
    print(f"  CPI_YoY_kz existe → milag48h = shift(48) → último valor: {last_val}")
else:
    print("  CPI_YoY_kz NO existe → no se puede construir")

# Whale_Proxy_Volume_USD_milag500h
if "Whale_Proxy_Volume_USD" in df.columns:
    candidate = df["Whale_Proxy_Volume_USD"].shift(500)
    last_val = candidate.dropna().iloc[-1] if not candidate.dropna().empty else None
    print(f"  Whale_Proxy_Volume_USD existe → milag500h = shift(500) → último valor: {last_val}")
else:
    print("  Whale_Proxy_Volume_USD NO existe → no se puede construir")

# Stablecoins_Delta_30d_milag12h
if "Stablecoins_Delta_30d" in df.columns:
    candidate = df["Stablecoins_Delta_30d"].shift(12)
    last_val = candidate.dropna().iloc[-1] if not candidate.dropna().empty else None
    print(f"  Stablecoins_Delta_30d existe → milag12h = shift(12) → último valor: {last_val}")
else:
    print("  Stablecoins_Delta_30d NO existe → no se puede construir")

print("\n" + "=" * 70)
print("[INVESTIGACION-O4] COMPLETADO")
print("=" * 70)
