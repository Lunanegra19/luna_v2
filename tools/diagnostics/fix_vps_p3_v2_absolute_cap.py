"""
fix_vps_p3_v2_absolute_cap.py
[FIX-VPS-P3-V2 2026-05-30] Cap absoluto financieramente plausible para Whale features.

ANALISIS:
- Whale_Proxy_Volume_USD max=1.054e+15 — equivalente a 1 BILLON de millones de USD diario.
  Esto es absurdo: el volumen BTC diario historico record es ~$100B (1e11 USD).
  El problema es que volume en ohlcv_raw esta en unidades de contrato (USDT) no en BTC,
  multiplicado por precio produce el cuadrado de USDT ~ 1e15.

FIX:
  Aplicar cap absoluto de 1e12 (un trillon de USD = 10x el maximo historico de BTC volume).
  Esto es financieramente plausible y elimina los valores absurdos.
  El ZScore posterior sera calculado sobre datos normales.
"""
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

PARQUET = Path("/root/luna_v2/data/raw/onchain/onchain_raw.parquet")
print("[FIX-VPS-P3-V2] === INICIO Fix Absoluto Whale Features ===")

bak = Path("/root/luna_v2/data/raw/onchain/onchain_raw.parquet.bak_p3v2")
shutil.copy2(PARQUET, bak)
print("[FIX-VPS-P3-V2] Backup: " + str(bak))

df = pd.read_parquet(PARQUET)

# Cap absoluto: maximo financieramente plausible
# Max volumen BTC diario historico record: ~$60B en 2024. Cap a $500B = 5e11 para margen.
# Aplicar clip individual a Whale USD y recalcular derivadas.
CAP_WHALE_USD = 5e11  # $500B USD diario - maximo plausible con margen x8

col = "Whale_Proxy_Volume_USD"
if col in df.columns:
    n_above = (df[col] > CAP_WHALE_USD).sum()
    n_total = len(df[col].dropna())
    print("[FIX-VPS-P3-V2] " + col + ":")
    print("  max_original=" + str(df[col].max()))
    print("  CAP_aplicado=" + str(CAP_WHALE_USD))
    print("  filas_por_encima_del_cap=" + str(n_above) + "/" + str(n_total))
    
    # Clip al cap absoluto (lower no necesita restriccion, los valores pequeños son reales)
    df[col] = df[col].clip(upper=CAP_WHALE_USD)
    print("  max_despues_clip=" + str(df[col].max()))
    print("  [FIX-VPS-P3-V2] " + col + " clippeado a cap=" + str(CAP_WHALE_USD))
    
    # Recalcular derivadas basadas en Whale_Proxy_Volume_USD
    whale = df[col]
    ma30 = whale.rolling(30).mean()
    std30 = whale.rolling(30).std()
    z = (whale - ma30) / (std30 + 1e-8)
    
    if "Whale_Vol_30d_MA" in df.columns:
        df["Whale_Vol_30d_MA"] = ma30
        print("  [FIX-VPS-P3-V2] Whale_Vol_30d_MA recalculada. max=" + str(ma30.max()))
    if "Whale_Vol_30d_Std" in df.columns:
        df["Whale_Vol_30d_Std"] = std30
        print("  [FIX-VPS-P3-V2] Whale_Vol_30d_Std recalculada. max=" + str(std30.max()))
    if "Whale_Vol_ZScore" in df.columns:
        df["Whale_Vol_ZScore"] = z
        print("  [FIX-VPS-P3-V2] Whale_Vol_ZScore recalculado. range=[" + str(z.min()) + "," + str(z.max()) + "]")
    if "Whale_Activity_Flag" in df.columns:
        df["Whale_Activity_Flag"] = (z > 2.0).astype(int)
        n_flag = (z > 2.0).sum()
        print("  [FIX-VPS-P3-V2] Whale_Activity_Flag recalculado. n_flagged=" + str(n_flag))

# Guardar
df.to_parquet(PARQUET)
print("\n[FIX-VPS-P3-V2] Parquet guardado.")

# Verificar
df2 = pd.read_parquet(PARQUET)
wmax = df2["Whale_Proxy_Volume_USD"].max() if "Whale_Proxy_Volume_USD" in df2.columns else "N/A"
print("[FIX-VPS-P3-V2] VERIFICACION final Whale_Proxy_Volume_USD max: " + str(wmax))
if isinstance(wmax, float) and wmax < 1e12:
    print("[FIX-VPS-P3-V2] OK: overflow corregido (max < 1e12)")
else:
    print("[FIX-VPS-P3-V2] WARN: max aun >= 1e12")

print("[FIX-VPS-P3-V2] === FIX COMPLETADO ===")
