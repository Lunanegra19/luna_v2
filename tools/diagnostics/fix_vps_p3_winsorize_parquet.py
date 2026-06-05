"""
fix_vps_p3_winsorize_parquet.py
[FIX-VPS-P3-OVERFLOW 2026-05-30] Winsoriza features con overflow en el parquet onchain_raw.
Aplica p0.5-p99.5 clipping directamente al parquet persistido para evitar el clip de emergencia
en ensemble_live_inference.py en cada ciclo de inferencia.

FEATURES A CORREGIR:
- Whale_Proxy_Volume_USD: max=1.054e+15 (deberia estar en rango 1e9-1e12)
  Causa: volume_contrato * precio puede producir valores absurdos en datos historicos.
  Fix: clip al p0.5-p99.5 de la serie completa (95th pct ~ 1e11, razonable).
  
- Stablecoin_Cap: verificar si tiene overflow similar
"""
import pandas as pd
import shutil
from pathlib import Path

PARQUET = Path("/root/luna_v2/data/raw/onchain/onchain_raw.parquet")
print("[FIX-VPS-P3] === INICIO Winsorización Parquet Onchain ===")

# Backup
bak = PARQUET.with_suffix(".parquet.bak_p3")
shutil.copy2(PARQUET, bak)
print("[FIX-VPS-P3] Backup: " + str(bak))

df = pd.read_parquet(PARQUET)
print("[FIX-VPS-P3] Shape original: " + str(df.shape))

features_to_check = [
    "Whale_Proxy_Volume_USD",
    "Stablecoin_Cap",
    "Whale_Vol_30d_MA",
    "Whale_Vol_30d_Std",
]

for col in features_to_check:
    if col not in df.columns:
        print("[FIX-VPS-P3] " + col + ": no existe en parquet, skipping")
        continue
    
    original_max = df[col].max()
    original_min = df[col].min()
    p005 = df[col].quantile(0.005)
    p995 = df[col].quantile(0.995)
    n_above = (df[col] > p995).sum()
    n_below = (df[col] < p005).sum()
    
    print("[FIX-VPS-P3] " + col + ":")
    print("  max=" + str(original_max) + " min=" + str(original_min))
    print("  p0.5=" + str(p005) + " p99.5=" + str(p995))
    print("  valores_por_encima_p99.5=" + str(n_above) + " valores_por_debajo_p0.5=" + str(n_below))
    
    # Solo winsorizar si hay outliers significativos (max > 10x p99.5)
    if original_max > p995 * 10 and n_above > 0:
        df[col] = df[col].clip(lower=p005, upper=p995)
        new_max = df[col].max()
        print("  [FIX] Winsorizado. Nuevo max=" + str(new_max))
        # Recalcular MA y Std si existen las columnas derivadas
        if col == "Whale_Proxy_Volume_USD":
            if "Whale_Vol_30d_MA" in df.columns:
                df["Whale_Vol_30d_MA"] = df["Whale_Proxy_Volume_USD"].rolling(30).mean()
                print("  [FIX] Whale_Vol_30d_MA recalculada")
            if "Whale_Vol_30d_Std" in df.columns:
                df["Whale_Vol_30d_Std"] = df["Whale_Proxy_Volume_USD"].rolling(30).std()
                print("  [FIX] Whale_Vol_30d_Std recalculada")
            if "Whale_Vol_ZScore" in df.columns:
                ma = df["Whale_Proxy_Volume_USD"].rolling(30).mean()
                std = df["Whale_Proxy_Volume_USD"].rolling(30).std()
                df["Whale_Vol_ZScore"] = (df["Whale_Proxy_Volume_USD"] - ma) / (std + 1e-8)
                print("  [FIX] Whale_Vol_ZScore recalculado")
            if "Whale_Activity_Flag" in df.columns:
                df["Whale_Activity_Flag"] = (df["Whale_Vol_ZScore"] > 2.0).astype(int)
                print("  [FIX] Whale_Activity_Flag recalculado")
    else:
        print("  [OK] Dentro del rango esperado, no se requiere winsorización")

# Guardar parquet corregido
df.to_parquet(PARQUET)
print("\n[FIX-VPS-P3] Parquet guardado: " + str(PARQUET))

# Verificar resultado
df_verify = pd.read_parquet(PARQUET)
if "Whale_Proxy_Volume_USD" in df_verify.columns:
    final_max = df_verify["Whale_Proxy_Volume_USD"].max()
    print("[FIX-VPS-P3] VERIFICACION Whale_Proxy_Volume_USD max FINAL: " + str(final_max))
    if final_max < 1e12:
        print("[FIX-VPS-P3] OK: max < 1e12, overflow corregido.")
    else:
        print("[FIX-VPS-P3] WARN: max aun alto: " + str(final_max))

print("[FIX-VPS-P3] === FIX COMPLETADO ===")
