"""check_whale_parquet.py — Verifica el estado del parquet de onchain en el VPS"""
import pandas as pd
from pathlib import Path

parquet = Path("/root/luna_v2/data/raw/onchain/onchain_raw.parquet")
if not parquet.exists():
    print("onchain_raw.parquet: NO EXISTE")
else:
    df = pd.read_parquet(parquet)
    col = "Whale_Proxy_Volume_USD"
    if col in df.columns:
        print("Whale_Proxy_Volume_USD max: " + str(df[col].max()))
        print("Whale_Proxy_Volume_USD p99.5: " + str(df[col].quantile(0.995)))
        print("Whale_Proxy_Volume_USD p0.5: " + str(df[col].quantile(0.005)))
        n_big = (df[col] > 1e9).sum()
        print("Whale_Proxy_Volume_USD filas >1e9: " + str(n_big))
    else:
        print("columna Whale_Proxy_Volume_USD: NO EXISTE")
    col2 = "Stablecoins_Delta_30d"
    if col2 in df.columns:
        print("Stablecoins_Delta_30d max: " + str(df[col2].max()))
    else:
        # Intentar encontrar columna similar
        stable_cols = [c for c in df.columns if "Stablecoin" in c or "stable" in c.lower()]
        print("Columnas Stablecoin disponibles: " + str(stable_cols))
    print("Shape onchain_raw: " + str(df.shape))
    print("Columnas: " + str(list(df.columns[:10])) + "...")
