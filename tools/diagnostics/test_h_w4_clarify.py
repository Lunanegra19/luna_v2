import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"g:\Mi unidad\ia\luna_v2")
RUNS = BASE / "data" / "runs"

# H-W4-REGIME: DESCARTADA pero el resultado es paradojico
# W3 prob_range=0.0086 (casi cero) PERO genera trades
# W4 prob_range=0.7151 (alto)     PERO genera 0 trades
# Esto solo tiene sentido si prob_range NO es la probabilidad del HMM de regimen
# sino la probabilidad del AGENTE XGBoost RANGE de que la barra sea ganadora
# Necesitamos entender el significado exacto de prob_bull/bear/range en raw_probs

pq_w3 = RUNS / "WFB_20260602_002420_seed42" / "seed42" / "W3" / "oos_raw_probs.parquet"
pq_w3_trades = RUNS / "WFB_20260602_002420_seed42" / "seed42" / "W3" / "oos_trades.parquet"

df_w3 = pd.read_parquet(pq_w3)
df_trades = pd.read_parquet(pq_w3_trades)

print("=== W3/seed42 oos_raw_probs.parquet ===")
print(f"Shape: {df_w3.shape}")
print(f"Index type: {type(df_w3.index)}")
print(f"Index sample: {df_w3.index[:3].tolist()}")
print()
print("Estadisticas por columna:")
print(df_w3.describe().to_string())
print()
print("Muestra (10 filas):")
print(df_w3.head(10).to_string())
print()
print("=== W3/seed42 oos_trades.parquet ===")
print(f"Columnas: {list(df_trades.columns)}")
print(df_trades.to_string())
print()

# Ver W4 candidatos
pq_w4_list = list(RUNS.glob("*/*/W4/oos_raw_probs.parquet"))
print(f"W4 raw_probs encontrados: {len(pq_w4_list)}")
if pq_w4_list:
    df_w4 = pd.read_parquet(pq_w4_list[0])
    if "timestamp" in df_w4.columns:
        df_w4 = df_w4.set_index("timestamp")
    print(f"W4 Shape: {df_w4.shape}")
    print(f"W4 Index sample: {df_w4.index[:3].tolist()}")
    print()
    print("W4 estadisticas:")
    print(df_w4.describe().to_string())
    print()
    print("W4 muestra (10 filas):")
    print(df_w4.head(10).to_string())
