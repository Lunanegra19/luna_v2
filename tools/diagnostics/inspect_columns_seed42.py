import pandas as pd
from pathlib import Path

wfb_dir = Path("g:/Mi unidad/ia/luna_v2/data/reports/wfb")
for w in [1, 2, 3]:
    p = wfb_dir / f"oos_trades_W{w}_seed42.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        print(f"\n--- VENTANA {w} ---")
        print(f"Shape: {df.shape}")
        print("Columnas:")
        print(df.columns.tolist())
        print("Muestra de las primeras 2 filas:")
        print(df.head(2).to_string())
    else:
        print(f"No existe {p}")
