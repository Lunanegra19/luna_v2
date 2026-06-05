import pandas as pd
from pathlib import Path

ROOT = Path("g:/Mi unidad/ia/luna_v2")
WFB_DIR = ROOT / "data" / "reports" / "wfb"

for w in [3, 4]:
    raw_path = WFB_DIR / f"oos_raw_probs_W{w}_seed42.parquet"
    if not raw_path.exists():
        print(f"Ventana W{w}: No se encontró el archivo de raw probs.")
        continue
    
    df = pd.read_parquet(raw_path)
    print(f"\n==================================================")
    print(f"ANALISIS DE RAW PROBS EN VENTANA W{w}")
    print(f"==================================================")
    print(f"Total registros: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    print("\nPrimeras 5 filas:")
    print(df.head(5))
    
    # Si contiene meta_v2_prob, ver cuántos están entre 0.50 y 0.55
    meta_cols = [c for c in df.columns if "meta" in c.lower()]
    print(f"\nColumnas relacionadas con MetaLabeler: {meta_cols}")
    
    for c in meta_cols:
        print(f"\nEstadísticas de {c}:")
        print(df[c].describe())
        n_50_55 = ((df[c] >= 0.50) & (df[c] < 0.55)).sum()
        n_gte_55 = (df[c] >= 0.55).sum()
        print(f"Cantidad entre 0.50 y 0.55: {n_50_55}")
        print(f"Cantidad >= 0.55: {n_gte_55}")
