import pandas as pd
import numpy as np

def simulate_pipeline():
    # Simular dataframe crudo
    df = pd.DataFrame({
        "time": range(100),
        "raw_feat": np.random.randn(100)
    })
    
    # Simular la creacion temprana de lags (donde esta ahora)
    requested_lags = ["ae_feat_39_milag24h", "btc_drawdown_from_ath_milag6h"]
    
    # 1. Intento de inyeccion actual (Buggy)
    print("Intento de inyeccion en apply_derived_features()...")
    lags_fallidos = []
    for col in requested_lags:
        parts = col.split("_milag")
        src = parts[0]
        lag = int(parts[1][:-1])
        if src in df.columns:
            df[col] = df[src].shift(lag)
        else:
            lags_fallidos.append(src)
    print(f"Lags fallidos en posicion actual: {lags_fallidos}")
    
    # 2. Creacion de variables derivadas tardias
    print("Creando btc_drawdown_from_ath...")
    df["btc_drawdown_from_ath"] = np.random.randn(100)
    
    # 3. Creacion de AE (Paso 9D)
    print("Creando ae_feat_39 (AutoEncoder)...")
    df["ae_feat_39"] = np.random.randn(100)
    
    # 4. Intento de inyeccion corregido (Al final del pipeline)
    print("Intento de inyeccion al final del pipeline (Fix Propuesto)...")
    lags_exitosos = []
    for col in requested_lags:
        parts = col.split("_milag")
        src = parts[0]
        lag = int(parts[1][:-1])
        if src in df.columns:
            df[col] = df[src].shift(lag)
            lags_exitosos.append(col)
            
    print(f"Lags inyectados exitosamente: {lags_exitosos}")
    print(f"Columnas finales: {df.columns.tolist()}")

if __name__ == '__main__':
    simulate_pipeline()
