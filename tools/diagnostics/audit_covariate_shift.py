import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def analyze_covariate_shift(train_path: Path, top_n: int = 20):
    """
    Analiza el Covariate Shift en el dataset de entrenamiento
    usando la misma logica del Adversarial Z-Filter (Fase 0).
    """
    if not train_path.exists():
        print(f"Error: No se encontro el archivo {train_path}")
        return
        
    print(f"--- Auditando Covariate Shift en {train_path.name} ---")
    df = pd.read_parquet(train_path)
    
    mid = len(df) // 2
    df_first = df.iloc[:mid]
    df_second = df.iloc[mid:]
    
    shifts = []
    
    for col in df.columns:
        if col in ['timestamp', 'target', 'close'] or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        s1 = df_first[col]
        s2 = df_second[col]
        
        mu_1, std_1 = s1.mean(), s1.std()
        mu_2 = s2.mean()
        
        if pd.isna(mu_1) or pd.isna(mu_2) or pd.isna(std_1) or std_1 == 0:
            continue
            
        z_shift = abs((mu_2 - mu_1) / std_1)
        shifts.append((col, z_shift))
        
    shifts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} features mas inestables (Z-Shift Extremo > 2.0 es peligroso):")
    for i, (col, z) in enumerate(shifts[:top_n]):
        status = "❌ PURGADA" if z > 2.0 else "✅ SEGURA"
        print(f"{i+1:02d}. {col:30s} Z-Shift: {z:5.2f} [{status}]")
        
    n_purged = sum(1 for _, z in shifts if z > 2.0)
    print(f"\nResumen: {n_purged} variables serian purgadas por el SFI Fase 0.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditoria de Covariate Shift")
    parser.add_argument("--window", type=str, default="W3", help="ID de la ventana WFB (ej. W3)")
    args = parser.parse_args()
    
    wfb_cache = Path("data/wfb_cache") / args.window / "features"
    train_file = wfb_cache / "features_train.parquet"
    
    analyze_covariate_shift(train_file)
