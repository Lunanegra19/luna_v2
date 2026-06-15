import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

def calculate_drift(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula Z-Score Shift y Ratio de Varianza entre IS y OOS."""
    # Find common numerical features (exclude targets, metadata, etc.)
    exclude_cols = ['timestamp', 'target', 'HMM_Regime', 'HMM_Semantic', 'Target_TBM_Bin']
    features = [c for c in train_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    
    results = []
    
    for f in features:
        if f not in holdout_df.columns:
            continue
            
        train_mean = train_df[f].mean()
        train_std = train_df[f].std()
        
        if pd.isna(train_std) or train_std == 0:
            continue
            
        holdout_mean = holdout_df[f].mean()
        holdout_std = holdout_df[f].std()
        
        # Z-Score Shift: ¿Cuántas desviaciones estándar IS se ha movido la media OOS?
        z_shift = (holdout_mean - train_mean) / train_std
        
        # Variance Ratio: OOS_Var / IS_Var
        var_ratio = (holdout_std ** 2) / (train_std ** 2) if train_std > 0 else np.nan
        
        results.append({
            'Feature': f,
            'IS_Mean': train_mean,
            'OOS_Mean': holdout_mean,
            'IS_Std': train_std,
            'OOS_Std': holdout_std,
            'Z_Shift': z_shift,
            'Abs_Z_Shift': abs(z_shift),
            'Var_Ratio': var_ratio
        })
        
    df_res = pd.DataFrame(results)
    return df_res.sort_values('Abs_Z_Shift', ascending=False)

def main():
    window = "W3"
    base_path = Path(f"data/wfb_cache/{window}/features")
    
    train_path = base_path / "features_train.parquet"
    holdout_path = base_path / "features_holdout.parquet"
    
    if not train_path.exists() or not holdout_path.exists():
        logger.error(f"Faltan parquets en {base_path}")
        return
        
    logger.info(f"Cargando datos IS ({train_path.name}) y OOS ({holdout_path.name})")
    df_train = pd.read_parquet(train_path)
    df_holdout = pd.read_parquet(holdout_path)
    
    logger.info(f"IS Shape: {df_train.shape} | OOS Shape: {df_holdout.shape}")
    
    df_drift = calculate_drift(df_train, df_holdout)
    
    logger.info(f"Análisis completo. Variables analizadas: {len(df_drift)}")
    
    print("\n" + "="*80)
    print("🚨 TOP 15 TOXIC FEATURES (Mayor Covariate Shift OOS) 🚨")
    print("================================================================================")
    print(df_drift[['Feature', 'Z_Shift', 'Var_Ratio']].head(15).to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ TOP 15 STABLE FEATURES (Menor Covariate Shift OOS)")
    df_clean = df_drift.dropna(subset=['Z_Shift'])
    print(df_clean[['Feature', 'Z_Shift', 'Var_Ratio']].tail(15).to_string(index=False))

if __name__ == "__main__":
    main()
