import pandas as pd
import numpy as np
from loguru import logger
import os

def add_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Integra y procesa las métricas de Order Flow Imbalance (OFI) provenientes de Bybit
    en el DataFrame principal. Calcula métricas derivadas (medias móviles, z-scores)
    para alimentar los modelos predictivos.
    
    Args:
        df: DataFrame principal (index = datetime)
        
    Returns:
        DataFrame con las nuevas columnas OFI añadidas.
    """
    ofi_path = "data/raw/orderflow/bybit_ofi_1h.parquet"
    
    if not os.path.exists(ofi_path):
        logger.warning(f"[OFI] Archivo no encontrado en {ofi_path}. Devolviendo DataFrame sin cambios.")
        return df

    logger.info("[OFI] Cargando Order Flow Imbalance data...")
    df_ofi = pd.read_parquet(ofi_path)
    
    # Asegurar que el index coincida en zona horaria (UTC)
    if df_ofi.index.tz is None:
        df_ofi.index = df_ofi.index.tz_localize('UTC')
    elif df_ofi.index.tz.zone != 'UTC':
        df_ofi.index = df_ofi.index.tz_convert('UTC')
        
    initial_cols = df.shape[1]
    
    # === INGENIERIA DE FEATURES OFI ===
    # 1. Suavizados del Imbalance Neto (Ruido de 1H es muy alto)
    df_ofi['ofi_imb_ema_4h'] = df_ofi['ofi_imbalance_1h'].ewm(span=4, adjust=False).mean()
    df_ofi['ofi_imb_ema_24h'] = df_ofi['ofi_imbalance_1h'].ewm(span=24, adjust=False).mean()
    
    # 2. Suavizados del Imbalance Institucional (Whales / Large orders)
    df_ofi['ofi_large_imb_ema_4h'] = df_ofi['ofi_large_imbalance'].ewm(span=4, adjust=False).mean()
    df_ofi['ofi_large_imb_ema_24h'] = df_ofi['ofi_large_imbalance'].ewm(span=24, adjust=False).mean()
    
    # 3. Ratio Retail vs Institucional (¿Están los grandes comprando mientras los pequeños venden?)
    df_ofi['ofi_divergence_inst_retail'] = df_ofi['ofi_large_imbalance'] - df_ofi['ofi_imbalance_1h']
    
    # 4. Intensidad Estructural (Volumen direccionalizado)
    df_ofi['ofi_buy_pressure_ratio'] = df_ofi['ofi_buy_vol_1h'] / (df_ofi['ofi_buy_vol_1h'] + df_ofi['ofi_sell_vol_1h'] + 1e-9)
    df_ofi['ofi_buy_pressure_ema_24h'] = df_ofi['ofi_buy_pressure_ratio'].ewm(span=24, adjust=False).mean()
    
    # 5. Delta Imbalance (Aceleración del Order Flow)
    df_ofi['ofi_imb_delta_1h'] = df_ofi['ofi_imbalance_1h'].diff()
    df_ofi['ofi_imb_delta_4h'] = df_ofi['ofi_imbalance_1h'].diff(4)
    
    # === PREPARACIÓN DE JOIN ===
    # Seleccionar columnas a incluir
    features_to_keep = [
        'ofi_imbalance_1h', 'ofi_imb_ema_4h', 'ofi_imb_ema_24h',
        'ofi_large_imbalance', 'ofi_large_imb_ema_4h', 'ofi_large_imb_ema_24h',
        'ofi_divergence_inst_retail', 'ofi_buy_pressure_ratio', 'ofi_buy_pressure_ema_24h',
        'ofi_imb_delta_1h', 'ofi_imb_delta_4h', 'ofi_avg_trade_btc'
    ]
    
    df_ofi_subset = df_ofi[features_to_keep].copy()
    
    # IMPORTANTE (Regla V10 R1 - Causalidad Estricta): 
    # Los datos de Order Flow representan el volumen durante la vela.
    # Deben desplazarse 1 periodo (shift 1) para evitar leakage.
    df_ofi_shifted = df_ofi_subset.shift(1)
    
    # Unir con el dataframe principal
    df = df.join(df_ofi_shifted, how='left')
    
    # Fill NaN para el periodo anterior a 2020 (donde no hay Order Flow de Bybit)
    # Rellenamos con 0 (estado neutro) para que los árboles no se vuelvan inestables
    df[features_to_keep] = df[features_to_keep].fillna(0)
    
    added_cols = df.shape[1] - initial_cols
    logger.info(f"[OFI] Integradas {added_cols} features de Order Flow exitosamente.")
    
    return df
