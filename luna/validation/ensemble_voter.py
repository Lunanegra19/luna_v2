"""
Ensemble Voter para Luna V1 WFB

Este script combina las probabilidades marginales de los modelos base XGBoost OOS 
generados por cada Seed del archivo `oos_raw_probs_W*_seed*.parquet`.
Aplica Soft Voting para mitigar la inanición operativa y la inestabilidad de
los modelos altamente restringidos.
"""
from pathlib import Path
import pandas as pd
from loguru import logger

def aggregate_wfb_seeds(wfb_out_dir: Path, out_path: Path, active_seeds: list = None):
    """
    Busca todos los archivos `oos_raw_probs_*.parquet`, agrupa por Timestamp (entry_time),
    hace un Promedio Ponderado (Soft Voting) de la clase predominante a través
    de las semillas activas configuradas.
    """
    files = list(wfb_out_dir.glob("oos_raw_probs_W*_seed*.parquet"))
    if not files:
        logger.warning(f"No se encontraron archivos raw_probs en {wfb_out_dir}. El ensamble no generará resultados.")
        return None

    # [FIX-ENSEMBLE-EVAL-FILT] Filtrar únicamente semillas configuradas en active_seeds
    if active_seeds is not None:
        filtered_files = []
        for f in files:
            try:
                parts = f.stem.split("_seed")
                if len(parts) == 2:
                    seed = int(parts[1])
                    if seed in active_seeds:
                        filtered_files.append(f)
            except Exception:
                pass
        files = filtered_files
        print(f"[FIX-ENSEMBLE-EVAL-FILT] Probabilidades de Soft Voting filtradas para semillas activas {active_seeds}: {len(files)} archivos.")
        logger.info(f"[FIX-ENSEMBLE-EVAL-FILT] Archivos de probabilidades filtrados: {len(files)}")

    logger.info(f"Procesando {len(files)} sub-arqueros OOS para ensamble...")
    
    df_list = []
    for f in files:
        df_sub = pd.read_parquet(f)
        if not df_sub.empty:
            # Asumimos estructura: timestamp, prob_bull, prob_bear, prob_range
            if 'timestamp' in df_sub.columns:
                df_sub = df_sub.set_index('timestamp')
            df_list.append(df_sub)
            
    if not df_list:
        logger.warning("Todos los parquets OOS raw estaban vacíos. Ensamble falló.")
        return None
        
    df_merged = pd.concat(df_list)
    
    # Agrupar colisiones por Timestamp tomando la media de las probabilidades inter-semillas
    df_ensemble = df_merged.groupby(df_merged.index).mean()
    
    logger.info(f"Ensamble completado: {len(df_ensemble)} instantes horarios agregados.")
    
    # Derivar predicciones duras del Soft Voting
    prob_cols = [c for c in df_ensemble.columns if "prob" in c.lower()]
    if prob_cols:
        logger.info(f"Guardando Dataframe Maestro del Ensamble Ponderado...")
        df_ensemble.to_parquet(out_path)
    
    return df_ensemble

if __name__ == "__main__":
    _root = Path(r"G:\Mi unidad\ia\Luna v1")
    _wfb_dir = _root / "data" / "wfb_ensemble_tests"
    _out = _wfb_dir / "master_ensemble_probs.parquet"
    if _wfb_dir.exists():
        aggregate_wfb_seeds(_wfb_dir, _out)
