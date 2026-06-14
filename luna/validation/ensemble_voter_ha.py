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
    # [H-A 2026-06-12] Pruning de Overfitters (Inverse Sharpe)
    import json
    import yaml
    
    try:
        with open("config/settings.yaml", "r") as f:
            cfg = yaml.safe_load(f)
            max_dsr = float(cfg.get("wfb", {}).get("ensemble_max_is_dsr", 0.0))
    except Exception as e:
        logger.warning(f"[ENSEMBLE-PRUNING] Error leyendo settings, usando default 0.0: {e}")
        max_dsr = 0.0

    wfb_cache_dir = wfb_out_dir.parent.parent / "wfb_cache"

    if active_seeds is not None:
        filtered_files = []
        pruned_count = 0
        for f in files:
            try:
                # El archivo tiene formato oos_raw_probs_W{window}_seed{seed}.parquet
                parts = f.stem.split("_seed")
                if len(parts) == 2:
                    seed = int(parts[1])
                    window = parts[0].split("_W")[-1] # Extrae el número o "W1"
                    if "W" not in window:
                        window = "W" + window
                        
                    if seed in active_seeds:
                        # Buscar firma de esta semilla/ventana
                        sig_path = wfb_cache_dir / f"seed{seed}" / window / "models" / "xgboost_meta_bull_long_signature.json"
                        dsr_is = -1.0
                        if sig_path.exists():
                            with open(sig_path, "r") as sf:
                                sdata = json.load(sf)
                                dsr_is = float(sdata.get("dsr_cpcv_best", -1.0))
                        
                        if dsr_is <= max_dsr:
                            filtered_files.append(f)
                        else:
                            pruned_count += 1
            except Exception as e:
                pass
        files = filtered_files
        print(f"[FIX-ENSEMBLE-EVAL-FILT] Probabilidades de Soft Voting filtradas: {len(files)} archivos (Podados por Overfitting: {pruned_count}).")
        logger.info(f"[FIX-ENSEMBLE-EVAL-FILT] Archivos de probabilidades filtrados: {len(files)} (Podados: {pruned_count})")

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
