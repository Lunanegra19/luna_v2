import pandas as pd
import numpy as np
from loguru import logger
from luna.features.tbm import apply_triple_barrier

def calculate_online_threshold(
    df_recent: pd.DataFrame, 
    meta_probs: np.ndarray, 
    pt_mult: float, 
    sl_mult: float, 
    tbm_min_return: float, 
    vb_max_h: int, 
    vb_min_h: int,
    cost_pct_meta: float = None  # [FIX-A] None para forzar lectura desde cfg
) -> tuple[float, float]:
    """
    Realiza un EV-Sweep sobre una ventana de datos recientes para encontrar el umbral óptimo.
    
    Args:
        df_recent: DataFrame con columnas 'close' (y eventos en el índice temporal).
        meta_probs: Array 1D con las predicciones del MetaLabelerV2 alineadas a df_recent.
        pt_mult: Multiplicador Take Profit.
        sl_mult: Multiplicador Stop Loss.
        tbm_min_return: Retorno mínimo para TBM.
        vb_max_h: Horizonte máximo TBM.
        vb_min_h: Horizonte mínimo dinámico TBM.
        cost_pct_meta: Costo de transacción.
        
    Returns:
        tuple: (best_threshold, best_ev)
    """
    try:
        from config.settings import cfg as _cfg
        _lin_decay = bool(_cfg.xgboost.linear_decay_pt)
        _pt_decay_frac = float(_cfg.xgboost.pt_decay_fraction)
        # [FIX-A] leer cost_pct de cfg si no fue pasado como argumento
        if cost_pct_meta is None:
            cost_pct_meta = float(_cfg.sop.cost_pct)
        # [FIX-A] n_target desde metalabeler.meta_min_trades (ya existe en settings)
        _n_target_cfg = int(_cfg.metalabeler.meta_min_trades)
    except Exception as e:
        if "cost_pct" in str(e) or cost_pct_meta is None:
            raise RuntimeError(f"Falta cfg.sop.cost_pct en settings.yaml. Política No-Fallback (SOP R6): {e}")
        _lin_decay = False
        _pt_decay_frac = 0.75
        _n_target_cfg = 30  # fallback documentado
        print(f"[FIX-A] WARN: No se pudo leer meta_min_trades de cfg. n_target fallback={_n_target_cfg}")

    # 1. Aplicar TBM
    tbm = apply_triple_barrier(
        price_series=df_recent["close"],
        event_times=df_recent.index,
        pt_sl_multiplier=[pt_mult, sl_mult],
        min_return=tbm_min_return,
        vertical_barrier_hours=vb_max_h,
        dynamic_barrier=True,
        dynamic_horizon_min_h=vb_min_h,
        dynamic_horizon_max_h=vb_max_h,
        linear_decay_pt=_lin_decay,
        pt_decay_fraction=_pt_decay_frac,
    )
    
    # Alinear meta_probs ANTES del join para evitar crash de Pandas por indices desalineados (Length of values != length of index)
    df_recent = df_recent.copy()
    if len(meta_probs) != len(df_recent):
        logger.warning(f"Longitud meta_probs ({len(meta_probs)}) != df_recent ({len(df_recent)}). Truncando...")
        df_recent["meta_prob"] = meta_probs[:len(df_recent)]
    else:
        df_recent["meta_prob"] = meta_probs
        
    df_tbm = df_recent.join(tbm[["bin", "ret"]], how="inner")
    df_tbm["target"] = (df_tbm["bin"] == 1).astype(int)
    
    df_tbm = df_tbm.dropna(subset=["target", "ret", "meta_prob"])
    
    if len(df_tbm) < 50:
        logger.warning("Velas insuficientes tras TBM en la ventana de recalibración (< 50).")
        return 0.50, -1.0

    y_seq = df_tbm["target"].values
    cal_probs = df_tbm["meta_prob"].values

    # 2. EV Sweep
    mt_min, mt_max, mt_step = 0.25, 0.65, 0.01
    best_ev = -np.inf
    best_t = 0.50
    # [FIX-A] n_target leído de metalabeler.meta_min_trades en settings.yaml (antes: 30 hardcodeado)
    n_target = _n_target_cfg
    print(f"[FIX-A] online_recalibrator: n_target={n_target} (meta_min_trades desde cfg)")
    
    for mt in np.arange(mt_min, mt_max + mt_step, mt_step):
        mask = cal_probs > mt
        n_trades = int(mask.sum())
        if n_trades < 5: 
            continue
            
        wins = int(y_seq[mask].sum())
        
        # FIX: Calcular el Expected Value (EV) real usando los retornos empíricos de la barrera
        # dinámica en lugar de un proxy estático (tbm_min_return).
        selected_returns = df_tbm["ret"].values[mask]
        ev = float(selected_returns.mean()) - cost_pct_meta
        score = ev * min(1.0, n_trades / n_target)
        
        if ev > best_ev and score > best_ev:
            best_ev = ev
            best_t = float(mt)
            
    return best_t, best_ev
