"""
Custom Objective Function - Monetary Loss
===================================================
A diferencia del log-loss estándar (binary:logistic) que trata a los Falsos Positivos
y Falsos Negativos con igual severidad, el Monetary Loss penaliza el modelo
base según las métricas asimétricas de rentabilidad del Triple Barrier Method.

Falso Positivo (FP): Entra al trade (pred=1) y pierde (y=0) -> Costo = SL + Comisiones
Falso Negativo (FN): No entra (pred=0) y gana (y=1) -> Costo de Oportunidad = PT - Comisiones

Se inyecta calculando el gradiente y hessiano analíticos ajustados por factores de asimetría.
"""

import numpy as np

def get_monetary_pnl_loss():
    """
    Fábrica que genera la función objetivo leyendo los costos operativos
    desde la configuración (settings.yaml).
    """
    try:
        from config.settings import cfg
        pt_mult = float(cfg.xgboost.pt_mult_min)
        sl_mult = float(cfg.xgboost.sl_mult_min)
        min_return = float(cfg.xgboost.tbm_min_return)
        cost_pct = float(cfg.sop.cost_pct)
    except Exception as e:
        raise RuntimeError(f"Falta cfg.sop.cost_pct en settings.yaml. Política No-Fallback (SOP R6): {e}")
        
    # Calcular el valor esperado (Expected Value) asimétrico de los trades
    reward = (pt_mult * min_return) - cost_pct
    risk = (sl_mult * min_return) + cost_pct
    
    # Normalizar riesgo-recompensa para que sirvan de gamma (multiplicador de error)
    penalty_fp = risk / min_return       # ~1.3
    penalty_fn = reward / min_return     # ~1.3
    
    # Extra Asimmetry factor: FPs directly damage equity curve. 
    # FNs only damage opportunity. PnL dictates avoiding DD.
    risk_aversion = 1.5 
    gamma_fp = penalty_fp * risk_aversion
    gamma_fn = penalty_fn

    def monetary_pnl_loss(y_true, y_pred, sample_weight=None):
        """
        Calcula Gradiente y Hessiano de la Loss Function Monetaria.
        XGBoost/LightGBM llama a esto con (labels, pre-activation_preds).
        """
        # 1. Sigmoid activation para pasar log-odds a probabilidad p
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-5, 1.0 - 1e-5)
        
        # 2. Base gradient del Binary Cross Entropy
        grad = p - y_true
        
        # 3. Base hessian del Binary Cross Entropy
        hess = p * (1.0 - p)
        
        # 4. Inyectar asimetría monetaria
        # Si observamos y_true == 0, el gradiente empuja p hacia 0.
        # Si p es alto (FP), el gradiente positivo se multiplica por gamma_fp.
        grad[y_true == 0] *= gamma_fp
        hess[y_true == 0] *= gamma_fp
        
        # Si observamos y_true == 1, el gradiente empuja p hacia 1.
        # Si p es bajo (FN), el gradiente negativo se multiplica por gamma_fn.
        grad[y_true == 1] *= gamma_fn
        hess[y_true == 1] *= gamma_fn
        
        if sample_weight is not None:
            grad *= sample_weight
            hess *= sample_weight
            
        hess = np.clip(hess, 1e-4, None)
        return grad, hess

    return monetary_pnl_loss
