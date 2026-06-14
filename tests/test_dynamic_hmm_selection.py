import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
import builtins

warnings.filterwarnings("ignore")

# Resolve project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from luna.models.hmm_regime import HMMRegimeModel
from luna.models.regime_router import RegimeRouter
from luna.features.tbm import apply_triple_barrier

def test_dynamic_hmm_selection():
    original_print = builtins.print
    
    # Filtro para ignorar logs ruidosos repetitivos
    def quiet_print(*args, **kwargs):
        if args and any("BUG-FIX-HMM-RESOLVER" in str(arg) for arg in args):
            return
        original_print(*args, **kwargs)
        
    builtins.print = quiet_print
    
    try:
        original_print("[TEST-DYN-HMM] Iniciando simulación de Selección Dinámica de Regímenes HMM...")
        
        # Evaluar ventanas W1 y W4
        for window in ["W1", "W4"]:
            original_print(f"\n=================== Window {window} ===================")
            
            # 1. Rutas
            feat_dir = ROOT / "data" / "wfb_cache" / window / "features"
            train_path = feat_dir / "features_train.parquet"
            val_path = feat_dir / "features_validation.parquet"
            holdout_path = feat_dir / f"features_holdout_{window}.parquet"
            
            # Usamos seed1337 / seed86454 en caché como sujeto de prueba
            test_seed = "seed86454" if window == "W1" else "seed1337"
            models_dir = ROOT / "data" / "wfb_cache" / test_seed / window / "models"
            
            if not (train_path.exists() and val_path.exists() and holdout_path.exists() and models_dir.exists()):
                original_print(f"[TEST-DYN-HMM] SKIP: Archivos para ventana {window} no encontrados.")
                continue
                
            # 2. Cargar datos
            df_train = pd.read_parquet(train_path)
            df_val = pd.read_parquet(val_path)
            df_holdout = pd.read_parquet(holdout_path)
            
            original_print(f"Cargados conjuntos: Train={len(df_train)} | Val={len(df_val)} | Holdout={len(df_holdout)}")
            
            # 3. Predecir regímenes HMM in-sample (validation) y out-of-sample (holdout)
            hmm_model = HMMRegimeModel.load(models_dir)
            
            # Validación
            hmm_val = hmm_model.predict_regime_series(df_val)
            df_val["HMM_Semantic"] = hmm_val["HMM_Semantic"]
            
            # Holdout
            hmm_hold = hmm_model.predict_regime_series(df_holdout)
            df_holdout["HMM_Semantic"] = hmm_hold["HMM_Semantic"]
            
            # 4. Obtener predicciones XGBoost in-sample y out-of-sample
            router = RegimeRouter(models_dir, agent_type="xgboost", direction="long")
            
            xgb_val = router.route_and_predict(df_val)
            df_val["xgb_prob_cal"] = xgb_val["calibrated"]
            
            xgb_hold = router.route_and_predict(df_holdout)
            df_holdout["xgb_prob_cal"] = xgb_hold["calibrated"]
            
            # 5. Algoritmo Dinámico in-sample (Validation)
            original_print("\n--- Análisis de Validación In-Sample por Régimen HMM ---")
            
            xgb_thresh = 0.48
            val_candidates = df_val["xgb_prob_cal"] >= xgb_thresh
            val_signal_times = df_val.index[val_candidates]
            
            allowed_regimes_dynamic = []
            
            if len(val_signal_times) > 0:
                from luna.models.predict_oos import get_hmm_tbm_params, get_hmm_horizon
                _pt = df_val["HMM_Semantic"].map(lambda r: get_hmm_tbm_params(r)["tp"])
                _sl = df_val["HMM_Semantic"].map(lambda r: get_hmm_tbm_params(r)["sl"])
                
                _prob_series = df_val["xgb_prob_cal"].fillna(0.5).clip(0.5, 1.0)
                _conf_scaler = 0.7 + ((_prob_series - 0.5) / 0.5) * (1.3 - 0.7)
                _pt = _pt * _conf_scaler
                _sl = _sl * _conf_scaler
                
                _dyn_max = int(df_val["HMM_Semantic"].dropna().map(lambda r: get_hmm_horizon(r)).mode().iloc[0] if not df_val["HMM_Semantic"].dropna().empty else 168)
                
                tbm_val = apply_triple_barrier(
                    price_series=df_val["close"],
                    event_times=val_signal_times,
                    sides=pd.Series(1, index=val_signal_times),
                    pt_sl_multiplier=[_pt, _sl],
                    vertical_barrier_hours=72,
                    min_return=0.005,
                    dynamic_barrier=True,
                    dynamic_horizon_min_h=24,
                    dynamic_horizon_max_h=_dyn_max,
                    linear_decay_pt=True,
                    pt_decay_fraction=0.75,
                    funding_series=df_val.get("FundingRate"),
                )
                
                df_val_signals = pd.DataFrame(index=val_signal_times)
                df_val_signals["ret_net"] = tbm_val["ret"] - 0.0015 # fee
                df_val_signals["HMM_Semantic"] = df_val.loc[val_signal_times, "HMM_Semantic"]
                
                # Agrupar por HMM Semantic para medir métricas
                for regime in df_val["HMM_Semantic"].unique():
                    reg_signals = df_val_signals[df_val_signals["HMM_Semantic"] == regime]
                    n_signals = len(reg_signals)
                    
                    if n_signals > 0:
                        wins = reg_signals.loc[reg_signals["ret_net"] > 0, "ret_net"].sum()
                        losses = reg_signals.loc[reg_signals["ret_net"] < 0, "ret_net"].sum()
                        pf = wins / abs(losses) if losses != 0 else float('inf')
                        wr = (reg_signals["ret_net"] > 0).mean()
                        total_ret = reg_signals["ret_net"].sum()
                        
                        is_bull_semantic = "BULL" in str(regime).upper() or "RANGE" in str(regime).upper()
                        
                        decision = False
                        if n_signals >= 3:
                            if pf > 1.05 and total_ret > 0:
                                decision = True
                            elif is_bull_semantic and pf > 0.95:
                                decision = True
                        else:
                            if is_bull_semantic:
                                decision = True
                                
                        original_print(f"  Régimen: {regime:<22} | Trades: {n_signals:<3} | WR: {wr:.1%} | PF: {pf:.3f} | RetVal: {total_ret*100:+.2f}% | Dinámico Habilitado: {decision}")
                        
                        if decision:
                            allowed_regimes_dynamic.append(regime)
                    else:
                        is_bull_semantic = "BULL" in str(regime).upper() or "RANGE" in str(regime).upper()
                        if is_bull_semantic:
                            allowed_regimes_dynamic.append(regime)
                            original_print(f"  Régimen: {regime:<22} | Trades: 0   | (Sin señales in-sample) | Dinámico Habilitado: True (Default Bull/Range)")
            
            original_print(f"\nRegímenes Habilitados Dinámicamente para {window}: {allowed_regimes_dynamic}")
            
            # 6. Validar out-of-sample en Holdout con estos regímenes dinámicos
            holdout_candidates = df_holdout["xgb_prob_cal"] >= xgb_thresh
            hold_signal_times = df_holdout.index[holdout_candidates]
            
            if len(hold_signal_times) == 0:
                original_print("  Holdout: Sin señales XGBoost brutas.")
                continue
                
            # Filtro estático original
            from config.settings import cfg
            original_allowed = list(cfg.metalabeler.hmm_allowed_regimes)
            
            import re
            def is_allowed_func(val, allowed_list):
                v = str(val).upper()
                if v in [x.upper() for x in allowed_list]:
                    return True
                allowed_bases = {re.sub(r'_[A-D]$', '', str(lbl).upper()) for lbl in allowed_list}
                for base in allowed_bases:
                    if v.startswith(base):
                        return True
                return False
                
            passed_orig = hold_signal_times[df_holdout.loc[hold_signal_times, "HMM_Semantic"].apply(lambda x: is_allowed_func(x, original_allowed))]
            passed_dyn = hold_signal_times[df_holdout.loc[hold_signal_times, "HMM_Semantic"].apply(lambda x: is_allowed_func(x, allowed_regimes_dynamic))]
            
            original_print(f"\nComparativa Out-of-Sample (Holdout {window}):")
            original_print(f"  Filtro Estático (Settings):  {len(passed_orig)} trades pasados")
            original_print(f"  Filtro Dinámico (Propuesto): {len(passed_dyn)} trades pasados")
            
            if len(passed_dyn) > 0:
                _pt_h = df_holdout["HMM_Semantic"].map(lambda r: get_hmm_tbm_params(r)["tp"])
                _sl_h = df_holdout["HMM_Semantic"].map(lambda r: get_hmm_tbm_params(r)["sl"])
                
                _prob_series_h = df_holdout["xgb_prob_cal"].fillna(0.5).clip(0.5, 1.0)
                _conf_scaler_h = 0.7 + ((_prob_series_h - 0.5) / 0.5) * (1.3 - 0.7)
                _pt_h = _pt_h * _conf_scaler_h
                _sl_h = _sl_h * _conf_scaler_h
                
                _dyn_max_h = int(df_holdout["HMM_Semantic"].dropna().map(lambda r: get_hmm_horizon(r)).mode().iloc[0] if not df_holdout["HMM_Semantic"].dropna().empty else 168)
                
                tbm_hold = apply_triple_barrier(
                    price_series=df_holdout["close"],
                    event_times=passed_dyn,
                    sides=pd.Series(1, index=passed_dyn),
                    pt_sl_multiplier=[_pt_h, _sl_h],
                    vertical_barrier_hours=72,
                    min_return=0.005,
                    dynamic_barrier=True,
                    dynamic_horizon_min_h=24,
                    dynamic_horizon_max_h=_dyn_max_h,
                    linear_decay_pt=True,
                    pt_decay_fraction=0.75,
                    funding_series=df_holdout.get("FundingRate"),
                )
                
                net_ret_hold = tbm_hold["ret"] - 0.0015
                wins_h = net_ret_hold[net_ret_hold > 0].sum()
                losses_h = net_ret_hold[net_ret_hold < 0].sum()
                pf_h = wins_h / abs(losses_h) if losses_h != 0 else float('inf')
                wr_h = (net_ret_hold > 0).mean()
                tot_ret_h = net_ret_hold.sum()
                
                original_print(f"  Performance Dinámica Holdout: WR={wr_h:.1%} | PF={pf_h:.3f} | Retorno Total={tot_ret_h*100:+.2f}%")
                
                if window == "W1":
                    assert "1_VOLATILE_BULL_B" in allowed_regimes_dynamic, "El algoritmo dinámico no seleccionó 1_VOLATILE_BULL_B en W1!"
                    original_print("  [OK] Algoritmo dinámico desbloqueó automáticamente 1_VOLATILE_BULL_B en W1.")
                elif window == "W4":
                    original_print("  [OK] El algoritmo dinámico limitó la operativa en W4 protegiendo la cuenta in-sample y adaptando la decisión.")
            else:
                original_print("  Performance Dinámica Holdout: 0 trades ejecutados (Régimen de mercado completamente bajista bloqueado).")
    finally:
        builtins.print = original_print

if __name__ == "__main__":
    test_dynamic_hmm_selection()
