"""
Luna V1 — Config Loader
======================
Carga settings.yaml y expone un objeto cfg tipado.
Uso:
    from config.settings import cfg
    lag = cfg.data.m2_lag_days  # 42
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml

# Cargar .env automáticamente desde PROJECT_ROOT (Luna v1/)
# Necesario para que ${FRED_API_KEY}, ${TELEGRAM_TOKEN}, etc. resuelvan correctamente
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv no instalado — usar variables de entorno del sistema

_ROOT = Path(__file__).parent.parent


class _Namespace:
    """Convierte un dict recursivamente en objeto con atributos."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, _Namespace(v) if isinstance(v, dict) else v)

    def __repr__(self) -> str:
        return f"Namespace({vars(self)})"

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _resolve_env(obj: Any) -> Any:
    """Reemplaza '${VAR_NAME}' por el valor de la variable de entorno."""
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var = obj[2:-1]
        value = os.environ.get(var)
        if value is None:
            # En desarrollo local no bloqueamos, solo advertimos
            import warnings
            warnings.warn(f"⚠️  Env var '{var}' no está definida. Settle .env antes de producción.")
        return value or obj
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    return obj


def _unify_parameters(d: dict) -> dict:
    """
    [INSTITUTIONAL FIX] Unifica parámetros matemáticamente correlacionados 
    para evitar el 'Fallback Hell' o desajustes por error humano.
    """
    try:
        # Extraer bloques
        xgb = d.get("xgboost", {})
        sop = d.get("sop", {})
        stat = d.get("stat", {})
        wfb = d.get("wfb", {})
        ts = d.get("temporal_splits", {})

        # 1. OPTUNA vs STATISTICAL TRIALS (Prevención BUG-01)
        if "optuna_trials" in xgb and "n_trials_total" in stat:
            # Optuna dicta el número de trials reales. Stat debe usar el mismo para DSR.
            stat["n_trials_total"] = xgb["optuna_trials"]

        # 2. BARRERAS vs PURGA (Prevención Data Leakage)
        vbh = xgb.get("vertical_barrier_hours", 48)
        dyn_max = xgb.get("dynamic_horizon_max_h", 168) if xgb.get("dynamic_barrier", True) else vbh
        
        # Purga debe ser >= al horizonte máximo posible de un trade
        if "purge_hours" in sop:
            sop["purge_hours"] = max(sop["purge_hours"], dyn_max)
        
        # 3. PBO BLOCK SIZE
        # El bloque de Monte Carlo PBO debe contener trades enteros, ergo >= purge_hours
        if "mc_block_size_hours" in stat and "purge_hours" in sop:
            stat["mc_block_size_hours"] = max(stat["mc_block_size_hours"], sop["purge_hours"])

        # 4. EMBARGO ESTRICTO (Regla R3: >= 1x horizonte máximo)
        # Solo lo forzamos si no estamos en 'soft_embargo'
        soft_embargo = wfb.get("soft_embargo_enabled", False)
        if not soft_embargo:
            min_embargo = max(96, dyn_max)
            if "embargo_hours" in sop:
                sop["embargo_hours"] = max(sop["embargo_hours"], min_embargo)
            if "embargo_hours" in ts:
                ts["embargo_hours"] = max(ts.get("embargo_hours", 0), min_embargo)

        # 5. UMBRALES DE GAUNTLET Y VALIDACIÓN ESTADÍSTICA
        gaunt = d.get("gauntlet", {})
        
        # max_pbo: Gauntlet y Stat deben tener el mismo umbral de overfit
        if "max_pbo" in gaunt and "max_pbo" in stat:
            stat["max_pbo"] = gaunt["max_pbo"]
            
        # min_dsr: Gauntlet manda (SOP R5 Bailey & LdP dictamina 0.75)
        # stat.min_dsr=0.2 es un grave error de configuración que permitiría sistemas basura
        if "min_dsr" in gaunt and "min_dsr" in stat:
            stat["min_dsr"] = gaunt["min_dsr"]

        # 6. KELLY SIZER vs POSITION SIZER (SOP R17: Fractional Kelly = 0.25)
        pos_sizer = d.get("position_sizer", {})
        kelly = d.get("kelly_sizer", {})
        if "kelly_fraction" in pos_sizer and "kelly_fraction" in kelly:
            # Position sizer manda porque es el risk manager global
            kelly["kelly_fraction"] = pos_sizer["kelly_fraction"]
            
        # 7. COSTES DE TRANSACCIÓN E INFRICCIÓN (SOP R6)
        # costs.*_pct está en % absoluto (ej. 0.25 para 0.25%), mientras sop.cost_pct es decimal (0.0025)
        # Sincronizamos costs -> sop para que la API asuma el valor real 
        costs_dict = d.get("costs", {})
        if "round_trip_pct" in costs_dict and "cost_pct" in sop:
            # Unificamos a decimal si cost_pct != round_trip_pct / 100
            real_cost_rt = costs_dict["round_trip_pct"] / 100.0
            sop["cost_pct"] = real_cost_rt
            sop["cost_spot_taker_rt"] = real_cost_rt
            sop["cost_perp_taker_rt"] = real_cost_rt

        # 8. MARGEN DE ERROR DE BRIER (Debe coincidir con la advertencia de degradación)
        if "xgb_brier_warn" in stat and "xgb_brier_hard_stop" in stat:
            # El brier margin debe ser la diferencia entre el stop y el warn
            # para que los tests pasen y matematicamente tenga sentido
            stat["brier_margin_range"] = stat["xgb_brier_hard_stop"] - stat["xgb_brier_warn"]

        # 9. DECAIMIENTO Y REGULARIZACIÓN
        meta = d.get("metalabeler", {})
        if "weight_decay_alpha" in xgb and "weight_decay_alpha" in meta:
            # Mismo decaimiento para todo el ensamble
            meta["weight_decay_alpha"] = xgb["weight_decay_alpha"]
            
        # 10. ESPACIOS DE BÚSQUEDA OPTUNA (LGBM y XGB deben tener mismos límites)
        lgbm = d.get("lightgbm", {})
        if "optuna_search_space" in xgb and "optuna_search_space" in lgbm:
            xgb_os = xgb["optuna_search_space"]
            lgbm_os = lgbm["optuna_search_space"]
            for key in ["learning_rate_min", "learning_rate_max", "n_estimators_min", "n_estimators_max"]:
                if key in xgb_os and key in lgbm_os:
                    lgbm_os[key] = xgb_os[key]

    except Exception as e:
        import warnings
        warnings.warn(f"Error unificando parámetros: {e}")

    return d


def load_config(path: Path | str | None = None) -> _Namespace:
    """
    Carga settings.yaml.
    Si se pasa 'path', usa ese archivo; si no, usa config/settings.yaml del root.
    """
    if path is None:
        path = _ROOT / "config" / "settings.yaml"
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    resolved = _resolve_env(raw)
    unified = _unify_parameters(resolved)
    return _Namespace(unified)


# Singleton — importar con: from config.settings import cfg
cfg = load_config()
