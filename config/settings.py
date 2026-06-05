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
    return _Namespace(resolved)


# Singleton — importar con: from config.settings import cfg
cfg = load_config()
