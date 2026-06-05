"""
project_root.py — Luna V1 (OP-05)
===================================
Función centralizada get_project_root().
Antes de este fix (OP-05), cada módulo declaraba su propia versión inline:

    def get_project_root() -> Path:
        return Path(__file__).resolve().parents[N]

Esto causaba duplicación en al menos 4 archivos:
 - core/models/calibrate_probabilities.py
 - core/models/train_metalabeler.py
 - core/models/train_xgboost.py
 - core/models/ood_guard.py
 - core/models/hmm_regime.py

Uso:
    from luna.utils.project_root import get_project_root
    ROOT = get_project_root()
"""
from __future__ import annotations
from pathlib import Path


def get_project_root() -> Path:
    """
    Devuelve la raíz del proyecto Luna V1 de forma determinista.
    Sube 3 niveles desde este archivo (core/utils/project_root.py → raíz).
    """
    return Path(__file__).resolve().parents[2]
