"""
luna/utils/determinism.py
=========================
[FIX-AE-DETERMINISM-01 2026-06-26] Helper central de reproducibilidad.

Contexto: el pipeline seedea HMM / SFI / genetico / XGBoost individualmente, pero los
componentes basados en torch (AutoEncoder de features, DenoisingAutoEncoder del OOD Guard,
clf del MetaLabeler) quedaron SIN seedear -> resultados NO reproducibles run-a-run.
Sintoma: misma seed y misma ventana daban WR 29% vs 75% porque el AE generaba ae_feat_*
distintas cada run y el SFI seleccionaba features distintas.
Ver docs/hallazgos_run_baseline_20260626.md (6.6).

IMPORTANTE: esto da determinismo POR-seed (seed 42 siempre el mismo resultado) PRESERVANDO
la diversidad ENTRE seeds (42 != 100 != 777, base del ensamble). NO es overfitting: es
determinismo de computo, ortogonal a la capacidad de generalizacion.
"""
import os
import random


def seed_everything(seed: "int | None" = None) -> int:
    """Seedea random / numpy / torch / cuda y fuerza cuDNN determinista. Idempotente.

    Si seed es None, lo lee de la env var LUNA_SEED (fallback 42).
    Devuelve la seed efectiva (util para crear torch.Generator de DataLoaders).
    """
    if seed is None:
        seed = int(os.environ.get("LUNA_SEED") or 42)
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    print(f"[FIX-AE-DETERMINISM-01] seed_everything(seed={seed}) -> torch/cuda/cudnn determinista")  # RULE[fixbugsprints.md]
    return seed


def seeded_generator(seed: "int | None" = None):
    """torch.Generator seedeado para DataLoaders con shuffle=True (shuffle reproducible)."""
    import torch as _torch
    if seed is None:
        seed = int(os.environ.get("LUNA_SEED") or 42)
    g = _torch.Generator()
    g.manual_seed(int(seed))
    return g
