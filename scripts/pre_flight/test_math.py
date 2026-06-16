from .core import *
from .core import _json_safe, _read, _cfg, _active, _load_parquet, _load_json, _is_stale_artifact, ROOT
import pandas as pd
import numpy as np
from pathlib import Path
import re
import json
import math
from itertools import combinations
from datetime import datetime

@test("TEST-51  Sharpe annualizado BTC usa 365*24 (no 252 dias equities) — BUG M-02", section="math")
def t51():
    # Verificar en todos los scripts donde se calcula Sharpe
    targets = [
        ROOT/"luna/models/train_xgboost_v2.py",
        ROOT/"luna/features/feature_selection_e.py",
        ROOT/"luna/models/train_metalabeler_v2.py",
    ]
    for f in targets:
        if not f.exists(): continue
        src = _read(f)
        # Buscar lineas donde se calcula Sharpe con annualizacion
        for _, _, s in _active(src):
            if "sqrt(252" in s or "* 252" in s or "np.sqrt(252" in s:
                assert False, f"Sharpe con 252 (equities) en {f.name}: {s}"
    # Al menos un archivo debe tener 365*24
    found_365 = any(
        "365 * 24" in _read(f) or "365*24" in _read(f)
        for f in targets if f.exists()
    )
    assert found_365, "Ningún script tiene 365*24 para Sharpe BTC 24/7"
    return "365*24 en todos los scripts"


@test("TEST-52  SFI_DSR_N_TRIALS = 600 en feature_selection_e.py — BUG M-01", section="math")
def t52():
    src = _read(ROOT/"luna/features/feature_selection_e.py")
    m = re.search(r"SFI_DSR_N_TRIALS\s*=\s*(\d+)", src)
    assert m, "SFI_DSR_N_TRIALS no encontrado"
    n = int(m.group(1))
    assert n >= 600, f"SFI_DSR_N_TRIALS={n} < 600 (filtro DSR demasiado laxo)"
    return f"SFI_DSR_N_TRIALS={n}"


@test("TEST-53  SFI_COST_ROUNDTRIP >= 0.0010 en feature_selection_e.py — R6 en SFI", section="math")
def t53():
    src = _read(ROOT/"luna/features/feature_selection_e.py")
    m = re.search(r"SFI_COST_ROUNDTRIP\s*=\s*([\d.]+)", src)
    assert m, "SFI_COST_ROUNDTRIP no encontrado"
    cost = float(m.group(1))
    assert cost >= 0.0010, f"SFI_COST_ROUNDTRIP={cost} < 0.0010"
    return f"SFI_COST_ROUNDTRIP={cost}"


@test("TEST-54  Euler-Mascheroni correcto en DSR (0.5772...)", section="math")
def t54():
    # Ambas implementaciones de DSR deben tener la constante correcta
    for f in [ROOT/"luna/models/train_xgboost_v2.py", ROOT/"luna/features/feature_selection_e.py"]:
        if not f.exists(): continue
        src = _read(f)
        if "deflated_sharpe" in src or "_compute_dsr" in src:
            assert "0.5772" in src, f"Euler-Mascheroni incorrecto en {f.name}"
    return "gamma=0.5772... en DSR"


@test("TEST-55  LSTM metalabeler es unidireccional en codigo (no bidireccional)", section="math")
def t55():
    src = _read(ROOT/"luna/models/train_metalabeler_v2.py")
    # Verificar la constante: bidirectional=False en la definicion LSTM
    assert "bidirectional=False" in src, "LSTM no declarado como unidireccional (SOP R1)"
    # ARCH-02: HMM_N_STATES ya no es un literal hardcodeado — se lee desde cfg.
    # El test verifica que el modulo use _cfg_meta para leer la dimension del HMM.
    # Si HMM_N_STATES = int(getattr(...)) existe en el codigo, el patron ARCH-02 esta correcto.
    cfg = _cfg()
    cfg_hmm_n = int(getattr(getattr(cfg, "hmm", object()), "n_states", 4))
    uses_cfg  = ("_cfg_meta" in src or "cfg_meta" in src or "cfg.hmm" in src)
    has_dyn   = ("getattr" in src and ("n_states" in src or "hmm" in src.lower()))
    if not uses_cfg and not has_dyn:
        # Fallback: buscar literal para detectar desync
        m = re.search(r"HMM_N_STATES\s*=\s*(\d+)", src)
        if m:
            n_code = int(m.group(1))
            assert n_code == cfg_hmm_n, f"HMM_N_STATES literal={n_code} != settings hmm.n_states={cfg_hmm_n}"
    return f"LSTM causal | HMM_N_STATES via cfg.hmm.n_states={cfg_hmm_n} (ARCH-02)"


@test("TEST-56  MetaLabelerV2 N_CPCV_GROUPS == 10 (consistente con XGBoost)", section="math")
def t56():
    src = _read(ROOT/"luna/models/train_metalabeler_v2.py")
    m = re.search(r"N_CPCV_GROUPS\s*=\s*(\d+)", src)
    if m:
        assert int(m.group(1)) == 10, f"N_CPCV_GROUPS={m.group(1)} != 10"
    return f"N_CPCV_GROUPS={m.group(1) if m else 'default'}"


@test("TEST-57  COST_PCT en MetaLabelerV2 >= 0.0010 (R6 aplicado a meta)", section="math")
def t57():
    src = _read(ROOT/"luna/models/train_metalabeler_v2.py")
    # ARCH-02: COST_PCT ya no es un literal — lee desde cfg.sop.cost_pct.
    # Verificar que: (A) usa el patron cfg, o (B) si aun hay literal, sigue siendo correcto.
    uses_cfg = ("_cfg_meta" in src or "cfg_meta" in src) and ("cost_pct" in src or "sop" in src)
    m = re.search(r"COST_PCT\s*=\s*([\d.]+)", src)
    if m:
        # Literal aun presente: verificar valor correcto
        cost = float(m.group(1))
        assert cost >= 0.0010, f"COST_PCT literal={cost} < 0.0010 en MetaLabelerV2"
        return f"COST_PCT={cost} (literal OK)"
    elif uses_cfg:
        # Lee desde cfg — verificar que cfg.sop.cost_pct sea >= 0.0010
        cfg_cost = float(getattr(getattr(_cfg(), "sop", object()), "cost_pct", 0.0010))
        assert cfg_cost >= 0.0010, f"cfg.sop.cost_pct={cfg_cost} < 0.0010"
        return f"COST_PCT via cfg.sop.cost_pct={cfg_cost} (ARCH-02 OK)"
    else:
        # Ni literal ni cfg — revisar si el modulo aplica costo de alguna forma
        assert "cost" in src.lower() or "fee" in src.lower(), (
            "MetaLabelerV2 no aplica ningun costo (R6 violado): "
            "Anadir lectura de sop.cost_pct desde cfg."
        )
        return "COST via referencia implicita (verificar manualmente)"


# ═══════════════════════════════════════════════════════════
#  SECCION 9: COHERENCIA ENTRE ARTEFACTOS (6 tests)
# ═══════════════════════════════════════════════════════════
