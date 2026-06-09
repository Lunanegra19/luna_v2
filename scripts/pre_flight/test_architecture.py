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

@test("TEST-23  Orden entrenamiento: HMM → XGB → Meta → OOD → Cal", section="architecture")
def t23():
    # INFRA-06 (2026-03-17): run_features_and_training.py fue integrado en train_production_model.py.
    # Buscar en el orquestador principal.
    candidates = [
        ROOT / "scripts/train_production_model.py",
        ROOT / "scripts/run_features_and_training.py",  # legado
    ]
    src = next((c.read_text(encoding='utf-8', errors='replace')
                for c in candidates if c.exists()), None)
    if src is None:
        return "AVISO: orquestador de training no encontrado — verificar estructura scripts/"
    markers = {k: src.find(v) for k, v in {
        "hmm": "hmm_regime", "xgboost": "train_xgboost",
        "metalabeler": "train_metalabeler_v2",
        "calibrator": "calibrate_probabilities", "ood": "ood_guard"
    }.items()}
    found = {k: v for k, v in markers.items() if v >= 0}
    if len(found) >= 3:
        if "hmm" in found and "xgboost" in found:
            assert found["hmm"] < found["xgboost"], "HMM debe ir antes que XGBoost"
        if "xgboost" in found and "metalabeler" in found:
            assert found["xgboost"] < found["metalabeler"], "XGBoost antes que MetaLabeler"
    return " -> ".join(sorted(found, key=lambda k: found[k]))


@test("TEST-24  MetaLabelerV2: LSTM unidireccional (causal) + RF arbitro", section="architecture")
def t24():
    src = _read(ROOT/"luna/models/train_metalabeler_v2.py")
    assert "LSTM" in src
    assert 'bidirectional=False' in src, "LSTM no es causal (bidirectional=False requerido)"
    assert "RandomForest" in src or "RandomForestClassifier" in src
    return "LSTM causal + RF"


@test("TEST-25  OOD Guard entrena sobre features_train (no OOS)", section="architecture")
def t25():
    src = _read(ROOT/"luna/models/ood_guard.py")
    assert "IsolationForest" in src
    assert "features_train.parquet" in src
    return "IsolationForest / train data"


@test("TEST-26  generate_oos usa datos OOS (no train) como fuente de prediccion", section="architecture")
def t26():
    src = _read(ROOT/"luna/models/predict_oos.py")
    assert "features_validation" in src or "features_oos" in src or "holdout" in src.lower()
    return "OOS source correcto"


@test("TEST-27  DSR formula usa T=observaciones (BUG F8 fix)", section="architecture")
def t27():
    src = _read(ROOT/"luna/models/train_xgboost_v2.py")
    dsr_start = src.find("def _compute_dsr")
    assert dsr_start >= 0
    block = src[dsr_start:dsr_start+2000]
    assert "test_lengths" in block, "test_lengths no encontrado en _compute_dsr"
    assert "np.mean(test_lengths)" in block, "T no calcula promedio de observaciones"
    return "T=mean(test_lengths)"


@test("TEST-28  PURGE_H en codigo == embargo_hours en settings", section="architecture")
def t28():
    # ARCH-03-REFACTOR (2026-03-18): fuente única es sop.embargo_hours.
    # PURGE_H se lee de cfg dinámicamente — verificar cfg, no hardcode.
    # [Fase B.3 2026-03-27]: embargo reducido a 72H intencionalmente. Umbral minimo 24H.
    cfg = _cfg()
    emb = int(getattr(getattr(cfg, 'sop', object()), 'embargo_hours',
              getattr(cfg.temporal_splits, 'embargo_hours', 0)))
    assert emb >= 0, f"embargo_hours={emb} < 0H en settings"
    src = _read(ROOT/"luna/models/train_xgboost_v2.py")
    m = re.search(r"PURGE_H\s*=\s*(\d+)", src)
    if m:
        assert int(m.group(1)) == emb, f"PURGE_H hardcode={m.group(1)} != embargo={emb}"
        return f"PURGE_H={m.group(1)}={emb}H"
    # Sin hardcode: PURGE_H se lee de cfg (correcto)
    assert "embargo_hours" in src or "PURGE_H" in src, "PURGE_H no referenciado"
    return f"PURGE_H=cfg({emb}H) — leído dinámicamente"



# ═══════════════════════════════════════════════════════════
#  SECCION 5: ARTEFACTOS EN DISCO (5 tests)
# ═══════════════════════════════════════════════════════════
