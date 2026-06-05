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

@test("TEST-01  CPCV produce exactamente 45 splits C(10,2)", section="legacy")
def t01():
    df = pd.read_parquet(ROOT / "data/features/features_train.parquet", columns=["close"])
    mask = df.index <= "2024-06-30"
    ts, n = df.index[mask], mask.sum()
    assert n > 10_000
    groups = np.array_split(np.arange(n), 10)
    ok = sum(
        1 for gidx in combinations(range(10), 2)
        if (lambda ti, tr: (
            np.ones(len(tr), bool) &
            ~np.logical_or.reduce([
                ((ts[tr] >= ts[groups[gi][0]] - pd.Timedelta(hours=96)) &
                 (ts[tr] <= ts[groups[gi][-1]] + pd.Timedelta(hours=96)))
                for gi in gidx
            ])
        ).sum() >= 100 and len(ti) >= 50)(
            np.concatenate([groups[i] for i in gidx]),
            np.concatenate([groups[i] for i in range(10) if i not in gidx])
        )
    )
    assert ok == 45, f"Esperado 45, obtenidos {ok}"
    return f"{ok}/45"


@test("TEST-02  TBM pt_mult/sl_mult sin hardcode en archivos clave", section="legacy")
def t02():
    files = [ROOT/"luna/models/train_xgboost_v2.py",
             ROOT/"luna/models/train_metalabeler_v2.py",
             ROOT/"luna/models/predict_oos.py"]
    bad = ["pt_sl=[2.", "pt_sl=[1.", "pt_sl_multiplier=[1.", "pt_sl_multiplier=[2."]
    for f in files:
        for _, _, s in _active(_read(f)):
            for p in bad:
                assert p not in s, f"TBM hardcodeado en {f.name}: {s}"
    return "dinamicos en 3 archivos"


@test("TEST-03  FracDiff train_cutoff dinamico (fallbacks defensivos OK)", section="legacy")
def t03():
    for _, _, s in _active(_read(ROOT/"luna/features/feature_pipeline.py")):
        for d in ['"2023-12-31"', "'2023-12-31'"]:
            if d in s and "fallback" not in s.lower() and "warning" not in s.lower():
                assert False, f"Fecha hardcodeada activa: {s}"
    return "dinamico"


@test("TEST-04  HMM train_cutoff dinamico (no hardcode en activo)", section="legacy")
def t04():
    for _, _, s in _active(_read(ROOT/"luna/models/hmm_regime.py")):
        for d in ['"2023-12-31"', "'2023-12-31'"]:
            if d in s and not s.startswith("def ") and "fallback" not in s.lower():
                assert False, f"Fecha hardcodeada activa: {s}"
    return "dinamico"


@test("TEST-05  Sin imports BiLSTMv1 en scripts activos", section="legacy")
def t05():
    dirs = [ROOT/"scripts", ROOT/"luna/models", ROOT/"luna/features", ROOT/"luna/live"]
    forbidden = ["MetaLabelerBiLSTM", "from luna.models.train_metalabeler import",
                 "build_metalabeler_dataset"]
    for d in dirs:
        for py in d.glob("*.py"):
            if "_legacy" in str(py) or py.name == "pre_flight_check.py":
                continue
            for _, _, s in _active(_read(py)):
                for pat in forbidden:
                    assert pat not in s, f"Import BiLSTMv1 en {py.name}: {s}"
    return "0 imports"


@test("TEST-06  calibrate_probabilities.py es MetaLabelerV2Calibrator", section="legacy")
def t06():
    src = _read(ROOT/"luna/models/calibrate_probabilities.py")
    assert "MetaLabelerV2Calibrator" in src
    assert "CalibratedClassifierCV" in src
    for _, _, s in _active(src):
        assert "MetaLabelerBiLSTM" not in s, f"BiLSTMv1 en activo: {s}"
    return "V2Calibrator"


@test("TEST-07  feature_pipeline no invoca add_dynamic_target()", section="legacy")
def t07():
    for _, _, s in _active(_read(ROOT/"luna/features/feature_pipeline.py")):
        if ("add_dynamic_target()" in s and not s.startswith("def ") and
                not s.startswith('"') and not s.startswith("'") and
                "DeprecationWarning" not in s and "raise" not in s):
            assert False, f"Invocado activamente: {s}"
    return "no invocado"


@test("TEST-08  BiLSTMv1 en _legacy/, no en ruta activa", section="legacy")
def t08():
    assert not (ROOT/"luna/models/train_metalabeler.py").exists()
    assert (ROOT/"luna/models/_legacy/train_metalabeler_bilstm_v1.py").exists()
    return "_legacy/ OK"


@test("TEST-09  features_train.parquet — integridad minima", section="legacy")
def t09():
    df = _load_parquet("features_train.parquet")
    assert len(df) > 30_000, f"Solo {len(df)} rows"
    assert df.shape[1] > 15, f"Solo {df.shape[1]} cols"
    leak = [c for c in df.columns if any(k in c for k in ["future_ret","fwd_return","next_ret"])]
    assert not leak, f"Leakage columns: {leak}"
    return f"{len(df):,} x {df.shape[1]}"


@test("TEST-10  settings.yaml tiene claves criticas", section="legacy")
def t10():
    cfg = _cfg()
    assert getattr(getattr(cfg,"temporal_splits",None),"train_end",None)
    xgb = getattr(cfg,"xgboost",None)
    assert xgb and float(getattr(xgb,"pt_mult_min",0)) > 0
    assert float(getattr(xgb,"sl_mult_min",0)) > 0
    cfg2 = _cfg()
    return f"train_end={cfg2.temporal_splits.train_end}"


# ═══════════════════════════════════════════════════════════
#  SECCION 2: SOP IRON RULES (7 tests)
# ═══════════════════════════════════════════════════════════
