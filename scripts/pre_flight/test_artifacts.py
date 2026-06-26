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

@test("TEST-29  Modelo XGBoost + signature existen", section="artifacts")
def t29():
    import os
    if os.environ.get("LUNA_SKIP_ARTIFACT_CHECKS") == "1":
        return "WARN -- skipped artifact check (pre-training env active)"
    cfg = _cfg()
    f2 = getattr(cfg, "fase2", None)
    use_regimes = getattr(f2, "use_regime_agents", False) if f2 else False
    direction = getattr(f2, "direction_mode", "long") if f2 else "long"
    
    if use_regimes:
        regimes = ["bull", "bear", "range"]
        # [DUAL-BOT-PREFLIGHT-FIX 2026-06-24] en modo 'both' cada regimen entrena su direccion
        # NATIVA (bear->short, resto->long), no '_both'. Ver train_xgboost_v2.py:303-323.
        def _native(r):
            return ("short" if r == "bear" else "long") if direction == "both" else direction
        found_models = []
        for r in regimes:
            dd = _native(r)
            model = ROOT / f"data/models/xgboost_meta_{r}_{dd}.model"
            sig   = ROOT / f"data/models/xgboost_meta_{r}_{dd}_signature.json"
            assert model.exists() and sig.exists(), f"Regime model/sig {r} missing: {model.name}"
            d = json.loads(sig.read_text(encoding="utf-8"))
            assert any(k in d for k in ["features","selected_features","feature_names"])
            found_models.append(f"{r}_{dd}:{model.stat().st_size//1024}KB")
        return " | ".join(found_models)
    else:
        model = ROOT/"data/models/xgboost_meta.model"
        sig   = ROOT/"data/models/xgboost_meta_signature.json"
        assert model.exists() and sig.exists()
        d = json.loads(sig.read_text(encoding="utf-8"))
        assert any(k in d for k in ["features","selected_features","feature_names"])
        return f"{model.stat().st_size//1024}KB"


@test("TEST-30  Modelo HMM existe en data/models/", section="artifacts")
def t30():
    import os
    if os.environ.get("LUNA_SKIP_ARTIFACT_CHECKS") == "1":
        return "WARN -- skipped artifact check (pre-training env active)"
    candidates = [ROOT/"data/models/hmm_model.pkl", ROOT/"data/models/hmm_regime.pkl"]
    found = [p for p in candidates if p.exists()]
    assert found, f"Ningun HMM encontrado"
    return f"{found[0].name}"


@test("TEST-31  Modelo MetaLabelerV2 existe en data/models/", section="artifacts")
def t31():
    import os
    if os.environ.get("LUNA_SKIP_ARTIFACT_CHECKS") == "1":
        return "WARN -- skipped artifact check (pre-training env active)"
    cfg = _cfg()
    f2 = getattr(cfg, "fase2", None)
    direction = getattr(f2, "direction_mode", "long") if f2 else "long"
    use_regimes = getattr(f2, "use_regime_agents", False) if f2 else False

    if use_regimes:
        # [DUAL-BOT-PREFLIGHT-FIX 2026-06-24] en modo 'both' existen MetaLabelers long Y short
        _dirs = ["long", "short"] if direction == "both" else [direction]
        candidates = []
        for _dd in _dirs:
            candidates += [
                ROOT / f"data/models/metalabeler_v2_{_dd}_lstm.pt",
                ROOT / f"data/models/metalabeler_v2_{_dd}_rf.joblib",
                ROOT / f"data/models/metalabeler_v2_{_dd}_config.json"
            ]
    else:
        candidates = [ROOT/"data/models/metalabeler_v2.pt",
                      ROOT/"data/models/metalabeler_v2.pkl",
                      ROOT/"data/models/meta_v2_rf.pkl"]
                      
    found = [p for p in candidates if p.exists()]
    if len(found) == len(candidates):
        return f"MetaLabelerV2 {direction} OK ({found[0].stat().st_size//1024}KB)"
    
    # Detectar si hay un entrenamiento en progreso:
    # XGBoost existe pero MetaLabeler no -> pipeline corriendo, no es un error
    if use_regimes:
        # [DUAL-BOT-PREFLIGHT-FIX 2026-06-24] bull es nativo long (tambien en modo 'both')
        _bull_dir = "long" if direction == "both" else direction
        xgb_running = (ROOT / f"data/models/xgboost_meta_bull_{_bull_dir}.model").exists()
    else:
        xgb_running = (ROOT/"data/models/xgboost_meta.model").exists()
    optuna_db   = list(ROOT.glob("data/models/*.db")) + list(ROOT.glob("*.db"))
    training_in_progress = xgb_running or bool(optuna_db)
    if training_in_progress:
        # WARN premitido pre-run: el modelo se generara cuando termine el pipeline
        return ("AVISO: MetaLabelerV2 no existe aun - "
                "entrenamiento en progreso (XGBoost OK, MetaLabeler pendiente). "
                "Ejecutar pre-flight post-run para verificar.")
    assert len(found) == len(candidates), (
        "MetaLabelerV2 no encontrado y no hay pipeline activo. "
        f"Buscados: {[p.name for p in candidates]}"
    )


@test("TEST-32  OOD Guard modelo + signature existen", section="artifacts")
def t32():
    import os
    if os.environ.get("LUNA_SKIP_ARTIFACT_CHECKS") == "1":
        return "WARN -- skipped artifact check (pre-training env active)"
    assert (ROOT/"data/models/ood_guard.pkl").exists()
    sig = ROOT/"data/models/ood_guard_signature.json"
    assert sig.exists()
    d = json.loads(sig.read_text())
    assert d.get("contamination") is not None
    return f"contamination={d['contamination']}"


@test("TEST-33  selected_features.json tiene >= 10 features", section="artifacts")
def t33():
    d = _load_json("data/features/selected_features.json")
    feats = d.get("selected_features", d.get("features", []))
    if len(feats) < 10:
        _sel  = ROOT / "data" / "features" / "selected_features.json"
        _trn  = ROOT / "data" / "features" / "features_train.parquet"
        if _is_stale_artifact(_sel, _trn):
            return f"WARN: solo {len(feats)} features (residuo run anterior — SFI regenerara en run actual)"
    assert len(feats) >= 10, f"Solo {len(feats)} features seleccionadas — SFI demasiado agresivo"
    return f"{len(feats)} features"


# ═══════════════════════════════════════════════════════════
#  SECCION 6: PATRONES ANTI-LEAKAGE EN CODIGO (7 tests)
# ═══════════════════════════════════════════════════════════
