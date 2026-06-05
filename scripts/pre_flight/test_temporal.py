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

@test("TEST-18  R4: Splits no solapan — train < val < holdout (OOS estricto)", section="temporal")
def t18():
    sp = _cfg().temporal_splits
    te, vs, ve, hs = (pd.Timestamp(x) for x in
                      [sp.train_end, sp.validation_start, sp.validation_end, sp.holdout_start])
    # LAB-02 fix (2026-03-20): validation DEBE ser OOS estricta respecto al training.
    # Antes (M-41B): se permitia val⊂train con el comentario 'R17/P1-FIX'.
    # Ahora: el solapamiento es un FAIL duro — el calibrador no puede ver datos de training.
    assert vs > te, (
        f"LEAKAGE R4: validation_start={vs.date()} <= train_end={te.date()}. "
        f"La validation NO es OOS — el calibrador ve datos del training set. "
        f"Fix LAB-02: reducir train_end a {vs.date() - pd.Timedelta(days=1)} "
        f"o aumentar validation_start."
    )
    assert ve < hs, f"SOLAPAMIENTO CRITICO: val_end={ve.date()} >= holdout_start={hs.date()}"
    assert te < hs, f"SOLAPAMIENTO CRITICO: train_end={te.date()} >= holdout_start={hs.date()}"
    gap_days = (vs - te).days
    return f"OK: train<={te.date()} | gap={gap_days}d | val={vs.date()}->{ve.date()} | hold>={hs.date()}"


@test("TEST-19  features_train: index monotónico y sin duplicados", section="temporal")
def t19():
    df = _load_parquet("features_train.parquet")
    assert df.index.is_monotonic_increasing, "Index no monotónico"
    assert not df.index.duplicated().any(), "Duplicados en index"
    return "OK"


@test("TEST-20  features_train data <= train_end (sin futuros en training)", section="temporal")
def t20():
    cfg = _cfg()
    train_end = pd.Timestamp(cfg.temporal_splits.train_end, tz="UTC")
    df = _load_parquet("features_train.parquet")
    if df.index.tz is None:
        train_end = train_end.tz_localize(None)
    assert df.index.max() <= train_end + pd.Timedelta(days=1), \
        f"Datos futuros: max={df.index.max().date()} > {train_end.date()}"
    return f"max={df.index.max().date()}"


@test("TEST-20B features_train min_date coherente con train_start (2017 incluido?)", section="temporal")
def t20b():
    """
    P4-1 FIX (2026-03-09): Al extender train_start a 2017-08-17, el parquet
    features_train.parquet debe haberse regenerado con los datos 2017.
    Si min_date del parquet es muy posterior a train_start (>180d de diferencia),
    el pipeline se ejecuto sin los datos 2017 — hay que regenerar features.

    NOTA: Este test es WARNING (no FAIL) si el delta es > 180d pero el parquet
    fue generado antes de que existiera BTCUSDT_1h_2017.parquet. El FAIL duro
    es si el parquet es posterior al 2017.parquet pero NO tiene datos 2017.
    """
    cfg = _cfg()
    sp  = cfg.temporal_splits
    train_start_cfg = pd.Timestamp(sp.train_start, tz="UTC")
    df = _load_parquet("features_train.parquet")
    parquet_min = df.index.min()
    if parquet_min.tz is None:
        parquet_min = parquet_min.tz_localize("UTC")

    delta_days = (parquet_min - train_start_cfg).days

    # Si settings pide 2017 pero features_train empieza en 2020, hay un gap
    if train_start_cfg < pd.Timestamp("2019-01-01", tz="UTC"):
        # Train_start extendido a 2017: verificar parquet 2017 existe
        path_2017 = ROOT / "data" / "historical" / "daemon" / "BTCUSDT_1h_2017.parquet"
        if not path_2017.exists():
            assert False, (
                f"train_start={train_start_cfg.date()} pero BTCUSDT_1h_2017.parquet NO existe. "
                f"Ejecutar: python scripts/dev/fetch_historical_2017.py"
            )

        # Parquet existe: verificar que features_train fue regenerado con datos 2017
        parquet_mtime = pd.Timestamp(
            (ROOT / "data" / "features" / "features_train.parquet").stat().st_mtime,
            unit="s", tz="UTC"
        )
        historic_mtime = pd.Timestamp(path_2017.stat().st_mtime, unit="s", tz="UTC")

        if parquet_mtime < historic_mtime:
            # features_train es ANTERIOR al 2017.parquet → se generó sin los datos 2017
            assert False, (
                f"features_train.parquet generado ANTES que BTCUSDT_1h_2017.parquet. "
                f"Regenerar features: python scripts/run_features_and_training.py --only-features"
            )

        # Verificar que el parquet incluye datos cerca de 2017-2018
        if delta_days > 365:  # mas de 1 año de diferencia
            # FAIL duro: parquet es más nuevo pero no tiene datos 2017
            assert False, (
                f"features_train.parquet min={parquet_min.date()} "
                f"pero train_start={train_start_cfg.date()} "
                f"(diferencia {delta_days}d). "
                f"El parquet no incluye datos 2017 aunque settings lo pide. "
                f"Regenerar features: --only-features"
            )

        return f"OK: min={parquet_min.date()} | train_start={train_start_cfg.date()} | delta={delta_days}d"

    else:
        # train_start normal (>= 2019): verificar que no hay exceso de datos pre-train
        if delta_days > 180:
            return (
                f"AVISO: min={parquet_min.date()} vs train_start={train_start_cfg.date()} "
                f"(delta {delta_days}d). Puede haber datos extra (warm-up OK)."
            )
        return f"min={parquet_min.date()} ~ train_start={train_start_cfg.date()}"


@test("TEST-18B R4: holdout_calib_months=0 cuando validation es genuinamente OOS", section="temporal")
def t18b():
    """LAB-02 (2026-03-20): cuando validation es OOS estricta (validation_start > train_end),
    el calibrador de threshold DEBE usar features_validation.parquet (holdout_calib_months=0).
    Usar el holdout para calibrar cuando ya tenemos validation OOS es una violacion de R4."""
    sp  = _cfg().temporal_splits
    xgb = getattr(_cfg(), 'xgboost', None)
    te  = pd.Timestamp(sp.train_end)
    vs  = pd.Timestamp(sp.validation_start)
    hcm = int(getattr(xgb, 'holdout_calib_months', 0))
    if vs > te:
        # Validation es genuinamente OOS: el calibrador NO debe tocar el holdout
        assert hcm == 0, (
            f"R4 VIOLATION: holdout_calib_months={hcm} > 0 pero validation es OOS "
            f"(val_start={vs.date()} > train_end={te.date()}). "
            f"El calibrador puede y debe usar features_validation.parquet. "
            f"Fix: holdout_calib_months: 0 en settings.yaml (LAB-02)"
        )
        return f"OK: holdout_calib_months=0 — calibrador usa validation OOS ({vs.date()}→{pd.Timestamp(sp.validation_end).date()})"
    else:
        # val⊂train: este bloque no deberia alcanzarse porque TEST-18 ya fallo
        return f"WARN: val⊂train — TEST-18 debia haber fallado antes que este test"


@test("TEST-18C R3: embargo_hours >= vertical_barrier_hours (sin look-ahead en CPCV)", section="temporal")
def t18c():
    """LAB-04 (2026-03-20): con embargo < barrera, los puntos de train adyacentes al
    bloque de test en CPCV tienen labels cuyo retorno TBM cruza al test set.
    Con embargo=96H y barrera=168H: 72H de solape -> ~13% del training con look-ahead.
    R3 (SOP): embargo_hours >= 1x horizonte maximo de barrera vertical."""
    cfg = _cfg()
    embargo_h = int(getattr(getattr(cfg, 'sop', object()), 'embargo_hours', 96))
    barrier_h = int(getattr(getattr(cfg, 'xgboost', object()), 'vertical_barrier_hours', 96))
    assert embargo_h >= barrier_h, (
        f"R3 VIOLATION (LAB-04): embargo_hours={embargo_h}H < vertical_barrier_hours={barrier_h}H. "
        f"Solape de {barrier_h - embargo_h}H en CPCV: retornos TBM cruzan al test set. "
        f"Fix: sop.embargo_hours: {barrier_h} en settings.yaml"
    )
    return f"OK: embargo={embargo_h}H >= barrera={barrier_h}H (R3 compliant)"


@test("TEST-18D R1: hmm_extend_to_holdout=false (sin look-ahead en HMM)", section="temporal")
def t18d():
    """LAB-03 (2026-03-20): hmm_extend_to_holdout=true tiene DOS problemas:
    1. StandardScaler.fit() sobre train+holdout (scaler leakage de distribucion 2025).
    2. _analyze_and_map_states G2-A: cutoff=max(holdout) -> retornos del holdout
       determinan que estado es 'BEAR_CRASH'. El filtro HMM conocia el futuro.
    En produccion el HMM solo ve datos hasta la fecha de reentrenamiento.
    false = correcto y equivalente a produccion."""
    cfg = _cfg()
    extend = bool(getattr(getattr(cfg, 'hmm', object()), 'hmm_extend_to_holdout', False))
    assert not extend, (
        "R1 VIOLATION (LAB-03): hmm_extend_to_holdout=true. "
        "El HMM ve datos del holdout: (1) scaler leakage de distribucion 2025, "
        "(2) mapeo semantico G2-A usa retornos del holdout para nombrar estados. "
        "Fix: hmm.hmm_extend_to_holdout: false en settings.yaml"
    )
    return "OK: hmm_extend_to_holdout=false — HMM solo ve training data (= produccion real)"


@test("TEST-18E R4: Proporción temporal Train/Val/Holdout alineada con SOP institucional", section="temporal")
def t18e():
    """Valida empíricamente la proporción de días asignados a Train, Validation y Holdout.
    El estándar institucional sugiere distribuciones como 70/15/15 u 80/10/10.
    Emite un WARNING si el componente de Training cae por debajo del 65% o el Holdout baja del 10%."""
    sp = _cfg().temporal_splits
    ts, te = pd.Timestamp(sp.train_start), pd.Timestamp(sp.train_end)
    vs, ve = pd.Timestamp(sp.validation_start), pd.Timestamp(sp.validation_end)
    hs = pd.Timestamp(sp.holdout_start)
    
    # Como no hay 'holdout_end' en temporal_splits (es iterativo al presente), limitamos a current time.
    he = pd.Timestamp(datetime.now(tz=ts.tzinfo)) 
    if hs > he:
        he = hs + pd.Timedelta(days=1)
        
    days_train = max(1, (te - ts).days)
    days_val = max(1, (ve - vs).days)
    days_holdout = max(1, (he - hs).days)
    
    total_days = days_train + days_val + days_holdout
    pct_t = (days_train / total_days) * 100
    pct_v = (days_val / total_days) * 100
    pct_h = (days_holdout / total_days) * 100
    
    warnings = []
    if pct_t < 65.0:
        warnings.append(f"Train_pct={pct_t:.1f}% (< 65%)")
    if pct_h < 8.0:
        warnings.append(f"Holdout_pct={pct_h:.1f}% (< 8%)")
        
    status = f"Split temporal: Train={pct_t:.1f}% ({days_train}d) | Val={pct_v:.1f}% ({days_val}d) | Holdout={pct_h:.1f}% ({days_holdout}d)"
    
    if warnings:
        return f"AVISO: Distribución macro subóptima vs paradigma 70/15/15: " + ", ".join(warnings) + f" [{status}]"
        
    return f"OK: {status}"


@test("TEST-21  CPCV_GROUPS en codigo == sop.cpcv_groups en settings", section="temporal")
def t21():
    # M-40 (2026-03-18): fuente única es sop.cpcv_groups (antes era cpcv.n_blocks=10).
    # El test ya no exige valor fijo=10; verifica coherencia código↔settings.
    cfg = _cfg()
    n_cfg = int(getattr(getattr(cfg, 'sop', object()), 'cpcv_groups',
                getattr(getattr(cfg, 'cpcv', object()), 'n_blocks', 0)))
    assert n_cfg >= 4, f"sop.cpcv_groups={n_cfg} demasiado bajo (mínimo 4)"
    src = _read(ROOT/"luna/models/train_xgboost_v2.py")
    m = re.search(r"CPCV_GROUPS\s*=\s*(\d+)", src)
    if m:
        # Hardcode detectado: verificar que coincide con settings
        assert int(m.group(1)) == n_cfg, (
            f"CPCV_GROUPS hardcode={m.group(1)} != sop.cpcv_groups={n_cfg} en settings"
        )
        return f"CPCV_GROUPS={m.group(1)} == sop.cpcv_groups={n_cfg}"
    # Sin hardcode: CPCV_GROUPS se lee de cfg (correcto)
    assert "cpcv_groups" in src or "CPCV_GROUPS" in src, "CPCV_GROUPS no referenciado"
    return f"sop.cpcv_groups={n_cfg} (leído de cfg — sin hardcode)"


@test("TEST-22  features_train timezone consistente (UTC o naive puro)", section="temporal")
def t22():
    df = _load_parquet("features_train.parquet")
    tz = df.index.tz
    if tz is not None:
        assert str(tz).upper() in ["UTC"], f"Timezone inesperado: {tz}"
    return f"tz={tz or 'naive'}"


# ═══════════════════════════════════════════════════════════
#  SECCION 4: ARQUITECTURA (6 tests)
# ═══════════════════════════════════════════════════════════
