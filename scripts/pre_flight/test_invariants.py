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

@test("TEST-120 BUG-01: stat.n_trials_total == xgboost.optuna_trials (DSR correcto)", section="invariants")
def t120():
    """
    BUG-01 (2026-03-17): statistical_audit.py usaba n_trials_total=600 para calcular
    SR* en el DSR, pero Optuna solo corre optuna_trials=100 por run.
    Resultado: DSR artificialmente suprimido 0.15-0.25 puntos en cada run.
    Fix: statistical_audit.py ahora lee xgboost.optuna_trials directamente.
    Este test verifica que ambos valores están sincronizados en settings.yaml.
    """
    cfg = _cfg()
    n_stat   = getattr(getattr(cfg, 'stat',    None), 'n_trials_total', None)
    n_optuna = getattr(getattr(cfg, 'xgboost', None), 'optuna_trials',  None)

    if n_stat is None:
        return "SKIP -- stat.n_trials_total no en settings (puede estar eliminado tras BUG-01 fix)"

    if n_optuna is None:
        return "SKIP -- xgboost.optuna_trials no en settings.yaml"

    # Verificar que statistical_audit.py ya NO lee stat.n_trials_total
    src_audit = _read(ROOT / "luna/monitoring/statistical_audit.py")
    reads_old = "n_trials_total" in src_audit and "stat" in src_audit and "optuna_trials" not in src_audit
    if reads_old:
        assert False, (
            "BUG-01 REGRESION: statistical_audit.py aun lee stat.n_trials_total "
            f"({n_stat}) en vez de xgboost.optuna_trials ({n_optuna}). "
            "El DSR usara el valor incorrecto."
        )

    # Verificar que el auditor lee desde xgboost.optuna_trials
    assert "optuna_trials" in src_audit, (
        "BUG-01 FIX no encontrado: statistical_audit.py no lee xgboost.optuna_trials. "
        "La funcion _load_n_trials_from_settings() debe usar cfg['xgboost']['optuna_trials']."
    )

    return f"OK -- statistical_audit lee xgboost.optuna_trials={n_optuna} (BUG-01 fix activo)"



@test("TEST-121 BUG-02: oos_trades guardado con DatetimeIndex (WFV con fechas reales)", section="invariants")
def t121():
    """
    BUG-02 (2026-03-17): predict_oos.py guardaba oos_trades.parquet
    con index=False -> indice entero 0,1,2 -> _run_wfv() reportaba start=0, end=12
    en vez de fechas reales.
    Fix: guardar con index=True, usando timestamp como DatetimeIndex.
    """
    src_oos = _read(ROOT / "luna/models/predict_oos.py")

    # Verificar que no guarda con index=False (antiguo bug)
    has_old_bug = bool(re.search(r'to_parquet\s*\([^)]*index\s*=\s*False', src_oos))
    if has_old_bug:
        assert False, (
            "BUG-02 REGRESION: oos_trades.parquet se guarda con index=False. "
            "El WFV obtendra un indice entero 0,1,2 en vez de DatetimeIndex. "
            "Cambiar a df_trades.to_parquet(out_path, index=True) con timestamp como indice."
        )

    # Verificar que guarda con index=True
    assert "index=True" in src_oos or 'set_index("timestamp")' in src_oos, (
        "BUG-02 fix no encontrado: oos_trades.parquet debe guardarse con "
        "timestamp como DatetimeIndex (index=True)."
    )

    # Verificar que _run_wfv tiene el guard de DatetimeIndex
    src_val = _read(ROOT / "scripts/run_statistical_validation.py")
    assert "isinstance(trades_df.index, pd.DatetimeIndex)" in src_val, (
        "BUG-02 fix parcial: _run_wfv() no tiene guard de DatetimeIndex. "
        "Anadir: if not isinstance(trades_df.index, pd.DatetimeIndex): usar columna timestamp."
    )

    # Verificar que las claves del WFV usan start_date/end_date (no start/end)
    assert "start_date" in src_val, (
        "BUG-02 fix parcial: _run_wfv() usa claves 'start'/'end' en vez de "
        "'start_date'/'end_date'. Actualizar para claridad."
    )

    return "OK -- oos_trades con DatetimeIndex, _run_wfv con guard y claves start_date/end_date"



@test("TEST-122 BUG-03: threshold de emergencia configurable (no hardcode 0.45)", section="invariants")
def t122():
    """
    BUG-03 (2026-03-17): cuando n_xgb < 10, el threshold bajaba silenciosamente
    a 0.45 hardcodeado sin dejar rastro en el parquet ni en el veredicto.
    Fix: leer xgb_min_signals_threshold de settings.yaml y registrar
    threshold_was_lowered=True en cada trade_record.
    """
    src_oos  = _read(ROOT / "luna/models/predict_oos.py")
    src_yaml = _read(ROOT / "config/settings.yaml")

    # El hardcode viejo no debe estar presente
    # Patron: if n_xgb < 10: seguido de SIGNAL_CUTOFF = 0.45
    old_pattern = re.search(
        r"if n_xgb\s*<\s*10\s*:\s*\n\s*SIGNAL_CUTOFF\s*=\s*0\.45",
        src_oos
    )
    assert old_pattern is None, (
        "BUG-03 REGRESION: bloque 'if n_xgb < 10: SIGNAL_CUTOFF = 0.45' hardcodeado. "
        "Reemplazar por lectura de xgb_min_signals_threshold desde settings.yaml."
    )

    # Verificar que settings.yaml tiene los parametros configurables
    assert "xgb_min_signals_count" in src_yaml, (
        "BUG-03 fix incompleto: xgb_min_signals_count ausente en settings.yaml."
    )
    assert "xgb_min_signals_threshold" in src_yaml, (
        "BUG-03 fix incompleto: xgb_min_signals_threshold ausente en settings.yaml."
    )

    # Verificar trazabilidad en trade_records
    assert "threshold_was_lowered" in src_oos, (
        "BUG-03 fix incompleto: threshold_was_lowered no se registra en trade_records. "
        "Anadir campo 'threshold_was_lowered': THRESHOLD_WAS_LOWERED en el dict de trades."
    )

    # Leer los valores configurados
    cfg = _cfg()
    xgb = getattr(cfg, 'xgboost', None)
    min_count = getattr(xgb, 'xgb_min_signals_count', None)
    min_thr   = getattr(xgb, 'xgb_min_signals_threshold', None)

    return (
        f"OK -- threshold emergencia configurable: "
        f"min_count={min_count}, min_thr={min_thr}, threshold_was_lowered registrado en parquet"
    )



@test("TEST-123 BUG-04: hmm_allowed_regimes usa etiquetas semanticas (no indices)", section="invariants")
def t123():
    """
    BUG-04 (2026-03-17): hmm_allowed_regimes: [0, 1, 2] filtraba por indice numerico.
    El HMM puede reasignar indices en cada run segun la convergencia de EM.
    Estado 0 puede ser BULL en M-35 y BEAR en M-36 -> filtro se invierte silenciosamente.
    Fix: usar etiquetas semanticas: ['1_BULL_TREND', '2_VOLATILE_BULL'].
    """
    src_yaml = _read(ROOT / "config/settings.yaml")
    src_oos  = _read(ROOT / "luna/models/predict_oos.py")
    cfg = _cfg()

    # Verificar en settings.yaml que son strings semanticos
    _hmm_cfg = getattr(cfg, 'metalabeler', None)
    _allowed  = getattr(_hmm_cfg, 'hmm_allowed_regimes', None) if _hmm_cfg else None

    if _allowed is None:
        return "SKIP -- hmm_allowed_regimes es null (pass-through, sin filtro HMM)"

    int_entries = [x for x in _allowed if isinstance(x, int)]
    assert not int_entries, (
        f"BUG-04 REGRESION: hmm_allowed_regimes contiene enteros: {int_entries}. "
        f"Los indices HMM son inestables entre runs. "
        f"Usar etiquetas semanticas: ['1_BULL_TREND', '2_VOLATILE_BULL', ...]."
    )

    # Verificar que el codigo de filtro HMM en generate_oos ya no usa isin(lista_de_int)
    # y tiene el guard de retrocompatibilidad
    assert "isinstance(x, int)" in src_oos or "BUG-04" in src_oos, (
        "BUG-04 fix no encontrado en predict_oos.py. "
        "El filtro HMM debe comparar etiquetas semanticas, no indices numericos."
    )

    valid_prefixes = ["1_BULL", "2_VOLATILE", "3_BEAR", "4_CALM"]
    invalid = [x for x in _allowed if not any(x.startswith(p) for p in valid_prefixes)]
    if invalid:
        return (
            f"WARN -- hmm_allowed_regimes contiene etiquetas no estandar: {invalid}. "
            f"Etiquetas estandar: 1_BULL_TREND, 2_VOLATILE_BULL, 3_BEAR_CRASH, 4_CALM_BEAR."
        )

    return f"OK -- hmm_allowed_regimes={_allowed} (etiquetas semanticas, BUG-04 fix activo)"



@test("TEST-124 INVARIANTS: pipeline_invariants.py importable (Capa 1 + Capa 2)", section="invariants")
def t124():
    """
    Verifica que el modulo pipeline_invariants.py existe y es importable,
    y que sus funciones principales estan presentes.
    Si este test falla, ningun check de trazabilidad funcionara durante el pipeline.
    """
    inv_path = ROOT / "luna/utils/pipeline_invariants.py"
    assert inv_path.exists(), (
        "pipeline_invariants.py no encontrado en luna/utils/. "
        "Este modulo es el sistema de trazabilidad del pipeline. "
        "Crear: luna/utils/pipeline_invariants.py"
    )

    src = _read(inv_path)

    # Capa 1 — config checks
    assert "check_config_consistency" in src, (
        "check_config_consistency() ausente en pipeline_invariants.py. "
        "Esta funcion es la Capa 1 del sistema de trazabilidad (pre-flight)."
    )

    # Capa 2 — runtime checks
    assert "check_trades_df" in src, (
        "check_trades_df() ausente en pipeline_invariants.py. "
        "Esta funcion es la Capa 2 del sistema de trazabilidad (runtime trades)."
    )
    assert "check_oos_df" in src, (
        "check_oos_df() ausente en pipeline_invariants.py. "
        "Esta funcion es la Capa 2 del sistema de trazabilidad (runtime features OOS)."
    )

    # Verificar que check_config_consistency detecta BUG-01 y BUG-04
    assert "n_trials_total" in src and "optuna_trials" in src, (
        "check_config_consistency() no verifica BUG-01 (n_trials mismatch)."
    )
    assert "hmm_allowed_regimes" in src and "isinstance(x, int)" in src, (
        "check_config_consistency() no verifica BUG-04 (hmm_allowed_regimes enteros)."
    )

    # Verificar que esta llamado en algun script del pipeline (integracion real)
    src_val = _read(ROOT / "scripts/run_statistical_validation.py")
    src_oos = _read(ROOT / "luna/models/predict_oos.py")
    integrated = "pipeline_invariants" in src_val or "pipeline_invariants" in src_oos
    if not integrated:
        return (
            "WARN -- pipeline_invariants.py existe pero no esta importado en "
            "run_statistical_validation.py ni predict_oos.py. "
            "Integrar: from luna.utils.pipeline_invariants import check_trades_df"
        )

    return "OK -- pipeline_invariants.py importable, Capa 1 y Capa 2 presentes e integradas"



@test("TEST-125 ARCH-01: run_gauntlet usa holdout_hours (no n_trades) para n_obs en DSR", section="invariants")
def t125():
    """
    ARCH-01 fix (2026-03-17): run_gauntlet debe pasar n_obs=holdout_hours a _compute_dsr,
    NO n_obs=total_trades. Con n_obs=53 (trades tipicos) el DSR siempre es ~0 incluso con
    señal real. Con n_obs=8760 (horas holdout) el DSR es calibrado correctamente.
    Este test verifica estáticamente que el código del Gauntlet usa la variable holdout_hours
    y no pasa total_trades como n_obs al DSR.
    """
    audit_path = ROOT / "luna/monitoring/statistical_audit.py"
    assert audit_path.exists(), "statistical_audit.py no encontrado"
    src = _read(audit_path)

    # El fix debe declarar holdout_hours como variable
    assert "holdout_hours" in src, (
        "ARCH-01 NO aplicado: statistical_audit.run_gauntlet debe calcular holdout_hours "
        "desde los timestamps reales del trades_df y pasarlo como n_obs a _compute_dsr. "
        "Fix: if 'timestamp' in trades_df.columns: ts=pd.to_datetime(...); holdout_hours=... "
        "Ver implementation_plan.md ARCH-01."
    )

    # El call a _compute_dsr debe usar n_obs=holdout_hours, no n_obs=total_trades
    lines = src.split("\n")
    dsr_call_lines = [l for l in lines if "_compute_dsr(" in l or ("n_obs=" in l and "dsr" in l.lower())]
    for line in dsr_call_lines:
        assert "n_obs=total_trades" not in line, (
            f"ARCH-01 INCOMPLETO: linea '{line.strip()}' sigue usando n_obs=total_trades. "
            f"Debe ser n_obs=holdout_hours — ver implementation_plan.md ARCH-01."
        )

    # El veredicto JSON debe incluir n_obs_dsr para trazabilidad
    assert "n_obs_dsr" in src, (
        "ARCH-01 TRAZABILIDAD: statistical_verdict.json debe incluir 'n_obs_dsr' para "
        "que se pueda auditar qué n_obs se uso en el calculo DSR. "
        "Añadir: 'n_obs_dsr': int(holdout_hours) en el bloque statistical_audit del verdict."
    )

    return "OK -- ARCH-01 activo: run_gauntlet usa n_obs=holdout_hours, n_obs_dsr en verdict"



@test("TEST-126 ARCH-02: sample_weights con decaimiento exponencial configurable", section="invariants")
def t126():
    """
    ARCH-02 fix (2026-03-17): _compute_sample_weights debe usar decaimiento exponencial
    exp(-alpha * years_ago) en lugar del escalon rigido 5x/2x/1x.
    El parametro alpha debe ser configurable desde settings.yaml -> xgboost.weight_decay_alpha.
    """
    # 1. settings.yaml debe tener weight_decay_alpha
    cfg = _cfg()
    xgb_cfg = getattr(cfg, "xgboost", None)
    alpha = getattr(xgb_cfg, "weight_decay_alpha", None)
    assert alpha is not None, (
        "ARCH-02 NO aplicado: 'weight_decay_alpha' ausente en settings.yaml bajo xgboost. "
        "Anadir: weight_decay_alpha: 0.5  (0.0=uniforme, 0.5=suave, 1.6=equivalente anterior 5x)"
    )
    assert isinstance(alpha, (int, float)) and 0.0 <= float(alpha) <= 5.0, (
        f"ARCH-02: weight_decay_alpha={alpha} fuera de rango esperado [0.0, 5.0]. "
        f"Valor recomendado: 0.5"
    )

    # 2. _compute_sample_weights debe usar np.exp (decaimiento exponencial)
    src = _read(ROOT / "luna/models/train_xgboost_v2.py")
    assert "np.exp" in src or "math.exp" in src, (
        "ARCH-02 NO aplicado en codigo: _compute_sample_weights debe usar np.exp() "
        "para el decaimiento exponencial. Reemplazar el bloque 5x/2x/1x."
    )

    # 3. No debe haber el escalon duro hardcodeado (5.0 como peso fijo)
    # Buscar el patron de asignacion directa que reemplazamos
    import re
    hard_pattern = re.search(r"weights\[.*\]\s*=\s*5\.0", src)
    assert hard_pattern is None, (
        "ARCH-02 INCOMPLETO: sigue existiendo asignacion de peso fijo '= 5.0' en "
        "_compute_sample_weights. Reemplazar por formula exp(-alpha * years_ago)."
    )

    # 4. Verificar que weight_decay_alpha es leido en la funcion
    assert "weight_decay_alpha" in src, (
        "ARCH-02: weight_decay_alpha no se lee en train_xgboost_v2.py. "
        "Anadir: _alpha = float(_cfg_sw.xgboost.weight_decay_alpha)"
    )

    return (
        f"OK -- ARCH-02 activo: sample_weights exp(-{float(alpha):.2f} * years_ago) "
        f"[ano0={1.0:.3f}, ano-1={float(alpha)**0:.3f}... configurable en settings.yaml]"
    )



@test("TEST-127 BUG-05: btc_cycle_position calculada una sola vez via funcion centralizada", section="invariants")
def t127():
    """
    BUG-05 fix (2026-03-17): btc_cycle_position se calculaba dos veces en
    predict_oos.py con codigo identico (bloque R21 + LEGACY-04 GUARD).
    Fix: extraida a _calc_btc_cycle_position() y ambos bloques la llaman.
    """
    src = _read(ROOT / "luna/models/predict_oos.py")

    # 1. La funcion centralizada debe existir
    assert "_calc_btc_cycle_position" in src, (
        "BUG-05 NO aplicado: _calc_btc_cycle_position ausente en predict_oos.py. "
        "Esta funcion centraliza el calculo de btc_cycle_position (percentil rolling 365d). "
        "Ver implementation_plan.md BUG-05."
    )

    # 2. El GUARD ya no debe tener el bloque de rolling duplicado inline
    # Si el GUARD tiene 'rolling(window=8760' propio (no en la funcion), es duplicado
    guard_section = src.split("LEGACY-04 GUARD")
    if len(guard_section) > 1:
        guard_body = guard_section[1][:500]  # primeras 500 chars del guard
        assert "rolling(window=8760" not in guard_body, (
            "BUG-05 INCOMPLETO: el LEGACY-04 GUARD sigue teniendo rolling(window=8760) inline. "
            "El GUARD debe llamar a self._calc_btc_cycle_position(), no recalcular."
        )

    # 3. El bloque R21 tambien debe usar la funcion
    r21_section = src.split("R21")
    if len(r21_section) > 1:
        r21_body = r21_section[1][:1000]
        assert "_calc_btc_cycle_position" in r21_body or "btc_cycle_position" not in r21_body, (
            "BUG-05: el bloque R21 no llama a _calc_btc_cycle_position. "
            "Reemplazar el calculo inline por self._calc_btc_cycle_position(df_oos['close'], self.root)."
        )

    return "OK -- BUG-05 activo: _calc_btc_cycle_position centralizada, GUARD sin rolling duplicado"


# ─────────────────────────────────────────────────────────





@test("TEST-129 ARCH-03: warning CPCV robustez baja y metalabeler sincronizado", section="invariants")
def t129():
    """
    ARCH-03 fix (2026-03-17):
    1. _create_cpcv_splits debe emitir warning cuando CPCV_GROUPS < 8 (< 28 paths).
    2. metalabeler.n_cpcv_groups debe estar sincronizado con xgboost.n_purged_splits.
    3. settings.yaml debe documentar la tabla de referencia temporal de CPCV.
    """
    # 1. El warning de robustez baja debe existir en el codigo
    src = _read(ROOT / "luna/models/train_xgboost_v2.py")
    assert "ARCH-03" in src, (
        "ARCH-03 NO aplicado: falta el bloque warning 'ARCH-03' en train_xgboost_v2.py. "
        "Anadir warning en _create_cpcv_splits cuando CPCV_GROUPS < 8."
    )
    assert "ROBUSTEZ BAJA" in src or "robustez" in src.lower(), (
        "ARCH-03 incompleto: el warning no menciona robustez baja. "
        "El usuario debe saber que < 28 paths implica menor confianza estadistica."
    )

    # 2. metalabeler.n_cpcv_groups debe ser igual a xgboost.n_purged_splits
    cfg = _cfg()
    xgb_groups = getattr(getattr(cfg, 'xgboost', None), 'n_purged_splits', None)
    meta_groups = getattr(getattr(cfg, 'metalabeler', None), 'n_cpcv_groups', None)
    if xgb_groups is not None and meta_groups is not None:
        assert int(xgb_groups) == int(meta_groups), (
            f"ARCH-03 INCONSISTENCIA: xgboost.n_purged_splits={xgb_groups} != "
            f"metalabeler.n_cpcv_groups={meta_groups}. "
            f"Ambos deben ser iguales para coherencia IS/OOS. "
            f"Sincronizar en settings.yaml."
        )

    # 3. settings.yaml debe tener la tabla de referencia temporal
    src_yaml = _read(ROOT / "config/settings.yaml")
    assert "BajaRobustez" in src_yaml or "Baja Robustez" in src_yaml or "12.4H" in src_yaml, (
        "ARCH-03: falta tabla de referencia temporal en settings.yaml bajo n_purged_splits. "
        "Anadir comentario con ETAs para n=6/8/10 grupos."
    )

    return (
        f"OK -- ARCH-03: warning robustez en codigo, "
        f"xgboost.n_purged_splits={xgb_groups}==metalabeler.n_cpcv_groups={meta_groups}"
    )



@test("TEST-130 ARCH-04: calibracion threshold con jerarquia holdout-first", section="invariants")
def t130():
    """
    ARCH-04 fix (2026-03-17): _calibrate_threshold usaba siempre features_validation.parquet
    (2024-H2 semi-conocido). Fix: jerarquia holdout-first con holdout_calib_months meses.
    """
    src = _read(ROOT / "luna/models/train_xgboost_v2.py")

    # 1. El codigo debe tener el bloque ARCH-04
    assert "ARCH-04" in src, (
        "ARCH-04 NO aplicado: falta bloque 'ARCH-04' en _calibrate_threshold de train_xgboost_v2.py. "
        "Implementar jerarquia holdout-first para calibracion de threshold."
    )

    # 2. holdout_path debe ser verificado antes que val_path
    assert "holdout_path" in src, (
        "ARCH-04 incompleto: 'holdout_path' no encontrado en _calibrate_threshold. "
        "Anadir ruta a features_holdout.parquet como primera opcion de calibracion."
    )

    # 3. cal_source debe persistirse en el JSON de la firma
    assert "cal_source" in src, (
        "ARCH-04 trazabilidad: 'cal_source' no guardado en xgboost_meta_signature.json. "
        "Anadir cal_source al dict de firma para saber que periodo se uso en calibracion."
    )

    # 4. holdout_calib_months debe estar en settings.yaml
    cfg = _cfg()
    calib_months = getattr(getattr(cfg, 'xgboost', None), 'holdout_calib_months', None)
    assert calib_months is not None, (
        "ARCH-04: holdout_calib_months ausente en settings.yaml bajo xgboost. "
        "Anadir: holdout_calib_months: 3  (0=desactivar, 3=primeros 3 meses del holdout)"
    )

    return f"OK -- ARCH-04: calibracion holdout-first activa (holdout_calib_months={calib_months}), cal_source en firma"



@test("TEST-131 ARCH-05: monitor drift JSD regimenes HMM IS vs OOS", section="invariants")
def t131():
    """
    ARCH-05 fix (2026-03-17): HMM entrenado en IS aplicado a OOS sin rolling retrain
    (Fix F4 documenta por que no es viable). Fix: monitor JSD2 de drift de distribucion
    de regimenes IS→OOS. Si JSD2 > hmm.drift_alert_jsd emite WARNING.
    """
    src = _read(ROOT / "luna/models/hmm_regime.py")

    # 1. El bloque ARCH-05 debe existir en hmm_regime.py
    assert "ARCH-05" in src, (
        "ARCH-05 NO aplicado: falta bloque 'ARCH-05' en hmm_regime.py. "
        "Anadir monitor JSD2 de drift en generate_oos_features()."
    )

    # 2. jensenshannon debe ser importado en el bloque
    assert "jensenshannon" in src, (
        "ARCH-05 incompleto: 'jensenshannon' no encontrado en hmm_regime.py. "
        "El calculo JSD2 requiere scipy.spatial.distance.jensenshannon."
    )

    # 3. drift_alert_jsd debe estar en settings.yaml bajo hmm:
    cfg = _cfg()
    hmm_cfg = getattr(cfg, 'hmm', None)
    jsd_thr = getattr(hmm_cfg, 'drift_alert_jsd', None)
    assert jsd_thr is not None, (
        "ARCH-05: hmm.drift_alert_jsd ausente en settings.yaml. "
        "Anadir: drift_alert_jsd: 0.15  (JSD2 umbral, rango [0,1])"
    )
    assert 0.0 < float(jsd_thr) < 1.0, (
        f"ARCH-05: drift_alert_jsd={jsd_thr} fuera de rango (0,1). Valor recomendado: 0.15"
    )

    # 4. _regime_drift_jsd debe persistirse en self
    assert "_regime_drift_jsd" in src, (
        "ARCH-05 trazabilidad: self._regime_drift_jsd no asignado en hmm_regime.py. "
        "Anadir: self._regime_drift_jsd = _jsd2  para persistir el JSD2 calculado."
    )

    return f"OK -- ARCH-05: monitor JSD2 activo (drift_alert_jsd={jsd_thr}), _regime_drift_jsd persistido"



@test("TEST-132 REPRO-01: Optuna TPESampler con seed determinista", section="invariants")
def t132():
    """
    REPRO-01 fix (2026-03-17): Optuna.create_study sin sampler explicito es no determinista.
    Dos runs con los mismos datos pueden producir hiperparametros XGBoost distintos.
    Fix: TPESampler(seed=optuna_seed) donde optuna_seed se lee de settings.yaml.
    """
    src = _read(ROOT / "luna/models/train_xgboost_v2.py")
    assert "TPESampler" in src, (
        "REPRO-01 NO aplicado: TPESampler ausente en train_xgboost_v2.py. "
        "Anadir: _sampler = optuna.samplers.TPESampler(seed=_optuna_seed) "
        "y sampler=_sampler en create_study()."
    )
    assert "REPRO-01" in src, (
        "REPRO-01: comentario de fix ausente en train_xgboost_v2.py."
    )
    cfg = _cfg()
    xgb_seed = getattr(getattr(cfg, 'xgboost', None), 'optuna_seed', None)
    assert xgb_seed is not None, (
        "REPRO-01: xgboost.optuna_seed ausente en settings.yaml. "
        "Anadir: optuna_seed: 42"
    )
    return f"OK -- REPRO-01: Optuna TPESampler(seed={xgb_seed}) activo — runs deterministas"



@test("TEST-133 REPRO-02: warning de cache stale cuando --skip-sfi", section="invariants")
def t133():
    """
    REPRO-02 fix (2026-03-17): cuando --skip-sfi, selected_features.json de un run previo
    se reutiliza silenciosamente aunque los datos hayan cambiado.
    Fix: comparar mtime del JSON contra features_train.parquet y emitir WARNING si es stale.
    """
    src = _read(ROOT / "scripts/train_production_ensemble.py")
    assert "REPRO-02" in src, (
        "REPRO-02 NO aplicado: 'REPRO-02' ausente en train_production_ensemble.py. "
        "Anadir comparacion mtime de selected_features.json vs features_train.parquet "
        "en el bloque skip_sfi."
    )
    assert "CACHE STALE" in src or "st_mtime" in src, (
        "REPRO-02 incompleto: no se detecta comparacion mtime en train_production_ensemble.py. "
        "Verificar que el bloque compara _sel_mtime < _train_mtime."
    )
    return "OK -- REPRO-02: stale-cache warning activo cuando --skip-sfi con JSON anterior al parquet"


@test("TEST-134 ARCH-06: btc_cycle_position estable en holdout parquet (no se recalcula)", section="invariants")
def t134():
    """
    ARCH-06 fix (2026-03-17): feature_pipeline.py (Paso 7B/P4-1-1) calcula y persiste
    btc_cycle_position en features_holdout.parquet. predict_oos.py debe
    reutilizarla directamente (no recalcular desde features_train.parquet completo).
    Este test verifica:
      1. btc_cycle_position existe en features_holdout.parquet con rango [0,1] y < 5% NaN.
      2. predict_oos.py tiene el guard ARCH-06 (no recalcula si ya existe).
    """
    # -- Check 1: predict_oos.py tiene el guard ARCH-06 ───────────
    src_oos = _read(ROOT / "luna/models/predict_oos.py")
    assert "ARCH-06" in src_oos, (
        "ARCH-06 NO aplicado: falta guard 'ARCH-06' en predict_oos.py. "
        "El bloque M08a debe reutilizar btc_cycle_position si ya existe en df_oos "
        "(feature_pipeline.py la calcula en Paso 7B y la guarda en holdout parquet)."
    )
    assert "_cycle_present" in src_oos, (
        "ARCH-06: '_cycle_present' no encontrado en predict_oos.py. "
        "El guard debe verificar si btc_cycle_position ya existe antes de recalcular."
    )

    # -- Check 2: btc_cycle_position en features_holdout.parquet ───────────────
    holdout_path = ROOT / "data" / "features" / "features_holdout.parquet"
    if not holdout_path.exists():
        return (
            "SKIP: features_holdout.parquet no existe aún — "
            "ejecutar feature_pipeline.py para generar. "
            "ARCH-06 guard en predict_oos.py: OK (check 1 pasado)"
        )

    try:
        df_h = pd.read_parquet(holdout_path, columns=["btc_cycle_position"])
    except Exception as e:
        # Columna no existe en el parquet — pipeline previo no la calculó
        return (
            f"WARN: btc_cycle_position ausente en features_holdout.parquet ({e}). "
            "Re-ejecutar feature_pipeline.py — Paso 7B genera la columna. "
            "El fallback en predict_oos.py la calculará en el próximo run."
        )

    nan_pct = df_h["btc_cycle_position"].isna().mean()
    assert nan_pct < 0.05, (
        f"ARCH-06: btc_cycle_position tiene {nan_pct:.1%} NaN en holdout — "
        "requiere ≥720H de historia para el rolling 8760H. "
        "Verificar que features_holdout incluye historia desde 2024 o reduce min_periods."
    )

    vmin = df_h["btc_cycle_position"].min()
    vmax = df_h["btc_cycle_position"].max()
    assert 0.0 <= vmin and vmax <= 1.0, (
        f"ARCH-06: btc_cycle_position fuera de rango [0,1]: min={vmin:.4f}, max={vmax:.4f}. "
        "El percentil rolling debe estar siempre en [0,1] (clip aplicado en feature_pipeline)."
    )

    n_rows = len(df_h)
    return (
        f"OK -- ARCH-06: guard en predict_oos.py activo + "
        f"btc_cycle_position válida en holdout ({n_rows} filas, "
        f"NaN={nan_pct:.1%}, range=[{vmin:.3f},{vmax:.3f}])"
    )

@test("TEST-135 DRIFT-01: audit_parametros_fijos.py sin inconsistencias", "consistency")
def test_135_configuration_drift():
    import subprocess
    import sys
    from pre_flight.core import ROOT
    
    script_path = ROOT / "tools" / "diagnostics" / "audit_parametros_fijos.py"
    if not script_path.exists():
        return "WARN -- script de auditoria no encontrado"
        
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, (
        "BUG-DRIFT REGRESION: Configuration drift detectado (inconsistencia de "
        "parametros hardcodeados en multiples archivos). "
        "Revisa la salida de tools/diagnostics/audit_parametros_fijos.py"
    )
    return "Cero inconsistencias detectadas por auditoria estatica"


#  MAIN
# ─────────────────────────────────────────────────────────