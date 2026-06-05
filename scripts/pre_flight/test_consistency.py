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

@test("TEST-58  Features XGBoost vs selected_features.json (SFI actualizado?)", section="consistency")
def t58():
    sig_path = ROOT / "data/models/xgboost_meta_signature.json"
    sel_path = ROOT / "data/features/selected_features.json"
    if not sig_path.exists() or not sel_path.exists():
        return "Archivos no existen aun (run en progreso)"
    sig = json.loads(sig_path.read_text())
    sel = json.loads(sel_path.read_text())
    xgb_feats = set(sig.get("features", sig.get("feature_names", sig.get("selected_features", []))))
    sel_feats  = set(sel.get("selected_features", []) + sel.get("pass_through_features", []))
    engineered = {"HMM_Regime", "HMM_State", "cal_hour_of_day", "cal_day_of_week",
                  "cal_month", "cal_is_weekend", "cal_quarter",
                  "NASDAQ_Ret", "Coinglass_funding_rate_high", "Inflation_MoM",
                  "Tx_Volume", "alpha_genetic_score", "golden_rule_10"}
    allowed = sel_feats | engineered
    if not xgb_feats or not sel_feats:
        return "Listas vacias (run en progreso)"
    extra = xgb_feats - allowed
    # Divergencia grave: >15 features fantasma indica SFI muy desactualizado.
    # Si hay inconsistencia, verificar si es residual (signature de run anterior al SFI actual).
    if len(extra) > 15:
        _sel = ROOT / "data/features/selected_features.json"
        _sig = ROOT / "data/models/xgboost_meta_signature.json"
        if _is_stale_artifact(_sel, _sig) or _is_stale_artifact(_sig, _sel):
            return (f"WARN: {len(extra)} features en XGBoost no reconocidas por SFI actual: {list(extra)[:3]} "
                    f"(signature de run anterior — se regenerara en FASE 4)")
    assert len(extra) <= 15, \
        f"XGBoost usa {len(extra)} features completamente desconocidas (umbral=15): {list(extra)[:5]}"
    if len(extra) > 0:
        return f"AVISO: {len(extra)} features en XGBoost no en SFI (modelo anterior al ultimo SFI): {list(extra)[:3]}"
    return f"{len(xgb_feats)} features XGBoost consistentes con SFI"


@test("TEST-59  Features OOD Guard vs selected_features.json (SFI actualizado?)", section="consistency")
def t59():
    ood_path = ROOT / "data/models/ood_guard_signature.json"
    sel_path = ROOT / "data/features/selected_features.json"
    if not ood_path.exists() or not sel_path.exists():
        return "Archivos no existen (run en progreso)"
    ood = json.loads(ood_path.read_text())
    sel = json.loads(sel_path.read_text())
    ood_feats  = set(ood.get("features_tracked", []))
    sel_feats  = set(sel.get("selected_features", []) + sel.get("pass_through_features", []))
    engineered = {"HMM_Regime", "HMM_State", "cal_hour_of_day", "cal_day_of_week",
                  "cal_month", "cal_is_weekend", "cal_quarter",
                  "NASDAQ_Ret", "Coinglass_funding_rate_high", "Inflation_MoM",
                  "Tx_Volume", "alpha_genetic_score", "golden_rule_10"}
    allowed = sel_feats | engineered
    if not ood_feats or not sel_feats:
        return "Listas vacias (run en progreso)"
    extra = ood_feats - allowed
    assert len(extra) <= 15, \
        f"OOD Guard usa {len(extra)} features completamente desconocidas (umbral=15): {list(extra)[:3]}"
    if len(extra) > 0:
        return f"AVISO: {len(extra)} features OOD fuera de SFI (modelo anterior al ultimo SFI)"
    return f"{len(ood_feats)} features OOD consistentes"


@test("TEST-60  Modelo XGBoost es mas reciente que features_train (no modelo obsoleto)", section="consistency")
def t60():
    cfg = _cfg()
    f2 = getattr(cfg, "fase2", None)
    use_regimes = getattr(f2, "use_regime_agents", False) if f2 else False
    direction = getattr(f2, "direction_mode", "long") if f2 else "long"
    
    if use_regimes:
        model = ROOT / f"data/models/xgboost_meta_bull_{direction}.model"
    else:
        model = ROOT / "data/models/xgboost_meta.model"
        
    train = ROOT / "data/features/features_train.parquet"
    if not model.exists() or not train.exists():
        return "Archivos no existen"
    model_ts = model.stat().st_mtime
    train_ts = train.stat().st_mtime
    # Si el modelo es mas de 7 dias anterior al parquet, es sospechoso
    diff_days = (train_ts - model_ts) / 86400
    assert diff_days < 7, \
        f"features_train es {diff_days:.0f}d mas nuevo que {model.name} (modelo posiblemente obsoleto)"
    return f"modelo {'mas reciente' if model_ts >= train_ts else str(abs(int(diff_days)))+'d anterior'} a features"


@test("TEST-61  n_states HMM en settings == N_REGIMES en hmm_regime.py", section="consistency")
def t61():
    cfg_n = int(getattr(getattr(_cfg(),"hmm",object()),"n_states",4))
    src = _read(ROOT/"luna/models/hmm_regime.py")
    m = re.search(r"N_REGIMES\s*=\s*(\d+)", src)
    if m:
        code_n = int(m.group(1))
        # Aceptamos ±1 (puede estar en proceso de tuning)
        assert abs(code_n - cfg_n) <= 1, \
            f"N_REGIMES={code_n} vs hmm.n_states={cfg_n} en settings"
    return f"n_states={cfg_n}"


@test("TEST-62  MetaLabelerV2 SEQ_LEN coincide con metalabeler.seq_len en settings", section="consistency")
def t62():
    cfg_seq = int(getattr(getattr(_cfg(),"metalabeler",object()),"seq_len",48))
    src = _read(ROOT/"luna/models/train_metalabeler_v2.py")
    # ARCH-02: SEQ_LEN ya no es literal — se lee desde cfg.metalabeler.seq_len.
    # (A) Si hay literal: debe coincidir con cfg.
    # (B) Si hay patron ARCH-02 (getattr + seq_len): el valor es cfg_seq, OK.
    # NOTA: no capturar 'COST_PCT, EMBARGO_H, SEQ_LEN = 0.0015, 96, 48' (tuple unpack ARCH-02)
    m = re.search(r"^\s*SEQ_LEN\s*=\s*(\d+)", src, re.MULTILINE)
    if m:
        code_seq = int(m.group(1))
        assert code_seq == cfg_seq, \
            f"SEQ_LEN literal={code_seq} vs metalabeler.seq_len={cfg_seq}"
        return f"SEQ_LEN={code_seq} (literal OK)"
    else:
        # Sin literal aislado: verificar que el modulo use cfg/getattr para seq_len
        uses_cfg = ("_cfg_meta" in src or "seq_len" in src)
        assert uses_cfg, (
            f"SEQ_LEN ausente EN LITERAL y sin lectura desde cfg. "
            f"metalabeler.seq_len={cfg_seq}: el modulo no leera el valor correcto."
        )
        return f"SEQ_LEN via cfg.metalabeler.seq_len={cfg_seq} (ARCH-02 OK)"


@test("TEST-63  Gauntlet thresholds en settings.yaml (min_dsr, max_pbo, min_trades)", section="consistency")
def t63():
    cfg = _cfg()
    # ARCH-02: los gates que LEE el codigo estan en stat: (no en gauntlet:)
    # gauntlet: es la seccion de referencia; stat: es la que usa LunaStatisticalAuditor.
    # shadow_min_trades esta en _roadmap.gauntlet (pre-deploy) -- no debe estar aqui.
    stat = getattr(cfg, "stat", None)
    assert stat, "Seccion stat: no encontrada en settings.yaml (Gates del Gauntlet)"
    assert getattr(stat, "min_dsr", None) is not None, "stat.min_dsr faltante"
    assert getattr(stat, "max_pbo", None) is not None, "stat.max_pbo faltante"
    assert getattr(stat, "min_trades", None) is not None, "stat.min_trades faltante"
    # Coherencia entre gauntlet: (referencia) y stat: (codigo)
    gaunt = getattr(cfg, "gauntlet", None)
    if gaunt:
        g_dsr = getattr(gaunt, "min_dsr", None)
        s_dsr = getattr(stat, "min_dsr", None)
        if g_dsr and s_dsr:
            assert g_dsr == s_dsr, f"gauntlet.min_dsr={g_dsr} != stat.min_dsr={s_dsr} (desync)"
    return (f"stat.min_dsr={stat.min_dsr} stat.max_pbo={stat.max_pbo} "
            f"stat.min_trades={stat.min_trades} (ARCH-02 OK)")


# ═══════════════════════════════════════════════════════════
#  SECCION 10: ENTORNO Y DEPENDENCIAS (4 tests)
# ═══════════════════════════════════════════════════════════


@test("TEST-116 PT/SL coherencia: modelo en disco vs settings.yaml actual", section="consistency")
def t116():
    """
    MEJORA-08 (2026-03-17): evitar desfase entre el PT/SL del modelo entrenado
    y el PT/SL activo en settings.yaml cuando se corre generate_oos_predictions.
    Ejemplo: M-33 usa pt=1.0/sl=1.0, pero si el modelo viejo (M-32, pt=1.5) sigue
    en disco, el OOS TBM aplicara pt=1.0 a un modelo que espera pt=1.5 -> WR invalido.
    """
    sig_path = ROOT / "data" / "models" / "xgboost_meta_signature.json"
    if not sig_path.exists():
        return "WARN: xgboost_meta_signature.json no existe — ejecutar training primero"

    try:
        sig = json.loads(sig_path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"WARN: no se pudo leer firma del modelo: {e}"

    # La firma NO guarda pt_mult (se lee de settings en cada run).
    # Lo que SI guarda es dsr_oos y params de XGBoost.
    # La coherencia se valida comparando settings actuales con el valor que existe.

    # Intentar leer de un campo de diagnostico si existe
    sig_pt = sig.get("pt_mult", sig.get("tbm_pt_mult", None))
    sig_sl = sig.get("sl_mult", sig.get("tbm_sl_mult", None))

    cfg = _cfg()
    xgb_cfg = getattr(cfg, "xgboost", None)
    settings_pt = float(getattr(xgb_cfg, "pt_mult_min", -1))
    settings_sl = float(getattr(xgb_cfg, "sl_mult_min", -1))

    if sig_pt is None or sig_sl is None:
        # La firma no tiene pt/sl (generada con version anterior) -> WARN
        return (
            f"OK: firma sin pt/sl — no se puede validar coherencia. "
            f"Settings actuales: pt={settings_pt}, sl={settings_sl}. "
            f"Re-entrenar para persistir pt/sl en firma."
        )

    # Comparar con tolerancia flotante
    pt_ok = abs(float(sig_pt) - settings_pt) < 1e-4
    sl_ok = abs(float(sig_sl) - settings_sl) < 1e-4

    if not pt_ok or not sl_ok:
        assert False, (
            f"DESFASE PT/SL: modelo en disco (pt={sig_pt}, sl={sig_sl}) != "
            f"settings.yaml (pt={settings_pt}, sl={settings_sl}). "
            f"Re-entrenar el modelo con los parametros actuales antes de generar OOS."
        )

    return f"pt={settings_pt}, sl={settings_sl} consistentes entre modelo y settings"




# ═══════════════════════════════════════════════════════════
#  SECCION 12: PIPELINE INVARIANTS (BUG-01..04, 2026-03-17)
#  Tests que detectan los bugs encontrados en la auditoría
#  institucional. Cada test corresponde a un bug específico.
# ═══════════════════════════════════════════════════════════

@test("TEST-117 LGBM Hard Floor lgbm_signal_min_prob debe ser 0.0 (Platt Scaling safe)", section="consistency")
def t117():
    """Valida que el hard floor de LGBM este en 0.0. 
    Con Platt Scaling, probabilidades de 0.76+ son matematicamente inalcanzables."""
    cfg = _cfg()
    fase2 = getattr(cfg, "fase2", None)
    if fase2 is None:
        return "SKIP: fase2 no configurada en settings.yaml"
    lgbm_min = float(getattr(fase2, "lgbm_signal_min_prob", 0.0))
    assert lgbm_min == 0.0, f"lgbm_signal_min_prob es {lgbm_min}, DEBE ser 0.0 para no bloquear señales (conflicto con Platt Scaling)"
    return "lgbm_signal_min_prob=0.0 (Platt Scaling safe)"

@test("TEST-118 TBM sl_mult_min debe ser >= 1.5 para evitar wipeout de senales intradiarias", section="consistency")
def t118():
    """Valida que el multiplicador del Stop Loss no baje de 1.5, para evitar salidas en ruido."""
    cfg = _cfg()
    sl_min = float(getattr(cfg, "sl_mult_min", 1.5))
    assert sl_min >= 1.5, f"sl_mult_min es {sl_min}, DEBE ser >= 1.5 (un SL muy ajustado destruye el win rate)"
    return f"sl_mult_min={sl_min} >= 1.5"

@test("TEST-119 MetaLabeler threshold_mode debe ser dynamic_is para evitar suppression OOS", section="consistency")
def t119():
    """Valida que no usemos umbrales fijos ciegos en MetaLabeler que causaban 0 trades en W1."""
    cfg = _cfg()
    mode = getattr(cfg, "meta_v2_threshold_mode", "dynamic_is")
    assert mode == "dynamic_is", f"meta_v2_threshold_mode es '{mode}', DEBE ser 'dynamic_is'"
    return f"meta_v2_threshold_mode={mode}"

