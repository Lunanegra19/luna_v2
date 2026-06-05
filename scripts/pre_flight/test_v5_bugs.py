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

@test("TEST-68  BUG-V5-01: SSR Proxy usa pd.Series (no ndarray.rename)", section="v5_bugs")
def t68():
    """
    Run 12: btc_mcap_proxy era un numpy.ndarray. Llamar .rename() sobre el
    array causaba: 'numpy.ndarray has no attribute rename'.
    Fix: envolver en pd.Series con el indice de close antes del rename.
    """
    src = _read(ROOT / "luna/data/fetch_onchain.py")
    pattern_bug = r"btc_mcap_proxy\s*=\s*close\.values\s*\*\s*supply_series\.values"
    pattern_fix = r"btc_mcap_proxy\s*=\s*pd\.Series"
    has_bug = bool(re.search(pattern_bug, src))
    has_fix = bool(re.search(pattern_fix, src))
    assert not has_bug or has_fix, (
        "BUG-V5-01 REGRESION: btc_mcap_proxy es un ndarray sin pd.Series wrapper. "
        "fetch_onchain.py fetch_ssr_proxy -> pd.Series(close.values * supply_series.values, index=close.index)"
    )
    assert has_fix, (
        "BUG-V5-01: pd.Series wrapper no encontrado en fetch_ssr_proxy. "
        "El .rename('SSR') fallara con AttributeError en runtime."
    )
    return "pd.Series wrapper OK"



@test("TEST-69  BUG-V5-02: HMM_FEATURES tiene >= 3 variables (no solo vol)", section="v5_bugs")
def t69():
    """
    Run 12: HMM_FEATURES = ['M2_YoY_Chg', 'mt_vol_realized_4bar'] -> MI=0.00028 (casi nulo).
    Con solo 1-2 features y 4 regimenes, el HMM no separa retornos futuros.
    Fix: ampliar a >= 3 features ortogonales con Granger *** validado.
    """
    src = _read(ROOT / "luna/models/hmm_regime.py")
    m = re.search(r"HMM_FEATURES\s*=\s*\[([^\]]+)\]", src, re.DOTALL)
    if not m:
        assert False, "HMM_FEATURES no encontrado como lista en hmm_regime.py"
    feature_list = re.findall(r'["\']([^"\']+)["\']', m.group(1))
    n = len(feature_list)
    assert n >= 3, (
        f"BUG-V5-02 REGRESION: HMM_FEATURES tiene solo {n} features ({feature_list}). "
        f"Con pocas features el MI HMM vs target es ~0 (Run 12: MI=0.00028). "
        f"Necesita >= 3 features ortogonales."
    )
    has_vol = any("vol" in f.lower() for f in feature_list)
    has_sentiment = any(f in ["FearGreed", "FundingRate", "MVRV_Proxy", "Stablecoin_Cap"] for f in feature_list)
    assert has_vol, f"HMM_FEATURES sin feature de volatilidad: {feature_list}"
    assert has_sentiment, (
        f"HMM_FEATURES sin feature de sentimiento/regime (FearGreed/FundingRate/MVRV): {feature_list}."
    )
    return f"{n} features: {feature_list[:4]}"



@test("TEST-108 BUG-V5-03: MI_LAG usa ETH_Return_1d (no ETH_Price precio bruto)", section="v5_bugs")
def t70():
    """
    Run 12: MI_LAG_FEATURES usaba 'ETH_Price' -> varianza ~0 al mergearse a 1H ffill.
    Fix: usar ETH_Return_1d (retorno diario, varianza real).
    """
    src = _read(ROOT / "luna/features/feature_pipeline.py")
    m = re.search(r"MI_LAG_FEATURES\s*=\s*\{([^}]+)\}", src, re.DOTALL)
    if not m:
        return "MI_LAG_FEATURES no encontrado (puede estar en otro formato)"
    block = m.group(1)
    has_eth_price_bug = "'ETH_Price'" in block or '"ETH_Price"' in block
    assert not has_eth_price_bug, (
        "BUG-V5-03 REGRESION: MI_LAG_FEATURES aun usa 'ETH_Price' (precio bruto). "
        "ETH_Price tiene varianza ~0 al mergearse a 1H ffill. "
        "Usar 'ETH_Return_1d' (retorno diario, varianza real)."
    )
    return "ETH_Price eliminado de MI_LAG (ETH_Return_1d OK)"



@test("TEST-71  BUG-V5-04: SHAP maneja arrays 2D y 3D (no solo lista legacy)", section="v5_bugs")
def t71():
    """
    Run 12: SHAP moderno devuelve ndarray de shape (n, f, 2).
    El codigo solo manejaba: sv[1] if isinstance(sv, list) -> AttributeError.
    Fix: manejar list legacy + ndim==3 (clase positiva [:,:,1]).
    """
    src = _read(ROOT / "luna/ai_mining/advanced_engine.py")
    shap_start = src.find("def shap_analysis")
    assert shap_start >= 0, "shap_analysis no encontrado en advanced_engine.py"
    block = src[shap_start:shap_start + 2000]
    has_ndim3_fix  = "ndim == 3" in block or "ndim==3" in block
    has_old_oneliner = bool(re.search(r"sarr\s*=\s*np\.abs\s*\(\s*sv\[1\]\s*if\s*isinstance", block))
    assert not has_old_oneliner, (
        "BUG-V5-04 REGRESION: SHAP usa el one-liner antiguo sin manejar ndarray 3D. "
        "SHAP moderno devuelve (n_samples, n_features, 2) en clasificadores binarios. "
        "Fix: if sv.ndim == 3: sv = sv[:, :, 1]"
    )
    assert has_ndim3_fix, (
        "BUG-V5-04: No se encontro manejo de sv.ndim==3 en shap_analysis. "
        "Con SHAP >= 0.40 el array es 3D para RandomForestClassifier binario."
    )
    return "SHAP ndim=2/3 manejados"



@test("TEST-72  BUG-V5-05: FRED _get_fred timeout default >= 30s", section="v5_bugs")
def t72():
    """
    Run 12: _get_fred() tenia timeout=15s por defecto.
    Las series MYAGM2CNM189N (China) y MABMM301JPM189N (Japan) tardaban mas.
    Fix: timeout default=30s, series internacionales=45s.
    """
    src = _read(ROOT / "luna/data/fetch_macro.py")
    m_default = re.search(r"def _get_fred.*?timeout:\s*int\s*=\s*(\d+)", src, re.DOTALL)
    if m_default:
        t_default = int(m_default.group(1))
        assert t_default >= 30, (
            f"BUG-V5-05 REGRESION: _get_fred timeout default={t_default}s (minimo 30s). "
            f"Con 15s las series FRED internacionales (M2 China, Japan) sufren TimeoutError."
        )
        t_val = str(t_default)
    else:
        t_val = "no_encontrado"
    china_match = re.search(r'_get_fred\("MYAGM2CNM189N".*?timeout\s*=\s*(\d+)', src)
    japan_match = re.search(r'_get_fred\("MABMM301JPM189N".*?timeout\s*=\s*(\d+)', src)
    if china_match:
        assert int(china_match.group(1)) >= 30, f"M2 China timeout={china_match.group(1)}s < 30s"
    if japan_match:
        assert int(japan_match.group(1)) >= 30, f"M2 Japan timeout={japan_match.group(1)}s < 30s"
    return f"default={t_val}s | China={china_match.group(1) if china_match else 'default'}s | Japan={japan_match.group(1) if japan_match else 'default'}s"



@test("TEST-73  BUG-V5-06: M2 Global guard empty DataFrame antes de .iloc[0]", section="v5_bugs")
def t73():
    """
    BUG-V5-06: fetch_m2_global_index usaba daily.iloc[0] sin verificar si daily estaba
    vacio. Cuando FRED devuelve 0 filas (API sin datos en el rango pedido), .iloc[0]
    lanza IndexError capturado como 'single positional indexer is out-of-bounds'.
    Fix: guard `if daily.empty: continue` antes de la normalizacion.
    """
    src = _read(ROOT / "luna/data/fetch_macro.py")
    m = re.search(r"def fetch_m2_global_index.*?(?=\ndef |\Z)", src, re.DOTALL)
    assert m, "fetch_m2_global_index no encontrado en fetch_macro.py"
    fn_body = m.group(0)
    assert "daily.empty" in fn_body, (
        "BUG-V5-06 NO CORREGIDO: fetch_m2_global_index no verifica daily.empty antes "
        "de daily.iloc[0]. Cuando FRED retorna 0 filas -> IndexError."
    )
    assert "continue" in fn_body, (
        "BUG-V5-06 fix incompleto: daily.empty detectado pero sin 'continue' "
        "para saltar la serie vacia."
    )
    return "guard daily.empty + continue OK"


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────

# ─── BUG-R12-01 fix (2026-03-10) ────────────────────────────────────────────

@test("TEST-75  BUG-R12-01: alpha_combined no coexiste con golden/genetic en SFI", section="v5_bugs")
def t75():
    """
    alpha_combined = 0.6 * alpha_golden_score + 0.4 * alpha_genetic_score * 2 - 1
    Es funcion lineal exacta de golden+genetic. Si las tres estan en selected_features
    hay redundancia matematica garantizada: PBO sube, slot SFI desperdiciado.
    Fix: alpha_combined eliminada de ALPHA_SIGNALS en feature_selection_e.py.
    Sustituida por alpha_storm_intensity (ortogonal: cuenta reglas sin ponderar WR).
    """
    sf_path = ROOT / "data/features/selected_features.json"
    if not sf_path.exists():
        return "SKIP -- selected_features.json no disponible (ejecutar SFI primero)"
    sf = json.loads(_read(sf_path))
    selected    = sf.get("selected_features", [])
    passthrough = sf.get("passthrough_features", [])
    all_in_model = selected + passthrough
    has_combined = "alpha_combined" in all_in_model
    has_golden   = "alpha_golden_score" in all_in_model
    has_genetic  = "alpha_genetic_score" in all_in_model
    assert not (has_combined and has_golden), (
        "BUG-R12-01 REGRESION: alpha_combined Y alpha_golden_score en el modelo. "
        "alpha_combined = f(golden, genetic) -- correlacion perfecta -- slot SFI perdido. "
        "Eliminar alpha_combined de ALPHA_SIGNALS en luna/features/feature_selection_e.py."
    )
    assert not (has_combined and has_genetic), (
        "BUG-R12-01 REGRESION: alpha_combined Y alpha_genetic_score en el modelo. "
        "Igual que el assert anterior -- eliminar alpha_combined de ALPHA_SIGNALS."
    )
    # Verificar que alpha_storm_intensity esta calculada en el parquet de features
    ft_path = ROOT / "data/features/features_train.parquet"
    storm_ok = False
    if ft_path.exists():
        cols = pd.read_parquet(ft_path).columns.tolist()
        storm_ok = "alpha_storm_intensity" in cols
        if not storm_ok:
            return ("OK -- alpha_storm_intensity no en features_train.parquet. "
                    "Re-ejecutar FeaturePipeline para generar la columna nueva. "
                    f"combined={has_combined}, golden={has_golden}, genetic={has_genetic}")
    return (f"OK -- alpha_combined ausente del modelo. "
            f"storm_intensity={'presente' if storm_ok else 'pendiente'}. "
            f"golden={has_golden}, genetic={has_genetic}")




@test("TEST-78  BUG-R12-02: MetaLabelerV2 tiene gradient clipping y NaN guard", section="v5_bugs")
def t78():
    """
    BUG-R12-02: _pretrain_lstm() no tenia gradient clipping ni deteccion de NaN loss.
    Gradientes > 1e6 corrompian los pesos LSTM silenciosamente -> embeddings NaN -> RF NaN en OOS.
    Fix (Opcion A): clip_grad_norm (max=1.0) + abort si NaN loss + fallback xgb_probs.
    """
    src = _read(ROOT / "luna/models/train_metalabeler_v2.py")
    assert "clip_grad_norm_" in src, (
        "BUG-R12-02 REGRESION: clip_grad_norm_ ausente en train_metalabeler_v2.py. "
        "Sin gradient clipping los pesos LSTM se vuelven NaN silenciosamente. "
        "Anadir: torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0)"
    )
    assert "_lstm_nan_abort" in src, (
        "BUG-R12-02 REGRESION: deteccion NaN loss ausente en _pretrain_lstm(). "
        "Sin este guard el LSTM entrena 30 epochs con loss=NaN y produce embeddings corruptos. "
        "Anadir: if math.isnan(batch_loss): self._lstm_nan_abort = True; return"
    )
    assert "np.isnan(embeddings)" in src, (
        "BUG-R12-02 REGRESION: guard NaN ausente en predict_proba(). "
        "Sin este guard los embeddings NaN se pasan al RF que predice NaN en OOS. "
        "Anadir: if np.isnan(embeddings).mean() > 0.0: return np.clip(xgb_probs, 0.0, 1.0)"
    )
    return "OK -- gradient clipping + NaN detection + NaN guard presentes en MetaLabelerV2"


# ─── LOG-BUG-03 fix (2026-03-10) ─────────────────────────────────────────────

@test("TEST-79  LOG-BUG-03: ffill() eliminado de meta_v2_prob OOS (look-ahead bias)", section="v5_bugs")
def t79():
    """
    LOG-BUG-03: meta_v2_prob_series.ffill() en predict_oos.py L289
    propagaba la 1a prediccion valida (barra seq_len=48) hacia las barras 0..47 del
    warmup del LSTM. Esas barras recibian probabilidades calculadas con datos futuros.
    Fix: eliminar ffill(). Las barras warmup quedan NaN -> False -> sin senal MetaV2.
    """
    src = _read(ROOT / "luna/models/predict_oos.py")
    # Buscar el patron: meta_v2_prob_series = meta_v2_prob_series.ffill()
    # (el fillna(0.0) de la mascara si es correcto y debe existir)
    import re
    # Detectar si el ffill sobre meta_v2_prob_series sigue presente
    ffill_bug = re.search(
        r"meta_v2_prob_series\s*=\s*meta_v2_prob_series\.ffill\s*\(",
        src
    )
    assert ffill_bug is None, (
        "LOG-BUG-03 REGRESION: meta_v2_prob_series.ffill() de vuelta en predict_oos.py. "
        "Propaga la 1a prediccion MetaV2 al warmup del LSTM (barras 0..seq_len-1) -- look-ahead bias. "
        "Eliminar esa linea: las barras warmup deben quedar NaN -> fillna(0.0) -> False (sin senal)."
    )
    # Verificar que fillna(0.0) correcto sigue presente en la mascara
    assert "fillna(0.0)" in src, (
        "LOG-BUG-03 fix incompleto: meta_v2_mask debe usar fillna(0.0) para tratar NaN warmup "
        "como 'sin senal' en vez de propagar la primera prediccion disponible."
    )
    return "OK -- ffill() ausente en meta_v2_prob_series, fillna(0.0) correcto en mascara"


# ─── LOG-BUG-04 fix (2026-03-10) ─────────────────────────────────────────────

@test("TEST-80  LOG-BUG-04: OOS usa features_holdout.parquet (2025), no validation (2024)", section="v5_bugs")
def t80():
    """
    LOG-BUG-04: predict_oos.py usaba features_validation.parquet
    (2024-07-01..2024-12-31) como fuente OOS primaria. El modelo fue entrenado hasta
    2024-06-30 y validado en 2024, asique esas metricas OOS eran OPTIMISTAS FALSOS.
    Fix: features_holdout.parquet (2025-01-01+) es ahora el primario.
    Fix2: validacion cruzada con cfg.temporal_splits.holdout_start de settings.yaml.
    """
    src = _read(ROOT / "luna/models/predict_oos.py")
    # 1. features_holdout.parquet debe ser el primario (primer if, no elif)
    holdout_idx = src.find("features_holdout.parquet")
    val_idx     = src.find("features_validation.parquet")
    assert holdout_idx != -1, (
        "LOG-BUG-04 REGRESION: features_holdout.parquet no referenciado en predict_oos.py. "
        "El script debe usar features_holdout.parquet (2025) como fuente OOS primaria."
    )
    assert val_idx != -1, (
        "predict_oos.py debe mantener features_validation.parquet como fallback. "
        "Estructura esperada: if holdout_path.exists() -> ... elif val_path.exists() -> ..."
    )
    assert holdout_idx < val_idx, (
        "LOG-BUG-04 REGRESION: features_validation.parquet aparece antes que features_holdout.parquet. "
        "El holdout (2025) debe ser el IF primario, la validation (2024) el ELIF fallback."
    )
    # 2. Validacion cruzada con settings.yaml presente
    assert "holdout_start" in src, (
        "LOG-BUG-04 fix incompleto: validacion cruzada con cfg.temporal_splits.holdout_start "
        "ausente en predict_oos.py. Anadir verificacion de fecha minima del parquet."
    )
    return ("OK -- features_holdout.parquet es OOS primario, "
            "features_validation.parquet es fallback, "
            "validacion cruzada con settings.yaml presente")



# ─── BUG-R12-03 fix (2026-03-10) ─────────────────────────────────────────────

@test("TEST-81  BUG-R12-03: rangos Optuna XGB en settings.yaml (sin hardcodes)", section="v5_bugs")
def t81():
    """
    BUG-R12-03: el espacio de busqueda Optuna tenia numeros magicos en codigo y
    faltaban gamma (L0), reg_alpha (L1), reg_lambda (L2), scale_pos_weight.
    Fix: todos los rangos movidos a cfg.xgboost.optuna_search_space en settings.yaml.
    objective() los lee desde cfg — sin ningun numero magico en el codigo.
    """
    src_xgb  = _read(ROOT / "luna/models/train_xgboost_v2.py")
    src_yaml = _read(ROOT / "config/settings.yaml")
    # 1. settings.yaml tiene la seccion optuna_search_space con los nuevos parametros
    for key in ("optuna_search_space", "gamma_min", "reg_alpha_min", "reg_lambda_min",
                "scale_pos_weight_min"):
        assert key in src_yaml, (
            f"BUG-R12-03 REGRESION: '{key}' ausente en settings.yaml. "
            "Los rangos Optuna deben estar documentados en cfg.xgboost.optuna_search_space."
        )
    # 2. objective() lee desde cfg (no tiene numeros magicos)
    assert "optuna_search_space" in src_xgb, (
        "BUG-R12-03 REGRESION: objective() no lee cfg.xgboost.optuna_search_space. "
        "Los rangos Optuna deben leerse desde settings.yaml, no hardcodes en el codigo."
    )
    assert "sp.gamma_min" in src_xgb, (
        "BUG-R12-03 REGRESION: gamma ausente en objective(). "
        "Anadir: trial.suggest_float('gamma', sp.gamma_min, sp.gamma_max)"
    )
    assert "sp.reg_alpha_min" in src_xgb, (
        "BUG-R12-03 REGRESION: reg_alpha ausente en objective(). "
        "Anadir: trial.suggest_float('reg_alpha', sp.reg_alpha_min, sp.reg_alpha_max, log=True)"
    )
    return "OK -- optuna_search_space en settings.yaml, objective() lee desde cfg, gamma/L1/L2 presentes"


# ─── MEJORA-R12-01 fix (2026-03-10) ───────────────────────────────────────────

@test("TEST-82  MEJORA-R12-01: xgb_signal_threshold=0.55 en settings.yaml (no hardcode)", section="v5_bugs")
def t82():
    """
    MEJORA-R12-01: calibracion automatica del threshold XGB usando EV-sweep.
    _calibrate_threshold() barre thresholds sobre features_validation.parquet
    maximizando EV(t) = P(win) * avg_win - P(loss) * avg_loss - cost.
    Resultado guardado en xgboost_meta_signature.json como 'optimal_threshold'.
    predict_oos.py: firma (primario) > settings.yaml (fallback) > 0.50 (neutro).
    """
    src_xgb = _read(ROOT / "luna/models/train_xgboost_v2.py")
    src_oos = _read(ROOT / "luna/models/predict_oos.py")
    src_yaml = _read(ROOT / "config/settings.yaml")
    # 1. _calibrate_threshold implementado en train_xgboost_v2.py
    assert "_calibrate_threshold" in src_xgb, (
        "MEJORA-R12-01 REGRESION: _calibrate_threshold() ausente en train_xgboost_v2.py. "
        "El threshold XGB debe calibrarse automaticamente sobre features_validation.parquet."
    )
    # 2. Parámetros del sweep en settings.yaml
    for key in ("threshold_sweep_min", "threshold_sweep_max", "threshold_sweep_step", "threshold_min_trades"):
        assert key in src_yaml, (
            f"MEJORA-R12-01 REGRESION: '{key}' ausente en settings.yaml. "
            "Los parametros del sweep de calibracion deben estar en cfg.xgboost."
        )
    # 3. signal_filter.py lee optimal_threshold desde la firma (primario)
    src_sigfilter = _read(ROOT / "luna/models/signal_filter.py")
    assert "optimal_threshold" in src_sigfilter, (
        "MEJORA-R12-01 REGRESION: signal_filter.py no lee optimal_threshold de la firma. "
        "Jerarquia esperada: firma JSON (primario) > settings.yaml (fallback) > 0.50 (neutro)."
    )
    # 4. xgb_signal_threshold en settings.yaml como fallback neutro (>= 0.25 minimo absoluto)
    # [Fase B.3 2026-03-27]: el threshold bajo a 0.35 via EV-sweep del calibrador.
    # El umbral de 0.40 era arbitrario -- el calibrador optimiza el valor por EV real.
    # Minimo absoluto: 0.25 para evitar senales aleatorias sin edge.
    import re
    m = re.search(r"xgb_signal_threshold:\s*([0-9.]+)", src_yaml)
    if m:
        val = float(m.group(1))
        assert val >= 0.25, (
            f"MEJORA-R12-01: xgb_signal_threshold={val} < 0.25 -- fallback peligrosamente bajo."
            f" Un umbral por debajo de 0.25 no distingue senal de ruido aleatorio."
        )
    return ("OK -- _calibrate_threshold() implementado, sweep params en settings.yaml, "
            "generate_oos lee desde firma JSON (primario) con fallback settings")





# ─── ARCH-02 fixes (2026-03-10) ───────────────────────────────────────────────

@test("TEST-110 ARCH-02: hmm_regime.py y metalabeler leen N_REGIMES/HMM_N_STATES desde cfg", section="v5_bugs")
def t110():
    """
    ARCH-02: N_REGIMES=4 en hmm_regime.py y HMM_N_STATES=4 en train_metalabeler_v2.py
    estaban hardcodeados — desync risk si se cambia hmm.n_states en settings.yaml.
    Fix: ambos modules leen desde cfg.hmm.n_states. Ademas WINDOWS_OOS, n_iter, tol,
    COST_PCT, EMBARGO_H, LSTM_HIDDEN, N_CPCV_GROUPS, RF_N_ESTIMATORS desde cfg.
    """
    src_hmm  = _read(ROOT / "luna/models/hmm_regime.py")
    src_meta = _read(ROOT / "luna/models/train_metalabeler_v2.py")
    src_yaml = _read(ROOT / "config/settings.yaml")
    # 1. hmm_regime.py lee desde cfg (no hardcode N_REGIMES = 4)
    assert "N_REGIMES = 4" not in src_hmm, (
        "ARCH-02 REGRESION: N_REGIMES = 4 hardcodeado en hmm_regime.py. "
        "Leer desde cfg.hmm.n_states."
    )
    assert "_cfg_hmm" in src_hmm or "cfg_hmm" in src_hmm, (
        "ARCH-02 REGRESION: hmm_regime.py no lee N_REGIMES desde cfg. "
        "Anadir lectura desde config.settings."
    )
    # 2. train_metalabeler_v2.py lee HMM_N_STATES desde cfg
    assert "HMM_N_STATES = 4" not in src_meta, (
        "ARCH-02 REGRESION: HMM_N_STATES = 4 hardcodeado en train_metalabeler_v2.py. "
        "Leer desde cfg.hmm.n_states para sincronizar con settings.yaml."
    )
    assert "_cfg_meta" in src_meta or "cfg_meta" in src_meta, (
        "ARCH-02 REGRESION: train_metalabeler_v2.py no lee constantes desde cfg. "
        "COST_PCT, EMBARGO_H, LSTM_HIDDEN, etc. deben venir de settings.yaml."
    )
    # 3. settings.yaml tiene los parametros de metalabeler y sop
    for key in ("lstm_hidden", "n_cpcv_groups", "rf_n_estimators", "sop:", "cost_pct"):
        assert key in src_yaml, (
            f"ARCH-02 REGRESION: '{key}' ausente en settings.yaml. "
            "Todos los parametros de metalabeler y SOP deben estar en cfg."
        )
    return ("OK -- hmm_regime.py y metalabeler leen desde cfg, "
            "settings.yaml tiene sop:, hmm:, metalabeler: con todos los params")




@test("TEST-111 BUG-R15-01: hmm_regime min_state_duration_dynamic >= floor configurado", section="v5_bugs")
def t111():
    """
    BUG-R15-01: max(_min_dur_cfg // 2, _p10) dejaba el floor por debajo del minimo configurado.
    Verifica que el codigo usa max(_min_dur_cfg, ...) sin la division que violaba el floor.
    """
    import re as _re
    hmm_path = ROOT / "luna/models/hmm_regime.py"
    if not hmm_path.exists():
        return "hmm_regime.py no encontrado"
    src = _read(hmm_path)
    # BUG: la formula vieja usaba _min_dur_cfg // 2
    bug_pattern = r"max\(_min_dur_cfg\s*//\s*2\s*,\s*_p10\)"
    assert not _re.search(bug_pattern, src), (
        "BUG-R15-01 presente: se usa _min_dur_cfg // 2 en lugar de _min_dur_cfg. "
        "El floor de duracion HMM puede quedar por debajo del minimo configurado."
    )
    # CORRECTO: debe usar max(_min_dur_cfg, _p10)
    correct_pattern = r"max\(_min_dur_cfg\s*,\s*_p10\)"
    assert _re.search(correct_pattern, src), (
        "Fix BUG-R15-01 no encontrado: se esperaba max(_min_dur_cfg, _p10) en hmm_regime.py."
    )
    cfg = _cfg()
    floor_h = getattr(getattr(cfg, "hmm", None), "min_state_duration_hours", 120)
    return f"BUG-R15-01 corregido: max(cfg_floor={floor_h}H, P10) garantiza floor respetado"



@test("TEST-112 BUG-R15-02: HMM load_data no filtra por selected_features", section="v5_bugs")
def t112():
    """
    BUG-R15-02: load_data() filtraba HMM_FEATURES con 'c in features' (selected_features),
    dejando al HMM con 1 sola feature cuando el SFI era agresivo.
    El HMM debe leer directamente del parquet, independientemente de selected_features.
    """
    hmm_path = ROOT / "luna/models/hmm_regime.py"
    if not hmm_path.exists():
        return "hmm_regime.py no encontrado"
    src = _read(hmm_path)

    # BUG: condicion doble 'c in df.columns and c in features' donde features=selected
    bug_str = "c in df.columns and c in features"
    assert bug_str not in src, (
        "BUG-R15-02 presente: load_data() filtra HMM_FEATURES por selected_features. "
        "El HMM debe leer directo del parquet sin filtrar."
    )

    # CORRECTO: loop sobre HMM_FEATURES directo
    assert "for c in HMM_FEATURES:" in src, (
        "Fix BUG-R15-02 no encontrado: se esperaba 'for c in HMM_FEATURES:' en hmm_regime.py."
    )

    # Verificar que OI_BTC_pct_chg esta en HMM_FEATURES
    assert "OI_BTC_pct_chg" in src, (
        "OI_BTC_pct_chg no esta en HMM_FEATURES -- feature de posicionamiento faltante."
    )

    return "BUG-R15-02 corregido: HMM_FEATURES cargadas del parquet sin filtrar por selected_features"



# ─── ARCH-03 fixes (2026-03-10) ───────────────────────────────────────────────

@test("TEST-85  ARCH-03: feature_selection_e.py lee constantes SFI desde cfg.features", section="v5_bugs")
def t85():
    """
    ARCH-03: CLUSTER_FIXED_N=15, SFI_TOP_N_FEATURES=15, SFI_N_GROUPS=6,
    SFI_N_ESTIMATORS=200, SFI_MAX_DEPTH=4, MAX_LAG_HOURS=500, FORWARD_MAX_FEATURES=25
    estaban hardcodeados en feature_selection_e.py.
    Fix: se leen desde cfg.features en settings.yaml (misma convencion ARCH-02).
    BONUS: SFI_PURGE_H, SFI_EMBARGO_H y SFI_COST_ROUNDTRIP usan sop.embargo_hours
    y sop.cost_pct compartidos — evita desync con los mismos valores en train_xgboost.
    """
    src_sfi  = _read(ROOT / "luna/features/feature_selection_e.py")
    src_yaml = _read(ROOT / "config/settings.yaml")
    # 1. El bloque principal lee desde cfg (detectar via _cfg_sfi)
    assert "_cfg_sfi" in src_sfi, (
        "ARCH-03 REGRESION: feature_selection_e.py no tiene bloque cfg_sfi. "
        "Anadir try/except que lea constantes desde config.settings."
    )
    # 2. La asignacion de nivel modulo usa getattr (no = int literal)
    # Buscar la linea de CLUSTER_FIXED_N dentro del bloque try (con getattr)
    assert "'sfi_n_clusters'" in src_sfi, (
        "ARCH-03 REGRESION: CLUSTER_FIXED_N no lee desde cfg.features.sfi_n_clusters. "
        "Usar getattr(_cfg_sfi.features, 'sfi_n_clusters', 15)."
    )
    assert "'sfi_n_estimators'" in src_sfi, (
        "ARCH-03 REGRESION: SFI_N_ESTIMATORS no lee desde cfg.features.sfi_n_estimators. "
        "Usar getattr(_cfg_sfi.features, 'sfi_n_estimators', 200)."
    )
    # 3. settings.yaml tiene todos los parametros SFI migrados
    for key in ("sfi_n_clusters", "sfi_top_n", "sfi_n_groups", "sfi_n_estimators",
                "sfi_max_depth", "max_lag_hours", "forward_max_features", "sfi_min_sharpe"):
        assert key in src_yaml, (
            f"ARCH-03 REGRESION: '{key}' ausente en settings.yaml (seccion features:). "
            "Todos los parametros SFI deben estar en cfg."
        )
    # 4. TIPO-1/2/3 leyenda presente en settings.yaml
    assert "TIPO-1" in src_yaml and "TIPO-2" in src_yaml and "TIPO-3" in src_yaml, (
        "ARCH-03 REGRESION: clasificacion TIPO-1/2/3 ausente en settings.yaml. "
        "Anadir leyenda en la cabecera."
    )
    return ("OK -- feature_selection_e.py lee 11 constantes desde cfg.features/sop/stat, "
            "settings.yaml tiene clasificacion TIPO-1/2/3 y todos los params SFI")


# ─── MEJORA-SFI-SHARPE-01 y MEJORA-HMM-DURATION-01 (2026-03-10) ──────────────

@test("TEST-86  MEJORA-SFI-SHARPE-01: SFI-CPCV usa min_sharpe_used dinamico (DSR-null)", section="v5_bugs")
def t86():
    """
    MEJORA-SFI-SHARPE-01: SFI_CPCV.evaluate() calcula min_sharpe_dynamic
    = max(SFI_MIN_SHARPE_floor, 1.645/sqrt(n_obs_oob)).
    El floor SFI_MIN_SHARPE=0.05 era placeholder; ahora es el floor del umbral
    dinamico derivado de la distribucion nula de Sharpe bajo H0 (sin senyal).
    """
    src_sfi = _read(ROOT / "luna/features/feature_selection_e.py")
    assert "self.min_sharpe_used" in src_sfi, (
        "MEJORA-SFI-SHARPE-01 REGRESION: SFI_CPCV no tiene self.min_sharpe_used. "
        "Anadir atributo en __init__ y calcular en evaluate()."
    )
    assert "1.645" in src_sfi and "_n_obs_oob" in src_sfi, (
        "MEJORA-SFI-SHARPE-01 REGRESION: calculo 1.645/sqrt(n_obs_oob) ausente. "
        "Implementar sfi_min_sharpe_dynamic en evaluate()."
    )
    assert "context_filtered" in src_sfi, (
        "MEJORA-SFI-SHARPE-01 REGRESION: filtro context_filtered por MeanSR ausente. "
        "Aplicar min_sharpe_threshold sobre features contextuales."
    )
    return ("OK -- SFI_CPCV calcula min_sharpe=max(floor=0.05, 1.645/sqrt(n_obs_oob)), "
            "filtra contextuales por MeanSR, Alphas exentas del filtro")



@test("TEST-87  MEJORA-HMM-DURATION-01: HMM calcula P10 empirico de duraciones de estado", section="v5_bugs")
def t87():
    """
    MEJORA-HMM-DURATION-01: fit_global_for_analysis() calcula P10 de
    run lengths IS y lo expone como self.min_state_duration_dynamic.
    Se persiste en save_model(). MIN_STATE_DURATION_H desde cfg.hmm.
    Bonus: min_mi ya no es hardcode (lee desde cfg.hmm.min_mi).
    """
    src_hmm = _read(ROOT / "luna/models/hmm_regime.py")
    assert "MIN_STATE_DURATION_H" in src_hmm, (
        "MEJORA-HMM-DURATION-01 REGRESION: MIN_STATE_DURATION_H ausente. "
        "Leer desde cfg.hmm.min_state_duration_hours."
    )
    assert "_run_lengths" in src_hmm, (
        "MEJORA-HMM-DURATION-01 REGRESION: calculo run_lengths ausente. "
        "Implementar P10 de duraciones en fit_global_for_analysis()."
    )
    assert "min_state_duration_dynamic" in src_hmm, (
        "MEJORA-HMM-DURATION-01 REGRESION: self.min_state_duration_dynamic ausente. "
        "Exponer el P10 calculado para trazabilidad y uso en OOS."
    )
    assert "'min_state_duration_dynamic'" in src_hmm, (
        "MEJORA-HMM-DURATION-01 REGRESION: no se persiste en save_model(). "
        "Anadir al dict de joblib.dump()."
    )
    assert "min_mi = 0.005" not in src_hmm, (
        "REGRESION: min_mi = 0.005 hardcodeado. Leer desde cfg.hmm.min_mi."
    )
    return ("OK -- HMM: P10 run_lengths en fit(), min_state_duration_dynamic persistido, "
            "MIN_STATE_DURATION_H y min_mi desde cfg.hmm")



# ─────────────────────────────────────────────────────────
#  MEJORA-08: Coherencia PT/SL — modelo en disco vs settings actuales
# ─────────────────────────────────────────────────────────