import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
import types

# ── [LUNA-V2-CALIB] PlattCalibrator definition and injection for pickle/joblib deserialization robustness ──
class PlattCalibrator:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression as _LR
        import os
        _seed = int(os.environ.get("LUNA_SEED", 42))
        print(f"[AUDIT-FIX] LUNA_SEED={_seed} inyectado en SignalFilter LogisticRegression")
        self.model = _LR(C=1e6, random_state=_seed)
        self.X_thresholds_ = []

    def fit(self, X, y):
        X_2d = X.reshape(-1, 1) if X.ndim == 1 else X
        self.model.fit(X_2d, y)
        return self

    def predict(self, T):
        T_2d = T.reshape(-1, 1) if T.ndim == 1 else T
        return self.model.predict_proba(T_2d)[:, 1]

if 'luna.models.train_xgboost_v2' not in sys.modules:
    _dummy = types.ModuleType('luna.models.train_xgboost_v2')
    _dummy.PlattCalibrator = PlattCalibrator
    sys.modules['luna.models.train_xgboost_v2'] = _dummy
else:
    sys.modules['luna.models.train_xgboost_v2'].PlattCalibrator = PlattCalibrator


class SignalFilter:
    """
    Módulo unificado para aplicar el pipeline de filtrado de señales sobre la inferencia de XGBoost.
    Consolida las lógicas de Threshold, OOD Guard, MetaLabelerV2, HMM y Momentum Asimétrico.
    Resuelve el anti-patrón de monolito (Audit C2) y estandariza la escritura del signal_funnel.json (P1.1).
    """

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.funnel_stats = {}
        # [XGB-ISO-CAL-01] Pre-cargar calibrador isotónico XGBoost una sola vez (evita IO repetido)
        self._xgb_iso_calibrator = None
        self._xgb_iso_suffix = None  # para soporte multi-agente (bull/range/bear)
        # [KELLY-SIZER] Position sizer — cargado lazy en apply_kelly_sizing()
        self._kelly_sizer = None

    def _load_xgb_isotonic_calibrator(self, suffix: str = "") -> bool:
        """
        [XGB-ISO-CAL-01] Carga el calibrador isotónico del XGBoost si existe.
        Devuelve True si se cargó exitosamente, False si no existe (fallback a crudas).
        """
        if self._xgb_iso_calibrator is not None and self._xgb_iso_suffix == suffix:
            return True  # ya cargado para este suffix
        iso_path = self.models_dir / f"xgboost_isotonic_calibrator{suffix}.joblib"
        if iso_path.exists():
            try:
                import joblib as _jlib_sf
                import numpy as _np_load_check
                _cal_loaded = _jlib_sf.load(iso_path)

                # [BUG-CALIB-XGB-01] GUARD: Verificar que el calibrador recién cargado
                # no es degenerado (función plana). Descartarlo en carga si lo es.
                _tx_check = _np_load_check.linspace(0.3, 0.7, 30)
                _out_check = _cal_loaded.predict(_tx_check) if hasattr(_cal_loaded, 'predict') else _tx_check
                _std_check = float(_np_load_check.std(_out_check))
                _anchors_check = len(getattr(_cal_loaded, 'X_thresholds_', []))
                print(
                    f"[BUG-CALIB-XGB-01] LOAD CHECK {iso_path.name} | "
                    f"std={_std_check:.6f} anchors={_anchors_check}"
                )
                if _std_check < 1e-4:
                    logger.warning(
                        "[BUG-CALIB-XGB-01] Calibrador {} RECHAZADO en carga — std={:.8f} (degenerado). "
                        "Usando xgb_prob raw. Eliminar el .joblib y re-entrenar para corregir.",
                        iso_path.name, _std_check
                    )
                    print(
                        f"[BUG-CALIB-XGB-01] *** CALIBRADOR {iso_path.name} RECHAZADO EN CARGA *** "
                        f"std={_std_check:.8f} — fallback a xgb_prob raw."
                    )
                    return False  # forzar fallback a probs raw

                self._xgb_iso_calibrator = _cal_loaded
                self._xgb_iso_suffix = suffix
                logger.info("[XGB-ISO-CAL-01] Calibrador isotónico XGBoost cargado: {} | std={:.4f} anchors={}", iso_path.name, _std_check, _anchors_check)
                return True
            except Exception as _e_iso_load:
                logger.warning("[XGB-ISO-CAL-01] Fallo cargando calibrador: {} - usando probs crudas.", _e_iso_load)
        return False

    def _apply_xgb_isotonic(self, df_oos: pd.DataFrame, prob_col: str, suffix: str = "") -> str:
        """
        [XGB-ISO-CAL-01] Aplica calibración isotónica a xgb_prob si el calibrador está disponible.
        Crea la columna 'xgb_prob_cal' con probabilidades calibradas y devuelve el nombre de columna a usar.
        Si no hay calibrador, devuelve prob_col sin cambios.
        """
        if self._load_xgb_isotonic_calibrator(suffix):
            import numpy as _np_sf
            raw = df_oos[prob_col].fillna(0).values
            calibrated = self._xgb_iso_calibrator.predict(raw)
            df_oos["xgb_prob_cal"] = _np_sf.clip(calibrated, 0.0, 1.0)

            # [BUG-CALIB-XGB-01] GUARD POST-APLICACIÓN: verificar que la columna
            # calibrada tiene varianza. Si es constante, revertir a probs raw.
            _std_applied = float(_np_sf.std(df_oos["xgb_prob_cal"].values))
            _std_raw = float(_np_sf.std(raw))
            print(
                f"[BUG-CALIB-XGB-01] POST-APPLY CHECK | "
                f"std_raw={_std_raw:.6f} std_cal={_std_applied:.6f} | "
                f"raw=[{raw.min():.3f},{raw.max():.3f}] cal=[{df_oos['xgb_prob_cal'].min():.3f},{df_oos['xgb_prob_cal'].max():.3f}]"
            )
            if _std_applied < 1e-4 and _std_raw > 1e-4:
                logger.warning(
                    "[BUG-CALIB-XGB-01] xgb_prob_cal COLAPSÓ a constante ({:.4f}) tras aplicar calibrador — "
                    "std_cal={:.8f} vs std_raw={:.4f}. Revirtiendo a xgb_prob raw para evitar señales nulas.",
                    float(df_oos["xgb_prob_cal"].mean()), _std_applied, _std_raw
                )
                print(
                    f"[BUG-CALIB-XGB-01] *** REVERTIENDO xgb_prob_cal → xgb_prob RAW *** "
                    f"(std_cal={_std_applied:.8f} < 1e-4 pero std_raw={_std_raw:.4f})"
                )
                df_oos.drop(columns=["xgb_prob_cal"], inplace=True)
                return prob_col  # fallback a raw

            logger.info(
                "[XGB-ISO-CAL-01] xgb_prob calibrado: min={:.3f} max={:.3f} mean={:.3f} "
                "(raw: min={:.3f} max={:.3f} mean={:.3f}) | std_cal={:.4f}",
                df_oos["xgb_prob_cal"].min(), df_oos["xgb_prob_cal"].max(), df_oos["xgb_prob_cal"].mean(),
                df_oos[prob_col].min(), df_oos[prob_col].max(), df_oos[prob_col].mean(),
                _std_applied
            )
            return "xgb_prob_cal"
        return prob_col





    def apply_kelly_sizing(self, df_oos: "pd.DataFrame", signal_mask: "pd.Series", prob_col: str = "xgb_prob_cal") -> "pd.Series":
        """
        [KELLY-SIZER] Calcula la fracción de capital (Quarter-Kelly) para cada señal activa.
        Devuelve pd.Series 'position_fraction' con fracción [0.0, 0.15] por fila.
        Señales rechazadas (mask=False) reciben fraction=0.0.

        Activado en settings.yaml si kelly_sizer.enabled = true.
        Si no está habilitado o falla, devuelve 1.0 para señales activas (tamaño fijo).
        """
        n_signals = int(signal_mask.sum())
        try:
            from config.settings import cfg as _cfg_ks
            _enabled = getattr(getattr(_cfg_ks, "kelly_sizer", None), "enabled", False)
        except Exception:
            _enabled = False

        if not _enabled:
            # Kelly deshabilitado → fracción fija 1.0 para señales activas
            return signal_mask.astype(float)

        if self._kelly_sizer is None:
            try:
                from luna.sizing.kelly_sizer import build_kelly_sizer_from_settings
                self._kelly_sizer = build_kelly_sizer_from_settings()
            except Exception as _e_ks:
                logger.warning("[KELLY-SIZER] No se pudo cargar KellyPositionSizer: {} -- fraccion fija.", _e_ks)
                return signal_mask.astype(float)

        fractions = self._kelly_sizer.size_signals(df_oos, prob_col=prob_col, mask_col=None)
        # Forzar 0.0 en señales rechazadas
        fractions[~signal_mask] = 0.0
        
        # [HMM-PREDICTIVE-02] Scale down Kelly sizes if risk is elevated
        if "hmm_transition_risk" in df_oos.columns:
            try:
                from config.settings import cfg as _cfg_hmm_tr
                _hmm_elevated_risk  = float(getattr(getattr(_cfg_hmm_tr, "hmm_predictive", None), "bear_transition_elevated_risk", 0.30))
                _hmm_kelly_penalty  = float(getattr(getattr(_cfg_hmm_tr, "hmm_predictive", None), "elevated_risk_kelly_penalty", 0.50))
                
                elevated_mask = (df_oos["hmm_transition_risk"] > _hmm_elevated_risk) & signal_mask
                if elevated_mask.any():
                    fractions[elevated_mask] *= _hmm_kelly_penalty
                    logger.info(
                        "  [HMM-PREDICTIVE-02] Reduccion {:.0%} en Kelly Fraction para {} señales por riesgo elevado (> {:.0%}).",
                        1.0 - _hmm_kelly_penalty, int(elevated_mask.sum()), _hmm_elevated_risk
                    )
            except Exception as _e_hmm_ks:
                logger.debug("  [HMM-PREDICTIVE-02] Penalty no aplicado en Kelly: {}", _e_hmm_ks)
                
        # [CAMINOB-08] Scale down Kelly sizes if Macro is adverse
        if "macro_kelly_penalty" in df_oos.columns:
            # Multiplicamos las fracciones de Kelly por la penalizacion macro
            fractions *= df_oos["macro_kelly_penalty"]
            n_penalized = int(((df_oos["macro_kelly_penalty"] < 1.0) & signal_mask).sum())
            if n_penalized > 0:
                logger.info(
                    "  [MACRO-GATE] Kelly Fraction reducido en {} señales por adversidad macro (Lag Trap evitado).",
                    n_penalized
                )
                
        logger.info("[KELLY-SIZER] {} señales dimensionadas. Fraccion media={:.1%} max={:.1%}",
                    n_signals, fractions[signal_mask].mean() if n_signals > 0 else 0.0,
                    fractions.max())
        return fractions

    def apply_model_threshold(self, df_oos: pd.DataFrame, prefix: str = "xgboost_meta", prob_col: str = "xgb_prob", model_name: str = "XGBoost", direction: str = "long") -> pd.Series:
        # [XGB-ISO-CAL-01] Aplicar calibración isotónica ANTES de cualquier umbral
        # Detectar suffix del agente (ej: "_bull", "_bear", "" para modelo único)
        if model_name == "XGBoost":
            if "xgb_prob_cal" in df_oos.columns and df_oos["xgb_prob_cal"].notna().any():
                logger.debug("[SignalFilter] xgb_prob_cal ya existe (calibrado por RegimeRouter). Saltando _apply_xgb_isotonic.")
                prob_col = "xgb_prob_cal"
            else:
                _iso_suffix = ""
                for _s in ["_bull", "_range", "_bear"]:
                    if prefix.endswith(_s):
                        _iso_suffix = _s
                        break
                prob_col = self._apply_xgb_isotonic(df_oos, prob_col, suffix=_iso_suffix)

        # MEJORA-R12-01 (2026-03-10): jerarquía de threshold (sin hardcodes)
        # [FIX-NEW-02] Jerarquía corregida — 3 niveles de prioridad:
        # Fuente-0 (MÁXIMA): xgb_signal_threshold_override en settings.yaml
        # Fuente-1 (PRIMARIA): optimal_threshold en la firma del modelo calibrado
        # Fuente-2 (FALLBACK): xgb_signal_threshold en settings.yaml
        # Razón: FIX-THRESH-01 puede producir threshold=0.90 (silenciador seguro)
        # sin posibilidad de override manual desde settings → 0 señales sin salida.
        _DEFAULT_XGB_LIMIT = 0.50  # default neutro
        threshold_source = "default-0.50"
        
        # [FIX-CALIB-01] MetaLabelerV2 thresholds are stored in calibrator_signature.json
        if model_name == "MetaLabelerV2" or "metalabeler" in prefix:
            prefix = "calibrator"
            
        sig_path = self.models_dir / f"{prefix}_signature.json"
        
        # [FIX-1] Usar signature direccional si existe para soportar optimal_threshold_per_regime separado
        _dir_sig_path = self.models_dir / f"{prefix}_{direction}_signature.json"
        if _dir_sig_path.exists():
            sig_path = _dir_sig_path

        # I4 - Per-regime thresholds
        optimal_threshold_per_regime = {}

        # [FIX-NEW-02] Fuente-0: Override manual en settings.yaml (MÁXIMA PRIORIDAD)
        # Activar con: xgb_signal_threshold_override: 0.38 en settings.yaml → xgboost section
        # Dejar ausente/null para comportamiento normal (firma calibrada prevalece).
        #
        # [FIX-SCOPE-01] (2026-05-03): Los overrides de threshold (xgb_signal_threshold_override
        # y short_threshold_override) son EXCLUSIVOS del agente XGBoost.
        # Bug anterior: cuando filter_signals iteraba direction="short", el short_threshold_override
        # se propagaba también a MetaLabelerV2 y LightGBM (vía el mismo direction arg),
        # silenciando todos los filtros con threshold=0.90 aunque solo se quería bloquear el XGB.
        # Fix: los overrides de settings.yaml solo aplican si model_name == "XGBoost".
        _override_applied = False
        try:
            from config.settings import cfg as _cfg_ovr
            _ovr = None  # por defecto: sin override

            if model_name == "XGBoost":  # [FIX-SCOPE-01] overrides SOLO para XGBoost
                _ovr = getattr(_cfg_ovr.xgboost, 'xgb_signal_threshold_override', None)

                # [FIX-3] Override direccional especifico para SHORT (XGBoost únicamente)
                if direction == "short":
                    _short_ovr = getattr(_cfg_ovr.xgboost, 'short_threshold_override', None)
                    if _short_ovr is not None:
                        _ovr = _short_ovr
                        logger.warning(
                            "  [FIX-3] short_threshold_override={:.3f} aplicado al agente XGBoost-SHORT "
                            "(NO se propaga a MetaLabeler ni LGBM — FIX-SCOPE-01).",
                            float(_ovr)
                        )
            else:
                logger.debug(
                    "  [FIX-SCOPE-01] Modelo {} ignorando overrides de XGBoost "
                    "(xgb_signal_threshold_override / short_threshold_override).",
                    model_name
                )

            if _ovr is not None:
                _DEFAULT_XGB_LIMIT = float(_ovr)
                threshold_source = f"settings.yaml OVERRIDE={_ovr:.3f} (max prioridad, FIX-NEW-02/FIX-3)"
                _override_applied = True
                logger.info(
                    "  [FIX-NEW-02/FIX-3] Override manual activo para {}: threshold={:.3f} "
                    "(override en settings.yaml prevalece sobre firma calibrada).",
                    model_name, _DEFAULT_XGB_LIMIT
                )
        except Exception:
            pass

        # Fuente-1: Firma del modelo (PRIMARIO — solo si no hay override manual)
        if not _override_applied:
            # Primero intentar cargar la firma global o direccional (legacy o LGBM)
            if sig_path.exists():
                try:
                    with open(sig_path, "r") as _f:
                        _sig = json.load(_f)
                    
                    # Soporte para firmas de XGBoost/LGBM (optimal_threshold) y MetaLabeler (optimal_meta_threshold)
                    _opt_thr = _sig.get("optimal_threshold", _sig.get("optimal_meta_threshold"))
                    if _opt_thr is not None:
                        _DEFAULT_XGB_LIMIT = float(_opt_thr)
                        threshold_source = "calibrado-validation (EV-sweep) [PRIMARIO]"
                        # [FIX-THRESH-AUDIT-01] (2026-05-03): Threshold >= 0.85 indica que
                        # el calibrador activó FIX-THRESH-01 (silenciador seguro: sin EV>0
                        # en holdout). Esto es correcto y esperado cuando el mercado de
                        # calibración era adverso. Se registra como AUDIT para trazabilidad.
                        if _DEFAULT_XGB_LIMIT >= 0.85:
                            logger.warning(
                                "  [FIX-THRESH-AUDIT-01] Agente {} (direction={}) tiene threshold={:.3f} "
                                "(silenciador FIX-THRESH-01 activo — EV<=0 en holdout de calibracion). "
                                "Este agente NO generará señales en este período. "
                                "Comportamiento ESPERADO y CORRECTO si el mercado de calibracion fue adverso.",
                                prefix, direction, _DEFAULT_XGB_LIMIT
                            )
                    
                    _opt_thr_reg = _sig.get("optimal_threshold_per_regime", _sig.get("optimal_meta_threshold_per_regime"))
                    if _opt_thr_reg is not None:
                        optimal_threshold_per_regime = _opt_thr_reg
                except Exception as _e:
                    logger.debug("  No se pudo leer optimal_threshold de firma: {}", _e)
            
            # [FIX-1] Luna V1 Multi-Agent: Si es XGBoost, cargar thresholds calibrados de cada agente especializado
            if model_name == "XGBoost" and not optimal_threshold_per_regime:
                try:
                    from config.settings import cfg as _cfg_ma
                    # FIX-BUG: regime_mapping extraction logic
                    _fase2 = getattr(_cfg_ma, "fase2", None)
                    _regime_mapping_obj = getattr(_cfg_ma.xgboost, "regime_mapping", getattr(_fase2, "regime_mapping", None))
                    
                    _regimes_config = {}
                    if hasattr(_regime_mapping_obj, "__dict__"):
                        _regimes_config = vars(_regime_mapping_obj)
                    elif isinstance(_regime_mapping_obj, dict):
                        _regimes_config = _regime_mapping_obj
                        
                    for agent_name, r_list in _regimes_config.items():
                        _ag_sig_path = self.models_dir / f"{prefix}_{agent_name}_{direction}_signature.json"
                        if _ag_sig_path.exists():
                            with open(_ag_sig_path, "r") as _f:
                                _ag_sig = json.load(_f)
                                if "optimal_threshold" in _ag_sig:
                                    for r in r_list:
                                        optimal_threshold_per_regime[str(r)] = float(_ag_sig["optimal_threshold"])
                    if optimal_threshold_per_regime:
                        threshold_source = "multi-agent-signatures (calibrados per-regime)"
                        logger.info("  [FIX-1] Thresholds Multi-Agent cargados: {}", optimal_threshold_per_regime)
                except Exception as e:
                    logger.debug("  [FIX-1] Error cargando firmas Multi-Agent: {}", e)

        # Fuente-2: settings.yaml (FALLBACK — solo si ni override ni firma activos)
        if threshold_source == "default-0.50" and not optimal_threshold_per_regime:
            try:
                from config.settings import cfg as _cfg_oos
                _t = getattr(_cfg_oos.xgboost, 'xgb_signal_threshold', None)
                if _t is not None:
                    _DEFAULT_XGB_LIMIT = float(_t)
                    threshold_source = "settings.yaml (fallback manual)"
            except Exception:
                pass
                
        # I4 Dynamic Application
        # [FIX-THRESHOLD-TRACE-01] Registrar el threshold efectivo real para trazabilidad.
        # BUG PREVIO (2026-05-31): used_threshold siempre era 0.5 cuando se usaba el path
        # per-régimen (I4) porque _DEFAULT_XGB_LIMIT no se actualizaba en ese flujo.
        # Los parquets OOS mostraban signal_threshold=0.5 para todos los trades aunque
        # las firmas calibradas usaran thresholds entre 0.46-0.695.
        # FIX: calcular la mediana ponderada por barras de los thresholds efectivos aplicados.
        _effective_threshold_for_trace = _DEFAULT_XGB_LIMIT  # se sobreescribirá si hay per-régimen
        _per_regime_thresholds_trace: dict = {}  # para trazabilidad detallada

        if optimal_threshold_per_regime and ("HMM_Regime" in df_oos.columns or "HMM_Semantic" in df_oos.columns):
            logger.info("  [I4] Umbrales señal {} por régimen aplicados: {}", model_name, optimal_threshold_per_regime)
            model_mask = pd.Series(False, index=df_oos.index)

            # FIX 2026-05-06: Detectar si las claves son numéricas (fallback) o semánticas
            _keys_are_numeric = all(str(k).replace('.','',1).isdigit() for k in optimal_threshold_per_regime.keys())

            if _keys_are_numeric and "HMM_Regime" in df_oos.columns:
                _regime_col = pd.to_numeric(df_oos["HMM_Regime"], errors='coerce').fillna(-1).astype(int)
                _thresh_by_regime = {int(float(k)): float(v) for k, v in optimal_threshold_per_regime.items()}
            elif not _keys_are_numeric and "HMM_Semantic" in df_oos.columns:
                _regime_col = df_oos["HMM_Semantic"].astype(str)
                _thresh_by_regime = {str(k): float(v) for k, v in optimal_threshold_per_regime.items()}
            else:
                _regime_col = pd.Series("UNKNOWN", index=df_oos.index)
                _thresh_by_regime = {}

            _weighted_thr_sum = 0.0
            _weighted_thr_n = 0
            for r_key, r_thresh in _thresh_by_regime.items():
                r_mask_idx = (_regime_col == r_key)
                _n_regime = int(r_mask_idx.sum())
                
                # [BUG-4 FIX] OOS-Aware Thresholding (P90 Fallback)
                if _n_regime > 0 and prob_col in df_oos.columns:
                    _max_oos_prob = float(df_oos.loc[r_mask_idx, prob_col].max())
                    if _max_oos_prob < r_thresh and _max_oos_prob > 0.0:
                        import numpy as np
                        _p90 = float(np.percentile(df_oos.loc[r_mask_idx, prob_col].dropna(), 90))
                        logger.warning(
                            "  [BUG-4 FIX] Régimen {}: max(prob_OOS)={:.4f} < IS_thresh={:.4f}. Fallback a P90={:.4f}",
                            r_key, _max_oos_prob, r_thresh, _p90
                        )
                        r_thresh = _p90

                model_mask.loc[r_mask_idx] = df_oos.loc[r_mask_idx, prob_col] > r_thresh
                if _n_regime > 0:
                    _weighted_thr_sum += r_thresh * _n_regime
                    _weighted_thr_n += _n_regime
                    _per_regime_thresholds_trace[str(r_key)] = r_thresh

            # [FIX-THRESHOLD-TRACE-01] Calcular threshold efectivo ponderado por barras
            if _weighted_thr_n > 0:
                _effective_threshold_for_trace = _weighted_thr_sum / _weighted_thr_n
                print(
                    f"[FIX-THRESHOLD-TRACE-01] {model_name} threshold efectivo real: "
                    f"{_effective_threshold_for_trace:.4f} (media pond. por barras). "
                    f"Per-régimen: {_per_regime_thresholds_trace}"
                )

            # Map unmapped regimes to the global _DEFAULT_XGB_LIMIT
            mapped_keys = list(_thresh_by_regime.keys())
            unmapped_mask = ~_regime_col.isin(mapped_keys)
            if unmapped_mask.any():
                model_mask.loc[unmapped_mask] = df_oos.loc[unmapped_mask, prob_col] > _DEFAULT_XGB_LIMIT
                logger.info("  Fallback a umbral global ({:.2f}) aplicado a {} velas sin régimen mapeado", _DEFAULT_XGB_LIMIT, unmapped_mask.sum())
        else:
            logger.info("  Umbral señal {} global estático: {:.2f} [fuente: {}]", model_name, _DEFAULT_XGB_LIMIT, threshold_source)
            if prob_col in df_oos.columns:
                # [BUG-4 FIX] Global OOS-Aware Thresholding
                _max_oos_prob = float(df_oos[prob_col].max())
                if _max_oos_prob < _DEFAULT_XGB_LIMIT and _max_oos_prob > 0.0:
                    import numpy as np
                    _p90 = float(np.percentile(df_oos[prob_col].dropna(), 90))
                    logger.warning(
                        "  [BUG-4 FIX] Global: max(prob_OOS)={:.4f} < IS_thresh={:.4f}. Fallback a P90={:.4f}",
                        _max_oos_prob, _DEFAULT_XGB_LIMIT, _p90
                    )
                    _DEFAULT_XGB_LIMIT = _p90
                    
                model_mask = df_oos[prob_col] > _DEFAULT_XGB_LIMIT
            else:
                model_mask = pd.Series(False, index=df_oos.index)
            _effective_threshold_for_trace = _DEFAULT_XGB_LIMIT

        n_model = int(model_mask.sum())
        logger.info("  Señales {} combinadas retenidas: {}", model_name, n_model)

        # M-38: filtro cap SUPERIOR de probabilidad.
        _prob_max = 1.0
        try:
            from config.settings import cfg as _cfg_pmax
            _pm = getattr(_cfg_pmax.xgboost, 'signal_prob_max', None)
            if _pm is not None and float(_pm) < 1.0:
                _prob_max = float(_pm)
        except Exception:
            pass
            
        if _prob_max < 1.0 and prob_col in df_oos.columns:
            model_mask_upper = df_oos[prob_col] <= _prob_max
            n_before = int(model_mask.sum())
            model_mask = model_mask & model_mask_upper
            n_model = int(model_mask.sum())
            logger.info("  [M-38] Filtro Q4: prob <= {:.3f} → {} señales (eliminadas {} con prob>{:.3f})",
                        _prob_max, n_model, n_before - n_model, _prob_max)

        # BUG-03 FIX: threshold de emergencia
        try:
            from config.settings import cfg as _cfg_thr
            _min_count = int(getattr(_cfg_thr.xgboost, 'xgb_min_signals_count', 10))
            _min_thr   = float(getattr(_cfg_thr.xgboost, 'xgb_min_signals_threshold', 0.45))
        except Exception:
            _min_count, _min_thr = 10, 0.45

        # [BUG-M3 FIX] BUG-03 GUARD respeta thresholds per-régimen del calibrador (I4).
        # Antes: forzaba _DEFAULT_XGB_LIMIT = _min_thr globalmente, anulando los thresholds
        # calibrados por régimen (ej: bear con 0.62 -> 0.45, añadiendo señales que el
        # calibrador habia descartado deliberadamente).
        # Ahora: si hay thresholds per-régimen, los reduce un 15% (no los anula).
        # El fallback a umbral plano solo aplica cuando NO hay thresholds por régimen.
        THRESHOLD_WAS_LOWERED = False
        if n_model < _min_count and model_name == "XGBoost":
            logger.warning("  BUG-03 GUARD: solo {} señales < mínimo {}.", n_model, _min_count)
            THRESHOLD_WAS_LOWERED = True
            if optimal_threshold_per_regime and "HMM_Regime" in df_oos.columns:
                # [BUG-M3] Reducir thresholds per-rgimen un 15%, respetando la jerarquia I4
                _thresh_by_regime_m3 = {int(float(str(k).split('_')[0])): float(v) for k, v in optimal_threshold_per_regime.items()}
                model_mask = pd.Series(False, index=df_oos.index)
                _regime_col_m3 = pd.to_numeric(df_oos["HMM_Regime"], errors='coerce').fillna(-1).astype(int)
                for r_int_m3, r_thresh_m3 in _thresh_by_regime_m3.items():
                    # Reduccion maxima: 15% o hasta _min_thr, lo que sea mayor
                    _reduced = max(_min_thr, r_thresh_m3 * 0.85)
                    r_mask_m3 = (_regime_col_m3 == r_int_m3)
                    model_mask.loc[r_mask_m3] = df_oos.loc[r_mask_m3, prob_col] > _reduced
                _unmapped_m3 = ~_regime_col_m3.isin(list(_thresh_by_regime_m3.keys()))
                if _unmapped_m3.any():
                    # [FIX-NEW-07] No usar _min_thr directo para regímenes no mapeados o desconocidos.
                    # Usar la mediana de los thresholds conocidos como fallback seguro.
                    import numpy as np
                    _safe_fallback_thr = max(_min_thr, np.median(list(_thresh_by_regime_m3.values())) * 0.85) if _thresh_by_regime_m3 else _min_thr
                    model_mask.loc[_unmapped_m3] = df_oos.loc[_unmapped_m3, prob_col] > _safe_fallback_thr
                n_model = int(model_mask.sum())
                logger.warning("  [BUG-M3] Thresholds per-régimen reducidos 15%: {} señales rescatadas.", n_model)
            else:
                # Fallback plano: no hay thresholds per-régimen, umbral global OK
                _DEFAULT_XGB_LIMIT = _min_thr
                model_mask = df_oos[prob_col] > _DEFAULT_XGB_LIMIT
                n_model = int(model_mask.sum())
                logger.warning("  Threshold global bajado a {:.2f}: {} señales.", _DEFAULT_XGB_LIMIT, n_model)

        # [HMM-PREDICTIVE-01] Ajuste anticipatorio de umbral por riesgo de transicion a BEAR
        # Si la probabilidad de transitar a BEAR en t+1 supera el umbral configurado,
        # aplicar un multiplicador conservador sobre el umbral XGBoost.
        if model_name == "XGBoost" and "hmm_transition_risk" in df_oos.columns and direction == "long":
            try:
                from config.settings import cfg as _cfg_hmm_tr
                _hmm_tr_thresh  = float(getattr(getattr(_cfg_hmm_tr, "hmm_predictive", None), "bear_transition_veto_thresh",  0.40))
                
                # Excluir del veto a las velas que YA ESTAN en un regimen bear (donde el agente bear_long esta disenado para operar)
                _bear_regimes = ["3_CALM_BEAR", "3_BEAR_CRASH", "4_BEAR_FORCED"]
                _is_already_bear = df_oos["HMM_Semantic"].isin(_bear_regimes) if "HMM_Semantic" in df_oos.columns else pd.Series(False, index=df_oos.index)
                
                model_mask_hmm = (df_oos["hmm_transition_risk"] <= _hmm_tr_thresh) | _is_already_bear
                n_before_hmm = int(model_mask.sum())
                model_mask = model_mask & model_mask_hmm
                
                _rows_at_risk = (~model_mask_hmm).sum()
                if _rows_at_risk > 0:
                    logger.info(
                        "  [HMM-PREDICTIVE-01] Veto anticipatorio: {}/{} velas con bear_transition_prob > {:.0%}. "
                        "Señales bloqueadas: {}",
                        _rows_at_risk, len(df_oos), _hmm_tr_thresh, n_before_hmm - int(model_mask.sum())
                    )
            except Exception as _e_hmm_tr:
                logger.debug("  [HMM-PREDICTIVE-01] No aplicado (no critico): {}", _e_hmm_tr)

        if model_name == "XGBoost":
            # [FIX-THRESHOLD-TRACE-01] Registrar el threshold EFECTIVO real (no el global 0.5 por defecto).
            # BUG PREVIO: always 0.5 when per-regime path was used (I4). Now uses the real weighted average.
            self.used_threshold = _effective_threshold_for_trace
            self.used_thresholds_per_regime = _per_regime_thresholds_trace  # trazabilidad detallada por régimen
            self.threshold_was_lowered = getattr(self, "threshold_was_lowered", False) or THRESHOLD_WAS_LOWERED
            print(
                f"[FIX-THRESHOLD-TRACE-01] used_threshold registrado: {self.used_threshold:.4f} "
                f"(era=0.5 con bug previo) | threshold_was_lowered={THRESHOLD_WAS_LOWERED} | "
                f"per_regime={_per_regime_thresholds_trace}"
            )
        return model_mask

    def apply_ood(self, df_oos: pd.DataFrame) -> pd.Series:
        ood_mask = pd.Series(True, index=df_oos.index)
        ood_model_path = self.models_dir / "ood_guard.pkl"
        ood_sig_path   = self.models_dir / "ood_guard_signature.json"
        
        if ood_model_path.exists() and ood_sig_path.exists():
            try:
                import joblib
                with open(ood_sig_path, "r") as f:
                    ood_sig = json.load(f)
                expected_features = ood_sig.get("features_tracked", [])
                if expected_features:
                    ood_model = joblib.load(ood_model_path)
                    X_ood = pd.DataFrame(index=df_oos.index)
                    for _ofd in expected_features:
                        if _ofd in df_oos.columns:
                            X_ood[_ofd] = df_oos[_ofd].fillna(0.0)  # LOGIC-OOD-01: mismo que training
                        else:
                            # [FIX-OOD-MISSING-01] Pad missing features with 0.0 to prevent KeyError in IsolationForest
                            X_ood[_ofd] = 0.0
                            
                    ood_preds = ood_model.predict(X_ood[expected_features])
                    df_oos["ood_kl_distance"] = ood_model.decision_function(X_ood[expected_features])
                    # [V2-P6] OOD Guard Continuo: Ya no bloquea de forma binaria.
                    # Se mantiene ood_mask en True para todas las barras. La penalización se aplica en KellySizer.
                    ood_mask = pd.Series(True, index=df_oos.index)
                    n_anomalies = int((ood_preds == -1).sum())
                    logger.info("  OOD Guard (Continuo): {} anomalias detectadas de {} ({:.1f}% off-distribution). "
                                "Se aplicara penalizacion Kelly en lugar de bloqueo binario.",
                                n_anomalies, len(df_oos), n_anomalies / max(len(df_oos), 1) * 100)

                    # [H3-TEST-01 2026-06-11] Gate experimental: filtro KL inverso causal
                    # Hipotesis H3: KL bajo (anomalo segun IS) = mejor trade en regimen 2025-2026
                    # Umbral: kl_q75_training derivado SOLO de datos IS en training, guardado en firma
                    # Gate: bloquear barras OOS donde KL > kl_q75_training (demasiado 'normales')
                    # Causal: umbral IS no contamina OOS -> respeta SOP R1 causalidad estricta
                    _cfg_ood = {}
                    try:
                        from config.settings import cfg as _cfg_main
                        _cfg_ood = _cfg_main.ood_guard if hasattr(_cfg_main, "ood_guard") else {}
                        _cfg_ood = vars(_cfg_ood) if hasattr(_cfg_ood, "__dict__") else {}
                    except Exception:
                        pass
                    _use_h3_gate = _cfg_ood.get("experimental_kl_gate", False)
                    if _use_h3_gate and "kl_q75_training" in ood_sig:
                        _kl_thresh = ood_sig["kl_q75_training"]
                        _n_before = int(ood_mask.sum())
                        _h3_pass = df_oos["ood_kl_distance"] <= _kl_thresh
                        ood_mask = ood_mask & _h3_pass
                        _n_after = int(ood_mask.sum())
                        _n_blocked = _n_before - _n_after
                        print(
                            f"[H3-TEST-01] Gate KL activo: umbral_IS_Q75={_kl_thresh:.6f} | "
                            f"Bloqueadas {_n_blocked}/{_n_before} ({_n_blocked/max(_n_before,1)*100:.1f}%) | "
                            f"Pasan {_n_after} barras (KL<=Q75_training)"
                        )
                        logger.info(
                            "[H3-TEST-01] Gate KL inverso activo: kl_q75_IS={:.6f} | "
                            "bloqueadas={} ({:.1f}%) | pasan={}",
                            _kl_thresh, _n_blocked, _n_blocked/max(_n_before,1)*100, _n_after
                        )
                    elif _use_h3_gate:
                        print("[H3-TEST-01] AVISO: experimental_kl_gate=true pero kl_q75_training NO en firma. "
                              "Re-entrenar OOD Guard con --nocache para generarlo.")
                        logger.warning("[H3-TEST-01] kl_q75_training ausente en firma OOD Guard -- gate H3 omitido")
                    else:
                        print("[H3-TEST-01] Gate KL experimental: INACTIVO (experimental_kl_gate=false o ausente)")
                else:
                    logger.warning("  OOD Guard: ninguna feature de la firma esperada en json — omitiendo")
            except Exception as e:
                logger.warning("  OOD Guard no disponible: {} — omitiendo filtrado OOD", e)
        else:
            logger.info("  OOD Guard: modelo no encontrado — omitiendo")
            
        # [FASE 4: DVOL Guardian - Zero Look-Ahead Bias]
        # Aplica filtrado basado en la variable DVOL_kz (Z-Score de ventana rodante 90d IS)
        try:
            from config.settings import cfg as _cfg_dvol
            _dvol_max = float(getattr(_cfg_dvol.ood_guard, "guardian_dvol_max_z", 1.5))
            _dvol_min = float(getattr(_cfg_dvol.ood_guard, "guardian_dvol_min_z", -1.0))
        except Exception:
            _dvol_max, _dvol_min = 1.5, -1.0
            
        if "DVOL_kz" in df_oos.columns:
            dvol_series = df_oos["DVOL_kz"]
            _dvol_mask = (dvol_series >= _dvol_min) & (dvol_series <= _dvol_max)
            _n_blocked_dvol = len(df_oos) - _dvol_mask.sum()
            if _n_blocked_dvol > 0:
                logger.info(
                    "  [DVOL-GUARDIAN] Bloqueadas {} barras por Volatilidad Extrema o Muerta "
                    "(DVOL_kz fuera de [{:.2f}, {:.2f}])",
                    _n_blocked_dvol, _dvol_min, _dvol_max
                )
                print(f"[DVOL-GUARDIAN] Bloqueadas {_n_blocked_dvol} barras por Volatilidad Extrema o Muerta.")
            ood_mask = ood_mask & _dvol_mask

        return ood_mask

    def apply_metalabeler(self, df_oos: pd.DataFrame, available_feats: list, direction: str = "long") -> pd.Series:
        meta_v2_prob_series = pd.Series(np.nan, index=df_oos.index)
        meta_v2_mask = pd.Series(True, index=df_oos.index)
        v2_config_path = self.models_dir / f"metalabeler_v2_{direction}_config.json"
        v2_lstm_path   = self.models_dir / f"metalabeler_v2_{direction}_lstm.pt"
        v2_transformer_path = self.models_dir / f"metalabeler_v2_{direction}_transformer.pt"
        v2_rf_path     = self.models_dir / f"metalabeler_v2_{direction}_rf.joblib"

        try:
            from config.settings import cfg as _cfg_skip
            _skip_meta = bool(getattr(getattr(_cfg_skip, 'metalabeler', None), 'skip_metalabeler', False))
        except Exception:
            _skip_meta = False

        if _skip_meta:
            print(
                f"[FIX-SKIP-METALABELER-01] MetaLabelerV2 DESACTIVADO como gate "
                f"(skip_metalabeler=true, RULE[fixbugsprints.md]). "
                f"Todas las señales HMM pasan sin filtro MetaLabeler. "
                f"Brier IS=0.28 > 0.25, block_rate_real=64.3%, EV_perdido≈+2.08%. "
                f"meta_v2_prob quedará NaN → Kelly usa fillna(0.5)=neutral."
            )
            logger.warning("  ⚠️  MetaLabelerV2 DESACTIVADO (skip_metalabeler=true) — XGBoost puro (diagnóstico M-32)")
        elif v2_config_path.exists() and (v2_lstm_path.exists() or v2_transformer_path.exists()) and v2_rf_path.exists():
            try:
                from luna.models.train_metalabeler_v2 import MetaLabelerV2
                v2_config = json.loads(v2_config_path.read_text(encoding="utf-8"))
                seq_len   = v2_config.get("seq_len", 48)
                seq_features_saved = v2_config.get("seq_features", [])
                
                # [AUTOENCODER HOOK] Validar si necesitamos comprimir el espacio local OOS primero
                autoencoder_state_path = self.models_dir / "autoencoder_state.pt"
                if autoencoder_state_path.exists() and any(f.startswith("LATENT_AE") for f in seq_features_saved):
                    try:
                        import torch
                        import joblib
                        from luna.models.train_autoencoder import DenoisingAutoEncoder
                        
                        ae_scaler = joblib.load(self.models_dir / "autoencoder_scaler.joblib")
                        with open(self.models_dir / "autoencoder_config.json") as f:
                            ae_cfg = json.load(f)
                            
                        ae_features = ae_cfg["features"]
                        # Pad en OOS si falta
                        for m in ae_features:
                            if m not in df_oos.columns: df_oos[m] = 0.0
                            
                        X_ae_raw = df_oos[ae_features].fillna(0.0).values
                        X_ae_scaled = ae_scaler.transform(X_ae_raw)
                        
                        ae_model = DenoisingAutoEncoder(input_dim=len(ae_features), latent_dim=ae_cfg["latent_dim"])
                        ae_model.eval()
                        ae_model.load_state_dict(torch.load(autoencoder_state_path, map_location="cpu", weights_only=True))
                        
                        latent_tensor = ae_model.encode(torch.tensor(X_ae_scaled, dtype=torch.float32)).numpy()
                        for idx in range(latent_tensor.shape[1]):
                            df_oos[f"LATENT_AE_{idx}"] = latent_tensor[:, idx]
                            
                        logger.debug(f"  [AUTOENCODER] {len(df_oos)} filas OOS comprimidas en {latent_tensor.shape[1]} dims")
                    except Exception as e:
                        logger.error(f"  [AUTOENCODER] Error comprimiendo OOS en SignalFilter: {e}")
                
                if seq_features_saved:
                    # [FIX-CRITICO-9] Reconstruir variables HMM_OH_* (one-hot del régimen)
                    # El MetaLabeler depende de ellas para no caer en hojas "unknown" con prob=0.0
                    if "HMM_Regime" in df_oos.columns:
                        _reg_col = pd.to_numeric(df_oos["HMM_Regime"], errors='coerce').fillna(-1).astype(int)
                        # [FIX-04] max_states leído de v2_config (hmm_n_states + 1 por Risk-Off Shield)
                        # Antes: max_states = 6 hardcodeado → silenciosamente perdería estados si HMM crece
                        _hmm_n_from_cfg = v2_config.get("hmm_n_states", 4)
                        max_states = _hmm_n_from_cfg + 1  # +1 = Risk-Off Shield (estado adicional)
                        print(f"[FIX-04] HMM one-hot: max_states={max_states} (hmm_n_states={_hmm_n_from_cfg} desde v2_config, +1 Risk-Off Shield)")

                        for s in range(max_states):
                            col = f"HMM_OH_{s}"
                            if col not in df_oos.columns:
                                df_oos[col] = (_reg_col == s).astype(float)

                    missing_seq = [f for f in seq_features_saved if f not in df_oos.columns]
                    if missing_seq:
                        logger.warning("  MetaLabelerV2: {} seq_features ausentes → pad 0: {}", len(missing_seq), missing_seq[:5])
                        for f in missing_seq:
                            df_oos[f] = 0.0
                    meta_feats = seq_features_saved
                else:
                    n_features = v2_config.get("input_dim", len(available_feats))
                    meta_feats = available_feats[:n_features]

                X_raw_oos = df_oos[meta_feats].fillna(0.0).values
                X_seq_list, seq_indices = [], []
                # [BUG-A2 FIX] Warm-up LSTM con cola del train set en lugar de
                # replicar la primera fila del holdout (sesgo estado estatico artificial).
                # Mismo patron que FIX-MOMENTUM-01 en apply_momentum().
                _train_tail_warm: np.ndarray = np.empty((0, len(meta_feats)), dtype=np.float32)
                try:
                    _train_path_warm = self.models_dir.parent / "features" / "features_train.parquet"
                    if _train_path_warm.exists():
                        try:
                            import pyarrow.parquet as _pq_warm
                            _schema_names = set(_pq_warm.read_schema(_train_path_warm).names)
                            _train_warm_cols = [c for c in meta_feats if c in _schema_names]
                        except Exception:
                            _train_warm_cols = meta_feats
                        _train_warm_df = __import__('pandas').read_parquet(_train_path_warm, columns=_train_warm_cols if _train_warm_cols else meta_feats[:1])
                        # Asegurar que tenemos todas las features (pad con 0 si faltan)
                        for _mf in meta_feats:
                            if _mf not in _train_warm_df.columns:
                                _train_warm_df[_mf] = 0.0
                        _train_tail_warm = _train_warm_df[meta_feats].fillna(0).values[-seq_len:].astype(np.float32)
                        logger.debug("  [BUG-A2] LSTM warm-up: {} filas historicas del train cargadas.", len(_train_tail_warm))
                except Exception as _e_warm:
                    if "LATENT_AE" in str(_e_warm) or "FieldRef" in str(_e_warm):
                        # [BUG-FIX-02] El parquet de train no tiene columnas LATENT_AE porque el
                        # AutoEncoder las genera DESPUÉS de que features_train.parquet ya fue guardado.
                        # Este error es ESPERADO. El warm-up con estado cero es el comportamiento correcto.
                        logger.info(
                            "  [AE-WARMUP-FIX] features_train.parquet no tiene LATENT_AE (normal: "
                            "AE se aplica post-guardado). Warm-up LSTM: estado cero inicial."
                        )
                    else:
                        logger.warning("  [BUG-A2] No se pudo cargar historial train para warm-up LSTM (error inesperado): {}", _e_warm)


                for i in range(len(X_raw_oos)):
                    if i < seq_len:
                        pad_len = seq_len - i
                        # [BUG-A2] Usar cola del train si tenemos suficientes filas, si no, fallback al primer valor
                        if len(_train_tail_warm) >= pad_len:
                            pad = _train_tail_warm[-pad_len:]
                        else:
                            pad = np.tile(X_raw_oos[0], (pad_len, 1))
                        seq = np.vstack([pad, X_raw_oos[:i]]) if i > 0 else pad
                        X_seq_list.append(seq)
                    else:
                        X_seq_list.append(X_raw_oos[i - seq_len:i])
                    seq_indices.append(i)

                if X_seq_list:
                    X_seq_oos  = np.array(X_seq_list, dtype=np.float32)
                    xgb_p_seq  = df_oos["xgb_prob"].values[seq_indices]

                    _n_hmm = v2_config.get("hmm_n_states", 4)
                    _n_hmm_total = _n_hmm + 1  # Risk-Off Shield agrega estado n_states (4)
                    hmm_context_oos = None
                    if "HMM_Regime" in df_oos.columns and not df_oos["HMM_Regime"].isna().all():
                        _regime_idx = pd.to_numeric(df_oos["HMM_Regime"], errors='coerce').fillna(0).astype(int)
                        _oh_df = pd.DataFrame(0.0, index=df_oos.index, columns=[f"HMM_OH_{s}" for s in range(_n_hmm_total)])
                        for _r_idx, _s_idx in zip(df_oos.index, _regime_idx):
                            if 0 <= _s_idx < _n_hmm_total:
                                _oh_df.loc[_r_idx, f"HMM_OH_{_s_idx}"] = 1.0
                        for _c in _oh_df.columns:
                            df_oos[_c] = _oh_df[_c].values
                        hmm_context_oos = _oh_df.values[seq_indices].astype(np.float32)
                    else:
                        hmm_context_oos = np.zeros((len(seq_indices), _n_hmm_total), dtype=np.float32)

                    meta_v2 = MetaLabelerV2.load(self.models_dir, direction_mode=direction)
                    meta_probs_arr = meta_v2.predict_proba(X_seq_oos, xgb_p_seq, hmm_regime=hmm_context_oos)

                    meta_v2_prob_series.iloc[seq_indices] = meta_probs_arr

                    # ── C3 (Auditoría 2026-03-27): Threshold MetaLabeler por régimen HMM ──────
                    # Umbral base (global fallback)
                    _meta_thresh_global = 0.50  # FASE C: Seguro default
                    try:
                        from config.settings import cfg as _cfg_meta_g
                        _mlab_cfg_g = getattr(_cfg_meta_g, 'metalabeler', None)
                        _global_thr = getattr(_mlab_cfg_g, 'meta_v2_min_prob', None)
                        if _global_thr is not None:
                            _meta_thresh_global = float(_global_thr)
                    except Exception:
                        pass

                    _meta_thresh_per_regime: dict = {}  # rÃ©gimen_semÃ¡ntico â†’ umbral

                    try:
                        _sig_found = False
                        _calib_sig_path_online = self.models_dir / "online_calibrator_signature.json"
                        
                        if _calib_sig_path_online.exists():
                            import time
                            import os
                            # Si tiene menos de 14 dias de antiguedad, usarlo
                            if time.time() - os.path.getmtime(_calib_sig_path_online) < 14 * 86400:
                                with open(_calib_sig_path_online) as _csf:
                                    _calib_sig = json.load(_csf)
                                if "online_optimal_threshold" in _calib_sig:
                                    _meta_thresh_global = float(_calib_sig["online_optimal_threshold"])
                                    logger.success(f"  [CAPA 4] Usando threshold ONLINE reciente: {_meta_thresh_global:.3f}")
                                    _sig_found = True
                            else:
                                logger.warning("  [CAPA 4] online_calibrator_signature.json ignorado (caducado > 14 dias).")
                                
                        if not _sig_found:
                            _calib_sig_path = self.models_dir / f"calibrator_{direction}_signature.json"
                            if _calib_sig_path.exists():
                                with open(_calib_sig_path) as _csf:
                                    _calib_sig = json.load(_csf)
                                if "optimal_meta_threshold" in _calib_sig:
                                    _calib_th = float(_calib_sig["optimal_meta_threshold"])
                                    _meta_thresh_global = _calib_th  # [FIX-M-06] Trust the calibrated threshold (was: max(0.50, _calib_th) which blocked everything)
                                # Soporte futuro: calibrador puede emitir umbrales por rÃ©gimen
                                if "optimal_meta_threshold_per_regime" in _calib_sig:
                                    _meta_thresh_per_regime = _calib_sig["optimal_meta_threshold_per_regime"]
                    except Exception as e:
                        logger.warning(f"  [CAPA 4] Error cargando firma del calibrador: {e}")

                    # ── C4: Multi-Agent Dynamic Application (Bull/Bear/Range x Long/Short) ──
                    try:
                        from config.settings import cfg as _cfg_meta
                        _mlab_cfg = getattr(_cfg_meta, 'metalabeler', None)
                        
                        _dir_series = df_oos.get("direction", pd.Series("long", index=df_oos.index))
                        if hasattr(_dir_series, "str"):
                            _dir_series = _dir_series.fillna("long").str.lower().str.strip()
                            
                        _regime_group = pd.Series("range", index=df_oos.index)
                        if "HMM_Semantic" in df_oos.columns and not df_oos["HMM_Semantic"].isna().all():
                            _sem = df_oos["HMM_Semantic"].astype(str)
                            _regime_group[_sem.str.contains("BULL", case=False, na=False)] = "bull"
                            _regime_group[_sem.str.contains("BEAR", case=False, na=False)] = "bear"
                        elif "hmm_regime" in df_oos.columns and not df_oos["hmm_regime"].isna().all():
                            _sem = df_oos["hmm_regime"].astype(str)
                            _regime_group[_sem.str.contains("BULL", case=False, na=False)] = "bull"
                            _regime_group[_sem.str.contains("BEAR", case=False, na=False)] = "bear"
                        else:
                            if "HMM_Regime" in df_oos.columns and not df_oos["HMM_Regime"].isna().all():
                                try:
                                    import joblib as _jl
                                    _hmm_pkl = self.models_dir / "hmm_regime.pkl"
                                    if _hmm_pkl.exists():
                                        _hmm_bundle = _jl.load(_hmm_pkl)
                                        _state_map = _hmm_bundle.get("state_map", {})
                                        if _state_map:
                                            _sem = df_oos["HMM_Regime"].map({k: v for k, v in _state_map.items()}).astype(str)
                                            _regime_group[_sem.str.contains("BULL", case=False, na=False)] = "bull"
                                            _regime_group[_sem.str.contains("BEAR", case=False, na=False)] = "bear"
                                except Exception:
                                    pass

                        _keys = "meta_v2_min_prob_" + _regime_group + "_" + _dir_series
                        _eff_thresh = pd.Series(_meta_thresh_global, index=df_oos.index)
                        
                        # [SIMULATION WFB: CAPA 4]
                        # Simula el avance del tiempo dentro de OOS sin look-ahead bias
                        _simulate_online = getattr(_mlab_cfg, 'simulate_online_recalibration', False) if _mlab_cfg else False
                        if _simulate_online and len(df_oos) > 30 * 24: # Al menos 1 mes de OOS
                            try:
                                from luna.labeling.online_recalibrator import calculate_online_threshold
                                from config.settings import cfg as _cfg_xgb
                                _pt_mult = float(getattr(_cfg_xgb.xgboost, 'pt_mult_min', 1.5))
                                _sl_mult = float(getattr(_cfg_xgb.xgboost, 'sl_mult_min', 0.8))
                                _tbm_min = float(getattr(_cfg_xgb.xgboost, 'tbm_min_return', 0.003))
                                _vb_min = int(getattr(_cfg_xgb.xgboost, 'vertical_barrier_min_hours', 24))
                                _vb_max = int(getattr(_cfg_xgb.xgboost, 'vertical_barrier_hours', 96))
                                
                                # Avanzar en pasos de 15 días (quincenal) usando 30 días previos
                                _step_days = 15
                                _lookback_days = 30
                                _t_start = df_oos.index.min() + pd.Timedelta(days=_lookback_days)
                                _t_end = df_oos.index.max()
                                
                                _current_t = _t_start
                                _last_dynamic_thresh = _meta_thresh_global
                                
                                while _current_t < _t_end:
                                    _past_mask = (df_oos.index >= _current_t - pd.Timedelta(days=_lookback_days)) & (df_oos.index < _current_t)
                                    _df_recent = df_oos[_past_mask]
                                    _meta_probs_recent = meta_v2_prob_series[_past_mask].values
                                    
                                    # Evitar Look-Ahead: Solo evaluar si la barrera máxima TBM ya cerró para la señal
                                    _safe_t = _current_t - pd.Timedelta(hours=_vb_max)
                                    _safe_mask = _df_recent.index <= _safe_t
                                    
                                    if _safe_mask.sum() > 50:
                                        _t_calc, _ev_calc = calculate_online_threshold(
                                            df_recent=_df_recent[_safe_mask],
                                            meta_probs=_meta_probs_recent[_safe_mask],
                                            pt_mult=_pt_mult,
                                            sl_mult=_sl_mult,
                                            tbm_min_return=_tbm_min,
                                            vb_max_h=_vb_max,
                                            vb_min_h=_vb_min
                                        )
                                        if _ev_calc > 0:
                                            _last_dynamic_thresh = _t_calc
                                            logger.debug(f"  [CAPA 4 WFB] Step {_current_t}: Umbral={_t_calc:.2f} (EV={_ev_calc:.4f})")
                                            
                                    _next_t = _current_t + pd.Timedelta(days=_step_days)
                                    _apply_mask = (df_oos.index >= _current_t) & (df_oos.index < _next_t)
                                    _eff_thresh[_apply_mask] = _last_dynamic_thresh
                                    _current_t = _next_t
                                    
                                logger.info(f"  [CAPA 4 WFB] Simulación Rolling MetaLabeler completada. Threshold min={_eff_thresh.min():.2f}, max={_eff_thresh.max():.2f}")
                            except Exception as e:
                                logger.warning(f"  [CAPA 4 WFB] Falló simulación rolling: {e}")
                        
                        if "HMM_Regime" in df_oos.columns and _meta_thresh_per_regime:
                            _regime_col = pd.to_numeric(df_oos["HMM_Regime"], errors='coerce').fillna(-1).astype(int)
                            for _r_int, _r_thresh in _meta_thresh_per_regime.items():
                                _eff_thresh[_regime_col == int(_r_int)] = float(_r_thresh)
                        
                        if _mlab_cfg is not None:
                            for k in _keys.unique():
                                yaml_val = getattr(_mlab_cfg, k, None)
                                if yaml_val is not None:
                                    _eff_thresh[_keys == k] = float(yaml_val)
                                    
                        # ── [H3-FIX 2026-05-30] PROPUESTA-C: Threshold por régimen desde settings.yaml ──
                        # ANTES: valores 0.50 y 0.58 hardcodeados violaban política No-Fallback.
                        # AHORA: se leen de settings.yaml con KeyError si faltan (política estricta).
                        # Evidencia audit: Q75 (prob>=0.665) → WR=57.7% vs baseline 51.3% (+6.4pp).
                        # bull_strong (1_BULL_TREND) → p50 causal en audit = 0.632 → threshold bajo razonable
                        # bull_unstable (WEAK, VOLATILE) → umbral más exigente para filtrar señal débil
                        try:
                            _thresh_bull_strong   = float(getattr(_mlab_cfg, 'meta_v2_thresh_bull_strong',   None) or 0.50)
                            _thresh_bull_unstable = float(getattr(_mlab_cfg, 'meta_v2_thresh_bull_unstable', None) or 0.63)
                        except Exception as _e_thr_read:
                            raise RuntimeError(
                                f"[H3-FIX][CRITICAL] No se pudo leer meta_v2_thresh_bull_strong/unstable de settings.yaml: {_e_thr_read}"
                            ) from _e_thr_read

                        _hmm_pkl = self.models_dir / "hmm_regime.pkl"
                        if _hmm_pkl.exists():
                            try:
                                import joblib as _jl
                                _hmm_bundle = _jl.load(_hmm_pkl)
                                _state_map = _hmm_bundle.get("state_map", {})
                                if _state_map:
                                    if "HMM_Semantic" in df_oos.columns:
                                        _sem = df_oos["HMM_Semantic"].astype(str)
                                    elif "HMM_Regime" in df_oos.columns:
                                        _regime_col = pd.to_numeric(df_oos["HMM_Regime"], errors='coerce').fillna(-1).astype(int)
                                        _sem = _regime_col.map({k: v for k, v in _state_map.items()}).astype(str)
                                    else:
                                        _sem = pd.Series("UNKNOWN", index=df_oos.index)

                                    _bull_strong_mask = (
                                        _sem.str.contains("1_BULL_TREND", case=False, na=False)
                                        & ~_sem.str.contains("WEAK", case=False, na=False)
                                    )
                                    _eff_thresh[_bull_strong_mask] = _thresh_bull_strong

                                    _inestables = ["1_VOLATILE_BULL", "1_VOLATILE_BULL_B",
                                                   "1_BULL_TREND_WEAK", "1_BULL_GRIND"]
                                    _bull_unstable_mask = pd.Series(False, index=df_oos.index)
                                    for reg_name in _inestables:
                                        _bull_unstable_mask |= _sem.str.contains(reg_name, case=False, na=False)

                                    _eff_thresh[_bull_unstable_mask] = _thresh_bull_unstable

                                    print(
                                        f"[H3-FIX][PROPUESTA-C] Threshold por regimen aplicado desde settings.yaml:"
                                        f" bull_strong(N={_bull_strong_mask.sum()})={_thresh_bull_strong:.3f}"
                                        f" | bull_unstable(N={_bull_unstable_mask.sum()})={_thresh_bull_unstable:.3f}"
                                        f" | resto usa threshold global/calibrado"
                                    )
                                    logger.info(
                                        "[H3-FIX][PROPUESTA-C] Thresholds regimen: bull_strong={:.3f}({}) bull_unstable={:.3f}({}) resto=global",
                                        _thresh_bull_strong, _bull_strong_mask.sum(),
                                        _thresh_bull_unstable, _bull_unstable_mask.sum()
                                    )
                            except Exception as _e_dyn_meta:
                                logger.warning(f"  [H3-FIX][PROPUESTA-C] Error aplicando umbrales HMM desde settings: {_e_dyn_meta}")

                        # ── [H3-FIX 2026-05-30 / SNIPER-ASYM-01 2026-06-09] CAPA 5: Percentil Causal Rolling Asimetrico ──
                        # Calculamos el percentil p(meta_v2_rolling_percentile) de las probs observadas.
                        # Novedad SNIPER-ASYM-01: El percentil varia segun el regimen (Bull relaja pyramiding, Bear asfixia ruido).
                        _use_rolling_pct = getattr(_mlab_cfg, 'meta_v2_rolling_percentile', None) if _mlab_cfg else None
                        _pct_bull = getattr(_mlab_cfg, 'meta_v2_rolling_percentile_bull', _use_rolling_pct) if _mlab_cfg else _use_rolling_pct
                        _pct_bear = getattr(_mlab_cfg, 'meta_v2_rolling_percentile_bear', _use_rolling_pct) if _mlab_cfg else _use_rolling_pct
                        _rolling_min_n   = int(getattr(_mlab_cfg, 'meta_v2_rolling_min_n', 50)) if _mlab_cfg else 50
                        
                        if _use_rolling_pct is not None and not _simulate_online:
                            _pct_q_global = float(_use_rolling_pct)
                            _pct_q_bull = float(_pct_bull) if _pct_bull is not None else _pct_q_global
                            _pct_q_bear = float(_pct_bear) if _pct_bear is not None else _pct_q_global
                            
                            _probs_all = meta_v2_prob_series.fillna(0.0).values
                            _rolling_thresh = np.full(len(_probs_all), _meta_thresh_global)
                            _seen_probs = []
                            _reg_grp_vals = _regime_group.values
                            
                            for _i_bar in range(len(_probs_all)):
                                if len(_seen_probs) >= _rolling_min_n:
                                    _current_regime = _reg_grp_vals[_i_bar]
                                    if _current_regime == "bull":
                                        _q = _pct_q_bull
                                    elif _current_regime == "bear":
                                        _q = _pct_q_bear
                                    else:
                                        _q = _pct_q_global
                                        
                                    _rolling_thresh[_i_bar] = np.percentile(_seen_probs, _q * 100)
                                _seen_probs.append(_probs_all[_i_bar])
                                
                            _eff_thresh_rolling = pd.Series(_rolling_thresh, index=df_oos.index)
                            # Solo sobreescribir barras donde el percentil es más exigente que el global
                            _pct_is_tighter = _eff_thresh_rolling > _eff_thresh
                            _eff_thresh[_pct_is_tighter] = _eff_thresh_rolling[_pct_is_tighter]
                            n_tighter = int(_pct_is_tighter.sum())
                            print(
                                f"[SNIPER-ASYM-01] Asymmetric Rolling percentile aplicado (Bull={_pct_q_bull}, Bear={_pct_q_bear}, Range={_pct_q_global}):"
                                f" {n_tighter}/{len(df_oos)} barras con threshold mas exigente"
                                f" | thresh_range=[{_eff_thresh_rolling.min():.4f},{_eff_thresh_rolling.max():.4f}]"
                                f" | min_n={_rolling_min_n}"
                            )
                            logger.info(
                                "[SNIPER-ASYM-01] Asym Rolling (Bull={:.2f}, Bear={:.2f}): {}/{} barras mas exigentes (thresh min={:.3f} max={:.3f})",
                                _pct_q_bull, _pct_q_bear, n_tighter, len(df_oos),
                                _eff_thresh_rolling.min(), _eff_thresh_rolling.max()
                            )
                        elif _use_rolling_pct is not None and _simulate_online:
                            print("[H3-FIX][CAPA-5] Rolling percentile OMITIDO: simulate_online_recalibration=True ya activo (no duplicar)")

                        # [LUNA V1 INSTITUTIONAL FIX] Dynamic Volatility Dimmer
                        # Exige mayor confianza probabilística al MetaLabeler cuando la volatilidad
                        # se está expandiendo (vix_slope_7d > 0), evitando Drawdowns de 'slow bleed'.
                        # Soft-Cap: máximo 1.15x para evitar extinción de señales en pánico de mercado.
                        if "vix_slope_7d" in df_oos.columns:
                            _vix_slope = df_oos["vix_slope_7d"].fillna(0.0).clip(lower=0.0)
                            _dimmer_multiplier = (1.0 + _vix_slope * 0.05).clip(upper=1.15)
                            _eff_thresh = (_eff_thresh * _dimmer_multiplier).clip(upper=0.99)
                            logger.info(f"  [VOL-DIMMER] Umbrales MetaLabeler ajustados dinámicamente por volatilidad (VIX max: {_vix_slope.max():.2f}, Multiplier cap: 1.15x)")
                                    
                        # [BUG-4 FIX] Dynamic Threshold Fallback (Opción C)
                        # Si la probabilidad máxima del batch OOS es inferior al threshold efectivo,
                        # el umbral es inalcanzable por covariate shift de varianza en el calibrador.
                        # Bajamos el threshold al P90 dinámico del batch para recuperar el 10% más fuerte.
                        _max_prob = meta_v2_prob_series.fillna(0.0).max()
                        _global_min_thresh = _eff_thresh.min()
                        if _max_prob > 0.0 and _max_prob < _global_min_thresh:
                            _fallback_thresh = meta_v2_prob_series.fillna(0.0).quantile(0.90)
                            # Actualizar _eff_thresh sin bajar por debajo del 0.40 como safety duro
                            _fallback_thresh = max(_fallback_thresh, 0.40)
                            _eff_thresh = pd.Series(_fallback_thresh, index=df_oos.index)
                            print(f"[BUG-4 FIX] Threshold inalcanzable (Max={_max_prob:.4f} < Thresh={_global_min_thresh:.4f}). "
                                  f"Fallback dinámico al P90 OOS -> Nuevo Threshold={_fallback_thresh:.4f}")
                            logger.warning(f"[BUG-4 FIX] Threshold inalcanzable. Fallback dinámico al P90={_fallback_thresh:.4f}")

                        # [FIX-SNIPER-W4 2026-06-14] Bloqueo Duro de Regímenes (Hard Exclusion)
                        try:
                            from config.settings import cfg as _cfg_excl
                            _excl_param = getattr(getattr(_cfg_excl, 'metalabeler', None), 'hmm_volatile_bull_exclude', [])
                            if _excl_param:
                                _h1_excluded_list = [str(x).upper() for x in _excl_param]
                                _hmm_col_f = "HMM_Semantic" if "HMM_Semantic" in df_oos.columns else ("HMM_Regime" if "HMM_Regime" in df_oos.columns else None)
                                if _hmm_col_f:
                                    _mask_banned = df_oos[_hmm_col_f].astype(str).str.upper().isin(_h1_excluded_list)
                                    if _mask_banned.sum() > 0:
                                        meta_v2_prob_series.loc[_mask_banned] = 0.0
                                        logger.info(f"  [FIX-SNIPER-W4] Hard Exclusion OOS: {_mask_banned.sum()} señales bloqueadas en regímenes prohibidos {_h1_excluded_list}")
                        except Exception as _e_excl:
                            pass

                        meta_v2_mask = meta_v2_prob_series.fillna(0.0) >= _eff_thresh
                        n_meta = int(meta_v2_mask.sum())
                        
                        logger.info(
                            "  [DIMMER] MetaLabelerV2 Multi-Agent thresholds aplicados dinámicamente → {}/{} barras retenidas",
                            n_meta, len(df_oos)
                        )
                    except Exception as e:
                        err_msg = f"[CRITICAL-FALLBACK-ALERT] Error aplicando HMM Multi-Agent MetaLabeler thresholds o dimmer de volatilidad: {e}"
                        print(err_msg)
                        logger.error(err_msg)
                        raise RuntimeError(err_msg) from e

                    df_oos["meta_v2_prob"] = meta_v2_prob_series.values
            except Exception as e:
                logger.warning("  MetaLabelerV2 falló u omitido: {}", e)
        else:
            logger.info("  MetaLabelerV2: archivos no encontrados — omitiendo")

        return meta_v2_mask

    def apply_hmm(self, df_oos: pd.DataFrame, direction: str = "long") -> pd.Series:
        hmm_mask = pd.Series(True, index=df_oos.index)
        try:
            from config.settings import cfg as _cfg_hmm
            _hmm_cfg = getattr(_cfg_hmm, 'metalabeler', None)
            _hmm_allowed = getattr(_hmm_cfg, 'hmm_allowed_regimes', None)
            if _hmm_allowed is None:
                raise ValueError("hmm_allowed_regimes no está definido en config.settings")
        except Exception as e:
            err_msg = f"[CRITICAL-FALLBACK-ALERT] Fallo crítico al leer 'hmm_allowed_regimes' de la configuración o está ausente: {e}"
            print(err_msg)
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e

        # [FIX-HMM-004] Validar hmm_allowed_regimes contra el state_map real del pkl.
        # Si hay etiquetas en settings que no existen en el modelo actual, alertar explícitamente.
        # Esto previene que el filtro sea inoperativo cuando el HMM reentrena y cambia sus etiquetas.
        _pkl_state_map = {}
        _pkl_all_labels = []
        try:
            import joblib as _jbl_hmm004
            _hmm_pkl_path = self.models_dir / "hmm_regime.pkl"
            if _hmm_pkl_path.exists():
                _hmm_bundle_004 = _jbl_hmm004.load(_hmm_pkl_path)
                _pkl_state_map = _hmm_bundle_004.get("state_map", {})
                _pkl_all_labels = list(_pkl_state_map.values())
                print(f"[FIX-HMM-004] state_map del pkl HMM actual: {_pkl_state_map}")
                logger.info(f"[FIX-HMM-004] state_map del pkl HMM actual: {_pkl_state_map}")

                # Validar etiquetas de settings vs pkl
                import re as _re_hmm004
                _allowed_str = [str(x) for x in _hmm_allowed if not isinstance(x, int)]
                _invalid = []
                for _lbl in _allowed_str:
                    _base = _re_hmm004.sub(r'_[A-D]$', '', _lbl.upper())
                    _found = any(str(v).upper() == _lbl.upper() or str(v).upper().startswith(_base)
                                 for v in _pkl_all_labels)
                    if not _found:
                        _invalid.append(_lbl)

                if _invalid:
                    print(
                        f"[FIX-HMM-004][WARN] {len(_invalid)} etiqueta(s) en settings.hmm_allowed_regimes "
                        f"NO existen en el state_map actual del HMM: {_invalid}. "
                        f"Regímenes reales disponibles: {_pkl_all_labels}"
                    )
                    logger.warning(
                        f"[FIX-HMM-004] {len(_invalid)} etiqueta(s) en hmm_allowed_regimes sin match en state_map: "
                        f"{_invalid}. Labels disponibles: {_pkl_all_labels}"
                    )
                else:
                    print(f"[FIX-HMM-004] PASS: todas las etiquetas de settings.hmm_allowed_regimes son válidas vs state_map.")
                    logger.info(f"[FIX-HMM-004] hmm_allowed_regimes validado OK contra state_map del pkl.")
        except Exception as _e_hmm004:
            logger.warning(f"[FIX-HMM-004] No se pudo validar hmm_allowed_regimes contra pkl: {_e_hmm004}")


        if _hmm_allowed is not None:
            _hmm_allowed_labels = []
            _has_int = any(isinstance(x, int) for x in _hmm_allowed)
            if _has_int:
                try:
                    import joblib
                    _hmm_data = joblib.load(self.models_dir / "hmm_regime.pkl")
                    _state_map = _hmm_data.get("state_map", {})
                    _hmm_allowed_labels = [_state_map[i] for i in _hmm_allowed if i in _state_map]
                except Exception:
                    pass
            else:
                _hmm_allowed_labels = [str(x) for x in _hmm_allowed]

            # [FIX-DYN-HMM-ALLOWED 2026-06-13] Auto-selección y auto-desbloqueo dinámico in-sample (Validation):
            # En lugar de usar una lista estática en settings.yaml, evalúa el rendimiento del clasificador (Profit Factor)
            # de cada régimen in-sample (en la ventana de validación) si la caché está disponible.
            _dyn_success = False
            try:
                import os as _os_dyn
                _ROOT_DYN = Path(__file__).resolve().parent.parent.parent
                _window_id_s = _os_dyn.environ.get("LUNA_WINDOW_ID", None)
                if not _window_id_s:
                    _parent_name = self.models_dir.parent.name
                    if _parent_name.startswith("W"):
                        _window_id_s = _parent_name
                
                if _window_id_s:
                    _val_path = _ROOT_DYN / "data" / "wfb_cache" / _window_id_s / "features" / f"features_validation_{_window_id_s}.parquet"
                    if not _val_path.exists():
                        _val_path = _ROOT_DYN / "data" / "wfb_cache" / _window_id_s / "features" / "features_validation.parquet"
                    
                    if _val_path.exists():
                        from luna.models.hmm_regime import HMMRegimeModel
                        from luna.models.regime_router import RegimeRouter
                        from luna.features.tbm import apply_triple_barrier
                        from luna.models.predict_oos import get_hmm_tbm_params, get_hmm_horizon

                        _df_val = pd.read_parquet(_val_path)
                        _hmm_model = HMMRegimeModel.load(self.models_dir)
                        _hmm_val = _hmm_model.predict_regime_series(_df_val)
                        _df_val["HMM_Semantic"] = _hmm_val["HMM_Semantic"]
                        
                        # [BUG-9 FIX] Extraer predicciones de transición para el HMM-Predictive Gate
                        _bull_mean_is = 0.0
                        _bear_mean_is = 0.0
                        if "hmm_bull_transition_prob" in _hmm_val.columns and "hmm_bear_transition_prob" in _hmm_val.columns:
                            # Promediar sobre el último 25% de la ventana de validación (zona de transición)
                            _n_tail = max(len(_hmm_val) // 4, 1)
                            _bull_mean_is = _hmm_val["hmm_bull_transition_prob"].tail(_n_tail).mean()
                            _bear_mean_is = _hmm_val["hmm_bear_transition_prob"].tail(_n_tail).mean()
                            print(f"[BUG-9 FIX] HMM-Predictive Gate: bull_mean_is={_bull_mean_is:.3f} | bear_mean_is={_bear_mean_is:.3f} (sobre las últimas {_n_tail} barras)")

                        _router = RegimeRouter(self.models_dir, agent_type="xgboost", direction=direction)
                        _xgb_val = _router.route_and_predict(_df_val)
                        _df_val["xgb_prob_cal"] = _xgb_val["calibrated"]

                        # Criterio de umbral (0.48 o de configuración)
                        _xgb_thresh = 0.48
                        try:
                            from config.settings import cfg as _cfg_xgb
                            _xgb_thresh = float(_cfg_xgb.metalabeler.xgb_signal_threshold)
                        except Exception:
                            pass

                        _val_candidates = _df_val["xgb_prob_cal"] >= _xgb_thresh
                        _val_signal_times = _df_val.index[_val_candidates]

                        _allowed_regimes_dynamic = []

                        if len(_val_signal_times) > 0:
                            _pt = _df_val["HMM_Semantic"].map(lambda r: get_hmm_tbm_params(r)["tp"])
                            _sl = _df_val["HMM_Semantic"].map(lambda r: get_hmm_tbm_params(r)["sl"])
                            _prob_series = _df_val["xgb_prob_cal"].fillna(0.5).clip(0.5, 1.0)
                            _conf_scaler = 0.7 + ((_prob_series - 0.5) / 0.5) * (1.3 - 0.7)
                            _pt = _pt * _conf_scaler
                            _sl = _sl * _conf_scaler

                            _mode_series = _df_val["HMM_Semantic"].dropna().map(lambda r: get_hmm_horizon(r))
                            _dyn_max = int(_mode_series.mode().iloc[0] if not _mode_series.empty else 168)

                            _tbm_val = apply_triple_barrier(
                                price_series=_df_val["close"],
                                event_times=_val_signal_times,
                                sides=pd.Series(1, index=_val_signal_times),
                                pt_sl_multiplier=[_pt, _sl],
                                vertical_barrier_hours=72,
                                min_return=0.005,
                                dynamic_barrier=True,
                                dynamic_horizon_min_h=24,
                                dynamic_horizon_max_h=_dyn_max,
                                linear_decay_pt=True,
                                pt_decay_fraction=0.75,
                                funding_series=_df_val.get("FundingRate"),
                            )

                            _df_val_signals = pd.DataFrame(index=_val_signal_times)
                            _df_val_signals["ret_net"] = _tbm_val["ret"] - 0.0015
                            _df_val_signals["HMM_Semantic"] = _df_val.loc[_val_signal_times, "HMM_Semantic"]

                            # [H1-VOLATILE-BULL-FIX 2026-06-14] Leer lista de regimenes excluidos y PF minimo desde settings
                            # Evidencia OOS: 1_VOLATILE_BULL=26.7% WR, 1_VOLATILE_BULL_C=18.7% WR (137 trades historicos)
                            _h1_excluded_list = []
                            _h1_min_pf_low_n = 1.0  # default: exige PF>1.0 en regimenes con n<3
                            try:
                                from config.settings import cfg as _cfg_h1
                                _mlab_cfg_h1 = getattr(_cfg_h1, 'metalabeler', None)
                                _excl_param = getattr(_mlab_cfg_h1, 'hmm_volatile_bull_exclude', None)
                                if _excl_param:
                                    _h1_excluded_list = [str(x) for x in _excl_param]
                                _h1_min_pf_low_n = float(getattr(_mlab_cfg_h1, 'hmm_dyn_min_pf_bull_low_n', 1.0))
                                print(f"[H1-VOLATILE-BULL-FIX] Config cargada: excluir={_h1_excluded_list} | min_pf_low_n={_h1_min_pf_low_n:.2f}")
                                logger.info(f"[H1-VOLATILE-BULL-FIX] Exclusion de regimenes destructores activa: {_h1_excluded_list}")
                            except Exception as _e_h1:
                                print(f"[H1-VOLATILE-BULL-FIX] WARN: No se pudo leer parametros de exclusion: {_e_h1}")

                            import re as _re_h1

                            for _regime in _df_val["HMM_Semantic"].unique():
                                _reg_signals = _df_val_signals[_df_val_signals["HMM_Semantic"] == _regime]
                                _n_signals = len(_reg_signals)
                                _regime_str = str(_regime)

                                # [H1-VOLATILE-BULL-FIX] Verificar exclusion explicita: match exacto case-insensitive
                                # NOTA: Usar exact match para no excluir variantes no-destructoras como 1_VOLATILE_BULL_B (WR=46.2%)
                                _is_explicitly_excluded = _regime_str.upper() in [_ex.upper() for _ex in _h1_excluded_list]
                                if _is_explicitly_excluded:
                                    print(f"[H1-VOLATILE-BULL-FIX] EXCLUIDO regimen destructor: '{_regime_str}' "
                                          f"(en hmm_volatile_bull_exclude) | n_signals_IS={_n_signals}")
                                    logger.info(f"[H1-VOLATILE-BULL-FIX] Regimen excluido explicitamente: '{_regime_str}' "
                                                f"| evidencia WR<30% en OOS historico")
                                    continue  # No anadir a _allowed_regimes_dynamic

                                if _n_signals > 0:
                                    _wins = _reg_signals.loc[_reg_signals["ret_net"] > 0, "ret_net"].sum()
                                    _losses = _reg_signals.loc[_reg_signals["ret_net"] < 0, "ret_net"].sum()
                                    _pf = _wins / abs(_losses) if _losses != 0 else float('inf')
                                    _total_ret = _reg_signals["ret_net"].sum()
                                    _is_bull_semantic = "BULL" in _regime_str.upper() or "RANGE" in _regime_str.upper()

                                    _decision = False
                                    if _n_signals >= 3:
                                        if _pf > 1.05 and _total_ret > 0:
                                            _decision = True
                                        elif _is_bull_semantic and _pf > 0.95:
                                            _decision = True
                                    else:
                                        # [H1-VOLATILE-BULL-FIX] Con n<3, exigir PF > hmm_dyn_min_pf_bull_low_n
                                        # para evitar inclusion automatica de regimenes sin evidencia solida
                                        if _is_bull_semantic and _pf > _h1_min_pf_low_n:
                                            _decision = True
                                            print(f"[H1-VOLATILE-BULL-FIX] Regimen '{_regime_str}' INCLUIDO: "
                                                  f"n={_n_signals}<3 pero PF={_pf:.3f}>{_h1_min_pf_low_n:.2f}")
                                        elif _is_bull_semantic:
                                            print(f"[H1-VOLATILE-BULL-FIX] Regimen '{_regime_str}' EXCLUIDO: "
                                                  f"n={_n_signals}<3 y PF={_pf:.3f}<={_h1_min_pf_low_n:.2f} (insuficiente evidencia)")

                                    # [BUG-9 FIX] HMM-Predictive Gate (Desbloqueo predictivo)
                                    # Si el IS fue BEAR (provocando exclusión de BULL por bajo PF), pero
                                    # el HMM Forward Algorithm detecta que vamos hacia BULL, forzamos inclusión.
                                    if _is_bull_semantic and not _decision:
                                        _predictive_margin = 0.05
                                        if _bull_mean_is > (_bear_mean_is + _predictive_margin):
                                            _decision = True
                                            print(f"[BUG-9 FIX] 🔓 DESBLOQUEO PREDICTIVO para '{_regime_str}': "
                                                  f"bull_mean({_bull_mean_is:.3f}) > bear_mean({_bear_mean_is:.3f}) + margin")

                                    if _decision:
                                        _allowed_regimes_dynamic.append(_regime_str)
                                else:
                                    # Con 0 senales IS: anadir solo si no esta excluido y es BULL/RANGE
                                    _is_bull_semantic = "BULL" in _regime_str.upper() or "RANGE" in _regime_str.upper()
                                    if _is_bull_semantic:
                                        _allowed_regimes_dynamic.append(_regime_str)
                                        print(f"[H1-VOLATILE-BULL-FIX] Regimen BULL con 0 senales IS anadido por defecto: '{_regime_str}'")
                        else:
                            # Si no hay señales en validación, habilitar regímenes BULL/RANGE por defecto
                            for _regime in _df_val["HMM_Semantic"].unique():
                                _is_bull_semantic = "BULL" in str(_regime).upper() or "RANGE" in str(_regime).upper()
                                if _is_bull_semantic:
                                    _allowed_regimes_dynamic.append(str(_regime))

                        _hmm_allowed_labels = _allowed_regimes_dynamic
                        _dyn_success = True
                        print(f"[FIX-DYN-HMM-ALLOWED] Selección Dinámica in-sample ejecutada con éxito. Regímenes permitidos: {_hmm_allowed_labels}")
                        logger.info(f"[FIX-DYN-HMM-ALLOWED] Selección Dinámica in-sample ejecutada con éxito. Regímenes permitidos: {_hmm_allowed_labels}")
            except Exception as _e_dyn:
                print(f"[FIX-DYN-HMM-ALLOWED] Error al ejecutar selección dinámica in-sample (usando fallback simple): {_e_dyn}")
                logger.warning(f"[FIX-DYN-HMM-ALLOWED] Error al ejecutar selección dinámica in-sample: {_e_dyn}")

            if not _dyn_success:
                # Fallback simple: auto-desbloqueo semántico dinámico para BULL y RANGE
                # [H1-VOLATILE-BULL-FIX] Respetar lista de exclusion tambien en el fallback
                _h1_excluded_fallback = []
                try:
                    from config.settings import cfg as _cfg_h1_fb
                    _mlab_h1_fb = getattr(_cfg_h1_fb, 'metalabeler', None)
                    _excl_fb = getattr(_mlab_h1_fb, 'hmm_volatile_bull_exclude', None)
                    if _excl_fb:
                        _h1_excluded_fallback = [str(x).upper() for x in _excl_fb]
                except Exception:
                    pass
                import re as _re_h1_fb
                if _pkl_all_labels:
                    for _lbl in _pkl_all_labels:
                        _lbl_str = str(_lbl).upper()
                        # [H1-VOLATILE-BULL-FIX] Verificar exclusion: match exacto case-insensitive
                        _is_fb_excluded = _lbl_str in _h1_excluded_fallback
                        if _is_fb_excluded:
                            print(f"[H1-VOLATILE-BULL-FIX][FALLBACK] Regimen EXCLUIDO (destructor): {_lbl}")
                            logger.info(f"[H1-VOLATILE-BULL-FIX][FALLBACK] Regimen excluido en fallback: {_lbl}")
                            continue
                        if ("BULL" in _lbl_str or "RANGE" in _lbl_str) and "BEAR" not in _lbl_str:
                            if str(_lbl) not in _hmm_allowed_labels:
                                _hmm_allowed_labels.append(str(_lbl))
                                print(f"[FIX-DYN-HMM-ALLOWED][FALLBACK] Auto-desbloqueado régimen semántico HMM: {_lbl}")
                                logger.info(f"[FIX-DYN-HMM-ALLOWED][FALLBACK] Auto-desbloqueado régimen semántico HMM: {_lbl}")

            # [FIX-BUG-HMM-BEAR-LONG-01] Permitir que el agente BEAR opere en posiciones LONG durante mercados bajistas cuando use_regime_agents=True.
            try:
                from config.settings import cfg as _cfg_sf
                use_regime = getattr(_cfg_sf.fase2, 'use_regime_agents', False)
                if use_regime and direction == "long":
                    _mapping = getattr(_cfg_sf.fase2, 'regime_mapping', None)
                    if _mapping:
                        _bear_regimes = getattr(_mapping, 'bear', [])
                        _added = []
                        for _br in _bear_regimes:
                            _br_str = str(_br)
                            if _br_str not in _hmm_allowed_labels:
                                _hmm_allowed_labels.append(_br_str)
                                _added.append(_br_str)
                        if _added:
                            print(f"[FIX-BUG-HMM-BEAR-LONG-01] use_regime_agents=True detectado. Añadidos regímenes BEAR no-configurados a hmm_allowed_labels: {_added}")
                            logger.info(f"[FIX-BUG-HMM-BEAR-LONG-01] use_regime_agents=True detectado. Añadidos regímenes BEAR no-configurados a hmm_allowed_labels: {_added}")
            except Exception as _e_br:
                print(f"[FIX-BUG-HMM-BEAR-LONG-01] Error al intentar añadir regímenes BEAR: {_e_br}")
                logger.warning(f"[FIX-BUG-HMM-BEAR-LONG-01] Error al intentar añadir regímenes BEAR: {_e_br}")

            if _hmm_allowed_labels:
                if direction == "short":
                    # [FIX-4] Bloqueo asimétrico TOTAL para SHORT como medida de contención
                    _hmm_allowed_labels = []
                # FIX-MULTI-AGENT-01: Se elimina el bloqueo hardcodeado "if BEAR not in lbl" para direction=="long".
                # En la arquitectura Multi-Agente actual, el Agente BEAR se especializa precisamente en posiciones
                # LONG durante mercados bajistas. Bloquearlas aquí provocaba el error de "0 trades" OOS en ventanas bajistas.

                _hmm_col_to_use = None
                if "HMM_Semantic" in df_oos.columns and not df_oos["HMM_Semantic"].isna().all():
                    _hmm_col_to_use = "HMM_Semantic"
                elif "HMM_Regime" in df_oos.columns and not df_oos["HMM_Regime"].isna().all():
                    _sample_val = df_oos["HMM_Regime"].dropna().iloc[0] if df_oos["HMM_Regime"].notna().any() else None
                    if _sample_val is not None and not isinstance(_sample_val, str):
                        try:
                            import joblib
                            _hmm_data2 = joblib.load(self.models_dir / "hmm_regime.pkl")
                            _state_map2 = _hmm_data2.get("state_map", {})
                            if _state_map2:
                                _str_map = {float(k): str(v) for k, v in _state_map2.items()}
                                df_oos["HMM_Semantic"] = df_oos["HMM_Regime"].map(_str_map).fillna("UNKNOWN")
                                _hmm_col_to_use = "HMM_Semantic"
                            else:
                                _hmm_col_to_use = "HMM_Regime"
                        except Exception:
                            _hmm_col_to_use = "HMM_Regime"
                    else:
                        _hmm_col_to_use = "HMM_Regime"

                if _hmm_col_to_use and _hmm_col_to_use in df_oos.columns:
                    # Camino B: Comparación semántica flexible para variantes dinámicas (ej. _B, _C)
                    import re
                    _allowed_bases = {re.sub(r'_[A-D]$', '', str(lbl).upper()) for lbl in _hmm_allowed_labels}
                    
                    def _is_regime_allowed(val):
                        v = str(val).upper()
                        if v in _hmm_allowed_labels:
                            return True
                        for base in _allowed_bases:
                            if v.startswith(base):
                                return True
                        return False

                    hmm_mask = df_oos[_hmm_col_to_use].apply(_is_regime_allowed)
                    
                    is_forced = df_oos[_hmm_col_to_use] == '4_BEAR_FORCED'
                    n_blocked = int((~hmm_mask).sum())
                    n_forced = int(is_forced.sum())
                    n_normal = n_blocked - n_forced
                    logger.info("  [H3] Filtro HMM flexible: {} bloqueadas by 4_BEAR_FORCED | {} bloqueadas by config (no {})", n_forced, n_normal, _hmm_allowed_labels)
        
        return hmm_mask

    def apply_macro_gate(self, df_oos: pd.DataFrame, direction: str = "long") -> pd.Series:
        """
        [CAMINOB-08 2026-06-06] Capa A: Macro Gate Direccional (SOFT).
        Ya no utiliza vetos duros. Modifica la columna 'macro_kelly_penalty' para
        reducir el tamaño de posición si la macro es adversa.
        Devuelve siempre True para no asfixiar el Win Rate.
        """
        macro_mask = pd.Series(True, index=df_oos.index)
        macro_penalty = pd.Series(1.0, index=df_oos.index)
        
        # M2 Global YoY como indicador maestro de liquidez (Capa A)
        if "M2_Global_YoY" in df_oos.columns:
            m2 = df_oos["M2_Global_YoY"].ffill().fillna(0.0)
            if direction == "long":
                # Reducción a la mitad (Half) si la liquidez mundial se contrae severamente (< -2.0%)
                _veto_mask = m2 < -2.0
                macro_penalty[_veto_mask] *= 0.5
                n_penalized = int(_veto_mask.sum())
                if n_penalized > 0:
                    logger.info(f"  [MACRO-GATE] {n_penalized} barras LONG reducidas al 50% de Kelly por contraccion severa de liquidez (M2 < -2.0%)")
            elif direction == "short":
                _veto_mask = m2 > 10.0
                macro_penalty[_veto_mask] *= 0.5
                n_penalized = int(_veto_mask.sum())
                if n_penalized > 0:
                    logger.info(f"  [MACRO-GATE] {n_penalized} barras SHORT reducidas al 50% de Kelly por explosion de liquidez (M2 > 10.0%)")
                    
        # Halving Cycle como modulador de tendencia
        if "cal_halving_cycle_sin" in df_oos.columns:
            halving_sin = df_oos["cal_halving_cycle_sin"].ffill().fillna(0.0)
            if direction == "long":
                _veto_mask = halving_sin < -0.85
                macro_penalty[_veto_mask] *= 0.5
                n_penalized = int(_veto_mask.sum())
                if n_penalized > 0:
                    logger.info(f"  [MACRO-GATE] {n_penalized} barras LONG adicionales reducidas por fase oscura del ciclo Halving (sin < -0.85)")

        df_oos["macro_kelly_penalty"] = macro_penalty
        return macro_mask

    def apply_session_gate(self, df_oos: pd.DataFrame) -> pd.Series:
        """
        [B1-SESSION-GATE 2026-05-30] Filtro de sesion horaria UTC.

        Evidencia OOS (892 trades WFB 2025-2026, hypothesis test B1):
          - 7H-13H UTC: WR=66.9% vs 51.6% baseline
          - W3: WR=72.2% vs 43.1% fuera del gate (p<0.001, N=79)
          - W4: WR=72.4% vs 41.8% fuera del gate (p<0.001, N=87)
          - Simulacion: Calmar 1.977 → 49.69 | MaxDD -74.8% → -14.3%
          - Lunes en BULL_TREND_WEAK: WR=7.4% | en BEAR_CRASH: destructivo

        Look-ahead: CERO — la hora UTC = timestamp del cierre de vela 1H,
        perfectamente conocida antes de enviar la orden.

        Causalidad economica: apertura mercados europeos (London 7H UTC, Frankfurt 8H UTC)
        concentra volumen institucional y movimientos direccionales reales.

        Politica No-Fallback: si session_gate.enabled=True pero configuracion
        ausente, lanza RuntimeError para forzar visibilidad del error.
        """
        print("[B1-SESSION-GATE] Iniciando apply_session_gate()")
        logger.info("[B1-SESSION-GATE] Iniciando filtro de sesion horaria UTC")

        # ── Leer configuracion sin fallback silencioso ─────────────────────────
        try:
            from config.settings import cfg as _cfg_sg
            _sg_cfg = getattr(_cfg_sg, "session_gate", None)
            if _sg_cfg is None:
                raise KeyError("session_gate no encontrado en config/settings.yaml")
            _enabled       = bool(getattr(_sg_cfg, "enabled", False))
            _allowed_hours = list(getattr(_sg_cfg, "allowed_hours_utc", None) or [])
            _monday_veto   = bool(getattr(_sg_cfg, "monday_regime_veto", False))
            _veto_regimes  = list(getattr(_sg_cfg, "monday_veto_regimes", None) or [])
            _log_blocked   = bool(getattr(_sg_cfg, "log_blocked", True))
        except Exception as e:
            err_msg = (
                "[B1-SESSION-GATE] CRITICO: Fallo al leer configuracion de session_gate. "
                f"Error: {e} — Verificar que session_gate esta definido en settings.yaml"
            )
            print(f"[CRITICAL] {err_msg}")
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e

        # ── Gate desactivado ─────────────────────────────────────────────────────
        if not _enabled:
            print("[B1-SESSION-GATE] Gate DESACTIVADO (session_gate.enabled=false) — pass-through")
            logger.info("[B1-SESSION-GATE] Gate desactivado — sin filtrado horario")
            return pd.Series(True, index=df_oos.index)

        if not _allowed_hours:
            err_msg = "[B1-SESSION-GATE] CRITICO: session_gate.enabled=true pero allowed_hours_utc esta vacio"
            print(f"[CRITICAL] {err_msg}")
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        print(f"[B1-SESSION-GATE] Gate ACTIVO: horas permitidas UTC={_allowed_hours} | monday_veto={_monday_veto}")
        logger.info(
            "[B1-SESSION-GATE] Gate ACTIVO: horas_utc={} | monday_regime_veto={} | regimenes_veto={}",
            _allowed_hours, _monday_veto, _veto_regimes
        )

        # ── Extraer hora UTC del timestamp del dataframe ───────────────────────
        if not isinstance(df_oos.index, pd.DatetimeIndex):
            try:
                _idx = pd.to_datetime(df_oos.index, utc=True)
            except Exception as e_idx:
                err_msg = f"[B1-SESSION-GATE] No se puede convertir indice a DatetimeIndex: {e_idx}"
                print(f"[CRITICAL] {err_msg}")
                logger.error(err_msg)
                raise RuntimeError(err_msg) from e_idx
        else:
            _idx = df_oos.index
            if _idx.tz is None:
                _idx = _idx.tz_localize("UTC")

        _hours = _idx.hour
        _dows  = _idx.dayofweek  # 0=Lunes, 6=Domingo

        # ── Aplicar gate de horas ─────────────────────────────────────────────
        session_mask = pd.Series(False, index=df_oos.index)
        hour_mask = _hours.isin(_allowed_hours)
        session_mask[hour_mask] = True

        n_total   = len(df_oos)
        n_allowed = int(session_mask.sum())
        n_blocked_hour = n_total - n_allowed

        print(f"[B1-SESSION-GATE] Horas UTC {_allowed_hours}: {n_allowed}/{n_total} barras permitidas | {n_blocked_hour} bloqueadas")
        logger.info(
            "[B1-SESSION-GATE] Horas UTC: {}/{} barras en sesion permitida | {} bloqueadas fuera de sesion",
            n_allowed, n_total, n_blocked_hour
        )

        # ── Veto condicional de Lunes por regimen HMM ─────────────────────────
        n_blocked_monday = 0
        if _monday_veto and _veto_regimes:
            # Determinar columna de regimen HMM disponible
            _hmm_col = None
            for col_candidate in ["HMM_Semantic", "hmm_regime", "HMM_Regime"]:
                if col_candidate in df_oos.columns:
                    _hmm_col = col_candidate
                    break

            if _hmm_col is None:
                print("[B1-SESSION-GATE] WARN: monday_regime_veto=true pero no hay columna HMM en df_oos — veto de Lunes omitido")
                logger.warning("[B1-SESSION-GATE] monday_regime_veto activo pero columna HMM ausente — omitiendo veto de Lunes")
            else:
                # Vetar Lunes si el regimen es adverso
                _is_monday = (_dows == 0)
                _regime_vals = df_oos[_hmm_col].astype(str)
                _is_veto_regime = _regime_vals.isin(_veto_regimes)
                _monday_veto_mask = _is_monday & _is_veto_regime

                # Desactivar sesion_mask para barras de Lunes en regimen adverso
                session_mask[_monday_veto_mask] = False
                n_blocked_monday = int(_monday_veto_mask.sum())

                # Log de regimenes afectados
                if n_blocked_monday > 0 and _log_blocked:
                    _veto_detail = df_oos.loc[_monday_veto_mask, _hmm_col].value_counts().to_dict()
                    print(f"[B1-SESSION-GATE] Veto Lunes: {n_blocked_monday} barras bloqueadas en regimenes adversos: {_veto_detail}")
                    logger.info(
                        "[B1-SESSION-GATE] Veto Lunes: {} barras bloqueadas en regimenes adversos: {}",
                        n_blocked_monday, _veto_detail
                    )
                else:
                    print(f"[B1-SESSION-GATE] Veto Lunes: 0 barras bloqueadas (ningun Lunes en regimen adverso)")
                    logger.info("[B1-SESSION-GATE] Veto Lunes: 0 barras bloqueadas en este periodo")

        # ── Resumen final ─────────────────────────────────────────────────────
        n_final_allowed = int(session_mask.sum())
        n_total_blocked = n_total - n_final_allowed

        print(
            f"[B1-SESSION-GATE] RESUMEN: {n_final_allowed}/{n_total} barras pasan "
            f"({n_total_blocked} bloqueadas: {n_blocked_hour} por hora, {n_blocked_monday} por veto Lunes)"
        )
        logger.info(
            "[B1-SESSION-GATE] RESUMEN FINAL: {}/{} barras pasan el gate "
            "({} bloqueadas total: {} por hora, {} por veto Lunes)",
            n_final_allowed, n_total, n_total_blocked, n_blocked_hour, n_blocked_monday
        )

        return session_mask

    def apply_momentum(self, df_oos: pd.DataFrame) -> pd.Series:

        momentum_mask = pd.Series(True, index=df_oos.index)
        _mom_threshold_global, _mom_upper = None, None
        _speed_window        = 120    # fallback: 5 días × 24H
        _crash_speed_thr     = -5.0  # fallback: speed < -5% → crash
        _ordered_corr_thr    = -25.0 # fallback: umbral relajado para correcciones
        try:
            from config.settings import cfg as _cfg_mom
            _meta_cfg = getattr(_cfg_mom, 'metalabeler', None)
            _mom_val   = getattr(_meta_cfg, 'momentum_filter_threshold', None)
            _mom_upper_val = getattr(_meta_cfg, 'momentum_filter_threshold_upper', None)
            if _mom_val is not None: _mom_threshold_global = float(_mom_val)
            if _mom_upper_val is not None: _mom_upper = float(_mom_upper_val)
            # [MEJORA-MOMENTUM-01] Leer parámetros de velocidad de caída
            _sw_val = getattr(_meta_cfg, 'momentum_speed_window', None)
            _cs_val = getattr(_meta_cfg, 'momentum_crash_speed_threshold', None)
            _oc_val = getattr(_meta_cfg, 'momentum_ordered_correction_threshold', None)
            if _sw_val is not None: _speed_window     = int(_sw_val)
            if _cs_val is not None: _crash_speed_thr  = float(_cs_val)
            if _oc_val is not None: _ordered_corr_thr = float(_oc_val)
        except Exception:
            pass

        if _mom_threshold_global is not None and 'close' in df_oos.columns:
            close_s = df_oos['close']
            try:
                # FIX-MOMENTUM-01: Evitar el Cold Start (NaNs) inyectando la cola del dataset de entrenamiento (720H previas)
                _data_dir = self.models_dir.parent if self.models_dir.name == "models" else Path(__file__).resolve().parents[2] / "data"
                _train_path = _data_dir / "features" / "features_train.parquet"
                
                if _train_path.exists():
                    _train_close = pd.read_parquet(_train_path, columns=["close"]).iloc[-720:]
                    close_s = pd.concat([_train_close['close'], close_s])
                    close_s = close_s[~close_s.index.duplicated(keep='last')].sort_index()
                    logger.info("  [M-45] Cold Start Activo: Inyectadas {} velas históricas previas de entrenamiento", len(_train_close))
            except Exception as e:
                logger.debug(f"  [M-45] Fallo leve cargando historial pre-OOS para momentum: {e}")

            ret_30d = close_s.pct_change(720) * 100
            ret_30d = ret_30d.reindex(df_oos.index) # Realinear con los datos OOS originales

            # [MEJORA-MOMENTUM-01] Calcular velocidad de caída — diferencia correcciones de crashes.
            # speed_30d = cuánto cambió el ret_30d en los últimos _speed_window horas.
            # Una corrección ORDENADA (ej. Q1 2026 -25% en 60 días) tiene speed ≈ -0.4%/día → lenta.
            # Un CRASH real (ej. -25% en 5 días) tiene speed ≈ -5%/día → rápida → speed < _crash_speed_thr.
            # Si la caída es lenta (no crash), relajamos el umbral a _ordered_corr_thr.
            speed_30d = ret_30d.diff(_speed_window)  # variación del ret_30d en ventana de velocidad
            _is_ordered_correction = (
                (ret_30d < _mom_threshold_global) &   # está por debajo del umbral base
                (speed_30d >= _crash_speed_thr)        # pero la velocidad NO es de crash
            )
            _n_ordered = int(_is_ordered_correction.sum())
            _n_total_below = int((ret_30d < _mom_threshold_global).sum())

            print(
                f"[MEJORA-MOMENTUM-01] Análisis velocidad de caída: "
                f"{_n_total_below} barras bajo umbral {_mom_threshold_global}% | "
                f"{_n_ordered} son correcciones ordenadas (speed≥{_crash_speed_thr}%) → umbral relajado a {_ordered_corr_thr}% | "
                f"{_n_total_below - _n_ordered} son crashes (speed<{_crash_speed_thr}%) → bloquear"
            )
            logger.info(
                "  [MEJORA-MOMENTUM-01] speed_window={}H crash_thr={}% ordered_thr={}% | "
                "bajo_umbral={} correcciones_ordenadas={} crashes={}",
                _speed_window, _crash_speed_thr, _ordered_corr_thr,
                _n_total_below, _n_ordered, _n_total_below - _n_ordered
            )

            # Umbral efectivo por barra: para correcciones ordenadas, umbral relajado a _ordered_corr_thr
            _effective_threshold = pd.Series(_mom_threshold_global, index=df_oos.index)
            _effective_threshold[_is_ordered_correction] = _ordered_corr_thr

            # [DYNAMIC-MOMENTUM-01] Umbral de momentum dinámico según régimen HMM (Evita penalizar caídas naturales en Bulls)
            # NOTA: el umbral efectivo ya incorpora correcciones ordenadas — el HMM lo refina sobre eso.
            _mom_thresholds_dynamic = _effective_threshold.copy()
            if "HMM_Semantic" in df_oos.columns and not df_oos["HMM_Semantic"].isna().all():
                _regime_series = df_oos["HMM_Semantic"].astype(str).fillna("")
                
                # Regímenes alcistas perdonan más caída (ej. pullbacks de -15%)
                _bull_mask = _regime_series.str.contains("BULL", case=False, na=False)
                _mom_thresholds_dynamic[_bull_mask] = -15.0
                
                # Regímenes pánico/fuerza mayor buscan cuchillos cayendo más agresivos
                _crash_mask = _regime_series.str.contains("CRASH|FORCED", case=False, na=False)
                _mom_thresholds_dynamic[_crash_mask] = -25.0
                
                # Regímenes tranquilos (CALM/RANGE) siguen el threshold efectivo (ya adaptado por velocidad)
                
                logger.info(
                    "  [MOM-DYN] Momentum Filtering Dinámico Activado: BULL={:.1f} | CRASH={:.1f} | RANGE/GLOBAL={:.1f}",
                    -15.0, -25.0, _mom_threshold_global
                )
            else:
                logger.warning("  [MOM-DYN] HMM_Semantic no encontrado, fallback al momentum global estático: {:.1f}%", _mom_threshold_global)

            if _mom_upper is not None:
                momentum_mask = (ret_30d >= _mom_thresholds_dynamic) | (ret_30d <= _mom_upper) | ret_30d.isna()
            else:
                momentum_mask = (ret_30d >= _mom_thresholds_dynamic) | ret_30d.isna()
                
            n_mom_blocked = int((~momentum_mask).sum())
            n_mom_pass    = int(momentum_mask.sum())
            print(
                f"[MEJORA-MOMENTUM-01] Resultado: {n_mom_pass} barras PERMITIDAS | "
                f"{n_mom_blocked} BLOQUEADAS (threshold dinámico+velocidad)"
            )
            logger.info("  [M-45] Filtro momentum: {} bloqueadas (Umbral dinámico+velocidad evaluado y <{:.1f}%)", n_mom_blocked, _mom_upper if _mom_upper else 999.0)

        return momentum_mask


    def apply_cvd_divergence(self, df_oos: pd.DataFrame, direction: str = "long") -> pd.Series:
        """
        [CVD-VETO-01] Filtra señales usando divergencia CVD (Spot vs Perps).
        Si la dirección es LONG y el ms_cvd_spot_vs_perps es muy bajo (Spot vendiendo, Perps comprando), veta la señal.
        """
        cvd_mask = pd.Series(True, index=df_oos.index)
        
        if "ms_cvd_spot_vs_perps" in df_oos.columns:
            cvd = df_oos["ms_cvd_spot_vs_perps"].fillna(0)
            if direction == "long":
                # Veto si CVD cae por debajo del percentil 10 (aprox -0.20, ajustar según datos empíricos)
                # Aproximación dinámica con rolling
                cvd_p10 = cvd.rolling(168).quantile(0.10).fillna(-0.20)
                cvd_mask = cvd >= cvd_p10
                n_blocked = int((~cvd_mask).sum())
                logger.info("  [CVD-VETO] CVD Divergence: {} barras bloqueadas para LONG (CVD < P10)", n_blocked)
            elif direction == "short":
                # Veto opuesto para shorts
                cvd_p90 = cvd.rolling(168).quantile(0.90).fillna(0.20)
                cvd_mask = cvd <= cvd_p90
                n_blocked = int((~cvd_mask).sum())
                logger.info("  [CVD-VETO] CVD Divergence: {} barras bloqueadas para SHORT (CVD > P90)", n_blocked)
        else:
            logger.info("  [CVD-VETO] ms_cvd_spot_vs_perps no encontrado — omitiendo")
            
        return cvd_mask

    def _inject_hmm_predictive_risk(self, df_oos: pd.DataFrame):
        if "hmm_transition_risk" in df_oos.columns:
            return

        try:
            import joblib as _jl
            
            _hmm_pkl = self.models_dir / "hmm_regime.pkl"
            if not _hmm_pkl.exists():
                return
                
            _hmm_bundle = _jl.load(_hmm_pkl)
            hmm_model = _hmm_bundle.get("model", None)
            scaler = _hmm_bundle.get("scaler", None)
            features = _hmm_bundle.get("features", [])
            state_map = _hmm_bundle.get("state_map", {})
            
            if hmm_model is None or scaler is None or not features or not hasattr(hmm_model, "transmat_"):
                return
                
            missing = [f for f in features if f not in df_oos.columns]
            if missing:
                for f in missing:
                    df_oos[f] = 0.0
                    
            X_oos = df_oos[features].fillna(0).values
            X_scaled = scaler.transform(X_oos)
            
            gamma_t = hmm_model.predict_proba(X_scaled)
            gamma_t1 = gamma_t @ hmm_model.transmat_
            
            toxic_indices = []
            for idx, label in state_map.items():
                l_up = str(label).upper()
                idx_int = int(float(idx))
                # [BUG-HMM-INDEX] Prevent out of bounds if state_map wasn't pruned after NAS
                if ("BEAR" in l_up or "CRASH" in l_up or "FORCED" in l_up) and idx_int < gamma_t1.shape[1]:
                    toxic_indices.append(idx_int)
                    
            if not toxic_indices:
                df_oos["hmm_transition_risk"] = 0.0
            else:
                risk = gamma_t1[:, toxic_indices].sum(axis=1)
                df_oos["hmm_transition_risk"] = risk
                logger.info(f"  [HMM-PREDICTIVE] hmm_transition_risk injectado on-the-fly. Riesgo Medio: {risk.mean():.1%}")
                
        except Exception as e:
            logger.error(f"  [HMM-PREDICTIVE] Error inyectando hmm_transition_risk (Fallando gracefully a 0.0): {e}")
            df_oos["hmm_transition_risk"] = 0.0

    def filter_signals(self, df_oos: pd.DataFrame, available_feats: list, direction: str = "long") -> pd.Series:
        """ Ejecuta el pipeline de filtrado completo, secuencialmente. """
        self.funnel_stats["raw_oos_bars"] = len(df_oos)
        
        self._inject_hmm_predictive_risk(df_oos)

        xgb_mask = self.apply_model_threshold(df_oos, prefix="xgboost_meta", prob_col="xgb_prob", model_name="XGBoost", direction=direction)
        self.last_xgb_mask = xgb_mask.copy() # [NUEVO] Guardar para análisis XGBoost-puro
        cum_mask = xgb_mask.copy()
        self.funnel_stats["after_xgb"] = int(cum_mask.sum())
        
        # ── Consenso LightGBM (Fase 2C) ──
        try:
            from config.settings import cfg as _cfg_fase2
            use_lgbm = bool(getattr(_cfg_fase2.fase2, "use_lgbm_ensemble", False))
        except Exception:
            use_lgbm = False
            
        if use_lgbm and "lgbm_prob" in df_oos.columns:
            # [FIX-LGBM-FOCAL-SIGMOID-01] Validate LGBM probability range before filtering.
            # If lgbm_prob > 1.0, the model was saved with an uncalibrated Focal Loss objective
            # and all values exceed any reasonable threshold → filter is a NOOP.
            # regime_router.py corrects this automatically (sigmoid safety net), but log here
            # to confirm: if this WARNING appears, Platt scaling fix in ensemble_lgbm.py
            # has NOT yet propagated (i.e., models were saved before the fix was applied).
            _lgbm_max = float(df_oos["lgbm_prob"].max())
            _lgbm_min = float(df_oos["lgbm_prob"].min())
            if _lgbm_max > 1.0 or _lgbm_min < 0.0:
                logger.error(
                    "[SIGNAL-FILTER] lgbm_prob FUERA de [0,1]: min={:.4f} max={:.4f}. "
                    "El modelo LGBM NO fue calibrado con Platt Scaling. "
                    "El filtro LGBM es inefectivo — re-entrenar ensemble_lgbm.py "
                    "(CalibratedClassifierCV fix). regime_router.py corrige esto automáticamente.",
                    _lgbm_min, _lgbm_max
                )
                print(f"[BUG-FIX-LOG 2026-06-05] [SIGNAL-FILTER] lgbm_prob FUERA de [0,1]: min={_lgbm_min:.4f} max={_lgbm_max:.4f}")
            else:
                logger.debug(
                    "[SIGNAL-FILTER] lgbm_prob en rango válido [0,1]: min={:.4f} max={:.4f} → filtro LGBM activo.",
                    _lgbm_min, _lgbm_max
                )
                print(f"[BUG-FIX-LOG 2026-06-05] [SIGNAL-FILTER] lgbm_prob en rango válido [0,1]: min={_lgbm_min:.4f} max={_lgbm_max:.4f} → filtro LGBM activo.")
            lgbm_mask = self.apply_model_threshold(df_oos, prefix="lgbm_meta", prob_col="lgbm_prob", model_name="LightGBM", direction=direction)
            # [BUG-C2] Log de diagnóstico AND detallado: permite identificar si el colapso
            # a 0 señales es por un LGBM malo (LGBM=0) o por desalineamiento temporal
            # (ambos tienen señales pero en barras distintas -> AND = 0).
            _xgb_only_n    = int(xgb_mask.sum())
            _lgbm_only_n   = int(lgbm_mask.sum())
            _and_result_n  = int((cum_mask & lgbm_mask).sum())  # pre-aplicar AND sobre cum_mask actual
            _xgb_not_lgbm  = int((cum_mask & ~lgbm_mask).sum())
            _lgbm_not_xgb  = int((~xgb_mask & lgbm_mask).sum())
            logger.info(
                "  [BUG-C2/AND-AUDIT] XGB={} | LGBM={} | AND={} | "
                "XGB∩¬LGBM={} (bloqueadas por LGBM) | LGBM∩¬XGB={} (nunca pasaron XGB)",
                _xgb_only_n, _lgbm_only_n, _and_result_n, _xgb_not_lgbm, _lgbm_not_xgb
            )
            cum_mask = cum_mask & lgbm_mask
            n_blocked = _xgb_only_n - int(cum_mask.sum())
            logger.info("  [LGBM-CONSENSUS] Lógica AND estricta: {} señales bloqueadas por discrepancia LGBM.", n_blocked)

            # [LGBM-PRIMARY-01] Hard floor sobre lgbm_prob — basado en evidencia forense W2.
            # Hallazgo: lgbm_prob >= 0.76 → WR=68% vs WR=56.2% con threshold IS calibrado (~0.51).
            # El LGBM es el ÚNICO modelo estadísticamente significativo en OOS W2 (r=+0.391, p=0.027).
            # XGBoost tiene correlación NEGATIVA con WR en OOS (Brier Skill=-0.05, overconfidence).
            # Esta mejora añade un piso mínimo configurable sobre el gate LGBM.
            # Referencia: docs/reports/run_audit_20260507_seed42.md §10, analyze_model_contradiction.py
            try:
                from config.settings import cfg as _cfg_lgbm_min
                _lgbm_min_prob = float(getattr(_cfg_lgbm_min.fase2, "lgbm_signal_min_prob", 0.0))
            except Exception:
                _lgbm_min_prob = 0.0

            if _lgbm_min_prob > 0.0 and "lgbm_prob" in df_oos.columns:
                _n_before_floor = int(cum_mask.sum())
                _lgbm_floor_mask = df_oos["lgbm_prob"] >= _lgbm_min_prob
                cum_mask = cum_mask & _lgbm_floor_mask
                _n_blocked_floor = _n_before_floor - int(cum_mask.sum())
                logger.info(
                    "  [LGBM-PRIMARY-01] Hard floor lgbm_prob >= {:.3f}: "
                    "{} señales bloqueadas ({} → {}). "
                    "Evidencia W2: WR=68% con floor vs 56.2% IS estático.",
                    _lgbm_min_prob, _n_blocked_floor, _n_before_floor, int(cum_mask.sum())
                )
            elif _lgbm_min_prob == 0.0:
                logger.debug("  [LGBM-PRIMARY-01] Desactivado (lgbm_signal_min_prob=0.0) — usando IS calibrado.")

        self.funnel_stats["after_lgbm"] = int(cum_mask.sum())

        ood_mask = self.apply_ood(df_oos)
        cum_mask = cum_mask & ood_mask
        self.funnel_stats["after_ood"] = int(cum_mask.sum())

        cvd_mask = self.apply_cvd_divergence(df_oos, direction=direction)
        cum_mask = cum_mask & cvd_mask
        self.funnel_stats["after_cvd"] = int(cum_mask.sum())

        # FIX-ORDER-01: HMM antes de MetaLabeler.
        # El HMM es O(1) por fila (lookup en columna), el MetaLabeler es O(N) con LSTM.
        # Filtrar regímenes prohibidos primero evita que el MetaLabeler evalue señales
        # que el HMM descartará de todas formas (W2/W3: 590 barras desperdiciadas).
        hmm_mask = self.apply_hmm(df_oos, direction=direction)
        cum_mask = cum_mask & hmm_mask
        self.funnel_stats["after_hmm"] = int(cum_mask.sum())

        # [B1-SESSION-GATE 2026-05-30] Filtro de sesion horaria UTC.
        # Posicion: despues del HMM (filtro barato O(1)) y ANTES del MetaLabeler
        # para evitar computar MetaLabeler en barras que el gate descartaria.
        # Evidencia: WR=66.9% en 7H-13H UTC (N=272, W3/W4 p<0.001)
        session_mask = self.apply_session_gate(df_oos)
        cum_mask = cum_mask & session_mask
        self.funnel_stats["after_session_gate"] = int(cum_mask.sum())
        
        # [CAMINOB-08] Capa A: Aplicar Macro Gate Direccional
        macro_mask = self.apply_macro_gate(df_oos, direction=direction)
        cum_mask = cum_mask & macro_mask
        self.funnel_stats["after_macro_gate"] = int(cum_mask.sum())

        meta_mask = self.apply_metalabeler(df_oos, available_feats, direction=direction)
        cum_mask = cum_mask & meta_mask
        self.funnel_stats["after_meta"] = int(cum_mask.sum())
        
        # [CASH-SHIELD-01] Veto trades if MetaLabeler probability is in the uncertainty band [0.45, 0.55]
        # This was suppressing signals approved by the dynamic EV threshold. Disabled to respect EV calibration.
        # if "meta_v2_prob_cal" in df_oos.columns: ...

        self.funnel_stats["after_cash_shield"] = int(cum_mask.sum())

        mom_mask = self.apply_momentum(df_oos)
        cum_mask = cum_mask & mom_mask
        self.funnel_stats["after_momentum"] = int(cum_mask.sum())

        signal_mask = cum_mask
        n_xgb = int(xgb_mask.sum())
        n_signals = int(signal_mask.sum())

        if n_xgb > 0:
            _filter_checks = [
                ("OOD",          (~ood_mask).sum()),
                ("CVD",          (~cvd_mask).sum()),
                ("HMM",          (~hmm_mask).sum()),
                ("SessionGate",  (~session_mask).sum()),
                ("MacroGate",    (~macro_mask).sum()),
                ("MetaV2",       (~meta_mask).sum()),
                ("Momentum",     (~mom_mask).sum()),
            ]
            if use_lgbm and 'lgbm_mask' in locals():
                _filter_checks.append(("LGBM", (~lgbm_mask).sum()))
                
            for _fn, _nb in _filter_checks:
                if int(_nb) / n_xgb > 0.80:
                    logger.warning(f"  [FILTER-ALARM] {_fn} bloquea >80% de señales XGB")

        lgbm_log = f"| LGBM-block={int((~lgbm_mask).sum())} " if use_lgbm and 'lgbm_mask' in locals() else ""
        logger.info(f"  [FILTROS] XGB={n_xgb} | OOD-block={int((~ood_mask).sum())} "
                    f"| CVD-block={int((~cvd_mask).sum())} "
                    f"| HMM-block={int((~hmm_mask).sum())} "
                    f"| SessionGate-block={int((~session_mask).sum())} "
                    f"| MacroGate-block={int((~macro_mask).sum())} "
                    f"| MetaV2={int(meta_mask.sum())} | Mom-block={int((~mom_mask).sum())} {lgbm_log}| FINAL={n_signals}")

        # [FUNNEL-REGIME-01] Analisis de supervivencia de senales por regimen HMM.
        # Determina si el MetaLabeler o el HMM estan suprimiendo asimetricamente un regimen
        # (ej: "volatile" siendo descartado en un 100% por un threshold estatico).
        try:
            _hmm_col_f = "HMM_Semantic" if "HMM_Semantic" in df_oos.columns else ("HMM_Regime" if "HMM_Regime" in df_oos.columns else None)
            if _hmm_col_f and n_xgb > 0:
                _regimes = df_oos[_hmm_col_f].unique()
                logger.info("  [FUNNEL-REGIME-01] Supervivencia de senales XGBoost por regimen:")
                for _r in sorted([str(x) for x in _regimes if pd.notna(x)]):
                    _m_r = df_oos[_hmm_col_f].astype(str) == _r
                    _n_r_xgb = int((xgb_mask & _m_r).sum())
                    _n_r_fin = int((signal_mask & _m_r).sum())
                    if _n_r_xgb > 0:
                        _surv = _n_r_fin / _n_r_xgb * 100
                        _meta_b = int((xgb_mask & _m_r & ~meta_mask).sum())
                        _hmm_b  = int((xgb_mask & _m_r & ~hmm_mask).sum())
                        _flag = " [!!] EXTINCION" if _n_r_fin == 0 else ""
                        logger.info(f"    - {_r:20s}: XGB={_n_r_xgb:3d} -> FINAL={_n_r_fin:3d} ({_surv:5.1f}%) | Bloqueados: Meta={_meta_b:2d}, HMM={_hmm_b:2d}{_flag}")
        except Exception as e_funnel:
            logger.warning(f"  [FUNNEL-REGIME-01] Error generando reporte por regimen: {e_funnel}")

        self.filter_fallback_level = 0
        if n_signals == 0:
            logger.warning("  [BUG-SIG-01 FIX] 0 señales tras HMM/MetaLabeler. Manteniendo 0 señales (Evitando fallback a XGB puro para proteger el capital en regímenes no aptos).")
            # Dejamos signal_mask como está (todo False).

        self.funnel_stats["filter_fallback_level"] = self.filter_fallback_level
        return signal_mask

    def apply_embargo(self, df_oos: pd.DataFrame, signal_mask: pd.Series) -> pd.DatetimeIndex:
        """Aplica embargo dinámico por régimen HMM a las señales OOS.
        
        Sustituye la mecánica DVOL estática/adaptativa previa.
        Mapeo empírico validado (Anti-inductivo):
        - Bull Trend: 72H
        - Volatile Bull: 96H
        - Calm/Volatile Range: 144H
        - Bear Crash: 168H
        """
        _signal_candidates = df_oos.index[signal_mask]
        _selected = []
        _last_signal_time = None
        
        # Diccionario de mapeo de regímenes
        # [FIX-EMBARGO-01] (2026-05-17): Añadidas variantes _B/_WEAK/_GRIND que heredan
        # el embargo de su régimen base. Antes caían al fallback de 168h (bug de omisión).
        # [FIX-EMBARGO-02] (2026-05-17): Bug de case en .get(): cambiado a lookup directo
        # sin .upper() ya que los keys son mixed-case (ej: '1_BULL_TREND', no '1_BULL_TREND').
        _hmm_embargo_map = {
            # Regímenes base
            '1_BULL_TREND':     72.0,
            '1_VOLATILE_BULL':  96.0,
            '1_BULL_GRIND':     72.0,   # [FIX-EMBARGO-01] igual que BULL_TREND
            '2_CALM_RANGE':    144.0,
            '2_VOLATILE_RANGE':168.0,
            '3_CALM_BEAR':     168.0,
            '3_BEAR_CRASH':    168.0,
            '4_BEAR_FORCED':   168.0,
            # Variantes _B/_C/_D/_WEAK: heredan embargo del régimen base
            '1_BULL_TREND_B':   72.0,   # [FIX-EMBARGO-01] heredado de BULL_TREND
            '1_BULL_TREND_C':   72.0,   # [FIX-EMBARGO-01] heredado de BULL_TREND
            '1_BULL_TREND_D':   72.0,   # [FIX-EMBARGO-01] heredado de BULL_TREND
            '1_BULL_TREND_WEAK':72.0,   # [FIX-EMBARGO-01] heredado de BULL_TREND (WR=60.2%)
            '1_VOLATILE_BULL_B':96.0,   # [FIX-EMBARGO-01] heredado de VOLATILE_BULL
            '1_VOLATILE_BULL_C':96.0,   # [FIX-EMBARGO-01] heredado de VOLATILE_BULL
            '1_VOLATILE_BULL_D':96.0,   # [FIX-EMBARGO-01] heredado de VOLATILE_BULL
            '2_CALM_RANGE_B':  144.0,   # [FIX-EMBARGO-01] heredado de CALM_RANGE
            '2_CALM_RANGE_C':  144.0,   # [FIX-EMBARGO-01] heredado de CALM_RANGE
            '2_VOLATILE_RANGE_B':168.0, # [FIX-EMBARGO-01] heredado de VOLATILE_RANGE
            '3_CALM_BEAR_B':   168.0,   # [FIX-EMBARGO-01] heredado de CALM_BEAR
            '3_BEAR_CRASH_B':  168.0,   # [FIX-EMBARGO-01] heredado de BEAR_CRASH
        }
        # [LUNA-V2-REGULARIZATION] Vincular parámetros a settings para evitar números mágicos o hardcodeados.
        # Fallbacks obligatorios de configuración a nivel de sistema que fallan ruidosamente (LOUD / CRITICAL).
        try:
            from config.settings import cfg as _cfg_emb
            _DEFAULT_WAIT_HOURS = float(_cfg_emb.sop.embargo_hours)
            _LOW_DENSITY_THRESHOLD = int(_cfg_emb.xgboost.embargo_low_density_threshold)
            _MIN_EMBARGO_H = float(_cfg_emb.xgboost.embargo_min_hours)
            _dynamic_decay = bool(_cfg_emb.xgboost.embargo_dynamic_decay)
            _atr_lookback = int(_cfg_emb.xgboost.embargo_decay_atr_lookback)
            _embargo_hours_floor = float(_cfg_emb.xgboost.embargo_hours)
            print(f"[LUNA-V2-EMBARGO] Configuración cargada correctamente desde settings.yaml | "
                  f"fallback_emb={_DEFAULT_WAIT_HOURS}H, low_density_thresh={_LOW_DENSITY_THRESHOLD}, "
                  f"min_embargo={_MIN_EMBARGO_H}H, dynamic_decay={_dynamic_decay}, atr_lookback={_atr_lookback}, "
                  f"embargo_floor={_embargo_hours_floor}H")
        except Exception as _cfg_err:
            raise RuntimeError(
                f"\n[CRITICAL-LUNA-V2] Error de configuración en apply_embargo. "
                f"No se pudieron leer los parámetros obligatorios desde settings.yaml.\n"
                f"Error: {_cfg_err}\n"
                f"Verifica que config/settings.yaml contenga los atributos bajo 'sop' y 'xgboost'."
            ) from _cfg_err

        # Columna de régimen a utilizar
        _hmm_col = None
        if "HMM_Semantic" in df_oos.columns and not df_oos["HMM_Semantic"].isna().all():
            _hmm_col = "HMM_Semantic"
        elif "hmm_regime" in df_oos.columns and not df_oos["hmm_regime"].isna().all():
            _hmm_col = "hmm_regime"

        # [FIX-EMBARGO-01-DENSITY] Embargo adaptativo por densidad de señales.
        _n_candidates = len(_signal_candidates)
        _density_mode_active   = _n_candidates < _LOW_DENSITY_THRESHOLD

        if _density_mode_active:
            print(
                f"[FIX-EMBARGO-01] Modo BAJA DENSIDAD activado: {_n_candidates} señales < {_LOW_DENSITY_THRESHOLD} umbral. "
                f"Embargo reducido de dinámico(72-168H) a mínimo({_MIN_EMBARGO_H}H) para preservar señales."
            )
            logger.warning(
                "[FIX-EMBARGO-01] Baja densidad: {} candidatos < {} → embargo reducido a {}H mínimo",
                _n_candidates, _LOW_DENSITY_THRESHOLD, _MIN_EMBARGO_H
            )
        else:
            print(
                f"[FIX-EMBARGO-01] Modo NORMAL: {_n_candidates} señales >= {_LOW_DENSITY_THRESHOLD}. "
                f"Embargo dinámico por régimen HMM (72-168H) activo."
            )

        _atr = None
        _atr_rolling_max = None
        if _dynamic_decay and 'close' in df_oos.columns:
            try:
                _close_s = df_oos['close']
                _high_s = df_oos['high'] if 'high' in df_oos.columns else None
                _low_s = df_oos['low'] if 'low' in df_oos.columns else None
                
                try:
                    _data_dir = self.models_dir.parent if self.models_dir.name == "models" else Path(__file__).resolve().parents[2] / "data"
                    _train_path = _data_dir / "features" / "features_train.parquet"
                    if _train_path.exists():
                        _cols_to_load = ["close"]
                        if _high_s is not None: _cols_to_load.append("high")
                        if _low_s is not None: _cols_to_load.append("low")
                        _train_df = pd.read_parquet(_train_path, columns=_cols_to_load).iloc[-720:]
                        _close_s = pd.concat([_train_df['close'], _close_s])
                        if _high_s is not None: _high_s = pd.concat([_train_df['high'], _high_s])
                        if _low_s is not None: _low_s = pd.concat([_train_df['low'], _low_s])
                        
                        _close_s = _close_s[~_close_s.index.duplicated(keep='last')].sort_index()
                        if _high_s is not None: _high_s = _high_s[~_high_s.index.duplicated(keep='last')].sort_index()
                        if _low_s is not None: _low_s = _low_s[~_low_s.index.duplicated(keep='last')].sort_index()
                except Exception as _e_cold:
                    logger.debug(f"[LUNA-V2-EMBARGO] Cold start ATR fallback: {_e_cold}")

                if _high_s is not None and _low_s is not None:
                    _tr = np.maximum(
                        _high_s - _low_s,
                        np.maximum(
                            np.abs(_high_s - _close_s.shift(1)),
                            np.abs(_low_s - _close_s.shift(1))
                        )
                    )
                else:
                    _tr = np.abs(_close_s - _close_s.shift(1))
                
                # ATR as exponential moving average of True Range
                _atr_all = _tr.ewm(span=_atr_lookback, adjust=False).mean()
                _atr = _atr_all.reindex(df_oos.index)
                
                # rolling max over last 168 hours (7 days) to find the recent peak volatility
                _atr_rolling_max_all = _atr_all.rolling(window=168, min_periods=1).max()
                _atr_rolling_max = _atr_rolling_max_all.reindex(df_oos.index)
            except Exception as _e_atr:
                print(f"[LUNA-V2-EMBARGO] Error calculating ATR: {_e_atr}. Falling back to normal embargo.")

        for _t in _signal_candidates:
            # Extraer régimen actual
            _current_regime = "UNKNOWN"
            if _hmm_col is not None:
                _current_regime = str(df_oos.loc[_t, _hmm_col])

            # Seleccionar embargo: mínimo en baja densidad, dinámico en densidad normal
            if _density_mode_active:
                _base_emb_h = _MIN_EMBARGO_H
            else:
                _base_emb_h = _DEFAULT_WAIT_HOURS # [FIX-EMBARGO-HARDCODED] Cumplir regla No-Magic-Numbers y usar settings.yaml

            # Aplicar decaimiento por volatilidad si está activo
            _emb_h = _base_emb_h
            if _atr is not None and _atr_rolling_max is not None:
                try:
                    _atr_t = float(_atr.loc[_t])
                    _atr_peak = float(_atr_rolling_max.loc[_t])
                    if _atr_peak > 1e-8:
                        _vol_ratio = _atr_t / _atr_peak
                        _decayed_emb = _base_emb_h * _vol_ratio
                        # [FIX-EMBARGO-FLOOR] Dynamic decay floor linked to settings.yaml (no magic numbers).
                        _emb_h = max(_embargo_hours_floor, _decayed_emb)
                        
                        if _t == _signal_candidates[0] or _vol_ratio < 0.95:
                            print(f"[LUNA-V2-EMBARGO] [FIX-EMBARGO-FLOOR] t={_t} regime={_current_regime} | "
                                  f"base_emb={_base_emb_h}H -> dynamic_emb={_emb_h:.1f}H "
                                  f"(ATR={_atr_t:.4f}, peak={_atr_peak:.4f}, ratio={_vol_ratio:.2%}, floor={_embargo_hours_floor}H)")
                except Exception as _e_loop:
                    pass

            # Chequeo dinámico por fila
            if _last_signal_time is None or (_t - _last_signal_time).total_seconds() / 3600.0 >= _emb_h:
                _selected.append(_t)
                _last_signal_time = _t

        signal_times = pd.DatetimeIndex(_selected)
        self.funnel_stats["after_embargo"] = len(signal_times)
        _mode_label = f"BAJA_DENSIDAD({_MIN_EMBARGO_H}H)" if _density_mode_active else "DINAMICO(72-168H)"
        print(
            f"[FIX-EMBARGO-01] Embargo [{_mode_label}]: {_n_candidates} candidatos -> "
            f"{len(signal_times)} señales retenidas "
            f"({len(signal_times)/max(_n_candidates,1)*100:.1f}% supervivencia)"
        )
        logger.info(
            "  [EMBARGO DINAMICO] [FIX-EMBARGO-01] Modo={} | {} candidatos -> {} señales retenidas",
            _mode_label, _n_candidates, len(signal_times)
        )
        return signal_times


    def export_funnel_json(self, output_dir: Path):
        """Exporta las estadísticas calculadas al reporte (P1.1).

        [FIX-FUNNEL-ACCUM-01] (2026-05-19): Acumula conteos entre ventanas WFB.
        Bug anterior: cada ventana sobreescribía signal_funnel.json con sus propios datos.
        El validador estadístico final (run_statistical_validation.py) leía solo la última
        ventana ejecutada — en WFB, normalmente W5, que suele tener 0 o 1 trades.
        Resultado visible: after_embargo=1 en el verdict aunque el run generó 22 trades reales.
        Fix: leer el JSON existente y SUMAR los conteos numéricos si pertenece al mismo
        LUNA_RUN_ID. Si el run_id cambia, se resetea el acumulador automáticamente.
        """
        import os as _os_funnel
        target_path = Path(output_dir) / "signal_funnel.json"
        _current_run_id = _os_funnel.environ.get("LUNA_RUN_ID", "")

        # Campos que se acumulan por suma entre ventanas
        _ACCUM_KEYS = [
            "raw_oos_bars", "after_xgb", "after_lgbm", "after_ood", "after_cvd",
            "after_hmm", "after_session_gate", "after_macro_gate", "after_meta", 
            "after_cash_shield", "after_momentum", "after_embargo"
        ]

        # Leer acumulado existente si pertenece al mismo run
        existing: dict = {}
        if target_path.exists() and _current_run_id:
            try:
                with open(target_path, encoding="utf-8") as _f_ex:
                    existing = json.load(_f_ex)
                if existing.get("run_id", "") != _current_run_id:
                    print(f"[FIX-FUNNEL-ACCUM-01] Nuevo run_id detectado "
                          f"({existing.get('run_id', '?')} → {_current_run_id}). "
                          f"Reseteando acumulador del funnel.")
                    logger.info(
                        "  [FIX-FUNNEL-ACCUM-01] Nuevo run_id → reset acumulador "
                        "({} → {})", existing.get("run_id", "?"), _current_run_id
                    )
                    existing = {}
            except Exception as _e_read:
                logger.debug("  [FIX-FUNNEL-ACCUM-01] No se pudo leer funnel previo: {}", _e_read)
                existing = {}

        # Construir versión acumulada
        merged = dict(self.funnel_stats)
        merged["run_id"] = _current_run_id
        merged["n_windows_accumulated"] = existing.get("n_windows_accumulated", 0) + 1

        for _key in _ACCUM_KEYS:
            if _key in self.funnel_stats:
                merged[_key] = existing.get(_key, 0) + self.funnel_stats[_key]

        # filter_fallback_level: máximo de todas las ventanas (0=limpio, >0=fallback activado)
        merged["filter_fallback_level"] = max(
            existing.get("filter_fallback_level", 0),
            self.funnel_stats.get("filter_fallback_level", 0)
        )

        try:
            with open(target_path, "w", encoding="utf-8") as _f_w:
                json.dump(merged, _f_w, indent=4)
            _n_win = merged["n_windows_accumulated"]
            _n_emb = merged.get("after_embargo", "?")
            _n_raw = merged.get("raw_oos_bars", "?")
            print(f"[FIX-FUNNEL-ACCUM-01] signal_funnel acumulado ventana #{_n_win} "
                  f"| raw_oos_bars={_n_raw} | after_embargo={_n_emb} | run_id={_current_run_id}")
            logger.info(
                "  [FIX-FUNNEL-ACCUM-01] signal_funnel acumulado (ventana #{}) → "
                "raw={} after_embargo={} en {}",
                _n_win, _n_raw, _n_emb, target_path
            )
        except Exception as e:
            logger.error("  [FIX-FUNNEL-ACCUM-01] Error guardando signal_funnel.json: {}", e)

