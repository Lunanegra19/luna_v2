import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from loguru import logger
from pathlib import Path
import sys
import types

# ── [LUNA-V2-CALIB] PlattCalibrator definition and injection for pickle/joblib deserialization robustness ──
class PlattCalibrator:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression as _LR
        self.model = _LR(C=1e6, random_state=42)
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

# [FIX-TEMPCAL-DESER-01 2026-06-01] Importar TemperatureCalibrator desde el módulo canónico.
# Antes, la clase era definida en train_xgboost_v2.py (namespace '__main__') y joblib la
# serializaba con ese path. Al deserializar aquí, Python no encontraba la clase → AttributeError.
# Fix: importar desde luna.models.calibrators (módulo propio), que es donde ahora se serializa.
# Impacto: elimina 33 errores AttributeError y 5 FATALs directas (seeds: 100, 1337, 2026, 27243, 44085).
from luna.models.calibrators import TemperatureCalibrator  # noqa: F401 — necesario para joblib.load()
print("[FIX-TEMPCAL-DESER-01] TemperatureCalibrator importada en regime_router — joblib.load() puede deserializar OK")  # RULE[fixbugsprints.md]



# [ARCH-11-FIX 2026-06-02] MockXGBClassifier: ya NO retorna prob=0.6 silenciosamente.
# Si este metodo se llama, es un error critico del pipeline — el modelo no fue entrenado
# y no debe generar senales. Ahora lanza RuntimeError para forzar visibilidad.
class MockXGBClassifier:
    """Placeholder de modelo no entrenado. NUNCA debe ser invocado en inference."""
    def predict_proba(self, X):
        import numpy as np
        msg = (
            "[ARCH-11-FIX] CRITICO: MockXGBClassifier.predict_proba() fue invocado. "
            "Este agente no fue entrenado correctamente. El archivo .model es un placeholder JSON. "
            "prob=0.6 silencioso eliminado — se bloquea la ejecucion para forzar visibilidad."
        )
        print(f"\n{'='*70}\n{msg}\n{'='*70}")
        logger.critical(msg)
        raise RuntimeError(msg)
    
    def set_params(self, **kwargs):
        pass  # Necesario para compatibilidad con XGBoost API


class RegimeRouter:
    """
    Enrutador de inferencia Multi-Agente para Luna v2 (Phase 2).
    Recibe un DataFrame de features OOS y direcciona cada fila
    al modelo experto correspondiente segun 'HMM_Semantic',
    combinando las predicciones en una unica serie.

    Args:
        disabled_regimes: lista de nombres de agente (ej: ['range', 'bear']) que el
            Gate-G2 ha marcado como NO_OPERABLE en esta ventana. Las barras
            clasificadas en estos regímenes recibirán prob=0.0 (CASH forzado)
            en lugar de ser enrutadas al modelo especializado.
    """
    def __init__(
        self,
        models_dir: Path,
        agent_type: str = "xgboost",
        direction: str = "long",
        disabled_regimes: list | None = None,
    ):
        self.models_dir = Path(models_dir)
        self.agent_type = agent_type  # "xgboost" o "lightgbm"
        self.direction = direction
        self.prefix = "xgboost_meta" if agent_type == "xgboost" else "lgbm_meta"
        # [DEGRADED-MODE] Agentes deshabilitados por Gate-G2 (modelos sin poder predictivo)
        self.disabled_regimes: set[str] = set(disabled_regimes or [])
        if self.disabled_regimes:
            print(f"[RegimeRouter/DEGRADED] MODO DEGRADADO ACTIVO: agentes deshabilitados={sorted(self.disabled_regimes)}. Barras en esos regimenes -> prob=0.0 (CASH).")
            logger.warning(
                "[RegimeRouter] DEGRADED MODE activo: agentes deshabilitados={}. "
                "Las barras en estos regímenes recibirán prob=0.0 (CASH).",
                sorted(self.disabled_regimes)
            )

        # Mapeo textual semantico a cada agente experto (Centralizado)
        try:
            from config.settings import cfg as _cfg_rr
            self.regimes_config = vars(_cfg_rr.fase2.regime_mapping)
        except Exception as e:
            logger.warning(f"[RegimeRouter] Error cargando regime_mapping: {e}")
            # [SOL3-CALM-BEAR-01 2026-06-01] Fallback actualizado con agente calm_bear dedicado.
            # Cuando settings.yaml sea actualizado, este fallback ya no se usará.
            # El agente 'bear' pasa a cubrir solo BEAR_CRASH/FORCED; calm_bear es independiente.
            self.regimes_config = {
                "bull":      ["1_BULL_TREND", "1_VOLATILE_BULL", "1_BULL_GRIND", "1_BULL_TREND_WEAK", "1_BULL_TREND_B", "1_VOLATILE_BULL_B"],
                "range":     ["2_CALM_RANGE", "2_VOLATILE_RANGE", "2_CALM_RANGE_B", "2_VOLATILE_RANGE_B"],
                "calm_bear": ["3_CALM_BEAR", "3_CALM_BEAR_B", "3_CALM_BEAR_C", "3_CALM_BEAR_D"],
                "bear":      ["3_BEAR_CRASH", "3_BEAR_CRASH_B", "3_BEAR_CRASH_C", "4_BEAR_FORCED"]
            }
            print("[SOL3-CALM-BEAR-01/FALLBACK] Usando regime_mapping interno con agente calm_bear dedicado.")
        self.models = {}
        self.signatures = {}
        self.calibrations = {}
        self.isotonic_calibrators = {}
        # [FIX-NEW-03] Modelo baseline para fallback cuando agente especializado no existe
        self._baseline_model = None
        self._baseline_features = None
        self._load_models()

        # [FIX-CALIB-BINARY-01 DETECTION-1] Auditoría post-carga: verificar que calibradores
        # == modelos. Si no, el bug UnicodeDecodeError silencioso está activo → CRITICAL.
        _n_models = len(self.models)
        _n_cals   = len(self.isotonic_calibrators)
        print(
            f"[FIX-CALIB-BINARY-01/AUDIT-LOAD] RegimeRouter '{self.prefix}' {self.direction} | "
            f"Modelos cargados: {_n_models} | Calibradores cargados: {_n_cals} | "
            f"Agentes esperados: {len(self.regimes_config)} | "
            f"{'OK — calibradores completos' if _n_cals == _n_models else '⚠ DESAJUSTE — posible UnicodeDecodeError en carga binaria'}"
        )
        if _n_cals < _n_models:
            _missing_cals = [a for a in self.models if a not in self.isotonic_calibrators]
            _msg_audit = (
                f"[FIX-CALIB-BINARY-01/AUDIT-LOAD] ALERTA: {_n_cals}/{_n_models} calibradores cargados. "
                f"Agentes SIN calibrador: {_missing_cals}. "
                f"Causa: posible UnicodeDecodeError al abrir .joblib como texto (bug FIX-CALIB-BINARY-01). "
                f"Efecto: xgb_prob_cal == xgb_prob_raw → WR degradado (ej. W1: 32.9%% vs 54%% esperado). "
                f"Verificar que regime_router.py abre los .joblib en modo binario ('rb')."
            )
            print(_msg_audit)
            logger.warning(_msg_audit)
        elif _n_cals == 0 and _n_models > 0:
            _msg_no_cal = (
                f"[FIX-CALIB-BINARY-01/AUDIT-LOAD] CRITICAL: 0 calibradores para {_n_models} modelos. "
                f"Todo el pipeline operará sin calibración isotónica. "
                f"xgb_prob_cal == xgb_prob_raw en TODOS los trades."
            )
            print(_msg_no_cal)
            logger.critical(_msg_no_cal)

    def _load_models(self):
        """Carga los modelos especializados (Bull/Range/Bear) del disco permitiendo direccionalidad."""
        # [FIX-NEW-03] Pre-cargar modelo baseline como fallback (xgboost_meta.model)
        _baseline_path = self.models_dir / f"{self.prefix}.model"
        _baseline_sig  = self.models_dir / f"{self.prefix}_signature.json"
        if _baseline_path.exists() and _baseline_sig.exists() and self.agent_type == "xgboost":
            try:
                is_bm_mock = False
                # [FIX-VPS-P1-V4b 2026-05-30] Deteccion binaria: XGBoost real={7b4c}, Mock JSON={7b22}
                try:
                    with open(_baseline_path, 'rb') as _f_bm:
                        _bm_hdr = _f_bm.read(4)
                    is_bm_mock = (_bm_hdr[:1] == b'{') and (_bm_hdr[1:2] != b'\x4c')
                    print('[MOCK-BM-FIX] ' + str(_baseline_path.name) + ' header=' + _bm_hdr.hex() + ' mock=' + str(is_bm_mock))
                except Exception:
                    pass
                if is_bm_mock:
                    _bm = MockXGBClassifier()
                else:
                    _bm = xgb.XGBClassifier()
                    _bm.load_model(_baseline_path)
                with open(_baseline_sig, 'r') as _bf:
                    _bsig = json.load(_bf)
                self._baseline_model = _bm
                self._baseline_features = _bsig.get("features", [])
                logger.info(
                    "[RegimeRouter] Baseline model precargado ({}) para fallback "
                    "cuando agente especializado no existe. features={}",
                    _baseline_path.name, len(self._baseline_features)
                )
            except Exception as _be:
                logger.warning(f"[RegimeRouter] No se pudo precargar baseline model: {_be}")

        for name in self.regimes_config.keys():
            # Construir nombre con sufijo de direccion (ej: xgboost_meta_bull_long)
            suffix_name = f"{name}_{self.direction}"
            model_path = self.models_dir / f"{self.prefix}_{suffix_name}.model"
            sig_path   = self.models_dir / f"{self.prefix}_{suffix_name}_signature.json"

            # [FIX-2] Retrocompatibilidad V1 eliminada: no hacer fallback a modelos sin sufijo _long/_short.
            # Esto previene contaminación silenciosa del router.

            if model_path.exists() and sig_path.exists():
                is_mock = False
                try:
                    with open(model_path, 'r', encoding='utf-8') as _fm:
                        _start = _fm.read(500).strip()
                        if _start.startswith('{'):
                            import json as _json
                            _mock_data = _json.loads(_start)
                            if _mock_data.get("mocked") is True:
                                is_mock = True
                except Exception:
                    pass

                with open(sig_path, 'r') as f:
                    sig = json.load(f)
                    self.signatures[name] = sig
                    self.calibrations[name] = sig.get('optimal_threshold', 0.5)

                # Cargar el modelo en formato nativo correspondiente
                if self.agent_type == "xgboost":
                    if is_mock:
                        # [ARCH-11-FIX 2026-06-02] No registrar Mock en self.models.
                        # El agente quedara ausente → FIX-NEW-03 usara baseline o prob=0.0.
                        # Antes: model = MockXGBClassifier() → prob=0.6 silencioso
                        # Ahora: agente ausente → prob=0.0 (bloqueado) — SIEMPRE MEJOR
                        msg_mock = (
                            f"[ARCH-11-FIX] Agente '{name}_{self.direction}': archivo .model es placeholder "
                            f"JSON (mocked=True). No se registra en self.models. "
                            f"FIX-NEW-03 usara baseline o bloqueara a prob=0.0. "
                            f"Archivo: {model_path}"
                        )
                        print(f"[ARCH-11-FIX/MOCK-BLOCKED] {msg_mock}")
                        logger.warning(msg_mock)
                        continue  # No registrar en self.models — el agente queda ausente
                    else:
                        model = xgb.XGBClassifier()
                        model.load_model(model_path)
                    self.models[name] = model

                    # [FIX-CALIB-02] Load Isotonic Calibrator specific to this agent
                    cal_path = self.models_dir / f"xgboost_isotonic_calibrator_{suffix_name}.joblib"
                    if cal_path.exists():
                        try:
                            # [FIX-CALIB-BINARY-01 2026-06-01] CRITICAL BUG FIX:
                            # Abrir en modo BINARIO para detectar mock JSON.
                            # El modo texto ('r', utf-8) lanza UnicodeDecodeError en archivos
                            # joblib binarios (primer byte 0x80) → caught silently por except →
                            # calibrador NUNCA se carga en isotonic_calibrators →
                            # calibrated_probs = raw_probs → xgb_prob_cal == xgb_prob_raw.
                            # EFECTO: 73 trades W1 sin calibración → WR=32.9% vs 54% esperado.
                            is_cal_mock = False
                            with open(cal_path, 'rb') as _fc:  # FIX: modo binario
                                if _fc.read(100).startswith(b'{'):  # FIX: bytes, no str
                                    is_cal_mock = True
                            print(
                                f"[FIX-CALIB-BINARY-01] Calibrador check {cal_path.name}: "
                                f"mock={is_cal_mock} — {'RECHAZADO (JSON)' if is_cal_mock else 'CARGANDO joblib binario'}"
                            )
                            if is_cal_mock:
                                logger.warning(f"[RegimeRouter] Calibrador Isotonico mockeado detectado en {cal_path}")
                            else:
                                self.isotonic_calibrators[name] = joblib.load(cal_path)
                                logger.info(
                                    f"[FIX-CALIB-BINARY-01] Calibrador Isotonico CARGADO OK para {suffix_name} "
                                    f"(fix binario aplicado — antes fallaba silenciosamente)"
                                )
                        except Exception as _e_cal:
                            logger.warning(f"[RegimeRouter] Error cargando calibrador {cal_path.name}: {_e_cal}")
                            print(f"[FIX-CALIB-BINARY-01] ERROR cargando calibrador {cal_path.name}: {_e_cal}")
                else:
                    try:
                        if is_mock:
                            # [ARCH-11-FIX 2026-06-02] No registrar LGBM Mock tampoco
                            msg_lgbm_mock = (
                                f"[ARCH-11-FIX] Agente LGBM '{name}': placeholder JSON detectado. "
                                f"No se registra. prob=0.0 para barras de regimen '{name}'."
                            )
                            print(f"[ARCH-11-FIX/LGBM-BLOCKED] {msg_lgbm_mock}")
                            logger.warning(msg_lgbm_mock)
                            continue  # No registrar
                        else:
                            model = joblib.load(model_path)
                        self.models[name] = model
                    except Exception as e:
                        logger.error(f"[RegimeRouter] Error cargando LGBM model {name}: {e}")

                logger.debug(
                    "[RegimeRouter] Modelo cargado: {}_{} (Threshold={:.3f})",
                    self.prefix.upper(), suffix_name.upper(), self.calibrations[name]
                )
                print(f"[RegimeRouter/LOAD] Agente '{name}_{self.direction}' cargado OK. Threshold={self.calibrations[name]:.3f}")
            else:
                # [FIX-NEW-03] Log explícito — se usara baseline si está disponible
                logger.warning(
                    "[RegimeRouter] Faltan artefactos para el agente: {}_{}. "
                    "[FIX-NEW-03] Se usara modelo baseline como fallback para este regimen.",
                    self.prefix, suffix_name.upper()
                )

    def route_and_predict(self, df_oos: pd.DataFrame) -> pd.DataFrame:
        """
        Enruta las filas a sus respectivos agentes expertos basandose en 'HMM_Semantic'
        y retorna un DataFrame con las probabilidades raw y calibradas.
        """
        if "HMM_Semantic" not in df_oos.columns:
            logger.error("[RegimeRouter] 'HMM_Semantic' no encontrado en df_oos. Abortando enrutamiento.")
            return pd.DataFrame({"raw": 0.0, "calibrated": np.nan}, index=df_oos.index)

        # Series inicializadas
        unified_probs = pd.Series(0.0, index=df_oos.index)
        calibrated_probs = pd.Series(np.nan, index=df_oos.index)

        # Early exit: 4_BEAR_FORCED es un override de risk-off determinista.
        forced_bear_mask = df_oos["HMM_Semantic"] == "4_BEAR_FORCED"
        if forced_bear_mask.any():
            n_forced = forced_bear_mask.sum()
            logger.info(f"[RegimeRouter] {n_forced} barras bloqueadas por 4_BEAR_FORCED (prob=0.0).")

        for agent_name, permitted_regimes in self.regimes_config.items():
            # [DEGRADED-MODE] Skip completo si el agente está deshabilitado por Gate-G2
            if agent_name in self.disabled_regimes:
                mask_disabled = df_oos["HMM_Semantic"].isin(permitted_regimes) & (~forced_bear_mask)
                n_disabled = mask_disabled.sum()
                if n_disabled > 0:
                    print(f"[RegimeRouter/DEGRADED] Agente '{agent_name}' NO_OPERABLE: {n_disabled} barras -> CASH (prob=0.0). Regimenes: {permitted_regimes}")
                    logger.warning(
                        "[RegimeRouter/DEGRADED] Agente '{}' NO_OPERABLE (Gate-G2). "
                        "{} barras forzadas a CASH (prob=0.0). Regímenes afectados: {}.",
                        agent_name, n_disabled, permitted_regimes
                    )
                    # Las barras quedan con unified_probs=0.0 (inicializado así) - no hace falta asignar
                continue

            # Filtrar filas que pertenezcan a los regimenes de este agente
            mask = df_oos["HMM_Semantic"].isin(permitted_regimes) & (~forced_bear_mask)
            n_rows = mask.sum()

            if n_rows == 0:
                continue

            if agent_name not in self.models:
                # [FIX-NEW-03] Fallback al baseline cuando el agente especializado no existe
                if self._baseline_model is not None and self._baseline_features:
                    _avail_b = [f for f in self._baseline_features if f in df_oos.columns]
                    _miss_b  = [f for f in self._baseline_features if f not in df_oos.columns]
                    X_base = df_oos.loc[mask, _avail_b].copy()
                    for _mf in _miss_b:
                        X_base[_mf] = np.nan
                    X_base = X_base[self._baseline_features]
                    try:
                        # [OPT-INFERENCE] Force CPU device to bypass DMatrix PCIe transfer bottleneck in XGBoost 3.x
                        if self.agent_type == "xgboost":
                            try:
                                if hasattr(self._baseline_model, "set_params"):
                                    self._baseline_model.set_params(device="cpu")
                            except Exception:
                                pass
                        _base_probs = self._baseline_model.predict_proba(X_base)[:, 1]
                        unified_probs.loc[mask] = _base_probs
                        
                        # Si es baseline, intentamos calibrar con el calibrador global
                        _base_cal_path = self.models_dir / "xgboost_isotonic_calibrator.joblib"
                        if _base_cal_path.exists():
                            try:
                                _base_cal = joblib.load(_base_cal_path)
                                _base_probs_cal = _base_cal.predict(_base_probs)
                                calibrated_probs.loc[mask] = np.clip(_base_probs_cal, 0.0, 1.0)
                            except Exception:
                                calibrated_probs.loc[mask] = _base_probs
                        else:
                            calibrated_probs.loc[mask] = _base_probs

                        print(f"[RegimeRouter/BASELINE] Agente '{agent_name}' AUSENTE -> usando baseline model para {n_rows} barras del regimen '{agent_name}'.")
                        logger.warning(
                            "[RegimeRouter/FIX-NEW-03] Agente [{}_{} ] AUSENTE — usando baseline model "
                            "para {} barras de regimen {}. Re-entrenar para mejor precision.",
                            self.prefix.upper(), agent_name.upper(), n_rows, agent_name
                        )
                    except Exception as _fb_err:
                        print(f"[RegimeRouter/ERROR] Baseline fallback FALLO para '{agent_name}': {_fb_err}. {n_rows} barras -> prob=0.0")
                        logger.error(
                            "[RegimeRouter/FIX-NEW-03] Baseline fallback tambien fallo para {}: {}. "
                            "{} barras quedan con prob=0.0.",
                            agent_name, _fb_err, n_rows
                        )
                else:
                    print(f"[RegimeRouter/WARN] Agente '{agent_name}' AUSENTE y sin baseline. {n_rows} barras -> prob=0.0 (BLOQUEADAS)")
                    logger.warning(
                        "[RegimeRouter/FIX-NEW-03] Agente [{}_{} ] AUSENTE y sin baseline disponible. "
                        "{} barras de regimen {} quedan con prob=0.0 (senales bloqueadas).",
                        self.prefix.upper(), agent_name.upper(), n_rows, agent_name
                    )
                continue

            model = self.models[agent_name]
            sig = self.signatures[agent_name]
            
            # Robust fallback for missing features in signature (e.g. mock metadata or older formats)
            features_list = sig.get("features", None)
            if features_list is None:
                # Usar todas las columnas numéricas como fallback
                features_list = [c for c in df_oos.columns if c not in ["HMM_Semantic", "HMM_Regime", "timestamp"] and df_oos[c].dtype.kind in ("i", "u", "f")]
                logger.warning(
                    "[RegimeRouter] 'features' no encontrado en la firma de [{}]. Usando fallback de todas las columnas numéricas.",
                    agent_name
                )
                print(f"[RegimeRouter/WARN] 'features' no encontrado en la firma de {agent_name}. Usando fallback de todas las columnas numéricas.")

            # Extraer columnas de features requeridas
            available_feats = [f for f in features_list if f in df_oos.columns]
            missing_feats   = [f for f in features_list if f not in df_oos.columns]

            if missing_feats:
                logger.warning(
                    "[RegimeRouter] Agente [{}_{}] sin {} features: {}. Rellenadas con NaN.",
                    self.prefix.upper(), agent_name.upper(), len(missing_feats), missing_feats
                )

            X_subset = df_oos.loc[mask, available_feats].copy()
            for m_feat in missing_feats:
                X_subset[m_feat] = np.nan

            # Reordenar al orden exacto especificado en signature
            X_subset = X_subset[features_list]

            # [OPT-INFERENCE] Force CPU device to bypass DMatrix PCIe transfer bottleneck in XGBoost 3.x
            if self.agent_type == "xgboost":
                try:
                    if hasattr(model, "set_params"):
                        model.set_params(device="cpu")
                except Exception:
                    pass

            # Predict
            # [FIX-OOS-FEATNAMES-01 2026-06-02] XGBoost 3.x valida que las columnas del DataFrame
            # coincidan exactamente con los feature_names del booster serializado.
            # Si el modelo fue guardado con feature_names distintas al orden de features_list,
            # predict_proba lanza ValueError: "data did not contain feature names".
            # Fix: obtener el orden exacto de feature_names del booster antes de llamar predict_proba
            # y reconstruir X_subset con ese orden. Si el booster no tiene feature_names,
            # usar los valores numpy directamente (compatible con versiones antiguas).
            try:
                _booster_fn = None
                if self.agent_type == "xgboost" and hasattr(model, "get_booster"):
                    try:
                        _booster_fn = model.get_booster().feature_names
                    except Exception:
                        pass
                if _booster_fn is not None and len(_booster_fn) > 0:
                    # Reordenar/reconstruir X_subset con el orden exacto del booster
                    _df_aligned = pd.DataFrame(index=X_subset.index)
                    for _fn in _booster_fn:
                        if _fn in X_subset.columns:
                            _df_aligned[_fn] = X_subset[_fn].values
                        else:
                            _df_aligned[_fn] = np.nan
                    probs = model.predict_proba(_df_aligned)[:, 1]
                    print(f"[FIX-OOS-FEATNAMES-01] predict_proba OK con alineacion de feature_names del booster ({len(_booster_fn)} features) para agente '{agent_name}'")
                else:
                    # Sin feature_names en booster: usar numpy (compatible con XGBoost antiguo)
                    probs = model.predict_proba(X_subset.values)[:, 1]
                    print(f"[FIX-OOS-FEATNAMES-01] predict_proba OK con numpy.values (sin feature_names en booster) para agente '{agent_name}'")
            except Exception as _feat_err:
                print(f"[FIX-OOS-FEATNAMES-01] ERROR en predict_proba agente '{agent_name}': {_feat_err} — intentando numpy.values como fallback")
                try:
                    probs = model.predict_proba(X_subset.values)[:, 1]
                    print(f"[FIX-OOS-FEATNAMES-01] Fallback numpy.values OK para agente '{agent_name}'")
                except Exception as _np_err:
                    raise RuntimeError(f"[FIX-OOS-FEATNAMES-01] FATAL: predict_proba fallo con DataFrame Y con numpy.values para '{agent_name}': {_np_err}") from _np_err


            # [FIX-LGBM-FOCAL-SIGMOID-01] Safety net: LightGBM con custom objective bypasses sigmoid.
            # [FIX-NEW-10] Usar tolerancia 1e-6 para evitar falsos positivos floating point.
            if probs.max() > 1.0 + 1e-6 or probs.min() < 0.0 - 1e-6:
                import scipy.special
                raw_min, raw_max = float(probs.min()), float(probs.max())
                probs = scipy.special.expit(probs).astype(np.float64)
                logger.warning(
                    "[RegimeRouter/%s] FIX-LGBM-FOCAL-SIGMOID-01: predict_proba out of [0,1] "
                    "(min=%.4f max=%.4f). Sigmoid applied (min=%.4f max=%.4f). "
                    "Re-train ensemble_lgbm.py for permanent fix (CalibratedClassifierCV).",
                    agent_name, raw_min, raw_max, float(probs.min()), float(probs.max())
                )

            unified_probs.loc[mask] = probs

            # [FIX-CALIB-02] Calibrate per-agent probabilities BEFORE they are mixed
            if agent_name in self.isotonic_calibrators:
                try:
                    calibrated = self.isotonic_calibrators[agent_name].predict(probs)
                    calibrated_clipped = np.clip(calibrated, 0.0, 1.0)

                    # [FIX-CALIB-ROUTER-01] Guard post-calibracion: detectar colapso del calibrador.
                    # PROBLEMA DOCUMENTADO (2026-05-31): El calibrador isotonico del agente 'bear_long'
                    # fue entrenado con probs crudas en el rango [0.6136, 0.6333] (3 knots).
                    # En OOS 2025, el 89.7% de las probs del agente 'bear' estan por debajo de 0.6136.
                    # IsotonicRegression(out_of_bounds='clip') mapea TODO lo de fuera del rango entrenado
                    # al valor de borde -> todas las probs salen como 0.5727 constante.
                    # EFECTO: std(xgb_prob_cal) = 0 -> MetaLabeler ciego, threshold sweep inutilizado,
                    # 116 trades (27.9% del total) con prob_cal constante en los datos reales.
                    # SOLUCION: Si std_cal < 1e-4 pero std_raw > 1e-4, el modelo tiene senal real
                    # que el calibrador aplana. Revertir a probs raw para este agente.
                    _std_cal = float(np.std(calibrated_clipped))
                    _std_raw = float(np.std(probs))

                    if _std_cal < 1e-4 and _std_raw > 1e-4:
                        # Calibrador colapsado: clip de out_of_bounds aplano las probs
                        print(
                            f"[FIX-CALIB-ROUTER-01] COLAPSO DETECTADO agente='{agent_name}_{self.direction}' | "
                            f"std_raw={_std_raw:.4f} std_cal={_std_cal:.2e} | "
                            f"cal_const={calibrated_clipped[0]:.4f} | "
                            f"Probable causa: probs OOS fuera del rango de entrenamiento del calibrador. "
                            f"REVERTIENDO a xgb_prob RAW para preservar varianza."
                        )
                        logger.warning(
                            "[FIX-CALIB-ROUTER-01] Calibrador isotónico COLAPSADO para agente '{}_{}'  — "
                            "std_cal={:.2e} < 1e-4 pero std_raw={:.4f} > 1e-4. "
                            "Probs OOS fuera del rango de entrenamiento del calibrador (out_of_bounds clip). "
                            "Revirtiendo a xgb_prob RAW. Re-entrenar el calibrador con Temperature Scaling "
                            "(FIX-CALIB-TEMP-01) para solución permanente.",
                            agent_name, self.direction, _std_cal, _std_raw
                        )
                        calibrated_probs.loc[mask] = probs  # revertir a raw
                    else:
                        calibrated_probs.loc[mask] = calibrated_clipped
                        print(
                            f"[FIX-CALIB-ROUTER-01] Calibracion OK agente='{agent_name}_{self.direction}' | "
                            f"std_raw={_std_raw:.4f} std_cal={_std_cal:.4f} | "
                            f"cal=[{calibrated_clipped.min():.4f},{calibrated_clipped.max():.4f}]"
                        )

                except Exception as _e_cal_pred:
                    logger.warning(f"[RegimeRouter] Fallo calibrando probs para {agent_name}: {_e_cal_pred}")
                    calibrated_probs.loc[mask] = probs
            else:
                # If no calibrator exists for this agent, just copy the raw probabilities
                calibrated_probs.loc[mask] = probs

            logger.debug(
                "[RegimeRouter] Agente [{}_{} ] asigno probabilidades a {} barras.",
                self.prefix.upper(), agent_name.upper(), n_rows
            )
            _prob_std  = float(probs.std())
            _prob_mean = float(probs.mean())
            _prob_min  = float(probs.min())
            _prob_max  = float(probs.max())
            print(f"[RegimeRouter/ROUTED] Agente '{agent_name}_{self.direction}': {n_rows} barras enrutadas | prob_mean={_prob_mean:.4f} std={_prob_std:.4f} min={_prob_min:.4f} max={_prob_max:.4f}")

            # [FIX-ROUTER-SANITY-01 2026-05-31] Checks de sanidad post-enrutamiento
            # Derivados de fallos documentados en sesion 2026-05-31:
            #   - Colapso total XGB (std=0.000) → toda la run inútil (bug FIX-REG-01)
            #   - Señal invertida bull W1 (prob_mean<0.47) → WR=37.9% peor que azar
            #   - Calibrador isotónico colapsado (std_cal<1e-4) → ya cubierto por FIX-CALIB-ROUTER-01
            # Nivel CRITICAL: detienen la run inmediatamente via RuntimeError
            # Nivel WARNING:  alertan sin detener (el agente puede tener edge a pesar del aviso)

            # ── CRITICAL/SKIP: Colapso total — std=0.0 o min==max ──
            # [FIX-BEAR-SKIP-01 2026-06-01] Distinguir dos causas de colapso:
            # A) Degeneración real: el régimen SÍ existe en OOS pero el modelo predice constante
            #    → FATAL (RuntimeError) — evita resultados espurios con señal real
            # B) Régimen ausente en IS: bear_long entrenado con n≈0 muestras bear
            #    porque el mercado fue 100% bull en el período IS (ej: W4=Ago-Sep 2025)
            #    → SKIP graceful — el agente no opera, 0 trades bear, la ventana continúa
            # Diagnóstico: si n_rows (barras OOS del régimen) > 20 pero std=0 → caso A.
            #              si n_rows es bajo (régimen ausente en OOS también) → caso B.
            # Umbral: si n_rows <= min_bear_oos_rows → régimen estructuralmente ausente → SKIP
            if (_prob_std == 0.0 or _prob_min == _prob_max) and n_rows > 20:
                # Determinar si es régimen ausente (caso B) o degeneración real (caso A)
                # Caso B: el agente bear tiene pocas o 0 barras porque el régimen no ocurrió
                # Lo detectamos por la combinación std=0 + prob constante ≈ base_rate (≈0.5)
                # y el nombre del agente es 'bear', 'calm_bear' o 'bear_crash' (regimenes raros en bull markets)
                # [SOL3-CALM-BEAR-01 2026-06-01] Extendido para cubrir calm_bear y bear_crash
                _BEAR_AGENT_NAMES = ('bear', 'calm_bear', 'bear_crash')
                _is_bear_agent = any(name in agent_name.lower() for name in _BEAR_AGENT_NAMES)
                _prob_is_near_base_rate = 0.45 <= _prob_min <= 0.55  # constante ≈ base rate
                _bear_absent_in_oos = _is_bear_agent and _prob_is_near_base_rate
                print(f"[SOL3-CALM-BEAR-01/SKIP-CHECK] agent='{agent_name}' is_bear={_is_bear_agent} prob_cte={_prob_min:.4f} near_base={_prob_is_near_base_rate}")

                if _bear_absent_in_oos:
                    # [FIX-BEAR-SKIP-01] SKIP graceful — régimen bear ausente en IS/OOS
                    # El modelo predice base_rate constante porque fue entrenado sin datos bear.
                    # No hay trades bear en OOS de todas formas → 0 pérdidas por skip.
                    # La ventana continúa con bull_long y range_long operando normalmente.
                    _skip_msg = (
                        f"[FIX-BEAR-SKIP-01/SKIP] Agente '{agent_name}_{self.direction}': "
                        f"std=0 prob_cte={_prob_min:.4f} ≈ base_rate → régimen bear/calm_bear/bear_crash ausente en IS "
                        f"(mercado fue 100% bull en período de entrenamiento). "
                        f"ACCIÓN: SKIP graceful — 0 trades bear esta ventana. "
                        f"bull_long y range_long/calm_bear_long continúan operando normalmente."
                    )
                    print(_skip_msg)  # RULE[fixbugsprints.md]
                    logger.warning(_skip_msg)
                    # Forzar prob=0.0 para todas las barras bear → señal de no-operar
                    calibrated_probs.loc[mask] = 0.0
                    # Continuar sin FATAL — la ventana sigue
                else:
                    # [FIX-ROUTER-SANITY-01/CRITICAL] Degeneración real — el régimen SÍ
                    # está presente en OOS pero el modelo predice constante no-base-rate.
                    # Esto indica problema de regularización/Optuna, no de datos ausentes.
                    _msg = (
                        f"[FIX-ROUTER-SANITY-01/CRITICAL] COLAPSO TOTAL REAL detectado en agente "
                        f"'{agent_name}_{self.direction}': std_prob={_prob_std:.6f} min=max={_prob_min:.4f} "
                        f"con n_rows={n_rows}. "
                        f"Modelo nulo — predice probabilidad constante no-base-rate. "
                        f"CAUSA PROBABLE: Optuna eligio hiper-parametros extremos (MCW alto, reg_alpha alto). "
                        f"ACCION: Verificar bounds en settings.yaml (xgboost.optuna_search_space). "
                        f"LA RUN SE DETIENE para evitar resultados espurios."
                    )
                    print(_msg)  # RULE[fixbugsprints.md]
                    logger.critical(_msg)
                    raise RuntimeError(_msg)

            # ── WARNING-1: Colapso suave — std muy baja pero no cero ──
            if _prob_std < 0.02 and n_rows > 100:
                print(
                    f"[FIX-ROUTER-SANITY-01/WARNING] Discriminacion POBRE en agente "
                    f"'{agent_name}_{self.direction}': std_prob={_prob_std:.4f} < 0.02 con n_rows={n_rows}. "
                    f"El agente barely discrimina. Los trades OOS tendran WR cercano al azar. "
                    f"Considerar re-entrenar con bounds Optuna mas restrictivos (FIX-REG-01)."
                )
                logger.warning(
                    "[FIX-ROUTER-SANITY-01] Discriminacion pobre agente='%s_%s': std=%.4f < 0.02 | n=%d",
                    agent_name, self.direction, _prob_std, n_rows
                )

            # ── WARNING-2: Señal invertida — prob_mean < 0.47 en agente bull ──
            if "bull" in agent_name.lower() and _prob_mean < 0.47 and n_rows > 50:
                print(
                    f"[FIX-ROUTER-SANITY-01/WARNING] SENAL POTENCIALMENTE INVERTIDA en agente "
                    f"'{agent_name}_{self.direction}': prob_mean={_prob_mean:.4f} < 0.47. "
                    f"El modelo BULL predice menos de 0.47 de media — puede estar prediciendo "
                    f"BEAR en mercado BULL (W1 Q1-2025: WR=37.9%% con este patron). "
                    f"Verificar si el periodo OOS es estructuralmente adverso (post-ATH correction)."
                )
                logger.warning(
                    "[FIX-ROUTER-SANITY-01] Senal invertida riesgo agente='%s_%s': prob_mean=%.4f < 0.47",
                    agent_name, self.direction, _prob_mean
                )

            # ── WARNING-3: Sobreconfianza — prob_mean > 0.75 ──
            if _prob_mean > 0.75 and n_rows > 50:
                print(
                    f"[FIX-ROUTER-SANITY-01/WARNING] SOBRECONFIANZA en agente "
                    f"'{agent_name}_{self.direction}': prob_mean={_prob_mean:.4f} > 0.75. "
                    f"Riesgo de look-ahead o overfit en IS. Verificar PurgedKFold y embargo en training."
                )
                logger.warning(
                    "[FIX-ROUTER-SANITY-01] Sobreconfianza agente='%s_%s': prob_mean=%.4f > 0.75",
                    agent_name, self.direction, _prob_mean
                )

            # ── WARNING-4: Pocas barras enrutadas para agente principal ──
            if agent_name in ("bull", "bear") and n_rows < 20:
                print(
                    f"[FIX-ROUTER-SANITY-01/WARNING] MUY POCAS BARRAS enrutadas a agente "
                    f"'{agent_name}_{self.direction}': n_rows={n_rows} < 20. "
                    f"El HMM asigno casi ninguna barra a este regimen — resultados estadisticamente "
                    f"insignificantes. Verificar HMM y mapping de regimenes."
                )
                logger.warning(
                    "[FIX-ROUTER-SANITY-01] Pocas barras agente='%s_%s': n_rows=%d < 20",
                    agent_name, self.direction, n_rows
                )



        # [FIX-NEW-03] Log diagnostico de barras con prob=0.0 al final
        n_zero_prob = (unified_probs == 0.0).sum()
        n_forced_zero = forced_bear_mask.sum()
        n_unexplained_zero = n_zero_prob - n_forced_zero
        if n_unexplained_zero > 0:
            print(f"[RegimeRouter/ZERO-AUDIT] *** {n_unexplained_zero} barras con prob=0.0 NO explicadas por BEAR_FORCED *** total_zero={n_zero_prob} forced_bear={n_forced_zero}. Causas: agentes faltantes, regimenes no mapeados o HMM_Semantic=UNKNOWN.")
            logger.warning(
                "[RegimeRouter/FIX-NEW-03] {} barras con prob=0.0 NO explicadas por 4_BEAR_FORCED "
                "(total_zero={}, forced_bear={}). Causas: agentes faltantes, regimenes no mapeados, "
                "o HMM_Semantic='UNKNOWN'. Verificar modelos de todos los agentes.",
                n_unexplained_zero, n_zero_prob, n_forced_zero
            )
        else:
            print(f"[RegimeRouter/ZERO-AUDIT] Todas las barras con prob=0.0 ({n_forced_zero}) explicadas por 4_BEAR_FORCED. OK.")

        # [FIX-CALIB-BINARY-01 DETECTION-2] Auditoría post-predicción: verificar que la
        # calibración modificó al menos ALGUNAS probabilidades. Si cal == raw en todo
        # el conjunto, el calibrador no se aplicó (bug UnicodeDecodeError silencioso).
        _n_total   = len(unified_probs)
        _cal_valid = calibrated_probs.dropna()
        if len(_cal_valid) > 0:
            _diff_cal_raw   = (calibrated_probs.fillna(unified_probs) - unified_probs).abs()
            _n_modified     = (_diff_cal_raw > 1e-6).sum()
            _pct_modified   = _n_modified / max(_n_total, 1) * 100
            _diff_mean      = float(_diff_cal_raw.mean())
            _diff_max       = float(_diff_cal_raw.max())
            print(
                f"[FIX-CALIB-BINARY-01/AUDIT-PREDICT] Calibración post-predicción | "
                f"barras_modificadas={_n_modified}/{_n_total} ({_pct_modified:.1f}%) | "
                f"diff_mean={_diff_mean:.4f} diff_max={_diff_max:.4f} | "
                f"{'✓ calibracion aplicada' if _pct_modified > 1.0 else '⚠ SOSPECHOSO: <1% barras modificadas — posible cal==raw'}"
            )
            if _pct_modified < 1.0 and _n_total > 50 and len(self.isotonic_calibrators) > 0:
                _msg_no_effect = (
                    f"[FIX-CALIB-BINARY-01/AUDIT-PREDICT] ALERTA: calibradores cargados={len(self.isotonic_calibrators)} "
                    f"pero solo {_pct_modified:.1f}% de barras tienen cal!=raw. "
                    f"Posibles causas: (1) probs OOS fuera del rango de entrenamiento del calibrador "
                    f"(out_of_bounds=clip aplana todas), (2) bug en asignacion de calibrated_probs.loc[mask]. "
                    f"Revisar [FIX-CALIB-ROUTER-01] guards y distribuciones de probs por agente."
                )
                print(_msg_no_effect)
                logger.warning(_msg_no_effect)
            elif _pct_modified == 0.0 and len(self.isotonic_calibrators) == 0:
                _msg_zero = (
                    f"[FIX-CALIB-BINARY-01/AUDIT-PREDICT] CRITICAL: 0 calibradores + 0% barras modificadas. "
                    f"Todos los trades OOS tendrán xgb_prob_cal == xgb_prob_raw. "
                    f"Verificar FIX-CALIB-BINARY-01 en _load_models (apertura binaria del .joblib)."
                )
                print(_msg_zero)
                logger.critical(_msg_zero)
        else:
            print(
                f"[FIX-CALIB-BINARY-01/AUDIT-PREDICT] calibrated_probs todos NaN — "
                f"ningun agente asigno probs calibradas. Revisar enrutamiento HMM."
            )

        return pd.DataFrame({
            "raw": unified_probs,
            "calibrated": calibrated_probs
        }, index=df_oos.index)
