import json
import logging
import time
from pathlib import Path
import sys
import numpy as np

# Reconfigure stdout for UTF-8 encoding on Windows to prevent charmap crashes
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import xgboost as xgb
import joblib
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# [BUGFIX-UNPICKLE-02] Inyectar adapters en __main__ para evitar AttributeErrors de Pickle/Joblib
try:
    from luna.models.calibrate_probabilities import _RFWithAdapter, _IdentityWrapper, _TSAdapter
    sys.modules['__main__']._RFWithAdapter = _RFWithAdapter
    sys.modules['__main__']._IdentityWrapper = _IdentityWrapper
    sys.modules['__main__']._TSAdapter = _TSAdapter
except Exception:
    pass

from luna.models.hmm_regime import HMMRegimeModel
from luna.models.train_metalabeler_v2 import MetaLabelerV2
from luna.models.regime_router import RegimeRouter

class EnsembleRegimeRouter(RegimeRouter):
    """
    [LUNA-V2-ENSEMBLE] Subclass of RegimeRouter that supports both production-style named models
    (e.g., 'xgboost_meta_bull_long.model') and simplified dry-run/mock models
    (e.g., 'xgboost_meta_1_BULL_TREND.model' or 'xgboost_meta_2_CALM_RANGE.model').
    """
    def _load_models(self):
        # 1. Intento de carga estandar con super()
        super()._load_models()
        
        # 2. Si algun modelo de agente mapeado sigue ausente, buscamos nombres simplificados (dry-run/mock)
        for agent_name, permitted_regimes in self.regimes_config.items():
            if agent_name in self.models:
                continue
                
            logger.info(f"[EnsembleRegimeRouter] Buscando modelos simplificados para el agente '{agent_name}' ({self.direction})...")
            print(f"[EnsembleRegimeRouter/SEARCH] Buscando modelos simplificados para el agente '{agent_name}' ({self.direction})...")
            
            # Buscar si alguno de los regimenes permitidos tiene un archivo de modelo en disco
            for regime in permitted_regimes:
                # Caso A: xgboost_meta_{regime}_{direction}.model
                model_name = f"{self.prefix}_{regime}_{self.direction}.model"
                sig_name = f"{self.prefix}_{regime}_{self.direction}_signature.json"
                model_path = self.models_dir / model_name
                sig_path = self.models_dir / sig_name
                
                # Caso B: xgboost_meta_{regime}.model (sin sufijo de direccion - ej: dry-run)
                if not model_path.exists():
                    model_name = f"{self.prefix}_{regime}.model"
                    sig_name = f"{self.prefix}_{regime}_signature.json"
                    model_path = self.models_dir / model_name
                    sig_path = self.models_dir / sig_name
                    
                if model_path.exists() and sig_path.exists():
                    try:
                        with open(sig_path, 'r') as f:
                            sig = json.load(f)
                            self.signatures[agent_name] = sig
                            self.calibrations[agent_name] = sig.get('optimal_threshold', 0.5)
                            
                        is_mock = False
                        try:
                            # [FIX-VPS-P1-V3 2026-05-30] Deteccion Mock en modo BINARIO.
                            # XGBoost real: header 0x7b 0x4c 0x00 0x00 (msgpack)
                            # Mock JSON: header 0x7b 0x22 o 0x7b 0x20
                            with open(model_path, 'rb') as _fm:
                                _hdr = _fm.read(4)
                            is_mock = (_hdr[:1] == b'{') and (_hdr[1:2] != b'\x4c')
                            print('[MOCK-DETECT-FIX] ' + str(model_path.name) + ': header=' + _hdr[:4].hex() + ' | is_mock=' + str(is_mock))
                        except Exception:
                            pass
                            
                        if is_mock:
                            from luna.models.regime_router import MockXGBClassifier
                            model = MockXGBClassifier()
                            logger.warning(f"[EnsembleRegimeRouter] Cargando XGBoost model mockeado desde {model_path}")
                        else:
                            model = xgb.XGBClassifier()
                            model.load_model(str(model_path))
                        
                        # [OPT-INFERENCE] Forzar CPU para evitar PCIe overhead
                        try:
                            if hasattr(model, "set_params"):
                                model.set_params(device="cpu")
                        except Exception:
                            pass
                            
                        self.models[agent_name] = model
                        
                        # Cargar calibrador isotonico especifico si existe
                        cal_name = f"xgboost_isotonic_calibrator_{regime}_{self.direction}.joblib"
                        cal_path = self.models_dir / cal_name
                        if not cal_path.exists():
                            cal_name = f"xgboost_isotonic_calibrator_{regime}.joblib"
                            cal_path = self.models_dir / cal_name
                        if cal_path.exists():
                            is_cal_mock = False
                            try:
                                with open(cal_path, 'r', encoding='utf-8') as _fc:
                                    if _fc.read(100).strip().startswith('{'):
                                        is_cal_mock = True
                            except Exception:
                                pass
                            if not is_cal_mock:
                                self.isotonic_calibrators[agent_name] = joblib.load(cal_path)
                                logger.info(f"[EnsembleRegimeRouter] Calibrador Isotonico cargado para {regime}")
                                print(f"[EnsembleRegimeRouter/CALIB] Calibrador Isotonico cargado para {regime}")
                            else:
                                logger.warning(f"[EnsembleRegimeRouter] Calibrador Isotonico mockeado detectado y omitido.")
                            
                        logger.info(f"[EnsembleRegimeRouter] Agente simplificado '{agent_name}' mapeado a '{regime}' cargado OK. Path: {model_path.name}")
                        print(f"[EnsembleRegimeRouter/LOAD] Agente simplificado '{agent_name}' mapeado a '{regime}' cargado OK. Path: {model_path.name}")
                        break  # Modelo encontrado para este agente
                    except Exception as e:
                        logger.error(f"[EnsembleRegimeRouter] Error cargando modelo simplificado {model_path.name}: {e}")
                        print(f"[EnsembleRegimeRouter/ERROR] Error cargando modelo simplificado {model_path.name}: {e}")


class LunaEnsembleLiveInference:
    """
    [LUNA-V2-LIVE] Motor unificado de inferencia de ensamble multi-semilla en vivo.
    Responsable de orquestar las 5 semillas campeonas, aplicar Soft Voting, quórum
    de consenso y Consensus-Soft Embargo de 24h.
    """
    def __init__(self, models_prod_dir: Path | str = None):
        self.root = Path(__file__).resolve().parent.parent.parent
        
        # Localizar el directorio prod/ de los modelos
        if models_prod_dir is None:
            self.models_prod_dir = self.root / "data" / "models" / "prod"
        else:
            self.models_prod_dir = Path(models_prod_dir)
            
        self.manifest_path = self.models_prod_dir / "ensemble_metadata.json"
        
        # Valores de settings.yaml por defecto
        try:
            from config.settings import cfg
            self.settings = cfg
            self.active_seeds = list(cfg.wfb.active_seeds)
            self.consensus_threshold = int(cfg.wfb.ensemble_consensus_threshold)
            self.soft_embargo_hours = float(cfg.wfb.soft_embargo_hours)
            self.soft_embargo_enabled = bool(cfg.wfb.soft_embargo_enabled)
            # Leer direction_mode dinámicamente de fase2 (RULE[settingsyfallvack.md] - No-Fallback Silencioso)
            self.direction_mode = getattr(cfg.fase2, "direction_mode", "long").lower()
            
            logger.info(f"[EnsembleLive] Configuración cargada desde settings.yaml: seeds={self.active_seeds} | quorum={self.consensus_threshold} | direction_mode={self.direction_mode}")
            print(f"[EnsembleLive/BOOT] Configuración cargada desde settings.yaml: seeds={self.active_seeds} | quorum={self.consensus_threshold} | direction_mode={self.direction_mode}")
        except Exception as e:
            # [RULE-SETTINGS] PROHIBIDO FALLBACK SILENCIOSO EN PARÁMETROS CRÍTICOS
            err_msg = f"[EnsembleLive] CRITICAL ERROR: Falló la carga de settings.yaml o parámetros del ensamble: {e}"
            logger.critical(err_msg)
            print(f"[EnsembleLive/CRITICAL] ERROR: {err_msg}")
            raise RuntimeError(err_msg)
            
        # Sobreescribir con ensemble_metadata.json del entrenamiento real si existe
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    manifest = json.load(f)
                self.active_seeds = manifest.get("active_seeds", self.active_seeds)
                self.consensus_threshold = manifest.get("ensemble_consensus_threshold", self.consensus_threshold)
                self.soft_embargo_hours = manifest.get("soft_embargo_hours", self.soft_embargo_hours)
                self.soft_embargo_enabled = manifest.get("soft_embargo_enabled", self.soft_embargo_enabled)
                logger.info(f"[EnsembleLive] Manifiesto ensemble_metadata.json cargado con éxito. active_seeds={self.active_seeds} | consensus={self.consensus_threshold}")
                print(f"[EnsembleLive/MANIFEST] Manifiesto cargado: active_seeds={self.active_seeds} | consensus={self.consensus_threshold} | soft_embargo={self.soft_embargo_enabled}")
            except Exception as e:
                # [RULE-SETTINGS] No-Fallback en manifiesto si no hay alternativas legibles
                err_msg = f"[EnsembleLive] CRITICAL ERROR: Error leyendo el manifiesto ensemble_metadata.json: {e}"
                logger.critical(err_msg)
                print(f"[EnsembleLive/CRITICAL] {err_msg}")
                raise RuntimeError(err_msg)
                
        print(f"[EnsembleLive/BOOT] Semillas activas del ensamble: {self.active_seeds}")
        
        self.seeds_models = {}
        self._load_all_seeds()
        
    def _load_all_seeds(self):
        """Carga en memoria los 14 modelos/configuraciones de cada una de las semillas activas."""
        for seed in self.active_seeds:
            seed_dir = self.models_prod_dir / f"seed{seed}"
            if not seed_dir.exists():
                logger.warning(f"[EnsembleLive] Directorio de semilla {seed_dir} ausente. Saltando.")
                print(f"[EnsembleLive/WARN] Directorio ausente: {seed_dir}")
                continue
                
            logger.info(f"[EnsembleLive] Cargando componentes de la semilla {seed} desde {seed_dir.name} (mode: {self.direction_mode})...")
            print(f"[EnsembleLive/LOAD] Cargando componentes de la semilla {seed} (mode: {self.direction_mode})...")
            
            try:
                # 1. HMM Regime Model
                hmm_model = HMMRegimeModel.load(seed_dir)
                
                # 2. Routers
                router_long = None
                router_short = None
                
                if "long" in self.direction_mode or "both" in self.direction_mode:
                    router_long = EnsembleRegimeRouter(seed_dir, direction="long")
                if "short" in self.direction_mode or "both" in self.direction_mode:
                    router_short = EnsembleRegimeRouter(seed_dir, direction="short")
                
                # 3. MetaLabelers
                meta_long = None
                meta_short = None
                
                if "long" in self.direction_mode or "both" in self.direction_mode:
                    meta_long = MetaLabelerV2.load(seed_dir, direction_mode="long")
                    
                    # [BUGFIX-ML-SEQ] Cargar seq_features del config.json ya que el cargador legacy los omite
                    long_cfg_path = seed_dir / "metalabeler_v2_long_config.json"
                    if long_cfg_path.exists():
                        try:
                            with open(long_cfg_path, 'r', encoding='utf-8') as f_cfg:
                                long_cfg = json.load(f_cfg)
                                meta_long._seq_features = long_cfg.get("seq_features", [])
                                print(f"  [MetaLabeler] Semilla {seed} LONG seq_features cargadas: {len(meta_long._seq_features)} columnas (RULE[fixbugsprints.md]).")
                        except Exception as e_cfg:
                            print(f"  [MetaLabeler] Error leyendo seq_features LONG: {e_cfg}")
                    
                    # [HMM-ONEHOT-FIX] Auditoria HMM al boot — verificacion de alineacion correcta.
                    # DIAGNOSTICO 2026-05-27: HMM n_components=5 + estado Risk-Off Shield (estado 5
                    # en state_map) = 6 columnas one-hot. Esto coincide exactamente con n_features_in_=73:
                    #   73 (n_features_in_) - 66 (22 seq_features × 3) - 1 (signal_strength) = 6 cols HMM
                    # El state_map del pkl lo confirma: {0,1,2,3,4: estados HMM, 5: '4_BEAR_FORCED' Risk-Off}
                    # No hay mismatch real — el BUGFIX-ML-SHIELD (padding) era incorrecto.
                    # La logica correcta esta en predict_cycle(): hmm_numeric puede ser 0..5 naturalmente.
                    try:
                        if hasattr(meta_long, 'rf') and meta_long.rf is not None and hasattr(meta_long, '_seq_features'):
                            _n_seq = len(meta_long._seq_features)
                            if _n_seq > 0:
                                _n_features_total = meta_long.rf.n_features_in_
                                _expected_hmm_states = _n_features_total - (_n_seq * 3) - 1
                                _hmm_n_components = getattr(hmm_model, 'n_components', None)
                                if _hmm_n_components is None and hasattr(hmm_model, 'model'):
                                    _hmm_n_components = getattr(hmm_model.model, 'n_components', None)
                                # Obtener el numero de estados reales del state_map (incluye Risk-Off Shield)
                                _hmm_dict = getattr(hmm_model, '_hmm_data', None)
                                _state_map = {}
                                if _hmm_dict and isinstance(_hmm_dict, dict):
                                    _state_map = _hmm_dict.get('state_map', {})
                                _max_state_in_map = max(_state_map.keys()) + 1 if _state_map else _hmm_n_components
                                print(
                                    f"  [HMM-ONEHOT-FIX/BOOT] Semilla {seed} LONG: "
                                    f"HMM n_components={_hmm_n_components} | "
                                    f"state_map max_state={_max_state_in_map} (incluye Risk-Off Shield) | "
                                    f"MetaLabeler espera {_expected_hmm_states} cols one-hot | "
                                    f"Alineado: {'OK' if _max_state_in_map == _expected_hmm_states else 'REVISAR'}"
                                )
                                meta_long._hmm_expected_states = _expected_hmm_states
                                meta_long._hmm_live_states = _hmm_n_components
                                meta_long._hmm_max_state = _max_state_in_map
                    except Exception as e_hmm_audit:
                        print(f"  [HMM-ONEHOT-FIX/BOOT] WARN: Error en verificacion HMM boot: {e_hmm_audit}")
                    
                    # Cargar calibrador del metalabeler de forma opcional (Platt scaling)
                    long_cal_path = seed_dir / "metalabeler_v2_long_calibrator.joblib"
                    if long_cal_path.exists():
                        is_long_cal_mock = False
                        try:
                            with open(long_cal_path, 'r', encoding='utf-8') as _flc:
                                if _flc.read(100).strip().startswith('{'):
                                    is_long_cal_mock = True
                        except Exception:
                            pass
                        if not is_long_cal_mock:
                            meta_long.calibrator = joblib.load(long_cal_path)
                            logger.info(f"  [Calibrador] Semilla {seed} LONG calibrator cargado.")
                            print(f"  [Calibrador/LOAD] Semilla {seed} LONG calibrator cargado.")
                        else:
                            logger.warning(f"  [Calibrador] Semilla {seed} LONG calibrator mockeado detectado y omitido.")

                    
                if "short" in self.direction_mode or "both" in self.direction_mode:
                    meta_short = MetaLabelerV2.load(seed_dir, direction_mode="short")
                    
                    # [BUGFIX-ML-SEQ] Cargar seq_features del config.json ya que el cargador legacy los omite
                    short_cfg_path = seed_dir / "metalabeler_v2_short_config.json"
                    if short_cfg_path.exists():
                        try:
                            with open(short_cfg_path, 'r', encoding='utf-8') as f_cfg:
                                short_cfg = json.load(f_cfg)
                                meta_short._seq_features = short_cfg.get("seq_features", [])
                                print(f"  [MetaLabeler] Semilla {seed} SHORT seq_features cargadas: {len(meta_short._seq_features)} columnas (RULE[fixbugsprints.md]).")
                        except Exception as e_cfg:
                            print(f"  [MetaLabeler] Error leyendo seq_features SHORT: {e_cfg}")
                    
                    short_cal_path = seed_dir / "metalabeler_v2_short_calibrator.joblib"
                    if short_cal_path.exists():
                        is_short_cal_mock = False
                        try:
                            with open(short_cal_path, 'r', encoding='utf-8') as _fsc:
                                if _fsc.read(100).strip().startswith('{'):
                                    is_short_cal_mock = True
                        except Exception:
                            pass
                        if not is_short_cal_mock:
                            meta_short.calibrator = joblib.load(short_cal_path)
                            logger.info(f"  [Calibrador] Semilla {seed} SHORT calibrator cargado.")
                            print(f"  [Calibrador/LOAD] Semilla {seed} SHORT calibrator cargado.")
                        else:
                            logger.warning(f"  [Calibrador] Semilla {seed} SHORT calibrator mockeado detectado y omitido.")
                    
                # [BUGFIX-MOCK-GUARD-01] SECURITY GUARD: Prohibido usar modelos Mock en producción/demo en vivo
                from luna.models.regime_router import MockXGBClassifier
                for r_name, router in [("LONG", router_long), ("SHORT", router_short)]:
                    if router is None:
                        continue
                    # 1. Verificar modelos de régimen en el router
                    for model_regime, model_obj in router.models.items():
                        if isinstance(model_obj, MockXGBClassifier):
                            err_msg = (
                                f"[EnsembleLive/SECURITY-GUARD] CRITICAL ERROR: Se detectó modelo Mock "
                                f"({model_regime}) en router_{r_name.lower()} para la semilla {seed}. "
                                f"Bajo las reglas SOP V10.0 y la política de No-Fallback Silencioso, "
                                f"el trading en vivo/demo tiene prohibido el uso de mocks en producción."
                            )
                            logger.critical(err_msg)
                            print(f"\n[BUG-MOCK-PANIC] {err_msg}\n")
                            raise RuntimeError(err_msg)
                    # 2. Verificar modelo baseline en el router
                    if hasattr(router, "_baseline_model") and isinstance(router._baseline_model, MockXGBClassifier):
                        err_msg = (
                            f"[EnsembleLive/SECURITY-GUARD] CRITICAL ERROR: Se detectó Baseline model Mock "
                            f"en router_{r_name.lower()} para la semilla {seed}. "
                            f"Bajo las reglas SOP V10.0 y la política de No-Fallback Silencioso, "
                            f"el trading en vivo/demo tiene prohibido el uso de mocks en producción."
                        )
                        logger.critical(err_msg)
                        print(f"\n[BUG-MOCK-PANIC] {err_msg}\n")
                        raise RuntimeError(err_msg)

                self.seeds_models[seed] = {
                    "hmm": hmm_model,
                    "router_long": router_long,
                    "router_short": router_short,
                    "meta_long": meta_long,
                    "meta_short": meta_short,
                    "meta_long_config": meta_long.extractor.save_state() if meta_long else None,
                    "meta_short_config": meta_short.extractor.save_state() if meta_short else None
                }
                logger.success(f"[EnsembleLive] Semilla {seed} cargada con éxito. Componentes OK.")
                print(f"[EnsembleLive/SUCCESS] Semilla {seed} cargada con éxito.")
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"[EnsembleLive] Error fatal al cargar la semilla {seed}: {e}")
                print(f"[EnsembleLive/ERROR] Error fatal cargando la semilla {seed}: {e}")
                if "SECURITY-GUARD" in str(e):
                    raise e
                
        if not self.seeds_models:
            raise RuntimeError("[EnsembleLive] Ninguna semilla activa pudo ser cargada en el ensamble!")

        # [OPT-HMM-SHARED-01] Detectar si todos los HMMs son idénticos para precálculo compartido
        self._detect_shared_hmm()
            
    def _detect_shared_hmm(self):
        """
        [OPT-HMM-SHARED-01] Detecta si todos los HMMs del ensamble comparten el mismo state_map.
        Si son idénticos, predict_cycle reutilizará un único resultado de predict_regime_series
        en lugar de ejecutarlo 12 veces. Fallback automático a per-seed si divergen.
        """
        if len(self.seeds_models) < 2:
            self._hmm_shared = False
            print("[OPT-HMM-SHARED-01] Solo 1 seed activa — sin optimización de HMM compartido necesaria.")
            return

        state_map_signatures = []
        for seed, models in self.seeds_models.items():
            hmm = models["hmm"]
            hmm_data = getattr(hmm, '_hmm_data', None)
            sm = hmm_data.get('state_map', {}) if isinstance(hmm_data, dict) else {}
            state_map_signatures.append(str(sorted(sm.items())))

        unique_signatures = set(state_map_signatures)
        self._hmm_shared = len(unique_signatures) == 1

        if self._hmm_shared:
            print(
                f"[OPT-HMM-SHARED-01] ✅ HMM COMPARTIDO detectado: los {len(self.seeds_models)} seeds "
                f"tienen state_map idéntico. predict_regime_series se ejecutará UNA sola vez por ciclo. "
                f"Reducción estimada: ~{len(self.seeds_models) - 1}x ahorro de tiempo HMM."
            )
            logger.info(
                f"[OPT-HMM-SHARED-01] HMM compartido: {len(self.seeds_models)} seeds con state_map idéntico. "
                f"Activando precálculo HMM único por ciclo."
            )
        else:
            print(
                f"[OPT-HMM-SHARED-01] ⚠️ HMM DIFERENTE por seed: {len(unique_signatures)} state_maps únicos "
                f"entre {len(self.seeds_models)} seeds. Usando inferencia HMM individual por seed (fallback seguro)."
            )
            logger.warning(
                f"[OPT-HMM-SHARED-01] HMMs divergentes — fallback a inferencia per-seed. "
                f"Unique state_maps={len(unique_signatures)}"
            )

    def predict_cycle(self, df: pd.DataFrame) -> dict:
        """
        Ejecuta el ciclo de inferencia unificado sobre el ensamble multisemilla.
        Soporta Soft Voting y evaluación de Consensus-Soft Embargo.
        
        Args:
            df: DataFrame con las features incrementales en tiempo real.
            
        Returns:
            Dict con decisión colectiva, quórum de semillas, embargo y traza de auditoría.
        """
        print(f"\n[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] 🧠 Inferencia de Ensamble de Producción (Luna V2)...")
        logger.info("[EnsembleLive] Iniciando inferencia de ensamble...")
        
        if df is None or df.empty:
            logger.warning("[EnsembleLive] DataFrame de inferencia vacío.")
            return {"action": "HOLD", "confidence": 0.0, "reason": "DataFrame vacio.", "xgb_prob": 0.0, "regime": "UNKNOWN"}
            
        # Registro del precio actual de cierre
        current_price = float(df['close'].iloc[-1]) if 'close' in df.columns else 0.0
        
        # Historial de votos de cada semilla
        votes = []
        seed_details = {}

        # [OPT-HMM-SHARED-01] Precálculo HMM compartido si todos los seeds tienen state_map idéntico
        shared_hmm_result = None
        if getattr(self, '_hmm_shared', False) and self.seeds_models:
            _t_hmm_start = time.time()
            _first_seed = next(iter(self.seeds_models))
            _first_hmm = self.seeds_models[_first_seed]["hmm"]
            try:
                shared_hmm_result = _first_hmm.predict_regime_series(df)
                _t_hmm_elapsed = time.time() - _t_hmm_start
                print(
                    f"[OPT-HMM-SHARED-01] HMM precalculado en {_t_hmm_elapsed:.2f}s | "
                    f"Reutilizado para {len(self.seeds_models)} seeds | "
                    f"Ahorro estimado: {_t_hmm_elapsed * (len(self.seeds_models) - 1):.1f}s"
                )
                logger.info(
                    f"[OPT-HMM-SHARED-01] HMM precalculado: {_t_hmm_elapsed:.2f}s | "
                    f"seeds={len(self.seeds_models)} | último_regime={shared_hmm_result['HMM_Semantic'].iloc[-1]}"
                )
            except Exception as _e_shared:
                print(f"[OPT-HMM-SHARED-01] ERROR en precálculo HMM compartido: {_e_shared} — fallback a per-seed.")
                logger.warning(f"[OPT-HMM-SHARED-01] Fallback a per-seed HMM: {_e_shared}")
                shared_hmm_result = None

        for seed, models in self.seeds_models.items():
            try:
                # 1. HMM Regime Projection
                # [OPT-HMM-SHARED-01] Reutilizar resultado compartido si está disponible, sino calcular per-seed
                hmm_model = models["hmm"]
                if shared_hmm_result is not None:
                    hmm_res = shared_hmm_result
                else:
                    hmm_res = hmm_model.predict_regime_series(df)
                hmm_semantic = hmm_res["HMM_Semantic"].iloc[-1]
                hmm_numeric = int(hmm_res["HMM_Regime"].iloc[-1])
                
                # Copia local enriquecida con HMM
                df_seed = df.copy()
                df_seed["HMM_Semantic"] = hmm_res["HMM_Semantic"]
                df_seed["HMM_Regime"] = hmm_res["HMM_Regime"]
                
                # 2. XGBoost Specialist Routers (LONG & SHORT)
                xgb_prob_long = 0.0
                xgb_prob_short = 0.0
                
                if models["router_long"] is not None:
                    preds_long = models["router_long"].route_and_predict(df_seed)
                    xgb_prob_long = float(preds_long["calibrated"].iloc[-1])
                    if np.isnan(xgb_prob_long): xgb_prob_long = 0.0
                    
                if models["router_short"] is not None:
                    preds_short = models["router_short"].route_and_predict(df_seed)
                    xgb_prob_short = float(preds_short["calibrated"].iloc[-1])
                    if np.isnan(xgb_prob_short): xgb_prob_short = 0.0
                
                # Decisión direccional de XGBoost de la semilla
                # Usamos una comparación de máxima probabilidad para determinar la dirección de XGBoost
                if xgb_prob_long >= 0.48 and xgb_prob_long > xgb_prob_short:
                    direction = "LONG"
                    xgb_prob = xgb_prob_long
                    active_meta = models["meta_long"]
                    meta_dir = "long"
                elif xgb_prob_short >= 0.48 and xgb_prob_short > xgb_prob_long:
                    direction = "SHORT"
                    xgb_prob = xgb_prob_short
                    active_meta = models["meta_short"]
                    meta_dir = "short"
                else:
                    direction = "HOLD"
                    xgb_prob = 0.0
                    active_meta = None
                    meta_dir = None
                    
                # 3. MetaLabelerV2 Filter
                meta_prob = 1.0  # Pass-through por defecto
                seed_decision = "HOLD"
                
                if direction != "HOLD" and active_meta is not None:
                    # Preparar secuencias temporales para el extractor rolling
                    seq_len_cfg = 48  # seq_len estandar de settings.yaml
                    if len(df_seed) >= seq_len_cfg:
                        seq_features = getattr(active_meta, "_seq_features", None)
                        if not seq_features:
                            # [BUGFIX-MEMBER-SHIELD] Filtrar estrictamente columnas numéricas para evitar castear HMM_Semantic (RULE[fixbugsprints.md])
                            seq_features = [c for c in df_seed.columns if c not in ["HMM_Semantic", "HMM_Regime", "timestamp"] and df_seed[c].dtype.kind in ("i", "u", "f")]
                            
                        # Limitar a columnas que realmente existan y sean numéricas
                        avail_seq = [f for f in seq_features if f in df_seed.columns and f not in ["HMM_Semantic", "HMM_Regime", "timestamp"] and df_seed[f].dtype.kind in ("i", "u", "f")]
                        
                        # [BUGFIX-OVERFLOW-CEILING] Acotar valores numéricos extremos para prevenir desbordamientos float32 en std()
                        # Si una columna numérica tiene valores extremadamente altos (como quote_volume=1e21 o M2=1e12),
                        # al calcular std() se elevan al cuadrado desbordando el límite float32 (3.4e38) y resultando en 'inf'.
                        # Acotamos a [-1e9, 1e9] para mantener la proporcionalidad y evitar el desbordamiento (RULE[fixbugsprints.md]).
                        # [FIX-P3-OVERFLOW-DIAG] Diagnosticar exactamente que columnas generan el overflow para winsorizar en pipeline
                        df_seed_clean = df_seed[avail_seq].copy()
                        _overflow_mask = (df_seed_clean.abs() > 1e9).any(axis=0)
                        _overflow_cols = _overflow_mask[_overflow_mask].index.tolist()
                        if _overflow_cols:
                            _overflow_info = []
                            for _col in _overflow_cols[:5]:  # Limitar a 5 para no saturar el log
                                _max_abs = df_seed_clean[_col].abs().max()
                                _overflow_info.append(f"{_col}(max={_max_abs:.3e})")
                            print(
                                f"[BUGFIX-OVERFLOW-CEILING] [FIX-P3-OVERFLOW-DIAG] Seed {seed}: "
                                f"{len(_overflow_cols)} features con valores > 1e9: "
                                f"{', '.join(_overflow_info)}"
                                + (" ..." if len(_overflow_cols) > 5 else "")
                                + " — Clip aplicado. Accion requerida: winsorizar estas features en el pipeline de features."
                            )
                        df_seed_clean = df_seed_clean.clip(lower=-1e9, upper=1e9)
                        seq_arr = df_seed_clean.iloc[-seq_len_cfg:].values.astype(np.float32)
                        
                        # Relleno (padding) si faltan columnas
                        if seq_arr.shape[1] < len(seq_features):
                            pad = np.zeros((seq_len_cfg, len(seq_features) - seq_arr.shape[1]), dtype=np.float32)
                            seq_arr = np.concatenate([seq_arr, pad], axis=1)
                            
                        X_seq = seq_arr[np.newaxis, ...]  # (1, seq_len, n_feat)
                        
                        # [HMM-ONEHOT-FIX] Generación correcta del vector one-hot HMM para MetaLabeler.
                        #
                        # CAUSA RAÍZ DEL MISMATCH (diagnosticada 2026-05-27):
                        # Durante el training (train_metalabeler_v2.py L951-952):
                        #   n_states_total = max(HMM_N_STATES, max_state_empirico) + 1
                        # El "+1" agrega el estado Risk-Off Shield como estado extra (estado N).
                        # El state_map del pkl lo confirma: tiene 6 entradas (0-5) con n_components=5.
                        # El estado 5 = '4_BEAR_FORCED' es el Risk-Off Shield activo.
                        #
                        # En live, el HMM puede predecir estado 5 via predict_regime_series()
                        # cuando vol > vol_p90 AND fund < fund_p05 (shield_quantiles del pkl).
                        # Por tanto, el one-hot debe tener n_hmm_states = n_features_in_ - (n_seq*3) - 1
                        # columnas, y el estado actual (hmm_numeric) puede ser 0..5.
                        #
                        # NO hay padding: si hmm_numeric == 5, se activa la columna Risk-Off Shield.
                        n_hmm_states = active_meta.rf.n_features_in_ - (len(avail_seq) * 3) - 1
                        hmm_oh = np.zeros((1, n_hmm_states), dtype=np.float32)
                        # Clamp defensivo: si por alguna razón el estado excede el rango esperado,
                        # asignarlo a la última columna (Risk-Off) en lugar de indexar fuera de rango.
                        hmm_state_clamped = min(hmm_numeric, n_hmm_states - 1)
                        if hmm_state_clamped != hmm_numeric:
                            print(
                                f"[HMM-ONEHOT-FIX] WARN Seed {seed}: estado HMM {hmm_numeric} "
                                f"excede n_hmm_states={n_hmm_states} → usando columna {hmm_state_clamped}"
                            )
                        hmm_oh[0, hmm_state_clamped] = 1.0
                        print(
                            f"[HMM-ONEHOT-FIX] Seed {seed}: estado={hmm_numeric} "
                            f"({hmm_semantic}) | n_hmm_states={n_hmm_states} | "
                            f"one-hot col={hmm_state_clamped} | "
                            f"Risk-Off={'YES' if hmm_state_clamped == n_hmm_states - 1 and n_hmm_states > 5 else 'NO'}"
                        )


                        # Inferencia MetaLabeler V2 con bloque try-except de diagnóstico y recuperación robusta
                        try:
                            meta_prob = float(active_meta.predict_proba(
                                X_seq, np.array([xgb_prob]), hmm_regime=hmm_oh
                            )[0])
                        except Exception as e_predict:
                            print(f"\n[BUGFIX-ML-DEBUG] ERROR en predict_proba para Semilla {seed}: {e_predict}")
                            logger.error(f"[BUGFIX-ML-DEBUG] Error en predict_proba para Semilla {seed}: {e_predict}")
                            if isinstance(X_seq, np.ndarray):
                                print(f"  X_seq shape: {X_seq.shape}")
                                print(f"  X_seq has NaNs: {np.isnan(X_seq).any()}")
                                print(f"  X_seq has Infs: {np.isinf(X_seq).any()}")
                                if X_seq.size > 0:
                                    try:
                                        print(f"  X_seq max: {np.nanmax(X_seq)}")
                                        print(f"  X_seq min: {np.nanmin(X_seq)}")
                                    except Exception:
                                        pass
                                    # Diagnóstico de qué columna contiene el valor problemático
                                    for idx, col_name in enumerate(avail_seq):
                                        if idx < seq_arr.shape[1]:
                                            col_vals = seq_arr[:, idx]
                                            if np.isnan(col_vals).any() or np.isinf(col_vals).any() or (col_vals.size > 0 and np.abs(col_vals).max() > 1e10):
                                                print(f"    -> Columna '{col_name}' en index {idx} tiene anomalías: "
                                                      f"NaN={np.isnan(col_vals).any()}, Inf={np.isinf(col_vals).any()}")
                            print(f"  xgb_prob: {xgb_prob} (type: {type(xgb_prob)})")
                            print(f"  hmm_oh: {hmm_oh}")
                            # Fallback seguro al estimador base de XGBoost calibrado para no colapsar la producción
                            print("[BUGFIX-ML-DEBUG] Forzando fallback seguro al estimador base XGBoost calibrado (xgb_prob).")
                            meta_prob = float(xgb_prob)
                    else:
                        logger.warning(f"[EnsembleLive] Historial insuficiente para MetaLabeler ({len(df_seed)} < {seq_len_cfg}).")
                        meta_prob = 0.0  # HOLD preventivo
                        
                    # Umbral dinámico o fijo de aceptación del MetaLabeler
                    # Leemos de settings.yaml para respetar no-hardcoding
                    try:
                        from config.settings import cfg as _cfg_meta
                        if hmm_semantic.startswith("1_BULL_TREND") and meta_dir == "long":
                            meta_threshold = float(getattr(_cfg_meta.metalabeler, "meta_v2_min_prob_bull_long", 0.55))
                        else:
                            meta_threshold = float(getattr(_cfg_meta.metalabeler, "meta_v2_min_prob", 0.38))
                    except Exception:
                        meta_threshold = 0.38  # Fallback documentado en docs/parametros_fijos.md
                        
                    # Decisión de la semilla
                    if meta_prob >= meta_threshold:
                        seed_decision = direction
                    else:
                        seed_decision = "HOLD"
                        
                # Auditoría individual de la semilla
                votes.append(seed_decision)
                seed_details[seed] = {
                    "regime": hmm_semantic,
                    "xgb_dir": direction,
                    "xgb_prob": xgb_prob,
                    "meta_prob": meta_prob,
                    "decision": seed_decision,
                    "xgb_prob_raw": xgb_prob
                }
                
                print(f"  [Seed {seed}] HMM={hmm_semantic} | XGB={direction} (p={xgb_prob:.4f}) | Meta ML={meta_prob:.4f} -> Voto={seed_decision}")
                logger.info(f"Seed {seed}: HMM={hmm_semantic} | XGB={direction} (prob={xgb_prob:.4f}) | Meta={meta_prob:.4f} -> Voto={seed_decision}")
                
            except Exception as e:
                logger.error(f"[EnsembleLive] Error procesando inferencia en semilla {seed}: {e}")
                print(f"  [Seed {seed}] ERROR: {e}")
                votes.append("HOLD")
                seed_details[seed] = {"decision": "HOLD", "error": str(e)}
                
        # 4. Soft Voting & Quórum Consolidado (Consensus)
        long_votes = votes.count("LONG")
        short_votes = votes.count("SHORT")
        hold_votes = votes.count("HOLD")
        
        # Encontrar la dirección colectiva mayoritaria
        if long_votes >= self.consensus_threshold and long_votes >= short_votes:
            consensus_direction = "LONG"
            consensus_count = long_votes
        elif short_votes >= self.consensus_threshold and short_votes >= long_votes:
            consensus_direction = "SHORT"
            consensus_count = short_votes
        else:
            consensus_direction = "HOLD"
            consensus_count = hold_votes
            
        # Calcular confianza y probabilidad promedio de las semillas alineadas
        aligned_meta_probs = []
        aligned_xgb_probs = []
        aligned_regimes = []
        raw_xgb_probs = [details.get("xgb_prob_raw", 0.0) for details in seed_details.values() if "xgb_prob_raw" in details]
        
        for seed, details in seed_details.items():
            if details.get("decision") == consensus_direction and consensus_direction != "HOLD":
                aligned_meta_probs.append(details["meta_prob"])
                aligned_xgb_probs.append(details["xgb_prob"])
                aligned_regimes.append(details["regime"])
                
        # Promedio consolidado (Soft Voting)
        if aligned_meta_probs:
            consolidated_confidence = float(np.mean(aligned_meta_probs))
            consolidated_xgb_prob = float(np.mean(aligned_xgb_probs))
            majority_regime = max(set(aligned_regimes), key=aligned_regimes.count)
        else:
            consolidated_confidence = 0.0
            consolidated_xgb_prob = float(np.mean(raw_xgb_probs)) if raw_xgb_probs else 0.5
            print(f"[Consensus] No hay semillas alineadas. Fallback XGB Prob = {consolidated_xgb_prob:.4f}")
            # Régimen mayoritario HMM entre todos
            all_regimes = [det["regime"] for det in seed_details.values() if "regime" in det]
            majority_regime = max(set(all_regimes), key=all_regimes.count) if all_regimes else "2_CALM_RANGE"

        # [FIX-XGB-TRAZABILIDAD] Siempre loguear el xgb_prob real vs consolidado para trazabilidad
        raw_xgb_mean = float(np.mean(raw_xgb_probs)) if raw_xgb_probs else 0.0
        print(f"[FIX-XGB-TRAZABILIDAD] Media XGB real ({len(raw_xgb_probs)} seeds): {raw_xgb_mean:.4f} "
              f"| XGB consolidado en DB: {consolidated_xgb_prob:.4f} | Consenso: {consensus_direction}")
        logger.info(f"[FIX-XGB-TRAZABILIDAD] raw_xgb_mean={raw_xgb_mean:.4f} | "
                    f"consolidated_xgb_prob={consolidated_xgb_prob:.4f} | consensus={consensus_direction}")

        # 5. Consensus-Soft Embargo (24 Horas atenuadas si hay consenso adaptativo)
        # [BUGFIX-EMBARGO-CEILING] Consenso de embargo adaptativo según la cantidad de semillas activas para evitar hardcodeos (RULE[settingsyfallvack.md])
        # Alinear con la fórmula de evaluate_ensemble_wfb.py para consistencia estricta backtest/live
        soft_embargo_active = False
        if self.soft_embargo_enabled and consensus_direction != "HOLD":
            n_active = len(self.active_seeds)
            soft_threshold = 4 if n_active >= 5 else 2 if n_active == 3 else max(2, n_active - 1)
            if consensus_count >= soft_threshold:
                soft_embargo_active = True
                logger.info(f"[Consensus-Soft Embargo] ¡QUORUM ALTO DETECTADO ({consensus_count}/{n_active} semillas)! Embargo atenuado a {self.soft_embargo_hours}H activo.")
                print(f"[Consensus-Soft Embargo/ACTIVE] ¡QUORUM ALTO DETECTADO ({consensus_count}/{n_active} semillas)! Embargo atenuado a {self.soft_embargo_hours}H activo.")
                print(f"[BUGFIX-EMBARGO-CEILING] Consensus-Soft Embargo adaptativo disparado: {consensus_count}/{n_active} semillas concurrentes >= soft_threshold {soft_threshold}.")
                
        print(f"[Consensus/RESULT] Quorum: {consensus_direction} (Consenso={consensus_count}/{len(self.active_seeds)}) | "
              f"Confianza Soft={consolidated_confidence:.2%} | Regime={majority_regime}")
        logger.info(f"Consensus Result: {consensus_direction} | count={consensus_count} | confidence={consolidated_confidence:.4f} | regime={majority_regime}")
        
        # Volatilidad realizada del DataFrame (Targeting dinámico)
        pct_changes = df['close'].pct_change()
        current_vol = float(pct_changes.tail(7 * 24).std())
        historical_vol = float(pct_changes.rolling(30 * 24).std().iloc[-1])
        if np.isnan(historical_vol) or historical_vol <= 0:
            historical_vol = current_vol if current_vol > 0 else 1e-6
            
        return {
            "action": consensus_direction,
            "confidence": consolidated_confidence,
            "xgb_prob": consolidated_xgb_prob,
            "raw_xgb_probs_per_seed": raw_xgb_probs,
            "regime": majority_regime,
            "consensus_count": consensus_count,
            "soft_embargo_active": soft_embargo_active,
            "seeds_breakdown": seed_details,
            "price": current_price,
            "current_vol": current_vol,
            "historical_vol": historical_vol
        }


if __name__ == "__main__":
    print("🌙 [TEST] Probando carga de Ensemble Live Inference...")
    try:
        engine = LunaEnsembleLiveInference()
        print("✅ [TEST/SUCCESS] LunaEnsembleLiveInference cargado con éxito y listo para producción.")
    except Exception as e:
        print(f"❌ [TEST/ERROR] Fallo al instanciar LunaEnsembleLiveInference: {e}")
        sys.exit(1)
