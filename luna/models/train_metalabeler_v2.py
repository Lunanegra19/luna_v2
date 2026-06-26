from __future__ import annotations
from loguru import logger
"""
train_metalabeler_v2.py — Luna V1 (P1-9 / MEJORA 2)
======================================================
Inversión de la cascada XGBoost → BiLSTM según MEJORA 2 del research v3.

Nueva arquitectura (Opción A recomendada):
  [Features temporales]
       ↓
  [TemporalFeatureExtractor — LSTM ligero como extractor de embeddings]
       ↓ (vector oculto h_n de 32 dims, NO clasificación directa)
  [Random Forest — árbitro final con bagging anti-overfitting]
       ↓
  [Señal calibrada]

Ventajas sobre BiLSTM profunda:
  1. RF no memoriza ruido — Bagging promedia múltiples árboles decorrelados
  2. LSTM solo extrae embeddings temporales (no predice directamente)
  3. Menos parámetros: LSTM-32 (~6K params) vs BiLSTM-64 (~400K params)
  4. Sin NAS costoso — RF con max_features='sqrt' ya regulariza

SOP Aplicado:
  - R1: CPCV purgeado para XGBoost probs que alimentan RF (P1-8 replicado)
  - R3: Embargo 96H entre train y val split
  - R5: DSR como criterio de aceptación del modelo final
  - R6: Costos 0.25% round-trip en evaluación de retornos

NOTA: Este módulo coexiste con train_metalabeler.py (original BiLSTM).
Activar cambiando la importación en run_features_and_training.py.
"""
import sys
import os as _os_meta
import json
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

import numpy as np
import pandas as pd
from pathlib import Path


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from itertools import combinations

# ── Rutas del proyecto ──────────────────────────────────────────────────────
def _get_root() -> Path:
    """Detecta la raíz del proyecto de forma robusta."""
    try:
        from luna.utils.project_root import get_project_root
        return get_project_root()
    except ImportError:
        return Path(__file__).parent.parent.parent

ROOT = _get_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Constantes SOP ────────────────────────────────────────────
# ARCH-02 (2026-03-10): todas las constantes ahora se leen desde settings.yaml.
# Sin hardcodes: si cambias el valor en settings.yaml, el módulo lo recibe automáticamente.
try:
    from config.settings import cfg as _cfg_meta
    COST_PCT           = float(_cfg_meta.sop.cost_pct)
    EMBARGO_H          = int(_cfg_meta.sop.embargo_hours)
    SEQ_LEN            = int(_cfg_meta.metalabeler.seq_len)
    LSTM_HIDDEN        = int(_cfg_meta.metalabeler.lstm_hidden)
    N_CPCV_GROUPS      = int(_cfg_meta.metalabeler.n_cpcv_groups)
    RF_N_ESTIMATORS    = int(_cfg_meta.metalabeler.rf_n_estimators)
    HMM_N_STATES       = int(_cfg_meta.hmm.n_states)
    # M-40 (2026-03-18): decaimiento exponencial de sample_weights por año (ARCH-02)
    # peso_i = exp(-alpha × años_desde_train_end). 0.0 = uniforme, 0.8 = moderado (ratio ~2.2:1)
    WEIGHT_DECAY_ALPHA = float(_cfg_meta.metalabeler.weight_decay_alpha)
except Exception as _e:
    # ARCH-FAIL-LOUD (2026-03-18): NO silenciar errores de configuración.
    # Si settings.yaml no carga, el pipeline correría con valores incorrectos
    # (p.ej. N_CPCV_GROUPS=10 en vez de 8, COST_PCT=0.0015 podría desync).
    # Principio: Fail Loud > Fail Silent.
    # Para tests unitarios: mockear config.settings, no bypasear silenciosamente.
    raise RuntimeError(
        f"\n[CRITICAL] train_metalabeler_v2.py no pudo cargar settings.yaml.\n"
        f"  Error: {_e}\n"
        f"  Verifica: sintaxis YAML, PYTHONPATH, existencia de config/settings.py"
    ) from _e


# ============================================================================
# COMPONENTE 0: RollingStatsExtractor (determinista, sin parámetros)
# ============================================================================
class RollingStatsExtractor:
    """
    R17-FIX-LSTM-01: Reemplaza el LSTM extractor por estadísticas rolling
    deterministas sobre la ventana temporal.

    Motivación: El LSTM con val_loss≈0.693 (log(2)) no aprende representaciones
    útiles — los embeddings son constantes en OOS 2025 causando que el RF arbiter
    predice siempre la prob marginal (~0.47).

    Diseño: para cada secuencia (seq_len, n_features) calcula 3 estadísticas:
      - mean   : tendencia central de la feature en la ventana
      - std    : volatilidad / dispersión de la feature
      - slope  : tendencia lineal (coeficiente de regresión, normalizado)
    Output: (N, n_features * 3) — completamente determinista, sin overfitting.
    """
    def __init__(self, input_dim: int):
        self.input_dim  = input_dim
        self.output_dim = input_dim * 3  # mean + std + slope por feature
        # Pesos de regresión lineal precalculados (se recalculan en extract si cambia seq_len)
        self._slope_weights: np.ndarray | None = None
        self._last_seq_len: int = -1

    def _get_slope_weights(self, seq_len: int) -> np.ndarray:
        """Coeficientes de regresión lineal normalizados para la ventana temporal."""
        if self._last_seq_len != seq_len:
            t = np.arange(seq_len, dtype=np.float32)
            t = (t - t.mean()) / (t.std() + 1e-8)  # normalizado para escala consistente
            self._slope_weights = t
            self._last_seq_len = seq_len
        return self._slope_weights

    def extract_embeddings(self, X_seq: np.ndarray) -> np.ndarray:
        """
        X_seq: (N, seq_len, n_features) — secuencias temporales
        Returns: (N, n_features * 3) — [mean, std, slope] por feature
        """
        N, seq_len, n_feat = X_seq.shape
        t = self._get_slope_weights(seq_len)  # (seq_len,)

        # Calcular estadísticas por feature de forma vectorizada
        mean_feats  = X_seq.mean(axis=1)                          # (N, n_feat)
        std_feats   = X_seq.std(axis=1) + 1e-8                    # (N, n_feat)

        # Slope lineal: cov(t, X) / var(t) → coef de regresión simple
        X_centered   = X_seq - mean_feats[:, np.newaxis, :]       # (N, seq_len, n_feat)
        t_bc         = t[np.newaxis, :, np.newaxis]                # (1, seq_len, 1)
        slope_feats  = (X_centered * t_bc).mean(axis=1)           # (N, n_feat) — cov(t,x)/E[t**2]

        embeddings = np.concatenate([mean_feats, std_feats, slope_feats], axis=1)  # (N, 3*n_feat)
        return embeddings.astype(np.float32)

    def save_state(self) -> dict:
        return {"type": "RollingStatsExtractor", "input_dim": self.input_dim}

    @classmethod
    def from_state(cls, state: dict) -> "RollingStatsExtractor":
        return cls(input_dim=state["input_dim"])


# ============================================================================
# COMPONENTE 1: TemporalFeatureExtractor (LSTM ligero — legacy, mantenido para compatibilidad)
# ============================================================================
class TemporalFeatureExtractor(nn.Module):
    """
    LSTM ligero que extrae representaciones temporales (embeddings).
    NO predice directamente — solo genera el vector de estado oculto final.

    La simplicidad es intencional: 1 capa, unidireccional (causal en producción).
    hidden_dim=32 da ~6K parámetros vs ~400K del BiLSTM profundo original.
    """

    def __init__(self, input_dim: int, hidden_dim: int = LSTM_HIDDEN, dropout: float = 0.5):
        super().__init__()
        # LEGACY-02 (2026-03-17): reemplazado por RollingStatsExtractor. Mantenido solo
        # para backward-compat de archivos .pt en disco. Ver R17-FIX-LSTM-01.
        import warnings as _wn
        _wn.warn(
            "TemporalFeatureExtractor (LSTM) obsoleto desde R17-FIX-LSTM-01. "
            "Usar RollingStatsExtractor.",
            DeprecationWarning, stacklevel=2,
        )
        self.input_dim = input_dim  # guardado para reconstruccion en load()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # dropout solo con >1 capa
            bidirectional=False,  # causal (SOP R1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch, seq_len, input_dim)
        Returns:
            h_n: Tensor (batch, hidden_dim) — embeddings temporales
        """
        _, (h_n, _) = self.lstm(x)
        return self.dropout(h_n.squeeze(0))  # (batch, hidden_dim)

    def extract_embeddings(self, X_seq: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Extrae embeddings en modo inferencia (sin gradientes).
        Retorna numpy array para uso con scikit-learn (RF).
        """
        self.eval()
        embeddings = []
        tensor = torch.FloatTensor(X_seq)
        with torch.no_grad():
            for i in range(0, len(tensor), batch_size):
                batch = tensor[i:i + batch_size]
                emb = self(batch)
                embeddings.append(emb.numpy())
        return np.vstack(embeddings)


# ============================================================================
# COMPONENTE 2: MetaLabelerV2 (LSTM extractor + RF árbitro)
# ============================================================================
class MetaLabelerV2:
    """
    P1-9: MetaLabeler con arquitectura invertida — LSTM extractor → RF árbitro.

    El RF (Random Forest) es el árbitro final porque:
    - Bagging: promedia múltiples árboles decorrelados → menor varianza
    - max_features='sqrt': decorrelación adicional entre árboles
    - min_samples_leaf=5: evita overfitting en nodos hoja
    - No memoriza ruido residual del LSTM (a diferencia de BiLSTM profunda)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = LSTM_HIDDEN,
                 n_estimators: int = RF_N_ESTIMATORS):
        from sklearn.ensemble import RandomForestClassifier

        # P6-FIX: leer parámetros RF desde settings.yaml (sin hardcodes)
        try:
            from config.settings import cfg as _cfg_rf
            _rf_min_leaf   = int(_cfg_rf.metalabeler.rf_min_samples_leaf) # FASE C: bajado default
            _rf_max_depth  = int(_cfg_rf.metalabeler.rf_max_depth)
            _rf_max_samp   = float(_cfg_rf.metalabeler.rf_max_samples)
            
            # FASE C: Asimetria R:R
            _pt_mult       = float(_cfg_rf.xgboost.pt_mult_min)
        except Exception:
            _rf_min_leaf, _rf_max_depth, _rf_max_samp, _pt_mult = 20, 8, 0.8, 1.6

        # [V2-FIX-3] Topology-Aware Regularization
        # ELIMINADO: _rf_min_leaf = min(_rf_min_leaf, 20)  ← hardcode que invalidaba settings.yaml
        # El min_leaf baseline viene de settings.yaml (ahora default=30).
        # La lógica topology-aware en train() ajustará dinámicamente según n_regime.
        # Guardamos los baselines del YAML para que train() los use como punto de partida.
        self._rf_min_leaf_base  = _rf_min_leaf
        self._rf_max_depth_base = _rf_max_depth

        # FASE C: Ponderacion asimetrica dictada por R:R
        self._base_pt_mult = _pt_mult
        _class_weight = {0: 1.0, 1: _pt_mult}


        # R17-FIX-LSTM-01: usar RollingStatsExtractor (deterministico) en lugar
        # de TemporalFeatureExtractor (LSTM). El LSTM se instancia por compatibilidad
        # con .save()/.load() legacy pero NO se entrena ni se usa en inferencia.
        self.extractor = RollingStatsExtractor(input_dim=input_dim)
        # P2-6-FIX (2026-03-30): instanciar TemporalFeatureExtractor solo cuando se necesita
        # (en save/load para retrocompatibilidad). Evita DeprecationWarning en cada __init__.
        self._lstm_hidden_dim = hidden_dim  # guardado para lazy-init en save()
        self._lstm_extractor  = None        # instanciado lazily en save() si se necesita
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features='sqrt',
            min_samples_leaf=_rf_min_leaf,
            max_depth=_rf_max_depth,
            max_samples=_rf_max_samp,
            class_weight=_class_weight, # FASE C
            # [FIX-RANDOM-STATE-02 2026-05-28] Usar LUNA_SEED para diversidad entre seeds del ensemble
            random_state=int(_os_meta.environ.get('LUNA_SEED', 42)),
            n_jobs=-1,
        )
        self.calibrator = None
        self._trained = False
        self._use_rolling = True  # R17-FIX: flag para save/load

    def train(self,
              X_seq: np.ndarray,
              xgb_probs: np.ndarray,
              y: np.ndarray,
              hmm_regime: np.ndarray | None = None,
              train_extractor: bool = True,
              lstm_epochs: int = 30,
              lstm_lr: float = 1e-3,
              timestamps: pd.DatetimeIndex = None) -> dict:
        """
        Entrena el pipeline: RollingStats extractor → RF árbitro.
        R17-FIX-LSTM-01: el LSTM ya no se entrena. El extractor es RollingStatsExtractor
        (deterministico, sin parámetros aprendidos).
        """
        logger.info(f"MetaLabelerV2.train: {len(y)} muestras, {X_seq.shape[2]} features")
        logger.info("[R17-FIX-LSTM-01] Usando RollingStatsExtractor (mean+std+slope) — sin LSTM")
        if hmm_regime is not None:
            logger.info(f"Contexto HMM activo: {hmm_regime.shape[1]} estados (P4-0-2)")

        # [V2-FIX-3] Topology-Aware RF Regularization
        # Calcular bounds dinámicos basados en la topología real de los datos de entrenamiento.
        # Protege contra sobreajuste en regímenes pequeños sin destruir el modelo en grandes.
        _n_samples     = max(1, len(y))
        _n_minority    = max(1, int((np.array(y) == 1).sum()))
        # max_depth: log2(n_minority) con floor=2, cap=8
        _topo_max_depth = int(np.clip(int(np.log2(max(_n_minority, 4))), 2, 8))
        _topo_min_leaf  = int(np.clip(int(_n_samples * 0.04), 10, max(50, int(_n_samples * 0.03))))
        
        # FIX-AUDIT-02: Erradicar fallback hardcodeado y leer desde config strict
        from config.settings import cfg as _cfg
        _rf_max_depth_base = float(_cfg.metalabeler.rf_max_depth_cap)
        _rf_min_leaf_base = float(_cfg.metalabeler.rf_min_leaf_base)
        
        _effective_max_depth = int(min(_topo_max_depth, _rf_max_depth_base))
        _effective_min_leaf  = int(max(_topo_min_leaf, _rf_min_leaf_base))

        logger.info(
            "[V2-FIX-3] Topology-Aware RF: n_samples={}, n_minority={} → "
            "max_depth={} (base={}), min_samples_leaf={} (base={})",
            _n_samples, _n_minority,
            _effective_max_depth, _rf_max_depth_base,
            _effective_min_leaf,  _rf_min_leaf_base
        )
        print(f"[BUG-FIX-LOG 2026-06-05] [V2-FIX-3] Topology-Aware RF: n_samples={_n_samples}, n_minority={_n_minority} -> max_depth={_effective_max_depth} (base={_rf_max_depth_base}), min_samples_leaf={_effective_min_leaf} (base={_rf_min_leaf_base})")
        # Re-instanciar RF con bounds topológicos
        from sklearn.ensemble import RandomForestClassifier as _RFC_topo
        self.rf.max_depth         = _effective_max_depth
        self.rf.min_samples_leaf  = _effective_min_leaf

        # [V2-FIX-4] Empirical EV (Vol-Adjusted) Class Weights
        # REPARADO: Anclaje estricto de la ponderación al Risk:Reward base.
        # Inflar el peso de los Wins (clase 1) dinámicamente cuando escasean CAUSA Falsos Positivos,
        # bajando el umbral de confianza del RF en el peor momento.
        _n_wins = max(1, int((np.array(y) == 1).sum()))
        _n_losses = max(1, int((np.array(y) == 0).sum()))
        _imbalance_ratio = float(_n_losses) / _n_wins
        
        _base_pt = float(_cfg.xgboost.pt_mult_min)
        
        # Ponderación Asimétrica Constante: dictada por la economía del Setup, NO por el desbalance.
        _empirical_weight = float(_base_pt)
        
        logger.info(
            "[V2-FIX-4] Ponderación Asimétrica Constante = {:.2f} "
            "(imbalance_regimen={:.1f}, wins={}, losses={})",
            _empirical_weight, _imbalance_ratio, _n_wins, _n_losses
        )
        self.rf.class_weight = {0: 1.0, 1: _empirical_weight}
        
        return self.train_rf_arbitro(X_seq, xgb_probs, y, hmm_regime=hmm_regime, timestamps=timestamps)

    def train_rf_arbitro(self, X_seq: np.ndarray, xgb_probs: np.ndarray, y: np.ndarray, hmm_regime: np.ndarray = None, timestamps: pd.DatetimeIndex = None):
        """
        1) Construye Meta-Features.
        2) Entrena RF usando sample_weights (más peso a datos recientes).
        3) Retorna el RF entrenado.
        """
        # [FIX-02] _contested_threshold calculado dinámicamente desde la distribución real del XGBoost
        # Antes: 0.35 hardcodeado (y 0.25 como emergencia) — arbitrarios, sin vínculo a la ventana actual
        # Ahora: percentile_25(xgb_probs[xgb_probs > 0.5]) — zona disputada real del agente en esta ventana
        # Si no hay suficientes positivos fuertes (< 50), usa percentile_25 de todas las probs > 0
        _xgb_positives = xgb_probs[xgb_probs > 0.5]
        if len(_xgb_positives) >= 50:
            _contested_threshold = float(np.percentile(_xgb_positives, 25))
            _threshold_source = f"p25(xgb_probs>0.5, n={len(_xgb_positives)})"
        else:
            # Fallback: usar p25 de todas las probs positivas (>0)
            _xgb_nonzero = xgb_probs[xgb_probs > 0]
            if len(_xgb_nonzero) >= 50:
                _contested_threshold = float(np.percentile(_xgb_nonzero, 25))
                _threshold_source = f"p25(xgb_probs>0, n={len(_xgb_nonzero)}) [fallback: pocos positivos]"
            else:
                _contested_threshold = 0.35  # último recurso: fallback documentado
                _threshold_source = "hardcode_fallback_0.35 [dataset muy pequeño]"

        # Clamp al rango razonable [0.20, 0.55] — evitar extremos que vacíen el dataset
        _contested_threshold = float(np.clip(_contested_threshold, 0.20, 0.55))
        print(f"[FIX-02] _contested_threshold={_contested_threshold:.4f} (fuente: {_threshold_source})")  # debug
        logger.info(
            "[FIX-02] Event-Driven threshold dinamico={:.4f} (antes hardcode 0.35). Fuente: {}",
            _contested_threshold, _threshold_source
        )

        _mask_contested = (xgb_probs >= _contested_threshold)

        # Fallback de seguridad: Si la zona disputada es demasiado pequeña (< 500 velas),
        # bajamos progresivamente hasta el 20% del threshold base para garantizar masa crítica.
        if _mask_contested.sum() < 500 and len(xgb_probs) > 1000:
            _contested_threshold_emergency = max(0.20, _contested_threshold * 0.80)
            _mask_contested = (xgb_probs >= _contested_threshold_emergency)
            print(f"[FIX-02] Zona disputada pequeña. Bajando threshold emergencia: {_contested_threshold:.4f} -> {_contested_threshold_emergency:.4f}")  # debug
            logger.warning(
                "[FIX-02] Zona disputada pequeña ({} muestras). Threshold emergencia: {:.4f} -> {:.4f}",
                _mask_contested.sum(), _contested_threshold, _contested_threshold_emergency
            )
            _contested_threshold = _contested_threshold_emergency

        if _mask_contested.sum() < 100:
            # Si sigue siendo minúscula (o la ventana de entrenamiento es enana), ignoramos el filtro
            _mask_contested = np.ones(len(xgb_probs), dtype=bool)

            logger.warning("[V2-META-DRIVEN] Peligro de colapso de muestra. Filtro Event-Driven desactivado.")
            
        _n_original = len(y)
        _n_filtered = int(_mask_contested.sum())
        
        logger.info(
            f"[V2-META-DRIVEN] Event-Driven Filter (>= {_contested_threshold}): "
            f"reteniendo {_n_filtered}/{_n_original} secuencias ({(1.0 - _n_filtered/max(1, _n_original)):.1%} ruido purgado)."
        )
        
        X_seq_filtered = X_seq[_mask_contested]
        xgb_probs_filtered = xgb_probs[_mask_contested]
        y_filtered = y[_mask_contested]
        hmm_regime_filtered = hmm_regime[_mask_contested] if hmm_regime is not None else None
        timestamps_filtered = timestamps[_mask_contested] if timestamps is not None else None

        # 1. Extraer embeddings rolling (deterministico)
        logger.info("Extrayendo rolling stats (mean/std/slope)...")
        embeddings = self.extractor.extract_embeddings(X_seq_filtered)
        logger.info(f"  Rolling embeddings: {embeddings.shape} ")

        # 2. Concatenar embeddings + probas XGBoost [+ HMM one-hot] → matriz para RF
        parts = [
            embeddings,                          # (N, n_feat*3) — rolling context
            xgb_probs_filtered.reshape(-1, 1),   # (N, 1)  — XGBoost signal strength
        ]
        if hmm_regime_filtered is not None:
            parts.append(hmm_regime_filtered)    # (N, n_states) — market regime context
        X_combined = np.hstack(parts)
        logger.info(f"X_combined para RF: {X_combined.shape} "
                    f"(rolling={embeddings.shape[1]} + xgb=1"
                    f"{ ' + hmm=' + str(hmm_regime_filtered.shape[1]) if hmm_regime_filtered is not None else ''})") 

        # 3. Calcular sample_weight por decaimiento temporal REAL (MEJORA-WEIGHT-01)
        try:
            from config.settings import cfg as _cfg_ml_sw
            _ml_floor = float(_cfg_ml_sw.metalabeler.weight_decay_floor)
        except Exception:
            _ml_floor = 0.3

        if timestamps_filtered is not None:
            t_max = timestamps_filtered.max()
            days_from_end = (t_max - timestamps_filtered).days.values
            _sw = np.exp(-WEIGHT_DECAY_ALPHA * (days_from_end / 365.0))
        else:
            _positions = np.linspace(0.0, 1.0, _n_filtered)
            _sw = np.exp(-WEIGHT_DECAY_ALPHA * (1.0 - _positions))
            
        # [FIX-HMM-AMNESIA 2026-06-14] Floor para prevenir amnesia total en regímenes antiguos
        _sw = np.clip(_sw, _ml_floor, 1.0)
            
        # [FIX-CAPA2-ASYM] Asimetria Dinamica por Regimen
        # Evita la dilucion de regimenes minoritarios y muy hostiles (ej. Bear Crash).
        # Incrementa el peso de los WINS en regimenes donde los LOSSES dominan abrumadoramente.
        _regime_booster = np.ones(len(y_filtered))
        if hmm_regime_filtered is not None:
            for r in range(hmm_regime_filtered.shape[1]):
                mask_r = (hmm_regime_filtered[:, r] == 1)
                if mask_r.sum() > 0:
                    y_r = y_filtered[mask_r]
                    wins_r = (y_r == 1).sum()
                    losses_r = (y_r == 0).sum()
                    if wins_r > 0 and losses_r > 0:
                        # Ratio de dificultad del regimen
                        local_ratio = float(losses_r) / wins_r
                        # Usar raiz cuadrada para suavizar (booster entre 1x y 5x)
                        booster = max(1.0, min(5.0, np.sqrt(local_ratio)))
                        win_mask_r = mask_r & (y_filtered == 1)
                        _regime_booster[win_mask_r] *= booster
                        print(f"[BUG-FIX-LOG 2026-06-07] [CAPA2-ASYM] Regime {r}: wins={wins_r}, losses={losses_r}, ratio={local_ratio:.1f}x -> Win Booster={booster:.2f}x")
                        logger.info(f"[CAPA2-ASYM] Regime {r} Win Booster: {booster:.2f}x (ratio={local_ratio:.1f})")

        _sw = _sw * _regime_booster
        _sw = _sw / _sw.mean()
        self._training_sample_weights = _sw
        logger.info(
            f"[M-40] sample_weight MetaLabeler: alpha={WEIGHT_DECAY_ALPHA}, "
            f"ratio_reciente/antiguo={_sw[-1]/_sw[0]:.2f}x ({_n_filtered} muestras)"
        )

        # 3b. Entrenar RF árbitro con énfasis temporal
        logger.info("Entrenando Random Forest árbitro...")
        self.rf.fit(X_combined, y_filtered, sample_weight=self._training_sample_weights)

        # 5. Evaluación RF: IS + OOS temporal (FIX-RF-OOS-01)
        # Bug anterior: solo se reportaba IS accuracy (self.rf.predict(X_combined) == y).
        # Fix: añadir TimeSeriesSplit(3) sobre X_combined para obtener OOS real.
        from sklearn.ensemble import RandomForestClassifier as _RFC
        from sklearn.model_selection import TimeSeriesSplit as _TSS
        is_acc = np.mean(self.rf.predict(X_combined) == y_filtered)
        logger.info(f"RF accuracy IS (solo referencia): {is_acc:.2%}")

        # [FIX-EMBARGO-META-CV-01 2026-05-28] gap calculado dinámicamente desde settings.yaml
        # Lógica: TBM vertical barrier (embargo_hours) + MetaLabeler sequence length (seq_len)
        # Antes: 144H hardcodeado — desync si embargo_hours o seq_len cambian en settings.
        # SOP R3: embargo temporal crítico — política No-Fallback LOUD.
        try:
            from config.settings import cfg as _cfg_embargo_cv
            _embargo_h_cv = int(_cfg_embargo_cv.sop.embargo_hours)
            _seq_len_cv   = int(_cfg_embargo_cv.metalabeler.seq_len)
            _embargo_h    = _embargo_h_cv + _seq_len_cv
            print(f"[FIX-EMBARGO-META-CV-01] MetaLabeler CV gap={_embargo_h}H (embargo={_embargo_h_cv}H + seq_len={_seq_len_cv}H)")  # RULE[fixbugsprints.md]
        except Exception as _e_emb_cv:
            raise RuntimeError(
                f"[CRITICAL][FIX-EMBARGO-META-CV-01] No se pudo leer embargo_hours/seq_len de settings.yaml. "
                f"Política No-Fallback SOP R3. Error: {_e_emb_cv}"
            ) from _e_emb_cv
        _tscv = _TSS(n_splits=3, gap=_embargo_h)
        # MEJ-META-02: gap=embargo_h asume frecuencia 1H perfecta — correcto para luna
        _oos_accs = []
        for _tr_i, _val_i in _tscv.split(X_combined):
            _rf_cv = _RFC(
                n_estimators=min(self.rf.n_estimators, 100),  # mas rapido en CV
                max_features=self.rf.max_features, 
                min_samples_leaf=self.rf.min_samples_leaf,
                max_depth=self.rf.max_depth,             # CRÍTICO: prevenir crecimiento infinito
                max_samples=self.rf.max_samples,         # CRÍTICO: mantener bagging idéntico
                class_weight=self.rf.class_weight, 
                # [FIX-RANDOM-STATE-02] CV RF usa mismo seed que el RF principal
                random_state=int(_os_meta.environ.get('LUNA_SEED', 42)), n_jobs=-1
            )
            # LOGIC-META-01 FIX (2026-04-06): aplicar sample_weight con decaimiento temporal
            _sw_cv = self._training_sample_weights[_tr_i]
            _sw_cv = _sw_cv / _sw_cv.mean()
            _rf_cv.fit(X_combined[_tr_i], y_filtered[_tr_i], sample_weight=_sw_cv)
            _oos_accs.append(np.mean(_rf_cv.predict(X_combined[_val_i]) == y_filtered[_val_i]))
        oos_acc_mean = float(np.mean(_oos_accs))
        oos_acc_std  = float(np.std(_oos_accs))

        logger.info(
            f"[FIX-RF-OOS-01] RF temporal CV (3-fold): "
            f"OOS_acc={oos_acc_mean:.2%} +/- {oos_acc_std:.2%} "
            f"(IS={is_acc:.2%} — gap={is_acc - oos_acc_mean:.2%})"
        )
        if is_acc - oos_acc_mean > 0.10:
            logger.warning(
                f"[FIX-RF-OOS-01] POSIBLE OVERFITTING RF: IS-OOS gap={is_acc - oos_acc_mean:.2%} > 10%"
            )

        # ══════════════════════════════════════════════════════════════════════
        # [GUARDIAN-03] MetaLabeler Overfit Guardian (Memorización)
        # Si el gap entre Train y CV es > X%, el RF memorizó el ruido y su CV
        # es peor que lanzar una moneda. Destruirá las señales en OOS.
        # ══════════════════════════════════════════════════════════════════════
        try:
            from config.settings import cfg as _cfg_g
            _max_overfit_gap = float(_cfg_g.metalabeler.guardian_max_overfit_gap)
        except Exception as e:
            raise RuntimeError(f"[CRITICAL-SOP] Falta metalabeler.guardian_max_overfit_gap en settings.yaml: {e}")

        if is_acc - oos_acc_mean > _max_overfit_gap:
            logger.error(
                f"[GUARDIAN-03] MetaLabeler OVERFIT DETECTADO: IS_acc={is_acc:.2%} vs CV_acc={oos_acc_mean:.2%} "
                f"(Gap = {is_acc - oos_acc_mean:.2%} > {_max_overfit_gap:.2%}). El árbitro no generaliza. Abortando."
            )
            print(f"[GUARDIAN-03] FATAL: RF MetaLabeler Overfit Extremo (Gap > {_max_overfit_gap:.2%}). Abortando.")
            import sys
            sys.exit(3)

        self._trained = True
        self._hmm_context_used = (hmm_regime is not None)  # BUG-4 FIX: flag para save()
        return {
            "train_accuracy_is":  float(is_acc),
            "oos_accuracy_cv":    oos_acc_mean,
            "oos_accuracy_std":   oos_acc_std,
            "n_samples":          _n_filtered,
            "lstm_hidden":        self._lstm_hidden_dim,
            "rf_n_estimators":    self.rf.n_estimators,
        }

    def _pretrain_lstm(self, X_seq: np.ndarray, y: np.ndarray,
                       epochs: int = 100, lr: float = 3e-4):
        """
        Pre-entrena el LSTM extractor como clasificador binario de una capa.
        La capa de clasificación se descarta al terminar — solo se retienen
        los pesos del LSTM.
        """
        from torch.optim import Adam

        # Añadir capa de clasificación temporal
        class _TempClassifier(nn.Module):
            def __init__(self, extractor, hidden_dim):
                super().__init__()
                self.extractor = extractor
                self.head = nn.Linear(hidden_dim, 1)
                self.sig = nn.Sigmoid()

            def forward(self, x):
                return self.sig(self.head(self.extractor(x)))

        # [FIX-AE-DETERMINISM-01 2026-06-26] Seedear init del clf torch (sin seed -> meta-filtrado
        # no reproducible). El loader (~628) es shuffle=False, no necesita generador.
        from luna.utils.determinism import seed_everything as _seed_meta_clf
        _seed_meta_clf()
        clf = _TempClassifier(self.extractor, self.extractor.hidden_dim)
        optimizer = Adam(clf.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # FIX-LSTM-SPLIT-01: split temporal IS/OOS para early stopping en val_loss.
        # Bug anterior: early stopping basado en train_loss (puede overfit sin saberlo).
        # Fix: tomar el 20% mas reciente (con embargo 96 timesteps) como val set.
        _n_lstm = len(X_seq)
        _val_split_lstm = int(_n_lstm * 0.80)
        _val_start_lstm = min(_val_split_lstm + 96, _n_lstm)  # embargo 96 timesteps
        has_val = _val_start_lstm < _n_lstm and (_n_lstm - _val_start_lstm) >= 32

        if has_val:
            X_train_lstm = X_seq[:_val_split_lstm]
            y_train_lstm = y[:_val_split_lstm]
            X_val_lstm   = X_seq[_val_start_lstm:]
            y_val_lstm   = y[_val_start_lstm:]
            ds_train_lstm = TensorDataset(
                torch.FloatTensor(X_train_lstm),
                torch.FloatTensor(y_train_lstm).unsqueeze(1)
            )
            X_val_t = torch.FloatTensor(X_val_lstm)
            y_val_t = torch.FloatTensor(y_val_lstm).unsqueeze(1)
            logger.info(
                f"[FIX-LSTM-SPLIT-01] LSTM split temporal: train={len(y_train_lstm)} "
                f"/ embargo=96 / val={len(y_val_lstm)} — early stop en val_loss"
            )
        else:
            ds_train_lstm = TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(y).unsqueeze(1))
            X_val_t, y_val_t = None, None
            logger.debug("[FIX-LSTM-SPLIT-01] Dataset demasiado pequeno para split — early stop en train_loss")

        ds = ds_train_lstm
        loader = DataLoader(ds, batch_size=128, shuffle=False)  # sin shuffle — temporal

        clf.train()
        best_loss = float('inf')
        patience_count = 0
        # P5-FIX: leer patience desde settings.yaml
        try:
            from config.settings import cfg as _cfg_p
            PATIENCE = int(_cfg_p.metalabeler.patience)
        except Exception:
            PATIENCE = 15  # fallback
        # BUG-R12-02 fix: gradient clipping para prevenir exploding gradients.
        # Sin este guard, gradientes > 1e6 vuelven los pesos inf/NaN silenciosamente.
        # Los embeddings NaN pasan al RF sin error, que luego predice NaN en OOS.
        # P2-2-FIX (2026-03-30): cfg_gradient_clip era undefined — NameError si se invocaba.
        try:
            from config.settings import cfg as _cfg_grad
            MAX_GRAD_NORM = float(_cfg_grad.metalabeler.gradient_clip)
        except Exception:
            MAX_GRAD_NORM = 1.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = clf(xb)
                loss = criterion(pred, yb)
                loss.backward()
                # BUG-R12-02 fix A: clip gradientes antes de actualizarlos
                torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()
                batch_loss = loss.item()
                # BUG-R12-02 fix A: detectar NaN loss por batch
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    logger.error(
                        f"LSTM loss={batch_loss} en epoch {epoch+1} batch — "
                        f"gradientes explosivos detectados. Abortando pre-entrenamiento."
                    )
                    self._lstm_nan_abort = True
                    return  # Extractor LSTM no entrenado — RF usará solo xgb_probs
                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            # BUG-R12-02 fix A: detectar NaN loss por epoch
            if math.isnan(avg_loss):
                logger.error(
                    f"LSTM avg_loss=NaN en epoch {epoch+1} — abortando pre-entrenamiento."
                )
                self._lstm_nan_abort = True
                return

            # FIX-LSTM-SPLIT-01: usar val_loss para early stopping si hay val set disponible
            if has_val and X_val_t is not None:
                clf.eval()
                with torch.no_grad():
                    val_pred = clf(X_val_t)
                    val_loss_v = criterion(val_pred, y_val_t).item()
                clf.train()
                monitor_loss = val_loss_v
            else:
                monitor_loss = avg_loss  # fallback a train_loss

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    logger.info(f"LSTM early stop en epoch {epoch+1} — monitor={'val' if has_val else 'train'}_loss")
                    break

        self._lstm_nan_abort = False
        logger.info(f"LSTM pre-entrenamiento finalizado. Final loss: {best_loss:.4f}")
        self.best_lstm_loss = best_loss  # Guardado en self para acceso en run()
        # El self.extractor ya tiene los pesos entrenados (compartidos con _TempClassifier)

    def predict_proba(self, X_seq: np.ndarray, xgb_probs: np.ndarray,
                      hmm_regime: np.ndarray | None = None) -> np.ndarray:
        """Retorna probabilidades de la clase positiva (LONG).

        BUG-R12-02 fix A: si los embeddings LSTM son NaN (extractor no convergió),
        se retorna xgb_probs como fallback — transparente para el pipeline OOS.
        El log registra el fallback para diagnóstico.
        """
        if getattr(self, "mocked", False):
            return np.ones(len(X_seq), dtype=np.float32) * 0.60
            
        if not self._trained:
            raise RuntimeError("MetaLabelerV2.train() debe llamarse antes de predict_proba()")
        embeddings = self.extractor.extract_embeddings(X_seq)

        # [BUGFIX-ML-SHIELD] Auto-ajustar hmm_regime para evitar desajustes de dimensiones (SOP V10 compliance)
        expected_hmm_states = self.rf.n_features_in_ - embeddings.shape[1] - 1
        if expected_hmm_states > 0:
            if hmm_regime is None:
                hmm_regime = np.zeros((len(X_seq), expected_hmm_states), dtype=np.float32)
                print(f"[BUGFIX-ML-SHIELD] Generando hmm_regime vacío de {expected_hmm_states} columnas (era None) (RULE[fixbugsprints.md]).")
                logger.info(f"MetaLabelerV2: Generando hmm_regime vacío de {expected_hmm_states} columnas.")
            elif hmm_regime.shape[1] != expected_hmm_states:
                print(f"[BUGFIX-ML-SHIELD] Ajustando hmm_regime de {hmm_regime.shape[1]} a {expected_hmm_states} columnas para coincidir con n_features_in={self.rf.n_features_in_} (RULE[fixbugsprints.md]).")
                logger.info(f"MetaLabelerV2: Ajustando hmm_regime de {hmm_regime.shape[1]} a {expected_hmm_states} columnas.")
                if hmm_regime.shape[1] < expected_hmm_states:
                    pad_width = expected_hmm_states - hmm_regime.shape[1]
                    hmm_regime = np.pad(hmm_regime, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    hmm_regime = hmm_regime[:, :expected_hmm_states]

        # BUG-R12-02 guard NaN: si el LSTM produjo embeddings inválidos, usar xgb_probs
        nan_frac = np.isnan(embeddings).mean()
        if nan_frac > 0.0:
            logger.warning(
                f"MetaLabelerV2: {nan_frac:.1%} de embeddings son NaN "
                f"(LSTM no convergió — grad explosion). "
                f"Fallback a xgb_probs solo. Revisar BUG-R12-02."
            )
            return np.clip(xgb_probs, 0.0, 1.0)  # xgb_probs ya calibradas

        parts = [embeddings, xgb_probs.reshape(-1, 1)]
        if expected_hmm_states > 0 and hmm_regime is not None:
            parts.append(hmm_regime)
        X_combined = np.hstack(parts)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X_combined)[:, 1]
        return self.rf.predict_proba(X_combined)[:, 1]

    def predict(self, X_seq: np.ndarray, xgb_probs: np.ndarray,
                threshold: float = 0.50,
                hmm_regime: np.ndarray | None = None) -> np.ndarray:
        """Predicción binaria con umbral configurable."""
        return (self.predict_proba(X_seq, xgb_probs, hmm_regime) >= threshold).astype(int)

    def save(self, output_dir: str | Path, direction_mode: str = "long") -> None:
        """Guarda RollingStatsExtractor state y RF árbitro (joblib).
        R17-FIX-LSTM-01: ya no guarda LSTM .pt — solo el config del extractor rolling.
        Mantiene compatibilidad: guarda un .pt vacío del _lstm_extractor para que
        generate_oos_predictions.py (que chequea si el .pt existe) no falle.
        """
        import joblib
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Guardar .pt del LSTM legacy (vacío) para compatibilidad con generate_oos_predictions
        # P2-6-FIX: instanciar lazily (solo en save) para evitar DeprecationWarning en __init__
        if self._lstm_extractor is None:
            import warnings as _wsave
            with _wsave.catch_warnings():
                _wsave.simplefilter("ignore", DeprecationWarning)
                self._lstm_extractor = TemporalFeatureExtractor(
                    input_dim=self.extractor.input_dim,
                    hidden_dim=self._lstm_hidden_dim,
                )
        torch.save(self._lstm_extractor.state_dict(),
                   output_dir / f"metalabeler_v2_{direction_mode}_lstm.pt")
        # RF
        joblib.dump(self.rf, output_dir / f"metalabeler_v2_{direction_mode}_rf.joblib")
        # Config con extractor_type para que load() sepa qué extractor usar
        config = {
            "extractor_type":  "rolling_stats",         # R17-FIX: nuevo campo
            "lstm_hidden":     self._lstm_hidden_dim,
            "input_dim":       self.extractor.input_dim,
            "rf_n_estimators": self.rf.n_estimators,
            "seq_len":         SEQ_LEN,
            "arch":            "RollingStats-extractor + RF-arbitro",
            "hmm_context":     getattr(self, '_hmm_context_used', False),
            "hmm_n_states":    HMM_N_STATES,
            "trained":         self._trained,
            "seq_features":    getattr(self, '_seq_features', []),
        }
        (output_dir / f"metalabeler_v2_{direction_mode}_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        logger.info(f"MetaLabelerV2 guardado en {output_dir}")

    @classmethod
    def load(cls, model_dir: str | Path, input_dim: int = None, direction_mode: str = "long") -> "MetaLabelerV2":
        """Carga un MetaLabelerV2 previamente guardado.
        R17-FIX-LSTM-01: detecta extractor_type en config para usar
        RollingStatsExtractor (nuevo) o LSTM (legacy backward-compat).
        """
        import joblib
        model_dir = Path(model_dir)
        
        # Check if loading in mock/dry-run mode
        config_path = model_dir / f"metalabeler_v2_{direction_mode}_config.json"
        is_mock = False
        if not config_path.exists():
            is_mock = True
        else:
            try:
                content = config_path.read_text(encoding='utf-8')
                if "mocked" in content:
                    is_mock = True
            except Exception:
                pass
                
        if is_mock:
            logger.warning(f"[MetaLabelerV2] Cargando MetaLabelerV2 mockeado para {direction_mode} desde {model_dir}")
            print(f"[MetaLabelerV2/MOCK] Cargando MetaLabelerV2 mockeado para {direction_mode} desde {model_dir}")
            obj = cls.__new__(cls)
            obj.input_dim = input_dim if input_dim is not None else 10
            obj.hidden_dim = 32
            obj._lstm_hidden_dim = 32
            obj._trained = True
            obj.mocked = True
            
            class MockExtractor:
                def __init__(self):
                    self.input_dim = 10
                def save_state(self):
                    return {"extractor_type": "mock_stats", "input_dim": 10}
                def extract_embeddings(self, X_seq):
                    import numpy as np
                    return np.zeros((len(X_seq), 32))
            
            obj.extractor = MockExtractor()
            obj._lstm_extractor = None
            obj.rf = None
            obj.calibrator = None
            return obj

        config = json.loads(config_path.read_text())
        saved_input_dim = config.get("input_dim", input_dim)
        if saved_input_dim is None:
            raise ValueError("input_dim no encontrado en config ni pasado como parámetro")
        obj = cls(input_dim=saved_input_dim, hidden_dim=config["lstm_hidden"],
                  n_estimators=config.get("rf_n_estimators", RF_N_ESTIMATORS))

        extractor_type = config.get("extractor_type", "lstm")  # default legacy
        if extractor_type == "rolling_stats":
            # R17-FIX: el extractor es RollingStatsExtractor (ya instanciado en __init__)
            logger.info("[R17-FIX-LSTM-01] Cargando con RollingStatsExtractor")
        else:
            # Legacy: cargar pesos LSTM del .pt
            lstm_path = model_dir / f"metalabeler_v2_{direction_mode}_lstm.pt"
            if lstm_path.exists():
                # Instanciar silenciosamente para no disparar DeprecationWarning
                import warnings as _wload
                with _wload.catch_warnings():
                    _wload.simplefilter("ignore", DeprecationWarning)
                    _lstm_extr = TemporalFeatureExtractor(
                        input_dim=saved_input_dim, hidden_dim=config["lstm_hidden"]
                    )
                _lstm_extr.load_state_dict(
                    torch.load(lstm_path, weights_only=False)
                )
                obj._lstm_extractor = _lstm_extr
                obj.extractor = obj._lstm_extractor  # usar LSTM legacy
                logger.info("[legacy] Cargando TemporalFeatureExtractor (LSTM)")

        obj.rf = joblib.load(model_dir / f"metalabeler_v2_{direction_mode}_rf.joblib")
        obj._trained = True
        logger.info(f"MetaLabelerV2 cargado desde {model_dir} (input_dim={saved_input_dim}, extractor={extractor_type})")
        return obj


# ============================================================================
# ORQUESTADOR DE ENTRENAMIENTO
# ============================================================================
class MetaLabelerV2Trainer:
    """
    Orquesta el entrenamiento del MetaLabelerV2 con CPCV purgeado.

    Usa los mismos datos de training que el BiLSTM original pero con:
    1. CPCV C(10,2) para generar probs XGBoost OOS (P1-8)
    2. Arquitectura LSTM→RF en lugar de BiLSTM profunda (P1-9)
    3. Validación temporal con embargo 96H (SOP R3)
    """

    def __init__(self, direction: str = "long"):
        self.root = ROOT
        self.direction = direction

    def _load_data(self) -> tuple:
        """Carga y prepara datos. Idéntico a MetaLabelerTrainer._prepare_data()."""
        df = pd.read_parquet(self.root / "data" / "features" / "features_train.parquet")

        # ── [CAPA-1] Rolling Window de 3 años (Filtro de Memoria) ──────────────
        try:
            from config.settings import cfg as _cfg_rw
            _t_mode = str(_cfg_rw.wfb.training_mode)
            if _t_mode == 'rolling':
                _rw_years = int(_cfg_rw.wfb.rolling_window_years)
                _train_end_val = str(_cfg_rw.temporal_splits.train_end)
                if _train_end_val:
                    _train_end_dt = pd.to_datetime(_train_end_val, utc=True)
                    _rolling_start = _train_end_dt - pd.DateOffset(years=_rw_years)
                    
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                        
                    _len_before = len(df)
                    df = df[df.index >= _rolling_start]

                    logger.info(
                        f"[CAPA-1] training_mode='rolling' ({_rw_years} años): "
                        f"Descartadas {_len_before - len(df)} velas anteriores a {_rolling_start.date()}."
                    )
        except Exception as e:

            logger.warning(f"[CAPA-1] Error aplicando Rolling Window en MetaLabeler: {e}. Fallback a 'expanding'.")
        # ───────────────────────────────────────────────────────────────────────

        # FIX-WFB-META-01 (2026-03-28): Respetar train_end de la ventana WFB activa.
        # Bug anterior: features_train.parquet es un archivo estático — si WFB lo regenera
        # con train_end=2025-03-31, el MetaLabeler de W4 entrenaba con datos del futuro
        # respecto al holdout de W3. Esto causaba WR=47% y 364 trades en W3-W5.
        # Fix: corte temporal explícito sobre el dataframe, igual que hace XGBoost.
        try:
            from config.settings import cfg as _cfg_wfb
            _train_end_str = getattr(_cfg_wfb.temporal_splits, "train_end", None)
            if _train_end_str is not None:
                _train_end_ts = pd.Timestamp(_train_end_str, tz='UTC')
                _rows_before = len(df)
                df = df[df.index <= _train_end_ts]
                logger.info(
                    f"[FIX-WFB-META-01] train_end cutoff aplicado: {_train_end_str} "
                    f"({_rows_before} → {len(df)} filas, -{_rows_before - len(df)} futuras eliminadas)"
                )
            else:
                logger.warning("[FIX-WFB-META-01] train_end no encontrado en settings — usando dataset completo")
        except Exception as _e:
            logger.warning(f"[FIX-WFB-META-01] No se pudo aplicar cutoff temporal: {_e}")

        with open(self.root / "data" / "features" / "selected_features.json") as f:
            _sfi_data = json.load(f)
            # P1-3-FIX (2026-03-30): incluir pass_through_features (KMeans_Tribe_ID, Master_Causal_Signal, etc.)
            # Antes: solo se cargaba "selected_features" — las features de paso directo quedaban excluidas
            # del MetaLabeler aunque sean las señales primarias del pipeline post-AI Mining.
            _sfi_feats = _sfi_data.get("selected_features", [])
            _pt_feats  = _sfi_data.get("pass_through_features", [])
            features_list = list(dict.fromkeys(_sfi_feats + _pt_feats))  # union sin duplicados
            if _pt_feats:
                logger.info(
                    f"[P1-3-FIX] MetaLabeler features: {len(_sfi_feats)} SFI + "
                    f"{len(_pt_feats)} pass-through = {len(features_list)} total"
                )

        df_hmm = pd.read_parquet(self.root / "data" / "features" / "hmm_regime_labels.parquet")
        # [FIX-HMM-JOIN-01] Mismo fix que en train_xgboost_v2.py:
        # features_train.parquet ya contiene HMM_Regime (integrado en Paso 3B del FP).
        # Hacer df.join(df_hmm) cuando ambos tienen HMM_Regime → ValueError.
        _hmm_meta_overlap = [c for c in df_hmm.columns if c in df.columns]
        if _hmm_meta_overlap:
            from loguru import logger as _log_meta_join
            _log_meta_join.info(
                f"[FIX-HMM-JOIN-01][MetaLabeler] Columnas solapadas eliminadas del df_hmm "
                f"antes del join: {_hmm_meta_overlap}"
            )
            print(f"[META][FIX-HMM-JOIN-01] Eliminando columnas duplicadas del join HMM: {_hmm_meta_overlap}")  # debug
            df_hmm = df_hmm.drop(columns=_hmm_meta_overlap)
        df = df.join(df_hmm)
        df["HMM_Regime"] = df["HMM_Regime"].ffill().bfill()


        # [FIX-P2-TIMING] Calcular timing features in-line (2026-03-26)
        # Sin estas features, el CPCV del XGBoost (que usa 27 features) se desvirtúa respecto 
        # al modelo real (30 features), arruinando la predictibilidad del MetaLabeler.
        if "FundingRate" in df.columns:
            df["timing_funding_acum8h"] = df["FundingRate"].ewm(span=8, min_periods=1).mean()
        if "close" in df.columns:
            _r24h = df["close"].pct_change(24)
            _r7d  = df["close"].pct_change(168)
            df["timing_momentum_div"] = _r24h - _r7d
        if "close" in df.columns and "volume" in df.columns:
            _r24h_abs   = df["close"].pct_change(24).abs()
            _vol_ma     = df["volume"].rolling(window=720, min_periods=48).mean()
            _vol_ratio  = df["volume"] / (_vol_ma + 1e-6)
            df["timing_vol_divergence"] = (_r24h_abs / (_vol_ratio + 1e-6)).clip(upper=5.0)

        # [A2] Calcular btc_drawdown_from_ath para el MetaLabeler
        if "close" in df.columns:
            rolling_ath = df["close"].rolling(window=90*24, min_periods=24).max().ffill().bfill()
            df["btc_drawdown_from_ath"] = (df["close"] / rolling_ath) - 1.0

        # Features disponibles (SFI base)
        cols = [c for c in features_list if c in df.columns]
        
        # [A2] Inyección explícita de features críticas que el SFI (features_list) ignora
        extra_features = [
            "timing_funding_acum8h", "timing_momentum_div", "timing_vol_divergence", 
            "btc_drawdown_from_ath", "HMM_Regime"
        ]
        for ef in extra_features:
            if ef in df.columns and ef not in cols:
                cols.append(ef)
                logger.debug(f"[A2] Feature OOS inyectada en seq_features de MetaLabeler: {ef}")

        # FIX-HMM-ALIGN-01 (2026-03-26): Orden topológico estricto garantizado.
        # NUNCA usar pd.get_dummies() aquí. Genera columnas según el orden en el DataFrame.
        # Si las columnas no tienen un orden determinístico [0, 1, 2, 3, 4], el array numpy OOS
        # entrará cruzado al RF en SignalFilter provocando un colapso en el WinRate y volumen del MetaLabeler.
        # [FIX-DUMMIES-DYNAMIC] Dynamically compute n_states_total based on maximum integer key in df["HMM_Regime"] to prevent dropping custom states
        max_state_empirico = int(df["HMM_Regime"].dropna().max()) if "HMM_Regime" in df.columns and len(df["HMM_Regime"].dropna()) > 0 else HMM_N_STATES
        n_states_total = max(HMM_N_STATES, max_state_empirico) + 1
        print(f"[FIX-DUMMIES-DYNAMIC] Computed n_states_total={n_states_total} dynamically (HMM_N_STATES={HMM_N_STATES}, max_state_empirico={max_state_empirico})")  # debug
        logger.info("[FIX-DUMMIES-DYNAMIC] Computed n_states_total={} dynamically (HMM_N_STATES={}, max_state_empirico={})", n_states_total, HMM_N_STATES, max_state_empirico)
        hmm_dummies = []
        for s in range(n_states_total):
            col = f"HMM_OH_{s}"
            df[col] = (df["HMM_Regime"].fillna(-1).astype(int) == s).astype(float)
            hmm_dummies.append(col)
        
        all_cols = list(set(cols + hmm_dummies + ["close", "HMM_Semantic"]))
        df = df[[c for c in df.columns if c in all_cols]].copy()
        df = df.dropna(subset=["close"])

        # MATH-META-01: Validar robustez cronológica
        assert df.index.is_monotonic_increasing, "MATH-META-01: MetaLabeler dataset must be sorted by time"

        return df, cols


    def _build_sequences(self, df: pd.DataFrame, feature_cols: list[str],
                         y: np.ndarray, seq_len: int = SEQ_LEN,
                         return_sequences: bool = True):
        """Construye matrices de secuencias (N, seq_len, n_features).

        FIX-META-NAN-01 (2026-03-21): preservar NaN nativos en lugar de fillna(0).
        El RollingStatsExtractor propaga NaN a mean/std/slope → el RF los trata como
        valores ausentes (consistente con el comportamiento en producción).

        MEM-FIX-01 (2026-03-30): parámetro return_sequences=False para obtener solo
        valid_idx sin materializar el array (~250 MB), permitiendo diferir la RAM
        hasta después del CPCV intensivo.
        """
        X = df[[c for c in feature_cols if c in df.columns]].values
        valid_idx = np.arange(seq_len, len(X))
        if return_sequences:
            from numpy.lib.stride_tricks import sliding_window_view
            # (MEJ-META-01) Optimización extrema de memoria y CPU para arrays 3D
            sequences = sliding_window_view(X, window_shape=(seq_len, X.shape[1])).squeeze(axis=1)[:-1]
            return sequences, valid_idx
        return None, valid_idx

    def run(self, epochs: int = None, lstm_lr: float = None) -> dict:
        """Ejecuta el pipeline de entrenamiento completo."""
        # P5-FIX: leer epochs y lstm_lr desde settings.yaml si no se pasan explícitamente
        try:
            from config.settings import cfg as _cfg_meta
            if epochs is None:
                epochs = int(_cfg_meta.metalabeler.max_epochs)
            if lstm_lr is None:
                lstm_lr = float(_cfg_meta.metalabeler.lstm_lr)
            logger.debug(f"[P5-FIX] MetaLabeler epochs={epochs} lr={lstm_lr} (de settings.yaml)")
        except Exception:
            epochs   = epochs   or 100   # fallback por si settings no disponible
            lstm_lr  = lstm_lr  or 3e-4
        logger.info("=" * 60)
        logger.info("MetaLabelerV2Trainer — P1-9 (LSTM→RF Arquitectura Invertida)")
        logger.info("=" * 60)

        # 1. Cargar datos
        df, feature_cols = self._load_data()
        logger.info(f"Dataset: {len(df)} filas, {len(feature_cols)} features base")

        # 2. Generar labels TBM
        # BUG-3 FIX (P4-0-4, 2026-03-08): leer pt/sl de settings.yaml en lugar de [2.0, 1.0] fijo
        # TBM-REGIME-01 (2026-05-05): MetaLabelerV2 debe usar los MISMOS multiplicadores
        # PT/SL que XGBoost por régimen. Si usa un TBM estático (ej. 1.6/1.0), generará
        # labels incompatibles con los agentes entrenados para otros perfiles, causando
        # que aprenda que los agentes "se equivocan" cuando en realidad son correctos.
        try:
            from config.settings import cfg
            _pt_base = float(cfg.xgboost.pt_mult_min)
            _sl_base = float(cfg.xgboost.sl_mult_min)
            logger.info(f"TBM MetaLabelerV2: Base pt_mult={_pt_base}, sl_base={_sl_base} (de settings.yaml)")
        except Exception as _e_tbm_base:
            raise RuntimeError(f"Falta config TBM Base en settings. Política No-Fallback: {_e_tbm_base}") from _e_tbm_base

        from luna.features.tbm import apply_triple_barrier
        try:
            from config.settings import cfg as _cfg_tbm
            # Ajuste dinámico de nombres de variables para no usar getattr obsoleto
            _vbh_ml  = int(_cfg_tbm.xgboost.dynamic_horizon_max_h) # Se usa el máximo del horizonte como barrera vertical histórica
            _minr_ml = float(_cfg_tbm.xgboost.tbm_min_return)
            _dyn_barrier_ml = bool(_cfg_tbm.xgboost.dynamic_barrier)
            _dyn_min_ml = int(_cfg_tbm.xgboost.dynamic_horizon_min_h)
            _dyn_max_ml = int(_cfg_tbm.xgboost.dynamic_horizon_max_h)
        except Exception as _e_dyn:
            raise RuntimeError(f"Falta config TBM Dynamic en settings. Política No-Fallback: {_e_dyn}") from _e_dyn

        pt_mult_series = pd.Series(_pt_base, index=df.index)
        sl_mult_series = pd.Series(_sl_base, index=df.index)

        if "HMM_Semantic" in df.columns:
            try:
                from config.settings import cfg as _cfg_reg
                _regime_profiles_raw = getattr(_cfg_reg.xgboost, "regime_tbm_profiles", None)
                if _regime_profiles_raw is not None:
                    try:
                        _regime_profiles = vars(_regime_profiles_raw)
                    except TypeError:
                        _regime_profiles = dict(_regime_profiles_raw)
                        
                    for _pk, _pv in _regime_profiles.items():
                        try:
                            _prof_dict = vars(_pv)
                        except TypeError:
                            _prof_dict = dict(_pv)
                            
                        _pt_val = float(_prof_dict.get('pt_mult_min', _pt_base))
                        _sl_val = float(_prof_dict.get('sl_mult_min', _sl_base))
                        
                        _mask = df["HMM_Semantic"].astype(str).str.lower().str.startswith(str(_pk).lower())
                        pt_mult_series.loc[_mask] = _pt_val
                        sl_mult_series.loc[_mask] = _sl_val
                        
                    logger.info("[TBM-REGIME-01] MetaLabeler usando perfiles PT/SL adaptativos por régimen!")
            except Exception as e:
                logger.warning(f"[TBM-REGIME-01] Error aplicando PT/SL por régimen en MetaLabeler: {e}")

        try:
            from config.settings import cfg as _cfg_ml
            _lin_decay_ml = bool(_cfg_ml.xgboost.linear_decay_pt)
            _pt_decay_frac_ml = float(_cfg_ml.xgboost.pt_decay_fraction)
        except Exception as _e_dec:
            raise RuntimeError(f"Falta config TBM Decay en settings. Política No-Fallback: {_e_dec}") from _e_dec

        _side_val = -1.0 if self.direction == "short" else 1.0
        _sides_series = pd.Series(_side_val, index=df.index)

        tbm = apply_triple_barrier(
            price_series=df["close"],
            event_times=df.index,
            sides=_sides_series,
            pt_sl_multiplier=[pt_mult_series, sl_mult_series],
            min_return=_minr_ml,
            vertical_barrier_hours=_vbh_ml,
            dynamic_barrier=_dyn_barrier_ml,
            dynamic_horizon_min_h=_dyn_min_ml,
            dynamic_horizon_max_h=_dyn_max_ml,
            linear_decay_pt=_lin_decay_ml,
            pt_decay_fraction=_pt_decay_frac_ml,
        )
        df_labeled = df.join(tbm[["bin", "ret"]], how="inner")
        # Since 'sides' correctly applies the direction into TBM (inverting PT and SL for short),
        # a successful trade in either direction is marked as bin == 1.
        df_labeled["target"] = (df_labeled["bin"] == 1).astype(int)
        df_labeled = df_labeled.dropna(subset=["target", "ret"])

        feature_cols_available = [c for c in feature_cols if c in df_labeled.columns]
        y_all = df_labeled["target"].values
        close_rets = df_labeled["ret"]

        # 3. Obtener valid_idx — X_seq se materializa DESPUÉS del CPCV [MEM-FIX-01]
        # El array X_seq (shape ~62k×48×21, ≈250 MB float32) no se necesita hasta el
        # split temporal (post-CPCV). Diferirlo evita contención RAM con los 45+ fits
        # CPCV consecutivos que causan el WATCHDOG kernel hang en GPU.
        _, valid_idx = self._build_sequences(df_labeled, feature_cols_available, y_all,
                                             return_sequences=False)
        y = y_all[valid_idx]
        logger.info(
            f"[MEM-FIX-01] valid_idx: {len(valid_idx)} secuencias ({y.mean():.1%} positivos) "
            f"— X_seq se materializa post-CPCV para liberar RAM"
        )

        if len(y) < 200:
            logger.error("Datos insuficientes para MetaLabelerV2 (mín 200 secuencias)")
            return {}

        # [FASE 2C] Soporte para LightGBM Unified Proxy o RegimeRouter en CPCV
        use_lgbm = False
        try:
            from config.settings import cfg as _cfg_lgbm
            use_lgbm = bool(_cfg_lgbm.fase2.use_lgbm_ensemble)
        except Exception:
            pass

        use_regime = False
        try:
            from config.settings import cfg as _cfg_regime
            use_regime = bool(_cfg_regime.fase2.use_regime_agents)
        except Exception:
            pass

        if use_regime and "HMM_Semantic" not in df_labeled.columns:
            logger.warning("[FIX-P1-V4-3] use_regime=True pero HMM_Semantic no encontrado. Desactivando Multi-Agente en CPCV.")
            use_regime = False
            
        hmm_semantic_xgb = df_labeled["HMM_Semantic"].values[valid_idx] if use_regime else None

        # [FIX-P1-V4-3] Cargar firmas dinámicamente según arquitectura activa (Global o Multi-Agente)
        agent_sigs = {}
        try:
            from config.settings import cfg as _cfg_ml
            regimes_config = vars(_cfg_ml.fase2.regime_mapping)
        except Exception as e:
            logger.warning(f"Error cargando regime_mapping: {e}. Fallback interno.")
            regimes_config = {
                "bull":  ["1_BULL_TREND", "1_VOLATILE_BULL", "1_BULL_GRIND", "1_BULL_TREND_WEAK", "1_BULL_TREND_B", "1_VOLATILE_BULL_B"],
                "range": ["2_CALM_RANGE", "2_VOLATILE_RANGE", "2_CALM_RANGE_B", "2_VOLATILE_RANGE_B"],
                "bear":  ["3_CALM_BEAR", "3_BEAR_CRASH", "4_BEAR_FORCED"]
            }
        
        xgb_meta_features = []
        prefix = "lgbm_meta" if use_lgbm else "xgboost_meta"
        if use_regime:
            logger.info(f"  [FIX-P1-V4-3] Cargando MetaLabeler CPCV vía RegimeRouter (Multi-Agent, use_lgbm={use_lgbm})...")
            for name in regimes_config.keys():
                sig_p = self.root / "data" / "models" / f"{prefix}_{name}_{self.direction}_signature.json"
                if sig_p.exists():
                    with open(sig_p) as f:
                        agent_sigs[name] = json.load(f)
                        xgb_meta_features.extend(agent_sigs[name]["features"])
                else:
                    logger.warning(f"  No se encontró {sig_p.name} - agente omitido.")
            xgb_features = list(dict.fromkeys(xgb_meta_features))
        else:
            sig_path = self.root / "data" / "models" / f"{prefix}_{self.direction}_signature.json"
            if sig_path.exists():
                with open(sig_path) as f:
                    sig_data = json.load(f)
                agent_sigs["global"] = sig_data
                xgb_features = sig_data["features"]
            else:
                xgb_features = []

        xgb_features = [f for f in xgb_features if f in df_labeled.columns]

        # FIX-META-NAN-01 (2026-03-21): NO hacer fillna(0)
        X_xgb_df = df_labeled[xgb_features].iloc[valid_idx]
        y_xgb = y_all[valid_idx]
        n = len(X_xgb_df)
        oos_probs = np.full(n, np.nan)

        # [LOGIC-06/FIX-WEIGHT-01] Sample weights consistentes con ensemble_lgbm.py (decaimiento por año)
        timestamps_xgb = X_xgb_df.index
        try:
            from config.settings import cfg as _cfg_sw
            _train_end_year = pd.Timestamp(_cfg_sw.temporal_splits.train_end).year
        except Exception:
            _train_end_year = timestamps_xgb.year.max()
            
        years_ago = np.clip(_train_end_year - timestamps_xgb.year.to_numpy(), 0, None).astype(float)
        _sw_xgb = np.exp(-WEIGHT_DECAY_ALPHA * years_ago)
        _sw_xgb = _sw_xgb / _sw_xgb.mean()

        # CPCV C(10,2) inline
        n_groups = N_CPCV_GROUPS
        group_size = n // n_groups
        groups = [list(range(i * group_size, min((i + 1) * group_size, n))) for i in range(n_groups)]
        cpcv_splits = []
        timestamps_xgb = X_xgb_df.index
        
        for test_gidxs in combinations(range(n_groups), 2):
            test_flat = [i for gi in test_gidxs for i in groups[gi]]
            test_set = set(test_flat)
            
            # [MEJORA-EMBARGO-01] Purge temporal en lugar de posicional, y por cada bloque de test
            train_idx_all = np.array([i for i in range(n) if i not in test_set])
            if len(train_idx_all) == 0:
                continue
                
            train_mask = np.ones(len(train_idx_all), dtype=bool)
            for gi in test_gidxs:
                block = groups[gi]
                if not block:
                    continue
                block_start = timestamps_xgb[block[0]]
                block_end   = timestamps_xgb[block[-1]]
                purge_lo    = block_start - pd.Timedelta(hours=EMBARGO_H)
                purge_hi    = block_end   + pd.Timedelta(hours=EMBARGO_H)
                
                in_purge_zone = (
                    (timestamps_xgb[train_idx_all] >= purge_lo) &
                    (timestamps_xgb[train_idx_all] <= purge_hi)
                )
                train_mask &= ~in_purge_zone
                
            train_idx = train_idx_all[train_mask].tolist()

            if len(train_idx) >= 100:
                cpcv_splits.append((train_idx, test_flat))
                
        print(f"[META][MEJORA-EMBARGO-01] CPCV generado con {len(cpcv_splits)} splits efectivos usando purge temporal ({EMBARGO_H}H)")  # debug
        logger.info(f"[MEJORA-EMBARGO-01] CPCV generado con {len(cpcv_splits)} splits efectivos usando purge temporal ({EMBARGO_H}H)")

        import xgboost as xgb
        _c_total = len(list(combinations(range(n_groups), 2)))  # C(N_CPCV_GROUPS,2) teórico
        logger.info(f"CPCV: {len(cpcv_splits)}/{_c_total} splits válidos. Arquitectura: {'Multi-Agente XGBoost' if use_regime else ('LightGBM Global' if use_lgbm else 'XGBoost Global')}")
        
        for train_idx, test_idx in cpcv_splits:
            train_sw = _sw_xgb[train_idx]
            train_sw = train_sw / train_sw.mean()
            
            if not use_regime:
                if "global" not in agent_sigs:
                    logger.warning(f"Firma global no encontrada (agentes: {list(agent_sigs.keys())}). Fallback omitido.")
                    continue
                b_params = agent_sigs["global"].get("params", agent_sigs["global"].get("best_params", {}))
                
                if use_lgbm:
                    import lightgbm as lgb
                    cv_clf = lgb.LGBMClassifier(
                        objective="binary", metric="auc", n_estimators=150,
                        learning_rate=0.05, num_leaves=31, random_state=int(_os_meta.environ.get('LUNA_SEED', 42)), n_jobs=4,  # [FIX-RANDOM-STATE-02b]
                        verbose=-1, min_child_samples=50
                    )
                    cv_clf.set_params(**{k: v for k, v in b_params.items() if k in cv_clf.get_params()})
                else:
                    import xgboost as xgb
                    cv_clf = xgb.XGBClassifier(
                        objective="binary:logistic", tree_method="hist", device="cpu", n_jobs=4, random_state=int(_os_meta.environ.get('LUNA_SEED', 42)), verbosity=0,  # [FIX-RANDOM-STATE-02b]
                        **{k: v for k, v in b_params.items() if k not in ["objective", "tree_method", "device", "n_jobs", "random_state", "verbosity"]}
                    )
                
                _feats = [f for f in agent_sigs["global"]["features"] if f in X_xgb_df.columns]
                X_train_sub = X_xgb_df.iloc[train_idx][_feats].values
                X_test_sub  = X_xgb_df.iloc[test_idx][_feats].values
                
                # [FIX-SINGLE-CLASS-FOLD 2026-06-17] Guard contra ValueError
                _y_tr = y_xgb[train_idx]
                if len(np.unique(_y_tr)) < 2:
                    oos_probs[test_idx] = float(_y_tr[0])
                    continue
                    
                cv_clf.fit(X_train_sub, _y_tr, sample_weight=train_sw)
                oos_probs[test_idx] = cv_clf.predict_proba(X_test_sub)[:, 1]
                
            else:
                # OPTION 2: FULL Multi-Agent CPCV Routing
                # Para cada régimen disponible, entrenamos un clf aislado y resolvemos solo los test set correspondientes.
                for agent_name, permitted_regimes in regimes_config.items():
                    if agent_name not in agent_sigs: continue
                    sig = agent_sigs[agent_name]
                    a_feats = [f for f in sig["features"] if f in X_xgb_df.columns]
                    b_params = sig.get("params", sig.get("best_params", {}))
                    
                    if use_lgbm:
                        import lightgbm as lgb
                        cv_clf = lgb.LGBMClassifier(
                            objective="binary", metric="auc", n_estimators=150,
                            learning_rate=0.05, num_leaves=31, random_state=int(_os_meta.environ.get('LUNA_SEED', 42)), n_jobs=4,  # [FIX-RANDOM-STATE-02b]
                            verbose=-1, min_child_samples=50
                        )
                        cv_clf.set_params(**{k: v for k, v in b_params.items() if k in cv_clf.get_params()})
                    else:
                        import xgboost as xgb
                        cv_clf = xgb.XGBClassifier(
                            objective="binary:logistic", tree_method="hist", device="cpu", n_jobs=4, random_state=int(_os_meta.environ.get('LUNA_SEED', 42)), verbosity=0,  # [FIX-RANDOM-STATE-02b]
                            **{k: v for k, v in b_params.items() if k not in ["objective", "tree_method", "device", "n_jobs", "random_state", "verbosity"]}
                        )
                    
                    # Máscaras de enrutamiento
                    train_mask_r = np.isin(hmm_semantic_xgb[train_idx], permitted_regimes)
                    if train_mask_r.sum() < 10:
                        continue # insuficientes datos para entrenar el agente del régimen en este fold
                        
                    train_idx_r = np.array(train_idx)[train_mask_r]
                    train_sw_r  = train_sw[train_mask_r]
                    if train_sw_r.sum() > 0:
                        train_sw_r = train_sw_r / train_sw_r.mean()
                    
                    test_mask_r = np.isin(hmm_semantic_xgb[test_idx], permitted_regimes)
                    if test_mask_r.sum() == 0:
                        continue # Nada que predecir
                        
                    test_idx_r = np.array(test_idx)[test_mask_r]
                    
                    X_train_sub = X_xgb_df.iloc[train_idx_r.tolist()][a_feats].values # Fallback list subset
                    X_test_sub  = X_xgb_df.iloc[test_idx_r.tolist()][a_feats].values
                    
                    # [FIX-SINGLE-CLASS-FOLD 2026-06-17] Guard contra ValueError
                    _y_tr = y_xgb[train_idx_r]
                    if len(np.unique(_y_tr)) < 2:
                        oos_probs[test_idx_r] = float(_y_tr[0])
                        continue
                        
                    cv_clf.fit(X_train_sub, _y_tr, sample_weight=train_sw_r)
                    
                    # Mapear predicciones de vuelta al offset original en oos_probs
                    agent_probs = cv_clf.predict_proba(X_test_sub)[:, 1]
                    oos_probs[test_idx_r] = agent_probs

        nan_mask = np.isnan(oos_probs)
        if nan_mask.any():
            _uncovered = int(nan_mask.sum())
            _uncov_pct = _uncovered / len(oos_probs) * 100
            
            if hmm_semantic_xgb is not None:
                _unique_regimes = np.unique([str(x) for x in hmm_semantic_xgb if pd.notna(x)])
                _uncov_by_regime = {
                    r: int((nan_mask & (hmm_semantic_xgb == r)).sum())
                    for r in _unique_regimes
                }
                logger.warning(
                    "[CPCV-COVERAGE] {} muestras ({:.1f}%) sin cobertura CPCV -> prob=0.50. "
                    "Por régimen: {}. Si >10% investigar splits insuficientes.",
                    _uncovered, _uncov_pct, _uncov_by_regime
                )
                print(f"[BUG-FIX-LOG 2026-06-05] [CPCV-COVERAGE] {_uncovered} muestras ({_uncov_pct:.1f}%) sin cobertura CPCV -> prob=0.50. Por régimen: {_uncov_by_regime}")
            else:
                logger.warning("[CPCV-COVERAGE] {} muestras ({:.1f}%) sin cobertura CPCV -> Fallback IS.", _uncovered, _uncov_pct)
                print(f"[BUG-FIX-LOG 2026-06-05] [CPCV-COVERAGE] {_uncovered} muestras ({_uncov_pct:.1f}%) sin cobertura CPCV -> Fallback IS.")

            try:
                # [FIX-P1-V4-7-OPCION-B] Entrenar un modelo In-Sample rapido para rellenar
                # las muestras sin cobertura en vez de asignar 0.50 (que asume probabilidad neutral).
                logger.info("  -> Entrenando XGBClassifier In-Sample (Fallback) para rellenar {} muestras", _uncovered)
                fallback_clf = xgb.XGBClassifier(
                    objective="binary:logistic", tree_method="hist", device="cpu", n_jobs=4,
                    n_estimators=50, max_depth=3, random_state=int(_os_meta.environ.get('LUNA_SEED', 42)), verbosity=0  # [FIX-RANDOM-STATE-02b]
                )
                
                is_train_idx = np.where(~nan_mask)[0]
                is_test_idx  = np.where(nan_mask)[0]
                if len(is_train_idx) > 100:
                    _f = list(X_xgb_df.columns)
                    
                    _is_sw = _sw_xgb[is_train_idx]
                    if _is_sw.sum() > 0:
                        _is_sw = _is_sw / _is_sw.mean()
                        
                    fallback_clf.fit(
                        X_xgb_df.iloc[is_train_idx][_f].values, 
                        y_xgb[is_train_idx], 
                        sample_weight=_is_sw
                    )
                    
                    nan_indices = np.where(nan_mask)[0]
                    X_fallback = X_xgb_df.iloc[nan_indices][_f].values
                    fallback_probs = fallback_clf.predict_proba(X_fallback)[:, 1]
                    oos_probs[nan_mask] = fallback_probs
                    logger.success("  -> Relleno IS exitoso (mean_prob={:.3f})", float(fallback_probs.mean()))
                    print(f"[BUGFIX 2026-06-02] Corregido formato string en Metalabeler Fallback IS (mean_prob={float(fallback_probs.mean()):.3f})")  # debug print trace
                else:
                    logger.warning("  -> Insuficientes datos IS para Fallback. Usando 0.50")
                    oos_probs[nan_mask] = 0.50
            except Exception as e:
                logger.error("  -> Fallo en Fallback IS: {}. Usando 0.50", str(e))
                oos_probs[nan_mask] = 0.50

        logger.info(f"Probs Base OOS generadas: {(~nan_mask).sum()}/{n} via CPCV")

        # [MEM-FIX-01] Materializar X_seq ahora — post-CPCV — para minimizar contención RAM
        # durante los fits intensivos del loop anterior. Peak ~250 MB solo desde este punto.
        logger.info("[MEM-FIX-01] Construyendo X_seq post-CPCV (~250 MB peak)...")
        X_seq, _ = self._build_sequences(df_labeled, feature_cols_available, y_all)
        logger.info(f"[MEM-FIX-01] X_seq: {X_seq.shape} — listo para split temporal")

        # 5. Extraer HMM one-hot context (P4-0-2) — el RF aprenderá por régimen
        hmm_dummies = [c for c in df_labeled.columns if c.startswith("HMM_OH_")]
        if hmm_dummies:
            hmm_onehot_all = df_labeled[hmm_dummies].values[valid_idx].astype(float)
            logger.info(f"Contexto HMM disponible: {len(hmm_dummies)} estados — {hmm_dummies}")
        else:
            hmm_onehot_all = None
            logger.warning("HMM_OH_* columns no encontradas — entrenando sin contexto de régimen")

        # 6. Split temporal con embargo (SOP R3)
        # [FIX-SPLIT-ML-01] Hacer ratio de split temporal train/val configurable
        try:
            from config.settings import cfg as _cfg_ml
            _val_ratio = float(_cfg_ml.metalabeler.val_split_ratio)
            _train_ratio = 1.0 - _val_ratio
            print(f"[FIX-SPLIT-ML-01] Split temporal cargado de config: train_ratio={_train_ratio:.2f} (val_ratio={_val_ratio:.2f})")  # debug
        except Exception as _e_ml_sp:
            _train_ratio = 0.80
            print(f"[FIX-SPLIT-ML-01] WARN: No se pudo leer metalabeler.val_split_ratio ({_e_ml_sp}). Usando fallback train_ratio={_train_ratio:.2f}")  # debug
        
        # [FIX-EMBARGO-ROBUST] Expand split validation temporal embargo to 144 hours (TBM vertical barrier 96H + sequence length 48H)
        ML_EMBARGO_H = 144
        split_idx = int(n * _train_ratio)
        val_start = min(split_idx + ML_EMBARGO_H, n)
        print(f"[FIX-EMBARGO-ROBUST] Expanded split validation temporal embargo to {ML_EMBARGO_H} hours (was {EMBARGO_H} hours). split_idx={split_idx}, val_start={val_start}")  # debug
        logger.info("[FIX-EMBARGO-ROBUST] Expanded split validation temporal embargo to {} hours (was {} hours). split_idx={}, val_start={}", ML_EMBARGO_H, EMBARGO_H, split_idx, val_start)
        
        timestamps_all = df_labeled.index[valid_idx]
        ts_train = timestamps_all[:split_idx]
        
        X_seq_train, X_seq_val = X_seq[:split_idx], X_seq[val_start:]
        xgb_p_train, xgb_p_val = oos_probs[:split_idx], oos_probs[val_start:]
        y_train, y_val = y[:split_idx], y[val_start:]
        hmm_train = hmm_onehot_all[:split_idx] if hmm_onehot_all is not None else None
        hmm_val = hmm_onehot_all[val_start:] if hmm_onehot_all is not None else None

        logger.info(f"Split: train={len(y_train)}, embargo={val_start-split_idx}H, val={len(y_val)}")

        # 7. Entrenar MetaLabelerV2 con contexto de régimen HMM
        n_features = X_seq.shape[2]
        model = MetaLabelerV2(input_dim=n_features)
        model.train(X_seq_train, xgb_p_train, y_train,
                    hmm_regime=hmm_train,
                    train_extractor=True, lstm_epochs=epochs, lstm_lr=lstm_lr,
                    timestamps=ts_train)

        # 8. Evaluación OOS
        if len(y_val) > 0:
            preds_val = model.predict(X_seq_val, xgb_p_val, hmm_regime=hmm_val)
            probs_val = model.predict_proba(X_seq_val, xgb_p_val, hmm_regime=hmm_val)
            wr_val = np.mean(preds_val == y_val)
            pos_val = np.mean(preds_val)
            logger.info(f"Validación OOS: WR={wr_val:.2%}, Predictions LONG={pos_val:.2%}")
        else:
            probs_val = np.array([])
            wr_val = 0.0

        # 8b. META-AUTO-01 (2026-03-23): EV-sweep sobre validación para encontrar
        # el threshold óptimo del MetaLabeler (análogo al EV-sweep de XGBoost).
        # Resultado escrito en calibrator_signature.json, que signal_filter.py
        # ya lee en L196-205 — si el campo existe, sustituye el hardcode 0.40.
        #
        # Formula EV = sum(ret_i × pos_i) - cost / n_trades  donde pos_i ∈ {0,1}
        # Barremos thresholds [0.30, 0.73) en pasos de 0.01.
        # Para que el threshold sea valido: EV>0 y n_trades >= n_baseline*0.20
        # (al menos 20% de trades del baseline t=0.30, para no colapsar la base)
        _optimal_meta_thresh = 0.30  # fallback si sweep falla (FASE C: adaptado de 0.40 a 0.30)
        _meta_calib_source   = "fallback_default"
        META_THRESH_MIN      = 0.30
        META_THRESH_MAX      = 0.73
        META_THRESH_STEP     = 0.01
        META_MIN_DENSITY_PCT = 0.20  # min 20% de trades del baseline

        # [FIX-CPCV-EV-SWEEP] META-AUTO-01 rewritten to run over the entire out-of-sample cross-validation probabilities (oos_probs)
        # generated by CPCV, avoiding local overfitting on a narrow validation slice.
        _close_rets_cpcv = close_rets.values[valid_idx]
        if len(oos_probs) > 0 and len(_close_rets_cpcv) == len(oos_probs):
            try:
                _thresholds = np.arange(META_THRESH_MIN, META_THRESH_MAX, META_THRESH_STEP)
                _n_baseline  = int((oos_probs >= 0.30).sum())
                _min_n       = max(5, int(_n_baseline * META_MIN_DENSITY_PCT))

                _best_ev   = -np.inf
                _best_thresh_m = None

                for _t in _thresholds:
                    _mask = oos_probs >= _t
                    _n_t  = int(_mask.sum())
                    if _n_t < _min_n:
                        continue
                    # FASE C: restar el COST_PCT para reflejar costo real de spread/fees
                    _ev = float(np.mean(_close_rets_cpcv[_mask] - COST_PCT))
                    if _ev > 0 and _ev > _best_ev:
                        _best_ev = _ev
                        _best_thresh_m = float(round(_t, 2))

                if _best_thresh_m is not None:
                    _optimal_meta_thresh = _best_thresh_m
                    _meta_calib_source   = "ev_sweep_cpcv"
                    print(f"[FIX-CPCV-EV-SWEEP] CPCV EV-Sweep optimal threshold found: {_optimal_meta_thresh:.2f} | EV={_best_ev:.5f}")  # debug
                    logger.success(
                        "[FIX-CPCV-EV-SWEEP] MetaLabeler CPCV threshold optimo={:.2f} | EV={:.5f} "
                        "(sweep {:.2f}→{:.2f} paso={:.2f})",
                        _optimal_meta_thresh, _best_ev,
                        META_THRESH_MIN, META_THRESH_MAX, META_THRESH_STEP,
                    )
                    print(f"[BUG-FIX-LOG 2026-06-05] [FIX-CPCV-EV-SWEEP] MetaLabeler CPCV threshold optimo={_optimal_meta_thresh:.2f} | EV={_best_ev:.5f} (sweep {META_THRESH_MIN:.2f}→{META_THRESH_MAX:.2f} paso={META_THRESH_STEP:.2f})")
                else:
                    print(f"[FIX-CPCV-EV-SWEEP] CPCV EV-sweep yielded no positive EV threshold, using fallback: {_optimal_meta_thresh:.2f}")  # debug
                    logger.warning(
                        "[FIX-CPCV-EV-SWEEP] CPCV EV-sweep sin resultado positivo — usando fallback={:.2f}.",
                        _optimal_meta_thresh,
                    )
                    print(f"[BUG-FIX-LOG 2026-06-05] [FIX-CPCV-EV-SWEEP] CPCV EV-sweep sin resultado positivo — usando fallback={_optimal_meta_thresh:.2f}.")
            except Exception as _e_meta:
                print(f"[FIX-CPCV-EV-SWEEP] CPCV EV-sweep crashed: {_e_meta}, using fallback: {_optimal_meta_thresh:.2f}")  # debug
                logger.warning("[FIX-CPCV-EV-SWEEP] CPCV EV-sweep fallido ({}) — usando fallback={:.2f}",
                               _e_meta, _optimal_meta_thresh)
        else:
            print(f"[FIX-CPCV-EV-SWEEP] CPCV EV-sweep skipped due to empty or misaligned arrays, using fallback: {_optimal_meta_thresh:.2f}")  # debug
            logger.warning("[FIX-CPCV-EV-SWEEP] Sin datos de CPCV suficientes — usando fallback={:.2f}",
                           _optimal_meta_thresh)

        # [P2-META-01] Umbral dinámico IS-calibrado (doc §5 Propuesta 4, 2026-05-06)
        # Cuando meta_v2_threshold_mode='dynamic_is', el floor del threshold se calcula
        # como max(floor_abs, IS_base_rate × (1 + edge_pct)) usando solo datos IS.
        # Esto evita que el EV-sweep seleccione thresholds permisivos cuando la base rate
        # de validation cae (régimen adverso), lo que causa pass-through masivo del MetaLabeler.
        try:
            from config.settings import cfg as _cfg_dynmeta
            _thresh_mode   = str(str(_cfg_dynmeta.metalabeler.meta_v2_threshold_mode)).lower()
            _edge_pct_dyn  = float(_cfg_dynmeta.metalabeler.meta_v2_dynamic_edge_pct)
            _floor_abs_dyn = float(_cfg_dynmeta.metalabeler.meta_v2_min_prob)
        except Exception:
            _thresh_mode = 'fixed'
            _edge_pct_dyn, _floor_abs_dyn = 0.05, 0.38

        if _thresh_mode == 'dynamic_is':
            # IS base rate = fracción de positivos en el training set
            _is_base_rate = float(y_train.mean()) if len(y_train) > 0 else 0.50
            _dyn_floor = max(_floor_abs_dyn, _is_base_rate * (1.0 + _edge_pct_dyn))
            _dyn_floor = min(_dyn_floor, 0.65)  # cap para no matar todas las señales

            # [FIX P2-META-01/CEIL] Techo dinámico basado en P90 de predicciones IS val.
            # El EV-sweep IS puede seleccionar thresholds (ej. 0.634) que son alcanzables
            # en validation IS (donde el modelo está más calibrado) pero NO en OOS (donde
            # las probabilidades se comprimen hacia 0.5). El P90 de probs_val es un proxy
            # conservador del máximo alcanzable en OOS, garantizando que ~10% de señales
            # pasen el gate MetaLabeler incluso en ventanas de baja confianza.
            if len(probs_val) > 10:
                _p90_val      = float(np.percentile(probs_val, 90))
                _dyn_ceiling  = max(_dyn_floor + 0.02, min(_p90_val, 0.65))
            else:
                _dyn_ceiling  = 0.65  # fallback conservador

            if _optimal_meta_thresh < _dyn_floor:
                logger.info(
                    "[P2-META-01] dynamic_is: EV-sweep thresh={:.3f} < IS-floor={:.3f} "
                    "(IS_base_rate={:.3f}, edge={:.0%}, floor_abs={:.2f}). "
                    "Subiendo threshold al floor.",
                    _optimal_meta_thresh, _dyn_floor,
                    _is_base_rate, _edge_pct_dyn, _floor_abs_dyn,
                )
                _optimal_meta_thresh = _dyn_floor
                _meta_calib_source   = f"dynamic_is_floor_{_dyn_floor:.3f}"
            elif _optimal_meta_thresh > _dyn_ceiling:
                logger.warning(
                    "[P2-META-01/CEIL] dynamic_is: EV-sweep thresh={:.3f} > P90_IS={:.3f} "
                    "(ceiling={:.3f}) — capando para asegurar pass-through OOS. "
                    "Sin este cap, 0 trades pasarían el gate MetaLabeler en OOS.",
                    _optimal_meta_thresh, _p90_val, _dyn_ceiling,
                )
                _optimal_meta_thresh = _dyn_ceiling
                _meta_calib_source   = f"dynamic_is_ceil_{_dyn_ceiling:.3f}"
            else:
                logger.info(
                    "[P2-META-01] dynamic_is: EV-sweep thresh={:.3f} en rango "
                    "[floor={:.3f}, ceil={:.3f}]. OK.",
                    _optimal_meta_thresh, _dyn_floor, _dyn_ceiling,
                )

        # Escribir calibrator_signature.json con optimal_meta_threshold
        model_dir = self.root / "data" / "models"   # definido aqui para META-AUTO-01 y save()
        _calib_sig_path = model_dir / f"calibrator_{self.direction}_signature.json"
        try:
            _calib_sig_existing = {}
            if _calib_sig_path.exists():
                import json as _json_cs
                _calib_sig_existing = _json_cs.loads(_calib_sig_path.read_text(encoding="utf-8"))
        except Exception:
            _calib_sig_existing = {}
        _calib_sig_existing["optimal_meta_threshold"] = _optimal_meta_thresh
        _calib_sig_existing["meta_calib_source"]      = _meta_calib_source
        
        # [P2-BUGFIX] Eliminar umbrales por régimen basados en ROC-AUC del calibrador isotónico.
        # Estos umbrales maximizan métricas estadísticas pero destruyen la rentabilidad (Sharpe negativo)
        # al anular el EV-Sweep threshold global en signal_filter.py.
        if "optimal_meta_threshold_per_regime" in _calib_sig_existing:
            del _calib_sig_existing["optimal_meta_threshold_per_regime"]
            
        _calib_sig_path.write_text(json.dumps(_calib_sig_existing, indent=2), encoding="utf-8")
        logger.info("[META-AUTO-01] calibrator_signature.json actualizado: optimal_meta_threshold={:.2f} [{}]",
                    _optimal_meta_thresh, _meta_calib_source)

        # 8c. Guardar modelo
        # FIX seq_features (2026-03-09): inyectar la lista exacta de features en el modelo
        # antes de save(), para que se persista en metalabeler_v2_config.json.
        model._seq_features = feature_cols_available
        model.save(model_dir, direction_mode=self.direction)

        metrics = {
            "val_win_rate": float(wr_val),
            "n_train": len(y_train),
            "n_val": len(y_val),
            "lstm_hidden": LSTM_HIDDEN,
            "rf_n_estimators": RF_N_ESTIMATORS,
            "cpcv_splits": len(cpcv_splits),
            "arch": "MetaLabelerV2 (LSTM-extractor + RF-arbitro)",
        }
        logger.info(f"MetaLabelerV2 entrenado. Métricas: {metrics}")

        # 9. Escribir metalabeler_signature.json unificado para el generador de reportes.
        #    Este archivo es leído por generate_validation_report.py para:
        #    a) detectar si se usó V1 (BiLSTM) o V2 (LSTM→RF) via campo "version"
        #    b) construir el nombre del archivo de reporte con la arquitectura correcta
        # FIX seq_features (2026-03-09): leer best_lstm_loss del objeto model._pretrain_lstm
        # El atributo se guarda en self.extractor dentro del model, no en el trainer.
        _best_lstm_loss = getattr(model, 'best_lstm_loss',
                           getattr(model.extractor, 'best_lstm_loss', 0))
        signature = {
            "version": "v2",
            "arch": "LSTM-extractor + RF-arbitro",
            "lstm_hidden": LSTM_HIDDEN,
            "rf_n_estimators": RF_N_ESTIMATORS,
            "val_loss": float(_best_lstm_loss),
            # val_loss del LSTM pre-training (aproximación; no es binary-cross-entropy del pipeline completo)
            "n_features_seq": int(X_seq.shape[2]),
            "seq_len": SEQ_LEN,
            "nas_params": {
                "hidden_size": LSTM_HIDDEN,
                "num_layers": 1,
                "dropout": 0.5,
                "lr": lstm_lr,
            },
            "oos_win_rate": float(wr_val),
            "cpcv_splits": len(cpcv_splits),
            # FIX seq_features: guardar en signature para diagnóstico rápido
            "seq_features": feature_cols_available,
        }
        sig_path = model_dir / f"metalabeler_{self.direction}_signature.json"
        sig_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        logger.info(f"metalabeler_{self.direction}_signature.json (v2) escrito en {sig_path}")

        return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, default="long", choices=["long", "short"])
    args = parser.parse_args()

    import os as _os
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    _log_dir = ROOT / "logs"
    _log_dir.mkdir(exist_ok=True)
    _ts_ml  = _dt.now().strftime("%Y%m%d_%H%M%S")
    _rid_ml = _os.environ.get("LUNA_RUN_ID", "")
    _lname_ml = f"train_metalabeler_{args.direction}_{_ts_ml}_{_rid_ml}.log" if _rid_ml else f"train_metalabeler_{args.direction}_{_ts_ml}.log"
    logger.add(_log_dir / _lname_ml, rotation="50 MB", level="DEBUG")

    try:
        trainer = MetaLabelerV2Trainer(direction=args.direction)
        results = trainer.run()
        print(results)  # debug
        if not results:
            sys.exit(1)
        sys.exit(0)
    except Exception as e:
        import traceback
        logger.error(f"[FATAL UNCAUGHT] Script crashed at main level: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
