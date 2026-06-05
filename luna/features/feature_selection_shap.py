"""
feature_selection_shap.py — Luna V1 (P1-10 / MEJORA 3)
=========================================================
SHAPFeatureSelector: selector complementario al SFI-CPCV que captura
interacciones sinérgicas entre features (lo que SFI no puede detectar).

Ventajas sobre SFI:
  - TreeSHAP: O(N*T) — igual de rápido que SFI para XGBoost
  - Captura contribuciones en contexto del modelo completo (no aisladas)
  - detect_synergistic_features: identifica features inútiles en solitario
    pero valiosas en combinación (SFI las descartaría incorrectamente)
  - SHAP drift monitoring: compara distribuciones SHAP entre períodos

SOP Aplicado:
  - R1 (Causalidad): SHAP se calcula sobre X_train únicamente
  - R7: complementa (no reemplaza) FracDiff dinámico
  - R5: selección final combina SFI + SHAP ranking

Referencia: Lundberg & Lee (2017), Lundberg et al. (2020) TreeSHAP.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    logger.warning("shap no instalado — SHAPFeatureSelector usará importancia de ganancias como fallback")


class SHAPFeatureSelector:
    """
    P1-10: Selector de features basado en SHAP Values.

    Complementa el SFI-CPCV evaluando features en el contexto del modelo
    completo entrenado. Detecta synergias que SFI ignora.

    Uso:
        selector = SHAPFeatureSelector(n_top_features=25, min_abs_shap=0.001)
        selected = selector.select(xgb_model, X_train, feature_names)
    """

    def __init__(self, n_top_features: int = 25, min_abs_shap: float = 1e-4):
        """
        Args:
            n_top_features: Máximo de features a seleccionar
            min_abs_shap:   SHAP medio absoluto mínimo para inclusión
        """
        self.n_top_features = n_top_features
        self.min_abs_shap = min_abs_shap
        self.shap_df_: pd.DataFrame | None = None
        self.synergy_scores_: pd.Series | None = None

    # ── Selector principal ──────────────────────────────────────────────────

    def select(self,
               model_xgb,
               X_train: pd.DataFrame | np.ndarray,
               feature_names: list[str] | None = None) -> list[str]:
        """
        Selecciona features por importancia SHAP (mean |SHAP|).

        Args:
            model_xgb:     Modelo XGBoost ya entrenado (XGBClassifier o Booster)
            X_train:       Datos de entrenamiento (NO de validación — SOP R1)
            feature_names: Nombres de features (si X_train es np.ndarray)

        Returns:
            Lista de features seleccionadas ordenadas por importancia SHAP
        """
        if not SHAP_OK:
            return self._fallback_gain_importance(model_xgb, X_train, feature_names)

        if isinstance(X_train, pd.DataFrame):
            feature_names = list(X_train.columns)
            X_arr = X_train.values
        else:
            X_arr = X_train
            feature_names = feature_names or [f"f{i}" for i in range(X_arr.shape[1])]

        logger.info(f"SHAPFeatureSelector: calculando SHAP values para {X_arr.shape[0]} muestras × {len(feature_names)} features...")

        try:
            explainer = shap.TreeExplainer(model_xgb)
            # Usar muestra representativa para velocidad (máx 5000 muestras)
            n_sample = min(len(X_arr), 5000)
            # SEED-FIX: usar RNG con seed fijo para reproducibilidad del subsampling SHAP
            _rng_shap = np.random.default_rng(42)
            idx_sample = _rng_shap.choice(len(X_arr), n_sample, replace=False)
            shap_values = explainer.shap_values(X_arr[idx_sample])

            # Importancia = mean(|SHAP|) por feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            self.shap_df_ = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_shap': mean_abs_shap,
            }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

            # Filtrar por umbral mínimo y top N
            selected_df = self.shap_df_[self.shap_df_['mean_abs_shap'] >= self.min_abs_shap]
            selected_df = selected_df.head(self.n_top_features)

            selected = selected_df['feature'].tolist()
            logger.info(
                f"SHAPFeatureSelector: {len(selected)}/{len(feature_names)} features seleccionadas "
                f"(top SHAP={self.shap_df_.iloc[0]['mean_abs_shap']:.6f}, "
                f"umbral={self.min_abs_shap})"
            )
            return selected

        except Exception as e:
            logger.warning(f"SHAPFeatureSelector: error en SHAP ({e}), usando fallback gain importance")
            return self._fallback_gain_importance(model_xgb, X_train, feature_names)

    # ── Detección de sinergias ──────────────────────────────────────────────

    def detect_synergistic_features(self,
                                    model_xgb,
                                    X_train: pd.DataFrame | np.ndarray,
                                    feature_names: list[str] | None = None,
                                    max_samples: int = 1000) -> pd.Series:
        """
        Detecta features con SHAP individual bajo pero interacciones altas.
        Estas son exactamente las que SFI descartaría incorrectamente.

        Args:
            model_xgb:     Modelo XGBoost entrenado
            X_train:       Datos de entrenamiento
            feature_names: Nombres de features
            max_samples:   Máximo muestras para SHAP interactions (costoso)

        Returns:
            Serie con synergy_score por feature (ratio interacción/individual)
        """
        if not SHAP_OK:
            logger.warning("shap no disponible — detect_synergistic_features requiere shap instalado")
            return pd.Series(dtype=float)

        if isinstance(X_train, pd.DataFrame):
            feature_names = list(X_train.columns)
            X_arr = X_train.values
        else:
            X_arr = X_train
            feature_names = feature_names or [f"f{i}" for i in range(X_arr.shape[1])]

        # SHAP interaction es costoso — limitar muestras
        n_sample = min(len(X_arr), max_samples)
        # SEED-FIX: usar RNG con seed fijo para reproducibilidad del subsampling SHAP interactions
        _rng_shap_inter = np.random.default_rng(42)
        idx_sample = _rng_shap_inter.choice(len(X_arr), n_sample, replace=False)
        X_sub = X_arr[idx_sample]

        logger.info(f"Calculando SHAP interaction values ({n_sample} muestras × {len(feature_names)} features)...")
        try:
            explainer = shap.TreeExplainer(model_xgb)
            shap_interaction = explainer.shap_interaction_values(X_sub)

            # diag[i] = importancia individual de feature i
            # off_diag[i] = suma de interacciones de feature i con todas las demás
            diag = np.abs(np.diagonal(shap_interaction, axis1=1, axis2=2)).mean(axis=0)
            off_diag = (np.abs(shap_interaction).sum(axis=(0, 1)) - diag) / 2

            synergy_score = off_diag / (diag + 1e-8)
            self.synergy_scores_ = pd.Series(synergy_score, index=feature_names).sort_values(ascending=False)

            # Features sinérgicas = baja importancia individual pero alta interacción
            top_synergic = self.synergy_scores_.head(10)
            logger.info(f"Top 5 features sinérgicas (ratio interacción/individual):\n{top_synergic.head(5).to_string()}")

            return self.synergy_scores_

        except Exception as e:
            logger.warning(f"SHAP interaction fallido: {e}")
            return pd.Series(dtype=float)

    # ── SHAP Drift Monitor ──────────────────────────────────────────────────

    def compute_shap_drift(self,
                           model_xgb,
                           X_ref: pd.DataFrame | np.ndarray,
                           X_cur: pd.DataFrame | np.ndarray,
                           feature_names: list[str] | None = None) -> pd.DataFrame:
        """
        Compara distribuciones SHAP entre períodos (drift monitoring post-deploy).

        Detecta features cuya contribución cambió significativamente respecto
        al período de referencia (training). Un cambio > 2σ indica drift.

        Args:
            model_xgb: Modelo XGBoost
            X_ref:     Datos de referencia (training)
            X_cur:     Datos actuales (últimas N horas de producción)
            feature_names: Nombres de features

        Returns:
            DataFrame con columnas: feature, shap_ref_mean, shap_cur_mean, drift_z, drifted
        """
        if not SHAP_OK:
            return pd.DataFrame()

        if isinstance(X_ref, pd.DataFrame):
            feature_names = list(X_ref.columns)
            X_ref = X_ref.values
            X_cur = X_cur.values if hasattr(X_cur, 'values') else X_cur

        try:
            explainer = shap.TreeExplainer(model_xgb)
            shap_ref = explainer.shap_values(X_ref[:min(len(X_ref), 2000)])
            shap_cur = explainer.shap_values(X_cur[:min(len(X_cur), 2000)])

            ref_mean = np.mean(shap_ref, axis=0)
            cur_mean = np.mean(shap_cur, axis=0)
            ref_std = np.std(shap_ref, axis=0) + 1e-8

            drift_z = np.abs(cur_mean - ref_mean) / ref_std

            drift_df = pd.DataFrame({
                'feature': feature_names,
                'shap_ref_mean': ref_mean,
                'shap_cur_mean': cur_mean,
                'drift_z': drift_z,
                'drifted': drift_z > 2.0,  # > 2σ = drift significativo
            }).sort_values('drift_z', ascending=False)

            n_drifted = drift_df['drifted'].sum()
            if n_drifted > 0:
                logger.warning(
                    f"SHAP Drift: {n_drifted}/{len(feature_names)} features con drift significativo (>2σ):\n"
                    f"{drift_df[drift_df['drifted']]['feature'].tolist()}"
                )
            else:
                logger.info("SHAP Drift: sin drift significativo detectado.")

            return drift_df

        except Exception as e:
            logger.warning(f"SHAP drift computation fallido: {e}")
            return pd.DataFrame()

    # ── Combinación SFI + SHAP ──────────────────────────────────────────────

    def combine_with_sfi(self,
                         shap_selected: list[str],
                         sfi_selected: list[str],
                         priority: str = "union") -> list[str]:
        """
        Combina la selección SHAP con el ranking SFI existente.

        Args:
            shap_selected: Features seleccionadas por SHAP
            sfi_selected:  Features seleccionadas por SFI-CPCV
            priority:      'union' = unión de ambos conjuntos (recomendado)
                           'intersection' = solo features en ambos (más restrictivo)
                           'shap' = SHAP sobreescribe SFI

        Returns:
            Lista de features combinada
        """
        shap_set = set(shap_selected)
        sfi_set = set(sfi_selected)

        if priority == "union":
            # SHAP primero (mayor importancia), luego SFI únicos
            combined = list(shap_selected)
            for f in sfi_selected:
                if f not in shap_set:
                    combined.append(f)
            result = combined[:self.n_top_features]  # cap en n_top_features
        elif priority == "intersection":
            # Solo features que ambos métodos aprueban
            result = [f for f in shap_selected if f in sfi_set]
        else:  # 'shap'
            result = shap_selected

        logger.info(
            f"SFI+SHAP ({priority}): {len(shap_set)} SHAP + {len(sfi_set)} SFI → {len(result)} combinadas"
        )
        return result

    # ── Reporte ─────────────────────────────────────────────────────────────

    def get_shap_report(self) -> pd.DataFrame:
        """Retorna el ranking SHAP completo de la última llamada a select()."""
        if self.shap_df_ is None:
            return pd.DataFrame(columns=['feature', 'mean_abs_shap'])
        return self.shap_df_.copy()

    # ── Fallback ────────────────────────────────────────────────────────────

    def _fallback_gain_importance(self,
                                  model_xgb,
                                  X_train,
                                  feature_names: list[str] | None = None) -> list[str]:
        """Fallback: usa importancia de ganancia de XGBoost si SHAP no está disponible."""
        try:
            if hasattr(model_xgb, 'feature_importances_'):
                importances = model_xgb.feature_importances_
            elif hasattr(model_xgb, 'get_score'):
                score = model_xgb.get_score(importance_type='gain')
                if feature_names is None:
                    return list(score.keys())[:self.n_top_features]
                importances = np.array([score.get(f, 0) for f in feature_names])
            else:
                return feature_names[:self.n_top_features] if feature_names else []

            if feature_names is None:
                feature_names = [f"f{i}" for i in range(len(importances))]

            df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            df = df.sort_values('importance', ascending=False)
            selected = df.head(self.n_top_features)['feature'].tolist()
            logger.info(f"Fallback gain importance: {len(selected)} features seleccionadas")
            return selected

        except Exception as e:
            logger.error(f"Fallback gain importance fallido: {e}")
            return feature_names[:self.n_top_features] if feature_names else []
