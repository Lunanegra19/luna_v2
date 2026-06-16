"""
ood_feature_guard.py
====================
Luna V2 — Detector Genérico de Features Degeneradas en OOS

Detecta automáticamente features cuya distribución colapsa en datos OOS
(Out-Of-Sample), previniendo predicciones estáticas del modelo sin
hardcodear nombres específicos de features.

PROBLEMA QUE RESUELVE
---------------------
Features como KMeans_Tribe_ID o Master_Causal_Signal son calculadas
sobre el training set y propagadas al OOS via ffill. Si el gap temporal
entre train y OOS es mayor que el límite de ffill, o si el modelo de
clustering no se reentrena, estas features degeneran a un valor constante.

Cuando XGBoost/LGBM recibe un input con varianza ~0 en una feature
importante, devuelve la misma probabilidad para todas las filas OOS
(ej: xgb_prob = 0.8076 constante para 2425 filas consecutivas).

CRITERIOS DE DETECCIÓN (genéricos, sin nombres hardcodeados)
-------------------------------------------------------------
1. CONSTANT:  > `max_constant_pct` de filas con el mismo valor
2. LOW_STD:   std_oos < `min_std_ratio` * std_train  (colapso relativo)
3. LOW_UNIQUE: < `min_unique_values` valores únicos en OOS
   (solo se aplica a features con > `min_unique_in_train` en training)

USO
---
    from luna.utils.ood_feature_guard import OOSFeatureGuard

    guard = OOSFeatureGuard()
    valid_features, report = guard.filter(
        X_train=df_train[feature_cols],
        X_oos=df_val[feature_cols],
        context="XGBoost/Bull"
    )
    # valid_features: lista filtrada, sin features degeneradas
    # report: lista de OODFeatureReport con diagnóstico por feature

THRESHOLDS (ajustables en settings.yaml bajo 'ood_guard:')
----------------------------------------------------------
    max_constant_pct:    0.95   # >95% mismo valor → bloqueada
    min_std_ratio:       0.05   # std_oos < 5% de std_train → bloqueada
    min_unique_values:   3      # <3 valores únicos en OOS → warning/block
    min_unique_in_train: 5      # aplicar min_unique solo si train tiene ≥5
    block_on_low_unique: True   # si False, solo warn (no bloquear)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


# ── Defaults configurables ────────────────────────────────────────────────────
_DEFAULT_MAX_CONSTANT_PCT    = 0.95   # >95% mismo valor → BLOQUEADA
_DEFAULT_MIN_STD_RATIO       = 0.05   # std_oos < 5% std_train → BLOQUEADA
_DEFAULT_MIN_UNIQUE_VALUES   = 3      # <3 unique en OOS → candidata a bloqueo
_DEFAULT_MIN_UNIQUE_IN_TRAIN = 5      # solo aplica min_unique si train tiene >=5
_DEFAULT_BLOCK_ON_LOW_UNIQUE = True   # bloquear o solo advertir por low unique


@dataclass
class OODFeatureReport:
    """Resultado del análisis de una feature individual."""
    feature: str
    blocked: bool
    reason: str              # "CONSTANT" | "LOW_STD" | "LOW_UNIQUE" | "OK"
    severity: str            # "ERROR" | "WARNING" | "OK"

    # Métricas del training
    train_std: float
    train_nunique: int

    # Métricas del OOS
    oos_std: float
    oos_nunique: int
    oos_constant_pct: float  # fracción de filas con el valor más común

    # Ratio derivado
    std_ratio: float         # oos_std / train_std (0=colapsada, 1=igual varianza)

    def __str__(self) -> str:
        return (
            f"[{self.severity}] {self.feature}: {self.reason} | "
            f"std_ratio={self.std_ratio:.4f} "
            f"(train={self.train_std:.4f}, oos={self.oos_std:.4f}) | "
            f"oos_unique={self.oos_nunique} | "
            f"oos_const={self.oos_constant_pct:.1%}"
        )


class OOSFeatureGuard:
    """
    Detector genérico de features con distribución degenerada en OOS.

    Compara la distribución de cada feature entre el training set y el
    OOS set, y bloquea aquellas que colapsan a valores constantes o
    con varianza extremadamente baja.

    No hardcodea nombres de features — funciona sobre cualquier columna.
    """

    def __init__(
        self,
        max_constant_pct: float    = _DEFAULT_MAX_CONSTANT_PCT,
        min_std_ratio: float       = _DEFAULT_MIN_STD_RATIO,
        min_unique_values: int     = _DEFAULT_MIN_UNIQUE_VALUES,
        min_unique_in_train: int   = _DEFAULT_MIN_UNIQUE_IN_TRAIN,
        block_on_low_unique: bool  = _DEFAULT_BLOCK_ON_LOW_UNIQUE,
    ):
        """
        Args:
            max_constant_pct:    Fracción máxima de filas con el mismo valor en OOS.
                                 Por encima de este umbral → feature BLOQUEADA (CONSTANT).
            min_std_ratio:       Ratio mínimo entre std_oos y std_train.
                                 Por debajo → feature BLOQUEADA (LOW_STD).
            min_unique_values:   Mínimo de valores únicos exigidos en OOS.
                                 Si no se cumple → BLOQUEADA o WARNING (LOW_UNIQUE).
            min_unique_in_train: Aplica min_unique solo si la feature tiene ≥N
                                 valores únicos en training (evita falsos positivos
                                 en features binarias legítimas como indicadores 0/1).
            block_on_low_unique: Si True, bloquear en LOW_UNIQUE. Si False, solo warn.
        """
        self.max_constant_pct   = max_constant_pct
        self.min_std_ratio      = min_std_ratio
        self.min_unique_values  = min_unique_values
        self.min_unique_in_train = min_unique_in_train
        self.block_on_low_unique = block_on_low_unique

        self.structural_features: list = []  # [FIX-CRIT-03] features exentas del OOD Guard

        # Intentar leer thresholds de settings.yaml (override de instancia)
        self._load_from_settings()

    def _load_from_settings(self) -> None:
        """Override de thresholds desde settings.yaml si existe 'ood_guard:' section."""
        try:
            from config.settings import cfg as _cfg
            _ood = int(_cfg.ood_guard)
            if _ood is not None:
                self.max_constant_pct   = float(getattr(_ood, 'max_constant_pct',   self.max_constant_pct))
                self.min_std_ratio      = float(getattr(_ood, 'min_std_ratio',      self.min_std_ratio))
                self.min_unique_values  = int(  getattr(_ood, 'min_unique_values',  self.min_unique_values))
                self.min_unique_in_train = int( getattr(_ood, 'min_unique_in_train',self.min_unique_in_train))
                self.block_on_low_unique = bool(getattr(_ood, 'block_on_low_unique',self.block_on_low_unique))
                logger.debug("[OOSFeatureGuard] Thresholds cargados desde settings.yaml [ood_guard]")
                self.structural_features = list(getattr(_ood, 'structural_features', []))
                if self.structural_features:
                    print(f"[FIX-CRIT-03] OOD Guard structural_features exentas: {self.structural_features}")  # RULE[fixbugsprints.md]
                    logger.info(
                        f"[FIX-CRIT-03] OOD Guard: {len(self.structural_features)} features "
                        f"estructurales exentas del bloqueo CONSTANT/LOW_STD: {self.structural_features}"
                    )
        except Exception:
            pass  # usar defaults — no es crítico

    def _analyze_feature(
        self,
        feat: str,
        train_series: pd.Series,
        oos_series: pd.Series,
    ) -> OODFeatureReport:
        """Analiza una feature individual. Retorna OODFeatureReport."""

        # ── Métricas training ──────────────────────────────────────────────
        train_clean = train_series.dropna()
        train_std     = float(train_clean.std()) if len(train_clean) > 1 else 0.0
        train_nunique = int(train_clean.nunique())

        # ── Métricas OOS ───────────────────────────────────────────────────
        oos_clean = oos_series.dropna()
        oos_std     = float(oos_clean.std()) if len(oos_clean) > 1 else 0.0
        oos_nunique = int(oos_clean.nunique())

        # % de filas con el valor más frecuente (moda)
        if len(oos_clean) > 0:
            oos_constant_pct = float(oos_clean.value_counts(normalize=True).iloc[0])
        else:
            oos_constant_pct = 1.0

        # Ratio std_oos / std_train (protegido contra división por cero)
        if train_std > 1e-10:
            std_ratio = oos_std / train_std
        elif oos_std < 1e-10:
            std_ratio = 1.0   # ambas constantes → ratio neutro (no bloquear)
        else:
            std_ratio = 0.0   # train constante pero OOS no → anómalo, dejar pasar

        # ── Evaluación de criterios ────────────────────────────────────────

        # [FIX-CRIT-03] Exencion para features estructurales (e.g. HMM_Regime)
        # Una feature constante en OOS puede ser informacion legitima del regimen actual,
        # no degeneracion. HMM_Regime=4 constante en bull trend puro es correcto y util.
        if feat in getattr(self, 'structural_features', []):
            print(f"[FIX-CRIT-03] Feature estructural '{feat}': exenta OOD (constante en OOS = info de regimen legitima)")  # RULE[fixbugsprints.md]
            logger.info(
                f"[FIX-CRIT-03] OOD Guard: '{feat}' exenta -- structural feature "
                f"(oos_unique={oos_nunique}, oos_const={oos_constant_pct:.1%})"
            )
            return OODFeatureReport(
                feature=feat, blocked=False, reason="STRUCTURAL_EXEMPT", severity="OK",
                train_std=train_std, train_nunique=train_nunique,
                oos_std=oos_std, oos_nunique=oos_nunique,
                oos_constant_pct=oos_constant_pct, std_ratio=std_ratio,
            )

        # Criterio 1: CONSTANT — demasiado % con el mismo valor en OOS
        if oos_constant_pct >= self.max_constant_pct:
            return OODFeatureReport(
                feature=feat, blocked=True, reason="CONSTANT", severity="ERROR",
                train_std=train_std, train_nunique=train_nunique,
                oos_std=oos_std, oos_nunique=oos_nunique,
                oos_constant_pct=oos_constant_pct, std_ratio=std_ratio,
            )

        # Criterio 2: LOW_STD — varianza OOS colapsa vs training
        # Solo aplica si la feature tiene varianza real en training
        if train_std > 1e-6 and std_ratio < self.min_std_ratio:
            return OODFeatureReport(
                feature=feat, blocked=True, reason="LOW_STD", severity="ERROR",
                train_std=train_std, train_nunique=train_nunique,
                oos_std=oos_std, oos_nunique=oos_nunique,
                oos_constant_pct=oos_constant_pct, std_ratio=std_ratio,
            )

        # Criterio 3: LOW_UNIQUE — pocos valores únicos en OOS vs training
        # Solo aplica si training tiene suficiente diversidad (evita falsos
        # positivos en features binarias legítimas: HMM_Regime, indicadores 0/1)
        if (train_nunique >= self.min_unique_in_train
                and oos_nunique < self.min_unique_values):
            blocked = self.block_on_low_unique
            severity = "WARNING" if not blocked else "ERROR"
            return OODFeatureReport(
                feature=feat,
                blocked=blocked,
                reason="LOW_UNIQUE",
                severity=severity,
                train_std=train_std, train_nunique=train_nunique,
                oos_std=oos_std, oos_nunique=oos_nunique,
                oos_constant_pct=oos_constant_pct, std_ratio=std_ratio,
            )

        # Sin problemas
        return OODFeatureReport(
            feature=feat, blocked=False, reason="OK", severity="OK",
            train_std=train_std, train_nunique=train_nunique,
            oos_std=oos_std, oos_nunique=oos_nunique,
            oos_constant_pct=oos_constant_pct, std_ratio=std_ratio,
        )

    def filter(
        self,
        X_train: pd.DataFrame,
        X_oos: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        context: str = "model",
    ) -> tuple[list[str], list[OODFeatureReport]]:
        """
        Filtra el feature set eliminando columnas con distribución degenerada en OOS.

        Args:
            X_train:      DataFrame del training set (filas × features).
            X_oos:        DataFrame del OOS/validation set (filas × features).
            feature_cols: Lista de features a evaluar. Si None, usa la intersección
                          de columnas de X_train y X_oos.
            context:      Etiqueta de contexto para logs (ej: "XGBoost/Bull").

        Returns:
            (valid_features, reports)
            - valid_features: lista de features sin problemas OOD
            - reports: lista de OODFeatureReport (una por feature evaluada)
        """
        if feature_cols is None:
            feature_cols = list(set(X_train.columns) & set(X_oos.columns))

        # Filtrar features que existen en ambos datasets
        available = [f for f in feature_cols if f in X_train.columns and f in X_oos.columns]
        missing   = [f for f in feature_cols if f not in available]
        if missing:
            logger.debug(
                f"[OOSFeatureGuard/{context}] {len(missing)} features no disponibles en ambos datasets (se mantienen): {missing}"
            )

        reports: list[OODFeatureReport] = []
        blocked_features: list[str]    = []
        valid_features: list[str]      = []

        for feat in feature_cols:
            if feat not in available:
                # Feature no evaluable → mantener (evitar eliminar HMM_Regime si está ausente en val)
                valid_features.append(feat)
                continue

            report = self._analyze_feature(
                feat=feat,
                train_series=X_train[feat],
                oos_series=X_oos[feat],
            )
            reports.append(report)

            if report.blocked:
                blocked_features.append(feat)
            else:
                valid_features.append(feat)

        # ── Logging de resultados ──────────────────────────────────────────
        n_blocked = len(blocked_features)
        n_total   = len(feature_cols)

        if n_blocked == 0:
            logger.info(
                f"[OOSFeatureGuard/{context}] ✓ Todas las {n_total} features pasan el control OOD."
            )
        else:
            logger.warning(
                f"[OOSFeatureGuard/{context}] ⚠ {n_blocked}/{n_total} features BLOQUEADAS por distribución degenerada en OOS:"
            )
            for r in reports:
                if r.blocked:
                    logger.warning(f"  → {r}")

            # Log de features OK con métricas relevantes a nivel DEBUG (no spam)
            ok_reports = [r for r in reports if not r.blocked and r.reason != "OK"]
            for r in ok_reports:
                logger.debug(f"  ~ {r}")

        # Resumen de features bloqueadas en un solo log de INFO para trazabilidad
        if n_blocked > 0:
            logger.warning(
                f"[OOSFeatureGuard/{context}] Features excluidas del training: {blocked_features} | "
                "Causa: distribución OOS degenerada detectada automáticamente. "
                "Ver logs anteriores para detalles por feature."
            )

        return valid_features, reports


# ── Función de conveniencia para uso directo ──────────────────────────────────

def filter_ood_features(
    X_train: pd.DataFrame,
    X_oos: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    context: str = "model",
    max_constant_pct: float   = _DEFAULT_MAX_CONSTANT_PCT,
    min_std_ratio: float      = _DEFAULT_MIN_STD_RATIO,
    min_unique_values: int    = _DEFAULT_MIN_UNIQUE_VALUES,
    min_unique_in_train: int  = _DEFAULT_MIN_UNIQUE_IN_TRAIN,
    block_on_low_unique: bool = _DEFAULT_BLOCK_ON_LOW_UNIQUE,
) -> tuple[list[str], list[OODFeatureReport]]:
    """
    Función de conveniencia que instancia OOSFeatureGuard y filtra features.

    Útil para llamadas únicas sin necesidad de instanciar la clase.
    Lee thresholds de settings.yaml si están disponibles (override de args).

    Returns:
        (valid_features, reports)
    """
    guard = OOSFeatureGuard(
        max_constant_pct=max_constant_pct,
        min_std_ratio=min_std_ratio,
        min_unique_values=min_unique_values,
        min_unique_in_train=min_unique_in_train,
        block_on_low_unique=block_on_low_unique,
    )
    return guard.filter(X_train=X_train, X_oos=X_oos,
                        feature_cols=feature_cols, context=context)
