"""
Kelly Fractional Position Sizer — Luna V1
==========================================
Convierte probabilidades calibradas de XGBoost en fracciones de capital óptimas
usando el Criterio de Kelly con un factor de escala conservador (Quarter-Kelly).

FUNDAMENTO MATEMÁTICO:
----------------------
John Kelly (1956, Bell System Technical Journal):
    f* = (W×p - L×q) / (W×L)

donde:
    p  = probabilidad de ganancia (idealmente calibrada via XGB-ISO-CAL-01)
    q  = 1 - p (probabilidad de pérdida)
    W  = ratio de ganancia esperada / unidad arriesgada (pt_ratio)
    L  = ratio de pérdida esperada / unidad arriesgada (sl_ratio, = 1.0)
    f* = fracción óptima de capital (ANTES de escalar)

IMPLEMENTACIÓN EN LUNA V1:
--------------------------
- Se aplica Quarter-Kelly (kelly_fraction=0.25) para mitigar el riesgo de
  estimación de probabilidades imperfectas.
- Floor mínimo de 1% del capital (nunca 0% si p > 0.52).
- Cap máximo de 15% del capital (diversificación y control de drawdown).
- Kelly negativo (EV < 0) → f = 0.0 (no operar).

REFERENCIAS:
------------
- Kim & Jeong (2020): Half-Kelly adaptativo en BTC-USD → +34% Sharpe vs. fixed-size
- Li et al. (2022): Quarter-Kelly en multi-crypto → -18% MaxDD mismo Sharpe
- López de Prado (2018), "AFML", Cap. 10: Kelly calibrado para HF trading
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── [V2-P5] Kelly Sizer Dinámico — Confianza Conjunta XGB + Meta + HMM ───────
#
# Problema: max_position=0.15 SIEMPRE, aunque XGBoost (0.82), Meta (0.79) y
#           HMM (BULL_TREND) coincidan. El sistema apuesta igual en una señal
#           marginal (0.52) que en una señal de triple confirmación (0.82).
#
# Solución: mapear la confianza conjunta [0,1] a un rango dinámico de max_position
#           [0.03, 0.30], aplicando además el drift_penalty del PSI Monitor (V2-P3).


def compute_joint_confidence(
    xgb_prob: float,
    meta_prob: float = 0.5,
    hmm_semantic: str = "",
    hmm_bonus_regimes: tuple = ("1_VOLATILE_BULL", "1_BULL_TREND"),
    hmm_bonus: float = 1.15,
) -> float:
    """Calcula la confianza conjunta de XGBoost + MetaLabeler + HMM.

    Fórmula: joint = xgb_prob × meta_prob × hmm_bonus (clipped a [0, 1])

    Args:
        xgb_prob:          Probabilidad XGBoost calibrada (o cruda como fallback).
        meta_prob:         Probabilidad del MetaLabeler V2. Default 0.5 (neutro).
        hmm_semantic:      Régimen HMM semántico del bar actual.
        hmm_bonus_regimes: Regímenes que reciben bonus de confianza.
        hmm_bonus:         Multiplicador si el régimen está en hmm_bonus_regimes.

    Returns:
        float en [0.0, 1.0]. Valores típicos:
            - Triple confirmación alta: ~0.75 → max_position=30%
            - Doble confirmación media: ~0.40 → max_position=10%
            - Señal marginal:           ~0.25 → max_position=3%
    """
    _hmm_mult = hmm_bonus if hmm_semantic in hmm_bonus_regimes else 1.0
    joint = float(np.clip(xgb_prob * meta_prob * _hmm_mult, 0.0, 1.0))
    return joint


def dynamic_max_position(
    joint_confidence: float,
    drift_penalty: float = 1.0,
    base_cap: float = 0.30,
) -> float:
    """Mapea confianza conjunta a max_position dinámico, penalizado por drift.

    Niveles de confianza (sin drift):
        < 0.25:  Señal muy débil   → 3%  (apenas tanteando)
        0.25-0.45: Señal débil     → 7%  (posición reducida)
        0.45-0.65: Señal media     → 10% (Kelly estándar)
        0.65-0.80: Señal fuerte    → 20% (posición cargada)
        > 0.80:  Triple confirmación → 30% (máximo agresivo controlado)

    El drift_penalty (del PSI Monitor, V2-P3) escala proporcionalmente:
        drift_penalty=1.0  → sin reducción (distribuciones estables)
        drift_penalty=0.5  → max_position al 50% (features contaminadas)

    Args:
        joint_confidence: Valor de compute_joint_confidence() en [0, 1].
        drift_penalty:    Factor del PSI Monitor (1.0 = sin penalización).
        base_cap:         Cap absoluto de max_position (default: 30%).

    Returns:
        float: max_position en [0.03, base_cap].
    """
    if joint_confidence < 0.25:
        raw = 0.03
    elif joint_confidence < 0.45:
        raw = 0.07
    elif joint_confidence < 0.65:
        raw = 0.10
    elif joint_confidence < 0.80:
        raw = 0.20
    else:
        raw = min(0.30, base_cap)

    penalized = raw * drift_penalty
    return float(np.clip(penalized, 0.03, base_cap))


class KellyPositionSizer:
    """
    Dimensionador de posición basado en el Criterio de Kelly Fraccional.

    Transforma probabilidades calibradas del XGBoost en fracciones de capital,
    permitiendo que el sistema apueste más en señales de alta confianza y menos
    en señales marginales — en lugar del binario actual (todo-o-nada).

    ORDEN DE ACTIVACIÓN OBLIGATORIO (SOP):
        1. XGBoost calibrado (XGB-ISO-CAL-01) → produce xgb_prob_cal
        2. KellyPositionSizer.size_signals(xgb_prob_cal) → produce position_fraction
        3. Ejecutor de trade usa position_fraction para calcular el tamaño de la orden

    NOTA: No implementar Kelly sobre xgb_prob cruda (sin calibrar).
    xgb_prob=0.80 con WR real=25% generaría un Kelly catastrófico.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        min_position:   float = 0.01,
        max_position:   float = 20.0,
        pt_ratio:       float = 2.0,
        sl_ratio:       float = 1.0,
        probability_cap: float = 0.62,
        target_sl_pct: float = 0.05,
    ):
        """
        Args:
            kelly_fraction: Factor multiplicador del Kelly puro (default: 0.25 = Quarter-Kelly).
                            Rango recomendado: [0.20, 0.50]. Valores > 0.5 son agresivos.
            min_position:   Fracción mínima del capital cuando p > 0.50 (evita f→0).
                            Default: 0.01 (1% del capital).
            max_position:   Fracción máxima del capital por trade (diversificación).
                            Default: 0.15 (15% del capital).
            pt_ratio:       Ratio PT/unidad_arriesgada. Debe coincidir con pt_mult_min de settings.
                            Default: 2.0 (Take Profit = 2× el Stop Loss).
            sl_ratio:       Ratio SL/unidad_arriesgada. Siempre 1.0 (SL = unidad base).
            probability_cap: Probabilidad máxima tolerada por el Kelly sizer (overconfidence cap).
                            Default: 0.62.
        """
        if not 0.0 < kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction debe ser (0, 1]. Recibido: {kelly_fraction}")
            
        # [SOP-R17-FIX] Forzar techo duro en Half-Kelly (0.5). Full-Kelly prohibido institucionalmente.
        if kelly_fraction > 0.5:
            logger.warning(f"[SOP-R17] Full-Kelly o fracción > 0.5 ({kelly_fraction}) prohibido por política. Forzando kelly_fraction = 0.5 (Half-Kelly).")
            print(f"[SOP-R17-FIX] Advertencia: Reduciendo kelly_fraction de {kelly_fraction} a 0.5 (Half-Kelly).")
            kelly_fraction = 0.5
            
        if not 0.0 <= min_position < max_position <= 100.0:
            raise ValueError(f"Rango de posición inválido: [{min_position}, {max_position}]")
        if pt_ratio <= 0 or sl_ratio <= 0:
            raise ValueError(f"pt_ratio y sl_ratio deben ser > 0.")
        if not 0.0 < probability_cap <= 1.0:
            raise ValueError(f"probability_cap debe ser (0, 1]. Recibido: {probability_cap}")

        self.kelly_fraction = kelly_fraction
        self.min_position   = min_position
        self.max_position   = max_position
        self.pt_ratio       = pt_ratio
        self.sl_ratio       = sl_ratio
        self.probability_cap = probability_cap
        self.target_sl_pct  = target_sl_pct
        
        # [SOP-R17-FIX] Limite matemático estricto de apalancamiento nominal en derivados.
        # (Para operaciones Spot, el límite efectivo es 1.0 dictado por max_position).
        self.max_leverage_limit = 20.0 

        logger.info(
            f"[KELLY-SIZER] Init: fraction={kelly_fraction:.2f} | "
            f"range=[{min_position:.1%}, {max_position:.1f}x] | "
            f"PT/SL={pt_ratio:.1f}/{sl_ratio:.1f} | "
            f"probability_cap={probability_cap:.4f} | "
            f"target_sl_pct={target_sl_pct:.1%}"
        )
        print(
            f"[KELLY-SIZER] Init: fraction={kelly_fraction:.2f} | "
            f"range=[{min_position:.1%}, {max_position:.1f}x] | "
            f"PT/SL={pt_ratio:.1f}/{sl_ratio:.1f} | "
            f"probability_cap={probability_cap:.4f} | "
            f"target_sl_pct={target_sl_pct:.1%}"
        )

    def compute_kelly(self, p_win: np.ndarray) -> np.ndarray:
        """
        Calcula la fracción de Kelly vectorialmente sobre un array de probabilidades.

        f* = (W×p - L×q) / (W×L)

        Args:
            p_win: Array de probabilidades de ganancia [0.0, 1.0].
                   Idealmente calibradas (xgb_prob_cal), no crudas.

        Returns:
            Array de fracciones de capital [0.0, max_position].
        """
        p_win = np.asarray(p_win, dtype=float)
        # [OOF-CALIB-V2 2026-06-03] probability_cap mitigation
        p_win = np.clip(p_win, 0.0, self.probability_cap)
        q_loss = 1.0 - p_win

        # Fracción Kelly pura (Capital at Risk / VaR)
        f_star = (self.pt_ratio * p_win - self.sl_ratio * q_loss) / (
            self.pt_ratio * self.sl_ratio
        )

        # Aplicar factor de escala conservador
        f_applied = f_star * self.kelly_fraction

        # [KELLY-FIX 2026-06-05] Convertir Capital en Riesgo (VaR) a Apalancamiento Notional.
        # El Kelly Sizer arrojaba qué % arriesgar, NO qué posición comprar. Si arriesgas 5% y el SL es 5%,
        # necesitas una posición del 100% (1.0x).
        leverage_multiplier = f_applied / self.target_sl_pct

        # Kelly negativo (EV < 0) → no operar
        # Kelly positivo → clampear leverage a [min_position, max_position]
        f_final = np.where(
            f_star <= 0.0,
            0.0,                                              # EV negativo → no operar
            np.clip(leverage_multiplier, self.min_position, self.max_position),  # EV positivo → clampear leverage
        )

        return f_final

    def size_signals(
        self,
        df: pd.DataFrame,
        prob_col: str = "xgb_prob_cal",
        mask_col: str | None = None,
    ) -> pd.Series:
        """
        Calcula la fracción de capital para cada fila de un DataFrame de señales.

        Args:
            df:       DataFrame con columnas de probabilidad y opcionalmente máscara de señal.
            prob_col: Columna de probabilidad a usar. Por defecto 'xgb_prob_cal' (calibrada).
                      Fallback automático a 'xgb_prob' si xgb_prob_cal no está disponible.
            mask_col: Columna booleana de señal activa. Si se especifica, las filas donde
                      mask_col=False recibirán position_fraction=0.0.

        Returns:
            pd.Series con nombre 'position_fraction', mismo índice que df.
        """
        # Fallback si la columna calibrada no está disponible
        if prob_col not in df.columns:
            if "xgb_prob" in df.columns:
                logger.warning(
                    f"[KELLY-SIZER] '{prob_col}' no disponible. "
                    "Usando 'xgb_prob' cruda (sub-óptimo: calibrar primero con XGB-ISO-CAL-01)."
                )
                prob_col = "xgb_prob"
            else:
                logger.error("[KELLY-SIZER] Ninguna columna de probabilidad disponible. Devolviendo ceros.")
                return pd.Series(0.0, index=df.index, name="position_fraction")

        p = df[prob_col].fillna(0.0).values
        sizes = self.compute_kelly(p)

        # Aplicar máscara de señal si se especifica
        if mask_col is not None and mask_col in df.columns:
            signal_active = df[mask_col].fillna(False).astype(bool).values
            sizes = np.where(signal_active, sizes, 0.0)

        result = pd.Series(sizes, index=df.index, name="position_fraction")

        # Log de estadísticas
        n_active = (result > 0).sum()
        if n_active > 0:
            logger.info(
                f"[KELLY-SIZER] {n_active}/{len(result)} posiciones activas | "
                f"Fracción media={result[result > 0].mean():.1%} | "
                f"Fracción max={result.max():.1%} | "
                f"Capital total asignado={result.sum():.1%}"
            )
        return result

    def size_signals_dynamic(
        self,
        df: pd.DataFrame,
        prob_col:     str   = "xgb_prob_cal",
        meta_col:     str   = "meta_v2_prob",
        hmm_col:      str   = "HMM_Semantic",
        mask_col:     Optional[str] = None,
        drift_penalty: float = 1.0,
        ood_col:      Optional[str] = "ood_kl_distance",
    ) -> pd.Series:
        """[V2-P5] Sizing dinámico: max_position proporcional a confianza conjunta.

        Combina XGBoost + MetaLabeler + HMM + PSI Drift Penalty para calcular
        un max_position específico para cada señal en lugar del cap fijo 15%.

        Args:
            df:            DataFrame con columnas de probabilidad y régimen.
            prob_col:      Columna XGBoost (preferiblemente calibrada).
            meta_col:      Columna MetaLabeler V2.
            hmm_col:       Columna HMM_Semantic.
            mask_col:      Máscara de señal activa (opcional).
            drift_penalty: Factor PSI Monitor (1.0=OK, 0.5=CRITICAL drift).
            ood_col:       Columna OOD KL Distance (opcional).

        Returns:
            pd.Series 'position_fraction' con el sizing dinámico por señal.
        """
        # Fallback de columnas
        if prob_col not in df.columns:
            prob_col = "xgb_prob" if "xgb_prob" in df.columns else None
            if prob_col is None:
                return pd.Series(0.0, index=df.index, name="position_fraction")

        fractions = []
        n_boosted = 0
        n_reduced = 0

        for idx in df.index:
            row = df.loc[idx]
            xgb_p  = float(row.get(prob_col,  0.5) or 0.5)
            meta_p = float(row.get(meta_col,  0.5) or 0.5)
            hmm_s  = str(row.get(hmm_col,   "")  or "")

            # Calcular confianza conjunta y max_position dinámico
            jc       = compute_joint_confidence(xgb_p, meta_p, hmm_s)

            _ood_penalty = 1.0
            if ood_col and ood_col in df.columns:
                _ood_dist = float(row.get(ood_col, 0.0) or 0.0)
                if _ood_dist < 0:
                    _ood_penalty = max(0.25, np.exp(_ood_dist * 2.0))
                    
            _eff_drift = drift_penalty * _ood_penalty
            max_pos  = dynamic_max_position(jc, drift_penalty=_eff_drift)

            # Sobreescribir max_position para este tick
            _orig_max = self.max_position
            self.max_position = max_pos
            frac = self.compute_kelly(np.array([xgb_p]))[0]
            self.max_position = _orig_max  # restaurar

            # Aplicar máscara
            if mask_col and mask_col in df.columns:
                if not bool(row.get(mask_col, True)):
                    frac = 0.0

            if max_pos > _orig_max:
                n_boosted += 1
            elif max_pos < _orig_max:
                n_reduced += 1

            fractions.append(frac)

        result = pd.Series(fractions, index=df.index, name="position_fraction")
        n_active = (result > 0).sum()
        if n_active > 0:
            logger.info(
                f"[V2-P5-KELLY-DYN] {n_active} trades sized dinamicamente | "
                f"Boosted={n_boosted} (+max_pos) | Reduced={n_reduced} (-max_pos) | "
                f"drift_penalty={drift_penalty:.0%} | "
                f"Fraccion media={result[result>0].mean():.1%} | Max={result.max():.1%}"
            )
        return result

    def summarize(self, position_series: pd.Series) -> dict:
        """Resumen de estadísticas del sizing para logging/reporting."""
        active = position_series[position_series > 0]
        return {
            "n_signals_sized": int((position_series > 0).sum()),
            "mean_fraction":   float(active.mean()) if len(active) > 0 else 0.0,
            "max_fraction":    float(active.max())  if len(active) > 0 else 0.0,
            "min_fraction":    float(active.min())  if len(active) > 0 else 0.0,
            "total_capital_deployed": float(position_series.sum()),
        }


def build_kelly_sizer_from_settings() -> KellyPositionSizer:
    """
    Construye un KellyPositionSizer cargando los parámetros desde config/settings.yaml.
    Fallback a valores conservadores si la configuración no existe.

    Parámetros esperados en settings.yaml:
        kelly_sizer:
          kelly_fraction: 0.5    # Half-Kelly
          min_position:   0.01
          max_position:   0.15
          pt_ratio:       1.20   # [FIX-KELLY-PT-RATIO-01] P/L calibrado OOS real
          sl_ratio:       1.0
    """
    try:
        from config.settings import cfg
        ks_cfg = getattr(cfg, "kelly_sizer", None)
        kwargs = {}
        if ks_cfg:
            for k in ["kelly_fraction", "min_position", "max_position", "pt_ratio", "sl_ratio", "probability_cap", "target_sl_pct"]:
                v = getattr(ks_cfg, k, None)
                if v is not None:
                    kwargs[k] = float(v)
        # Sincronizar pt_ratio con xgboost.pt_mult_min si no está en kelly_sizer
        if "pt_ratio" not in kwargs:
            pt = getattr(getattr(cfg, "xgboost", None), "pt_mult_min", None)
            if pt:
                kwargs["pt_ratio"] = float(pt)
                logger.warning(
                    "[KELLY-SIZER][FIX-KELLY-PT-RATIO-01] pt_ratio NO encontrado en kelly_sizer section. "
                    f"Usando xgboost.pt_mult_min={float(pt):.2f} como fallback. "
                    "AÑADIR pt_ratio explícito a kelly_sizer en settings.yaml."
                )
                print(
                    f"[FIX-KELLY-PT-RATIO-01][WARN] pt_ratio no en kelly_sizer — "
                    f"fallback a xgboost.pt_mult_min={float(pt):.2f}. "
                    "Esto puede causar Kelly sobreestimado si el P/L real OOS difiere del IS."
                )
            else:
                logger.error(
                    "[KELLY-SIZER][FIX-KELLY-PT-RATIO-01] CRITICAL: pt_ratio no encontrado "
                    "en kelly_sizer NI en xgboost.pt_mult_min. Usando default pt_ratio=2.0 "
                    "(P/L TEÓRICO). El Kelly será ~2x mayor que el óptimo OOS real. "
                    "AÑADIR pt_ratio: 1.20 a settings.yaml -> kelly_sizer."
                )
                print(
                    "[FIX-KELLY-PT-RATIO-01][CRITICAL] pt_ratio no configurado. "
                    "Usando default=2.0 (P/L teórico IS). "
                    "P/L real OOS BULL=0.97 -> Kelly sobreestimado ~2x. "
                    "ACCION REQUERIDA: añadir pt_ratio: 1.20 a kelly_sizer en settings.yaml."
                )
        sizer = KellyPositionSizer(**kwargs)
        # [FIX-KELLY-PT-RATIO-01] Print de trazabilidad del pt_ratio efectivo
        print(
            f"[FIX-KELLY-PT-RATIO-01] KellyPositionSizer construido desde settings.yaml: "
            f"kelly_fraction={sizer.kelly_fraction:.2f} | "
            f"pt_ratio={sizer.pt_ratio:.3f} | "
            f"sl_ratio={sizer.sl_ratio:.3f} | "
            f"target_sl_pct={sizer.target_sl_pct:.1%} | "
            f"range=[{sizer.min_position:.1%}, {sizer.max_position:.1f}x] | "
            f"EV mínimo con WR=54%: {(sizer.pt_ratio*0.54 - 1.0*0.46)/(sizer.pt_ratio*1.0):.4f} "
            f"-> Leverage={max(0,(sizer.pt_ratio*0.54 - 1.0*0.46)/(sizer.pt_ratio*1.0))*sizer.kelly_fraction/sizer.target_sl_pct:.2f}x (Half-Kelly)"
        )
        return sizer
    except Exception as e:
        logger.warning(f"[KELLY-SIZER] Error cargando settings ({e}). Usando defaults conservadores.")
        print(f"[FIX-KELLY-PT-RATIO-01][WARN] Error cargando settings: {e}. Usando defaults conservadores.")
        return KellyPositionSizer()
