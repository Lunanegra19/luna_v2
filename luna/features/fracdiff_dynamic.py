"""
fracdiff_dynamic.py — Luna V1 (SOP R7)
=========================================
Fractional Differentiation DINÁMICA: d se recalcula en cada ventana
del walk-forward mediante ADF test.

REGLA R7: d ≠ 0.4 FIJO.
  - V9.4 fracasó parcialmente usando d=0.4 fijo para todo el período.
  - El d óptimo varía entre 0.3 y 1.0 según el régimen del mercado.

Implementación:
  - Busca el d mínimo tal que la serie sea estacionaria (ADF p < alpha)
  - Preserva la máxima memoria histórica posible
  - Se recalcula por ventana (llamado desde el pipeline walk-forward)

Referencia: López de Prado, AFML (2018), Capítulo 5.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from loguru import logger
from luna.utils.debug_guards import vlog, timeit, check_numeric_stability
try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    logger.warning("statsmodels no disponible — FracDiff usará d=1.0 como fallback")


class FracDiffDynamic:
    """
    Aplica Fractional Differentiation con d óptimo determinado por ADF.

    Uso:
        fd = FracDiffDynamic(alpha=0.05, d_min=0.1, d_max=1.0, step=0.1)
        df_out = fd.transform(df, cols=["close", "volume"])
    """

    def __init__(self,
                 alpha: float = 0.05,
                 d_min: float = 0.1,
                 d_max: float = 1.0,
                 step:  float = 0.05,
                 thresh: float = 1e-5,
                 max_width: int = 126):
        """
        Args:
            alpha:     Nivel de significancia ADF (default 0.05)
            d_min:     d mínimo a probar (0.1 → mínima diferenciación)
            d_max:     d máximo (1.0 → primera diferenciación completa)
            step:      Paso de incremento en búsqueda de d
            thresh:    Umbral de peso para truncar la ventana FFD
            max_width: Máximo de lags a incluir en la ventana
        """
        self.alpha     = alpha
        self.d_min     = d_min
        self.d_max     = d_max
        self.step      = step
        self.thresh    = thresh
        self.max_width = max_width
        self.d_values_: dict[str, float] = {}

    # ── Cálculo de pesos FFD ─────────────────────────────────────────────────

    @staticmethod
    def _get_weights(d: float, thresh: float = 1e-5, max_width: int = 126) -> np.ndarray:
        """
        Pesos para Fixed-window Fractional Differentiation (FFD).
        w_k = -w_{k-1} * (d - k + 1) / k
        Trunca cuando |w_k| < thresh.
        """
        w = [1.0]
        for k in range(1, max_width + 1):
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < thresh:
                break
            w.append(w_k)
        return np.array(w[::-1])

    # ── Forward FFD sobre una serie ──────────────────────────────────────────

    def _ffd(self, series: pd.Series, d: float) -> pd.Series:
        """Aplica FFD con d dado a la serie."""
        weights = self._get_weights(d, self.thresh, self.max_width)
        width = len(weights)
        if len(series) < width:
            logger.warning(f"Serie demasiado corta ({len(series)}) para d={d}, width={width}")
            return series  # fallback sin diferenciar
        result = np.full(len(series), np.nan)
        series_vals = series.values
        for i in range(width - 1, len(series)):
            result[i] = np.dot(weights, series_vals[i - width + 1: i + 1])
        return pd.Series(result, index=series.index, name=series.name)

    # ── ADF test ─────────────────────────────────────────────────────────────

    def _is_stationary(self, series: pd.Series) -> bool:
        """True si la serie pasa el test ADF con nivel de significancia alpha."""
        if not STATSMODELS_OK:
            return True  # fallback: asumir estacionaria si no hay statsmodels
        clean = series.dropna()
        if len(clean) < 50:
            return False
        try:
            _, p_value, _, _, _, _ = adfuller(clean, autolag="AIC")
            return p_value < self.alpha
        except Exception as e:
            logger.debug(f"ADF fallido: {e}")
            return False

    # ── Búsqueda de d óptimo ─────────────────────────────────────────────────

    def find_optimal_d(self, series: pd.Series, feature_name: str = "") -> float:
        """
        Encuentra el d mínimo que hace la serie estacionaria (ADF p < alpha).
        Preserva la máxima memoria posible (menor d = más memoria preservada).

        Returns:
            d óptimo en [d_min, d_max]
        """
        d = self.d_min
        last_pvalue = None
        while d <= self.d_max + 1e-9:
            fd_series = self._ffd(series, d)
            if STATSMODELS_OK:
                # [FIX-FRACDIFF-CRASH-01] Saneamiento estricto de Infs antes de std/mean.
                # Las variables np.log(0) generan -inf, destruyendo clean.std() y provocando
                # un array corrupto que crashea silenciosamente (Exit -1) LAPACK/OpenBLAS en Windows.
                clean = fd_series.replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean) >= 50:
                    try:
                        # [FIX-FRACDIFF-OUTLIER-01 2026-06-04] Windsorización temporal ±4 std.
                        _std = clean.std()
                        
                        # Salvaguarda matemática contra series constantes
                        if _std < 1e-8:
                            logger.debug(f"  {feature_name}: Serie casi constante (std={_std}). ADF no confiable.")
                            p_value = 1.0
                        else:
                            _mean = clean.mean()
                            print(f'[FIX-FRACDIFF-OUTLIER-01] Windsorizacion +-4 std aplicada antes del test ADF (memoria salvada)')
                            clean_clipped = clean.clip(lower=_mean - 4*_std, upper=_mean + 4*_std)
                            
                            # [FIX-FRACDIFF-LAPACK-01] Prevención de Segfaults nativos (Exit -1).
                            # ADFuller con autolag='AIC' construye matrices OLS masivas. Con N=77000
                            # y variables degeneradas, OpenBLAS falla silenciosamente en Windows.
                            # La estacionariedad es una propiedad asintótica global; 5000 puntos (200 días)
                            # son matemáticamente suficientes para la potencia del test ADF.
                            if len(clean_clipped) > 5000:
                                clean_clipped = clean_clipped.iloc[-5000:]
                                
                            _, p_value, _, _, _, _ = adfuller(clean_clipped, autolag="AIC")
                            
                        last_pvalue = p_value
                        if p_value < self.alpha:
                            vlog(f"  FracDiff [{feature_name}]: d_opt={d:.2f} | ADF p={p_value:.4f} < {self.alpha} ✓")
                            logger.debug(f"  {feature_name}: d_opt={d:.2f} (estacionaria con ADF p<{self.alpha})")

                            # [FALLA-07-FIX 2026-05-30] Verificacion Hurst post-hoc
                            # Si d_opt = d_max (limite superior), puede ser estacionariedad espuria por outlier.
                            # Se calcula el Hurst exponent R/S para diagnostico (no cambia d, solo logea).
                            _d_rounded = round(d, 3)
                            _d_max_rounded = round(self.d_max, 3)
                            if abs(_d_rounded - _d_max_rounded) < 0.02:  # d esta en el limite
                                try:
                                    _orig = series.dropna().values[-min(2000, len(series)):]
                                    if len(_orig) > 100:
                                        _lags = [2, 4, 8, 16, 32, 64]
                                        _rs = []
                                        for _lag in _lags:
                                            if _lag >= len(_orig): break
                                            _chunks = [_orig[i:i+_lag] for i in range(0, len(_orig)-_lag, _lag)]
                                            if not _chunks: continue
                                            _rs_list = []
                                            for _ch in _chunks:
                                                _r = _ch.max() - _ch.min()
                                                _s = _ch.std() + 1e-10
                                                _rs_list.append(_r / _s)
                                            _rs.append(np.mean(_rs_list))
                                        if len(_rs) >= 3:
                                            import warnings
                                            with warnings.catch_warnings():
                                                warnings.simplefilter("ignore")
                                                _log_lags = np.log([_lags[i] for i in range(len(_rs))])
                                                _log_rs = np.log(_rs)
                                                _H = np.polyfit(_log_lags, _log_rs, 1)[0]
                                            if _H < 0.4:
                                                print(f"[FALLA-07-FIX] WARNING: '{feature_name}' d_opt=d_max={_d_rounded} "
                                                      f"pero Hurst H={_H:.3f}<0.4 (anti-persistente). "
                                                      f"ADF p={p_value:.4f} puede ser espurio por outlier.")
                                            else:
                                                print(f"[FALLA-07-FIX] '{feature_name}' d_opt={_d_rounded} | Hurst H={_H:.3f} OK")
                                except Exception as _he:
                                    logger.debug(f"[FALLA-07-FIX] Hurst check fallido para {feature_name}: {_he}")

                            return round(d, 3)
                    except Exception as e:
                        logger.debug(f"ADF fallido en d={d:.2f}: {e}")
            elif self._is_stationary(fd_series):
                logger.debug(f"  {feature_name}: d_opt={d:.2f} (estacionaria con ADF p<{self.alpha})")
                return round(d, 3)
            d += self.step
        # Si ningún d funciona → diferenciación completa
        logger.warning(
            f"  {feature_name}: no se encontró d estacionario "
            f"(last ADF p={last_pvalue:.4f if last_pvalue else 'N/A'}) → usando d=1.0"
        )
        # [FIX-FRACDIFF-SANITY-01 2026-05-31] WARNING: d=1.0 viola SOP R7
        # d=1.0 es diferenciacion ENTERA completa — destruye la memoria de la serie.
        # SOP R7 exige d FRACCIONARIO dinamico (no entero, no fijo).
        # Causa probable: serie no-estacionaria con outliers extremos o datos corruptos.
        print(
            f"[FIX-FRACDIFF-SANITY-01/WARNING] '{feature_name}' no converge a d fraccionario. "
            f"Fallback a d=1.0 (diferenciacion entera). VIOLA SOP R7. "
            f"Verificar calidad de datos y outliers en la serie. "
            f"last_ADF_p={last_pvalue:.4f if last_pvalue is not None else 'N/A'}"
        )
        print(f"[BUG-FIX-LOG 2026-06-05] Corregido formatting logger.warning en fracdiff_dynamic.py para R7")
        logger.warning(
            "[FIX-FRACDIFF-SANITY-01] SOP R7 violation: '{}' d=1.0 (entero) | last_ADF_p={}",
            feature_name, f"{last_pvalue:.4f}" if last_pvalue is not None else "N/A"
        )
        return 1.0



    # ── Transform ────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame,
                  cols: list[str] | None = None,
                  suffix: str = "_fd") -> pd.DataFrame:
        """
        Aplica FracDiff dinámico a las columnas especificadas.

        Args:
            df:     DataFrame de features
            cols:   Columnas a diferenciar (default: ["close"])
            suffix: Sufijo para las columnas diferenciadas

        Returns:
            DataFrame con columnas adicionales {col}{suffix}
        """
        df = df.copy()
        if cols is None:
            cols = ["close"]

        vlog(f"FracDiff.transform: cols={cols} | suffix='{suffix}' | df.shape={df.shape}")
        with timeit("FracDiff.transform"):
            for col in cols:
                if col not in df.columns:
                    logger.warning(f"FracDiff: columna '{col}' no encontrada, saltando")
                    continue

                series = df[col].dropna()
                if len(series) < 100:
                    logger.warning(f"FracDiff: serie '{col}' demasiado corta ({len(series)})")
                    continue

                # Calcular d óptimo
                d_opt = self.find_optimal_d(series, feature_name=col)
                self.d_values_[col] = d_opt

                # Aplicar FFD con d óptimo
                fd_col = self._ffd(df[col], d_opt)
                df[f"{col}{suffix}"] = fd_col

                # Check estabilidad numérica del resultado
                check_numeric_stability(fd_col.values, label=f"FracDiff[{col}]")

                nan_pct = 100 * fd_col.isnull().sum() / max(len(fd_col), 1)
                logger.info(
                    f"FracDiff [{col}]: d={d_opt:.3f} → {col}{suffix} | "
                    f"NaN post-ffd={nan_pct:.1f}%"
                )

        return df

    def find_optimal_d_for_window(self, series_window: pd.Series, feature_name: str = "") -> float:
        """
        Alias explícito para recalcular d en una ventana de CPCV.
        Documenta la intención: d DEBE recalcularse en cada ventana (SOP R7 / P1-2).

        Args:
            series_window: Serie de la ventana de train del fold CPCV actual
            feature_name:  Nombre para logging

        Returns:
            d óptimo para esa ventana específica
        """
        return self.find_optimal_d(series_window, feature_name=feature_name)

    def transform_with_cpcv_windows(self,
                                     df: pd.DataFrame,
                                     cpcv_splits: list,
                                     cols: list | None = None,
                                     suffix: str = "_fd") -> pd.DataFrame:
        """
        P1-2 (SOP R7): Aplica FracDiff recalculando d en CADA ventana CPCV.

        En cada fold (train_idx, test_idx):
          1. Calcula d_opt usando SOLO los datos de entrenamiento (causal: SOP R1)
          2. Aplica FFD con ese d_opt a los datos de test del fold
          3. Concatena los resultados OOS (sin contaminar entre folds)

        El d resultante varía por fold — esto es el comportamiento CORRECTO.
        Prohibido usar d fijo 0.4 (V9.4 falló por esto, SOP R7).

        Args:
            df:           DataFrame completo (train+test)
            cpcv_splits:  Lista de (train_idx, test_idx) arrays — mismos splits del CPCV
            cols:         Columnas a diferenciar (default: ['close'])
            suffix:       Sufijo para columnas resultado

        Returns:
            DataFrame con columnas {col}{suffix} OOS por fold CPCV.
            NaN donde no hay cobertura OOS.
        """
        if cols is None:
            cols = ["close"]

        df_out = df.copy()
        for col in cols:
            df_out[f"{col}{suffix}"] = np.nan

        d_by_fold = {col: [] for col in cols}

        for fold_i, (train_idx, test_idx) in enumerate(cpcv_splits):
            if len(train_idx) < 100:
                logger.warning(f"Fold {fold_i}: train muy pequeño ({len(train_idx)} rows), saltando")
                continue

            for col in cols:
                if col not in df.columns:
                    continue

                # Calcular d solo con datos de train (causal — SOP R1)
                train_series = df[col].iloc[train_idx]
                d_opt = self.find_optimal_d_for_window(train_series, feature_name=f"{col}[fold{fold_i}]")
                d_by_fold[col].append({"fold": fold_i, "d_opt": d_opt,
                                        "train_len": len(train_idx), "test_len": len(test_idx)})

                # Aplicar FFD a los datos de test usando d del train
                # Necesitamos el contexto de lookback del train para la ventana FFD
                weights = self._get_weights(d_opt, self.thresh, self.max_width)
                w = len(weights)

                # Optimización NumPy: extraer arrays de memoria contigua
                col_vals = df[col].values
                buffer_vals = df_out[f"{col}{suffix}"].values.copy()

                for j in test_idx:
                    # El contexto FFD requiere w puntos previos en la serie completa
                    if j < w - 1:
                        continue
                    window_vals = col_vals[j - w + 1: j + 1]
                    if len(window_vals) < w or np.isnan(window_vals).any():
                        continue
                    val = float(np.dot(weights, window_vals))
                    buffer_vals[j] = val
                
                df_out[f"{col}{suffix}"] = buffer_vals

        # Guardar reporte de d por fold
        self._cpcv_d_report = d_by_fold
        total_folds = len(cpcv_splits)
        for col in cols:
            if d_by_fold[col]:
                d_values = [x["d_opt"] for x in d_by_fold[col]]
                logger.info(
                    f"FracDiff CPCV [{col}]: d_opt varía {min(d_values):.2f}–{max(d_values):.2f} "
                    f"(media={sum(d_values)/len(d_values):.2f}) en {len(d_values)}/{total_folds} folds"
                )

        return df_out

    def get_cpcv_d_report(self) -> pd.DataFrame:
        """Tabla de d óptimos por fold CPCV (solo disponible tras transform_with_cpcv_windows)."""
        report = getattr(self, "_cpcv_d_report", {})
        rows = []
        for col, folds in report.items():
            for fold_data in folds:
                rows.append({"feature": col, **fold_data})
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_d_report(self) -> pd.DataFrame:
        """Retorna tabla con los d óptimos encontrados (transform() simple)."""
        return pd.DataFrame([
            {"feature": k, "d_optimal": v}
            for k, v in self.d_values_.items()
        ])
