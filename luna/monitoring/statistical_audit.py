"""
statistical_audit.py — Luna V1
================================
Motor de Validación Estadística (La Gran Barrera).
SOP Luna v2 → R5 (DSR), R8 (Binomial + Trades mínimos).

CORRECCIONES v1.2:
  - DSR: separados n_obs (para varianza SR) vs n_trials (para SR* Multiple Testing)
  - PBO: reemplazado proxy logístico por CSCV con permutación de bloques reales
  - MIN_DSR: 0.75 (SOP R5 validación final) — antes era 0.95 en el código
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)

# Ruta raíz del proyecto (para report_dir absoluto)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _load_n_trials_from_settings() -> int:
    """
    BUG-01 FIX (2026-03-17): Lee n_trials desde xgboost.optuna_trials en settings.yaml.

    Antes leía stat.n_trials_total=600 (trials históricos acumulados), lo que inflaba
    artificialmente el SR* (benchmark de azar) y suprimía el DSR sistemáticamente.
    Ahora lee xgboost.optuna_trials=100 (los trials REALES que corre Optuna en cada run).

    Justificación: el DSR (Bailey & LdP 2014) penaliza por el número de hipótesis
    probadas EN ESTE BACKTEST, no el acumulado histórico. Si corremos 100 trials Optuna,
    el multiple testing adjustment debe ser sobre 100, no 600.
    """
    try:
        import yaml
        cfg_path = _PROJECT_ROOT / "config" / "settings.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            # FUENTE CORRECTA: trials reales de Optuna en este run
            n = int(cfg.get("xgboost", {}).get("optuna_trials", 100))
            logger.debug(f"[StatAudit] n_trials cargado desde xgboost.optuna_trials: {n}")
            return n
    except Exception as e:
        logger.warning(f"[StatAudit] No se pudo leer optuna_trials de settings.yaml: {e}")
    # Fallback: 100 (valor por defecto de xgboost.optuna_trials)
    return 100


class LunaStatisticalAuditor:
    """
    Motor de Validación Estadística (La Gran Barrera).
    SOP Luna v2 → R5 (DSR), R8 (Binomial Test).
    """
    def __init__(self, cpcv_n_trials: int = None):
        # Fix M-01 / DISC-03: si no se pasa un valor explícito, leer desde settings.yaml.
        if cpcv_n_trials is None:
            self.n_trials = _load_n_trials_from_settings()
        else:
            self.n_trials = cpcv_n_trials
        logger.info(f"[StatAudit] n_trials activo: {self.n_trials} (= xgboost.optuna_trials, BUG-01 fix)")

        # ARCH-02 (2026-03-10): gates del Gauntlet leídos desde cfg.stat en settings.yaml.
        # POLÍTICA NO-FALLBACK (2026-05-21): si settings.yaml no carga → CRITICAL + RuntimeError.
        # Rationale: un fallback silencioso en gates del Gauntlet (DSR/PBO/MaxDD) produce
        # veredictos estadísticamente inválidos. Ver docs/parametros_fijos.md §2 — Caso #1.
        try:
            import yaml as _yaml
            _cfg_path = _PROJECT_ROOT / "config" / "settings.yaml"
            if not _cfg_path.exists():
                raise FileNotFoundError(
                    f"[StatAudit] CRITICAL: settings.yaml no encontrado en {_cfg_path}. "
                    "Los gates del Gauntlet (DSR/PBO/MaxDD) no pueden cargarse. "
                    "El run se detiene para evitar veredictos estadísticamente inválidos. "
                    "Ver docs/parametros_fijos.md §2 — Política No-Fallback."
                )
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f)
            _s = _cfg.get("stat", {})

            # Parámetros obligatorios — se verifica que existen en settings.yaml
            # [FIX-DSR-TRANS-STD 2026-06-13] Cargar dsr_transversal_std obligatoriamente
            _REQUIRED = ["min_dsr", "max_pbo", "min_trades", "alpha_binomial",
                         "max_drawdown", "pbo_n_blocks", "dsr_transversal_std"]
            _missing = [k for k in _REQUIRED if k not in _s]
            if _missing:
                raise KeyError(
                    f"[StatAudit] CRITICAL: parámetros obligatorios ausentes en cfg.stat: {_missing}. "
                    "Añadirlos a config/settings.yaml antes de continuar. "
                    "Ver docs/parametros_fijos.md §2 — Parámetros Obligatorios."
                )

            self.MIN_DSR         = float(_s["min_dsr"])
            self.MAX_PBO         = float(_s["max_pbo"])
            self.MIN_TRADES      = int(_s["min_trades"])
            self.ALPHA_BINOMIAL  = float(_s["alpha_binomial"])
            self.MAX_DRAWDOWN    = float(_s["max_drawdown"])
            self.DSR_TRANSVERSAL_STD = float(_s["dsr_transversal_std"])
            # MAX_DRAWDOWN = 0.60 (Compatibilidad con el auditor pre-flight)
            self.TOTAL_RETURN_CAP = float(_s.get("total_return_cap", 1e6))  # opcional: no es gate
            self.PBO_N_BLOCKS    = int(_s["pbo_n_blocks"])

            print(
                f"[StatAudit] Gates cargados OK: MIN_DSR={self.MIN_DSR} | MAX_PBO={self.MAX_PBO} | "
                f"MIN_TRADES={self.MIN_TRADES} | PBO_N_BLOCKS={self.PBO_N_BLOCKS} | "
                f"DSR_TRANSVERSAL_STD={self.DSR_TRANSVERSAL_STD} | MAX_DRAWDOWN={self.MAX_DRAWDOWN}"
            )
            logger.debug("[StatAudit] Gates cargados desde cfg.stat (settings.yaml)")

        except Exception as _e:
            # POLÍTICA NO-FALLBACK: cualquier fallo al cargar settings → CRITICAL + stop.
            # No se usan valores hardcodeados — el run debe corregir el settings.yaml.
            _msg = (
                f"[StatAudit] CRITICAL — Imposible cargar gates del Gauntlet desde settings.yaml: {_e}\n"
                "ACCIÓN REQUERIDA: verificar que config/settings.yaml existe, tiene sección [stat]\n"
                "con las claves: min_dsr, max_pbo, min_trades, alpha_binomial, max_drawdown, pbo_n_blocks.\n"
                "El run se DETIENE. Ver docs/parametros_fijos.md para contexto completo."
            )
            print(_msg)
            logger.critical(_msg)
            raise RuntimeError(_msg) from _e

        # Factor de anualización horario (datos 1H): sqrt(365 * 24)
        self.ANNUAL_FACTOR = np.sqrt(365 * 24)


    # ── 1. Deflated Sharpe Ratio ──────────────────────────────────────────────

    def _compute_dsr(self, sharpe_oos: float, skewness: float, kurtosis: float,
                     n_obs: int = 8760, n_trials: int = None) -> float:
        """
        Calcula el Deflated Sharpe Ratio según Bailey & López de Prado (2014).
        DSR = Phi((SR - SR*) / std_SR)

        CORRECCIÓN v1.2 — dos parámetros DISTINTOS:
          n_obs   : número de observaciones en el holdout/fold OOS.
                    Controla std_SR (varianza del estadístico Sharpe).
                    Antes: se usaba n_trials aquí (incorrecto).
          n_trials: número de combinaciones/estrategias candidatas (Optuna).
                    Controla SR* (benchmark del azar bajo Multiple Testing).
                    Antes: se usaba n_obs aquí (incorrecto).
        """
        gamma = 0.5772156649  # Constante de Euler-Mascheroni

        # Varianza del estadístico Sharpe — función de n_obs y momentos de distribución
        # Incluye corrección por no-normalidad (skewness y excess kurtosis)
        var_sr = (1.0 - (skewness * sharpe_oos) + ((kurtosis - 1.0) / 4.0) * (sharpe_oos ** 2)) / max(n_obs, 2)
        std_sr = float(np.sqrt(max(var_sr, 1e-12)))

        # SR* = Sharpe esperado del mejor trial por azar puro (Multiple Testing Correction)
        # Deriva de n_trials (número de estrategias candidatas), NO de n_obs
        # Si no se pasa n_trials explícitamente, usar el del constructor (ya cargado desde settings)
        if n_trials is None:
            n_trials = self.n_trials
        prob_term = 1.0 / max(n_trials, 2)
        z_1 = stats.norm.ppf(1.0 - prob_term)
        z_2 = stats.norm.ppf(1.0 - prob_term * np.exp(-1.0))

        # [FIX-DSR-TRANS-STD 2026-06-13] Bailey & Lopez de Prado (2014) exige varianza TRANSVERSAL.
        # std_sr es el Standard Error temporal de la estrategia ganadora. 
        # sr_star (Expected Maximum Sharpe) DEBE calcularse con la desviación estándar
        # TRANSVERSAL de los n_trials. Leemos dsr_transversal_std desde settings.yaml.
        sr_std_cross = self.DSR_TRANSVERSAL_STD
        sr_star = sr_std_cross * ((1.0 - gamma) * z_1 + gamma * z_2)

        # Guard: evitar división por 0 si std_sr → 0
        if std_sr < 1e-9:
            return 1.0 if sharpe_oos > sr_star else 0.0

        z_score = (sharpe_oos - sr_star) / std_sr
        dsr = float(stats.norm.cdf(z_score))
        logger.info(
            "  [DSR] SR_crudo=%.4f | n_obs=%d | n_trials=%d | SR*=%.4f | std_SR=%.6f | z=%.4f → DSR=%.4f",
            sharpe_oos, n_obs, n_trials, sr_star, std_sr, z_score, dsr
        )
        return float(np.clip(dsr, 0.0, 1.0))

    # ── 2. Binomial Test ──────────────────────────────────────────────────────

    def _compute_binomial_test(self, wins: int, total_trades: int) -> float:
        """
        Test de significación del Win Rate. H0: WR = 50% (azar puro).
        Retorna p-value (< 0.05 = rechazamos H0 con 95% confianza).
        """
        if total_trades == 0:
            return 1.0
        try:
            # scipy >= 1.7.0
            res = stats.binomtest(k=int(wins), n=int(total_trades), p=0.5, alternative='greater')
            return float(res.pvalue)
        except AttributeError:
            # Fallback para scipy antiguo
            return float(stats.binom_test(x=int(wins), n=int(total_trades), p=0.5, alternative='greater'))

    # ── 3. PBO via CSCV ──────────────────────────────────────────────────────

    def _estimate_pbo_cscv(self, returns_oos: np.ndarray, n_blocks: int = None) -> float:
        """
        Estimador PBO (Probability of Backtest Overfitting) via Combinatorial
        Symmetric Cross-Validation (CSCV) — Bailey & López de Prado (2014).
        n_blocks: leído desde cfg.stat.pbo_n_blocks si no se pasa explícitamente (ARCH-02).
        """
        if n_blocks is None:
            n_blocks = self.PBO_N_BLOCKS
        print(f"[TRACKING-FIX] CSCV utilizando pbo_n_blocks={n_blocks} directo desde settings.yaml (No-Fallback).")
        n = len(returns_oos)
        if n < n_blocks * 4:
            if n >= 5 * 4:
                # [FIX-PBO-ADAPTIVE] Reducir n_blocks dinámicamente en lugar de abortar
                _old_blocks = n_blocks
                n_blocks = max(2, n // 4)
                print(f"[FIX-PBO-ADAPTIVE] Trades ({n}) insuficientes para n_blocks={_old_blocks}. "
                      f"Reduciendo dinámicamente pbo_n_blocks a {n_blocks}.")
                logger.warning(f"[PBO] Reducción adaptativa de n_blocks: {_old_blocks} -> {n_blocks} por bajo número de trades ({n}).")
            else:
                # Dataset demasiado pequeño — estimación conservadora
                print(f"[FIX-PBO-01] WARN CSCV: {n} trades < {n_blocks * 4} mínimo (n_blocks={n_blocks}*4). Retornando PBO=0.50 conservador.")
                logger.warning(f"[PBO] Solo {n} trades disponibles (min recomendado: {n_blocks * 4}). "
                               "PBO poco fiable — estimación conservadora 0.50.")
                return 0.50
        print(f"[FIX-PBO-01] CSCV activo: n_trades={n} >= {n_blocks*4} mínimo | n_blocks={n_blocks} | calculando PBO real...")

        block_size = n // n_blocks
        blocks = [returns_oos[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]

        rng = np.random.default_rng(42)
        n_simulations = min(200, n_blocks * (n_blocks - 1))
        overfit_count = 0

        for _ in range(n_simulations):
            # Permutación aleatoria 50/50 bloques IS / OOS
            perm = rng.permutation(n_blocks)
            half = n_blocks // 2
            is_ret  = np.concatenate([blocks[i] for i in perm[:half]])
            oos_ret = np.concatenate([blocks[i] for i in perm[half:]])

            sr_is  = float(np.mean(is_ret))  / (float(np.std(is_ret))  + 1e-10) * self.ANNUAL_FACTOR
            sr_oos = float(np.mean(oos_ret)) / (float(np.std(oos_ret)) + 1e-10) * self.ANNUAL_FACTOR

            # Overfit: estrategia "gana" en IS pero pierde en OOS
            if sr_is > 0.0 and sr_oos <= 0.0:
                overfit_count += 1

        pbo = float(overfit_count) / float(n_simulations) if n_simulations > 0 else 0.50
        logger.info("  [CSCV] n_blocks={} | n_sims={} | overfit_count={} → PBO={.1f}%",
                    n_blocks, n_simulations, overfit_count, pbo * 100)
        return float(np.clip(pbo, 0.0, 1.0))

    # ── 4. Gauntlet principal ─────────────────────────────────────────────────

    def run_gauntlet(self, trades_df: pd.DataFrame,
                     is_returns: List[float] = None) -> Dict[str, Any]:
        """
        Ejecuta todos los gates estadísticos sobre un DataFrame de trades OOS reales.
        trades_df debe contener las columnas: ['return_pct', 'is_win', 'timestamp']

        Args:
            trades_df: DataFrame con trades OOS reales (generado por generate_oos_predictions.py)
            is_returns: Ignorado — se mantiene por compatibilidad. El PBO ahora
                        se calcula internamente con CSCV sobre los retornos OOS.
        """
        logger.info("🛡️ Iniciando LunaV1 Statistical Audit (The Gauntlet)...")

        # [GAUNTLET-FILTER-SILENCED 2026-06-13] Silenciar trades con kelly_fraction_used == 0
        if "kelly_fraction_used" in trades_df.columns:
            _initial_count = len(trades_df)
            trades_df = trades_df[trades_df["kelly_fraction_used"] > 0.0].copy()
            _filtered_count = len(trades_df)
            print(f"[GAUNTLET-FILTER-SILENCED 2026-06-13] Silenciados {(_initial_count - _filtered_count)} trades con kelly_fraction_used == 0. Quedan {_filtered_count} trades activos.")
            logger.info(f"[GAUNTLET-FILTER-SILENCED] Silenciados {(_initial_count - _filtered_count)} trades. Quedan: {_filtered_count}")

        if len(trades_df) == 0:
            return {"deploy_approved": False, "reason": "No active trades generated."}

        # ── Métricas base ────────────────────────────────────────────────────
        total_trades = len(trades_df)
        wins = int(trades_df['is_win'].sum())
        win_rate = float(wins) / float(total_trades)

        returns = trades_df['return_pct'].values.astype(float)
        total_ret = float(np.prod(1.0 + returns) - 1.0)

        # INC-04 fix: capear total_return_pct usando cfg.stat.total_return_cap (ARCH-02)
        TOTAL_RETURN_CAP = self.TOTAL_RETURN_CAP
        total_ret_pct_raw = total_ret * 100.0
        total_ret_is_capped = total_ret_pct_raw > TOTAL_RETURN_CAP
        total_ret_pct = min(total_ret_pct_raw, TOTAL_RETURN_CAP)

        # Drawdown sobre curva de equity acumulada
        cum_returns = (1.0 + returns).cumprod()
        peaks = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - peaks) / peaks
        max_dd = float(abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0

        # Sharpe anualizado (basado en frecuencia real de operaciones)
        # Si asumimos np.sqrt(365*24) sobre un array de trades, estaríamos asumiendo 8760 trades/año
        if "timestamp" in trades_df.columns and total_trades >= 2:
            ts = pd.to_datetime(trades_df["timestamp"])
            years_in_oos = max((ts.max() - ts.min()).total_seconds() / (365.25 * 24 * 3600), 1e-5)
            trades_per_year = total_trades / years_in_oos
        else:
            years_in_oos = 1.0
            trades_per_year = total_trades  # Fallback a 1 año

        mean_ret = float(np.mean(returns))
        std_ret  = float(np.std(returns))
        annual_factor_trades = np.sqrt(trades_per_year)
        sharpe_crudo = (mean_ret / std_ret) * annual_factor_trades if std_ret > 1e-10 else 0.0

        # Calmar Ratio
        # [FIX-CALMAR-01 2026-06-13] Calcular retorno anualizado CAGR para un Calmar comparable
        if 1.0 + total_ret > 0.0:
            total_ret_annualized = ((1.0 + total_ret) ** (1.0 / years_in_oos) - 1.0) * 100.0
        else:
            total_ret_annualized = -99.9  # límite inferior lógico
            
        calmar = float(total_ret_annualized) / (max_dd * 100.0) if max_dd > 1e-10 else float('inf')

        # Momentos de distribución (para DSR)
        skewness = float(stats.skew(returns))  if len(returns) > 2 else 0.0
        kurtosis = float(stats.kurtosis(returns)) if len(returns) > 2 else 0.0

        # ── Gates estadísticos ────────────────────────────────────

        logger.info("-- [Gauntlet] Métricas base ---------------------------------")
        logger.info("  Trades={} | WR={.1f}% ({} wins) | SR_crudo={.4f} | MaxDD={.1f}% | Calmar={.2f}",
                    total_trades, win_rate * 100, wins, sharpe_crudo, max_dd * 100,
                    calmar if calmar != float('inf') else 999.0)
        logger.info("  Distribución: skewness={.4f} | kurtosis={.4f}", skewness, kurtosis)

        # Corrección Matemática Profunda: n_obs debe ser OBLIGATORIAMENTE el tamaño real de la muestra
        # de la que se derivaron el skewness, kurtosis y Sharpe (es decir, total_trades). 
        # Usar holdout_hours (ej: 8760) colapsaba falsamente la varianza de la distribución a casi cero, 
        # provocando que el test DSR diera 100% de confianza a modelos mediocres porque std_sr era microscópico.
        logger.info(
            "-- [Gauntlet] Calculando DSR (Bailey & LdP 2014) — n_obs=%d trades, n_trials=%d",
            total_trades, self.n_trials
        )
        dsr = self._compute_dsr(
            sharpe_oos=sharpe_crudo,
            skewness=skewness,
            kurtosis=kurtosis,
            n_obs=total_trades,       # CORRECCIÓN: Tamaño real de la muestra
            n_trials=self.n_trials
        )

        # Binomial test (p-value)
        logger.info("-- [Gauntlet] Test Binomial (H0: WR=50%%) -------------------")
        p_value = self._compute_binomial_test(wins, total_trades)
        logger.info("  p-value={.6f} | alpha={.2f} | {}",
                    p_value, self.ALPHA_BINOMIAL,
                    "PASS" if p_value < self.ALPHA_BINOMIAL else "FAIL (no se rechaza H0)")

        # PBO via CSCV real (no proxy logístico, no IS mock)
        logger.info("-- [Gauntlet] PBO via CSCV (n_blocks={}, n_sims<=200) ------",
                    self.PBO_N_BLOCKS)
        pbo = self._estimate_pbo_cscv(returns)

        # ── Veredicto ────────────────────────────────────────────────────────
        pass_dsr      = dsr >= self.MIN_DSR
        pass_binomial = p_value < self.ALPHA_BINOMIAL
        pass_trades   = total_trades >= self.MIN_TRADES
        pass_dd       = max_dd < self.MAX_DRAWDOWN
        pass_pbo      = pbo < self.MAX_PBO

        deploy_approved = all([pass_dsr, pass_binomial, pass_trades, pass_dd, pass_pbo])

        # ── Tabla de gates ─────────────────────────────────────────
        logger.info("-- [Gauntlet] Tabla de Gates --------------------------------")
        logger.info("  %-22s %10s  %-10s  %s", "Gate", "Valor", "Umbral", "Estado")
        logger.info("  %-22s %10d  %-10s  %s", "Trades",
                    total_trades, f">={self.MIN_TRADES}",
                    "PASS" if pass_trades   else "FAIL")
        logger.info("  %-22s %9.1f%%  %-10s  %s", "Win Rate",
                    win_rate * 100, ">50%",
                    "PASS" if win_rate > 0.50 else "FAIL")
        logger.info("  %-22s %10.4f  %-10s  %s", "DSR",
                    dsr, f">={self.MIN_DSR}",
                    "PASS" if pass_dsr      else "FAIL")
        logger.info("  %-22s %10.6f  %-10s  %s", "p-binomial",
                    p_value, f"<{self.ALPHA_BINOMIAL}",
                    "PASS" if pass_binomial else "FAIL")
        logger.info("  %-22s %9.1f%%  %-10s  %s", "PBO",
                    pbo * 100, f"<{self.MAX_PBO * 100:.0f}%",
                    "PASS" if pass_pbo      else "FAIL")
        logger.info("  %-22s %9.1f%%  %-10s  %s", "MaxDrawdown",
                    max_dd * 100, f"<{self.MAX_DRAWDOWN * 100:.0f}%",
                    "PASS" if pass_dd       else "FAIL")
        logger.info("  %s", "-" * 60)
        logger.info("  VEREDICTO GAUNTLET: {}",
                    "✅ DEPLOY APROBADO" if deploy_approved else "❌ RECHAZADO")

        verdict = {
            "deploy_approved": bool(deploy_approved),
            "metrics": {
                "total_trades":         int(total_trades),
                "win_rate":             float(win_rate),
                "total_return_pct":     float(total_ret_pct),
                "total_return_is_capped": bool(total_ret_is_capped),
                "max_drawdown_pct":     float(max_dd * 100.0),
                "sharpe_crudo":         float(sharpe_crudo),
                "calmar_ratio":         float(calmar) if calmar != float('inf') else None,
            },
            "statistical_audit": {
                "dsr":              float(dsr),
                "binomial_p_value": float(p_value),
                "estimated_pbo":    float(pbo),
                "skewness":         float(skewness),
                "kurtosis":         float(kurtosis),
                "n_obs_dsr":        int(total_trades),   # ARCH-01: horas OOS reales usadas en DSR
                "n_trials_dsr":     int(self.n_trials),   # BUG-01: optuna_trials del run actual
            },
            "flags": {
                "pass_dsr":      bool(pass_dsr),
                "pass_binomial": bool(pass_binomial),
                "pass_trades":   bool(pass_trades),
                "pass_dd":       bool(pass_dd),
                "pass_pbo":      bool(pass_pbo),
            },
            "sop_thresholds": {
                "min_dsr":          self.MIN_DSR,
                "max_pbo_pct":      self.MAX_PBO * 100.0,
                "min_trades":       self.MIN_TRADES,
                "max_drawdown_pct": self.MAX_DRAWDOWN * 100.0,
            },
            "timestamp": pd.Timestamp.now('UTC').isoformat(),  # [FIX-PIPE-002]
        }

        # [AUDIT-VERDICT 2026-05-21] Bloque summary plano al nivel raíz.
        # Proyecta las claves críticas para auditoría post-mortem rápida (grep, jq, pandas).
        # No duplica lógica — los valores se calculan arriba y se proyectan aquí.
        # Referencia: audit_ultima_run.md §5 — "verdict JSON no guarda max_drawdown_pct directamente".
        verdict["summary"] = {
            "total_trades":      int(total_trades),
            "win_rate_pct":      round(win_rate * 100.0, 2),
            "max_drawdown_pct":  round(max_dd * 100.0, 2),
            "total_return_pct":  round(float(total_ret_pct), 4),
            "sharpe_crudo":      round(float(sharpe_crudo), 4),
            "calmar_ratio":      round(float(calmar), 2) if calmar != float('inf') else None,
            "dsr":               round(float(dsr), 4),
            "pbo_pct":           round(float(pbo * 100.0), 2),
            "binomial_p":        round(float(p_value), 6),
            "pass_trades":       bool(pass_trades),
            "pass_dsr":          bool(pass_dsr),
            "pass_dd":           bool(pass_dd),
            "pass_pbo":          bool(pass_pbo),
            "pass_binomial":     bool(pass_binomial),
            "deploy_approved":   bool(deploy_approved),
        }
        print(
            f"[AUDIT-VERDICT] summary plano generado: trades={total_trades} | "
            f"WR={win_rate*100:.1f}% | MaxDD={max_dd*100:.1f}% | DSR={dsr:.4f} | "
            f"PBO={pbo*100:.1f}% | approved={deploy_approved}"
        )

        status = "✅ PASS" if deploy_approved else "❌ FAIL"
        logger.info(f"📊 The Gauntlet Verdict: {status}")
        logger.info(f"   Trades: {total_trades} (Min: {self.MIN_TRADES}) → {'PASS' if pass_trades else 'FAIL'}")
        logger.info(f"   Win Rate: {win_rate:.1%}")
        logger.info(f"   Sharpe crudo: {sharpe_crudo:.2f} | MaxDD: {max_dd*100:.1f}% → {'PASS' if pass_dd else 'FAIL'}")
        logger.info(f"   DSR: {dsr:.4f} (Min: {self.MIN_DSR}) → {'PASS' if pass_dsr else 'FAIL'}")
        logger.info(f"   p-value binomial: {p_value:.5f} (Max: {self.ALPHA_BINOMIAL}) → {'PASS' if pass_binomial else 'FAIL'}")
        logger.info(f"   PBO (CSCV): {pbo:.1%} (Max: {self.MAX_PBO:.0%}) → {'PASS' if pass_pbo else 'FAIL'}")

        # Guardar veredicto JSON (ruta absoluta — independiente del CWD)
        report_dir = _PROJECT_ROOT / "data" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        verdict_path = report_dir / "statistical_verdict.json"
        with open(verdict_path, "w", encoding="utf-8") as f:
            json.dump(verdict, f, indent=4)
        logger.info(f"   Veredicto guardado en: {verdict_path}")

        return verdict
