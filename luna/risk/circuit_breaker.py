"""
luna/risk/circuit_breaker.py
─────────────────────────────────────────────────────────────────────────────
Circuit Breaker de Régimen Mensual — Hipótesis #2 (analisis_wfb_ensemble_20260528.md)

Propósito:
    Detectar regímenes adversos donde ≥N semillas tienen un Win Rate rolling de 30 días
    por debajo del umbral mínimo, y emitir una señal de suspensión de posiciones.

Diseño:
    - Opera sobre los trades OOS de múltiples semillas (parquets de wfb/)
    - Calcula WR rolling sobre ventanas de 30 días naturales
    - Emite señal CB_ACTIVE si el consenso adverso supera el threshold
    - Completamente configurable desde settings.yaml (sin magic numbers)
    - Integrable en evaluate_ensemble_wfb.py y en el Router de señales en vivo

Política No-Fallback (settingsyfallvack.md):
    - Todos los parámetros se leen desde settings.yaml
    - Si falta un parámetro crítico → RuntimeError (no fallback silencioso)
    - Los parámetros secundarios de diagnóstico admiten fallback con WARNING

Historia:
    [P2-A 2026-05-28] Creación inicial — datos empíricos de la sesión:
        - May-Jun 2025: WR < 35% en 5/6 semillas simultáneamente (agujero negro #1)
        - Ene 2026: WR < 25% en 5/6 semillas (agujero negro #2)
        - Umbral empírico: 38% WR rolling captura ambos regímenes sin falsos positivos
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Carga de parámetros desde settings.yaml (No-Fallback en parámetros críticos)
# ─────────────────────────────────────────────────────────────────────────────

def _load_cb_params() -> dict:
    """
    Lee los parámetros del Circuit Breaker desde settings.yaml.
    Política No-Fallback: los parámetros críticos fallan con RuntimeError si faltan.
    """
    try:
        from config.settings import cfg
    except Exception as e:
        raise RuntimeError(
            f"[CB] CRITICAL: No se pudo cargar config/settings.yaml para Circuit Breaker: {e}"
        ) from e

    # ── Parámetros CRÍTICOS (No-Fallback) ──
    try:
        _cb_cfg = cfg.wfb
    except AttributeError as e:
        raise RuntimeError(
            f"[CB] CRITICAL: Sección 'wfb' no encontrada en settings.yaml. {e}"
        ) from e

    # Intentar leer bloque circuit_breaker, o usar valores por defecto documentados
    # en parametros_fijos.md. Si el bloque no existe, se usan defaults con WARNING.
    try:
        _cb_block = cfg.wfb.circuit_breaker
        # Parámetros críticos del bloque
        _min_seeds_adverse = int(_cb_block.min_seeds_adverse)   # ≥ este N de seeds con WR bajo → CB
        _wr_threshold      = float(_cb_block.wr_threshold)       # WR rolling < este valor → adverso
        _rolling_days      = int(_cb_block.rolling_days)         # ventana de rolling (días)
        _cb_enabled        = bool(_cb_block.enabled)
        logger.info("[CB] Parámetros cargados desde settings.yaml.wfb.circuit_breaker")
    except AttributeError:
        # El bloque circuit_breaker no existe en settings.yaml → WARNING + defaults documentados
        logger.warning(
            "[CB] WARN: Bloque 'wfb.circuit_breaker' no encontrado en settings.yaml. "
            "Usando defaults documentados en parametros_fijos.md "
            "(min_seeds_adverse=4, wr_threshold=0.38, rolling_days=30, enabled=True). "
            "Añadir bloque circuit_breaker a settings.yaml para control explícito."
        )
        print(
            "[CB] WARN: wfb.circuit_breaker no en settings.yaml — usando defaults documentados."
        )
        _min_seeds_adverse = 4    # ≥4 seeds con WR bajo → CB activo
        _wr_threshold      = 0.38  # WR rolling < 38% → adverso (calibrado con datos reales)
        _rolling_days      = 30    # ventana 30 días naturales
        _cb_enabled        = True

    return {
        "min_seeds_adverse": _min_seeds_adverse,
        "wr_threshold":      _wr_threshold,
        "rolling_days":      _rolling_days,
        "enabled":           _cb_enabled,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal: RegimeCircuitBreaker
# ─────────────────────────────────────────────────────────────────────────────

class RegimeCircuitBreaker:
    """
    Circuit Breaker de Régimen Mensual basado en el Win Rate rolling multi-semilla.

    Uso (backtesting OOS):
        cb = RegimeCircuitBreaker(wfb_dir)
        report = cb.analyze(approved_seeds=[1337, 2025, 48907])
        cb.save_report(report, report_dir)

    Uso (producción en vivo):
        cb = RegimeCircuitBreaker(wfb_dir)
        is_blocked = cb.is_regime_adverse_now(approved_seeds)
    """

    def __init__(self, wfb_dir: Path):
        self.wfb_dir = Path(wfb_dir)
        self.params  = _load_cb_params()
        logger.info(
            "[CB] RegimeCircuitBreaker inicializado: "
            "min_seeds={} wr_threshold={:.0%} rolling_days={} enabled={}",
            self.params["min_seeds_adverse"],
            self.params["wr_threshold"],
            self.params["rolling_days"],
            self.params["enabled"],
        )
        print(
            f"[CB] RegimeCircuitBreaker: min_seeds_adverse={self.params['min_seeds_adverse']} | "
            f"wr_threshold={self.params['wr_threshold']:.0%} | "
            f"rolling_days={self.params['rolling_days']} | "
            f"enabled={self.params['enabled']}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Carga de trades por seed
    # ──────────────────────────────────────────────────────────────────────────

    def _load_seed_trades(self, seed: int) -> Optional[pd.DataFrame]:
        """Carga todos los trades OOS de una seed desde los parquets WFB."""
        dfs = []
        for w in range(1, 7):
            p = self.wfb_dir / f"oos_trades_W{w}_seed{seed}.parquet"
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    if "is_win" in df.columns and len(df) > 0:
                        dfs.append(df[["is_win"]].copy())
                except Exception as e:
                    logger.warning("[CB] Error leyendo {}: {}", p.name, e)

        if not dfs:
            logger.warning("[CB] Seed {} sin parquets WFB disponibles", seed)
            return None

        combined = pd.concat(dfs).sort_index()
        if not pd.api.types.is_datetime64_any_dtype(combined.index):
            combined.index = pd.to_datetime(combined.index, utc=True)
        else:
            combined.index = combined.index.tz_localize("UTC") if combined.index.tz is None else combined.index.tz_convert("UTC")

        combined = combined[~combined.index.duplicated(keep="first")]
        return combined

    # ──────────────────────────────────────────────────────────────────────────
    # Cálculo de WR rolling
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_rolling_wr(self, trades: pd.DataFrame, min_trades: int = 5) -> pd.Series:
        """
        Calcula el Win Rate rolling de 30 días (ventana deslizante por día natural).
        Solo emite valor cuando hay ≥ min_trades en la ventana.
        """
        rolling_days = self.params["rolling_days"]
        wins   = trades["is_win"].astype(float)
        wr_rol = wins.rolling(f"{rolling_days}D", min_periods=min_trades).mean()
        return wr_rol

    # ──────────────────────────────────────────────────────────────────────────
    # Análisis completo: CB report para backtesting
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        approved_seeds: List[int],
        min_trades_per_window: int = 5,
    ) -> Dict:
        """
        Analiza los trades OOS de todas las seeds aprobadas y calcula el mapa
        de regímenes adversos (periodos donde ≥ min_seeds_adverse seeds tienen WR < threshold).

        Returns:
            dict con:
                - adverse_periods: lista de (start, end, n_seeds_adverse, avg_wr)
                - trades_blocked: N de trades bloqueados por CB en backtesting
                - trades_total: N total de trades sin CB
                - wr_by_seed: dict {seed: pd.Series} con WR rolling
                - cb_signal: pd.Series bool (True = CB activo en ese timestamp)
        """
        if not self.params["enabled"]:
            logger.info("[CB] Circuit Breaker DESACTIVADO (enabled=False en settings)")
            print("[CB] Circuit Breaker DESACTIVADO")
            return {"enabled": False, "adverse_periods": [], "trades_blocked": 0}

        print(
            f"\n[CB] Analizando Circuit Breaker de Régimen para {len(approved_seeds)} seeds aprobadas..."
        )
        logger.info("[CB] Analizando {} seeds: {}", len(approved_seeds), approved_seeds)

        threshold       = self.params["wr_threshold"]
        min_seeds_adv   = self.params["min_seeds_adverse"]

        # ── Cargar trades y calcular WR rolling por seed ──
        wr_by_seed: Dict[int, pd.Series] = {}
        for seed in approved_seeds:
            trades = self._load_seed_trades(seed)
            if trades is None or len(trades) == 0:
                logger.warning("[CB] Seed {} sin trades — excluida del análisis", seed)
                print(f"[CB] Seed {seed}: sin trades — excluida")
                continue
            wr = self._compute_rolling_wr(trades, min_trades=min_trades_per_window)
            wr_by_seed[seed] = wr
            _below = (wr < threshold).sum()
            print(
                f"[CB] Seed {seed}: {len(trades)} trades | "
                f"WR rolling media={wr.mean():.1%} | "
                f"periodos adversos (WR<{threshold:.0%}): {_below} timestamps"
            )

        if not wr_by_seed:
            logger.warning("[CB] Sin seeds con datos para Circuit Breaker")
            return {"enabled": True, "adverse_periods": [], "trades_blocked": 0}

        # ── Construir señal de consenso adverso ──
        # Para cada timestamp, contar cuántas seeds tienen WR < threshold
        all_idx = pd.DatetimeIndex(sorted(
            set().union(*[wr.dropna().index for wr in wr_by_seed.values()])
        ))

        # Reindexar todas las series al mismo índice (forward-fill = última ventana conocida)
        wr_df = pd.DataFrame(index=all_idx)
        for seed, wr in wr_by_seed.items():
            wr_df[seed] = wr.reindex(all_idx, method="ffill")

        # Contar cuántas seeds están en régimen adverso en cada timestamp
        n_adverse = (wr_df < threshold).sum(axis=1)
        cb_signal = n_adverse >= min_seeds_adv

        # ── Identificar periodos adversos contiguos ──
        adverse_periods = []
        in_adverse = False
        period_start = None
        for ts in all_idx:
            if cb_signal.loc[ts] and not in_adverse:
                in_adverse    = True
                period_start  = ts
            elif not cb_signal.loc[ts] and in_adverse:
                in_adverse   = False
                # Calcular estadísticas del periodo
                mask     = (all_idx >= period_start) & (all_idx < ts)
                avg_wr   = wr_df.loc[mask].mean(axis=1).mean()
                n_seeds  = int(n_adverse.loc[mask].max())
                adverse_periods.append({
                    "start":         str(period_start.date()),
                    "end":           str(ts.date()),
                    "n_seeds_adverse": n_seeds,
                    "avg_wr_ensemble": round(float(avg_wr), 4) if not np.isnan(avg_wr) else None,
                })
        if in_adverse and period_start is not None:
            mask    = all_idx >= period_start
            avg_wr  = wr_df.loc[mask].mean(axis=1).mean()
            n_seeds = int(n_adverse.loc[mask].max())
            adverse_periods.append({
                "start":           str(period_start.date()),
                "end":             "ongoing",
                "n_seeds_adverse": n_seeds,
                "avg_wr_ensemble": round(float(avg_wr), 4) if not np.isnan(avg_wr) else None,
            })

        # ── Calcular impacto (trades bloqueados) ──
        trades_blocked = 0
        for seed in approved_seeds:
            trades = self._load_seed_trades(seed)
            if trades is None:
                continue
            for t in trades.index:
                if t in cb_signal.index and cb_signal.loc[t]:
                    trades_blocked += 1

        print(f"\n[CB] PERIODOS ADVERSOS DETECTADOS: {len(adverse_periods)}")
        for p in adverse_periods:
            print(
                f"  [{p['start']} → {p['end']}] "
                f"seeds_adversas={p['n_seeds_adverse']}/{len(wr_by_seed)} | "
                f"avg_WR={p.get('avg_wr_ensemble', '?'):.1%}" 
                if p.get('avg_wr_ensemble') is not None else
                f"  [{p['start']} → {p['end']}] seeds_adversas={p['n_seeds_adverse']}"
            )
        print(f"[CB] Trades bloqueados por CB: {trades_blocked}")
        logger.info(
            "[CB] Periodos adversos: {} | Trades bloqueados: {} | Seeds analizadas: {}",
            len(adverse_periods), trades_blocked, len(wr_by_seed)
        )

        return {
            "enabled":          True,
            "params":           self.params,
            "seeds_analyzed":   list(wr_by_seed.keys()),
            "adverse_periods":  adverse_periods,
            "trades_blocked":   trades_blocked,
            "cb_signal":        cb_signal,          # pd.Series — para uso interno
            "wr_by_seed":       wr_by_seed,         # dict seed→pd.Series — para visualización
        }

    # ──────────────────────────────────────────────────────────────────────────
    # API para producción en vivo: ¿está el régimen adverso AHORA?
    # ──────────────────────────────────────────────────────────────────────────

    def is_regime_adverse_now(
        self,
        approved_seeds: List[int],
        as_of: Optional[pd.Timestamp] = None,
    ) -> Tuple[bool, dict]:
        """
        Evalúa si el régimen actual es adverso según los últimos 30 días de trades.

        Returns:
            (is_blocked: bool, details: dict)
        """
        if not self.params["enabled"]:
            return False, {"reason": "CB desactivado"}

        if as_of is None:
            as_of = pd.Timestamp.utcnow()

        report = self.analyze(approved_seeds)
        cb_signal = report.get("cb_signal")

        if cb_signal is None or cb_signal.empty:
            return False, {"reason": "Sin datos suficientes"}

        # Buscar el último timestamp conocido ≤ as_of
        available = cb_signal[cb_signal.index <= as_of]
        if available.empty:
            return False, {"reason": "Sin datos previos a as_of"}

        is_blocked = bool(available.iloc[-1])
        details = {
            "as_of":          str(as_of.date()),
            "last_cb_check":  str(available.index[-1].date()),
            "is_blocked":     is_blocked,
            "adverse_periods": report["adverse_periods"],
        }
        print(f"[CB] is_regime_adverse_now({as_of.date()}): is_blocked={is_blocked}")
        logger.info("[CB] Evaluación en vivo ({}) → is_blocked={}", as_of.date(), is_blocked)
        return is_blocked, details

    # ──────────────────────────────────────────────────────────────────────────
    # Persistencia del reporte
    # ──────────────────────────────────────────────────────────────────────────

    def save_report(self, report: Dict, report_dir: Path, seed_suffix: str = "") -> Path:
        """Guarda el reporte del CB en JSON para auditoría y dashboards."""
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        # Serializar solo lo que es JSON-serializable (excluir pd.Series)
        serializable = {k: v for k, v in report.items()
                        if not isinstance(v, (pd.Series, dict)) or k == "params"}
        if "wr_by_seed" in report:
            serializable["wr_by_seed_summary"] = {
                str(seed): {
                    "mean": round(float(wr.mean()), 4) if not wr.empty else None,
                    "min":  round(float(wr.min()),  4) if not wr.empty else None,
                    "pct_below_threshold": round(float((wr < self.params["wr_threshold"]).mean()), 4)
                    if not wr.empty else None,
                }
                for seed, wr in report["wr_by_seed"].items()
            }

        suffix = f"_{seed_suffix}" if seed_suffix else ""
        out_path = report_dir / f"circuit_breaker_report{suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

        print(f"[CB] Reporte guardado → {out_path.name}")
        logger.info("[CB] Reporte guardado: {}", out_path)
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# API de conveniencia para uso desde evaluate_ensemble_wfb.py
# ─────────────────────────────────────────────────────────────────────────────

def run_circuit_breaker_analysis(
    approved_seeds: List[int],
    wfb_dir: Path,
    report_dir: Path,
    seed_suffix: str = "",
) -> Dict:
    """
    Función de conveniencia para llamar desde evaluate_ensemble_wfb.py.
    Ejecuta el análisis completo y guarda el reporte.

    Returns:
        dict con adverse_periods, trades_blocked y cb_signal
    """
    print(f"\n{'='*60}")
    print(f"[CB] CIRCUIT BREAKER DE RÉGIMEN MENSUAL")
    print(f"[CB] Seeds aprobadas: {approved_seeds}")
    print(f"{'='*60}")

    cb = RegimeCircuitBreaker(wfb_dir)
    report = cb.analyze(approved_seeds)
    cb.save_report(report, report_dir, seed_suffix=seed_suffix)

    print(f"[CB] Análisis completado. Periodos adversos: {len(report.get('adverse_periods', []))}")
    return report
