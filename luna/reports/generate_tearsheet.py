"""
generate_tearsheet.py — Luna V1 Fase 5  v8.0
=============================================
TearSheet OOS completo con 13 paneles en 10 filas.

Layout (10 filas):
  Row 0 (full) : Precio BTC/USDT + markers por trade + HMM background shading (M1)
  Row 1 (full) : Curva de equity acumulado + BTC Buy&Hold benchmark (M2)
  Row 2 (full) : Retorno individual por trade (barras win/loss)
  Row 3 left   : Underwater Plot (drawdown)
  Row 3 right  : Rolling Sharpe + Rolling Win Rate eje derecho (M3)
  Row 4 left   : Monthly Returns Heatmap / Win-Loss bars (fallback)
  Row 4 right  : Distribución de retornos por trade + VaR 95%
  Row 5 left   : Gate Status Semaphore [PASS]/[FAIL]
  Row 5 right  : Metric Gauges (DSR / WR / PBO vs umbrales)
  Row 6 left   : [Panel A] XGB Prob Cuartiles (señal monotónica?)
  Row 6 right  : [Panel B] Holding Time Distribution (barrera vertical)
  Row 7 (full) : [Panel C] HMM Regime Distribution Map
  Row 8 (full) : [Panel D] Walk-Forward Validation — WR por ventana (M4)
  Row 9 (full) : [Panel E] Timeline Data Overlap Audit (IS vs VAL vs OOS)

Banner superior: 2 líneas — título+veredicto / WR·MaxDD·Sharpe·DSR·PBO·Trades (M6)

Artefactos:
  data/reports/YYYY-MM-DD_THHMM_tearsheet_oos.png  — archivado
  data/reports/tearsheet_oos.png                    — copia «latest»
"""
from __future__ import annotations

import json
from loguru import logger
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binom

# logger already imported from loguru
# ── Paleta Luna dark ──────────────────────────────────────────────────────────
_BG     = "#0d1117"
_PANEL  = "#161b22"
_BORDER = "#30363d"
_GREEN  = "#00ffb3"
_RED    = "#ff4d6d"
_YELLOW = "#ffcc00"
_BLUE   = "#4ea8de"
_GRAY   = "#8b949e"
_PURPLE = "#c792ea"
_WHITE  = "#e6edf3"
_GRIDΑ  = 0.15

# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct_fmt(ax, decimals: int = 1):
    """Formateador % para eje Y."""
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.{decimals}f}%")
    )

def _date_fmt(ax, n_ticks: int = 6):
    """Formateador de fechas en eje X — formato DD/MM/AAAA."""
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=n_ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=7)

def _caption(ax, text: str):
    """Texto de ayuda gris bajo la gráfica — fuera del área del subplot pero dentro de la figura."""
    ax.annotate(
        text, xy=(0.0, -0.13), xycoords="axes fraction",
        fontsize=7.0, color=_GRAY, va="top", ha="left",
        style="italic", wrap=True,
        annotation_clip=False,  # necesario: la coord es negativa (debajo del subplot)
    )

def _style_ax(ax):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_GRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_BORDER)

def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


# ── Carga de datos OHLCV ──────────────────────────────────────────────────────

def _load_btc_price(project_root: Path, start: pd.Timestamp, end: pd.Timestamp
                    ) -> Optional[pd.Series]:
    """
    Intenta cargar precio de cierre BTC desde múltiples fuentes.
    Devuelve Series con DatetimeIndex UTC o None si no se encuentra.
    """
    candidates = [
        project_root / "data" / "raw" / "ohlcv" / "ohlcv_raw.parquet",
        project_root / "data" / "historical" / "daemon" / "BTCUSDT_1h.parquet",
        project_root / "data" / "features" / "features_oos.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["close"] if "close" in
                                 pd.read_parquet(path, columns=[]).columns else None)
            if "close" not in df.columns:
                continue
            df.index = pd.to_datetime(df.index, utc=True)
            mask = (df.index >= start) & (df.index <= end)
            sliced = df.loc[mask, "close"]
            if len(sliced) > 0:
                return sliced
        except Exception as e:
            logger.debug("OHLCV fallback error {}: {}", path, e)
    return None


# =============================================================================
# LunaTearSheet
# =============================================================================

class LunaTearSheet:
    """
    Generador del TearSheet visual para The Gauntlet (Fase 5).

    Usage
    -----
    ts = LunaTearSheet(project_root=ROOT, output_dir="data/reports")
    out = ts.generate(trades_df, timestamp="2026-03-08_T1145")
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        output_dir: str = "data/reports",
    ):
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent
        self.root    = project_root
        self.out_dir = self.root / output_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_drawdowns(self, cum: pd.Series) -> pd.Series:
        peaks = cum.cummax()
        return ((cum - peaks) / peaks.replace(0, np.nan)).fillna(0.0)

    def _rolling_sharpe(self, ret: pd.Series, window: int = 30) -> pd.Series:
        r = ret.rolling(window)
        return (r.mean() / r.std().replace(0, np.nan)) * np.sqrt(252 * 24)

    def _load_verdict(self) -> dict:
        return _load_json(self.out_dir / "statistical_verdict.json")

    def _apply_global_style(self):
        plt.rcParams.update({
            "figure.facecolor": _BG,
            "axes.facecolor":   _BG,
            "axes.edgecolor":   _BORDER,
            "axes.labelcolor":  _GRAY,
            "xtick.color":      _GRAY,
            "ytick.color":      _GRAY,
            "text.color":       _WHITE,
            "grid.color":       _GRAY,
            "font.family":      "DejaVu Sans",
            "font.size":        9,
        })

    # ── Paleta HMM regimes ────────────────────────────────────────────────────
    # Colores diferenciados del verde/rojo de markers para no confundir al lector.
    _HMM_COLORS = {
        "1_BULL_TREND":              "#00c4ff",  # azul cian
        "1_BULL_TREND_B":            "#7ecfff",  
        "1_VOLATILE_BULL":           "#00ffb3",  # verde neon
        "1_VOLATILE_BULL_B":         "#7effd9",
        "2_CALM_RANGE":              "#ffd166",  # amarillo
        "2_CALM_RANGE_B":            "#ffe099",
        "2_VOLATILE_RANGE":          "#ffb347",  # naranja
        "2_VOLATILE_RANGE_B":        "#ffcc80",
        "3_CALM_BEAR":               "#c792ea",  # lila
        "3_CALM_BEAR_B":             "#dcbef0",
        "3_BEAR_CRASH":              "#ff4d6d",  # rojo coral
        "3_BEAR_CRASH_B":            "#ff8098",
        "4_BEAR_FORCED":             "#555555",
        "UNKNOWN":                   "#555555",
    }

    def _load_hmm_regimes(self) -> Optional[pd.Series]:
        """
        Carga hmm_regime_labels.parquet y lo extiende al período holdout si es necesario.

        Estrategia en cascada:
        1. Parquet base (hmm_regime_labels.parquet) — entrenamiento + validación
        2. Si no cubre el holdout → predict_regime_series() sobre features_holdout.parquet
        3. Si HMM no disponible → ffill con el último estado conocido hasta fin del holdout
        """
        path = self.root / "data" / "features" / "hmm_regime_labels.parquet"
        if not path.exists():
            logger.warning("[HMM] hmm_regime_labels.parquet no encontrado en {}", path)
            return None
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index, utc=True)
            if "HMM_Semantic" in df.columns:
                s = df["HMM_Semantic"].astype(str)
            else:
                s = df.iloc[:, 0].astype(str)
            s = s.sort_index()
            parquet_end = s.index.max()
            logger.debug("[HMM] Base: {} filas | {} → {}",
                         len(s), s.index.min().date(), parquet_end.date())

            # ── Detectar si el holdout queda fuera del rango base ──────────────
            # AUDIT Tier 3 (BUG-HOLDOUT-PATH): usar parquet especifico de ventana si existe
            import os as _os_ts
            _win_ts = _os_ts.environ.get("LUNA_WINDOW_ID", "")
            _hp_ts = self.root / "data" / "features" / f"features_holdout_{_win_ts}.parquet"
            holdout_path = _hp_ts if (_win_ts and _hp_ts.exists()) else self.root / "data" / "features" / "features_holdout.parquet"
            if holdout_path.exists():
                try:
                    df_hld = pd.read_parquet(holdout_path)
                    df_hld.index = pd.to_datetime(df_hld.index, utc=True)
                    holdout_end = df_hld.index.max()

                    if holdout_end > parquet_end + pd.Timedelta(hours=24):
                        # Intentar predict_regime_series para extender
                        try:
                            from luna.models.hmm_regime import HMMRegimeModel as _HMM
                            _hmm_model = _HMM.load(self.root / "data" / "models")
                            _predicted_df = _hmm_model.predict_regime_series(df_hld)
                            _predicted = _predicted_df["HMM_Regime"]
                            # predict devuelve entero → mapear a string semántico vía state_map
                            _state_map = getattr(_hmm_model, "state_names_", None) or {}
                            if hasattr(_predicted, "map") and _state_map:
                                _predicted = _predicted.map(_state_map).fillna(_predicted.astype(str))
                            else:
                                _predicted = _predicted.astype(str)
                            _predicted.index = pd.to_datetime(_predicted.index, utc=True)
                            # Solo el delta que no está en el base
                            _delta = _predicted[_predicted.index > parquet_end]
                            if len(_delta) > 0:
                                s = pd.concat([s, _delta]).sort_index()
                                logger.info("[HMM] Extendido via predict_regime_series: +{} filas hasta {}",
                                            len(_delta), s.index.max().date())
                        except Exception as _he:
                            # Fallback: ffill con el último estado conocido
                            logger.warning("[HMM] predict_regime_series falló ({}) — forward-fill hasta {}",
                                           _he, holdout_end.date())
                            _extra_idx = pd.date_range(
                                start=parquet_end + pd.Timedelta(hours=1),
                                end=holdout_end,
                                freq="1h",
                                tz="UTC",
                            )
                            _last_state = s.iloc[-1]
                            _extra = pd.Series(_last_state, index=_extra_idx, name=s.name)
                            s = pd.concat([s, _extra]).sort_index()
                            logger.info("[HMM] Forward-fill aplicado: estado='%s' hasta %s",
                                        _last_state, holdout_end.date())
                except Exception as _hld_e:
                    logger.debug("[HMM] No se pudo leer features_holdout: {}", _hld_e)

            return s
        except Exception as e:
            logger.warning("[HMM] Load error: {}", e)
            return None


    # ── Subplot 0: BTC price + trade markers ────────────────────────────────

    def _plot_btc_trades(self, ax, trades_df: pd.DataFrame,
                         start: pd.Timestamp, end: pd.Timestamp):
        """
        Carga precio BTC y dibuja un punto verde (win) / rojo (loss) por trade.
        El sombreado HMM se muestra en el panel dedicado _plot_hmm_regime_map.
        """
        btc = _load_btc_price(self.root, start, end)
        has_btc = btc is not None and len(btc) > 10

        if has_btc:
            # M1: HMM regime background shading
            try:
                _hmm_bg = self._load_hmm_regimes()
                if _hmm_bg is not None and len(_hmm_bg) > 0:
                    _hoos = _hmm_bg[(_hmm_bg.index >= start) & (_hmm_bg.index <= end)].sort_index()
                    _pr, _ps = None, None
                    for _ts, _reg in _hoos.items():
                        if _reg != _pr:
                            if _pr is not None and _ps is not None:
                                ax.axvspan(_ps, _ts, color=self._HMM_COLORS.get(str(_pr), "#555555"),
                                           alpha=0.11, zorder=0, linewidth=0)
                            _ps, _pr = _ts, _reg
                    if _pr is not None and _ps is not None and len(_hoos) > 0:
                        ax.axvspan(_ps, _hoos.index[-1], color=self._HMM_COLORS.get(str(_pr), "#555555"),
                                   alpha=0.11, zorder=0, linewidth=0)
            except Exception:
                pass
            ax.plot(btc.index, btc.values / 1000, color="#3d5a80",
                    linewidth=1.0, alpha=0.85, zorder=1)
            ax.set_ylabel("BTC Price (k USDT)", color=_GRAY, fontsize=9)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"${v:.0f}k"))
        else:
            ax.set_ylabel("(Sin precio BTC — datos no encontrados)", color=_GRAY, fontsize=8)
            ax.set_ylim(0, 1)

        wins  = trades_df[trades_df["return_pct"] > 0]
        loses = trades_df[trades_df["return_pct"] <= 0]

        def _scatter(subset, color, label, zorder):
            if len(subset) == 0:
                return
            ts_list = pd.DatetimeIndex(subset.index).tz_localize("UTC") \
                if pd.DatetimeIndex(subset.index).tz is None \
                else pd.DatetimeIndex(subset.index).tz_convert("UTC")
            if has_btc:
                prices = []
                for t in ts_list:
                    idx = btc.index.get_indexer([t], method="nearest")
                    prices.append(btc.iloc[idx[0]] / 1000 if idx[0] >= 0 else np.nan)
                ax.scatter(ts_list, prices, color=color, s=18, alpha=0.75,
                           label=label, zorder=zorder, edgecolors="none")
            else:
                ax.scatter(ts_list, [0.5] * len(ts_list), color=color, s=18,
                           alpha=0.75, label=label, zorder=zorder, edgecolors="none")

        _scatter(wins,  _GREEN, f"Wins   ({len(wins)})",   5)
        _scatter(loses, _RED,   f"Losses ({len(loses)})", 4)

        _date_fmt(ax, n_ticks=8)
        ax.set_xlabel("")
        ax.grid(alpha=_GRIDΑ, axis="y")

        days  = (end - start).days
        weeks = days // 7
        ax.set_title(
            f"BTC/USDT — Trades OOS Marcados  "
            f"({len(trades_df)} trades  |  {start.date()} → {end.date()}  |  ~{weeks}w)",
            fontsize=11, color=_WHITE, pad=7, fontweight="bold"
        )
        ax.legend(fontsize=8, framealpha=0.25,
                  facecolor=_PANEL, edgecolor=_BORDER, labelcolor="white",
                  loc="upper left", markerscale=1.5)
        _caption(ax,
                 "Precio BTC/USDT 1H con cada trade del set OOS marcado."
                 " Puntos verdes = trade ganador (TP). Puntos rojos = trade perdedor (SL/barrera)."
                 " La distribución por régimen HMM se muestra en el Panel C inferior.")
        _style_ax(ax)

    # ── Panel C: HMM Regime Distribution Map ────────────────────────────────

    def _plot_hmm_regime_map(self, ax, trades_df: pd.DataFrame,
                             start: pd.Timestamp, end: pd.Timestamp):
        """
        Mapa de distribución de trades por régimen HMM.
        Muestra para cada estado: barras horizontales Wins (verde) / Losses (rojo),
        Win Rate anotado, y cobertura del régimen en % del período OOS.
        """
        if len(trades_df) == 0:
            ax.set_axis_off()
            return

        # --- Mapear régimen a cada trade ---
        t_idx = pd.DatetimeIndex(trades_df.index)
        if t_idx.tz is None:
            t_idx = t_idx.tz_localize("UTC")
        else:
            t_idx = t_idx.tz_convert("UTC")

        df = trades_df.copy()
        
        # [FIX-TEARSHEET-HMM-02] Priorizar columna hmm_regime validada
        if "hmm_regime" in df.columns:
            df["_regime"] = df["hmm_regime"].astype(str)
            hmm = self._load_hmm_regimes()
            hmm_sorted = hmm.sort_index() if hmm is not None and len(hmm) > 0 else None
        else:
            hmm = self._load_hmm_regimes()
            if hmm is None or len(hmm) == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "HMM regimes no disponibles",
                        ha="center", va="center", color=_GRAY, fontsize=11,
                        transform=ax.transAxes)
                ax.set_title("Régimen HMM — Distribución de Trades",
                             fontsize=11, color=_GRAY, pad=6)
                return

            hmm_sorted = hmm.sort_index()
            mapped = hmm_sorted.reindex(t_idx, method="nearest", tolerance=pd.Timedelta("2H"))
            cov = mapped.notna().mean()
            if cov < 0.50:
                logger.warning(
                    "HMM regime map: cobertura baja (%.0f%%). "
                    "trades rango=%s→%s | HMM parquet=%s→%s | "
                    "Verificar timezone y rango de hmm_regime_labels.parquet.",
                    cov * 100,
                    t_idx.min().date() if len(t_idx) > 0 else "N/A",
                    t_idx.max().date() if len(t_idx) > 0 else "N/A",
                    hmm_sorted.index.min().date(), hmm_sorted.index.max().date(),
                )
            df["_regime"] = mapped.values


        # Orden de estados (de mejor a peor WR esperado)
        ORDER = [
            "1_BULL_TREND", "1_BULL_TREND_B", "1_VOLATILE_BULL", "1_VOLATILE_BULL_B",
            "2_CALM_RANGE", "2_CALM_RANGE_B", "2_VOLATILE_RANGE", "2_VOLATILE_RANGE_B",
            "3_CALM_BEAR", "3_CALM_BEAR_B", "3_BEAR_CRASH", "3_BEAR_CRASH_B", "4_BEAR_FORCED"
        ]
        short_map = {
            "1_BULL_TREND":       "BULL\nTREND",
            "1_BULL_TREND_B":     "BULL\nTRND B",
            "1_VOLATILE_BULL":    "VOL\nBULL",
            "1_VOLATILE_BULL_B":  "VOL\nBULL B",
            "2_CALM_RANGE":       "CALM\nRNGE",
            "2_CALM_RANGE_B":     "CALM\nRNGE B",
            "2_VOLATILE_RANGE":   "VOL\nRNGE",
            "2_VOLATILE_RANGE_B": "VOL\nRNGE B",
            "3_CALM_BEAR":        "CALM\nBEAR",
            "3_CALM_BEAR_B":      "CALM\nBEAR B",
            "3_BEAR_CRASH":       "BEAR\nCRASH",
            "3_BEAR_CRASH_B":     "BEAR\nCRASH B",
            "4_BEAR_FORCED":      "BEAR\nFRCD"
        }

        # Agregar stats por régimen
        regimes_present = [r for r in ORDER if r in df["_regime"].values]
        # También incluir los que estén pero no en ORDER (numéricos, desconocidos)
        for r in df["_regime"].unique():
            if r not in regimes_present:
                regimes_present.append(str(r))

        rows = []
        for reg in regimes_present:
            g = df[df["_regime"] == reg]
            n_total  = len(g)
            n_wins   = int((g["return_pct"] > 0).sum())
            n_losses = n_total - n_wins
            wr       = n_wins / n_total if n_total > 0 else 0.0
            # Cobertura temporal: qué % del OOS estuvo en este régimen
            cov = (hmm == reg).mean() if n_total > 0 and hmm is not None else 0.0
            rows.append({
                "reg": reg,
                "label": short_map.get(str(reg), str(reg)),
                "n_total": n_total, "n_wins": n_wins, "n_losses": n_losses,
                "wr": wr, "cov": cov,
                "color": self._HMM_COLORS.get(str(reg), "#555555"),
            })

        if not rows:
            ax.set_axis_off()
            return

        # --- Dibujar barras horizontales agrupadas ---
        n_regs  = len(rows)
        y_pos   = np.arange(n_regs)
        bar_h   = 0.35
        max_n   = max(r["n_total"] for r in rows) or 1

        ax.set_facecolor(_BG)
        ax.set_xlim(-max_n * 0.02, max_n * 1.40)   # extra espacio derecha para anotaciones
        ax.set_ylim(-0.6, n_regs - 0.4)

        for i, row in enumerate(rows):
            y      = y_pos[i]
            color  = row["color"]
            nw, nl = row["n_wins"], row["n_losses"]
            wr     = row["wr"]

            # Fondo suave del régimen (full row)
            ax.barh(y, max_n * 1.35, height=bar_h * 2.6,
                    left=-max_n * 0.02, color=color, alpha=0.06, zorder=0)

            # Barra wins (verde)
            ax.barh(y + bar_h / 2, nw, height=bar_h,
                    color=_GREEN, alpha=0.80, zorder=2,
                    label="Wins" if i == 0 else "")

            # Barra losses (rojo)
            ax.barh(y - bar_h / 2, nl, height=bar_h,
                    color=_RED, alpha=0.80, zorder=2,
                    label="Losses" if i == 0 else "")

            # Anotación WR  (lado derecho)
            wr_color = _GREEN if wr >= 0.50 else _RED
            ax.text(max_n * 1.06, y,
                    f"WR {wr*100:.1f}%",
                    va="center", ha="left",
                    fontsize=10, color=wr_color, fontweight="bold")

            # Números exactos dentro de las barras
            if nw > 0:
                ax.text(nw / 2, y + bar_h / 2, str(nw),
                        va="center", ha="center", fontsize=8,
                        color=_BG, fontweight="bold")
            if nl > 0:
                ax.text(nl / 2, y - bar_h / 2, str(nl),
                        va="center", ha="center", fontsize=8,
                        color=_BG, fontweight="bold")

            # Cobertura temporal
            ax.text(max_n * 1.28, y,
                    f"{row['cov']*100:.0f}%\ncov",
                    va="center", ha="center", fontsize=7, color=_GRAY)

        # Etiquetas Y (nombres cortos de régimen con parche de color)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r["label"] for r in rows],
                           fontsize=9, color=_WHITE, ha="right")
        for tick, row in zip(ax.get_yticklabels(), rows):
            tick.set_color(row["color"])

        # Línea vertical en 50% WR de referencia
        ax.axvline(max_n * 0.50, color=_YELLOW, linewidth=0.9,
                   linestyle=":", alpha=0.4, zorder=1)

        ax.set_xlabel("Número de trades", color=_GRAY, fontsize=9)
        ax.grid(alpha=_GRIDΑ, axis="x")

        # Calcular WR global para el título
        total_t = sum(r["n_total"] for r in rows)
        total_w = sum(r["n_wins"] for r in rows)
        global_wr = total_w / total_t * 100 if total_t > 0 else 0

        ax.set_title(
            f"Distribución de Trades por Régimen HMM  —  "
            f"{len(regimes_present)} regímenes activos  |  WR global: {global_wr:.1f}%",
            fontsize=11, color=_WHITE, pad=7, fontweight="bold"
        )

        # Leyenda Wins/Losses
        win_p  = mpatches.Patch(color=_GREEN, alpha=0.80, label="Wins")
        loss_p = mpatches.Patch(color=_RED,   alpha=0.80, label="Losses")
        ax.legend(handles=[win_p, loss_p], fontsize=8, framealpha=0.25,
                  facecolor=_PANEL, edgecolor=_BORDER, labelcolor="white",
                  loc="lower right")

        _caption(ax,
                 "Distribución de trades por régimen HMM activo en el momento de entrada."
                 " Barras verdes = wins, rojas = losses. WR% por régimen anotado a la derecha."
                 " Cov = % del período OOS que el régimen estuvo activo.")
        _style_ax(ax)

    # ── Subplot 1: Equity curve ───────────────────────────────────────────────

    def _plot_equity(self, ax, cum_ret: pd.Series, ret: pd.Series,
                     start: pd.Timestamp, end: pd.Timestamp):
        x = cum_ret.index if isinstance(cum_ret.index, pd.DatetimeIndex) \
            else range(len(cum_ret))
        has_dates = isinstance(cum_ret.index, pd.DatetimeIndex)

        ax.fill_between(x, cum_ret.values, 1.0,
                        where=cum_ret.values >= 1.0, color=_GREEN, alpha=0.07)
        ax.fill_between(x, cum_ret.values, 1.0,
                        where=cum_ret.values < 1.0, color=_RED, alpha=0.12)
        ax.plot(x, cum_ret.values, color=_GREEN, linewidth=1.8, zorder=3, label="Luna V1")
        # M2: BTC buy & hold benchmark
        if has_dates:
            try:
                _btc_bm = _load_btc_price(self.root, start, end)
                if _btc_bm is not None and len(_btc_bm) > 10:
                    _btc_r = _btc_bm.reindex(cum_ret.index, method="nearest").dropna()
                    if len(_btc_r) > 2 and _btc_r.iloc[0] > 0:
                        _btc_norm = _btc_r / _btc_r.iloc[0]
                        ax.plot(_btc_r.index, _btc_norm.values, color=_YELLOW,
                                linewidth=1.0, linestyle="--", alpha=0.45, zorder=2, label="BTC B&H")
            except Exception:
                pass
        ax.axhline(1.0, color=_GRAY, linewidth=0.8, linestyle="--", alpha=0.5)

        final_pct = (cum_ret.iloc[-1] - 1) * 100
        color = _GREEN if final_pct >= 0 else _RED

        # Anotar retorno final
        ax.text(x[-1] if not has_dates else x[-1],
                cum_ret.iloc[-1], f" {final_pct:+.1f}%",
                color=color, fontsize=9, va="center", fontweight="bold")

        n_w = int((ret > 0).sum())
        wr  = n_w / len(ret) * 100

        ax.set_title(
            f"Curva de Equity Acumulado — Retorno Total: {final_pct:+.1f}%  "
            f"|  {len(ret)} trades  |  WR: {wr:.1f}%",
            fontsize=11, color=color, pad=7, fontweight="bold"
        )
        ax.set_ylabel("Multiplicador de Equity (1.0 = capital inicial)", color=_GRAY, fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.2f}×"))

        if has_dates:
            _date_fmt(ax)
        else:
            ax.set_xlabel("Número de Trade", color=_GRAY, fontsize=9)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"T{int(v):,}"))

        ax.legend(fontsize=8, framealpha=0.2, facecolor=_PANEL, edgecolor=_BORDER,
                  labelcolor="white", loc="upper left")
        ax.grid(alpha=_GRIDΑ)
        _caption(ax,
                 "Crecimiento del capital acumulativo. Línea verde = Luna V1. Línea amarilla punteada = BTC B&H normalizado al mismo capital inicial."
                 " Si Luna V1 supera al BTC B&H el sistema añade alpha real al mercado.")
        _style_ax(ax)

    # ── Subplot 2: Drawdown ───────────────────────────────────────────────────

    def _plot_drawdown(self, ax, dd: pd.Series, ret: pd.Series):
        x = dd.index if isinstance(dd.index, pd.DatetimeIndex) else range(len(dd))
        has_dates = isinstance(dd.index, pd.DatetimeIndex)

        dd_pct = dd.values * 100
        ax.fill_between(x, dd_pct, 0, color=_RED, alpha=0.40)
        ax.plot(x, dd_pct, color=_RED, linewidth=1.0)

        max_dd = dd_pct.min()
        ax.axhline(max_dd, color=_YELLOW, linewidth=0.8, linestyle=":", alpha=0.8)
        ax.text(x[0] if not has_dates else x[int(len(x) * 0.01)],
                max_dd * 1.05, f"Max DD = {max_dd:.1f}%",
                color=_YELLOW, fontsize=7.5, va="top")

        ax.set_title(f"Underwater Plot — Max Drawdown: {max_dd:.1f}%",
                     fontsize=11, color=_RED, pad=6)
        ax.set_ylabel("Drawdown (%)", color=_GRAY, fontsize=9)
        _pct_fmt(ax, 1)

        if has_dates:
            _date_fmt(ax)
        else:
            ax.set_xlabel("Número de Trade", color=_GRAY, fontsize=9)

        ax.grid(alpha=_GRIDΑ)
        _caption(ax,
                 "Caída porcentual desde el máximo previo (peak-to-trough)."
                 " Una DD de -8.5% significa que el capital llegó a estar un 8.5% por"
                 " debajo de su máximo histórico. SOP límite: <20%.")
        _style_ax(ax)

    # ── Subplot 3: Rolling Sharpe ─────────────────────────────────────────────

    def _plot_rolling_sharpe(self, ax, ret: pd.Series, window: int = 30):
        roll = self._rolling_sharpe(ret, window).fillna(0)
        x = roll.index if isinstance(roll.index, pd.DatetimeIndex) else range(len(roll))
        has_dates = isinstance(roll.index, pd.DatetimeIndex)
        vals = roll.values

        # Colored line segments
        for i in range(1, len(vals)):
            c = _GREEN if vals[i] >= 0 else _RED
            xi = [x[i - 1], x[i]]
            ax.plot(xi, [vals[i - 1], vals[i]], color=c, linewidth=1.2, alpha=0.85)

        ax.fill_between(x, vals, 0, where=vals >= 0, color=_GREEN, alpha=0.05)
        ax.fill_between(x, vals, 0, where=vals < 0,  color=_RED,   alpha=0.08)

        ax.axhline(0,    color=_WHITE,  linestyle="--", alpha=0.3, linewidth=0.8)
        ax.axhline(1.5,  color=_GREEN,  linestyle=":",  alpha=0.5, linewidth=0.8,
                   label="SOP mín (1.5)")
        ax.axhline(-1.5, color=_RED,    linestyle=":",  alpha=0.5, linewidth=0.8,
                   label="Zona mala (<-1.5)")

        # Clamped Y view para no distorsionar cuando hay picos en test mode
        q95 = np.nanpercentile(np.abs(vals), 95)
        ylim = max(q95 * 1.2, 3.0)
        ax.set_ylim(-ylim, ylim)

        ax.set_title(f"Rolling {window}-Trade Sharpe Ratio",
                     fontsize=11, color=_YELLOW, pad=6)
        ax.set_ylabel(f"Sharpe (ventana {window} trades)", color=_GRAY, fontsize=9)

        if has_dates:
            _date_fmt(ax)
        else:
            ax.set_xlabel("Número de Trade", color=_GRAY, fontsize=9)

        ax.legend(fontsize=7, framealpha=0.2, facecolor=_PANEL,
                  edgecolor=_BORDER, labelcolor="white")
        # M3: Rolling Win Rate overlay (eje Y derecho)
        ax2 = ax.twinx()
        _roll_wr = (ret > 0).rolling(window).mean() * 100
        ax2.plot(x, _roll_wr.fillna(50).values, color=_PURPLE, linewidth=1.1,
                 alpha=0.55, linestyle=":", zorder=2, label=f"WR {window}T")
        ax2.axhline(50, color=_PURPLE, linewidth=0.4, linestyle=":", alpha=0.3)
        ax2.set_ylim(15, 85)
        ax2.set_ylabel(f"WR% ({window}T)", color=_PURPLE, fontsize=7)
        ax2.tick_params(colors=_PURPLE, labelsize=7, right=True)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        _style_ax(ax2)
        ax.grid(alpha=_GRIDΑ)
        _caption(ax,
                 f"Sharpe rolling {window}T (izq, verde/rojo) y Win Rate rolling {window}T (der, lila punteada)."
                 " Win Rate por encima de 50% indica alpha consistente en esa ventana. SOP Sharpe mínimo: 1.5.")
        _style_ax(ax)

    # ── Subplot NEW: Retorno por Trade (no acumulado) ─────────────────────────

    def _plot_trade_returns(self, ax, ret: pd.Series, trades_df: pd.DataFrame):
        """Barras de retorno individual por trade, coloreadas win/loss. No acumulado."""
        has_dates = isinstance(ret.index, pd.DatetimeIndex)
        vals = ret.values * 100  # en porcentaje
        n = len(vals)
        x = np.arange(n)

        # Barras coloreadas por resultado
        colors = [_GREEN if v > 0 else _RED for v in vals]
        ax.bar(x, vals, color=colors, alpha=0.75, width=0.8, zorder=2)

        # Media móvil 15 trades
        win_size = min(15, max(3, n // 10))
        ma = pd.Series(vals).rolling(win_size, center=True).mean()
        ax.plot(x, ma.values, color=_YELLOW, linewidth=1.4, alpha=0.9,
                label=f"MA{win_size} trades", zorder=3)

        # Línea de 0
        ax.axhline(0, color=_WHITE, linewidth=0.7, linestyle="--", alpha=0.4)

        # Expectancy horizontal
        exp_val = np.mean(vals)
        ax.axhline(exp_val, color=_BLUE, linewidth=1.0, linestyle=":", alpha=0.7,
                   label=f"Expectancy: {exp_val:+.2f}%")

        # Anotación ganancia/pérdida máxima
        max_idx, min_idx = int(np.argmax(vals)), int(np.argmin(vals))
        ax.annotate(f"+{vals[max_idx]:.1f}%", xy=(max_idx, vals[max_idx]),
                    xytext=(max_idx, vals[max_idx] + np.ptp(vals) * 0.05),
                    color=_GREEN, fontsize=7, ha="center",
                    arrowprops=dict(arrowstyle="-", color=_GREEN, lw=0.6))
        ax.annotate(f"{vals[min_idx]:.1f}%", xy=(min_idx, vals[min_idx]),
                    xytext=(min_idx, vals[min_idx] - np.ptp(vals) * 0.05),
                    color=_RED, fontsize=7, ha="center",
                    arrowprops=dict(arrowstyle="-", color=_RED, lw=0.6))

        # Etiquetas fecha en x si tiene timestamps (cada N ticks)
        if has_dates and len(ret.index) > 0:
            date_strs = [t.strftime("%d/%m") for t in ret.index]
            step = max(1, n // 10)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(date_strs[::step], rotation=30, ha="right",
                               fontsize=7, color=_GRAY)
        else:
            ax.set_xlabel("Número de Trade", color=_GRAY, fontsize=9)

        avg_w = float(np.mean(vals[vals > 0])) if (vals > 0).any() else 0.0
        avg_l = float(np.mean(vals[vals <= 0])) if (vals <= 0).any() else 0.0
        ratio = abs(avg_w / avg_l) if avg_l != 0 else float("inf")

        ax.set_title(
            f"Retorno por Trade (no acumulado)  |  "
            f"Avg Win: +{avg_w:.2f}%  |  Avg Loss: {avg_l:.2f}%  |  R:R = {ratio:.2f}x",
            fontsize=11, color=_BLUE, pad=7, fontweight="bold"
        )
        ax.set_ylabel("Retorno por Trade (%)", color=_GRAY, fontsize=9)
        ax.legend(fontsize=7, framealpha=0.2, facecolor=_PANEL,
                  edgecolor=_BORDER, labelcolor="white")
        ax.grid(alpha=_GRIDΑ)
        _caption(ax,
                 "Retorno individual de cada trade sin efecto de capitalización acumulada."
                 " Verde = ganador, Rojo = perdedor. La línea amarilla es la media móvil"
                 " de retornos. La línea azul es la expectancy promedio por trade.")
        _style_ax(ax)

    # ── Subplot NEW: Stats Box ────────────────────────────────────────────────

    def _plot_stats_box(self, ax, ret: pd.Series, verdict: dict):
        """Tabla de métricas clave complementarias no visibles en los otros plots."""
        ax.set_axis_off()
        ax.set_facecolor(_PANEL)

        vals = ret.values * 100
        wins_arr  = vals[vals > 0]
        loss_arr  = vals[vals <= 0]
        avg_w = float(np.mean(wins_arr)) if len(wins_arr) > 0 else 0.0
        avg_l = float(np.mean(loss_arr)) if len(loss_arr) > 0 else 0.0
        ratio = abs(avg_w / avg_l) if avg_l != 0 else float("inf")
        wr = len(wins_arr) / len(vals) * 100 if len(vals) > 0 else 0.0
        expectancy = (wr / 100 * avg_w) + (1 - wr / 100) * avg_l
        std_ret = float(np.std(vals))
        skew    = float(pd.Series(vals).skew())
        kurt    = float(pd.Series(vals).kurt())

        # Consecutive wins / losses
        def _max_consec(mask):
            best = cur = 0
            for v in mask:
                cur = cur + 1 if v else 0
                best = max(best, cur)
            return best

        max_cw = _max_consec(vals > 0)
        max_cl = _max_consec(vals <= 0)

        # Calmar: return total / MaxDD
        cum = (1 + ret).cumprod()
        max_dd_pct = self._compute_drawdowns(cum).min() * 100
        total_ret_pct = (cum.iloc[-1] - 1) * 100
        calmar = abs(total_ret_pct / max_dd_pct) if max_dd_pct != 0 else float("inf")

        # SFI from verdict
        sfi = verdict.get("statistical_audit", {}).get("brier_score")
        dsr = verdict.get("statistical_audit", {}).get("dsr") or 0.0
        pbo = verdict.get("statistical_audit", {}).get("estimated_pbo") or 0.0

        # OOS Health from verdict
        oos_health = verdict.get("oos_health", {})
        cusum = oos_health.get("cusum_max_drift")
        sharpe_rec = oos_health.get("sharpe_decay_recent")

        rows = [
            ("── Rentabilidad",    "",               False),
            ("Avg Win",            f"+{avg_w:.2f}%",  avg_w > 0),
            ("Avg Loss",           f"{avg_l:.2f}%",   False),
            ("Ratio R:R",          f"{ratio:.2f}×",   ratio >= 1.5),
            ("Expectancy/trade",   f"{expectancy:+.2f}%", expectancy > 0),
            ("── Riesgo",          "",               False),
            ("Max Consec Wins",    str(max_cw),       max_cw >= 5),
            ("Max Consec Losses", str(max_cl),       max_cl <= 5),
            ("Calmar Ratio",       f"{calmar:.2f}",   calmar >= 2.0),
            ("── Estadístico",     "",               False),
            ("Skewness",           f"{skew:+.2f}",    skew > 0),
            ("DSR (Bailey)",       f"{dsr:.4f}",      dsr >= 0.75),
            ("PBO (CSCV)",         f"{pbo*100:.1f}%", pbo < 0.10),
            ("── OOS Health",      "",               False),
            ("CUSUM Max Drift",    f"{cusum:.2f}" if cusum is not None else "N/A", cusum is not None and cusum <= 4.5),
            ("Sharpe Decay (2W)",  f"{sharpe_rec:.2f}" if sharpe_rec is not None else "N/A", sharpe_rec is not None and sharpe_rec >= -2.0),
        ]

        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(rows))
        ax.set_title("Métricas Complementarias", fontsize=11,
                     color=_BLUE, pad=6, fontweight="bold")

        for i, (label, value, is_good) in enumerate(reversed(rows)):
            y = i + 0.5
            is_header = label.startswith("──")
            lbl_color = _YELLOW if is_header else _GRAY
            val_color = _GREEN if (is_good and not is_header) else \
                        (_RED   if (not is_good and not is_header and value) else _GRAY)
            fsize = 8.5 if not is_header else 7.5
            fw    = "bold" if is_header else "normal"

            ax.text(0.02, y, label, color=lbl_color, fontsize=fsize,
                    va="center", fontweight=fw, transform=ax.transData)
            if value:
                ax.text(0.98, y, value, color=val_color, fontsize=fsize,
                        va="center", ha="right", fontweight="bold",
                        transform=ax.transData)
            if is_header:
                ax.axhline(y - 0.4, color=_BORDER, linewidth=0.5, alpha=0.5)

    # ── Subplot 4: Monthly heatmap or Win/Loss bars ───────────────────────────

    def _plot_monthly_heatmap(self, ax, ret: pd.Series):
        has_dates = isinstance(ret.index, pd.DatetimeIndex)
        n_months = 0

        if has_dates:
            try:
                monthly = ret.resample("ME").apply(lambda x: (np.prod(1 + x) - 1) * 100)
                monthly.index = monthly.index.to_period("M")
                n_months = len(monthly)
            except Exception:
                pass

        if n_months >= 2:
            df_m = pd.DataFrame({
                "Year":  monthly.index.year,
                "Month": monthly.index.month,
                "Ret":   monthly.values,
            })
            pivot = df_m.pivot(index="Year", columns="Month", values="Ret")
            MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                      "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
            pivot.columns = [MONTHS[c - 1] for c in pivot.columns]
            cmap = LinearSegmentedColormap.from_list("luna", [_RED, "#1a1a2e", _GREEN])
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap=cmap, center=0,
                        ax=ax, cbar=False,
                        annot_kws={"size": 8, "color": "white"},
                        linewidths=0.5, linecolor=_BORDER)
            ax.set_title("Monthly Returns Heatmap (%)",
                         fontsize=11, color=_BLUE, pad=6)
            ax.set_xlabel("Mes", color=_GRAY, fontsize=9)
            ax.set_ylabel("Año", color=_GRAY, fontsize=9)
            _caption(ax,
                     "Retorno total (%) agrupado por mes y año. Verde = mes rentable,"
                     " rojo = mes con pérdidas. Permite detectar estacionalidad y meses"
                     " sistemáticamente malos.")
        else:
            # Fallback: Win vs Loss histogram superpuesto
            wins   = ret[ret > 0] * 100
            losses = ret[ret <= 0] * 100
            bins   = np.linspace(ret.min() * 100, ret.max() * 100, 30)
            ax.hist(wins.values,   bins=bins, color=_GREEN, alpha=0.70,
                    label=f"Wins   ({len(wins)} trades)")
            ax.hist(losses.values, bins=bins, color=_RED,   alpha=0.70,
                    label=f"Losses ({len(losses)} trades)")
            ax.axvline(0, color=_WHITE, linewidth=0.9, linestyle="--", alpha=0.5)
            ax.set_xlabel("Retorno por trade (%)", color=_GRAY, fontsize=9)
            ax.set_ylabel("Número de trades", color=_GRAY, fontsize=9)
            ax.legend(fontsize=8, framealpha=0.2, facecolor=_PANEL,
                      edgecolor=_BORDER, labelcolor="white")
            ax.set_title("Distribución Win vs. Loss",
                         fontsize=11, color=_BLUE, pad=6)
            _caption(ax,
                     "Histograma separado de trades ganadores (verde) y perdedores (rojo)."
                     " Se muestra cuando el período de datos es < 2 meses."
                     " Idealmente los wins deben tener cola derecha más larga (mayor R:R).")
            ax.grid(alpha=_GRIDΑ)

        _style_ax(ax)

    # ── Subplot 5: Returns distribution ──────────────────────────────────────

    def _plot_returns_dist(self, ax, ret: pd.Series):
        r = ret.values * 100
        sns.histplot(r, bins=50, kde=True, color=_BLUE, ax=ax,
                     alpha=0.50, line_kws={"linewidth": 1.4, "color": _PURPLE})

        mean_r  = np.mean(r)
        median_r = np.median(r)
        var95   = np.percentile(r, 5)
        std_r   = np.std(r)

        ax.axvline(mean_r,   color=_YELLOW, linestyle="--", linewidth=1.2, zorder=4)
        ax.axvline(median_r, color=_BLUE,   linestyle=":",  linewidth=1.0, zorder=4)
        ax.axvline(var95,    color=_RED,     linestyle=":",  linewidth=1.1, zorder=4)
        ax.axvline(0,        color=_GRAY,    linestyle="--", linewidth=0.7, alpha=0.5)

        ymax = ax.get_ylim()[1]
        for val, label, col, offset in [
            (mean_r,  f"Media\n{mean_r:.2f}%",   _YELLOW, 0.90),
            (var95,   f"VaR95\n{var95:.2f}%",    _RED,    0.75),
            (median_r,f"Med.\n{median_r:.2f}%",  _BLUE,   0.60),
        ]:
            ax.text(val, ymax * offset, label,
                    color=col, fontsize=6.5, ha="center", va="top",
                    bbox=dict(facecolor=_BG, edgecolor="none", alpha=0.7, pad=1))

        # Stats box
        stats_txt = f"N={len(r):,}  |  std={std_r:.2f}%  |  Skew={pd.Series(r).skew():.2f}"
        ax.text(0.99, 0.98, stats_txt, transform=ax.transAxes,
                fontsize=7, color=_GRAY, ha="right", va="top")

        ax.set_xlabel("Retorno por trade (%)", color=_GRAY, fontsize=9)
        ax.set_ylabel("Número de trades", color=_GRAY, fontsize=9)
        ax.set_title("Distribución de Retornos por Trade",
                     fontsize=11, color=_BLUE, pad=6)
        ax.grid(alpha=_GRIDΑ)
        _caption(ax,
                 "Distribución estadística de los retornos individuales (%).  "
                 "VaR95 = peor retorno esperado el 5% del tiempo.  "
                 "Una distribución con cola derecha > cola izquierda indica buen ratio beneficio/riesgo.")
        _style_ax(ax)

    # ── Subplot 6: Gate Semaphore ─────────────────────────────────────────────

    def _plot_gate_semaphore(self, ax, verdict: dict):
        """Gate Status con mini progress bar + threshold marker por row."""
        flags   = verdict.get("flags", {})
        metrics = verdict.get("metrics", {})
        audit   = verdict.get("statistical_audit", {})
        thresh  = verdict.get("sop_thresholds", {})

        # (name, passed, value_raw, threshold_raw, scale_max, higher_better, disp_val, disp_thr)
        n_tr   = metrics.get("total_trades", 0)
        min_tr = thresh.get("min_trades", 100)
        wr     = metrics.get("win_rate", 0)
        # Calculate dynamic win rate threshold passing p-value
        alpha  = thresh.get("alpha_binomial", 0.05)  # [FIX-07] Fallback corregido 0.15→0.05 (α estándar 5%)
        print(f"[FIX-07] Gate semaphore: alpha_binomial={alpha:.3f} (fuente: sop_thresholds={'alpha_binomial' in thresh})")
        min_wins = binom.ppf(1 - alpha, max(n_tr, 1), 0.5)
        min_wr = min_wins / max(n_tr, 1)

        dsr    = audit.get("dsr", 0)
        min_dsr = thresh.get("min_dsr", 0.75)
        pbo    = audit.get("estimated_pbo", 1.0)
        max_pbo = thresh.get("max_pbo_pct", 10) / 100
        dd     = metrics.get("max_drawdown_pct", 100)
        max_dd_t = thresh.get("max_drawdown_pct", 20)
        p_val  = audit.get("binomial_p_value", 1.0)

        gate_defs = [
            # (name, passed, value, threshold, scale_max, higher_better, disp_val, disp_thr)
            ("TRADES",   flags.get("pass_trades", False),
             n_tr,         min_tr,         max(n_tr * 1.5, min_tr * 2), True,
             f"{n_tr:,}", f">= {min_tr:,}"),
            ("WIN RATE", wr > min_wr,
             wr,           min_wr,         1.0, True,
             f"{wr*100:.1f}%", f"> {min_wr*100:.1f}%"),
            ("BINOMIAL", flags.get("pass_binomial", False),
             p_val,        alpha,           1.0, False,
             f"p={p_val:.3f}", f"<= {alpha:.2f}"),
            ("DSR",      flags.get("pass_dsr", False),
             dsr,          min_dsr,        1.0, True,
             f"{dsr:.4f}", f">= {min_dsr}"),
            ("PBO",      flags.get("pass_pbo", False),
             pbo,          max_pbo,        0.50, False,    # lower is better
             f"{pbo*100:.1f}%", f"< {max_pbo*100:.0f}%"),
            ("MAX DD",   flags.get("pass_dd", False),
             dd / 100,     max_dd_t / 100, 0.50, False,   # lower is better
             f"{dd:.1f}%", f"< {max_dd_t:.0f}%"),
        ]

        n_pass = sum(1 for _, p, *_ in gate_defs if p)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.9, len(gate_defs) - 0.1)
        ax.axis("off")
        ax.set_title(f"Gate Status — SOP Thresholds ({n_pass}/{len(gate_defs)} cumplidos)",
                     fontsize=11, color=_WHITE, pad=6, fontweight="bold")

        # Layout X zones
        BADGE_X0, BADGE_W = 0.00, 0.09
        NAME_X           = 0.11
        BAR_X0, BAR_W    = 0.32, 0.38
        VAL_X            = 0.72
        _LABEL_X         = 0.72  # [FIX-08] Renombrado de THR_X: es coordenada X de layout, NO un threshold del modelo

        for i, (name, passed, value, thr, scale, higher, lval, lthr) in \
                enumerate(reversed(gate_defs)):
            col   = _GREEN if passed else _RED
            label = "PASS" if passed else "FAIL"
            y     = i

            # ── Fila background ──────────────────────────────────────────────
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.0, y - 0.35), 1.0, 0.70,
                boxstyle="round,pad=0.015",
                facecolor=col, alpha=0.08,
                edgecolor=col, linewidth=0.5))

            # ── Badge [PASS/FAIL] ─────────────────────────────────────────────
            ax.add_patch(mpatches.FancyBboxPatch(
                (BADGE_X0, y - 0.28), BADGE_W, 0.56,
                boxstyle="round,pad=0.01",
                facecolor=col, alpha=0.85, edgecolor="none"))
            ax.text(BADGE_X0 + BADGE_W / 2, y, label,
                    va="center", ha="center",
                    fontsize=6.5, color=_BG if passed else _WHITE,
                    fontweight="bold")

            # ── Gate name ────────────────────────────────────────────────────
            ax.text(NAME_X, y, name,
                    va="center", ha="left",
                    fontsize=10, color=col, fontweight="bold")

            # ── Mini progress bar ─────────────────────────────────────────────
            # normalize value and threshold to [0, scale]
            norm_v = min(max(value, 0), scale) / scale  # 0..1
            norm_t = min(thr, scale) / scale            # 0..1

            # background track
            ax.add_patch(mpatches.Rectangle(
                (BAR_X0, y - 0.14), BAR_W, 0.28,
                facecolor="#21262d", edgecolor=_BORDER, linewidth=0.5))
            # fill
            ax.add_patch(mpatches.Rectangle(
                (BAR_X0, y - 0.14), BAR_W * norm_v, 0.28,
                facecolor=col, alpha=0.68))

            # threshold vertical line + diamond (same visual as Gauges)
            thr_px = BAR_X0 + BAR_W * norm_t
            ax.plot([thr_px, thr_px], [y - 0.26, y + 0.26],
                    color=_YELLOW, linewidth=2.0, zorder=5,
                    solid_capstyle="round")
            ax.plot([thr_px], [y], marker="D",
                    color=_YELLOW, markersize=5, zorder=6,
                    markeredgecolor=_BG, markeredgewidth=0.5)

            # ── Valor y umbral (derecha) ──────────────────────────────────────
            ax.text(VAL_X + 0.27, y + 0.16, lval,
                    va="center", ha="right",
                    fontsize=10.5, color=_WHITE, fontweight="bold")
            ax.text(_LABEL_X + 0.27, y - 0.20, f"Umbral: {lthr}",
                    va="center", ha="right",
                    fontsize=7.0, color=_GRAY)

        # Eje X de referencia (0%..100%) debajo de todas las barras
        ref_y = -0.72
        ax.plot([BAR_X0, BAR_X0 + BAR_W], [ref_y, ref_y],
                color=_BORDER, linewidth=0.8)
        for frac, label in [(0, "0"), (0.5, "50%"), (1.0, "100%")]:
            ax.text(BAR_X0 + BAR_W * frac, ref_y - 0.08, label,
                    ha="center", fontsize=6.5, color=_GRAY)

        # Leyenda del threshold marker (coherente con Gauges)
        from matplotlib.lines import Line2D
        mkr = Line2D([0], [0], marker="D", color=_YELLOW,
                     linestyle="-", linewidth=1.5, markersize=5,
                     label="Umbral SOP")
        ax.legend(handles=[mkr], loc="lower right",
                  fontsize=7, framealpha=0.2,
                  facecolor=_PANEL, edgecolor=_BORDER, labelcolor="white")

        _caption(ax,
                 "Los 5 gates del Gauntlet SOP deben cumplirse TODOS para aprobar el deploy. "
                 "La barra muestra el valor actual; el marcador amarillo [diamante] es el umbral mínimo/máximo. "
                 "PASS = verde; FAIL = rojo.")

    # ── Subplot 7: Metric Gauges ──────────────────────────────────────────────

    def _plot_audit_gauges(self, ax, verdict: dict):
        audit   = verdict.get("statistical_audit", {})
        metrics = verdict.get("metrics", {})
        thresh  = verdict.get("sop_thresholds", {})
        
        n_tr = metrics.get("total_trades", 0)
        alpha = thresh.get("alpha_binomial", 0.05)  # [FIX-07] Fallback corregido 0.15→0.05 (α estándar 5%)
        print(f"[FIX-07] Audit gauges: alpha_binomial={alpha:.3f} (fuente: sop_thresholds={'alpha_binomial' in thresh})")
        min_wins = binom.ppf(1 - alpha, max(n_tr, 1), 0.5)
        min_wr = min_wins / max(n_tr, 1)

        items = [
            # (label, value, threshold, scale_max, higher_better, display_val, display_thr)
            ("DSR",
             float(audit.get("dsr", 0)), thresh.get("min_dsr", 0.75),
             1.0, True,
             f"{audit.get('dsr', 0):.4f}", f">= {thresh.get('min_dsr', 0.75)}"),
            ("WIN RATE",
             float(metrics.get("win_rate", 0)), min_wr,
             1.0, True,
             f"{metrics.get('win_rate', 0)*100:.1f}%", f"> {min_wr*100:.1f}%"),
            ("PBO",
             float(audit.get("estimated_pbo", 1.0)),
             thresh.get("max_pbo_pct", 10) / 100,
             1.0, False,
             f"{audit.get('estimated_pbo', 1.0)*100:.1f}%",
             f"< {thresh.get('max_pbo_pct', 10):.0f}%"),
        ]

        ax.set_xlim(-0.25, 1.20)
        ax.set_ylim(-0.9, len(items) - 0.1)
        ax.axis("off")
        ax.set_title("Métricas Clave vs Umbrales SOP",
                     fontsize=11, color=_WHITE, pad=6, fontweight="bold")

        for i, (name, value, thr, scale, higher, lval, lthr) in enumerate(reversed(items)):
            y = i
            passed = (value >= thr) if higher else (value <= thr)
            col    = _GREEN if passed else _RED
            norm_v = min(max(value, 0), scale)
            norm_t = min(thr, scale)

            # Fondo sombreado de fila (igual que Gate Status)
            ax.add_patch(mpatches.FancyBboxPatch(
                (-0.25, y - 0.38), 1.45, 0.76,
                boxstyle="round,pad=0.015",
                facecolor=col, alpha=0.08,
                edgecolor=col, linewidth=0.5,
                zorder=0))

            # Track background
            ax.barh(y, 1.0, height=0.40, color="#21262d", left=0, zorder=1)
            # Valor
            ax.barh(y, norm_v, height=0.40, color=col, alpha=0.70, left=0, zorder=2)
            # Threshold marker
            ax.plot([norm_t, norm_t], [y - 0.28, y + 0.28],
                    color=_YELLOW, linewidth=2.5, zorder=4, solid_capstyle="round")
            ax.scatter([norm_t], [y], color=_YELLOW, s=55,
                       marker="D", zorder=5, edgecolors=_BG, linewidth=0.5)

            # Label izquierda
            ax.text(-0.02, y, name, va="center", ha="right",
                    fontsize=9.5, color=_WHITE, fontweight="bold")
            # Valor derecha (grande)
            ax.text(1.03, y + 0.15, lval, va="center", ha="left",
                    fontsize=10, color=col, fontweight="bold")
            ax.text(1.03, y - 0.18, lthr, va="center", ha="left",
                    fontsize=7.5, color=_GRAY)

            # Descripción breve de la métrica (dentro de la barra, si cabe)
            desc = {"DSR": "Deflated Sharpe Ratio",
                    "WIN RATE": "Porcentaje de trades ganadores",
                    "PBO": "Prob. de Overfitting (CSCV)"}.get(name, "")
            ax.text(0.01, y + 0.18, desc, va="center", ha="left",
                    fontsize=6.5, color=_BG if passed else _WHITE, alpha=0.85)

        # Eje X de porcentaje al fondo
        ax.text(0.0,  -0.7, "0%", ha="left",  fontsize=7, color=_GRAY)
        ax.text(0.5,  -0.7, "50%", ha="center", fontsize=7, color=_GRAY)
        ax.text(1.0,  -0.7, "100%", ha="right", fontsize=7, color=_GRAY)
        ax.plot([0, 1], [-0.6, -0.6], color=_BORDER, linewidth=0.8)

        # Leyenda del threshold marker
        marker_legend = Line2D([0], [0], marker="D", color=_YELLOW,
                               linestyle="-", linewidth=1.5, markersize=6,
                               label="Umbral SOP")
        ax.legend(handles=[marker_legend], loc="lower right",
                  fontsize=7, framealpha=0.2, facecolor=_PANEL,
                  edgecolor=_BORDER, labelcolor="white")

        _caption(ax,
                 "Barras de progreso de las 3 métricas estadísticas principales."
                 " El marcador amarillo [diamante] indica el umbral SOP mínimo/máximo."
                 " DSR mide si la señal es real (>0.75). PBO mide overfitting (<10%).")

    # ── Panel A: XGB Prob Cuartiles ──────────────────────────────────────────

    def _plot_xgb_quartiles(self, ax, trades_df: pd.DataFrame):
        """
        HALLAZGO-2 (M-36): muestra el WR por cuartil de probabilidad XGBoost.
        Si la señal está invertida (Q4 WR < Q1 WR) es firma de sobreajuste.
        """
        if "xgb_prob" not in trades_df.columns or "is_win" not in trades_df.columns:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "xgb_prob no disponible",
                    ha="center", va="center", color=_GRAY, fontsize=10,
                    transform=ax.transAxes)
            ax.set_title("XGB Prob Cuartiles — N/A", fontsize=11, color=_GRAY, pad=6)
            return

        df = trades_df.copy()
        try:
            df["q"] = pd.qcut(df["xgb_prob"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        except ValueError:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Datos insuficientes para cuartiles",
                    ha="center", va="center", color=_GRAY, fontsize=10,
                    transform=ax.transAxes)
            return

        results = []
        for q_label in ["Q1", "Q2", "Q3", "Q4"]:
            g = df[df["q"] == q_label]
            wr_q   = (g["is_win"] == True).mean() if len(g) > 0 else 0.0
            n_q    = len(g)
            prob_m = g["xgb_prob"].median() if len(g) > 0 else 0.0
            results.append((q_label, wr_q, n_q, prob_m))

        labels   = [f"{r[0]}\n(p~{r[3]:.2f})" for r in results]
        wrs      = [r[1] * 100 for r in results]
        ns       = [r[2] for r in results]
        colors_b = [_GREEN if w >= 50 else _RED for w in wrs]

        xs = np.arange(len(results))
        bars = ax.bar(xs, wrs, color=colors_b, alpha=0.72, width=0.55, zorder=2)

        # Línea 50% (random)
        ax.axhline(50.0, color=_YELLOW, linewidth=1.0, linestyle="--", alpha=0.7,
                   label="50% (azar)")

        # Anotaciones: WR% y n sobre cada barra
        for bar, (ql, wr_q, n_q, _) in zip(bars, results):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{wr_q*100:.1f}%\n(n={n_q})",
                    ha="center", va="bottom", fontsize=8, color=_WHITE, fontweight="bold")

        # Monotonía: check si la señal es monotónica (buena) o invertida (mala)
        is_monotonic = all(wrs[i] <= wrs[i+1] for i in range(len(wrs)-1)) or \
                       all(wrs[i] >= wrs[i+1] for i in range(len(wrs)-1))
        is_inverted  = wrs[0] > wrs[-1]  # Q1 mejor que Q4 → sobreajuste
        signal_qual  = ("✓ Monotónica" if is_monotonic and not is_inverted else
                        ("⚠ Invertida — sobreajuste" if is_inverted else "~ Irregular"))
        signal_col   = _GREEN if (is_monotonic and not is_inverted) else \
                       (_RED if is_inverted else _YELLOW)

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, color=_GRAY, fontsize=9)
        ax.set_ylabel("Win Rate (%)", color=_GRAY, fontsize=9)
        ax.set_ylim(0, max(wrs) * 1.25 + 5)
        ax.set_title(
            f"XGB Prob Cuartiles — WR por Confianza  |  Señal: {signal_qual}",
            fontsize=11, color=signal_col, pad=6, fontweight="bold"
        )
        ax.legend(fontsize=8, framealpha=0.2, facecolor=_PANEL,
                  edgecolor=_BORDER, labelcolor="white")
        ax.grid(alpha=_GRIDΑ, axis="y")
        _caption(ax,
                 "Win Rate por cuartil de probabilidad XGBoost. Q1=menor confianza, Q4=mayor confianza.\n"
                 "Una señal SANA es monotónica (WR sube con la confianza).\n"
                 "Señal INVERTIDA (Q4 WR < Q1 WR) = sobreajuste: el modelo está más convicto cuando se equivoca.")
        _style_ax(ax)

    # ── Panel B: Holding Time Distribution ───────────────────────────────────

    def _plot_holding_time(
        self,
        ax,
        trades_df: pd.DataFrame,
        vertical_barrier_hours: float = 168.0,
    ):
        """
        HALLAZGO-1 (M-36): distribución de holding time en horas.
        Si la mayoría de trades se cierran cerca de la barrera vertical,
        el R:R real colapsa porque el PT no se alcanza a tiempo.
        """
        entry_col = next(
            (c for c in trades_df.columns if any(k in c.lower() for k in ["entry", "t_in", "open_time"])),
            None
        )
        exit_col = next(
            (c for c in trades_df.columns if any(k in c.lower() for k in ["exit", "t_out", "close_time"])),
            None
        )

        # Intentar reconstruir desde timestamp si no hay entry/exit explícitos
        if (entry_col is None or exit_col is None) and "timestamp" in trades_df.columns:
            entry_col = exit_col = None

        if entry_col and exit_col:
            hold_hours = (
                pd.to_datetime(trades_df[exit_col]) -
                pd.to_datetime(trades_df[entry_col])
            ).dt.total_seconds() / 3600
        else:
            ax.set_axis_off()
            ax.text(0.5, 0.5,
                    "Holding time no disponible\n(añadir entry_time/exit_time a oos_trades)",
                    ha="center", va="center", color=_GRAY, fontsize=9,
                    transform=ax.transAxes)
            ax.set_title(
                f"Holding Time — N/A  |  Barrera vertical: {vertical_barrier_hours:.0f}H",
                fontsize=11, color=_GRAY, pad=6
            )
            return

        hold_hours = hold_hours.dropna()
        # Buckets de 24H con np.histogram + bar() manual (matplotlib no acepta color array en hist)
        max_h  = max(hold_hours.max(), vertical_barrier_hours * 1.1)
        bins   = np.arange(0, max_h + 24, 24)
        counts, edges = np.histogram(hold_hours.values, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        widths  = np.diff(edges) * 0.85

        colors_h = [
            _RED if (edges[i] >= vertical_barrier_hours - 12) else _BLUE
            for i in range(len(counts))
        ]
        ax.bar(centers, counts, width=widths, color=colors_h,
               alpha=0.75, edgecolor=_BORDER, linewidth=0.4, zorder=2)


        # Marcador de barrera vertical
        ax.axvline(vertical_barrier_hours,
                   color=_YELLOW, linewidth=1.8, linestyle="--", alpha=0.9,
                   label=f"Barrera vertical ({vertical_barrier_hours:.0f}H)", zorder=4)

        # % que llega a barrera
        n_vertical = (hold_hours >= vertical_barrier_hours * 0.95).sum()
        pct_vert   = n_vertical / len(hold_hours) * 100
        ax.text(vertical_barrier_hours + 2,
                ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1,
                f"{pct_vert:.1f}%\nen barrera",
                color=_YELLOW, fontsize=8, va="center", fontweight="bold")

        ax.set_xlabel("Horas hasta cierre del trade", color=_GRAY, fontsize=9)
        ax.set_ylabel("Número de trades", color=_GRAY, fontsize=9)
        ax.set_title(
            f"Holding Time  |  Media: {hold_hours.mean():.1f}H  |  "
            f"Barrera: {vertical_barrier_hours:.0f}H  |  "
            f"{pct_vert:.1f}% cierran en barrera",
            fontsize=11, color=_BLUE, pad=6, fontweight="bold"
        )
        ax.legend(fontsize=8, framealpha=0.2, facecolor=_PANEL,
                  edgecolor=_BORDER, labelcolor="white")
        ax.grid(alpha=_GRIDΑ, axis="y")
        _caption(ax,
                 "Distribución del tiempo que cada trade permanece abierto.\n"
                 "Barras azules = PT o SL alcanzados. Barras rojas = cerrados por barrera vertical (tiempo máximo).\n"
                 "Si >30% cierran en barrera vertical el R:R real colapsa — considerar ampliar la barrera o el PT.")
        _style_ax(ax)

    # ── Panel D: Walk-Forward Validation ──────────────────────────────────────

    def _plot_wfv(self, ax, verdict: dict):
        """M4: Panel Walk-Forward Validation — WR por ventana temporal."""
        wfv = verdict.get("wfv_results", {})
        if not wfv:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "WFV no disponible (insuficientes trades)",
                    ha="center", va="center", color=_GRAY, fontsize=11,
                    transform=ax.transAxes)
            ax.set_title("Walk-Forward Validation — N/A", fontsize=11, color=_GRAY, pad=6)
            return

        keys   = list(wfv.keys())
        wrs    = [(wfv[k].get("win_rate") or 0.0) * 100 for k in keys]
        ns     = [wfv[k].get("n_trades", 0) for k in keys]
        starts = [str(wfv[k].get("start_date", wfv[k].get("start", "?")))[:10] for k in keys]
        ends   = [str(wfv[k].get("end_date",   wfv[k].get("end",   "?")))[:10] for k in keys]

        x          = np.arange(len(keys))
        bar_colors = [_GREEN if w >= 50 else (_YELLOW if w >= 40 else _RED) for w in wrs]
        bars       = ax.bar(x, wrs, color=bar_colors, alpha=0.75, width=0.55, zorder=2)

        ax.axhline(50, color=_YELLOW, linewidth=1.2, linestyle="--", alpha=0.7, label="50% (señal mínima)")
        ax.axhline(40, color=_RED,    linewidth=0.8, linestyle=":",  alpha=0.5, label="40% (zona crítica)")

        for bar, k, wr_v, n_v, s, e in zip(bars, keys, wrs, ns, starts, ends):
            ax.text(bar.get_x() + bar.get_width() / 2, wr_v + 0.8,
                    f"{wr_v:.1f}%\n(n={n_v})",
                    ha="center", va="bottom", fontsize=9, color=_WHITE, fontweight="bold")
            ax.text(bar.get_x() + bar.get_width() / 2, -4.5,
                    f"{s}\n{e}", ha="center", va="top", fontsize=6.5, color=_GRAY)

        # Tendencia: flecha visual si hay degradación W1→W4
        if len(wrs) >= 2:
            trend = wrs[-1] - wrs[0]
            trend_sym  = "▼ Degradando" if trend < -5 else ("▲ Mejorando" if trend > 5 else "→ Estable")
            trend_col  = _RED if trend < -5 else (_GREEN if trend > 5 else _YELLOW)
            ax.text(0.99, 0.97, trend_sym, transform=ax.transAxes,
                    ha="right", va="top", fontsize=10, color=trend_col, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=10, color=_WHITE)
        ax.set_ylabel("Win Rate (%)", color=_GRAY, fontsize=9)
        ax.set_ylim(0, max(max(wrs) if wrs else 60, 60) * 1.25 + 8)
        _pct_fmt(ax, 0)
        _date_fmt_disabled = True  # no-op
        ax.set_title(
            f"Walk-Forward Validation — WR por Ventana Temporal  "
            f"|  {len(keys)} ventanas  |  Media: {np.mean(wrs):.1f}%  "
            f"|  Trend: {wrs[0]:.0f}% → {wrs[-1]:.0f}%" if len(wrs) >= 2 else
            f"Walk-Forward Validation — {len(keys)} ventanas",
            fontsize=11, color=_BLUE, pad=8, fontweight="bold"
        )
        ax.legend(fontsize=8, framealpha=0.2, facecolor=_PANEL, edgecolor=_BORDER, labelcolor="white")
        ax.grid(alpha=_GRIDΑ, axis="y")
        _caption(ax,
                 "Win Rate por ventana temporal (WFV). Verde ≥50% | Amarillo 40-50% | Rojo <40%."
                 " Una degradación W1→W4 indica drift de régimen: la señal no generaliza al período reciente."
                 " Tendencia ▼ Degradando requiere revisión del pipeline antes de deploy.")
        _style_ax(ax)

    # ── Panel E: Timeline Audit ───────────────────────────────────────────────

    def _plot_timeline_audit(self, ax, trades_df: pd.DataFrame):
        """Panel E: Visualización de superposición de datasets (Leakage Audit) y trades distribuidos."""
        try:
            df_train = pd.read_parquet(self.root / "data/features/features_train.parquet", columns=['close'])
            df_val = pd.read_parquet(self.root / "data/features/features_validation.parquet", columns=['close'])
            # AUDIT Tier 3: usar holdout especifico de ventana
            import os as _os_ts_1433
            _win_ts2 = _os_ts_1433.environ.get('LUNA_WINDOW_ID', '')
            _ts_ho = self.root / 'data' / 'features' / (f'features_holdout_{_win_ts2}.parquet' if _win_ts2 else 'features_holdout.parquet')
            if not _ts_ho.exists(): _ts_ho = self.root / 'data' / 'features' / 'features_holdout.parquet'
            df_holdout = pd.read_parquet(_ts_ho, columns=['close'])
            
            df_all = pd.concat([df_train, df_val, df_holdout]).sort_index()
            df_all = df_all[~df_all.index.duplicated(keep='first')]

            ax.plot(df_all.index, df_all['close'] / 1000, color=_GRAY, linewidth=1.2, alpha=0.5)
            
            regions = [
                ("Training (In-Sample)", df_train.index.min(), df_train.index.max(), _BLUE, 0.15),
                ("Validation", df_val.index.min(), df_val.index.max(), _YELLOW, 0.20),
                ("Holdout OOS", df_holdout.index.min(), df_holdout.index.max(), _PURPLE, 0.20)
            ]
            for label, start, end, color, alpha in regions:
                ax.axvspan(start, end, color=color, alpha=alpha, label=f"{label} ({start.strftime('%y-%m')} → {end.strftime('%y-%m')})")
                
            if "timestamp" in trades_df.columns and len(trades_df) > 0:
                ts_list = pd.DatetimeIndex(trades_df["timestamp"])
                if ts_list.tz is None:
                    ts_list = ts_list.tz_localize("UTC")
                else:
                    ts_list = ts_list.tz_convert("UTC")
                
                prices = []
                for t in ts_list:
                    idx = df_all.index.get_indexer([t], method="nearest")
                    prices.append(df_all.iloc[idx[0], 0] / 1000 if idx[0] >= 0 else np.nan)
                
                ax.scatter(ts_list, prices, color=_GREEN, marker='^', s=45, zorder=5, label=f"Trades OOS ({len(trades_df)})")

            ax.set_title("Data Overlap Audit (IS vs VAL vs OOS)  |  "
                         f"Train: {regions[0][1].date()}  |  Val: {regions[1][1].date()}  |  OOS: {regions[2][1].date()}",
                         fontsize=11, color=_BLUE, pad=8, fontweight="bold")
            ax.set_ylabel("BTC Price (k USDT)", color=_GRAY, fontsize=9)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}k"))
            
            # Check overlap strings
            overlap_warnings = []
            if df_train.index.max() > df_val.index.min():
                overlap_warnings.append(f"Leakage Train/Val: {df_train.index.max()}")
            if df_val.index.max() > df_holdout.index.min():
                overlap_warnings.append(f"Leakage Val/Holdout: {df_val.index.max()}")
            
            loc = "upper left"
            if overlap_warnings:
                ax.text(0.5, 0.9, "ALERTA LEAKAGE:\n" + "\n".join(overlap_warnings), color=_RED, fontsize=12, fontweight="bold", transform=ax.transAxes, ha="center")
                
            ax.legend(fontsize=8, framealpha=0.2, facecolor=_PANEL, edgecolor=_BORDER, labelcolor="white", loc=loc)
            _date_fmt(ax, n_ticks=20)   # More ticks for an 8 year chart
            ax.grid(alpha=_GRIDΑ)
            _caption(ax, "Auditoría estructural de particiones [Panel E]. Si las regiones de color se solapan, hay data leakage futuro en el pipeline de entrenamiento. Los triángulos verdes representan las operaciones de validación cayendo exclusivamente en el Holdout OOS (morado).")
            _style_ax(ax)
            
        except Exception as e:
            logger.warning("[Timeline] Error generando subplot de auditoría: {}", e)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"Timeline no disponible: {e}", ha="center", va="center", color=_GRAY, fontsize=11)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        trades_df: pd.DataFrame,
        title:     str = "Luna V1 — The Gauntlet OOS TearSheet",
        timestamp: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Genera el TearSheet PNG completo.

        Parameters
        ----------
        trades_df : DataFrame con columnas 'timestamp', 'return_pct' (float), 'is_win' (bool)
        title     : Título principal del reporte
        timestamp : Cadena YYYY-MM-DD_THHMM compartida con el run. Si None → hora actual.

        Returns
        -------
        Path del PNG archivado (con timestamp).
        """
        if len(trades_df) == 0:
            logger.error("trades_df vacío — TearSheet no generado.")
            return None

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_T%H%M")

        # Preparar datos
        df = trades_df.copy()
        if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = df.index
            
        has_ts = "timestamp" in df.columns

        if has_ts:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            df_ts = df.set_index("timestamp").sort_index()
            ret       = df_ts["return_pct"]
            cum_ret   = (1 + ret).cumprod()
            drawdown  = self._compute_drawdowns(cum_ret)
            t_start   = ret.index.min()
            t_end     = ret.index.max()
        else:
            df_sorted = df.sort_index()
            ret       = df_sorted["return_pct"]
            cum_ret   = (1 + ret).cumprod()
            drawdown  = self._compute_drawdowns(cum_ret)
            t_start   = pd.Timestamp("2025-01-01", tz="UTC")
            t_end     = pd.Timestamp("2025-12-31", tz="UTC")

        verdict = self._load_verdict()

        # Metadata del header
        dsr       = verdict.get("statistical_audit", {}).get("dsr", 0)
        approved  = verdict.get("deploy_approved", False)
        v_label   = "APROBADO ✓" if approved else "RECHAZADO ✗"
        v_color   = _GREEN if approved else _RED
        n_gates   = len(verdict.get("flags", {}))
        n_pass    = sum(verdict.get("flags", {}).values())
        days_span = (t_end - t_start).days if t_end > t_start else 0
        # M6: métricas adicionales para banner
        _m        = verdict.get("metrics", {})
        _wr_pct   = _m.get("win_rate", 0) * 100
        _n_trades = _m.get("total_trades", 0)
        _mdd      = _m.get("max_drawdown_pct", 0)
        _sharpe   = _m.get("sharpe_crudo", 0)
        _pbo      = verdict.get("statistical_audit", {}).get("estimated_pbo", 0) * 100

        # ── Layout ───────────────────────────────────────────────────────────
        self._apply_global_style()
        fig = plt.figure(figsize=(20, 62), facecolor=_BG)  # v8.0: 56 → 62 por 10 filas (Panel E: Timeline Audit)

        # Suptitle / banner — M6: dos líneas con métricas clave
        _run_id_str = os.environ.get("LUNA_RUN_ID", "?")
        banner_l1 = f"Luna V1 — The Gauntlet OOS TearSheet  |  {v_label}  |  {_run_id_str}"
        banner_l2 = (
            f"WR {_wr_pct:.1f}%  ·  MaxDD {_mdd:.1f}%  ·  Sharpe {_sharpe:.2f}"
            f"  ·  DSR {dsr:.4f}  ·  PBO {_pbo:.1f}%  ·  Trades {_n_trades}"
            f"  ·  Gates {n_pass}/{n_gates}  ·  {t_start.strftime('%d/%m/%Y')} → {t_end.strftime('%d/%m/%Y')} ({days_span}d)"
            f"  ·  {timestamp}"
        )
        fig.text(0.5, 0.991, banner_l1, ha="center", va="top",
                 fontsize=13, fontweight="bold", color=v_color)
        fig.text(0.5, 0.984, banner_l2, ha="center", va="top",
                 fontsize=9, color=_GRAY,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=_PANEL,
                           edgecolor=v_color, linewidth=1.5, alpha=0.9))

        # v8.0: 10 filas — nueva fila 9 = Panel E (Timeline Audit)
        gs = fig.add_gridspec(
            10, 2,
            height_ratios=[1.4, 1.4, 1.2, 1.1, 1.1, 1.6, 1.1, 1.0, 0.9, 1.1],
            hspace=0.90,
            wspace=0.30,
            left=0.07, right=0.97,
            top=0.975, bottom=0.02,
        )

        ax_btc  = fig.add_subplot(gs[0, :])   # BTC price + trades
        ax_eq   = fig.add_subplot(gs[1, :])   # Equity acumulada
        ax_tr   = fig.add_subplot(gs[2, :])   # Retorno por trade (no acumulado)
        ax_dd   = fig.add_subplot(gs[3, 0])   # Drawdown
        ax_sr   = fig.add_subplot(gs[3, 1])   # Rolling Sharpe
        ax_hm   = fig.add_subplot(gs[4, 0])   # Monthly heatmap / Win-Loss hist
        ax_rd   = fig.add_subplot(gs[4, 1])   # Returns dist
        ax_gate = fig.add_subplot(gs[5, 0])   # Gate semaphore
        ax_sb   = fig.add_subplot(gs[5, 1])   # Stats box
        ax_xgbq = fig.add_subplot(gs[6, 0])   # [Panel A] XGB Prob Cuartiles
        ax_hold = fig.add_subplot(gs[6, 1])   # [Panel B] Holding Time Distribution
        ax_hmm  = fig.add_subplot(gs[7, :])   # [Panel C] HMM Regime Distribution Map
        ax_wfv  = fig.add_subplot(gs[8, :])   # [Panel D] Walk-Forward Validation
        ax_tla  = fig.add_subplot(gs[9, :])   # [Panel E] Timeline Data Overlap Audit

        for ax in [ax_btc, ax_eq, ax_tr, ax_dd, ax_sr, ax_hm, ax_rd, ax_xgbq, ax_hold, ax_hmm, ax_wfv, ax_tla]:
            ax.set_facecolor(_BG)
        ax_sb.set_facecolor(_PANEL)

        self._plot_btc_trades(ax_btc, df, t_start, t_end)
        self._plot_equity(ax_eq, cum_ret, ret, t_start, t_end)
        self._plot_trade_returns(ax_tr, ret, df)
        self._plot_drawdown(ax_dd, drawdown, ret)
        self._plot_rolling_sharpe(ax_sr, ret)
        self._plot_monthly_heatmap(ax_hm, ret)
        self._plot_returns_dist(ax_rd, ret)
        self._plot_gate_semaphore(ax_gate, verdict)
        self._plot_stats_box(ax_sb, ret, verdict)
        self._plot_xgb_quartiles(ax_xgbq, df)             # [Panel A] v5.0
        # barrera vertical desde verdict o default
        _vb_h = verdict.get("statistical_audit", {}).get(
            "vertical_barrier_hours",
            verdict.get("config", {}).get("vertical_barrier_hours", 168.0)
        )
        self._plot_holding_time(ax_hold, df, vertical_barrier_hours=float(_vb_h))  # [Panel B] v5.0
        # FIX-HMM-TS-02: asegurar que df['timestamp'] tiene tz=UTC antes del panel HMM
        df_for_hmm = df.copy()
        if "timestamp" in df_for_hmm.columns:
            df_for_hmm["timestamp"] = pd.to_datetime(df_for_hmm["timestamp"], utc=True)
        self._plot_hmm_regime_map(ax_hmm, df_for_hmm, t_start, t_end)              # [Panel C] v6.0
        self._plot_wfv(ax_wfv, verdict)                                             # [Panel D] v7.0
        self._plot_timeline_audit(ax_tla, df)                                       # [Panel E] v8.0

        # ── Guardar ──────────────────────────────────────────────────────────
        run_id = os.environ.get("LUNA_RUN_ID", "DEV")
        ts_name    = f"{timestamp}_{run_id}_tearsheet_oos.png"
        out_ts     = self.out_dir / ts_name
        out_latest = self.out_dir / "tearsheet_oos.png"

        plt.savefig(out_ts,     dpi=150, bbox_inches="tight", facecolor=_BG)
        shutil.copy2(out_ts, out_latest)
        plt.close(fig)

        logger.info("TearSheet guardado: {}  (copia: {})", out_ts, out_latest)
        return out_ts


# ── Entry point de testing ────────────────────────────────────────────────────

if __name__ == "__main__":
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=821, freq="h", tz="UTC")
    rets  = rng.normal(-0.001, 0.020, 821)
    df    = pd.DataFrame({
        "timestamp":  dates,
        "return_pct": rets,
        "is_win":     rets > 0,
    })
    ts = LunaTearSheet()
    out = ts.generate(df, timestamp="2025-01-01_T0000")
    print(f"TearSheet generado: {out}")
