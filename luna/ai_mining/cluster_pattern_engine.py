"""
Cluster Pattern Engine — Luna V1 AI Mining (Engine 6/6)
====================================================================
Propósito: Agrupamiento K-Means de "tribus de mercado" y análisis de
estacionalidad macro para enriquecer el contexto de predicción.

Dos análisis complementarios:
  A. K-Means 4 Tribus: agrupa barras en 4 "perfiles de mercado" según
     las condiciones simultáneas del mercado.

  B. Estacionalidad Macro: calcula retorno medio de BTC por:
     - Hora del día (0-23 UTC)
     - Día de la semana (Lun-Dom)
     - Semana del mes (1-4)
     - Semana del ciclo FOMC (±3 días)

Output:
  - `KMeans_Tribe_ID` (0-7) añadida al dataset
  - `data/ai_mining/reports/cluster_pattern_report.md`
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import os
import joblib
import numpy as np
import pandas as pd
from loguru import logger

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_FEATURES = PROJECT_ROOT / "data" / "features"
REPORTS_DIR   = PROJECT_ROOT / "data" / "ai_mining" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
N_TRIBES    = 4
HORIZON_H   = 24

# Features para K-Means de "momentos de mercado"
TRIBE_FEATURES = [
    "FedFundsRate", "YieldCurve_10Y3M", "VIX", "DXY",
    "FundingRate", "DangerZone", "OI_BTC",
    "MVRV_Proxy", "FearGreed", "SSR",
    "Master_Causal_Signal",
    "eth_btc_corr_24h", "alt_season_proxy",
]

# Fechas FOMC hardcoded 2020-2026 (de FederalReserve.gov)
FOMC_DATES = [
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-11-05", "2025-12-17",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]


# ─────────────────────────────────────────────────────────────────────────────
class ClusterPatternEngine:

    def __init__(self, cutoff_date=None):
        # cutoff_date: pd.Timestamp | None
        # Si se inyecta (--mode dev), load_data() filtra df a <= cutoff_date.
        self.cutoff_date = pd.Timestamp(cutoff_date, tz='UTC') if cutoff_date else None

    def load_data(self) -> pd.DataFrame:
        for name in ["features_train_causal.parquet",
                     "features_train_kshape.parquet",
                     "features_train.parquet"]:
            p = DATA_FEATURES / name
            if p.exists():
                df = pd.read_parquet(p)
                df.index = pd.to_datetime(df.index, utc=True)
                # [MODO DEV] Aplicar cutoff_date si está definido
                if self.cutoff_date is not None and df.index.max() > self.cutoff_date:
                    before = len(df)
                    df = df[df.index <= self.cutoff_date]
                    logger.info(f"Cluster: cutoff_date={self.cutoff_date.date()} aplicado — {before} → {len(df)} filas")
                logger.info(f"Cluster: cargado {df.shape} desde {name}")
                return df
        raise FileNotFoundError("No dataset encontrado")


    # ── A. K-Means Tribus ─────────────────────────────────────────────────────

    def fit_tribes(self, df: pd.DataFrame, train_cutoff: str | None = None) -> pd.Series:
        """
        K-Means de N_TRIBES sobre las features de contexto de mercado.
        train_cutoff: si None, se lee de settings.yaml (cfg.temporal_splits.train_end).

        Fix CL-01: el StandardScaler se ajusta SOLO sobre el training set (train_cutoff).
        Ajustar sobre todo el dataset contamina la normalización con datos futuros (leakage).
        """
        if train_cutoff is None:
            try:
                from config.settings import cfg as _cfg_cl
                train_cutoff = _cfg_cl.temporal_splits.train_end
            except Exception:
                raise RuntimeError("Cluster: train_cutoff no disponible en settings.yaml — no se puede ajustar scaler")
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        available = [f for f in TRIBE_FEATURES if f in df.columns]
        if not available:
            logger.warning("Cluster: ninguna feature para K-Means")
            return pd.Series(0, index=df.index, name="KMeans_Tribe_ID")

        X = df[available].dropna()
        if len(X) < N_TRIBES * 3:
            return pd.Series(0, index=df.index, name="KMeans_Tribe_ID")

        # Fix CL-01: determinar ventana de training para el scaler
        cutoff = pd.Timestamp(train_cutoff, tz="UTC")
        if X.index.tz is None:
            cutoff = cutoff.tz_localize(None)
        X_train = X[X.index <= cutoff]
        if len(X_train) < N_TRIBES * 3:
            logger.warning(f"CL-01: Datos insuficientes antes de {train_cutoff} para ajustar scaler — usando todo X como fallback.")
            X_train = X

        # FIX-ALTO-4: Estabilidad de KMeans Tribus en W2-W5 (Anchoring de Centroides)
        run_id = os.environ.get("LUNA_RUN_ID", "")
        models_dir = PROJECT_ROOT / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = models_dir / "kmeans_scaler.pkl"
        kmeans_path = models_dir / "kmeans_model.pkl"

        is_wfb_subsequent = run_id.startswith("WFB_") and not run_id.endswith("_W1")
        
        if is_wfb_subsequent and scaler_path.exists() and kmeans_path.exists():
            # W2-W5: Cargar modelos anclados de W1 para prevenir Tribe Drift
            logger.info(f"Cluster [FIX-ALTO-4]: Run {run_id}. Cargando centroides y scaler anclados de W1.")
            scaler = joblib.load(scaler_path)
            km = joblib.load(kmeans_path)
            X_s = scaler.transform(X)
            labels = km.predict(X_s)
        else:
            # W1 o entrenamiento normal: Entrenar desde cero y exportar anclas
            scaler = StandardScaler()
            scaler.fit(X_train)          # fit SOLO en train
            X_s = scaler.transform(X)   # transform en todo (causal: sólo usa estadísticos del train)

            km = KMeans(n_clusters=N_TRIBES, n_init=5, random_state=42)
            labels = km.fit_predict(X_s)
            
            # Guardar anclajes
            joblib.dump(scaler, scaler_path)
            joblib.dump(km, kmeans_path)
            logger.info(f"Cluster [FIX-ALTO-4]: Modelos scaler y KMeans (N={N_TRIBES}) guardados en {models_dir.name}")

        tribe_series = pd.Series(labels, index=X.index, dtype=int, name="KMeans_Tribe_ID")
        tribe_series = tribe_series.reindex(df.index).ffill().bfill()
        logger.info(f"Cluster: {N_TRIBES} tribus ajustadas/asignadas sobre {len(X)} barras")
        return tribe_series

    def _tribe_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula Win Rate y Sharpe por tribu con CI estadístico de Wilson.
        
        [FALLA-08-FIX 2026-05-30] Añadido Wilson score CI al 95%.
        Una tribu con WR=63.9% y n=936 tiene IC95=[60.8%, 67.0%] -> significativo.
        Una tribu con WR=60% y n=50 tiene IC95=[46%, 73%] -> NO significativo (cruza 0.5).
        Se emite WARNING y se descarta la tribu de LARGA si CI_lower < 0.50.
        """
        if "KMeans_Tribe_ID" not in df.columns or "close" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["ret_24h"] = df["close"].pct_change(24)

        rows = []
        for tid in sorted(df["KMeans_Tribe_ID"].dropna().unique()):
            grp = df[df["KMeans_Tribe_ID"] == tid]["ret_24h"].dropna()
            if len(grp) < 10:
                continue
            mu    = grp.mean()
            sigma = grp.std()
            wr    = (grp > 0).mean()
            sharpe = mu / sigma * np.sqrt(365) if sigma > 0 else 0

            # [FALLA-08-FIX] Wilson score CI al 95% para el Win Rate
            n = len(grp)
            z = 1.96  # z para 95% CI
            # Formula Wilson: CI = (wr + z²/2n ± z*sqrt(wr*(1-wr)/n + z²/4n²)) / (1 + z²/n)
            _center = wr + z**2 / (2 * n)
            _margin = z * np.sqrt(wr * (1 - wr) / n + z**2 / (4 * n**2))
            _denom  = 1 + z**2 / n
            ci_lower = max(0.0, (_center - _margin) / _denom)
            ci_upper = min(1.0, (_center + _margin) / _denom)

            # Determinar régimen con CI estadístico
            regime_raw = "LARGA" if mu > 0.003 else "CORTA" if mu < -0.003 else "NEUTRAL"
            # [FALLA-08-FIX] Si la tribu es LARGA pero CI_lower < 0.50, degradar a NEUTRAL
            regime = regime_raw
            if regime_raw == "LARGA" and ci_lower < 0.50:
                regime = "NEUTRAL"
                print(f"[FALLA-08-FIX] Tribu {tid}: WR={wr:.1%} PERO CI_lower={ci_lower:.1%}<50% "
                      f"(n={n}) -> degradada de LARGA a NEUTRAL (no significativo)")
            elif n < 300:
                print(f"[FALLA-08-FIX] Tribu {tid}: n={n}<300 | WR={wr:.1%} CI95=[{ci_lower:.1%},{ci_upper:.1%}] "
                      f"-> {regime} con CI estrecho, interpretacion con cautela")
            else:
                print(f"[FALLA-08-FIX] Tribu {tid}: n={n} | WR={wr:.1%} CI95=[{ci_lower:.1%},{ci_upper:.1%}] | {regime}")

            rows.append({
                "tribe":            int(tid),
                "n_bars":           len(grp),
                "win_rate":         round(wr * 100, 1),
                "mean_ret_24h":     round(mu * 100, 3),
                "sharpe_annual":    round(sharpe, 2),
                "ci_lower_95":      round(ci_lower * 100, 1),
                "ci_upper_95":      round(ci_upper * 100, 1),
                "regime":           regime,
            })
        return pd.DataFrame(rows).sort_values("sharpe_annual", ascending=False)


    # ── B. Estacionalidad Macro ───────────────────────────────────────────────

    def seasonality_analysis(self, df: pd.DataFrame) -> dict:
        """
        Calcula estadísticas de retorno BTC por:
          - Hora del día (0-23 UTC)
          - Día de la semana (0=Lunes)
          - Semana del mes (1-4)
          - Semana FOMC (±3 días antes/después de reunión)
        """
        if "close" not in df.columns:
            return {}

        df = df.copy()
        df["ret_1h"]       = df["close"].pct_change(1)
        df["hour"]         = df.index.hour
        df["dow"]          = df.index.dayofweek
        df["week_of_month"] = (df.index.day - 1) // 7 + 1

        # FOMC flag (±3 días)
        fomc_dts = pd.to_datetime(FOMC_DATES)

        def is_fomc_week(ts: pd.Timestamp) -> int:
            for fd in fomc_dts:
                if abs((ts - fd).days) <= 3:
                    return 1
            return 0

        df["is_fomc_week"] = df.index.map(lambda t: is_fomc_week(t.replace(tzinfo=None)))

        # Estacionalidad por hora
        by_hour = df.groupby("hour")["ret_1h"].agg(["mean", "std", lambda x: (x > 0).mean()])
        by_hour.columns = ["mean_ret", "std_ret", "win_rate"]

        # Estacionalidad por día de semana
        by_dow = df.groupby("dow")["ret_1h"].agg(["mean", "std", lambda x: (x > 0).mean()])
        by_dow.columns = ["mean_ret", "std_ret", "win_rate"]
        dow_names = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
        by_dow.index = [dow_names.get(i, str(i)) for i in by_dow.index]

        # FOMC vs no-FOMC
        fomc_wr = df[df["is_fomc_week"] == 1]["ret_1h"].agg(
            ["mean", lambda x: (x > 0).mean()]
        )
        nonfomc_wr = df[df["is_fomc_week"] == 0]["ret_1h"].agg(
            ["mean", lambda x: (x > 0).mean()]
        )

        # Best hours (top 3 por WR)
        best_hours = by_hour["win_rate"].sort_values(ascending=False).head(3)
        worst_hours = by_hour["win_rate"].sort_values().head(3)

        return {
            "by_hour":     by_hour,
            "by_dow":      by_dow,
            "best_hours":  best_hours.index.tolist(),
            "worst_hours": worst_hours.index.tolist(),
            "fomc_wr":     round(float(fomc_wr.iloc[1]) * 100, 1) if len(fomc_wr) > 1 else 50.0,
            "nonfomc_wr":  round(float(nonfomc_wr.iloc[1]) * 100, 1) if len(nonfomc_wr) > 1 else 50.0,
        }

    # ── Reporte ──────────────────────────────────────────────────────────────

    def _save_report(self, tribe_df: pd.DataFrame, season: dict) -> None:
        """Reporte estilo Correlaciones — K-Means Tribus + Estacionalidad."""
        ts = pd.Timestamp.now().strftime("%d %B %Y %H:%M")

        # Tribu/Régimen actual
        larga_tribes  = tribe_df[tribe_df["regime"] == "LARGA"]["tribe"].tolist() if not tribe_df.empty else []
        corta_tribes  = tribe_df[tribe_df["regime"] == "CORTA"]["tribe"].tolist() if not tribe_df.empty else []
        neutral_tribes = tribe_df[tribe_df["regime"] == "NEUTRAL"]["tribe"].tolist() if not tribe_df.empty else []

        report = [
            f"# 🧩 CLUSTER PATTERN ENGINE — Luna V1",
            f"**Generado:** {ts} | **Algoritmo:** K-Means Tribus | **N Tribus:** {N_TRIBES} | **Horizonte:** {HORIZON_H}H",
            "",
            "> *Agrupación de 'momentos de mercado' — diferente al KShape (topología temporal).*",
            "> *K-Means agrupa ESTADOS simultáneos de todas las features macro/on-chain.*",
            "> *Cada tribu representa una configuración macro que el mercado repite históricamente.*",
            "",
            "---",
            "",
            "## 🧭 1. Clasificación de Tribus Macro",
            "",
            f"**Tribus Alcistas (LARGA):** {larga_tribes} — Condiciones macro históricamente favorables a BTC.",
            f"**Tribus Bajistas/Neutras:** {corta_tribes + neutral_tribes} — Condiciones de cautela o sin sesgo.",
            "",
            "---",
            "",
            "## 📊 2. Performance por Tribu K-Means",
            "",
            "> **Retorno y Win Rate promedio 24H.** Sharpe anualizado (ret_24H × √365) — valores > 10 son esperables en BTC sobre ventanas largas.",
            "",
            "| Tribu | Barras | Win Rate | EV 24H | Sharpe | Régimen |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for _, r in tribe_df.iterrows():
            regime = r["regime"]
            emoji = {"LARGA": "🟢", "CORTA": "🔴", "NEUTRAL": "⚖️"}.get(regime, "")
            ev_str = f"+{r['mean_ret_24h']}%" if r['mean_ret_24h'] >= 0 else f"{r['mean_ret_24h']}%"
            report.append(
                f"| {emoji} Tribu {int(r['tribe'])} | {int(r['n_bars'])} "
                f"| **{r['win_rate']}%** | {ev_str} "
                f"| {r['sharpe_annual']} | **{regime}** |"
            )

        if season:
            best = season.get("best_hours", [])
            worst = season.get("worst_hours", [])
            fomc_wr = season.get("fomc_wr", 50.0)
            nonfomc_wr = season.get("nonfomc_wr", 50.0)
            fomc_delta = round(fomc_wr - nonfomc_wr, 1)
            fomc_note = f"{'FOMC SEMANA FAVORABLE' if fomc_delta > 0.5 else 'FOMC = Sin efecto notable'}"

            report += [
                "",
                "---",
                "",
                "## 🕐 3. Estacionalidad Macro",
                "",
                "### ⏰ Horario Óptimo de Trading (UTC)",
                "",
                f"| Métrica | Valor | Interpretación |",
                f"| --- | --- | --- |",
                f"| Mejores horas UTC | **{best}** | Horas con mayor Win Rate histórico 1H |",
                f"| Peores horas UTC | {worst} | Horas con menor Win Rate histórico 1H |",
                f"| Win Rate FOMC week | **{fomc_wr}%** | ±3 días de reunión Fed |",
                f"| Win Rate No-FOMC | {nonfomc_wr}% | Resto de semanas |",
                f"| Delta FOMC | {'+' if fomc_delta >= 0 else ''}{fomc_delta}pp | {fomc_note} |",
                "",
            ]

            if "by_dow" in season:
                report += [
                    "### 📅 Performance por Día de la Semana",
                    "",
                    "| Día | Win Rate 1H | Ret Medio 1H | Sesgo |",
                    "| --- | --- | --- | --- |",
                ]
                for dow, row in season["by_dow"].iterrows():
                    wr_pct = row['win_rate'] * 100
                    ret_pct = row['mean_ret'] * 100
                    sesgo = "🟢" if wr_pct > 51 else "🔴" if wr_pct < 49 else "⚖️"
                    report.append(
                        f"| **{dow}** | {wr_pct:.1f}% | {ret_pct:+.3f}% | {sesgo} |"
                    )

        report += [
            "",
            "---",
            "",
            "## 📓 4. Playbook Táctico — Tribus",
            "",
            "### 🛡️ Reglas de Operación por Régimen",
            "",
            f"- **Tribus {larga_tribes} (LARGA):** Exposición plena con gestión de riesgo estándar.",
            f"- **Tribus {neutral_tribes} (NEUTRAL):** Reducir sizing al 50% — esperar confirmación de Bayesian/Advanced.",
            f"- **Tribus {corta_tribes} (CORTA):** Evitar largos. Considerar cobertura o flat.",
            "",
            "### ⏱️ Horario de Ejecución",
            f"- **Abrir posiciones en horas:** {best} UTC — máxima probabilidad histórica",
            f"- **Evitar entradas en horas:** {worst} UTC — worst historical win rate",
            "",
            "> [!TIP]",
            "> **Nota de Ejecución:** Cluster Pattern tiene peso ×1 en el Meta-Oracle.",
            "> Úsalo como filtro horario/de-tribu secundario, no como señal primaria.",
            "> La combinación Tribu LARGA + hora óptima + `Master_Causal_Signal > 0` = máxima confluence.",
        ]

        path = REPORTS_DIR / "cluster_pattern_report.md"
        path.write_text("\n".join(report), encoding="utf-8")
        logger.success(f"Cluster: reporte mejorado guardado en {path}")

    # ── Pipeline principal ────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info("Cluster Pattern Engine — INICIO")
        logger.info("=" * 60)

        df = self.load_data()

        # A. Tribus K-Means — train_cutoff desde settings.yaml
        try:
            from config.settings import cfg as _cfg_run_cl
            _tc = _cfg_run_cl.temporal_splits.train_end
        except Exception:
            _tc = None
        tribes = self.fit_tribes(df, train_cutoff=_tc)
        df["KMeans_Tribe_ID"] = tribes

        tribe_analysis = self._tribe_analysis(df)
        if not tribe_analysis.empty:
            logger.info(f"\n{tribe_analysis.to_string(index=False)}")

        # B. Estacionalidad
        logger.info("Cluster: analizando estacionalidad macro...")
        season = self.seasonality_analysis(df)
        if season:
            logger.info(f"  Mejor horas UTC: {season.get('best_hours', [])}")
            logger.info(f"  FOMC WR: {season.get('fomc_wr', 0)}%  vs  no-FOMC: {season.get('nonfomc_wr', 0)}%")

        # Guardar parquet final enriquecido
        out_path = DATA_FEATURES / "features_train_final.parquet"
        df.to_parquet(out_path)
        logger.success(f"Cluster: dataset final guardado en {out_path}")

        self._save_report(tribe_analysis, season)

        logger.info("Cluster Pattern Engine — COMPLETADO")
        return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = ClusterPatternEngine()
    engine.run()
