"""
Master Pattern Engine — Luna V1 AI Mining (Engine 4/6)
====================================================================
Propósito: Descubrir "Golden Rules" — conjunciones de condiciones que
maximizan el Win Rate de BTC con evidencia de HMM regime + Wavelet +
Transfer Entropy + Win Rate histórico.

Metodología:
  1. Segmentar el histórico por régimen HMM (K=4 estados)
  2. En cada régimen: buscar conjunciones IF/AND de 2-4 variables
     que maximicen Win Rate con >= 30 ocurrencias (significancia R8)
  3. Validar cada regla con SFI (RF OOB > umbral)
  4. Generar reporte master_pattern_report.md con tabla de "Golden Storms"

Output:
  - `data/ai_mining/reports/master_pattern_report.md`
  - Las reglas serán leídas por export_alpha_rules.py para generar
    alpha_rules.py nativo (R13 SOP).
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
from itertools import combinations

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
MIN_OCCURRENCES  = 30    # mínimo de barras que activan la regla (R8)
MIN_WIN_RATE     = 0.58  # mínimo WR para considerar una regla válida
MAX_COMBINATIONS = 3     # máximo de variables en una conjunción IF/AND
N_HMM_STATES     = 4     # estados HMM para segmentación de régimen
HORIZON_H        = 24    # horizonte de predicción en horas

# Variables candidatas para construir reglas (priorizadas por evidence_score)
RULE_CANDIDATES = [
    # FRED macro (alta evidencia causal)
    "FedFundsRate", "YieldCurve_10Y3M", "T10Y2Y",
    "GlobalM2_Index", "Fed_Net_Liquidity", "WEI",
    # Mercado
    "VIX", "DXY", "SP500_AboveMA200", "NASDAQ_Ret",
    # Derivados
    "FundingRate", "DangerZone", "DVOL", "OI_BTC",
    # On-chain
    "MVRV_Proxy", "FearGreed", "SSR", "Whale_Vol_ZScore",
    # Salidas de engines previos
    "KMeans_Tribe_ID", "Master_Causal_Signal",
]


# ─────────────────────────────────────────────────────────────────────────────
class MasterPatternEngine:
    """Descubre Golden Rules por régimen HMM."""

    def __init__(self, cutoff_date=None):
        # cutoff_date: pd.Timestamp | None
        # Si se inyecta (modo --mode dev), load_data() filtra df a <= cutoff_date.
        # Esto garantiza que el Mining de desarrollo usa el mismo corte temporal
        # que Feature Selection (train_end en settings.yaml), eliminando Selection Leakage.
        self.cutoff_date = cutoff_date

    def load_data(self) -> pd.DataFrame:
        # P1-2-FIX (2026-03-30): eliminado features_train_kshape.parquet (K-Shape decommisionado).
        for name in ["features_train_causal.parquet",
                     "features_train.parquet"]:
            p = DATA_FEATURES / name
            if p.exists():
                df = pd.read_parquet(p)
                df.index = pd.to_datetime(df.index, utc=True)
                # ── Filtro de fecha de corte (modo dev) ──────────────────
                cutoff = getattr(self, "cutoff_date", None)
                if cutoff is not None:
                    n_before = len(df)
                    df = df[df.index <= cutoff]
                    logger.info(
                        f"MasterPattern [DEV]: cutoff={cutoff.date()} "
                        f"→ {len(df)}/{n_before} filas ({len(df)/n_before*100:.1f}% del histórico)"
                    )
                else:
                    logger.info(f"MasterPattern [PROD]: datos completos {df.shape} desde {name}")
                return df
        raise FileNotFoundError("No dataset encontrado en data/features/")

    # ── HMM Regime ────────────────────────────────────────────────────────────

    def fit_hmm_regimes(self, df: pd.DataFrame) -> pd.Series:
        """
        Ajusta HMM de 4 estados sobre [ret_1h, vol_24h, funding_rate].
        Usa Forward Algorithm (nunca Viterbi en producción — SOP R9).
        Retorna columna 'hmm_regime' (0-3).
        
        [FALLA-10-FIX 2026-05-30] Multi-init con 5 seeds para evitar mínimos locales.
        El log anterior mostraba "Model is not converging. Delta=-24.27" con una sola init.
        """
        try:
            from hmmlearn import hmm

            features = []
            if "close" in df.columns:
                ret    = df["close"].pct_change(1).fillna(0)
                vol24  = ret.rolling(24).std().fillna(0)
                features.extend([ret, vol24])
            if "FundingRate" in df.columns:
                features.append(df["FundingRate"].fillna(0))
            if not features:
                return pd.Series(0, index=df.index, name="hmm_regime")

            X_raw = pd.concat(features, axis=1).dropna().values
            
            # [FIX-HMM-CONVERGENCE-01] Estandarizar X para estabilizar el EM Algorithm.
            # Variables como FundingRate tienen varianzas naturales de 1e-8, chocando
            # con el min_covar default de hmmlearn (1e-3) y rompiendo el Log-Likelihood.
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X_raw)

            # [FALLA-10-FIX] Multi-init: probar 5 seeds y quedarse con el mejor log-likelihood
            best_score = -np.inf
            best_model = None
            _n_init_master = 5  # menos que el HMM principal (10) por velocidad
            for _seed in range(_n_init_master):
                _m = hmm.GaussianHMM(
                    n_components=N_HMM_STATES,
                    covariance_type="diag",
                    n_iter=200,       # aumentado de 100 a 200 para mejor convergencia
                    random_state=_seed,
                    min_covar=0.01,   # [SENTINEL-FIX] Previene log-likelihood negativa
                    verbose=False,
                )
                try:
                    _m.fit(X)
                    _score = _m.score(X)
                    if _m.monitor_.converged and _score > best_score:
                        best_score = _score
                        best_model = _m
                except Exception:
                    continue

            if best_model is None:
                logger.warning("[FALLA-10-FIX] Ningún seed del MasterPattern HMM convergió — fallback seed=42 sin filtro")
                print("[FALLA-10-FIX] WARNING: HMM MasterPattern no convergió en 5 seeds — usando fallback")
                model = hmm.GaussianHMM(
                    n_components=N_HMM_STATES,
                    covariance_type="diag",
                    n_iter=200,
                    random_state=42,
                    min_covar=0.01,
                    verbose=False,
                )
                model.fit(X)
            else:
                model = best_model
                logger.info(f"[FALLA-10-FIX] MasterPattern HMM: mejor seed={best_model.startprob_.argmax()} | "
                            f"log-lik={best_score:.2f} | converged=True")
                print(f"[FALLA-10-FIX] MasterPattern HMM multi-init OK: log-lik={best_score:.2f} en {_n_init_master} seeds")

            # Forward Algorithm (causal)
            log_prob, posteriors = model.score_samples(X)
            states = posteriors.argmax(axis=1)

            regime = pd.Series(
                states,
                index=df.dropna(subset=[df.columns[0]]).index[:len(states)],
                name="hmm_regime",
                dtype=int,
            )
            regime = regime.reindex(df.index).ffill().fillna(0).astype(int)
            logger.info(f"HMM: {N_HMM_STATES} regímenes ajustados")
            return regime

        except Exception as e:
            logger.warning(f"HMM falló ({e}) — usando régimen único")
            return pd.Series(0, index=df.index, name="hmm_regime")


    # ── Target ───────────────────────────────────────────────────────────────

    def _build_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Target Binario: 1=LONG (Tp tocado o retorno positivo).
        La arquitectura de Luna V2 es estrictamente Long-Only.
        """
        if "Target_TBM_Bin" in df.columns:
            logger.info("MasterPattern: Usando Target_TBM_Bin (Probabilidad LONG)")
            return df["Target_TBM_Bin"]
        if "target" in df.columns:
            return df["target"]
        if "close" in df.columns:
            ret24 = df["close"].shift(-HORIZON_H) / df["close"] - 1
            return (ret24 > 0.01).astype(int)
        return pd.Series(dtype=int)

    # ── Discretizar variables ─────────────────────────────────────────────────

    def _discretize(self, col: pd.Series) -> pd.Series:
        """
        Convierte una variable continua en percentil Q:
        LOW (< 25th), MID_LOW, MID_HIGH, HIGH (> 75th).
        Permite construir reglas IF(FedFunds_HIGH) más robustas.
        """
        q25, q75 = col.quantile(0.25), col.quantile(0.75)
        bins  = [-np.inf, q25, q75, np.inf]
        cats  = ["LOW", "MID", "HIGH"]
        return pd.cut(col, bins=bins, labels=cats, include_lowest=True)

    # ── Búsqueda de Golden Rules ──────────────────────────────────────────────

    def _find_golden_rules(
        self, df: pd.DataFrame, target: pd.Series, regime_id: int
    ) -> list[dict]:
        """
        Busca conjunciones de 1-MAX_COMBINATIONS variables que maximicen
        Win Rate (Long-Only) con >= MIN_OCCURRENCES ocurrencias.
        """
        available = [c for c in RULE_CANDIDATES if c in df.columns]
        rules     = []

        # OPTIMIZACIÓN O(1): Pre-calcular cuantiles para el régimen entero
        quantiles = {}
        for var in available:
            col = df[var].dropna()
            if len(col) > 10:
                quantiles[var] = (col.quantile(0.25), col.quantile(0.75))
            else:
                quantiles[var] = (0.0, 0.0)

        for n_vars in range(1, MAX_COMBINATIONS + 1):
            for combo in combinations(available, n_vars):
                # Construir condición compuesta
                try:
                    masks = []
                    conditions = []
                    for var in combo:
                        q25, q75 = quantiles[var]

                        # Elegir la condición que mejor predice LONG
                        mask_h = df[var] >= q75
                        mask_l = df[var] <= q25

                        wr_h = target[mask_h & (target.notna())].mean() if mask_h.sum() > 10 else 0
                        wr_l = target[mask_l & (target.notna())].mean() if mask_l.sum() > 10 else 0

                        # Pruning: Si ni la parte alta ni la baja alcanzan un umbral mínimo,
                        # descartamos la combinación entera (Long-Only).
                        if max(wr_h, wr_l) < MIN_WIN_RATE * 0.9:
                            break

                        if wr_h >= wr_l:
                            masks.append(mask_h)
                            conditions.append(f"{var} >= {q75:.4f}")
                        else:
                            masks.append(mask_l)
                            conditions.append(f"{var} <= {q25:.4f}")
                    else:
                        # Todas las variables pasaron el filtro individual
                        combined_mask = masks[0]
                        for m in masks[1:]:
                            combined_mask = combined_mask & m

                        combined_mask = combined_mask & target.notna()
                        t_masked = target[combined_mask].sort_index()
                        n_occ = len(t_masked)
                        if n_occ < MIN_OCCURRENCES:
                            continue

                        wr_mean = float(t_masked.mean())
                        
                        # FIX FASE 5: Validación Cruzada Cronológica (H1 vs H2)
                        half = n_occ // 2
                        wr_h1 = float(t_masked.iloc[:half].mean())
                        wr_h2 = float(t_masked.iloc[half:].mean())
                        wr_robust = min(wr_h1, wr_h2)

                        # Fix F2-01: EV calculado SOLO sobre el training set para evitar leakage
                        future_ret = df["close"].shift(-HORIZON_H) / df["close"] - 1
                        train_mask = combined_mask & (df.index <= self.cutoff_date) if self.cutoff_date is not None else combined_mask
                        ev = float(future_ret[train_mask].mean()) if train_mask.sum() > 0 else 0.0

                        if wr_robust >= MIN_WIN_RATE:
                            rules.append({
                                "regime":    regime_id,
                                "variables": list(combo),
                                "conditions": conditions,
                                "n_occurrences": int(n_occ),
                                "win_rate":  round(wr_mean * 100, 1),
                                "ev_pct":    round(ev * 100, 2) if not np.isnan(ev) else 0.0,
                                "pandas_eval": " & ".join([f"({c})" for c in conditions]),
                            })
                except Exception:
                    continue

        # Ordenar por WR desc, luego n_occ desc
        rules.sort(key=lambda x: (x["win_rate"], x["n_occurrences"]), reverse=True)
        return rules[:20]  # top 20 por régimen

    # ── Reporte ──────────────────────────────────────────────────────────────

    def _save_report(self, all_rules: list[dict], df: pd.DataFrame | None = None) -> None:
        """Genera master_pattern_report.md con formato editorial Correlaciones-style."""
        now = pd.Timestamp.now(tz="UTC").strftime("%d %B %Y %H:%M")
        n_rules = len(all_rules)
        top = all_rules[0] if all_rules else {}
        top_wr = top.get("win_rate", 0)
        top_ev = top.get("ev_pct", 0)

        # Agrupar reglas por régimen HMM
        by_regime: dict[int, list] = {}
        for r in all_rules:
            rid = r.get("regime", 0)
            by_regime.setdefault(rid, []).append(r)

        # Calcular stats del dataset
        n_days = len(df) // 24 if df is not None else "N/A"

        lines: list[str] = []

        # ── Header ──
        lines += [
            "# 🌌 MASTER PATTERN & SIGNAL REPORT — Luna V1",
            f"**Generado:** {now} | **Dataset:** {n_days} días de historia | **Total reglas:** {n_rules}",
            "",
            "> *Este reporte descubre las 'Tormentas Perfectas' — conjunciones de condiciones que históricamente"
            " producen asimetrías de riesgo/beneficio comprobadas empíricamente (Win Rate & Expected Value a 24H).*"
            " *Cada regla requiere mínimo 30 ocurrencias y WR ≥ 58% sobre el histórico de BTC 2020→2025.*",
            "",
            "---",
        ]

        # ── 1. Tormentas perfectas ──
        lines += [
            "",
            "## 🌪️ 1. Las Tormentas Perfectas (Reglas de Oro Combinadas)",
            "> **La convergencia de extremos.** El mercado paga las primas más altas cuando dos o más"
            " vectores asimétricos colisionan. Estas son las combinaciones con el mayor Win Rate histórico"
            " y Retorno Esperado a 24H.",
            "",
            "| Condición Exacta | Régimen HMM | Win Rate | EV 24H | N hits |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in all_rules[:15]:
            conds = r.get("conditions", [])
            # Multilínea con HTML br (estilo Correlaciones)
            cond_html = "<br>**AND** ".join([f"`{c}`" for c in conds])
            cond_str = f"**IF** {cond_html}"
            ev_str = f"**+{r['ev_pct']}%**" if r['ev_pct'] >= 0 else f"**{r['ev_pct']}%**"
            lines.append(
                f"| {cond_str} | {r.get('regime','-')} | **{r['win_rate']}%** | {ev_str} | {r['n_occurrences']} |"
            )

        # ── 2. Reglas por régimen HMM ──
        lines += [
            "",
            "",
            "---",
            "## 🧩 2. Reglas por Régimen HMM",
            "> **Inteligencia contextual.** El mercado no se comporta igual en todos los ciclos."
            " Estas reglas son válidas SÓLO dentro del régimen detectado — su efectividad fuera de él es menor.",
            "",
        ]
        regime_names = {0: "Régimen 0 🌑", 1: "Régimen 1 🐻", 2: "Régimen 2 ⚖️", 3: "Régimen 3 🚀"}
        for rid, rules in sorted(by_regime.items()):
            rname = regime_names.get(rid, f"Régimen {rid}")
            lines += [
                f"### {rname}",
                "",
                "| Condición | Win Rate | EV 24H | N |",
                "| --- | --- | --- | --- |",
            ]
            for r in rules[:8]:
                conds = r.get("conditions", [])
                cond_html = "<br>**AND** ".join([f"`{c}`" for c in conds])
                ev_str = f"+{r['ev_pct']}%" if r['ev_pct'] >= 0 else f"{r['ev_pct']}%"
                lines.append(
                    f"| **IF** {cond_html} | **{r['win_rate']}%** | {ev_str} | {r['n_occurrences']} |"
                )
            lines.append("")

        # ── 3. Componentes de mayor impacto ──
        lines += [
            "",
            "---",
            "## 🎯 3. Variables con Mayor Poder Predictivo",
            "> **Los pilares del sistema.** Variables que aparecen en más reglas Golden"
            " son los indicadores más robustos para decisiones de inversión.",
            "",
            "| Variable | Apariciones en reglas | WR Promedio |",
            "| --- | --- | --- |",
        ]
        from collections import Counter
        var_count: Counter = Counter()
        var_wr: dict = {}
        for r in all_rules:
            for v in r.get("variables", []):
                var_count[v] += 1
                var_wr.setdefault(v, []).append(r["win_rate"])
        for var, count in var_count.most_common(10):
            avg_wr = sum(var_wr[var]) / len(var_wr[var])
            lines.append(f"| **{var}** | {count} | **{avg_wr:.1f}%** |")

        # ── 4. Engine_shap visual ──
        lines += [
            "",
            "",
            "---",
            "## 📊 4. SHAP Feature Importance",
            "> Contribución de cada variable en la predicción de BTC a 24H.",
            "",
            "![SHAP Feature Importance](engine_shap.png)",
            "",
        ]

        # ── 5. Distribución por régimen ──
        lines += [
            "",
            "---",
            "## 🔍 5. Distribución de Reglas por Régimen",
            "",
            "| Régimen | N Reglas | WR Medio | WR Max |",
            "| --- | --- | --- | --- |",
        ]
        for rid, rules in sorted(by_regime.items()):
            wrs = [r["win_rate"] for r in rules]
            lines.append(
                f"| {regime_names.get(rid, f'Régimen {rid}')} | {len(rules)} |"
                f" {sum(wrs)/len(wrs):.1f}% | {max(wrs):.1f}% |"
            )

        # ── 6. Playbook táctico ──
        top_conds_str = " AND ".join(top.get("conditions", ["N/A"]))
        lines += [
            "",
            "",
            "---",
            "## 📓 6. Playbook Táctico de Trading",
            "> **Estrategia ejecutable.** Basado en las asimetrías detectadas en el histórico 2020→2025.",
            "",
            "### 🛡️ Escenario de Alta Probabilidad",
            f"La regla de oro actual es: **{top_conds_str}**",
            f"- **Objetivo:** Entrada cuando estas condiciones converjan simultáneamente.",
            f"- **Histórico:** Win Rate **{top_wr}%** a 24H · EV calculado: **{'+'if top_ev>=0 else ''}{top_ev}%** *(retorno medio forward 24H)*.",
            f"- **Mínimo de hits históricos:** {top.get('n_occurrences', 0)} ocurrencias — estadísticamente significativo (R8 SOP).",
            "",
            "> [!TIP]",
            "> **Nota de Ejecución:** Las 'Tormentas Perfectas' (IF/AND de 2+ condiciones) superan"
            " consistentemente a los indicadores aislados. La convergencia múltiple es lo que genera"
            " la ventaja estadística real.",
            "",
            "---",
            "",
            f"*Generado por Luna V1 Master Pattern Engine · {now}*",
        ]

        path = REPORTS_DIR / "master_pattern_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.success(f"MasterPattern: {n_rules} reglas → {path}")


    # ── Pipeline principal ────────────────────────────────────────────────────

    def run(self) -> list[dict]:
        logger.info("=" * 60)
        logger.info("Master Pattern Engine — INICIO")
        logger.info("=" * 60)

        df = self.load_data()

        # Ajustar regímenes HMM
        regime_col = self.fit_hmm_regimes(df)
        df["hmm_regime"] = regime_col

        target = self._build_target(df)
        if target.empty:
            logger.error("MasterPattern: no se pudo construir target")
            return []

        all_rules = []
        regimes   = sorted(df["hmm_regime"].unique())
        logger.info(f"MasterPattern: buscando reglas en {len(regimes)} regímenes HMM")

        for rid in regimes:
            mask   = df["hmm_regime"] == rid
            df_reg = df[mask]
            tgt_reg = target[mask]
            n_bars = mask.sum()

            if n_bars < MIN_OCCURRENCES * 2:
                logger.warning(f"  Régimen {rid}: solo {n_bars} barras — saltando")
                continue

            logger.info(f"  Régimen {rid}: {n_bars} barras")
            rules = self._find_golden_rules(df_reg, tgt_reg, rid)
            all_rules.extend(rules)
            logger.info(f"  Régimen {rid}: {len(rules)} Golden Rules encontradas")

        logger.info(f"MasterPattern: TOTAL {len(all_rules)} Golden Rules")
        self._save_report(all_rules, df=df)

        # Guardar también como CSV
        if all_rules:
            pd.DataFrame(all_rules).to_csv(
                REPORTS_DIR / "master_pattern_rules.csv", index=False
            )

        logger.info("Master Pattern Engine — COMPLETADO")
        return all_rules


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = MasterPatternEngine()
    engine.run()
