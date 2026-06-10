"""
Deep Discovery Engine v2 — Luna V1 AI Mining (Engine 5/6)  ★ UPGRADED
====================================================================
Módulo 0 NUEVO — RFE Pre-Selector [GAP 1]:
  Antes del AG, un Random Forest rankea todas las 133+ features por
  Permutation Importance y poda hasta retener el 90% de varianza.
  Solo las features supervivientes entran al Algoritmo Genético.

Algoritmo Genético (20 gen, pop=60, tournament+crossover+mutation).

DTW Fractal Matching: top-5 análogos históricos 72H, probabilidad alcista.

Visual Analytics [GAP 2]:
  - engine_rfe_importance.png  (supervivientes RFE — bar chart)
  - engine_dtw_fractals.png    (ventanas análogas sobre precio)

Output:
  - deep_discovery_report.md
  - deep_discovery_rules.csv
  - engine_rfe_importance.png
  - engine_dtw_fractals.png
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import random
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_FEATURES = PROJECT_ROOT / "data" / "features"
REPORTS_DIR   = PROJECT_ROOT / "data" / "ai_mining" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config AG (Fix F2-03: leer desde settings.yaml, fallback a hardcoded) ─────
try:
    from config.settings import cfg
    _dd = cfg.ai_mining.deep_discovery
    N_GENERATIONS    = int(getattr(_dd, "genetic_n_generations", 20))
    # [FALLA-03-FIX 2026-05-30] Key correcto en settings.yaml es 'genetic_population'
    POPULATION_SIZE  = int(getattr(_dd, "genetic_population", getattr(_dd, "genetic_population_size", 60)))
    TOURNAMENT_K     = int(getattr(_dd, "genetic_tournament_k", 5))
    MUTATION_RATE    = float(getattr(_dd, "genetic_mutation_rate", 0.25))
    CROSSOVER_RATE   = float(getattr(_dd, "genetic_crossover_rate", 0.70))
    # [FALLA-03-FIX] Cobertura máxima tolerada antes de penalizar fitness (evitar reglas triviales)
    MAX_COVERAGE_PENALTY = float(getattr(_dd, "genetic_max_coverage", 0.20))
    # [H10H11-FIX 2026-05-30] Mínimo hits absoluto para regla estadísticamente válida
    # n=42 con WR=95% = memorización pura. n=500 garantiza ~0.8% del IS (63037 pts)
    MIN_HITS_ABS     = int(getattr(_dd, "genetic_min_hits", 500))
except Exception:
    N_GENERATIONS    = 20
    POPULATION_SIZE  = 60
    TOURNAMENT_K     = 5
    MUTATION_RATE    = 0.25   # fallback — overridden by settings.yaml (0.10)
    CROSSOVER_RATE   = 0.70   # fallback
    MAX_COVERAGE_PENALTY = 0.20  # fallback: penalizar reglas que activan >20% del dataset
    MIN_HITS_ABS     = 500   # fallback mínimo hits absoluto
MIN_OCCURRENCES  = 25     # mínimo hits para regla válida (legacy, reemplazado por MIN_HITS_ABS)
MIN_WIN_RATE     = 0.56   # umbral genético (un poco más laxo que golden)
MAX_CONDITIONS   = 4      # máximo condiciones por regla genética
HORIZON_H        = 24
ELITE_FRAC       = 0.10   # top 10% pasan directamente a la siguiente gen
# [FALLA-03-FIX] Base WinRate del target (se calcula dinámicamente en run(), fallback 0.54)
_BASE_WIN_RATE   = 0.54   # actualizado en run() con el WR real del dataset
print(f"[FALLA-03-FIX] AG Config cargado: N_GEN={N_GENERATIONS}, POP={POPULATION_SIZE}, "
      f"MUTATION={MUTATION_RATE:.2f}, MAX_COVERAGE_PENALTY={MAX_COVERAGE_PENALTY:.2f}, "
      f"MIN_HITS_ABS={MIN_HITS_ABS}")

# Features disponibles para el AG
AG_FEATURES = [
    "FedFundsRate", "YieldCurve_10Y3M", "T10Y2Y",
    "GlobalM2_Index", "Fed_Net_Liquidity", "CPI_YoY",
    "VIX", "DXY", "SP500_AboveMA200", "NASDAQ_Ret",
    "FundingRate", "DangerZone", "DVOL", "OI_BTC",
    "MVRV_Proxy", "FearGreed", "SSR", "Whale_Vol_ZScore",
    "Stablecoin_Cap", "DeFi_WBTC_TVL",
    "eth_btc_corr_24h", "eth_ret_lag1", "alt_season_proxy",
    "KMeans_Tribe_ID", "Master_Causal_Signal",
    "hashrate_7d_ma", "active_addresses_7d_ma",
    "Tx_Fees_USD", "Wiki_BTC_Views",
]

# DTW Config
DTW_WINDOW_H    = 72    # ventana de comparación en horas
DTW_LOOKBACK_H  = 4380  # 6 meses de historia para buscar análogos
DTW_N_MATCHES   = 5     # top-5 ventanas más similares


# ─────────────────────────────────────────────────────────────────────────────
# Representación de un individuo (regla genética)
# Una "regla" es una lista de (variable, operator, percentile_value)
# ─────────────────────────────────────────────────────────────────────────────

class GeneticRule:
    """Una regla genética: lista de condiciones (var, op, val)."""

    def __init__(self, conditions: list[tuple]):
        # conditions: [(var_name, operator, threshold_value), ...]
        self.conditions = conditions
        self.win_rate   = 0.0
        self.ev_pct     = 0.0
        self.n_hits     = 0
        self.fitness    = -1.0

    def evaluate(self, df: pd.DataFrame, target: pd.Series) -> bool:
        """Evalúa la regla sobre el DataFrame. Calcula fitness."""
        try:
            if not self.conditions:
                return False
            mask = pd.Series(True, index=df.index)
            for var, op, val in self.conditions:
                if var not in df.columns:
                    return False
                col = df[var]
                if op == ">=":
                    mask = mask & (col >= val)
                elif op == "<=":
                    mask = mask & (col <= val)
                elif op == "==":
                    mask = mask & (col == val)

            mask = mask & target.notna()
            t_masked = target[mask].sort_index()
            n = len(t_masked)
            if n < MIN_OCCURRENCES:
                self.fitness = -1.0
                return False
            # [H10H11-FIX 2026-05-30] Mínimo hits absoluto: n=42 con WR=95% es memorización pura
            # Una regla válida debe cubrir al menos MIN_HITS_ABS puntos para tener
            # significancia estadística en OOS (Wilson CI lower bound decente)
            min_hits = getattr(self, '_min_hits_abs', MIN_HITS_ABS)
            if n < min_hits:
                self.fitness = -1.0
                print(f"[H10H11-FIX] Regla descartada: n={n} < min_hits={min_hits} (memorización)")
                return False

            wr_global = float(t_masked.mean())
            
            # FIX FASE 5: Validación Cruzada Cronológica (H1 vs H2)
            half = n // 2
            wr_h1 = float(t_masked.iloc[:half].mean())
            wr_h2 = float(t_masked.iloc[half:].mean())
            wr_robust = min(wr_h1, wr_h2)

            ev = 0.0
            if "close" in df.columns:
                # Fix F2-02: EV calc restringido al training set (informacional, no fitness).
                # shift(-HORIZON_H) introduce look-ahead; solo válido para análisis histórico
                # en el training set. La función de fitness usa WR (no EV).
                future_ret = df["close"].shift(-HORIZON_H) / df["close"] - 1
                train_mask = mask & (df.index <= self.cutoff_date) if hasattr(self, "cutoff_date") and self.cutoff_date is not None else mask
                ev = float(future_ret[train_mask].mean()) if train_mask.sum() > 0 else 0.0

            self.n_hits   = int(n)
            self.win_rate = round(wr_global * 100, 1)
            self.ev_pct   = round(ev * 100, 2) if not np.isnan(ev) else 0.0

            # [FALLA-03-FIX 2026-05-30] Fitness corregido — penalizar reglas triviales de alta cobertura
            # Problema anterior: fitness = wr_robust * log1p(n_hits) premiaba reglas con 15k+ hits
            # que activaban el 25% del dataset con WR=60.4% -> f=5.826 (imbatible = Gen1 y se congela)
            #
            # Nueva función:
            #   wr_incremental = wr_robust - base_winrate  (incremento real sobre el random)
            #   coverage = n_hits / total_samples
            #   coverage_penalty = max(0, coverage - MAX_COVERAGE_PENALTY)  (0 si < 20%)
            #   fitness = wr_incremental * log1p(n_hits) * (1 - coverage_penalty)
            #
            # Esto asegura que reglas precisas (5% cobertura, 68% WR) superen a reglas
            # triviales (25% cobertura, 60% WR) en el ranking de fitness.
            if wr_robust >= MIN_WIN_RATE:
                # Base WR del target (tasa base sin condición)
                base_wr = getattr(self, '_base_win_rate', _BASE_WIN_RATE)
                total_n = getattr(self, '_total_samples', n * 4)  # estimacion conservadora
                # Incremento real sobre la base
                wr_incremental = max(0.0, wr_robust - base_wr)
                # Penalización por cobertura excesiva
                coverage = n / max(total_n, 1)
                coverage_excess = max(0.0, coverage - MAX_COVERAGE_PENALTY)
                coverage_penalty = min(coverage_excess * 2.0, 0.5)  # penalizar hasta -50%
                self.fitness = wr_incremental * np.log1p(n) * (1.0 - coverage_penalty)
                if coverage > MAX_COVERAGE_PENALTY:
                    print(f"[FALLA-03-FIX] Regla penalizada: cov={coverage:.1%} > {MAX_COVERAGE_PENALTY:.0%} "
                          f"| WR={wr_robust:.3f} | f={self.fitness:.4f} (sin penalizar sería {wr_incremental * np.log1p(n):.4f})")
            else:
                self.fitness = -1.0
            return True
        except Exception:
            self.fitness = -1.0
            return False

    def to_dict(self) -> dict:
        # Formato legible: NASDAQ_Ret >= 0.0094 AND T10Y2Y >= 0.70
        cond_readable = " AND ".join([f"{v} {op} {val:.4f}" for v, op, val in self.conditions])
        return {
            "conditions":       cond_readable,
            "pandas_eval":      " & ".join([f"({v} {op} {val:.4f})" for v, op, val in self.conditions]),
            "n_conditions":     len(self.conditions),
            "win_rate":         self.win_rate,
            "ev_pct":           self.ev_pct,
            "n_hits":           self.n_hits,
            "fitness":          round(self.fitness, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
class DeepDiscoveryEngine:

    def __init__(self, cutoff_date=None):
        # cutoff_date: pd.Timestamp | None
        # En --mode dev, limita los datos a <= train_end para evitar Selection Leakage.
        self.cutoff_date        = cutoff_date
        self.available_features: list[str] = []
        self.percentiles: dict[str, dict]  = {}
        self.rfe_survivors: list[str]       = []   # features tras poda RFE

    def load_data(self) -> pd.DataFrame:
        # P1-2-FIX (2026-03-30): eliminado features_train_kshape.parquet (K-Shape decommisionado).
        for name in ["features_train_causal.parquet",
                     "features_train.parquet"]:
            p = DATA_FEATURES / name
            if p.exists():
                df = pd.read_parquet(p)
                df.index = pd.to_datetime(df.index, utc=True)
                cutoff = getattr(self, "cutoff_date", None)
                if cutoff is not None:
                    n_before = len(df)
                    df = df[df.index <= cutoff]
                    logger.info(
                        f"Deep [DEV]: cutoff={cutoff.date()} "
                        f"-> {len(df)}/{n_before} filas ({len(df)/n_before*100:.1f}%)"
                    )
                else:
                    logger.info(f"Deep [PROD]: datos completos {df.shape} desde {name}")
                return df
        raise FileNotFoundError("No dataset encontrado")

    # ── MÓDULO 0: RFE Pre-Selector [GAP 1] ───────────────────────────────────
    def rfe_preselect(self, df: pd.DataFrame) -> list[str]:
        """
        Permutation Importance con RandomForest para podar features.
        Acumula importancia de mayor a menor hasta capturar el 90%
        de la varianza explicada — retorna las features supervivientes.
        Fallback: usa todas las AG_FEATURES disponibles.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.inspection import permutation_importance

            all_feats = [f for f in AG_FEATURES if f in df.columns]
            if len(all_feats) < 3:
                return all_feats

            if "close" not in df.columns:
                return all_feats

            target = (df["close"].shift(-24) / df["close"] - 1 > 0).astype(int)
            X = df[all_feats].ffill().fillna(0)
            y = target
            idx = X.index.intersection(y.dropna().index)
            X, y = X.loc[idx], y.loc[idx]

            # Train en primeros 75%
            n = len(X)
            X_tr, y_tr = X.iloc[:int(n*0.75)], y.iloc[:int(n*0.75)]
            X_te, y_te = X.iloc[int(n*0.75):], y.iloc[int(n*0.75):]

            import os
            _seed = int(os.environ.get("LUNA_SEED", 42))
            logger.info(f"RFE: entrenando RF sobre {len(all_feats)} features...")
            rf = RandomForestClassifier(
                n_estimators=150, max_depth=5,
                min_samples_leaf=50, n_jobs=-1, random_state=_seed
            )
            rf.fit(X_tr, y_tr)
            acc = rf.score(X_te, y_te)
            logger.info(f"RFE RF OOS accuracy: {acc:.3f}")

            # Permutation Importance en test
            perm = permutation_importance(rf, X_te, y_te, n_repeats=5,
                                          random_state=_seed, n_jobs=-1)
            imp = pd.Series(perm.importances_mean, index=all_feats)
            imp = imp.clip(lower=0)  # ignorar negativas (sin efecto)
            imp_sorted = imp.sort_values(ascending=False)

            # Acumular hasta 90% de la varianza explicada
            total = imp_sorted.sum()
            if total <= 0:
                logger.warning("RFE: importancias todas cero — usando todas las features")
                return all_feats

            cumulative = 0.0
            survivors = []
            for feat, val in imp_sorted.items():
                survivors.append(feat)
                cumulative += val
                if cumulative / total >= 0.90:
                    break

            # Mínimo 5 features
            if len(survivors) < 5:
                survivors = imp_sorted.head(5).index.tolist()

            logger.info(f"RFE: {len(all_feats)} → {len(survivors)} features "
                        f"(90% varianza) — supervivientes: {survivors}")

            # Plot RFE importance [GAP 2]
            self._plot_rfe(imp_sorted, survivors)
            self.rfe_survivors = survivors
            return survivors

        except Exception as e:
            logger.warning(f"RFE pre-selector falló ({e}) — usando AG_FEATURES completo")
            fallback = [f for f in AG_FEATURES if f in df.columns]
            self.rfe_survivors = fallback
            return fallback

    # ── Visual: RFE importance bar chart [GAP 2] ─────────────────────────────
    def _plot_rfe(self, imp_sorted: pd.Series, survivors: list[str]) -> None:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            top = imp_sorted.head(20)
            colors = ["#2ecc71" if f in survivors else "#e74c3c" for f in top.index]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(top)), top.values[::-1], color=colors[::-1])
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.index[::-1], fontsize=9)
            ax.set_xlabel("Permutation Importance")
            ax.set_title("RFE — Feature Survival (verde=superviviente, rojo=podada)",
                         fontweight="bold")
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color="#2ecc71", label="Superviviente (AG)"),
                               Patch(color="#e74c3c", label="Podada")],
                      loc="lower right")
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "engine_rfe_importance.png", dpi=120, bbox_inches="tight")
            plt.close()
            logger.success("engine_rfe_importance.png generado")
        except Exception as e:
            logger.warning(f"RFE plot: {e}")

    # ── Visual: DTW fractals [GAP 2] ─────────────────────────────────────────
    def _plot_dtw(self, df: pd.DataFrame, dtw_res: dict) -> None:
        try:
            if not dtw_res.get("similar_windows") or "close" not in df.columns:
                return
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            close = df["close"].dropna()
            n = len(close)
            n_analogs = len(dtw_res["similar_windows"])
            fig, axes = plt.subplots(n_analogs + 1, 1,
                                     figsize=(12, 3 * (n_analogs + 1)))
            # Panel 0: ventana actual
            curr = close.iloc[-DTW_WINDOW_H:].values
            curr_n = (curr - curr.mean()) / (curr.std() + 1e-8)
            axes[0].plot(curr_n, color="#3498db", linewidth=2.0)
            axes[0].set_title("Ventana Actual (últimas 72H) — normalizada",
                              fontweight="bold")
            axes[0].set_ylabel("Z-score")
            # Paneles análogos
            for k, win_info in enumerate(dtw_res["similar_windows"]):
                win_date = win_info["date"]
                emoji = "🟢" if win_info["bullish"] else "🔴"
                color = "#2ecc71" if win_info["bullish"] else "#e74c3c"
                # Encontrar índice de esa fecha
                matches = close.index[close.index.date == pd.Timestamp(win_date).date()]
                if len(matches) == 0:
                    axes[k+1].set_visible(False)
                    continue
                idx_pos = close.index.get_loc(matches[0])
                hist = close.iloc[max(0, idx_pos):min(n, idx_pos + DTW_WINDOW_H)].values
                hist_n = (hist - hist.mean()) / (hist.std() + 1e-8)
                axes[k+1].plot(hist_n, color=color, linewidth=1.5)
                axes[k+1].set_title(
                    f"{emoji} Análogo {win_date} | DTW={win_info['distance']:.4f} "
                    f"| Ret={win_info['future_ret']:+.1f}%"
                )
            plt.suptitle(f"DTW Fractal Matching — Bull Prob = {dtw_res['dtw_bull_prob']*100:.0f}%",
                         fontsize=12, fontweight="bold", y=1.01)
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "engine_dtw_fractals.png", dpi=100, bbox_inches="tight")
            plt.close()
            logger.success("engine_dtw_fractals.png generado")
        except Exception as e:
            logger.warning(f"DTW plot: {e}")

    def _build_target(self, df: pd.DataFrame) -> pd.Series:
        # [BUG-TARGET-FIX] Priorizando Target_TBM_Bin como objetivo causal/genético para consistencia con TBM
        if "Target_TBM_Bin" in df.columns:
            print("[BUG-TARGET-FIX] DeepDiscovery: Detectada y priorizada columna Target_TBM_Bin")
            logger.info("[BUG-TARGET-FIX] DeepDiscovery: Usando Target_TBM_Bin como target principal para el algoritmo genético.")
            return df["Target_TBM_Bin"]
        if "target" in df.columns:
            return df["target"]
        if "close" in df.columns:
            return (df["close"].shift(-HORIZON_H) / df["close"] - 1 > 0.005).astype(int)
        return pd.Series(dtype=int)

    def _compute_percentiles(self, df: pd.DataFrame) -> None:
        """Precomputa percentiles 10, 25, 50, 75, 90 de cada feature.
        
        [H10H11-FIX 2026-05-30] Ampliado de 3 percentiles (q25/q50/q75) a 5 (q10/q25/q50/q75/q90).
        Espacio de genes anterior: 5_features × 2_percentiles = 10 umbrales posibles.
        Pop=60 con 10 genes únicos → colapso de diversidad garantizado en Gen1.
        Espacio nuevo: 5_features × 4_umbrales (q10<=, q25<=, q75>=, q90>=) = 20 genes.
        Con pop=60 y 20 genes × combinaciones de 2-3 condiciones → diversidad mantenida.
        """
        print(f"[H10H11-FIX] Calculando percentiles extendidos para {len(self.available_features)} features")
        for feat in self.available_features:
            col = df[feat].dropna()
            self.percentiles[feat] = {
                "q10": col.quantile(0.10),
                "q25": col.quantile(0.25),
                "q50": col.quantile(0.50),
                "q75": col.quantile(0.75),
                "q90": col.quantile(0.90),
            }

    # ── AG: Inicialización ────────────────────────────────────────────────────

    def _random_condition(self) -> tuple:
        """Genera una condición aleatoria: (var, op, percentile_val).
        
        [H10H11-FIX 2026-05-30] Expandido el espacio de genes de 2 a 4 umbrales.
        Anterior: sólo q25 (<=) o q75 (>=) → 2 operadores × n_features genes únicos.
        Nuevo: q10/q25 (<=, extremo bajo/bajo) y q75/q90 (>=, alto/extremo alto).
        Esto cuadruplica el espacio de búsqueda sin añadir complejidad computacional.
        """
        var = random.choice(self.available_features)
        # [H10H11-FIX] 4 umbrales posibles: extremo bajo, bajo, alto, extremo alto
        pct_choice = random.choice(["q10", "q25", "q75", "q90"])
        op  = "<=" if pct_choice in ("q10", "q25") else ">="
        val = self.percentiles[var][pct_choice]
        return (var, op, val)

    def _random_individual(self) -> GeneticRule:
        n_cond = random.randint(1, MAX_CONDITIONS)
        return GeneticRule([self._random_condition() for _ in range(n_cond)])

    def _initialize_population(self) -> list[GeneticRule]:
        return [self._random_individual() for _ in range(POPULATION_SIZE)]

    # ── AG: Selección, Cruce, Mutación ───────────────────────────────────────

    def _tournament_select(self, population: list[GeneticRule]) -> GeneticRule:
        tournament = random.sample(population, min(TOURNAMENT_K, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: GeneticRule, parent2: GeneticRule) -> GeneticRule:
        """Cruce genético mejorado.
        
        [H10H11-FIX 2026-05-30] Bug anterior: concatenar condiciones de ambos padres
        y tomar c[:n_cond] siempre tomaba las condiciones del padre con más fitness
        (que dominaba el torneo), produciendo copias del mismo individuo.
        
        Nuevo: cruce por intercalado posicional + punto de corte aleatorio.
        Si padre1 = [A, B] y padre2 = [C, D, E]:
          - Punto corte en p1: e.g. 1 → toma [A] de padre1
          - Completa con condiciones de padre2 no duplicadas: [A, C, D] (max 4)
        Esto garantiza que el hijo siempre hereda algo de cada padre.
        """
        if random.random() > CROSSOVER_RATE:
            return GeneticRule(parent1.conditions[:])
        # [H10H11-FIX] Cruce por punto de corte + completar con padre2
        c1 = parent1.conditions
        c2 = parent2.conditions
        if not c1:
            return GeneticRule(c2[:])
        if not c2:
            return GeneticRule(c1[:])
        # Punto de corte aleatorio en padre1
        cut = random.randint(1, len(c1))
        child_conds = c1[:cut]
        # Añadir condiciones de padre2 que no dupliquen variable ya presente
        existing_vars = {v for v, _, _ in child_conds}
        for cond in c2:
            if len(child_conds) >= MAX_CONDITIONS:
                break
            if cond[0] not in existing_vars:  # no duplicar variable
                child_conds.append(cond)
                existing_vars.add(cond[0])
        if not child_conds:
            child_conds = c1[:]
        print(f"[H10H11-FIX] Crossover: p1({len(c1)}cond)+p2({len(c2)}cond) → hijo({len(child_conds)}cond)") if random.random() < 0.01 else None
        return GeneticRule(child_conds)

    def _mutate(self, individual: GeneticRule) -> GeneticRule:
        if random.random() > MUTATION_RATE:
            return individual
        conditions = individual.conditions[:]
        mutation_type = random.choice(["replace", "add", "remove"])
        if mutation_type == "replace" and conditions:
            idx = random.randint(0, len(conditions) - 1)
            conditions[idx] = self._random_condition()
        elif mutation_type == "add" and len(conditions) < MAX_CONDITIONS:
            conditions.append(self._random_condition())
        elif mutation_type == "remove" and len(conditions) > 1:
            conditions.pop(random.randint(0, len(conditions) - 1))
        return GeneticRule(conditions)

    # ── AG: Evolución ────────────────────────────────────────────────────────

    def _evolve(self, df: pd.DataFrame, target: pd.Series) -> list[GeneticRule]:
        """Corre el algoritmo genético por N_GENERATIONS generaciones."""
        population = self._initialize_population()
        best_rules: list[GeneticRule] = []

        for gen in range(N_GENERATIONS):
            # Evaluar
            for ind in population:
                if ind.fitness < 0:  # solo si no evaluado o inválido
                    ind.evaluate(df, target)

            # Ordenar por fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            elite_n = max(1, int(POPULATION_SIZE * ELITE_FRAC))

            # Guardar élite
            for ind in population[:elite_n]:
                if ind.fitness > 0:
                    best_rules.append(ind)

            best_gen = population[0]
            logger.info(
                f"  Gen {gen+1:02d}/{N_GENERATIONS}  "
                f"Best: WR={best_gen.win_rate:.1f}%  "
                f"hits={best_gen.n_hits}  f={best_gen.fitness:.3f}"
            )

            # Nueva generación
            new_population = population[:elite_n]  # élite sobrevive

            while len(new_population) < POPULATION_SIZE:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                child.fitness = -1.0  # marcado para re-evaluación
                new_population.append(child)

            population = new_population

        # Deduplicar reglas (por pandas_eval)
        seen: set[str] = set()
        unique_rules: list[GeneticRule] = []
        for r in sorted(best_rules, key=lambda x: x.fitness, reverse=True):
            key = r.to_dict()["pandas_eval"]
            if key not in seen and r.fitness > 0:
                seen.add(key)
                unique_rules.append(r)
        return unique_rules[:50]  # top-50 únicas

    # ── DTW Fractal Matching ───────────────────────────────────────────────

    def dtw_fractal_match(self, df: pd.DataFrame) -> dict:
        """
        Busca en el histórico las ventanas de 72H más similares a
        las últimas 72H usando DTW distance sobre precio normalizado.

        Retorna probabilidad de éxito empírica y ventanas análogas.
        """
        if "close" not in df.columns or len(df) < DTW_WINDOW_H * 3:
            return {"dtw_bull_prob": 0.5, "n_matches": 0, "similar_windows": []}

        try:
            from dtaidistance import dtw as dtw_lib
            _dtw_fn = dtw_lib.distance
        except ImportError:
            # Fallback: Euclidean distance sobre ventana normalizada
            def _dtw_fn(a: np.ndarray, b: np.ndarray) -> float:
                a_n = (a - a.mean()) / (a.std() + 1e-8)
                b_n = (b - b.mean()) / (b.std() + 1e-8)
                return float(np.sqrt(np.mean((a_n - b_n) ** 2)))

        close = df["close"].dropna()
        n     = len(close)

        # Ventana actual (últimas 72H)
        current_window = close.iloc[-DTW_WINDOW_H:].values
        curr_norm = (current_window - current_window.mean()) / (current_window.std() + 1e-8)

        # Búsqueda en el histórico
        distances: list[tuple[float, int]] = []
        start_idx = max(0, n - DTW_LOOKBACK_H - DTW_WINDOW_H)

        for i in range(start_idx, n - DTW_WINDOW_H * 2, 6):  # stride=6H
            hist_window = close.iloc[i:i + DTW_WINDOW_H].values
            hist_norm = (hist_window - hist_window.mean()) / (hist_window.std() + 1e-8)
            dist = _dtw_fn(curr_norm, hist_norm)
            distances.append((dist, i))

        distances.sort(key=lambda x: x[0])
        top_matches = distances[:DTW_N_MATCHES]

        # Calcular probabilidad de éxito (sube en siguiente HORIZON_H)
        n_bull = 0
        similar_windows = []
        for dist, idx in top_matches:
            future_idx = idx + DTW_WINDOW_H + HORIZON_H
            if future_idx < n:
                future_ret = close.iloc[future_idx] / close.iloc[idx + DTW_WINDOW_H] - 1
                bullish = future_ret > 0
                n_bull += int(bullish)
                similar_windows.append({
                    "date": str(close.index[idx].date()),
                    "distance": round(dist, 4),
                    "future_ret": round(float(future_ret) * 100, 2),
                    "bullish": bool(bullish),
                })

        bull_prob = n_bull / len(top_matches) if top_matches else 0.5
        return {
            "dtw_bull_prob":    round(bull_prob, 3),
            "n_matches":        len(top_matches),
            "similar_windows":  similar_windows,
        }

    # ── Reporte ──────────────────────────────────────────────────────────────

    def _save_report(self, rules: list[GeneticRule], dtw_res: dict) -> None:
        """Reporte estilo Correlaciones: DE + DTW + RFE narrativo."""
        ts = pd.Timestamp.now().strftime("%d %B %Y %H:%M")
        dtw_prob = dtw_res.get('dtw_bull_prob', 0.5)
        dtw_emoji = "🟢 ALCISTA" if dtw_prob >= 0.6 else "🔴 BAJISTA" if dtw_prob <= 0.4 else "⚖️ NEUTRAL"
        top_rule = rules[0].to_dict() if rules else {}
        top_wr = top_rule.get("win_rate", 0)
        top_ev = top_rule.get("ev_pct", 0)
        top_ev_str = f"+{top_ev}%" if top_ev >= 0 else f"{top_ev}%"

        report = [
            f"# 🧬 DEEP DISCOVERY ENGINE — Luna V1",
            f"**Generado:** {ts} | **Generaciones:** {N_GENERATIONS} × Pob={POPULATION_SIZE} | **Reglas únicas:** {len(rules)} | **RFE features:** {len(self.rfe_survivors)}",
            "",
            "> *Tres módulos complementarios de descubrimiento profundo:*",
            "> *① RFE Pre-Selector (RandomForest Permutation Importance) → ② Algoritmo Genético (WR × log(N)) → ③ DTW Fractal Matching (análogos históricos 72H)*",
            "",
            "---",
            "",
            "## 🔬 1. Supervivientes RFE — Features relevantes para el AG",
            "",
            f"> De las {len(self.rfe_survivors) + 5}+ features candidatas, **{len(self.rfe_survivors)} supervivieron** el filtro de Permutation Importance (capturando ≥90% de varianza).",
            "",
            f"**Features supervivientes:** `{'`, `'.join(self.rfe_survivors[:10])}`{'...' if len(self.rfe_survivors) > 10 else ''}",
            "",
            "> [!NOTE]",
            "> Las features podadas por RFE tienen importancia near-zero en el Random Forest — incluirlas introduciría ruido al AG.",
            "> Ver imagen `engine_rfe_importance.png` para el ranking visual completo.",
            "",
            "---",
            "",
            "## 🕰️ 2. DTW Fractal Matching — Análogos Históricos 72H",
            "",
            f"> **Probabilidad alcista DTW: {dtw_prob*100:.1f}% → {dtw_emoji}**",
            f"> Basado en los {dtw_res.get('n_matches', len(dtw_res.get('similar_windows', [])))} análogos históricos más similares a la ventana actual (últimas 72H).",
            "",
            "| Fecha Análogo | Distancia DTW | Retorno 24H Post | Resultado |",
            "| --- | --- | --- | --- |",
        ]
        for w in dtw_res.get("similar_windows", []):
            bull_emoji = "🟢" if w["bullish"] else "🔴"
            ret_str = f"+{w['future_ret']}%" if w['future_ret'] >= 0 else f"{w['future_ret']}%"
            report.append(
                f"| {w['date']} | {w['distance']:.4f} | {ret_str} | {bull_emoji} |"
            )

        report += [
            "",
            "> [!NOTE]",
            "> Distancia DTW normalizada — menor = más similar en forma. Los análogos con Ret < 0 no invalidan la señal",
            "> si la mayoría es alcista. La probabilidad es empírica sobre los top-5 análogos.",
            "",
            "---",
            "",
            "## 🧬 3. Reglas Genéticas — Top 20 (Algoritmo Evolutivo)",
            "",
            "> **Reglas descubiertas por selección evolutiva.** Cada regla es una conjunción de condiciones IF/AND",
            "> sobre las features supervivientes del RFE. EV = retorno forward real (shift corregido).",
            "",
            "| Condiciones (IF/AND) | Win Rate | EV 24H | N Hits |",
            "| --- | --- | --- | --- |",
        ]
        for r in rules[:20]:
            d = r.to_dict()
            cond_cells = "<br>**AND** ".join([f"`{c.strip()}`" for c in d["conditions"].split(" AND ")])
            ev = d['ev_pct']
            ev_str = f"+{ev}%" if ev >= 0 else f"{ev}%"
            wr_str = f"**{d['win_rate']}%**" if d['win_rate'] >= 60 else f"{d['win_rate']}%"
            report.append(f"| {cond_cells} | {wr_str} | {ev_str} | {d['n_hits']} |")

        # Top rule playbook
        if top_rule:
            cond_str = " AND ".join(top_rule.get("conditions", "N/A").split(" AND ")[:2])
            report += [
                "",
                "---",
                "",
                "## 📓 4. Playbook Táctico — Deep Discovery",
                "",
                f"### 🛡️ Regla Genética Óptima",
                f"- **Condición:** `{top_rule.get('conditions', 'N/A')[:120]}`",
                f"- **Histórico:** Win Rate **{top_wr}%** | EV forward 24H: **{top_ev_str}** | {top_rule.get('n_hits', 0)} hits",
                f"- **DTW confirma:** {dtw_emoji} ({dtw_prob*100:.0f}% análogos alcistas)",
                "",
                f"> [!TIP]",
                f"> **Uso:** El AG descubre reglas que el motor Golden Rules (Master Pattern) puede perder",
                f"> por buscar en espacio de features diferente. Crosscheck ambos para máxima confluence.",
            ]

        path = REPORTS_DIR / "deep_discovery_report.md"
        path.write_text("\n".join(report), encoding="utf-8")
        logger.success(f"Deep: reporte mejorado guardado en {path}")

    # ── Pipeline principal ────────────────────────────────────────────────────

    def run(self) -> list[GeneticRule]:
        logger.info("=" * 60)
        logger.info("Deep Discovery Engine v2 — INICIO (RFE + AG + DTW)")
        logger.info("=" * 60)

        import os
        _seed = int(os.environ.get("LUNA_SEED", 42))
        random.seed(_seed)
        np.random.seed(_seed)
        logger.info(f"Deep: random_state configurado con LUNA_SEED={_seed}")

        df = self.load_data()
        target = self._build_target(df)

        # [FALLA-03-FIX 2026-05-30] Calcular base_winrate dinámico y total_samples
        # para inyectarlos en GeneticRule.evaluate() via atributos de clase global
        global _BASE_WIN_RATE
        _total_valid = int(target.notna().sum())
        _base_wr = float(target.dropna().mean()) if _total_valid > 0 else 0.54
        _BASE_WIN_RATE = _base_wr
        logger.info(f"[FALLA-03-FIX] Base WinRate dinámico del target: {_base_wr:.4f} ({_base_wr*100:.1f}%) | "
                    f"Total samples válidos: {_total_valid}")
        print(f"[FALLA-03-FIX] Base WinRate: {_base_wr:.4f} — reglas deben superar este umbral para fitness>0")

        # ── MÓDULO 0: RFE Pre-Selector [GAP 1] ───────────────────────────────
        logger.info("Deep: Módulo 0 — RFE Permutation Importance...")
        rfe_features = self.rfe_preselect(df)

        # Usar supervivientes RFE (intersección con AG_FEATURES disponibles)
        self.available_features = rfe_features
        if not self.available_features:
            logger.error("Deep: ninguna feature disponible para el AG")
            return []
        logger.info(f"Deep: AG con {len(self.available_features)} features tras RFE")

        self._compute_percentiles(df)

        # [FALLA-03-FIX + H10H11-FIX] Inyectar total_samples, base_wr y min_hits_abs en cada GeneticRule
        # Para que evaluate() pueda calcular cobertura = n_hits / total_samples y validar min_hits
        _total_samples_for_fitness = _total_valid
        original_init = GeneticRule.__init__
        def _patched_init(self_rule, conditions):
            original_init(self_rule, conditions)
            self_rule._total_samples = _total_samples_for_fitness
            self_rule._base_win_rate = _BASE_WIN_RATE
            self_rule._min_hits_abs  = MIN_HITS_ABS  # [H10H11-FIX] mínimo estadístico
        GeneticRule.__init__ = _patched_init
        logger.info(f"[FALLA-03-FIX] GeneticRule parcheado: total_samples={_total_samples_for_fitness}, "
                    f"base_wr={_BASE_WIN_RATE:.4f}, min_hits_abs={MIN_HITS_ABS}")
        print(f"[FALLA-03-FIX] GeneticRule.__init__ parcheado con total_samples={_total_samples_for_fitness}, "
              f"min_hits_abs={MIN_HITS_ABS}")

        # ── Evolución genética ────────────────────────────────────────────────
        logger.info(f"Deep: evolucionando {N_GENERATIONS} generaciones, "
                    f"población={POPULATION_SIZE}")
        rules = self._evolve(df, target)
        logger.success(f"Deep: {len(rules)} reglas genéticas únicas encontradas")

        # ── DTW Fractal Matching ──────────────────────────────────────────────
        logger.info("Deep: ejecutando DTW Fractal Matching...")
        dtw_res = self.dtw_fractal_match(df)
        logger.info(f"Deep: DTW Bull Probability = {dtw_res['dtw_bull_prob']*100:.1f}%")

        # ── Visual Analytics [GAP 2] ─────────────────────────────────────────
        self._plot_dtw(df, dtw_res)

        # ── Guardar CSV ───────────────────────────────────────────────────────
        if rules:
            rules_df = pd.DataFrame([r.to_dict() for r in rules])
            rules_df.to_csv(REPORTS_DIR / "deep_discovery_rules.csv", index=False)

        self._save_report(rules, dtw_res)
        logger.info("Deep Discovery Engine v2 — COMPLETADO")
        return rules


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = DeepDiscoveryEngine()
    engine.run()
