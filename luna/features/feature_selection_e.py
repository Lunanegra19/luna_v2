"""
Feature Selection Pipeline — Luna V1 Fase E
============================================================
Implementa el pipeline completo de 5 etapas de López de Prado
(Advances in Financial Machine Learning, Ch. 7-8).

PROPÓSITO: Reducir 133+ features a ~15-20 features óptimas
verificadas estadísticamente antes de entrenar XGBoost.

SECUENCIA OBLIGATORIA:
  [A] FracDiff Dinámico (externo, en feature_pipeline.py)
  [B] Clustering Jerárquico      → ~40 representantes únicos
  [C] Automatic Lag Discovery MI → ~45 features (+ alpha signals)
  [D] SFI-CPCV                   → ~20-30 supervivientes (Deflated Sharpe ≥ 0)
  [E] Forward Feature Selection  → ~15-20 features óptimas

REGLA R22 (actualizado BUG-R12-01): Las señales alpha de Fase D Sí pasan por Etapa D
(SFI-CPCV) — saltan Etapa C (lags ya baked-in por el Mining).
NOTA: alpha_combined fue eliminada — era función lineal de alpha_golden_score y
alpha_genetic_score (corr≈1.0). Sustituida por alpha_storm_intensity (ortogonal).

TIEMPO ESTIMADO EN CPU: ~8 minutos (26 candidatos tras clustering jerárquico B → 15 clusters).

OUTPUT: data/features/selected_features.json
        data/features/feature_selection_report.md

ARQUITECTURA DE PROCESO AISLADO (SOP - RESOLUCIÓN CASO 1):
  Este script y sus clases primarias (SFI_CPCV y AdversarialValidator) están estructurados
  para ser orquestados por el cerebro central (luna/pipeline_executor.py) mediante llamadas
  a nivel de sistema operativo como subproceso independiente ("python -u").
  Esto garantiza la total liberación de memoria RAM y prevención de fugas entre ciclos WFB.
  NOTA AST/GRAPHIFY: Al ejecutarse como un script ejecutable autónomo con su propio bloque 
  __main__, no existen declaraciones de "import" estáticas de estas clases desde otros archivos,
  por lo que herramientas de análisis AST estático las listarán como "isolated nodes" (falso positivo
  de código muerto). Ambas clases se instancian y ejecutan en tiempo de corrida de forma activa.

Uso:
    python core/features/feature_selection_e.py
    python core/features/feature_selection_e.py --resume  # retoma desde C si hay checkpoint
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import os
_LUNA_SEED = int(os.environ.get("LUNA_SEED", 42))
print(f"[AUDIT-FIX] LUNA_SEED={_LUNA_SEED} inyectado en feature_selection_e.py para stochastic ensemble")
from loguru import logger
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import hashlib
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from luna.utils.encoding_fix import fix_stdout_encoding; fix_stdout_encoding()

DATA_DIR     = PROJECT_ROOT / "data" / "features"
OUTPUT_FILE  = DATA_DIR / "selected_features.json"
REPORT_FILE  = DATA_DIR / "feature_selection_report.md"
CHECKPOINT   = DATA_DIR / "_fs_checkpoint.json"
DSR_CACHE    = DATA_DIR / "_dsr_cache.json"   # MEJORA-SFI-B03: cache DSR adaptativo

# ————————————————— Alpha signals de Fase D (saltan Etapa C, sí pasan Etapa D — R22) —————————————————
# BUG-R12-01 fix (2026-03-10): alpha_combined ELIMINADA.
#   Era = 0.6×alpha_golden_score + 0.4×alpha_genetic_score×2 - 1
#   → función lineal exacta de golden+genetic → corr(A,C) y corr(B,C) ≈ 1.0
#   → redundancia matemática garantizada, slot SFI desperdiciado.
#   Sustituida por alpha_storm_intensity (ortogonal: cuenta reglas, no pondera por WR).
ALPHA_SIGNALS = [
    # Señales principales de Mining (mutuamente informativas, no linealmente dependientes)
    # [SFI-BALANCE-01 2026-06-03] alpha_golden_score y alpha_genetic_score MOVIDAS a PASSTHROUGH_FEATURES.
    # Motivo: competían en los Top-N slots contextuales ganando 2/13 slots (15% del presupuesto)
    # pese a tener DSR≈0 individual. Al estar en PASSTHROUGH siempre entran en XGBoost sin coste.
    # Esto libera 2 slots para features macro/onchain OOS-estables con DSR genuino.
    "alpha_storm_intensity",    # fracción Golden Rules simultáneas [0,1]    — concentración (ORTOGONAL)
    "alpha_dtw_signal",         # señal DTW + momentum ∈ [-1,1]
    "alpha_tribe_bias",         # sesgo de tribu K-Shape (-1/0/1)
    # M1: Meta-Oracle Score (síntesis de 6 motores de Mining)
    "meta_oracle_score",
    # M2: Reglas genéticas como binarios independientes (pass-through también en PASSTHROUGH)
    "genetic_rule_0",
    "genetic_rule_1",
    "genetic_rule_2",
    # ELIMINADAS:
    # "alpha_combined",  # BUG-R12-01: f(golden,genetic) — correlación perfecta, sin información nueva
]

# ————————————————— Pass-Through Features (saltan SFI — van directo a XGBoost training) —————————————————
# Señales raras (N<500 activaciones en 5 años) que el SFI-CPCV destruye
# por N insuficiente para CV estable. XGBoost las valida internamente via
# regularización (min_child_weight, L1/L2). Validadas con Bootstrap WR en
# core/features/rule_validator.py antes del training.
# ————————————————— Columnas a excluir explícitamente del pool raw_cols —————————————————
# BUG-M23-01 (2026-03-16): estas columnas son OHLC raw (correlación ~1.0 con close)
# o fueron eliminadas por diseño (alpha_combined). Sin esta blacklist, el clustering
# desperdicia slots y el SFI las evalúa con MeanSR fortemente negativo.
RAW_COLS_BLACKLIST: set[str] = {
    "open", "high", "low", "volume",   # OHLC raw — ruidosas, nunca mejores que features derivadas
    "alpha_combined",                   # BUG-R12-01: f(golden,genetic) — correlación perfecta
    "Target_TBM_Bin",                   # [BUG-LEAK-01] Target OOS, leak prevencion
}

# [SANEAMIENTO-V2 2026-06-07] Prohibición estructural de variables TIPO-1
# En la nueva arquitectura de alta frecuencia (barrera de 48H), las variables macro/onchain
# ultra lentas (mensuales/semanales) introducen ruido y violan la causalidad de alta rotación.
TIPO1_SLOW_SUBSTRINGS: tuple[str, ...] = (
    "unemploy", "m2", "cpi", "inflation", "fed_net_liquidity", 
    "lth_supply", "vix", "t10y2y", "gold_ret",
    "g3_net", "rrp", "pmi", "gdp", "dxy", "puell", "mvrv", "nvt", 
    "sopr", "hash_ribbon", "russell", "bito", "treasury", "fomc",
    "oil_ret"
)

LOW_FREQ_PREFIX = ("genetic_rule_", "storm_intensity", "golden_rule_")
LOW_FREQ_LIMIT = 0.05

PASSTHROUGH_FEATURES: list[str] = [
    # [SFI-BALANCE-01 2026-06-03] alpha_golden_score y alpha_genetic_score añadidas aquí.
    # Antes estaban en ALPHA_SIGNALS compitiendo en top-N y ocupando 2/13 slots contextuales.
    # Al moverlas a PASSTHROUGH: siempre entran en XGBoost (garantizado) y liberan slots para macro.
    "alpha_golden_score",  # promedio WR Golden Rules [0,1] — señal calidad (WR media ~65%)
    "alpha_genetic_score", # promedio WR Genetic Rules [0,1] — señal calidad
    # Golden Rules individuales (WR 69-89%, N=30-58 hits cada una)
    "golden_rule_0",   # Fed_Net_Liq ≤ -727k AND SSR ≥ 2.66      WR=88.9%
    "golden_rule_1",   # Fed_Net_Liq AND SP500_AboveMA200 AND SSR  WR=88.9%
    "golden_rule_2",   # MVRV≥2.22 AND FearGreed≥84 AND SSR        WR=82.6%
    "golden_rule_3",   # YieldCurve≥1.6 AND FundingRate≤-0.0002   WR=81.2%
    "golden_rule_4",   # Fed_Net_Liq AND FundingRate≤-0.0002       WR=80.6%
    "golden_rule_5",   # YieldCurve AND Fed_Net_Liq AND Funding    WR=80.6%
    "golden_rule_6",   # DXY≥99.58 AND FundingRate≤-0.0002        WR=80.0%
    "golden_rule_7",   # Funding AND Whale_Vol AND MasterCausal     WR=76.5%
    "golden_rule_8",   # DXY≥99.58 AND Whale_Vol_ZScore             WR=71.9%
    "golden_rule_9",   # Fed_Net_Liq AND DXY≥99.58                 WR=70.2%
    "golden_rule_10",  # T10Y2Y≥1.14 AND MVRV_Proxy≥2.22          WR=70.0%
    "golden_rule_11",  # T10Y2Y AND SP500_AboveMA200 AND MVRV      WR=70.0%
    "golden_rule_12",  # T10Y2Y AND MVRV AND SSR                   WR=70.0%
    "golden_rule_13",  # FundingRate AND Master_Causal              WR=69.8%
    "golden_rule_14",  # MVRV_Proxy≥2.22 AND SSR≥2.66              WR=69.0%
    # Genetic Rules individuales (WR 56-60%, N=234-336 hits)
    "genetic_rule_0",  # NASDAQ_Ret≥0.009 AND active_addr AND T10Y2Y  WR=59.5%
    "genetic_rule_1",  # NASDAQ_Ret AND SSR AND T10Y2Y AND KShape     WR=56.4%
    "genetic_rule_2",  # NASDAQ_Ret AND KShape AND SSR AND T10Y2Y     WR=56.4%
    "HMM_Regime",      # [FIX-HMM-01] Passthrough obligatorio para condicionamiento causal
    "HMM_Semantic",    # [FIX-HMM-01] Etiqueta string
    "close_fd",        # [FIX-PASSTHROUGH-FRAC-01] FracDiff es estructural y analiticamente comprobado. Salta SFI.
]

# ————————————————— Parámetros del pipeline —————————————————
# ARCH-03 (2026-03-10): todas las constantes se leen desde cfg.features en settings.yaml.
# Sin hardcodes: cambiar el valor en settings.yaml lo propaga automáticamente al pipeline.
try:
    from config.settings import cfg as _cfg_sfi
    CLUSTER_FIXED_N       = int(getattr(_cfg_sfi.features, 'sfi_n_clusters',    15))
    MAX_LAG_HOURS         = int(getattr(_cfg_sfi.features, 'max_lag_hours',     500))
    SFI_TOP_N_FEATURES    = int(getattr(_cfg_sfi.features, 'sfi_top_n',         15))
    SFI_N_GROUPS          = int(getattr(_cfg_sfi.features, 'sfi_n_groups',       6))
    # LOGIC-SFI-01 FIX (2026-04-06): Separar las fuentes de Purge y Embargo.
    # Antes: ambos leían de 'embargo_hours' → purge == embargo == 96H
    # → 192H de exclusión total (96H antes + 96H después de cada test fold).
    # Correcto: purge = horizonte TBM (hacia atrás), embargo = cooldown post-test (menor).
    SFI_PURGE_H           = int(getattr(_cfg_sfi.sop,      'purge_hours',    96))  # LdP: hacia atrás del test
    SFI_EMBARGO_H         = int(getattr(_cfg_sfi.sop,      'embargo_hours',  24))  # LdP: cooldown post-test
    SFI_N_ESTIMATORS      = int(getattr(_cfg_sfi.features, 'sfi_n_estimators', 200))
    SFI_MAX_DEPTH         = int(getattr(_cfg_sfi.features, 'sfi_max_depth',      4))
    SFI_COST_ROUNDTRIP    = 0.0025; SFI_COST_ROUNDTRIP = float(_cfg_sfi.sop.cost_pct)  # R6: >= 0.0025
    SFI_MIN_SHARPE        = float(getattr(_cfg_sfi.features,'sfi_min_sharpe',  0.05))  # floor DSR-null (MEJORA-SFI-SHARPE-01)
    SFI_DSR_N_TRIALS      = 600; SFI_DSR_N_TRIALS = int(getattr(_cfg_sfi.stat, 'n_trials_total', 600))  # BUG M-01
    FORWARD_ENABLED       = False  # Etapa E: controlado por código (no por cfg aún)
    FORWARD_MAX_FEATURES  = int(getattr(_cfg_sfi.features, 'forward_max_features', 25))
    FORWARD_MIN_IMPROVE   = 0.00   # Etapa E: mejora mínima 0% — por diseño, no heurístico
    # [SFI-BALANCE-01 2026-06-03] Slots mínimos garantizados por categoría.
    # 3 cuotas independientes: macro estructural, onchain valuation, calendar/halving.
    # Total garantizado: 3+1+1=5 slots de 13. Resto: competición libre por DSR.
    SFI_MACRO_MIN_SLOTS     = int(getattr(_cfg_sfi.features, 'sfi_macro_min_slots',     0))
    SFI_ONCHAIN_MIN_SLOTS   = int(getattr(_cfg_sfi.features, 'sfi_onchain_min_slots',   0))
    SFI_CALENDAR_MIN_SLOTS  = int(getattr(_cfg_sfi.features, 'sfi_calendar_min_slots',  0))
    print(
        f"[SFI-BALANCE-01] Cuotas cargadas: "
        f"macro={SFI_MACRO_MIN_SLOTS} onchain={SFI_ONCHAIN_MIN_SLOTS} "
        f"calendar={SFI_CALENDAR_MIN_SLOTS} (total garantizado="
        f"{SFI_MACRO_MIN_SLOTS+SFI_ONCHAIN_MIN_SLOTS+SFI_CALENDAR_MIN_SLOTS}/{SFI_TOP_N_FEATURES})"
    )
except Exception as _e_sfi:
    raise RuntimeError(f"CRITICAL: settings.yaml no disponible o faltan parametros criticos (SOP No-Fallback). Error: {_e_sfi}")



# =============================================================================
# ETAPA PRE-A: FILTRO DE ESTACIONARIEDAD (ADF) - PILAR 3 DARWINIANO
# =============================================================================

class ADFStationarityFilter:
    """
    Etapa PRE-A de LdP: Aplica la prueba de Dickey-Fuller Aumentado a todas las
    features continuas para detectar Caminatas Aleatorias (Random Walks).
    Aplica el 'Soft Penalty' a las mÃ©tricas tendenciales que generen regresiones espurias.
    """
    def __init__(self, p_value_threshold: float = 0.05, penalty_multiplier: float = 0.10):
        self.p_value_threshold = p_value_threshold
        self.penalty_multiplier = penalty_multiplier

    def evaluate(self, df: pd.DataFrame, alpha_cols: List[str]) -> Dict[str, float]:
        import statsmodels.api as sm
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        penalties = {}
        cands = [c for c in df.columns if c not in alpha_cols and pd.api.types.is_numeric_dtype(df[c])]
        
        logger.info(f"[PRE-A] Evaluando Estacionariedad (ADF Test) en {len(cands)} features crudas...")
        
        def _check_adf(col: str) -> Tuple[str, float, float]:
            s = df[col].dropna()
            if len(s) < 100 or s.nunique() < 10:
                return col, 0.0, 1.0 # Eximir a categÃ³ricas o features con muy pocos datos
            try:
                # maxlag=1, autolag=None para O(N) de extrema velocidad
                adf_res = sm.tsa.stattools.adfuller(s.values, maxlag=1, autolag=None)
                p_val = float(adf_res[1])
                return col, p_val, 1.0 if p_val <= self.p_value_threshold else self.penalty_multiplier
            except Exception:
                return col, 1.0, self.penalty_multiplier

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_check_adf, c): c for c in cands}
            for future in as_completed(futures):
                col_name, p_val, pen = future.result()
                penalties[col_name] = pen

        for ac in alpha_cols:
            penalties[ac] = 1.0
            
        n_punished = sum(1 for v in penalties.values() if v < 1.0)
        logger.info(f"[PRE-A] Filtro ADF completado: {n_punished} features No-Estacionarias penalizadas con {self.penalty_multiplier}x")
        return penalties

# =============================================================================
# ETAPA B: Clustering JerÃ¡rquico Anti-Redundancia
# =============================================================================

class FeatureClusterer:
    """
    Etapa B de LdP: reduce multicolinealidad agrupando features correlacionadas
    y seleccionando el mejor representante de cada cluster.

    Problema: RSI_14 y RSI_15 al 99% de correlaciÃÂ³n confunden el modelo.
    SoluciÃÂ³n: un representante por cluster (mayor corr con target).
    """

    def __init__(self, n_clusters: int = CLUSTER_FIXED_N, n_reps: int = 2):
        self.n_clusters = n_clusters
        self.n_reps     = n_reps       # MEJORA-SFI-B04: top-N representantes por cluster
        self.cluster_assignments: Dict[str, int] = {}
        self.selected: List[str] = []

    @staticmethod
    def _icir_score(
        col_vals: np.ndarray,
        fwd_ret: np.ndarray,
        nan_pct: float,
        window: int = 720,
        min_windows: int = 3,
    ) -> float:
        """
        MEJORA-SFI-B01: Information Coefficient IR x (1 - NaN%).

        IC = Spearman(feature_t, fwd_return_{t+H}) en ventanas rodantes.
        ICIR = mean(IC) / std(IC)  -- analogo al Sharpe de la senal cruda.
        Score final = abs(ICIR) x (1 - nan_pct):
          - abs(ICIR): captura senal tanto positiva como negativa
          - (1-nan_pct): penaliza features con datos incompletos (IBIT=93%NaN -> score~0)

        Parametros:
            window: filas por ventana IC (~720H = 1 mes de datos horarios)
            min_windows: minimo de ventanas validas para calcular ICIR
        """
        from scipy.stats import spearmanr
        ics = []
        step = window // 2  # solapamiento 50%
        n = len(col_vals)
        for start in range(0, n - window, step):
            end = start + window
            x = col_vals[start:end]
            y = fwd_ret[start:end]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 30:
                continue
            ic, _ = spearmanr(x[mask], y[mask])
            if not np.isnan(ic):
                ics.append(ic)
        if len(ics) < min_windows:
            return 0.0
        mean_ic = float(np.mean(ics))
        std_ic  = float(np.std(ics)) + 1e-10
        icir    = mean_ic / std_ic
        return abs(icir) * (1.0 - nan_pct)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prices: Optional[pd.Series] = None,
    ) -> List[str]:
        """
        Agrupa features y selecciona representantes por ICIR x (1-NaN%).

        Criterio MEJORA-SFI-B01: el representante de cada cluster es la feature
        con mayor abs(ICIR) x (1-NaN%), donde ICIR = mean(IC)/std(IC) calculado
        sobre ventanas rodantes de Spearman(feature, fwd_return).
        Esto prioriza features predictivas Y con datos completos.

        Args:
            X:      DataFrame con features raw (sin alpha signals)
            y:      Target binario TBM
            prices: Serie de precios close para calcular retornos forward.
                    Si no se provee, se usa y directamente como proxy.
        Returns:
            Lista de features representantes (una por cluster)
        """
        logger.info(f"[B] Clustering jerÃÂ¡rquico sobre {len(X.columns)} features "
                    f"(n_clusters={self.n_clusters})...")

        # Eliminar columnas constantes
        X = X.loc[:, X.std() > 1e-10]

        # BUG-SFI-NaN-01: Eliminar features con NaN excesivo antes del clustering.
        # Umbral configurable desde settings.yaml (sfi_max_nan_pct, default=0.85).
        # 0.85 filtra: IBIT/ARKB (93% NaN, ESET-bloqueadas) y ShortAccount (83% NaN).
        # 0.85 preserva: Transactions/Unique_Addresses (80.5% NaN, historico desde 2023).
        try:
            _max_nan = float(getattr(_cfg_sfi.features, 'sfi_max_nan_pct', 0.85))
        except Exception:
            _max_nan = 0.85
        nan_mask = X.isna().mean() < _max_nan
        n_dropped_nan = (~nan_mask).sum()
        if n_dropped_nan > 0:
            dropped = X.columns[~nan_mask].tolist()
            logger.info(f"[B] Filtro NaN>{_max_nan:.0%}: {n_dropped_nan} features eliminadas: {dropped}")
            X = X.loc[:, nan_mask]

        # [PURGA-MACRO-SFI] 2026-06-06: Arquitectura 2-capas: Capa A (Macro), Capa B (Micro/Timing).
        # Para que el SFI no se contamine con lag macroeconómico, las purgamos del pool de XGBoost.
        macro_keywords = ['M2_', 'FedFunds', 'CPI', 'YieldCurve', 'T10Y2Y', 'DXY', 'SP500', 'Gold']
        macro_cols_to_drop = [c for c in X.columns if any(mk.lower() in c.lower() for mk in macro_keywords)]
        if macro_cols_to_drop:
            logger.info(f"[PURGA-MACRO-SFI] Excluyendo {len(macro_cols_to_drop)} features macro del pool SFI para enfocar XGBoost en Micro/Timing.")
            print(f"[FIX-ARQUITECTURA-2CAPAS] Purgando {len(macro_cols_to_drop)} variables MACRO del SFI. XGBoost se enfocará en timing.")
            X = X.drop(columns=macro_cols_to_drop)

        # MEJORA-SFI-C-MI: Matriz de InformaciÃ³n Mutua Condicionada (Pilar 4)
        # Reemplazamos Pearson/Spearman por MI para agrupar variables no-linealmente equivalentes.
        # Para evitar el colapso O(N^2) en tiempo, usamos un Fast Binning en 20 percentiles.
        try:
            import time
            from sklearn.metrics import mutual_info_score
            t0 = time.time()
            n_features = len(X.columns)
            logger.info(f"[B] Calculando matriz C-MI Pairwise ({n_features}x{n_features} = {n_features*n_features//2} pares) vÃ­a Fast Binning...")
            
            # 1. Discretizar (Binning) todas las features para acelerar EntropÃ­a Discreta
            X_binned = pd.DataFrame(index=X.index)
            for col in X.columns:
                # Saneamiento de tipo blindado: Sin infinitos y sin NaNs.
                col_no_na = X[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                # AÃ±adir ruido infinitesimal para evitar bins idÃ©nticos
                noise = np.random.normal(0, 1e-10, size=len(col_no_na))
                try:
                    # pd.qcut puede generar pandas float array con NaNs si hay anomalÃÂ­as. Forzar int puro.
                    binned = pd.qcut(col_no_na + noise, q=20, labels=False, duplicates='drop')
                    X_binned[col] = binned.fillna(0).astype(int)
                except Exception:
                    X_binned[col] = 0 # Feature degenerada
            
            # 2. Pre-calcular EntropÃ­as Marginales H(X)
            # MEJ-SFI-01 FIX (2026-04-06): reemplazar mutual_info_score(X,X) por scipy.entropy.
            # mutual_info_score(X,X) == H(X) por definiciÃ³n de MI, pero es O(NÂ²) internamente.
            # scipy.entropy con bincount es O(N) â mismo resultado, 10-50x mÃ¡s rÃ¡pido.
            from scipy.stats import entropy as _scipy_entropy
            features = X.columns
            entropies = np.zeros(n_features)
            for i, f in enumerate(features):
                counts = np.bincount(X_binned[f].values.astype(int).clip(0))
                probs = counts[counts > 0] / counts.sum()
                entropies[i] = float(_scipy_entropy(probs))

            # 3. Calcular la matriz de redundancia
            # MEJ-SFI-03 FIX (2026-04-06): Paralelizar el bucle externo con joblib + Fast numpy.bincount
            dist_arr = np.zeros((n_features, n_features))
            
            # Matriz NumPy pre-calculada para velocidad
            X_arr = X_binned.values

            def _nmi_row(i):
                """Calcula una fila de la matriz NMI para el Ã­ndice i."""
                row = np.zeros(n_features)
                h_i = entropies[i]
                arr_i = X_arr[:, i] * 20
                for j in range(i + 1, n_features):
                    h_j = entropies[j]
                    if h_i < 1e-6 or h_j < 1e-6:
                        row[j] = 1.0
                        continue
                    
                    combined = arr_i + X_arr[:, j]
                    counts = np.bincount(combined)
                    probs = counts[counts > 0] / len(combined)
                    h_ij = -np.sum(probs * np.log(probs))
                    
                    mi = max(0.0, h_i + h_j - h_ij)
                    norm_mi = mi / max(h_i, h_j) if max(h_i, h_j) > 0 else 0.0
                    row[j] = float(np.clip(1.0 - norm_mi, 0.0, 1.0))
                return i, row

            try:
                from joblib import Parallel as _Parallel, delayed as _delayed
                _results = _Parallel(n_jobs=-1, backend='loky')(
                    _delayed(_nmi_row)(i) for i in range(n_features)
                )
            except ImportError:
                _results = [_nmi_row(i) for i in range(n_features)]

            for _i, _row in _results:
                dist_arr[_i, _i + 1:] = _row[_i + 1:]
                dist_arr[_i + 1:, _i] = _row[_i + 1:]

            np.fill_diagonal(dist_arr, 0.0)
            dist = pd.DataFrame(dist_arr, index=features, columns=features)
            t1 = time.time()
            logger.info(f"[B] Matriz C-MI ensamblada en {t1-t0:.2f}s.")
        except ImportError:
            logger.warning("[B] sklearn no disponible. Fallback a correlaciÃ³n Pearson.")
            corr = X.corr().abs().fillna(0)
            dist_arr = (1 - corr.to_numpy()).clip(0)
            np.fill_diagonal(dist_arr, 0)
            dist_arr = (dist_arr + dist_arr.T) / 2
            dist = pd.DataFrame(dist_arr, index=corr.index, columns=corr.columns)

        try:
            condensed = squareform(dist.values, checks=False)
            Z = linkage(condensed, method="ward")

            # MEJORA-SFI-B02: k automatico por Dendrogram Gap.
            # El mayor salto en las distancias de fusion indica donde el dendrograma
            # tiene la frontera mas natural entre grupos. Coste: 0 (usa Z ya computado).
            # Floor/cap desde settings.yaml para que Etapa D no explote en tiempo.
            try:
                _k_min = int(getattr(_cfg_sfi.features, 'sfi_n_clusters_min', 8))
                _k_max = int(getattr(_cfg_sfi.features, 'sfi_n_clusters_max', 30))
                # FIX: Permitir que k_max sea al menos el 85% de las features totales, evitando compresiÃ³n destructiva
                _k_max = max(_k_max, int(len(X.columns) * 0.85))
            except Exception:
                _k_min, _k_max = 8, max(30, int(len(X.columns) * 0.85))

            merge_dists = Z[:, 2]                          # distancias de fusion sucesivas
            gaps        = np.diff(merge_dists)              # salto entre fusiones consecutivas
            # El mayor gap ocurre en la posicion i -> cortar ahÃÂ­ da k = N - i - 1 clusters
            # (N = num features, cada fusiÃÂ³n reduce en 1 el nÃÂºmero de clusters)
            n_feats = len(X.columns)
            if len(gaps) > 0:
                # Ignorar los ultimos saltos (fusions globales que siempre son grandes)
                # y los primeros (fusiones de features casi identicas)
                trim_lo = max(0, int(len(gaps) * 0.20))   # ignorar 20% menor
                trim_hi = max(trim_lo + 1, int(len(gaps) * 0.90))  # ignorar 10% mayor
                gaps_trimmed = gaps[trim_lo:trim_hi]
                best_gap_idx = trim_lo + int(np.argmax(gaps_trimmed))
                k_auto = n_feats - best_gap_idx - 1
            else:
                k_auto = self.n_clusters  # fallback si solo 1 feature

            n_clusters_target = int(np.clip(k_auto, _k_min, min(_k_max, n_feats - 1)))
            logger.info(
                f"[B] Dendrogram Gap: k_auto={k_auto} Ã¢â â k_final={n_clusters_target} "
                f"(floor={_k_min}, cap={_k_max}) | features={n_feats}"
            )
            clusters = fcluster(Z, t=n_clusters_target, criterion="maxclust")
        except Exception as e:
            logger.warning(f"[B] Clustering fallÃÂ³: {e}. Retornando todas las features.")
            self.selected = list(X.columns)
            return self.selected

        self.cluster_assignments = dict(zip(X.columns, clusters))
        n_clusters = max(clusters)

        # MEJORA-SFI-B03: cargar cache DSR del run anterior para score adaptativo.
        # Primera ejecucion: cache vacio -> usa ICIR solo.
        # Runs posteriores: max(ICIR_norm, DSR_previo_norm) dentro de cada cluster.
        try:
            logger.info("[FIX-SHADOW-JSON-01] Cargo cache DSR sin import local redundante.")
            with open(DSR_CACHE) as f:
                dsr_prev: dict = json.load(f)
            logger.info(f"[B] Cache DSR cargado: {len(dsr_prev)} features del run anterior")
        except Exception:
            dsr_prev = {}  # primera ejecucion o cache no disponible

        # Calcular retorno forward para ICIR (alineado con horizonte dinÃ¡mico promedio del TBM: 120H)
        if prices is not None and len(prices) == len(X):
            # FIX-LAB-01B (2026-05-15): shift(-120) vectorizado sin pct_change previo para evitar fugas.
            # Mide el retorno desde T hasta T+120H correctamente, reflejando causalidad hacia el TBM macro.
            # Al usar prices.shift(-120) / prices, los retornos futuros desaparecen al final del dataset automÃ¡ticamente como NaN.
            _horizon = 120
            fwd_ret = (prices.shift(-_horizon) / prices) - 1.0
            # Se reindexa para mantener el mismo largo que X y se preservan los NaNs 
            # ya que _icir_score internamente hace mask = ~(np.isnan(x) | np.isnan(y))
            fwd_ret = fwd_ret.reindex(X.index).values  
        else:
            # Fallback: usar y como proxy de retorno forward (binario 0/1)
            fwd_ret = y.reindex(X.index).fillna(0.5).values.astype(float)

        selected = []
        for cid in range(1, n_clusters + 1):
            members = [f for f, c in self.cluster_assignments.items() if c == cid]
            if not members:
                continue
            # MEJORA-SFI-B03: score = max(ICIR_norm, DSR_previo_norm) dentro del cluster.
            # Normalizar ambas metricas al rango [0,1] dentro del cluster para comparabilidad.
            # Si DSR_previo no existe (primera ejecucion), usa ICIR solo.
            icir_scores   = {}
            dsr_prev_vals = {}
            for col in members:
                nan_pct   = float(X[col].isna().mean())
                # [FIX-H-FP-02 2026-05-30] Cambio fillna(0) -> ffill().fillna(0) para ICIR score.
                # Antes: NaN en periodos sin dato (ej: Stablecoin pre-2019, ETF pre-2024) se rellenaba
                # con 0, creando correlaciones espurias en ICIR (feature=0 != feature=ausente).
                # Ahora: ffill propaga el ultimo valor conocido (mas informativo). Solo el NaN inicial
                # (sin historia previa) se rellena con 0 como valor de base.
                col_vals  = X[col].ffill().fillna(0).values.astype(float)
                icir_scores[col]   = self._icir_score(col_vals, fwd_ret, nan_pct)
                dsr_prev_vals[col] = float(dsr_prev.get(col, 0.0))

            # Normalizar ICIR al [0,1] dentro del cluster
            icir_max = max(icir_scores.values()) or 1e-10
            icir_norm = {c: v / icir_max for c, v in icir_scores.items()}

            # Normalizar DSR_previo al [0,1] dentro del cluster (solo valores positivos)
            dsr_pos = {c: max(v, 0.0) for c, v in dsr_prev_vals.items()}
            dsr_max = max(dsr_pos.values()) or 1e-10
            dsr_norm = {c: v / dsr_max for c, v in dsr_pos.items()}

            # MEJORA-SFI-B04: seleccionar top-N representantes por cluster.
            # n_reps=1 Ã¢â â comportamiento original (1 rep/cluster)
            # n_reps=2 Ã¢â â 50 candidatos raw con k=25 Ã¢â â captura Unique_Addresses, WEI, etc.
            combined = {c: max(icir_norm[c], dsr_norm[c]) for c in members}
            # [ARCH-21/23-FIX-A 2026-06-02] Macro-stable boost en SFI cluster ranking.
            # Features macro de largo plazo (M2, Fed_Liq, CPI, yield_curve, MOVE, Puell)
            # pueden tener ICIR ligeramente menor en ventanas especificas pero son
            # MAS estables cross-window que features tecnicas. El boost garantiza
            # que el SFI las favorezca en el ranking frente a features inestables.
            # El boost es ADITIVO (no multiplicativo) para preservar el ordinal relativo.
            try:
                from config.settings import cfg as _cfg_macro
                _macro_boost = float(getattr(_cfg_macro.features, 'sfi_macro_stable_boost', 0.15))
                _macro_feats = set(getattr(_cfg_macro.features, 'sfi_macro_stable_features', []) or [])
            except Exception:
                _macro_boost = 0.15
                _macro_feats = set()
            if _macro_feats:
                _boosted = []
                for feat_c, score_c in combined.items():
                    # Aplicar boost solo si la feature esta en la whitelist macro
                    _base_name = feat_c.split('_milag')[0] if '_milag' in feat_c else feat_c
                    _in_macro  = feat_c in _macro_feats or _base_name in _macro_feats
                    combined[feat_c] = min(1.0, score_c + (_macro_boost if _in_macro else 0.0))
                    if _in_macro and combined[feat_c] > score_c:
                        _boosted.append(feat_c)
                if _boosted:
                    print(  # RULE[fixbugsprints.md]
                        f'[ARCH-21/23-FIX-A] SFI macro-stable boost +{_macro_boost:.2f} aplicado a '
                        f'{len(_boosted)} features: {_boosted}'
                    )
            sorted_members = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
            top_n_reps = [feat for feat, _ in sorted_members[:self.n_reps]]
            for rank, rep in enumerate(top_n_reps):
                via = "DSR_prev" if dsr_norm[rep] >= icir_norm[rep] else "ICIR"
                logger.debug(
                    f"  [B] Cluster {cid} #{rank+1}: {rep} "
                    f"(combined={combined[rep]:.4f}, ICIR={icir_norm[rep]:.3f}, "
                    f"DSR_prev={dsr_norm[rep]:.3f}) [{via}]"
                )
            selected.extend(top_n_reps)

        self.selected = list(dict.fromkeys(selected))  # deduplicar preservando orden
        logger.info(
            f"[B] {len(X.columns)} Ã¢â â {len(self.selected)} features "
            f"({n_clusters} clusters Ãâ top-{self.n_reps} reps)"
        )
        return self.selected

    def get_report(self) -> pd.DataFrame:
        rows = []
        for feat, cid in self.cluster_assignments.items():
            rows.append({
                "feature": feat,
                "cluster": cid,
                "selected": feat in self.selected,
            })
        return pd.DataFrame(rows).sort_values("cluster")


# =============================================================================
# ETAPA C: Automatic Lag Discovery por Mutual Information
# =============================================================================

class AutoLagDiscovery:
    """
    Etapa C de LdP: encuentra el lag ÃÂ³ptimo de CADA feature por Mutual Information.

    Complementa a Granger (que detecta SI hay causalidad) detectando CUÃÂNDO
    es mÃÂ¡xima esa causalidad.

    Si MI-lag Ã¢â°Â  Granger-lag conocido Ã¢â â genera DOS versiones de la feature.
    Las alpha signals saltan esta etapa (lags ya baked en sus reglas).
    """

    # Lags conocidos por Granger/TE en Luna V1 (feature_pipeline.py)
    KNOWN_GRANGER_LAGS: Dict[str, int] = {
        "ETH":            13 * 24,   # 312H
        "M2":             42 * 24,   # 1008H
        "CPI":            14 * 24,   # 336H
        "FedFunds":       14 * 24,
        "Unemploy":       14 * 24,
        "onchain":        24,
        "defi":           24,
    }

    # Candidatos logarÃÂ­tmicos para el scan de lag
    LAG_CANDIDATES = [1, 2, 4, 6, 12, 24, 48, 72, 96, 120, 168, 240, 336, 500]

    def __init__(self, max_lag: int = MAX_LAG_HOURS, n_samples: int = 8000,
                 random_state: int = 42):
        self.max_lag = max_lag
        self.n_samples = n_samples
        self.random_state = random_state  # MI-FIX: seed determinista para reproducibilidad
        self.optimal_lags: Dict[str, int] = {}
        self.mi_scores: Dict[str, float] = {}

    def _known_lag(self, feature_name: str) -> Optional[int]:
        """Retorna el lag Granger conocido si el nombre de feature lo contiene."""
        for key, lag in self.KNOWN_GRANGER_LAGS.items():
            if key.lower() in feature_name.lower():
                return lag
        return None

    def find_lag(self, series: pd.Series, target: pd.Series,
                 name: str, hmm_regime: Optional[pd.Series] = None) -> Dict:
        """Busca el lag ÃÂ³ptimo por MI para una feature.

        MI-FIX (2026-03-16): usa RNG seeded derivado del nombre de feature para
        garantizar reproducibilidad entre runs con el mismo dataset.
        Antes de este fix: np.random.choice sin seed Ã¢â â lags distintos cada run
        (DeFi_WBTC_TVL variaba entre 1H y 500H entre M-26 y M-27).
        """
        candidates = [l for l in self.LAG_CANDIDATES if l <= self.max_lag]
        mi_results: Dict[int, float] = {}

        # GUARD: si target estÃ¡ vacÃ­o (ej: llamada desde D.2 holdout-adversarial
        # con y=pd.Series(dtype=float)), no hay nada que calcular â early return.
        if len(target) == 0:
            return {"optimal_lag": 1, "mi_score": 0.0, "all_mi": {}}

        # [FIX-SFI-RNG-01] RNG determinista por feature usando hash criptográfico SHA-256
        # para evitar el salteo process-randomized de la función hash() nativa de Python.
        name_hash = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16)
        feature_seed = (self.random_state + name_hash) % (2**31)
        # Log determinism for absolute traceability (RULE[fixbugsprints.md])
        logger.debug(f"[FIX-SFI-RNG-01] Deterministic lag discovery seed for '{name}': {feature_seed}")
        rng = np.random.default_rng(feature_seed)

        for lag in candidates:
            shifted = series.shift(lag)
            # MI-FIX: Convertir a numpy boolean puro para evitar DataFrame Masking degenerado
            mask = ~(shifted.isna() | target.isna()).values.astype(bool)
            Xv = shifted.values[mask].reshape(-1, 1)
            yv = target.values[mask]
            
            if len(Xv) > self.n_samples:
                try:
                    # Si hmm_regime esta disponible, estratificar por el regimen (Fase C: Muestreo Estratificado)
                    _stratify = hmm_regime.values[mask] if hmm_regime is not None else yv
                    
                    idx, _ = train_test_split(
                        np.arange(len(Xv)),
                        train_size=self.n_samples,
                        stratify=_stratify,
                        random_state=feature_seed
                    )
                except ValueError:
                    idx = rng.choice(len(Xv), self.n_samples, replace=False)
                Xv, yv = Xv[idx], yv[idx]
            if len(Xv) < 50:
                continue
            try:
                mi = mutual_info_classif(Xv, yv, random_state=self.random_state)[0]
                mi_results[lag] = mi
            except Exception:
                continue

        if not mi_results:
            return {"optimal_lag": 1, "mi_score": 0.0, "all_mi": {}}

        opt_lag = max(mi_results, key=mi_results.get)
        self.optimal_lags[name] = opt_lag
        self.mi_scores[name] = mi_results[opt_lag]
        return {"optimal_lag": opt_lag, "mi_score": mi_results[opt_lag],
                "all_mi": mi_results}


    def transform(self, X: pd.DataFrame, y: pd.Series,
                  lag_cache: Optional[Dict[str, int]] = None,
                  hmm_regime: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Aplica lag discovery a cada feature.
        Si MI-lag difiere del Granger conocido Ã¢â â aÃÂ±ade la versiÃÂ³n con MI-lag
        como feature adicional (con sufijo _milag).

        Args:
            lag_cache: dict opcional {feature_name: optimal_lag_H} de un run
                anterior. Si existe para una feature, evita recalcular el MI.

        Returns:
            DataFrame con features lagged (y posibles versiones adicionales MI)
        """
        logger.info(f"[C] Lag Discovery MI sobre {len(X.columns)} features "
                    f"(max_lag={self.max_lag}H, seed={self.random_state})...")
        result = X.copy()
        extra_cols: Dict[str, pd.Series] = {}
        cache_hits = 0

        for col in X.columns:
            # MI-FIX: usar lag cacheado si existe (mismo dataset = mismo lag)
            if lag_cache and col in lag_cache:
                opt_lag = lag_cache[col]
                self.optimal_lags[col] = opt_lag
                self.mi_scores[col] = 0.0  # no recalculado
                cache_hits += 1
            else:
                info = self.find_lag(X[col], y, col, hmm_regime=hmm_regime)
                opt_lag = info["optimal_lag"]

            # LOGIC-SFI-03 (2026-04-06): aplicar lag IS (opt_lag) al dataset completo.
            # El lag Ã³ptimo se determina en training (causal) y se aplica tambiÃ©n al
            # OOS/holdout con el mismo valor. Esto es CORRECTO: el lag es un parÃ¡metro
            # del feature engineering, no una etiqueta futura. No es look-ahead.
            # Si el lag Ã³ptimo cambia entre regimenes (nonstationarity de lag), el WFB
            # re-calibra el SFI en cada ventana, preservando la adaptabilidad temporal.
            result[col] = X[col].shift(opt_lag)

            # Si hay un lag Granger conocido diferente -> aÃ±adir columna adicional
            granger_lag = self._known_lag(col)
            if granger_lag and granger_lag != opt_lag and "_milag" not in col:
                new_name = f"{col}_milag{opt_lag}h"
                extra_cols[new_name] = X[col].shift(opt_lag)
                logger.debug(f"  {col}: Granger={granger_lag}H | MI={opt_lag}H -> "
                             f"ambas versiones")

        for name, s in extra_cols.items():
            result[name] = s

        if cache_hits > 0:
            logger.info(f"[C] Lag cache: {cache_hits}/{len(X.columns)} features "
                        f"reutilizaron lag anterior (seed={self.random_state})")
        logger.info(f"[C] Lag discovery completado. "
                    f"Features: {len(X.columns)} Ã¢â â {len(result.columns)} "
                    f"(+{len(extra_cols)} versiones MI-lag adicionales)")
        return result

    def get_report(self) -> pd.DataFrame:
        rows = [{"feature": n, "optimal_lag_H": self.optimal_lags.get(n, 0),
                 "mi_score": self.mi_scores.get(n, 0)}
                for n in self.optimal_lags]
        return pd.DataFrame(rows).sort_values("mi_score", ascending=False)


# =============================================================================
# UTILS: CPCV
# =============================================================================

class PurgedCPCV:
    """
    Combinatorial Purged Cross-Validation con Purge y Embargo.
    (LÃƒÂ³pez de Prado, AFML Ch. 7)
    """

    def __init__(self, n_groups: int = SFI_N_GROUPS,
                 purge: int = SFI_PURGE_H,
                 embargo: int = SFI_EMBARGO_H,
                 n_test_groups: int = 2):
        self.n_groups = n_groups
        self.purge = purge
        self.embargo = embargo
        self.n_test_groups = n_test_groups

    def split(self, n: int):
        from itertools import combinations
        size = n // self.n_groups
        groups = [(i * size, (i + 1) * size if i < self.n_groups - 1 else n)
                  for i in range(self.n_groups)]

        # [FIX-H-SFI-13 2026-05-30] SPEEDUP print eliminado (era un print() por cada fold de cada feature
        # generando 1305+ lineas de ruido en el log durante el SFI). Verificable en el logger.info del _thread_worker.

        for test_gidx in combinations(range(self.n_groups), self.n_test_groups):
            train_s = set()
            test_idx = []
            for gi, (s, e) in enumerate(groups):
                if gi in test_gidx:
                    test_idx.extend(range(s, e))
                else:
                    train_s.update(range(s, e))
            
            # Optimized Purge & Embargo using contiguous group boundary coordinates (100% mathematically equivalent)
            for gi in test_gidx:
                s, e = groups[gi]
                # Purge before the start of the test interval s
                for i in range(max(0, s - self.purge), s):
                    train_s.discard(i)
                # Embargo after the end of the test interval e
                for i in range(e, min(n, e + self.embargo)):
                    train_s.discard(i)
                    
            train_idx = sorted(train_s)
            if len(train_idx) > 100 and len(test_idx) > 100:
                yield np.array(train_idx), np.array(test_idx)

    @staticmethod
    def deflated_sharpe(sharpes: List[float], n_trials: int, n_obs: Optional[int] = None, freq: Optional[float] = None) -> float:
        """
        Deflated Sharpe Ratio CORRECTO (Bailey & López de Prado, 2014).

        DSR = Φ[(SR - SR*) / sqrt(Var(SR))]

        donde:
          SR  = media del Sharpe entre folds
          SR* = Sharpe esperado del mejor trial por azar puro (benchmark)
                = sr_std_cross * [(1-γ) * z1 + γ * z2]
          Var(SR) = varianza estadística del estimado de Sharpe (1.0 + 0.5 * SR^2) / n_obs

        FIX-MATH-05: El cálculo previo cometía un error gravísimo al usar la varianza
        temporal (folds) como proxy de la varianza transversal (trials). Esto causaba
        que features mediocres pero estables tuvieran un SR* de casi 0, otorgándoles
        un DSR = 1.0 (eliminando por completo la penalización del Multiple Testing Bias).

        FIX-MATH-06-V2: Corregir la varianza del Sharpe Ratio anualizado dividiendo por
        n_obs si se proporciona, de lo contrario usar fallback de 6 grados de libertad.
        """
        import math
        from scipy.stats import norm

        if len(sharpes) < 2:
            return 0.0

        sr_mean = float(np.mean(sharpes))
        sr_std  = float(np.std(sharpes, ddof=1)) or 1e-10

        # [FIX-DSR-CROSS-VAR-01 2026-06-04] Bailey & Lopez de Prado (2014) exige la varianza TRANSVERSAL.
        print(f'[FIX-DSR-CROSS-VAR-01] SFI: DSR calculado con Varianza Transversal constante (std_cross=1.0).')
        # Usar sr_std (varianza temporal de los folds) borraba la penalización de Multiple Testing
        # para features muy estables en el tiempo, inflando su DSR a 1.0 artificialmente.
        # Asignamos una varianza transversal teórica y conservadora (std=1.0) para que el hurdle rate
        # crezca pura e implacablemente en base al número de pruebas (n_trials).
        sr_std_cross = 1.0

        em_gamma = 0.5772156649
        n_trials_eff = max(2, n_trials)
        p = 1.0 / n_trials_eff
        z1 = norm.ppf(1 - p)
        z2 = norm.ppf(1 - p / math.e)
        exp_max = (1 - em_gamma) * z1 + em_gamma * z2
        sr_star = sr_std_cross * exp_max

        # Adjust variance using n_obs if provided
        # [FIX-DSR-FORMULA-01 2026-05-30] Bailey & Lopez de Prado (2014) exact formula:
        #   Var(SR_anualizado) = (1 + 0.5 * SR^2) / n_obs_bloques_independientes
        # ANTERIOR (INCORRECTO): var_sr = (freq_eff + 0.5*SR^2) / n_obs
        #   → inflaba var_sr ~55x-300x → DSR insensible a calidad de feature
        # CORRECTO: el factor freq ya esta implicito en sr_mean (anualizado con sqrt(8760))
        #   → no se suma al numerador de la varianza
        # El parametro freq se mantiene en la firma para compatibilidad futura pero no se usa aqui.
        if n_obs is not None and n_obs > 0:
            var_sr = (1.0 + 0.5 * (sr_mean ** 2)) / n_obs
            if var_sr <= 0:
                var_sr = 1e-10
            dsr = float(norm.cdf((sr_mean - sr_star) / math.sqrt(var_sr)))
        else:
            efectos_independientes = 6.0
            dsr = float(norm.cdf((sr_mean - sr_star) / (sr_std / math.sqrt(efectos_independientes))))
        
        # [FIX-H-SFI-13 2026-05-30] print() eliminado — generaba 2610+ lineas al stdout
        # mezclando con el log estructurado de loguru. La trazabilidad queda en logger.debug.
        logger.debug(f"[FIX-DSR-FORMULA-01] deflated_sharpe: sr_mean={sr_mean:.4f}, sr_star={sr_star:.4f}, n_obs={n_obs}, dsr={dsr:.4f}")
        return dsr


# =============================================================================
# ETAPA D: Single Feature Importance con CPCV
# =============================================================================

class SFI_CPCV:
    """
    Etapa D de LdP: evalúa cada feature EN SOLITARIO con CPCV para RANKING.
    V1.3b: Se eliminan umbrales mágicos. Se evalúa el Deflated Sharpe y MeanSR
    como métrica de ordenación para conservar las Top N features más útiles
    y rechazar features constantes o con errores.
    Incluye las alpha signals de Fase D (R22), que se conservan obligatoriamente.
    """

    def __init__(self, top_n: int = SFI_TOP_N_FEATURES,
                 n_groups: int = SFI_N_GROUPS):
        self.top_n = top_n
        self.n_groups = n_groups
        self.scores: Dict[str, Dict] = {}
        self.selected: List[str] = []
        self.min_sharpe_used: float = SFI_MIN_SHARPE

    def _eval_one(self, X1: np.ndarray, y: np.ndarray,
                  prices: np.ndarray, cpcv: PurgedCPCV,
                  feature_name: str = "", mask: Optional[np.ndarray] = None,
                  n_trials_sfi: int = 2,
                  n_estimators: Optional[int] = None,
                  n_obs_base: Optional[int] = None) -> Dict:
        """Evalúa una feature en solitario con CPCV.

        R6 SOP: aplica costos 0.25% round-trip a los retornos de cada fold.
        R5 SOP: usa DSR correcto Bailey & LdP (no la variante hi-expected_max).

        M4 (2026-03-04): Si mask != None, evaluar solo en las filas activas
        (señales de baja frecuencia como genetic_rule_N, storm_intensity).

        [FIX-H-SFI-09 2026-05-30]: n_obs_base es el tamaño del dataset ANTES de
        aplicar la mask. Se usa para n_obs_eff en DSR, evitando penalizar 5.7x más
        las features de baja frecuencia que las siempre activas.
        """
        # M4: aplicar máscara condicional para señales de baja frecuencia
        if mask is not None and mask.sum() > 200:
            X1     = X1[mask]
            y      = y[mask]
            prices = prices[mask]

        fold_sharpes = []
        for tr, te in cpcv.split(len(X1)):
            if len(tr) < 100 or len(te) < 100:
                continue
            try:
                n_est = n_estimators if n_estimators is not None else SFI_N_ESTIMATORS
                model = XGBClassifier(
                    n_estimators=n_est,
                    max_depth=SFI_MAX_DEPTH,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=1.0,
                    random_state=_LUNA_SEED,
                    verbosity=0,
                    n_jobs=1,  # Avoid core thrashing inside ThreadPoolExecutor
                    tree_method='hist',
                    device='cpu'  # CPU-only machine, avoid CUDA runtime loading overhead
                )
                model.fit(X1[tr], y[tr])
                proba = model.predict_proba(X1[te])[:, 1]

                # BUG-SFI-01 FIX (2026-04-06): Alinear modelo de retorno del SFI con
                # el sistema real de trading (TBM long-only).
                try:
                    _tbm_h = int(getattr(_cfg_sfi.xgboost, 'vertical_barrier_hours', 96))
                except Exception:
                    _tbm_h = 96  # default TBM horizon

                # Long-only: 1 = entramos, 0 = cash (no operamos)
                sigs_lo = np.where(proba > 0.5, 1.0, 0.0)  # long-only

                # Circuit Breaker: rechazar predicción constante (y penalizar supervivencia)
                if sigs_lo.sum() == 0 or sigs_lo.sum() == len(sigs_lo):
                    fold_sharpes.append(0.0)
                    logger.debug(f"[FIX-MATH-SURVIVAL-01] Circuit Breaker: prediccion constante detectada para fold en '{feature_name}'. Se añade Sharpe 0.0.")
                    continue

                # [FIX-CPCV-GAP-JUMP] Calculate 1H returns on contiguous prices first
                # to avoid massive artificial jumps across K-fold non-contiguous gaps in te.
                # [FIX-H-SFI-13]: Per-fold debug log eliminado (1305+ lineas). Se emite 1 vez por feature al final.
                fwd_ret_full = np.zeros(len(prices))
                fwd_ret_full[:-1] = np.diff(prices) / prices[:-1]
                fwd_ret_1h = fwd_ret_full[te[:-1]]
                sigs_eval = sigs_lo[:-1]
                
                n = min(len(sigs_eval), len(fwd_ret_1h))
                if n < 10:
                    fold_sharpes.append(0.0)
                    logger.debug(f"[FIX-MATH-SURVIVAL-01] Longitud insuficiente (n={n}) para fold en '{feature_name}'. Se añade Sharpe 0.0.")
                    continue
                    
                # Detectar entradas (transicion de 0 a 1) para cobrar el round-trip
                entradas = (np.diff(sigs_eval[:n], prepend=0) > 0).astype(float)
                
                strat_ret = sigs_eval[:n] * fwd_ret_1h[:n] - entradas * SFI_COST_ROUNDTRIP

                if np.std(strat_ret) < 1e-10:
                    fold_sharpes.append(0.0)
                    logger.debug(f"[FIX-MATH-SURVIVAL-01] Varianza nula en retornos de estrategia para fold en '{feature_name}'. Se añade Sharpe 0.0.")
                    continue
                
                # Al ser retornos reales de 1H, la anualizacion vuelve a ser la estandar (sqrt(8760))
                ann_factor = np.sqrt(365 * 24)
                sharpe = float(np.mean(strat_ret) / np.std(strat_ret) * ann_factor)
                
                fold_sharpes.append(np.clip(sharpe, -10, 10))
            except Exception as e:
                logger.debug(f"  Fold error: {e}")
                fold_sharpes.append(0.0)
                logger.debug(f"[FIX-MATH-SURVIVAL-01] Excepcion en fold de '{feature_name}': {e}. Se añade Sharpe 0.0.")
                continue

        if not fold_sharpes:
            return {"mean_sharpe": 0.0, "deflated_sharpe": 0.0,
                    "passed": False, "n_folds": 0, "n_constant_folds": 0}

        mean_sr = float(np.mean(fold_sharpes))
        # [FIX-PBOGRD-QUALITY-01 2026-05-30] Track constant fold ratio for PBO-GUARD gate
        _n_constant = sum(1 for s in fold_sharpes if s == 0.0)
        _constant_ratio = _n_constant / max(len(fold_sharpes), 1)
        if _constant_ratio > 0.3:
            print(f"[FIX-PBOGRD-QUALITY-01] Feature '{feature_name}': {_n_constant}/{len(fold_sharpes)} folds constantes (ratio={_constant_ratio:.2f}). MeanSR puede estar inflado por zeros.")

        try:
            _tbm_h = int(getattr(_cfg_sfi.xgboost, 'vertical_barrier_hours', 96))
        except Exception:
            _tbm_h = 96
        # [FIX-H-SFI-09 2026-05-30] Usar n_obs_base (pre-mask) si se pasa, para evitar
        # penalizar features de baja frecuencia (mask reduce len(X1) de 63034 a ~3274).
        # Sin este fix: n_obs_eff = 3274//72 = 45 en vez de 63034//72 = 875 → var 5.7x mayor.
        _n_obs_for_dsr = n_obs_base if n_obs_base is not None else len(X1)
        n_obs_eff = max(10, _n_obs_for_dsr // _tbm_h)
        print(f"[FIX-H-SFI-09] '{feature_name}': n_obs_base={n_obs_base}, n_obs_post_mask={len(X1)}, n_obs_eff={n_obs_eff}")
        
        # [FIX-DSR-UNIT-01] Scale frequency to effective block scale matching n_obs_eff
        freq_eff = 8760.0 / _tbm_h
        dsr = PurgedCPCV.deflated_sharpe(fold_sharpes, n_trials_sfi, n_obs=n_obs_eff, freq=freq_eff)
        # Traceability logs (RULE[fixbugsprints.md])
        logger.info(f"[FIX-DSR-UNIT-01] DSR recalculado con n_obs_eff={n_obs_eff} (original={len(X1)}, TBM_H={_tbm_h}), freq_eff={freq_eff:.2f}. DSR={dsr:.4f}")

        passed = True

        return {"mean_sharpe": mean_sr, "deflated_sharpe": dsr,
                "passed": passed, "n_folds": len(fold_sharpes),
                "fold_sharpes": fold_sharpes,
                "n_constant_folds": _n_constant, "constant_folds_ratio": round(_constant_ratio, 3)}


    def _eval_temporal_stability(
        self,
        X1_col: np.ndarray,
        y: np.ndarray,
        prices: np.ndarray,
        timestamps: pd.Series | pd.DatetimeIndex,
        cpcv_small: PurgedCPCV,
        feature_name: str = "",
    ) -> dict:
        """
        [LIFECYCLE-01 2026-05-29] Evaluacion temporal consciente del ciclo de vida.

        Distingue 4 patrones de disponibilidad de datos por ano:
          PRE-BORN  : anos antes de que la fuente existiera    no es fallo, excluir
          POST-DEATH: anos tras la desconexion de la fuente    penalizar con -1.0
          GAP       : huecos internos (fuente poco fiable)     penalizar con -0.5
          NORMAL    : anos con datos reales                    evaluar con _eval_one

        Bug corregido de WEIGHTED-STABILITY-01: _current_year ahora usa ts_max_yr
        en vez de max(yearly_dsrs), corrigiendo el sesgo hacia features muertas.
        """
        print(f"[LIFECYCLE-01] Evaluating temporal stability for: '{feature_name}'")
        logger.debug("[LIFECYCLE-01] Two-phase lifecycle-aware temporal stability.")

        try:
            _stab_min_dsr      = float(getattr(_cfg_sfi.features, 'stability_min_dsr', 0.05))
            _stab_min_positive = int(getattr(_cfg_sfi.features, 'stability_min_positive', 3))
            _half_life_yr      = float(getattr(_cfg_sfi.features, 'stability_half_life_years', 2.0))
            _recent_window_yr  = int(getattr(_cfg_sfi.features, 'stability_recent_window_years', 2))
            _trend_window_yr   = int(getattr(_cfg_sfi.features, 'stability_trend_window_years', 4))
            _trend_thresh      = float(getattr(_cfg_sfi.features, 'stability_trend_threshold', 0.10))
            _var_thresh        = float(getattr(_cfg_sfi.features, 'stability_variance_threshold', 1e-6))
            _min_real_years    = int(getattr(_cfg_sfi.features, 'stability_min_real_years', 2))
            _min_mature_years  = int(getattr(_cfg_sfi.features, 'stability_maturity_min_years', 3))
            _dead_thresh_years = int(getattr(_cfg_sfi.features, 'stability_dead_threshold_years', 2))
            _gap_penalty       = float(getattr(_cfg_sfi.features, 'stability_gap_penalty', -0.5))
        except Exception as _cfg_e:
            logger.warning(f"[LIFECYCLE-01] Config read error: {_cfg_e} — using defaults")
            _stab_min_dsr, _stab_min_positive = 0.05, 3
            _half_life_yr, _recent_window_yr, _trend_window_yr, _trend_thresh = 2.0, 2, 4, 0.10
            _var_thresh, _min_real_years, _min_mature_years = 1e-6, 2, 3
            _dead_thresh_years, _gap_penalty = 2, -0.5

        if isinstance(timestamps, pd.Series):
            ts_years = timestamps.dt.year.values
        else:
            ts_years = pd.DatetimeIndex(timestamps).year

        ts_min_yr = int(ts_years.min())
        ts_max_yr = int(ts_years.max())
        years = list(range(ts_min_yr, ts_max_yr + 1))

        # ── FASE 1: Pre-scan — detectar ventana de vida real ────────────────────────
        _first_real_year = None
        _last_real_year  = None
        _years_with_variance = set()

        for yr in years:
            mask_yr = np.array(ts_years == yr)
            if mask_yr.sum() < 500:
                continue
            x1_yr    = X1_col[mask_yr]
            _yr_std  = float(np.std(x1_yr))
            _yr_uniq = int(np.unique(x1_yr).size)
            if _yr_std >= _var_thresh and _yr_uniq > 2:
                _years_with_variance.add(yr)
                if _first_real_year is None:
                    _first_real_year = yr
                _last_real_year = yr

        _n_real_years = len(_years_with_variance)

        if _first_real_year is None:
            print(f"[LIFECYCLE-01] '{feature_name}': CONSTANT sin varianza real")
            return {
                "stability_score": 0.0, "positive_years": 0, "yearly_dsrs": {},
                "stable": False, "trend": "Insufficient",
                "lifecycle": "CONSTANT", "n_real_years": 0,
                "first_real_year": None, "last_real_year": None,
                "weighted_stability": 0.0, "recent_stability": 0.0,
            }

        _years_dead_count = ts_max_yr - _last_real_year
        _is_young  = _first_real_year > (ts_min_yr + 2) and _years_dead_count < _dead_thresh_years
        _is_dead   = _years_dead_count >= _dead_thresh_years
        _lifecycle = "YOUNG" if _is_young else ("DEAD" if _is_dead else "MATURE")

        print(f"[LIFECYCLE-01] '{feature_name}': {_lifecycle}"
              f" first_real={_first_real_year} last_real={_last_real_year}"
              f" n_real={_n_real_years} ts_max={ts_max_yr} dead_yrs={_years_dead_count}")

        # ── FASE 2: Loop con 4 casos ────────────────────────────────────────────────
        yearly_dsrs: dict[str, float] = {}

        for yr in years:
            mask_yr = np.array(ts_years == yr)
            if mask_yr.sum() < 500:
                continue

            # CASO 1: PRE-BORN — feature no existia, excluir (no es fallo)
            if yr < _first_real_year:
                logger.debug(f"[LIFECYCLE-01] '{feature_name}' yr={yr}: PRE-BORN excluido")
                continue

            # CASO 2: POST-DEATH — fuente desconectada, penalizar con maximo peso reciente
            if yr > _last_real_year:
                yearly_dsrs[str(yr)] = -1.0
                logger.debug(f"[LIFECYCLE-01] '{feature_name}' yr={yr}: POST-DEATH DSR=-1.0")
                continue

            # CASO 3: GAP INTERNO — hueco en fuente fiable, penalizar parcialmente
            if yr not in _years_with_variance:
                yearly_dsrs[str(yr)] = _gap_penalty
                logger.debug(f"[LIFECYCLE-01] '{feature_name}' yr={yr}: GAP DSR={_gap_penalty}")
                continue

            # CASO 4: ANO NORMAL con datos reales
            try:
                _yr_X1 = X1_col[mask_yr]
                _yr_y  = y[mask_yr]
                _yr_p  = prices[mask_yr]
                res_yr = self._eval_one(
                    _yr_X1, _yr_y, _yr_p, cpcv_small,
                    feature_name=f"{feature_name}_yr{yr}", n_trials_sfi=2, n_estimators=20,
                    n_obs_base=len(_yr_X1),  # [FIX-H-SFI-09-NULL 2026-05-30] n_obs_base=yearly slice size
                )

                yearly_dsrs[str(yr)] = round(res_yr.get("mean_sharpe", 0.0), 4)
            except Exception as _ye:
                logger.debug(f"[LIFECYCLE-01] '{feature_name}' yr={yr}: eval failed: {_ye}")
                yearly_dsrs[str(yr)] = -1.0

        if _n_real_years < _min_real_years:
            print(f"[LIFECYCLE-01] '{feature_name}': INSUFFICIENT DATA"
                  f" n_real={_n_real_years} min={_min_real_years} score=0.0")
            return {
                "stability_score": 0.0, "positive_years": 0, "yearly_dsrs": yearly_dsrs,
                "stable": False, "trend": "Insufficient",
                "lifecycle": _lifecycle, "n_real_years": _n_real_years,
                "first_real_year": _first_real_year, "last_real_year": _last_real_year,
                "weighted_stability": 0.0, "recent_stability": 0.0,
            }

        if not yearly_dsrs:
            return {
                "stability_score": 0.5, "positive_years": 0, "yearly_dsrs": {},
                "stable": True, "trend": "Insufficient",
                "lifecycle": _lifecycle, "n_real_years": _n_real_years,
                "first_real_year": _first_real_year, "last_real_year": _last_real_year,
                "weighted_stability": 0.5, "recent_stability": 0.5,
            }

        # ── WEIGHTED-STABILITY-01 + LIFECYCLE-01 FIX ──────────────────────────────
        # BUG CORREGIDO: _current_year usa ts_max_yr no max(yearly_dsrs).
        # Antes, features muertas (last_real=2022) usaban 2022 como ref → sesgo +
        _current_year = ts_max_yr  # referencia fija al maximo del dataset

        _exp_weights = {yr: float(np.exp(-((_current_year - int(yr)) / _half_life_yr)))
                        for yr in yearly_dsrs}
        _total_w = sum(_exp_weights.values()) or 1.0
        _weighted_pos = sum(_exp_weights[yr] for yr, msr in yearly_dsrs.items() if msr > 0.0)
        _weighted_stability = _weighted_pos / _total_w

        _recent_yrs  = {yr: msr for yr, msr in yearly_dsrs.items()
                        if _current_year - int(yr) <= _recent_window_yr}
        _recent_stab = (sum(1 for v in _recent_yrs.values() if v > 0)
                        / max(len(_recent_yrs), 1)) if _recent_yrs else 0.0

        _composite = 0.60 * _weighted_stability + 0.40 * _recent_stab

        _trend_data = sorted(
            [(int(yr), msr) for yr, msr in yearly_dsrs.items()
             if _current_year - int(yr) <= _trend_window_yr],
            key=lambda x: x[0]
        )
        _trend_label, _trend_mod = "Stable", 1.00
        if len(_trend_data) >= 3:
            _xs = np.array([x[0] for x in _trend_data], dtype=float)
            _ys = np.array([x[1] for x in _trend_data], dtype=float)
            _slope = float(np.polyfit(_xs, _ys, 1)[0]) if np.std(_xs) > 0 else 0.0
            _std_y = float(np.std(_ys))
            if _slope > _trend_thresh:
                _trend_label, _trend_mod = "Rising", 1.10
            elif _slope < -_trend_thresh:
                _trend_label, _trend_mod = "Declining", 0.70
            elif _std_y >= 0.30:
                _trend_label, _trend_mod = "Volatile", 0.85
        else:
            _trend_label, _trend_mod = "Insufficient", 1.00

        stability_score = float(min(_composite * _trend_mod, 1.0))
        positive_years  = sum(1 for msr in yearly_dsrs.values() if msr > 0.0)

        # Descuento de madurez para YOUNG (incertidumbre binomial con pocos anos reales)
        # IC binomial 95% para p=1.0: n=2 -> [0.158,1.0]; n=3 -> [0.292,1.0]
        # Factor: 1yr=0.60, 2yr=0.80, 3yr+=1.00
        _maturity_factor = 1.00
        if _is_young and _n_real_years < _min_mature_years:
            _maturity_factor = min(1.0, max(0.60, _n_real_years / float(_min_mature_years)))
            _score_before    = stability_score
            stability_score  = stability_score * _maturity_factor
            print(f"[LIFECYCLE-01] '{feature_name}': YOUNG maturity discount"
                  f" n_real={_n_real_years} factor={_maturity_factor:.2f}"
                  f" {_score_before:.3f} -> {stability_score:.3f}")

        _simple_score = positive_years / max(len(yearly_dsrs), 1)
        print(f"[LIFECYCLE-01+WS-01] '{feature_name}' [{_lifecycle}]:"
              f" simple={_simple_score:.3f}"
              f" weighted={_weighted_stability:.3f} recent={_recent_stab:.3f}"
              f" trend={_trend_label}(x{_trend_mod}) mat={_maturity_factor:.2f}"
              f" => score={stability_score:.3f}")
        logger.info(
            f"[LIFECYCLE-01] '{feature_name}' [{_lifecycle}]:"
            f" weighted={_weighted_stability:.3f} recent={_recent_stab:.3f}"
            f" trend={_trend_label}(mod={_trend_mod}) maturity={_maturity_factor:.2f}"
            f" final={stability_score:.3f} n_real={_n_real_years}"
            f" first={_first_real_year} last={_last_real_year} ts_max={ts_max_yr}"
        )

        return {
            "stability_score":    round(stability_score, 4),
            "positive_years":     positive_years,
            "yearly_dsrs":        yearly_dsrs,
            "stable":             stability_score >= 0.30,
            "trend":              _trend_label,
            "weighted_stability": round(_weighted_stability, 4),
            "recent_stability":   round(_recent_stab, 4),
            "lifecycle":          _lifecycle,
            "n_real_years":       _n_real_years,
            "first_real_year":    _first_real_year,
            "last_real_year":     _last_real_year,
        }


    def _bootstrap_importance_rank(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_boots: int = 20,
        top_k: int = 20,
    ) -> Dict[str, float]:
        """
        [V2-FIX-2] MDI Bootstrap Stability Ranker (non-linear, IS-only).

        Entrena n_boots submodelos XGBoost sobre subsamples aleatorios del
        dataset y rankea las features por su frecuencia de apariciÃ³n en el
        Top-K de importancias (Mean Decrease Impurity).

        NO usa DSR ni OOS en ningÃºn momento â†’ puramente In-Sample.
        Evita el problem de Feature Selection Snooping de la metodologÃ­a DSR.

        Returns:
            Dict {feature: freq_score} donde freq_score âˆˆ [0, 1] representa
            la fracciÃ³n de submodelos en los que la feature apareciÃ³ en el Top-K.
        """
        from sklearn.utils import resample as _resample
        freq_counts: Dict[str, int] = {col: 0 for col in X.columns}
        n_features = len(X.columns)

        # n_boots adaptativo: mÃ¡s boots en datasets grandes (cap=20)
        _n_boots_eff = min(n_boots, max(10, len(y) // 500))

        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        lock = threading.Lock()

        def _run_bootstrap(boot_i):
            try:
                # Bootstrap subsample (con reemplazo, seed determinista)
                X_boot, y_boot = _resample(
                    X.values, y,
                    replace=True, n_samples=len(y), random_state=_LUNA_SEED + boot_i
                )
                _boot_clf = XGBClassifier(
                    n_estimators=SFI_N_ESTIMATORS,
                    max_depth=SFI_MAX_DEPTH,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=_LUNA_SEED + boot_i,
                    n_jobs=1,
                    verbosity=0
                )
                _boot_clf.fit(X_boot, y_boot)
                imp = _boot_clf.feature_importances_
                # Top-K features por importancia en este boot
                top_k_eff = min(top_k, n_features)
                top_k_idx = np.argsort(imp)[-top_k_eff:]
                with lock:
                    for idx in top_k_idx:
                        freq_counts[X.columns[idx]] += 1
            except Exception as _be:
                logger.debug(f"  [V2-FIX-2] Bootstrap {boot_i} error: {_be}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(_run_bootstrap, range(_n_boots_eff))

        # Normalizar a [0, 1]
        freq_scores = {
            col: freq_counts[col] / _n_boots_eff
            for col in X.columns
        }
        print(f"[BUG-FIX-LOG 2026-06-05] Corregido formatting logger.debug en feature_selection_e.py [V2-FIX-2]")
        logger.debug(
            "[V2-FIX-2] MDI Bootstrap ({} boots): top-5 por frecuencia: {}",
            _n_boots_eff,
            sorted(freq_scores.items(), key=lambda x: -x[1])[:5]
        )
        return freq_scores


    def evaluate(self, X: pd.DataFrame, y: pd.Series,

                 prices: pd.Series,
                 adf_penalties: Optional[Dict[str, float]] = None,
                 adv_penalties: Optional[Dict[str, float]] = None) -> List[str]:
        """
        EvalÃƒÂºa todas las features y retorna las que pasan.

        Args:
            X: Features candidatas (representantes de cluster + alpha signals)
            y: Target binario
            prices: Serie de precios BTC para calcular retornos

        Returns:
            Lista de features que superan el test de fuego
        """
        logger.info(f"[D] SFI-CPCV Rankeando {len(X.columns)} features "
                    f"y reteniendo el Top {self.top_n} (+ Alphas)...")
        logger.info(f"    XGB: {SFI_N_ESTIMATORS} estimators, max_depth={SFI_MAX_DEPTH}, "
                    f"costs={SFI_COST_ROUNDTRIP:.3%} round-trip")

        cpcv      = PurgedCPCV(n_groups=self.n_groups)
        # MEJORA-SFI-02: CPCV pequeÃƒÂ±o (n_groups=4) para evaluaciones anuales.
        # Menos grupos = menos tiempo por aÃƒÂ±o (datos mÃƒÂ¡s escasos por aÃƒÂ±o).
        cpcv_small = PurgedCPCV(n_groups=4)

        # Ã¢â€ â‚¬Ã¢â€ â‚¬ MEJORA-SFI-SHARPE-01 (2026-03-10) Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬
        # Bajo H0 (sin seÃƒÂ±al), SR_OOB ~ N(0, 1/sqrt(n_obs)) (aproximaciÃƒÂ³n iid).
        # Umbral significativo al 95% (z=1.645): SR* = 1.645 / sqrt(n_obs_oob)
        # Aplica max(floor=SFI_MIN_SHARPE, SR_dynamic): el floor es la cota mÃƒÂ­nima.
        # n_obs_oob Ã¢â€°Ë† len(datos_alineados) Ãƒâ€” (k_test / n_groups)
        # con CPCV (n=6, k=2): k_test/n_groups = 2/6 = 0.333
        _n_obs_common_preview = max(
            len(X.dropna(how="all").index.intersection(y.dropna().index)), 1
        )
        # k_test desde cfg.cpcv.k_test_blocks (ya en settings.yaml como k_test_blocks: 2)
        try:
            _k_test = int(getattr(_cfg_sfi.cpcv, 'k_test_blocks', 2))
        except Exception:
            _k_test = 2  # fallback — C(n_groups, 2) diseño estándar CPCV
        _n_obs_oob = max(int(_n_obs_common_preview * _k_test / max(self.n_groups, 1)), 30)

        # FIX-MATH-01: El _sr_dynamic es un threshold "per-period" (hourly) anualizado.
        ann_factor = np.sqrt(365 * 24)
        _sr_dynamic_ensemble = (1.645 / np.sqrt(_n_obs_oob)) * ann_factor
        
        # FIX-SHARPE-03: SFI evalúa Weak Learners individuales, no la estrategia final.
        # Un Weak Learner no necesita superar el umbral de significancia del 95% individualmente
        # (que sería _sr_dynamic_ensemble ~ 1.06). Solo necesita tener esperanza matemática positiva.
        # Imponemos un filtro base de 0.10 para limpiar puro ruido, pero permitimos que el ensamble
        # combine múltiples features débiles (SR 0.2, 0.4) para alcanzar el >1.5 global.
        _min_sharpe_threshold = 0.10
        self.min_sharpe_used = _min_sharpe_threshold
        _mode = f"weak learner threshold (SR>0.10) [Ensemble 95% = {_sr_dynamic_ensemble:.4f}]"
        logger.info(
            f"[D] FIX-SHARPE-03: n_obs_oob≈{_n_obs_oob} | "
            f"=> min_sharpe={_min_sharpe_threshold:.4f} [{_mode}]"
        )
        # -------------------------------------------------------------------------
        import pandas as pd
        common_idx = X.dropna(how="all").index.intersection(y.dropna().index)
        if len(common_idx) < 100:
            logger.error("[D] SFI falló: <100 filas comunes.")
            return []

        X_a = X.loc[common_idx]
        y_a = y.loc[common_idx].values
        p_a = prices.loc[common_idx].values
        ts_a = pd.Series(common_idx)

        total = len(X.columns)
        from joblib import Parallel, delayed
        import time

        # Extraemos variables de instancia para evitar pickling de 'self' si es pesado,
        # pero 'self._eval_one' necesita self. Lo ideal es no capturar 'logger' ni open files.
        _eval_one_fn = self._eval_one
        _eval_temporal_stability_fn = getattr(self, "_eval_temporal_stability", None)
        _min_sharpe = self.min_sharpe_used
        _top_n = self.top_n

        try:
            _stab_penalty_w = float(getattr(_cfg_sfi.features, 'stability_penalty_weight', 0.5))
        except Exception:
            _stab_penalty_w = 0.5

        logger.info(
            "[D] [FIX-BUGS-PRINTS] SFI-CPCV Pipeline con correcciones matemáticas activas:\n"
            "    - Sesgo de supervivencia corregido (se añaden Sharpes de 0.0 en predicción constante en folds)\n"
            "    - Autocorrelación serial corregida amortiguando el factor anualizador por sqrt(8760 / TBM_H)\n"
            "    - Varianza transversal DSR corregida usando n_obs reales de validación (len(X1))\n"
            f"    - Penalización de estabilidad temporal activa con peso: {_stab_penalty_w:.2f}"
        )

        def _process_feature(i, col):
            t0 = time.time()
            _nan_pct = float(X_a[col].isna().mean())
            _nunique = X_a[col].nunique()
            if _nan_pct > 0.85 or _nunique <= 1:
                return col, i, {"passed": False, "deflated_sharpe": 0.0, "mean_sharpe": 0.0, "n_folds": 0}, {}, t0, None, 0.0, 0.0

            # [FIX-IC-GUARD 2026-06-07] Mathematical Information Coefficient Guard
            # Previene que XGBoost intente ajustar reglas espurias sobre ruido puro.
            from scipy.stats import spearmanr
            rho, pval = spearmanr(X_a[col].fillna(0).values, y_a)
            if abs(rho) < 0.015 and col not in ALPHA_SIGNALS:
                logger.debug(f"[IC-GUARD] '{col}' purgada pre-XGBoost: IC nulo (rho={rho:+.4f})")
                return col, i, {"passed": False, "deflated_sharpe": 0.0, "mean_sharpe": 0.0, "n_folds": 0}, {}, t0, None, -999.0, -999.0


            # [FIX-H-FP-02 2026-05-30] ffill().fillna(0) en lugar de fillna(0) para XGBoost SFI.
            # XGBoost puede manejar NaN nativamente, pero para features con ausencia historica
            # (ETF pre-2024, DeFi pre-2020, Stablecoins pre-2019) el valor=0 es semanticamente
            # incorrecto: ese activo no existia, el mercado no estaba en 0%.
            # ffill propaga el ultimo valor valido. Periodo inicial sin historia -> 0 como base.
            _nan_pct_orig = float(X_a[col].isna().mean())
            x1 = X_a[col].ffill().fillna(0).values.reshape(-1, 1)
            if _nan_pct_orig > 0.0:
                _nan_filled = int(_nan_pct_orig * len(x1))
                print(f"[FIX-H-FP-02] '{col}': {_nan_pct_orig:.1%} NaN ({_nan_filled} filas) -> ffill+fillna(0)")

            mask_cond = None
            if any(col.startswith(p) for p in LOW_FREQ_PREFIX):
                activation_rate = (X_a[col].fillna(0) > 0).mean()
                if activation_rate < LOW_FREQ_LIMIT:
                    mask_cond = (X_a[col].fillna(0) > 0).values

            # [FIX-H-SFI-09 2026-05-30]: Guardar n_obs_base antes de que mask reduzca X1
            # para que el DSR no penalice 5.7x más las features de baja frecuencia.
            _n_obs_base = len(x1)  # tamaño completo pre-mask del dataset alineado

            res = _eval_one_fn(x1, y_a, p_a, cpcv, feature_name=col, mask=mask_cond,
                               n_trials_sfi=total, n_obs_base=_n_obs_base)

            stab_info: dict = {}
            adjusted_dsr = res["deflated_sharpe"]
            adjusted_mean_sharpe = res["mean_sharpe"]
            if col not in ALPHA_SIGNALS and res["n_folds"] > 0:
                try:
                    if _eval_temporal_stability_fn:
                        stab_info = _eval_temporal_stability_fn(x1, y_a, p_a, ts_a, cpcv_small, feature_name=col)
                    _stab_score = stab_info.get("stability_score", 1.0)
                    adjusted_dsr = res["deflated_sharpe"] * ((1 - _stab_penalty_w) + _stab_penalty_w * _stab_score)
                    adjusted_mean_sharpe = res["mean_sharpe"] * ((1 - _stab_penalty_w) + _stab_penalty_w * _stab_score)
                    
                    if _stab_score < 0.50:
                        adjusted_dsr = adjusted_dsr * (_stab_score / 0.50)**2
                        adjusted_mean_sharpe = adjusted_mean_sharpe * (_stab_score / 0.50)**2

                    if adf_penalties and col in adf_penalties:
                        adjusted_dsr *= adf_penalties[col]
                        adjusted_mean_sharpe *= adf_penalties[col]
                    if adv_penalties and col in adv_penalties:
                        adv_pen = adv_penalties[col]
                        if adv_pen == 0.0:
                            res = {**res, "passed": False}
                            adjusted_dsr = -9999.0
                            adjusted_mean_sharpe = -9999.0
                        else:
                            adjusted_dsr *= adv_pen
                            adjusted_mean_sharpe *= adv_pen
                        
                    stab_info["adjusted_dsr"] = round(adjusted_dsr, 4)
                    stab_info["adjusted_mean_sharpe"] = round(adjusted_mean_sharpe, 4)
                except Exception as _se2:
                    logger.exception(f"  SFI-02: stability eval fallida para {col}")

            return col, i, res, stab_info, t0, mask_cond, adjusted_dsr, adjusted_mean_sharpe

        logger.info(f"    [SFI] Iniciando procesamiento PARALELO de {total} features (ThreadPoolExecutor)...")
        from concurrent.futures import ThreadPoolExecutor
        _raw_results = []
        
        def _thread_worker(idx_col):
            i, col = idx_col
            res_tuple = _process_feature(i, col)
            # Unpack para log progresivo inmediato
            col_ret, i_ret, res, stab_info, t0, mask_cond, adjusted_dsr, adjusted_mean_sharpe = res_tuple
            status = "[OK]" if res["passed"] else "[XX]"
            cond_tag = " [COND]" if mask_cond is not None else ""
            stab_tag = ""
            if stab_info:
                p_years = stab_info.get("positive_years", "?")
                tot_years = len(stab_info.get("yearly_dsrs", [1,2,3,4,5]))
                s_score = stab_info.get("stability_score", 0.0)
                stab_tag = f" [STAB={s_score:.2f} {p_years}/{tot_years}]"

            elapsed = time.time() - t0
            # Log on every feature for deep traceability, or if slow
            logger.info(f"  [{i_ret+1:3d}/{total}] {col_ret:<40} DSR={res['deflated_sharpe']:+.3f} adjDSR={adjusted_dsr:+.3f} MeanSR={res['mean_sharpe']:+.3f} adjMeanSR={adjusted_mean_sharpe:+.3f} Folds={res['n_folds']} {status}{cond_tag}{stab_tag} ({elapsed:.1f}s)")
            return res_tuple

        with ThreadPoolExecutor(max_workers=8) as executor:
            _raw_results = list(executor.map(_thread_worker, enumerate(X.columns)))
            
        for res_tuple in _raw_results:
            col_ret, i_ret, res, stab_info, t0, mask_cond, adjusted_dsr, adjusted_mean_sharpe = res_tuple
            self.scores[col_ret] = {
                **res, 
                "stability": stab_info, 
                "adjusted_dsr": round(adjusted_dsr, 4),
                "adjusted_mean_sharpe": round(adjusted_mean_sharpe, 4)
            }

        # Seleccionar las TOP N features rankeadas por adjusted_dsr + Alphas
        ranking = self.get_ranking()
        valid_features = ranking[ranking["passed"] == True]

        # MEJORA-SFI-SHARPE-01: filtrar contextuales por umbral MeanSR dinámico.
        # Alphas exentas del filtro: validándose via Bootstrap WR en Mining.
        context_all = valid_features[~valid_features["feature"].isin(ALPHA_SIGNALS)]
        context_filtered = context_all[context_all["mean_sharpe"] >= self.min_sharpe_used]
        n_dropped = len(context_all) - len(context_filtered)
        if n_dropped > 0:
            logger.info(
                f"[D] SFI-SHARPE-01: {n_dropped} feature(s) contextual(es) descartadas "
                f"por MeanSR < {self.min_sharpe_used:.4f} "
                f"(n_obs_oob≈{_n_obs_oob}, SR_null_95={_sr_dynamic_ensemble:.4f})"
            )

        # 1. Conservar Alpha Signals validadas con MeanSR >= 0 (bypass DSR pero no sentido comun)
        # FIX-ALPHA-BYPASS-01: antes se conservaban TODAS (passed=True incluye MeanSR=-7.330)
        # Ahora: alpha entra si tiene n_folds>0 Y mean_sharpe>=0 (no destructiva)
        alpha_all = valid_features[valid_features["feature"].isin(ALPHA_SIGNALS)]
        alpha_selected_df = alpha_all[alpha_all["mean_sharpe"] >= 0.0]
        alpha_rejected_bypass = alpha_all[alpha_all["mean_sharpe"] < 0.0]
        if not alpha_rejected_bypass.empty:
            for _, row in alpha_rejected_bypass.iterrows():
                logger.warning(
                    f"  [D] Alpha rechazada por MeanSR<0: {row['feature']} "
                    f"(MeanSR={row['mean_sharpe']:+.3f}) — bypass denegado"
                )
        alpha_selected = alpha_selected_df["feature"].tolist()

        # [V2-FIX-2] MDI Bootstrap blend: leer peso desde settings
        try:
            _stab_sel_w = float(getattr(_cfg_sfi.features, 'stability_selection_weight', 0.0))
        except Exception:
            _stab_sel_w = 0.0  # default conservador: si no está en YAML, usar solo DSR (legacy)

        # 2. Top N contextuales rankeadas por score compuesto
        if _stab_sel_w > 0.0:
            # [V2-FIX-2] Calcular Bootstrap MDI IS rank y blendear con DSR adjusted
            logger.info(
                "[V2-FIX-2] MDI Bootstrap IS Selection activo "
                f"(stability_selection_weight={_stab_sel_w:.1%})"
            )
            _bootstrap_freq = self._bootstrap_importance_rank(X_a, y_a, n_boots=20, top_k=20)

            # Normalizar DSR adjusted al [0,1] para blend justo con bootstrap [0,1]
            _all_dsr_vals = [self.scores[c].get("adjusted_dsr", 0.0) for c in X_a.columns if c in self.scores]
            _dsr_max = max(_all_dsr_vals) if _all_dsr_vals else 1.0
            _dsr_min = min(_all_dsr_vals) if _all_dsr_vals else 0.0
            _dsr_range = max(_dsr_max - _dsr_min, 1e-9)

            _blend_scores = {}
            for _col in X_a.columns:
                if _col not in self.scores:
                    continue
                _dsr_raw = self.scores[_col].get("adjusted_dsr", 0.0)
                _dsr_norm = (_dsr_raw - _dsr_min) / _dsr_range
                _boot_freq = _bootstrap_freq.get(_col, 0.0)
                _blend_scores[_col] = _stab_sel_w * _boot_freq + (1 - _stab_sel_w) * _dsr_norm

            # Filtrar solo las features que pasaron el filtro MeanSR
            _context_cols = context_filtered["feature"].tolist()
            _blend_context = {c: _blend_scores.get(c, 0.0) for c in _context_cols}
            top_context_features = sorted(
                _blend_context.keys(), key=lambda c: -_blend_context[c]
            )[:self.top_n]
            logger.info(
                f"[V2-FIX-2] Top-{self.top_n} contextuales (blend DSR+Bootstrap): "
                f"{top_context_features[:5]}..."
            )
        else:
            # FIX-RANKING-01: Rankear Weak Learners por adjusted_mean_sharpe, no por mean_sharpe bruto o DSR.
            # El DSR saturado en 1.0/0.0 impide discriminar el top-15 y elegía features al azar.
            # Al usar adjusted_mean_sharpe, aplicamos correctamente penalizaciones ADF, temporales y adversarias.
            top_context_features = context_filtered.sort_values(
                "adjusted_mean_sharpe", ascending=False
            )["feature"].head(self.top_n).tolist()


        self.selected = list(set(alpha_selected + top_context_features))
        logger.info(f"[D] {len(self.selected)} features seleccionadas: "
                    f"{len(alpha_selected)} Alphas + {len(top_context_features)} Contextuales Top")

        # MEJORA-SFI-B03: guardar cache DSR para Etapa B del PROXIMO run.
        # El cache almacena el DSR de TODAS las features evaluadas (no solo las seleccionadas).
        # Etapa B usara este cache como segundo criterio en max(ICIR, DSR_previo).
        try:
            dsr_cache = {feat: round(scores["deflated_sharpe"], 6)
                         for feat, scores in self.scores.items()
                         if scores.get("n_folds", 0) > 0}
            DSR_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(DSR_CACHE, "w") as f:
                logger.info("[FIX-SHADOW-JSON-01] Guardo cache DSR sin import local redundante.")
                json.dump(dsr_cache, f, indent=2)
            logger.debug(f"[D] Cache DSR guardado: {len(dsr_cache)} features -> {DSR_CACHE}")
        except Exception as _ce:
            logger.debug(f"[D] Cache DSR no guardado: {_ce}")

        return self.selected



    def get_ranking(self) -> pd.DataFrame:
        rows = []
        for n, s in self.scores.items():
            rows.append({
                "feature":          n,
                "deflated_sharpe":  s["deflated_sharpe"],
                # MEJORA-SFI-02: usar adjusted_dsr como clave de ranking si disponible
                "adjusted_dsr":     s.get("adjusted_dsr", s["deflated_sharpe"]),
                "mean_sharpe":      s["mean_sharpe"],
                "adjusted_mean_sharpe": s.get("adjusted_mean_sharpe", s["mean_sharpe"]),
                "passed":           s["passed"],
                "n_folds":          s["n_folds"],
                "stability_score":  s.get("stability", {}).get("stability_score", None),
                "positive_years":   s.get("stability", {}).get("positive_years", None),
            })
        return pd.DataFrame(rows).sort_values("adjusted_mean_sharpe", ascending=False)


# =============================================================================
# ETAPA E: Forward Feature Selection
# =============================================================================

class ShapRFEFeatureSelector:
    """
    Etapa E de LdP adaptada (V2): Recursive Feature Elimination basado en SHAP Values.
    Entrena un modelo base, calcula la importancia de cada variable considerando
    interacciones n-dimensionales (SHAP), y poda el 10% inferior iterativamente.
    Evita la Ceguera Sinérgica del Greedy Forward Selection.
    """
    """
    Etapa E de LdP: encuentra la combinaciÃƒÂ³n ÃƒÂ³ptima de features.
    Parte de la feature con mayor Deflated Sharpe individual y aÃƒÂ±ade
    las siguientes si mejoran el Sharpe combinado Ã¢â€°Â¥ min_improve.
    """

    def __init__(self, max_features: int = FORWARD_MAX_FEATURES,
                 min_improve: float = FORWARD_MIN_IMPROVE,
                 n_groups: int = 5):
        self.max_features = max_features
        self.min_improve = min_improve
        self.n_groups = n_groups
        self.history: List[Dict] = []
        self.optimal: List[str] = []

    def _eval_combo(self, X: pd.DataFrame, y: pd.Series,
                    prices: pd.Series) -> float:
        """Sharpe OOS de una combinacion de features con CV walk-forward.

        FIX-FORWARD-03 (2026-05-15): Alinear horizonte de retorno con TBM.
        ANTES: retorno 1H con ann=sqrt(8760) -> Sharpe 9.8x inflado vs SFI-CPCV.
        AHORA: retorno forward al horizonte TBM configurado (default 96H) con
               ann=sqrt(8760/horizon_h), identico al evaluador SFI_CPCV._eval_one().
        Ademas: long-only (0/1) en lugar de long/short (1/-1), consistente con
               el sistema real de trading.
        """
        common = X.dropna(how="all").index.intersection(
            y.dropna().index).intersection(prices.index)
        if len(common) < 500:
            return -999.0

        Xv = X.loc[common].values
        yv = y.loc[common].values
        pv = prices.loc[common].values
        n  = len(Xv)
        fold_size = n // self.n_groups

        # FIX-FORWARD-03: leer horizonte TBM desde settings (mismo que SFI_CPCV)
        try:
            _tbm_h = int(getattr(_cfg_sfi.xgboost, 'vertical_barrier_hours', 96))
        except Exception:
            _tbm_h = 96
        ann_factor = np.sqrt((365 * 24) / _tbm_h)  # sqrt(8760/96)=9.55, NO sqrt(8760)=93.6

        fold_sharpes = []
        for i in range(1, self.n_groups):
            te_s = i * fold_size + SFI_EMBARGO_H
            te_e = min((i + 1) * fold_size, n)
            if te_s >= te_e:
                continue
            try:
                model = XGBClassifier(
                    n_estimators=100, max_depth=4,
                    learning_rate=0.1, random_state=_LUNA_SEED,
                    n_jobs=-1, verbosity=0
                )
                model.fit(Xv[:i * fold_size], yv[:i * fold_size])
                proba = model.predict_proba(Xv[te_s:te_e])[:, 1]

                # FIX-FORWARD-03: long-only (0/1) + retorno horizonte TBM
                sigs_lo = np.where(proba > 0.5, 1.0, 0.0)
                if sigs_lo.sum() == 0 or sigs_lo.sum() == len(sigs_lo):
                    continue  # prediccion constante: descartamos

                n_te = len(pv[te_s:te_e])
                if n_te <= _tbm_h:
                    if n_te < 2:
                        continue
                    fwd_ret   = np.diff(pv[te_s:te_e]) / pv[te_s:te_e][:-1]
                    sigs_eval = sigs_lo[:-1]
                    _ann = np.sqrt(365 * 24)  # fallback 1H si el fold es muy pequeno
                else:
                    fwd_ret   = pv[te_s:te_e][_tbm_h:] / pv[te_s:te_e][:-_tbm_h] - 1
                    sigs_eval = sigs_lo[:-_tbm_h]
                    _ann = ann_factor

                mn = min(len(sigs_eval), len(fwd_ret))
                if mn < 10:
                    continue
                sr_ret = sigs_eval[:mn] * fwd_ret[:mn] - sigs_eval[:mn] * SFI_COST_ROUNDTRIP
                if np.std(sr_ret) < 1e-10:
                    continue
                s = float(np.mean(sr_ret) / np.std(sr_ret) * _ann)
                fold_sharpes.append(np.clip(s, -10, 10))
            except Exception:
                continue

        return float(np.mean(fold_sharpes)) if fold_sharpes else -999.0

    def select(self, X: pd.DataFrame, y: pd.Series, prices: pd.Series,
               ranking: List[str]) -> List[str]:
        """
        Selección recursiva guiada por SHAP values.
        """
        try:
            import shap
        except ImportError:
            import logging
            logging.getLogger().warning("[E] SHAP no instalado. Fallback a RFE tradicional.")
            shap = None
            
        from lightgbm import LGBMClassifier
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("[E] SHAP-RFE Feature Selection...")
        logger.info(f"    max_features={self.max_features}, "
                    f"min_improve={self.min_improve*100:.0f}%")

        valid = [f for f in ranking if f in X.columns]
        if not valid:
            return []
            
        current_features = valid.copy()
        best_sharpe = self._eval_combo(X[current_features], y, prices)
        logger.info(f"    Baseline Sharpe (Todas {len(current_features)} features): {best_sharpe:.3f}")
        
        best_features = current_features.copy()
        step = 1
        
        import numpy as np
        
        while len(current_features) > self.max_features or len(current_features) > 5:
            X_curr = X[current_features]
            common = X_curr.dropna(how="all").index.intersection(y.dropna().index)
            if len(common) < 500:
                break
                
            X_train = X_curr.loc[common]
            y_train = y.loc[common]
            
            model = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, 
                                   subsample=0.8, colsample_bytree=0.8, random_state=_LUNA_SEED, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train)
            
            try:
                if shap is not None:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_train)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] 
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    shap_importance = dict(zip(current_features, mean_abs_shap))
                else:
                    shap_importance = dict(zip(current_features, model.feature_importances_))
            except Exception as e:
                logger.debug(f"Error calculando SHAP: {e}. Fallback a feature_importances_")
                shap_importance = dict(zip(current_features, model.feature_importances_))
                
            sorted_features = sorted(current_features, key=lambda f: shap_importance.get(f, 0), reverse=True)
            
            n_prune = max(1, int(len(current_features) * 0.10))
            next_features = sorted_features[:-n_prune]
            
            sharpe = self._eval_combo(X[next_features], y, prices)
            improve = ((sharpe - best_sharpe) / (abs(best_sharpe) + 1e-8) if best_sharpe > -900 else 1.0)
            
            logger.info(f"  [Ronda {step}] Eliminadas {n_prune} features (menor SHAP). "
                        f"Features restantes: {len(next_features)}. Sharpe={sharpe:.3f}")
            
            self.history.append({
                "step": step, 
                "n_features": len(next_features),
                "sharpe": sharpe, 
                "improvement": improve,
                "added": False,
                "pruned": sorted_features[-n_prune:]
            })
            
            if sharpe >= best_sharpe or (best_sharpe - sharpe) < 0.05:
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_features = next_features.copy()
                    logger.info(f"    ✅ Nuevo best_sharpe: {best_sharpe:.3f}")
                current_features = next_features.copy()
            else:
                logger.info(f"    ⚠️ Poda degradó el Sharpe significativamente. Se detiene el RFE.")
                break
                
            step += 1
            
        self.optimal = best_features
        logger.info(f"[E] {len(self.optimal)} features seleccionadas mediante SHAP-RFE — Sharpe={best_sharpe:.3f}")
        return self.optimal

# =============================================================================
# ETAPA D.2: ValidaciÃ³n Adversaria Univariable
# =============================================================================

class AdversarialValidator:
    """
    Etapa D.2 de LdP: Penaliza features cuyo comportamiento estadÃ­stico haya
    sufrido un Covariate Shift entre el set de entrenamiento y el de holdout.

    FIX-ADV-01 (2026-05-15): Dos niveles de acciÃ³n:
      1. Soft penalty (AUC 0.55â€“0.90): multiplica adjDSR por un factor < 1.
         Ãštil cuando la feature aÃºn aporta seÃ±al a pesar del drift moderado.
      2. Hard block (AUC > hard_block_auc=0.90): marca passed=False directamente.
         El multiplier 0.10x era inerte cuando DSR=0 (0 Ã— 0.10 = 0 = sin efecto).
         Con AUC>0.90, el clasificador distingue casi perfectamente train de holdout:
         la feature es una variable fantasma â€” no tiene sentido mantenerla.
    """
    # Umbral de exclusiÃ³n directa: AUC > 0.95 = Covariate Shift severo (Relajado de 0.90)
    HARD_BLOCK_AUC: float = 0.95

    def __init__(self, auc_threshold: float = 0.55, penalty_multiplier: float = 2.5):
        # Parametros depreciados, la clase ahora es binaria
        pass

    def evaluate_feature(self, X_train_s: pd.Series, X_holdout_s: pd.Series) -> Tuple[float, float]:
        # [FIX] Data Availability Window: ignorar el periodo antes de que la feature existiera
        # (ej. ETFs recientes que eran 100% NaNs al inicio de train).
        first_valid_train_idx = X_train_s.first_valid_index()
        if first_valid_train_idx is not None:
            X_train_s = X_train_s.loc[first_valid_train_idx:]
        else:
            return 1.0, 0.5

        X = pd.concat([X_train_s, X_holdout_s], axis=0).to_frame()
        y = np.array([0]*len(X_train_s) + [1]*len(X_holdout_s))

        mask = X[X.columns[0]].notna()
        X = X[mask]
        y = y[mask]

        if len(np.unique(y)) < 2 or min(np.bincount(y)) < 50:
            return 1.0, 0.5

        from sklearn.model_selection import StratifiedKFold as Adv_KF
        from sklearn.metrics import roc_auc_score
        from xgboost import XGBClassifier

        cv = Adv_KF(n_splits=3, shuffle=True, random_state=_LUNA_SEED)
        aucs = []
        for tr, te in cv.split(X, y):
            model = XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.1, n_jobs=1, missing=np.nan, verbosity=0)
            model.fit(X.iloc[tr].values, y[tr])
            preds = model.predict_proba(X.iloc[te].values)[:, 1]
            try:
                aucs.append(roc_auc_score(y[te], preds))
            except Exception:
                aucs.append(0.5)

        mean_auc = float(np.mean(aucs))

        # FIX-ADV-ARCH-04: Filtro puramente binario (Hard Block)
        # Se devuelve penalty=0.0 si hay shift severo, 1.0 de lo contrario.
        if mean_auc > self.HARD_BLOCK_AUC:
            return 0.0, mean_auc

        return 1.0, mean_auc

# =============================================================================
# PIPELINE COMPLETO
# =============================================================================

class FeatureSelectionPipelineE:
    """
    Pipeline completo de 5 etapas LdP para Luna V1 Fase E.

    Uso:
        pipeline = FeatureSelectionPipelineE()
        result = pipeline.run(features_parquet, resume=False)

        # Luego en train_xgboost.py:
        selected = pipeline.load_selected()
        X_train = df_train[selected]
    """

    def __init__(self):
        _n_reps = int(getattr(_cfg_sfi.features, 'sfi_n_reps_per_cluster', 2))
        self.clusterer  = FeatureClusterer(n_clusters=CLUSTER_FIXED_N, n_reps=_n_reps)
        self.lag_disc   = AutoLagDiscovery(max_lag=MAX_LAG_HOURS)
        self.sfi        = SFI_CPCV(top_n=SFI_TOP_N_FEATURES)
        self.forward    = ShapRFEFeatureSelector(max_features=FORWARD_MAX_FEATURES,
                                                 min_improve=FORWARD_MIN_IMPROVE)
        self.results: Dict = {}

    # Ã¢â€ â‚¬Ã¢â€ â‚¬ Helpers Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬

    @staticmethod
    def load_selected() -> List[str]:
        """Carga la lista de features seleccionadas + pass-through (para usar en entrenamiento)."""
        if not OUTPUT_FILE.exists():
            raise FileNotFoundError(
                f"No se encontrÃƒÂ³ {OUTPUT_FILE}. "
                "Ejecutar feature_selection_e.py primero."
            )
        with open(OUTPUT_FILE) as f:
            data = json.load(f)
        # Combinar SFI-selected + pass-through (sin duplicados)
        selected = data.get("selected_features", [])
        passthrough = data.get("pass_through_features", [])
        return list(dict.fromkeys(selected + passthrough))

    def _save_checkpoint(self, stage: str, data: dict):
        """Guarda checkpoint para poder retomar desde una etapa especÃƒÂ­fica."""
        existing = {}
        if CHECKPOINT.exists():
            with open(CHECKPOINT) as f:
                existing = json.load(f)
        existing[stage] = data
        with open(CHECKPOINT, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        logger.debug(f"Checkpoint guardado: etapa={stage}")

    def _load_checkpoint(self, stage: str) -> Optional[dict]:
        if not CHECKPOINT.exists():
            return None
        with open(CHECKPOINT) as f:
            data = json.load(f)
        return data.get(stage)

    def _write_output(self, selected: List[str],
                      alpha_passed: List[str], alpha_rejected: List[str],
                      stage_summaries: Dict,
                      passthrough: List[str] = None):
        """Guarda selected_features.json y el reporte markdown."""
        passthrough = passthrough or []
        out = {
            "timestamp":            datetime.utcnow().isoformat() + "Z",
            "pipeline_version":     "LunaV1-LdP-5Etapas",
            "n_input":              self.results.get("n_input", 0),
            "n_after_clustering":   self.results.get("n_after_B", 0),
            "n_after_lag_discovery":self.results.get("n_after_C", 0),
            "n_after_sfi":          self.results.get("n_after_D", 0),
            "n_final":              len(selected),
            "selected_features":    selected,
            "pass_through_features": passthrough,
            "n_passthrough":        len(passthrough),
            "alpha_signals_passed": alpha_passed,
            "alpha_signals_rejected": alpha_rejected,
            "stage_summaries":      stage_summaries,
            # MEJORA-SFI-02 (Run 14): temporal stability data para TEST-106
            "temporal_stability": {
                feat: {
                    "stability_score": self.sfi.scores.get(feat, {}).get("stability", {}).get("stability_score"),
                    "positive_years":  self.sfi.scores.get(feat, {}).get("stability", {}).get("positive_years"),
                    "yearly_dsrs":     self.sfi.scores.get(feat, {}).get("stability", {}).get("yearly_dsrs", {}),
                    "stable":          self.sfi.scores.get(feat, {}).get("stability", {}).get("stable", True),
                    "adjusted_dsr":    self.sfi.scores.get(feat, {}).get("adjusted_dsr"),
                }
                for feat in self.sfi.scores
                if self.sfi.scores[feat].get("stability")  # solo features con evaluacion temporal
            },
            "params": {
                "cluster_fixed_n":   CLUSTER_FIXED_N,
                "max_lag_hours":     MAX_LAG_HOURS,
                "sfi_top_n":         SFI_TOP_N_FEATURES,
                "sfi_n_groups":      SFI_N_GROUPS,
                "sfi_purge_h":       SFI_PURGE_H,
                "sfi_embargo_h":     SFI_EMBARGO_H,
                "forward_max_feat":  FORWARD_MAX_FEATURES,
                "forward_min_impr":  FORWARD_MIN_IMPROVE,
            }
        }
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.success(f"Ã¢Å“â€¦ selected_features.json guardado: {OUTPUT_FILE}")

        # [FIX-SFI-SANITY-01 2026-05-31] Check de sanidad post-guardado
        # Si el SFI selecciona <5 features, el XGB entrena con ruido puro.
        _n_sel   = len(selected)
        _n_pass  = len(passthrough)
        _n_total = _n_sel + _n_pass
        print(
            f"[FIX-SFI-SANITY-01] SFI completado: {_n_sel} features seleccionadas + "
            f"{_n_pass} pass-through = {_n_total} total para XGB"
        )
        if _n_sel < 5:
            _msg = (
                f"[FIX-SFI-SANITY-01/CRITICAL] SFI selecciono solo {_n_sel} features "
                f"(minimo requerido: 5). El XGBoost entrenaria con ruido puro. "
                f"Causas probables: DSR threshold alto, datos insuficientes IS, "
                f"o features con varianza=0. LA RUN SE DETIENE."
            )
            print(_msg)
            logger.critical(_msg)
            raise RuntimeError(_msg)
        if _n_sel < 10:
            print(
                f"[FIX-SFI-SANITY-01/WARNING] Solo {_n_sel} features < 10. "
                f"Riesgo de sobreajuste y baja generalizacion OOS."
            )
            print(f"[BUG-FIX-LOG 2026-06-05] Corregido formatting logger.warning en feature_selection_e.py [FIX-SFI-SANITY-01]")
            logger.warning(
                "[FIX-SFI-SANITY-01] Pocas features: n_sel={} | n_pass={} | total={}",
                _n_sel, _n_pass, _n_total
            )

        # Reporte Markdown
        sfi_ranking = self.sfi.get_ranking()
        fwd_hist = pd.DataFrame(self.forward.history)
        md_lines = [
            "# Feature Selection Report Ã¢â‚¬â€  Luna V1",
            f"**Fecha:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
            f"**Pipeline:** LÃƒÂ³pez de Prado 5 Etapas  ",
            "",
            "## Resumen",
            f"| Etapa | Input | Output |",
            f"|---|---|---|",
            f"| [A] FracDiff DinÃƒÂ¡mico | Ã¢â‚¬â€  | (externo) |",
            f"| [B] Clustering JerÃƒÂ¡rquico (N={CLUSTER_FIXED_N}) | {out['n_input']} | {out['n_after_clustering']} |",
            f"| [C] Lag Discovery MI | {out['n_after_clustering']} + 5ÃŽÂ± | {out['n_after_lag_discovery']} |",
            f"| [D] SFI-CPCV (Top {SFI_TOP_N_FEATURES} Contextuales) | {out['n_after_lag_discovery']} | {out['n_after_sfi']} |",
            f"| [E] Forward Selection (Ã¢â€°Â¥{FORWARD_MIN_IMPROVE*100:.0f}%) | {out['n_after_sfi']} | **{out['n_final']}** |",
            "",
            "## Alpha Signals (R22)",
            f"**Pasaron SFI:** {', '.join(alpha_passed) if alpha_passed else 'Ninguna'}  ",
            f"**Rechazadas:** {', '.join(alpha_rejected) if alpha_rejected else 'Ninguna'}  ",
            "",
            "## Features Seleccionadas Finales",
        ]
        for i, f in enumerate(selected, 1):
            md_lines.append(f"{i}. `{f}`")
        md_lines += [
            "",
            "## Ranking SFI (todas las features evaluadas)",
            "",
            sfi_ranking.to_markdown(index=False) if len(sfi_ranking) > 0 else "N/A",
            "",
            "## Forward Selection Ã¢â‚¬â€  Historial",
            "",
            fwd_hist.to_markdown(index=False) if len(fwd_hist) > 0 else "N/A",
        ]
        REPORT_FILE.write_text("\n".join(md_lines), encoding="utf-8")
        logger.success(f"Ã¢Å“â€¦ Reporte guardado: {REPORT_FILE}")

    # Ã¢â€ â‚¬Ã¢â€ â‚¬ Pipeline principal Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬Ã¢â€ â‚¬

    def run(self, features_parquet: Optional[Path] = None,
            resume: bool = False) -> Dict:
        """
        Ejecuta las 5 etapas del pipeline.

        Args:
            features_parquet: Ruta al parquet de features (default: features_train.parquet)
            resume: Si True, retoma desde el checkpoint mÃƒÂ¡s avanzado disponible

        Returns:
            Dict con resultados de cada etapa
        """
        logger.info("=" * 65)
        logger.info("Feature Selection Pipeline Ã¢â‚¬â€  Luna V1 (LÃƒÂ³pez de Prado 5 Etapas)")
        logger.info("=" * 65)
        logger.info(f"ADVERTENCIA: tiempo estimado en CPU: ~8 min (26 candidatos tras clustering B)")

        # FIX: Siempre borrar el checkpoint al inicio de un run fresco (resume=False).
        # AsÃƒÂ­ se garantiza que cada ejecuciÃƒÂ³n sin --resume recalcula TODO desde cero.
        # El checkpoint solo se preserva si el usuario pasa explÃƒÂ­citamente --resume.
        if not resume and CHECKPOINT.exists():
            CHECKPOINT.unlink()
            logger.info(f"Ã°Å¸â€”â€˜Ã¯Â¸Â   Checkpoint previo eliminado Ã¢â‚¬â€  run limpio desde Etapa B (resume=False)")

        # Cargar datos
        fp = features_parquet or (DATA_DIR / "features_train.parquet")
        if not Path(fp).exists():
            raise FileNotFoundError(f"No encontrado: {fp}")

        logger.info(f"Cargando features desde {fp}...")
        df = pd.read_parquet(fp)

        # [WFB-CAUSAL-FIX-SFI] Cortar el DataFrame al train_end de la ventana activa.
        # features_train.parquet puede contener datos mÃ¡s allÃ¡ del train_end de la ventana WFB
        # (el parquet es compartido entre ventanas). Sin este corte, el SFI evalÃºa features
        # usando datos OOS de la ventana, produciendo rankings DSR distintos segÃºn la fecha
        # mÃ¡xima del parquet, lo que viola la causalidad y hace no-reproducible el WFB.
        try:
            from config.settings import cfg as _cfg_sfi_fix
            _wfb_train_end = getattr(getattr(_cfg_sfi_fix, 'temporal_splits', None), 'train_end', None)
            if _wfb_train_end:
                _cutoff_sfi = pd.Timestamp(str(_wfb_train_end), tz='UTC')
                if df.index.tz is None:
                    _cutoff_sfi = _cutoff_sfi.tz_localize(None)
                _orig_len = len(df)
                df = df[df.index <= _cutoff_sfi]
                logger.info(
                    f"[WFB-CAUSAL-FIX-SFI] DataFrame cortado a train_end={_wfb_train_end}: "
                    f"{_orig_len} â†’ {len(df)} filas. Datos post-cutoff excluidos de evaluaciÃ³n SFI."
                )
            else:
                logger.debug("[WFB-CAUSAL-FIX-SFI] No se encontrÃ³ train_end en settings â€” usando parquet completo.")
        except Exception as _sfi_cut_err:
            logger.warning(f"[WFB-CAUSAL-FIX-SFI] No se pudo aplicar corte train_end: {_sfi_cut_err}. Continuando sin corte.")

        # Loguea el fingerprint de features_train + features_holdout al inicio
        # de cada SFI. Si los datos cambiaron respecto al run anterior, avisa
        # explicitamente "CACHE INVALIDATED" para que el usuario sepa que se
        # espera varianza en el feature set resultante.
        #
        # El fingerprint usa: nrows | fecha_inicio | fecha_fin | ncols
        # (misma funcion que ya usan los caches de lag MI en Etapa C).
        # Se persiste en _fp_history.json junto a _lag_cache.json y _dsr_cache.json.
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        def _make_fp(df_: "pd.DataFrame") -> str:
            idx = df_.index
            s = idx[0].isoformat() if hasattr(idx[0], "isoformat") else str(idx[0])
            e = idx[-1].isoformat() if hasattr(idx[-1], "isoformat") else str(idx[-1])
            return f"{len(df_)}rows|{s}|{e}|{len(df_.columns)}cols"

        _fp_history_path = DATA_DIR / "_fp_history.json"
        _train_fp   = _make_fp(df)
        _holdout_fp = "(no existe)"
        _holdout_path_fp = DATA_DIR / "features_holdout.parquet"
        if _holdout_path_fp.exists():
            try:
                import pyarrow.parquet as _pq_fp
                _ho_meta = _pq_fp.read_metadata(_holdout_path_fp)
                # Leemos solo metadata para no recargar todo el archivo
                _ho_rows = _ho_meta.num_rows
                _ho_cols = len(_pq_fp.read_schema(_holdout_path_fp).names)
                _holdout_fp = f"{_ho_rows}rows|{_ho_cols}cols"
            except Exception as _fp_err:
                _holdout_fp = f"(error leyendo metadata: {_fp_err})"

        # Comparar con fingerprints del run anterior
        logger.info("[FIX-SHADOW-JSON-01] Comparando fingerprints del run anterior con modulo json global correcto.")
        _prev_fp_data: dict = {}
        if _fp_history_path.exists():
            try:
                _prev_fp_data = json.loads(_fp_history_path.read_text(encoding="utf-8"))
            except Exception:
                _prev_fp_data = {}

        _prev_train_fp   = _prev_fp_data.get("features_train_fp",   "")
        _prev_holdout_fp = _prev_fp_data.get("features_holdout_fp", "")
        _train_changed   = _train_fp   != _prev_train_fp   and bool(_prev_train_fp)
        _holdout_changed = _holdout_fp != _prev_holdout_fp and bool(_prev_holdout_fp)

        logger.info(
            f"[R2-FP-01] features_train   fp = {_train_fp}"
            + (" â†  MISMO QUE RUN ANTERIOR" if not _train_changed and _prev_train_fp else
               " â†  NUEVO (sin historial)" if not _prev_train_fp else
               " â†  âš  CAMBIO DETECTADO")
        )
        logger.info(
            f"[R2-FP-01] features_holdout fp = {_holdout_fp}"
            + (" â†  MISMO QUE RUN ANTERIOR" if not _holdout_changed and _prev_holdout_fp else
               " â†  NUEVO (sin historial)" if not _prev_holdout_fp else
               " â†  âš  CAMBIO DETECTADO")
        )

        if _train_changed or _holdout_changed:
            logger.warning(
                "[R2-FP-01] âš  CACHE INVALIDATED â€” los parquets cambiaron respecto al run anterior. "
                "Se espera varianza en el feature set: clustering diferente, lags MI recalculados, "
                "features macro/FRED con NaN% distinto. "
                f"{'features_train cambio. ' if _train_changed else ''}"
                f"{'features_holdout cambio.' if _holdout_changed else ''}"
            )
        elif _prev_train_fp:
            logger.info("[R2-FP-01] âœ” Datos SIN cambios â€” caches de lags y DSR reutilizables. Varianza baja esperada.")

        # Persistir fingerprints actuales para el prÃ³ximo run
        try:
            _fp_history_path.write_text(
                json.dumps({
                    "timestamp":           datetime.utcnow().isoformat() + "Z",
                    "features_train_fp":   _train_fp,
                    "features_holdout_fp": _holdout_fp,
                    "features_train_path": str(fp),
                    "run_id":              __import__("os").environ.get("LUNA_RUN_ID", "unknown"),
                }, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as _fp_save_err:
            logger.debug(f"[R2-FP-01] No se pudo guardar _fp_history.json: {_fp_save_err}")
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # Target dinÃ¡mico (R17)
        if "Target_TBM_Bin" in df.columns:
            # [FIX-SFI-TBM-01] Usar el etiquetado Triple Barrera generado por Feature Pipeline
            logger.info("[SFI] Usando Target_TBM_Bin (Single Source of Truth) para selecciÃ³n predictiva.")
            y = df["Target_TBM_Bin"]
        elif "target" in df.columns:
            y = df["target"]
        elif "close" in df.columns:
            # Fallback (no recomendado) si se estÃ¡ corriendo con datos antiguos
            logger.warning("[SFI] Target_TBM_Bin no encontrado! Fallback a target proxy 24H (riesgo de Dual-Labeling Trap).")
            fwd_ret = df["close"].pct_change(24).shift(-24)
            y_float = (fwd_ret > 0).astype(float)
            y_float[fwd_ret.isna()] = np.nan
            y = y_float
        else:
            raise ValueError("No se encontrÃƒÂ³ 'Target_TBM_Bin', 'target' ni 'close' en el parquet")

        # Precios para cÃƒÂ¡lculo de retornos en SFI
        prices = df["close"] if "close" in df.columns else y.cumsum()

        # Separar raw features de alpha signals
        alpha_cols = [c for c in ALPHA_SIGNALS if c in df.columns]
        # BUG-M23-01 (2026-03-16): excluir del pool raw:
        #   - ALPHA_SIGNALS (van a Etapa D directamente)
        #   - PASSTHROUGH_FEATURES (van directo al JSON output, saltan SFI)
        #   - RAW_COLS_BLACKLIST: OHLC raw (corr~1.0 con close) + alpha_combined (BUG-R12-01)
        _excluded = set(ALPHA_SIGNALS + PASSTHROUGH_FEATURES + ["target", "close"]) | RAW_COLS_BLACKLIST
        raw_cols   = [c for c in df.columns
                      if c not in _excluded
                      and not any(sub in c.lower() for sub in TIPO1_SLOW_SUBSTRINGS)
                      and df[c].dtype in [np.float32, np.float64, np.int32, np.int64]]

        # â”€â”€ Etapa PRE-A: Estacionariedad ADF (Pilar 3) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        adf_penalties = {c: 1.0 for c in df.columns}
        chk_PreA = self._load_checkpoint("PreA") if resume else None
        
        if chk_PreA:
            adf_penalties = chk_PreA["penalties"]
            logger.info(f"[PRE-A] Retomando ADF Test desde checkpoint: {len(adf_penalties)} features procesadas")
        else:
            try:
                adf_filter = ADFStationarityFilter()
                adf_penalties = adf_filter.evaluate(df, alpha_cols)
                
                _bad_macros = [(k, v) for k, v in adf_penalties.items() if v < 1.0]
                if _bad_macros:
                    logger.warning(f"[PRE-A] ðŸš¨ {len(_bad_macros)} variables sufren de RaÃ­z Unitaria (Random Walk). Top examples:")
                    for k, v in _bad_macros[:5]:
                        logger.warning(f"      {k:<30} -> Penalty: {v:.2f}x")
                    
                self._save_checkpoint("PreA", {"penalties": adf_penalties})
            except Exception as e:
                logger.error(f"[PRE-A] Error en test ADF (Â¿statsmodels no instalado?). Filtro omitido: {e}")

        # â”€â”€ FIX-SFI-HOLDOUT-NAN-01 (Actualizado R4 2026-03-25) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Filtrar raw_cols usando NaN% en features_holdout.parquet.
        # R4: Distinguir NaNs estructurales de NaNs por "Publication Lag" (FRED).
        # features macro suelen tener trailing NaNs al final del holdout por retraso en publicaciÃ³n.
        # Si los NaNs son estrictamente trailing (hasta un limite, ej 45%), se conservan.
        try:
            _hmax_nan = float(getattr(_cfg_sfi.features, 'sfi_holdout_max_nan_pct', 0.10))
        except Exception:
            _hmax_nan = 0.10
            
        _max_trailing_nan_pct = 0.45  # Permite hasta 45% de trailing NaNs (aprox 2.5 meses en un holdout de 6m)
        _holdout_path = DATA_DIR / "features_holdout.parquet"
        
        if _holdout_path.exists():
            try:
                import pyarrow.parquet as _pq
                _holdout_schema_cols = set(_pq.read_schema(_holdout_path).names)
                _cols_to_check = [c for c in raw_cols if c in _holdout_schema_cols]
                
                if _cols_to_check:
                    _holdout_df = pd.read_parquet(_holdout_path, columns=_cols_to_check)
                    _ho_len = len(_holdout_df)
                    
                    _holdout_high_nan = set()
                    _lag_pardoned = set()
                    _broken_internal = set()
                    _broken_lag = set()
                    
                    for c in _cols_to_check:
                        s = _holdout_df[c]
                        total_nan_pct = s.isna().mean()
                        
                        if total_nan_pct > _hmax_nan:
                            # Verificar si los NaNs son por publication lag (trailing_nans)
                            last_valid_idx = s.last_valid_index()
                            if last_valid_idx is None:
                                # Feature completamente vacÃ­a
                                _broken_internal.add(c)
                                _holdout_high_nan.add(c)
                                continue
                                
                            # Extraer la porciÃ³n "interna" de los datos (hasta el Ãºltimo dato vÃ¡lido)
                            # y verificar si esa porciÃ³n es densa o tiene huecos.
                            s_internal = s.loc[:last_valid_idx]
                            internal_nan_pct = s_internal.isna().mean()
                            
                            # Cantidad de NaNs en la cola (despuÃ©s del Ãºltimo vÃ¡lido)
                            trailing_nans = _ho_len - len(s_internal)
                            trailing_nan_pct = trailing_nans / _ho_len if _ho_len > 0 else 0
                            
                            if internal_nan_pct > _hmax_nan:
                                # Tiene demasiados agujeros internos -> estructuralmente rota
                                _broken_internal.add(c)
                                _holdout_high_nan.add(c)
                            elif trailing_nan_pct > _max_trailing_nan_pct:
                                # Feature muy atrasada o discontinuada (supera lÃ­mite de lag)
                                _broken_lag.add(c)
                                _holdout_high_nan.add(c)
                                logger.warning(
                                    f"[FIX-SFI-HOLDOUT-NAN-01] ðŸš¨ ALERTA DEPRECACIÃ“N: Feature '{c}' eliminada. "
                                    f"SuperÃ³ el lÃ­mite de trailing NaNs ({trailing_nan_pct:.1%} > {_max_trailing_nan_pct:.1%}). "
                                    f"Â¿Datos discontinuados en origen o API rota?"
                                )
                            else:
                                # Perdonada: Es una feature sana pero con publication lag macroeconÃ³mico
                                _lag_pardoned.add(c)

                    _before = len(raw_cols)
                    raw_cols = [c for c in raw_cols if c not in _holdout_high_nan]

                    if _holdout_high_nan:
                        logger.info(
                            f"[FIX-SFI-HOLDOUT-NAN-01] Filtro estructural NaN: "
                            f"{_before}â†’{len(raw_cols)} features | "
                            f"Rotas internas ({len(_broken_internal)}) | "
                            f"Deprecadas por lag extremo ({len(_broken_lag)}) | "
                            f"Perdonadas por Publication Lag ({len(_lag_pardoned)}): {', '.join(sorted(_lag_pardoned)) if _lag_pardoned else '0'}"
                        )
                    else:
                        logger.info(
                            f"[FIX-SFI-HOLDOUT-NAN-01] Todas pasan el filtro estructural (Perdonadas por Lag: {len(_lag_pardoned)})."
                        )
            except Exception as _e_hnan:
                logger.warning(f"[FIX-SFI-HOLDOUT-NAN-01] Error aplicando filtro R4: {_e_hnan}")
        else:
            logger.warning(f"[FIX-SFI-HOLDOUT-NAN-01] {_holdout_path} no existe â€” filtro omitido.")
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # â”€â”€ [OOD-GUARD-SFI] DetecciÃ³n de features con distribuciÃ³n degenerada en OOS â”€â”€
        # POSICIÃ“N CRÃ TICA: justo antes del paso B (clustering).
        # Si el guard se aplica despuÃ©s del SFI, solo reduce la lista final sin dar
        # oportunidad a que otras features llenen los huecos. Aplicarlo aquÃ­ permite
        # que el SFI seleccione las N mejores features VÃ LIDAS en OOS.
        # -- [ARCH-26-FIX-A 2026-06-02] OOD-GUARD-SFI con IS reciente (MISMO PATRON QUE ARCH-25) --
        # PROBLEMA: OOD-GUARD-SFI usaba features_validation.parquet = Ene-Abr 2025 (100% BULL extremo).
        # Features macro (M2, T10Y2Y, CPI, yield_curve) tienen baja varianza en bull puro ->
        # OOD las bloqueaba por LOW_STD -> 79/93 candidatas eliminadas (85% del pool SFI).
        # SOLUCION: usar ultimo 20% del IS como referencia OOD (mismo regimen, sin sesgo de distribucion).
        # Consistente con ARCH-25-FIX-A (XGBoost OOD Guard usa IS reciente).
        _ood_split_pct = 0.20   # ultimo 20% del IS como referencia OOD
        _ood_n_ref   = max(int(len(df) * _ood_split_pct), 200)
        _ood_n_train = len(df) - _ood_n_ref
        _ood_is_train = df.iloc[:_ood_n_train]
        _ood_is_ref   = df.iloc[_ood_n_train:]
        print(   # RULE[fixbugsprints.md]
            f"[ARCH-26-FIX-A] OOD-GUARD-SFI: IS reciente como ref OOD "
            f"(train={_ood_n_train} filas, ref={_ood_n_ref} filas, split={_ood_split_pct:.0%})"
        )
        logger.info(
            f"[ARCH-26-FIX-A] OOD-GUARD-SFI: referencia OOD = ultimas {_ood_n_ref} filas IS "
            f"({_ood_split_pct:.0%}). ANTES usaba features_validation.parquet (regimen BULL 2025)."
        )
        try:
            from luna.utils.ood_feature_guard import filter_ood_features as _ood_sfi_filter
            _ood_checkable = [c for c in raw_cols
                              if c in _ood_is_ref.columns and c in _ood_is_train.columns]
            if _ood_checkable:
                _valid_raw, _ood_rpts = _ood_sfi_filter(
                    X_train=_ood_is_train[_ood_checkable],
                    X_oos=_ood_is_ref[_ood_checkable],
                    context="SFI/CandidatePool-IS",
                )
                _n_ood_blocked = sum(1 for r in _ood_rpts if r.blocked)
                if _n_ood_blocked > 0:
                    _ood_blocked_names = [r.feature for r in _ood_rpts if r.blocked]
                    logger.warning(
                        f"[ARCH-26-FIX-A][OOD-GUARD-SFI] {_n_ood_blocked} features bloqueadas "
                        f"por distribucion degenerada en IS reciente: {_ood_blocked_names}. "
                        f"Estas features colapsan dentro del IS -> eliminar es correcto."
                    )
                    print(   # RULE[fixbugsprints.md]
                        f"[ARCH-26-FIX-A] OOD-GUARD-SFI bloqueo {_n_ood_blocked} features "
                        f"(degeneradas en IS reciente): {_ood_blocked_names}"
                    )
                    
                    # [FIX-WHITELIST-01] Rescate de características con estacionariedad analítica
                    _critical_features = {"close_fd", "hmm_regime", "hmm_velocity_bull", "hmm_acceleration_bull", "volatility_fd"}
                    _blocked_criticals = _critical_features.intersection(set(_ood_blocked_names))
                    if _blocked_criticals:
                        _rescue_msg = f"[FIX-WHITELIST-01] Rescatando {_blocked_criticals} del bloqueo OOD (estacionariedad analitica p-value garantizada)."
                        logger.success(_rescue_msg)
                        print(_rescue_msg)
                        _ood_blocked_names = [n for n in _ood_blocked_names if n not in _critical_features]
                        
                    _ood_not_checked = [c for c in raw_cols if c not in _ood_checkable]
                    raw_cols = _valid_raw + _ood_not_checked
                    
                    # Eliminar las bloqueadas del dataset
                    if _ood_blocked_names:
                        raw_cols = [c for c in raw_cols if c not in _ood_blocked_names]
                else:
                    logger.info(
                        f"[ARCH-26-FIX-A][OOD-GUARD-SFI] OK: {len(_ood_checkable)} candidatos "
                        f"pasan el control OOD (IS reciente). Pool SFI intacto."
                    )
                    print(   # RULE[fixbugsprints.md]
                        f"[ARCH-26-FIX-A] OOD-GUARD-SFI: {len(_ood_checkable)} candidatos OK "
                        f"(sin degeneracion en IS reciente)"
                    )
            else:
                logger.debug("[ARCH-26-FIX-A][OOD-GUARD-SFI] No hay columnas evaluables -> guard omitido.")
        except Exception as _ood_sfi_err:
            logger.warning(
                f"[ARCH-26-FIX-A][OOD-GUARD-SFI] Error en guard OOD -> pool sin filtrar: {_ood_sfi_err}"
            )
            print(f"[ARCH-26-FIX-A] OOD-GUARD-SFI ERROR (pool conservado): {_ood_sfi_err}")
        # -----------------------------------------------------------------------------------------

        # [Paso B] Clustering JerÃ¡rquico & SelecciÃ³n Representantes
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        X_raw = df[raw_cols].copy()
        X_alpha = df[alpha_cols]
        prices = df['close'] if 'close' in df.columns else y.cumsum()

        chk_B = self._load_checkpoint('B') if resume else None
        if chk_B:
            repr_features = chk_B['selected']
            logger.info(f'[B] Retomando desde checkpoint: {len(repr_features)} features')
        else:
            repr_features = self.clusterer.fit_transform(X_raw, y, prices=prices)
            self._save_checkpoint('B', {'selected': repr_features})

        self.results['n_after_clustering'] = len(repr_features)

        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # [Paso C] Automatic Lag Discovery & Alignment
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        _lag_cache_path = Path(DATA_DIR) / '_lag_cache.json'

        def _data_fingerprint(df_: pd.DataFrame) -> str:
            import pandas as pd
            idx = df_.index
            s = idx[0].isoformat() if hasattr(idx[0], 'isoformat') else str(idx[0])
            e = idx[-1].isoformat() if hasattr(idx[-1], 'isoformat') else str(idx[-1])
            return f'{len(df_)}rows|{s}|{e}|{len(df_.columns)}cols'

        _domain_lags = {}
        _effective_cache = None
        _mi_lags_valid = {}
        _dsr_verified_lags = {}
        current_fp = _data_fingerprint(X_raw)

        logger.info("[FIX-SHADOW-JSON-01] Leyendo _lag_cache.json sin import local redundante.")
        if _lag_cache_path.exists():
            try:
                _cache_data = json.loads(_lag_cache_path.read_text(encoding='utf-8'))
                if current_fp == _cache_data.get('mi_data_fingerprint'):
                    _mi_lags_valid = _cache_data.get('mi_lags', {})
                    _effective_cache = {**_domain_lags, **_mi_lags_valid}
            except Exception as e:
                logger.warning(f'[C] Error leyendo _lag_cache.json: {e}')

        X_lagged = self.lag_disc.transform(X_raw[repr_features], y,
                                           lag_cache=_effective_cache,
                                           hmm_regime=df["HMM_Regime"] if "HMM_Regime" in df.columns else None)
        # AÃƒÂ±adir alpha signals (sin lag discovery Ã¢â‚¬â€  R22, lags ya baked)
        for c in alpha_cols:
            X_lagged[c] = X_alpha[c]
        col_lags_saved = {col: int(lag)
                          for col, lag in self.lag_disc.optimal_lags.items()
                          if col in repr_features}
        extra_saved = {col: int(self.lag_disc.optimal_lags.get(col.rsplit("_milag", 1)[0].split("h")[0], 0))
                       for col in X_lagged.columns
                       if col not in repr_features + alpha_cols}
        self._save_checkpoint("C", {
            "col_lags":  col_lags_saved,
            "extra_cols": extra_saved,
            "alpha_cols": alpha_cols,
            "columns":   list(X_lagged.columns),
        })
        # Persistir cache v3.0: domain_lags preservados + mi_lags del run actual
        try:
            # Rescatar domain_lags del cache previo (si existe v3.0) o usar vacÃƒÂ­o
            prev_domain = _domain_lags if _domain_lags else {}
            # mi_lags: los reciÃƒÂ©n calculados (excluir los que eran domain)
            new_mi_lags = {k: v for k, v in col_lags_saved.items()
                           if k not in prev_domain}
            # Merge con mi_lags previos vÃƒÂ¡lidos (si mismo dataset, sumar features nuevas)
            if current_fp == _cache_data.get("mi_data_fingerprint", "") if _lag_cache_path.exists() else False:
                merged_mi = {**_mi_lags_valid, **new_mi_lags}  # nuevos overrides previos
            else:
                merged_mi = new_mi_lags  # dataset nuevo Ã¢â€ â€™ solo lags frescos

            # Merge dsr_verified: preservar verificados previos + aÃƒÂ±adir nuevos
            _prev_dsr_verified = _cache_data.get("dsr_verified_lags", {}) \
                if _lag_cache_path.exists() and '_cache_data' in dir() else {}
            _merged_dsr_verified = {**_prev_dsr_verified, **_dsr_verified_lags}

            _lag_cache_path.write_text(
                json.dumps({
                    "version": "3.0",
                    "description": (
                        "Lag cache v3.0 dos niveles: "
                        "domain_lags=teoricos inmutables, "
                        "mi_lags=empiricos seed=42, ligados al fingerprint, "
                        "dsr_verified_lags=empiricos por DSR-Lag Scan (M-29+)."
                    ),
                    "domain_lags":          prev_domain,
                    "mi_lags":              merged_mi,
                    "dsr_verified_lags":    _merged_dsr_verified,
                    "mi_data_fingerprint":  current_fp,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8")
            logger.info(f"[C] Lag cache v3.0 guardado: "
                        f"{len(prev_domain)} domain + {len(merged_mi)} MI "
                        f"+ {len(_merged_dsr_verified)} DSR-verificados "
                        f"| fp={current_fp[:60]}")
        except Exception as e:
            logger.warning(f"[C] No se pudo guardar _lag_cache.json: {e}")
        self.results["n_after_C"] = len(X_lagged.columns)
        logger.info(f"[C] Total candidatos para SFI: {len(X_lagged.columns)} "
                    f"({self.results['n_after_C'] - len(alpha_cols)} raw + "
                    f"{len(alpha_cols)} alpha signals)")

        # â”€â”€ Etapa D.2: ValidaciÃ³n Adversaria Univariable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        adv_penalties = {c: 1.0 for c in X_lagged.columns}
        adv_aucs = {c: 0.5 for c in X_lagged.columns}
        
        chk_D2 = self._load_checkpoint("D2") if resume else None
        if chk_D2:
            adv_penalties = chk_D2.get("penalties", adv_penalties)
            adv_aucs = chk_D2.get("aucs", adv_aucs)
            logger.info(f"[D.2] Retomando Validación Adversaria desde checkpoint: {len(adv_penalties)} features procesadas")
        else:
            if _holdout_path.exists():
                import pyarrow.parquet as _pq
                _ho_cols = set(_pq.read_schema(_holdout_path).names)
                _raw_needed = [c for c in raw_cols if c in _ho_cols]
                if _raw_needed:
                    logger.info("[D.2] Iniciando Validación Adversaria Univariable (Covariate Shift)...")
                    _ho_df = pd.read_parquet(_holdout_path, columns=_raw_needed)
                    logger.info("[D.2] Aplicando Lags OOS al Holdout para el cruce adversario...")
                    # FIX-ADV-ARCH-03: Utilizar self.lag_disc.optimal_lags para prevenir re-cálculo de lags asimétricos (Lag 1H) en OOS
                    _ho_lagged = self.lag_disc.transform(_ho_df[[c for c in repr_features if c in _ho_df.columns]], y=pd.Series(dtype=float), lag_cache=self.lag_disc.optimal_lags)
                    
                    adv_val = AdversarialValidator()
                    _cands_adv = [c for c in X_lagged.columns if c not in alpha_cols]

                    
                    from joblib import Parallel, delayed
                    
                    def eval_adv(f):
                        if f in _ho_lagged.columns:
                            pen, auc = adv_val.evaluate_feature(X_lagged[f], _ho_lagged[f])
                            return f, pen, auc
                        return None
                    
                    results = Parallel(n_jobs=-1, batch_size=10)(
                        delayed(eval_adv)(f) for f in _cands_adv
                    )
                    
                    for res in results:
                        if res is not None:
                            f, pen, auc = res
                            adv_penalties[f] = pen
                            adv_aucs[f] = auc
                    
                    _pen_list = [(k, v, adv_aucs[k]) for k, v in adv_penalties.items() if v < 0.99]
                    _pen_list.sort(key=lambda x: x[1])
                    if _pen_list:
                        logger.warning(f"[D.2] ðŸš¨ Detectadas {len(_pen_list)} Features mutantes. Top Sancionadas:")
                        for k, v, a in _pen_list[:10]:
                            logger.warning(f"      {k:<30} | AUC: {a:.3f} -> Penalty: {v:.2f}x")
                    else:
                        logger.info("[D.2] âœ… Ninguna feature superÃ³ el umbral de mutaciÃ³n. Todo limpio.")
                else:
                    logger.warning("[D.2] No se pudieron cruzar columnas con el holdout.")
            else:
                logger.warning("[D.2] features_holdout.parquet no encontrado. Se omite.")
                
            self._save_checkpoint("D2", {"penalties": adv_penalties, "aucs": adv_aucs})

        # â”€â”€ Etapa D: SFI-CPCV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        chk_D = self._load_checkpoint("D") if resume else None
        if chk_D:
            sfi_passed = chk_D["passed"]
            self.sfi.scores = chk_D["scores"]
            self.sfi.selected = sfi_passed
            logger.info(f"[D] Retomando desde checkpoint: {len(sfi_passed)} features")
        else:
            logger.info(f"[D] Evaluando {len(X_lagged.columns)} features con "
                        f"XGBoost SFI-CPCV (n_groups={SFI_N_GROUPS}, "
                        f"purge/embargo={SFI_PURGE_H}H)...")
            # FIX-ADV-ARCH-01 (2026-05-15): D.2 se desacopla del feature selection.
            # adv_penalties ya NO se pasa a sfi.evaluate() para evitar indirect look-ahead:
            # el holdout futuro NO debe influir en el ranking de poder predictivo.
            #
            # En su lugar: aplicar hard-block PRE-SFI para features con AUC>HARD_BLOCK_AUC.
            # Esto es transparente (documentado como exclusion de auditoria) y causal
            # (se elimina la feature del pool ANTES de evaluar, no se penaliza su Sharpe).
            _hard_blocked = {f for f, pen in adv_penalties.items() if pen == 0.0}
            if _hard_blocked:
                logger.warning(
                    f"[FIX-ADV-ARCH-01] Hard-block PRE-SFI: {len(_hard_blocked)} features "
                    f"excluidas del pool por Covariate Shift severo (AUC>"
                    f"{AdversarialValidator.HARD_BLOCK_AUC:.2f}): {sorted(_hard_blocked)}"
                )
                
                # [FIX-WHITELIST-01] Rescate de características con estacionariedad analítica
                _critical_features = {"close_fd", "hmm_regime", "hmm_velocity_bull", "hmm_acceleration_bull", "volatility_fd"}
                _blocked_criticals = _critical_features.intersection(_hard_blocked)
                if _blocked_criticals:
                    _rescue_msg = f"[FIX-WHITELIST-01] Rescatando {_blocked_criticals} del bloqueo Covariate Shift (estacionariedad analitica garantizada)."
                    logger.success(_rescue_msg)
                    print(_rescue_msg)
                    _hard_blocked = _hard_blocked - _critical_features
                    
                X_lagged_sfi = X_lagged.drop(columns=[c for c in _hard_blocked if c in X_lagged.columns])
            else:
                X_lagged_sfi = X_lagged
            # D.2 es solo auditoria â€” no afecta el scoring DSR:
            sfi_passed = self.sfi.evaluate(X_lagged_sfi, y, prices, adf_penalties, adv_penalties=None)
            self._save_checkpoint("D", {"passed": sfi_passed,
                                        "scores": self.sfi.scores})

        self.results["n_after_D"] = len(sfi_passed)

        # Separar alpha results
        alpha_passed   = [c for c in sfi_passed if c in ALPHA_SIGNALS]
        alpha_rejected = [c for c in alpha_cols if c not in sfi_passed]
        if alpha_rejected:
            logger.warning(f"[D] Alpha signals rechazadas en validación extrema: "
                           f"{alpha_rejected}")

        # ── Etapa E: Tribunal Multivariado Darwiniano (Time-Fold Variance Penalty) ──
        # MEJORA-SFI-ADAPTIVO: El SFI univariable (Fase D) es engañado por features fuertemente
        # cíclicas que funcionan aisladas pero causan PBO masivo al combinarse.
        # Se evalúa la importancia conjunta en 4 épocas cronológicas para purgar la inestabilidad.
        chk_E = self._load_checkpoint("E") if resume else None
        if chk_E:
            final_features = chk_E["optimal"]
            logger.info(f"[E] Retomando desde checkpoint: {len(final_features)} features")
        else:
            logger.info(f"[E] Iniciando Tribunal Multivariado (Time-Fold Variance) sobre {len(sfi_passed)} candidatos...")
            _candidates = sfi_passed[:]
            # Asegurar alineación
            _c_idx = X_lagged[_candidates].dropna(how="all").index.intersection(y.dropna().index)
            _X_mc = X_lagged.loc[_c_idx, _candidates]
            _y_mc = y.loc[_c_idx]

            # Dividir cronológicamente en 4 pliegues
            _n_folds = 4
            _fold_size = len(_X_mc) // _n_folds
            
            _imp_matrix = []
            
            if len(_X_mc) > 400 and len(_candidates) > 0:
                from xgboost import XGBClassifier
                
                logger.info(f"    Entrenando XGBoost Multivariable en {_n_folds} pliegues cronológicos (max_depth=3, n_estimators=40)...")
                for _f in range(_n_folds):
                    start_i = _f * _fold_size
                    end_i = (_f + 1) * _fold_size if _f < _n_folds - 1 else len(_X_mc)
                    X_f = _X_mc.iloc[start_i:end_i]
                    y_f = _y_mc.iloc[start_i:end_i]
                    
                    if len(np.unique(y_f)) < 2:
                        continue  # Skip fold if monotonic (extreme case)
                        
                    _m = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, random_state=_LUNA_SEED + _f, n_jobs=1)
                    _m.fit(X_f.values, y_f.values)
                    
                    # Normalizar importancias del fold para que sumen 1
                    _imps = _m.feature_importances_
                    if _imps.sum() > 0:
                        _imps = _imps / _imps.sum()
                    _imp_matrix.append(_imps)
                
                if len(_imp_matrix) > 1:
                    _imp_matrix = np.array(_imp_matrix)
                    _mean_imp = np.mean(_imp_matrix, axis=0)
                    _std_imp = np.std(_imp_matrix, axis=0)
                    
                    # Score Darwiniano: Importancia Media penalizada fuertemente por su Varianza CronolÃ³gica
                    # FIX-ADV-ARCH-01: _adv_mults eliminado del scoring Darwiniano.
                    # adv_penalties se conserva abajo solo para logging de auditoria (AdvPen display).
                    _adf_mults = np.array([adf_penalties.get(c, 1.0) for c in _candidates])
                    
                    _darwin_scores = (_mean_imp / (1.0 + _std_imp * 5.0)) * _adf_mults  # FIX-ADV-ARCH-01: _adv_mults eliminado (no usar holdout en scoring Darwiniano)
                    
                    _feat_scores = [(cand, score, m, s, adv_penalties.get(cand, 1.0), adf_penalties.get(cand, 1.0)) for cand, score, m, s in zip(_candidates, _darwin_scores, _mean_imp, _std_imp)]
                    _feat_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    logger.info("    Top Features por Robustez CronolÃ³gica, Estructural y Estacionaria (Darwin Score):")
                    for cand, sc, m, s, advp, adfp in _feat_scores[:10]:
                        logger.info(f"      {cand[:30]:<30} | Score: {sc:.4f} (Mean: {m:.4f}, Std: {s:.4f}, AdvPen: {advp:.2f}x, ADFPen: {adfp:.2f}x)")
                        
                    # Retener el Top N
                    _target_features = min(SFI_TOP_N_FEATURES, len(_candidates))
                    _raw_top = [x[0] for x in _feat_scores[:max(_target_features*2, 20)]]
                    
                    # DeduplicaciÃ³n estricta Nominal y Correlacional (Pre-Flight TEST-49 Guard)
                    _dedup = []
                    _seen_bases = set()
                    
                    for f in _raw_top:
                        if len(_dedup) >= _target_features:
                            break
                        
                        # Sacar el nombre base ignorando sufijos de lag
                        _base = f.rsplit('_milag', 1)[0]
                        if _base in _seen_bases:
                            logger.info(f"    [PBO-GUARD] Descartando {f} por redundancia nominal (versiÃ³n base ya seleccionada)")
                            continue
                            
                        if not _dedup:
                            _dedup.append(f)
                            _seen_bases.add(_base)
                            continue
                        
                        # Filtro por correlaciÃ³n dura
                        _c = _X_mc[_dedup + [f]].corr(method='pearson').abs().iloc[:-1, -1]
                        _max_corr = float(_c.max()) if not _c.isna().all() else 0.0
                        
                        if _max_corr < 0.95:
                            _dedup.append(f)
                            _seen_bases.add(_base)
                        else:
                            logger.info(f"    [PBO-GUARD] Descartando {f} por redundancia (corr={_max_corr:.3f})")
                            
                    final_features = _dedup

                    # [FIX-PBOGRD-QUALITY-01 2026-05-30] Gate de calidad antes de forzar alpha al pool
                    # CAUSA RAIZ H-SFI-15: la version anterior de este loop NO tenia gate de calidad,
                    # el bloque chk_E nuevo reemplazaba el bloque else pero el PBO-GUARD interno
                    # del Tribunal seguia siendo el viejo. Este es el fix correcto.
                    _PBO_MIN_MEAN_SR   = 0.05   # MeanSR minimo para forzar alpha al pool
                    _PBO_MAX_CONST_R   = 0.40   # ratio maximo de folds constantes tolerados
                    for a_col in alpha_cols:
                        if a_col in sfi_passed and a_col not in final_features:
                            _a_scores = self.sfi.scores.get(a_col, {})
                            _a_mean_sr = _a_scores.get("mean_sharpe", 0.0)
                            # Calcular constant_ratio de forma robusta: usar clave directa
                            # y fallback inline desde n_constant_folds/n_folds por si el
                            # checkpoint no serializo la clave (H-SFI-15 root cause)
                            _a_const_ratio = _a_scores.get("constant_folds_ratio", None)
                            if _a_const_ratio is None:
                                _n_c = _a_scores.get("n_constant_folds", 0)
                                _n_f = max(_a_scores.get("n_folds", 1), 1)
                                _a_const_ratio = _n_c / _n_f
                            print(f"[FIX-PBOGRD-QUALITY-01] PBO-GUARD gate: '{a_col}' | MeanSR={_a_mean_sr:.3f} | const_ratio={_a_const_ratio:.2f}")
                            if _a_mean_sr < _PBO_MIN_MEAN_SR:
                                print(f"[FIX-PBOGRD-QUALITY-01] Alpha '{a_col}' BLOQUEADA: MeanSR={_a_mean_sr:.3f} < {_PBO_MIN_MEAN_SR} (umbral minimo).")
                                logger.warning(f"[FIX-PBOGRD-QUALITY-01] Alpha '{a_col}' bloqueada: MeanSR insuficiente ({_a_mean_sr:.3f}).")
                                continue
                            if _a_const_ratio > _PBO_MAX_CONST_R:
                                print(f"[FIX-PBOGRD-QUALITY-01] Alpha '{a_col}' BLOQUEADA: const_ratio={_a_const_ratio:.2f} > {_PBO_MAX_CONST_R} (feature degenerada).")
                                logger.warning(f"[FIX-PBOGRD-QUALITY-01] Alpha '{a_col}' bloqueada: {_a_const_ratio:.0%} folds constantes.")
                                continue
                            _base_alpha = a_col.rsplit('_milag', 1)[0]
                            if _base_alpha not in _seen_bases:
                                final_features.append(a_col)
                                _seen_bases.add(_base_alpha)
                                logger.info(f"    [PBO-GUARD] Forzando Alpha Signal al pool final: {a_col}")

                            
                else:
                    logger.warning("[E] Fallo en K-Fold, usando fallback (todas las SFI passed)")
                    final_features = sfi_passed[:SFI_TOP_N_FEATURES]
            else:
                logger.warning("[E] Datos insuficientes para Tribunal K-Fold, usando fallback.")
                final_features = sfi_passed[:SFI_TOP_N_FEATURES]
                
            self._save_checkpoint("E", {"optimal": final_features, "history": []})

        # ── [SFI-BALANCE-01 2026-06-03] Cuotas mínimas por categoría (3 checks independientes) ──
        # Cada categoría tiene su propia whitelist y cuota mínima de slots.
        # El mecanismo es idéntico para las 3: si hay déficit en final_features,
        # se swapea el tail (menor rango) por candidatos de esa categoría en sfi_passed.
        # Total garantizado: macro(3) + onchain(1) + calendar(1) = 5/13 slots estructurales.
        # Los 8 slots restantes compiten libremente por DSR/ICIR.

        def _apply_category_quota(
            feats: list, passed: list, whitelist: set, min_slots: int, cat_name: str
        ) -> list:
            """Aplica cuota mínima de una categoría a la lista de features seleccionadas.
            Swapea tail de feats por candidatos de whitelist si hay déficit.
            Devuelve la lista actualizada.
            """
            if min_slots <= 0 or not whitelist:
                return feats

            def _in_cat(name):
                base = name.split('_milag')[0] if '_milag' in name else name
                if name in whitelist or base in whitelist:
                    return True
                # [FIX-SFI-BALANCE-02] Búsqueda flexible por substring para reconocer 
                # features derivadas (ej. 'M2_YoY_Chg_z90d_milag72h' coincidirá con 'M2_YoY_Chg')
                for w in whitelist:
                    if w in name:
                        return True
                return False

            in_final  = [f for f in feats if _in_cat(f)]
            deficit   = min_slots - len(in_final)
            print(
                f"[SFI-BALANCE-01] Cuota {cat_name}: "
                f"{len(in_final)}/{min_slots} slots | déficit={max(0,deficit)}"
            )
            if deficit <= 0:
                print(f"[SFI-BALANCE-01] Cuota {cat_name} OK — sin cambios.")
                return feats

            candidates  = [f for f in passed if _in_cat(f) and f not in feats]
            non_cat_tail = [f for f in reversed(feats) if not _in_cat(f)]
            swapped_in, swapped_out = [], []
            for i in range(min(deficit, len(candidates), len(non_cat_tail))):
                _in, _out = candidates[i], non_cat_tail[i]
                feats.remove(_out)
                feats.append(_in)
                swapped_in.append(_in)
                swapped_out.append(_out)
                logger.info(f"[SFI-BALANCE-01] {cat_name} SWAP: +{_in} / -{_out}")

            if swapped_in:
                print(
                    f"[SFI-BALANCE-01] Cuota {cat_name} aplicada: "
                    f"+{len(swapped_in)} → {swapped_in} | sacadas: {swapped_out}"
                )
            else:
                print(
                    f"[SFI-BALANCE-01] ADVERTENCIA {cat_name}: déficit={deficit} "
                    f"pero sin candidatos en sfi_passed. Cuota no satisfecha."
                )
                logger.warning(
                    f"[SFI-BALANCE-01] {cat_name} cuota no satisfecha — "
                    f"0 candidatos disponibles en sfi_passed."
                )
            return feats

        # Cargar las 3 whitelists desde settings (con fallback vacío = backward compatible)
        try:
            from config.settings import cfg as _cfg_quota
            _wl_macro    = set(getattr(_cfg_quota.features, 'sfi_macro_features',    []) or [])
            _wl_onchain  = set(getattr(_cfg_quota.features, 'sfi_onchain_features',  []) or [])
            _wl_calendar = set(getattr(_cfg_quota.features, 'sfi_calendar_features', []) or [])
        except Exception as _eq:
            logger.warning(f"[SFI-BALANCE-01] No se pudieron cargar whitelists de settings: {_eq}")
            _wl_macro = _wl_onchain = _wl_calendar = set()

        # Aplicar las 3 cuotas en orden: macro → onchain → calendar
        # El orden importa: macro va primero porque tiene mayor cuota (3 slots).
        final_features = _apply_category_quota(
            final_features, sfi_passed, _wl_macro,    SFI_MACRO_MIN_SLOTS,    "MACRO"
        )
        final_features = _apply_category_quota(
            final_features, sfi_passed, _wl_onchain,  SFI_ONCHAIN_MIN_SLOTS,  "ONCHAIN"
        )
        final_features = _apply_category_quota(
            final_features, sfi_passed, _wl_calendar, SFI_CALENDAR_MIN_SLOTS, "CALENDAR"
        )

        # BUG-LAG-NAMING-01 FIX (2026-05-05)
        # LagDiscoveryTransformer aplica shift(opt_lag) pero el JSON
        # guardaba el nombre BASE sin sufijo _milagNh cuando no hay
        # Granger-lag conocido. Fix: renombrar antes de _write_output.
        _lag_disc_map = getattr(getattr(self, "lag_disc", None), "optimal_lags", {})
        _final_renamed = []
        for _ff in final_features:
            _opt_lag = _lag_disc_map.get(_ff, 0)
            if _opt_lag > 0 and "_milag" not in _ff:
                _new_name = "{}_milag{}h".format(_ff, _opt_lag)
                logger.debug("[BUG-LAG-NAMING-01] {} -> {} (lag in parquet)".format(_ff, _new_name))
                _final_renamed.append(_new_name)
            else:
                _final_renamed.append(_ff)
        final_features = _final_renamed

        # ══════════════════════════════════════════════════════════════════════
        # [GUARDIAN-10] SFI Feature Starvation Guardian
        # ══════════════════════════════════════════════════════════════════════
        try:
            from config.settings import cfg as _cfg_g
            _sfi_min_features = int(_cfg_g.stat.sfi_min_features)
        except Exception as e:
            logger.warning(f"[CRITICAL-SOP] Falta stat.sfi_min_features en settings.yaml, usando fallback 5: {e}")
            _sfi_min_features = 5
            
        if len(final_features) < _sfi_min_features:
            logger.error(
                f"[GUARDIAN-10] Feature Starvation DETECTADO: SFI seleccionó solo {len(final_features)} "
                f"features (mínimo exigido: {_sfi_min_features}). El mercado es puro ruido y purgó "
                f"todas las variables. El modelo no convergerá. Abortando."
            )
            print(f"[GUARDIAN-10] FATAL: SFI Starvation. {len(final_features)} features < {_sfi_min_features}. Abortando.")
            import sys
            sys.exit(3)

        # ── Output ──────────────────────────────────────────────────────────
        
        # Resumen final de distribución para trazabilidad en logs
        _all_wl = _wl_macro | _wl_onchain | _wl_calendar
        def _cat(f):
            b = f.split('_milag')[0] if '_milag' in f else f
            if f in _wl_macro    or b in _wl_macro:    return 'macro'
            if f in _wl_onchain  or b in _wl_onchain:  return 'onchain'
            if f in _wl_calendar or b in _wl_calendar: return 'calendar'
            # [FIX-SFI-BALANCE-02] Coincidencia por substring para el log de distribución
            for w in _wl_macro:
                if w in f: return 'macro'
            for w in _wl_onchain:
                if w in f: return 'onchain'
            for w in _wl_calendar:
                if w in f: return 'calendar'
            return 'libre'
        _dist = {c: sum(1 for f in final_features if _cat(f)==c) for c in ['macro','onchain','calendar','libre']}
        print(
            f"[SFI-BALANCE-01] Distribución final ({len(final_features)} features): "
            f"macro={_dist['macro']} onchain={_dist['onchain']} "
            f"calendar={_dist['calendar']} libre={_dist['libre']}"
        )
        # ── Fin [SFI-BALANCE-01] ─────────────────────────────────────────────────────

        # -- Pass-Through: Detectar reglas disponibles en df con varianza suficiente

        # FIX-PASSTHROUGH-VAR-01 (2026-03-21): Antes se inyectaban todas las reglas
        # de PASSTHROUGH_FEATURES que existian en df. Ahora se exige varianza minima:
        # min_hits = max(10, int(n_rows * 0.001)) = 0.1% de activaciones minimas.
        # Reglas con 0-9 hits en IS son efectivamente constantes -> noise en OOS.
        _n_rows_pt = max(len(df), 1)
        _min_hits_pt = max(10, int(_n_rows_pt * 0.001))
        
        # [NEW LOGIC] Exigir varianza en los Ãºltimos 180 dÃ­as (4320 horas)
        # Si una regla estuvo activa hace 4 aÃ±os pero lleva 6 meses congelada, es tÃ³xica para el WFB actual.
        _tail_len = min(_n_rows_pt, 4320)

        passthrough_available = []
        passthrough_rejected_var = []

        for _pt_f in PASSTHROUGH_FEATURES:
            if _pt_f not in df.columns:
                continue
            if _pt_f in final_features:
                continue  # ya seleccionada por SFI, no duplicar
            try:
                _pt_hits_total = int(df[_pt_f].sum())
                _pt_tail = df[_pt_f].iloc[-_tail_len:]
                _pt_hits_tail = int(_pt_tail.sum())
                # Regla congelada temporalmente si tiene 0 hits o 100% hits en los Ãºltimos 180 dÃ­as
                _is_frozen_recently = (_pt_hits_tail == 0) or (_pt_hits_tail == _tail_len)
            except Exception:
                _pt_hits_total = 0
                _is_frozen_recently = True
                
            if _pt_hits_total >= _min_hits_pt and not _is_frozen_recently:
                passthrough_available.append(_pt_f)
            else:
                passthrough_rejected_var.append((_pt_f, _pt_hits_total))

        if passthrough_rejected_var:
            logger.warning(
                f"[PT] FIX-PASSTHROUGH-VAR-01: {len(passthrough_rejected_var)} reglas descartadas "
                f"(min_hits={_min_hits_pt}): {passthrough_rejected_var[:10]}"
            )
        if passthrough_available:
            logger.info(
                f"[PT] {len(passthrough_available)} features pass-through anyadidas al output (saltan SFI): {passthrough_available}"
            )
        else:
            logger.warning("[PT] No se encontraron pass-through features con varianza suficiente.")

        # BUG-LAG-NAMING-01 FIX (2026-05-05)
        # LagDiscoveryTransformer aplica shift(opt_lag) pero el JSON
        # guardaba el nombre BASE sin sufijo _milagNh cuando no hay
        # Granger-lag conocido. Fix: renombrar antes de _write_output.
        _lag_disc_map = getattr(getattr(self, "lag_disc", None), "optimal_lags", {})
        _final_renamed = []
        for _ff in final_features:
            _opt_lag = _lag_disc_map.get(_ff, 0)
            if _opt_lag > 0 and "_milag" not in _ff:
                _new_name = "{}_milag{}h".format(_ff, _opt_lag)
                logger.debug("[BUG-LAG-NAMING-01] {} -> {} (lag in parquet)".format(_ff, _new_name))
                _final_renamed.append(_new_name)
            else:
                _final_renamed.append(_ff)
        final_features = _final_renamed

        # Ã¢ââ¬Ã¢ââ¬ Output Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬Ã¢ââ¬
        self._write_output(
            selected=final_features,
            alpha_passed=alpha_passed,
            alpha_rejected=alpha_rejected,
            passthrough=passthrough_available,
            stage_summaries={
                "B": {"n_clusters": len(repr_features),
                      "n_input": self.results.get("n_input", self.results.get("n_after_clustering", 0))},
                "C": {"n_final": self.results["n_after_C"],
                      "optimal_lags": self.lag_disc.optimal_lags},
                "D": {"n_passed": len(sfi_passed),
                      "top_n_contextuales": SFI_TOP_N_FEATURES},
                "E": {"n_final": len(final_features),
                      "selection_history": self.forward.history},
            }
        )

        logger.info("=" * 65)
        logger.success(f"â Pipeline completo: {self.results.get('n_input', self.results.get('n_after_clustering', '?'))} â {len(final_features)} features SFI "
                       f"+ {len(passthrough_available)} pass-through")
        logger.info(f"   Features SFI: {final_features}")
        logger.info(f"   Pass-Through: {passthrough_available}")
        logger.info(f"   Output: {OUTPUT_FILE}")
        logger.info("=" * 65)

        return {"selected_features": final_features,
                "stage_results": self.results}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Feature Selection Pipeline LdP 5 Etapas â Luna V1"
    )
    parser.add_argument("--parquet", type=str, default=None,
                        help="Ruta al parquet de features (default: features_train.parquet)")
    parser.add_argument("--resume", action="store_true",
                        help="Retomar desde el checkpoint mÃ¡s avanzado")
    parser.add_argument("--mode", type=str, default="dev", choices=["dev", "prod"],
                        help="Modo de ejecuciÃ³n: dev (trainâ¤2023-12-31) o prod (todos los datos)")
    args = parser.parse_args()

    from loguru import logger as _log
    import os as _os
    import sys

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _run_id_suffix = _os.environ.get("LUNA_RUN_ID", "")
    _log_name = f"feature_selection_{ts}_{_run_id_suffix}.log" if _run_id_suffix else f"feature_selection_{ts}.log"
    _log.add(log_dir / _log_name, rotation="50 MB", level="DEBUG")

    try:
        pipeline = FeatureSelectionPipelineE()
        result = pipeline.run(
            features_parquet=Path(args.parquet) if args.parquet else None,
            resume=args.resume
        )
        print(f"\n[OK] Seleccionadas: {len(result['selected_features'])} features")
        print(f"   {result['selected_features']}")
        sys.exit(0)
    except Exception as e:
        import traceback
        _log.error(f"[FATAL UNCAUGHT] Script crashed at main level: {e}")
        _log.debug(traceback.format_exc())
        sys.exit(1)
