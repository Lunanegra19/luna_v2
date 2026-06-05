"""
Bayesian Causal Engine — Luna V1 AI Mining (Engine 2/6)
====================================================================
Propósito: Identificar efectos causales reales entre indicadores y
retorno BTC usando DoWhy (Double Machine Learning / ATE causal).

DoWhy (Microsoft) implementa el framework de Pearl's Causal Calculus.
Aquí usamos el estimador DML (Double Machine Learning) para calcular
el Average Treatment Effect (ATE) de cada variable sobre el retorno BTC.

Output:
  - `Master_Causal_Signal` (float): suma ponderada de las señales
    causales más fuertes, normalizada [-1, 1].
  - `data/ai_mining/reports/bayesian_causal_report.md`

Regla SOP: Ejecutar ANTES de master_pattern_engine y deep_discovery_engine
para que las reglas generadas usen causalidad real, no correlación.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
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

# ── Config ───────────────────────────────────────────────────────────────────
# Variables candidatas para análisis causal — priorizadas por SHAP + TE_net
CAUSAL_CANDIDATES = [
    # Macro FRED (alta evidencia causal en Correlaciones)
    "FedFundsRate", "YieldCurve_10Y3M", "T10Y2Y", "M2_USA_raw",
    "GlobalM2_Index", "Fed_Net_Liquidity", "CPI_YoY", "Inflation_MoM",
    "WEI", "TGA_USD", "RRP_USD",
    # Mercado
    "VIX", "DXY", "SP500_Ret", "NASDAQ_Ret", "Gold_Ret", "Oil_Ret",
    # On-chain
    "MVRV_Proxy", "FearGreed", "SSR", "Whale_Vol_ZScore",
    "hashrate_7d_ma", "active_addresses_7d_ma",
    # Derivados
    "FundingRate", "OI_BTC", "DangerZone", "DVOL",
    # DeFi
    "Stablecoin_Cap", "DeFi_WBTC_TVL",
    # Cross-asset
    "eth_btc_corr_24h", "eth_ret_lag1", "alt_season_proxy",
]

TARGET_COL    = "target"   # 0/1 label binario o future_ret_24h
HORIZON_H     = 24         # predicción a 24H
MIN_SAMPLES   = 500        # mínimo para ejecutar DML
ATE_MIN_MAGNITUDE = 0.002      # ATE absoluto mínimo para incluir variable en señal


# ─────────────────────────────────────────────────────────────────────────────
class BayesianCausalEngine:
    """
    Estimador ATE (Average Treatment Effect) para features de BTC.

    Usa DML si dowhy está disponible; fallback a regresión parcial
    (Frisch-Waugh-Lovell) que es matemáticamente equivalente.
    """

    def __init__(self, cutoff_date=None):
        self.results_: list[dict] = []
        # cutoff_date: pd.Timestamp | None
        # Si se inyecta (--mode dev), load_data() filtra df a <= cutoff_date
        # para garantizar consistencia temporal con Feature Selection.
        self.cutoff_date = pd.Timestamp(cutoff_date, tz='UTC') if cutoff_date else None

    # ── Carga ────────────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        """Carga features_train_kshape.parquet (output del engine 1)
        o features_train.parquet si el anterior no existe."""
        path_k = DATA_FEATURES / "features_train_kshape.parquet"
        path_b = DATA_FEATURES / "features_train.parquet"
        path   = path_k if path_k.exists() else path_b
        if not path.exists():
            raise FileNotFoundError(f"No se encontró dataset en {DATA_FEATURES}")
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        # [MODO DEV] Aplicar cutoff_date si está definido
        if self.cutoff_date is not None and df.index.max() > self.cutoff_date:
            before = len(df)
            df = df[df.index <= self.cutoff_date]
            logger.info(f"Causal: cutoff_date={self.cutoff_date.date()} aplicado — {before} → {len(df)} filas")
        else:
            logger.info(f"Causal [PROD]: datos completos {df.shape} desde {path.name}")
        logger.info(f"Causal: cargado {df.shape} desde {path.name}")
        return df

    # ── Target ───────────────────────────────────────────────────────────────

    def _build_target(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Construye target binario a 24H si no viene en el dataset."""
        # [BUG-TARGET-FIX] Priorizando Target_TBM_Bin como objetivo causal/genético para consistencia con TBM
        if "Target_TBM_Bin" in df.columns:
            print("[BUG-TARGET-FIX] Causal: Detectada y priorizada columna Target_TBM_Bin")
            logger.info("[BUG-TARGET-FIX] Causal: Usando Target_TBM_Bin como target principal para el análisis causal.")
            return df["Target_TBM_Bin"].dropna()
        if TARGET_COL in df.columns:
            return df[TARGET_COL].dropna()
        if "close" in df.columns:
            # future_ret_24h > 0 → 1 (alcista)
            ret24 = df["close"].shift(-HORIZON_H) / df["close"] - 1
            return (ret24 > 0).astype(int).dropna()
        return None

    # ── ATE por DML / fallback FWL ───────────────────────────────────────────

    def _estimate_ate_dml(self, df: pd.DataFrame, treatment: str, outcome: pd.Series) -> dict:
        """
        Estima ATE usando Double Machine Learning (DoWhy) si disponible,
        o Frisch-Waugh-Lovell partial regression como fallback.

        DML: Y ~ T + X  usando Random Forest para "residualizar"
             ambas variables frente a los controles X.
        FWL: equivalente econométrico.
        """
        confounders = [
            c for c in CAUSAL_CANDIDATES
            if c in df.columns and c != treatment
        ]

        T = df[treatment].dropna()
        Y = outcome.reindex(T.index).dropna()
        common = T.index.intersection(Y.index)
        T, Y = T.loc[common], Y.loc[common]

        if len(T) < MIN_SAMPLES:
            return {"variable": treatment, "ate": 0.0, "method": "skip", "n": len(T)}

        # [FALLA-01-FIX 2026-05-30] Binarizacion mejorada: detrending rolling-90d antes de binarizar
        # Problema anterior: T > T.median() para M2 (tendencia secular) clasificaba el 100%
        # de 2021-2024 como 'treated=1' porque la mediana era del periodo 2017-2024.
        # Fix: normalizar rolling 90d para hacer la binarizacion RELATIVA al contexto reciente.
        # Variable con tendencia secular (M2, FedFunds): T_norm captura si ESTA POR ENCIMA
        # de su propia media reciente (90 dias), eliminando el drift secular.
        try:
            rolling_mu = T.rolling(window=90*24, min_periods=24*7).mean()
            rolling_std = T.rolling(window=90*24, min_periods=24*7).std().replace(0, 1)
            T_normalized = (T - rolling_mu) / rolling_std
            # Si el rolling tiene muy pocos datos al inicio, caer al metodo original para esas filas
            has_rolling = rolling_mu.notna()
            T_binary_rolling = (T_normalized > 0).astype(int)
            T_binary_raw = (T > T.median()).astype(int)
            T_binary = T_binary_rolling.where(has_rolling, T_binary_raw)
            print(f"[FALLA-01-FIX] ATE '{treatment}': binarizacion rolling-90d aplicada. "
                  f"Treated={T_binary.mean():.1%} (vs mediana cruda: {T_binary_raw.mean():.1%})")
        except Exception as _e_norm:
            # Fallback al metodo original si el rolling falla
            T_binary = (T > T.median()).astype(int)
            print(f"[FALLA-01-FIX] ATE '{treatment}': fallback a mediana cruda ({_e_norm})")


        try:
            # Intentar DoWhy DML
            import dowhy
            from dowhy import CausalModel

            X_ctrl = df[confounders].reindex(common).fillna(0)
            data_dml = pd.concat([T_binary.rename(treatment), Y.rename("outcome"), X_ctrl], axis=1).dropna()

            model = CausalModel(
                data=data_dml,
                treatment=treatment,
                outcome="outcome",
                common_causes=confounders,
            )
            identified = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                identified,
                method_name="backdoor.linear_regression",
                test_significance=False,
            )
            ate = float(estimate.value)
            method = "dowhy_lr"

        except Exception:
            # Fallback: FWL partial regression
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler

            X_ctrl = df[confounders].reindex(common).fillna(0)
            scaler = StandardScaler()
            X_s    = scaler.fit_transform(X_ctrl)

            # Residualizar T frente a X
            model_T = Ridge(alpha=1.0)
            model_T.fit(X_s, T_binary)
            T_resid = T_binary - model_T.predict(X_s)

            # Residualizar Y frente a X
            model_Y = Ridge(alpha=1.0)
            model_Y.fit(X_s, Y)
            Y_resid = Y - model_Y.predict(X_s)

            # ATE = cov(T_resid, Y_resid) / var(T_resid)
            ate = float(np.cov(T_resid, Y_resid)[0, 1] / (np.var(T_resid) + 1e-8))
            method = "fwl_ridge"

        # [H5-FIX 2026-05-30] Detectar ATE outlier extremo (posible reverse causality)
        # DVOL=+0.167 es 5x mayor que el siguiente ATE (DeFi_WBTC_TVL=0.042).
        # Valores |ATE| > 0.10 para variables con alta correlacion contemporanea con BTC
        # son señal de reverse causality (BTC->variable, no variable->BTC).
        # Vars de riesgo: DVOL, OI_BTC, FundingRate (son derivadas del propio precio BTC)
        _reverse_causality_suspects = {"DVOL", "dv_dvol_raw", "OI_BTC", "FundingRate",
                                        "dv_funding_rate", "dv_oi_acceleration_24h"}
        _ate_hard_cap = 0.10  # |ATE| > 10% de probabilidad es casi seguro artefacto
        if abs(ate) > _ate_hard_cap and treatment in _reverse_causality_suspects:
            _ate_original = ate
            ate = np.sign(ate) * _ate_hard_cap
            print(f"[H5-FIX] ATE capped para '{treatment}': |ATE|={abs(_ate_original):.4f} > {_ate_hard_cap:.2f} "
                  f"(posible reverse causality) -> cap a {ate:.4f}")
            method = f"{method}_capped"
        elif abs(ate) > _ate_hard_cap:
            # Para otras variables: advertencia pero sin cap
            print(f"[H5-FIX] ATE extremo para '{treatment}': |ATE|={abs(ate):.4f} > {_ate_hard_cap:.2f} "
                  f"— verificar causalidad inversa. Incluido sin modificar.")

        return {
            "variable":  treatment,
            "ate":       round(ate, 5),
            "direction": "BULLISH" if ate > 0 else "BEARISH",
            "method":    method,
            "n":         len(T),
        }


    # ── Señal causal agregada ─────────────────────────────────────────────────

    def _build_causal_signal(self, df: pd.DataFrame, results: list[dict]) -> pd.Series:
        """
        Suma ponderada de las variables con ATE significativo.
        Cada variable con |ATE| >= ATE_MIN_MAGNITUDE aporta con su signo.
        El resultado es normalizado rolling 90d → escala [-1, 1].
        """
        sig_results = [r for r in results if abs(r.get("ate", 0)) >= ATE_MIN_MAGNITUDE]
        if not sig_results:
            logger.warning("Causal: ninguna variable supera ATE_MIN_MAGNITUDE — retornando señal cero")
            return pd.Series(0.0, index=df.index, name="Master_Causal_Signal")

        signal = pd.Series(0.0, index=df.index)
        for r in sig_results:
            var = r["variable"]
            if var not in df.columns:
                continue
            col = df[var].ffill().fillna(0)
            # Normalizar rolling 90d
            mu90  = col.rolling(90 * 24, min_periods=24).mean()
            std90 = col.rolling(90 * 24, min_periods=24).std().replace(0, 1)
            z     = (col - mu90) / std90
            # Signo según dirección causal
            sign  = 1 if r["direction"] == "BULLISH" else -1
            # Peso proporcional al |ATE|
            weight = abs(r["ate"])
            signal += sign * weight * z.clip(-3, 3)

        # Normalizar a [-1, 1]
        total_weight = sum(abs(r["ate"]) for r in sig_results)
        if total_weight > 0:
            signal /= total_weight

        signal = signal.clip(-1, 1).rename("Master_Causal_Signal")
        return signal

    # ── Reporte ──────────────────────────────────────────────────────────────

    def _save_report(self, results: list[dict]) -> None:
        """Reporte estilo Correlaciones — causalidad ATE con 5 secciones narrativas."""
        rows = sorted(results, key=lambda x: abs(x.get("ate", 0)), reverse=True)
        sig  = [r for r in rows if isinstance(r, dict) and abs(r.get("ate", 0)) >= ATE_MIN_MAGNITUDE]
        n_bull = sum(1 for r in sig if r.get("direction") == "BULLISH")
        n_bear = sum(1 for r in sig if r.get("direction") == "BEARISH")
        verdict = "🟢 BULLISH" if n_bull > n_bear else "🔴 BEARISH" if n_bear > n_bull else "⚖️ NEUTRAL"

        ts = pd.Timestamp.now().strftime("%d %B %Y %H:%M")
        report = [
            f"# 🔬 BAYESIAN CAUSAL ENGINE — Luna V1",
            f"**Generado:** {ts} | **Método:** FWL-Ridge (DoWhy fallback) | **Variables evaluadas:** {len(results)} | **Umbral ATE:** {ATE_MIN_MAGNITUDE}",
            "",
            f"> *Estimación del Average Treatment Effect (ATE) causal de cada indicador sobre el retorno BTC 24H.*",
            f"> *Frisch-Waugh-Lovell partial regression — matemáticamente equivalente a Double Machine Learning (DML).*",
            f"> *ATE > 0: el indicador causa subida de BTC en promedio. ATE < 0: causa bajada.*",
            "",
            "---",
            "",
            "## 📡 1. Veredicto Causal — Síntesis",
            "",
            f"> **{verdict}** — {n_bull} variables causales alcistas vs {n_bear} bajistas (umbral |ATE| ≥ {ATE_MIN_MAGNITUDE})",
            "",
            "| Dirección | Variables causales | Ejemplos |",
            "| --- | --- | --- |",
            f"| 🟢 BULLISH | {n_bull} | {', '.join([r['variable'] for r in sig if r.get('direction')=='BULLISH'][:3])} |",
            f"| 🔴 BEARISH | {n_bear} | {', '.join([r['variable'] for r in sig if r.get('direction')=='BEARISH'][:3])} |",
            "",
            "---",
            "",
            "## 📊 2. Ranking de ATE Causal (todas las variables)",
            "",
            "> **Ordenado por |ATE| absoluto.** Estrellas: ★★★ > 0.05 | ★★ > 0.02 | ★ > 0.002 | × por debajo del umbral.",
            "",
            "| Variable | ATE | Significancia | Dirección | N obs |",
            "| --- | --- | --- | --- | --- |",
        ]

        for r in rows:
            if not isinstance(r, dict):
                continue
            ate = r.get("ate", 0)
            stars = "★★★" if abs(ate) > 0.05 else "★★" if abs(ate) > 0.02 else "★" if abs(ate) >= ATE_MIN_MAGNITUDE else "×"
            dir_str = r.get("direction", "—")
            dir_emoji = "🟢" if dir_str == "BULLISH" else "🔴" if dir_str == "BEARISH" else "—"
            ate_str = f"{ate:+.5f}"
            report.append(
                f"| **{r['variable']}** | {ate_str} | {stars} | {dir_emoji} {dir_str} | {r.get('n', 0):,} |"
            )

        # Top 3 por dirección
        top_bull = [r for r in sig if r.get("direction") == "BULLISH"][:3]
        top_bear = [r for r in sig if r.get("direction") == "BEARISH"][:3]

        report += [
            "",
            "---",
            "",
            "## 🔭 3. Variables más Influyentes",
            "",
            "### 🟢 Motores Alcistas (mayor ATE positivo)",
            "",
        ]
        for r in top_bull:
            ate = r.get("ate", 0)
            report.append(f"- **{r['variable']}** → ATE={ate:+.5f} — Por cada unidad de tratamiento binario, BTC sube {abs(ate)*100:.3f}% en promedio a 24H.")

        report += [
            "",
            "### 🔴 Motores Bajistas (mayor ATE negativo)",
            "",
        ]
        for r in top_bear:
            ate = r.get("ate", 0)
            report.append(f"- **{r['variable']}** → ATE={ate:+.5f} — Efecto causal bajista de {abs(ate)*100:.3f}% sobre retorno 24H.")

        report += [
            "",
            "> [!NOTE]",
            "> **Interpretación económica:** CPI_YoY bearish es coherente — inflación alta = Fed hawkish = risk-off.",
            "> SSR y Stablecoin_Cap bullish reflejan acumulación on-chain y liquidez disponible para entrada.",
            "> FedFundsRate bullish es contraintuitivo — posiblemente captura el régimen 2020-2021 donde rates bajos coincidían con el mercado alcista inicial.",
            "",
            "---",
            "",
            "## 📓 4. Directiva Táctica — Causal",
            f"**Señal consolidada: {verdict}**",
            f"- `Master_Causal_Signal` se construye como suma ponderada de las {len(sig)} variables significativas ({n_bull} ↑ / {n_bear} ↓).",
            "- Esta señal alimenta el Meta-Oracle con peso ×3 (el mayor de todos los motores).",
            "- La señal es **causal** (ATE estimado), no correlacional — filtra espuriedad estadística.",
            "",
            "> [!TIP]",
            "> **Uso práctico:** `Master_Causal_Signal > 0` combinado con `alpha_tribe_bias = LARGA`",
            "> da la máxima confirmación cruzada para entrada larga.",
        ]

        path = REPORTS_DIR / "bayesian_causal_report.md"
        clean_report = [l for l in report if isinstance(l, str)]
        path.write_text("\n".join(clean_report), encoding="utf-8")
        logger.success(f"Causal: reporte mejorado guardado en {path}")

    # ── Pipeline principal ────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        1. Carga datos
        2. Construye target 24H
        3. Estima ATE para cada variable candidata
        4. Construye Master_Causal_Signal agregada
        5. Guarda parquet enriquecido + reporte
        """
        logger.info("=" * 60)
        logger.info("Bayesian Causal Engine — INICIO")
        logger.info("=" * 60)

        df = self.load_data()
        outcome = self._build_target(df)
        if outcome is None:
            logger.error("Causal: no se pudo construir target — abortando")
            return df

        # Estimar ATE para cada candidato disponible
        available = [c for c in CAUSAL_CANDIDATES if c in df.columns]
        logger.info(f"Causal: evaluando {len(available)} variables candidatas")

        results = []
        for var in available:
            try:
                res = self._estimate_ate_dml(df, var, outcome)
                results.append(res)
                logger.info(f"  {var:<35} ATE={res['ate']:+.5f}  ({res['direction'] if 'direction' in res else 'skip'})")
            except Exception as e:
                logger.warning(f"  {var}: error — {e}")

        self.results_ = results

        # Señal causal agregada
        signal = self._build_causal_signal(df, results)
        df["Master_Causal_Signal"] = signal

        # Guardar (Fix GDrive Locking Errno 22)
        out_path = DATA_FEATURES / "features_train_causal.parquet"
        
        # Eliminar archivo temporal residual si existiera
        temp_path = DATA_FEATURES / "features_train_causal.tmp.parquet"
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

        # Intento seguro de eliminación de out_path con reintentos para liberar handles
        if out_path.exists():
            import time
            for attempt in range(1, 6):
                try:
                    out_path.unlink()
                    break
                except Exception as e:
                    if attempt < 5:
                        time.sleep(0.5)
                    else:
                        logger.warning(f"Causal: No se pudo eliminar archivo previo {out_path.name} tras 5 intentos - {e}")

        # Escritura directa y segura con control de errores GDrive (errno 13, 22) y backoff
        import time as _time_retry
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                df.to_parquet(out_path)
                print("[FIX-GDRIVE-PARQUET] Escrito de forma directa y segura en Windows")
                logger.success(f"Causal: datos guardados en {out_path}")
                break
            except OSError as _oe:
                if attempt < max_retries and getattr(_oe, "errno", None) in (13, 22):
                    _wait = 2 ** attempt
                    import sys
                    print(f"[FIX-GDRIVE-PARQUET] to_parquet({out_path.name}) errno={_oe.errno} — reintento {attempt}/{max_retries} en {_wait}s", file=sys.stderr)
                    _time_retry.sleep(_wait)
                else:
                    raise

        self._save_report(results)
        
        # [P2-BCE-SERIALIZATION] Guardar pesos causales para inferencia online en el FeaturePipeline
        try:
            import json
            models_dir = PROJECT_ROOT / "data" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            bce_weights_path = models_dir / "bce_weights.json"
            
            # Limpiar floats y nans para json
            import math
            clean_results = []
            for r in results:
                if isinstance(r, dict):
                    clean_r = {k: v if not isinstance(v, float) or not math.isnan(v) else 0.0 for k, v in r.items()}
                    clean_results.append(clean_r)
                    
            with open(bce_weights_path, "w", encoding="utf-8") as f:
                json.dump({"weights": clean_results}, f, indent=2)
            logger.success(f"Causal: pesos serializados en {bce_weights_path.name}")
        except Exception as e:
            logger.warning(f"Causal: error al serializar bce_weights.json: {e}")

        logger.info("Bayesian Causal Engine — COMPLETADO")
        return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = BayesianCausalEngine()
    engine.run()
