"""
feature_pipeline.py
===================
Luna V1 â€” Feature Engineering (NÃšCLEO)
Pipeline que transforma los datos raw (8 categorÃ­as) en features listas para el modelo.

ORDEN OBLIGATORIO de transformaciones (no alterar el orden):
1. Carga de las 8 categorÃ­as raw
2. Safety Lags por categorÃ­a (tabla TE validada â€” R1)
3. Merge en Ã­ndice 1H UTC (join por timestamp exacto)
4. Rolling Z-Score 90d en macro vars (R14)
5. MultiTimeframe features (15min intra-hora, 4H momentum)
6. Calendar features (FOMC, Deribit expiry, US session)
7. Cross-Asset features (ETH/BTC corr, ETH lead, BTC Dominance, DangerZone)
8. FracDiff dinÃ¡mico por ventana (R7)
9. alpha_rules.py import â†’ columnas binarias
10. Target dinÃ¡mico future_ret_24h (R17)
11. SFI Filter (RF OOB > 0.05 Sharpe)
12. Clustering de correlaciÃ³n (threshold 0.70)
13. Split Train/Val/Holdout â†’ guardar parquets

Reglas crÃ­ticas:
- R1:  Safety lags ANTES del merge
- R7:  FracDiff dinÃ¡mico por ventana
- R13: alpha_rules.py nativo â€” NUNCA JSON bridge
- R14: Rolling Z-Score 90d obligatorio
- R17: Target dinÃ¡mico â€” NUNCA features_labeled.parquet estÃ¡tico de V8.3
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

# AÃ±adir root al path
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from luna.utils.encoding_fix import fix_stdout_encoding; fix_stdout_encoding()

from config.settings             import cfg
from luna.data.data_collector    import DataCollector
from luna.features.rolling_normalization import RollingZScoreNormalizer, MACRO_COLUMNS
from luna.features.calendar_features     import CalendarFeatures
from luna.security.guard_pipeline        import purge_leakage_columns  # R1 Anti-Leakage
from luna.data.btc_supply                import BTCDynamicSupply       # BUG M-03: supply dinÃ¡mico con halvings

_FEATURES_DIR = _ROOT / "data" / "features"


class FeaturePipeline:
    """
    Pipeline principal de feature engineering de Luna V1.
    Produce los DataFrames train/validation/holdout listos para el modelo.
    """

    def __init__(self):
        self.collector   = DataCollector()
        self.normalizer  = RollingZScoreNormalizer()
        self.calendar_fe = CalendarFeatures()

    # â”€â”€ PASO 1+2+3: Carga + Safety Lags + Merge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def load_and_merge_raw(self) -> pd.DataFrame:
        """
        Carga las 8 categorÃ­as raw, aplica safety lags y hace merge en 1H UTC.

        Safety Lags (tabla validada por Transfer Entropy):
        - Macro (M2):            shift +42d  (R1 â€” correlaciÃ³n BTC-M2 documentada)
        - Macro (CPI/FFR/UNEM):  shift +14d
        - On-chain:              shift +24H
        - DeFi:                  shift +24H
        - Altcoins (ETH):        shift +13d
        - Derivatives:           shift 0 (datos en tiempo real de mercado)
        - Mempool:               shift 0
        - ETF:                   shift 0
        - OHLCV 1H base:         shift 0
        """
        raw = self.collector.load_all()

        # â”€â”€ Base: OHLCV 1H â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        base = raw.get("ohlcv_1h", pd.DataFrame())
        if base.empty:
            raise ValueError("OHLCV 1H no disponible. Ejecutar DataCollector.fetch_all primero.")

        # 🔸 Futures (Perps) — sin lag 🔸
        if "ohlcv_futures_1h" in raw and not raw["ohlcv_futures_1h"].empty:
            futures = raw["ohlcv_futures_1h"].copy()
            futures.index = pd.to_datetime(futures.index, utc=True).as_unit("ns")
            futures = futures.resample("1h").last().ffill()
            # Renombrar columnas esenciales para evitar sobreescribir el spot
            fut_cols = {
                "close": "close_perps",
                "volume": "volume_perps",
                "taker_buy_base": "taker_buy_base_perps",
                "trades": "trades_perps"
            }
            futures = futures.rename(columns=fut_cols)[list(fut_cols.values())]
            base = base.join(futures, how="left")
        # Normalizar a ns para garantizar compatibilidad de join con todas las fuentes
        base.index = pd.to_datetime(base.index, utc=True).as_unit("ns")

        # â”€â”€ Macro â€” resamplear a 1H + safety lags â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if "macro" in raw and not raw["macro"].empty:
            macro = raw["macro"].copy()
            macro.index = pd.to_datetime(macro.index, utc=True).as_unit("ns")
            macro = macro.resample("1h").last().ffill()

            # Safety lag M2 (42 dÃ­as = 42*24 horas)
            m2_cols = [c for c in macro.columns if "M2" in c]
            if m2_cols:
                macro[m2_cols] = macro[m2_cols].shift(cfg.data.m2_lag_days * 24)

            # Safety lag CPI, FedFunds, Unemploy, WEI (14 dÃ­as)
            lag14_cols = [c for c in macro.columns
                         if any(k in c for k in ["CPI", "FedFunds", "Unemploy", "WEI"])]
            if lag14_cols:
                macro[lag14_cols] = macro[lag14_cols].shift(cfg.data.cpi_lag_days * 24)

            base = base.join(macro, how="left", rsuffix="_macro")

        # â”€â”€ On-chain â€” lag +24H â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if "onchain" in raw and not raw["onchain"].empty:
            onchain = raw["onchain"].copy()
            onchain.index = pd.to_datetime(onchain.index, utc=True).as_unit("ns")
            onchain = onchain.resample("1h").last().ffill()
            onchain = onchain.shift(cfg.data.onchain_lag_hours)  # +24H
            base = base.join(onchain, how="left", rsuffix="_onchain")

        # â”€â”€ Derivatives â€” sin lag â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if "derivatives" in raw and not raw["derivatives"].empty:
            deriv = raw["derivatives"].copy()
            deriv.index = pd.to_datetime(deriv.index, utc=True).as_unit("ns")
            deriv = deriv.resample("1h").last().ffill(limit=72)  # FIX-DERIV-GAP-01: limitar a 72H para evitar propagacion de gaps largos (ej: FundingRate gap 933d)
            base = base.join(deriv, how="left", rsuffix="_deriv")

        # â”€â”€ Altcoins â€” ETH lag +13d, resto sin lag â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if "altcoins" in raw and not raw["altcoins"].empty:
            alt = raw["altcoins"].copy()
            alt.index = pd.to_datetime(alt.index, utc=True).as_unit("ns")
            alt = alt.resample("1h").last().ffill()
            # ETH lag (con validación dinámica C8)
            eth_cols = [c for c in alt.columns if "ETH" in c]
            if eth_cols and "ETH_Price" in alt.columns and "close" in base.columns:
                try:
                    from luna.features.eth_lag_validator import validate_eth_lag
                    validate_eth_lag(base["close"], alt["ETH_Price"], cfg.data.eth_lag_days)
                except Exception as e:
                    logger.warning(f"No se pudo validar ETH Lag dinámico (C8): {e}")

            if eth_cols:
                alt[eth_cols] = alt[eth_cols].shift(cfg.data.eth_lag_days * 24)
            base = base.join(alt, how="left", rsuffix="_alt")

        # â”€â”€ DeFi â€” lag +24H â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if "defi" in raw and not raw["defi"].empty:
            defi = raw["defi"].copy()
            defi.index = pd.to_datetime(defi.index, utc=True).as_unit("ns")
            defi = defi.resample("1h").last().ffill()
            defi = defi.shift(cfg.data.onchain_lag_hours)  # +24H (igual que on-chain)
            base = base.join(defi, how="left", rsuffix="_defi")

        # â”€â”€ Mempool â€” sin lag â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if "mempool" in raw and not raw["mempool"].empty:
            mem = raw["mempool"].copy()
            mem.index = pd.to_datetime(mem.index, utc=True).as_unit("ns")
            mem = mem.resample("1h").last().ffill()
            base = base.join(mem, how="left", rsuffix="_mem")

        # â”€â”€ ETF â€” sin lag â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if "etf" in raw and not raw["etf"].empty:
            etf = raw["etf"].copy()
            etf.index = pd.to_datetime(etf.index, utc=True).as_unit("ns")
            etf = etf.resample("1h").last().ffill()
            base = base.join(etf, how="left", rsuffix="_etf")

        # â”€â”€ Stablecoin_Cap + M2 Global â€” lag +24H â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if "stablecoin_m2" in raw and not raw["stablecoin_m2"].empty:
            stab = raw["stablecoin_m2"].copy()
            stab.index = pd.to_datetime(stab.index, utc=True).as_unit("ns")
            stab = stab.resample("1h").last().ffill()
            stab = stab.shift(cfg.data.onchain_lag_hours)  # +24H (publicaciÃ³n externa)
            # Calcular SSR Z-Score aquÃ­ ya tenemos close disponible
            if "Stablecoin_Cap" in stab.columns:
                stab["Stablecoin_Cap_Delta"] = stab["Stablecoin_Cap"].diff(7*24)   # 7d delta horario
                stab["Stablecoin_Cap_YoY"]   = stab["Stablecoin_Cap"].pct_change(365*24)
            # Renombrar columnas que colisionarÃ¡n con macro para evitar _stab suffix
            # Columnas de stablecoin_m2 que NO existen en base â†’ sin problema
            # Columnas que SÃ existen (Stablecoin_Cap se crea vacÃ­a del macro join) â†’ forzar overwrite
            stab_only_cols = [c for c in stab.columns if c not in base.columns]
            stab_collision_cols = [c for c in stab.columns if c in base.columns]
            # Sobrescribir directamente las columnas que colisionan con los valores reales de stab
            for c in stab_collision_cols:
                if stab[c].notna().any():
                    stab_mapped = stab[c].reindex(base.index, method='ffill')
                    base[c] = base[c].fillna(stab_mapped)
            # Añadir las columnas únicas de stablecoin (sin rsuffix)
            if stab_only_cols:
                stab_mapped_only = stab[stab_only_cols].reindex(base.index, method='ffill')
                base = base.join(stab_mapped_only, how="left")

        # ── M2 Global Composite (Fase 4) ─────────────────────────────────
        m2_global_path = _ROOT / "data" / "raw" / "macro" / "m2_global.parquet"
        if m2_global_path.exists():
            try:
                m2_global = pd.read_parquet(m2_global_path)
                m2_global.index = pd.to_datetime(m2_global.index, utc=True).as_unit("ns")
                # El lag estructural de 42 días ya está aplicado en el fetcher.
                base = base.join(m2_global, how="left")
                logger.info("M2 Global Composite (YoY) integrado con éxito.")
            except Exception as e:
                logger.warning(f"Error cargando m2_global.parquet: {e}")

        # SSR Z-Score (Stablecoin Supply Ratio) â€” requiere close + Stablecoin_Cap
        if "Stablecoin_Cap" in base.columns and "close" in base.columns:
            # Fix BUG M-03: supply dinámico por halvings (evita error ~9% con 19.8M fijo)
            _supply_calc = BTCDynamicSupply()
            supply_series = _supply_calc.get_supply_series(base.index).reindex(base.index)
            btc_mktcap_proxy = base["close"] * supply_series
            ssr = base["Stablecoin_Cap"] / btc_mktcap_proxy
            ssr_mean = ssr.rolling(90*24).mean()    # 90d rolling
            ssr_std  = ssr.rolling(90*24).std()
            base["SSR_ZScore"] = (ssr - ssr_mean) / ssr_std.clip(lower=1e-9)
            logger.info("SSR Z-Score calculado con BTCDynamicSupply (halvings-aware, BUG M-03)")

        # â”€â”€ Deduplicar columnas tÃ©cnicas _mem / _etf â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Cuando hay col 'X' y 'X_mem'/'X_etf', conservar la que tenga menos NaN
        # y eliminar la redundante. Sucede al hacer join con rsuffix cuando ambas
        # fuentes (onchain y mempool) tienen la misma columna.
        for suffix in ["_mem", "_etf", "_onchain", "_defi"]:
            suffixed_cols = [c for c in base.columns if c.endswith(suffix)]
            for sc in suffixed_cols:
                original = sc[: -len(suffix)]  # nombre sin el sufijo
                if original in base.columns:
                    # Elegir la que tenga menos NaN (= mÃ¡s datos)
                    nan_orig    = base[original].isna().mean()
                    nan_suffixed = base[sc].isna().mean()
                    if nan_suffixed <= nan_orig:
                        # La versiÃ³n _mem tiene igual o menos NaN â†’ reemplazar
                        base[original] = base[original].fillna(base[sc])
                    # Eliminar la columna tÃ©cnica con sufijo siempre
                    base = base.drop(columns=[sc])
                    logger.debug(f"Dedup {sc} -> {original} (NaN: {nan_orig:.1%} â†’ {base[original].isna().mean():.1%})")
                else:
                    # No hay columna base, renombrar quitando el sufijo
                    base = base.rename(columns={sc: original})
                    logger.debug(f"Renombrado {sc} -> {original}")

        # Eliminar columnas con nombre identico duplicado (keep first)
        base = base.loc[:, ~base.columns.duplicated(keep="first")]

        # ── DEAD-FEAT-PRUNE (2026-04-19) ─────────────────────────────────────
        # Grupo 1: features 100% NaN que nunca fueron seleccionadas por SFI
        # (verificado en 120 logs históricos). Eliminadas aquí para:
        #   a) Reducir raw features de ~316 → ~295 (k_auto: 214 → ~40)
        #   b) Evitar lag cache invalidation (SFI tarda +90 min extra sin cache)
        #   c) Eliminar columnas completamente vacías que saturan el clustering
        #
        # Grupo 3: duplicados internos — la información ya está disponible en
        # columnas calculadas por apply_derived_features() con mejor calidad.
        _COLS_TO_PRUNE = [
            # ── Grupo 1A: ETF APIs muertas (100% NaN, nunca seleccionadas) ──
            "BITO_Price",      # duplicado inferior de BITO_Close (que sí tiene datos)
            "FBTC_Price",      # API FBTC no actualizada; GBTC_Discount_Pct cubre ETF
            "ARKB_Price",      # ARK Bitcoin ETF — API inactiva
            "ARKB_Volume",     # ARK Bitcoin ETF — API inactiva
            "BITB_Price",      # Bitwise ETF — API inactiva
            "BITB_Volume",     # Bitwise ETF — API inactiva

            # ── Grupo 1B: Macro APIs muertas (100% NaN, nunca seleccionadas) ──
            "Breakeven_Inflation_5Y",  # FRED T5YIE — série descontinuada en raw;
                                       # CPI_YoY (31% NaN) cubre expectativas de inflación
            "NatGas",          # Gas Natural — sin actualización; Oil (sano) cubre energía
            "USD",             # índice USD crudo — DXY es el estándar y está disponible

            # ── Grupo 1C: Mempool APIs muertas (100% NaN, nunca seleccionadas) ──
            "avgFees",         # mempool_raw — API completamente inactiva
            "avgHeight",       # mempool_raw — API completamente inactiva

            # ── Grupo 1D: Derivatives legacy (100% NaN, nunca seleccionadas) ──
            "ls_ratio_ema_24h",  # EMA del LongShortRatio — derivado de LSR que
                                 # también está deteriorado; LongAccount lo reemplaza

            # ── Grupo 3: Duplicados internos exactos ─────────────────────────
            # BTC_SP500_Ratio: columna cruda del raw. apply_derived_features()
            # calcula mc_btc_sp500_ratio = close/SP500 con mejor calidad (ffill).
            # Mantener ambas duplica la información y crea features correlacionadas 1.0.
            "BTC_SP500_Ratio",

            # BTC_Gold_Ratio: 100% NaN y sin alternativa directa calculable.
            # El pipeline no tiene fuente de Gold en raw (Gold_Ret también 100% NaN).
            "BTC_Gold_Ratio",
        ]

        pruned = [c for c in _COLS_TO_PRUNE if c in base.columns]
        if pruned:
            base = base.drop(columns=pruned)
            logger.info(
                f"DEAD-FEAT-PRUNE: eliminadas {len(pruned)} columnas vacías/duplicadas "
                f"({pruned}). Raw features: {base.shape[1]} cols"
            )

        logger.info(f"Raw merged: {base.shape[0]} rows x {base.shape[1]} cols")
        return base


    # â”€â”€ PASO 3B: Mining Outputs (K_Shape_Cluster_ID + Master_Causal_Signal) â”€â”€

    def _check_parquet_freshness(self, path: "Path", max_days: int = 7) -> bool:
        """Fix F20: Verifica que el parquet no supere max_days de antigÃ¼edad."""
        if not path.exists():
            return False
        mtime = pd.Timestamp(path.stat().st_mtime, unit='s', tz='UTC')
        age_days = (pd.Timestamp.now('UTC') - mtime).days  # [FIX-PIPE-002] utcnow() deprecated
        if age_days > max_days:
            logger.warning(
                f"[F20 Stale-Check] {path.name} tiene {age_days} dÃ­as de antigÃ¼edad "
                f"(mÃ¡x: {max_days}d). SeÃ±al posiblemente obsoleta. "
                f"Ejecutar run_weekly_mining.py para actualizar."
            )
        return True  # Advertir pero no bloquear (mining es opcional)

    def integrate_mining_outputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Integra los outputs del AI Mining al DataFrame principal.

        FIX: Antes, KShapeClusterEngine guardaba features_train_kshape.parquet
        SEPARADO del features_train.parquet, dejando K_Shape_Cluster_ID fuera
        del pipeline. Este mÃ©todo une ambos.

        Si los parquets enriquecidos no existen (primer run), genera versiones
        simplificadas inline para no bloquear el pipeline.

        Ejecutar run_weekly_mining.py primero para obtener la versiÃ³n precisa.
        """
        # ── Master_Causal_Signal ─────────────────────────────────────────
        if "Master_Causal_Signal" not in df.columns:
            bce_weights_path = _ROOT / "data" / "models" / "bce_weights.json"
            _mcs_calculated = False
            
            # [P2-BCE-SERIALIZATION] Inferencia Online Dinámica
            if bce_weights_path.exists():
                try:
                    import json
                    with open(bce_weights_path, "r", encoding="utf-8") as f:
                        bce_data = json.load(f)
                    weights = bce_data.get("weights", [])
                    
                    ATE_MIN_MAGNITUDE = 0.002
                    sig_results = [r for r in weights if abs(r.get("ate", 0)) >= ATE_MIN_MAGNITUDE]
                    
                    if sig_results:
                        signal = pd.Series(0.0, index=df.index)
                        for r in sig_results:
                            var = r["variable"]
                            if var not in df.columns:
                                continue
                            col = df[var].ffill().fillna(0)
                            mu90  = col.rolling(90 * 24, min_periods=24).mean()
                            std90 = col.rolling(90 * 24, min_periods=24).std().replace(0, 1)
                            z     = (col - mu90) / std90
                            sign  = 1 if r.get("direction") == "BULLISH" else -1
                            weight = abs(r.get("ate", 0))
                            signal += sign * weight * z.clip(-3, 3)
                        
                        total_weight = sum(abs(r.get("ate", 0)) for r in sig_results)
                        if total_weight > 0:
                            signal /= total_weight
                            
                        df["Master_Causal_Signal"] = signal.clip(-1, 1).fillna(0.0)
                        _mcs_calculated = True
                        
                        _mcs_std = df["Master_Causal_Signal"].std()
                        logger.info(f"[A3alt/BCE-Online] Master_Causal_Signal calculada en tiempo real | std={_mcs_std:.4f}")
                        print(f"[LUNA][A3alt/BCE-Online] Master_Causal_Signal calculada en tiempo real desde pesos ATE serializados. std={_mcs_std:.4f}")
                    else:
                        logger.warning("[A3alt/BCE-Online] No hay pesos significativos en bce_weights.json")
                except Exception as e:
                    logger.warning(f"[A3alt/BCE-Online] Fallo al usar pesos serializados: {e}.")
            
            # Fallback al Proxy si no se pudo calcular
            if not _mcs_calculated:
                logger.warning("Usando Proxy inline MVRV+StablecoinCap para Master_Causal_Signal.")
                proxy = pd.Series(0.0, index=df.index)
                if "MVRV_Proxy" in df.columns:
                    mz = (df["MVRV_Proxy"] - df["MVRV_Proxy"].rolling(90*24, min_periods=24).mean()) / \
                         df["MVRV_Proxy"].rolling(90*24, min_periods=24).std().clip(lower=1e-9)
                    proxy = proxy + mz.fillna(0) * 0.5
                if "Stablecoin_Cap" in df.columns:
                    proxy = proxy + df["Stablecoin_Cap"].pct_change(7*24).fillna(0) * 0.5
                df["Master_Causal_Signal"] = proxy.clip(-3, 3)
                logger.info("Master_Causal_Signal proxy inline (MVRV+StablecoinCap)")


        # â”€â”€ K_Shape_Cluster_ID â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if "KMeans_Tribe_ID" not in df.columns:
            tribe_parquet = _FEATURES_DIR / "features_train_final.parquet"
            if tribe_parquet.exists():
                self._check_parquet_freshness(tribe_parquet)
                try:
                    src = pd.read_parquet(tribe_parquet)
                    src = purge_leakage_columns(src)
                    src.index = pd.to_datetime(src.index, utc=True)
                except Exception as e:
                    logger.error(f"[CORRUPT] Fallo al leer {tribe_parquet.name}: {e}. Eliminando archivo corrupto.")
                    try: tribe_parquet.unlink()
                    except: pass
                    src = None
                
                if src is not None and "KMeans_Tribe_ID" in src.columns:
                    # [BUG-L3-TRIM 2026-05-30] Verificar fechas > train_end y TRIMMAR en lugar de fallback=0.
                    # PROBLEMA ANTERIOR: src_max > train_end -> KMeans_Tribe_ID=0 para TODAS las filas IS.
                    # Eso destruia la info tribal del 100% del IS (63034 filas -> tribe=0 constante).
                    # CORRECCION: TRIM src a <= train_end antes del join. IS obtiene tribe_ID real.
                    # Las filas OOS (> train_end) quedan NaN y son cubiertas por [A1/KMeans-Online].
                    _src_max_date = src.index.max()
                    try:
                        from config.settings import cfg as _cfg_l3
                        _train_end_l3 = pd.Timestamp(getattr(_cfg_l3.temporal_splits, 'train_end', '2099-01-01'), tz='UTC')
                    except Exception:
                        _train_end_l3 = pd.Timestamp('2099-01-01', tz='UTC')
                    if _src_max_date > _train_end_l3:
                        _n_rows_before = len(src)
                        src = src[src.index <= _train_end_l3]
                        _n_trimmed = _n_rows_before - len(src)
                        print(f"[BUG-FIX-LOG 2026-06-05] Corregido formatting logger.warning en feature_pipeline.py [BUG-L3-TRIM]")
                        logger.warning(
                            "[BUG-L3-TRIM] features_train_final.parquet tenia fechas > train_end "
                            "({} > {}). TRIM: {} filas OOS eliminadas del src ({}->{}). "
                            "IS rows preservadas con KMeans_Tribe_ID real.",
                            _src_max_date.date(), _train_end_l3.date(),
                            _n_trimmed, _n_rows_before, len(src)
                        )
                        print(f"[FIX-BUG-L3-TRIM] src trimado: {_n_rows_before}->{len(src)} filas "
                              f"({_n_trimmed} OOS eliminadas). IS preservado hasta {_train_end_l3.date()}")
                    df = df.join(src[["KMeans_Tribe_ID"]], how="left", rsuffix="_kt")

                    # ── [A1: KMeans Online Inference] ────────────────────────────────────
                    # PROBLEMA: el join + ffill() propaga el último cluster del training
                    # indefinidamente al holdout. Si el gap train→holdout es grande (ej:
                    # 1488h en WFB W1), el 100% del holdout queda en un solo cluster.
                    #
                    # SOLUCIÓN: si existe el pkl del modelo KMeans (guardado por
                    # ClusterPatternEngine con centroides ajustados SOLO en train), cargar
                    # ese modelo y predecir el cluster de cada barra OOS.
                    #
                    # GARANTÍA CAUSAL: los centroides son parámetros fijos del training.
                    # km.predict(X_oos_t) solo usa datos de la barra t. Sin look-ahead.
                    # Es el mismo mecanismo que usa el HMM en _run_hmm_enrichment().
                    _km_pkl   = _ROOT / "data" / "models" / "kmeans_model.pkl"
                    _sc_pkl   = _ROOT / "data" / "models" / "kmeans_scaler.pkl"
                    _km_used_online = False

                    print(f"[LUNA][A1/KMeans] Verificando inferencia online... "
                          f"pkl={'SI' if _km_pkl.exists() else 'NO'} "
                          f"scaler={'SI' if _sc_pkl.exists() else 'NO'}")

                    if _km_pkl.exists() and _sc_pkl.exists():
                        try:
                            import joblib as _jbl_km
                            from luna.ai_mining.cluster_pattern_engine import TRIBE_FEATURES as _TF
                            _km = _jbl_km.load(_km_pkl)
                            _sc = _jbl_km.load(_sc_pkl)

                            # Filas OOS: las que KMeans_Tribe_ID es NaN tras el join
                            # (= barras del holdout/validation que no estaban en features_train_final)
                            _oos_mask = df["KMeans_Tribe_ID"].isna()
                            _n_oos = _oos_mask.sum()
                            _n_train = (~_oos_mask).sum()

                            print(f"[LUNA][A1/KMeans] Barras train (join directo): {_n_train} | "
                                  f"Barras OOS (sin cluster): {_n_oos}")

                            if _n_oos > 0:
                                _avail_tf = [f for f in _TF if f in df.columns]
                                if len(_avail_tf) >= 3:   # mínimo razonable de features
                                    _X_oos = df.loc[_oos_mask, _avail_tf].fillna(0)
                                    _pred  = _km.predict(_sc.transform(_X_oos))
                                    df.loc[_oos_mask, "KMeans_Tribe_ID"] = _pred.astype(float)
                                    _km_used_online = True
                                    _tribe_dist = {int(t): int((_pred == t).sum())
                                                   for t in sorted(set(_pred))}
                                    print(f"[LUNA][A1/KMeans] OK - {_n_oos} barras OOS reclasificadas "
                                          f"con centroides congelados. Distribución: {_tribe_dist}")
                                    logger.info(
                                        f"[A1/KMeans-Online] {_n_oos} barras OOS reclasificadas "
                                        f"con centroides congelados del training. "
                                        f"Distribución tribus OOS: {_tribe_dist}"
                                    )
                                else:
                                    print(f"[LUNA][A1/KMeans] WARN - Solo {len(_avail_tf)}/{len(_TF)} "
                                          f"TRIBE_FEATURES disponibles. Usando ffill como fallback.")
                                    logger.warning(
                                        f"[A1/KMeans-Online] Solo {len(_avail_tf)} de {len(_TF)} "
                                        f"TRIBE_FEATURES disponibles — usando ffill como fallback."
                                    )
                            else:
                                print(f"[LUNA][A1/KMeans] Sin barras OOS (todas en training set). "
                                      f"No se requiere inferencia online.")
                        except Exception as _km_e:
                            print(f"[LUNA][A1/KMeans] ERROR en inferencia online: {_km_e} - usando ffill")
                            logger.warning(f"[A1/KMeans-Online] Error en inferencia online: {_km_e} — usando ffill.")

                    if not _km_used_online:
                        # Fallback: ffill clásico (degradado pero sin look-ahead si src <= train_end)
                        df["KMeans_Tribe_ID"] = df["KMeans_Tribe_ID"].ffill().fillna(0)
                        print(f"[LUNA][A1/KMeans] FALLBACK ffill activo (pkl no disponible). "
                              f"Único valor propagado: {df['KMeans_Tribe_ID'].value_counts().idxmax()}")
                        logger.debug("[A1/KMeans-Online] Fallback ffill activo (pkl no disponible).")

                    df["KMeans_Tribe_ID"] = df["KMeans_Tribe_ID"].fillna(0).astype(int)
                    _n_unique_tribes = df['KMeans_Tribe_ID'].nunique()
                    _vc = df['KMeans_Tribe_ID'].value_counts().to_dict()
                    print(f"[LUNA][A1/KMeans] FINAL: {_n_unique_tribes} tribus activas | "
                          f"distribución={_vc} | online_inference={_km_used_online}")
                    logger.info(f"KMeans_Tribe_ID integrado | tribus totales: {df['KMeans_Tribe_ID'].nunique()} | online_inference={_km_used_online}")
                    # ── fin [A1] ─────────────────────────────────────────────────────────


            else:
                logger.warning("Ningún parquet con KMeans_Tribe_ID. KMeans fallback inline.")
                # FIX-V5-01: Anulamos el fallback inline KMeans porque fit_predict sobre
                # toda la serie de tiempo (X) incluye el dataset OOS -> Look-Ahead Bias.
                # Al fallar el proxy, asignamos Tribu 0 a todos para no filtrar datos futuros.
                df["KMeans_Tribe_ID"] = 0
                logger.info("KMeans_Tribe_ID fallback causal asignado a 0 (neutro)")

        # â”€â”€ Features derivadas requeridas por selected_features.json â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # vix_slope_7d: slope porcentual del VIX en 7d (datos horarios: 7*24 periodos)
        if "vix_slope_7d" not in df.columns:
            if "VIX_Slope7d" in df.columns:
                df["vix_slope_7d"] = df["VIX_Slope7d"]
            elif "VIX" in df.columns:
                df["vix_slope_7d"] = df["VIX"].pct_change(7 * 24)
            else:
                logger.warning("vix_slope_7d no generada: VIX no disponible.")

        # yield_curve_pct_1y: cambio YoY% del spread de la curva de rendimientos
        if "yield_curve_pct_1y" not in df.columns:
            if "YieldCurve_10Y3M" in df.columns:
                df["yield_curve_pct_1y"] = df["YieldCurve_10Y3M"].pct_change(365 * 24)
            elif "T10Y2Y" in df.columns:
                df["yield_curve_pct_1y"] = df["T10Y2Y"].pct_change(365 * 24)
            else:
                logger.warning("yield_curve_pct_1y no generada: YieldCurve_10Y3M no disponible.")

        # M2_YoY_Chg_milag12h: M2_YoY_Chg con lag extra de 12H (mitigaciÃ³n de retraso de publicaciÃ³n)
        if "M2_YoY_Chg_milag12h" not in df.columns:
            if "M2_YoY_Chg" in df.columns:
                df["M2_YoY_Chg_milag12h"] = df["M2_YoY_Chg"].shift(12)
            else:
                logger.warning("M2_YoY_Chg_milag12h no generada: M2_YoY_Chg no disponible.")

        # Stablecoin_Cap_z90d: Rolling Z-Score de 90 dias de Stablecoin_Cap
        if "Stablecoin_Cap" in df.columns:
            sc = df["Stablecoin_Cap"]
            if "Stablecoin_Cap_z90d" not in df.columns:
                sc_m = sc.rolling(90 * 24).mean()
                sc_s = sc.rolling(90 * 24).std().clip(lower=1e-9)
                df["Stablecoin_Cap_z90d"] = (sc - sc_m) / sc_s
        else:
            if "Stablecoin_Cap_z90d" not in df.columns:
                logger.warning("Stablecoin_Cap_z90d no generada: Stablecoin_Cap no disponible.")

        # ── HMM_Regime + HMM_Semantic (Fase 2: Asimetría Causal) ─────────
        # [FIX-PIPE-001] Inyectar HMM_Regime Y HMM_Semantic ANTES de SFI para selección
        # causal condicionada. CRÍTICO: HMM_Semantic (string) es requerida por el filtro
        # isin(hmm_allowed_regimes) en predict_oos.py. Sin ella, el filtro rechaza TODAS
        # las señales y el WFB produce 0 trades. Bug raíz: join solo incluía HMM_Regime.
        hmm_parquet = _FEATURES_DIR / "hmm_regime_labels.parquet"

        if "HMM_Regime" not in df.columns:
            if hmm_parquet.exists():
                try:
                    hmm_src = pd.read_parquet(hmm_parquet)
                    if "HMM_Regime" in hmm_src.columns:
                        # [FIX-PIPE-001] Incluir HMM_Semantic si está disponible en el parquet.
                        # El parquet generado por hmm_regime.py tiene shape=(N, 2) con ambas columnas.
                        cols_to_join = ["HMM_Regime"]
                        if "HMM_Semantic" in hmm_src.columns:
                            cols_to_join.append("HMM_Semantic")
                        df = df.join(hmm_src[cols_to_join], how="left")
                        hmm_cov = df["HMM_Regime"].notna().mean()
                        sem_cov = df["HMM_Semantic"].notna().mean() if "HMM_Semantic" in df.columns else 0.0
                        logger.info(
                            f"[FIX-PIPE-001] HMM labels integrados desde {hmm_parquet.name} | "
                            f"HMM_Regime cov={hmm_cov:.1%} | HMM_Semantic cov={sem_cov:.1%}"
                        )
                        print(
                            f"[FIX-PIPE-001] HMM_Regime+HMM_Semantic inyectados | "
                            f"cols={cols_to_join} | HMM_Regime_cov={hmm_cov:.1%} | HMM_Semantic_cov={sem_cov:.1%}"
                        )
                except Exception as e:
                    logger.error(f"[FIX-PIPE-001][CORRUPT] Fallo al leer {hmm_parquet.name}: {e}")
                    print(f"[FIX-PIPE-001][ERROR] Fallo cargando hmm_regime_labels.parquet: {e}")
            else:
                logger.warning(
                    f"[FIX-PIPE-001][HMM] {hmm_parquet.name} no encontrado en integrate_mining_outputs. "
                    f"SFI evaluará sin régimen (Global)."
                )
                print(f"[FIX-PIPE-001][WARN] hmm_regime_labels.parquet no existe todavía (normal en 1ª pasada del pipeline).")

        elif "HMM_Semantic" not in df.columns:
            # [FIX-PIPE-001-B] Guard secundario: HMM_Regime ya está en df (de una pasada anterior del pipeline)
            # pero HMM_Semantic no. Esto ocurre cuando el pipeline se ejecutó antes de este fix.
            # Intentar derivarla del parquet o del pkl del HMM.
            print(
                f"[FIX-PIPE-001-B] HMM_Regime presente pero HMM_Semantic ausente. "
                f"Intentando inyectar HMM_Semantic desde parquet o pkl HMM..."
            )
            _semantic_injected = False
            if hmm_parquet.exists():
                try:
                    hmm_src = pd.read_parquet(hmm_parquet)
                    if "HMM_Semantic" in hmm_src.columns:
                        # Drop primero para evitar conflicto en join
                        df = df.join(hmm_src[["HMM_Semantic"]], how="left")
                        sem_cov = df["HMM_Semantic"].notna().mean()
                        logger.info(f"[FIX-PIPE-001-B] HMM_Semantic inyectada desde parquet. cov={sem_cov:.1%}")
                        print(f"[FIX-PIPE-001-B] HMM_Semantic inyectada desde parquet | cov={sem_cov:.1%}")
                        _semantic_injected = True
                except Exception as e_b:
                    logger.warning(f"[FIX-PIPE-001-B] No se pudo inyectar HMM_Semantic desde parquet: {e_b}")

            if not _semantic_injected:
                # Intentar derivar desde el pkl del HMM con state_map
                _hmm_pkl = _FEATURES_DIR.parent / "models" / "hmm_regime.pkl"
                if _hmm_pkl.exists():
                    try:
                        import joblib as _jbl_fp
                        _saved_fp = _jbl_fp.load(_hmm_pkl)
                        _smap_fp = _saved_fp.get("state_map", {})
                        if _smap_fp:
                            df["HMM_Semantic"] = df["HMM_Regime"].map(_smap_fp).fillna("UNKNOWN")
                            sem_cov_pkl = (df["HMM_Semantic"] != "UNKNOWN").mean()
                            logger.info(f"[FIX-PIPE-001-B] HMM_Semantic derivada desde pkl state_map. cov={sem_cov_pkl:.1%}")
                            print(f"[FIX-PIPE-001-B] HMM_Semantic derivada desde pkl state_map | cov={sem_cov_pkl:.1%} | state_map={_smap_fp}")
                            _semantic_injected = True
                    except Exception as e_pkl:
                        logger.warning(f"[FIX-PIPE-001-B] No se pudo derivar HMM_Semantic desde pkl: {e_pkl}")

            if not _semantic_injected:
                logger.warning(
                    "[FIX-PIPE-001-B] No se pudo inyectar HMM_Semantic por ningún método. "
                    "El filtro de régimen en predict_oos producirá 0 señales."
                )
                print("[FIX-PIPE-001-B][WARN] HMM_Semantic no disponible. 0 trades probable en esta ventana.")

        return df

    # â”€â”€ PASO 4: Rolling Z-Score (R14) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def apply_rolling_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica Rolling Z-Score 90d (R14) + Kalman Z-Score adaptativo (V2-P4).

        [V2-P4] Las features '_kz' se añaden ADICIONALMENTE al rolling 90d original.
        El SFI decidirá cuáles son más predictivas en cada ventana WFB.

        Rolling Z-Score 90d: ventana fija, estable, retrocompatible (R14).
        Kalman Z-Score (_kz): adaptativo, detecta cambios de régimen en ~14-21d.
        """
        # ── Paso 1: Rolling Z-Score 90d (comportamiento original — R14) ──
        df = self.normalizer.transform(df, columns=MACRO_COLUMNS)

        # ── Paso 2: Kalman Z-Score adaptativo (V2-P4) ────────────────────
        try:
            from luna.features.kalman_normalizer import KalmanZScoreNormalizer, KALMAN_COLUMNS
            from config.settings import cfg as _cfg_kz
            _kz_q = float(getattr(getattr(_cfg_kz, 'features', {}), 'kalman_q', 1e-4))
            _kz_r = float(getattr(getattr(_cfg_kz, 'features', {}), 'kalman_r', 0.1))
        except Exception:
            _kz_q, _kz_r = 1e-4, 0.1
            from luna.features.kalman_normalizer import KalmanZScoreNormalizer, KALMAN_COLUMNS

        _kalman = KalmanZScoreNormalizer(process_noise=_kz_q, obs_noise=_kz_r)
        df = _kalman.transform_df(df, columns=KALMAN_COLUMNS, suffix="_kz")

        return df


    # â”€â”€ PASO 5: MultiTimeframe features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def apply_multitimeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera features multi-timeframe desde OHLCV 15min y 4H.
        Importa MultitimeframeFeatures si disponible.
        """
        try:
            from luna.features.multitimeframe_features import MultitimeframeFeatures
            mtf = MultitimeframeFeatures()
            df = mtf.transform(df)
        except ImportError:
            logger.warning("multitimeframe_features.py no disponible aÃºn. Saltando.")
        return df

    # â”€â”€ PASO 6: Calendar Features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def apply_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera las 13 features cal_* (FOMC, Deribit, Halving, Sesiones)."""
        return self.calendar_fe.transform(df)

    # â”€â”€ PASO 7: Cross-Asset features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def apply_crossasset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera ETH/BTC corr, BTC Dominance, DangerZone."""
        # ETH/BTC rolling correlation 24H
        if "ETH_Price" in df.columns and "close" in df.columns:
            eth_ret = df["ETH_Price"].pct_change(1)
            btc_ret = df["close"].pct_change(1)
            df["ETH_BTC_Corr24H"] = eth_ret.rolling(24).corr(btc_ret)
            df["ETH_Lead_1H"] = df["ETH_Price"].pct_change(1).shift(1)  # ETH 1H antes como predictor

        # DangerZone ya se computa en DerivativesFetcher si OI y Funding disponibles
        # AquÃ­ aseguramos que estÃ¡ presente
        if "DangerZone" not in df.columns:
            if "FundingRate" in df.columns and "OI_BTC" in df.columns:
                fr_pct = df["FundingRate"].rolling(90*24, min_periods=48).rank(pct=True)
                oi_pct = df["OI_BTC"].rolling(90*24, min_periods=48).rank(pct=True)
                df["DangerZone"] = (fr_pct + oi_pct) / 2

        return df

    # â”€â”€ PASO 7B: Derived Macro + On-Chain features (mc_* / oc_*) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Estos features cubren el gap vs Luna v2 + Correlaciones.
    # Todos se calculan desde datos raw ya presentes en el DataFrame.
    # Referencia: Luna v2/core/data/fetch_macro.py, fetch_onchain.py

    def apply_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera features derivadas mc_* y oc_* equivalentes a Luna v2 V10.
        No requiere fetchers adicionales â€” usa VIX, DXY, SP500, close, volume,
        YieldCurve_10Y3M, M2_YoY_Chg, Hashrate, etc. ya presentes en df.
        """
        import numpy as np

        # â”€â”€ Macro Market (mc_*) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # VIX derived
        if "VIX" in df.columns:
            vix = df["VIX"].ffill()
            df["mc_vix_raw_z90d"]  = (vix - vix.rolling(90*24).mean()) / (vix.rolling(90*24).std().clip(lower=1e-9))
            df["mc_vix_slope_7d"]  = vix.diff(7*24)
            df["mc_vix_regime"]    = np.select(
                [vix < 20, vix < 30], [0, 1], default=2).astype(float)
            logger.info("mc_vix_* derivadas")

        # DXY derived
        if "DXY" in df.columns:
            dxy = df["DXY"].ffill()
            df["mc_dxy_raw_z90d"] = (dxy - dxy.rolling(90*24).mean()) / (dxy.rolling(90*24).std().clip(lower=1e-9))
            df["dxy_pct_6m"]      = dxy.pct_change(180 * 24)   # cambio 6m horario
            df["mc_dxy_slope_30d"] = dxy.pct_change(30 * 24)
            if "close" in df.columns:
                btc_dxy = df["close"] / dxy.replace(0, np.nan)
                df["mc_btc_dxy_ratio"] = btc_dxy
                df["mc_btc_dxy_ratio_z30d"] = (btc_dxy - btc_dxy.rolling(30*24).mean()) / btc_dxy.rolling(30*24).std().clip(lower=1e-9)



        # SP500 derived
        if "SP500" in df.columns:
            sp = df["SP500"].ffill()
            ma200h = sp.rolling(200*24).mean()
            df["mc_sp500_raw_z90d"]   = (sp - sp.rolling(90*24).mean()) / (sp.rolling(90*24).std().clip(lower=1e-9))
            df["mc_sp500_above_ma200"] = (sp > ma200h).astype(float)
            df["mc_sp500_ret_1m"]      = sp.pct_change(21*24)  # retorno 1 mes em horas
            if "close" in df.columns:
                btc_sp500 = df["close"] / sp.replace(0, np.nan)
                df["mc_btc_sp500_ratio"] = btc_sp500
                df["mc_btc_sp500_ratio_z30d"] = (btc_sp500 - btc_sp500.rolling(30*24).mean()) / btc_sp500.rolling(30*24).std().clip(lower=1e-9)

                # SFI Phase 5: Rolling Beta BTC vs SP500 (Desacople)
                sp500_ret = sp.pct_change(1)
                btc_ret = df["close"].pct_change(1)
                covar_30d = btc_ret.rolling(30*24).cov(sp500_ret)
                var_sp_30d = sp500_ret.rolling(30*24).var().clip(lower=1e-9)
                df["ca_btc_sp500_beta_30d"] = covar_30d / var_sp_30d

        # Risk Premium Proxy Diaria (SP500_Ret - DXY_Ret)
        if "SP500" in df.columns and "DXY" in df.columns:
            sp500_ret_1d = df["SP500"].ffill().pct_change(24)
            dxy_ret_1d = df["DXY"].ffill().pct_change(24)
            rp_proxy = sp500_ret_1d - dxy_ret_1d
            df["mc_risk_premium_proxy"] = rp_proxy
            df["mc_risk_premium_regime_z30d"] = (rp_proxy - rp_proxy.rolling(30*24).mean()) / rp_proxy.rolling(30*24).std().clip(lower=1e-9)
            logger.info("mc_risk_premium_proxy y mc_risk_premium_regime_z30d generados")

        # Yield Curve derived (YieldCurve_10Y3M o T10Y2Y)
        for yc_col in ["YieldCurve_10Y3M", "T10Y2Y"]:
            if yc_col in df.columns:
                yc = df[yc_col].ffill()
                df["mc_yield_curve_inverted"] = (yc < 0).astype(float)
                df["mc_yield_curve_pct_1y"]   = yc.rolling(252*24).rank(pct=True)
                # SFI Phase 5: Yield Curve Velocity (Bear Steepening)
                df["mc_yield_curve_velocity_30d"] = yc.diff(30 * 24).abs()
                break

        # Unemployment derived
        for unc_col in ["UnemployRate", "Unemploy_Rate"]:
            if unc_col in df.columns:
                unr = df[unc_col].ffill()
                df["mc_unemploy_rate_raw_z90d"]      = (unr - unr.rolling(90*24).mean()) / (unr.rolling(90*24).std().clip(lower=1e-9))
                df["mc_unemploy_rate_delta_3m_z90d"]  = unr.diff(3*30*24)  # delta 3 meses en horas
                break

        # M2 USA z90d
        for m2_col in ["M2_USA", "M2_YoY_Chg"]:
            if m2_col in df.columns:
                m2 = df[m2_col].ffill()
                df["mc_m2_usa_raw_z90d"] = (m2 - m2.rolling(90*24).mean()) / (m2.rolling(90*24).std().clip(lower=1e-9))
                break

        # M2 UK z90d
        for m2uk_col in ["M2_UK", "M2_UK_Raw"]:
            if m2uk_col in df.columns:
                m2uk = df[m2uk_col].ffill()
                df["mc_m2_uk_raw_z90d"] = (m2uk - m2uk.rolling(90*24).mean()) / (m2uk.rolling(90*24).std().clip(lower=1e-9))
                df["mc_m2_uk_yoy_z90d"] = m2uk.pct_change(365*24)
                break

        # â”€â”€ On-Chain proxies (oc_*) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # Hashrate derived â€” ya tenemos Hashrate_TH_s y Hashrate_7d_MA en onchain_raw
        for hr_col in ["Hashrate_7d_MA", "Hashrate_TH_s", "hashrate_7d_ma", "hashrate_th"]:
            if hr_col in df.columns:
                hr = df[hr_col].ffill()
                df["oc_hashrate_7d_ma"]   = hr
                df["oc_hashrate_chg_30d"] = hr.pct_change(30*24)
                break
                
        # Hash Ribbon Signal & Miner Capitulation (SFI Phase 4)
        if "hash_ribbon_signal" in df.columns:
            df["oc_hash_ribbon_inversion"] = df["hash_ribbon_signal"].ffill()
            logger.info("oc_hash_ribbon_inversion pass-through y ffill realizada")

        # Active Addresses â€” varios posibles nombres de columna desde onchain_raw
        aa_series = None
        for aa_col in ["ActiveAddresses_7d", "active_addresses_7d_ma", "active_addresses_7d",
                       "Active_Addresses", "ActiveAddresses"]:
            if aa_col in df.columns:
                aa = df[aa_col].ffill()
                df["oc_active_addresses_7d"] = aa
                aa_series = aa
                break

        # SFI Phase 5: NVT Proxy (Dynamic Network Value to Transactions)
        if "close" in df.columns:
            # Fallback a volumen de exchange (7d MA) si ActiveAddresses estÃ¡ corrupto (< 10% datos)
            if aa_series is not None and aa_series.notna().mean() > 0.1:
                utility_proxy = aa_series
            else:
                utility_proxy = df["volume"].rolling(7*24).mean()
                
            nvt_proxy = df["close"] / utility_proxy.replace(0, np.nan)
            df["oc_nvt_proxy_z90d"] = (nvt_proxy - nvt_proxy.rolling(90*24).mean()) / nvt_proxy.rolling(90*24).std().clip(lower=1e-9)

        # Exchange Flow Proxy â€” OBV momentum desde OHLCV (calculable siempre)
        if "close" in df.columns and "volume" in df.columns:
            close_s = df["close"]
            vol_s   = df["volume"]
            sign = np.sign(close_s.diff()).fillna(0)
            obv  = pd.Series((sign * vol_s.values).cumsum(), index=close_s.index)
            obv_ma7  = obv.rolling(7*24).mean()
            obv_ma30 = obv.rolling(30*24).mean()
            df["oc_exchange_flow_proxy"] = (obv_ma7 - obv_ma30) / (obv_ma30.abs().clip(lower=1e-8))
            df["oc_obv_momentum"]        = obv_ma7 - obv_ma30
            # FIX: OBV_Momentum del bridge histÃ³rico tiene 100% NaN (columna no en daemon parquet).
            # Rellenar con oc_obv_momentum calculado aquÃ­ (mismo concepto, datos completos).
            if "OBV_Momentum" not in df.columns or df["OBV_Momentum"].isna().mean() > 0.5:
                df["OBV_Momentum"] = df["oc_obv_momentum"]
                logger.info("OBV_Momentum backfilled con oc_obv_momentum (columna legacy inicialmente NaN)")


        # STH-SOPR proxy — precio actual / media rolling 90d
        if "close" in df.columns:
            close_s = df["close"]
            realized_90d = close_s.rolling(90*24).mean()
            df["oc_sth_sopr_proxy"] = close_s / (realized_90d.clip(lower=1e-8))

            # Puell Multiple proxy (halving-aware)
            HALVINGS = [
                (pd.Timestamp("2012-11-28", tz="UTC"), 25.0),
                (pd.Timestamp("2016-07-09", tz="UTC"), 12.5),
                (pd.Timestamp("2020-05-11", tz="UTC"), 6.25),
                (pd.Timestamp("2024-04-20", tz="UTC"), 3.125),
            ]
            reward_s = pd.Series(50.0, index=close_s.index)
            for halving_date, reward in HALVINGS:
                reward_s.loc[close_s.index >= halving_date] = reward
            daily_price    = close_s.resample("D").mean().reindex(close_s.index, method="ffill")
            daily_reward_h = reward_s
            daily_issuance = daily_price * daily_reward_h
            ma365h = daily_issuance.rolling(365*24).mean()
            pm_proxy = daily_issuance / (ma365h.clip(lower=1e-8))
            df["oc_puell_multiple_proxy"] = pm_proxy
            # Estacionaried para Machine Learning (SFI Phase 4)
            df["oc_puell_multiple_proxy_z90d"] = (pm_proxy - pm_proxy.rolling(90*24).mean()) / pm_proxy.rolling(90*24).std().clip(lower=1e-9)

        # ── [H4-ATH 2026-06-03] ATH Regime Features ──────────────────────────
        # Motivo: el modelo no distingue W3 (BTC ~60K, rally normal) de W4 (BTC
        # en ATH historico). HMM clasifica ambos como BULL_TREND_WEAK. El modelo
        # es 10% mas confiante en W4 pero el WR es 28pp peor (W3=74% vs W4=46%).
        # Estas features dan al XGBoost informacion causal sobre el sub-regimen ATH.
        # Causalidad garantizada: cummax() y rolling() solo usan datos hasta t.
        # SFI decidira si tienen DSR > 0 en IS. Sin hardcodes — todo desde close.
        if "close" in df.columns:
            _close = df["close"].ffill()

            # 1. Distancia al ATH historico acumulado hasta t [0, inf)
            #    0 = precio en ATH exacto; 0.20 = 20% por debajo del ATH
            _ath_cummax = _close.cummax()
            df["ath_dist_pct"] = (_ath_cummax - _close) / _ath_cummax.clip(lower=1e-8)
            print("[LUNA][H4-ATH] ath_dist_pct generada: distancia % al ATH historico acumulado")

            # 2. Racha de horas consecutivas haciendo nuevos ATH (vectorizado)
            #    0 en correccion, >0 en expansion de ATH (W4 tipicamente >700h)
            _new_ath_flag = (_close >= _ath_cummax.shift(1)).astype(int)
            # Vectorized streak: asignar un group_id cada vez que la racha se rompe
            _group_id = (_new_ath_flag == 0).cumsum()
            _streak = _new_ath_flag.groupby(_group_id).cumcount().where(_new_ath_flag == 1, 0).astype(float)
            df["ath_streak_h"] = _streak
            # Version normalizada Z90d para el SFI (estacionaria)
            _str_m = _streak.rolling(90 * 24, min_periods=24).mean()
            _str_s = _streak.rolling(90 * 24, min_periods=24).std().clip(lower=1e-9)
            df["ath_streak_z90d"] = (_streak - _str_m) / _str_s
            print(f"[LUNA][H4-ATH] ath_streak_h y ath_streak_z90d generadas (vectorizado): max_streak={_streak.max():.0f}h")


            # 3. Z-score del precio vs ultimo ano (~regimen historico de precio)
            #    >3 sigma = territorio sin precedentes en 252d (tipico de ATH explosive)
            _p_m252 = _close.rolling(252 * 24, min_periods=30 * 24).mean()
            _p_s252 = _close.rolling(252 * 24, min_periods=30 * 24).std().clip(lower=1e-9)
            df["price_z_score_252d"] = (_close - _p_m252) / _p_s252
            print("[LUNA][H4-ATH] price_z_score_252d generada: z-score precio vs ano anterior")

            # 4. Ratio de volatilidad realizada: vol_30d / vol_252d
            #    >1 = vol actual mayor que la historica (tipico en ATH con compresion)
            #    <1 = vol comprimida respecto al pasado (atencion antes de ruptura)
            _ret1h = _close.pct_change(1)
            _vol30d  = _ret1h.rolling(30 * 24,  min_periods=24).std()
            _vol252d = _ret1h.rolling(252 * 24, min_periods=30 * 24).std().clip(lower=1e-9)
            df["realized_vol_ratio"] = _vol30d / _vol252d
            # Version Z90d para mejor estacionariedad en el SFI
            _rvr = df["realized_vol_ratio"]
            _rvr_m = _rvr.rolling(90 * 24, min_periods=24).mean()
            _rvr_s = _rvr.rolling(90 * 24, min_periods=24).std().clip(lower=1e-9)
            df["realized_vol_ratio_z90d"] = (_rvr - _rvr_m) / _rvr_s
            print("[LUNA][H4-ATH] realized_vol_ratio y realized_vol_ratio_z90d generadas")

            _ath_feats_ok = [f for f in ["ath_dist_pct","ath_streak_h","ath_streak_z90d",
                                          "price_z_score_252d","realized_vol_ratio",
                                          "realized_vol_ratio_z90d"] if f in df.columns]
            logger.info(f"[H4-ATH] {len(_ath_feats_ok)} features ATH generadas: {_ath_feats_ok}")
            print(f"[LUNA][H4-ATH] TOTAL {len(_ath_feats_ok)} features ATH listas para SFI: {_ath_feats_ok}")

        # SFI Phase 4: ms_stablecoin_flow_30d
        if "Stablecoin_Cap" in df.columns:
            if "ms_stablecoin_flow_30d" not in df.columns:
                df["ms_stablecoin_flow_30d"] = df["Stablecoin_Cap"].pct_change(30 * 24)
                logger.info("ms_stablecoin_flow_30d generada en derivadas")
            if "ms_stablecoin_flow_yoy" not in df.columns:
                df["ms_stablecoin_flow_yoy"] = df["Stablecoin_Cap"].pct_change(365 * 24)


        # Cross-Asset (ca_*)
        if "ETH_BTC_Corr24H" in df.columns:
            mv = df["MVRV_Proxy"].ffill()
            df["oc_mvrv_pct_6m"] = mv.rolling(180*24).rank(pct=True)

        # ── Derivatives (dv_*) ──────────────────────────────────────────────

        # Funding rate derived
        for fr_col in ["FundingRate", "Funding_Rate", "funding_rate"]:
            if fr_col in df.columns:
                fr = df[fr_col].ffill()
                df["dv_funding_rate"]    = fr
                df["dv_funding_pct_90d"] = fr.rolling(90*24).rank(pct=True)
                
                # FASE 2: Funding Rate suavizado (Regimen Largo Plazo)
                df["FundingRate_30d_MA"] = fr.rolling(30*24).mean()
                df["FundingRate_ZScore_90d"] = (fr - fr.rolling(90*24).mean()) / fr.rolling(90*24).std().clip(lower=1e-9)
                break

        # [FIX-SKEW-01] ALIASES DE NOMBRE: historical_data_bridge.py renombra estas columnas
        # de snake_case (fetchers) a PascalCase durante el entrenamiento. El pipeline live
        # nunca aplicaba ese renombre -> training-serving skew en 7 features.
        # Causa confirmada: audit_live_full.py 2026-05-25.
        # FIX-SKEW-01 v2 (2026-05-25): el df live en ciclo incremental no tiene funding_ema_3
        # (columna generada por DerivativesFetcher en barras 8H, no disponible en 1H live).
        # Prioridad: funding_ema_3 > dv_funding_rate > FundingRate
        # Nota: span=3 en fetch_derivatives.py son 3 barras 8H = equivalente a ewm(span=24, min_periods=1) en 1H
        #
        # BUG 1/3: FundingRate_EMA3 y FundingRate_Pct90d
        _skew_fr_fixed = 0
        if "FundingRate_EMA3" not in df.columns:
            # Prioridad 1: alias directo desde funding_ema_3 (generado por DerivativesFetcher)
            if "funding_ema_3" in df.columns:
                df["FundingRate_EMA3"] = df["funding_ema_3"]
                _skew_fr_fixed += 1
                print("[FIX-SKEW-01] FundingRate_EMA3 <- funding_ema_3 (alias aplicado)")
            # Prioridad 2: derivar desde FundingRate o dv_funding_rate (siempre disponibles)
            else:
                _fr_src = df.get("FundingRate", df.get("dv_funding_rate"))
                if _fr_src is not None:
                    # span=24 en 1H ≡ span=3 en 8H (definición original del fetcher)
                    df["FundingRate_EMA3"] = _fr_src.ewm(span=24, min_periods=1, adjust=False).mean()
                    _skew_fr_fixed += 1
                    print("[FIX-SKEW-01] FundingRate_EMA3 derivada desde FundingRate.ewm(span=24h, 1H-bars) [equiv. span=3 en 8H-bars]")
                else:
                    print("[FIX-SKEW-01/WARN] FundingRate_EMA3: sin fuente disponible (funding_ema_3, FundingRate, dv_funding_rate)")

        if "FundingRate_Pct90d" not in df.columns:
            # Prioridad 1: alias directo desde funding_pct_90d
            if "funding_pct_90d" in df.columns:
                df["FundingRate_Pct90d"] = df["funding_pct_90d"]
                _skew_fr_fixed += 1
                print("[FIX-SKEW-01] FundingRate_Pct90d <- funding_pct_90d (alias aplicado)")
            # Prioridad 2: alias desde dv_funding_pct_90d
            elif "dv_funding_pct_90d" in df.columns:
                df["FundingRate_Pct90d"] = df["dv_funding_pct_90d"]
                _skew_fr_fixed += 1
                print("[FIX-SKEW-01] FundingRate_Pct90d <- dv_funding_pct_90d (alias aplicado)")
            # Prioridad 3: derivar desde FundingRate base
            else:
                _fr_src2 = df.get("FundingRate", df.get("dv_funding_rate"))
                if _fr_src2 is not None:
                    df["FundingRate_Pct90d"] = _fr_src2.rolling(90*24, min_periods=48).rank(pct=True)
                    _skew_fr_fixed += 1
                    print("[FIX-SKEW-01] FundingRate_Pct90d derivada desde FundingRate.rolling(90*24).rank(pct=True) [fallback]")
                else:
                    print("[FIX-SKEW-01/WARN] FundingRate_Pct90d: sin fuente disponible")

        if _skew_fr_fixed > 0:
            print(f"[FIX-SKEW-01] {_skew_fr_fixed} alias/derivadas aplicadas. FundingRate_EMA3={'✅' if 'FundingRate_EMA3' in df.columns else '❌'} | FundingRate_Pct90d={'✅' if 'FundingRate_Pct90d' in df.columns else '❌'}")


        # LongShortRatio histÃ³rico â€” Coinglass CSV como fuente primaria 2023+
        # Binance Futures L/S solo disponible desde 2025â†’ no cubre training (2020-2024)
        # Coinglass CSV cubre 2023-05-04â†’2026-02 (1000 filas diarias)
        if "LongShortRatio" not in df.columns or df["LongShortRatio"].isna().mean() > 0.9:
            cg_csv = _ROOT / "data" / "historical" / "correlaciones" / "coinglass_long_short.csv"
            if cg_csv.exists():
                try:
                    cg = pd.read_csv(cg_csv, parse_dates=["date"])
                    cg["date"] = pd.to_datetime(cg["date"], utc=True)
                    cg = cg.set_index("date").sort_index()
                    if "long_short_ratio" in cg.columns:
                        lsr_daily = cg["long_short_ratio"].dropna()
                        if len(lsr_daily) > 0:
                            # Expandir de diario a horario via ffill
                            lsr_1h = lsr_daily.resample("1h").last().reindex(df.index, method="ffill")
                            if "LongShortRatio" in df.columns:
                                df["LongShortRatio"] = df["LongShortRatio"].fillna(lsr_1h)
                            else:
                                df["LongShortRatio"] = lsr_1h
                            logger.info(f"LongShortRatio completado con Coinglass CSV ({len(lsr_daily)} dias)")
                except Exception as e:
                    logger.warning(f"Coinglass CSV no cargado: {e}")

        # LongAccount/ShortAccount — derivar desde LongShortRatio
        # LSR = LongAccount / ShortAccount, LongAccount + ShortAccount = 1
        # → LongAccount = LSR / (1 + LSR), ShortAccount = 1 / (1 + LSR)
        for lsr_col in ["LongShortRatio", "ls_ratio"]:
            if lsr_col in df.columns:
                lsr = df[lsr_col].ffill()
                nan_la = df["LongAccount"].isna().mean() if "LongAccount" in df.columns else 1.0
                if nan_la > 0.5:
                    # [P2-3-FIX] Validación de anomalías en Coinglass (ej. si el csv trae % long en lugar de ratio)
                    if lsr.max() > 5.0:
                        logger.warning(f"[P2-3-FIX] LongShortRatio sospechosamente alto (max={lsr.max():.1f} > 5). Verificar csv.")
                    df["LongAccount"]  = lsr / (1.0 + lsr.clip(lower=1e-9))
                    df["ShortAccount"] = 1.0 / (1.0 + lsr.clip(lower=1e-9))
                    # Coinglass ratios como alias del mismo cálculo (para backward compat)
                    df["Coinglass_long_ratio"]  = df["LongAccount"]
                    df["Coinglass_short_ratio"] = df["ShortAccount"]
                    logger.info("LongAccount/ShortAccount/Coinglass_ratios derivados desde LongShortRatio")
                break


        # [FIX-SKEW-02] BUG 2/3: OI_High_USD, OI_Low_USD, OI_Open_USD
        # bridge.py: 'oi_open' -> 'OI_Open_USD', 'oi_high' -> 'OI_High_USD', 'oi_low' -> 'OI_Low_USD'
        # En el parquet live estos viven como 'Coinglass_oi_open/high/low' (nombres del fetcher)
        _oi_skew_map = {
            "OI_Open_USD": ["Coinglass_oi_open", "oi_open"],
            "OI_High_USD": ["Coinglass_oi_high", "oi_high"],
            "OI_Low_USD":  ["Coinglass_oi_low",  "oi_low"],
        }
        for canonical, sources in _oi_skew_map.items():
            if canonical not in df.columns:
                for src in sources:
                    if src in df.columns:
                        df[canonical] = df[src].ffill()
                        print(f"[FIX-SKEW-02] {canonical} <- {src} (alias aplicado)")
                        break
                else:
                    print(f"[FIX-SKEW-02/WARN] {canonical}: ninguna fuente encontrada {sources}")

        # [FIX-SKEW-03] BUG 3/3: ETF_Flow_Proxy y dv_etf_flow_proxy
        # bridge.py: 'etf_flow_proxy' -> 'ETF_Flow_Proxy'
        # En el parquet live existe como 'etf_flow_proxy' (snake_case del fetcher)
        if "ETF_Flow_Proxy" not in df.columns and "etf_flow_proxy" in df.columns:
            df["ETF_Flow_Proxy"] = df["etf_flow_proxy"]
            print("[FIX-SKEW-03] ETF_Flow_Proxy <- etf_flow_proxy (alias aplicado)")
        elif "ETF_Flow_Proxy" not in df.columns:
            print("[FIX-SKEW-03/WARN] ETF_Flow_Proxy: ni alias snake_case ni columna base encontrados")

        # ETF Flow proxy (ya existe como ETF_Flow_Proxy, añadimos dv_ alias)
        if "ETF_Flow_Proxy" in df.columns and "dv_etf_flow_proxy" not in df.columns:
            df["dv_etf_flow_proxy"] = df["ETF_Flow_Proxy"]
            print("[FIX-SKEW-03] dv_etf_flow_proxy <- ETF_Flow_Proxy (alias aplicado)")

        # SFI Phase 5: OI Acceleration
        if "OI_BTC" in df.columns:
            oi_m = df["OI_BTC"].ffill()
            dv_oi_velocity_24h = oi_m.diff(24)
            df["dv_oi_acceleration_24h"] = dv_oi_velocity_24h.diff(24)
            logger.info("OI Acceleration (dv_oi_acceleration_24h) generada")

        # ── DVOL derived (Volatilidad Implícita Deribit) ────────────────────────
        if "DVOL" in df.columns:
            dvol = df["DVOL"].ffill()
            df["dv_dvol_raw"] = dvol
            df["dv_dvol_pct_24h"] = dvol.pct_change(24)
            df["dv_dvol_z7d"] = (dvol - dvol.rolling(7*24).mean()) / dvol.rolling(7*24).std().clip(lower=1e-9)
            df["dv_dvol_z30d"] = (dvol - dvol.rolling(30*24).mean()) / dvol.rolling(30*24).std().clip(lower=1e-9)
            
            # SFI Phase 5: VRP (Variance Risk Premium)
            if "close" in df.columns:
                realized_vol_30d = df["close"].pct_change(1).rolling(30*24).std() * np.sqrt(365*24) * 100
                df["dv_vrp_30d"] = dvol - realized_vol_30d
            logger.info("DVOL features derivadas (dv_dvol_*) generadas")

        # ── Micro-Structure (ms_*) ──────────────────────────────────────────────
        if "volume" in df.columns and "taker_buy_base" in df.columns:
            vol = df["volume"]
            taker_buy = df["taker_buy_base"]
            taker_sell = vol - taker_buy
            # Taker Ratio: agresividad de mercado (buy vs sell)
            taker_ratio = taker_buy / taker_sell.replace(0, np.nan)
            
            df["ms_taker_buy_vol"] = taker_buy
            df["ms_taker_sell_vol"] = taker_sell
            df["ms_taker_ratio_1h"] = taker_ratio
            df["ms_taker_ratio_4h"] = taker_ratio.rolling(4).mean()
            df["ms_taker_ratio_12h"] = taker_ratio.rolling(12).mean()
            df["ms_taker_ratio_24h"] = taker_ratio.rolling(24).mean()
            
            # SFI Phase 4: CVD Spot vs Perps
            if "Futures_Volume" in df.columns and "Futures_Taker_Buy" in df.columns:
                fut_vol = df["Futures_Volume"]
                fut_buy = df["Futures_Taker_Buy"]
                # [M3-FIX] Verificar que las columnas no sean 100% NaN (API puede fallar)
                _cvd_nan_pct = max(fut_vol.isna().mean(), fut_buy.isna().mean())
                if _cvd_nan_pct > 0.90:
                    logger.warning(
                        f"[M3-CVD] ms_cvd_spot_vs_perps NO generada: Futures_Volume/Futures_Taker_Buy "
                        f"son {_cvd_nan_pct:.0%} NaN. El endpoint de futuros no esta sirviendo datos. "
                        f"Verificar conector Binance Futures o Coinglass API. "
                        f"XGBoost manejara NaN internamente pero la señal CVD se pierde."
                    )
                else:
                    fut_sell = fut_vol - fut_buy
                    taker_ratio_fut = fut_buy / fut_sell.replace(0, np.nan)
                    df["ms_futures_taker_ratio_12h"] = taker_ratio_fut.rolling(12).mean()
                    df["ms_cvd_spot_vs_perps"] = df["ms_taker_ratio_12h"] - df["ms_futures_taker_ratio_12h"]
                    logger.info("CVD Spot vs Perps (ms_cvd_spot_vs_perps) generado a partir de Taker Volume de Spot y Perps")
            else:
                logger.warning(
                    "[M3-CVD] ms_cvd_spot_vs_perps NO generada: columnas Futures_Volume o Futures_Taker_Buy "
                    "ausentes del DataFrame. Verificar que el fetcher de derivados está activo y el "
                    "endpoint de Binance Futures (GET /fapi/v1/trades) está funcionando."
                )
                logger.info("Micro-Structure features (ms_*) generadas: Taker Ratios desde crudo L1")

        # ── Timing Short-term Features (timing_*) ──
        if "FundingRate" in df.columns:
            df["timing_funding_acum8h"] = df["FundingRate"].ewm(span=8, min_periods=1).mean()

        if "close" in df.columns:
            ret_24h = df["close"].pct_change(24)
            ret_7d  = df["close"].pct_change(168)
            df["timing_momentum_div"] = ret_24h - ret_7d

        if "close" in df.columns and "volume" in df.columns:
            ret_24h_abs = df["close"].pct_change(24).abs()
            vol_ma_30d  = df["volume"].rolling(window=720, min_periods=48).mean()
            vol_ratio   = df["volume"] / (vol_ma_30d + 1e-6)
            df["timing_vol_divergence"] = (ret_24h_abs / (vol_ratio + 1e-6)).clip(upper=5.0)
            
        # ── Momentum multi-TF (mt_*) ──────────────────────────────────────────
        if "close" in df.columns:
            close_s = df["close"]
            df["mt_momentum_1h"] = close_s.pct_change(1)   # retorno 1H
            if "mt_momentum_4h" not in df.columns:
                df["mt_momentum_4h"] = close_s.pct_change(4)

        # â”€â”€ Calendar extras (cal_*) â€” completar gap vs Luna v2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # SeÃ±ales de periodicidad semanal/mensual documentadas
        if isinstance(df.index, pd.DatetimeIndex):
            df["cal_is_monday"]    = (df.index.dayofweek == 0).astype(int)
            df["cal_is_first_week"] = (df.index.day <= 7).astype(int)

        logger.info(f"apply_derived_features: {sum(1 for c in df.columns if c.startswith(('mc_','oc_','dv_')))} mc_/oc_/dv_ features generadas")

        # â”€â”€ MI-Lag Features (lags optimos descubiertos por Feature Selection) â”€â”€
        # El SFI (Etapa C - Lag Discovery MI) detecta que ciertas features tienen
        # su maxima causalidad con BTC a un lag distinto del Granger conocido.
        # Estos lags se persisten aqui para que XGBoost los consuma directamente.
        # IMPORTANTE: si el SFI produce un nombre 'X_milagNh', debe existir aqui.
        MI_LAG_FEATURES = {
            # feature_source: (lag_horas, nombre_output)
            'M2_MoM_Chg_z90d': (120, 'M2_MoM_Chg_z90d_milag120h'),  # MI=120H vs Granger=1008H
            # ETH_Return_1d en lugar de ETH_Price (precio bruto con varianza ~0 en 1H ffill)
            'ETH_Return_1d':    (96,  'ETH_Return_1d_milag96h'),      # MI=96H vs Granger=312H
            # FIX TEST-116 (2026-03-11): CPI_YoY en selected_features.json con lag 24H
            'CPI_YoY':         (24,  'CPI_YoY_milag24h'),            # MI=24H (timing publicacion CPI)
            # FIX RUN-0035 (2026-03-25): mc_unemploy_rate_delta_3m_z90d
            'mc_unemploy_rate_delta_3m_z90d': (24, 'mc_unemploy_rate_delta_3m_z90d_milag24h'),

            # [FIX-B3] LAGS EXACTOS DEL LGBM BULL V1 (dsr_oos=0.248, lgbm_prob_media=0.7884)
            # Evidencia: lgbm_meta_bull_long_signature.json seed777 W1
            # Todas las columnas base EXISTEN en V2 -- solo faltaban los lags especificos.
            'cal_day_of_week':   (72,  'cal_day_of_week_milag72h'),    # Ciclo semanal con lag 3d
            'ETH_Return_1d':     (336, 'ETH_Return_1d_milag336h'),     # ETH momentum 14d
            'NASDAQ_Ret':        (12,  'NASDAQ_Ret_milag12h'),          # Macro equities 12H lag
            'M2_YoY_Chg_z90d':  (72,  'M2_YoY_Chg_z90d_milag72h'),   # Liquidez M2 con 3d lag
            'ECBASSETS':         (336, 'ECBASSETS_milag336h'),          # Balance BCE 14d lag
            'SP500_Ret':         (72,  'SP500_Ret_milag72h'),           # S&P 500 retorno 3d lag
            'DXY_Slope30d_z90d': (72,  'DXY_Slope30d_z90d_milag72h'), # Tendencia DXY 3d lag
            'UnemployRate_z90d': (1,   'UnemployRate_z90d_milag1h'),   # Desempleo normalizado 1H
            'pi_cycle_ma111':    (72,  'pi_cycle_ma111_milag72h'),      # Pi Cycle indicator 3d
            'EURUSD':            (500, 'EURUSD_milag500h'),             # EUR/USD tipo de cambio
            'GBTC_Low':          (48,  'GBTC_Low_milag48h'),            # GBTC minimo 2d lag
            'CPI_YoY':           (1,   'CPI_YoY_milag1h'),              # CPI publicacion 1H lag

            # [FIX-B7] Features bear-especificas: el SFI global tiene sesgo bull.
            # Estas features tienen maxima causalidad en mercados bajistas (evidencia V1).
            # Se inyectan como candidatos adicionales para que el agente BEAR las pueda usar.
            'FearGreed':          (48,  'FearGreed_milag48h'),            # Miedo extremo precede recuperacion
            'Stablecoin_Cap':     (72,  'Stablecoin_Cap_milag72h'),       # Acumulacion estables antes de rebote
            'MVRV_Proxy_z90d':   (168, 'MVRV_Proxy_z90d_milag168h'),     # MVRV<1 = zona de capitulacion
            'oc_nvt_proxy_z90d': (120, 'oc_nvt_proxy_z90d_milag120h'),   # NVT alto = sobrevaloracion bear
            'realized_vol_ratio': (48, 'realized_vol_ratio_milag48h'),   # Aceleracion vol = bear confirmation
            'btc_drawdown_from_ath': (24, 'btc_drawdown_from_ath_milag24h'), # Distancia ATH = profundidad bear

            # [FIX-O4-MISSING-MILAGS] 4 features solicitadas por TODOS los modelos XGBoost
            # (seed99, seed1337, seed2025) en sus *_signature.json pero ausentes en el
            # pipeline live. Las features base existen con 0% NaN en el parquet live.
            # Causa raiz: los shifts se entrenaron pero no se registraron en MI_LAG_FEATURES.
            # Impacto corregido: elimina 4 WARNINGs por ciclo en el stderr de regime_router.py.
            # Verificado: shift() positivo -> solo usa datos del pasado, sin look-ahead bias (R1).
            'DXY_z90d':              (96,  'DXY_z90d_milag96h'),              # DXY z-score 90d con lag 96H (~4 dias)
            'Whale_Proxy_Volume_USD': (500, 'Whale_Proxy_Volume_USD_milag500h'), # Volumen ballenas lag 500H (~20 dias)
            'ms_stablecoin_flow_30d': (12,  'Stablecoins_Delta_30d_milag12h'), # Delta stablecoins 30d lag 12H
            'CPI_YoY_kz':            (48,  'CPI_YoY_kz_milag48h'),            # CPI YoY Kalman-z lag 48H (~2 dias)
        }

        for src_col, (lag_h, out_col) in MI_LAG_FEATURES.items():
            if src_col in df.columns and out_col not in df.columns:
                df[out_col] = df[src_col].shift(lag_h)
                logger.debug(f"MI-lag: {src_col} (lag={lag_h}H) -> {out_col}")

        # [FIX WFB Strict Causality] Auto-generar cualquier lag dictado por SFI que no este hardcodeado
        try:
            sf_path = _ROOT / "data" / "features" / "selected_features.json"
            parquet_path = _ROOT / "data" / "features" / "features_train.parquet"
            if sf_path.exists():
                import json as _json_lag, os as _os_lag
                # [FIX-SCHEMA-01] Detectar artefacto obsoleto: si el parquet es mas reciente
                # que el selected_features.json, el JSON es de un run anterior y sus lags
                # podrian referirse a features que ya no existen en el parquet actual.
                if parquet_path.exists():
                    _sf_mtime   = _os_lag.path.getmtime(sf_path)
                    _pqt_mtime  = _os_lag.path.getmtime(parquet_path)
                    if _pqt_mtime > _sf_mtime + 300:  # >5min de diferencia = artefacto obsoleto
                        import datetime as _dt_lag
                        _sf_age  = _dt_lag.datetime.fromtimestamp(_sf_mtime).strftime('%Y-%m-%d %H:%M')
                        _pqt_age = _dt_lag.datetime.fromtimestamp(_pqt_mtime).strftime('%Y-%m-%d %H:%M')
                        logger.warning(
                            f"[FIX-SCHEMA-01] selected_features.json ({_sf_age}) es MAS ANTIGUO "
                            f"que features_train.parquet ({_pqt_age}). "
                            f"Los lags del JSON pueden no existir en el parquet actual. "
                            f"Esto se resolvera automaticamente cuando el SFI actual complete y "
                            f"genere un nuevo selected_features.json para esta ventana."
                        )
                        print(f"[FP][WARN][FIX-SCHEMA-01] selected_features.json obsoleto -- los lags dinamicos se omiten hasta que el SFI regenere el JSON.")
                with open(sf_path, "r") as f:
                    _d = _json_lag.load(f)
                    _req = _d.get("selected_features", []) + _d.get("pass_through_features", [])
                _lags_inyectados = 0
                _lags_fallidos   = []
                for col in _req:
                    if "_milag" in col and col.lower().endswith("h"):
                        try:
                            parts = col.split("_milag")
                            current_src = parts[0]
                            for lag_str in parts[1:]:
                                if not lag_str.lower().endswith("h"):
                                    break
                                lag = int(lag_str[:-1])
                                next_col = f"{current_src}_milag{lag_str}"
                                if current_src in df.columns and next_col not in df.columns:
                                    df[next_col] = df[current_src].shift(lag)
                                    logger.info(f"MI-lag Dinamico inyectado (Cascade Fix): {current_src} (lag={lag}H) -> {next_col}")
                                    _lags_inyectados += 1
                                elif current_src not in df.columns:
                                    _lags_fallidos.append(f"{col} (base '{current_src}' no en df)")
                                current_src = next_col
                        except Exception as _e_lag:
                            logger.warning(f"[FIX-SCHEMA-01] Error inyectando MI-lag para '{col}': {_e_lag}")
                if _lags_fallidos:
                    logger.warning(
                        f"[FIX-SCHEMA-01] {len(_lags_fallidos)} MI-lags NO inyectados (base ausente en parquet): "
                        f"{_lags_fallidos[:5]}" + (" ..." if len(_lags_fallidos) > 5 else "")
                    )
                    print(f"[FP][WARN][FIX-SCHEMA-01] MI-lags no inyectados: {_lags_fallidos}")
                elif _lags_inyectados > 0:
                    logger.info(f"[FIX-SCHEMA-01] {_lags_inyectados} MI-lags dinamicos inyectados correctamente.")
        except Exception as e:
            logger.warning(f"[FIX-SCHEMA-01] Error en bloque MI-lag cascade: {e}")

        # â”€â”€ HashRate_14dMA: complementar con onchain_raw si mempool_raw incompleto â”€â”€
        # mempool_raw.HashRate_14dMA arranca en mar-2021 (NaN=23.8% en training).
        # onchain_raw.Hashrate_7dMA cubre desde 2020-01-01 con 0% NaN.
        # Estrategia: si HashRate_14dMA tiene NaN, rellenar con Hashrate_7dMA (fuente alternativa).
        if 'HashRate_14dMA' in df.columns and 'Hashrate_7dMA' in df.columns:
            nan_before = df['HashRate_14dMA'].isna().mean()
            df['HashRate_14dMA'] = df['HashRate_14dMA'].fillna(df['Hashrate_7dMA'])
            nan_after  = df['HashRate_14dMA'].isna().mean()
            if nan_before > nan_after + 0.001:
                logger.info(f"HashRate_14dMA complementado con Hashrate_7dMA: NaN {nan_before:.1%} -> {nan_after:.1%}")
        elif 'HashRate_14dMA' not in df.columns and 'Hashrate_7dMA' in df.columns:
            # Si la columna del mempool no existe, crear desde onchain directamente
            hr = df['Hashrate_7dMA'].ffill()
            df['HashRate_14dMA'] = hr.rolling(14 * 24, min_periods=1).mean()
            logger.info("HashRate_14dMA creado desde Hashrate_7dMA (rolling 14d)")

        # â”€â”€ P4-1-1: 5 Features de DetecciÃ³n de RÃ©gimen Actual â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # DiagnÃ³stico Run9: el modelo no detectaba el cambio de rÃ©gimen 2025.
        # Estas features capturan el CONTEXTO DE MERCADO ACTUAL (no solo histÃ³rico).
        if "close" in df.columns:
            close_r = df["close"].ffill()

            # 1. btc_trend_regime: ratio EMA50H / EMA200H â€” tendencia estructural
            #    >0 = precio con momentum alcista, <0 = momentum bajista
            ema50  = close_r.ewm(span=50,  adjust=False).mean()
            ema200 = close_r.ewm(span=200, adjust=False).mean()
            df["btc_trend_regime"] = (ema50 / ema200.clip(lower=1e-8)) - 1.0

            # 2. vol_percentile_90d: Percentil de la volatilidad EWMA en ventana 90d
            #    0 = vol en mÃ­nimos histÃ³ricos (calm), 1 = vol en mÃ¡ximos (crisis)
            ewm_vol = close_r.pct_change().ewm(span=24, adjust=False).std()
            df["vol_percentile_90d"] = ewm_vol.rolling(90 * 24).rank(pct=True)

            # 3. realized_vol_ratio: vol_7d / vol_30d â€” aceleraciÃ³n de volatilidad
            #    >1 = volatilidad subiendo (incertidumbre), <1 = calmÃ¡ndose
            vol_7d  = close_r.pct_change().rolling(7 * 24).std()
            vol_30d = close_r.pct_change().rolling(30 * 24).std().clip(lower=1e-9)
            df["realized_vol_ratio"] = vol_7d / vol_30d

            # 4. btc_drawdown_from_ath: distancia desde el ATH de 90d
            #    0 = en mÃ¡ximos recientes, -0.30 = 30% por debajo del ATH de 90d
            ath_90d = close_r.rolling(90 * 24).max().clip(lower=1e-8)
            df["btc_drawdown_from_ath"] = (close_r / ath_90d) - 1.0

            # 5. btc_cycle_position: percentil del precio en ventana 365d (MEJORA-B01)
            #    0.0 = mÃ­nimo anual (bottom de ciclo), 1.0 = mÃ¡ximo anual (techo de ciclo)
            #    Captura la posiciÃ³n en el ciclo BTC â€” Ãºtil para discriminar bull-trend real
            #    de rally dentro de bajista. Se persiste aquÃ­ para que OOS generator la tenga.
            _rolling_365 = close_r.rolling(window=8760, min_periods=720)
            _ath_365 = _rolling_365.max()
            _min_365 = _rolling_365.min()
            _rng_365 = (_ath_365 - _min_365).replace(0, float("nan"))
            df["btc_cycle_position"] = ((close_r - _min_365) / _rng_365).clip(0.0, 1.0)

            logger.info("Regime features (P4-1-1): btc_trend_regime, vol_percentile_90d, "
                        "realized_vol_ratio, btc_drawdown_from_ath, btc_cycle_position")

        # 5. funding_regime: Percentil 30d del funding rate â€” sesgo del mercado retail
        #    >0.8 = mercado muy largo (contrarian bear signal), <0.2 = muy corto (contrarian bull)
        for fr_col in ["FundingRate", "Funding_Rate", "funding_rate", "dv_funding_rate"]:
            if fr_col in df.columns:
                fr = df[fr_col].ffill()
                df["funding_regime"] = fr.rolling(30 * 24).rank(pct=True)
                logger.info("Regime feature (P4-1-1): funding_regime")
                break

        # ── [HMM-FIX-ZSCORE-OOS 2026-06-07] Generación Global de Features HMM ───────────
        try:
            hmm_z_cols = ['btc_drawdown_from_ath', 'mt_vol_realized_4bar']
            for col in hmm_z_cols:
                if col in df.columns:
                    z_col = f"{col}_z90d"
                    if z_col not in df.columns:
                        import numpy as np
                        _roll_mean = df[col].rolling(window=2160, min_periods=24).mean()
                        _roll_std  = df[col].rolling(window=2160, min_periods=24).std().replace(0, np.nan).bfill()
                        df[z_col] = (df[col] - _roll_mean) / _roll_std
                        df[z_col] = df[z_col].fillna(0.0)
                        logger.info(f"[HMM-FIX-ZSCORE-OOS] Transformación {z_col} inyectada globalmente.")
        except Exception as _e_hmmz:
            logger.warning(f"[HMM-FIX-ZSCORE-OOS] Error inyectando variables z90d globalmente: {_e_hmmz}")

        # ── [P2] HMM Transition Velocity (2026-04-09) ──────────────────────────────
        # Derivada suavizada de la probabilidad HMM del régimen BULL.
        # Convierte el vector de probabilidades del HMM (normalmente sólo categórico)
        # en una señal continua que captura la VELOCIDAD DE ROTACIÓN de régimen.
        # Key insight: cuando P(BULL) cae rápidamente (velocidad<0), el modelo debe
        # ser más conservador aunque el HMM aún no haya cambiado de etiqueta.
        #
        # Causalidad garantizada:
        # - El HMM se carga desde el pkl IS (entrenado hasta train_cutoff)
        # - predict_proba se ejecuta sobre toda la serie, pero el modelo IS nunca "ve" el futuro
        # - El diff() opera sólo sobre el pasado causal en cada timestep
        # - El EWM de suavizado (span=24h) también es causal (adjust=False)
        try:
            _hmm_pkl = _ROOT / "data" / "models" / "hmm_regime.pkl"
            if _hmm_pkl.exists():
                import joblib as _jl
                _hmm_data = _jl.load(_hmm_pkl)
                _hmm_model = _hmm_data.get("model", None)
                _hmm_scaler = _hmm_data.get("scaler", None)
                _hmm_features = _hmm_data.get("features", None)
                _state_map = _hmm_data.get("state_map", {})

                # Leer span de suavizado desde settings (default 24h)
                try:
                    from config.settings import cfg as _cfg_hmm_v
                    _vel_span = int(getattr(_cfg_hmm_v.xgboost, 'hmm_velocity_ewm_span', 24))
                except Exception:
                    _vel_span = 24

                if _hmm_model is not None and _hmm_scaler is not None and _hmm_features is not None:
                    # Preparar features HMM disponibles en el df actual
                    _hmm_feats_avail = [f for f in _hmm_features if f in df.columns]
                    if len(_hmm_feats_avail) == len(_hmm_features):  # requiere todas las features del modelo
                        _X_hmm = df[_hmm_feats_avail].ffill().fillna(0).values
                        _X_scaled = _hmm_scaler.transform(_X_hmm)

                        # predict_proba: shape (n_timesteps, n_states)
                        _proba = _hmm_model.predict_proba(_X_scaled)  # range [0,1] por estado

                        # Identificar el índice del estado BULL (1_BULL_TREND o similar)
                        _bull_state_idx = None
                        for _s_idx, _s_label in _state_map.items():
                            if str(_s_label).startswith("1_BULL"):
                                _bull_state_idx = int(_s_idx)
                                break

                        if _bull_state_idx is not None and _bull_state_idx < _proba.shape[1]:
                            _p_bull = pd.Series(_proba[:, _bull_state_idx], index=df.index, name="hmm_prob_bull")
                            # Velocidad: diff(1) + suavizado EWM causal (adjust=False garantiza causalidad)
                            _vel = _p_bull.diff(1).ewm(span=_vel_span, adjust=False).mean()
                            df["hmm_velocity_bull"] = _vel
                            # Aceleración (segunda derivada suavizada) para capturar inflexiones
                            df["hmm_acceleration_bull"] = _vel.diff(1).ewm(span=_vel_span, adjust=False).mean()
                            logger.info(
                                f"[P2] HMM Transition Velocity generada: hmm_velocity_bull, "
                                f"hmm_acceleration_bull | span={_vel_span}H | estado BULL={_bull_state_idx}"
                            )
                        else:
                            logger.warning("[P2] HMM Velocity: no se encontró estado BULL en state_map — feature omitida")
                    else:
                        logger.warning(f"[P2] HMM Velocity: solo {len(_hmm_feats_avail)} de {len(_hmm_features)} features disponibles en df")
            else:
                logger.debug("[P2] HMM Velocity: hmm_regime.pkl no encontrado — feature omitida (normal en primer run)")
        except Exception as _e_p2:
            logger.warning(f"[P2] HMM Velocity falló silenciosamente (no crítico): {_e_p2}")

        # ── [P4] Macro-OHL Tension Index (2026-04-09) ──────────────────────────────
        # Índice compuesto que articula la divergencia entre señal técnica (precio alcista)
        # y fundamentales (liquidez fiat, valoración on-chain, dólar fuerte).
        #
        # Fórmula: macro_ohl_tension = NVT_z × |M2_trend| × sign(-DXY_slope)
        # Truncado a [-5, 5] para evitar outliers en periodos de crisis extrema.
        # Alta tensión (>1σ): divergencia técnica/fundamental → XGBoost penaliza entradas Long.
        # Baja tensión (<0): confluencia macro + técnica → máxima convicción.
        #
        # Robustez: todos los componentes pasan por z-score rolling 90d antes de combinarse,
        # lo que hace el índice invariante ante cambios de escala absoluta entre ciclos BTC.
        try:
            _tension_components_available = []

            # Componente 1: NVT proxy (sobrevaloración on-chain relativa)
            # oc_nvt_proxy_z90d ya calculado en apply_derived_features
            _nvt_series = None
            if "oc_nvt_proxy_z90d" in df.columns:
                _nvt_series = df["oc_nvt_proxy_z90d"].ffill()
                _tension_components_available.append("nvt")

            # Componente 2: M2 trend magnitude (retiro/inyección de liquidez fiat)
            # Usar M2_MoM_Chg_z90d_milag120h si está disponible (ya tiene safety lag correcto)
            _m2_series = None
            for _m2_cand in ["M2_MoM_Chg_z90d_milag120h", "M2_MoM_Chg_z90d", "M2_MoM_Chg"]:
                if _m2_cand in df.columns:
                    _m2_raw = df[_m2_cand].ffill()
                    # Normalizar por ventana 90d para hacerlo invariante entre ciclos
                    _m2_z = (_m2_raw - _m2_raw.rolling(90 * 24).mean()) / _m2_raw.rolling(90 * 24).std().clip(lower=1e-9)
                    _m2_series = _m2_z.abs()  # magnitud: nos interesa la fuerza del movimiento M2
                    _tension_components_available.append("m2")
                    break

            # Componente 3: DXY slope (dólar fuerte = negativo para crypto)
            # sign(-DXY_slope) > 0 cuando el dólar se debilita (favorable crypto)
            _dxy_series = None
            if "DXY_Ret" in df.columns:
                _dxy_slope = df["DXY_Ret"].ewm(span=96, adjust=False).mean()  # EWM 4 días
                _dxy_series = np.sign(-_dxy_slope)  # +1 si DXY bajando (crypto-favorable)
                _tension_components_available.append("dxy")
            elif "DXY" in df.columns:
                _dxy_raw = df["DXY"].ffill()
                _dxy_slope = _dxy_raw.pct_change(96)
                _dxy_series = np.sign(-_dxy_slope)
                _tension_components_available.append("dxy")

            if len(_tension_components_available) >= 2:
                # Construir el índice con los componentes disponibles
                _tension = pd.Series(1.0, index=df.index)  # neutral
                if _nvt_series is not None:
                    _tension = _tension * _nvt_series.ffill().fillna(1.0)
                if _m2_series is not None:
                    _tension = _tension * _m2_series.ffill().fillna(1.0)
                if _dxy_series is not None:
                    _tension = _tension * pd.Series(_dxy_series, index=df.index).ffill().fillna(1.0)

                # Z-score final rolling 90d para estandarizar la magnitud del índice compuesto
                _t_mean = _tension.rolling(90 * 24, min_periods=24 * 7).mean()
                _t_std  = _tension.rolling(90 * 24, min_periods=24 * 7).std().clip(lower=1e-9)
                df["macro_ohl_tension_z"] = ((_tension - _t_mean) / _t_std).clip(-5, 5)

                logger.info(
                    f"[P4] Macro-OHL Tension Index generado: macro_ohl_tension_z | "
                    f"componentes={_tension_components_available}"
                )
            else:
                logger.warning(
                    f"[P4] Macro-OHL Tension: solo {len(_tension_components_available)} componentes disponibles "
                    f"(mín 2). Feature omitida. Verificar NVT/M2/DXY en los datos raw."
                )
        except Exception as _e_p4:
            logger.warning(f"[P4] Macro-OHL Tension falló silenciosamente (no crítico): {_e_p4}")

        # ── [DXY-HMM-01 2026-06-03] DXY Condicional al Régimen HMM ──────────────────────
        # MOTIVACIÓN: El DXY tiene ambigüedad de signo según el régimen de mercado.
        # - Bull/Risk-On (HMM alto): DXY↑ → BTC↓ (correlación negativa clara)
        # - Bear/Risk-Off (HMM bajo): la correlación se debilita o invierte
        #   (deleveraging forzado, todos los activos caen juntos)
        # XGBoost no puede modelar esta interacción sin features de árbol muy profundos.
        # La feature condicional pre-calcula la señal ajustada al régimen.
        #
        # CAUSALIDAD: HMM_Regime usa Forward Algorithm (no Viterbi) — sin look-ahead (R1).
        # Disponible desde integrate_mining_outputs (Paso 3B) — antes de este paso (7B).
        #
        # FEATURES GENERADAS:
        # - DXY_HMM_cond:     DXY_Ret × regime_sign  → positivo cuando DXY es bearish para BTC
        # - DXY_HMM_interact: DXY_Zscore × HMM_Regime → interacción suave para árboles
        # - DXY_HMM_bull_neg: DXY_Ret × bull_flag × (-1) → señal pura en régimen bull
        try:
            _has_dxy_ret   = "DXY_Ret"    in df.columns
            _has_dxy_z     = "DXY_Zscore" in df.columns or "DXY_z90d" in df.columns
            _has_hmm       = "HMM_Regime" in df.columns
            _n_states_dxh  = 5  # Estados HMM esperados (0=crash, 4=bull)

            if _has_dxy_ret and _has_hmm:
                _dxy_ret = df["DXY_Ret"].ffill()
                _hmm_reg = df["HMM_Regime"].fillna(df["HMM_Regime"].median())

                # Normalizar régimen a [-1, +1]: 0=crash(bear) → -1, 4=bull → +1
                _regime_norm = (_hmm_reg / (_n_states_dxh - 1)) * 2 - 1  # [-1, +1]

                # Feature 1: Señal condicional — DXY amplificado por régimen
                # En bull (+1): DXY_Ret puro (correlación negativa con BTC activa)
                # En crash (-1): DXY_Ret invertido (correlación menos confiable)
                df["DXY_HMM_cond"] = _dxy_ret * (-_regime_norm)
                # Interpretación: positivo → condición bearish para BTC (DXY sube en bull)
                #                 negativo → condición bullish para BTC

                # Feature 2: Bull-flag binario × DXY — más interpretable para el árbol
                _bull_flag = (_hmm_reg >= _n_states_dxh // 2).astype(float)  # 1 en estados >= 2
                df["DXY_HMM_bull_neg"] = _dxy_ret * _bull_flag * (-1.0)
                # En régimen bull: -DXY_Ret (cae cuando DXY sube → señal bearish)
                # En régimen bear: 0 (no hay señal confiable)

                # Feature 3: Interacción suave (continua) para gradiente más suave
                if _has_dxy_z:
                    _dxy_z_col = "DXY_Zscore" if "DXY_Zscore" in df.columns else "DXY_z90d"
                    _dxy_z = df[_dxy_z_col].ffill()
                    df["DXY_HMM_interact"] = _dxy_z * _regime_norm
                    n_interact = df["DXY_HMM_interact"].notna().sum()
                    logger.debug(f"[HMM-DXY-01] DXY_HMM_interact: N={n_interact} | src={_dxy_z_col} x regime_norm")

                n_cond = df["DXY_HMM_cond"].notna().sum()
                n_bull = df["DXY_HMM_bull_neg"].notna().sum()
                _last_reg  = int(_hmm_reg.iloc[-1]) if not _hmm_reg.empty else -1
                _last_cond = float(df["DXY_HMM_cond"].dropna().iloc[-1]) if not df["DXY_HMM_cond"].dropna().empty else float("nan")
                logger.info(
                    f"[HMM-DXY-01] Features condicionales generadas | "
                    f"DXY_HMM_cond: N={n_cond} last={_last_cond:.4f} | "
                    f"DXY_HMM_bull_neg: N={n_bull} | HMM_actual=estado_{_last_reg}"
                )
            else:
                _missing = []
                if not _has_dxy_ret: _missing.append("DXY_Ret")
                if not _has_hmm:     _missing.append("HMM_Regime")
                logger.warning(f"[HMM-DXY-01] DXY condicional omitido: faltan {_missing}")
        except Exception as _e_dxh:
            logger.warning(f"[HMM-DXY-01] DXY condicional fallo silenciosamente: {_e_dxh}")
        # ── Fin [DXY-HMM-01] ────────────────────────────────────────────────────────────

        return df




    # â”€â”€ PASO 8: FracDiff dinÃ¡mico (R7) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def apply_fracdiff_dynamic(self, df: pd.DataFrame,
                                train_cutoff: str | None = None) -> pd.DataFrame:
        """
        FracDiff dinÃ¡mico: d se recalcula por ventana via ADF test.
        R7: NUNCA d_val=0.4 fijo.

        Fix C-03: el ADF (que determina d) se ejecuta SOLO sobre el training set
        (train_cutoff). El d Ã³ptimo encontrado en train se aplica luego al dataset
        completo via transform. AsÃ­ evitamos que los datos futuros influyan en el
        parÃ¡metro de diferenciaciÃ³n selÃ©ccionado.

        BUG-1 FIX (P4-0-1, 2026-03-08): train_cutoff se lee de settings.yaml en lugar
        de estar hardcodeado a '2023_12_31'. Con train_end=2024-06-30, el d Ã³ptimo
        ahora se calcula sobre el training set extendido completo.
        """
        # Leer train_cutoff de settings si no se pasa explÃ­citamente
        if train_cutoff is None:
            try:
                from config.settings import cfg
                train_cutoff = cfg.temporal_splits.train_end
                logger.info(f"FracDiff train_cutoff leÃ­do de settings.yaml: {train_cutoff}")
            except Exception:
                raise RuntimeError("FracDiff: no se pudo leer settings.yaml.temporal_splits.train_end â€” requerido para evitar look-ahead en FracDiff (C-03)")

        try:
            from luna.features.fracdiff_dynamic import FracDiffDynamic
            fd = FracDiffDynamic()

            # Fix C-03: detectar los Ã­ndices del training set para la bÃºsqueda de d
            cutoff = pd.Timestamp(train_cutoff, tz="UTC")
            if df.index.tz is None:
                cutoff = cutoff.tz_localize(None)
            df_train = df[df.index <= cutoff]

            if len(df_train) < 100:
                logger.warning("C-03: df_train insuficiente para ADF â€” usando todo el df para FracDiff (fallback)")
                # Aplicamos log manual al fallback tmb
                df_log = df.copy()
                if "close" in df_log.columns: df_log["close"] = np.log(df_log["close"])
                if "volume" in df_log.columns: df_log["volume"] = np.log1p(df_log["volume"])
                
                df_log = fd.transform(df_log, cols=["close", "volume"])
                # Recuperar las columnas fd al df original
                if "close_fd" in df_log.columns: df["close_fd"] = df_log["close_fd"]
                if "volume_fd" in df_log.columns: df["volume_fd"] = df_log["volume_fd"]
            else:
                # 1) Encontrar d Ã³ptimo sobre el training set (causal)
                for col in ["close", "volume"]:
                    if col not in df.columns:
                        continue
                    # [FIX-FRACDIFF-LOG-01] AFML: Aplicar FracDiff a log-precios para estabilizar varianza
                    series_train = df_train[col].dropna()
                    if col == "close":
                        series_train = np.log(series_train)
                    elif col == "volume":
                        series_train = np.log1p(series_train)
                        
                    d_opt = fd.find_optimal_d(series_train, feature_name=f"{col}[train]")
                    fd.d_values_[col] = d_opt
                    logger.info(f"FracDiff C-03 [{col}]: d_opt={d_opt:.3f} (calculado sobre train â‰¤ {train_cutoff})")

                # 2) Aplicar ese d al dataset completo (sin look-ahead: d fue calculado en train)
                for col in ["close", "volume"]:
                    if col not in df.columns or col not in fd.d_values_:
                        continue
                    
                    series_full = df[col]
                    if col == "close":
                        series_full = np.log(series_full)
                    elif col == "volume":
                        series_full = np.log1p(series_full)
                        
                    d_opt = fd.d_values_[col]
                    fd_col = fd._ffd(series_full, d_opt)
                    df[f"{col}_fd"] = fd_col
                    logger.info(f"FracDiff [{col}]: d={d_opt:.3f} â†’ {col}_fd (log-transformed)")

        except ImportError:
            logger.warning("fracdiff_dynamic.py no disponible aÃºn. Saltando FracDiff.")
        return df

    # â”€â”€ PASO 9: Alpha Rules (R13 â€” NATIVO, sin JSON bridge) â”€â”€â”€â”€â”€

    def apply_alpha_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Importa alpha_rules.py (generado por export_alpha_rules.py del AI Mining).
        R13: NUNCA cargar alpha_rules.json ni usar var_map.
        """
        # [FIX-ALPHA-PATH] La ruta original apuntaba a _ROOT/core/features/ que no existe en V2.
        # alpha_rules.py esta en luna/features/alpha_rules.py -- corregido para restaurar
        # alpha_dtw_signal (feature #1 por Gain en V1 XGBoost y LGBM Bull).
        alpha_path = _ROOT / "luna" / "features" / "alpha_rules.py"
        if not alpha_path.exists():
            # Fallback legacy (por compatibilidad con estructura V1)
            alpha_path = _ROOT / "core" / "features" / "alpha_rules.py"
        if not alpha_path.exists():
            logger.warning("alpha_rules.py no existe aÃºn. Ejecutar AI Mining primero (run_weekly_mining.py).")
            return df

        try:
            import importlib.util
            import sys
            
            # P3-N1-FIX (2026-03-30): Caché del módulo importado dinámicamente.
            # Evita re-ejecución si el pipeline se corre múltiples veces en el mismo proceso (tests).
            if "alpha_rules" in sys.modules:
                mod = sys.modules["alpha_rules"]
            else:
                spec = importlib.util.spec_from_file_location("alpha_rules", alpha_path)
                mod  = importlib.util.module_from_spec(spec)
                sys.modules["alpha_rules"] = mod
                spec.loader.exec_module(mod)

            # get_alpha_features es la función principal (genera las 5 alpha signals)
            if hasattr(mod, "get_alpha_features"):
                df = mod.get_alpha_features(df)
            else:
                # Fallback por compatibilidad con versiones anteriores del módulo
                if hasattr(mod, "get_golden_rules"):
                    df = mod.get_golden_rules(df)
                if hasattr(mod, "get_genetic_rules"):
                    df = mod.get_genetic_rules(df)
                if hasattr(mod, "get_fractal_rules"):
                    df = mod.get_fractal_rules(df)

            alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
            
            # [P3-DeepDiscovery-Online] Verbose tracking
            print(f"[LUNA][A4/DeepDiscovery] OK - Módulo dinámico alpha_rules.py cargado exitosamente. "
                  f"Generó {len(alpha_cols)} señales alpha para esta ventana.")
            logger.info(f"[A4/DeepDiscovery] Alpha Rules: {len(alpha_cols)} señales alpha dinámicas integradas -> {alpha_cols}")

            # [DTW-DISABLE-01 2026-06-03] Desactivar alpha_dtw_signal si use_dtw_signal=false
            # CAUSA: DTW_BULL_PROB=1.0 hardcodeado -> dtw_direction=+1 siempre -> signal=tanh(mom_24H*20)
            # El "trigger DTW" era puro momentum-chasing: WR=47.3% vs 56.0% sin trigger (p<0.0001, N=3767)
            # FIX: zerear alpha_dtw_signal para que (a) no active el trigger y (b) XGBoost reciba 0
            try:
                from config.settings import cfg as _cfg_dtw
                _use_dtw = bool(getattr(getattr(_cfg_dtw, 'fase2', _cfg_dtw), 'use_dtw_signal', True))
                if not _use_dtw and 'alpha_dtw_signal' in df.columns:
                    _n_nonzero = (df['alpha_dtw_signal'] != 0).sum()
                    df['alpha_dtw_signal'] = 0.0
                    print(f"[LUNA][DTW-DISABLE-01] alpha_dtw_signal ZEROEADO "
                          f"(use_dtw_signal=false) — {_n_nonzero} valores anulados. "
                          f"Causa: DTW_BULL_PROB=1.0 hardcodeado => momentum chaser con WR=47.3%")
                    logger.warning(f"[DTW-DISABLE-01] alpha_dtw_signal desactivado por settings. "
                                   f"El trigger DTW no disparará en esta ventana.")
                elif _use_dtw:
                    print(f"[LUNA][DTW-DISABLE-01] use_dtw_signal=true -> alpha_dtw_signal activo "
                          f"(rango [{df['alpha_dtw_signal'].min():.3f},{df['alpha_dtw_signal'].max():.3f}])")
            except Exception as _e_dtw:
                print(f"[LUNA][DTW-DISABLE-01] No se pudo leer use_dtw_signal de settings: {_e_dtw} — DTW activo por defecto")

        except Exception as e:
            print(f"[LUNA][A4/DeepDiscovery] ERROR importando alpha_rules.py: {e}")
            logger.error(f"Error importando alpha_rules.py: {e}")

        return df


    # ──── PASO 10: Limpieza Pre-Split (R17) ──────────────────────

    def add_tbm_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [FIX-SFI-TBM-01] Single Source of Truth para etiquetado.
        Genera Target_TBM_Bin usando parámetros globales estándar para que el SFI
        evalúe las características contra las mismas reglas TBM que XGBoost usará después.
        Elimina la regresión naive de 24H que causaba el Dual-Labeling Trap.
        """
        if "close" not in df.columns:
            logger.warning("No se puede generar TBM Target: falta 'close'")
            return df
            
        logger.info("[FP] Generando Target Dinámico TBM Centralizado (Target_TBM_Bin)...")
        from luna.features.tbm import apply_triple_barrier
        from config.settings import cfg
        
        # Parámetros Base para SFI
        _pt = 1.5
        _sl = 1.5
        _min_ret = float(getattr(cfg.sop, 'tbm_min_return', 0.005)) if hasattr(cfg, 'sop') else 0.005
        
        events_idx = df.index
        price_series = df["close"]
        _sides_series = pd.Series(1.0, index=events_idx) # Para SFI asumimos vector direccional Long
        
        tbm_result = apply_triple_barrier(
            price_series=price_series,
            event_times=events_idx,
            sides=_sides_series,
            pt_sl_multiplier=[_pt, _sl],
            min_return=_min_ret,
            dynamic_barrier=True
        )
        
        if "bin" in tbm_result.columns:
            # Convertir bin (-1, 1) a probabilidad binaria (0, 1) para el SFI
            # Conservamos los NaN (operaciones sin resolución o gaps)
            df["Target_TBM_Bin"] = np.nan
            
            valid_mask = tbm_result["ret"].notna()
            valid_bins = (tbm_result.loc[valid_mask, "bin"] > 0).astype(float)
            df.loc[valid_bins.index, "Target_TBM_Bin"] = valid_bins
            
            logger.info(f"[FP] Target_TBM_Bin generado exitosamente. WinRate global estático: {df['Target_TBM_Bin'].mean():.1%}")
        else:
            logger.warning("[FP] Error: 'bin' no encontrado en el resultado TBM.")
            
        return df

    # â”€â”€ PASO 13: Split y guardado â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def split_and_save(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Split temporal Train/Validation/Holdout (R4 Triple Frontera).
        NUNCA mezcla los splits.
        """
        splits = cfg.temporal_splits
        _FEATURES_DIR.mkdir(parents=True, exist_ok=True)

        train_end     = pd.Timestamp(splits.train_end,        tz="UTC")
        val_start     = pd.Timestamp(splits.validation_start, tz="UTC")
        val_end       = pd.Timestamp(splits.validation_end,   tz="UTC")
        holdout_start = pd.Timestamp(splits.holdout_start,    tz="UTC")

        # FIX-WFB-HOLDOUT-END-01 (2026-03-29): leer holdout_end para limitar el parquet
        # holdout al OOS exacto de cada ventana WFB. Sin esto, W1-W4 exponen datos de
        # ventanas futuras al modelo en inferencia. Si no está configurado → sin límite.
        _holdout_end_raw = getattr(splits, 'holdout_end', None)
        if _holdout_end_raw:
            holdout_end = pd.Timestamp(_holdout_end_raw, tz="UTC")
            # FIX-WFB-TBM-OOB (2026-03-29): Añadimos 10 días de colchón a la descarga de features
            # para que las barreras del TBM de los últimos trades de la ventana puedan resolverse.
            # isolate_window_trades() descartará luego las señales cuya entry_time > holdout_end.
            holdout_end_extended = holdout_end + pd.Timedelta(days=10)
            holdout = df[(df.index >= holdout_start) & (df.index <= holdout_end_extended)]
            logger.info("[FIX-WFB-HOLDOUT-END-01] holdout acotado (con +10d cushion para TBM): {} → {} ({} filas)",
                        holdout_start.date(), holdout_end_extended.date(), len(holdout))
        else:
            holdout = df[df.index >= holdout_start]

        train      = df[df.index <= train_end]
        validation = df[(df.index >= val_start) & (df.index <= val_end)]

        datasets = {
            "train":      train,
            "validation": validation,
            "holdout":    holdout,
        }

        def _safe_to_parquet(df_part, out_path, max_retries=5):
            """Retry+backoff para tolerar bloqueos transitorios de Google Drive (errno 22/13)."""
            import time as _time_retry
            for _attempt in range(1, max_retries + 1):
                try:
                    df_part.to_parquet(out_path)
                    return
                except OSError as _oe:
                    if _attempt < max_retries and getattr(_oe, "errno", None) in (13, 22):
                        _wait = 2 ** _attempt
                        import sys as _sys_retry
                        print(f"[DRIVE-RETRY] to_parquet({out_path.name}) errno={_oe.errno} — reintento {_attempt}/{max_retries} en {_wait}s")  # RULE[fixbugsprints.md]
                        _time_retry.sleep(_wait)
                    else:
                        raise

        for name, data in datasets.items():
            if data.empty:
                logger.warning("[DATAFLOW-EXPORT] Split [{}] esta VACIO â€” revisar temporal_splits en settings.yaml", name)
                continue

            # [FIX-OBS-03-HMM-DTYPE 2026-05-30] Convertir HMM_Regime float64->int ANTES de guardar.
            # BUG-HMM-FILTER-01: filtro semantico compara int keys del state_map vs float64 -> fallo silencioso.
            for _hmm_dtype_col in ["HMM_Regime"]:
                if _hmm_dtype_col in data.columns and str(data[_hmm_dtype_col].dtype) in ("float64", "float32"):
                    data = data.copy()
                    data[_hmm_dtype_col] = data[_hmm_dtype_col].fillna(-1).astype(int)
                    print(  # RULE[fixbugsprints.md]
                        f"[FIX-OBS-03-HMM-DTYPE] Split '{name}': "
                        f"{_hmm_dtype_col} float64->int (BUG-HMM-FILTER-01 prevenido)"
                    )
                    logger.info(
                        f"[FIX-OBS-03-HMM-DTYPE] Split '{name}': {_hmm_dtype_col} "
                        f"dtype float64->int -- evita BUG-HMM-FILTER-01"
                    )
                    # [FIX-OBS-03-B 2026-06-02] Actualizar datasets dict tras la conversion.
                    # BUG: data = data.copy() crea referencia local; datasets.get('holdout')
                    # en DATAFLOW-EXPORT-FP-01 leia el objeto original (float64) -> falsa ALERTA.
                    datasets[name] = data
                    print(  # RULE[fixbugsprints.md]
                        f"[FIX-OBS-03-B] datasets['{name}'] actualizado dtype int64"
                        f" -> DATAFLOW check leera objeto correcto (no falsa ALERTA)"
                    )

            path = _FEATURES_DIR / f"features_{name}.parquet"
            _safe_to_parquet(data, path)
            # AUDIT Paso 1 (BUG-HOLDOUT-PATH fix): double-write con ID de ventana especifico.
            # Cada ventana WFB escribe su propio features_{name}_W{N}.parquet inmutable.
            if name in ("holdout", "validation"):
                import os as _os_fp_dw
                _win_id_dw = _os_fp_dw.environ.get("LUNA_WINDOW_ID", "")
                if _win_id_dw:
                    _window_path = _FEATURES_DIR / f"features_{name}_{_win_id_dw}.parquet"
                    _safe_to_parquet(data, _window_path)
                    logger.info(f"[AUDIT-HOLDOUT-PATH] features_{name}_{_win_id_dw}.parquet escrito ({len(data)} filas).")
            # Resumen NaN en columnas críticas del split
            crit = [c for c in ["close", "FundingRate", "FearGreed", "ETH_Price", "M2_China_YoY"]
                    if c in data.columns]
            nan_crit = {c: f"{data[c].isna().mean():.0%}" for c in crit if data[c].isna().any()}
            nan_str = f" | NaN crit: {nan_crit}" if nan_crit else " | NaN crit: OK"
            print(f"[BUG-FIX-LOG 2026-06-05] Corregido formatting logger.success en feature_pipeline.py [DATAFLOW-EXPORT]")
            logger.success(
                "[DATAFLOW-EXPORT] Split [{}]: {} x {} | {} -> {}{}",
                name, data.shape[0], data.shape[1],
                data.index.min().date(), data.index.max().date(),
                nan_str
            )

        # ——— [DATAFLOW-EXPORT-FP-01] Cross-split consistency checks ——————————————————————
        # Detecta problemas de dataflow entre splits ANTES de que fallen los modelos.
        train_df   = datasets.get("train",      pd.DataFrame())
        holdout_df = datasets.get("holdout",    pd.DataFrame())
        val_df     = datasets.get("validation", pd.DataFrame())

        # CHECK 1: Columnas HMM en holdout — su ausencia causa BUG-HMM-FILTER-01
        _hmm_cols_expected = ["HMM_Regime", "HMM_Semantic"]
        for _hc in _hmm_cols_expected:
            if not holdout_df.empty:
                if _hc not in holdout_df.columns:
                    logger.warning(
                        f"  [DATAFLOW-EXPORT-FP-01] {_hc} NO EXISTE en features_holdout.parquet. "
                        f"El filtro HMM en generate_oos_predictions.py usara predict_regime_series() "
                        f"que devuelve indices numericos â€” asegurarse de que el fix BUG-HMM-FILTER-01 esta activo."
                    )
                else:
                    _cov = holdout_df[_hc].notna().mean()
                    _dtype = holdout_df[_hc].dtype
                    logger.info(
                        f"  [DATAFLOW-EXPORT-FP-01] {_hc} en holdout: dtype={_dtype} cov={_cov:.1%}"
                    )
                    # [FIX-DATAFLOW-INT-01 2026-06-02] Warning solo si dtype es FLOAT.
                    # ANTES: alertaba si dtype in (float64, float32, int64, int32)
                    # BUG: int64 ES el dtype correcto post-FIX-OBS-03-HMM-DTYPE.
                    # El BUG-HMM-FILTER-01 ocurre solo con float64 (float 1.0 != int 1).
                    # AHORA: warning solo si float, print-OK si ya es int (correcto).
                    if str(_dtype) in ("float64", "float32"):
                        logger.warning(
                            f"  [DATAFLOW-EXPORT-FP-01] ALERTA: {_hc} en holdout es FLOAT ({_dtype}). "
                            f"Causara BUG-HMM-FILTER-01 (filtro regime ==1 fallara). "
                            f"Verificar que FIX-OBS-03-HMM-DTYPE esta activo."
                        )
                        print(  # RULE[fixbugsprints.md]
                            f"[FIX-DATAFLOW-INT-01] ALERTA: {_hc} dtype={_dtype} (float) -> BUG-HMM-FILTER-01 riesgo"
                        )
                    elif str(_dtype) in ("int64", "int32", "int16"):
                        print(  # RULE[fixbugsprints.md]
                            f"[FIX-DATAFLOW-INT-01] {_hc} dtype={_dtype} (int) -> OK, FIX-OBS-03 activo"
                        )

        # CHECK 2: Consistencia de columnas train vs holdout
        if not train_df.empty and not holdout_df.empty:
            _only_train   = set(train_df.columns) - set(holdout_df.columns)
            _only_holdout = set(holdout_df.columns) - set(train_df.columns)
            if _only_train:
                logger.warning(
                    f"  [DATAFLOW-EXPORT-FP-01] {len(_only_train)} cols en TRAIN pero NO en HOLDOUT: "
                    f"{sorted(_only_train)[:10]}{'...' if len(_only_train) > 10 else ''}. "
                    f"Columnas faltantes en holdout pueden causar errores en generate_oos_predictions."
                )
            else:
                logger.info("  [DATAFLOW-EXPORT-FP-01] Columnas train vs holdout: OK (mismo conjunto)")

        # CHECK 3: No hay solapamiento de fechas entre splits (BLOQUEANTE desde LAB-02 â€” R4)
        if not train_df.empty and not val_df.empty:
            _overlap_tv = set(train_df.index) & set(val_df.index)
            if _overlap_tv:
                raise RuntimeError(
                    f"[DATAFLOW-EXPORT-FP-01] LEAKAGE R4: {len(_overlap_tv)} timestamps solapan "
                    f"entre train y validation. El calibrador verÃ­a datos del training set. "
                    f"Revisar temporal_splits.train_end y validation_start en settings.yaml. "
                    f"(LAB-02 fix: train_end debe ser < validation_start)"
                )
            else:
                logger.info("  [DATAFLOW-EXPORT-FP-01] Solapamiento train/validation: OK (cero overlap â€” R4 OK)")
        if not val_df.empty and not holdout_df.empty:
            _overlap_vh = set(val_df.index) & set(holdout_df.index)
            if _overlap_vh:
                logger.warning(
                    f"  [DATAFLOW-EXPORT-FP-01] LEAKAGE: {len(_overlap_vh)} timestamps solapan entre validation y holdout!"
                )

        # CHECK 4: Holdout cubre el periodo esperado por settings.yaml
        if not holdout_df.empty:
            _hd_start = holdout_df.index.min()
            _hd_end   = holdout_df.index.max()
            _expected_start = holdout_start
            _gap_hours = abs((_hd_start - _expected_start).total_seconds() / 3600)
            if _gap_hours > 48:
                logger.warning(
                    f"  [DATAFLOW-EXPORT-FP-01] Holdout empieza en {_hd_start.date()} "
                    f"pero settings.yaml dice {_expected_start.date()} (gap={_gap_hours:.0f}h). "
                    f"Revisar temporal_splits.holdout_start en settings.yaml."
                )
            else:
                logger.info(
                    f"  [DATAFLOW-EXPORT-FP-01] Holdout fechas OK: {_hd_start.date()} -> {_hd_end.date()} "
                    f"(expected_start={_expected_start.date()})"
                )
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        return datasets



    # â”€â”€ MÃ©todo maestro â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run(
        self,
        skip_fracdiff: bool = False,
        skip_sfi: bool = False,
        live_mode: bool = False,
        fast_inject: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Ejecuta el pipeline completo de features.

        Args:
            skip_fracdiff: saltar FracDiff (útil en desarrollo)
            skip_sfi: saltar SFI filter (útil en prototipos rápidos)
            live_mode: BUG-LIVE-01 FIX. Si es True, no hace split temporal y 
                       devuelve el DataFrame completo hasta el presente para inferencia live.

        Returns:
            dict con train/validation/holdout DataFrames (o 'live' si live_mode=True)
        """
        import time as _time
        _t_total = _time.monotonic()
        logger.info("â”" * 60)
        logger.info("ðŸŒ™ Luna V1 FeaturePipeline.run() comenzando...")
        logger.info("â”" * 60)

        # [FAST-INJECT] Fase 4: Optimization de Pipeline Redundancy
        if fast_inject:
            logger.info("[FP] FAST-INJECT MODE ACTIVO: Saltando reconstruccion de variables base.")
            parts = []
            for p in ["features_train.parquet", "features_validation.parquet", "features_holdout.parquet"]:
                pth = _FEATURES_DIR / p
                if pth.exists():
                    try:
                        import pandas as pd
                        parts.append(pd.read_parquet(pth))
                    except Exception as e:
                        logger.warning(f"[FP] FAST-INJECT no pudo leer {p}: {e}")
            if parts:
                import pandas as pd
                df = pd.concat(parts).sort_index()
                cols_to_drop = ["KMeans_Tribe_ID", "Master_Causal_Signal", "HMM_Regime", "HMM_Semantic"]
                df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
                
                df = self.integrate_mining_outputs(df)
                
                logger.info(f"[FP] FAST-INJECT FINALIZADO | {_time.monotonic() - _t_total:.1f}s")
                if live_mode:
                    return {"live": df}
                return self.split_and_save(df)
            else:
                logger.warning("[FP] FAST-INJECT Fallo: No se encontraron parquets base. Fallback a pipeline completo.")

        # Pasos 1-3: Carga + Safety Lags + Merge
        _t = _time.monotonic()
        df = self.load_and_merge_raw()
        logger.info("[FP] Paso 1-3 (load+merge raw): {}Ã—{} | {} â†’ {} | {:.1f}s",
                    df.shape[0], df.shape[1],
                    df.index.min().date() if not df.empty else '?',
                    df.index.max().date() if not df.empty else '?',
                    _time.monotonic() - _t)

        # Paso 3B: Mining Outputs (K_Shape_Cluster_ID + Master_Causal_Signal)
        _n_before = df.shape[1]
        df = self.integrate_mining_outputs(df)
        logger.info("[FP] Paso 3B (mining outputs): +{} cols â†’ {} total",
                    df.shape[1] - _n_before, df.shape[1])

        # Paso 4: Rolling Z-Score (R14)
        _n_before = df.shape[1]
        df = self.apply_rolling_normalization(df)
        logger.info("[FP] Paso 4 (rolling zscore): +{} cols â†’ {} total",
                    df.shape[1] - _n_before, df.shape[1])

        # Paso 5: Multi-Timeframe
        _n_before = df.shape[1]
        df = self.apply_multitimeframe_features(df)
        logger.info("[FP] Paso 5 (multi-timeframe): +{} cols â†’ {} total",
                    df.shape[1] - _n_before, df.shape[1])

        # Paso 6: Calendar Features â† NUEVO (FOMC, Deribit, Halving, Sesiones)
        _n_before = df.shape[1]
        df = self.apply_calendar_features(df)
        logger.info("[FP] Paso 6 (calendar features): +{} cols â†’ {} total",
                    df.shape[1] - _n_before, df.shape[1])

        # Paso 7: Cross-Asset
        _n_before = df.shape[1]
        df = self.apply_crossasset_features(df)
        logger.info("[FP] Paso 7 (cross-asset): +{} cols â†’ {} total",
                    df.shape[1] - _n_before, df.shape[1])

        # Paso 7B: Derived Macro + On-Chain features (mc_* / oc_* / dv_*)
        # Genera features equivalentes a Luna v2 V10 sin fetchers adicionales
        _n_before = df.shape[1]
        df = self.apply_derived_features(df)
        logger.info("[FP] Paso 7B (derived mc_/oc_/dv_): +{} cols â†’ {} total",
                    df.shape[1] - _n_before, df.shape[1])

        # Paso 7C: Order Flow Imbalance (OFI) - Fase 2D
        try:
            from config.settings import cfg as _cfg
            if getattr(_cfg, "fase2", None) and getattr(_cfg.fase2, "use_ofi_features", False):
                from luna.features.ofi_features import add_ofi_features
                _n_before = df.shape[1]
                df = add_ofi_features(df)
                logger.info("[FP] Paso 7C (Order Flow Imbalance): +{} cols -> {} total",
                            df.shape[1] - _n_before, df.shape[1])
            else:
                logger.info("[FP] Paso 7C (Order Flow Imbalance): OMITIDO (cfg.fase2.use_ofi_features = false o no configurado)")
        except Exception as e:
            logger.warning(f"[FP] Paso 7C (Order Flow Imbalance) Fallo: {e}")

        # Paso 7D: Kalman Z-Score Adaptativo (Segunda Pasada para Features Derivadas)
        # [FIX-KALMAN-01] Las features derivadas (mc_*) se generan en 7B, por lo que la
        # primera pasada de Kalman en Paso 4 las omitía. Esta segunda pasada atrapa el resto.
        try:
            from luna.features.kalman_normalizer import KalmanZScoreNormalizer, KALMAN_COLUMNS
            from config.settings import cfg as _cfg_kz
            _kz_q = float(getattr(getattr(_cfg_kz, 'features', {}), 'kalman_q', 1e-4))
            _kz_r = float(getattr(getattr(_cfg_kz, 'features', {}), 'kalman_r', 0.1))
            _n_before = df.shape[1]
            _kalman = KalmanZScoreNormalizer(process_noise=_kz_q, obs_noise=_kz_r)
            df = _kalman.transform_df(df, columns=KALMAN_COLUMNS, suffix="_kz")
            if df.shape[1] > _n_before:
                logger.info("[FP] Paso 7D (Kalman Z-Score post-derivadas): +{} cols -> {} total",
                            df.shape[1] - _n_before, df.shape[1])
        except Exception as e:
            logger.error(f"[FP] Error en Paso 7D (Kalman post-derivadas): {e}")

        # Paso 8: FracDiff dinÃ¡mico (R7)
        if not skip_fracdiff:
            _t = _time.monotonic()
            _n_before = df.shape[1]
            df = self.apply_fracdiff_dynamic(df)
            logger.info("[FP] Paso 8 (fracdiff): +{} cols â†’ {} total | {:.1f}s",
                        df.shape[1] - _n_before, df.shape[1], _time.monotonic() - _t)
        else:
            logger.info("[FP] Paso 8 (fracdiff): OMITIDO (--skip-fracdiff)")

        # Paso 9: Alpha Rules (R13)
        _n_before = df.shape[1]
        df = self.apply_alpha_rules(df)
        logger.info("[FP] Paso 9 (alpha rules): +{} cols â†’ {} total",
                    df.shape[1] - _n_before, df.shape[1])

        # Paso 9B: Tribe Features derivadas (Mejora M3)
        # Convierte KMeans_Tribe_ID de entero ordinal a features continuas/binarias
        # que XGBoost puede interpretar correctamente (tribe_wr_historical, tribe_in_larga, tribe_wr_zscore)
        try:
            from luna.features.alpha_rules import apply_tribe_features
            _n_before = df.shape[1]
            df = apply_tribe_features(df)
            tribe_cols = [c for c in df.columns if c.startswith('tribe_')]
            if tribe_cols:
                logger.info(f"Tribe features (M3): {tribe_cols} (+{df.shape[1] - _n_before} cols)")
        except Exception as e:
            logger.warning(f"apply_tribe_features fallÃ³ (no bloquea pipeline): {e}")


        # Paso 9C: Meta-Oracle Score (Mejora M1 â€” 2026-03-04)
        # El Meta-OrÃ¡culo sintetiza 6 motores de Mining en un score [0,1].
        # Se lee del oracle_verdict.md (actualizado semanalmente por run_weekly_mining.py)
        # y se propaga como constante horaria â€” el mismo score se aplica a todas las filas
        # del dataset hasta el prÃ³ximo mining run.
        try:
            import re as _re
            _oracle_path = Path(__file__).resolve().parents[2] / 'data' / 'ai_mining' / 'reports' / 'oracle_verdict.md'  # Fix A-04: ruta absoluta
            if _oracle_path.exists():
                _oracle_text = _oracle_path.read_text(encoding='utf-8')
                # Buscar Score Final en el markdown
                _m = _re.search(r'Score Final.*?`([\d.]+)`', _oracle_text)
                if _m:
                    _oracle_score = float(_m.group(1))
                    # Centrar en 0: [0,1] â†’ [-1,1] para que el SFI vea seÃ±al direccional
                    _oracle_feat_val = (_oracle_score - 0.5) * 2.0
                    # [AUDIT-A4 FIX 2026-05-08] meta_oracle_score es constante (var=0).
                    # XGBoost la ignora (GAIN=0). No inyectar para no desperdiciar features.
                    # Re-habilitar cuando se implemente oracle_score rolling por timestamp.
                    print(f"[BUG-FIX-LOG 2026-06-05] Corregido formatting logger.warning en feature_pipeline.py [AUDIT-A4]")
                    logger.warning(
                        '[AUDIT-A4] meta_oracle_score={:.4f} (feature={:.4f}) NO inyectada '
                        '(constante global, var=0, XGBoost GAIN=0 en todos los splits). '
                        'Implementar oracle_score rolling para re-habilitar.',
                        _oracle_score, _oracle_feat_val
                    )
                    logger.info(f"Meta-Oracle score integrado: {_oracle_score:.4f} â†’ feature={(_oracle_score-0.5)*2.0:.4f}")
                else:
                    logger.warning("meta_oracle_score: no se pudo parsear el score de oracle_verdict.md")
            else:
                logger.warning("meta_oracle_score: oracle_verdict.md no encontrado â€” feature omitida")
        except Exception as _e:
            logger.warning(f"meta_oracle_score falló (no bloquea pipeline): {_e}")

        # Paso 9D: Meta-Feature AutoEncoder Compression (Fase 3)
        # Comprime las 300+ features crudas en representaciones densas no-lineales.
        try:
            from luna.features.autoencoder_features import apply_autoencoder
            from config.settings import cfg as _cfg
            
            _use_ae = getattr(_cfg.fase2, "use_autoencoder", True) if hasattr(_cfg, "fase2") else True
            if _use_ae:
                _t_ae = _time.monotonic()
                _train_end = getattr(_cfg.temporal_splits, "train_end", "2023-12-31") # fallback warning
                # [FIX-AE-BOTTLENECK-01 2026-05-30] Bottleneck era 32 (8.6x compresion para 275 features
                # -> MSE=0.1376 alto). Leemos de settings con default=64 (4.3x, ratio optimo ~n/4).
                _ae_bottleneck = 64
                _ae_epochs = 60
                try:
                    from config.settings import cfg as _cfg_ae_bn
                    _ae_bottleneck = int(getattr(getattr(_cfg_ae_bn, "autoencoder", None) or type("_", (), {"bottleneck_size": 64})(), "bottleneck_size", 64))
                    _ae_epochs = int(getattr(getattr(_cfg_ae_bn, "autoencoder", None) or type("_", (), {"max_epochs": 60})(), "max_epochs", 60))
                except Exception:
                    pass
                print(f"[FIX-AE-BOTTLENECK-01] AE bottleneck={_ae_bottleneck} (era 32), epochs={_ae_epochs} (era 30)")
                df = apply_autoencoder(df, train_end_date=_train_end, bottleneck_size=_ae_bottleneck, epochs=_ae_epochs, live_mode=live_mode)
                logger.info(f"[FP] Paso 9D (AutoEncoder): Extracción densa completada | bottleneck={_ae_bottleneck} | {_time.monotonic() - _t_ae:.1f}s")
            else:
                logger.info("[FP] Paso 9D (AutoEncoder): OMITIDO por config")
        except Exception as e:
            import traceback
            logger.warning(f"[FP] Paso 9D (AutoEncoder) Falló: {e}\n{traceback.format_exc()}")

        # Paso 10: Centralized Target Labeling (Single Source of Truth)
        # [FIX-SFI-TBM-01] Asegura que SFI y XGBoost trabajen con el mismo target de Triple Barrera
        _t_tbm = _time.monotonic()
        df = self.add_tbm_target(df)
        logger.info(f"[FP] Paso 10 (TBM Target): Etiquetado dinámico finalizado | {_time.monotonic() - _t_tbm:.1f}s")

        # Eliminar NaN excesivos (head artifacts de lags largos)
        before = len(df)
        df = df.dropna(subset=["close"])
        logger.info("[FP] Dropna(close): {} â†’ {} filas (-{})", before, len(df), before - len(df))

        # Paso 11: PURGA anti-leakage (R1) â€” eliminar cualquier columna de look-ahead
        #          antes de guardar. Esto incluye future_ret_24h si por error se persistÃ­Ã³.
        _n_before = df.shape[1]
        df = purge_leakage_columns(df)
        if df.shape[1] < _n_before:
            logger.warning("[FP] Paso 11 (purge leakage): eliminadas {} columnas con look-ahead",
                           _n_before - df.shape[1])
        else:
            logger.info("[FP] Paso 11 (purge leakage): sin columnas de look-ahead detectadas")

        # â”€â”€ NaN summary final (features con >20% NaN en el dataset completo) â”€â”€â”€â”€â”€â”€
        numeric_cols = df.select_dtypes(include=["number"]).columns
        nan_pct = df[numeric_cols].isna().mean()
        high_nan = nan_pct[nan_pct > 0.20].sort_values(ascending=False)
        if not high_nan.empty:
            logger.warning("[FP] NaN summary: {} cols con >20% NaN: {} (max: {}={:.0f}%)",
                           len(high_nan), high_nan.index.tolist()[:8],
                           high_nan.idxmax(), high_nan.max() * 100)
        else:
            logger.info("[FP] NaN summary: ninguna columna numérica con >20%% NaN ✅")

        logger.info("[FP] Dataset final: {} filas × {} cols | tiempo total: {:.1f}s",
                    df.shape[0], df.shape[1], _time.monotonic() - _t_total)
        logger.info("— " * 60)


        # FIX-NAN-COLS: Limpieza de columnas 100% NaN (evita warnings y acelera SFI)
        cols_before = df.shape[1]
        df = df.dropna(axis=1, how='all')
        if df.shape[1] < cols_before:
            logger.info(f"[FP] Limpieza: eliminadas {cols_before - df.shape[1]} columnas que eran 100% NaN")

        # [FIX-SKEW-FINAL] Alias bridge post-dropna para garantizar nombres canónicos de entrenamiento
        # en features_live.parquet. Los aliases se crean en apply_derived_features (FIX-SKEW-01/02/03)
        # pero si la fuente tiene datos solo en periodos recientes, la columna es 100%NaN en el df
        # histórico completo y dropna(how='all') la elimina. Este bloque re-aplica los aliases
        # DESPUÉS del dropna usando las fuentes disponibles con datos válidos.
        # NOTA: Solo se ejecuta si la columna canónica (train) no existe y la live sí existe.
        if live_mode:
            _FINAL_ALIAS_MAP = {
                'FundingRate_EMA3':  ['funding_ema_3'],
                'FundingRate_Pct90d': ['funding_pct_90d', 'dv_funding_pct_90d'],
                'OI_Open_USD':       ['Coinglass_oi_open', 'oi_open'],
                'OI_High_USD':       ['Coinglass_oi_high', 'oi_high'],
                'OI_Low_USD':        ['Coinglass_oi_low', 'oi_low'],
                'ETF_Flow_Proxy':    ['etf_flow_proxy', 'ETF_IBIT_Flow_Proxy'],
                # [FIX-SKEW-FINAL-DV] dv_etf_flow_proxy: FIX-SKEW-03 solo lo crea en memoria
                # durante inference (no se persiste a parquet). Aquí se añade al bridge
                # post-dropna para que quede en features_live.parquet con la misma fuente.
                'dv_etf_flow_proxy': ['etf_flow_proxy', 'ETF_Flow_Proxy'],
            }
            _final_aliased = []
            _final_missing = []
            for _train_col, _live_srcs in _FINAL_ALIAS_MAP.items():
                if _train_col not in df.columns:
                    _applied = False
                    for _src in _live_srcs:
                        if _src in df.columns:
                            df[_train_col] = df[_src]
                            _final_aliased.append(f"{_src}→{_train_col}")
                            _applied = True
                            break
                    if not _applied:
                        _final_missing.append(f"{_train_col} (intentado: {_live_srcs})")
            if _final_aliased:
                print(f"[FIX-SKEW-FINAL] Post-dropna aliases aplicados ({len(_final_aliased)}): {_final_aliased}")
                logger.info(f"[FIX-SKEW-FINAL] Post-dropna aliases para features_live.parquet: {_final_aliased}")
            if _final_missing:
                logger.warning(f"[FIX-SKEW-FINAL] {len(_final_missing)} aliases sin fuente: {_final_missing}")
                print(f"[FIX-SKEW-FINAL] WARN: sin fuente: {_final_missing}")

        # Paso 13: Split + Guardar (o Bypass en Live Mode)
        # BUG-LIVE-01 FIX: Si estamos en live inference, devolver el DataFrame completo
        if live_mode:
            logger.info("[FP] LIVE MODE: Saltando split_and_save — enviando dataset fresco a inferencia.")
            return {"live": df}
            
        return self.split_and_save(df)




if __name__ == "__main__":
    """
    Punto de entrada para run_full_pipeline.py Fase 3A.
    Invocado como: python core/features/feature_pipeline.py [--skip-preflight]

    --skip-preflight: omitir pre-flight check (el orquestador ya lo ejecuto en Fase 0).
    """
    import argparse as _ap
    import os as _os
    _parser = _ap.ArgumentParser(description="Luna V1 â€” Feature Pipeline")
    _parser.add_argument("--skip-preflight", action="store_true",
                         help="Omitir pre-flight check (ya ejecutado por el orquestador)")
    _parser.add_argument("--skip-fracdiff", action="store_true",
                         help="Saltar FracDiff (util en desarrollo rapido)")
    _parser.add_argument("--fast-inject", action="store_true",
                         help="Fase 4: Inyectar HMM/BCE sobre parquets existentes, omitiendo Feature Generation.")
    _parser.add_argument("--window-id", type=str, default=None,
                         help="ID de ventana WFB (ej: W1, W2). Si se pasa, exporta features_holdout_{window_id}.parquet")
    _args = _parser.parse_args()

    # Propagar window-id al entorno para que split_and_save lo herede
    if _args.window_id:
        _os.environ["LUNA_WINDOW_ID"] = _args.window_id

    # â”€â”€ Log file propio del subproceso (trazabilidad por RUN_ID) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from datetime import datetime as _dt
    _log_dir = _ROOT / "logs"
    _log_dir.mkdir(exist_ok=True)
    _ts_fp  = _dt.now().strftime("%Y%m%d_%H%M%S")
    _rid_fp = _os.environ.get("LUNA_RUN_ID", "")
    _lname_fp = f"feature_pipeline_{_ts_fp}_{_rid_fp}.log" if _rid_fp else f"feature_pipeline_{_ts_fp}.log"
    logger.add(_log_dir / _lname_fp, rotation="50 MB", level="DEBUG", encoding="utf-8")
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    logger.info("â€™ feature_pipeline.py __main__ â€” regenerando features con nuevas fechas")
    fp = FeaturePipeline()
    splits = fp.run(skip_fracdiff=_args.skip_fracdiff, fast_inject=_args.fast_inject)
    for name, df_s in splits.items():
        if not df_s.empty:
            logger.success(
                f"  {name:12s}: {df_s.index.min().date()} -> {df_s.index.max().date()} "
                f"({len(df_s):,} rows x {df_s.shape[1]} cols)"
            )
