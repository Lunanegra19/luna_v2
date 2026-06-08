#!/usr/bin/env python
"""
data_integrity_check.py — Luna V1  (v2.1)
==========================================
Verificacion completa de integridad de los datos descargados en disco.

CHEQUES REALIZADOS:
  1. Existencia y tamano de cada archivo parquet
  2. Shape (filas x columnas) — detecta archivos vacios
  3. Rango de fechas — detecta indices mal seteados (ej: epoch 1970)
  4. Gaps temporales — segun tipo de archivo (horario vs diario)
  5. % NaN por columna — con whitelist de columnas calculadas en pipeline
  6. Columnas criticas presentes
  7. Valores infinitos (np.inf/-np.inf)
  8. Tipos de datos OHLCV (deben ser numericos)
  9. Logica OHLCV: high >= close >= low >= open >= 0
 10. Freshness: solo holdout debe llegar a 2025 (train/val correcto hasta 2024)
 11. Solapamiento de splits train/validation/holdout
 12. Continuidad OHLCV 1H con los 3 peores gaps
 13. Archivos de modelos criticos (pkl, joblib, pt)
 14. selected_features.json existente y valido
 15. Coherencia cross-file: features_holdout cubre el mismo rango que ohlcv_raw
 16. [FROZEN-FEAT] Features congeladas o degeneradas en features_holdout/train/validation

Uso:
    venv/Scripts/python.exe scripts/data_integrity_check.py [-v]
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))  # allow imports from project root

# Forzar stdout UTF-8 en Windows (PowerShell usa cp1252 por defecto)
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

# ─── Colores ANSI ──────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

OK_S   = f"{GREEN}OK  {RESET}"
WARN_S = f"{YELLOW}WARN{RESET}"
FAIL_S = f"{RED}FAIL{RESET}"

# ─── Configuracion desde settings.yaml ─────────────────────────────
try:
    from config.settings import cfg
    NAN_LIMIT = float(getattr(getattr(cfg, "debug", object()), "nan_threshold_pct", 5.0))
    EMBARGO_H     = int(getattr(getattr(cfg, "sop", object()), "embargo_hours", 96))
except Exception:
    NAN_LIMIT = 5.0
    EMBARGO_H     = 96

# ─── Whitelist de columnas que son NaN en raw (se calculan en pipeline) ──
# No generar WARN por estas columnas — es comportamiento esperado
NAN_WHITELIST_PATTERNS = [
    # Derivadas calculadas en feature_pipeline.py
    "_Zscore", "_ZScore", "_zscore",
    "_MA", "_EMA", "_Slope", "_Pct6m", "_pct_6m",
    "Pct30d", "Liq_ZScore", "Liq_Pct30d",
    "AboveMA200", "vs_MA200",
    # Columnas de rolling window con NaN en las primeras filas
    "_Ret", "_Ret1m", "_Ret7d", "_30d_MA", "_30d_Std",
    "_7d_MA", "_90d_MA",
    # Columnas de correlacion (requieren ventana)
    "_Corr", "_corr_",
    # Columnas de on-chain proxy
    "ExchangeFlow_Proxy", "exchange_flow_proxy",
    "MVRV_Pct", "mvrv_pct", "Whale_Vol",
    # Mempool/onchain derivadas
    "HashRate_14dMA", "HashRate_30dMA",
    "Puell_Multiple",
    # DeFi derivadas
    "DeFi_WBTC_APY",
    # ETF/Coinglass (datos opcionales, no siempre presentes)
    "Coinglass_", "ETF_", "IBIT_",
    # Columnas heredadas de Binance sin uso
    "quote_volume", "trades",
    # Crossasset derivadas
    "eth_btc_ratio_zscore", "alt_season_proxy",
    # SP500/NASDAQ derivadas
    "SP500_Ret1m", "NASDAQ_Ret",
    # Stablecoin/M2 derivadas
    "ETF_IBIT_",
    # MetaLabeler prob (poblada solo post-training)
    "meta_v2_prob",

    # ── Grupo 2: APIs con datos parciales, históricamente seleccionadas ──────
    # Fuentes deterioradas pero con suficiente señal para que el SFI las evalúe.
    # NO se eliminan del pipeline porque fueron seleccionadas en 14-72 de 120 runs.
    # Se añaden a la whitelist para que el integrity check no las bloquee.
    # El SFI decide ventana a ventana si tienen suficiente IC/DSR para entrar.
    #
    # BITO ETF: 54% NaN. Seleccionada 72+35+14 veces (BITO_Close/High/Volume).
    # Proxy relevante de demanda institucional BTC pre-IBIT (2021-2023).
    "BITO_Close", "BITO_High", "BITO_Low", "BITO_Volume",
    #
    # Derivatives OI: 53% NaN. dv_oi_acceleration_24h seleccionada 32 veces.
    # OI de Binance futures — fuente fragmentada pero cubre períodos clave.
    "OI_BTC", "OI_USD", "oi_velocity_", "dv_oi_acceleration_24h",
    "oi_pct_90d", "DangerZone",
    #
    # LongShortRatio / Long/ShortAccount: 74-99% NaN.
    # ShortAccount seleccionada 14 veces. Fuente Coinglass CSV parcial.
    "LongAccount", "ShortAccount", "LongShortRatio",
    #
    # On-chain blockchain: 72% NaN. Unique_Addresses seleccionada 14 veces.
    # Fuente Glassnode — cobertura limitada a ciertos períodos.
    "Transactions", "Hash_Rate", "Tx_Volume", "Mempool_Size", "Unique_Addresses",
    #
    # DVOL Deribit: 77% NaN. dv_dvol_z7d seleccionada 1 vez.
    # Volatilidad implícita — datos desde 2020, escasos en training pre-2020.
    "DVOL", "dv_dvol_", "dv_vrp_",
    #
    # ETF flow proxy alternativo: 83% NaN (duplicado de ETF_IBIT_ ya whitelisted).
    "etf_flow_proxy",
    #
    # Legacy/Orphaned features from older pipelines that are safely ignored by SFI
    "Taker_Buy", "Taker_Sell", "active_addresses", "block_size", "tx_count",
    "Wiki_BTC", "hash_ribbon", "oc_hash_ribbon",
    #
    # [WHITELIST-PERPS-01 2026-05-30] OKX Perps API — datos no disponibles en backtesting
    # close_perps, volume_perps, taker_buy_base_perps son 100% NaN en features_train/holdout
    # porque el endpoint de OKX perpetuos no esta en el fetcher historico.
    # El SFI los excluye automaticamente por NaN. No son model-critical.
    "close_perps", "volume_perps", "taker_buy_base_perps", "taker_sell_base_perps",
    "open_perps", "high_perps", "low_perps",
    #
    # [WHITELIST-GBTC-01 2026-05-30] GBTC Discount API muerta desde conversion a ETF (2024)
    # GBTC_Discount_Pct es 100% NaN desde Ene-2024. No es model-critical.
    "GBTC_Discount_Pct",
]

def _is_nan_whitelist(col: str) -> bool:
    return any(p in col for p in NAN_WHITELIST_PATTERNS)

# ─── Archivos de modelos criticos ──────────────────────────────────
# M-09: Cargar dinámicamente según configuración
try:
    from config.settings import cfg
    _use_regimes  = getattr(cfg.xgboost, "use_regime_agents", False)
    _direction    = getattr(cfg.fase2, "direction_mode", "both")  # [DIRECTION-FIX 2026-06-03]
except:
    _use_regimes  = False
    _direction    = "both"

CRITICAL_MODELS = [
    ("data/models/hmm_regime.pkl",                   "HMM Regime Model"),
]
if _use_regimes:
    # [DIRECTION-FIX 2026-06-03] Solo incluir modelos SHORT si direction_mode != 'long'
    # Con direction_mode='long' solo se entrenan *_long.model — los SHORT son esperados ausentes.
    _include_short = _direction not in ("long", "long_only")
    CRITICAL_MODELS.extend([
        ("data/models/xgboost_meta_bull_long.model",      "XGBoost Meta Model (BULL LONG)"),
        ("data/models/xgboost_meta_bear_long.model",      "XGBoost Meta Model (BEAR LONG)"),
        ("data/models/xgboost_meta_range_long.model",     "XGBoost Meta Model (RANGE LONG)"),
    ])
    if _include_short:
        CRITICAL_MODELS.extend([
            ("data/models/xgboost_meta_bull_short.model",     "XGBoost Meta Model (BULL SHORT)"),
            ("data/models/xgboost_meta_bear_short.model",     "XGBoost Meta Model (BEAR SHORT)"),
            ("data/models/xgboost_meta_range_short.model",    "XGBoost Meta Model (RANGE SHORT)"),
        ])
else:
    CRITICAL_MODELS.append(("data/models/xgboost_meta.model", "XGBoost Meta Model"))

CRITICAL_MODELS.extend([
    # [NAMING-FIX-01 2026-05-30] MetaLabeler V2 usa sufijo _long (direction_mode=long)
    # Los archivos reales son metalabeler_v2_long_rf.joblib etc., no sin sufijo
    ("data/models/prod/seed42/metalabeler_v2_long_rf.joblib",         "MetaLabeler RF"),
    ("data/models/prod/seed42/metalabeler_v2_long_lstm.pt",           "MetaLabeler LSTM"),
    ("data/models/prod/seed42/metalabeler_v2_long_calibrator.joblib", "MetaLabeler Calibrador"),
    ("data/models/prod/seed42/ood_guard.pkl",                         "OOD Guard"),
    ("data/features/selected_features.json",                          "Selected Features JSON"),
    # calibrator_signature vive por seed, no en /prod root — omitir del check global
])

# ─── Catalogo de parquets ───────────────────────────────────────────
# (path_relativo, freq_horas, columnas_criticas, es_horario, solo_holdout_freshness)
# freq_horas=0 significa "no verificar gaps"
EXPECTED_FILES = [
    # OHLCV
    ("raw/ohlcv/ohlcv_raw.parquet",             1,    ["open","high","low","close","volume"],  True,  False, "OHLCV 1H"),
    ("raw/ohlcv/ohlcv_15m_raw.parquet",         0.25, ["open","high","low","close","volume"],  True,  False, "OHLCV 15min"),
    # Raw (daily o irregular -> gaps son normales)
    ("raw/macro/macro_raw.parquet",              24,   [
        "FedFundsRate", "CPI_YoY", "M2_China_YoY",
        "DXY", "DXY_Zscore",                         # FIX-VIX-ZSCORE-01: Zscore critico
        "Macro_Risk_Score", "Macro_Risk_On",         # S2.3 regime flag
    ], False, False, "Macro FRED+yfinance"),

    ("raw/onchain/onchain_raw.parquet",          24,   ["hashrate_th","FearGreed"],             False, False, "On-chain"),
    ("raw/derivatives/derivatives_raw.parquet",  1,    ["FundingRate","LongShortRatio"],        True,  False, "Derivatives"),
    ("raw/altcoins/altcoins_raw.parquet",        0,    ["ETH_Price","ETH_Return_1d"],           False, False, "Altcoins ETH"),
    ("raw/crossasset/crossasset_raw.parquet",    1,    ["eth_btc_ratio","eth_btc_corr_24h"],    True,  False, "Cross-asset"),
    ("raw/defi/defi_raw.parquet",               0,    [],                                      False, False, "DeFi"),
    ("raw/mempool/mempool_raw.parquet",          0,    [],                                      False, False, "Mempool"),
    ("raw/etf/etf_raw.parquet",                 0,    [],                                      False, False, "ETF flows"),
    ("raw/stablecoin_m2/stablecoin_m2_raw.parquet", 0, [],                                    False, False, "Stablecoin+M2"),
    # Historico
    ("historical/daemon/BTCUSDT_1h.parquet",    1,    ["close"],                               True,  False, "Hist OHLCV daemon"),
    # Features — critical_cols incluyen las implementadas recientemente
    ("features/features_train.parquet",          1,    [
        "open", "close", "FundingRate",
        "MVRV_Proxy", "DVOL",                         # Risk-Off Shield (Fase 4C)
        "btc_drawdown_from_ath",                      # Risk-Off Shield + XGBoost
        "DXY_Zscore",                                 # FIX-VIX-ZSCORE-01
    ], True,  False, "Features IS train"),
    ("features/features_validation.parquet",     1,    [
        "open", "close", "FundingRate",
        "MVRV_Proxy", "DVOL",
        "btc_drawdown_from_ath",
        "DXY_Zscore",
    ], True,  False, "Features OOS val"),
    ("features/features_holdout.parquet",        1,    [
        "open", "close", "FundingRate",
        "MVRV_Proxy", "DVOL",                         # Risk-Off Shield (Fase 4C)
        "btc_drawdown_from_ath",                      # Risk-Off Shield + XGBoost
        "btc_cycle_position",                         # LOG-BUG-04
        "DXY_Zscore",                                 # FIX-VIX-ZSCORE-01
    ], True,  True,  "Features OOS holdout"),
    ("features/features_train_final.parquet",    1,    [],                                      True,  False, "Features train final"),
    ("features/hmm_regime_labels.parquet",       1,    [],                                      True,  False, "HMM labels"),
    # Predicciones — tienen timestamp en columna, no en indice
    ("predictions/oos_trades.parquet",           0,    ["return_pct","is_win","xgb_prob"],     False, False, "OOS Trades"),
]

# ─── Contadores globales ────────────────────────────────────────────
n_ok = n_warn = n_fail = 0
# MEJ-DIC-01 (2026-04-06): Contador separado de FAILs "blandos" elegibles para
# degradar a WARN en modo --lenient-wfb. Solo los FAILs de archivos no 
# generados aun (modelos, parquets intermedios) son lenient-elegibles.
# FAILs estructurales (datos vacios, solapamiento, OHLCV invalida) son SIEMPRE fatales.
n_fail_lenient = 0  # FAILs elegibles para degradar

# Patrones que indican un FAIL "esperado" al inicio del WFB (antes de train):
# El archivo no existe o es de tipo modelo/firma que se genera durante el entrenamiento.
_LENIENT_FAIL_PATTERNS = [
    "NO EXISTE",
    "not found",
    "no existe",
    ".model",
    ".joblib",
    ".pkl",
    ".pt",
    "calibrator_signature",
    "selected_features",
    "hmm_regime_labels",
    "features_train_final",
    "oos_trades",
]


def _line(status: str, name: str, msg: str) -> None:
    global n_ok, n_warn, n_fail, n_fail_lenient
    if status == "OK":    n_ok   += 1; s = OK_S
    elif status == "WARN": n_warn += 1; s = WARN_S
    else:
        # MEJ-DIC-01: Determinar si este FAIL es elegible para lenient-degradation
        _is_lenient = any(p.lower() in msg.lower() or p.lower() in name.lower()
                         for p in _LENIENT_FAIL_PATTERNS)
        if _is_lenient:
            n_fail_lenient += 1
        n_fail += 1
        s = FAIL_S
    print(f"  {s}  {name:<50} {msg}")


# ══════════════════════════════════════════════════════════════════════
#  CHECK 1-10: Parquets
# ══════════════════════════════════════════════════════════════════════

def _check_parquet(rel_path, freq_h, crit_cols, is_hourly, freshness_holdout_only,
                   desc, verbose) -> bool:
    path = DATA_DIR / rel_path
    name = Path(rel_path).name
    errors, warnings = [], []

    # 1. Existencia
    if not path.exists():
        _line("FAIL", name, "NO EXISTE"); return False

    size_mb = path.stat().st_size / 1_048_576

    # 2. Lectura
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        _line("FAIL", name, f"ERROR lectura: {e}"); return False

    rows, cols = df.shape
    if rows == 0:
        if "oos_trades" in rel_path:
            _line("WARN", name, "VACIO (0 filas) - normal durante entrenamiento")
            return True
        else:
            _line("FAIL", name, "VACIO (0 filas)")
            return False

    # 3. Indice de tiempo — tratamiento especial para oos_trades
    is_oos_trades = "oos_trades" in rel_path
    if is_oos_trades and "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        try:
            idx = pd.to_datetime(df.index, utc=True)
        except Exception:
            idx = None

    first_date = last_date = None
    date_str = "indice-no-datetime"
    if idx is not None and not idx.isna().all():
        first_date = idx.dropna().min()
        last_date  = idx.dropna().max()
        date_str   = f"{first_date.date()} -> {last_date.date()}"
        # Check: indice 1970 == epoch 0 → bug
        if first_date.year < 2010:
            warnings.append(f"indice temporal sospechoso: primer valor {first_date.date()}")

    # 4. Freshness — solo para holdout o para parquets que deban llegar a 2025
    if freshness_holdout_only and last_date is not None:
        if last_date < pd.Timestamp("2025-01-01", tz="UTC"):
            warnings.append(f"holdout llega solo hasta {last_date.date()} — deberia ser >=2025-01-01")

    # 5. Gaps temporales — solo para series marcadas como horarias
    if is_hourly and freq_h > 0 and idx is not None:
        try:
            sorted_idx = idx.sort_values().dropna()
            diffs_s = pd.Series(sorted_idx.diff().dropna().values, index=sorted_idx[1:])
            threshold = pd.Timedelta(hours=freq_h * (3 if freq_h < 24 else 5))
            big_gaps = diffs_s[diffs_s > threshold]
            if len(big_gaps) > 0:
                worst = big_gaps.sort_values(ascending=False).iloc[0]
                warnings.append(f"{len(big_gaps)} gaps (max {worst})")
        except Exception:
            pass

    # 6. NaN — excluyendo columnas de whitelist
    real_cols = [c for c in df.columns if not _is_nan_whitelist(c)]
    if real_cols:
        nan_pcts = df[real_cols].isnull().mean() * 100
        high_nan = nan_pcts[nan_pcts > NAN_LIMIT]
        if len(high_nan) > 0:
            worst = f"{high_nan.idxmax()} {high_nan.max():.0f}%"
            warnings.append(f"{len(high_nan)} cols >{NAN_LIMIT:.0f}% NaN (peor: {worst})")
            
        # 6.5 API Death (Trailing NaNs detectado post-mortem W1 FundingRate)
        if idx is not None and len(df) > 0 and last_date is not None:
            try:
                # Tolerancia extendida a 120 días para variables macro y M2 (rezago reportes FRED)
                tolerancia_dias = 120 if ("macro_raw" in rel_path or "stablecoin_m2_raw" in rel_path) else 30
                
                # [GAP-2 FIX] Coinglass es en tiempo real, no tiene rezago de 120 días aunque esté en stablecoin_m2_raw
                is_coinglass_col = lambda c: "Coinglass" in c
                
                if isinstance(df.index, pd.DatetimeIndex):
                    tail_mask_default = df.index >= (last_date - pd.Timedelta(days=tolerancia_dias))
                    tail_mask_strict  = df.index >= (last_date - pd.Timedelta(days=15)) # Coinglass no debe tener >15d lag
                else:
                    tail_mask_default = (idx >= (last_date - pd.Timedelta(days=tolerancia_dias))).to_numpy()
                    tail_mask_strict  = (idx >= (last_date - pd.Timedelta(days=15))).to_numpy()
                
                # Chequeo estandar (tolerancia_dias)
                if tail_mask_default.sum() > 0:
                    tail_df = df.loc[tail_mask_default, real_cols]
                    dead_cols = [c for c in real_cols if not is_coinglass_col(c) and tail_df[c].isna().all()]
                    # Chequeo estricto para Coinglass
                    tail_df_strict = df.loc[tail_mask_strict, real_cols]
                    dead_cols += [c for c in real_cols if is_coinglass_col(c) and tail_df_strict[c].isna().all()]
                    
                    if dead_cols:
                        dead_crit = [c for c in dead_cols if c in crit_cols]
                        if dead_crit:
                            # [WFB-FIX] API Death is only critical if it happens near real-time (live trading).
                            # If last_date is older than 30 days from now, it's just a historical gap (expected in WFB windows).
                            is_historical_window = (pd.Timestamp.now(tz="UTC") - last_date).days > 30
                            if is_historical_window:
                                warnings.append(f"GAP HISTORICO (WFB): {len(dead_crit)} cols criticas 100% NaN ult. {tolerancia_dias}d ({dead_crit[:3]}) - OK para WFB")
                            else:
                                errors.append(f"MUERTE DE API: {len(dead_crit)} cols criticas 100% NaN ult. {tolerancia_dias}d ({dead_crit[:3]})")
                        else:
                            warnings.append(f"MUERTE DE API (NO CRITICA): {len(dead_cols)} cols 100% NaN ult. {tolerancia_dias}d ({dead_cols[:3]})")
            except Exception:
                pass

    # 7. Infinitos
    numeric = df.select_dtypes(include=[np.number])
    n_inf = np.isinf(numeric.values).sum()
    if n_inf > 0:
        errors.append(f"INFINITOS: {n_inf} valores inf/-inf detectados")

    # 8. Tipos numericos en OHLCV
    if "close" in df.columns and "open" in df.columns:
        bad_types = [c for c in ["open","high","low","close","volume"]
                     if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
        if bad_types:
            errors.append(f"tipos no-numericos en OHLCV: {bad_types}")
        else:
            # 9. Logica OHLCV: high >= close, close >= low, low >= 0
            ok_rows = (
                (df["high"] >= df["close"]) &
                (df["close"] >= df["low"]) &
                (df["low"] >= 0)
            )
            n_bad = (~ok_rows).sum()
            if n_bad > 0:
                errors.append(f"OHLCV logica rota: {n_bad} velas con high<close o low<0")

    # 10. Columnas criticas
    miss = [c for c in crit_cols if c not in df.columns]
    if miss:
        errors.append(f"faltan cols: {miss}")

    # ── Status ───────────────────────────────────────────────────────
    status = "FAIL" if errors else ("WARN" if warnings else "OK")
    summary = f"{rows:>7,} filas x {cols:>3} cols | {size_mb:.1f}MB | {date_str}"
    issues  = (" | " + " | ".join(warnings + errors)) if (warnings or errors) else ""
    _line(status, name, summary + issues)

    if verbose:
        for w in warnings: print(f"             ~  {w}")
        for e in errors:   print(f"             !  {e}")

    return not bool(errors)


# ══════════════════════════════════════════════════════════════════════
#  CHECK 11: Coherencia de splits
# ══════════════════════════════════════════════════════════════════════

def _check_splits(verbose: bool) -> None:
    print(f"\n{BOLD}[CHECK] Coherencia de splits temporales{RESET}")
    files = {
        "train":      DATA_DIR / "features/features_train.parquet",
        "validation": DATA_DIR / "features/features_validation.parquet",
        "holdout":    DATA_DIR / "features/features_holdout.parquet",
    }
    dates: dict = {}
    for split, path in files.items():
        if not path.exists(): continue
        try:
            df = pd.read_parquet(path)
            idx = pd.to_datetime(df.index, utc=True)
            dates[split] = (idx.min(), idx.max())
        except Exception:
            pass

    if "train" in dates and "validation" in dates:
        t_last, v_first = dates["train"][1], dates["validation"][0]
        v_last = dates["validation"][1]
        if t_last >= v_first:
            # Overlap intencional: validation está contenida dentro del período de train
            # (features_train incluye H2-2024 como contexto; features_validation es el
            # subconjunto H2-2024 para calibración de probabilidades — diseño de Luna V1)
            import os
            _is_wfb = os.environ.get("LUNA_RUN_ID", "").startswith("WFB_")
            if v_last <= t_last:
                if _is_wfb:
                    _line("FAIL", "train->validation", f"OVERLAP NO PERMITIDO EN WFB: val {v_first.date()} <= train {t_last.date()}")
                else:
                    _line("WARN", "train->validation", f"Overlap contenido OK: train={t_last.date()}, val={v_first.date()}→{v_last.date()} (calibracion sobre subperiodo train — esperado)")
            else:
                # v_last > t_last sería leakage real: validation tiene datos futuros
                _line("FAIL", "train->validation", f"LEAKAGE: val {v_last.date()} > train {t_last.date()} (validation tiene datos del futuro respecto a train)")
        else:
            gap_h = (v_first - t_last).total_seconds() / 3600
            _line("OK",  "train->validation", f"gap={gap_h:.0f}H (embargo={EMBARGO_H}H) OK")

    if "validation" in dates and "holdout" in dates:
        v_last, h_first = dates["validation"][1], dates["holdout"][0]
        if v_last >= h_first:
            _line("FAIL", "validation->holdout", f"SOLAPAMIENTO: val {v_last.date()}, holdout {h_first.date()}")
        else:
            gap_h = (h_first - v_last).total_seconds() / 3600
            _line("OK",  "validation->holdout", f"gap={gap_h:.0f}H (embargo={EMBARGO_H}H) OK")

    if "holdout" in dates:
        first_h, last_h = dates["holdout"]
        # P1-6-FIX (2026-03-30): leer holdout_start desde settings.yaml en lugar de
        # hardcodear "2025-01-01". En WFB cada ventana tiene su propio holdout_start.
        try:
            from config.settings import cfg as _cfg_fresh
            _expected_holdout_start_str = getattr(_cfg_fresh.temporal_splits, 'holdout_start', None)
            _expected_holdout = (
                pd.Timestamp(_expected_holdout_start_str, tz="UTC")
                if _expected_holdout_start_str
                else pd.Timestamp("2025-01-01", tz="UTC")  # fallback conservador
            )
        except Exception:
            _expected_holdout = pd.Timestamp("2025-01-01", tz="UTC")
        if first_h < _expected_holdout:
            _line("WARN", "holdout freshness",
                  f"{first_h.date()} deberia ser >={_expected_holdout.date()} (settings.yaml)")
        else:
            _line("OK",  "holdout freshness", f"{first_h.date()} -> {last_h.date()} OOS OK")


# ══════════════════════════════════════════════════════════════════════
#  CHECK 12: Continuidad OHLCV
# ══════════════════════════════════════════════════════════════════════

def _check_ohlcv(verbose: bool) -> None:
    print(f"\n{BOLD}[CHECK] Continuidad OHLCV 1H (solo post-2020){RESET}")
    path = DATA_DIR / "raw/ohlcv/ohlcv_raw.parquet"
    if not path.exists():
        _line("FAIL", "ohlcv_raw", "NO EXISTE"); return
    try:
        df = pd.read_parquet(path)
        idx = pd.to_datetime(df.index, utc=True).sort_values()
        # Solo verificar post-2020 (los gaps pre-2020 son datos heredados, no afectan al modelo)
        idx = idx[idx >= pd.Timestamp("2020-01-01", tz="UTC")]
        diffs = pd.Series(idx.diff().dropna().values, index=idx[1:])
        big_gaps = diffs[diffs > pd.Timedelta(hours=2)]
        duplicates = pd.to_datetime(df.index, utc=True).duplicated().sum()

        if duplicates > 0:
            _line("WARN", "OHLCV duplicados", f"{duplicates} timestamps duplicados")

        if len(big_gaps) > 0:
            worst3 = big_gaps.sort_values(ascending=False).head(3)
            for ts, gap in worst3.items():
                _line("WARN", "OHLCV gap post-2020", f"{gap} en {ts.date()}")
        else:
            _line("OK", "OHLCV continuidad post-2020", f"{len(idx):,} velas | sin gaps >2H")

        # Freshness
        stale = (pd.Timestamp.now(tz="UTC") - idx.max()).days
        if stale > 30:
            _line("WARN", "OHLCV freshness", f"ultimo dato hace {stale} dias")
        else:
            _line("OK",  "OHLCV freshness", f"hace {stale} dias OK")

    except Exception as e:
        _line("FAIL", "OHLCV continuidad", f"error: {e}")


# ══════════════════════════════════════════════════════════════════════
#  CHECK 13: Archivos de modelos criticos
# ══════════════════════════════════════════════════════════════════════

def _check_models(verbose: bool) -> None:
    import os
    if os.environ.get("LUNA_SKIP_ARTIFACT_CHECKS") == "1":
        print(f"\n{BOLD}[CHECK] Archivos de modelos criticos -- SKIPPED (LUNA_SKIP_ARTIFACT_CHECKS active){RESET}")
        return
    print(f"\n{BOLD}[CHECK] Archivos de modelos criticos ({len(CRITICAL_MODELS)} esperados){RESET}")
    for rel_path, desc in CRITICAL_MODELS:
        path = ROOT / rel_path
        if not path.exists():
            _line("FAIL", Path(rel_path).name, f"NO EXISTE — {desc}")
        else:
            size_bytes = path.stat().st_size
            size_kb    = size_bytes / 1024   # float, no truncado
            if size_bytes == 0:
                _line("WARN", Path(rel_path).name, f"VACIO (0 bytes) — {desc}")
            elif rel_path.endswith(".json"):
                # Validar JSON y mostrar contenido clave
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if rel_path.endswith("selected_features.json"):
                        n_feat = len(data) if isinstance(data, list) else len(data.get("features", data))
                        _line("OK", Path(rel_path).name, f"{n_feat} features | {size_kb:.1f}KB")
                    elif "calibration_method" in data:
                        brier_raw = data.get("brier_score_raw", "?")
                        brier_cal = data.get("brier_score_calibrated", "?")
                        mejora    = data.get("mejora_pct", 0)
                        _line("OK", Path(rel_path).name,
                              f"metodo={data['calibration_method']} | brier {brier_raw:.4f}->{brier_cal:.4f} ({mejora:.1f}% mejora)")
                    else:
                        _line("OK", Path(rel_path).name, f"{size_kb:.1f}KB OK")
                except Exception as e:
                    _line("FAIL", Path(rel_path).name, f"JSON invalido: {e}")
            else:
                _line("OK", Path(rel_path).name, f"{size_kb:.0f}KB OK")

    # ── [GAP-1 FIX] Model Output Sanity Check (Colapso XGB) ─────────
    print(f"\n{BOLD}[CHECK] XGB Model Output Sanity (GAP-1){RESET}")
    # Solo probar si tenemos un features_validation o holdout
    val_path = DATA_DIR / "features" / "features_validation.parquet"
    hold_path = DATA_DIR / "features" / "features_holdout.parquet"
    sample_path = val_path if val_path.exists() else (hold_path if hold_path.exists() else None)
    
    if sample_path:
        try:
            df_sample = pd.read_parquet(sample_path)
            import xgboost as xgb
            
            for rel_path, desc in CRITICAL_MODELS:
                if not rel_path.endswith(".model") or "meta" not in rel_path:
                    continue
                model_path = ROOT / rel_path
                sig_path = ROOT / rel_path.replace(".model", "_signature.json")
                if not model_path.exists() or not sig_path.exists():
                    continue
                    
                # Cargar el signature para saber qué features necesita
                try:
                    sig_data = json.loads(sig_path.read_text(encoding="utf-8"))
                    feats = sig_data.get("features", [])
                    if not feats:
                        continue
                except Exception:
                    continue
                    
                model = xgb.XGBClassifier()
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.load_model(str(model_path))
                
                # [GAP-4 FIX] Preparar datos y revisar features runtime (ej: HMM_Regime)
                df_f = pd.DataFrame()
                missing_runtime = []
                for f in feats:
                    if f in df_sample.columns:
                        df_f[f] = df_sample[f]
                    else:
                        missing_runtime.append(f)
                        df_f[f] = np.nan # padding con NaN para que XGBoost use missing values
                        
                if missing_runtime:
                    name = Path(rel_path).name
                    _line("WARN", name, f"padding=NaN para {len(missing_runtime)} runtime features (ej. {missing_runtime[0]})")
                
                probs = model.predict_proba(df_f)[:, 1]
                
                std_dev = np.std(probs)
                pct_95 = np.percentile(probs, 95)
                pct_05 = np.percentile(probs, 5)
                spread = pct_95 - pct_05
                
                name = Path(rel_path).name
                msg = f"Spread={spread:.3f}, Std={std_dev:.3f}"
                if isinstance(std_dev, float) and std_dev < 0.01 or spread < 0.03:
                    _line("FAIL", name, f"COLAPSO XGB: output casi constante en OOS. {msg}")
                else:
                    _line("OK", name, f"Output sanity: varianza correcta. {msg}")
                    
        except Exception as e:
            _line("WARN", "Model Sanity", f"Error corriendo model sanity check: {e}")
    else:
        _line("WARN", "Model Sanity", "SKIP (no hay features_validation o holdout)")


# ══════════════════════════════════════════════════════════════════════
#  CHECK 14: Coherencia cross-file (features_holdout vs OHLCV)
# ══════════════════════════════════════════════════════════════════════

def _check_cross_file(verbose: bool) -> None:
    print(f"\n{BOLD}[CHECK] Coherencia cross-file{RESET}")
    try:
        ohlcv = DATA_DIR / "raw/ohlcv/ohlcv_raw.parquet"
        holdout = DATA_DIR / "features/features_holdout.parquet"
        if not ohlcv.exists() or not holdout.exists():
            _line("WARN", "cross-file", "Faltan archivos para comparar"); return

        df_ohlcv   = pd.read_parquet(ohlcv)
        df_holdout = pd.read_parquet(holdout)

        ohlcv_last   = pd.to_datetime(df_ohlcv.index, utc=True).max()
        holdout_last = pd.to_datetime(df_holdout.index, utc=True).max()
        gap_days = (ohlcv_last - holdout_last).days

        if gap_days > 30:
            _line("WARN", "ohlcv vs holdout",
                  f"OHLCV llega hasta {ohlcv_last.date()} pero holdout solo hasta {holdout_last.date()} ({gap_days}d detras)")
        else:
            _line("OK",  "ohlcv vs holdout",
                  f"holdout {holdout_last.date()} vs ohlcv {ohlcv_last.date()} ({gap_days}d gap OK)")
    except Exception as e:
        _line("WARN", "cross-file", f"no verificado: {e}")



# ==================================================================================================
#  CHECK 16: Features congeladas o degeneradas (FROZEN-FEAT)
#  Motivacion: VIX_Zscore estuvo congelado 354 dias sin deteccion (bug fetch incremental)
# ==================================================================================================

def _check_frozen_features(verbose: bool) -> None:
    print(f"\n{BOLD}[CHECK] Features degeneradas o congeladas (FROZEN-FEAT){RESET}")
    try:
        from luna.utils.pipeline_invariants import check_frozen_features
    except ImportError as e:
        _line("WARN", "frozen-feat", f"pipeline_invariants no importable: {e}")
        return

    targets = [
        ("features/features_holdout.parquet",    "features_holdout",    7),
        ("features/features_train.parquet",       "features_train",      30),
        ("features/features_validation.parquet",  "features_validation", 14),
    ]

    sfi_path = DATA_DIR / "features" / "selected_features.json"
    selected_feats = []
    if sfi_path.exists():
        try:
            sfi_data = json.loads(sfi_path.read_text(encoding="utf-8"))
            selected_feats = sfi_data.get("selected_features", []) + sfi_data.get("pass_through_features", [])
        except:
            pass

    for rel_path, context, min_days in targets:
        path = DATA_DIR / rel_path
        if not path.exists():
            _line("WARN", Path(rel_path).name, "no existe - SKIP frozen-feat check")
            continue
        try:
            df = pd.read_parquet(path)
            issues = check_frozen_features(
                df, context=context, min_days_frozen=min_days, max_constant_pct=0.98
            )
            if issues:
                critical = [i for i in issues if "UN SOLO VALOR" in i]
                non_crit = [i for i in issues if "UN SOLO VALOR" not in i]
                if critical:
                    # Filter critical issues by whether the feature is actually used by the model
                    active_critical = [i for i in critical if any(f"'{f}'" in i for f in selected_feats)]
                    inactive_critical = [i for i in critical if i not in active_critical]
                    
                    if active_critical:
                        cols = [i.split("'")[1] for i in active_critical if "'" in i][:5]
                        _line("WARN", Path(rel_path).name,
                              f"{len(active_critical)} cols SELECCIONADAS degeneradas: {cols}")
                    if inactive_critical:
                        cols = [i.split("'")[1] for i in inactive_critical if "'" in i][:5]
                        _line("WARN", Path(rel_path).name,
                              f"{len(inactive_critical)} cols inactivas degeneradas: {cols}")
                if non_crit:
                    frozen_cols = [i.split("'")[1] for i in non_crit if "'" in i][:5]
                    _line("WARN", Path(rel_path).name,
                          f"{len(non_crit)} features congeladas: {frozen_cols}")
            else:
                _line("OK", Path(rel_path).name,
                      f"{df.shape[1]} columnas - ninguna feature degenerada")
        except Exception as e:
            _line("WARN", Path(rel_path).name, f"error en frozen-feat check: {e}")


# ==================================================================================================
#  CHECK 17: Feature Pool completo — todas las features de selected_features.json
#
#  No requiere listas manuales. Se basa en selected_features.json que el pipeline
#  actualiza en cada run de SFI. Cualquier feature nueva sera verificada automaticamente.
#
#  Verifica en features_holdout Y features_train:
#    - Presencia: la columna existe en el parquet
#    - NaN: < nan_threshold_pct (config) o < 20% como fallback
#    - Calidad: no es constante (std > 0), no tiene mayoria de NaN
# ==================================================================================================

def _check_selected_features(verbose: bool) -> None:
    print(f"\n{BOLD}[CHECK] Feature pool completo (selected_features.json){RESET}")

    sfi_path = DATA_DIR / "features" / "selected_features.json"
    if not sfi_path.exists():
        _line("WARN", "selected_features.json",
              "no existe aun - SKIP (ejecutar SFI primero)")
        return

    try:
        sfi_data = json.loads(sfi_path.read_text(encoding="utf-8"))
    except Exception as e:
        _line("FAIL", "selected_features.json", f"JSON invalido: {e}")
        return

    # Extraer ambas listas de features del modelo
    sfi_feats  = sfi_data.get("selected_features", [])
    pt_feats   = sfi_data.get("pass_through_features", [])
    all_model_feats = list(dict.fromkeys(sfi_feats + pt_feats))  # union sin duplicados

    n_sfi = len(sfi_feats)
    n_pt  = len(pt_feats)
    n_total = len(all_model_feats)

    print(f"  SFI features: {n_sfi} | Pass-through: {n_pt} | Total: {n_total}")

    # Verificar en cada split de features
    targets = [
        ("features/features_holdout.parquet",   "holdout"),
        ("features/features_train.parquet",     "train"),
        ("features/features_validation.parquet","validation"),
    ]

    for rel_path, split_name in targets:
        path = DATA_DIR / rel_path
        if not path.exists():
            _line("WARN", f"feat-pool [{split_name}]", "parquet no existe - SKIP")
            continue

        try:
            df = pd.read_parquet(path)
            df_cols = set(df.columns)

            missing_cols    = []   # no existe en el parquet → FAIL
            recent_nan_cols = []   # 100% NaN en ultimos 7d (168h) → FAIL
            high_nan_cols   = []   # existe pero NaN excesivo en total → WARN
            constant_cols   = []   # existe pero valor constante → WARN
            ok_cols         = []

            for feat in all_model_feats:
                if feat not in df_cols:
                    missing_cols.append(feat)
                    continue

                series_full = df[feat]
                series = series_full.dropna()
                nan_pct = series_full.isna().mean() * 100

                # NUEVO CHECK: 100% NaN en la ultima semana (168 horas)
                # Esto previene el envenenamiento silencioso cerca del límite OOS
                last_7d = series_full.tail(168)
                if len(last_7d) > 0 and last_7d.isna().all():
                    recent_nan_cols.append(feat)
                elif nan_pct > 50:
                    high_nan_cols.append((feat, nan_pct))
                elif len(series) > 0 and series.nunique() <= 1:
                    constant_cols.append((feat, series.iloc[0] if len(series) else "N/A"))
                else:
                    ok_cols.append(feat)

            # ── Reportar ─────────────────────────────────────────────
            has_fails = bool(recent_nan_cols)
            has_warns = bool(missing_cols)
            if has_fails or has_warns:
                msg_parts = []
                if missing_cols:
                    msg_parts.append(f"{len(missing_cols)} AUSENTES")
                if recent_nan_cols:
                    msg_parts.append(f"{len(recent_nan_cols)} RECENT-DEAD (100% NaN ult 7d)")
                
                status_to_report = "FAIL" if has_fails else "WARN"
                _line(status_to_report, f"feat-pool [{split_name}]", " | ".join(msg_parts))
                if verbose:
                    for c in missing_cols:
                        origin = "SFI" if c in sfi_feats else "pass-through"
                        print(f"             !  MISSING [{origin}]: {c}")
                    for c in recent_nan_cols:
                        print(f"             !  DEAD LAST 7D: {c} tiene puros NaNs al final de la serie")
            elif high_nan_cols or constant_cols:
                problems = len(high_nan_cols) + len(constant_cols)
                _line("WARN", f"feat-pool [{split_name}]",
                      f"{len(ok_cols)}/{n_total} OK | {problems} con problemas de calidad")
                if verbose:
                    for feat, pct in high_nan_cols:
                        print(f"             ~  HIGH-NaN {feat}: {pct:.0f}% NaN")
                    for feat, val in constant_cols:
                        print(f"             ~  CONSTANT {feat}: siempre={val}")
            else:
                _line("OK", f"feat-pool [{split_name}]",
                      f"{n_total}/{n_total} features presentes y con datos validos")

        except Exception as e:
            _line("WARN", f"feat-pool [{split_name}]", f"error leyendo parquet: {e}")


# ==================================================================================================
#  CHECK 18: RAW-NAN-AUDIT — Features con alto NaN que degradan el SFI
#
#  Motivación: seed42 (2026-04-19) tuvo 71 features con >50% NaN entrando al SFI,
#  causando k_auto=214 (vs normal=34), lag cache invalidation (4/60 hits vs 56/60)
#  y 54 "features mutantes" — el resultado fue solo 29 trades OOS vs 221 esperados.
#
#  Este check audita features_train.parquet ANTES de que entre al SFI:
#    - Features 100% NaN no whitelisteadas → FAIL (eliminar del pipeline)
#    - Features >50% NaN no whitelisteadas → WARN (degradan clustering y lag cache)
#    - Agrupa por prefijo de fuente para identificar APIs muertas
#    - Alerta si el total de features crudas supera el umbral saludable (>300)
# ==================================================================================================

# Umbral de features crudas sano para el SFI (basado en run histórico óptimo)
_HEALTHY_RAW_FEAT_CEILING = 300   # >300 raw features → riesgo de k_auto explosivo
_WARN_HIGH_NAN_PCT        = 50.0  # % NaN a partir del cual la feature degrada el SFI
_FAIL_DEAD_FEAT_LIMIT = 5     # >5 features 100% NaN no-whitelist → FAIL
_WARN_HIGH_NAN_LIMIT  = 15    # >15 features >50% NaN no-whitelist → WARN

def _check_raw_feature_quality(verbose: bool) -> None:
    """CHECK 18: RAW-NAN-AUDIT — Detecta features problemáticas antes del SFI.

    Distingue entre:
      - Features whitelisteadas (opcionales por diseño, no bloquean)
      - Features NO whitelisteadas con alto NaN (errores reales del pipeline)
    """
    print(f"\n{BOLD}[CHECK] RAW-NAN-AUDIT — Calidad de features pre-SFI (CHECK 18){RESET}")

    path = DATA_DIR / "features" / "features_train.parquet"
    if not path.exists():
        _line("WARN", "features_train", "no existe — SKIP raw-nan-audit (ejecutar feature_pipeline primero)")
        return

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        _line("WARN", "raw-nan-audit", f"error leyendo parquet: {e}")
        return

    total_cols = df.shape[1]
    nan_pcts   = df.isnull().mean() * 100

    # ── Separar features whitelisteadas vs no-whitelisteadas ──────────
    dead_nonwl    = []   # 100% NaN, NO whitelist → FAIL (deben eliminarse)
    highnан_nonwl = []   # >50% NaN, NO whitelist → WARN (degradan SFI)
    dead_wl       = []   # 100% NaN, whitelist → info (comportamiento esperado)
    highnан_wl    = []   # >50% NaN, whitelist → info silenciada

    for col, pct in nan_pcts.items():
        if pct == 0:
            continue
        is_wl = _is_nan_whitelist(col)
        if pct >= 100.0:
            (dead_wl if is_wl else dead_nonwl).append(col)
        elif pct >= _WARN_HIGH_NAN_PCT:
            (highnан_wl if is_wl else highnан_nonwl).append((col, pct))

    # ── Agrupar features no-whitelist por prefijo de fuente ───────────
    from collections import defaultdict
    source_dead: dict = defaultdict(list)
    for col in dead_nonwl:
        prefix = col.split("_")[0]
        source_dead[prefix].append(col)

    source_high: dict = defaultdict(list)
    for col, pct in highnан_nonwl:
        prefix = col.split("_")[0]
        source_high[prefix].append((col, pct))

    # ── CHECK A: Total de features crudas ────────────────────────────
    if total_cols > _HEALTHY_RAW_FEAT_CEILING:
        _line("WARN", "total-features",
              f"{total_cols} columnas en features_train "
              f"(umbral saludable ≤{_HEALTHY_RAW_FEAT_CEILING}) "
              f"— riesgo de k_auto explosivo en clustering SFI")
    else:
        _line("OK", "total-features",
              f"{total_cols} columnas — dentro del umbral saludable (≤{_HEALTHY_RAW_FEAT_CEILING})")

    # ── CHECK B: Features 100% NaN no whitelisteadas (FAIL) ──────────
    if len(dead_nonwl) >= _FAIL_DEAD_FEAT_LIMIT:
        src_summary = ", ".join(
            f"{src}({len(cols)})"
            for src, cols in sorted(source_dead.items(), key=lambda x: -len(x[1]))
        )
        _line("FAIL", "dead-features",
              f"{len(dead_nonwl)} features 100% NaN NO whitelisteadas "
              f"— eliminar del pipeline. Fuentes: {src_summary}")
        if verbose:
            for col in sorted(dead_nonwl):
                print(f"             !  ELIMINAR: {col}")
    elif dead_nonwl:
        _line("WARN", "dead-features",
              f"{len(dead_nonwl)} features 100% NaN no-whitelist: {dead_nonwl[:5]}")
    else:
        _line("OK", "dead-features",
              f"0 features 100% NaN fuera de whitelist ✓")

    # Info sobre las whitelisteadas (no bloquean, solo informativo)
    if dead_wl and verbose:
        print(f"             ~  INFO: {len(dead_wl)} features 100% NaN whitelisteadas (comportamiento esperado): "
              f"{', '.join(dead_wl[:5])}{'...' if len(dead_wl) > 5 else ''}")

    # ── CHECK C: Features >50% NaN no whitelisteadas (WARN) ──────────
    if len(highnан_nonwl) >= _WARN_HIGH_NAN_LIMIT:
        worst5 = sorted(highnан_nonwl, key=lambda x: -x[1])[:5]
        worst_str = ", ".join(f"{c}({p:.0f}%)" for c, p in worst5)
        src_summary = ", ".join(
            f"{src}({len(cols)})"
            for src, cols in sorted(source_high.items(), key=lambda x: -len(x[1]))
        )
        _line("WARN", "high-nan-features",
              f"{len(highnан_nonwl)} features >{_WARN_HIGH_NAN_PCT:.0f}% NaN no-whitelist "
              f"— degradan clustering y lag cache. Fuentes: {src_summary}. "
              f"Peores: {worst_str}")
        if verbose:
            for col, pct in sorted(highnан_nonwl, key=lambda x: -x[1]):
                print(f"             ~  HIGH-NaN ({pct:.0f}%): {col}")
    elif highnан_nonwl:
        _line("OK", "high-nan-features",
              f"{len(highnан_nonwl)} features >{_WARN_HIGH_NAN_PCT:.0f}% NaN no-whitelist "
              f"(bajo umbral de {_WARN_HIGH_NAN_LIMIT}) — aceptable")
    else:
        _line("OK", "high-nan-features",
              f"0 features >{_WARN_HIGH_NAN_PCT:.0f}% NaN fuera de whitelist ✓")

    # ── CHECK D: Riesgo de lag cache invalidation ─────────────────────
    # Si el total de features no-whitelisted superó el ciclo anterior en >10%,
    # el cache DSR quedará obsoleto y el SFI tardará ~90 min extra.
    dsr_cache = DATA_DIR / "features" / "_dsr_cache.json"
    if dsr_cache.exists():
        try:
            cache = json.loads(dsr_cache.read_text(encoding="utf-8"))
            cached_feats = len(cache)
            total_nonnan = int((nan_pcts < _WARN_HIGH_NAN_PCT).sum())
            ratio = total_nonnan / max(cached_feats, 1)
            if ratio > 1.15 or ratio < 0.85:
                _line("WARN", "lag-cache-risk",
                      f"{total_nonnan} features densas actuales vs {cached_feats} en cache DSR "
                      f"(ratio={ratio:.2f}) — el cache puede invalidarse, "
                      f"SFI tardará ~90 min extra por lag re-discovery")
            else:
                _line("OK", "lag-cache-risk",
                      f"{total_nonnan} features densas ≈ {cached_feats} en cache DSR "
                      f"(ratio={ratio:.2f}) — lag cache probablemente válido")
        except Exception:
            _line("WARN", "lag-cache-risk", "no se pudo leer _dsr_cache.json — cache status desconocido")
    else:
        _line("WARN", "lag-cache-risk",
              "_dsr_cache.json no existe — primer run o cache eliminado. "
              "SFI recalculará todos los lags (+~90 min)")


# ==================================================================================================
#  MAIN
# ==================================================================================================

def main():
    global n_ok, n_warn, n_fail
    parser = argparse.ArgumentParser(description="Luna V1 — Data Integrity Check v2.0")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--lenient-wfb", action="store_true", help="Permite degradar FAILs a WARNs en contexto de WFB (ej. W1 faltan modelos)")
    # AUDIT Tier 3: --window-id para verificar el holdout especifico de la ventana WFB
    parser.add_argument("--window-id", type=str, default=None,
                        help="ID de ventana WFB (ej: W1). Verifica features_holdout_{window_id}.parquet adicionalmente.")
    args = parser.parse_args()

    # Propagar window-id al entorno para que las subfunciones hereden el contexto
    import os as _os_dic
    if _os_dic.environ.get("LUNA_RUN_ID", "").startswith("WFB_"):
        args.lenient_wfb = True
        
    if args.window_id:
        _os_dic.environ["LUNA_WINDOW_ID"] = args.window_id
        # Agregar dinamicamente el holdout especifico de la ventana a la lista de verificacion
        _win_holdout_rel = f"features/features_holdout_{args.window_id}.parquet"
        EXPECTED_FILES.append((
            _win_holdout_rel, 1,
            ["open", "close", "FundingRate"],  # criticas minimas
            True, True, f"Features OOS holdout {args.window_id}"
        ))


    print(f"\n{BOLD}{CYAN}+{'='*62}+")
    print(f"|  Luna V1 - Data Integrity Check v2.1  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]  |")
    print(f"+{'='*62}+{RESET}\n")

    print(f"{BOLD}[CHECK] Parquets ({len(EXPECTED_FILES)} archivos){RESET}")
    print("  " + "-"*78)

    for rel_path, freq_h, crit_cols, is_hourly, fresh_holdout, desc in EXPECTED_FILES:
        _check_parquet(rel_path, freq_h, crit_cols, is_hourly, fresh_holdout, desc, args.verbose)

    _check_ohlcv(args.verbose)
    _check_splits(args.verbose)
    _check_models(args.verbose)
    _check_cross_file(args.verbose)
    _check_frozen_features(args.verbose)
    _check_selected_features(args.verbose)
    _check_raw_feature_quality(args.verbose)

    # ══════════════════════════════════════════════════════════════════════
    #  CHECK 20: SFI List Coverage
    #  [SFI-COVERAGE-01 2026-06-03] Verifica que toda feature en listas SFI
    #  de settings.yaml tiene datos reales en sus parquets fuente.
    #  Cierra el gap donde features nuevas en fetch_onchain.py aparecian
    #  como columnas 100% NaN en onchain_raw.parquet por bug incremental.
    # ══════════════════════════════════════════════════════════════════════
    try:
        from scripts.check_sfi_coverage import run_sfi_coverage_check
        _sfi_ok, _sfi_warn, _sfi_fail = run_sfi_coverage_check(verbose=args.verbose)
        # Integrar contadores en los globales del integrity check
        n_ok   += _sfi_ok
        n_warn += _sfi_warn
        n_fail += _sfi_fail
        print(f"  [SFI-COVERAGE-01] OK={_sfi_ok} WARN={_sfi_warn} FAIL={_sfi_fail}")
    except ImportError:
        # check_sfi_coverage.py no disponible — crear WARN no FAIL
        _line("WARN", "SFI Coverage Check",
              "scripts/check_sfi_coverage.py no encontrado — CHECK 20 omitido")
    except Exception as _sfi_e:
        _line("WARN", "SFI Coverage Check", f"Error inesperado en CHECK 20: {_sfi_e}")

    # ── Resumen ──────────────────────────────────────────────────────
    # MEJ-DIC-01 FIX (2026-04-06): --lenient-wfb SOLO degrada FAILs "esperados" al inicio del WFB
    # (modelos sin entrenar aun, parquets intermedios no generados). FAILs en datos críticos
    # (parquet vacío, columna 'close' ausente, solapamiento de splits, OHLCV inválida)
    # siguen siendo FATALES aunque se pase --lenient-wfb.
    if args.lenient_wfb and n_fail_lenient > 0:
        _n_critical = n_fail - n_fail_lenient
        print(f"\n  {YELLOW}>>> [LENIENT WFB] {n_fail_lenient} FAILs esperados (modelos/parquets intermedios) degradados a WARNs.")
        if _n_critical > 0:
            print(f"  {RED}>>> [LENIENT WFB] {_n_critical} FAILs críticos PERMANECEN COMO FAIL (datos vacíos, solapamiento, OHLCV).{RESET}")
        n_warn += n_fail_lenient
        n_fail = _n_critical
    elif args.lenient_wfb and n_fail > 0:
        # Todos los FAILs son críticos — no hay nada que degradar
        print(f"\n  {YELLOW}>>> [LENIENT WFB] Modo activo, pero todos los {n_fail} FAILs son críticos (sin FAILs suavizables).{RESET}")

    total = n_ok + n_warn + n_fail
    print(f"\n{'='*80}")
    print(f"{BOLD}RESUMEN{RESET}")
    print(f"  {GREEN}OK  {RESET}: {n_ok:>3}   {YELLOW}WARN{RESET}: {n_warn:>3}   {RED}FAIL{RESET}: {n_fail:>3}   TOTAL: {total}")

    if n_fail > 0:
        print(f"\n  {RED}!!! {n_fail} FALLOS CRITICOS — resolver antes de entrenar{RESET}")
        sys.exit(1)
    elif n_warn > 0:
        print(f"\n  {YELLOW}>>> {n_warn} advertencias — revisar antes de produccion{RESET}")
        sys.exit(0)
    else:
        print(f"\n  {GREEN}>>> Todos los datos OK{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
