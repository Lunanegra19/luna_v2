"""
check_sfi_coverage.py
=====================
CHECK 20 — SFI List Coverage (post-fetch y post-pipeline)
==========================================================
Verifica que cada feature registrada en las listas SFI de settings.yaml
tiene datos reales en alguna de sus fuentes (onchain_raw, macro_raw,
features_train).

PROPÓSITO:
  Cierra el gap detectado el 2026-06-03 donde features onchain nuevas
  (Exchange_NetFlow, LTH_Supply_Change_30d) se añadieron al código de
  fetch_onchain.py y a settings.yaml sfi_onchain_features, pero el
  parquet histórico las tenía con 100% NaN por el bug del modo incremental.
  El data_integrity_check.py existente no detectaba esto.

ARQUITECTURA DE BÚSQUEDA:
  Prioridad de búsqueda por categoría:
  - sfi_onchain_features   → busca en: onchain_raw, features_train
  - sfi_macro_features     → busca en: macro_raw, features_train
  - sfi_calendar_features  → busca en: features_train (siempre calculadas)
  - sfi_macro_stable_features (boost) → busca en: features_train, macro_raw, onchain_raw

  Una feature se considera OK si N > MIN_REQUIRED en AL MENOS UNA fuente.
  FAIL si N == 0 en TODAS las fuentes.

INTEGRACIÓN:
  - Llamado desde data_integrity_check.py como CHECK 20
  - Ejecutable standalone: python scripts/check_sfi_coverage.py
  - Llamado desde sync_data_lake.py al final del pipeline
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

DATA_DIR = ROOT / "data"

# Mínimo de observaciones válidas para considerar una feature usable
MIN_REQUIRED = 200

# ─── Colores ─────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

OK_S   = f"{GREEN}OK  {RESET}"
WARN_S = f"{YELLOW}WARN{RESET}"
FAIL_S = f"{RED}FAIL{RESET}"

n_ok = n_warn = n_fail = 0


def _line(status: str, name: str, msg: str) -> None:
    global n_ok, n_warn, n_fail
    if status == "OK":    n_ok   += 1; s = OK_S
    elif status == "WARN": n_warn += 1; s = WARN_S
    else:                 n_fail += 1; s = FAIL_S
    print(f"  {s}  {name:<50} {msg}")


# Features que se generan SOLO en el feature_pipeline (no en parquets raw).
# Son OK aunque no estén en onchain_raw ni macro_raw.
PIPELINE_DERIVED_FEATURES = {
    "DXY_HMM_cond", "DXY_HMM_bull_neg", "DXY_HMM_interact",
    "btc_cycle_position", "halving_days_norm",
    "cal_days_to_next_halving",   # purgado como leakage — esperable no esté en features_train
    "hal_progress_sin", "hal_progress_cos",
    "btc_weekday_sin", "btc_month_sin",
    # Calendar features calculadas en pipeline
    "cal_halving_cycle_sin", "cal_halving_cycle_cos",
    "cal_days_since_halving", "cal_halving_cycle_pct",
    # Credit spreads: API no disponible actualmente — documentado en KI
    "CreditSpread_HY_IG", "CreditSpread_HY_IG_z90d", "CreditSpread_HY_z90d",
}

# Descripción de por qué están exentas (para el log)
PIPELINE_DERIVED_REASON = {
    "DXY_HMM_cond":          "HMM×DXY derivada en feature_pipeline",
    "DXY_HMM_bull_neg":      "HMM×DXY derivada en feature_pipeline",
    "DXY_HMM_interact":      "HMM×DXY derivada en feature_pipeline",
    "cal_days_to_next_halving": "purgado como leakage en pipeline (look-ahead)",
    "cal_halving_cycle_sin":  "calendar derivada en feature_pipeline",
    "cal_halving_cycle_cos":  "calendar derivada en feature_pipeline",
    "cal_days_since_halving": "calendar derivada en feature_pipeline",
    "cal_halving_cycle_pct":  "calendar derivada en feature_pipeline",
    "CreditSpread_HY_IG":     "API no disponible — pendiente integración",
    "CreditSpread_HY_IG_z90d":"API no disponible — pendiente integración",
    "CreditSpread_HY_z90d":   "API no disponible — pendiente integración",
}


def run_sfi_coverage_check(verbose: bool = False) -> tuple[int, int, int]:
    """
    Ejecuta el CHECK 20 SFI Coverage.
    Retorna (n_ok, n_warn, n_fail).
    """
    global n_ok, n_warn, n_fail
    n_ok = n_warn = n_fail = 0

    print(f"\n{BOLD}[CHECK 20] SFI List Coverage — features SFI vs datos reales{RESET}")
    print(f"  [SFI-COVERAGE-01] Verifica cobertura de features en listas SFI de settings.yaml")

    # 1. Leer listas SFI desde settings
    try:
        from config.settings import cfg as _cfg
        sfi_onchain  = list(getattr(_cfg.features, "sfi_onchain_features",  []))
        sfi_macro    = list(getattr(_cfg.features, "sfi_macro_features",    []))
        sfi_calendar = list(getattr(_cfg.features, "sfi_calendar_features", []))
        sfi_boost    = list(getattr(_cfg.features, "sfi_macro_stable_features", []))
    except Exception as e:
        _line("FAIL", "settings.yaml", f"No se pudo leer listas SFI: {e}")
        return n_ok, n_warn, n_fail

    # Todas las features (sin duplicados, con categoría)
    all_sfi: dict[str, str] = {}
    for f in sfi_onchain:  all_sfi.setdefault(f, "onchain")
    for f in sfi_macro:    all_sfi.setdefault(f, "macro")
    for f in sfi_calendar: all_sfi.setdefault(f, "calendar")
    for f in sfi_boost:    all_sfi.setdefault(f, "boost")

    total = len(all_sfi)
    if total == 0:
        _line("WARN", "sfi_lists", "Listas SFI vacías en settings.yaml")
        return n_ok, n_warn, n_fail

    print(f"  Total features en listas SFI: {total} "
          f"(onchain={len(sfi_onchain)} macro={len(sfi_macro)} "
          f"cal={len(sfi_calendar)} boost={len(sfi_boost)})")

    # 2. Cargar todas las fuentes disponibles
    source_paths = {
        "onchain_raw":    DATA_DIR / "raw/onchain/onchain_raw.parquet",
        "macro_raw":      DATA_DIR / "raw/macro/macro_raw.parquet",
        "features_train": DATA_DIR / "features/features_train.parquet",
    }
    dfs: dict[str, Optional[pd.DataFrame]] = {}
    for sname, spath in source_paths.items():
        if spath.exists():
            try:
                dfs[sname] = pd.read_parquet(spath)
            except Exception as e:
                print(f"  [WARN] {sname}: no legible ({e})")
                dfs[sname] = None
        else:
            dfs[sname] = None

    available_sources = [k for k, v in dfs.items() if v is not None]
    if not available_sources:
        print(f"  [WARN] No hay parquets disponibles — skip CHECK 20")
        return n_ok, n_warn, n_fail

    print(f"  Fuentes disponibles: {available_sources}")

    # 3. Verificar cada feature
    zero_data_features:    list[tuple[str, str]] = []   # (feat, category)
    hundred_nan_features:  list[tuple[str, str, str]] = []  # (feat, category, source)

    for feat, category in all_sfi.items():
        # Features pipeline-derivadas o con excepción conocida
        if feat in PIPELINE_DERIVED_FEATURES:
            reason = PIPELINE_DERIVED_REASON.get(feat, "derivada/excepción conocida")
            # Para estas, verificar en features_train si existe (no obligatorio)
            ft = dfs.get("features_train")
            if ft is not None and feat in ft.columns:
                n = ft[feat].notna().sum()
                if n > MIN_REQUIRED:
                    if verbose:
                        _line("OK", feat, f"derivada/exenta [{reason}] N={n:,} en features_train")
                elif n > 0:
                    if verbose:
                        _line("WARN", feat, f"derivada N={n} bajo en features_train")
                else:
                    if verbose:
                        _line("OK", feat, f"derivada/exenta [{reason}] (0 en features_train — esperado si purgada)")
            else:
                if verbose:
                    _line("OK", feat, f"derivada/exenta [{reason}]")
            n_ok += 1  # exenta — contar como OK
            continue

        # Buscar en TODAS las fuentes
        max_n = 0
        found_source = None
        all_nan_in_all = True  # True si en TODAS las fuentes donde aparece = 0 válidos

        for sname, df in dfs.items():
            if df is None or feat not in df.columns:
                continue
            n = df[feat].notna().sum()
            if n > max_n:
                max_n = n
                found_source = sname
            if n > 0:
                all_nan_in_all = False

        # Clasificar
        if max_n >= MIN_REQUIRED:
            nan_pct = 0.0
            df_src = dfs.get(found_source)
            if df_src is not None and feat in df_src.columns:
                nan_pct = df_src[feat].isna().mean() * 100
            _line("OK", feat,
                  f"[{category}] N={max_n:,} en '{found_source}' | NaN={nan_pct:.0f}%")

        elif max_n > 0 and max_n < MIN_REQUIRED:
            _line("WARN", feat,
                  f"[{category}] N={max_n} en '{found_source}' (mín={MIN_REQUIRED})")

        elif found_source and all_nan_in_all:
            # Columna presente pero 100% NaN en todas las fuentes donde aparece
            hundred_nan_features.append((feat, category, found_source))
            _line("FAIL", feat,
                  f"[{category}] COLUMNA PRESENTE pero 100% NaN en '{found_source}' — "
                  f"bug incremental o API muerta")
            print(f"         → [ONCHAIN-NEWCOL-FIX] Re-ejecutar fetcher con backfill completo")

        else:
            # No aparece en ninguna fuente
            zero_data_features.append((feat, category))
            _line("FAIL", feat,
                  f"[{category}] AUSENTE de todas las fuentes "
                  f"(onchain_raw, macro_raw, features_train)")
            print(f"         → Verificar implementación en fetch_onchain.py / fetch_macro.py")

    # 4. Resumen de problemas
    all_failures = zero_data_features + [(f, c) for f, c, _ in hundred_nan_features]
    print()
    if all_failures:
        print(f"  {RED}[SFI-COVERAGE-01] {len(all_failures)} features SFI sin datos — "
              f"el SFI NO puede evaluarlas:{RESET}")
        for item in all_failures[:10]:
            feat, cat = item[0], item[1]
            in_boost = " [BOOST]" if feat in sfi_boost else ""
            print(f"    → [{cat}] {feat}{in_boost}")
        if len(all_failures) > 10:
            print(f"    → ... y {len(all_failures) - 10} más")
    else:
        print(f"  {GREEN}[SFI-COVERAGE-01] Todas las features SFI tienen datos suficientes.{RESET}")

    return n_ok, n_warn, n_fail


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SFI Coverage Check — CHECK 20")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Mostrar también features OK y exentas")
    args = parser.parse_args()

    ok, warn, fail = run_sfi_coverage_check(verbose=args.verbose)
    print()
    print("=" * 60)
    print(f"SFI COVERAGE FINAL: OK={ok} WARN={warn} FAIL={fail}")
    if fail > 0:
        print(f"{RED}FALLO: {fail} features SFI sin datos válidos{RESET}")
        sys.exit(1)
    elif warn > 0:
        print(f"{YELLOW}AVISO: {warn} features con calidad reducida{RESET}")
        sys.exit(0)
    else:
        print(f"{GREEN}OK: cobertura SFI completa{RESET}")
        sys.exit(0)
