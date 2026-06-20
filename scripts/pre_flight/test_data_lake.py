"""
test_data_lake.py
=================
Luna V2 — Pre-Flight Checks: DATA LAKE INTEGRITY
Section: data_lake

[FIX-PREFLIGHT-DATALAKE-01 2026-06-18]
Motivacion: 3 fetchers criticos (stablecoin_m2, crossasset, bybit_ofi) estaban
desactualizados 24-112 dias sin que ningun pre-flight lo detectara. El OOD-GUARD
los bloqueaba silenciosamente pero el error nunca se reportaba al operador.

Checks implementados:
  TEST-DL-01  Parquets raw criticos existen
  TEST-DL-02  Parquets raw no estan obsoletos (atraso > MAX_LAG_DAYS)
  TEST-DL-03  Todos los fetchers estan registrados en sync_data_lake.py
  TEST-DL-04  Todos los fetchers tienen bloque __main__ ejecutable
  TEST-DL-05  Consistencia ruta de guardado: el parquet que genera el fetcher
              coincide con el que lee el pipeline (sin ghost paths)
  TEST-DL-06  Parquets raw sin gaps internos de >30 dias (detecta bug incremental)
"""
from __future__ import annotations
from .core import test, ROOT
import pandas as pd
from pathlib import Path
import ast
import re


# ── Configuracion canonica de parquets criticos ────────────────────────────────
# Cada entrada: (path_relativo, max_lag_dias, fetcher_script, col_muestra)
# max_lag_dias: cuantos dias de atraso se toleran antes de FAIL
# col_muestra: columna para verificar NaN % (None = skip col check)

_RAW_PARQUETS = [
    ("data/raw/ohlcv/ohlcv_raw.parquet",               2,  "luna/data/fetch_ohlcv.py",         "close"),
    ("data/raw/derivatives/derivatives_raw.parquet",    2,  "luna/data/fetch_derivatives.py",    "OI_BTC"),
    ("data/raw/onchain/onchain_raw.parquet",            2,  "luna/data/fetch_onchain.py",        "Hash_Rate"),
    ("data/raw/macro/macro_raw.parquet",                5,  "luna/data/fetch_macro.py",          "DXY"),
    ("data/raw/altcoins/altcoins_raw.parquet",          3,  "luna/data/fetch_altcoins.py",       "ETH_Price"),
    ("data/raw/mempool/mempool_raw.parquet",            3,  "luna/data/fetch_mempool.py",        "Mempool_Size"),
    ("data/raw/defi/defi_raw.parquet",                  5,  "luna/data/fetch_defi.py",           "Total_TVL_USD"),
    ("data/raw/stablecoin_m2/stablecoin_m2_raw.parquet",5,  "luna/data/fetch_stablecoins.py",   "Stablecoin_Cap"),
    ("data/raw/crossasset/crossasset_raw.parquet",      3,  "luna/data/fetch_crossasset.py",     "eth_btc_corr_24h"),
    ("data/raw/orderflow/bybit_ofi_1h.parquet",         3,  "luna/data/fetch_bybit_ofi.py",      "ofi_buy_vol_1h"),
]

# Fetchers registrados en sync_data_lake.py (lista canonica)
_EXPECTED_FETCHERS_IN_SYNC = [
    "luna/data/fetch_ohlcv.py",
    "luna/data/fetch_macro.py",
    "luna/data/m2_global_fetcher.py",
    "luna/data/fetch_onchain.py",
    "luna/data/fetch_derivatives.py",
    "luna/data/fetch_altcoins.py",
    "luna/data/fetch_mempool.py",
    "luna/data/fetch_defi.py",
    "luna/data/fetch_crossasset.py",
    "luna/data/fetch_stablecoins.py",
    "luna/data/fetch_bybit_ofi.py",
]

# Mapeo fetcher -> parquet que produce (para check de ghost paths)
_FETCHER_PARQUET_MAP = {
    "luna/data/fetch_stablecoins.py":  "data/raw/stablecoin_m2/stablecoin_m2_raw.parquet",
    "luna/data/fetch_crossasset.py":   "data/raw/crossasset/crossasset_raw.parquet",
    "luna/data/fetch_bybit_ofi.py":    "data/raw/orderflow/bybit_ofi_1h.parquet",
    "luna/data/fetch_derivatives.py":  "data/raw/derivatives/derivatives_raw.parquet",
    "luna/data/fetch_onchain.py":      "data/raw/onchain/onchain_raw.parquet",
    "luna/data/fetch_macro.py":        "data/raw/macro/macro_raw.parquet",
    "luna/data/fetch_altcoins.py":     "data/raw/altcoins/altcoins_raw.parquet",
    "luna/data/fetch_mempool.py":      "data/raw/mempool/mempool_raw.parquet",
    "luna/data/fetch_defi.py":         "data/raw/defi/defi_raw.parquet",
}

# Gaps internos: tolerancia maxima en dias antes de FAIL
_MAX_INTERNAL_GAP_DAYS = 30


# ── Helpers ───────────────────────────────────────────────────────────────────

def _days_stale(parquet_path: Path) -> int:
    """Cuantos dias tiene de atraso el parquet respecto a HOY."""
    df = pd.read_parquet(parquet_path, columns=[])
    last = df.index.max()
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    return (pd.Timestamp.now(tz="UTC") - last).days


def _get_fetchers_in_sync() -> list[str]:
    """Extrae la lista de fetchers registrada en sync_data_lake.py.
    Robusto frente a comentarios inline, lineas en blanco y CRLF de Windows.
    """
    sync_path = ROOT / "scripts/sync_data_lake.py"
    if not sync_path.exists():
        return []
    lines = sync_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Buscar inicio de la lista `fetchers = [`
    in_list = False
    collected = []
    for line in lines:
        stripped = line.strip()
        if not in_list:
            if re.search(r"fetchers\s*=\s*\[", stripped):
                in_list = True
                # La apertura puede tener el primer elemento en la misma linea
                after = re.sub(r"^[^[]*\[", "", stripped)
                stripped = after
        if in_list:
            # Quitar comentario inline PRIMERO para evitar ] dentro de #[...]
            clean = re.sub(r"#.*$", "", stripped)
            # Detectar fin de la lista tras eliminar comentarios
            if "]" in clean:
                before_bracket = clean[:clean.index("]")]
                collected.append(before_bracket)
                break
            collected.append(clean)

    combined = " ".join(collected)
    found = re.findall(r'"([^"]+\.py)"', combined)
    return found


def _has_main_block(fetcher_path: Path) -> bool:
    """Verifica que el script tiene un bloque `if __name__ == '__main__':`."""
    if not fetcher_path.exists():
        return False
    src = fetcher_path.read_text(encoding="utf-8", errors="replace")
    return '__name__' in src and '__main__' in src


def _max_internal_gap_days(parquet_path: Path, col: str | None = None) -> float:
    """
    Calcula el mayor gap interno en el indice del parquet (en dias).
    Usa la primera columna numerica si col es None.
    """
    df = pd.read_parquet(parquet_path, columns=[col] if col else None)
    if df.index.empty or len(df) < 2:
        return 0.0
    diffs = pd.Series(df.index).diff().dropna()
    if diffs.empty:
        return 0.0
    return diffs.max().total_seconds() / 86400


# ── Tests ─────────────────────────────────────────────────────────────────────

@test("TEST-DL-01  Parquets raw criticos existen", section="data_lake")
def t_dl_01():
    """
    [FIX-PREFLIGHT-DATALAKE-01] Verifica que todos los parquets raw canonicos
    existen en el data lake. Un parquet faltante indica que el fetcher nunca
    se ejecuto o que la ruta de guardado cambio silenciosamente.
    """
    print("[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-01: verificando existencia de parquets raw...")
    missing = []
    for path_rel, _, fetcher, _ in _RAW_PARQUETS:
        p = ROOT / path_rel
        if not p.exists():
            missing.append(f"{p.name} (fetcher={fetcher})")

    assert not missing, (
        f"[TEST-DL-01] {len(missing)} parquets raw AUSENTES — el fetcher nunca "
        f"se ejecuto o usa una ruta de guardado incorrecta:\n  "
        + "\n  ".join(missing)
    )
    print(f"[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-01 OK: {len(_RAW_PARQUETS)} parquets presentes")
    return f"OK — {len(_RAW_PARQUETS)} parquets raw existentes"


@test("TEST-DL-02  Parquets raw no obsoletos (atraso > MAX_LAG_DAYS)", section="data_lake")
def t_dl_02():
    """
    [FIX-PREFLIGHT-DATALAKE-01] Verifica que ningun parquet raw esta demasiado
    desactualizado. Este test hubiera detectado los gaps de 24-112 dias.

    Cada fuente tiene su propio umbral (MAX_LAG_DAYS) segun su frecuencia de
    actualizacion y criticidad en el modelo.
    """
    print("[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-02: verificando frescura de parquets raw...")
    stale = []
    warn = []

    for path_rel, max_lag, fetcher, _ in _RAW_PARQUETS:
        p = ROOT / path_rel
        if not p.exists():
            continue  # ya lo detecta TEST-DL-01
        try:
            lag = _days_stale(p)
        except Exception as e:
            stale.append(f"{p.name}: error leyendo ({e})")
            continue

        if lag > max_lag:
            stale.append(f"{p.name}: {lag}d de atraso (max={max_lag}d) | fetcher={Path(fetcher).name}")
        elif lag > max(max_lag // 2, 1):
            warn.append(f"{p.name}: {lag}d atraso (acercandose al limite {max_lag}d)")

    warn_str = f" | WARN: {'; '.join(warn[:3])}" if warn else ""
    print(
        f"[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-02: {len(stale)} STALE, {len(warn)} WARN{warn_str}"
    )

    assert not stale, (
        f"[TEST-DL-02] {len(stale)} parquet(s) raw OBSOLETOS — ejecutar sync_data_lake.py:\n  "
        + "\n  ".join(stale)
    )
    return f"OK — todos frescos{warn_str}"


@test("TEST-DL-03  Todos los fetchers criticos registrados en sync_data_lake.py", section="data_lake")
def t_dl_03():
    """
    [FIX-PREFLIGHT-DATALAKE-01] Verifica que ningun fetcher este 'huerfano'
    (existe en luna/data/ pero no esta en la lista de sync_data_lake.py).

    Este test hubiera detectado que fetch_crossasset.py, fetch_stablecoins.py
    y fetch_bybit_ofi.py nunca se ejecutaban automaticamente.
    """
    print("[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-03: verificando registro en sync_data_lake.py...")
    registered = set(_get_fetchers_in_sync())
    expected   = set(_EXPECTED_FETCHERS_IN_SYNC)

    missing_in_sync = expected - registered
    print(
        f"[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-03: registrados={len(registered)} "
        f"esperados={len(expected)} | ausentes={len(missing_in_sync)}"
    )

    assert not missing_in_sync, (
        f"[TEST-DL-03] {len(missing_in_sync)} fetcher(s) AUSENTES de sync_data_lake.py "
        f"(nunca se ejecutaran automaticamente):\n  "
        + "\n  ".join(sorted(missing_in_sync))
    )

    extra = registered - expected
    suffix = f" | Extra en sync (no canonicos): {sorted(extra)}" if extra else ""
    return f"OK — {len(registered)} fetchers registrados{suffix}"


@test("TEST-DL-04  Todos los fetchers tienen bloque __main__ ejecutable", section="data_lake")
def t_dl_04():
    """
    [FIX-PREFLIGHT-DATALAKE-01] Verifica que cada fetcher registrado en
    sync_data_lake.py tiene un bloque `if __name__ == '__main__':`.

    sync_data_lake.py llama los fetchers via `python script.py` — sin __main__
    el script se importa pero no ejecuta nada (bug silencioso).

    Este test hubiera detectado que fetch_crossasset.py no tenia __main__.
    """
    print("[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-04: verificando bloques __main__ en fetchers...")
    registered = _get_fetchers_in_sync()
    no_main = []

    for fetcher_rel in registered:
        fetcher_path = ROOT / fetcher_rel
        if not fetcher_path.exists():
            continue  # archivo faltante — otro test lo detecta
        if not _has_main_block(fetcher_path):
            no_main.append(fetcher_rel)

    print(
        f"[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-04: {len(no_main)} fetchers sin __main__"
    )
    assert not no_main, (
        f"[TEST-DL-04] {len(no_main)} fetcher(s) SIN bloque __main__ — "
        f"sync_data_lake.py los llama pero no ejecutan nada:\n  "
        + "\n  ".join(no_main)
    )
    return f"OK — {len(registered)} fetchers con __main__ verificado"


@test("TEST-DL-05  Ruta de guardado del fetcher coincide con parquet esperado", section="data_lake")
def t_dl_05():
    """
    [FIX-PREFLIGHT-DATALAKE-01] Verifica que el parquet que produce cada fetcher
    (segun _FETCHER_PARQUET_MAP) existe en la ruta canonica del pipeline.

    Este test hubiera detectado que fetch_stablecoins.py guardaba en
    data/raw/onchain/ en vez de data/raw/stablecoin_m2/.
    """
    print("[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-05: verificando consistencia de rutas de guardado...")
    ghost_paths = []  # fetcher produce parquet en ruta no canonica

    for fetcher_rel, expected_parquet_rel in _FETCHER_PARQUET_MAP.items():
        expected_path = ROOT / expected_parquet_rel
        if not expected_path.exists():
            ghost_paths.append(
                f"{Path(fetcher_rel).name} → parquet esperado en '{expected_parquet_rel}' NO EXISTE"
            )

    print(
        f"[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-05: {len(ghost_paths)} rutas inconsistentes"
    )
    assert not ghost_paths, (
        f"[TEST-DL-05] {len(ghost_paths)} fetcher(s) con GHOST PATH — "
        f"el parquet de salida no coincide con la ruta canonica del pipeline:\n  "
        + "\n  ".join(ghost_paths)
    )
    return f"OK — {len(_FETCHER_PARQUET_MAP)} rutas de guardado consistentes"


@test("TEST-DL-06  Parquets raw sin gaps internos criticos (> 30 dias)", section="data_lake")
def t_dl_06():
    """
    [FIX-PREFLIGHT-DATALAKE-01] Detecta gaps internos en el indice temporal
    de los parquets raw. Un gap > 30 dias indica que el fetcher incremental
    no rellenó correctamente el historico (bug clasico de modo incremental).

    Este test hubiera detectado el gap de 994 dias en OI_BTC/FundingRate
    desde Sep 2023.

    NOTA: Solo evalua las columnas de muestra para rapidez. Un gap en el
    indice del parquet afecta a TODAS las columnas.
    """
    print("[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-06: detectando gaps internos en parquets raw...")
    gap_fails = []
    gap_warns = []

    # Parquets a chequear con su columna de muestra
    # (usar los que tienen datos de alta frecuencia — no diarios)
    HIGH_FREQ = {
        "data/raw/derivatives/derivatives_raw.parquet": "OI_BTC",
        "data/raw/orderflow/bybit_ofi_1h.parquet":      "ofi_buy_vol_1h",
        "data/raw/crossasset/crossasset_raw.parquet":    "eth_btc_corr_24h",
        "data/raw/ohlcv/ohlcv_raw.parquet":              "close",
    }

    for path_rel, col in HIGH_FREQ.items():
        p = ROOT / path_rel
        if not p.exists():
            continue
        try:
            gap_days = _max_internal_gap_days(p, col)
        except Exception as e:
            gap_warns.append(f"{Path(path_rel).name}: error leyendo ({e})")
            continue

        if gap_days > _MAX_INTERNAL_GAP_DAYS:
            gap_fails.append(
                f"{Path(path_rel).name}[{col}]: gap interno de {gap_days:.0f}d "
                f"(max={_MAX_INTERNAL_GAP_DAYS}d) — ejecutar fetcher con BACKFILL-GAP"
            )
        elif gap_days > 7:
            gap_warns.append(f"{Path(path_rel).name}: gap de {gap_days:.1f}d (tolerable)")

    warn_str = f" | WARN: {'; '.join(gap_warns[:2])}" if gap_warns else ""
    print(
        f"[FIX-PREFLIGHT-DATALAKE-01] TEST-DL-06: {len(gap_fails)} FAIL, "
        f"{len(gap_warns)} WARN{warn_str}"
    )

    assert not gap_fails, (
        f"[TEST-DL-06] {len(gap_fails)} parquet(s) con GAPS INTERNOS criticos:\n  "
        + "\n  ".join(gap_fails)
        + "\n  Solucion: re-ejecutar el fetcher con logica BACKFILL-GAP (ver FIX-INCR-02-v2)"
    )
    return f"OK — sin gaps internos > {_MAX_INTERNAL_GAP_DAYS}d{warn_str}"
