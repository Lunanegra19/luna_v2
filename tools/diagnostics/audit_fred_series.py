"""
[FRED-AUDIT] Diagnóstico profundo de las series FRED que fallan.
Prueba cada serie individualmente con la clave real para identificar:
1. Series descontinuadas o con ID incorrecto (error 400 Bad Request)
2. Series con timeout (error: None = thread finaliza sin respuesta)
3. Series realmente disponibles con la clave actual
"""
import sys
import time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from config.settings import cfg

print("=" * 65)
print("  FRED API AUDIT — Diagnóstico de series")
print("=" * 65)

api_key = cfg.api_keys.fred_api_key
print(f"\n  API Key: '{api_key[:12]}...' (len={len(api_key)})")
print(f"  Válida:  {'✅ sí (32 chars)' if len(api_key) == 32 else '❌ NO'}")

try:
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    print("  fredapi: ✅ importado correctamente\n")
except ImportError:
    print("  fredapi: ❌ no instalado")
    sys.exit(1)

# Todas las series que usa fetch_macro.py
FRED_SERIES = [
    # M2 Global
    ("M2SL",             "M2 USA",              "mensual"),
    ("MYAGM2CNM189N",    "M2 China",            "mensual"),
    ("MABMM301EZM189N",  "M2 EU",               "mensual"),
    ("MABMM301JPM189N",  "M2 Japan",            "mensual"),
    # Monetaria / Liquidez
    ("FEDFUNDS",         "FedFunds",            "mensual"),
    ("WALCL",            "Fed Balance Sheet",   "semanal"),
    ("WTREGEN",          "TGA",                 "semanal"),
    ("RRPONTSYD",        "RRP",                 "diario"),
    # Macro USA
    ("CPIAUCSL",         "CPI YoY (raw)",       "mensual"),
    ("UNRATE",           "Unemployment",        "mensual"),
    ("WEI",              "WEI",                 "semanal"),
    # Yield Curves
    ("GS10",             "10Y Treasury",        "diario"),
    ("GS2",              "2Y Treasury",         "diario"),
    ("T5YIE",            "Breakeven 5Y",        "diario"),
    # Nuevas 2026-03-05
    ("PAYEMS",           "NFP",                 "mensual"),
    ("PCEPI",            "PCE Inflation",       "mensual"),
    ("DFII10",           "Real Yield 10Y",      "diario"),
    # Balance sheets bancos centrales
    ("ECBASSETS",        "ECB Assets",          "semanal"),
    ("JPNASSETS",        "Japan Assets",        "mensual"),
    ("CHNASSETS",        "China Assets",        "???"),  # <-- SOSPECHOSO
]

results = []
print(f"  {'Series ID':<22} {'Nombre':<22} {'Estado':<15} {'Últimas filas':<15} {'Último dato'}")
print("  " + "-" * 90)

for series_id, name, freq in FRED_SERIES:
    t0 = time.time()
    try:
        data = fred.get_series(series_id, observation_start="2020-01-01")
        elapsed = time.time() - t0
        if data is None or len(data) == 0:
            status = "VACÍA"
            last_val = "N/A"
            last_date = "N/A"
        else:
            status = "✅ OK"
            last_val = f"{data.iloc[-1]:.4f}"
            last_date = str(data.index[-1].date())
        print(f"  {series_id:<22} {name:<22} {status:<15} n={len(data) if data is not None else 0:<10} {last_date}  ({elapsed:.1f}s)")
        results.append((series_id, name, "OK", len(data) if data is not None else 0))
    except Exception as e:
        elapsed = time.time() - t0
        err_str = str(e)[:60]
        print(f"  {series_id:<22} {name:<22} ❌ ERROR        {err_str}  ({elapsed:.1f}s)")
        results.append((series_id, name, f"ERROR: {err_str}", 0))

print("\n" + "=" * 65)
ok = sum(1 for _, _, s, _ in results if s == "OK")
err = sum(1 for _, _, s, _ in results if s.startswith("ERROR"))
print(f"  RESUMEN: {ok}/{len(results)} series OK | {err} con errores")

if err > 0:
    print("\n  🔴 Series con error:")
    for sid, name, status, _ in results:
        if status.startswith("ERROR"):
            print(f"    {sid} ({name}): {status[7:]}")
print("=" * 65)
