"""
audit_raw_vs_gdrive.py
Compara todos los parquets raw del local vs G: para detectar datos faltantes.
"""
import pandas as pd
from pathlib import Path
import sys

RG = Path("G:/Mi unidad/ia/luna_v2/data/raw")
RL = Path("C:/Users/Usuario/Desktop/ia/luna_v2/data/raw")

ARCHIVOS = [
    "derivatives/derivatives_raw.parquet",
    "onchain/onchain_raw.parquet",
    "macro/macro_raw.parquet",
    "stablecoin_m2/stablecoin_m2_raw.parquet",
    "altcoins/altcoins_raw.parquet",
    "crossasset/crossasset_raw.parquet",
    "defi/defi_raw.parquet",
    "mempool/mempool_raw.parquet",
    "etf/etf_raw.parquet",
    "ohlcv/ohlcv_raw.parquet",
    "orderflow/bybit_ofi_1h.parquet",
    "macro/macro_raw_bak.parquet",
]

print(f"{'Archivo':<35} {'G: rows':>9} {'L: rows':>9} {'G:NaN%':>7} {'L:NaN%':>7}  Estado")
print("-" * 85)

problemas = []
for p in ARCHIVOS:
    pg = RG / p
    pl = RL / p
    name = p.split("/")[-1]
    if not pg.exists():
        print(f"{name:<35}  [SKIP] No existe en G:")
        continue
    if not pl.exists():
        print(f"{name:<35}  [MISS] No existe LOCAL")
        problemas.append(f"AUSENTE: {name}")
        continue
    dg = pd.read_parquet(pg)
    dl = pd.read_parquet(pl)
    ng = round(dg.isnull().mean().mean() * 100, 1)
    nl = round(dl.isnull().mean().mean() * 100, 1)
    rg = len(dg)
    rl = len(dl)

    if rl >= rg and nl <= ng + 2:
        st = "OK"
    elif rl < rg - 50:
        st = "LOCAL CORTO!"
        problemas.append(f"CORTO: {name} G={rg} L={rl}")
    elif nl > ng + 5:
        st = "LOCAL +NaN!"
        problemas.append(f"MAS NaN: {name} G={ng}% L={nl}%")
    else:
        st = "OK ~"

    print(f"{name:<35} {rg:>9,} {rl:>9,} {ng:>6}% {nl:>6}%  {st}")

print()
if problemas:
    print(f"PROBLEMAS DETECTADOS ({len(problemas)}):")
    for p in problemas:
        print(f"  - {p}")
else:
    print("Todo OK: local >= G: en todos los archivos.")
