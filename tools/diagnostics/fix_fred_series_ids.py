"""
[FIX-FRED-SERIES] Encuentra los IDs correctos para:
1. CHNASSETS — China Central Bank Assets (no existe en FRED con ese ID)
2. MYAGM2CNM189N — M2 China (devuelve 0 filas desde 2020)
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
from config.settings import cfg
from fredapi import Fred

fred = Fred(api_key=cfg.api_keys.fred_api_key)

# Candidatos para China Assets (balance hoja del banco central chino)
china_asset_candidates = [
    "CHNASSETS",        # Original — no existe
    "CBASSETS",         # Alternativo
    "BOPBCA",           # Balance of payments China  
    "DDDM01CNA066NWDB", # China money supply M1
    "MABMM301CNM189N",  # China M3 (si existe)
    "MYAGM2CNM189N",    # M2 China (actual — vacía)
    "MYAGM2USM052N",    # M2 USA alternativo  
]

# Candidatos para M2 China post-2020
m2_china_candidates = [
    "MYAGM2CNM189N",    # El que usamos — vacío
    "MABMM301CNM189N",  # M3 China
    "MANMM101CNM189S",  # Broad money China
    "MABMM301CNQ189N",  # China broad money quarterly
]

print("=== TEST China Assets (reemplazar CHNASSETS) ===")
for sid in china_asset_candidates:
    try:
        data = fred.get_series(sid, observation_start="2018-01-01")
        if data is not None and len(data) > 0:
            print(f"  ✅ {sid}: {len(data)} filas | último={data.index[-1].date()} | val={data.iloc[-1]:.4f}")
        else:
            print(f"  ⚠️  {sid}: vacía (0 filas)")
    except Exception as e:
        print(f"  ❌ {sid}: {str(e)[:70]}")

print("\n=== TEST M2 China (reemplazar MYAGM2CNM189N vacía) ===")
for sid in m2_china_candidates:
    try:
        data = fred.get_series(sid, observation_start="2018-01-01")
        if data is not None and len(data) > 0:
            print(f"  ✅ {sid}: {len(data)} filas | último={data.index[-1].date()} | val={data.iloc[-1]:.4f}")
        else:
            print(f"  ⚠️  {sid}: vacía (0 filas)")
    except Exception as e:
        print(f"  ❌ {sid}: {str(e)[:70]}")
