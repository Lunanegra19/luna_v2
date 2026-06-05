"""
Investiga por qué FundingRate_EMA3 y FundingRate_Pct90d fallan en live incremental.
El parquet de entrenamiento tiene 'funding_ema_3' pero el parquet live recién actualizado no.
"""
import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, "/root/luna_v2")

print("=" * 70)
print("INVESTIGACION SKEW-01 — funding_ema_3 en parquet live post-update")
print("=" * 70)

live = pd.read_parquet("/root/luna_v2/data/features/features_live.parquet")
print(f"Parquet: {len(live)} filas | {len(live.columns)} cols")
print(f"Index: {live.index[-1]}")

# Buscar cols de funding
fr_cols = [c for c in live.columns if "funding" in c.lower() or "FundingRate" in c]
print(f"\nColumnas funding encontradas: {sorted(fr_cols)}")
for c in sorted(fr_cols):
    v = live[c].iloc[-1]
    nan_pct = live[c].isna().mean()
    print(f"  {c}: last={v} | NaN={nan_pct:.1%}")

# ¿Dónde se genera funding_ema_3 en el pipeline?
print("\n" + "=" * 70)
print("¿Dónde se genera funding_ema_3 en el pipeline live?")
print("=" * 70)
import subprocess
PYTHON = "/root/miniconda3/envs/luna_env/bin/python"
result = subprocess.run(
    ["grep", "-rn", "funding_ema_3", "/root/luna_v2/luna/", "--include=*.py"],
    capture_output=True, text=True
)
print(result.stdout[:3000])

print("\n" + "=" * 70)
print("¿Cómo se genera FundingRate_EMA3 en el bridge histórico?")
print("=" * 70)
result2 = subprocess.run(
    ["grep", "-n", "-A", "10", "funding_ema_3", "/root/luna_v2/luna/data/historical_data_bridge.py"],
    capture_output=True, text=True
)
print(result2.stdout[:3000])

print("\n" + "=" * 70)
print("¿Qué columnas tiene el DerivativesFetcher en live?")
print("=" * 70)
# Verificar el fetcher de derivatives que genera funding_ema_3
result3 = subprocess.run(
    ["grep", "-n", "ema_3\|ema3\|funding.*ema\|EMA.*fund", "/root/luna_v2/luna/data/fetchers/fetch_derivatives.py"],
    capture_output=True, text=True
)
print(result3.stdout[:3000] if result3.stdout else "  No encontrado en fetch_derivatives.py")

# Buscar en todos los fetchers
result4 = subprocess.run(
    ["grep", "-rn", "funding_ema_3\|ema_3.*fund\|funding.*ema_3", "/root/luna_v2/luna/data/"],
    capture_output=True, text=True
)
print("\nEn todos los fetchers:")
print(result4.stdout[:2000] if result4.stdout else "  No encontrado")

print("\n" + "=" * 70)
print("CONCLUSION — ¿qué hay disponible para derivar FundingRate_EMA3?")
print("=" * 70)
# Ver si FundingRate base está disponible para hacer el EMA3 como fallback
fr = live["FundingRate"] if "FundingRate" in live.columns else None
fr_base = live.get("dv_funding_rate")
print(f"  FundingRate raw: {'✅' if fr is not None else '❌'}")
print(f"  dv_funding_rate: {'✅' if fr_base is not None else '❌'}")
if fr is not None:
    ema3 = fr.ewm(span=3*24, min_periods=1).mean()
    print(f"  FundingRate.ewm(span=72h) → último: {ema3.iloc[-1]}")
    print(f"  → El fallback EMA derivado FUNCIONA pero el FIX-SKEW-01 no lo ejecuta correctamente")
