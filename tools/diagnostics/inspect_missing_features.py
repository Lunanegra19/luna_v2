"""Inspecciona las columnas disponibles en el parquet live relevantes para los 3 bugs."""
import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, "/root/luna_v2")

df = pd.read_parquet("/root/luna_v2/data/features/features_live.parquet")
print(f"Parquet: {len(df)} filas | {len(df.columns)} cols")
print(f"Index range: {df.index[0]} → {df.index[-1]}\n")

# --- BUG 1: FundingRate disponible? ---
print("=== BUG-1: FundingRate cols disponibles ===")
fr_cols = [c for c in df.columns if "funding" in c.lower() or "FundingRate" in c]
for c in sorted(fr_cols):
    nan_pct = df[c].isna().mean()
    last = df[c].dropna().iloc[-1] if not df[c].dropna().empty else None
    print(f"  {c}: NaN={nan_pct:.1%}, last={last}")

# --- BUG 2: OI cols disponibles ---
print("\n=== BUG-2: OI cols disponibles ===")
oi_cols = [c for c in df.columns if c.startswith("OI") or "open_interest" in c.lower() or "_oi_" in c.lower()]
for c in sorted(oi_cols):
    nan_pct = df[c].isna().mean()
    last = df[c].dropna().iloc[-1] if not df[c].dropna().empty else None
    print(f"  {c}: NaN={nan_pct:.1%}, last={last}")

# --- BUG 3: ETF cols disponibles ---
print("\n=== BUG-3: ETF cols disponibles ===")
etf_cols = [c for c in df.columns if "etf" in c.lower() or "ETF" in c or "bito" in c.lower() or "BITO" in c or "flow" in c.lower()]
for c in sorted(etf_cols):
    nan_pct = df[c].isna().mean()
    last = df[c].dropna().iloc[-1] if not df[c].dropna().empty else None
    print(f"  {c}: NaN={nan_pct:.1%}, last={last}")

# --- qué tiene el live data path (pre-feature) ---
print("\n=== Archivos parquet en data/features/ ===")
for p in sorted(Path("/root/luna_v2/data/features").glob("*.parquet")):
    sz = p.stat().st_size // 1024
    print(f"  {p.name}: {sz} KB")

print("\n=== Archivos parquet en data/historical/ (buscando OI raw) ===")
for p in sorted(Path("/root/luna_v2/data/historical").rglob("*coinglass*")):
    print(f"  {p}")
