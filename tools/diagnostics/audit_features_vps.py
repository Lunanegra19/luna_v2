"""audit_features_vps.py — Audita todos los parquets de features en el VPS"""
import pandas as pd
from pathlib import Path

feat_dir = Path("/root/luna_v2/data/features")
print("=" * 90)
print("INVENTARIO FEATURES PARQUETS")
print("=" * 90)

for f in sorted(feat_dir.glob("*.parquet")):
    try:
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index, utc=True)
        hmm = "HMM_Regime" in df.columns or "HMM_Semantic" in df.columns
        has_2026 = (df.index >= "2026-01-01").sum()
        has_2025 = (df.index >= "2025-01-01").sum()
        print(f"{f.name:<45s} {str(df.index.min().date()):<12s} -> {str(df.index.max().date()):<12s} "
              f"rows={len(df):5d} hmm={hmm} 2026_bars={has_2026} 2025+={has_2025}")
    except Exception as e:
        print(f"{f.name}: ERROR {e}")
