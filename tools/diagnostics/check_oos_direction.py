"""
[SHORT-VIABILITY-TEST-01] Diagnóstico: verificar que los archivos OOS de la run Short
están correctamente etiquetados como short y separados de la run Long previa.
"""
import pandas as pd
import glob
import os

print("[CHECK-OOS-DIRECTION] Buscando archivos OOS de trades...")

# Buscar todos los parquets OOS en data/
patterns = [
    "data/oos_trades_W*.parquet",
    "data/**/oos_trades_W*.parquet",
    "data/seed*/**/oos_trades*.parquet",
]

files = []
for p in patterns:
    files += glob.glob(p, recursive=True)
files = sorted(set(files))

print(f"[CHECK-OOS-DIRECTION] Total parquets OOS encontrados: {len(files)}")

for f in files[:10]:
    try:
        df = pd.read_parquet(f)
        print(f"\n=== {os.path.basename(f)} ({len(df)} trades) ===")
        print(f"  Ruta: {f}")
        print(f"  Columnas: {list(df.columns[:12])}")

        # Buscar columnas de dirección
        dir_cols = ["side", "_side", "direction", "label", "signal_side", "pos_side"]
        for col in dir_cols:
            if col in df.columns:
                print(f"  [{col}]: {df[col].value_counts().to_dict()}")

        # Métricas de retorno
        ret_col = next((c for c in ["ret", "ret_net", "pnl", "return"] if c in df.columns), None)
        if ret_col:
            wr = (df[ret_col] > 0).mean()
            print(f"  ret({ret_col}): mean={df[ret_col].mean():.4f} | WR={wr:.1%} | n={len(df)}")
        
        # Fecha
        ts_col = next((c for c in ["ts", "timestamp", "open_time", "date"] if c in df.columns), None)
        if ts_col:
            print(f"  Período: {df[ts_col].min()} → {df[ts_col].max()}")

    except Exception as e:
        print(f"  ERROR leyendo {f}: {e}")

# Verificar si hay modelos LONG previos vs SHORT nuevos en el cache
print("\n\n[CHECK-OOS-DIRECTION] Verificando coexistencia long/short en wfb_cache...")
long_models = glob.glob("data/wfb_cache/**/xgboost_meta_*_long*.model", recursive=True)
short_models = glob.glob("data/wfb_cache/**/xgboost_meta_*_short*.model", recursive=True)
generic_models = glob.glob("data/wfb_cache/**/*.model", recursive=True)

print(f"  Modelos *_long*.model: {len(long_models)}")
print(f"  Modelos *_short*.model: {len(short_models)}")
print(f"  Modelos genéricos (.model): {len(generic_models)}")

for m in sorted(short_models)[:6]:
    print(f"  SHORT: {os.path.relpath(m, 'data')}")
for m in sorted(long_models)[:6]:
    print(f"  LONG:  {os.path.relpath(m, 'data')}")

# Verificar si hay archivos xgboost sin sufijo (que serían overwrite de long)
nosuffix = [m for m in generic_models if "_short" not in m and "_long" not in m and "_bear" not in m and "_bull" not in m and "_range" not in m]
if nosuffix:
    print(f"\n  ⚠️  MODELOS SIN SUFIJO DIRECCIÓN ({len(nosuffix)}) — posible SOBREESCRITURA:")
    for m in nosuffix[:5]:
        print(f"      {os.path.relpath(m, 'data')}")
else:
    print("\n  ✅ No hay modelos sin sufijo de dirección — no hay sobreescritura.")
