"""
[BUG-L3 DIAGNÓSTICO] Inspecciona el parquet features_train_final.parquet en el VPS.
Verifica si el rango de fechas excede train_end y cuántas filas son problemáticas.
Contexto: BUG-L3 produce tribe_id=0 (fallback causal) en TODOS los ciclos live
porque el parquet contiene fechas > train_end=2024-10-31.
Solución correcta: regenerar el parquet con run_weekly_mining.py filtrado,
O filtrar el parquet en disco directamente.
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from config.settings import cfg

TRAIN_END = pd.Timestamp(cfg.temporal_splits.train_end, tz='UTC')
PARQUET_PATH = PROJECT_ROOT / "data" / "features" / "features_train_final.parquet"

print("=" * 65)
print("  BUG-L3 DIAGNÓSTICO — features_train_final.parquet")
print("=" * 65)
print(f"  train_end (settings.yaml): {TRAIN_END.date()}")
print(f"  Parquet path: {PARQUET_PATH}")
print(f"  Parquet existe: {PARQUET_PATH.exists()}")

if not PARQUET_PATH.exists():
    print("[ERROR] Parquet no encontrado. No hay nada que hacer.")
    sys.exit(1)

src = pd.read_parquet(PARQUET_PATH)
src.index = pd.to_datetime(src.index, utc=True)

print(f"\n  Total filas en parquet: {len(src):,}")
print(f"  Fecha mínima:          {src.index.min()}")
print(f"  Fecha máxima:          {src.index.max()}")
print(f"  Tiene KMeans_Tribe_ID: {'KMeans_Tribe_ID' in src.columns}")
print(f"  train_end (configurado): {TRAIN_END.date()}")

mask_leakage = src.index > TRAIN_END
n_leakage = mask_leakage.sum()
n_ok = (~mask_leakage).sum()
print(f"\n  Filas <= train_end (OK):       {n_ok:,}")
print(f"  Filas > train_end (LEAKAGE):   {n_leakage:,}")

if n_leakage > 0:
    print(f"\n  ⚠️ BUG-L3 ACTIVO: {n_leakage:,} filas con fechas post-train_end")
    print(f"  → Solución: Filtrar el parquet en disco.")
    
    # FIX: Filtrar el parquet para que solo contenga filas <= train_end
    src_clean = src[~mask_leakage].copy()
    print(f"\n  Aplicando fix: guardando parquet filtrado...")
    print(f"  Filas originales: {len(src):,} → Filas limpias: {len(src_clean):,}")
    
    # Backup del original
    backup_path = PARQUET_PATH.parent / "features_train_final_backup_bugl3.parquet"
    src.to_parquet(backup_path)
    print(f"  [BUG-L3-FIX] Backup guardado en: {backup_path.name}")
    
    # Guardar el parquet limpio
    src_clean.to_parquet(PARQUET_PATH)
    print(f"  [BUG-L3-FIX] Parquet filtrado guardado. Nuevas fechas max: {src_clean.index.max()}")
    
    # Verificación post-fix
    src_verify = pd.read_parquet(PARQUET_PATH)
    src_verify.index = pd.to_datetime(src_verify.index, utc=True)
    still_leaking = (src_verify.index > TRAIN_END).sum()
    if still_leaking == 0:
        print(f"  [BUG-L3-FIX] ✅ VERIFICADO: 0 filas > train_end. BUG-L3 eliminado.")
    else:
        print(f"  [BUG-L3-FIX] ⚠️ ATENCIÓN: Quedan {still_leaking} filas > train_end tras el fix.")
else:
    print(f"\n  ✅ SIN LEAKAGE: todas las filas <= train_end. BUG-L3 no activo.")

print("\n" + "=" * 65)
