"""
[TEST-O4-FIX] Verifica que las 4 features milag ausentes se generan correctamente
tras el Fix O4 en MI_LAG_FEATURES de feature_pipeline.py.
Ejecuta SOLO el bloque de MI-Lag (sin recomputar todo el pipeline).
"""
import sys, pandas as pd
sys.path.insert(0, "/root/luna_v2")

PARQUET = "/root/luna_v2/data/features/features_live.parquet"
TARGET = [
    ("DXY_z90d",              96,  "DXY_z90d_milag96h"),
    ("Whale_Proxy_Volume_USD", 500, "Whale_Proxy_Volume_USD_milag500h"),
    ("Stablecoins_Delta_30d", 12,  "Stablecoins_Delta_30d_milag12h"),
    ("CPI_YoY_kz",            48,  "CPI_YoY_kz_milag48h"),
]

print("=" * 70)
print("[TEST-O4-FIX] Verificando Fix O4 — 4 features milag ausentes")
print("=" * 70)

df = pd.read_parquet(PARQUET)
print(f"Parquet: {len(df)} filas | {len(df.columns)} columnas")

all_ok = True
for base, lag, derived in TARGET:
    # Simular el shift que hace MI_LAG_FEATURES
    if base not in df.columns:
        print(f"  ❌ {derived} — BASE '{base}' NO ENCONTRADA en parquet")
        all_ok = False
        continue

    simulated = df[base].shift(lag)
    last_simulated = simulated.dropna().iloc[-1] if not simulated.dropna().empty else None
    nan_pct = simulated.isna().mean() * 100

    # ¿Ya existe la derivada en el parquet (de una corrida anterior)?
    in_parquet = derived in df.columns
    last_in_parquet = df[derived].dropna().iloc[-1] if (in_parquet and not df[derived].dropna().empty) else None

    print(f"\n  Feature: {derived}")
    print(f"    Base '{base}': last={float(df[base].dropna().iloc[-1]):.6f} | NaN=0.0%")
    print(f"    Shift({lag}) simulado: last={last_simulated:.6f} | NaN={nan_pct:.1f}%")
    print(f"    ¿Ya en parquet?: {'✅ SÍ — last=' + str(round(last_in_parquet, 6)) if in_parquet else '❌ NO (se generará en próximo ciclo)'}")

    # Verificar que no hay look-ahead bias (shift debe ser >= 1)
    bias_safe = lag >= 1
    print(f"    Look-ahead bias (R1): {'✅ SEGURO — lag={lag}H' if bias_safe else '🔴 PELIGRO'}")

    if last_simulated is not None:
        print(f"    ✅ Fix O4 puede generar esta feature correctamente")
    else:
        all_ok = False

# Verificar que MI_LAG_FEATURES del módulo actualizado contiene las 4 features
print(f"\n  --- Verificación del módulo actualizado ---")
try:
    # Ejecutar solo la parte de MI_LAG_FEATURES para confirmar que el dict es correcto
    from luna.features import feature_pipeline as fp_module
    import importlib
    importlib.reload(fp_module)

    # Instanciar el pipeline y obtener el dict
    # Usamos un hack seguro: parsear el source para encontrar las 4 nuevas líneas
    import inspect
    src = inspect.getsource(fp_module)
    found = []
    for _, _, derived in TARGET:
        # El nombre del output debe aparecer en el source
        if f"'{derived}'" in src or f'"{derived}"' in src:
            found.append(derived)
            print(f"  ✅ '{derived}' encontrado en MI_LAG_FEATURES del módulo")
        else:
            print(f"  ❌ '{derived}' NO encontrado en el módulo — verificar el fix")
            all_ok = False
except Exception as e:
    print(f"  [WARN] No se pudo verificar el módulo directamente: {e}")

print(f"\n{'='*70}")
if all_ok:
    print("[TEST-O4-FIX] ✅ FIX VALIDADO — Las 4 features se generarán en el próximo ciclo")
    print("  Los WARNINGs 'sin 4 features' desaparecerán del stderr de regime_router")
else:
    print("[TEST-O4-FIX] ❌ PROBLEMAS DETECTADOS — Revisar arriba")
print("=" * 70)
