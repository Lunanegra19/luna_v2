#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de diagnóstico para probar el resolvedor robusto de regímenes HMM
y asegurar que no se produzca el bug Silent Regime Fallback.
"""

import sys
import os
from pathlib import Path

# Añadir raíz del proyecto al path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

print(f"Raíz del proyecto detectada: {project_root}")

try:
    from luna.models.predict_oos import (
        get_hmm_tbm_params, 
        get_hmm_horizon,
        HMM_TBM_PARAMS,
        HMM_HORIZON_MAP,
        _HMM_TBM_FALLBACK,
        _HMM_HORIZON_FALLBACK
    )
    print("[IMPORT-TEST] OK: resolvedores de predict_oos importados con éxito.")
except Exception as e:
    print(f"[IMPORT-TEST] ERROR al importar resolvedores: {e}")
    sys.exit(1)

# Casos de prueba: (nombre_entrada, tp_esperado, sl_esperado, horizon_esperado)
test_cases = [
    # Coincidencia exacta
    ("1_BULL_TREND", 2.5, 1.5, 168),
    ("1_VOLATILE_BULL", 3.0, 2.5, 240),
    ("2_CALM_RANGE", 1.2, 0.6, 96),
    
    # Coincidencia por prefijo base
    ("1_BULL_TREND_WEAK", 2.5, 1.5, 168),
    ("2_CALM_RANGE_B", 1.2, 0.6, 96),
    ("3_BEAR_CRASH_C", 1.5, 1.5, 168),
    ("1_VOLATILE_BULL_B", 3.0, 2.5, 240),
    
    # Fallback (regímenes no-canónicos / neutrales esperados)
    ("UNKNOWN_REGIME_XYZ", 1.5, 0.8, 168),
    ("4_BEAR_FORCED", 1.5, 0.8, 168),
]

failed = 0
print("\n--- INICIANDO VERIFICACIÓN DE RESOLVEDORES ---")
for entry, exp_tp, exp_sl, exp_hor in test_cases:
    params = get_hmm_tbm_params(entry)
    hor = get_hmm_horizon(entry)
    
    tp_ok = abs(params["tp"] - exp_tp) < 1e-5
    sl_ok = abs(params["sl"] - exp_sl) < 1e-5
    hor_ok = hor == exp_hor
    
    status = "OK" if (tp_ok and sl_ok and hor_ok) else "FALLÓ"
    if status == "FALLÓ":
        failed += 1
        
    print(f"Régimen: {entry:<20} | TP: {params['tp']:.1f} (Esperado {exp_tp:.1f}) | SL: {params['sl']:.1f} (Esperado {exp_sl:.1f}) | Hor: {hor} (Esperado {exp_hor}) | [{status}]")

print("\n--- INICIANDO VERIFICACIÓN DE GUARDAS FAIL-FAST (EXCEPCIONES ESPERADAS) ---")
fail_fast_cases = ["1_BULL_ROTO", "2_RANGE_ROTO", "3_BEAR_ROTO"]
for invalid_entry in fail_fast_cases:
    # Probar get_hmm_tbm_params
    try:
        get_hmm_tbm_params(invalid_entry)
        print(f"Régimen: {invalid_entry:<20} (TBM Params) | Se esperaba ValueError pero NO se lanzó! | [FALLÓ]")
        failed += 1
    except ValueError as e:
        print(f"Régimen: {invalid_entry:<20} (TBM Params) | Lanzó ValueError esperado: {e} | [OK]")
    except Exception as e:
        print(f"Régimen: {invalid_entry:<20} (TBM Params) | Lanzó una excepción inesperada: {type(e).__name__}: {e} | [FALLÓ]")
        failed += 1

    # Probar get_hmm_horizon
    try:
        get_hmm_horizon(invalid_entry)
        print(f"Régimen: {invalid_entry:<20} (Horizon)   | Se esperaba ValueError pero NO se lanzó! | [FALLÓ]")
        failed += 1
    except ValueError as e:
        print(f"Régimen: {invalid_entry:<20} (Horizon)   | Lanzó ValueError esperado: {e} | [OK]")
    except Exception as e:
        print(f"Régimen: {invalid_entry:<20} (Horizon)   | Lanzó una excepción inesperada: {type(e).__name__}: {e} | [FALLÓ]")
        failed += 1

print("---------------------------------------------")
if failed == 0:
    print("✅ ¡Prueba exitosa! Todos los resolvedores mapearon correctamente y las guardas fail-fast arrojaron ValueError según lo previsto.")
    sys.exit(0)
else:
    print(f"❌ ¡Prueba fallida! {failed} casos de prueba fallaron.")
    sys.exit(1)
