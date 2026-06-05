#!/usr/bin/env python3
"""
[FIX-FRED-01] Actualiza las claves API reales en el .env de la VPS.
Recuperadas desde Luna v1 .env (fuente: proyecto hermano).
- FRED_API_KEY: 32-char hex — requerida para fetch_macro.py (CPI, M2, FedFunds, etc.)
- COINGLASS_API_KEY: requerida para fetch de datos de derivados
Ejecutar en VPS: python tools/diagnostics/fix_vps_env_keys.py
"""
import os
import re
from pathlib import Path

ENV_PATH = Path("/root/luna_v2/.env")

UPDATES = {
    "FRED_API_KEY": "956a61d76442b65a11d877b36686465a",
    "COINGLASS_API_KEY": "e886f40be88641699e63ced49ce91c79",
    # Limpiar referencias Kraken legacy (no bloquea, pero ordena el .env)
    "KRAKEN_API_KEY": None,         # None = eliminar si es mock
    "KRAKEN_API_SECRET": None,
}

print(f"[FIX-FRED-01] Actualizando {ENV_PATH}")
content = ENV_PATH.read_text(encoding="utf-8")
lines = content.splitlines()

updated_lines = []
updated_keys = set()

for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        updated_lines.append(line)
        continue

    # Parsear KEY=VALUE
    if "=" not in stripped:
        updated_lines.append(line)
        continue

    key = stripped.split("=", 1)[0].strip()

    if key in UPDATES:
        new_val = UPDATES[key]
        if new_val is None:
            # Eliminar si contiene "mock" o es Kraken legacy
            current_val = stripped.split("=", 1)[1]
            if "mock" in current_val.lower():
                print(f"[FIX-FRED-01] ELIMINANDO línea mock: {key}=<mock>")
                updated_keys.add(key)
                continue  # Saltar la línea
            else:
                updated_lines.append(line)
        else:
            current_val = stripped.split("=", 1)[1]
            if current_val == new_val:
                print(f"[FIX-FRED-01] {key}: ya tiene el valor correcto. Sin cambio.")
                updated_lines.append(line)
            else:
                print(f"[FIX-FRED-01] ACTUALIZANDO {key}: '{current_val[:10]}...' → '{new_val[:10]}...'")
                updated_lines.append(f"{key}={new_val}")
            updated_keys.add(key)
    else:
        updated_lines.append(line)

# Añadir keys que no existían en el archivo
for key, val in UPDATES.items():
    if key not in updated_keys and val is not None:
        print(f"[FIX-FRED-01] AÑADIENDO nueva key: {key}='{val[:10]}...'")
        updated_lines.append(f"{key}={val}")

new_content = "\n".join(updated_lines) + "\n"
ENV_PATH.write_text(new_content, encoding="utf-8")
print(f"[FIX-FRED-01] .env actualizado correctamente.")

# Verificación inmediata
print("\n[FIX-FRED-01] Verificación post-fix:")
from dotenv import load_dotenv, dotenv_values
vals = dotenv_values(ENV_PATH)
for key in ["FRED_API_KEY", "COINGLASS_API_KEY"]:
    v = vals.get(key, "")
    status = "✅ OK" if v and "mock" not in v.lower() and len(v) >= 10 else "❌ FALLO"
    print(f"  {status} {key}: '{v[:12]}...' (len={len(v)})")

print("\n[FIX-FRED-01] COMPLETADO. Reiniciar luna-v2-live-demo para que cargue las nuevas keys.")
