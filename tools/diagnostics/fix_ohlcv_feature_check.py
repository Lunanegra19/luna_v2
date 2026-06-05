"""
[FIX-OHLCV-FEATURE-CHECK] Corrige el grupo OHLCV en el dashboard.

PROBLEMA: Checks para returns_1h, returns_24h, atr_14h, volatility_24h que NO existen
ni en train ni en live (columnas aspiracionales). Resultado: siempre 5/9 WARN.

FIX: Actualizar a columnas reales verificadas en features_live.parquet.
Las features técnicas reales del pipeline usan nombres como mt_vol_realized_4bar, etc.
Para el grupo OHLCV del dashboard usamos solo las OHLCV base reales.
"""

SERVER_PATH = '/root/luna_v2/dashboard/server.py'

with open(SERVER_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

old_ohlcv = '''                        _check(["close", "open", "high", "low", "volume",
                                "returns_1h", "returns_24h", "atr_14h", "volatility_24h"],
                               "OHLCV + Derivadas Precio", "📊"),'''

new_ohlcv = '''                        # [FIX-OHLCV-FEATURE-CHECK] Columnas OHLCV reales del pipeline.
                        # returns_1h/24h, atr_14h, volatility_24h no existen en train ni live
                        # (nombres aspiracionales nunca implementados con esos nombres exactos).
                        # El pipeline usa Futures_Volume, close_perps y features de momentum
                        # con otros identificadores. Se verifica solo lo que realmente existe.
                        _check(["close", "open", "high", "low", "volume",
                                "Futures_Volume", "close_perps", "taker_buy_base_perps"],
                               "OHLCV + Derivadas Precio", "📊"),'''

if old_ohlcv in content:
    content = content.replace(old_ohlcv, new_ohlcv, 1)
    print('[FIX-OHLCV-FEATURE-CHECK] OK - OHLCV check actualizado a columnas reales')
else:
    print('[FIX-OHLCV-FEATURE-CHECK] ERROR - bloque no encontrado')
    exit(1)

with open(SERVER_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

import subprocess
result = subprocess.run(
    ['python3', '-c', f'import ast; ast.parse(open("{SERVER_PATH}").read()); print("SYNTAX OK")'],
    capture_output=True, text=True
)
print(f'[FIX-OHLCV-FEATURE-CHECK] Sintaxis: {result.stdout.strip() or result.stderr.strip()}')
