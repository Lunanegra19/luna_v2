import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path

# Buscamos un pool pre-embargo de la ultima run OOS
# Se guardaron en data/predictions/signal_pool_...
base_dir = Path("c:/Users/Usuario/Desktop/ia/luna_v2")
preds_dir = base_dir / "data" / "predictions"

pools = list(preds_dir.glob("signal_pool_*.parquet"))
if not pools:
    print("No se encontraron signal_pools. Por favor asegurese de tener LUNA_SAVE_SIGNAL_POOL=1 en alguna run.")
    sys.exit(1)

pool_file = max(pools, key=lambda x: x.stat().st_mtime)
print(f"Analizando dataset: {pool_file.name}")

df = pd.read_parquet(pool_file)
# Sort por tiempo por si acaso
df = df.sort_index()

# Simulador de Dynamic Hold
# Asumimos direccion LONG por defecto (el archivo dice _long o _short al final)
direction = 1 if 'long' in pool_file.name.lower() else -1

print(f"Direccion inferida: {'LONG' if direction == 1 else 'SHORT'}")
print(f"Total barras en pool: {len(df)}")

# Necesitamos simular 2 escenarios:
# Escenario 1 (Actual MFT): Entramos si signal (aqui estan todas filtradas) y salimos a la 1h.
# Escenario 2 (Dynamic Hold): Entramos, y mantenemos mientras lgbm_prob o meta_v2_prob > 0.50.

retornos_1h = []
retornos_dyn = []
holding_1h = []
holding_dyn = []

# Costo asumido (ahora que estamos en perpetuos)
cost_rt = 0.0004 

close_prices = df['close'].values
if 'lgbm_prob' in df.columns:
    probs = df['lgbm_prob'].values
elif 'xgb_prob' in df.columns:
    probs = df['xgb_prob'].values
else:
    probs = df['meta_v2_prob'].values

times = df.index

for i in range(len(df) - 1):
    # Condicion de entrada (todas las filas en este pool ya pasaron los filtros iniciales, pero para no sobre-solapar,
    # simulamos que cada señal es un trade independiente y medimos su PnL)
    
    entry_price = close_prices[i]
    
    # 1. Escenario 1h: cerramos en i+1
    exit_price_1h = close_prices[i+1]
    ret_1h = (exit_price_1h / entry_price - 1.0) * direction - cost_rt
    retornos_1h.append(ret_1h)
    holding_1h.append(1)
    
    # 2. Escenario Dynamic: cerramos cuando prob < 0.50 o a las 24h maximo
    dyn_exit_idx = i + 1
    max_hold = min(24, len(df) - i - 1)
    
    for j in range(1, max_hold):
        idx = i + j
        if probs[idx] < 0.50:
            dyn_exit_idx = idx
            break
        # Para evitar perder ganancias masivas, podemos poner un stop-loss dinamico o trailing,
        # pero la hipotesis basica es mantener mientras prob >= 0.50
        dyn_exit_idx = idx 
        
    exit_price_dyn = close_prices[dyn_exit_idx]
    ret_dyn = (exit_price_dyn / entry_price - 1.0) * direction - cost_rt
    retornos_dyn.append(ret_dyn)
    holding_dyn.append(dyn_exit_idx - i)

print("\n=== RESULTADOS SIMULACION ===")
print("Escenario 1 (Hold 1 Hora - Comportamiento actual del MFT):")
print(f"  Return Medio por trade: {np.mean(retornos_1h)*100:.4f}%")
print(f"  Win Rate: {np.mean(np.array(retornos_1h) > 0)*100:.1f}%")
print(f"  Holding Time Medio: {np.mean(holding_1h):.1f} horas")

print("\nEscenario 2 (Dynamic Hold - Mantener mientras prob > 0.50):")
print(f"  Return Medio por trade: {np.mean(retornos_dyn)*100:.4f}%")
print(f"  Win Rate: {np.mean(np.array(retornos_dyn) > 0)*100:.1f}%")
print(f"  Holding Time Medio: {np.mean(holding_dyn):.1f} horas")

# Comparacion
diff = np.sum(retornos_dyn) - np.sum(retornos_1h)
print(f"\nDiferencia Acumulada de Retornos Brutos: {diff*100:.2f}%")
if diff > 0:
    print("CONCLUSION: La hipotesis de Dynamic Hold mejora el PnL.")
else:
    print("CONCLUSION: La hipotesis de Dynamic Hold NO mejora el PnL con reglas simples.")
