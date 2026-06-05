"""
Análisis profundo: PBO con diferentes n_blocks y por qué PBO=50% es tan frecuente.
También verifica configuración de settings.yaml.
"""
import yaml
import numpy as np

# 1. Verificar settings.yaml
with open(r'G:\Mi unidad\ia\luna_v2\config\settings.yaml') as f:
    cfg = yaml.safe_load(f)

stat = cfg.get('stat', {})
print('[AUDIT] === stat section from settings.yaml ===')
for k, v in stat.items():
    print(f'  {k}: {v}')

PBO_NUM_CHUNKS = stat.get('PBO_NUM_CHUNKS', 'NOT FOUND -> fallback=16')
min_trades = stat.get('min_trades', 'NOT FOUND')
print()
print(f'[AUDIT] PBO_NUM_CHUNKS en settings: {PBO_NUM_CHUNKS}')
print(f'[AUDIT] min_trades en settings: {min_trades}')

# 2. Diagnostico del threshold PBO=50%
print()
print('[AUDIT] === DIAGNÓSTICO PBO=50% ===')
print('El código retorna 0.50 cuando n_trades < n_blocks*4')
print()

for nb in [8, 16]:
    print(f'Con n_blocks={nb}: min trades necesarios = {nb*4}')
    for nt in [30, 35, 38, 45, 48, 55, 64]:
        meets = nt >= nb * 4
        result = 'CSCV REAL' if meets else '0.50 CONSERVADOR'
        print(f'  n_trades={nt}: cumple={meets} -> {result}')
    print()

# 3. Simulación de PBO con seed1337 aprobada
def simulate_pbo(returns, n_blocks=8, annual_factor=None):
    if annual_factor is None:
        annual_factor = np.sqrt(365 * 24)
    n = len(returns)
    if n < n_blocks * 4:
        print(f'  [WARN] n={n} < {n_blocks * 4} -> retorna 0.50 conservador')
        return 0.50

    block_size = n // n_blocks
    blocks = [returns[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    rng = np.random.default_rng(42)
    n_sims = min(200, n_blocks * (n_blocks - 1))
    overfit_count = 0
    for _ in range(n_sims):
        perm = rng.permutation(n_blocks)
        half = n_blocks // 2
        is_ret = np.concatenate([blocks[i] for i in perm[:half]])
        oos_ret = np.concatenate([blocks[i] for i in perm[half:]])
        sr_is = np.mean(is_ret) / (np.std(is_ret) + 1e-10) * annual_factor
        sr_oos = np.mean(oos_ret) / (np.std(oos_ret) + 1e-10) * annual_factor
        if sr_is > 0.0 and sr_oos <= 0.0:
            overfit_count += 1
    return overfit_count / n_sims

# Simulacion seed1337 aprobada: 38 trades, WR=50%, Sharpe=1.026
np.random.seed(42)
# Con Sharpe=1.026 y WR=50%, los retornos deben tener R:R > 1
returns_pos = np.array([0.012] * 19 + [-0.008] * 19)
np.random.shuffle(returns_pos)

print('[AUDIT] === SIMULACIÓN SEED1337 (aprobada): 38 trades, WR=50% ===')
pbo_8 = simulate_pbo(returns_pos, n_blocks=8)
print(f'  PBO con n_blocks=8 (settings.yaml): {pbo_8*100:.1f}%')
pbo_16 = simulate_pbo(returns_pos, n_blocks=16)
print(f'  PBO con n_blocks=16 (hardcode fallback): {pbo_16*100:.1f}%')

print()
print('[AUDIT] === CONCLUSIÓN CRÍTICA ===')
if isinstance(PBO_NUM_CHUNKS, int):
    print(f'  settings.yaml tiene PBO_NUM_CHUNKS={PBO_NUM_CHUNKS}')
    if PBO_NUM_CHUNKS == 8:
        min_needed = 8 * 4
        print(f'  Con n_blocks=8, se necesitan >= {min_needed} trades para CSCV real')
        print(f'  La mayoría de seeds ({30} a {55} trades) pasan el umbral mínimo')
    print(f'  PERO el código hardcodea PBO_NUM_CHUNKS=16 en el fallback (línea 95)')
    print(f'  Si settings.yaml FALLA al cargar -> PBO_NUM_CHUNKS=16 -> se necesitan 64 trades')
    print(f'  NINGUNA seed de esta run tiene 64+ trades -> TODAS reportan PBO=0.50')
else:
    print(f'  ALERTA: PBO_NUM_CHUNKS no encontrado en settings.yaml!')
