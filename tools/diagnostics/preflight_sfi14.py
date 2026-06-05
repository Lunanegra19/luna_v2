import sys
sys.path.insert(0, r'G:\Mi unidad\ia\luna_v2')
from config.settings import cfg

print('[PRE-FLIGHT SFI14] Verificando configuracion critica...')
print(f'  sfi_top_n      = {cfg.features.sfi_top_n}  (esperado: 13 -> ~14 features)')
print(f'  pbo_n_blocks   = {cfg.stat.pbo_n_blocks}  (esperado: 8)')
print(f'  max_pbo        = {cfg.stat.max_pbo}  (esperado: 0.22)')
print(f'  min_dsr        = {cfg.stat.min_dsr}  (esperado: 0.75)')
print(f'  max_drawdown   = {cfg.stat.max_drawdown}  (esperado: 0.60)')
print(f'  min_trades     = {cfg.stat.min_trades}  (esperado: 32)')
print(f'  alpha_binomial = {cfg.stat.alpha_binomial}  (esperado: 1.0)')
print(f'  cusum_thresh   = {cfg.stat.cusum_threshold}')
print(f'  embargo_hours  = {cfg.sop.embargo_hours}')
print()
print('[PRE-FLIGHT] Ventanas WFB:')
for w in cfg.wfb.windows:
    wid = getattr(w, 'id', '?')
    te  = getattr(w, 'train_end', '?')
    hs  = getattr(w, 'holdout_start', '?')
    he  = getattr(w, 'holdout_end', '?')
    print(f'  {wid}: train_end={te} | holdout={hs} -> {he}')
print()
print('[PRE-FLIGHT] OK - Listo para lanzar.')
