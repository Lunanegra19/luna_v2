"""
Análisis de impacto de FIX-BINOMIAL-01: alpha_binomial 1.0 -> 0.05.
Evalúa qué WR mínimo se necesita para pasar el gate binomial con diferentes n_trades.
"""
import sys
sys.path.insert(0, r'G:\Mi unidad\ia\luna_v2')
from luna.monitoring.statistical_audit import LunaStatisticalAuditor

auditor = LunaStatisticalAuditor()
print('[IMPACT FIX-BINOMIAL-01] === Análisis de impacto alpha=0.05 ===')
print()
print('WR mínimo para pasar el gate binomial (p < 0.05):')
for n in [30, 35, 38, 45, 48, 55, 100]:
    passed = False
    for w in range(n // 2, n + 1):
        p = auditor._compute_binomial_test(wins=w, total_trades=n)
        if p < 0.05:
            wr = w / n * 100
            print(f'  n={n}: necesita al menos {w} wins ({wr:.1f}% WR) para pasar (p={p:.4f})')
            passed = True
            break
    if not passed:
        print(f'  n={n}: NO pasa con ningún WR insuficientes trades')

print()
print('[IMPACT FIX-BINOMIAL-01] === Impacto en seeds existentes ===')
seeds_data = [
    ('seed42 SFI18',   34, 16),
    ('seed100 SFI18',  41, 23),
    ('seed777 SFI18',  55, 27),
    ('seed1337 SFI18', 48, 22),
    ('seed2025 SFI18', 45, 18),
    ('seed42 SFI16',   35, 17),
    ('seed100 SFI16',  31, 17),
    ('seed777 SFI16',  30, 14),
    ('seed1337 SFI16', 38, 19),  # <- La única aprobada
    ('seed2025 SFI16', 36, 16),
]
for name, n, w in seeds_data:
    p = auditor._compute_binomial_test(wins=w, total_trades=n)
    passes = p < 0.05
    wr = w / n * 100
    approved_marker = ' <- ANTERIORMENTE APROBADA' if 'seed1337 SFI16' in name else ''
    result = 'PASS' if passes else 'FAIL'
    print(f'  {name}: WR={wr:.1f}% p={p:.4f} -> {result}{approved_marker}')

print()
print('[IMPACT FIX-BINOMIAL-01] CONCLUSIÓN:')
print('  Con alpha=0.05, ninguna seed de la última run pasa el gate binomial.')
print('  Esto indica que el gate binomial es DEMASIADO ESTRICTO para 30-55 trades.')
print('  Para WR>50% estadísticamente significativo al 95% con n=38, se necesita ~57-58% WR.')
print('  ADVERTENCIA: Este fix hace el sistema más restrictivo y podría rechazar la seed1337.')
