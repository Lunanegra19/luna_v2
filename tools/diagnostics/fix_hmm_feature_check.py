"""
[FIX-HMM-FEATURE-CHECK] Corrige el grupo de features HMM en el dashboard.

PROBLEMA: El dashboard buscaba 5 columnas aspiracionales que nunca existieron:
  - hmm_regime_label, hmm_prob_bull, hmm_prob_bear, hmm_prob_volatile, hmm_state_duration_h
  → Resultado: siempre 0/5 ERROR aunque el HMM funciona perfectamente.

SOLUCIÓN: Actualizar el _check() para usar las columnas reales del pipeline:
  - HMM_Regime (numérico, siempre presente)
  - hmm_velocity_bull (derivada de P(bull), siempre presente)
  - hmm_acceleration_bull (segunda derivada, siempre presente)

IMPACTO EN TRADING: NINGUNO. Estas features son solo para monitoring del dashboard.
Los modelos XGB/Meta no usan las features aspiracionales.
"""

SERVER_PATH = '/root/luna_v2/dashboard/server.py'

with open(SERVER_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the HMM feature check
old_hmm_check = '''                        _check(["hmm_regime_label", "hmm_prob_bull", "hmm_prob_bear",
                                "hmm_prob_volatile", "hmm_state_duration_h"],
                               "Régimen HMM", "🧬"),'''

new_hmm_check = '''                        # [FIX-HMM-FEATURE-CHECK] Columnas reales del pipeline HMM en producción.
                        # Pipeline produce: HMM_Regime (numérico), hmm_velocity_bull, hmm_acceleration_bull.
                        # State map: {0:'2_VOLATILE_RANGE', 2:'1_BULL_TREND', 3:'3_BEAR_CRASH', 4:'1_VOLATILE_BULL', 5:'4_BEAR_FORCED'}
                        # Las columnas aspiracionales (hmm_prob_bear, hmm_prob_volatile, hmm_state_duration_h)
                        # nunca fueron implementadas en el pipeline y no existen en ningún modelo.
                        _check(["HMM_Regime", "hmm_velocity_bull", "hmm_acceleration_bull"],
                               "Régimen HMM", "🧬"),'''

if old_hmm_check in content:
    content = content.replace(old_hmm_check, new_hmm_check, 1)
    print('[FIX-HMM-FEATURE-CHECK] OK - HMM feature check actualizado a columnas reales')
else:
    print('[FIX-HMM-FEATURE-CHECK] ERROR - bloque no encontrado')
    idx = content.find('hmm_regime_label')
    if idx >= 0:
        print('Contexto:', repr(content[max(0,idx-100):idx+200]))
    exit(1)

with open(SERVER_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print('[FIX-HMM-FEATURE-CHECK] server.py guardado')

# Verify syntax
import subprocess
result = subprocess.run(
    ['python3', '-c', f'import ast; ast.parse(open("{SERVER_PATH}").read()); print("SYNTAX OK")'],
    capture_output=True, text=True
)
print(f'[FIX-HMM-FEATURE-CHECK] Sintaxis: {result.stdout.strip() or result.stderr.strip()}')

# Also verify the fix is in place
with open(SERVER_PATH, 'r') as f:
    check = f.read()
if '"HMM_Regime", "hmm_velocity_bull", "hmm_acceleration_bull"' in check:
    print('[FIX-HMM-FEATURE-CHECK] VERIFIED: columnas reales en el check ✓')
else:
    print('[FIX-HMM-FEATURE-CHECK] VERIFICATION FAILED')
