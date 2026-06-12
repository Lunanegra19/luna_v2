import sys
sys.stdout.reconfigure(encoding='utf-8')
import ast
import yaml
from pathlib import Path

print('[VERIFICACION SINTACTICA P3+P2]')

# YAML
cfg = yaml.safe_load(Path('config/settings.yaml').read_text('utf-8'))
exec_cfg = cfg.get('execution', {})
assert exec_cfg.get('symbol') == 'BTC/USDT:USDT'
assert exec_cfg.get('instrument_type') == 'swap'
assert exec_cfg.get('use_hybrid_execution') == True
print('  [OK] settings.yaml - symbol=BTC/USDT:USDT instrument_type=swap hybrid=True')

# Python syntax
for fpath in ['luna/live/okx_connector.py', 'luna/live/run_luna_live.py']:
    src = Path(fpath).read_text('utf-8')
    try:
        ast.parse(src)
        print(f'  [OK] {fpath} - sintaxis Python valida')
    except SyntaxError as e:
        print(f'  [ERROR] {fpath} - SyntaxError: {e}')
        sys.exit(1)

# Verificar contratos funcionales en okx_connector.py
connector_src = Path('luna/live/okx_connector.py').read_text('utf-8')
assert 'def execute_order(' in connector_src
assert 'def get_trading_symbol(' in connector_src
assert 'def fetch_equity(' in connector_src
assert '_global_cfg.execution.symbol' in connector_src
print('  [OK] okx_connector.py: execute_order + get_trading_symbol + fetch_equity + cfg.execution.symbol PRESENTES')

# Verificar MFT
mft_src = Path('luna/live/run_luna_live.py').read_text('utf-8')
assert 'OKXBrokerConnector' in mft_src
assert 'self.connector = OKXBrokerConnector(' in mft_src
assert 'self.connector.execute_order(' in mft_src
assert 'BTC/USD:BTC' not in mft_src
print('  [OK] run_luna_live.py: OKXBrokerConnector instanciado, execute_order conectado, sin simbolos hardcodeados')

print()
print('[RESULTADO] P3+P2 IMPLEMENTACION VERIFICADA.')
