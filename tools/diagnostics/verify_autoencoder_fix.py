"""Verify autoencoder_features.py syntax and alias bridge insertion"""
import ast, sys

path = '/root/luna_v2/luna/features/autoencoder_features.py'
with open(path, 'r') as f:
    content = f.read()

try:
    ast.parse(content)
    print('SYNTAX OK')
except SyntaxError as e:
    print(f'SYNTAX ERROR: {e}')
    sys.exit(1)

# Verify alias bridge is present
if '[FIX-ALIAS-TRAIN-LIVE]' in content:
    print('FIX-ALIAS-TRAIN-LIVE: PRESENT')
else:
    print('FIX-ALIAS-TRAIN-LIVE: NOT FOUND - upload may have failed')
    sys.exit(1)

# Count alias entries
alias_count = content.count("'FundingRate_EMA3'")
print(f'Alias bridge entries found: {alias_count}')

print('All checks passed')
