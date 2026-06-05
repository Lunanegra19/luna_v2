import yaml
d = yaml.safe_load(open('config/settings.yaml'))
sym = d.get('data', {}).get('trading_symbol', 'NOT FOUND')
dust = d.get('position_sizer', {}).get('spot_dust_threshold', 'NOT FOUND')
print(f'[VPS-OK] trading_symbol={sym} | spot_dust_CUTOFF = {dust}')
print('[VPS-OK] YAML valido, sin errores de parseo.')
