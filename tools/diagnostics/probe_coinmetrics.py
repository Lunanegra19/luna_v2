"""Probe CoinMetrics — testar metricas gratuitas reales para BTC."""
import sys, requests, json
sys.stdout.reconfigure(encoding='utf-8')

base = 'https://community-api.coinmetrics.io/v4'

# Testar las 4 metricas free que encontramos + otras candidatas
candidate_metrics = [
    'SplyCur',        # supply total — disponible
    'SplyExNtv',      # supply en exchanges — disponible
    'SplyExUSD',      # supply exchanges en USD
    'SplyExpFut10yr', # supply esperado 10yr
    'AdrActCnt',      # active addresses
    'TxCnt',          # tx count
    'FeeTotNtv',      # fees total
    'PriceBTC',       # precio en BTC
    'CapMrktCurUSD',  # market cap
    'ROI30d',         # return 30d
    'NVTAdj',         # NVT ajustado
    'SoprEnt',        # SOPR entidades
    'IssTotNtv',      # issuance
]

url_ts = f'{base}/timeseries/asset-metrics'
print('[PROBE] Testando metricas gratuitas en CoinMetrics Community...')
print()
available = []
for metric in candidate_metrics:
    params = {
        'assets': 'btc',
        'metrics': metric,
        'frequency': '1d',
        'start_time': '2023-01-01',
        'end_time': '2023-01-05',
    }
    try:
        r = requests.get(url_ts, params=params, timeout=10)
        if r.status_code == 200:
            rows = r.json().get('data', [])
            val = rows[0].get(metric, 'N/A') if rows else 'empty'
            print(f'  [OK]  {metric:20s}: {val}')
            available.append(metric)
        elif r.status_code == 403:
            print(f'  [---] {metric:20s}: PREMIUM (403)')
        else:
            print(f'  [???] {metric:20s}: HTTP {r.status_code}')
    except Exception as e:
        print(f'  [ERR] {metric:20s}: {e}')

print()
print(f'Metricas gratuitas disponibles: {available}')
print()

# Si SplyExNtv disponible, testar como proxy LTH
if 'SplyExNtv' in available and 'SplyCur' in available:
    params = {
        'assets': 'btc',
        'metrics': 'SplyCur,SplyExNtv',
        'frequency': '1d',
        'start_time': '2021-01-01',
        'end_time': '2021-01-15',
    }
    r = requests.get(url_ts, params=params, timeout=15)
    if r.status_code == 200:
        rows = r.json().get('data', [])
        print('=== SplyCur vs SplyExNtv (primeras 5 filas) ===')
        for row in rows[:5]:
            total = float(row.get('SplyCur', 0))
            exch  = float(row.get('SplyExNtv', 0))
            non_exch = total - exch
            pct_non_exch = non_exch / total * 100 if total > 0 else 0
            print(f'  {row["time"][:10]}: SplyCur={total/1e6:.3f}M | SplyEx={exch/1e6:.3f}M | '
                  f'Non-Exchange={non_exch/1e6:.3f}M ({pct_non_exch:.1f}%)')
