import sys
sys.stdout.reconfigure(encoding='utf-8')
import ccxt

ex = ccxt.okx({'enableRateLimit': True})
try:
    markets = ex.load_markets()
    sym = 'BTC/USDT:USDT'
    if sym in markets:
        m = markets[sym]
        cs = float(m.get('contractSize') or 0.01)
        min_c = float((m.get('limits') or {}).get('amount', {}).get('min') or 1)
        min_cost = (m.get('limits') or {}).get('cost', {}).get('min')
        precision_amt = (m.get('precision') or {}).get('amount')

        print('=== BTC/USDT:USDT (Futuros Perpetuos OKX) ===')
        print(f'  contract size (BTC/contrato): {cs}')
        print(f'  min amount (contratos):       {min_c}')
        print(f'  min cost USD:                 {min_cost}')
        print(f'  precision amount:             {precision_amt}')
        print(f'  type: {m.get("type")} | settle: {m.get("settle")}')

        ticker = ex.fetch_ticker(sym)
        price = float(ticker['last'])
        min_usd_noleverage = cs * min_c * price
        print()
        print('=== POSICION MINIMA ===')
        print(f'  Precio BTC actual: ${price:,.0f}')
        print(f'  Notional minimo (1 contrato, sin leverage): ${min_usd_noleverage:,.0f}')
        print(f'  Margin minimo (x5 leverage):  ${min_usd_noleverage/5:,.0f}')
        print(f'  Margin minimo (x10 leverage): ${min_usd_noleverage/10:,.0f}')
        print(f'  Margin minimo (x25 leverage): ${min_usd_noleverage/25:,.0f}')
        print()
        # Fraccionamiento
        print('=== FRACCIONAMIENTO ===')
        print(f'  Se pueden comprar 0.5 contratos? {"SI" if min_c < 1 else "NO - minimo 1 contrato"}')
        print(f'  Se pueden comprar 0.1 contratos? {"SI" if min_c <= 0.1 else "NO"}')
    else:
        print(f'Simbolo {sym} no encontrado. BTC disponibles:')
        for k in sorted(markets):
            if 'BTC' in k and 'USDT' in k:
                print(f'  {k}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
