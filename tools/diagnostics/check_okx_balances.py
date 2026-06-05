import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load environment variables
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

from luna.live.okx_connector import OKXBrokerConnector

def run():
    connector = OKXBrokerConnector(demo_mode=True)
    try:
        balance = connector.exchange.fetch_balance()
        print("--- FULL BALANCE ---")
        for asset, data in balance.items():
            if isinstance(data, dict) and data.get('total', 0) > 0:
                print(f"{asset}: {data}")
            elif asset == 'total':
                print(f"Total: { {k: v for k, v in data.items() if v > 0} }")
                
        # Also print list of all loaded markets to see which SOL or BTC or EUR markets are loaded
        markets = connector.exchange.markets
        eur_markets = [symbol for symbol in markets.keys() if 'EUR' in symbol]
        print(f"\n--- EUR MARKETS (Count: {len(eur_markets)}) ---")
        print(eur_markets[:20])
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run()
