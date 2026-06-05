import socket

# MONKEY-PATCH: Force IPv4 for all socket resolutions to match OKX whitelist
orig_getaddrinfo = socket.getaddrinfo
def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = ipv4_only_getaddrinfo

import ccxt
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("OKX_API_KEY")
secret = os.getenv("OKX_SECRET_KEY")
passphrase = os.getenv("OKX_PASSPHRASE")

print("--- TESTING DEMO MODE (x-simulated-trading: 1) with EEA HOST ---")
exchange_demo = ccxt.okx({
    'apiKey': api_key,
    'secret': secret,
    'password': passphrase,
    'hostname': 'eea.okx.com',  # Crucial for OKX Europe users!
})
exchange_demo.set_sandbox_mode(True)
try:
    balance = exchange_demo.fetch_balance()
    print("✅ Success in EEA Demo Mode! Balance keys:", balance.keys())
    if 'info' in balance and 'data' in balance['info']:
        print("Demo Account Equity (eqUsd):", balance['info']['data'][0].get('eqUsd'))
except Exception as e:
    print("❌ Failed in EEA Demo Mode:", e)

print("\n--- TESTING LIVE MODE (No simulated header) with EEA HOST ---")
exchange_live = ccxt.okx({
    'apiKey': api_key,
    'secret': secret,
    'password': passphrase,
    'hostname': 'eea.okx.com',  # Crucial for OKX Europe users!
})
try:
    balance = exchange_live.fetch_balance()
    print("✅ Success in EEA Live Mode! Balance keys:", balance.keys())
except Exception as e:
    print("❌ Failed in EEA Live Mode:", e)
