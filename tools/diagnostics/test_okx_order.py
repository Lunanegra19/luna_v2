import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Configure UTF-8 encoding for Windows to prevent charmap crashes
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load environment variables
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

from luna.live.okx_connector import OKXBrokerConnector

def run_diagnostic():
    print("="*60)
    print("🌙 [LUNA V2 DIAGNOSTICS] INICIANDO MULTI-PAIR PRUEBA DE CONECTOR OKX (SPOT)")
    print("="*60)
    
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_SECRET_KEY")
    passphrase = os.getenv("OKX_PASSPHRASE")
    
    if not api_key or not secret_key:
        print("❌ [ERROR] Faltan la clave API o el secreto en el archivo .env.")
        sys.exit(1)
        
    try:
        # 1. Initialize connector in Demo/Sandbox mode
        print("\n[STEP 1] Inicializando OKXBrokerConnector en modo demo (demo_mode=True)...")
        connector = OKXBrokerConnector(demo_mode=True)
        
        # 2. Retrieve net account equity
        print("\n[STEP 2] Consultando la equidad (Equity) unificada en OKX Demo...")
        equity = connector.fetch_equity()
        print(f"💰 [EQUITY_RESULT] Equidad neta de la cuenta Demo: ${equity:,.2f} USD")
        
        # We will try a few pairs to see which ones are allowed by local compliance
        # We specify small amounts for each (below standard thresholds but valid)
        pairs_to_try = [
            ("BTC/EUR", 0.0001),   # 0.0001 BTC is approx 6 EUR
            ("ETH/EUR", 0.001),    # 0.001 ETH is approx 3 EUR
            ("EUR/USDT", 2.0),     # 2 EUR
            ("USDT/EUR", 2.0),     # 2 USDT
            ("SOL/EUR", 0.05),     # 0.05 SOL is approx 6 EUR
        ]
        
        print("\n[STEP 3] Probando ordenes en diferentes pares para identificar restricciones...")
        
        allowed_pair = None
        for test_symbol, trade_amount in pairs_to_try:
            print(f"\n--- Probando par: {test_symbol} (Cantidad: {trade_amount}) ---")
            try:
                # Place order
                order = connector.execute_market_order(
                    symbol=test_symbol,
                    side="buy",
                    contracts=trade_amount
                )
                if order:
                    print(f"✅ [SUCCESS] ¡Orden SPOT de {test_symbol} ejecutada correctamente!")
                    print(f"Order ID: {order.get('id')} | Precio: {order.get('average', order.get('price', 'N/D'))}")
                    allowed_pair = (test_symbol, trade_amount)
                    break
                else:
                    print(f"❌ [FAILED] La orden en {test_symbol} no devolvió respuesta.")
            except Exception as e:
                err_str = str(e)
                print(f"❌ [FAILED] Excepción en {test_symbol}: {err_str}")
                if "51155" in err_str:
                    print(f"👉 [COMPLIANCE] El par {test_symbol} está bloqueado por restricciones de cumplimiento local.")
                elif "51004" in err_str or "Insufficient" in err_str or "balance" in err_str.lower():
                    print(f"👉 [BALANCE] ¡El par {test_symbol} SÍ está permitido por cumplimiento! Pero falló por saldo insuficiente.")
                    allowed_pair = (test_symbol, trade_amount)
                    # We can stop here since we found an allowed pair
                    break
                
        if allowed_pair:
            symbol, amount = allowed_pair
            print("\n" + "="*50)
            print(f"🎉 ¡PAR COMPATIBLE ENCONTRADO!: {symbol}")
            print(f"Este par puede usarse para las pruebas de trading en la VPS.")
            print("="*50)
        else:
            print("\n❌ No se encontraron pares Spot compatibles en el entorno de pruebas Demo.")
            
    except Exception as e:
        print(f"❌ [DIAGNOSTIC_FATAL] Error crítico durante la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostic()
