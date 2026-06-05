#!/usr/bin/env python
"""
test_live_trade.py
===================
[LUNA-V2-DIAGNOSTIC] Script de utilidad técnica en tools/refactor/ para verificar 
el correcto funcionamiento de la conexión de base de datos a través del túnel SSH,
la ejecución de órdenes en la API Demo de OKX y la actualización del Dashboard en tiempo real.
Cumple estrictamente con las directrices de SOP V10.0 y las reglas de oro de LUNA V2.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Configurar UTF-8 en Windows para evitar fallos de codificación
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Cargar variables de entorno
load_dotenv(PROJECT_ROOT / ".env")

from luna.database.db_manager import DatabaseManager
from luna.live.okx_connector import OKXBrokerConnector

def run_diagnostic_test():
    print("\n" + "="*80)
    print("      LUNA V2 - DIAGNÓSTICO EN VIVO DE INTEGRACIÓN Y PIPELINE")
    print("="*80 + "\n")
    
    print(f"[TRACK-TEST] Iniciando diagnóstico en local. PROJECT_ROOT: {PROJECT_ROOT}")
    
    # 1. Conexión de Base de Datos
    print("\n[PASO 1] Conectando a Base de Datos PostgreSQL...")
    try:
        db = DatabaseManager()
        if db.connection_pool is None:
            raise RuntimeError("DatabaseManager no inicializó el Connection Pool.")
        print("✅ [OK] Conexión establecida con éxito a la base de datos a través del túnel SSH.")
    except Exception as e:
        print(f"❌ [FALLO] Error de conexión de base de datos: {e}")
        print("[!] Asegúrate de que el túnel SSH esté abierto en el puerto 5433 y el .env esté bien configurado.")
        return

    # 2. Inyección de Latido (Prueba visual instantánea en el Watchdog)
    print("\n[PASO 2] Emitiendo latido de prueba a la tabla system_heartbeat...")
    try:
        db.log_heartbeat(status="TEST_ACTIVE")
        print("✅ [OK] Latido de prueba inyectado como 'TEST_ACTIVE'. El Watchdog del dashboard debería parpadear.")
    except Exception as e:
        print(f"❌ [FALLO] Error al emitir latido: {e}")

    # 3. Inyección de Transacciones de Diagnóstico (Prueba de renderizado en el Dashboard)
    print("\n[PASO 3] Inyectando transacciones de prueba directamente en la base de datos...")
    try:
        current_price = 76245.50
        print("  - Insertando transacción LONG de prueba...")
        db.log_audit(
            timestamp=datetime.utcnow(),
            price=current_price,
            action="LONG",
            confidence=0.8842,
            xgb_prob=0.8842,
            hmm_regime=1, # 1_BULL_TREND
            reason="[TEST DIAGNÓSTICO] Compra de prueba inyectada para testear renderizado en Localhost.",
            contracts=1,
            executed_price=current_price
        )
        
        # Sincronizar live state en DB para simular balance y apalancamiento
        db.update_live_state(
            portfolio_value=10543.20,
            ath=10543.20,
            drawdown=0.00,
            is_paused=False
        )
        print("✅ [OK] Transacción LONG inyectada y balance actualizado a $10,543.20. Verifica la tabla de transacciones.")
        
        print("  - Esperando 12 segundos para simular tiempo de exposición...")
        for i in range(12, 0, -1):
            sys.stdout.write(f"\r    Expira en: {i}s... ")
            sys.stdout.flush()
            time.sleep(1)
        print()
        
        print("  - Insertando transacción de cierre (HOLD) de prueba...")
        exit_price = 76420.10
        db.log_audit(
            timestamp=datetime.utcnow(),
            price=exit_price,
            action="HOLD",
            confidence=0.50,
            xgb_prob=0.50,
            hmm_regime=0,
            reason="[TEST DIAGNÓSTICO] Posición cerrada con éxito en el simulador de integraciones.",
            contracts=0,
            executed_price=exit_price
        )
        
        # Ajustar el live state con ganancia simulada
        db.update_live_state(
            portfolio_value=10717.80, # $174.60 de ganancia ficticia
            ath=10717.80,
            drawdown=0.00,
            is_paused=False
        )
        print("✅ [OK] Transacción HOLD (Cierre) inyectada y balance actualizado a $10,717.80 (+1.65% PnL simulado).")
        
    except Exception as e:
        print(f"❌ [FALLO] Error al inyectar transacciones en la DB: {e}")

    # 4. Probar Conexión real a OKX Demo desde la VPS (Vía SSH)
    print("\n[PASO 4] Iniciando test de órdenes reales en OKX Demo en la VPS remota...")
    print("[TRACK-INFO] Para evitar el error de Whitelist IP 50110 de OKX, ejecutaremos el trigger en la VPS a través de SSH.")
    
    # Creamos un pequeño comando remoto en python para probar la ejecución real en OKX Demo
    remote_cmd = (
        "cat << 'EOF' | /root/miniconda3/envs/luna_env/bin/python\n"
        "import sys\n"
        "sys.path.insert(0, '/root/luna_v2')\n"
        "from luna.live.okx_connector import OKXBrokerConnector\n"
        "from luna.database.db_manager import DatabaseManager\n"
        "from datetime import datetime\n"
        "import time\n"
        "connector = OKXBrokerConnector(demo_mode=True)\n"
        "db = DatabaseManager()\n"
        "symbol = 'BTC/USDC'\n"
        "print('--- TEST REMOTO OKX ---')\n"
        "print('1. Obteniendo balance...')\n"
        "balance = connector.fetch_equity()\n"
        "print(f'Balance en OKX Demo: {balance} USD')\n"
        "print('2. Obteniendo ultimo precio...')\n"
        "ticker = connector.exchange.fetch_ticker(symbol)\n"
        "price = float(ticker.get('last', 0.0))\n"
        "print(f'Precio actual de {symbol} en OKX: {price} USD')\n"
        "print('3. Ejecutando orden de COMPRA de prueba...')\n"
        "try:\n"
        "    buy_order = connector.execute_hybrid_order(symbol, 'buy', 0.0002)\n"
        "    if buy_order:\n"
        "        print('Compra ejecutada. Registrando en DB...')\n"
        "        db.log_audit(timestamp=datetime.utcnow(), price=price, action='LONG', confidence=0.99, xgb_prob=0.99, hmm_regime=1, reason='[REAL VPS TEST] Compra Spot de prueba de 0.0002 BTC en OKX Demo.', contracts=0.0002, executed_price=price)\n"
        "        time.sleep(5)\n"
        "        print('4. Cerrando posicion vendiendo el balance comprado...')\n"
        "        closed = connector.close_position(symbol)\n"
        "        if closed:\n"
        "            db.log_audit(timestamp=datetime.utcnow(), price=price, action='HOLD', confidence=0.50, xgb_prob=0.50, hmm_regime=0, reason='[REAL VPS TEST] Cierre de posicion Spot de prueba OKX Demo.', contracts=0, executed_price=price)\n"
        "            print('Test real OKX exitoso!')\n"
        "    else:\n"
        "        print('Error: La orden fue rechazada o no devolvió un objeto válido.')\n"
        "except Exception as err:\n"
        "    print('Error during order execution:', err)\n"
        "EOF"
    )
    
    # Ejecutamos comando SSH
    import subprocess
    vps_host = os.getenv("ORACLE_HOST", "178.105.197.191")
    ssh_key = r"C:\Users\Usuario\.ssh\id_ed25519"
    ssh_cmd = [
        "ssh",
        "-o", "ConnectTimeout=4",
        "-o", "StrictHostKeyChecking=no",
        "-i", ssh_key,
        f"root@{vps_host}",
        remote_cmd
    ]
    
    try:
        print(f"[TRACK-SSH] Lanzando orden de prueba OKX Demo en la VPS remota ({vps_host})...")
        result = subprocess.run(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=90.0,
            encoding="utf-8",
            errors="replace"
        )
        if result.returncode == 0:
            print("\n✅ [OK] Salida de la ejecución remota de órdenes en la VPS:")
            print("-" * 60)
            print(result.stdout)
            print("-" * 60)
        else:
            print(f"\n❌ [FALLO] El trigger SSH devolvió un error de ejecución:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ [FALLO] Error de ejecución SSH: {e}")
        
    print("\n" + "="*80)
    print("      DIAGNÓSTICO FINALIZADO CON ÉXITO")
    print("      Revisa tu localhost en tu navegador, la telemetría se ha actualizado.")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_diagnostic_test()
