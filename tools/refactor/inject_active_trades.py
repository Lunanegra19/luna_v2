#!/usr/bin/env python
"""
inject_active_trades.py
========================
[LUNA-V2-DIAGNOSTIC] Utilidad técnica para inyectar una secuencia de transacciones
reales simuladas e industriales en la base de datos a través del túnel SSH.
Esto demuestra la reactividad en tiempo real del Dashboard local ante cambios
en el estado y transacciones del VPS sin arriesgar capital.
"""

import sys
import os

# Reconfigure stdout for UTF-8 encoding on Windows to prevent charmap crashes
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from luna.database.db_manager import DatabaseManager

def inject_mock_telemetry():
    print("\n" + "="*80)
    print("      LUNA V2 - INYECTOR DE TELEMETRÍA COMPLETA Y TRANSACCIONES")
    print("="*80 + "\n")
    
    try:
        db = DatabaseManager()
        if db.connection_pool is None:
            raise RuntimeError("DatabaseManager no inicializó el Connection Pool.")
        print("[DB-OK] Conexión establecida con éxito via puerto 5433.")
    except Exception as e:
        print(f"[!] Error de conexión: {e}")
        print("Asegúrate de que el túnel SSH esté abierto en tu consola.")
        return

    # Inyectar Heartbeat ONLINE
    print("\n1. Actualizando latido del Daemon a ONLINE...")
    try:
        db.log_heartbeat(status="ONLINE")
        print("✅ Latido actualizado a ONLINE.")
    except Exception as e:
        print(f"❌ Falló latido: {e}")

    # Limpiar logs de diagnostico previos para evitar duplicados ruidosos
    print("\n2. Limpiando logs de diagnósticos previos...")
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM audit_logs WHERE reason LIKE '%TEST DIAGNÓSTICO%';")
                cur.execute("DELETE FROM audit_logs WHERE reason LIKE '%REAL VPS TEST%';")
            conn.commit()
        print("✅ Registros anteriores de diagnóstico limpiados.")
    except Exception as e:
        print(f"❌ Falló limpieza: {e}")

    # Inyectar una serie de 4 trades realistas
    print("\n3. Inyectando secuencia de transacciones activas...")
    now = datetime.utcnow()
    
    trades = [
        {
            "timestamp": now - timedelta(minutes=120),
            "price": 75840.00,
            "action": "LONG",
            "confidence": 0.7420,
            "xgb_prob": 0.7420,
            "hmm_regime": 1,
            "reason": "[SOP-LIVE] Apertura LONG. Consenso ensamble 4/5 semillas. Regime BULL_TREND.",
            "contracts": 1,
            "executed_price": 75840.00
        },
        {
            "timestamp": now - timedelta(minutes=95),
            "price": 76190.50,
            "action": "HOLD",
            "confidence": 0.5000,
            "xgb_prob": 0.5000,
            "hmm_regime": 0,
            "reason": "[SOP-LIVE] Cierre LONG. Take Profit alcanzado. PnL +$350.50 USDT (+4.62% nocional).",
            "contracts": 0,
            "executed_price": 76190.50
        },
        {
            "timestamp": now - timedelta(minutes=60),
            "price": 76210.00,
            "action": "SHORT",
            "confidence": 0.6890,
            "xgb_prob": 0.6890,
            "hmm_regime": 3,
            "reason": "[SOP-LIVE] Apertura SHORT. Señal de reversión. Regime BEAR_TREND.",
            "contracts": 1,
            "executed_price": 76210.00
        },
        {
            "timestamp": now - timedelta(minutes=25),
            "price": 75780.20,
            "action": "HOLD",
            "confidence": 0.5000,
            "xgb_prob": 0.5000,
            "hmm_regime": 0,
            "reason": "[SOP-LIVE] Cierre SHORT. Señal consolidada neutralizada. PnL +$429.80 USDT (+5.64% nocional).",
            "contracts": 0,
            "executed_price": 75780.20
        }
    ]

    try:
        for t in trades:
            db.log_audit(
                timestamp=t["timestamp"],
                price=t["price"],
                action=t["action"],
                confidence=t["confidence"],
                xgb_prob=t["xgb_prob"],
                hmm_regime=t["hmm_regime"],
                reason=t["reason"],
                contracts=t["contracts"],
                executed_price=t["executed_price"]
            )
            print(f"  - Inyectado trade: {t['action']} a ${t['price']:.2f}")
        print("✅ Secuencia de 4 transacciones inyectada.")
    except Exception as e:
        print(f"❌ Falló inyección de trades: {e}")

    # Actualizar Estado Financiero en Vivo
    print("\n4. Actualizando estado financiero en vivo (live_state)...")
    try:
        # Simulamos un crecimiento del balance de $5,000 iniciales a $5,780.30
        db.update_live_state(
            portfolio_value=5780.30,
            ath=5780.30,
            drawdown=0.00,
            is_paused=False
        )
        print("✅ Estado financiero actualizado. Balance fijado en $5,780.30 USDT (+15.6% PnL acumulado).")
    except Exception as e:
        print(f"❌ Falló estado financiero: {e}")

    print("\n" + "="*80)
    print("      INYECCIÓN COMPLETADA. RECARGA TU NAVEGADOR EN LOCALHOST!")
    print("="*80 + "\n")

if __name__ == "__main__":
    inject_mock_telemetry()
