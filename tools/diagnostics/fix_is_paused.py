"""
[FIX-IS-PAUSED] Script de emergencia para limpiar el flag is_paused en PostgreSQL.
Ejecutar en la VPS: python fix_is_paused.py
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv()
print("[FIX-IS-PAUSED] Iniciando script de limpieza de flag is_paused...")

sys.path.insert(0, "/root/luna_v2")
from luna.database.db_manager import DatabaseManager

db = DatabaseManager()
state = db.get_live_state()

print(f"[FIX-IS-PAUSED] Estado actual en DB: {dict(state) if state else 'None'}")

if state:
    portfolio_value = float(state['portfolio_value'])
    ath = float(state['ath'])
    drawdown = float(state['drawdown'])
    was_paused = bool(state['is_paused'])
    
    print(f"[FIX-IS-PAUSED] is_paused actual = {was_paused}")
    
    if was_paused:
        db.update_live_state(
            portfolio_value=portfolio_value,
            ath=ath,
            drawdown=drawdown,
            is_paused=False
        )
        state2 = db.get_live_state()
        print(f"[FIX-IS-PAUSED] Estado tras fix: {dict(state2) if state2 else 'None'}")
        print("[FIX-IS-PAUSED] [OK] Flag is_paused limpiado exitosamente. El bot inferira en el proximo ciclo.")
    else:
        print("[FIX-IS-PAUSED] El flag ya estaba en False. No se requiere accion.")
else:
    print("[FIX-IS-PAUSED] [ERROR] No se pudo leer live_state de la DB.")
    sys.exit(1)
