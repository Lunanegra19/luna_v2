#!/usr/bin/env python
"""
watchdog_rescue.py
==================
[LUNA-V2-WATCHDOG] Script automático de autocuración y rescate del daemon.
Escanea el estado del proceso en PM2 y el latido de vida (heartbeat) en PostgreSQL.
Si se detecta congelamiento o inactividad prolongada, reinicia el daemon y notifica por Telegram.
"""

import sys
import os
import subprocess
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Fix python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Env
from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env.sandbox"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv(PROJECT_ROOT / ".env")

from luna.database.db_manager import DatabaseManager
from luna.live.telegram_alerts import TelegramAlerts

def check_pm2_process_status(process_name="luna-v2-live-demo"):
    """
    Consulta a PM2 el estado del proceso.
    Retorna (is_online, memory_mb, cpu_pct)
    """
    try:
        # Run pm2 jlist to get process list in JSON format
        res = subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            for proc in data:
                if proc.get("name") == process_name:
                    status = proc.get("pm2_env", {}).get("status")
                    monit = proc.get("monit", {})
                    mem = monit.get("memory", 0) / (1024 * 1024) # Convert to MB
                    cpu = monit.get("cpu", 0)
                    is_online = (status == "online")
                    return is_online, mem, cpu, status
            return False, 0.0, 0.0, "NOT_FOUND"
    except Exception as e:
        print(f"[WATCHDOG-WARN] Error consultando PM2 jlist: {e}")
    return True, 0.0, 0.0, "UNKNOWN" # Fallback a asusmir online en caso de fallo de comando para evitar bucle de falsos positivos

def main():
    print("[LUNA-V2-WATCHDOG-START] Iniciando auditoría activa de salud del Daemon...")
    db = DatabaseManager()
    telegram = TelegramAlerts()
    
    # 1. Auditar latido en PostgreSQL
    last_hb = db.get_last_heartbeat()
    now_utc = datetime.utcnow()
    
    print(f"[WATCHDOG-AUDIT] Hora Actual UTC: {now_utc.strftime('%H:%M:%S')}")
    if last_hb:
        delta = now_utc - last_hb
        delta_seconds = delta.total_seconds()
        print(f"[WATCHDOG-AUDIT] Último Heartbeat DB: {last_hb.strftime('%Y-%m-%d %H:%M:%S')} (Hace {delta_seconds:.1f}s)")
    else:
        delta_seconds = 99999
        print("[WATCHDOG-AUDIT] ¡No se encontró registro de Heartbeat en DB!")

    # 2. Auditar estado en PM2
    is_online, mem, cpu, pm2_status = check_pm2_process_status("luna-v2-live-demo")
    print(f"[WATCHDOG-AUDIT] Estado en PM2: {pm2_status} | CPU: {cpu}% | RAM: {mem:.2f}MB")

    # 3. Evaluar criterios de rescate
    # [FIX-WATCHDOG-THRESHOLD-2026-05-26] Umbral aumentado de 420s (7min) a 4500s (75min).
    # El live trader duerme hasta 60 minutos entre ciclos operativos. El umbral antiguo
    # de 7 minutos causaba 421+ reinicios innecesarios con KeyboardInterrupt.
    # Nuevo umbral: 75 minutos > 60 min (ciclo) + 10 min (boot/carga de modelos) + 5 min (margen).
    STALE_THRESHOLD_SECONDS = 4500  # 75 minutos
    print(f"[WATCHDOG-AUDIT] Threshold de inactividad configurado: {STALE_THRESHOLD_SECONDS}s ({STALE_THRESHOLD_SECONDS/60:.0f} min)")

    # Solo marcar como stale si el proceso está online (evitar falsos positivos durante reinicio)
    # Si el proceso NO está online en PM2, es un crash real → siempre reiniciar
    # Si el proceso SÍ está online, solo reiniciar si el heartbeat es muy antiguo (> 75 min)
    is_stale = is_online and (delta_seconds > STALE_THRESHOLD_SECONDS)
    should_restart = not is_online or is_stale
    print(f"[WATCHDOG-AUDIT] Evaluación: is_online={is_online} | delta={delta_seconds:.1f}s | is_stale={is_stale} | should_restart={should_restart}")
    
    if should_restart:
        reason = f"Proceso PM2 en estado '{pm2_status}'" if not is_online else f"Latido DB inactivo por {delta_seconds/60:.1f} minutos"
        print(f"[WATCHDOG-CRITICAL] ¡Alerta de cuelgue detectada! Motivo: {reason}")
        
        # Enviar alerta previa a Telegram
        alert_msg = (
            f"🚨 *Fallo de Latido en Luna V2 Live Demo*\n"
            f"• Motivo: {reason}\n"
            f"• PM2 Status: {pm2_status}\n"
            f"• Acción: Ejecutando reinicio de emergencia en caliente..."
        )
        telegram.send_alert(alert_msg, priority="critical")
        
        # Reiniciar a través de PM2
        try:
            print("[WATCHDOG-ACTION] Ejecutando: pm2 restart luna-v2-live-demo")
            subprocess.run(["pm2", "restart", "luna-v2-live-demo"], check=True, timeout=10)
            print("[WATCHDOG-ACTION] Reinicio ejecutado con éxito en PM2.")
            
            # Notificar éxito del reinicio
            time.sleep(3) # Esperar a que levante
            success_msg = "🟢 *Daemon Autocurado y Revivido con éxito* mediante Watchdog de Rescate PM2."
            telegram.send_alert(success_msg, priority="info")
        except Exception as err:
            err_msg = f"❌ *Fallo al revivir Daemon* por Watchdog: `{err}`"
            print(f"[WATCHDOG-FATAL] Fallo al reiniciar: {err}")
            telegram.send_alert(err_msg, priority="critical")
    else:
        print("[WATCHDOG-AUDIT-OK] El Daemon se encuentra en perfectas condiciones y latiendo con fuerza.")

if __name__ == "__main__":
    main()
