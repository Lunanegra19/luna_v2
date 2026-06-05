import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load environment variables
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

from luna.live.telegram_alerts import TelegramAlerts

def run():
    print("="*60)
    print("🌙 [LUNA V2 DIAGNOSTICS] PROBANDO TELEGRAM ALERTS")
    print("="*60)
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    print(f"Token: {token[:10]}... | Chat ID: {chat_id}")
    
    tg = TelegramAlerts()
    print("Enviando mensaje de prueba...")
    tg.send_alert("🔔 *Prueba de conectividad desde el VPS!* \n\nEl sistema de alertas de Telegram de Luna V2 está funcionando correctamente en el VPS. 🚀", priority="info")
    print("¡Mensaje enviado!")

if __name__ == "__main__":
    run()
