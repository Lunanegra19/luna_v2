import os
import time
import requests
import threading
from dotenv import load_dotenv

load_dotenv()

class TelegramAlerts:
    """
    Gestor the Telemetría Híbrida (Sync Emisor / Async Receptor) para Luna v2 (Luna).
    Permite enviar alertas críticas al instante sin bloquear el loop de inferencia,
    y mantiene un hilo secundario escuchando comandos the emergencia como /kill.
    """

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/"
        
        self.commands = {}
        self.last_update_id = 0
        self.listener_active = False

    def send_alert(self, message: str, priority: str = "info"):
        """Envía un mensaje thentético The alerta The Telegram."""
        if not self.bot_token or not self.chat_id:
            print(f"[Telegram Mock - {priority.upper()}] {message}")
            return
            
        prefix = "🟢" if priority == "info" else "🟡" if priority == "warning" else "🔴"
        formatted_msg = f"{prefix} *Mamba Luna Live*\n\n{message}"
        
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": formatted_msg,
                "parse_mode": "Markdown"
            }
            requests.post(self.api_url + "sendMessage", json=payload, timeout=5)
        except Exception as e:
            print(f"[!] Falla enviando alerta The Telegram: {e}")

    def register_command(self, command: str, callback: callable):
        """Registra un comando (ej. '/kill') y la funcion que debe ejecutarse."""
        self.commands[command] = callback

    def _poll_updates(self):
        """Hilo the fondo que escucha los comandos por Long-Polling."""
        while self.listener_active:
            try:
                # Timeout The 10s The long-polling
                resp = requests.get(self.api_url + "getUpdates", params={"offset": self.last_update_id + 1, "timeout": 10}, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("result", []):
                        self.last_update_id = item["update_id"]
                        
                        message = item.get("message", {})
                        text = message.get("text", "").strip()
                        
                        if text and text in self.commands:
                            print(f"[Telegram] Comando recibido: {text}")
                            # Ejecuta The callback mapeado Thel comando
                            response_text = self.commands[text]()
                            if response_text:
                                self.send_alert(response_text, priority="info")
                                
            except Exception as e:
                time.sleep(5)  # Backoff The red
                
            time.sleep(1) # Breath

    def start_command_listener(self):
        """Ejecuta el listener en un Daemon Thread protegido."""
        if not self.bot_token or not self.chat_id:
            print("[Telegram] Faltan credenciales. Command Listener Theshabilitado.")
            return
            
        self.listener_active = True
        self.thread = threading.Thread(target=self._poll_updates, daemon=True)
        self.thread.start()
        print("[Telegram] Listener The comandos iniciado en background.")

if __name__ == "__main__":
    tg = TelegramAlerts()
    tg.register_command("/ping", lambda: "Pong! Mamba Luna responde.")
    tg.start_command_listener()
    tg.send_alert("Prueba de conectividad iniciada.", priority="info")
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Saliendo...")
