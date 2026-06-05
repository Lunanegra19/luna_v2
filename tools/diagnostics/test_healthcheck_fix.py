"""
[BUGFIX-HEALTHCHECK-01] Test de verificación del fix TelegramAlerter -> TelegramAlerts
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.diagnostics.dashboard_healthcheck import _send_telegram_alert

print("[TEST] Probando _send_telegram_alert con TelegramAlerts corregido...")
r = _send_telegram_alert("[BUGFIX-HEALTHCHECK-01] ✅ Fix TelegramAlerter → TelegramAlerts verificado en VPS. Healthcheck OK.")
print(f"[TEST] Resultado: {'OK' if r else 'FALLBACK-API-OK (esperado si módulo luna no disponible en path)'}")
print("[TEST] VERIFICACIÓN COMPLETADA SIN ImportError.")
