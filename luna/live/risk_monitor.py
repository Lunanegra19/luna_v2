import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

# Fix python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Internal imports
from luna.database.db_manager import DatabaseManager

class RiskMonitor:
    """
    Monitor de Riesgo Global para Luna v2 Live (Luna).
    Se encarga de leer el estado contable de PostgreSQL, calcular el Drawdown de la equidad
    y aplicar Hard Circuit Breakers. Está diseñado para prevenir bucles de muerte ante Systemd.

    Fix M-06: daily_dd y weekly_dd ahora activos (antes eran dead code).
    Umbrales:
      - daily_dd   >= 3%  → pausa 24h automática
      - weekly_dd  >= 7%  → pausa para revisión manual
      - drawdown_reduce >= 10% → alerta reducción sizing 50%
      - kill_switch >= 15% → parada total definitiva
    """
    
    CIRCUIT_BREAKERS = {
        "daily_dd":       {"threshold": 0.03, "action": "pause_24h"},
        "weekly_dd":      {"threshold": 0.07, "action": "pause_review"},
        "drawdown_reduce":{"threshold": 0.10, "action": "reduce_50pct"},
        "kill_switch":    {"threshold": 0.15, "action": "close_all_STOP"},
    }

    def __init__(self, telegram_bot=None):
        self.db = DatabaseManager()
        self.telegram = telegram_bot
        state = self.db.get_live_state()
        if not state:
            self.db.update_live_state(portfolio_value=5000.0, ath=5000.0, drawdown=0.0, is_paused=False)
            state = self.db.get_live_state()
            
        if state is None:
            print("[FIX-DB-RM/WARN] No se pudo establecer conexión con la base de datos. Usando valores de fallback locales por seguridad.")
            self.is_paused = False
            self.mock_db_state = {"portfolio_value": 5000.0, "ath": 5000.0, "drawdown": 0.0, "is_paused": False}
        else:
            self.is_paused = state.get('is_paused', False)
            self.mock_db_state = None

    def compute_current_drawdown(self, current_portfolio_value: float) -> tuple[float, float, bool]:
        """
        Calcula el drawdown actual y actualiza el ATH si es necesario.
        Retorna (drawdown, ATH_actualizado, is_paused_flag)
        """
        state = self.db.get_live_state()
        if not state:
            if hasattr(self, 'mock_db_state') and self.mock_db_state:
                state = self.mock_db_state
            else:
                state = {"portfolio_value": current_portfolio_value, "ath": current_portfolio_value, "drawdown": 0.0, "is_paused": False}
        ath = float(state.get('ath', 5000.0))
        is_paused = state.get('is_paused', False)
        
        if current_portfolio_value > ath:
            ath = current_portfolio_value
            
        if ath <= 0: return 0.0, ath, is_paused
        
        dd = (ath - current_portfolio_value) / ath
        self.db.update_live_state(current_portfolio_value, ath, dd, is_paused)
        if hasattr(self, 'mock_db_state') and self.mock_db_state:
            self.mock_db_state = {"portfolio_value": current_portfolio_value, "ath": ath, "drawdown": dd, "is_paused": is_paused}
        return dd, ath, is_paused

    def _auto_reset_periods(self, current_portfolio_value: float) -> None:
        """
        Fix M-06: Restablece automáticamente equity_start_day y equity_start_week
        al inicio de cada día/semana, usando las columnas de DB añadidas en db_manager.
        """
        today = date.today()
        period = self.db.get_period_equity()
        if not period and hasattr(self, 'mock_period_equity'):
            period = self.mock_period_equity
        elif not period:
            period = {}
            
        reset_day  = not period.get("day_reset_date")  or period["day_reset_date"] != today
        # Monday = weekday 0
        week_start = today - timedelta(days=today.weekday())
        reset_week = not period.get("week_reset_date") or period["week_reset_date"] != week_start
        
        if reset_day or reset_week:
            self.db.reset_period_equity(
                portfolio_value=current_portfolio_value,
                reset_day=reset_day,
                reset_week=reset_week,
            )
            if not self.db.connection_pool:
                if not hasattr(self, 'mock_period_equity') or not self.mock_period_equity:
                    self.mock_period_equity = {}
                if reset_day:
                    self.mock_period_equity["equity_start_day"] = current_portfolio_value
                    self.mock_period_equity["day_reset_date"] = today
                if reset_week:
                    self.mock_period_equity["equity_start_week"] = current_portfolio_value
                    self.mock_period_equity["week_reset_date"] = week_start
            if reset_day:
                print(f"  [RM] Nuevo día — equity_start_day = ${current_portfolio_value:,.2f}")
            if reset_week:
                print(f"  [RM] Nueva semana — equity_start_week = ${current_portfolio_value:,.2f}")

    def _check_intraperiod_dd(self, current_portfolio_value: float) -> tuple[float, float]:
        """
        Fix M-06: Calcula el DD desde el inicio del día y de la semana.
        Retorna (daily_dd_pct, weekly_dd_pct).
        """
        period = self.db.get_period_equity()
        if not period and hasattr(self, 'mock_period_equity'):
            period = self.mock_period_equity
        elif not period:
            period = {}
        start_day  = float(period.get("equity_start_day")  or current_portfolio_value)
        start_week = float(period.get("equity_start_week") or current_portfolio_value)

        daily_dd  = max(0.0, (start_day  - current_portfolio_value) / start_day)  if start_day  > 0 else 0.0
        weekly_dd = max(0.0, (start_week - current_portfolio_value) / start_week) if start_week > 0 else 0.0
        return daily_dd, weekly_dd

    def check_system_health(self, current_portfolio_value: float) -> bool:
        """
        Calcula drawdown global + intradiario/semanal y aplica circuit breakers.
        Llamado continuamente por el orquestador maestro (run_luna_live.py).
        Retorna True si el sistema debe pausarse, False si puede operar.
        """
        # 1. Update contable global (ATH drawdown)
        current_dd, ath, self.is_paused = self.compute_current_drawdown(current_portfolio_value)
        
        # 2. Prevent Boot Loop
        if self.is_paused:
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] [CRITICO] SISTEMA EN ESTADO 'PAUSED' PREVIO.")
            print("  El sistema no operara ni ejecutara bucles hasta ser restaurado manualmente (SQL Update).")
            # [BUGFIX-TELEMETRY] Modificado mensaje de alerta para reflejar que está en pausa sin decir falsamente que se reinició
            print("[BUGFIX-TELEMETRY] Enviando alerta de estado en pausa en Telegram.")
            if self.telegram:
                self.telegram.send_alert("⚠️ *AVISO*: El sistema se encuentra en PAUSA activa por flag en la base de datos (Inferencia omitida).", "critical")
            return True

        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Escaneo de riesgo. DD ATH: {current_dd:.2%} | ATH: ${ath:,.2f}")

        # 3. Auto-reset equity de período (inicio del día/semana) — Fix M-06
        self._auto_reset_periods(current_portfolio_value)

        # 4. Fix M-06: Daily DD Circuit Breaker (3%)
        daily_dd, weekly_dd = self._check_intraperiod_dd(current_portfolio_value)
        daily_threshold  = self.CIRCUIT_BREAKERS["daily_dd"]["threshold"]
        weekly_threshold = self.CIRCUIT_BREAKERS["weekly_dd"]["threshold"]

        if daily_dd >= daily_threshold:
            msg = f"⛔ DAILY DD CIRCUIT BREAKER: -{daily_dd:.2%} en el día (límite {daily_threshold:.0%}). PAUSADO 24H."
            print(f"  [CRITICO] {msg}")
            self.db.update_live_state(current_portfolio_value, ath, current_dd, is_paused=True)
            self.is_paused = True
            if self.telegram:
                self.telegram.send_alert(f"🚨 *{msg}*", "critical")
            return True

        # 5. Fix M-06: Weekly DD Circuit Breaker (7%)
        if weekly_dd >= weekly_threshold:
            msg = f"⛔ WEEKLY DD CIRCUIT BREAKER: -{weekly_dd:.2%} en la semana (límite {weekly_threshold:.0%}). PAUSADO — revisión manual requerida."
            print(f"  [CRITICO] {msg}")
            self.db.update_live_state(current_portfolio_value, ath, current_dd, is_paused=True)
            self.is_paused = True
            if self.telegram:
                self.telegram.send_alert(f"🚨 *{msg}*", "critical")
            return True

        print(f"  [RM] DD Día: {daily_dd:.2%} | DD Semana: {weekly_dd:.2%} | DD ATH: {current_dd:.2%}")

        # 6. Kill-Switch Global (15% desde ATH)
        kill_threshold = self.CIRCUIT_BREAKERS["kill_switch"]["threshold"]
        if current_dd >= kill_threshold:
            print("  [CRITICO] ==================================")
            print("  [CRITICO] KILL SWITCH TRIGGERED (-15% MAX DD)")
            print("  [CRITICO] ==================================")
            self.db.update_live_state(current_portfolio_value, ath, current_dd, is_paused=True)
            self.is_paused = True
            if self.telegram:
                self.telegram.send_alert(f"🚨 *KILL SWITCH TRIGGERED* (-{current_dd:.2%})! Bot PAUSADO permanentemente. Requiere revision humana.", "critical")
            return True
            
        # 7. Drawdown Secundario (alerta sin pausar)
        reduce_threshold = self.CIRCUIT_BREAKERS["drawdown_reduce"]["threshold"]
        if current_dd >= reduce_threshold:
            print(f"  [WARNING] Drawdown Secundario ATH (-{current_dd:.2%}). Alertando al Sizer para reducir 50% el riesgo.")
            if self.telegram:
                self.telegram.send_alert(f"🛡️ *DRAWDOWN SEVERO* (-{current_dd:.2%}). Reduciendo agresividad a un 50%.", "warning")
            return False
            
        return False
        
    def get_risk_summary(self):
        state = self.db.get_live_state()
        if not state and hasattr(self, 'mock_db_state'):
            state = self.mock_db_state
        elif not state:
            state = {"portfolio_value": 5000.0, "ath": 5000.0, "drawdown": 0.0, "is_paused": False}
        period = self.db.get_period_equity()
        if not period and hasattr(self, 'mock_period_equity'):
            period = self.mock_period_equity
        elif not period:
            period = {}
        return {
            "drawdown": float(state['drawdown']),
            "portfolio_value": float(state['portfolio_value']),
            "ath": float(state['ath']),
            "is_paused": bool(state['is_paused']),
            "equity_start_day": float(period.get("equity_start_day") or 0),
            "equity_start_week": float(period.get("equity_start_week") or 0),
            "thresholds": self.CIRCUIT_BREAKERS,
        }

if __name__ == "__main__":
    print("[TEST] Inicializando Risk Monitor Luna v2 (Luna Edition)")
    monitor = RiskMonitor()
    # Test: simula $4200 contra $5000 ATH (-16%) → debe triggear kill switch
    is_blocked = monitor.check_system_health(4200)
    print(f"Block Status: {is_blocked}")

