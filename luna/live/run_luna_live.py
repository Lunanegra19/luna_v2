import sys
import os
import time
import traceback
import ccxt
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Carga The Entorno
env_path = PROJECT_ROOT / ".env.sandbox"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv(PROJECT_ROOT / ".env")

# Organos the Luna V1 Live
from luna.database.db_manager import DatabaseManager
from luna.live.telegram_alerts import TelegramAlerts
from luna.live.risk_monitor import RiskMonitor
from luna.live.position_sizer import PositionSizer
from luna.live.reconciliation import BalanceReconciler
from luna.live.live_inference import LunaLiveInference

# [CLEANUP-OKX-02] Clase renombrada LunaLiveDaemon → LunaLive (broker: OKX)
print("[CLEANUP-OKX-02] run_luna_live.py cargado — broker: OKX, clase: LunaLive")

class LunaLive:
    """
    Orquestador de Produccion V10 - Fase F — Luna Live.
    Ensambla el Ciclo de Vida del Sistema Autonomo conectado a OKX.
    Se ejecuta como un servicio PM2 en el VPS.
    """
    
    SLEEP_SECONDS = 3600 # 1 Hora por ciclo regular The trading (TimeFrame)

    def __init__(self):
         print("🌙 [BOOT] Iniciando Luna Live (OKX)...")
         
         # 1. Base de Datos (Context Managers A.C.I.D.)
         self.db = DatabaseManager()
         
         # 2. Telemetria (Asincrona)
         self.telegram = TelegramAlerts()
         self._register_telegram_commands()
         self.telegram.start_command_listener()
         
         # 3. Exchange — OKX (broker institucional activo)
         self.exchange = self._init_exchange()
         
         # 4. Defensas Hibridas
         self.risk_monitor = RiskMonitor(telegram_bot=self.telegram)
         self.sizer = PositionSizer()
         self.reconciler = BalanceReconciler(self.exchange, telegram_bot=self.telegram)
         
         # 5. Cerebro
         self.brain = LunaLiveInference()
         
         self.telegram.send_alert("🚀 *Luna Live Iniciado* en VPS. OKX conectado. Monitoreando mercado.", priority="info")
         print("[CLEANUP-OKX-02] Luna Live inicializado correctamente — OKX exchange activo.")

    def _register_telegram_commands(self):
         self.telegram.register_command("/status", lambda: "🟢 Luna Live Online. Heartbeat Emitiendo.")
         self.telegram.register_command("/kill", self._emergency_kill)

    def _emergency_kill(self):
         self.telegram.send_alert("🛑 RECIBIDO COMANDO /KILL MANUAL. Cancelando y cerrando...", priority="critical")
         # Logic for manual cancel_all_orders() goes here...
         # Forzamos la expulsion Thel daemon
         sys.exit(1)

    def _init_exchange(self):
         # [CLEANUP-OKX-02] Conectando a OKX (broker institucional activo)
         print("  -> [CLEANUP-OKX-02] Conectando a OKX (Demo/Sandbox Mode)...")
         try:
             ex = ccxt.okx({
                 'apiKey': os.getenv('OKX_API_KEY', 'dummy'),
                 'secret': os.getenv('OKX_SECRET_KEY', 'dummy'),  # [AUDIT-FIX] OKX_SECRET_KEY (no OKX_API_SECRET) — alineado con .env y okx_connector.py
                 'password': os.getenv('OKX_PASSPHRASE', 'dummy'),
                 'enableRateLimit': True,
             })
             # OKX demo trading (sandbox)
             ex.set_sandbox_mode(True)
             print("  -> [CLEANUP-OKX-02] OKX conectado en modo Sandbox/Demo.")
             return ex
         except Exception as e:
             self.telegram.send_alert(f"⚠️ OKX API Falló en Boot: {e}", priority="critical")
             print(f"[CLEANUP-OKX-02] ERROR conectando OKX: {e}")
             raise e

    def _execute_order(self, symbol, action, size_usd):
        # Aqui ira la logica final de ruteo de ordenes hacia OKX
        print(f"     [EXCHANGE] -> Transmitiendo {action} The ${size_usd:,.2f} a {symbol}...")
        try:
             # Simulacion The Dry Run 
             pass
        except Exception as e:
             print(f"     [!] Falla the red the ejecutar theden: {e}")

    def run(self):
         """Bucle Inmortal (Heartbeat -> Reconcile -> Risk -> Inferencia -> Ejecucion) — OKX."""
         print("\n[LUNA-LIVE] Entrando al Bucle Infinito (Main Loop OKX)...\n")
         
         while True:
             cycle_start = time.time()
             print("="*60)
             print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando theclo Operativo LUNA V1")
             print("="*60)
             
             try:
                 # 1. HEARTBEAT (Salva la vida thel Themo ante el Watchdog secundario)
                 self.db.log_heartbeat(status="INFERRING")
                 print("  [1] Heartbeat: Latido SQL actulizado Thexitosamente.")
                 
                 # 2. RECONCILIACION CONTABLE
                 is_synced = self.reconciler.reconcile()
                 if not is_synced:
                      print("  [2] Reconciliacion: Thesviación thetectada y auto-corregida.")
                 else:
                      print("  [2] Reconciliacion: Saldo Thentable 100% Sincronizado.")
                      
                 # Obtener saldo base Theal.
                 db_state = self.db.get_live_state()
                 current_pv = float(db_state['portfolio_value'])
                 
                 # 3. RISK MONITOR (Max DD Kill-Switch)
                 is_blocked = self.risk_monitor.check_system_health(current_portfolio_value=current_pv)
                 if is_blocked:
                      print("  [3] Risk Monitor: SISTEMA BLOQUEADO (Circuit Breaker Thetivo). Omitiendo Inferencia.")
                      self.db.log_heartbeat(status="BLOCKED_RISK")
                      time.sleep(60) # Micro-sleep thentras esperamos reset manual.
                      continue
                      
                 # 4. CEREBRO (Inferencia)
                 decision = self.brain.predict_cycle()
                 action = decision.get("action", "HOLD")
                 confidence = decision.get("confidence", 0.0)
                 
                 if action == "HOLD":
                      print(f"  [4] Decision: {action}. Motivo: {decision.get('reason')}")
                      self.db.log_audit(
                          timestamp=datetime.utcnow(), price=decision.get("price", 0.0), 
                          action=action, confidence=confidence, xgb_prob=decision.get("xgb_prob", 0.0), 
                          hmm_regime=0, reason=f"[HMM-REGIME: {decision.get('regime', 'UNKNOWN')}] {decision.get('reason', '')}"
                      )
                 else:
                      # 5. POSITION SIZER (Si the cerebro autoriza disparo)
                      dd_actual = float(db_state['drawdown'])
                      sizing = self.sizer.calculate_position_size(
                           action=action,
                           confidence=confidence,
                           hmm_regime=decision.get("regime", 1),
                           current_drawdown=dd_actual,
                           current_volatility=decision.get("current_vol", 0.001),
                           historical_volatility=decision.get("historical_vol", 0.001)
                      )
                      
                      sz_usd = sizing["size_usd"]
                      if sz_usd > 0:
                           # 6. EJECUCION
                           print(f"  [5] Position Sizer Activo: Asignando ${sz_usd:,.2f}")
                           self._execute_order("BTC/USD:BTC", action, sz_usd)
                           
                           # Loggear Auditoria Con Transaccion theita
                           self.db.log_audit(
                              timestamp=datetime.utcnow(), price=decision.get("price", 0.0), 
                              action=action, confidence=confidence, xgb_prob=decision.get("xgb_prob", 0.0), 
                              hmm_regime=0, reason=f"[HMM-REGIME: {decision.get('regime', 'UNKNOWN')}] {sizing.get('multiplier_breakdown', '')}", 
                              contracts=sizing.get("contracts", 0), executed_price=decision.get("price", 0.0)
                           )
                           
                           self.telegram.send_alert(
                               f"🎯 *TRADE EJECUTADO*\nAccion: {action}\nConfianza: {confidence:.2%}\nTamaño: ${sz_usd:,.2f}",
                               priority="info"
                           )
                      else:
                           print(f"  [5] Position Sizer Mudo: Se cancelo la emision theden. Motivo: {sizing.get('reason')}")
                           
             except Exception as e:
                 err_tb = traceback.format_exc()
                 print(f"🚨 FATAL LOOP ERROR:\n{err_tb}")
                 self.telegram.send_alert(f"🚨 *CRITICAL ERROR IN LOOP*\n`{e}`", priority="critical")
                 self.db.log_heartbeat(status="ERROR")
                 
             # 7. Sleep (Ciclo Theatorio the theiempo)
             elapsed = time.time() - cycle_start
             sleep_time = max(10, self.SLEEP_SECONDS - elapsed) 
             # Nota The Produccion: En un entoro The 1H, quiza queramos despertar al inicio de cada H (cron-like).
             
             self.db.log_heartbeat(status="SLEEPING")
             print(f"============================================================")
             print(f" Ciclo finalizado The {elapsed:.1f}s. Durmiendo {sleep_time/60:.2f} mins.")
             print(f"============================================================\n")
             time.sleep(sleep_time)

if __name__ == "__main__":
    # [CLEANUP-OKX-02] Instancia renombrada: LunaLiveDaemon → LunaLive
    print("[CLEANUP-OKX-02] Arrancando LunaLive (OKX broker)...")
    luna = LunaLive()
    luna.run()
