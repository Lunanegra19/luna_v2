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

# Carga entorno
env_path = PROJECT_ROOT / ".env.sandbox"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv(PROJECT_ROOT / ".env")

# Modulos de Luna V2 Live
from luna.database.db_manager import DatabaseManager
from luna.live.telegram_alerts import TelegramAlerts
from luna.live.risk_monitor import RiskMonitor
from luna.live.position_sizer import PositionSizer
from luna.live.reconciliation import BalanceReconciler
from luna.live.live_inference import LunaLiveInference
# [P3+P2-2026-06-12] Conector OKX real con soporte Futuros Perpetuos y ejecucion hibrida Maker
from luna.live.okx_connector import OKXBrokerConnector

print("[P3+P2-BOOT] run_luna_live.py cargado — OKXBrokerConnector activo (Futuros Perps + Hybrid Maker)")

class LunaLive:
    """
    Orquestador de Produccion V10 - Fase F -- Luna Live.
    Ensambla el Ciclo de Vida del Sistema Autonomo conectado a OKX.
    Se ejecuta como un servicio PM2 en el VPS.
    """
    
    SLEEP_SECONDS = 3600  # 1 Hora por ciclo regular de trading (TimeFrame)

    def __init__(self):
        print("[BOOT] Iniciando Luna Live (OKX Futuros Perpetuos + Hybrid Maker)...")

        # 1. Base de Datos (Context Managers A.C.I.D.)
        self.db = DatabaseManager()

        # 2. Telemetria (Asincrona)
        self.telegram = TelegramAlerts()
        self._register_telegram_commands()
        self.telegram.start_command_listener()

        # 3. [P3+P2-2026-06-12] Conector OKX REAL — Futuros Perpetuos + Hybrid Maker
        #    OKXBrokerConnector carga symbol e instrument_type desde settings.yaml (No-Fallback)
        self.connector = OKXBrokerConnector(demo_mode=True)
        self.trading_symbol = self.connector.get_trading_symbol()
        print(f"[P3+P2-BOOT] Conector activo. Simbolo: {self.trading_symbol}")

        # 4. Defensas Hibridas
        self.risk_monitor = RiskMonitor(telegram_bot=self.telegram)
        self.sizer = PositionSizer()
        self.reconciler = BalanceReconciler(self.connector.exchange, telegram_bot=self.telegram)

        # 5. Cerebro
        self.brain = LunaLiveInference()

        self.telegram.send_alert(
            f"[BOOT] Luna Live activo. OKX Futuros ({self.trading_symbol}). Hybrid Maker ON.",
            priority="info"
        )
        print("[P3+P2-BOOT] Luna Live inicializado correctamente.")

    def _register_telegram_commands(self):
        self.telegram.register_command("/status", lambda: "Luna Live Online. Heartbeat Emitiendo.")
        self.telegram.register_command("/kill", self._emergency_kill)

    def _emergency_kill(self):
        self.telegram.send_alert("[KILL] RECIBIDO COMANDO /KILL MANUAL. Cancelando y cerrando...", priority="critical")
        sys.exit(1)

    def _execute_order(self, action: str, size_usd: float) -> dict:
        """
        [P3+P2-2026-06-12] Ejecuta una orden REAL en OKX usando OKXBrokerConnector.
        - Instrumento: leido de settings.yaml (Futuros Perpetuos BTC/USDT:USDT)
        - Ejecucion: Hybrid Maker (Limit con Order Chasing + Fallback Market)
        - Gestion de posicion: cierra posicion inversa antes de abrir
        action: 'LONG' | 'CLOSE' | 'HOLD'
        size_usd: capital en USD a asignar a la posicion
        """
        symbol = self.trading_symbol
        print(f"[P3+P2-EXEC] _execute_order llamado: action={action} size_usd={size_usd:.2f} symbol={symbol}")

        # Obtener precio actual para convertir USD a contratos
        try:
            ticker = self.connector.exchange.fetch_ticker(symbol)
            price = float(ticker.get('last', ticker.get('close', 0.0)) or 0.0)
            if price <= 0:
                raise RuntimeError(f"Precio invalido recibido para {symbol}: {price}")
        except Exception as e:
            print(f"[P3+P2-EXEC/ERROR] No se pudo obtener precio de {symbol}: {e}")
            return {}

        if action == "LONG":
            # Verificar posicion actual y cerrar si es necesario
            current_pos = self.connector.get_position(symbol)
            if current_pos["side"] == "SHORT":
                print(f"[P3+P2-EXEC] Posicion SHORT detectada. Cerrando antes de abrir LONG...")
                self.connector.close_position(symbol)
            elif current_pos["side"] == "LONG":
                # [P1-DYNAMIC-HOLD] Bloqueo de Pyramiding/Compounding infinito
                print(f"[P1-DYNAMIC-HOLD] Posicion LONG ya existe. Ignorando senal de re-compra para evitar compounding infinito y proteger margen.")
                return {}

            # Calcular contratos a partir de size_usd
            # [P3-MIN-CONTRACT] OKX requiere minimo 0.01 contratos y step size 0.0001
            contracts = round(size_usd / price, 4)  # Step size 0.0001
            
            if contracts < 0.01:
                print(f"[P3+P2-EXEC/WARN] Contratos calculados ({contracts}) menores al minimo de 0.01. "
                      f"Se requieren al menos ~${0.01 * price:,.2f} USD. Abortando orden.")
                return {}

            close_params = {'reduceOnly': False}  # Futuros: nueva posicion long
            print(f"[P3+P2-EXEC/LONG] Abriendo LONG: {contracts:.4f} contratos @ ~${price:,.2f} (${size_usd:,.2f} USD)")
            order = self.connector.execute_order(side="buy", contracts=contracts, params=close_params)

        elif action == "SHORT":
            current_pos = self.connector.get_position(symbol)
            if current_pos["side"] == "LONG":
                print(f"[P3+P2-EXEC] Posicion LONG detectada. Cerrando antes de abrir SHORT...")
                self.connector.close_position(symbol)
            elif current_pos["side"] == "SHORT":
                print(f"[P1-DYNAMIC-HOLD] Posicion SHORT ya existe. Ignorando re-venta.")
                return {}
            # Como fallback por si se activa short
            print(f"[P3+P2-EXEC/WARN] Accion SHORT ignorada temporalmente en entorno Only-Long.")
            return {}

        elif action == "CLOSE":
            current_pos = self.connector.get_position(symbol)
            if current_pos["side"] != "HOLD":
                print(f"[P3+P2-EXEC/CLOSE] Forzando cierre de posicion {current_pos['side']} en {symbol}...")
                order = self.connector.close_position(symbol)
            else:
                print(f"[P3+P2-EXEC/CLOSE] Sin posicion abierta. Nada que cerrar.")
                return {}
                
        elif action == "HOLD":
            current_pos = self.connector.get_position(symbol)
            if current_pos["side"] == "LONG":
                # [P1-DYNAMIC-HOLD] Histéresis de salida: dejar correr la ganancia si xgb_prob sigue siendo alcista
                try:
                    from config.settings import cfg as _cfg
                    exit_prob = float(getattr(_cfg.metalabeler, "dynamic_hold_exit_prob", 0.48))
                except Exception:
                    exit_prob = 0.48
                    
                xgb_prob = float(decision.get("xgb_prob", 0.5))
                if xgb_prob >= exit_prob:
                    print(f"[P1-DYNAMIC-HOLD/HOLD] Manteniendo ganancia de LONG activa. (xgb_prob {xgb_prob:.4f} >= {exit_prob}).")
                    return {}
                else:
                    print(f"[P1-DYNAMIC-HOLD/EXIT] Probabilidad alcista deteriorada (xgb_prob {xgb_prob:.4f} < {exit_prob}). Liquidando LONG para asegurar.")
                    order = self.connector.close_position(symbol)
                    
            elif current_pos["side"] == "SHORT":
                try:
                    from config.settings import cfg as _cfg
                    exit_prob = float(getattr(_cfg.metalabeler, "dynamic_hold_exit_prob", 0.48))
                except Exception:
                    exit_prob = 0.48
                xgb_prob = float(decision.get("xgb_prob", 0.5))
                if xgb_prob <= (1.0 - exit_prob):
                    print(f"[P1-DYNAMIC-HOLD/HOLD] Manteniendo ganancia de SHORT activa. (xgb_prob {xgb_prob:.4f} <= {(1.0 - exit_prob):.2f}).")
                    return {}
                else:
                    print(f"[P1-DYNAMIC-HOLD/EXIT] Probabilidad bajista deteriorada. Liquidando SHORT.")
                    order = self.connector.close_position(symbol)
            else:
                print(f"[P3+P2-EXEC/HOLD] Sin posicion abierta. Quedando plano.")
                return {}
        else:
            print(f"[P3+P2-EXEC/WARN] Accion desconocida: {action}. Ignorando.")
            return {}

        return order if order else {}


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
                      # 5. POSITION SIZER (Si el cerebro autoriza disparo)
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
                           # 6. [P3+P2-2026-06-12] EJECUCION REAL via OKXBrokerConnector
                           prod_cost_rt = self.connector.get_production_cost_rt()
                           print(f"  [5] Position Sizer Activo: Asignando ${sz_usd:,.2f} | symbol={self.trading_symbol} | cost_rt={prod_cost_rt:.4f}")
                           order = self._execute_order(action=action, size_usd=sz_usd)

                           # Loggear Auditoria
                           executed_price = float(order.get('avg_price', order.get('price', decision.get('price', 0.0)))) if order else decision.get('price', 0.0)
                           self.db.log_audit(
                              timestamp=datetime.utcnow(), price=decision.get("price", 0.0),
                              action=action, confidence=confidence, xgb_prob=decision.get("xgb_prob", 0.0),
                              hmm_regime=0, reason=f"[HMM-REGIME: {decision.get('regime', 'UNKNOWN')}] [P3+P2] cost_rt={prod_cost_rt:.4f} {sizing.get('multiplier_breakdown', '')}",
                              contracts=sizing.get("contracts", 0), executed_price=executed_price
                           )

                           self.telegram.send_alert(
                               f"[TRADE] {action} | {self.trading_symbol} | Confianza: {confidence:.2%} | Tamano: ${sz_usd:,.2f} | Precio: ${executed_price:,.2f} | Fee: {prod_cost_rt:.4f}",
                               priority="info"
                           )
                      else:
                           print(f"  [5] Position Sizer Mudo: Emision cancelada. Motivo: {sizing.get('reason')}")
                           
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
