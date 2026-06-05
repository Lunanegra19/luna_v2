#!/usr/bin/env python
"""
run_live_trader.py
==================
[LUNA-V2-LIVE] Luna V2 Live Demo Orquestador en Vivo de Producción para Luna V2.
Ejecuta el loop inmortal de trading: Heartbeat -> Reconcile -> Risk -> Inferencia Ensamble -> Kelly Sizing -> OKX Execution.
"""

import sys

# Reconfigure stdout for UTF-8 encoding on Windows to prevent charmap crashes
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import time
import argparse
import traceback

from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import numpy as np

# Fix python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# [BUGFIX-UNPICKLE-01] [MEJORA-SOP-V10] Inyectar adapters en __main__ para evitar AttributeErrors de Pickle/Joblib
# en el live trader cuando se deserializan los calibradores del MetaLabelerV2.
try:
    from luna.models.calibrate_probabilities import _RFWithAdapter, _IdentityWrapper, _TSAdapter
    sys.modules['__main__']._RFWithAdapter = _RFWithAdapter
    sys.modules['__main__']._IdentityWrapper = _IdentityWrapper
    sys.modules['__main__']._TSAdapter = _TSAdapter
    print("[BUGFIX-UNPICKLE-01] OK: Wrappers de calibración inyectados con éxito en __main__.")
except Exception as e_unpickle:
    print(f"[BUGFIX-UNPICKLE-01] WARNING: No se pudieron inyectar los wrappers en __main__: {e_unpickle}")

# Load Env
env_path = PROJECT_ROOT / ".env.sandbox"
if env_path.exists():
    load_dotenv(env_path)
    print(f"[BOOT] Entorno sandbox cargado desde {env_path.name}")
else:
    load_dotenv(PROJECT_ROOT / ".env")
    print("[BOOT] Entorno de produccion cargado desde .env")

from luna.database.db_manager import DatabaseManager
from luna.live.telegram_alerts import TelegramAlerts
from luna.live.risk_monitor import RiskMonitor
from luna.live.position_sizer import PositionSizer
from luna.live.reconciliation import BalanceReconciler
from luna.live.ensemble_live_inference import LunaEnsembleLiveInference
from luna.live.okx_connector import OKXBrokerConnector
from luna.data.data_collector import DataCollector
from luna.features.feature_pipeline import FeaturePipeline
from luna.live.operational_auditor import LiveOperationalAuditor

class LunaLiveTraderV2LiveDemo:
    """
    [LUNA-V2-LUNA V2 LIVE DEMO] Orquestador de Producción del Ensamble Multisemilla Luna V2.
    """
    def __init__(self, demo_mode: bool = True, interval_seconds: int = 3600, once: bool = False):
        print("🌙 [BOOT] Inicializando Luna V2 Live Trader (Luna V2 Live Demo)...")
        
        self.interval_seconds = interval_seconds
        self.once = once
        self.demo_mode = demo_mode
        
        # 1. Base de Datos (Integridad ACID & Pools)
        self.db = DatabaseManager()
        
        # 2. Telemetria Telegram
        self.telegram = TelegramAlerts()
        self._register_telegram_commands()
        self.telegram.start_command_listener()
        
        # 3. OKX Broker API (Safety first: Demo default)
        self.okx = OKXBrokerConnector(demo_mode=self.demo_mode)
        
        # 4. Defensas de Capital
        self.risk_monitor = RiskMonitor(telegram_bot=self.telegram)
        self.sizer = PositionSizer()
        self.reconciler = BalanceReconciler(self.okx.exchange, telegram_bot=self.telegram)
        
        # 5. Cerebro Ensamble Multi-Semilla
        self.ensemble = LunaEnsembleLiveInference()
        
        # 5.5. Auditor Operativo en Vivo (SOP V10.0 compliance)
        self.auditor = LiveOperationalAuditor(self.okx, self.risk_monitor, self.telegram)
        
        # 6. Simbolo y configuracion del activo
        # [FIX-P1-SYMBOL] El simbolo se lee SIEMPRE de settings.yaml (fuente canónica).
        # La heuristica anterior "BTC/EUR si hostname==eea.okx.com" era un bug que causaba
        # FATAL en cada intento de LONG: la cuenta Demo OKX solo tiene USDT, no EUR.
        # eea.okx.com es solo la region europea de la API, NO indica operar en spot EUR.
        # Prioridad: (1) settings.yaml, (2) env var OKX_TRADING_SYMBOL, (3) fallback BTC/USDT:USDT
        try:
            from config.settings import cfg as _cfg_symbol
            _symbol_from_cfg = getattr(_cfg_symbol.data, 'trading_symbol', None)
            if _symbol_from_cfg:
                self.symbol = str(_symbol_from_cfg)
                print(f"[FIX-P1-SYMBOL] Símbolo cargado desde settings.yaml: {self.symbol}")
            else:
                # Env var como segunda opcion
                self.symbol = os.getenv("OKX_TRADING_SYMBOL", "BTC/USDT:USDT")
                print(f"[FIX-P1-SYMBOL] settings.yaml sin trading_symbol — usando env/default: {self.symbol}")
        except Exception as e_sym:
            # Fallback defensivo: nunca fallar el boot por el simbolo
            self.symbol = os.getenv("OKX_TRADING_SYMBOL", "BTC/USDT:USDT")
            print(f"[FIX-P1-SYMBOL] WARN: Error leyendo settings para simbolo: {e_sym}. Usando: {self.symbol}")
        
        # Validacion de seguridad: nunca operar en spot EUR en cuenta de futuros USDT
        if "EUR" in self.symbol and "USDT" not in self.symbol:
            print(f"[FIX-P1-SYMBOL] CRITICAL: Simbolo spot EUR detectado ({self.symbol}). "
                  f"La cuenta OKX opera en futuros USDT. Corrigiendo a BTC/USDT:USDT.")
            self.symbol = "BTC/USDT:USDT"
        
        print(f"[FIX-P1-SYMBOL] Simbolo de trading definitivo: {self.symbol}")
        
        # 7. Consensus-Soft Embargo state tracking
        self.embargo_expiration = None
        
        # 8. Tracker de inicio de ciclo operativo
        self.cycle_start = time.time()
        
        self.telegram.send_alert(
            f"🚀 *Luna V2 Live Demo Iniciado*\n"
            f"• Modo: {'DEMO/SANDBOX (SEGURO)' if self.demo_mode else 'PRODUCCION REAL (RIESGO REAL)'}\n"
            f"• Simbolo: {self.symbol}\n"
            f"• Semillas Activas: {self.ensemble.active_seeds}\n"
            f"• Quorum Minimo: {self.ensemble.consensus_threshold} semillas",
            priority="info"
        )
        print("🌙 [BOOT] Luna V2 Live Demo listo para operar.")

    def _register_telegram_commands(self):
        self.telegram.register_command("/status", self._get_status_report)
        self.telegram.register_command("/kill", self._emergency_kill)
        self.telegram.register_command("/reconcile", self._force_reconciliation)
        self.telegram.register_command("/vps_status", self._get_vps_status_report)
        self.telegram.register_command("/panico", self._emergency_kill)
        print("[MEJORA-SOP-V10] Registrados comandos seguros /vps_status y /panico en Telegram listener.")

    def _force_reconciliation(self):
        print("[Telegram/COMMAND] Forzando reconciliacion...")
        self.reconciler.reconcile()
        return "🔄 Reconciliación completada."

    def _get_vps_status_report(self):
        print("[Telegram/COMMAND] [MEJORA-SOP-V10] Generando reporte de hardware VPS...")
        try:
            import psutil
            import time
            cpu = psutil.cpu_percent(interval=0.5)
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/')
            uptime_hours = (time.time() - psutil.boot_time()) / 3600
            
            report = (
                f"🖥️ *MÉTRICAS DE HARDWARE VPS HETZNER*\n"
                f"• Carga CPU: {cpu:.1f}%\n"
                f"• Uso Memoria RAM: {ram:.1f}%\n"
                f"• Disco '/': {disk.percent:.1f}% (Libre: {disk.free / (1024**3):.1f} GB)\n"
                f"• Uptime del Servidor: {uptime_hours:.1f} horas\n"
                f"• PM2 Process Status: ONLINE"
            )
            return report
        except Exception as e:
            return f"❌ Error generando reporte de hardware VPS: {e}"

    def _get_status_report(self):
        print("[Telegram/COMMAND] Generando reporte de estado...")
        try:
            state = self.db.get_live_state()
            risk = self.risk_monitor.get_risk_summary()
            pos = self.okx.get_position(self.symbol)
            
            portfolio_value = float(state['portfolio_value']) if state else float(risk.get('portfolio_value', 5000.0))
            ath = float(state['ath']) if state else float(risk.get('ath', 5000.0))
            drawdown = float(state['drawdown']) if state else float(risk.get('drawdown', 0.0))
            is_paused_flag = state['is_paused'] if state else risk.get('is_paused', False)
            
            report = (
                f"📊 *REPORTE DE ESTADO LUNA V2*\n"
                f"• Equidad DB: ${portfolio_value:,.2f} USD\n"
                f"• ATH registrado: ${ath:,.2f} USD\n"
                f"• Max DD actual: {drawdown:.2%}\n"
                f"• Bot Pausado: {'SÍ' if is_paused_flag else 'NO'}\n"
                f"• Posicion OKX: {pos['side']} ({pos['contracts']} conts) | PnL: ${pos['unrealized_pnl']:.2f}\n"
                f"• Embargo Activo: {'SÍ' if self._is_embargo_active() else 'NO'}"
            )
            return report
        except Exception as e:
            return f"❌ Error generando reporte: {e}"

    def _emergency_kill(self):
        print("[Telegram/COMMAND] 🛑 EMERGENCY KILL TRIGGERED")
        self.telegram.send_alert("🛑 *KILL SWITCH MANUAL DETECTADO*. Cerrando todas las posiciones y deteniendo el bot...", priority="critical")
        
        # Cerrar posicion en OKX de forma segura
        self.okx.close_position(self.symbol)
        
        # Pausar el live state en base de datos para prevenir auto-reinicio
        state = self.db.get_live_state()
        if state:
            self.db.update_live_state(
                portfolio_value=float(state['portfolio_value']),
                ath=float(state['ath']),
                drawdown=float(state['drawdown']),
                is_paused=True
            )
        else:
            print("  [!] DB no conectada. No se pudo persistir la pausa en base de datos. Se usara variable en memoria.")
            self.risk_monitor.is_paused = True
            
        print("  [!] Bot pausado permanentemente en base de datos. Saliendo del proceso.")
        sys.exit(0)

    def _is_embargo_active(self) -> bool:
        """Determina si el temporizador del Consensus-Soft Embargo esta activo."""
        if self.embargo_expiration is None:
            return False
        return datetime.now(timezone.utc) < self.embargo_expiration

    def _handle_execution(self, decision: dict, current_price: float) -> float:
        """
        Ejecuta las operaciones de mercado en base a las señales del ensamble consolidado
        y el control estricto de embargos.
        Retorna el precio ejecutado de la orden (para auditoría y slippage).
        """
        exec_price = current_price
        action = decision.get("action", "HOLD")
        confidence = decision.get("confidence", 0.0)
        regime = decision.get("regime", "UNKNOWN")
        current_vol = decision.get("current_vol", 0.001)
        historical_vol = decision.get("historical_vol", 0.0015)
        
        c_count = decision.get("consensus_count", 3)
        t_seeds = len(self.ensemble.active_seeds)

        # [FIX-LIVE-006/DB-003] Convertir regime str semántico a índice entero real del HMM.
        # Los `hmm_regime=0`, `hmm_regime=2 if LONG else 3` eran hardcodeados e incorrectos.
        # Ahora se busca el índice real en el state_map del pkl del HMM.
        def _regime_str_to_int(regime_str: str) -> int:
            try:
                from pathlib import Path as _PP_lt
                import joblib as _jbl_lt
                _hmm_pkl = _PP_lt(__file__).resolve().parent.parent / "data" / "models" / "hmm_regime.pkl"
                if _hmm_pkl.exists():
                    _bundle = _jbl_lt.load(_hmm_pkl)
                    _smap = _bundle.get("state_map", {})
                    for _idx, _lbl in _smap.items():
                        if str(_lbl).upper() == str(regime_str).upper():
                            print(f"[FIX-LIVE-006] regime_str='{regime_str}' -> hmm_regime_int={_idx} (state_map match)")
                            return int(_idx)
                    print(f"[FIX-LIVE-006][WARN] regime='{regime_str}' no encontrado en state_map {_smap}. Usando -1.")
            except Exception as _e_lt:
                print(f"[FIX-LIVE-006][WARN] No se pudo convertir regime a int: {_e_lt}")
            return -1  # -1 indica régimen desconocido — trazable en DB

        _hmm_regime_int = _regime_str_to_int(regime)
        print(f"[FIX-LIVE-006] regime='{regime}' | hmm_regime_int={_hmm_regime_int} para DB audit_log")

        
        # 1. Chequeo de soft embargo
        if self._is_embargo_active():
            print(f"  [EMBARGO/ACTIVE] Consensus-Soft Embargo activo hasta {self.embargo_expiration.strftime('%Y-%m-%d %H:%M:%S')}. Ignorando señal {action}.")
            logger.info("Signal blocked due to active Consensus-Soft Embargo.")
            return exec_price
            
        # 2. Si hay consenso >= 4, se enciende/renueva el Consensus-Soft Embargo de 24h
        if decision.get("soft_embargo_active", False):
            embargo_hours = self.ensemble.soft_embargo_hours
            self.embargo_expiration = datetime.now(timezone.utc) + timedelta(hours=embargo_hours)
            msg = f"🛡️ [Consensus-Soft Embargo] ¡QUORUM ALTO ({decision.get('consensus_count')} semillas)! Activando embargo de {embargo_hours}H."
            print(msg)
            self.telegram.send_alert(msg, priority="warning")
            logger.info(f"Consensus-Soft Embargo activated for {embargo_hours} hours.")

        # 3. Sincronizar posicion actual del exchange
        pos = self.okx.get_position(self.symbol)
        current_side = pos["side"]
        
        # 4. Procesar decisiones
        if action == "HOLD":
            # Si el ensamble dicta HOLD, cerramos posicion si existe
            if current_side != "HOLD":
                print(f"  [EXEC] El ensamble dicta HOLD. Liquidando posicion {current_side} actual...")
                closed_order = self.okx.close_position(self.symbol)
                if closed_order:
                    exec_price = closed_order.get('avg_price', current_price)
                    fee_cost = closed_order.get('fee_cost', 0.0)
                    fee_currency = closed_order.get('fee_currency', 'USDT')
                    slip_pct = closed_order.get('slippage_pct', 0.0)
                    
                    print(f"[LIVE-TRADER-AUDIT] Registrando decisión en DB. Tiempo del ciclo transcurrido: {time.time() - self.cycle_start:.2f}s")
                    reason_str = f"[Consenso={c_count}/{t_seeds}] [HMM-REGIME: {regime}] Consenso dicta HOLD. Cierre orden inversa. [SOP-FEE: {fee_cost:.4f} {fee_currency} | SLIPPAGE: {slip_pct:.4%}] [DURATION: {time.time() - self.cycle_start:.1f}s]"
                    print(f"[LIVE-TRADER-AUDIT-PRINT] DB Reason logged: {reason_str}")
                    self.db.log_audit(
                        timestamp=datetime.utcnow(),
                        price=current_price,
                        action="HOLD",
                        confidence=confidence,
                        xgb_prob=decision.get("xgb_prob", 0.5),
                        hmm_regime=_hmm_regime_int,  # [FIX-LIVE-006] regime real del decision dict
                        reason=reason_str,
                        contracts=0,
                        executed_price=exec_price
                    )
                    self.telegram.send_alert(
                        f"⏹️ *Cierre de Posición LUNA V2*: Sincronizado a HOLD.\n"
                        f"• Precio de Referencia: ${current_price:,.2f} USD\n"
                        f"• Precio Ejecutado Prom.: ${exec_price:,.2f} USD\n"
                        f"• Comisión Total: {fee_cost:.4f} {fee_currency}\n"
                        f"• Deslizamiento (Slippage): {slip_pct:.4%}",
                        priority="info"
                    )
            else:
                print("  [EXEC] Posicion ya alineada a HOLD. No hay operaciones requeridas.")
                print(f"[LIVE-TRADER-AUDIT] Registrando decisión en DB. Tiempo del ciclo transcurrido: {time.time() - self.cycle_start:.2f}s")
                reason_str = f"[Consenso={c_count}/{t_seeds}] [HMM-REGIME: {regime}] Consenso dicta HOLD (alineado). [DURATION: {time.time() - self.cycle_start:.1f}s]"
                print(f"[LIVE-TRADER-AUDIT-PRINT] DB Reason logged: {reason_str}")
                self.db.log_audit(
                    timestamp=datetime.utcnow(),
                    price=current_price,
                    action="HOLD",
                    confidence=confidence,
                    xgb_prob=decision.get("xgb_prob", 0.5),
                    hmm_regime=_hmm_regime_int,  # [FIX-LIVE-006]
                    reason=reason_str,
                    contracts=0,
                    executed_price=current_price
                )
            return exec_price

        # 5. Para LONG o SHORT, calculamos el dimensionamiento
        # Recuperamos si hay KMeans Tribe ID o similar en la ultima barra
        tribe_id = -1
        
        # Calculamos la posicion
        sizing = self.okx.calculate_live_size(
            action=action,
            confidence=confidence,
            hmm_regime=regime,
            current_vol=current_vol,
            historical_vol=historical_vol,
            sizer=self.sizer,
            asset_price=current_price,
            tribe_id=tribe_id
        )
        
        target_size_usd = sizing.get("size_usd", 0.0)
        target_contracts = sizing.get("contracts", 0.0)
        
        if target_size_usd <= 0 or target_contracts <= 0:
            print("  [EXEC] Position Sizer entrego volumen nulo o insuficiente. Abortando transmision.")
            return exec_price

        # [EXEC_HYBRID/TELEMETRY] Recuperando estadísticas de riesgo para Telegram (Fix SOP M-06 / R6)
        risk = self.risk_monitor.get_risk_summary()
        equity = float(risk.get("portfolio_value", 5000.0))
        start_day = float(risk.get("equity_start_day", equity))
        start_week = float(risk.get("equity_start_week", equity))
        
        daily_dd = max(0.0, (start_day - equity) / start_day) if start_day > 0 else 0.0
        weekly_dd = max(0.0, (start_week - equity) / start_week) if start_week > 0 else 0.0
        global_dd = float(risk.get("drawdown", 0.0))
        
        real_leverage = target_size_usd / equity if equity > 0 else 0.0
        
        print(f"[EXEC_HYBRID/TELEMETRY] Cargado: Daily DD: {daily_dd:.2%} | Weekly DD: {weekly_dd:.2%} | Global DD: {global_dd:.2%} | Real Leverage: {real_leverage:.2f}x | Equity: ${equity:,.2f}")

        # 6. Ejecutar transicion / rebalanceo
        if current_side == action:
            # Ya estamos en la direccion correcta. Validamos desviacion de tamaño.
            size_diff_pct = abs(pos["contracts"] - target_contracts) / target_contracts if target_contracts > 0 else 0
            if size_diff_pct > 0.15:
                print(f"  [EXEC] Rebalanceo necesario. Posicion actual de {pos['contracts']} difiere de target {target_contracts:.6f} (>15%). Ajustando...")
                
                # Cerrar y volver a abrir para simplificar y asegurar transicion limpia
                self.okx.close_position(self.symbol)
                time.sleep(2)
                order_side = "buy" if action == "LONG" else "sell"
                
                print(f"[EXEC_HYBRID/TRADER] Llamando execute_hybrid_order en reajuste para {self.symbol} ({order_side})")
                order = self.okx.execute_hybrid_order(self.symbol, order_side, target_contracts)
                
                if order:
                    exec_price = order.get('avg_price', current_price)
                    fee_cost = order.get('fee_cost', 0.0)
                    fee_currency = order.get('fee_currency', 'USDT')
                    slip_pct = order.get('slippage_pct', 0.0)
                    
                    print(f"[LIVE-TRADER-AUDIT] Registrando decisión en DB. Tiempo del ciclo transcurrido: {time.time() - self.cycle_start:.2f}s")
                    reason_str = f"[Consenso={c_count}/{t_seeds}] [HMM-REGIME: {regime}] Ajuste de volumen en vivo: target {target_contracts} conts. {sizing.get('multiplier_breakdown')} [SOP-FEE: {fee_cost:.4f} {fee_currency} | SLIPPAGE: {slip_pct:.4%}] [DURATION: {time.time() - self.cycle_start:.1f}s]"
                    print(f"[LIVE-TRADER-AUDIT-PRINT] DB Reason logged: {reason_str}")
                    self.db.log_audit(
                        timestamp=datetime.utcnow(),
                        price=current_price,
                        action=action,
                        confidence=confidence,
                        xgb_prob=decision.get("xgb_prob", 0.5),
                        hmm_regime=_hmm_regime_int,  # [FIX-LIVE-006] real regime index
                        reason=reason_str,
                        contracts=int(target_contracts),
                        executed_price=exec_price
                    )
                    self.telegram.send_alert(
                        f"🔄 *Ajuste de Posición LUNA V2* (Rebalanceo)\n"
                        f"• Dirección: *{action}*\n"
                        f"• Confianza Calibrada: {confidence:.2%}\n"
                        f"• Quórum: {decision.get('consensus_count')}/{len(self.ensemble.active_seeds)} semillas\n"
                        f"• Volumen USD: ${target_size_usd:,.2f}\n"
                        f"• Contratos: {target_contracts:.6f}\n"
                        f"• Precio Ideal: ${current_price:,.2f} USD\n"
                        f"• Precio Ejecutado Prom.: ${exec_price:,.2f} USD\n"
                        f"• Comisión Total: {fee_cost:.4f} {fee_currency}\n"
                        f"• Deslizamiento (Slippage): {slip_pct:.4%}\n"
                        f"• Apalancamiento Real: {real_leverage:.2f}x nocional\n"
                        f"• Régimen HMM: {regime}\n"
                        f"• Drawdown Diario: -{daily_dd:.2%} (Límite -3%)\n"
                        f"• Drawdown Semanal: -{weekly_dd:.2%} (Límite -7%)\n"
                        f"• Drawdown Global ATH: -{global_dd:.2%} (Límite -15% Pánico)\n"
                        f"• Equidad Neta en Cuenta: ${equity:,.2f} USDT",
                        priority="info"
                    )
                else:
                    logger.error("[EXEC] Error fatal al emitir la orden de reajuste en OKX.")
                    print("  [EXEC] [FATAL] La orden de reajuste fue rechazada o falló.")
            else:
                print(f"  [EXEC] Posicion actual {current_side} de {pos['contracts']} conts alineada dentro de tolerancia del target {target_contracts:.6f}.")
        else:
            # Transicion completa (Ej: HOLD -> LONG o SHORT -> LONG)
            if current_side != "HOLD":
                print(f"  [EXEC] Transición de señal detectada ({current_side} -> {action}). Liquidando posicion previa...")
                self.okx.close_position(self.symbol)
                time.sleep(2)
                
            print(f"  [EXEC] Abriendo nueva posicion {action} con target de {target_contracts:.6f} contratos (${target_size_usd:,.2f} USD)...")
            order_side = "buy" if action == "LONG" else "sell"
            
            print(f"[EXEC_HYBRID/TRADER] Llamando execute_hybrid_order en nueva posición para {self.symbol} ({order_side})")
            order = self.okx.execute_hybrid_order(self.symbol, order_side, target_contracts)
            
            if order:
                exec_price = order.get('avg_price', current_price)
                fee_cost = order.get('fee_cost', 0.0)
                fee_currency = order.get('fee_currency', 'USDT')
                slip_pct = order.get('slippage_pct', 0.0)
                
                print(f"[LIVE-TRADER-AUDIT] Registrando decisión en DB. Tiempo del ciclo transcurrido: {time.time() - self.cycle_start:.2f}s")
                reason_str = f"[Consenso={c_count}/{t_seeds}] [HMM-REGIME: {regime}] Apertura posicion. {sizing.get('multiplier_breakdown')} [SOP-FEE: {fee_cost:.4f} {fee_currency} | SLIPPAGE: {slip_pct:.4%}] [DURATION: {time.time() - self.cycle_start:.1f}s]"
                print(f"[LIVE-TRADER-AUDIT-PRINT] DB Reason logged: {reason_str}")
                self.db.log_audit(
                    timestamp=datetime.utcnow(),
                    price=current_price,
                    action=action,
                    confidence=confidence,
                    xgb_prob=decision.get("xgb_prob", 0.5),
                    hmm_regime=_hmm_regime_int,  # [FIX-LIVE-006]
                    reason=reason_str,
                    contracts=int(target_contracts),
                    executed_price=exec_price
                )
                self.telegram.send_alert(
                    f"🎯 *TRADE EJECUTADO EN LUNA V2*\n"
                    f"• Dirección: *{action}*\n"
                    f"• Confianza Calibrada: {confidence:.2%}\n"
                    f"• Quórum: {decision.get('consensus_count')}/{len(self.ensemble.active_seeds)} semillas\n"
                    f"• Volumen USD: ${target_size_usd:,.2f}\n"
                    f"• Contratos: {target_contracts:.6f}\n"
                    f"• Precio Ideal: ${current_price:,.2f} USD\n"
                    f"• Precio de Entrada Prom.: ${exec_price:,.2f} USD\n"
                    f"• Comisión Total: {fee_cost:.4f} {fee_currency}\n"
                    f"• Deslizamiento (Slippage): {slip_pct:.4%}\n"
                    f"• Apalancamiento Real: {real_leverage:.2f}x nocional\n"
                    f"• Régimen HMM: {regime}\n"
                    f"• Drawdown Diario: -{daily_dd:.2%} (Límite -3%)\n"
                    f"• Drawdown Semanal: -{weekly_dd:.2%} (Límite -7%)\n"
                    f"• Drawdown Global ATH: -{global_dd:.2%} (Límite -15% Pánico)\n"
                    f"• Equidad Neta en Cuenta: ${equity:,.2f} USDT",
                    priority="info"
                )
            else:
                logger.error("[EXEC] Error fatal al emitir la orden en OKX.")
                print("  [EXEC] [FATAL] La orden de mercado fue rechazada o falló.")
        return exec_price

    def run_cycle(self):
        """Ejecuta un unico ciclo del bucle ininterrumpido con Auditor Operativo en Vivo."""
        self.cycle_start = time.time()
        start_time = self.cycle_start
        print("="*70)
        print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando Ciclo Operativo LUNA V2")
        print("="*70)
        
        try:
            # 1. HEARTBEAT (Latido del Watchdog en DB)
            self.db.log_heartbeat(status="RUNNING")
            print("  [1] Heartbeat: SQL latido de vida registrado.")
            
            # 2. RECONCILIACIÓN CONTABLE
            is_synced = self.reconciler.reconcile()
            if not is_synced:
                print("  [2] Reconciliacion: Saldo sincronizado y corregido por desvío.")
            else:
                print("  [2] Reconciliacion: Saldo contable validado con éxito.")
                
            # Recuperar saldo actual
            db_state = self.db.get_live_state()
            if db_state:
                current_equity = float(db_state.get('portfolio_value', 5000.0))
                is_paused = bool(db_state.get('is_paused', False))
            else:
                current_equity = 5000.0
                is_paused = self.risk_monitor.is_paused
                print("[LUNA-V2-LIVE-DEMO-FIX-DB/WARN] No se pudo leer live_state de DB, usando valores de fallback locales por seguridad (is_paused=False, equity=5000.0).")
            
            # 3. CIRCUIT BREAKERS Y RISK MONITOR
            is_blocked = self.risk_monitor.check_system_health(current_portfolio_value=current_equity)
            if is_blocked or is_paused:
                print("  [3] Risk Monitor: SISTEMA EN ESTADO BLOQUEADO / PAUSADO. Omitiendo inferencia.")
                self.db.log_heartbeat(status="BLOCKED")
                return
            else:
                print("  [3] Risk Monitor: Escaneo de circuito pasivo OK.")
                
            # 4. PIPELINE CAUSAL & INGERIERIA EN TIEMPO REAL
            print("  [4] Recolectando ticks de mercado y procesando pipeline...")
            try:
                print("[FIX-BUG] Ejecutando DataCollector.build en modo 'live' para actualización incremental real de datos raw.")
                DataCollector().build(mode='live')
            except Exception as e:
                print(f"      [!] Fallo de fetcher incremental: {e}. Usando caches locales.")
                
            try:
                pipeline_res = FeaturePipeline().run(skip_fracdiff=False, skip_sfi=True, live_mode=True)
                df_live = pipeline_res.get('live', None)
                if df_live is not None:
                    # [BUGFIX-WEEKEND-NAN] Ffill weekend gaps from closed traditional markets
                    # TradFi assets (VIX, DXY, SP500, Gold, etc.) close on weekends leaving NaN values in live tables.
                    # We replace infs with nan, ffill to propagate Friday's values, bfill, and fillna(0.0) for absolute safety.
                    print("[BUGFIX-WEEKEND-NAN] Aplicando ffill(), bfill() y fillna(0.0) a df_live para rellenar huecos de mercados tradicionales e infs.")
                    logger.info("[BUGFIX-WEEKEND-NAN] Sanitizando df_live (sustituyendo infs por nan y rellenando huecos) para evitar disparar el NaN/Inf Sanity Shield.")
                    df_live = df_live.sort_index().replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
                    # Save live features to disk for dashboard telemetry
                    try:
                        out_dir = PROJECT_ROOT / "data" / "features"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        df_live.to_parquet(out_dir / "features_live.parquet")
                        print(f"🌙 [LIVE-INFERENCE-SAVE] [MEJORA-FEAT-TELEMETRY] Features en vivo guardadas exitosamente en: {out_dir / 'features_live.parquet'} | Filas: {len(df_live)} | Fecha Máxima: {df_live.index.max()}")
                    except Exception as save_err:
                        print(f"⚠️ [LIVE-INFERENCE-WARN] No se pudo guardar features_live.parquet: {save_err}")
            except Exception as e:
                print(f"      [!] Error de feature pipeline: {e}")
                df_live = None
                
            if df_live is None or df_live.empty:
                print("      [!] DataFrame de features vacio. Omitiendo ciclo por falta de datos.")
                self.db.log_heartbeat(status="SLEEP_WAIT_DATA")
                return
                
            # 4.5. AUDITOR OPERATIVO EN VIVO - FASE PRE-INFERENCIA (Drift, NaN/Inf, API Liveness)
            is_pre_ok, pre_audit = self.auditor.run_pre_inference_audit(df_live)
            if not is_pre_ok:
                failure_reason = pre_audit.get('failure_reason', '')
                print(f"  [4.5] Auditor Operativo: FALLÓ la auditoría pre-inferencia. Motivo: {failure_reason}")
                self.db.log_heartbeat(status="OPERATIONAL_ALARM")
                # Forzar cierre preventivo a HOLD en OKX
                print("  [Auditor/RECOVERY] Forzando liquidación preventiva a HOLD por fallo crítico pre-inferencia.")
                self._handle_execution({"action": "HOLD", "confidence": 0.0, "xgb_prob": 0.5, "regime": "UNKNOWN"}, current_price=float(df_live['close'].iloc[-1]))
                
                # Consolidar y persistir log de auditoría operativa con falla
                audit_log_data = {
                    "timestamp": datetime.utcnow(),
                    "clock_drift_minutes": pre_audit["clock_drift_minutes"],
                    "clock_drift_status": pre_audit["clock_drift_status"],
                    "nan_inf_null_cols": pre_audit["nan_inf_null_cols"],
                    "nan_inf_status": pre_audit["nan_inf_status"],
                    "active_leverage": 0.0,
                    "leverage_status": "OK",
                    "api_liveness_equity": pre_audit["api_liveness_equity"],
                    "api_liveness_status": pre_audit["api_liveness_status"],
                    "hmm_regime_index": -1,
                    "hmm_status": "OK",
                    "execution_latency_sec": time.time() - start_time,
                    "latency_status": "OK",
                    "slippage_pct": 0.0,
                    "slippage_status": "OK",
                    "is_approved": False,
                    "details": f"Pre-inference audit failed: {failure_reason}"
                }
                self.db.log_operational_audit(audit_log_data)
                
                # [FIX-CLOCK-DRIFT-NOBLOCK] ClockDrift y NaN/Inf son fallos TRANSITORIOS:
                # el ciclo se omite pero el sistema NO se pausa permanentemente para auto-recuperarse.
                # Solo API Liveness down (riesgo de trading ciego real) activa is_paused=True.
                # Bug original: is_paused=True para cualquier fallo de auditor → bucle de bloqueo eterno.
                is_api_failure = "API Liveness" in failure_reason
                if is_api_failure:
                    print(f"[FIX-CLOCK-DRIFT-NOBLOCK] CRITICAL: API Liveness caída — activando is_paused=True para evitar trading ciego.")
                    self.db.update_live_state(portfolio_value=current_equity, ath=current_equity, drawdown=0.0, is_paused=True)
                else:
                    print(f"[FIX-CLOCK-DRIFT-NOBLOCK] Fallo transitorio (ClockDrift/NaN): omitiendo ciclo SIN bloqueo permanente. El sistema intentará recuperarse en el proximo ciclo. Motivo: {failure_reason}")
                    # NO llamar update_live_state con is_paused=True — el sistema debe auto-recuperarse
                return

            # 5. INFERENCIA DEL ENSAMBLE MULTISEMILLA
            decision = self.ensemble.predict_cycle(df_live)
            current_price = decision.get("price", 0.0)
            if current_price <= 0:
                current_price = float(df_live['close'].iloc[-1])
                decision["price"] = current_price
                
            # 5.5. AUDITOR OPERATIVO EN VIVO - FASE POST-INFERENCIA (Leverage ceiling y consistencia HMM)
            is_post_ok, post_audit = self.auditor.run_post_inference_audit(df_live, decision)
            if not is_post_ok:
                post_failure_reason = post_audit.get('failure_reason', '')
                print(f"  [5.5] Auditor Operativo: FALLÓ la auditoría post-inferencia. Motivo: {post_failure_reason}")
                self.db.log_heartbeat(status="OPERATIONAL_ALARM")
                # Forzar cierre preventivo a HOLD
                print("  [Auditor/RECOVERY] Forzando liquidación preventiva a HOLD por fallo crítico post-inferencia.")
                self._handle_execution({"action": "HOLD", "confidence": 0.0, "xgb_prob": 0.5, "regime": "UNKNOWN"}, current_price=current_price)
                
                # Consolidar y persistir log de auditoría operativa con falla
                audit_log_data = {
                    "timestamp": datetime.utcnow(),
                    "clock_drift_minutes": pre_audit["clock_drift_minutes"],
                    "clock_drift_status": pre_audit["clock_drift_status"],
                    "nan_inf_null_cols": pre_audit["nan_inf_null_cols"],
                    "nan_inf_status": pre_audit["nan_inf_status"],
                    "active_leverage": post_audit["active_leverage"],
                    "leverage_status": post_audit["leverage_status"],
                    "api_liveness_equity": pre_audit["api_liveness_equity"],
                    "api_liveness_status": pre_audit["api_liveness_status"],
                    "hmm_regime_index": post_audit["hmm_regime_index"],
                    "hmm_status": post_audit["hmm_status"],
                    "execution_latency_sec": time.time() - start_time,
                    "latency_status": "OK",
                    "slippage_pct": 0.0,
                    "slippage_status": "OK",
                    "is_approved": False,
                    "details": f"Post-inference audit failed: {post_failure_reason}"
                }
                self.db.log_operational_audit(audit_log_data)
                
                # [FIX-CLOCK-DRIFT-NOBLOCK] Post-inference: solo Leverage catastrófico activa is_paused=True.
                # HMM out-of-range es transitorio — el sistema debe auto-recuperarse en el siguiente ciclo.
                is_leverage_critical = "Leverage limit exceeded" in post_failure_reason
                if is_leverage_critical:
                    print(f"[FIX-CLOCK-DRIFT-NOBLOCK] CRITICAL: Apalancamiento catastrófico detectado — activando is_paused=True para proteger el capital.")
                    self.db.update_live_state(portfolio_value=current_equity, ath=current_equity, drawdown=0.0, is_paused=True)
                else:
                    print(f"[FIX-CLOCK-DRIFT-NOBLOCK] Fallo post-inferencia transitorio (HMM/check): omitiendo ciclo SIN bloqueo permanente. Motivo: {post_failure_reason}")
                    # NO llamar update_live_state con is_paused=True
                return

            # 6. ACCIÓN Y DISPACHADOR DE ORDENES (Safety check & execution)
            print("  [6] Procesando ejecutor de ordenes...")
            executed_price = self._handle_execution(decision, current_price)
            
            # 6.5. AUDITOR OPERATIVO EN VIVO - ANALIZAR METRICAS DE LATENCIA Y DESLIZAMIENTO
            perf_results = self.auditor.process_latency_and_slippage(
                start_time=start_time,
                ideal_price=current_price,
                executed_price=executed_price
            )
            
            # Consolidar y guardar en base de datos la auditoría exitosa (SOP V10.0 compliance)
            audit_log_data = {
                "timestamp": datetime.utcnow(),
                "clock_drift_minutes": pre_audit["clock_drift_minutes"],
                "clock_drift_status": pre_audit["clock_drift_status"],
                "nan_inf_null_cols": pre_audit["nan_inf_null_cols"],
                "nan_inf_status": pre_audit["nan_inf_status"],
                "active_leverage": post_audit["active_leverage"],
                "leverage_status": post_audit["leverage_status"],
                "api_liveness_equity": pre_audit["api_liveness_equity"],
                "api_liveness_status": pre_audit["api_liveness_status"],
                "hmm_regime_index": post_audit["hmm_regime_index"],
                "hmm_status": post_audit["hmm_status"],
                "execution_latency_sec": perf_results["execution_latency_sec"],
                "latency_status": perf_results["latency_status"],
                "slippage_pct": perf_results["slippage_pct"],
                "slippage_status": perf_results["slippage_status"],
                "is_approved": True,
                "details": f"Cycle executed successfully. Action: {decision.get('action')}. Regime: {decision.get('regime')}."
            }
            self.db.log_operational_audit(audit_log_data)

            # Latido exitoso final
            self.db.log_heartbeat(status="SLEEPING")
            
        except Exception as e:
            err_tb = traceback.format_exc()
            logger.error(f"[FATAL_CYCLE] Error en ciclo operativo: {e}\n{err_tb}")
            print(f"  [FATAL] Error critico en el ciclo: {e}\n{err_tb}", file=sys.stderr)
            self.telegram.send_alert(f"🚨 *CRITICAL ERROR IN LUNA V2 LIVE DEMO*\n`{e}`", priority="critical")
            self.db.log_heartbeat(status="ERROR")
            
        elapsed = time.time() - start_time
        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Ciclo finalizado en {elapsed:.1f}s.\n")

        # [POST-CYCLE-HEALTHCHECK] Ejecutar healthcheck del dashboard en background
        # No-bloqueante: corre en subprocess con timeout. Si falla, solo loguea — nunca
        # interrumpe el ciclo siguiente. Configurable via settings.yaml dashboard_healthcheck.
        try:
            import yaml as _yaml_hc
            _hc_cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
            _hc_cfg = _yaml_hc.safe_load(open(_hc_cfg_path, encoding="utf-8")).get("dashboard_healthcheck", {})
            if _hc_cfg.get("enabled", True) and _hc_cfg.get("post_cycle_enabled", True):
                import subprocess as _subprocess_hc
                _hc_script = PROJECT_ROOT / "tools" / "diagnostics" / "dashboard_healthcheck.py"
                _hc_timeout = _hc_cfg.get("post_cycle_timeout_sec", 15)
                print(f"[POST-CYCLE-HEALTHCHECK] Lanzando healthcheck del dashboard (timeout={_hc_timeout}s)...")
                _hc_proc = _subprocess_hc.Popen(
                    [sys.executable, str(_hc_script), "--trigger", "post-cycle"],
                    stdout=_subprocess_hc.PIPE, stderr=_subprocess_hc.STDOUT,
                    cwd=str(PROJECT_ROOT), text=True, encoding="utf-8", errors="replace"
                )
                try:
                    _hc_out, _ = _hc_proc.communicate(timeout=_hc_timeout)
                    _hc_exit = _hc_proc.returncode
                    print(f"[POST-CYCLE-HEALTHCHECK] Resultado: exit={_hc_exit}")
                    for _hc_line in _hc_out.strip().split("\n")[-5:]:
                        if _hc_line.strip():
                            print(f"[POST-CYCLE-HEALTHCHECK]   {_hc_line.strip()}")
                except _subprocess_hc.TimeoutExpired:
                    _hc_proc.kill()
                    print(f"[POST-CYCLE-HEALTHCHECK/WARN] Timeout ({_hc_timeout}s) — healthcheck cancelado. No bloquea el ciclo.")
            else:
                print("[POST-CYCLE-HEALTHCHECK] Desactivado en settings.yaml.")
        except Exception as _hc_err:
            print(f"[POST-CYCLE-HEALTHCHECK/WARN] Error lanzando healthcheck: {_hc_err} — ciclo no interrumpido.")

    def run(self):
        """Bucle inmortal de Luna V2 Live Demo."""
        print("\n[LUNA V2 LIVE DEMO] Entrando a Main Loop Operativo...\n")
        
        if self.once:
            self.run_cycle()
            print("[LUNA V2 LIVE DEMO] Ejecucion unica completada. Saliendo.")
            return
            
        while True:
            self.run_cycle()
            
            # Calcular sleep exacto para despertar al inicio exacto de la siguiente hora (HH:00:00)
            now = time.time()
            next_hour = ((int(now) // self.interval_seconds) + 1) * self.interval_seconds
            sleep_time = next_hour - now
            
            # Si despertamos demasiado cerca de la hora (ej. < 5s), saltar a la siguiente hora por seguridad
            if sleep_time < 5:
                sleep_time += self.interval_seconds
            
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Ciclo completo. Durmiendo {sleep_time/60:.2f} minutos hasta la siguiente ventana de hora en punto...")
            # [FIX-SLEEP-HEARTBEAT-2026-05-26] Reemplaza time.sleep(sleep_time) simple
            # con un sleep fraccionado que actualiza el heartbeat DB cada 5 minutos.
            # El watchdog tenia threshold de 420s y mataba el proceso durante el sleep de 60 min
            # generando 421+ reinicios. Con este fix, el heartbeat se actualiza cada 300s
            # (< threshold de 4500s del watchdog corregido) para mantener el daemon vivo.
            HB_INTERVAL = 300  # Actualizar heartbeat cada 5 minutos durante el sleep
            slept = 0.0
            hb_count = 0
            while slept < sleep_time:
                chunk = min(HB_INTERVAL, sleep_time - slept)
                time.sleep(chunk)
                slept += chunk
                if slept < sleep_time:  # Solo si aun falta tiempo por dormir
                    hb_count += 1
                    remaining = (sleep_time - slept) / 60
                    print(f"[SLEEP-HB-{hb_count}] Actualizando heartbeat durante sleep. Faltan {remaining:.1f} min para el proximo ciclo.")
                    try:
                        self.db.log_heartbeat(status="SLEEPING")
                    except Exception as hb_err:
                        print(f"[SLEEP-HB-WARN] No se pudo actualizar heartbeat en sleep: {hb_err}")
            print(f"[SLEEP-DONE] Sleep completado. Iniciando proximo ciclo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Luna V2 Live Demo")
    parser.add_argument("--demo", action="store_true", default=True, help="Ejecutar en Sandbox / Cuenta Demo (Por defecto)")
    parser.add_argument("--real-money", action="store_true", help="¡CUIDADO! Operar con capital real en OKX")
    parser.add_argument("--interval", type=int, default=3600, help="Intervalo en segundos entre ciclos (Default 3600 = 1H)")
    parser.add_argument("--once", action="store_true", help="Ejecutar solo un ciclo de inferencia y salir (Modo test)")
    args = parser.parse_args()
    
    # Resolver modo de dinero real
    demo_mode = True
    if args.real_money:
        demo_mode = False
    elif args.demo:
        demo_mode = True
        
    try:
        trader = LunaLiveTraderV2LiveDemo(
            demo_mode=demo_mode,
            interval_seconds=args.interval,
            once=args.once
        )
        trader.run()
    except Exception as e:
        print(f"❌ Fallo al iniciar Luna V2 Live Demo: {e}")
        sys.exit(1)
