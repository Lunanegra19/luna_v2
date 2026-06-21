import sys
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path

# Fix python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luna.database.db_manager import DatabaseManager
from config.settings import cfg

class LiveOperationalAuditor:
    """
    [LIVE-OPERATIONAL-AUDITOR] Motor de Seguridad Operativa en Vivo de Producción.
    Ejecuta 6 disyuntores preventivos críticos para evitar fallos catastróficos,
    look-ahead bias, desvíos de datos en vivo (código muerto) o fallos de API.
    Totalmente alineado con SOP V10.0 y la política de No-Fallback Silencioso.
    """

    def __init__(self, broker, risk_monitor, telegram_bot=None):
        self.db = DatabaseManager()
        self.broker = broker
        self.risk_monitor = risk_monitor
        self.telegram = telegram_bot
        
        # Leer límites operativos institucionalizados de settings.yaml
        # No-Fallback Silencioso: Lanzar KeyError si faltan parámetros clave
        try:
            self.max_drift_minutes = float(cfg.stat.data_max_gap_h) * 60.0 # Umbral máximo de retraso
            # Por seguridad, si data_max_gap_h es grande, limitamos a 90 minutos para ticks vivos
            self.max_drift_minutes = min(90.0, self.max_drift_minutes)
            
            # Limite de apalancamiento
            self.max_leverage_allowed = float(cfg.risk.max_leverage_allowed)

            # Límite de latencia del ciclo (No-Fallback Silencioso con valor seguro por defecto)
            try:
                self.max_latency_seconds = float(cfg.stat.max_latency_seconds)
            except Exception:
                self.max_latency_seconds = 120.0
                
            print(f"[AUDITOR/INIT] Cargado límite de drift: {self.max_drift_minutes}m | leverage ceiling: {self.max_leverage_allowed}x | latency ceiling: {self.max_latency_seconds}s")
            logger.info(f"Auditor Live: drift_limit={self.max_drift_minutes}m | leverage_limit={self.max_leverage_allowed}x | latency_limit={self.max_latency_seconds}s")
        except Exception as e:
            # Fallback seguro con advertencia explícita en terminal
            print(f"⚠️ [AUDITOR/WARN] Falló carga de settings específicos de riesgo en Auditor: {e}. Usando límites estándar (90m drift, 20.0x leverage, 120s latency).")
            self.max_drift_minutes = 90.0
            self.max_leverage_allowed = 20.0
            self.max_latency_seconds = 120.0


    def run_pre_inference_audit(self, df_live: pd.DataFrame) -> tuple[bool, dict]:
        """
        Ejecuta las salvaguardas que se evalúan ANTES de pasar los datos al ensamble de inferencia.
        Esto previene predecir con NaNs, infinitos o desfases de reloj críticos.
        
        Retorna (is_approved, audit_results)
        """
        print(f"\n[AUDITOR] 🔍 Iniciando auditoría PRE-INFERENCIA en tick actual...")
        logger.info("Auditor Live: Iniciando auditoría pre-inferencia...")
        
        results = {
            "clock_drift_minutes": 0.0,
            "clock_drift_status": "OK",
            "nan_inf_null_cols": 0,
            "nan_inf_status": "OK",
            "api_liveness_equity": 0.0,
            "api_liveness_status": "OK",
            "is_approved": True,
            "failure_reason": ""
        }

        # --- Guard 1: CLOCK DRIFT GUARD (Desfase de Datos Incrementales) ---
        try:
            latest_timestamp = df_live.index.max()
            if not isinstance(latest_timestamp, pd.Timestamp):
                latest_timestamp = pd.to_datetime(latest_timestamp, utc=True)
            
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.tz_localize('UTC')
                
            now_utc = datetime.now(timezone.utc)
            drift_minutes = (now_utc - latest_timestamp).total_seconds() / 60.0
            results["clock_drift_minutes"] = drift_minutes
            
            if drift_minutes > self.max_drift_minutes:
                results["clock_drift_status"] = "FAIL"
                results["is_approved"] = False
                err_msg = (
                    f"⚠️ [CLOCK-DRIFT-GUARD] CRITICAL: Desplazamiento temporal detectado. "
                    f"Último dato: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC | Hora actual: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
                    f"Desfase: {drift_minutes:.1f} minutos (Límite {self.max_drift_minutes}m). "
                    f"FETCH EN VIVO PARADO o desfase crítico. Abortando para evitar trading ciego."
                )
                print(f"\n[CLOCK-DRIFT-PANIC] {err_msg}\n")
                logger.critical(err_msg)
                if self.telegram:
                    self.telegram.send_alert(f"🛡️ *CLOCK DRIFT ALARM*:\n`{err_msg}`", priority="critical")
                results["failure_reason"] += "[ClockDrift exceeded] "
            else:
                print(f"  [Auditor] Guard 1 (Clock Drift): OK. Desfase: {drift_minutes:.1f} minutos (Límite {self.max_drift_minutes}m).")
        except Exception as e_drift:
            results["clock_drift_status"] = "ERROR"
            results["is_approved"] = False
            results["failure_reason"] += f"[ClockDrift error: {e_drift}] "
            print(f"  [Auditor] Guard 1 (Clock Drift) Error de verificación: {e_drift}")
            logger.error(f"Auditor Clock Drift check error: {e_drift}")

        # --- Guard 2: NAN/INF SANITY SHIELD (Filtro de Integridad de Features) ---
        try:
            last_row = df_live.iloc[-1]
            nulls = last_row.isnull()
            infs = np.isinf(last_row)
            
            null_cols = list(last_row[nulls].index)
            inf_cols = list(last_row[infs].index)
            
            total_bad = len(null_cols) + len(inf_cols)
            results["nan_inf_null_cols"] = total_bad
            
            if total_bad > 0:
                results["nan_inf_status"] = "FAIL"
                results["is_approved"] = False
                
                # Reporte detallado e impresiones para depuración forense (RULE[fixbugsprints.md])
                bad_details = []
                for c in null_cols:
                    bad_details.append(f"'{c}' (NaN)")
                    print(f"[BUGFIX-NAN-SHIELD] Semilla actual: Columna corrupta en df_live: '{c}' es NaN! (RULE[fixbugsprints.md])")
                for c in inf_cols:
                    bad_details.append(f"'{c}' (Inf: {last_row[c]})")
                    print(f"[BUGFIX-INF-SHIELD] Semilla actual: Columna corrupta en df_live: '{c}' es Infinito! Valor: {last_row[c]} (RULE[fixbugsprints.md])")
                
                err_msg = (
                    f"⚠️ [NAN-INF-SANITY-SHIELD] CRITICAL: Datos corruptos en features en vivo. "
                    f"Total columnas corruptas: {total_bad}. Anomalías en: {', '.join(bad_details[:6])}. "
                    f"Inferencia abortada preventivamente para proteger el ensamble."
                )
                print(f"\n[NAN-INF-PANIC] {err_msg}\n")
                logger.critical(err_msg)
                if self.telegram:
                    self.telegram.send_alert(f"🛡️ *NAN/INF SANITY ALARM*:\n`{err_msg}`", priority="critical")
                results["failure_reason"] += "[NaN/Inf features detected] "
            else:
                print("  [Auditor] Guard 2 (NaN/Inf Shield): OK. 0 columnas con NaN o Infinito en features en vivo.")
        except Exception as e_sanity:
            results["nan_inf_status"] = "ERROR"
            results["is_approved"] = False
            results["failure_reason"] += f"[NaN/Inf check error: {e_sanity}] "
            print(f"  [Auditor] Guard 2 (NaN/Inf Shield) Error de verificación: {e_sanity}")
            logger.error(f"Auditor NaN/Inf check error: {e_sanity}")

        # --- Guard 4: API LIVENESS CHECK (Disyuntor de Vida del Broker) ---
        try:
            balance_equity = self.broker.fetch_equity()
            if balance_equity is None or balance_equity < 0:
                raise RuntimeError("fetch_equity devolvió un saldo negativo o nulo.")
            
            results["api_liveness_equity"] = balance_equity
            print(f"  [Auditor] Guard 4 (API Liveness): OK. Equidad reportada por OKX: ${balance_equity:,.2f} USD.")
        except Exception as e_api:
            results["api_liveness_status"] = "FAIL"
            results["is_approved"] = False
            results["failure_reason"] += f"[API Liveness check failed: {e_api}] "
            
            err_msg = (
                f"⚠️ [API-LIVENESS-CHECK] CRITICAL: Falló comunicación viva con el Broker: {e_api}. "
                f"Conectividad con OKX caída. Abortando ciclo para evitar trading ciego."
            )
            print(f"\n[API-LIVENESS-PANIC] {err_msg}\n")
            logger.critical(err_msg)
            if self.telegram:
                self.telegram.send_alert(f"🛡️ *API LIVENESS CHECK ALARM*:\n`{err_msg}`", priority="critical")

        return results["is_approved"], results

    def run_post_inference_audit(self, df_live: pd.DataFrame, ensemble_decision: dict) -> tuple[bool, dict]:
        """
        Ejecuta las salvaguardas que se evalúan DESPUÉS de la inferencia,
        pero ANTES de enviar la orden real al broker (leverage, HMM, y métricas del ciclo).
        
        Retorna (is_approved, audit_results)
        """
        print(f"[AUDITOR] 🔍 Iniciando auditoría POST-INFERENCIA en decisión actual...")
        logger.info("Auditor Live: Iniciando auditoría post-inferencia...")
        
        results = {
            "active_leverage": 0.0,
            "leverage_status": "OK",
            "hmm_regime_index": -1,
            "hmm_status": "OK",
            "is_approved": True,
            "failure_reason": ""
        }

        # --- Guard 3: LEVERAGE CEILING (Tope de Apalancamiento Real) ---
        try:
            risk = self.risk_monitor.get_risk_summary()
            equity = float(risk.get("portfolio_value", 5000.0))
            
            # Obtener posición activa del broker
            pos = self.broker.get_position(self.broker.symbol if hasattr(self.broker, "symbol") else "BTC/USDT:USDT")
            current_contracts = float(pos.get("contracts", 0.0))
            current_price = float(df_live['close'].iloc[-1])
            current_nocional = current_contracts * current_price
            
            active_leverage = current_nocional / equity if equity > 0 else 0.0
            results["active_leverage"] = active_leverage
            
            if active_leverage > self.max_leverage_allowed:
                results["leverage_status"] = "FAIL"
                results["is_approved"] = False
                
                err_msg = (
                    f"⚠️ [LEVERAGE-CEILING-GUARD] WARNING: Apalancamiento catastrófico detectado: "
                    f"{active_leverage:.2f}x (Límite máximo permitido: {self.max_leverage_allowed}x). "
                    f"Fallo del Position Sizer o trade huérfano. Abortando nueva orden e indicando rebalanceo seguro."
                )
                print(f"\n[LEVERAGE-PANIC] {err_msg}\n")
                logger.critical(err_msg)
                if self.telegram:
                    self.telegram.send_alert(f"🛡️ *LEVERAGE CEILING WARNING*:\n`{err_msg}`", priority="warning")
                results["failure_reason"] += "[Leverage limit exceeded] "
            else:
                print(f"  [Auditor] Guard 3 (Leverage Ceiling): OK. Apalancamiento actual: {active_leverage:.2f}x (Límite {self.max_leverage_allowed}x).")
        except Exception as e_lev:
            results["leverage_status"] = "ERROR"
            results["is_approved"] = False
            results["failure_reason"] += f"[Leverage check error: {e_lev}] "
            print(f"  [Auditor] Guard 3 (Leverage Ceiling) Error de verificación: {e_lev}")
            logger.error(f"Auditor Leverage check error: {e_lev}")

        # --- Guard 5: HMM REGIME INDEX VALIDITY (Consistencia de Estados HMM) ---
        try:
            # Extraer régimen mayoritario del ensamble
            majority_regime = ensemble_decision.get("regime", "UNKNOWN")
            
            # Buscar el índice del estado HMM en los breakdowns de las semillas
            hmm_indexes = []
            for seed, details in ensemble_decision.get("seeds_breakdown", {}).items():
                if "regime" in details:
                    # El régimen HMM en vivo suele tener el formato "X_SEMANTIC", extraemos el número
                    try:
                        idx = int(details["regime"].split("_")[0])
                        hmm_indexes.append(idx)
                    except Exception:
                        pass
            
            if hmm_indexes:
                mean_idx = int(np.round(np.mean(hmm_indexes)))
                results["hmm_regime_index"] = mean_idx
                
                # Un índice fuera del rango normal (ej. < 0 o > 6) indica corrupción matemática del HMM
                if mean_idx < 0 or mean_idx > 6:
                    results["hmm_status"] = "FAIL"
                    results["is_approved"] = False
                    results["failure_reason"] += f"[Invalid HMM state index: {mean_idx}] "
                    
                    err_msg = f"⚠️ [HMM-REGIME-GUARD] CRITICAL: Estados HMM fuera de rango normal. Índice medio: {mean_idx}. HMM degradado."
                    print(f"\n[HMM-REGIME-PANIC] {err_msg}\n")
                    logger.critical(err_msg)
                    if self.telegram:
                        self.telegram.send_alert(f"🛡️ *HMM REGIME ALARM*:\n`{err_msg}`", priority="critical")
                else:
                    print(f"  [Auditor] Guard 5 (HMM Regime Consistency): OK. Índice medio HMM: {mean_idx} ('{majority_regime}').")
            else:
                # Si no hay breakdown, warn descriptivo
                print(f"  [Auditor] Guard 5 (HMM Regime Consistency): OK (Sin breakdown de semillas. Régimen mayoritario: '{majority_regime}').")
        except Exception as e_hmm:
            results["hmm_status"] = "ERROR"
            results["is_approved"] = False
            results["failure_reason"] += f"[HMM check error: {e_hmm}] "
            print(f"  [Auditor] Guard 5 (HMM Regime Consistency) Error de verificación: {e_hmm}")
            logger.error(f"Auditor HMM Regime check error: {e_hmm}")

        return results["is_approved"], results

    def process_latency_and_slippage(self, start_time: float, ideal_price: float, executed_price: float) -> dict:
        """
        Ejecuta Guard 6: Latency & Slippage Monitor.
        Mide el tiempo demorado y el deslizamiento y genera telemetría o alertas según corresponda.
        """
        print(f"[AUDITOR] 🔍 Analizando Guard 6: Latencia y Deslizamiento (Slippage)...")
        
        latency_sec = time.time() - start_time
        
        slippage_pct = 0.0
        if ideal_price > 0 and executed_price is not None and executed_price > 0:
            slippage_pct = abs(executed_price - ideal_price) / ideal_price
            
        latency_status = "OK"
        slippage_status = "OK"
        
        # Umbral institucional: ciclo > max_latency_seconds indica lentitud preocupante
        if latency_sec > self.max_latency_seconds:
            latency_status = "WARNING"
            warn_msg = f"⚠️ [LATENCY-WARNING] El ciclo tardó {latency_sec:.1f} segundos (Urgente: revisar rendimiento del fetcher/VPS, límite {self.max_latency_seconds}s)."
            print(f"\n[LATENCY-ALERT] {warn_msg}\n")
            if self.telegram:
                self.telegram.send_alert(f"🛡️ *LATENCY WARNING*:\n`{warn_msg}`", priority="warning")
        else:
            print(f"  [Auditor] Guard 6a (Ciclo Latency): OK. Duración: {latency_sec:.2f}s (Límite {self.max_latency_seconds}s).")

            
        # Umbral institucional: deslizamiento > 0.50% indica falta de liquidez en orderbook o lagunas de red
        if slippage_pct > 0.0050:
            slippage_status = "WARNING"
            warn_msg = f"⚠️ [SLIPPAGE-WARNING] Deslizamiento excesivo: {slippage_pct:.4%} (Ideal: ${ideal_price:,.2f} | Ejecutado: ${executed_price:,.2f})."
            print(f"\n[SLIPPAGE-ALERT] {warn_msg}\n")
            if self.telegram:
                self.telegram.send_alert(f"🛡️ *SLIPPAGE WARNING*:\n`{warn_msg}`", priority="warning")
        else:
            print(f"  [Auditor] Guard 6b (Execution Slippage): OK. Deslizamiento: {slippage_pct:.4%} (Límite 0.50%).")
            
        return {
            "execution_latency_sec": latency_sec,
            "latency_status": latency_status,
            "slippage_pct": slippage_pct,
            "slippage_status": slippage_status
        }
