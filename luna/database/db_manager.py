import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor
from contextlib import contextmanager
from datetime import datetime
from dotenv import load_dotenv

# Asegurar carga de variables de entorno
load_dotenv()

class DatabaseManager:
    """
    Gestor de Base de Datos PostgreSQL para Luna V1 Live Orchestration.
    Cumple con SOP R12: Integridad A.C.I.D. estricta mediante Context Managers
    y Connection Pooling para evitar Thebordamientos (memory leaks) o bloqueos
    en el bucle de producción.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.connection_pool = None
            cls._instance._init_pool()
        elif cls._instance.connection_pool is None:
            print("[DB] Re-intentando inicialización del connection pool (puerto dinámico ahora disponible)...")
            cls._instance._init_pool()
        return cls._instance
        
    def _init_pool(self):
        """Inicializa el pool de conexiones."""
        db_url = os.getenv("DATABASE_URL")
        # Fallback local o usar parámetros Theglosados si no hay URL
        if not db_url:
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "5432")
            db_name = os.getenv("DB_NAME", "luna_db")
            db_user = os.getenv("DB_USER", "postgres")
            db_pass = os.getenv("DB_PASS", "postgres")
            db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            
        try:
            # Pool mínimo de 1, máximo de 10 conexiones simultáneas
            self.connection_pool = pool.SimpleConnectionPool(1, 10, db_url)
            if self.connection_pool:
                print("[DB] Connection pool creado con exito.")
                self._initialize_schema()
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"[!] Error conectando a PostgreSQL (Pool Initialization): {error}")
            self.connection_pool = None

    @contextmanager
    def get_connection(self):
        """
        SOP R12: Context Manager estricto para asegurar que las conexiones 
        siempre se devuelvan al pool, incluso si ocurre una excepcion.
        """
        if not self.connection_pool:
            raise Exception("Connection pool no ha sido inicializado.")
            
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)

    def _initialize_schema(self):
        """Crea las tablas maestras si no existen."""
        create_tables_sql = """
        -- 1. Tabla de Latidos (Dead Man's Switch)
        CREATE TABLE IF NOT EXISTS system_heartbeat (
            id SERIAL PRIMARY KEY,
            component VARCHAR(50) UNIQUE NOT NULL,
            last_heartbeat TIMESTAMP NOT NULL,
            status VARCHAR(20) NOT NULL
        );

        -- 2. Tabla de Auditoria de Decisiones
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            price NUMERIC(15, 2) NOT NULL,
            action VARCHAR(10) NOT NULL,
            confidence NUMERIC(6, 4),
            xgb_prob NUMERIC(6, 4),
            hmm_regime SMALLINT,
            reason TEXT,
            contracts INT DEFAULT 0,
            executed_price NUMERIC(15, 2)
        );

        -- 3. Tabla de Estado y Riesgo (para reiniciar desde fallos)
        CREATE TABLE IF NOT EXISTS live_state (
            id INT PRIMARY KEY DEFAULT 1,
            portfolio_value NUMERIC(15, 2) NOT NULL,
            ath NUMERIC(15, 2) NOT NULL,
            drawdown NUMERIC(6, 4) DEFAULT 0.0,
            is_paused BOOLEAN DEFAULT FALSE,
            -- Fix M-06: equity de inicio de día/semana para circuit breakers diarios/semanales
            equity_start_day  NUMERIC(15, 2) DEFAULT NULL,
            equity_start_week NUMERIC(15, 2) DEFAULT NULL,
            day_reset_date    DATE DEFAULT NULL,
            week_reset_date   DATE DEFAULT NULL,
            updated_at TIMESTAMP NOT NULL
        );

        -- 4. Tabla de Reconciliacion PnL
        CREATE TABLE IF NOT EXISTS reconciliation_log (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            db_pnl NUMERIC(15, 2) NOT NULL,
            exchange_pnl NUMERIC(15, 2) NOT NULL,
            delta_pct NUMERIC(6, 4) NOT NULL,
            status VARCHAR(10) NOT NULL
        );

        -- 5. Tabla de Auditoría de Seguridad Operativa en Vivo
        CREATE TABLE IF NOT EXISTS operational_audit_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            clock_drift_minutes NUMERIC(8, 2) NOT NULL,
            clock_drift_status VARCHAR(20) NOT NULL,
            nan_inf_null_cols INT NOT NULL,
            nan_inf_status VARCHAR(20) NOT NULL,
            active_leverage NUMERIC(6, 2) NOT NULL,
            leverage_status VARCHAR(20) NOT NULL,
            api_liveness_equity NUMERIC(15, 2),
            api_liveness_status VARCHAR(20) NOT NULL,
            hmm_regime_index SMALLINT,
            hmm_status VARCHAR(20) NOT NULL,
            execution_latency_sec NUMERIC(6, 2),
            latency_status VARCHAR(20) NOT NULL,
            slippage_pct NUMERIC(8, 6),
            slippage_status VARCHAR(20) NOT NULL,
            is_approved BOOLEAN NOT NULL,
            details TEXT
        );

        -- 6. [FIX-P4-HB-HISTORY] Historial completo de heartbeats para trazabilidad forense.
        -- system_heartbeat solo guarda el estado actual (1 fila, UPSERT).
        -- Esta tabla guarda cada latido con timestamp para detectar gaps silenciosos retrospectivamente.
        CREATE TABLE IF NOT EXISTS system_heartbeat_history (
            id SERIAL PRIMARY KEY,
            component VARCHAR(50) NOT NULL,
            heartbeat_ts TIMESTAMP NOT NULL,
            status VARCHAR(20) NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_hb_history_component_ts
            ON system_heartbeat_history (component, heartbeat_ts DESC);
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_tables_sql)
                conn.commit()
                print("[DB] Esquema inicializado/verificado.")
                
                # Insertar el registro base para live_state si no existe
                with conn.cursor() as cur:
                    # Asegurar que existen columnas M-06 en DB existente (migracion suave)
                    cur.execute("""
                        ALTER TABLE live_state
                        ADD COLUMN IF NOT EXISTS equity_start_day  NUMERIC(15, 2) DEFAULT NULL,
                        ADD COLUMN IF NOT EXISTS equity_start_week NUMERIC(15, 2) DEFAULT NULL,
                        ADD COLUMN IF NOT EXISTS day_reset_date    DATE DEFAULT NULL,
                        ADD COLUMN IF NOT EXISTS week_reset_date   DATE DEFAULT NULL;
                    """)
                    cur.execute("SELECT 1 FROM live_state WHERE id = 1")
                    if not cur.fetchone():
                        cur.execute("""
                            INSERT INTO live_state (id, portfolio_value, ath, updated_at)
                            VALUES (1, 5000.0, 5000.0, %s)
                        """, (datetime.utcnow(),))
                    
                    # Insertar el registro base para heartbeat
                    cur.execute("SELECT 1 FROM system_heartbeat WHERE component = 'luna_v2_live_demo'")
                    if not cur.fetchone():
                        cur.execute("""
                            INSERT INTO system_heartbeat (component, last_heartbeat, status) 
                            VALUES ('luna_v2_live_demo', %s, 'INITIALIZED')
                        """, (datetime.utcnow(),))
                conn.commit()

        except Exception as e:
            print(f"[!] Error inicializando el esquema: {e}")

    # --- METODOS DE TELEMETRIA Y HEARTBEAT ---
    
    def log_heartbeat(self, status: str = "ONLINE"):
        """Registra el latido (Heartbeat) del demonio principal.
        [FIX-P4-HB-HISTORY] Doble escritura:
          1. UPDATE en system_heartbeat (estado actual, UPSERT clásico)
          2. INSERT en system_heartbeat_history (historial forense completo)
        Esto permite detectar gaps silenciosos retrospectivamente auditando
        la tabla de historial sin perder el estado actual del watchdog.
        """
        ts_now = datetime.utcnow()
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # 1. UPDATE estado actual (mantiene retrocompatibilidad con watchdog)
                    cur.execute("""
                        UPDATE system_heartbeat 
                        SET last_heartbeat = %s, status = %s 
                        WHERE component = 'luna_v2_live_demo'
                    """, (ts_now, status))
                    # 2. [FIX-P4-HB-HISTORY] INSERT en historial forense
                    cur.execute("""
                        INSERT INTO system_heartbeat_history (component, heartbeat_ts, status)
                        VALUES ('luna_v2_live_demo', %s, %s)
                    """, (ts_now, status))
                conn.commit()
            print(f"[FIX-P4-HB-HISTORY] Heartbeat registrado: status={status} | ts={ts_now.strftime('%H:%M:%S')} UTC")
        except Exception as e:
            print(f"[!] DB Error en log_heartbeat: {e}")

    def get_last_heartbeat(self, component: str = 'luna_v2_live_demo') -> datetime:
        """Lee The ultimo latido (usado por el watchdog de sistema)."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT last_heartbeat FROM system_heartbeat WHERE component = %s", (component,))
                    res = cur.fetchone()
                    return res[0] if res else None
        except Exception as e:
            print(f"[!] DB Error en get_last_heartbeat: {e}")
            return None

    # --- METODOS DE AUDITORIA ---
    
    def log_audit(self, timestamp: datetime, price: float, action: str, confidence: float, xgb_prob: float, hmm_regime: int, reason: str, contracts: int = 0, executed_price: float = None):
        """Registra the log de auditoria de cada iteracion del modelo."""
        # 1. Intentar sincronizar datos pendientes offline en segundo plano
        self.sync_offline_audits()
        
        # 2. Intentar logear en base de datos
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO audit_logs 
                        (timestamp, price, action, confidence, xgb_prob, hmm_regime, reason, contracts, executed_price) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (timestamp, price, action, confidence, xgb_prob, hmm_regime, reason, contracts, executed_price))
                conn.commit()
        except Exception as e:
            print(f"[!] DB Error en log_audit: {e}. Desviando a caché local offline...")
            self._save_audit_to_local_cache(timestamp, price, action, confidence, xgb_prob, hmm_regime, reason, contracts, executed_price)

    def log_operational_audit(self, audit_data: dict):
        """
        [LIVE-OPERATIONAL-AUDITOR] Registra las métricas de la auditoría en la base de datos PostgreSQL.
        Soporta cola/caché local offline automática si el pool está bloqueado.
        """
        self.sync_offline_operational_audits()
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO operational_audit_logs 
                        (timestamp, clock_drift_minutes, clock_drift_status, nan_inf_null_cols, nan_inf_status, 
                         active_leverage, leverage_status, api_liveness_equity, api_liveness_status, 
                         hmm_regime_index, hmm_status, execution_latency_sec, latency_status, 
                         slippage_pct, slippage_status, is_approved, details) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        audit_data.get("timestamp", datetime.utcnow()),
                        audit_data.get("clock_drift_minutes", 0.0),
                        audit_data.get("clock_drift_status", "OK"),
                        audit_data.get("nan_inf_null_cols", 0),
                        audit_data.get("nan_inf_status", "OK"),
                        audit_data.get("active_leverage", 0.0),
                        audit_data.get("leverage_status", "OK"),
                        audit_data.get("api_liveness_equity", 0.0),
                        audit_data.get("api_liveness_status", "OK"),
                        audit_data.get("hmm_regime_index", -1),
                        audit_data.get("hmm_status", "OK"),
                        audit_data.get("execution_latency_sec", 0.0),
                        audit_data.get("latency_status", "OK"),
                        audit_data.get("slippage_pct", 0.0),
                        audit_data.get("slippage_status", "OK"),
                        audit_data.get("is_approved", True),
                        audit_data.get("details", "")
                    ))
                conn.commit()
                print("✨ [LIVE-TRADER-AUDIT] Persistidas métricas de seguridad operativa en base de datos PostgreSQL.")
        except Exception as e:
            print(f"[!] DB Error en log_operational_audit: {e}. Desviando a caché local offline...")
            self._save_operational_audit_to_local_cache(audit_data)

    def _save_operational_audit_to_local_cache(self, audit_data: dict):
        try:
            import json
            from pathlib import Path
            cache_file = Path("data/cache") / "offline_operational_audit_logs.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            records = []
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        records = json.load(f)
                except Exception:
                    records = []
            
            ts = audit_data.get("timestamp", datetime.utcnow())
            serializable_data = audit_data.copy()
            serializable_data["timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            records.append(serializable_data)
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"🌙 [DB-CACHE-OK] Grabada auditoría operativa localmente en caché. Total en cola: {len(records)}")
        except Exception as cache_err:
            print(f"[!] Fallo crítico al escribir caché local de auditoría operativa: {cache_err}")

    def sync_offline_operational_audits(self):
        try:
            import json
            from pathlib import Path
            cache_file = Path("data/cache") / "offline_operational_audit_logs.json"
            if not cache_file.exists():
                return
                
            print("🔄 [DB-SYNC] Encontrado archivo de caché de auditoría operativa. Sincronizando...")
            with open(cache_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                
            if not records:
                if cache_file.exists():
                    os.unlink(cache_file)
                return
                
            success_count = 0
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for rec in records:
                        cur.execute("""
                            INSERT INTO operational_audit_logs 
                            (timestamp, clock_drift_minutes, clock_drift_status, nan_inf_null_cols, nan_inf_status, 
                             active_leverage, leverage_status, api_liveness_equity, api_liveness_status, 
                             hmm_regime_index, hmm_status, execution_latency_sec, latency_status, 
                             slippage_pct, slippage_status, is_approved, details) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            datetime.fromisoformat(rec["timestamp"]),
                            rec.get("clock_drift_minutes", 0.0),
                            rec.get("clock_drift_status", "OK"),
                            rec.get("nan_inf_null_cols", 0),
                            rec.get("nan_inf_status", "OK"),
                            rec.get("active_leverage", 0.0),
                            rec.get("leverage_status", "OK"),
                            rec.get("api_liveness_equity", 0.0),
                            rec.get("api_liveness_status", "OK"),
                            rec.get("hmm_regime_index", -1),
                            rec.get("hmm_status", "OK"),
                            rec.get("execution_latency_sec", 0.0),
                            rec.get("latency_status", "OK"),
                            rec.get("slippage_pct", 0.0),
                            rec.get("slippage_status", "OK"),
                            rec.get("is_approved", True),
                            rec.get("details", "")
                        ))
                        success_count += 1
                conn.commit()
            
            if success_count == len(records):
                os.unlink(cache_file)
                print(f"🔄 [DB-SYNC-SUCCESS] ¡Sincronizados {success_count} registros de auditoría operativa con éxito!")
        except Exception as sync_err:
            print(f"[!] Fallo al sincronizar caché local de auditoría operativa: {sync_err}. Se mantendrán los datos offline.")

    def _save_audit_to_local_cache(self, timestamp: datetime, price: float, action: str, confidence: float, xgb_prob: float, hmm_regime: int, reason: str, contracts: int, executed_price: float):
        try:
            import json
            from pathlib import Path
            cache_dir = Path("data/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "offline_audit_logs.json"
            
            records = []
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        records = json.load(f)
                except Exception:
                    records = []
            
            new_record = {
                "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
                "price": price,
                "action": action,
                "confidence": confidence,
                "xgb_prob": xgb_prob,
                "hmm_regime": hmm_regime,
                "reason": f"[OFFLINE-CACHED] {reason}",
                "contracts": contracts,
                "executed_price": executed_price
            }
            records.append(new_record)
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"🌙 [DB-CACHE-OK] Grabada decisión horaria localmente en caché por fallo de DB. Total en cola: {len(records)}")
        except Exception as cache_err:
            print(f"[!] Fallo crítico al escribir caché local de base de datos: {cache_err}")

    def sync_offline_audits(self):
        try:
            import json
            from pathlib import Path
            cache_file = Path("data/cache") / "offline_audit_logs.json"
            if not cache_file.exists():
                return
            
            print("🔄 [DB-SYNC] Encontrado archivo de caché local. Intentando sincronizar con base de datos remota...")
            with open(cache_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            
            if not records:
                if cache_file.exists():
                    os.unlink(cache_file)
                return
                
            success_count = 0
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for rec in records:
                        cur.execute("""
                            INSERT INTO audit_logs 
                            (timestamp, price, action, confidence, xgb_prob, hmm_regime, reason, contracts, executed_price) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            datetime.fromisoformat(rec["timestamp"]),
                            rec["price"],
                            rec["action"],
                            rec["confidence"],
                            rec["xgb_prob"],
                            rec["hmm_regime"],
                            rec["reason"],
                            rec["contracts"],
                            rec["executed_price"]
                        ))
                        success_count += 1
                conn.commit()
            
            if success_count == len(records):
                os.unlink(cache_file)
                print(f"🔄 [DB-SYNC-SUCCESS] ¡Sincronizados {success_count} registros de auditoría offline con éxito!")
        except Exception as sync_err:
            print(f"[!] Fallo al sincronizar caché local con base de datos: {sync_err}. Se mantendrán los datos offline.")

    # --- METODOS DE ESTADO Y RIESGO ---

    def get_live_state(self):
        """Recupera el estado actual del portfolio para calculo de Drawdown y Sizing."""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute("SELECT portfolio_value, ath, drawdown, is_paused FROM live_state WHERE id = 1")
                    return dict(cur.fetchone()) if cur.rowcount > 0 else None
        except Exception as e:
            print(f"[!] DB Error en get_live_state: {e}")
            return None

    def update_live_state(self, portfolio_value: float, ath: float, drawdown: float, is_paused: bool):
        """Actualiza el estado consolidado de riesgo."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE live_state 
                        SET portfolio_value = %s, ath = %s, drawdown = %s, is_paused = %s, updated_at = %s 
                        WHERE id = 1
                    """, (portfolio_value, ath, drawdown, is_paused, datetime.utcnow()))
                conn.commit()
        except Exception as e:
            print(f"[!] DB Error en update_live_state: {e}")

    # --- FIX M-06: MÉTODOS PARA CIRCUIT BREAKERS DIARIO/SEMANAL ---

    def get_period_equity(self) -> dict:
        """Lee equity_start_day y equity_start_week para cálculo de DD intra-período."""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute("""
                        SELECT equity_start_day, equity_start_week,
                               day_reset_date, week_reset_date
                        FROM live_state WHERE id = 1
                    """)
                    row = cur.fetchone()
                    return dict(row) if row else {}
        except Exception as e:
            print(f"[!] DB Error en get_period_equity: {e}")
            return {}

    def reset_period_equity(self, portfolio_value: float, reset_day: bool = False, reset_week: bool = False):
        """Actualiza equity de inicio de día y/o semana (llamar al comenzar cada período)."""
        from datetime import date, timedelta
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if reset_day:
                        cur.execute("""
                            UPDATE live_state
                            SET equity_start_day = %s, day_reset_date = %s
                            WHERE id = 1
                        """, (portfolio_value, today))
                    if reset_week:
                        cur.execute("""
                            UPDATE live_state
                            SET equity_start_week = %s, week_reset_date = %s
                            WHERE id = 1
                        """, (portfolio_value, week_start))
                conn.commit()
        except Exception as e:
            print(f"[!] DB Error en reset_period_equity: {e}")
            
    # --- METODOS DE RECONCILIACION ---
    
    def log_reconciliation(self, db_pnl: float, exchange_pnl: float, delta_pct: float, status: str):
        """Guarda un registro del cotejo entre el libro contable de la BD y la realidad the Exchange."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO reconciliation_log 
                        (timestamp, db_pnl, exchange_pnl, delta_pct, status) 
                        VALUES (%s, %s, %s, %s, %s)
                    """, (datetime.utcnow(), db_pnl, exchange_pnl, delta_pct, status))
                conn.commit()
        except Exception as e:
            print(f"[!] DB Error en log_reconciliation: {e}")

    # --- CIERRE ---
    
    def close_pool(self):
        """Cierra todas las conexiones limpiamente al finalizar."""
        if self.connection_pool:
            self.connection_pool.closeall()
            print("[DB] Connection pool cerrado.")
