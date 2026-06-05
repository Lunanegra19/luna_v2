"""
[AUDIT-INSTITUCIONAL-V1] Auditoría profunda del sistema live Luna V2 en VPS.
Inspecciona 12 vectores de riesgo silenciosos:
  1. Zombie trades (posiciones OKX vs DB desalineadas)
  2. Heartbeat gaps (periodos sin latido = sistema colgado silenciosamente)
  3. Reconciliation history (grandes discrepancias históricas)
  4. Audit log patterns (ciclos fallidos, motivos, frecuencia)
  5. Memory leak tracking (proceso PM2 creciendo sin límite)
  6. NTP / clock skew del VPS
  7. Disco y filesystem health
  8. DB connection pool health
  9. Telegram listener alive (¿responde comandos?)
  10. RegimeRouter dead-code (prob=0.0 en todas las barras = modelo zombie)
  11. Reconciliation deltas históricos (BD vs OKX)
  12. Crash pattern en reinicios PM2 (10 reinicios actuales)
"""

import psycopg2
import os
import sys
import json
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

DATABASE_URL = "postgresql://luna_user:luna_secure_pass@localhost:5432/luna_db"
PROJECT_ROOT = Path("/root/luna_v2")
FEATURES_PATH = PROJECT_ROOT / "data/features/features_live.parquet"

# Añadir al path para imports de luna
sys.path.insert(0, str(PROJECT_ROOT))

SEPARATOR = "=" * 70
WARN = "⚠️ "
OK   = "✅ "
CRIT = "🔴 "
INFO = "ℹ️  "

issues = []   # Lista de problemas encontrados
warnings = [] # Lista de advertencias

def flag_issue(level, code, msg):
    if level == "CRIT":
        issues.append(f"[{code}] {msg}")
        print(f"{CRIT} CRÍTICO [{code}]: {msg}")
    elif level == "WARN":
        warnings.append(f"[{code}] {msg}")
        print(f"{WARN} ADVERTENCIA [{code}]: {msg}")
    else:
        print(f"{OK} OK [{code}]: {msg}")

print(SEPARATOR)
print("[AUDIT-INSTITUCIONAL] Iniciando auditoría completa del sistema live")
print(f"[AUDIT-INSTITUCIONAL] Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(SEPARATOR)

# =========================================================
# VECTOR 1: ZOMBIE TRADES — OKX vs DB desalineados
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V1] ZOMBIE TRADES: Comparación posición OKX vs DB")
print(f"{'─'*60}")
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from luna.live.okx_connector import OKXBrokerConnector
    from luna.database.db_manager import DatabaseManager

    db = DatabaseManager()
    okx = OKXBrokerConnector(demo_mode=True)

    # Estado DB
    db_state = db.get_live_state()
    db_portfolio = float(db_state.get('portfolio_value', 0)) if db_state else 0
    db_is_paused = bool(db_state.get('is_paused', False)) if db_state else False

    # Última acción registrada en DB
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT timestamp, action, contracts, price FROM audit_logs ORDER BY timestamp DESC LIMIT 1")
    last_action_row = cur.fetchone()
    db_last_action = last_action_row[1] if last_action_row else "UNKNOWN"
    db_last_contracts = float(last_action_row[2]) if last_action_row else 0
    db_last_ts = last_action_row[0] if last_action_row else None
    conn.close()

    # Estado OKX real
    symbol = os.getenv('OKX_TRADING_SYMBOL', 'BTC/USDT:USDT')
    pos = okx.get_position(symbol)
    okx_side = pos.get('side', 'HOLD')
    okx_contracts = float(pos.get('contracts', 0))
    okx_pnl = float(pos.get('unrealized_pnl', 0))

    print(f"  DB última acción: {db_last_action} ({db_last_contracts} contratos) | {db_last_ts}")
    print(f"  OKX posición real: {okx_side} ({okx_contracts} contratos) | PnL no realizado: ${okx_pnl:.2f}")
    print(f"  DB portfolio_value: ${db_portfolio:,.2f} | is_paused: {db_is_paused}")

    # Detectar zombie
    is_zombie = False
    if okx_side != "HOLD" and db_last_action == "HOLD":
        is_zombie = True
        flag_issue("CRIT", "V1-ZOMBIE", f"POSICIÓN ZOMBIE: OKX tiene {okx_side} ({okx_contracts} conts) pero DB dice último HOLD. Posición huérfana sin tracking!")
    elif okx_side == "HOLD" and db_last_action not in ("HOLD", "UNKNOWN") and db_last_contracts > 0:
        flag_issue("WARN", "V1-GHOST", f"POSICIÓN FANTASMA: DB registra {db_last_action} ({db_last_contracts} conts) pero OKX no tiene posición abierta. Trade puede haber sido liquidado sin notificación.")
    elif okx_side == db_last_action or (okx_side == "HOLD" and db_last_action == "HOLD"):
        flag_issue("OK", "V1-ALIGNED", f"Posición alineada: OKX={okx_side} | DB={db_last_action}")

    # Verificar si 1.0 BTC siempre presente (BUGFIX-DEMO-BOOT repetido)
    print(f"\n  [V1-DEMO-BOOT] Verificando balance BTC en cuenta Demo...")
    try:
        balance = okx.exchange.fetch_balance()
        btc_total = balance.get('BTC', {}).get('total', 0)
        print(f"  [V1-DEMO-BOOT] BTC en cuenta OKX Demo: {btc_total} BTC")
        if btc_total > 0.5:
            flag_issue("WARN", "V1-DEMO-BTC", f"1.0 BTC estático en cuenta Demo OKX ({btc_total} BTC). BUGFIX-DEMO-BOOT activo — puede enmascarar posiciones reales en futuros. Verificar que no se confunda con posición activa.")
    except Exception as e_bal:
        flag_issue("WARN", "V1-BAL-ERR", f"No se pudo obtener balance BTC: {e_bal}")

except Exception as e:
    flag_issue("CRIT", "V1-ERROR", f"Error en verificación de Zombie Trades: {e}")
    import traceback
    traceback.print_exc()


# =========================================================
# VECTOR 2: HEARTBEAT GAPS — periodos sin latido
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V2] HEARTBEAT GAPS: Periodos sin actividad de heartbeat")
print(f"{'─'*60}")
try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Ver estructura de system_heartbeat
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='system_heartbeat' ORDER BY ordinal_position")
    hb_cols = [r[0] for r in cur.fetchall()]
    print(f"  Columnas en system_heartbeat: {hb_cols}")

    # Obtener últimos heartbeats
    cur.execute(f"SELECT * FROM system_heartbeat ORDER BY {hb_cols[0]} DESC LIMIT 20")
    rows = cur.fetchall()
    print(f"  Últimos 20 heartbeats:")
    timestamps_hb = []
    for r in rows:
        print(f"    {dict(zip(hb_cols, r))}")
        # Extraer timestamp
        for col, val in zip(hb_cols, r):
            if 'time' in col.lower() or 'at' in col.lower() or col == hb_cols[0]:
                try:
                    ts = pd.Timestamp(val)
                    timestamps_hb.append(ts)
                    break
                except:
                    pass

    # Detectar gaps > 15 minutos entre heartbeats
    if len(timestamps_hb) >= 2:
        timestamps_hb_sorted = sorted(timestamps_hb, reverse=True)
        max_gap = timedelta(0)
        max_gap_pair = None
        for i in range(len(timestamps_hb_sorted)-1):
            gap = timestamps_hb_sorted[i] - timestamps_hb_sorted[i+1]
            if gap > max_gap:
                max_gap = gap
                max_gap_pair = (timestamps_hb_sorted[i+1], timestamps_hb_sorted[i])
        
        gap_min = max_gap.total_seconds() / 60
        if gap_min > 15:
            flag_issue("WARN", "V2-GAP", f"Gap de heartbeat detectado: {gap_min:.1f} min entre {max_gap_pair[0]} y {max_gap_pair[1]}")
        else:
            flag_issue("OK", "V2-HB", f"Heartbeats continuos. Gap máximo reciente: {gap_min:.1f} min")
    
    # Verificar último heartbeat
    cur.execute(f"SELECT * FROM system_heartbeat ORDER BY {hb_cols[0]} DESC LIMIT 1")
    last_hb = cur.fetchone()
    if last_hb:
        last_hb_dict = dict(zip(hb_cols, last_hb))
        # Encontrar timestamp
        for col in hb_cols:
            if 'time' in col.lower() or 'at' in col.lower() or col == hb_cols[0]:
                try:
                    last_ts = pd.Timestamp(last_hb_dict[col])
                    now = pd.Timestamp.now(tz='UTC')
                    minutes_ago = (now - last_ts).total_seconds() / 60
                    print(f"\n  Último heartbeat: hace {minutes_ago:.1f} minutos")
                    if minutes_ago > 15:
                        flag_issue("CRIT", "V2-STALE-HB", f"Último heartbeat hace {minutes_ago:.1f} min — sistema posiblemente colgado o en sleep extendido!")
                    else:
                        flag_issue("OK", "V2-FRESH-HB", f"Heartbeat fresco: hace {minutes_ago:.1f} min")
                    break
                except:
                    pass

    conn.close()
except Exception as e:
    flag_issue("WARN", "V2-ERROR", f"Error en verificación de Heartbeats: {e}")
    import traceback
    traceback.print_exc()


# =========================================================
# VECTOR 3: RECONCILIATION HISTORY — deltas históricos
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V3] RECONCILIATION: Historial de discrepancias BD vs OKX")
print(f"{'─'*60}")
try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='reconciliation_log' ORDER BY ordinal_position")
    rec_cols = [r[0] for r in cur.fetchall()]
    print(f"  Columnas en reconciliation_log: {rec_cols}")

    cur.execute(f"SELECT * FROM reconciliation_log ORDER BY {rec_cols[0]} DESC LIMIT 10")
    rows = cur.fetchall()
    if rows:
        print(f"  Últimas 10 reconciliaciones:")
        for r in rows:
            print(f"    {dict(zip(rec_cols, r))}")
    else:
        print(f"  (Sin registros de reconciliación)")
    conn.close()
except Exception as e:
    flag_issue("WARN", "V3-ERROR", f"Error en Reconciliation check: {e}")


# =========================================================
# VECTOR 4: AUDIT LOG PATTERNS — ciclos fallidos
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V4] AUDIT LOG PATTERNS: Ciclos fallidos y sus motivos")
print(f"{'─'*60}")
try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Contar totales
    cur.execute("SELECT COUNT(*) FROM operational_audit_logs WHERE is_approved = TRUE")
    total_ok = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM operational_audit_logs WHERE is_approved = FALSE")
    total_fail = cur.fetchone()[0]
    total = total_ok + total_fail
    fail_rate = total_fail / total * 100 if total > 0 else 0

    print(f"  Total ciclos: {total} | OK: {total_ok} | FALLIDOS: {total_fail} | Tasa fallo: {fail_rate:.1f}%")
    if fail_rate > 10:
        flag_issue("CRIT", "V4-FAIL-RATE", f"Tasa de fallos de auditoría del {fail_rate:.1f}% — demasiados ciclos bloqueados.")
    elif fail_rate > 3:
        flag_issue("WARN", "V4-FAIL-RATE", f"Tasa de fallos de auditoría del {fail_rate:.1f}% — revisar.")
    else:
        flag_issue("OK", "V4-FAIL-RATE", f"Tasa de fallos: {fail_rate:.1f}%")

    # Motivos de fallo
    cur.execute("""
        SELECT details, COUNT(*) as cnt 
        FROM operational_audit_logs 
        WHERE is_approved = FALSE 
        GROUP BY details 
        ORDER BY cnt DESC 
        LIMIT 10
    """)
    rows = cur.fetchall()
    if rows:
        print(f"\n  Top motivos de fallo:")
        for r in rows:
            print(f"    [{r[1]}x] {r[0][:100]}")

    # Ciclos de las últimas 24h
    cur.execute("""
        SELECT is_approved, COUNT(*) 
        FROM operational_audit_logs 
        WHERE timestamp > NOW() - INTERVAL '24 hours'
        GROUP BY is_approved
    """)
    rows_24h = cur.fetchall()
    print(f"\n  Últimas 24h: {dict((str(r[0]), r[1]) for r in rows_24h)}")

    # Distribución de estados de clock_drift
    cur.execute("""
        SELECT clock_drift_status, COUNT(*) 
        FROM operational_audit_logs 
        GROUP BY clock_drift_status
    """)
    print(f"  ClockDrift statuses: {dict(cur.fetchall())}")

    # Distribución de estados API
    cur.execute("""
        SELECT api_liveness_status, COUNT(*) 
        FROM operational_audit_logs 
        GROUP BY api_liveness_status
    """)
    print(f"  API Liveness statuses: {dict(cur.fetchall())}")

    conn.close()
except Exception as e:
    flag_issue("WARN", "V4-ERROR", f"Error en Audit Pattern check: {e}")
    import traceback
    traceback.print_exc()


# =========================================================
# VECTOR 5: MEMORY LEAK — proceso PM2 creciendo
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V5] MEMORY LEAK: Uso de memoria del proceso live trader")
print(f"{'─'*60}")
try:
    import psutil
    # Buscar el proceso del live trader
    live_proc = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'create_time']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'run_live_trader' in cmdline:
                live_proc = proc
                break
        except:
            pass

    if live_proc:
        mem_mb = live_proc.info['memory_info'].rss / 1024 / 1024
        uptime_min = (time.time() - live_proc.info['create_time']) / 60
        print(f"  PID: {live_proc.pid} | Memoria RSS: {mem_mb:.0f} MB | Uptime: {uptime_min:.0f} min")
        
        if mem_mb > 2000:
            flag_issue("CRIT", "V5-MEM-LEAK", f"Uso de memoria CRÍTICO: {mem_mb:.0f} MB. Posible leak acumulando DataFrames.")
        elif mem_mb > 1500:
            flag_issue("WARN", "V5-MEM-HIGH", f"Uso de memoria alto: {mem_mb:.0f} MB. Monitorear tendencia.")
        else:
            flag_issue("OK", "V5-MEM", f"Memoria en rango normal: {mem_mb:.0f} MB")
    else:
        flag_issue("WARN", "V5-PROC-NOT-FOUND", "No se encontró proceso run_live_trader activo. ¿PM2 todavía arrancando?")

    # Estado global de memoria del VPS
    vm = psutil.virtual_memory()
    print(f"\n  VPS RAM total: {vm.total/1024/1024:.0f} MB | Usado: {vm.used/1024/1024:.0f} MB | Disponible: {vm.available/1024/1024:.0f} MB | Uso: {vm.percent:.1f}%")
    if vm.percent > 90:
        flag_issue("CRIT", "V5-RAM-CRIT", f"RAM del VPS al {vm.percent:.1f}% — riesgo de OOM Killer.")
    elif vm.percent > 75:
        flag_issue("WARN", "V5-RAM-HIGH", f"RAM del VPS al {vm.percent:.1f}%.")
    else:
        flag_issue("OK", "V5-RAM", f"RAM VPS OK: {vm.percent:.1f}%")

except Exception as e:
    flag_issue("WARN", "V5-ERROR", f"Error en Memory check: {e}")


# =========================================================
# VECTOR 6: NTP / CLOCK SKEW
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V6] NTP / CLOCK SKEW: Sincronización de reloj del VPS")
print(f"{'─'*60}")
try:
    result = subprocess.run(['timedatectl', 'status'], capture_output=True, text=True, timeout=5)
    print(result.stdout)
    if 'synchronized: yes' in result.stdout.lower() or 'ntp service: active' in result.stdout.lower():
        flag_issue("OK", "V6-NTP", "Reloj NTP sincronizado.")
    else:
        flag_issue("WARN", "V6-NTP-UNSYNC", f"Posible desincronización NTP — verificar. Salida: {result.stdout[:200]}")
except Exception as e:
    try:
        result2 = subprocess.run(['chronyc', 'tracking'], capture_output=True, text=True, timeout=5)
        print(result2.stdout[:300])
        flag_issue("OK", "V6-NTP-CHRONY", "chronyc tracking ejecutado (ver detalles arriba).")
    except:
        flag_issue("WARN", "V6-NTP-ERR", f"No se pudo verificar NTP: {e}")


# =========================================================
# VECTOR 7: DISCO Y FILESYSTEM
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V7] DISCO: Estado del filesystem y logs crecientes")
print(f"{'─'*60}")
try:
    import psutil
    disk = psutil.disk_usage('/')
    disk_pct = disk.percent
    disk_free_gb = disk.free / 1024**3
    print(f"  Disco /: {disk_pct:.1f}% usado | Libre: {disk_free_gb:.1f} GB")
    if disk_pct > 85:
        flag_issue("CRIT", "V7-DISK-FULL", f"Disco al {disk_pct:.1f}% — riesgo de escritura fallida en DB/logs.")
    elif disk_pct > 70:
        flag_issue("WARN", "V7-DISK-HIGH", f"Disco al {disk_pct:.1f}% — monitorear.")
    else:
        flag_issue("OK", "V7-DISK", f"Disco OK: {disk_pct:.1f}% | {disk_free_gb:.1f} GB libre")

    # Tamaño de logs PM2 (pueden crecer indefinidamente)
    log_out = Path("/root/.pm2/logs/luna-v2-live-demo-out.log")
    log_err = Path("/root/.pm2/logs/luna-v2-live-demo-error.log")
    out_mb = log_out.stat().st_size / 1024/1024 if log_out.exists() else 0
    err_mb = log_err.stat().st_size / 1024/1024 if log_err.exists() else 0
    print(f"  PM2 log OUT: {out_mb:.1f} MB | PM2 log ERR: {err_mb:.1f} MB")
    if out_mb > 500:
        flag_issue("WARN", "V7-LOG-LARGE", f"Log OUT de PM2 muy grande: {out_mb:.1f} MB. Considerar logrotate o pm2 flush.")
    else:
        flag_issue("OK", "V7-LOG-SIZE", f"Logs PM2 en rango aceptable: {out_mb:.1f} MB OUT | {err_mb:.1f} MB ERR")

    # Tamaño de features_live.parquet
    if FEATURES_PATH.exists():
        feat_mb = FEATURES_PATH.stat().st_size / 1024/1024
        print(f"  features_live.parquet: {feat_mb:.1f} MB")
        if feat_mb > 500:
            flag_issue("WARN", "V7-FEAT-LARGE", f"features_live.parquet muy grande: {feat_mb:.1f} MB — revisar si se está acumulando historia innecesaria.")

except Exception as e:
    flag_issue("WARN", "V7-ERROR", f"Error en Disco check: {e}")


# =========================================================
# VECTOR 8: DB CONNECTION POOL HEALTH
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V8] DB CONNECTION POOL: Salud de PostgreSQL")
print(f"{'─'*60}")
try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Conexiones activas
    cur.execute("SELECT count(*) FROM pg_stat_activity WHERE datname='luna_db'")
    active_conns = cur.fetchone()[0]
    print(f"  Conexiones activas a luna_db: {active_conns}")

    # Conexiones idle
    cur.execute("SELECT state, count(*) FROM pg_stat_activity WHERE datname='luna_db' GROUP BY state")
    conn_states = dict(cur.fetchall())
    print(f"  Estado conexiones: {conn_states}")

    # Tamaño de la BD
    cur.execute("SELECT pg_size_pretty(pg_database_size('luna_db'))")
    db_size = cur.fetchone()[0]
    print(f"  Tamaño BD luna_db: {db_size}")

    # Tablas más grandes
    cur.execute("""
        SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) as size,
               n_live_tup as rows
        FROM pg_stat_user_tables
        ORDER BY pg_total_relation_size(relid) DESC
        LIMIT 5
    """)
    print(f"  Tablas más grandes:")
    for r in cur.fetchall():
        print(f"    {r[0]}: {r[1]} ({r[2]} filas)")

    # Verificar locks bloqueantes
    cur.execute("""
        SELECT count(*) FROM pg_locks l 
        JOIN pg_stat_activity a ON l.pid = a.pid
        WHERE NOT l.granted AND a.datname = 'luna_db'
    """)
    blocked = cur.fetchone()[0]
    if blocked > 0:
        flag_issue("CRIT", "V8-DB-LOCK", f"{blocked} transacciones bloqueadas en PostgreSQL!")
    else:
        flag_issue("OK", "V8-DB-LOCKS", "Sin locks bloqueantes en PostgreSQL.")

    if active_conns > 20:
        flag_issue("WARN", "V8-CONN-HIGH", f"Muchas conexiones activas: {active_conns}. Posible leak de conexiones.")
    else:
        flag_issue("OK", "V8-CONN", f"Conexiones DB en rango normal: {active_conns}")

    conn.close()
except Exception as e:
    flag_issue("CRIT", "V8-DB-ERROR", f"Error conectando/auditando PostgreSQL: {e}")


# =========================================================
# VECTOR 9: REGIME ROUTER DEAD-CODE (prob=0.0 silencioso)
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V9] REGIME ROUTER: Detección de agentes muertos (prob=0.0)")
print(f"{'─'*60}")
try:
    # Leer el último log completo para buscar patrones ZERO-AUDIT
    log_path = Path("/root/.pm2/logs/luna-v2-live-demo-out.log")
    if log_path.exists():
        # Leer las últimas 2000 líneas
        result = subprocess.run(['tail', '-2000', str(log_path)], capture_output=True, text=True)
        log_content = result.stdout

        # Buscar ZERO-AUDIT warnings
        zero_audit_old = [l for l in log_content.split('\n') if 'ZERO-AUDIT' in l and 'NO explicadas' in l]
        zero_audit_ok = [l for l in log_content.split('\n') if 'ZERO-AUDIT' in l and 'explicadas por 4_BEAR_FORCED. OK' in l]

        print(f"  ZERO-AUDIT con barras NO explicadas (últimas 2000 líneas): {len(zero_audit_old)} ocurrencias")
        print(f"  ZERO-AUDIT OK (BEAR_FORCED): {len(zero_audit_ok)} ocurrencias")

        if zero_audit_old:
            flag_issue("CRIT", "V9-DEAD-AGENT", f"{len(zero_audit_old)} casos de agentes RegimeRouter con prob=0.0 no explicados por BEAR_FORCED. Posibles agentes muertos enviando señal nula.")
            print(f"  Ejemplo: {zero_audit_old[0][:150]}")
        else:
            flag_issue("OK", "V9-AGENTS", "RegimeRouter: todos los prob=0.0 explicados por BEAR_FORCED. Sin agentes zombie.")

        # Contar BUGFIX-ML-SHIELD (ajuste HMM de 9 a 6 columnas)
        ml_shield_count = log_content.count('BUGFIX-ML-SHIELD')
        print(f"  BUGFIX-ML-SHIELD activaciones (últimas 2000 líneas): {ml_shield_count}")
        if ml_shield_count > 0:
            flag_issue("WARN", "V9-HMM-MISMATCH", f"El HMM genera 9 columnas pero el modelo espera 6 — esto indica discrepancia de features IS vs live en cada ciclo. {ml_shield_count} activaciones recientes.")

        # Contar BUGFIX-OVERFLOW-CEILING
        overflow_count = log_content.count('BUGFIX-OVERFLOW-CEILING')
        print(f"  BUGFIX-OVERFLOW-CEILING activaciones: {overflow_count}")
        if overflow_count > 3:
            flag_issue("WARN", "V9-OVERFLOW", f"Valores extremos sanitizados {overflow_count} veces — posibles NaN o Inf en features que no llegan al escudo de auditoría.")

except Exception as e:
    flag_issue("WARN", "V9-ERROR", f"Error en RegimeRouter check: {e}")


# =========================================================
# VECTOR 10: TELEGRAM LISTENER — ¿sigue vivo?
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V10] TELEGRAM LISTENER: Estado del listener de comandos")
print(f"{'─'*60}")
try:
    log_path = Path("/root/.pm2/logs/luna-v2-live-demo-out.log")
    result = subprocess.run(['tail', '-500', str(log_path)], capture_output=True, text=True)
    log_content = result.stdout

    # Verificar que el listener arrancó
    listener_started = 'Listener The comandos iniciado en background' in log_content or \
                       'Listener' in log_content and 'background' in log_content

    # Verificar el token de Telegram
    tg_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    tg_chat = os.getenv('TELEGRAM_CHAT_ID', '')
    has_token = len(tg_token) > 10
    has_chat = len(tg_chat) > 3

    print(f"  Telegram token configurado: {'SÍ' if has_token else 'NO'}")
    print(f"  Telegram chat_id configurado: {'SÍ' if has_chat else 'NO'}")
    print(f"  Listener arrancado: {'SÍ' if listener_started else 'NO'}")

    if not has_token or not has_chat:
        flag_issue("CRIT", "V10-TG-NOCREDS", "Credenciales de Telegram no configuradas — bot mudo, sin alertas ni comandos.")
    elif not listener_started:
        flag_issue("WARN", "V10-TG-LISTENER", "Telegram listener no detectado en logs recientes — comandos /status /kill pueden no responder.")
    else:
        flag_issue("OK", "V10-TG", "Telegram listener activo y con credenciales configuradas.")

    # Verificar si hay errores de conexión Telegram recientes
    tg_errors = [l for l in log_content.split('\n') if 'telegram' in l.lower() and ('error' in l.lower() or 'failed' in l.lower() or 'timeout' in l.lower())]
    if tg_errors:
        flag_issue("WARN", "V10-TG-ERRS", f"{len(tg_errors)} errores de Telegram en logs recientes: {tg_errors[0][:100]}")

except Exception as e:
    flag_issue("WARN", "V10-ERROR", f"Error en Telegram check: {e}")


# =========================================================
# VECTOR 11: PM2 RESTART PATTERN — análisis de los 10 reinicios
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V11] PM2 RESTARTS: Patrón de los 10 reinicios")
print(f"{'─'*60}")
try:
    # Buscar todos los momentos de arranque en el log
    log_path = Path("/root/.pm2/logs/luna-v2-live-demo-out.log")
    result = subprocess.run(['grep', '-n', 'BOOT.*Inicializando Luna\|Entorno.*cargado desde', str(log_path)], capture_output=True, text=True)
    boot_lines = result.stdout.strip().split('\n')
    print(f"  Arranques detectados en log completo: {len(boot_lines)}")
    if boot_lines:
        print("  Últimos arranques:")
        for l in boot_lines[-10:]:
            print(f"    {l[:120]}")

    # Analizar errores antes de cada reinicio (buscar Traceback/KeyboardInterrupt)
    result2 = subprocess.run(['grep', '-n', 'KeyboardInterrupt\|Traceback\|FATAL\|sys.exit', str(log_path)], capture_output=True, text=True)
    exit_lines = result2.stdout.strip().split('\n')
    print(f"\n  Salidas/crashes detectadas: {len([l for l in exit_lines if l.strip()])}")
    for l in exit_lines[-5:]:
        if l.strip():
            print(f"    {l[:120]}")

    # Buscar en error log
    err_log_path = Path("/root/.pm2/logs/luna-v2-live-demo-error.log")
    if err_log_path.exists():
        err_size_mb = err_log_path.stat().st_size / 1024/1024
        result3 = subprocess.run(['tail', '-50', str(err_log_path)], capture_output=True, text=True)
        err_recent = result3.stdout
        print(f"\n  Error log ({err_size_mb:.1f} MB) — últimas 50 líneas:")
        # Solo mostrar WARNINGs o superiores
        for l in err_recent.split('\n'):
            if 'WARNING\|ERROR\|CRITICAL' in l or 'warning' in l.lower() or 'error' in l.lower() or 'critical' in l.lower():
                print(f"    {l[:150]}")

    if len(boot_lines) > 10:
        flag_issue("WARN", "V11-RESTARTS", f"{len(boot_lines)} arranques totales del bot desde el inicio del log. Los 10 reinicios recientes pueden ser de hot-deploys o crashes.")
    else:
        flag_issue("OK", "V11-STABLE", f"Solo {len(boot_lines)} arranques en el historial del log.")

except Exception as e:
    flag_issue("WARN", "V11-ERROR", f"Error en PM2 Restart check: {e}")


# =========================================================
# VECTOR 12: FEATURES DRIFT HISTÓRICO — gaps en el parquet
# =========================================================
print(f"\n{'─'*60}")
print("[AUDIT-V12] FEATURES DRIFT: Continuidad temporal del parquet")
print(f"{'─'*60}")
try:
    if FEATURES_PATH.exists():
        df = pd.read_parquet(FEATURES_PATH)
        print(f"  features_live.parquet: {len(df)} filas | Rango: {df.index.min()} → {df.index.max()}")

        # Detectar gaps de >2h en el índice temporal
        if len(df) > 10:
            df_sorted = df.sort_index()
            diffs = df_sorted.index.to_series().diff().dropna()
            expected_freq = pd.Timedelta('1H')
            gaps = diffs[diffs > expected_freq * 2]
            if len(gaps) > 0:
                print(f"  Gaps detectados en series temporal:")
                for ts, gap in gaps.head(10).items():
                    print(f"    En {ts}: gap de {gap}")
                flag_issue("WARN", "V12-GAPS", f"{len(gaps)} gaps temporales en features_live.parquet (>2H). Pueden afectar la inferencia de ciclos futuros.")
            else:
                flag_issue("OK", "V12-CONTINUITY", "Series temporal de features sin gaps significativos.")

        # Drift actual
        last_ts = df.index.max()
        if hasattr(last_ts, 'tzinfo') and last_ts.tzinfo is None:
            last_ts_aware = pd.Timestamp(last_ts).tz_localize('UTC')
        else:
            last_ts_aware = pd.Timestamp(last_ts)
        now_utc = pd.Timestamp.now(tz='UTC')
        drift_min = (now_utc - last_ts_aware).total_seconds() / 60
        print(f"  Drift actual de features_live: {drift_min:.1f} min")
        if drift_min > 90:
            flag_issue("CRIT", "V12-DRIFT", f"Drift de features_live.parquet: {drift_min:.1f} min. El próximo ciclo disparará ClockDrift Guard (límite 90 min).")
        elif drift_min > 60:
            flag_issue("WARN", "V12-DRIFT-WARN", f"Drift de features_live.parquet: {drift_min:.1f} min. Cercano al límite de 90 min.")
        else:
            flag_issue("OK", "V12-DRIFT", f"Drift de features_live.parquet: {drift_min:.1f} min — dentro de límite.")
    else:
        flag_issue("CRIT", "V12-NO-FILE", "features_live.parquet NO EXISTE — primer ciclo va a fallar en FeaturePipeline.")
except Exception as e:
    flag_issue("WARN", "V12-ERROR", f"Error en Features Drift check: {e}")


# =========================================================
# RESUMEN FINAL
# =========================================================
print(f"\n{SEPARATOR}")
print("[AUDIT-INSTITUCIONAL] RESUMEN FINAL")
print(SEPARATOR)
print(f"\n{CRIT} PROBLEMAS CRÍTICOS ({len(issues)}):")
if issues:
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("  (Ninguno)")

print(f"\n{WARN} ADVERTENCIAS ({len(warnings)}):")
if warnings:
    for i, w in enumerate(warnings, 1):
        print(f"  {i}. {w}")
else:
    print("  (Ninguna)")

print(f"\n[AUDIT-INSTITUCIONAL] Auditoría completada: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"[AUDIT-INSTITUCIONAL] Salud general: {'🔴 CRÍTICA' if issues else ('⚠️ ADVERTENCIAS' if warnings else '✅ BUENA')}")
