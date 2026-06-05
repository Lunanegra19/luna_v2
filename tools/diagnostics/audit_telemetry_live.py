"""
[AUDIT-TELEMETRY] Script de auditoría completa de decisiones horarias del Live Trader.
Extrae audit_logs de PostgreSQL y genera reporte estructurado.
"""
import sys
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
sys.path.insert(0, "/root/luna_v2")

from luna.database.db_manager import DatabaseManager
from psycopg2.extras import DictCursor

db = DatabaseManager()

print("=" * 80)
print("[AUDIT-TELEMETRY] REPORTE COMPLETO DE DECISIONES - LUNA V2 LIVE DEMO")
print(f"[AUDIT-TELEMETRY] Generado: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
print("=" * 80)

try:
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:

            # 1. Resumen general de audit_logs
            cur.execute("SELECT COUNT(*) as total FROM audit_logs;")
            total = cur.fetchone()['total']
            print(f"\n[AUDIT-STATS] Total registros en audit_logs: {total}")

            # 2. Distribución de acciones
            cur.execute("""
                SELECT action, COUNT(*) as count
                FROM audit_logs
                GROUP BY action
                ORDER BY count DESC;
            """)
            rows = cur.fetchall()
            print("\n[AUDIT-STATS] Distribución de acciones:")
            for r in rows:
                print(f"  {r['action']:10s} -> {r['count']} veces")

            # 3. Últimas 48h de decisiones detalladas
            since = datetime.utcnow() - timedelta(hours=48)
            cur.execute("""
                SELECT timestamp, price, action, confidence, xgb_prob, hmm_regime, reason, contracts, executed_price
                FROM audit_logs
                WHERE timestamp >= %s
                ORDER BY timestamp ASC;
            """, (since,))
            rows = cur.fetchall()

            print(f"\n[AUDIT-TELEMETRY] Decisiones últimas 48h ({len(rows)} registros):")
            print("-" * 80)

            for r in rows:
                ts = r['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(r['timestamp'], 'strftime') else str(r['timestamp'])
                price = float(r['price']) if r['price'] else 0.0
                action = r['action']
                confidence = float(r['confidence']) if r['confidence'] else 0.0
                xgb_prob = float(r['xgb_prob']) if r['xgb_prob'] else 0.0
                hmm = r['hmm_regime'] if r['hmm_regime'] is not None else 'N/A'
                contracts = int(r['contracts']) if r['contracts'] else 0
                exec_price = float(r['executed_price']) if r['executed_price'] else 0.0
                reason = r['reason'] or ''

                print(f"\n  [{ts} UTC]")
                print(f"  Accion:       {action}")
                print(f"  Precio:       {price:,.2f} EUR | Ejecutado: {exec_price:,.2f} EUR")
                print(f"  XGB Prob:     {xgb_prob:.4f} | Confianza: {confidence:.4f}")
                print(f"  HMM Regime:   {hmm}")
                print(f"  Contratos:    {contracts}")
                print(f"  Razon:        {reason[:200]}")
                print(f"  {'─'*70}")

            # 4. Live state actual
            cur.execute("SELECT portfolio_value, ath, drawdown, is_paused, updated_at FROM live_state WHERE id=1;")
            state = cur.fetchone()
            if state:
                print(f"\n[AUDIT-LIVE-STATE] Estado actual del portfolio:")
                print(f"  Portfolio Value: ${float(state['portfolio_value']):,.2f}")
                print(f"  ATH:             ${float(state['ath']):,.2f}")
                print(f"  Drawdown:        {float(state['drawdown'])*100:.4f}%")
                print(f"  Is Paused:       {state['is_paused']}")
                print(f"  Updated At:      {state['updated_at']}")

            # 5. Último heartbeat
            cur.execute("SELECT last_heartbeat, status FROM system_heartbeat WHERE component = 'luna_v2_live_demo';")
            hb = cur.fetchone()
            if hb:
                age = (datetime.utcnow() - hb['last_heartbeat']).total_seconds() / 60
                print(f"\n[AUDIT-HEARTBEAT] Último latido: {hb['last_heartbeat']} UTC (hace {age:.1f} min) | Status: {hb['status']}")

            # 6. Audit operacional últimas 24h
            cur.execute("""
                SELECT timestamp, clock_drift_minutes, nan_inf_null_cols, active_leverage,
                       api_liveness_equity, hmm_regime_index, execution_latency_sec,
                       slippage_pct, is_approved, details
                FROM operational_audit_logs
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                ORDER BY timestamp DESC
                LIMIT 10;
            """)
            op_rows = cur.fetchall()
            print(f"\n[AUDIT-OPERATIONAL] Últimas {len(op_rows)} auditorías operacionales (24h):")
            for op in op_rows:
                ts = op['timestamp'].strftime('%H:%M:%S') if hasattr(op['timestamp'], 'strftime') else str(op['timestamp'])
                print(f"  [{ts}] Drift={op['clock_drift_minutes']}m | NaN={op['nan_inf_null_cols']} | Lev={op['active_leverage']}x | Lat={op['execution_latency_sec']}s | Slip={op['slippage_pct']} | OK={op['is_approved']}")

except Exception as e:
    import traceback
    print(f"[AUDIT-ERROR] {e}\n{traceback.format_exc()}")

print("\n" + "=" * 80)
print("[AUDIT-TELEMETRY] FIN DEL REPORTE")
print("=" * 80)
