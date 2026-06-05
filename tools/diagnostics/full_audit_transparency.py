"""
[FULL-AUDIT-TRANSPARENCY] Auditoría completa de transparencia y trazabilidad del proceso live.
Verifica TODAS las tablas de tracking, detecta gaps, errores y valida integridad end-to-end.
"""
import sys, os, json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "/root/luna_v2")

from luna.database.db_manager import DatabaseManager
from psycopg2.extras import DictCursor

db = DatabaseManager()
now_utc = datetime.now(timezone.utc)

print("=" * 80)
print("[FULL-AUDIT] AUDITORÍA COMPLETA DE TRANSPARENCIA — LUNA V2 LIVE DEMO")
print(f"[FULL-AUDIT] Ejecutado: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print("=" * 80)

issues = []

with db.get_connection() as conn:
    with conn.cursor(cursor_factory=DictCursor) as cur:

        # ── 1. Inventario de todas las tablas de tracking ──────────────────────
        print("\n[AUDIT-1] INVENTARIO DE TABLAS DE TRACKING")
        cur.execute("""
            SELECT table_name,
                   (SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_name = t.table_name AND table_schema = 'public') AS col_count
            FROM information_schema.tables t
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        tables = cur.fetchall()
        for t in tables:
            cur.execute(f"SELECT COUNT(*) AS n FROM {t['table_name']}")
            n = cur.fetchone()['n']
            print(f"  📋 {t['table_name']:45s} | {t['col_count']:2d} cols | {n:6d} registros")

        # ── 2. audit_logs — integridad y cobertura temporal ─────────────────────
        print("\n[AUDIT-2] INTEGRIDAD DE audit_logs")
        cur.execute("""
            SELECT
                COUNT(*) AS total,
                MIN(timestamp) AS primera,
                MAX(timestamp) AS ultima,
                COUNT(DISTINCT DATE(timestamp)) AS dias_cubiertos
            FROM audit_logs;
        """)
        al = cur.fetchone()
        print(f"  Total registros:      {al['total']}")
        print(f"  Primer registro:      {al['primera']}")
        print(f"  Último registro:      {al['ultima']}")
        print(f"  Días cubiertos:       {al['dias_cubiertos']}")

        # Gap stats en subquery
        cur.execute("""
            SELECT
                AVG(gap_min) AS avg_gap_min,
                MAX(gap_min) AS max_gap_min
            FROM (
                SELECT EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (ORDER BY timestamp)))/60 AS gap_min
                FROM audit_logs
            ) g WHERE gap_min IS NOT NULL;
        """)
        gap_row = cur.fetchone()
        print(f"  Gap promedio:         {float(gap_row['avg_gap_min'] or 0):.1f} min")
        print(f"  Gap máximo:           {float(gap_row['max_gap_min'] or 0):.1f} min")

        # Gaps > 2 horas (anomalías)
        cur.execute("""
            SELECT timestamp,
                   LAG(timestamp) OVER (ORDER BY timestamp) AS prev_ts,
                   EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (ORDER BY timestamp)))/3600 AS gap_hours
            FROM audit_logs
            ORDER BY timestamp
        """)
        all_rows = cur.fetchall()
        big_gaps = [(r['prev_ts'], r['timestamp'], float(r['gap_hours'])) for r in all_rows
                    if r['gap_hours'] and float(r['gap_hours']) > 2.0]
        if big_gaps:
            print(f"\n  ⚠️  Gaps > 2h detectados ({len(big_gaps)}):")
            for g in big_gaps:
                print(f"    {g[0]} → {g[1]} ({g[2]:.1f}h)")
            issues.append(f"audit_logs tiene {len(big_gaps)} gaps >2h")
        else:
            print(f"  ✅ Sin gaps >2h en audit_logs")

        # Distribución de acciones
        cur.execute("SELECT action, COUNT(*) AS n FROM audit_logs GROUP BY action ORDER BY n DESC")
        print(f"\n  Distribución acciones:")
        for r in cur.fetchall():
            print(f"    {r['action']:8s}: {r['n']}")

        # ── 3. operational_audit_logs ───────────────────────────────────────────
        print("\n[AUDIT-3] operational_audit_logs — Guards de seguridad")
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='operational_audit_logs' ORDER BY ordinal_position")
        op_cols = [r['column_name'] for r in cur.fetchall()]
        print(f"  Columnas: {op_cols}")

        cur.execute("""
            SELECT COUNT(*) AS total,
                   MIN(timestamp) AS primera,
                   MAX(timestamp) AS ultima,
                   SUM(CASE WHEN is_approved = false THEN 1 ELSE 0 END) AS ciclos_bloqueados,
                   AVG(clock_drift_minutes) AS avg_drift,
                   MAX(clock_drift_minutes) AS max_drift,
                   AVG(execution_latency_sec) AS avg_latency,
                   MAX(execution_latency_sec) AS max_latency,
                   AVG(slippage_pct) AS avg_slippage
            FROM operational_audit_logs;
        """)
        op = cur.fetchone()
        print(f"  Total auditorías:     {op['total']}")
        print(f"  Primer registro:      {op['primera']}")
        print(f"  Último registro:      {op['ultima']}")
        print(f"  Ciclos bloqueados:    {op['ciclos_bloqueados']} ({100*op['ciclos_bloqueados']/(op['total'] or 1):.1f}%)")
        print(f"  Clock drift avg/max:  {float(op['avg_drift'] or 0):.1f} / {float(op['max_drift'] or 0):.1f} min")
        print(f"  Latencia avg/max:     {float(op['avg_latency'] or 0):.1f} / {float(op['max_latency'] or 0):.1f} s")
        print(f"  Slippage avg:         {float(op['avg_slippage'] or 0):.6f}%")

        if op['ciclos_bloqueados'] and op['ciclos_bloqueados'] > 0:
            cur.execute("""
                SELECT timestamp, clock_drift_minutes, clock_drift_status, is_approved, details
                FROM operational_audit_logs WHERE is_approved = false ORDER BY timestamp
            """)
            blocked = cur.fetchall()
            print(f"\n  ⚠️  Ciclos bloqueados:")
            for b in blocked:
                print(f"    [{b['timestamp']}] drift={b['clock_drift_minutes']:.1f}m | {b['clock_drift_status']}")

        # ── 4. system_heartbeat ─────────────────────────────────────────────────
        print("\n[AUDIT-4] system_heartbeat")
        cur.execute("SELECT * FROM system_heartbeat ORDER BY last_heartbeat DESC")
        hbs = cur.fetchall()
        for hb in hbs:
            age_min = (now_utc - hb['last_heartbeat'].replace(tzinfo=timezone.utc)).total_seconds() / 60
            status = "✅" if age_min < 70 else "⚠️" if age_min < 130 else "🔴"
            print(f"  {status} {hb['component']:35s} | {hb['status']:12s} | hace {age_min:.1f} min")
            if age_min > 130:
                issues.append(f"Heartbeat stale: {hb['component']} hace {age_min:.0f}min")

        # ── 5. live_state ───────────────────────────────────────────────────────
        print("\n[AUDIT-5] live_state — Estado del portfolio")
        cur.execute("SELECT * FROM live_state WHERE id=1")
        ls = cur.fetchone()
        if ls:
            age_min = (now_utc - ls['updated_at'].replace(tzinfo=timezone.utc)).total_seconds() / 60
            print(f"  Portfolio:    ${float(ls['portfolio_value']):,.2f}")
            print(f"  ATH:          ${float(ls['ath']):,.2f}")
            print(f"  Drawdown:     {float(ls['drawdown'])*100:.4f}%")
            print(f"  Is Paused:    {ls['is_paused']} {'⚠️ ALERTA' if ls['is_paused'] else '✅'}")
            print(f"  Updated:      {ls['updated_at']} (hace {age_min:.1f} min)")
            if age_min > 130:
                issues.append(f"live_state no actualizado en {age_min:.0f}min")

        # ── 6. Consistencia audit_logs vs operational_audit_logs ────────────────
        print("\n[AUDIT-6] CONSISTENCIA CRUZADA DE TABLAS")
        cur.execute("SELECT COUNT(*) AS n FROM audit_logs WHERE timestamp >= NOW() - INTERVAL '24 hours'")
        al_24h = cur.fetchone()['n']
        cur.execute("SELECT COUNT(*) AS n FROM operational_audit_logs WHERE timestamp >= NOW() - INTERVAL '24 hours'")
        op_24h = cur.fetchone()['n']
        print(f"  audit_logs últimas 24h:              {al_24h} registros")
        print(f"  operational_audit_logs últimas 24h:  {op_24h} registros")
        if abs(al_24h - op_24h) > 2:
            issues.append(f"Discrepancia: audit_logs={al_24h} vs operational_audit_logs={op_24h}")
            print(f"  ⚠️  Discrepancia > 2 entre tablas")
        else:
            print(f"  ✅ Tablas alineadas (diff={abs(al_24h-op_24h)})")

        # ── 7. Últimas 5 decisiones completas ──────────────────────────────────
        print("\n[AUDIT-7] ÚLTIMAS 5 DECISIONES END-TO-END")
        cur.execute("""
            SELECT al.timestamp, al.price, al.action, al.confidence,
                   al.xgb_prob, al.hmm_regime, al.reason,
                   op.clock_drift_minutes, op.clock_drift_status,
                   op.execution_latency_sec, op.slippage_pct,
                   op.is_approved, op.nan_inf_null_cols, op.active_leverage
            FROM audit_logs al
            LEFT JOIN operational_audit_logs op ON DATE_TRUNC('minute', al.timestamp) = DATE_TRUNC('minute', op.timestamp)
            ORDER BY al.timestamp DESC
            LIMIT 5
        """)
        for r in cur.fetchall():
            ts = r['timestamp'].strftime('%m-%d %H:%M') if hasattr(r['timestamp'], 'strftime') else str(r['timestamp'])
            ok = "✅" if r['is_approved'] != False else "🔴"
            drift = f"{float(r['clock_drift_minutes'] or 0):.0f}m" if r['clock_drift_minutes'] else "N/A"
            lat = f"{float(r['execution_latency_sec'] or 0):.1f}s" if r['execution_latency_sec'] else "N/A"
            print(f"  {ok} [{ts}] {r['action']:5s} | XGB={float(r['xgb_prob'] or 0):.4f} | "
                  f"Drift={drift} | Lat={lat} | NaN={r['nan_inf_null_cols']} | Lev={r['active_leverage']}x")

# ── Resumen final ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("[FULL-AUDIT] RESUMEN FINAL")
print("=" * 80)
if issues:
    print(f"⚠️  {len(issues)} PROBLEMA(S) DETECTADO(S):")
    for i, issue in enumerate(issues, 1):
        print(f"  [{i}] {issue}")
else:
    print("✅ SISTEMA 100% TRANSPARENTE Y TRAZABLE — SIN PROBLEMAS DETECTADOS")
print("=" * 80)
