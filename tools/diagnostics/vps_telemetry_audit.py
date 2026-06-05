"""
[TELEMETRY-AUDIT-2026-05-28] Script de auditoría completa del sistema live en VPS.
Revisa: live_state, heartbeats, audit_logs (24h), decisiones, errores silenciosos.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("[TELEMETRY-AUDIT] ERROR: DATABASE_URL no encontrado en .env")
    sys.exit(1)

print("=" * 70)
print("  LUNA V2 — TELEMETRÍA VPS COMPLETA (ÚLTIMAS 24H)")
print(f"  Generado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
print("=" * 70)

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# ── 1. LIVE STATE ─────────────────────────────────────────────────────────────
print("\n📊 [1] LIVE STATE (PostgreSQL):")
print("-" * 50)
cur.execute("SELECT portfolio_value, ath, drawdown, is_paused, updated_at FROM live_state WHERE id=1;")
row = cur.fetchone()
if row:
    pv, ath, dd, paused, updated = row
    print(f"  Portfolio Value : ${pv:,.2f}")
    print(f"  ATH             : ${ath:,.2f}")
    print(f"  Drawdown        : {dd:.2%}")
    print(f"  is_paused       : {'⚠️ TRUE — SISTEMA PAUSADO' if paused else '✅ False'}")
    print(f"  Última actualiz.: {updated}")
else:
    print("  ❌ Sin datos en live_state!")

# ── 2. HEARTBEATS (últimas 24h) ───────────────────────────────────────────────
print("\n💓 [2] HEARTBEATS (últimas 24h):")
print("-" * 50)
try:
    cur.execute("""
        SELECT status, COUNT(*) as n, MIN(timestamp) as first, MAX(timestamp) as last
        FROM heartbeat_logs
        WHERE timestamp > NOW() - INTERVAL '24 hours'
        GROUP BY status ORDER BY last DESC;
    """)
    rows = cur.fetchall()
    if rows:
        for status, n, first, last in rows:
            print(f"  {status:<20} | n={n:<5} | first={first} | last={last}")
    else:
        print("  ⚠️ Sin heartbeats en las últimas 24h")
except Exception as e:
    conn.rollback()
    print(f"  [heartbeat_logs no existe o error]: {e}")

# ── 3. AUDIT LOGS RESUMEN (24h) ────────────────────────────────────────────────
print("\n🔍 [3] AUDIT LOGS — RESUMEN POR ACCIÓN (últimas 24h):")
print("-" * 50)
cur.execute("""
    SELECT action, COUNT(*) as n, MIN(timestamp) as first, MAX(timestamp) as last
    FROM audit_logs
    WHERE timestamp > NOW() - INTERVAL '24 hours'
    GROUP BY action ORDER BY n DESC;
""")
rows = cur.fetchall()
if rows:
    total = sum(r[1] for r in rows)
    for action, n, first, last in rows:
        print(f"  {action:<8} | n={n:<4} ({n/total:.0%}) | first={first} | last={last}")
    print(f"\n  TOTAL CICLOS: {total}")
else:
    print("  ❌ Sin registros en audit_logs en 24h")

# ── 4. ÚLTIMAS 20 DECISIONES ──────────────────────────────────────────────────
print("\n📋 [4] ÚLTIMAS 20 DECISIONES:")
print("-" * 70)
cur.execute("""
    SELECT timestamp, action, xgb_prob, confidence, hmm_regime, reason
    FROM audit_logs
    WHERE timestamp > NOW() - INTERVAL '24 hours'
    ORDER BY timestamp DESC LIMIT 20;
""")
rows = cur.fetchall()
if rows:
    for ts, action, xgb_prob, conf, hmm, reason in rows:
        ts_str = ts.strftime('%H:%M') if ts else 'N/A'
        xgb_str = f"{xgb_prob:.4f}" if xgb_prob is not None else "N/A"
        conf_str = f"{conf:.4f}" if conf is not None else "N/A"
        hmm_str = str(hmm) if hmm is not None else "N/A"
        reason_short = str(reason)[:60] if reason else ""
        print(f"  {ts_str} | {action:<5} | xgb={xgb_str} | conf={conf_str} | hmm={hmm_str:<3} | {reason_short}")
else:
    print("  ❌ Sin decisiones recientes")

# ── 5. TRADES EJECUTADOS (24h) ──────────────────────────────────────────────
print("\n💰 [5] TRADES EJECUTADOS (últimas 24h — action=LONG/SHORT):")
print("-" * 50)
cur.execute("""
    SELECT timestamp, action, xgb_prob, confidence, executed_price, contracts, reason
    FROM audit_logs
    WHERE timestamp > NOW() - INTERVAL '24 hours'
      AND action IN ('LONG', 'SHORT')
    ORDER BY timestamp DESC;
""")
rows = cur.fetchall()
if rows:
    for ts, action, xgb_prob, conf, price, contracts, reason in rows:
        ts_str = ts.strftime('%Y-%m-%d %H:%M') if ts else 'N/A'
        print(f"  {ts_str} | {action} | xgb={xgb_prob:.4f} | conf={conf:.4f} | price=${price:,.2f} | contracts={contracts}")
else:
    print("  ✅ Sin trades ejecutados (solo HOLD en 24h)")

# ── 6. RANGO DE COBERTURA TEMPORAL ───────────────────────────────────────────
print("\n🕐 [6] COBERTURA TEMPORAL DE AUDIT LOGS:")
print("-" * 50)
cur.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM audit_logs;")
row = cur.fetchone()
if row:
    mn, mx, cnt = row
    print(f"  Primer registro : {mn}")
    print(f"  Último registro : {mx}")
    print(f"  Total registros : {cnt}")

# ── 7. GAPS DETECTADOS (ciclos perdidos en 24h) ────────────────────────────
print("\n⚠️ [7] GAPS HORARIOS DETECTADOS (ciclos > 90min sin heartbeat):")
print("-" * 50)
cur.execute("""
    SELECT timestamp, action
    FROM audit_logs
    WHERE timestamp > NOW() - INTERVAL '24 hours'
    ORDER BY timestamp ASC;
""")
rows = cur.fetchall()
if len(rows) >= 2:
    gaps = []
    for i in range(1, len(rows)):
        delta = rows[i][0] - rows[i-1][0]
        if delta.total_seconds() > 5400:  # > 90 min
            gaps.append((rows[i-1][0], rows[i][0], delta))
    if gaps:
        for start, end, delta in gaps:
            print(f"  ⚠️ GAP: {start} → {end} ({delta.total_seconds()/60:.0f} min)")
    else:
        print("  ✅ Sin gaps detectados (cobertura horaria continua)")
else:
    print("  Insuficientes datos para análisis de gaps")

# ── 8. XGB PROB STATISTICS (24h) ─────────────────────────────────────────────
print("\n📈 [8] ESTADÍSTICAS XGB PROB (últimas 24h):")
print("-" * 50)
cur.execute("""
    SELECT
        AVG(xgb_prob) as avg_prob,
        MIN(xgb_prob) as min_prob,
        MAX(xgb_prob) as max_prob,
        STDDEV(xgb_prob) as std_prob,
        COUNT(*) as n
    FROM audit_logs
    WHERE timestamp > NOW() - INTERVAL '24 hours'
      AND xgb_prob IS NOT NULL;
""")
row = cur.fetchone()
if row:
    avg_p, min_p, max_p, std_p, n = row
    print(f"  Avg XGB Prob : {avg_p:.4f}")
    print(f"  Min / Max    : {min_p:.4f} / {max_p:.4f}")
    print(f"  Std Dev      : {std_p:.4f}" if std_p else "  Std Dev      : N/A")
    print(f"  N registros  : {n}")

# ── 9. PM2 RESTART COUNT ───────────────────────────────────────────────────────
print("\n🔄 [9] NOTA — Ver pm2 list para restart count.")

# ── 10. OPERATIONAL METRICS ───────────────────────────────────────────────────
print("\n🛡️ [10] OPERATIONAL METRICS (últimas 24h):")
print("-" * 50)
try:
    cur.execute("""
        SELECT
            AVG(cycle_duration_s) as avg_cycle,
            MAX(cycle_duration_s) as max_cycle,
            AVG(slippage_pct) as avg_slip,
            MAX(leverage_actual) as max_lev
        FROM operational_metrics
        WHERE timestamp > NOW() - INTERVAL '24 hours';
    """)
    row = cur.fetchone()
    if row and row[0]:
        avg_c, max_c, avg_s, max_l = row
        print(f"  Avg Cycle Duration : {avg_c:.1f}s")
        print(f"  Max Cycle Duration : {max_c:.1f}s")
        print(f"  Avg Slippage       : {avg_s:.4%}" if avg_s else "  Avg Slippage       : N/A")
        print(f"  Max Leverage Used  : {max_l:.2f}x" if max_l else "  Max Leverage Used  : N/A")
    else:
        print("  Sin datos de operational_metrics en 24h")
except Exception as e:
    print(f"  [operational_metrics no existe]: {e}")

cur.close()
conn.close()

print("\n" + "=" * 70)
print("  FIN DEL INFORME DE TELEMETRÍA")
print("=" * 70)
