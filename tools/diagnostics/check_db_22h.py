"""Check DB records for 22:00 local (20:00 UTC) slot"""
import psycopg2
import sys

try:
    conn = psycopg2.connect(dbname='luna_db', user='luna_user', host='127.0.0.1', port=5432)
    cur = conn.cursor()
    
    # Check 20:00 UTC = 22:00 local
    print("=== Records 20:00-21:00 UTC (22:00-23:00 local) ===")
    cur.execute("""
        SELECT timestamp, action, confidence, hmm_regime, reason
        FROM live_trading_decisions
        WHERE timestamp >= '2026-05-25 20:00:00'
          AND timestamp < '2026-05-25 21:00:00'
        ORDER BY timestamp
        LIMIT 10
    """)
    rows = cur.fetchall()
    if rows:
        for r in rows:
            print(f"  {r[0]} | {r[1]} | conf={r[2]} | regime={r[3]}")
            print(f"    reason: {str(r[4])[:100]}")
    else:
        print("  SIN DATOS en ese rango UTC")
    
    print("\n=== Últimos 5 records en la DB ===")
    cur.execute("""
        SELECT timestamp, action, confidence, hmm_regime
        FROM live_trading_decisions
        ORDER BY timestamp DESC LIMIT 5
    """)
    for r in cur.fetchall():
        print(f"  {r[0]} | {r[1]} | conf={r[2]} | regime={r[3]}")
    
    print("\n=== Total records hoy ===")
    cur.execute("SELECT COUNT(*) FROM live_trading_decisions WHERE timestamp::date = '2026-05-25'")
    print(f"  {cur.fetchone()[0]} records del 2026-05-25")
    
    cur.close()
    conn.close()
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
