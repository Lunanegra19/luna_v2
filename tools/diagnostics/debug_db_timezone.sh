#!/bin/bash
# Check audit_logs via psql using peer auth (no password needed as root)
sudo -u postgres psql -d luna_db -c "SELECT id, timestamp, action, xgb_prob, hmm_regime FROM audit_logs WHERE timestamp >= '2026-05-25 19:00:00' AND timestamp <= '2026-05-25 19:59:59' ORDER BY id DESC LIMIT 10;" 2>&1 || \
psql -U luna_user -d luna_db -c "SELECT id, timestamp, action, xgb_prob, hmm_regime FROM audit_logs ORDER BY id DESC LIMIT 10;" 2>&1 || \
echo "DB access failed - checking luna_v2 DatabaseManager..."
cd /root/luna_v2
/root/miniconda3/envs/luna_env/bin/python - <<'EOF'
import sys
sys.path.insert(0, '/root/luna_v2')
from luna.data.database import DatabaseManager
db = DatabaseManager()
with db.get_connection() as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        print("=== Últimos 5 registros en audit_logs ===")
        cur.execute("SELECT id, timestamp, action, xgb_prob, hmm_regime FROM audit_logs ORDER BY id DESC LIMIT 5")
        for r in cur.fetchall():
            print(f"  id={r[0]} ts={r[1]} action={r[2]} xgb={r[3]} hmm={r[4]}")
        
        print("\n=== Ventana 19:00-19:59 UTC (21:00 CEST) ===")
        cur.execute("SELECT id, timestamp, action FROM audit_logs WHERE timestamp >= '2026-05-25T19:00:00Z' AND timestamp <= '2026-05-25T19:59:59Z' ORDER BY id DESC LIMIT 5")
        rows = cur.fetchall()
        if rows:
            for r in rows:
                print(f"  ✅ id={r[0]} ts={r[1]} action={r[2]}")
        else:
            print("  ❌ VACÍO - ningún registro en ventana 19:00-19:59 UTC")
        
        print("\n=== DB timezone ===")
        cur.execute("SHOW timezone")
        print(f"  timezone: {cur.fetchone()[0]}")
        cur.execute("SELECT NOW()")
        print(f"  NOW(): {cur.fetchone()[0]}")
        
        print("\n=== Rango total hoy ===")
        cur.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM audit_logs WHERE timestamp::date = '2026-05-25'")
        r = cur.fetchone()
        print(f"  MIN={r[0]} MAX={r[1]} count={r[2]}")
EOF
