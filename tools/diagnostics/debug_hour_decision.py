"""Debug del endpoint hour-decision y operational_audit_logs para entender qué datos hay."""
import sys, requests, json
from datetime import datetime

sys.path.insert(0, "/root/luna_v2")

BASE = "http://localhost:5000"

for hour in [18, 19, 20]:
    print(f"\n{'='*60}")
    print(f"HORA: {hour}:00")
    print(f"{'='*60}")
    try:
        r = requests.get(f"{BASE}/api/vps/hour-decision", params={"hour": hour}, timeout=10)
        data = r.json()
        print(f"  status: {data.get('status')}")
        print(f"  hour_label: {data.get('hour_label')}")
        print(f"  action: {data.get('action')}")
        print(f"  consensus_count: {data.get('consensus_count')}")
        print(f"  total_seeds: {data.get('total_seeds')}")
        print(f"  inference_time: {data.get('inference_time_seconds')}")
        print(f"  hmm_regime: {data.get('hmm_regime')}")
        print(f"  ensemble_prob: {data.get('ensemble_prob')}")
        print(f"  xgb_prob: {data.get('xgb_prob')}")
        steps = data.get("step_logs", [])
        print(f"  step_logs count: {len(steps)}")
        for i, s in enumerate(steps):
            preview = str(s)[:80] if s else "(vacío)"
            print(f"    Paso {i+1}: {preview}")
        # Op audit
        op = data.get("operational_audit")
        if op:
            print(f"  operational_audit keys: {list(op.keys()) if isinstance(op, dict) else 'no-dict'}")
    except Exception as e:
        print(f"  ERROR: {e}")

# Ver directamente los datos en la BD
from luna.database.db_manager import DatabaseManager
from psycopg2.extras import DictCursor
db = DatabaseManager()
print(f"\n{'='*60}")
print("audit_logs — últimas 5 filas")
print(f"{'='*60}")
with db.get_connection() as conn:
    conn.autocommit = True
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT id, timestamp, action, consensus_count, hmm_regime, 
                   ensemble_prob, xgb_prob, inference_time_ms
            FROM audit_logs ORDER BY id DESC LIMIT 5
        """)
        for row in cur.fetchall():
            print(f"  {dict(row)}")

print(f"\n{'='*60}")
print("operational_audit_logs — últimas 3 filas (columnas)")
print(f"{'='*60}")
with db.get_connection() as conn:
    conn.autocommit = True
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'operational_audit_logs' ORDER BY ordinal_position
        """)
        cols = [r[0] for r in cur.fetchall()]
        print(f"  Columnas: {cols}")
        cur.execute("SELECT * FROM operational_audit_logs ORDER BY id DESC LIMIT 3")
        for row in cur.fetchall():
            d = dict(row)
            print(f"  {d}")
