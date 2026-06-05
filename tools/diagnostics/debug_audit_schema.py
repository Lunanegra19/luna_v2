"""Debug completo: schema real de audit_logs y operational_audit_logs."""
import sys
from pathlib import Path
sys.path.insert(0, "/root/luna_v2")
from luna.database.db_manager import DatabaseManager
from psycopg2.extras import DictCursor

db = DatabaseManager()

with db.get_connection() as conn:
    conn.autocommit = True
    with conn.cursor(cursor_factory=DictCursor) as cur:

        print("=== SCHEMA: audit_logs ===")
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'audit_logs' ORDER BY ordinal_position
        """)
        for r in cur.fetchall():
            print(f"  {r['column_name']}: {r['data_type']}")

        print("\n=== ULTIMAS 3 FILAS: audit_logs ===")
        cur.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 3")
        rows = cur.fetchall()
        desc = [d[0] for d in cur.description]
        for row in rows:
            d = dict(zip(desc, row))
            for k, v in d.items():
                print(f"  {k}: {v}")
            print("  ---")

        print("\n=== SCHEMA: operational_audit_logs ===")
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'operational_audit_logs' ORDER BY ordinal_position
        """)
        for r in cur.fetchall():
            print(f"  {r['column_name']}: {r['data_type']}")

        print("\n=== ULTIMA FILA: operational_audit_logs ===")
        cur.execute("SELECT * FROM operational_audit_logs ORDER BY id DESC LIMIT 1")
        rows = cur.fetchall()
        desc = [d[0] for d in cur.description]
        for row in rows:
            d = dict(zip(desc, row))
            for k, v in d.items():
                print(f"  {k}: {str(v)[:200]}")
