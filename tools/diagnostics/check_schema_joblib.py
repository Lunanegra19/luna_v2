"""Verifica schema real de system_heartbeat y comportamiento real de joblib.load con pkl."""
import sys
from pathlib import Path
sys.path.insert(0, "/root/luna_v2")

from luna.database.db_manager import DatabaseManager
from psycopg2.extras import DictCursor

db = DatabaseManager()
with db.get_connection() as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        # Schema real de system_heartbeat
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'system_heartbeat'
            ORDER BY ordinal_position
        """)
        print("=== system_heartbeat schema ===")
        cols = cur.fetchall()
        for c in cols:
            print(f"  {c[0]}: {c[1]}")

        # Última fila
        cur.execute("SELECT * FROM system_heartbeat ORDER BY id DESC LIMIT 2")
        rows = cur.fetchall()
        for r in rows:
            print(f"  row: {dict(zip([d[0] for d in cur.description], r))}")

import joblib
print("\n=== joblib.load test ===")
pkl_path = Path("/root/luna_v2/data/models/prod/seed99/ood_guard.pkl")
obj = joblib.load(pkl_path)
print(f"type(obj): {type(obj)}")
print(f"isinstance(int): {isinstance(obj, int)}")
print(f"repr: {repr(obj)[:100]}")
