"""
audit_db_trades.py — Audita los conteos de trades en la DB vs lo mostrado en el dashboard.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psycopg2
from dotenv import load_dotenv
load_dotenv("/root/luna_v2/.env")

db_url = os.environ.get("DATABASE_URL", "postgresql://luna_user:luna_pass@localhost:5432/luna_db")
conn = psycopg2.connect(db_url)
cur = conn.cursor()

print("=" * 60)
print("AUDIT DB TRADES")
print("=" * 60)

cur.execute("SELECT COUNT(*) FROM audit_logs")
total = cur.fetchone()[0]
print(f"\naudit_logs total filas: {total}")

cur.execute("SELECT action, COUNT(*) FROM audit_logs GROUP BY action ORDER BY COUNT(*) DESC")
rows = cur.fetchall()
print("\nDesglose por action:")
for r in rows:
    print(f"  {r[0]:<10} : {r[1]:>5}")

cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM audit_logs")
r = cur.fetchone()
print(f"\nRango temporal: {r[0]} -> {r[1]}")

cur.execute("SELECT COUNT(*) FROM audit_logs WHERE action IN ('LONG', 'SHORT')")
trades_real = cur.fetchone()[0]
print(f"\nTrades reales (LONG+SHORT): {trades_real}")

cur.execute("SELECT COUNT(*) FROM audit_logs WHERE action = 'HOLD'")
holds = cur.fetchone()[0]
print(f"HOLDs: {holds}")

# Ver si hay tabla de trades separada
cur.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
""")
tables = [r[0] for r in cur.fetchall()]
print(f"\nTablas en DB: {tables}")

# Buscar tabla de trades si existe
for t in ["trades", "transactions", "orders", "live_trades"]:
    if t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        cnt = cur.fetchone()[0]
        print(f"\nTabla '{t}': {cnt} filas")
        cur.execute(f"SELECT * FROM {t} ORDER BY id DESC LIMIT 3")
        for r in cur.fetchall():
            print(f"  {r}")

conn.close()
print("\n" + "=" * 60)
