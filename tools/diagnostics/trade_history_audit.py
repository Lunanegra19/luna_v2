"""
trade_history_audit.py — Ver qué trades hay en DB y si son test o shadow trading real
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dotenv import load_dotenv
load_dotenv("/root/luna_v2/.env")
import psycopg2

conn = psycopg2.connect(os.environ.get("DATABASE_URL"))
cur = conn.cursor()

print("=" * 80)
print("HISTORIAL COMPLETO DE TRADES (LONG + SHORT) EN DB")
print("=" * 80)

cur.execute("""
    SELECT timestamp, action, reason
    FROM audit_logs
    WHERE action IN ('LONG','SHORT')
    ORDER BY timestamp
""")
rows = cur.fetchall()
for r in rows:
    reason_short = str(r[2])[:100] if r[2] else ""
    print(f"{r[0]} | {r[1]:<6} | {reason_short}")

print(f"\nTotal: {len(rows)} trades")

# Ver primeros HOLD para contexto
cur.execute("""
    SELECT timestamp, action FROM audit_logs ORDER BY timestamp LIMIT 5
""")
print("\nPrimeros 5 registros (contexto de inicio):")
for r in cur.fetchall():
    print(f"  {r[0]} | {r[1]}")

cur.execute("""
    SELECT timestamp, action FROM audit_logs ORDER BY timestamp DESC LIMIT 5
""")
print("\nUltimos 5 registros:")
for r in cur.fetchall():
    print(f"  {r[0]} | {r[1]}")

conn.close()
