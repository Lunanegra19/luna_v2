"""Debug completo: qué hay en audit_logs para la ventana 19:00-20:00 UTC y por qué el endpoint retorna standby."""
import psycopg2, yaml
from datetime import datetime, timezone

cfg = yaml.safe_load(open('/root/luna_v2/config/settings.yaml'))
db_cfg = cfg.get('deployment', {}).get('postgres', {})

conn = psycopg2.connect(
    host=db_cfg.get('host', 'localhost'),
    port=db_cfg.get('port', 5432),
    dbname=db_cfg.get('dbname', 'luna_db'),
    user=db_cfg.get('user', 'luna_user'),
    password=db_cfg.get('password', '')
)
conn.autocommit = True
cur = conn.cursor()

print("=" * 60)
print("DIAGNÓSTICO COMPLETO: audit_logs hoy")
print("=" * 60)

# 1. Últimos 10 registros de audit_logs
print("\n--- ÚLTIMOS 10 registros de audit_logs ---")
cur.execute("""
    SELECT id, timestamp, action, xgb_prob, hmm_regime, confidence
    FROM audit_logs
    ORDER BY id DESC LIMIT 10
""")
for row in cur.fetchall():
    print(f"  id={row[0]} | ts={row[1]} | action={row[2]} | xgb={row[3]:.3f if row[3] else 'N/A'} | hmm={row[4]} | conf={row[5]}")

# 2. Ventana 19:00-19:59 UTC (21:00-21:59 CEST)
print("\n--- Registros en ventana 19:00-19:59 UTC ---")
cur.execute("""
    SELECT id, timestamp, action, xgb_prob, hmm_regime
    FROM audit_logs
    WHERE timestamp >= '2026-05-25T19:00:00Z' AND timestamp <= '2026-05-25T19:59:59Z'
    ORDER BY id DESC LIMIT 5
""")
rows = cur.fetchall()
if rows:
    for row in rows:
        print(f"  ✅ id={row[0]} | ts={row[1]} | action={row[2]} | xgb={row[3]} | hmm={row[4]}")
else:
    print("  ❌ NINGÚN registro en ventana 19:00-19:59 UTC")

# 3. Timezone de la DB
print("\n--- Timezone configuración DB ---")
cur.execute("SHOW timezone")
print(f"  DB timezone: {cur.fetchone()[0]}")

cur.execute("SELECT NOW(), NOW() AT TIME ZONE 'UTC'")
row = cur.fetchone()
print(f"  NOW() DB: {row[0]}")
print(f"  NOW() UTC: {row[1]}")

# 4. Rangos de timestamps en audit_logs
print("\n--- Rango de timestamps en audit_logs ---")
cur.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM audit_logs WHERE timestamp::date = '2026-05-25'")
row = cur.fetchone()
print(f"  Min: {row[0]} | Max: {row[1]} | Total hoy: {row[2]}")

# 5. ¿Son los timestamps UTC o locales?
cur.execute("SELECT timestamp, timestamp AT TIME ZONE 'UTC', timestamp AT TIME ZONE 'Europe/Berlin' FROM audit_logs ORDER BY id DESC LIMIT 3")
print("\n--- Comparación de timezones en audit_logs ---")
for row in cur.fetchall():
    print(f"  raw={row[0]} | as_utc={row[1]} | as_berlin={row[2]}")

cur.close()
conn.close()
