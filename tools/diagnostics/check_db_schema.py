import psycopg2
import os
from dotenv import load_dotenv

load_dotenv('/root/luna_v2/.env')
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Columnas de audit_logs
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='audit_logs' ORDER BY ordinal_position")
cols = [r[0] for r in cur.fetchall()]
print('Columnas audit_logs:', cols)

# Columnas de trade_log si existe
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
tables = [r[0] for r in cur.fetchall()]
print('Tablas publicas:', tables)

conn.close()
