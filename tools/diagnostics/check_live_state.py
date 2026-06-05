"""
[DIAG-LIVE-STATE] Diagnóstico completo del estado live_state en PostgreSQL.
Incluye is_paused, drawdown, portfolio, timestamps y motivo del bloqueo.
"""
import psycopg2
import json
from datetime import datetime, timezone

DATABASE_URL = "postgresql://luna_user:luna_secure_pass@localhost:5432/luna_db"

print("[DIAG-LIVE-STATE] Conectando a PostgreSQL...")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # 1. Estado completo de live_state
    cur.execute("SELECT * FROM live_state WHERE id=1")
    columns = [desc[0] for desc in cur.description]
    row = cur.fetchone()
    if row:
        print("\n=== LIVE_STATE (id=1) ===")
        for col, val in zip(columns, row):
            print(f"  {col}: {val}")
    else:
        print("[ERROR] No existe fila con id=1 en live_state!")

    # 2. Verificar si is_paused es True
    is_paused_val = None
    for col, val in zip(columns, row):
        if col == 'is_paused':
            is_paused_val = val
    
    print(f"\n[DIAG] is_paused = {is_paused_val}")
    if is_paused_val:
        print("[ALERTA] SISTEMA EN PAUSA - Esta es la causa del bloqueo en Telegram!")
        print("[DIAG] Drawdown actual:")
        for col, val in zip(columns, row):
            if 'draw' in col.lower() or 'dd' in col.lower():
                print(f"  {col} = {val}")

    # 3. Buscar columnas de drawdown y pausa
    print("\n=== TODAS LAS COLUMNAS DISPONIBLES ===")
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'live_state' 
        ORDER BY ordinal_position
    """)
    for col_name, col_type in cur.fetchall():
        print(f"  {col_name} ({col_type})")

    # 4. Verificar si hay tabla de circuit breakers o pausas
    cur.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    print("\n=== TABLAS EN BD ===")
    for t in cur.fetchall():
        print(f"  {t[0]}")

    # 5. Historial de heartbeats recientes
    try:
        cur.execute("""
            SELECT created_at, event_type, details 
            FROM live_events 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        rows = cur.fetchall()
        print("\n=== ÚLTIMOS 10 EVENTOS LIVE ===")
        for r in rows:
            print(f"  {r[0]} | {r[1]} | {r[2]}")
    except Exception as e:
        print(f"[INFO] No hay tabla live_events o error: {e}")

    conn.close()
    print("\n[DIAG-LIVE-STATE] Diagnóstico completado.")

except Exception as e:
    print(f"[ERROR] Fallo en diagnóstico: {e}")
    import traceback
    traceback.print_exc()
