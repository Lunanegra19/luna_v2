"""
[DIAG-AUDIT] Auditoría completa de operational_audit_logs y reconciliation_log.
Muestra el momento exacto en que el sistema entró en pausa y por qué.
"""
import psycopg2
from datetime import datetime, timezone

DATABASE_URL = "postgresql://luna_user:luna_secure_pass@localhost:5432/luna_db"

print("[DIAG-AUDIT] Conectando a PostgreSQL...")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # 1. Últimas 20 entradas del audit log operativo
    print("\n=== OPERATIONAL_AUDIT_LOGS (últimas 20 entradas) ===")
    try:
        cur.execute("""
            SELECT id, timestamp, clock_drift_status, nan_inf_status, 
                   leverage_status, api_liveness_status, hmm_status, 
                   latency_status, slippage_status, is_approved, details
            FROM operational_audit_logs 
            ORDER BY timestamp DESC 
            LIMIT 20
        """)
        rows = cur.fetchall()
        if rows:
            for r in rows:
                status = "✅ APROBADO" if r[9] else "❌ FALLIDO"
                print(f"  [{r[1]}] {status}")
                print(f"    clock={r[2]} | nan_inf={r[3]} | leverage={r[4]} | api={r[5]} | hmm={r[6]}")
                print(f"    latency={r[7]} | slippage={r[8]}")
                print(f"    Detalles: {r[10]}")
                print()
        else:
            print("  (Sin registros)")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # 2. Últimas 10 entradas del reconciliation_log
    print("\n=== RECONCILIATION_LOG (últimas 10 entradas) ===")
    try:
        cur.execute("""
            SELECT * FROM reconciliation_log 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        if rows:
            for r in rows:
                for col, val in zip(cols, r):
                    print(f"  {col}: {val}")
                print()
        else:
            print("  (Sin registros)")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # 3. Últimas 20 entradas de audit_logs (trades y decisiones)
    print("\n=== AUDIT_LOGS (últimas 20 decisiones) ===")
    try:
        cur.execute("""
            SELECT timestamp, price, action, confidence, reason
            FROM audit_logs 
            ORDER BY timestamp DESC 
            LIMIT 20
        """)
        rows = cur.fetchall()
        if rows:
            for r in rows:
                print(f"  [{r[0]}] {r[2]} @ ${r[1]:,.2f} | conf={r[3]:.3f} | {r[4][:80]}")
        else:
            print("  (Sin registros)")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # 4. Contar cuántas veces se bloqueó el sistema
    print("\n=== RESUMEN DE BLOQUEOS ===")
    try:
        cur.execute("""
            SELECT COUNT(*) FROM operational_audit_logs WHERE is_approved = FALSE
        """)
        count_failed = cur.fetchone()[0]
        cur.execute("""
            SELECT COUNT(*) FROM operational_audit_logs WHERE is_approved = TRUE
        """)
        count_ok = cur.fetchone()[0]
        print(f"  Ciclos APROBADOS: {count_ok}")
        print(f"  Ciclos FALLIDOS/BLOQUEADOS: {count_failed}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    conn.close()
    print("\n[DIAG-AUDIT] Auditoría completada.")

except Exception as e:
    print(f"[ERROR] Fallo: {e}")
    import traceback
    traceback.print_exc()
