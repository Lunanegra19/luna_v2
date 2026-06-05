"""
db_schema_migration_v3.py (v3 final)
=====================================
FIX-06/07: Migración de schema PostgreSQL no destructiva.
Usa la API correcta de DatabaseManager: `with db.get_connection() as conn:`
"""
import sys
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

print("[DB-MIGRATION-V3] Iniciando migración de schema PostgreSQL...")

try:
    from luna.database.db_manager import DatabaseManager
    db = DatabaseManager()
    print("[DB-MIGRATION-V3] Conexión a PostgreSQL establecida.")
except Exception as e:
    print(f"[DB-MIGRATION-V3][FATAL] No se pudo conectar: {e}")
    sys.exit(1)

# get_connection() es un generator/context manager decorado con @contextmanager
# Uso correcto: with db.get_connection() as conn:
with db.get_connection() as conn:
    print(f"[DB-MIGRATION-V3] Connection del pool activa: {type(conn).__name__}")

    try:
        # ── FIX-06: Añadir columnas a audit_logs ─────────────────────────────
        print("\n[FIX-06] Añadiendo columnas a audit_logs...")
        fix06_migrations = [
            ("cycle_id",         "ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS cycle_id VARCHAR(64);"),
            ("seed",             "ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS seed INTEGER;"),
            ("consensus_count",  "ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS consensus_count INTEGER;"),
            ("regime_semantic",  "ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS regime_semantic VARCHAR(64);"),
            ("meta_prob",        "ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS meta_prob FLOAT;"),
        ]

        for col_name, sql in fix06_migrations:
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()
                print(f"[FIX-06] OK: columna '{col_name}'")
            except Exception as e:
                conn.rollback()
                print(f"[FIX-06][ERROR] Fallo '{col_name}': {e}")

        # ── FIX-07: Crear tabla transactions ─────────────────────────────────
        print("\n[FIX-07] Creando tabla transactions...")
        fix07_sql = """
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            cycle_id VARCHAR(64),
            seed INTEGER,
            action VARCHAR(10),
            contracts FLOAT,
            entry_price FLOAT,
            exit_price FLOAT,
            pnl_usd FLOAT,
            regime VARCHAR(64),
            consensus_count INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        try:
            with conn.cursor() as cur:
                cur.execute(fix07_sql)
            conn.commit()
            print("[FIX-07] OK: Tabla transactions creada (o ya existía).")
        except Exception as e:
            conn.rollback()
            print(f"[FIX-07][ERROR]: {e}")

        # ── Verificación post-migración ───────────────────────────────────────
        print("\n[DB-MIGRATION-V3] Verificando schema...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'audit_logs'
                ORDER BY ordinal_position;
            """)
            cols = cur.fetchall()

        existing = [c[0] for c in cols]
        print(f"[DB-MIGRATION-V3] audit_logs tiene {len(cols)} columnas:")
        for cn, ct in cols:
            print(f"  - {cn}: {ct}")

        for nc in ['cycle_id', 'seed', 'consensus_count', 'regime_semantic', 'meta_prob']:
            status = "PASS" if nc in existing else "FAIL"
            print(f"[FIX-06] {status}: columna '{nc}'")

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='transactions';")
            has_tx = cur.fetchone()[0] > 0
        print(f"[FIX-07] {'PASS' if has_tx else 'FAIL'}: tabla transactions {'EXISTE' if has_tx else 'NO ENCONTRADA'}")

    except Exception as e:
        print(f"[DB-MIGRATION-V3][FATAL]: {e}")
        import traceback
        traceback.print_exc()

print("\n[DB-MIGRATION-V3] Migración completada. Conexión devuelta al pool.")
