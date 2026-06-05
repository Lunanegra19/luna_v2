"""
[DEEP-AUDIT-XGB] Investigación profunda de la anomalía xgb_prob=0.5000 y clock_drift=300min.
Ejecutar en la VPS: python tools/diagnostics/deep_audit_anomalies.py
"""
import sys, os, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

load_dotenv_ok = False
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv_ok = True
except:
    pass

sys.path.insert(0, "/root/luna_v2")

print("=" * 80)
print("[DEEP-AUDIT] INVESTIGACIÓN DE ANOMALÍAS — LUNA V2 LIVE")
print(f"[DEEP-AUDIT] Ejecutado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════
# ANOMALÍA 1: xgb_prob = 0.5000
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[DEEP-AUDIT-1] INVESTIGANDO: xgb_prob = 0.5000 constante")
print("─" * 60)

try:
    from config.settings import cfg
    seeds = list(cfg.wfb.active_seeds)
    direction_mode = getattr(cfg.fase2, 'direction_mode', 'long')
    print(f"[DEEP-AUDIT-1] Seeds activas: {seeds} | direction_mode: {direction_mode}")
except Exception as e:
    print(f"[DEEP-AUDIT-1] Error leyendo settings: {e}")
    seeds = [99, 1337, 2025]
    direction_mode = 'long'

# Inspeccionar los modelos cargados por seed
data_dir = Path("/root/luna_v2/data")
for seed in seeds:
    seed_dir = data_dir / f"seed_{seed}"
    print(f"\n[DEEP-AUDIT-1] === Semilla {seed} ===")

    # Verificar archivos de modelo
    model_files = list(seed_dir.glob("**/*.joblib")) + list(seed_dir.glob("**/*.pkl")) + list(seed_dir.glob("**/*.json"))
    xgb_files = [f for f in model_files if 'xgb' in f.name.lower() or 'lgbm' in f.name.lower() or 'model' in f.name.lower()]
    cal_files = [f for f in model_files if 'calib' in f.name.lower() or 'isoton' in f.name.lower()]

    print(f"  XGB/LGBM models: {[f.name for f in xgb_files[:5]]}")
    print(f"  Calibradores: {[f.name for f in cal_files[:5]]}")

    # Verificar si los modelos son mocks
    for mf in xgb_files[:3]:
        try:
            with open(mf, 'rb') as f:
                header = f.read(100)
            if b'"mocked": true' in header or b'"mocked":true' in header:
                print(f"  [DEEP-AUDIT-1] ⚠️ MOCK DETECTADO en {mf.name}")
            else:
                stat = mf.stat()
                print(f"  [DEEP-AUDIT-1] Modelo real: {mf.name} ({stat.st_size/1024:.1f} KB)")
        except Exception as e:
            print(f"  [DEEP-AUDIT-1] Error leyendo {mf.name}: {e}")

    # Intentar inferencia real con el RegimeRouter
    try:
        from luna.live.ensemble_live_inference import EnsembleRegimeRouter
        router = EnsembleRegimeRouter(seed_dir, direction='long')
        print(f"  [DEEP-AUDIT-1] RegimeRouter cargado. Agentes: {list(router.agents.keys()) if hasattr(router, 'agents') else 'N/A'}")

        # Crear DataFrame de prueba mínimo para ver qué calibrated devuelve
        # Cargar features reales del último ciclo
        try:
            from luna.live.live_inference import LiveDataCollector
            print(f"  [DEEP-AUDIT-1] Intentando cargar datos reales...")
            # Buscar parquet cacheado
            cache_files = list((data_dir / "cache").glob(f"*{seed}*.parquet")) if (data_dir / "cache").exists() else []
            if not cache_files:
                cache_files = list(data_dir.glob(f"**/*{seed}*.parquet"))
            if cache_files:
                cf = sorted(cache_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                df_test = pd.read_parquet(cf).tail(200)
                print(f"  [DEEP-AUDIT-1] Parquet cargado: {cf.name} ({len(df_test)} filas)")

                result = router.route_and_predict(df_test)
                last_raw = result['raw'].iloc[-1]
                last_cal = result['calibrated'].iloc[-1]
                print(f"  [DEEP-AUDIT-1] ✅ Predicción: raw={last_raw:.6f} | calibrated={last_cal:.6f}")
                if abs(last_cal - 0.5) < 0.001:
                    print(f"  [DEEP-AUDIT-1] 🔴 CONFIRMADO: calibrated=0.5000 → posible mock o NaN propagado")
                    # Inspeccionar distribución
                    print(f"  [DEEP-AUDIT-1] Distribución calibrated: {result['calibrated'].describe()}")
                    print(f"  [DEEP-AUDIT-1] NaN en calibrated: {result['calibrated'].isna().sum()}")
                else:
                    print(f"  [DEEP-AUDIT-1] ✅ Calibrated varía normalmente")
            else:
                print(f"  [DEEP-AUDIT-1] No se encontraron parquets para seed {seed}")
        except Exception as e2:
            print(f"  [DEEP-AUDIT-1] Error en inferencia: {e2}")
    except Exception as e:
        print(f"  [DEEP-AUDIT-1] Error cargando router: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# ANOMALÍA 2: Clock drift = 300 minutos
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[DEEP-AUDIT-2] INVESTIGANDO: Clock drift = 300 minutos a las 00:00 UTC")
print("─" * 60)

# El drift se calcula como: (now_utc - df_live.index.max()).total_seconds() / 60
# A las 00:00 UTC el drift fue 300 min = 5 horas
# Eso significa que el último dato del DataFrame tenía timestamp de ~19:00 UTC del día anterior
# Esto es un problema de fetch incremental: el DataCollector no obtuvo datos nuevos

# Verificar el código del DataCollector
try:
    from luna.live.live_inference import LiveDataCollector
    print(f"[DEEP-AUDIT-2] LiveDataCollector importado.")
    # Buscar el método de fetch
    import inspect
    source = inspect.getsource(LiveDataCollector)
    # Buscar lógica de fetch incremental
    if 'incremental' in source.lower() or 'since' in source.lower():
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if any(kw in line for kw in ['since', 'incremental', 'fetch', 'window', 'last_ts', 'utc']):
                print(f"  L{i}: {line}")
except Exception as e:
    print(f"[DEEP-AUDIT-2] Error inspeccionando LiveDataCollector: {e}")

# Verificar qué pasó en la DB a las 00:00
try:
    from luna.database.db_manager import DatabaseManager
    from psycopg2.extras import DictCursor
    db = DatabaseManager()
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # Buscar los operational_audit_logs alrededor de la medianoche
            cur.execute("""
                SELECT timestamp, clock_drift_minutes, clock_drift_status, is_approved, details, failure_reason
                FROM operational_audit_logs
                WHERE timestamp BETWEEN '2026-05-24 23:30:00' AND '2026-05-25 00:30:00'
                ORDER BY timestamp ASC;
            """)
            rows = cur.fetchall()
            print(f"\n[DEEP-AUDIT-2] Registros auditoría alrededor de medianoche:")
            for r in rows:
                ts = r['timestamp'].strftime('%H:%M:%S') if hasattr(r['timestamp'], 'strftime') else str(r['timestamp'])
                print(f"  [{ts}] Drift={r['clock_drift_minutes']:.1f}m | Status={r['clock_drift_status']} | OK={r['is_approved']}")
                if r['failure_reason']:
                    print(f"         Reason: {r['failure_reason']}")
                if r['details']:
                    try:
                        det = json.loads(r['details']) if isinstance(r['details'], str) else r['details']
                        print(f"         Details: {det}")
                    except:
                        print(f"         Details: {r['details']}")

            # ¿Hay gaps en audit_logs alrededor de medianoche?
            cur.execute("""
                SELECT timestamp, action, reason
                FROM audit_logs
                WHERE timestamp BETWEEN '2026-05-24 23:30:00' AND '2026-05-25 01:00:00'
                ORDER BY timestamp ASC;
            """)
            al_rows = cur.fetchall()
            print(f"\n[DEEP-AUDIT-2] audit_logs alrededor de medianoche: {len(al_rows)} registros")
            for r in al_rows:
                ts = r['timestamp'].strftime('%H:%M:%S') if hasattr(r['timestamp'], 'strftime') else str(r['timestamp'])
                print(f"  [{ts}] {r['action']} | {str(r['reason'])[:100]}")

except Exception as e:
    import traceback
    print(f"[DEEP-AUDIT-2] Error DB: {e}\n{traceback.format_exc()}")

print("\n" + "=" * 80)
print("[DEEP-AUDIT] FIN INVESTIGACIÓN")
print("=" * 80)
