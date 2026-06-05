"""
[FIX-CLOCK-DRIFT-PAUSE] Fix de emergencia: limpia is_paused=True causado por
ClockDrift en ciclo de medianoche del 2026-05-27 00:00 UTC.

Causa raíz: El LiveOperationalAuditor detectó ClockDrift porque el fetcher
incremental en el ciclo de medianoche tardó/falló, dejando features_live.parquet
con timestamp inferior a max_drift_minutes (90 min). Esto disparó is_paused=True
en run_live_trader.py línea 555. El fix anterior (fix_is_paused.py) resolvió
el estado de BD pero NO impidió que el auditor volviera a dispararlo.

Este script:
1. Lee el estado actual
2. Limpia is_paused=False
3. Verifica el timestamp del último feature vivo para diagnosticar si el drift persiste
"""

import psycopg2
import os
from datetime import datetime, timezone
import pyarrow.parquet as pq

DATABASE_URL = "postgresql://luna_user:luna_secure_pass@localhost:5432/luna_db"
FEATURES_PATH = "/root/luna_v2/data/features/features_live.parquet"

print("[FIX-CLOCK-DRIFT-PAUSE] === Iniciando fix de desbloqueo ===")
print(f"[FIX-CLOCK-DRIFT-PAUSE] Hora actual UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

# 1. Verificar features_live.parquet timestamp
print(f"\n[FIX-CLOCK-DRIFT-PAUSE] Verificando features_live.parquet...")
try:
    pf = pq.read_table(FEATURES_PATH)
    df_meta = pf.to_pandas()
    if not df_meta.empty:
        last_ts = df_meta.index.max()
        import pandas as pd
        if hasattr(last_ts, 'tzinfo') and last_ts.tzinfo is None:
            last_ts = pd.Timestamp(last_ts).tz_localize('UTC')
        now_utc = datetime.now(timezone.utc)
        drift_min = (now_utc - last_ts.to_pydatetime()).total_seconds() / 60.0
        print(f"[FIX-CLOCK-DRIFT-PAUSE] Último timestamp features_live: {last_ts}")
        print(f"[FIX-CLOCK-DRIFT-PAUSE] Drift actual: {drift_min:.1f} minutos")
        if drift_min > 90:
            print(f"[FIX-CLOCK-DRIFT-PAUSE] ⚠️ ALERTA: Drift = {drift_min:.1f} min > 90 min. Si se desbloquea el sistema, el Auditor VOLVERÁ a dispararse inmediatamente.")
            print(f"[FIX-CLOCK-DRIFT-PAUSE] El pipeline de datos deberá actualizarse antes de que el desbloqueo sea efectivo.")
        else:
            print(f"[FIX-CLOCK-DRIFT-PAUSE] ✅ Drift dentro de límite. El desbloqueo será estable.")
except Exception as e:
    print(f"[FIX-CLOCK-DRIFT-PAUSE] [ERROR] No se pudo leer features_live.parquet: {e}")

# 2. Limpiar is_paused en BD
print(f"\n[FIX-CLOCK-DRIFT-PAUSE] Conectando a PostgreSQL para limpiar is_paused...")
try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Leer estado actual
    cur.execute("SELECT id, portfolio_value, ath, drawdown, is_paused, updated_at FROM live_state WHERE id=1")
    row = cur.fetchone()
    if row:
        print(f"[FIX-CLOCK-DRIFT-PAUSE] Estado actual: portfolio={row[1]} | ath={row[2]} | dd={row[3]:.4f} | is_paused={row[4]} | updated={row[5]}")
        
        if row[4]:  # is_paused == True
            cur.execute("UPDATE live_state SET is_paused = FALSE, updated_at = NOW() WHERE id = 1")
            conn.commit()
            
            # Verificar
            cur.execute("SELECT is_paused FROM live_state WHERE id=1")
            new_paused = cur.fetchone()[0]
            print(f"[FIX-CLOCK-DRIFT-PAUSE] ✅ is_paused limpiado. Nuevo valor: {new_paused}")
        else:
            print(f"[FIX-CLOCK-DRIFT-PAUSE] is_paused ya era False. No se requiere acción.")
    else:
        print(f"[FIX-CLOCK-DRIFT-PAUSE] [ERROR] No existe fila id=1 en live_state.")
    
    conn.close()

except Exception as e:
    print(f"[FIX-CLOCK-DRIFT-PAUSE] [ERROR] Fallo en BD: {e}")
    import traceback
    traceback.print_exc()

print(f"\n[FIX-CLOCK-DRIFT-PAUSE] === Fix completado ===")
print("[FIX-CLOCK-DRIFT-PAUSE] SIGUIENTE PASO: Si el drift persiste, esperar al próximo ciclo de hora en punto")
print("[FIX-CLOCK-DRIFT-PAUSE] para que el DataCollector actualice features_live.parquet con datos frescos.")
print("[FIX-CLOCK-DRIFT-PAUSE] Si el drift sigue >90min tras 2 ciclos, revisar el fetcher incremental.")
