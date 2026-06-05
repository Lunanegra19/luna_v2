import psycopg2
import os
from dotenv import load_dotenv

load_dotenv('/root/luna_v2/.env')
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# === AUDIT_LOGS ===
cur.execute("SELECT COUNT(*) FROM audit_logs WHERE action NOT IN ('HOLD') AND action IS NOT NULL")
total_trades = cur.fetchone()[0]
print(f'[SHADOW-TRADES] Total acciones no-HOLD: {total_trades}')

cur.execute("SELECT action, COUNT(*) FROM audit_logs GROUP BY action ORDER BY COUNT(*) DESC")
print('[BREAKDOWN] Por tipo:')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]}')

cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM audit_logs")
mn, mx = cur.fetchone()
print(f'[RANGE] Desde: {mn} | Hasta: {mx}')

cur.execute("SELECT COUNT(*) FROM audit_logs WHERE action NOT IN ('HOLD') AND timestamp >= NOW() - INTERVAL '7 days'")
recent_7d = cur.fetchone()[0]
print(f'[7D] Ultimos 7 dias: {recent_7d} trades')

cur.execute("SELECT COUNT(*) FROM audit_logs WHERE action NOT IN ('HOLD') AND timestamp >= NOW() - INTERVAL '30 days'")
recent_30d = cur.fetchone()[0]
print(f'[30D] Ultimos 30 dias: {recent_30d} trades')

# Tasa diaria aproximada
if mn and mx:
    from datetime import timezone
    if mn.tzinfo is None:
        import datetime
        mn = mn.replace(tzinfo=timezone.utc)
        mx = mx.replace(tzinfo=timezone.utc)
    days_running = max((mx - mn).days, 1)
    rate = total_trades / days_running
    remaining_to_r8 = max(0, 100 - total_trades)
    eta_days = remaining_to_r8 / rate if rate > 0 else 9999
    print(f'[STATS] Dias en operacion: {days_running} | Tasa: {rate:.1f} trades/dia')
    print(f'[SOP-R8] Progreso: {total_trades}/100 trades | Faltan: {remaining_to_r8} | ETA: {eta_days:.0f} dias')

# Ultimas 10 acciones no-HOLD
print()
print('[ULTIMAS-10] Ultimos 10 trades no-HOLD:')
cur.execute("SELECT timestamp, action, xgb_prob, hmm_regime, confidence FROM audit_logs WHERE action NOT IN ('HOLD') ORDER BY timestamp DESC LIMIT 10")
for row in cur.fetchall():
    print(f'  {str(row[0])[:19]} | {row[1]:5s} | XGB={row[2]:.4f} | HMM={row[3]} | conf={row[4]:.2f}')

# === TRADE_LOG si existe ===
cur.execute("SELECT COUNT(*) FROM audit_logs WHERE contracts > 0")
with_contracts = cur.fetchone()[0]
print(f'\n[EXEC] Trades con contratos > 0 (ejecutados realmente): {with_contracts}')

conn.close()
print('[DONE] Consulta completada.')
