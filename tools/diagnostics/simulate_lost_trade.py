"""
Simulación del trade LONG perdido el 2026-05-24 21:26 UTC.
Calcula el P&L que habría tenido si la orden se hubiera ejecutado.
"""
import pandas as pd

# Datos del trade bloqueado (extraídos de los logs)
ENTRY_PRICE   = 53458.40   # precio límite de la orden
ENTRY_TIME    = "2026-05-24 21:00:00"
POSITION_USD  = 1577.47    # tamaño asignado por Kelly
LEVERAGE      = 0.02       # leverage asignado (muy conservador, casi spot)
MAKER_FEE     = 0.0002     # 0.02% maker OKX

# Cargar OHLCV
df = pd.read_parquet('/root/luna_v2/data/raw/ohlcv/ohlcv_raw.parquet')
df.index = pd.to_datetime(df.index, utc=True)

entry_dt = pd.Timestamp(ENTRY_TIME).tz_localize('UTC')
window = df[df.index >= entry_dt][['open','high','low','close']].head(72)  # 72h = 3 dias

# Calcular P&L por hora si se hubiera mantenido abierto
entry_fee = POSITION_USD * MAKER_FEE
exit_fee  = POSITION_USD * MAKER_FEE  # asumiendo también maker en cierre
total_fees = entry_fee + exit_fee

print("=" * 80)
print("  SIMULACIÓN: Trade LONG Perdido — 2026-05-24 21:26 UTC")
print(f"  Entry: ${ENTRY_PRICE:,.2f} | Tamaño: ${POSITION_USD:,.2f} | Leverage: {LEVERAGE:.2f}x")
print(f"  Comisiones totales (round-trip maker): ${total_fees:.4f} ({(total_fees/POSITION_USD)*100:.3f}%)")
print("=" * 80)
print(f"\n{'Hora (UTC)':<30} {'Close':>10} {'Δ% vs entry':>12} {'P&L bruto':>12} {'P&L neto':>12}")
print("-" * 78)

for ts, row in window.iterrows():
    pct = (row['close'] / ENTRY_PRICE - 1)
    pnl_gross = POSITION_USD * pct
    pnl_net   = pnl_gross - total_fees
    marker = ""
    if abs(pct) > 0.02:
        marker = " ←"
    print(f"{str(ts):<30} ${row['close']:>9,.2f} {pct*100:>+11.2f}% ${pnl_gross:>+10.2f} ${pnl_net:>+10.2f}{marker}")

print()
print("=" * 80)
# Resumen estadístico
all_pct = [(row['close'] / ENTRY_PRICE - 1) for _, row in window.iterrows()]
peak_pct = max(all_pct)
trough_pct = min(all_pct)
final_pct = all_pct[-1]

peak_pnl   = POSITION_USD * peak_pct - total_fees
trough_pnl = POSITION_USD * trough_pct - total_fees
final_pnl  = POSITION_USD * final_pct - total_fees

print(f"  PEAK (mejor momento):    {peak_pct*100:+.2f}%  →  P&L neto: ${peak_pnl:+.2f}")
print(f"  TROUGH (peor momento):   {trough_pct*100:+.2f}%  →  P&L neto: ${trough_pnl:+.2f}")
print(f"  CLOSE 72h después:       {final_pct*100:+.2f}%  →  P&L neto: ${final_pnl:+.2f}")
print(f"\n  Las comisiones round-trip fueron solo: ${total_fees:.4f}")
print("=" * 80)
