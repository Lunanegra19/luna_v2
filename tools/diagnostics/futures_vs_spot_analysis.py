"""
tools/diagnostics/futures_vs_spot_analysis.py
Analisis de las implicaciones reales de migrar de Spot a Futuros Perpetuos.
[FUTURES-ANALYSIS 2026-05-30]
"""
BTC_PRICE = 107_000
CONTRACT_BTC = 0.001
CONTRACT_USD = CONTRACT_BTC * BTC_PRICE

print("=== GRANULARIDAD CONTRATOS OKX BTC-USDT-SWAP ===")
print(f"1 contrato = {CONTRACT_BTC} BTC = {CONTRACT_USD:.0f} USD")
print(f"Minimo orden = 1 contrato = {CONTRACT_USD:.0f} USD")
print()

base_capital = 5000
base_risk = 0.20
regimes = {
    "BULL_TREND":      0.25,
    "BULL_TREND_WEAK": 0.12,
    "BULL_TREND_B":    0.20,
    "CALM_RANGE":      0.20,
    "BEAR_TREND":      0.10,
}

print(f"Capital base: {base_capital} USD | Risk fraction: {base_risk:.0%}")
print()
print(f"  {'Regimen':<22} | {'Size USD':>10} | {'Contratos':>10} | Estado")
print("-" * 65)
for regime, cap in regimes.items():
    size = base_capital * base_risk * cap
    n = size / CONTRACT_USD
    estado = "OK" if n >= 1 else "PROBLEMA < 1 contrato"
    print(f"  {regime:<22} | {size:>10.0f} | {n:>10.2f} | {estado}")

print()
print("=== FUNDING RATE (EL COSTE OCULTO REAL) ===")
print()
fund_8h   = 0.01       # 0.01% cada 8H — tasa tipica BTC en tendencia
fund_day  = fund_8h * 3
fund_week = fund_day * 7
print(f"Funding tipico: {fund_8h}% cada 8H = {fund_day:.2f}%/dia = {fund_week:.2f}%/semana")
print()

# Con embargo 72H (nuestro sweet spot)
for h in [72, 96, 168]:
    periods = h / 8
    total_fund = fund_8h * periods
    net = 0.0573 - total_fund
    print(f"Holding {h:3d}H: {periods:.0f} pagos x {fund_8h}% = {total_fund:.3f}% funding | "
          f"ret_medio_win=0.0573% | NETO={net:.4f}%")

print()
print("=== COMPARACION COSTES TOTALES ===")
print()
avg_win  = 0.0573 / 100
avg_loss = 0.0560 / 100

costs = {
    "Spot (actual)":          0.15 / 100,
    "Futures 72H holding":    (0.08 + 0.03 * 3) / 100,
    "Futures 96H holding":    (0.08 + 0.03 * 4) / 100,
}
for label, cost in costs.items():
    wr_be = (cost + avg_loss) / (avg_win + avg_loss)
    print(f"  {label:<28}: coste={cost*100:.3f}% | WR break-even={wr_be:.1%}")

print()
print("  WR bruto actual: 50.9%")
print("  WR ensemble (72H embargo): 57.1%")
print()

print("=== LOS 5 PROBLEMAS REALES DE PASAR A FUTUROS ===")
print()
issues = [
    ("1. Granularidad contratos", "OKX: 1 contrato = 0.001 BTC = ~107 USD",
     "MENOR | Minimo viable a partir de ~107 USD"),
    ("2. Funding rate acumulado", "0.01% c/8H = 0.09%/3dias = come el retorno medio",
     "MAYOR | Retorno medio por trade = 0.057%. Con 72H, funding = 0.09% -> NETO negativo"),
    ("3. Riesgo de liquidacion", "Margin call si posicion se mueve en contra",
     "MAYOR | Con apalancamiento x10-x20, volatilidad BTC puede liquidar en horas"),
    ("4. Retrenamiento completo SHORT", "TBM con side=-1, MetaLabeler SHORT, XGBoost SHORT",
     "MAYOR | Requiere run completa con direction_mode='both' - semanas de trabajo"),
    ("5. Cambio de arquitectura OKX", "Spot API vs Futures API son distintas en ccxt",
     "MEDIO | Cambiar instrument_type en el conector de BTC/USDT a BTC-USDT-SWAP"),
]

for name, detail, impacto in issues:
    print(f"  {name}")
    print(f"    Detalle: {detail}")
    print(f"    Impacto: {impacto}")
    print()

print("=== CONCLUSION ===")
print()
print("El contrato minimo NO es el problema principal.")
print("El problema principal es el FUNDING RATE:")
print(f"  Retorno medio por trade WIN:  +0.057%")
print(f"  Funding en 72H (9 pagos):     -0.090%")
print(f"  -> Con las barreras TBM actuales, los futuros son RENTABLES SOLO si:")
print(f"     a) Los holdings duran < 24H (< 3 pagos = 0.03% funding)")
print(f"     b) O se amplian las barreras TBM (profit target > 0.20%)")
print(f"     c) O el mercado es muy direccional (WR > 65% sostenido)")
