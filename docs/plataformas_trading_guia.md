# Guía de Plataformas de Trading para Luna V2

> **Contexto:** Cuenta retail no profesional desde España. Requisitos: API Python, soporte SHORT, comisiones compatibles con el modelo Luna V2.
> **Última actualización:** Mayo 2026

---

## Contexto Regulatorio: MiCA 2025

La regulación **MiCA (Markets in Crypto-Assets)** de la UE, en plena vigencia desde 2025, ha cambiado profundamente la disponibilidad de productos de derivados para usuarios retail en España.

**Consecuencias directas:**
- **Binance Futures** — BLOQUEADO para usuarios retail en España/UE
- **Bybit EU Futures** — BLOQUEADO. Solo spot disponible en la plataforma regulada bybit.eu
- Los futuros/perpetuales que no sean MiFID-II pasan a ser ilegales para retail

Alternativas reguladas que SÍ funcionan: **OKX (X-Perps)** y **Kraken (Margin)**.

---

## Comparativa General de Plataformas

| Plataforma | Futures/Shorts | API Python | Disponible España | Costo RT estimado | Valoracion |
|---|---|---|---|---|---|
| **OKX (X-Perps)** | SI (MiFID-II) | SI | SI | 0.14% | RECOMENDADA |
| **Kraken Pro (Margin)** | SI (10x) | SI | SI | 0.65-0.80% | Viable, cara |
| **Binance Futures** | BLOQUEADO | — | NO | 0.10-0.13% | No disponible |
| **Bybit EU** | BLOQUEADO | Solo Spot | NO futures | — | No disponible |
| **cTrader CFD (IC/Pepperstone)** | SI (CFDs) | SI | SI | 0.17-0.20% | Descartada (overnight) |
| **Kraken Spot Puro** | NO shorts | SI | SI | 0.65-0.80% | Descartada |

---

## Futuros vs Spot — Decision para Luna V2

Para la arquitectura Luna V2, los **Futuros (o Margin)** son obligatorios por las siguientes razones:

| Caracteristica | Spot | Futuros / Margin |
|---|---|---|
| Short Selling | NO | SI — esencial para operar en Bear |
| Eficiencia de capital | 100% del valor BTC | Solo margen (10-20%) |
| Cobertura en bajadas | Solo salirse del mercado | Ganar activamente en caidas |
| Funding nocturno | Sin coste | ~0.01% cada 8h (variable) |
| Compatible API Luna | SI | SI |

**Conclusion:** Sin capacidad de SHORT, Luna V2 pierde todo el alfa en regimenes `3_BEAR_CRASH` y `4_BEAR_FORCED`, que representan el ~30% del tiempo de mercado.

---

## Opcion 1 — OKX (RECOMENDADA)

### Por que OKX
- Unica plataforma que en 2025 combina: disponibilidad legal en Espana + costos compatibles con Luna + shorts reales + API robusta.
- Lanzaron **X-Perps** (Abril 2025): futuros con vencimiento a 5 anos que simulan perpetuales. Regulados bajo MiFID-II, disponibles para retail EU.

### Comisiones OKX
| Tipo de orden | Comision entrada | Comision salida | RT total |
|---|---|---|---|
| Taker (mercado) | 0.055% | 0.055% | **0.11%** |
| Maker (limite) | 0.020% | 0.020% | **0.04%** |
| + Slippage estimado | — | — | +0.02-0.03% |
| **Total real Taker** | — | — | **~0.14%** |
| **Total real Maker** | — | — | **~0.06%** |

> **IMPORTANTE:** El SOP de Luna V2 usa 0.25% como referencia para OKX Spot. Esto refleja el peor caso realista (doble Taker 0.10% + slippage 0.05%). Con OKX Taker (0.14%) en futuros estamos perfectamente alineados, pero en Spot aplicamos 0.25%.

### Integracion API Python
```bash
pip install python-okx
```

```python
from okx.Trade import TradeAPI
from okx.MarketData import MarketAPI

# Autenticacion
trade_api = TradeAPI(
    api_key="TU_API_KEY",
    api_secret_key="TU_SECRET",
    passphrase="TU_PASSPHRASE",
    flag="0"  # 0=produccion, 1=demo/testnet
)

# Orden de compra (LONG)
trade_api.place_order(
    instId="BTC-USDT-SWAP",
    tdMode="cross",      # cross-margin
    side="buy",
    ordType="limit",     # Maker = menor comision
    px="85000",
    sz="0.01"
)

# Orden de venta en corto (SHORT)
trade_api.place_order(
    instId="BTC-USDT-SWAP",
    tdMode="cross",
    side="sell",
    posSide="short",     # SHORT explícito
    ordType="market",
    sz="0.01"
)
```

### Pasos para activar la cuenta
1. Registro en okx.com (KYC obligatorio con DNI + prueba de residencia)
2. Habilitar "Trading de Derivados" en ajustes de cuenta
3. Completar test de idoneidad MiFID-II
4. Crear API Key en Ajustes > API > con permisos: Trade + Read
5. Probar primero en Demo/Testnet (flag="1")

---

## Opcion 2 — Kraken Pro (Margin)

### Comisiones Kraken (Margin Spot)
| Concepto | Coste |
|---|---|
| Comision apertura (Taker) | 0.40% |
| Comision cierre (Taker) | 0.40% |
| Fee apertura margin | 0.01-0.025% |
| Rollover cada 4 horas | ~0.01-0.02% |
| **Total estimado por trade 24h** | **~0.83-0.90%** |

### Impacto en Luna V2
Con 57 trades (semilla 1337) y un costo de 0.90% RT:
- Costo total: 57 x 0.90% = **51.3%**
- PnL bruto 1337: +14.66%
- **PnL neto: -36.64%** — INVIABLE

> Kraken solo seria viable si el PnL bruto por semilla supera el 50%. No compatible con la arquitectura actual.

---

## Impacto Real de Comisiones en Semilla 1337

| Plataforma | Costo RT | Total (57 trades) | PnL Neto |
|---|---|---|---|
| OKX Maker | 0.06% | 3.42% | **+11.24%** |
| OKX Taker | 0.14% | 7.98% | **+6.68%** |
| Luna SOP actual | 0.25% | 14.25% | **+0.41%** |
| cTrader CFD | 0.17% | 9.69% | +4.97% (antes overnight) |
| Kraken Margin | 0.90% | 51.3% | **-36.64%** |

---

## Estructura de Costos Completa por Plataforma

### Costos que NO son comisiones (pero cuentan)

| Concepto | Descripcion | Impacto en Luna V2 |
|---|---|---|
| **Spread (CFD/Spot)** | Diferencia bid/ask al ejecutar | 0.05-0.12% por trade |
| **Funding Rate (Futures)** | Pago entre longs y shorts cada 8h | ~0.01% cada 8h. Con trades de 24h medios: ~0.03% |
| **Slippage** | Diferencia entre precio esperado y ejecutado | 0.01-0.05% en BTC liquido |
| **Overnight Swap (CFD)** | Interes diario por mantener CFD abierto | 0.03-0.07% diario — DESTRUCTIVO |

> Para Luna V2 con horizonte medio de 24h: el Funding Rate en OKX es minimo (~0.01-0.03%). El Overnight Swap de CFD seria devastador (+0.05% diario) y es la razon principal para DESCARTAR cTrader para esta estrategia.

---

## Tabla de Decision Rapida

```
¿Estas en Espana/UE?
    |
    +--> SI
          |
          +--> ¿Necesitas SHORT?
                    |
                    +--> SI (RECOMENDADO para Luna V2)
                    |         |
                    |         +--> Bajo coste  --> OKX (X-Perps) PRIMERA OPCION
                    |         +--> Alta regulacion --> Kraken (Margin) CARA PERO LEGAL
                    |
                    +--> NO (solo LONG)
                              |
                              +--> Kraken Spot (mas caro) o Coinbase Advanced
```

---

## Configuracion Recomendada para Luna V2 en OKX

```yaml
# En config/settings.yaml — Seccion de ejecucion en produccion
execution:
  platform: okx
  instrument: BTC-USDT-SWAP      # Perpetual X-Perp en OKX EU
  order_type: limit               # Maker para minimizar comision (0.06% RT)
  slippage_tolerance_pct: 0.05   # Maximo slippage aceptable
  
cost_model:
  commission_rt_pct: 0.06        # Maker OKX (usar 0.14 si es Taker)
  funding_per_trade_pct: 0.03    # Estimado para trade de ~24h
  slippage_pct: 0.02             # Estimado conservador
  total_cost_rt_pct: 0.11        # Total conservador con Maker
```

---

## Referencias

- OKX X-Perps (MiFID-II EU): https://www.okx.com/es-es/trade-spot
- OKX API Python Docs: https://github.com/okxapi/python-okx
- Kraken API Docs: https://docs.kraken.com/rest/
- Regulacion MiCA (CNMV): https://www.cnmv.es/portal/home.aspx
- OKX Fee Schedule: https://www.okx.com/fees
