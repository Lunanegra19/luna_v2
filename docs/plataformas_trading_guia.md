# Guía de Plataformas de Trading para Luna V2

> **Contexto:** Cuenta retail no profesional desde España. Requisitos: API Python, operativas Spot Only Long, comisiones estrictamente mapeadas al SOP V11.0.
> **Última actualización:** Junio 2026

---

## Contexto Regulatorio: MiCA 2025 y Evolución a Spot

La regulación **MiCA (Markets in Crypto-Assets)** de la UE, en plena vigencia desde 2025, ha limitado la disponibilidad de productos de derivados apalancados para usuarios retail en España (Binance Futures y Bybit EU Futures están bloqueados).

**Decisión Arquitectónica (SOP V11.0):**
Ante este escenario regulatorio, la operativa de Luna V2 se ha consolidado en formato **Spot (Only Long)**. El motor cuantitativo se ha adaptado para prescindir de posiciones SHORT, maximizando la eficiencia predictiva en los rallies e inyectando embargos estrictos y el "Guardián OOD" para proteger el capital durante los regímenes bajistas (`3_BEAR_CRASH` y `4_BEAR_FORCED`).

---

## Plataforma Oficial — OKX Spot (RECOMENDADA)

### Por qué OKX Spot
- Plena disponibilidad legal en España y compatibilidad con MiCA.
- API robusta y liquidez profunda en los pares Spot (BTC/USDT).
- Permite aplicar el modelo estricto de costos de **0.25% Round-Trip** dictado por la Regla R6 del SOP V11.0.

### Comisiones Oficiales (Luna V2 SOP)
La directiva institucional impone evaluar los backtests bajo el peor escenario realista para garantizar robustez:

| Concepto | Costo Estimado |
|---|---|
| Comisión de entrada (Taker Spot) | ~0.10% |
| Comisión de salida (Taker Spot) | ~0.10% |
| Slippage máximo tolerado | ~0.05% |
| **Costo Total Round-Trip (R6)** | **0.25%** |

> **IMPORTANTE:** El pipeline orquestador (WFB) y los tests de pre-vuelo están configurados con un límite duro (`round_trip_pct: 0.25`). Cualquier run ejecutada por debajo de este umbral será invalidada por los scripts de integridad.

---

## Integración API Python (Spot Only Long)

```bash
pip install python-okx
```

```python
from okx.Trade import TradeAPI

# Autenticación
trade_api = TradeAPI(
    api_key="TU_API_KEY",
    api_secret_key="TU_SECRET",
    passphrase="TU_PASSPHRASE",
    flag="0"  # 0=produccion, 1=demo/testnet
)

# Orden de compra en Spot (LONG)
trade_api.place_order(
    instId="BTC-USDT",
    tdMode="cash",       # Spot trading
    side="buy",
    ordType="market",    # Ejecución Taker
    sz="0.01"
)
```

---

## Configuración Requerida en settings.yaml

```yaml
# En config/settings.yaml — Sección de ejecución
execution:
  platform: okx
  instrument: BTC-USDT          # Spot market
  order_type: market            # Taker (asume el peor caso)
  slippage_tolerance_pct: 0.05  # Máximo slippage aceptable
  
cost_model:
  round_trip_pct: 0.25          # SOP V11.0 Iron Rule R6 (Spot Only Long)
  funding_per_trade_pct: 0.00   # En Spot puro no existe Funding Drag
```

---

## Tabla de Decisión Rápida

```
¿Estás en España/UE?
    |
    +--> SI
          |
          +--> Operativa Luna V2 (Spot Only Long)
                    |
                    +--> Plataforma: OKX
                    +--> Instrumento: BTC-USDT (Spot)
                    +--> Costo Round-Trip: 0.25% (SOP R6)
```
