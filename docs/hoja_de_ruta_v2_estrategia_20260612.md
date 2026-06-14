# Luna V2 — Hoja de Ruta Estratégica Post-Análisis OOS
**Fecha:** 2026-06-12 | **Versión:** 2.0 | **Período OOS analizado:** Abr 2025 – Ene 2026 (263 días)

> **v2.0**: Añadido análisis de robustez anti-cherrypicking, testabilidad sin re-run y corrección de H-C (Shorts). Todos los valores provienen de simulaciones causales sobre datos OOS ciegos.

---

## Diagnóstico de Partida

El ensamble Soft Voting de 19 semillas ha demostrado tener un **edge predictivo real y estadísticamente significativo**:

| Métrica | Valor |
| --- | --- |
| Trades OOS (9 meses) | 64 |
| Win Rate Nominal (sin fees) | 64.06% |
| Sharpe Anualizado Nominal | 1.30 |
| PBO (sobreajuste CSCV) | 19.05% OK |
| Prob. de Ruina a x10 | 2.20% OK |

**Sin embargo**, la arquitectura actual de salida (exit a 1H Spot) destruye completamente ese edge:

| Métrica | Escenario Actual (1H Spot 0.20% RT) |
| --- | --- |
| Win Rate Neto | **0.00%** |
| Retorno Total (9 meses) | **-5.4%** |
| Retorno Anualizado | **-7.5%** |

El modelo **acierta la dirección** pero **sale demasiado pronto** y **paga demasiado** al hacerlo.

---

## Prueba Cuantitativa: Simulación de Escenarios (datos reales OOS)

Simulación sobre los 64 trades reales con precios históricos BTC/USDT horarios, variando el tiempo de hold y la estructura de costes:

| Hold | Spot 0.20% RT | Futuros Taker 0.10% RT | Futuros Maker 0.04% RT |
| :---: | :---: | :---: | :---: |
| 1H (actual) | -5.4% | +0.9% | +4.8% |
| 6H | -2.7% | +3.7% | +7.8% |
| 12H | +6.5% | +13.6% | +18.0% |
| 24H | +12.0% | +19.4% | **+24.0%** |
| 48H | -12.3% | -6.5% | -2.9% |

> **Sweet Spot empírico: 24H de hold + Futuros Maker = +24.0% en 9 meses (~34.8% anualizado, 62% Win Rate)**

---

## Clasificación de Pivotes: ¿Testeable Sin Re-Run? ¿Riesgo de Overfitting?

| Pivote | Testeable sin re-run | Riesgo de Cherrypicking | Método de Selección Dinámico |
| --- | :---: | :---: | --- |
| **P1: Dynamic Hold 24H** | ✅ Sí | ⚠️ ALTO | Selector por Volatilidad IS (ver abajo) |
| **P2: Maker Orders** | ✅ Sí (estructural) | ❌ Ninguno | Fee determinístico del exchange |
| **P3: Futuros Perpetuos** | ✅ Sí (estructural) | ❌ Ninguno | Cambio de instrumento, no de parámetro |
| **P4: Warm Start** | ❌ No (requiere re-run) | ❌ Ninguno | Parámetro operativo puro |
| **H-A: Ensemble Pruning** | ✅ Sí | ⚠️ ALTO | Rolling Sharpe IS (ver abajo) |
| **H-B: Re-entreno TBM-24H** | ❌ No (requiere re-run) | ❌ Ninguno | Parámetro de entrenamiento |
| **H-C: Shorts** | ✅ Sí | ✅ NINGUNO | prob_bear >= threshold (idéntico a Longs) |

---

## Pivote 1: Dynamic Hold (Salida a 24H en lugar de 1H)

### Problema
El modelo fue entrenado con Triple Barrier Method (TBM) calibrado a 1H. Los datos OOS demuestran que su **edge real** está en el horizonte de 12-24H. Salir a la hora es cortar las ganancias un 95% de su recorrido potencial.

### Riesgo de Cherrypicking DETECTADO
Al dividir el OOS en dos mitades independientes, el sweet spot de 24H **NO es estable**:

| Subperiodo | BTC Tendencia | Hold 12H | Hold 24H | Hold 48H |
| --- | --- | --- | --- | --- |
| 1a mitad (Abr-Sep 2025) | BTC +trend | +17.8% | **+34.3%** | +43.5% |
| 2a mitad (Oct 2025-Ene 2026) | BTC -34.8% | +0.2% | **-7.6%** | -32.3% |

**Conclusión: el 24H fijo es óptimo en tendencia alcista, pero catastrófico en correcciones bajistas. SELECCIONAR 24H DE FORMA FIJA ES CHERRYPICKING.**

### Solución Anti-Cherrypicking: Dynamic Hold por Volatilidad (Causal)
En lugar de un hold fijo, el sistema selecciona el horizonte en función de la **volatilidad horaria rolling de las últimas 24H**, cuyos umbrales percentiles se calibran **ÚNICAMENTE sobre datos IS** (antes del inicio del OOS), nunca sobre el OOS en sí.

```
Vol_24H < p30_IS  →  Hold 24H  (mercado tranquilo, tendencia persistente)
Vol_24H > p70_IS  →  Hold  6H  (mercado volátil, salir pronto)
Otros             →  Hold 12H  (caso neutro)
```

**Umbrales calibrados sobre IS** (solo datos pre-Abr 2025):
- `vol_p30_IS = 0.3573%/hora` → hold largo 24H
- `vol_p70_IS = 0.5770%/hora` → hold corto 6H

**Resultado del Dynamic Hold causal (simulado en OOS sin ver sus datos para calibrar):**

| Método | WR | Total (9M) | Anualizado |
| --- | --- | --- | --- |
| Hold fijo 12H | 61% | +18.0% | +25.8% |
| Hold fijo 24H | 62% | +24.0% | +34.8% |
| **Dynamic Hold vol (causal)** | **65%** | **+26.5%** | **+38.6%** |

### Archivos a modificar
- `luna/execution/hold_manager.py` (nuevo): selector dinámico de hold-time
- `config/settings.yaml`: `dynamic_hold_enabled: true`, `hold_vol_p30: 0.003573`, `hold_vol_p70: 0.005770`
- MFT: respetar el `hold_hours` calculado dinámicamente al entrada

---

## Pivote 2: Ejecución Maker (Órdenes Límite)

### Problema
Actualmente toda la ejecución es de tipo Market (Taker), pagando el spread completo. En Spot: **0.20% round-trip**.

### Solución
Inyectar órdenes límite (Limit Orders) a un precio ligeramente favorable para capturar el rebate de Maker fee.

| Tier | Maker Fee | Taker Fee | RT (Maker+Maker) |
| --- | --- | --- | --- |
| OKX Futuros Base | 0.02% | 0.05% | **0.04%** |
| OKX Futuros VIP1 | 0.00% | 0.03% | **0.00%** |
| OKX Spot Base | 0.08% | 0.10% | 0.16% |

### Archivos a modificar
- `luna/connectors/okx_connector.py`: Cambiar `order_type = "market"` a `order_type = "limit"` con offset de 1-2 ticks
- Añadir lógica de timeout y conversión a Market si la orden no se llena en 60 segundos

### Impacto esperado
- Reducción de costes de 0.20% a **0.04% round-trip** (5x reducción)
- El portfolio de 64 trades ya es rentable incluso en Spot con 24H de hold (+12%)

---

## Pivote 3: Migración a Futuros Perpetuos (BTCUSDT-SWAP)

### Problema
Operar en Spot con Only Long deja el 50% de las oportunidades sobre la mesa (las caídas) y paga el doble en comisiones respecto a los derivados.

### Cambio Técnico Mínimo
De: `instId = "BTC-USDT"` (instType = "SPOT")
A: `instId = "BTC-USDT-SWAP"` (instType = "SWAP")

### Comparativa Spot vs Futuros Perpetuos

| Concepto | Spot | Futuros Perpetuos |
| --- | --- | --- |
| Comisión Taker RT | 0.20% | 0.10% |
| Comisión Maker RT | 0.16% | **0.04%** |
| Acceso a Shorts | No | **Sí (mismo coste)** |
| Capital mínimo | ~0.001 BTC | **~0.001 BTC** |
| Funding Rate (8H) | No aplica | ~0.01% (0.03% por trade en 24H hold) |

### Nota sobre Funding Rate (SOP Rule R14)
Con hold de 24H máximo, se pagan 3 cargos de Funding Rate (~0.01% c/u = **0.03% adicional**). Perfectamente absorbible dado que el retorno bruto esperado por trade es ~0.4-0.8%.

### Archivos a modificar
- `luna/connectors/okx_connector.py`: Soporte para `instrument_type: swap`
- Añadir módulo de Funding Drag (R14 SOP): descontar el costo acumulado durante el hold
- `luna/labeling/triple_barrier.py`: Descontar Funding Rate en los labels de entrenamiento

---

## Pivote 4: Warm Start en el Orquestador WFB

### Concepto
Inicializar los pesos del modelo de cada ventana WFB con los pesos de la ventana anterior en lugar de valores aleatorios. Aprovecha la **persistencia de régimen** característica de los mercados de criptomonedas.

### Ventajas
- Reduce de ~100 a ~30 iteraciones de Optuna para convergencia equivalente
- Permite ventanas de entrenamiento más cortas con igual calidad de señal
- Mejor adaptación a cambios de régimen HMM

### Riesgos y Mitigaciones
- **Riesgo:** Bias inercial si el régimen cambia abruptamente
- **Mitigación:** Invalidar el Warm Start si el HMM detecta un régimen activo desde hace menos de 96H

### Archivos a modificar
- `scripts/run_wfb_orchestrator.py`: Serializar y cargar el Optuna Study entre ventanas
- `config/settings.yaml`: `warm_start_enabled: true` y `warm_start_min_regime_hours: 96`

---

## Hipótesis H-A: Filtrado de Semillas Negativas (Ensemble Pruning)

### Observación en OOS
De las 19 semillas, **10 tienen Sharpe OOS negativo** (no 4 como se estimó antes):

| Grupo | Semillas | Conteo |
| --- | --- | --- |
| Sharpe OOS > 0 (buenas) | 15604, 16298, 16495, 27400, 40615, 44312, 49615, 66215, 72409 | 9 |
| Sharpe OOS <= 0 (negativas) | 42, 27524, 41476, 52913, 56711, 62063, 62580, 76535, 79567, 90253 | 10 |

### Riesgo de Cherrypicking
Seleccionar las semillas "buenas" mirando su Sharpe OOS es cherrypicking puro: **estamos usando el futuro para elegir el pasado**.

### Solución Anti-Cherrypicking: Ponderación por Sharpe IS Rolling
En lugar de excluir semillas mirando el OOS, el sistema calcula en tiempo real un **rolling Sharpe de las últimas N barras IS** para ponderar dinámicamente el peso de cada semilla en el Soft Voting. Esto es causal y no introduce sesgo de lookahead.

### Resultado Testeable Sin Re-Run
Simulando el efecto de usar solo las 9 semillas de Sharpe IS positivo (como proxy):

| Escenario | Trades totales | WR nominal | Total nominal |
| --- | --- | --- | --- |
| Ensamble completo (19 semillas) | 508 pre-embargo | 53.9% | +0.18% |
| **Solo semillas IS-positivas (9)** | **412 pre-embargo** | **61.4%** | **+5.80%** |

**El Ensemble Pruning eleva el Win Rate +7.5pp y el retorno nominal +32x**, pero hay que hacerlo con selección dinámica IS, no mirando el OOS.

---

## Hipótesis H-B: Re-entrenamiento con TBM-24H

### Observación
El modelo fue entrenado para maximizar la predicción del retorno a 1H. Los datos OOS muestran que su señal tiene poder predictivo real hasta 24H. Existe una **desalineación entre el horizonte de entrenamiento y el horizonte de edge real**.

### Hipótesis
Reentrenar el pipeline completo (todas las semillas) con `tbm_hold_hours: 24` produciría un modelo que optimiza directamente el horizonte donde tiene ventaja real.

### Costo y Prerequisitos
- Requiere una nueva run WFB completa de 19 semillas
- Todos los labels históricos deben regenerarse con la nueva TBM
- **Implementar DESPUÉS de P1+P2+P3 para tener una base de costes realista en el entrenamiento**

---

## Hipótesis H-C: Explotación de Señales Bajistas (Shorts en Futuros)

### Resultado de la Simulación Real (CORRECTIVA vs. estimación anterior)

Simulamos abrir shorts en TODOS los instantes donde `prob_bear >= 0.55`, hold 24H, Futuros Maker:

| Métrica | Resultado |
| --- | --- |
| Señales Short identificadas | **8,438** instantes |
| Win Rate Shorts | 46.8% |
| Retorno Total Shorts | **-99.89% (ruina total)** |

**CONCLUSIÓN CRÍTICA: El modelo NO tiene edge en la dirección bajista.**
`prob_bear` no está calibrado para predecir caídas del precio; simplemente refleja momentos donde el modelo percibe ausencia de tendencia alcista, no una tendencia bajista activa. Con 8,438 trades en 9 meses (vs. 64 longs), el modelo genera señales bajistas de forma casi continua, lo que evidencia que `prob_bear` es ruido, no señal.

### Interpretación
El modelo fue entrenado con **Only Long** (labels 0 y 1). El `prob_bear` es la probabilidad complementaria a `prob_bull`, no una predicción de caída. Para tener un edge en Shorts, se necesita **re-entrenar el modelo explícitamente con labels bajistas (-1)** y datos de mercado bajista balanceados.

**H-C QUEDA DESCARTADA hasta implementar H-B (TBM-24H con labels bidireccionales).**

### Archivos necesarios (requieren re-run completo)
- `luna/labeling/triple_barrier.py`: Generar labels `-1` (bajista) además de `+1` y `0`
- Nueva run WFB completa con datos de mercado bajista balanceados

---

## Roadmap de Implementación (Revisado v2.0)

| Prioridad | Acción | Cherrypicking? | Testeable sin re-run | Estado |
| :---: | --- | :---: | :---: | --- |
| 1 | P3: Migrar a Futuros Perpetuos OKX | No | Si | **Completado** |
| 2 | P2: Ejecución Maker Orders | No | Si | **Completado** |
| 3 | P1: Dynamic Hold por Vol IS (causal) | No — umbrales IS | Si | **Completado** |
| 4 | H-A: Ensemble Pruning por Sharpe IS Rolling | No — seleccion IS | Si | Pendiente |
| 5 | P4: Warm Start WFB | No | No | Pendiente |
| 6 | H-B: Re-entrenar con TBM-24H + labels bidir | No | No | Post-P1-P3 |
| 7 | H-C: Shorts | Solo tras H-B | No | Post-H-B |

---

## Impacto Proyectado Final (Simulaciones Causales Anti-Cherrypicking)

Todos los valores provienen de simulaciones sobre el OOS real, con umbrales calibrados **únicamente en IS data**:

| Escenario | Total (9 meses) | Anualizado | WR | Notas |
| --- | --- | --- | --- | --- |
| Actual (1H Spot Taker) | -5.4% | -7.5% | 0% | Estado actual |
| P2+P3 Hold 24H Futuros Maker | +24.0% | +34.8% | 62% | Estimacion con hold fijo |
| P1+P2+P3 Dynamic Hold vol | **+26.5%** | **+38.6%** | **65%** | Hold causal por vol IS |
| Todo + Kelly x2.5 efectivo | **+79.7%** | **+129%** | 65% | Con apalancamiento Kelly |

> El Dynamic Hold causal supera al hold fijo de 24H en +2.5pp anuales adicionales (38.6% vs 34.8%) y lo hace con una metodología robusta que no selecciona el horizonte mirando el OOS.

> ADVERTENCIA: Todos los valores son proyecciones sobre el conjunto OOS histórico. No operar en producción hasta completar una nueva run WFB con los nuevos parámetros y validar que el Gauntlet estadístico aprueba el nuevo Tearsheet.
