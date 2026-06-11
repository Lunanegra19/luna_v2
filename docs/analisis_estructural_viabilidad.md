# Análisis Crítico de Viabilidad Económica y Estructural — Luna V2 (WFB)

**Fecha del documento:** 2026-06-06  
**Autor:** Antigravity (AI Pair Programmer)  
**Contexto:** Evaluación del pipeline multi-semilla Walk-Forward (W1–W5) sobre el periodo holdout OOS 2025.

---

## 1. Veredicto Cuantitativo Global (Estándar V2)

Para evaluar la viabilidad de la arquitectura actual, primero debemos consolidar los datos de rendimiento obtenidos en el holdout 2025 (W1-W5).

### 1.1 Rendimiento del Ensemble (20 Semillas)
* **Sharpe Ratio Anualizado (Ensemble):** `-0.306`
* **DSR Ajustado (Bailey & López de Prado):** `0.0000` (Umbral requerido: $\ge 0.75$)
* **PBO Estimado (Probabilidad de Overfitting):** `50.0%`
* **Número Medio de Trades (por Semilla/Año):** `8.0` (Inanición operativa severa)
* **Inconsistencia Temporal del Win Rate (Std Dev Cross-Window):** `25.8%`

### 1.2 Rendimiento Detallado (Semilla 100 — Caso Testigo)
* **Retorno Nominal Neto:** `-0.0512%`
* **Retorno Compuesto Neto:** `-0.0512%`
* **Máximo Drawdown (MaxDD %):** `0.16%`
* **Ratio de Sharpe Anualizado:** `-0.3469`
* **Calmar Ratio (Retorno / MaxDD):** `-218.78`
* **Total de Trades OOS:** `8`
* **Win Rate OOS:** `62.5%`
* **p-value binomial:** `0.363281` (No significativo estadísticamente, $p > 0.05$)
* **Asignación de Capital Kelly (Half-Kelly Estático):** `14.17%` (Penalizado al 50% por drift a un **3.5%** de Kelly efectivo, equivalente a ~$3,500 de exposición nominal sobre un capital base de $100K).

---

## 2. Los Tres Fallos Estructurales Críticos

El análisis de logs y los tests de hipótesis demuestran que el sistema actual no está perdiendo dinero por "mala suerte", sino por tres contradicciones de diseño matemático y arquitectónico:

### 🔴 Fallo #1: La Disonancia de Frecuencias Temporales (Macro vs. Inferencia Horaria)
La arquitectura actual mezcla flujos de información a escalas temporales incompatibles:

```
[Datos Macro/On-Chain] (M2 YoY, CPI, Liquidez Fed)  --> Cambios Mensuales/Semanales
                   │
                   ▼
       [Inferencia horaria (1H)]                    --> Genera señales continuas e idénticas (Clusters)
                   │
                   ▼
     [Embargo Dinámico (96H-168H)]                  --> Destruye el 98% de las señales para evitar autocorrelación
                   │
                   ▼
  [Inanición Operativa: 8-10 Trades/Año]            --> Ruido estadístico puro, inviable para pagar costes fijos
```

* **El problema de los clusters:** El XGBoost ve condiciones macro estables durante semanas, por lo que genera la misma señal hora tras hora. Al aplicar un embargo dinámico estricto de 96H a 168H (basado en ATR), el sistema tritura el 98% de las señales, reduciendo la muestra a niveles donde es estadísticamente imposible demostrar un *edge*.

### 🔴 Fallo #2: La Ceguera del HMM ante Cambios de Régimen (PSI = 4.51)
* El modelo HMM se entrena de forma estática en la fase histórica (IS). Al enfrentarse al holdout 2025 (un mercado alcista donde BTC subió un ~40%), el HMM clasifica el mercado de forma radicalmente distinta.
* **El "Gate" G2 de exclusión (`bull_gate_min_dsr = 0.20`) desactivó sistemáticamente al agente BULL** porque su DSR de validación era de 0.174. Como consecuencia, el sistema estuvo en **CASH el 84% del tiempo de mercado** durante el rally de 2025, perdiéndose el principal movimiento alcista histórico de la semilla.

### 🔴 Fallo #3: La Matemática del Payoff Inverso y Costos de Transacción
Con la actual configuración de Triple Barrera (TBM), el sistema tiene un comportamiento de retorno asimétrico negativo:

* **Retorno medio de ganadores (PT/VB wins):** `+0.72%`
* **Retorno medio de perdedores (VB losses):** `-1.59%`
* **Payoff Ratio real:** `0.45x` (Se gana menos de la mitad de lo que se pierde en promedio).

> [!WARNING]
> La pérdida media de `-1.59%` ocurre porque el Stop Loss dinámico de 1.0x ATR ($\approx 0.357\%$) nunca se toca ya que el precio deriva lentamente en contra sin volatilidad durante las 96H del horizonte vertical (VB). El trade se cierra al finalizar las 96H con una pérdida acumulada mucho mayor.
> Los datos de holding time demuestran que **cualquier trade que dure más de 48 horas en el mercado tiene un Win Rate del 0.0% y una pérdida media del -2.87%**.

* **La barrera de comisiones:** Con un costo por operación de **0.175%** (comisión + deslizamiento round-trip) y un payoff de 0.45x, el Win Rate mínimo para salir en breakeven es del **69%**. Mantener este Win Rate consistentemente en datos OOS no correlacionados es estadísticamente imposible para este tipo de modelos.

---

## 3. Desglose del Rendimiento por Régimen de Mercado (OOS)

El test de hipótesis demuestra que el modelo carece de consistencia transversal y que su rendimiento depende enteramente de si la ventana coincide con un régimen de rango volátil (su único sesgo positivo):

| Régimen HMM | Trades Ejecutados | Win Rate (%) | Retorno Medio (%) | p-value Binomial | ¿Edge Estadístico Real? |
|-------------|-------------------|--------------|-------------------|------------------|--------------------------|
| `3_BEAR_CRASH` | 111 | 34.2% | -0.61% | 0.9996 | ❌ NO (Pierde dinero sistemáticamente) |
| `2_CALM_RANGE` | 59 | 44.1% | -0.33% | 0.9429 | ❌ NO (Rendimiento peor que el azar) |
| `2_VOLATILE_RANGE` | 139 | 67.6% | +0.72% (bruto) | 0.0005 | ✅ SÍ (Único régimen con edge real) |
| `BULL_TREND` | 0 | — | — | — | ❌ N/A (Bloqueado por el Gate) |

La inconsistencia temporal (desviación estándar del WR entre ventanas del **25.8%**) confirma que el Sharpe positivo temporal en ciertas ventanas (como W3) es un artefacto de coincidencia de régimen, no un comportamiento robusto del modelo.

---

## 4. Dos Caminos de Reestructuración Arquitectónica

Para que el modelo sea económicamente viable en producción real, el diseño debe simplificarse eliminando la sobredimensión de filtros. Se proponen dos alternativas de diseño:

### 🗺️ Camino A: Alinear el Pipeline a Baja Frecuencia (Diario/Semanal)
*Este enfoque es el idóneo si se desea seguir explotando variables macroeconómicas y fundamentales (M2, CPI, liquidez global).*
1. **Inferencia Lenta:** Cambiar el intervalo de decisión de 1H a **1D (Diario)** o **1W (Semanal)**.
2. **Reducción de Complejidad:** Eliminar el MetaLabeler LSTM de alta frecuencia, Platt Scaling, y los embargos horarios.
3. **Rebalanceo de Portfolio:** Configurar un modelo de asignación de activos en base a las señales macro en lugar de trading direccional de alta frecuencia.
4. **Matemática Sólida:** Los costos de transacción (0.175%) dejan de ser un factor destructivo al buscar movimientos direccionales de mayor aliento (retornos del 3% al 10% por operación).

### ⚡ Camino B: Alinear el Pipeline a Alta Frecuencia (15m/1h)
*Este enfoque es el idóneo si se desea mantener la infraestructura de trading algorítmico activo en VPS con el motor de ejecución actual.*
1. **Saneamiento del Selector SFI:** Prohibir la entrada de variables macro/on-chain lentas en la Fase 3B.
2. **Inyección de Características de Microestructura:** Utilizar variables de ordenes y volumen intradía: *CVD (Cumulative Volume Delta) spot vs. perpetuos, spreads de order book, volatilidad implícita horaria (DVOL), perfiles de volumen por sesión, y momentum de marcos temporales cortos*.
3. **Optimización de Triple Barrera (TBM):**
   * Reducir el horizonte vertical de la barrera (VB) de 96H a **24H - 48H máximo** (para cortar inmediatamente los trades zombies que derivan a pérdidas).
   * Aumentar el umbral mínimo de retorno (`tbm_min_return = 0.005`) para asegurar que el payoff ratio neto supere cómodamente la fricción de comisiones.
4. **HMM Rolling Dinámico:** Re-entrenar y mapear los estados del HMM en cada ventana de Walk-Forward, en lugar de arrastrar un modelo estático e inconsistente.
