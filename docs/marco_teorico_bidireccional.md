# 📊 Marco Teórico y Diseño de Arquitectura Bidireccional (Long/Short) para Luna V2

Este documento establece los fundamentos matemáticos, estadísticos y operativos para la integración de sistemas bidireccionales en el pipeline de trading cuantitativo de **Luna V2**, bajo las estrictas directrices de la política institucional **SOP V11.0**.

---

## 1. Fundamentos Matemáticos y Asimetría del Mercado

El diseño de un sistema bidireccional exige modelar las diferencias intrínsecas de riesgo y distribución entre tomar posiciones largas (compras) y cortas (ventas) en criptoactivos.

### A. Ley de los Grandes Números (LLN) e Inferencia Estadística
La **Ley de los Grandes Números** establece que el promedio muestral de una variable aleatoria converge al valor esperado a medida que el tamaño de la muestra ($N$) tiende al infinito:

$$\bar{X}_N \xrightarrow{a.s.} \mu \quad \text{cuando } N \to \infty$$

En el contexto de trading de alta precisión (como el *Sniper-Mode*):
* **SOP R8 (Significancia)** exige un mínimo de **100 trades** out-of-sample (OOS) para validar que el Sharpe Ratio o el Win Rate del sistema no son fruto del ruido estadístico (p-value < 0.05).
* Si restringimos el sistema únicamente a posiciones largas (Long-Only), alcanzar $N \ge 100$ trades válidos en periodos bajistas prolongados (como 2022 o correcciones secas de 2025) puede demorar meses, retrasando la validación del modelo.
* La inclusión de la rama Short permite **duplicar el espacio muestral disponible**, acelerando la convergencia estadística del Sharpe Ratio observado hacia su Sharpe real esperado, reduciendo el error estándar de estimación:

$$\sigma_{SR} \approx \sqrt{\frac{1 + \frac{SR^2}{2}}{N}}$$

### B. Asimetría Alcista y Perfil de Retorno (Long vs. Short)
Matemáticamente, los perfiles de pago de ambas direcciones son asimétricos:

| Característica | Posición Larga (Long) | Posición Corta (Short) |
| :--- | :--- | :--- |
| **Pérdida Máxima Teórica** | $100\%$ del capital invertido (el precio cae a $0$). | **Ilimitada** (el precio de Bitcoin puede subir indefinidamente). |
| **Retorno Máximo Teórico** | Ilimitado. | Limitado al $100\%$ (si el precio cae a $0$). |
| **Distribución de Retornos** | Sesgo positivo (Right-tailed / Fat-tails alcistas). | Sesgo negativo (Left-tailed / Riesgo de Squeezes). |

Debido a los eventos de liquidación en cadena y rallies de cobertura de cortos (*Short Squeezes*), la volatilidad alcista en cortos puede ser extremadamente violenta. Por lo tanto, el sistema Short exige:
1. Multiplicadores de Stop Loss más ajustados basados en el Rango Medio Verdadero (ATR).
2. Un umbral de confianza predictiva más estricto para autorizar la entrada.

---

## 2. Arquitectura Multi-Agente HMM: Macro vs. Micro

Para evitar la redundancia computacional y maximizar la coherencia estadística, la arquitectura de Luna V2 divide sus componentes en dos niveles:

```
                  ┌────────────────────────────────────────┐
                  │          DATOS DE ENTRADA (OHLCV)      │
                  └───────────────────┬────────────────────┘
                                      │
                         [MACRO]      ▼
                  ┌────────────────────────────────────────┐
                  │       HMM Regime Classifier (Único)    │
                  └───────────────────┬────────────────────┘
                                      │
                                      ├────────────────────────┐
                                      ▼ [BULL / RANGE]         ▼ [BEAR]
                         [MICRO]      ▼                        ▼
                  ┌────────────────────────┐      ┌────────────────────────┐
                  │   XGBoost LONG Agent   │      │   XGBoost SHORT Agent  │
                  └───────────┬────────────┘      └───────────┬────────────┘
                              │                               │
                              ▼                               ▼
                  ┌────────────────────────┐      ┌────────────────────────┐
                  │  Post-Filters (LONG)   │      │  Post-Filters (SHORT)  │
                  │ (Meta, Calibrator, CVD)│      │ (Meta, Calibrator, CVD)│
                  └───────────┬────────────┘      └───────────┬────────────┘
                              │                               │
                              └───────────────┬───────────────┘
                                              │ [PORTFOLIO]
                                              ▼
                  ┌────────────────────────────────────────┐
                  │     Portfolio Kelly & Netting Sizer    │
                  └────────────────────────────────────────┘
```

### A. Modelo de Régimen HMM (Macro - Unificado)
El modelo de Mezclas de Gaussianas y Cadenas Ocultas de Markov (GMM-HMM) clasifica el estado de mercado en base a la volatilidad local y los retornos del precio. 
* El régimen del mercado es un **hecho macroeconómico unificado** (Bitcoin está en una fase alcista, bajista o lateral). No tiene sentido tener un HMM para Long y otro para Short.
* Ambos agentes de ejecución leen la etiqueta de régimen semántico resultante (`1_BULL_TREND`, `3_BEAR_CRASH`, etc.) desde el mismo modelo HMM compartido.

### B. Agentes XGBoost Base (Micro - Especializados)
En lugar de tener un único XGBoost monolítico que intente predecir ambas direcciones, se utilizan modelos separados y entrenados de forma independiente por dirección y régimen:
* **Especialización**: Las variables predictivas (como flujos on-chain y tasas de financiación) actúan con sentidos contrarios al predecir techos o suelos. Un modelo especializado aprende con mayor precisión el subconjunto específico de dinámicas para su dirección.
* **TBM Asimétrico**: El etiquetado por Triple Barrera (TBM) genera una clase $Y=1$ (éxito) y $Y=0$ (fracaso o salida por tiempo) con parámetros diferenciados:
  * Para Longs: $Y=1$ si toca la barrera superior ($Price_t \times (1 + \text{pt\_mult} \times \text{ATR})$) antes de tocar el Stop Loss o la barrera de tiempo.
  * Para Shorts: $Y=1$ si toca la barrera inferior ($Price_t \times (1 - \text{pt\_mult} \times \text{ATR})$).

---

## 3. Embudo de Filtrado Post-XGBoost Especializado

Una predicción cruda de XGBoost de $P(\text{Win}) \ge 0.50$ no es suficiente para arriesgar capital real. Debe pasar por un embudo de validación post-modelo específico por dirección.

### A. MetaLabeler V2 (LONG vs. SHORT)
El MetaLabeler V2 actúa como un árbitro de riesgo que predice si la señal generada por el XGBoost base tiene una alta probabilidad de éxito en las condiciones de mercado actuales.
* **Separación de Archivos**: Se entrenan de forma independiente `metalabeler_v2_long_lstm.pt` y `metalabeler_v2_short_lstm.pt`.
* **Entrada de Contexto**: El MetaLabeler de Cortos se nutre principalmente de variables de pánico (incremento de volumen en futuros, aumento de DVOL, divergencia CVD negativa, picos de tarifas de mempool), mientras que el de Longs prioriza variables de acumulación y liquidez macro.

### B. Calibración de Probabilidades
Las probabilidades devueltas por XGBoost no son directamente frecuencias de acierto empíricas debido al sobreajuste IS. La calibración (isotónica o sigmoide de Platt) transforma la salida en probabilidades reales:

$$P(\text{Acierto} \mid X) = \text{Calibrador}(P_{\text{xgb}})$$

* Se calibran de forma asimétrica, ya que el modelo de shorts suele presentar mayor sesgo hacia el sesgo de supervivencia en los splits temporales.

### C. CVD Veto Direccional
El Cumulative Volume Delta (CVD) spot vs perpetuos detecta anomalías de presión en el orderbook:
* **Filtro LONG**: Veta si el Spot está vendiendo agresivamente mientras el Perpetuo empuja el precio al alza de forma artificial (CVD < percentil 10). Evita comprar subidas sin soporte de dinero real.
* **Filtro SHORT**: Veta si los creadores de mercado en Spot acumulan agresivamente mientras el Perpetuo está sobrevendido (CVD > percentil 90). Evita vender en corto en zonas de absorción institucional.

---

## 4. Gestión de Riesgo e Integración en Portfolio (Kelly de Cartera)

La unificación definitiva ocurre en la capa de dimensionamiento del capital, donde se consolidan ambas colas de señales.

### A. Neteo de Señales Horario (Portfolio Netting)
Para optimizar las comisiones y evitar la fricción operativa en OKX/Kraken Futures, el Portfolio Manager procesa las señales en la misma hora antes de enviar órdenes:

$$\Delta Q_t = \text{Sign}(S_{Long, t} \cdot f^*_{Long, t} - S_{Short, t} \cdot f^*_{Short, t}) \cdot \text{Net\_Size}$$

Si el sistema genera simultáneamente una señal de compra de tamaño $0.8$ y una señal de venta de tamaño $0.3$, el sizer unificado simplemente ejecuta una orden de compra neta de tamaño $0.5$, ahorrando un $37.5\%$ en fees de transacción y reduciendo el deslizamiento de mercado.

### B. Kelly de Cartera y Prevención de Volatility Drag
El Criterio de Kelly maximiza la tasa de crecimiento logarítmico del capital, pero su versión pura (*Full Kelly*) es altamente vulnerable a la volatilidad, pudiendo destruir la cuenta en rachas de pérdidas consecutivas.

Bajo **SOP R17**, se aplica estrictamente el **Kelly Fraccional** ($\lambda = 0.25$):

$$f^{**} = \lambda \cdot f^* = \lambda \cdot \frac{p \cdot R - (1 - p)}{R}$$

Donde $p$ es la probabilidad calibrada por el modelo y $R$ es el ratio TakeProfit / StopLoss.

En un sistema bidireccional, para evitar el sobre-apalancamiento cuando ambas ramas se activan, la exposición total de la cartera se regula mediante la suma de la varianza conjunta:

$$\text{Leverage}_{\text{opt}} = \frac{E[R_p] - R_f}{\sigma^2_p}$$

Donde la covarianza entre las estrategias de Long y Short se descuenta de la varianza total de la cartera para optimizar el *sweet-spot* de apalancamiento (máximo x10 - x20 en futuros, prohibiendo el doble Kelly y los niveles de apalancamiento insostenibles debido al Volatility Drag y el funding drag).

### C. Embargo Temporal y Causalidad de Splits (SOP R2/R3)
El Embargo HMM evita el solapamiento temporal de operaciones correlacionadas (Overlap Drag). En un entorno bidireccional:
* El embargo se gestiona mediante dos colas independientes de bloqueo por dirección.
* Una señal de Long bloqueada por embargo no restringe una señal de Short (ya que son operaciones en sentidos opuestos y su correlación temporal de retorno suele ser negativa).

---

## 5. Mitigación del Overfitting mediante Walk-Forward Validation (WFB)

El sistema Walk-Forward Validation (WFB) de Luna V2 está diseñado específicamente para combatir el sesgo de selección y el curve-fitting:

* **Saneamiento Temporal (SOP R2)**: Se utiliza PurgedKFold en la validación cruzada IS para eliminar los datos adyacentes a las fronteras de los splits, evitando que la información leakée del conjunto de entrenamiento al de test.
* **Deflated Sharpe Ratio (SOP R5)**: Al evaluar múltiples semillas y combinaciones de parámetros en el orquestador, el Sharpe Ratio bruto observado tiende a inflarse de forma ficticia. El validador computa el DSR descontando el número de ensayos ($N$) y la varianza de los resultados de las semillas:

$$DSR = \Phi \left( \frac{(SR_{\text{obs}} - SR_0) \cdot \sqrt{T-1}}{\sqrt{1 - \gamma_1 \cdot SR_{\text{obs}} + \frac{\gamma_2}{4} \cdot SR_{\text{obs}}^2}} \right)$$

Donde $\Phi$ es la CDF de la distribución normal estándar y $SR_0$ es el Sharpe esperado bajo la hipótesis nula corregida por selección múltiple. Esto garantiza que solo las configuraciones bidireccionales con un Sharpe significativamente superior al ruido sean aprobadas para producción.
