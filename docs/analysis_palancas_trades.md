# 📊 Análisis de Palancas y Filtros (Semilla 42 - Ventanas W1 a W9)

Este análisis investiga la influencia de cada filtro de la arquitectura **Luna V2 (conforme a SOP V11.0)** sobre el número de trades resultantes, el Sharpe Ratio y la eficiencia global medida por el **Calmar Ratio** (Retorno Compuesto / MaxDD).

La investigación se realiza mediante retro-simulación estática utilizando los datos out-of-sample reales generados en la run activa (`WFB_20260619_173435_seed42`). Esto garantiza que no se perturbe el orquestador en ejecución.

---

## 🔍 Diagnóstico del Embudo de Filtrado (W1-W9)

Al analizar las **73 señales candidatas** que generó originalmente el modelo base de XGBoost en las ventanas 1 a 9, el comportamiento del embudo de filtrado clásico es el siguiente:

* **Total de señales baseline detectadas:** 73
* **Filtro MetaLabeler V2 (Probabilidad < 0.45):** **Bloquea 0 señales (0.0%)**.
  * *Explicación matemática:* Todas las señales baseline tienen una probabilidad asignada por el MetaLabeler de entre `0.698` y `0.801` (media de `0.749`). Por lo tanto, relajar este filtro (por ejemplo, a `0.38`) no añade ningún trade nuevo.
* **Filtro de Veto HMM (Regímenes `BEAR_CRASH` o `3_BEAR`):** **Bloquea 14 señales (19.2%)**.
* **Filtro de Embargo HMM (72H a 168H):** **Bloquea 30 señales (50.8% de las supervivientes)**.
  * El embargo temporal es la palanca más restrictiva de la arquitectura.

---

## 📈 Retro-Simulación de Escenarios de Sensibilidad

Evaluamos el impacto de modificar o desactivar cada una de las palancas en el dataset consolidado OOS (W1-W9):

| Escenario | Descripción | Trades | Win Rate | Retorno Raw | Retorno Compuesto Kelly | Max DD | Sharpe Anualizado | Calmar Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Standard (Simulado)** | Meta 0.45 + Veto HMM + Embargo 1.0x | 29 | 72.41% | +22.18% | +25.77% | 6.46% | 21.78 | 3.99 |
| **B: Relaxed Meta** | Meta 0.38 + Veto HMM + Embargo 1.0x | 29 | 72.41% | +22.18% | +25.77% | 6.46% | 21.78 | 3.99 |
| **C: Disabled Veto** | Meta 0.45 + **No Veto HMM** + Embargo 1.0x | 33 | 81.82% | +41.42% | **+39.48%** | **4.76%** | **27.98** | **8.29** |
| **D: Reduced Embargo** | Meta 0.45 + Veto HMM + **Embargo 0.5x** | 37 | 59.46% | +20.55% | +28.07% | 6.46% | 14.63 | 4.35 |
| **E: Disabled Embargo** | Meta 0.45 + Veto HMM + **Sin Embargo (0x)** | 59 | 57.63% | +26.40% | +33.40% | 7.65% | 12.01 | 4.36 |
| **F: No Veto & No Embargo**| Meta 0.45 + **No Veto** + **Sin Embargo** | 73 | 56.16% | +25.47% | +35.73% | 12.41% | 10.08 | 2.88 |
| **G: Relaxed + Reduced** | Meta 0.38 + Veto HMM + **Embargo 0.5x** | 37 | 59.46% | +20.55% | +28.07% | 6.46% | 14.63 | 4.35 |
| **H: Relaxed + No Embargo**| Meta 0.38 + Veto HMM + **Sin Embargo** | 57 | 56.14% | +26.40% | +33.40% | 7.65% | 12.01 | 4.36 |
| **I: Baseline Raw** | XGBoost Puro (No Veto, No Meta, No Embargo) | 73 | 56.16% | +25.47% | +35.73% | 12.41% | 10.08 | 2.88 |

> [!NOTE]
> Las métricas del **Histórico Filtrado Real (W1-W9)** en la run activa muestran **59 trades**, un **Win Rate del 57.63%**, un retorno compuesto del **+7.07%** y un **MaxDD de solo 0.81%** (Calmar = 8.72). Esto se debe a que el pipeline dinámicamente activa la reducción de embargo a 0 horas en ventanas de baja densidad (por ejemplo, W7) y activa el "Sniper Anomaly" (Fast-Track) para rescatar señales de alta confianza que luego son correctamente dimensionadas (o silenciadas a 0.00% si la probabilidad es dudosa) por el Kelly Sizer.

---

## 🛠️ Análisis de Palancas Viables

### 1. Desactivación de Veto HMM (Escenario C)
* **Resultados:** Incrementa los trades de 29 a 33. Eleva el retorno compuesto simulado al **+39.48%** y el Calmar Ratio a **8.29** (con un WR extraordinario de 81.82%).
* **Análisis Cuantitativo:** XGBoost es capaz de identificar excelentes puntos de entrada de media-reversión en zonas clasificadas como crash (`BEAR_CRASH` o `3_BEAR`) en la semilla 42. Al vetarlas el HMM por seguridad macro, nos perdemos estos retornos altamente lucrativos.
* **Viabilidad de SOP V11.0:** Desactivar por completo el veto HMM no es viable bajo los lineamientos institucionales de riesgo del pipeline en VPS (un crash real de bitcoin sin veto podría barrer posiciones no cubiertas). Sin embargo, relajar el veto en regímenes de transición podría ser una alternativa de estudio.

### 2. Relajación de Embargo (Escenario D y E)
* **Resultados:** Desactivar o reducir a la mitad el embargo temporal eleva el número de trades significativamente (hasta 59), y sube el retorno compuesto nominal (+33.40%).
* **Análisis Cuantitativo:** El costo de esta palanca es un **colapso del Win Rate** (cae del 72.41% al 57.63%) y una pérdida notable del Sharpe Ratio (cae de 21.78 a 12.01). Al eliminar el embargo temporal, tomamos señales consecutivas altamente correlacionadas (overlap), lo que duplica el riesgo en el mismo movimiento de mercado (Volatility & Drawdown Drag).

### 3. Modificación del Conformal Censor
* **Resultados:** El censor conformal causó la anulación de 2 trades en total (uno en W2 y otro en W6).
* **Análisis Cuantitativo:** Eliminar el censor conformal no rescataría trades de manera significativa, y su ausencia en ventanas con alta inestabilidad temporal (Covariate Shift) expondría al capital a fallos severos por sobreajuste del XGBoost.

---

## 📌 Conclusiones para la Toma de Decisiones

1. **La palanca del MetaLabeler está inactiva:** Sus predicciones son demasiado seguras y estables en esta semilla actual.
2. **El Embargo HMM es el verdadero regulador de volumen:** Desactivarlo aumenta los trades a costa de una degradación severa en el Sharpe Ratio y en el control del Overlap (se pierde la causalidad temporal robusta de SOP R2/R3).
3. **El Veto HMM en BEAR_CRASH es la única palanca que aumenta trades Y mejora el rendimiento:** En la semilla 42, XGBoost tiene un edge extraordinario comprando capitulaciones que el HMM veta estáticamente.
