# Análisis de Sigmas del Modelo (Ensamble W1-W12)

Este documento documenta la medición de los **Sigmas (desviaciones estándar)** del modelo tras el ensamble de las 29 semillas a lo largo de las 12 ventanas Walk-Forward (W1-W12) en 2025. 

Analizamos los resultados comparando dos métodos de agregación:
1. **Método B (Unison Portfolio)**: Agrega todas las señales de forma proporcional. Muestra máxima robustez por volumen de datos ($N = 387$ trades únicos).
2. **Método A (Consenso $K \ge 10$)**: Solo toma operaciones cuando al menos 10 semillas votan a favor simultáneamente. Muestra máxima calidad y filtrado de ruido ($N = 85$ trades consolidados).

---

## 1. Resumen Comparativo de Métricas de Entrada

| Métrica | Método B (Unison) - Bruto | Método B (Unison) - Neto | Método A ($K \ge 10$) - Bruto | Método A ($K \ge 10$) - Neto |
| :--- | :---: | :---: | :---: | :---: |
| **Trades Consolidados** | 387 | 387 | 85 | 85 |
| **Retorno Medio por Trade** | +0.1706% | +0.0534% | **+0.7181%** | **+0.1147%** |
| **1-Sigma (SD)** | 1.9865% | 0.2634% | 1.9562% | 0.2909% |
| **Win Rate** | 49.61% | 36.43% | **61.18%** | **57.65%** |
| **Sharpe por Trade** | 0.0859 | 0.2028 | **0.3671** | **0.3943** |
| **Z-score (Sigmas)** | **1.6866 Sigmas** | **3.9499 Sigmas** | **3.2760 Sigmas** | **3.5010 Sigmas** |
| **p-value (Azar)** | 9.2489% | **0.0093%** | **0.1531%** | **0.0745%** |
| **Significancia** | No significativa | **Extremadamente Alta** | **Muy Alta (Bruto)** | **Extremadamente Alta** |

---

## 2. Límites de Desviación Estándar (Sigmas de Riesgo / VaR)

Los sigmas de la distribución empírica nos permiten estimar las pérdidas/ganancias extremas esperadas por trade consolidado.

### Método B (Unison Portfolio)
* **Retornos en Bruto**:
  * $\pm$ 1-Sigma: $[-1.816\%, +2.157\%]$
  * $\pm$ 2-Sigmas (VaR 95%): $[-3.802\%, +4.144\%]$
  * $\pm$ 3-Sigmas (VaR 99.7%): $[-5.789\%, +6.130\%]$
* **Retornos Netos (Apalancados/Kelly)**:
  * $\pm$ 1-Sigma: $[-0.210\%, +0.317\%]$
  * $\pm$ 2-Sigmas (VaR 95%): $[-0.473\%, +0.580\%]$
  * $\pm$ 3-Sigmas (VaR 99.7%): $[-0.737\%, +0.844\%]$

### Método A (Consenso $K \ge 10$)
* **Retornos en Bruto**:
  * $\pm$ 1-Sigma: $[-1.238\%, +2.674\%]$
  * $\pm$ 2-Sigmas (VaR 95%): $[-3.194\%, +4.631\%]$
  * $\pm$ 3-Sigmas (VaR 99.7%): $[-5.150\%, +6.587\%]$
* **Retornos Netos (Apalancados/Kelly)**:
  * $\pm$ 1-Sigma: $[-0.176\%, +0.406\%]$
  * $\pm$ 2-Sigmas (VaR 95%): $[-0.467\%, +0.697\%]$
  * $\pm$ 3-Sigmas (VaR 99.7%): $[-0.758\%, +0.988\%]$

---

## 3. Explicación de las Sigmas de Confianza (Z-Score)

### ¿Cómo transforma el Consenso las Sigmas en bruto?
* **En el Método B (Unison)**, al promediar todas las entradas sin importar el nivel de acuerdo, los trades en bruto tienen una significancia muy baja (**1.69 Sigmas** con $p \approx 9.25\%$). Esto no supera el umbral científico estándar de significancia del 95% ($1.96$ Sigmas).
* **En el Método A ($K \ge 10$)**, exigir que al menos 10 semillas estén de acuerdo actúa como un potente filtro contra el ruido. Como predice el **Teorema del Jurado de Condorcet**, los errores individuales no correlacionados se cancelan y el Edge real se refuerza. El retorno medio bruto se cuadruplica (de +0.1706% a **+0.7181%**) y el Win Rate sube del 49.61% al **61.18%**. Esto permite que el modelo bruto obtenga por sí solo **3.2760 Sigmas de confianza** ($p \approx 0.15\%$), siendo extremadamente significativo incluso sin gestión de posición.

### Efecto del Position Sizer y MetaLabeler
Al inyectar el Position Sizer con el criterio de Kelly fraccional, la volatilidad (1-Sigma) del portafolio se contrae drásticamente (de 1.96% a 0.29% para $K \ge 10$). A nivel neto/apalancado:
* **Método B** alcanza **3.9499 Sigmas** debido al enorme tamaño muestral ($N = 387$).
* **Método A ($K \ge 10$)** alcanza **3.5010 Sigmas** con un win rate neto muy superior (**57.65%** vs 36.43%).

---

## 4. Implicaciones para la Gestión de Riesgos

> [!IMPORTANT]
> Bajo el **Método A ($K \ge 10$)**, la pérdida extrema a 3-sigmas netos por operación consolidada es de apenas **$-0.758\%$**. El circuito de parada de emergencia (`dd_kill_switch` en `settings.yaml` fijado en **15.0%**) está sumamente resguardado: se requerirían 20 pérdidas extremas consecutivas para disparar el freno, un evento con probabilidad nula bajo las condiciones modeladas.
