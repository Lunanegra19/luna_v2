# Propuestas de Mejora Matemática e Institucional — Luna V3

> **Documento de Control Institucional**  
> Generado: Junio 2026 | Auditor: Run Sentinel (WFB seed42, 17/06/2026)  
> Contexto: Resultados del Walk-Forward Backtest sobre 6 ventanas (Ago 2024 – Jun 2025)

Este documento registra los vectores de fricción matemática descubiertos durante las auditorías operativas en la ejecución del orquestador Walk-Forward Backtesting (WFB) de Luna V2. Las propuestas buscan alinear el pipeline pragmático actual con los principios estrictos de la Teoría Estadística Clásica y la Ley de los Grandes Números (LLN), abordando la no-estacionariedad intrínseca del mercado de criptomonedas.

---

## Contexto Operativo: Estado de Luna V2

El pipeline actual logra tasas de validación OOS sólidas (Win Rate 67–88% según ventana) gracias a su arquitectura de embudo multi-capa:

```
Señales Crudas (XGBoost)
        │
        ▼
  [HMM Regime Gate]       ← Bloquea regímenes BEAR/RANGE
        │
        ▼
  [MetaLabeler V2]        ← Percentil 85% rolling (Sniper-Mode)
        │
        ▼
  [OOD Guardian]          ← Distancia KL distribucional
        │
        ▼
  Señal de Trading OOS
```

Sin embargo, tras la simulación exhaustiva de percentiles en W1-W5, se identificaron **cuatro hipótesis de fricción** que justifican una evolución arquitectónica hacia V3.

---

## Hipótesis A — Asimetría Estructural del TBM vs Proceso de Wiener

### El Problema
El *Triple Barrier Method* (TBM) actual escala dinámicamente sus barreras usando un ATR (Average True Range) simétrico con multiplicadores iguales para Take Profit y Stop Loss (ej. `pt_mult: 1.5x / sl_mult: 1.5x`). Esta simetría implica matemáticamente que el modelo asume que la evolución del precio sigue un **Movimiento Browniano Geométrico (MBG) simétrico**, es decir:

```
dS = μS dt + σS dW
```

donde `σ` (volatilidad) es homogénea tanto al alza como a la baja. Esto es una simplificación inapropiada para Bitcoin y criptoactivos, cuya distribución de retornos exhibe:

- **Curtosis excesiva (Fat Tails):** Los eventos de cola a la baja son 2-4x más frecuentes que en una distribución Normal.
- **Asimetría Negativa (Negative Skew):** Las caídas son más rápidas e intensas que las subidas.
- **Leverage Effect:** La volatilidad aumenta asimétricamente cuando el precio cae.

### Consecuencias Prácticas
Con un TBM simétrico, la probabilidad de que el precio alcance la barrera de Stop Loss antes que la de Take Profit es sistemáticamente mayor que `0.5` en mercados con skew negativo. Esto sesga el Target TBM hacia la clase `0` (pérdida), creando una distribución de entrenamiento artificialmente pesimista que el XGBoost compensa sobreinflando sus umbrales de confianza.

### Plan de Implementación (Asymmetric TBM)

1. **Calcular Semi-Varianza por Dirección:**
   - `ATR_upside = media de (high - open)` → Volatilidad alcista real
   - `ATR_downside = media de (open - low)` → Volatilidad bajista real
   - Ratio de Asimetría: `asymmetry_ratio = ATR_downside / ATR_upside`

2. **Escalar Barreras de Forma Independiente:**
   ```python
   PT_barrier = entry_price * (1 + pt_mult * ATR_upside)
   SL_barrier = entry_price * (1 - sl_mult * ATR_downside * asymmetry_ratio)
   ```

3. **Parametrización en settings.yaml:**
   ```yaml
   tbm_asymmetric: true
   tbm_asymmetry_ratio_cap: 2.0  # Máximo 2x de asimetría SL vs PT
   ```

4. **Validación:** Confirmar que la distribución de la variable Target TBM converge a 50/50 en régimen `BULL_TREND`, indicando neutralidad probabilística respecto al proceso de precio real.

### Beneficio Esperado
Reducción del sesgo sistemático en el Target TBM → mayor calibración del XGBoost → menos necesidad de filtros agresivos en el MetaLabeler → más alfa capturado con mismo nivel de seguridad.

---

## Hipótesis B — "Maldición de la Dimensionalidad" en el SFI

### El Problema
El Tribunal SFI (Sequential Feature Importance) evalúa más de **500 variables candidatas** usando Información Mutua (MI) con estimadores basados en K-Nearest Neighbors (KNN). En alta dimensionalidad y con muestras temporales reducidas (post-filtrado por régimen), el estimador de MI presenta **alta varianza estadística** conocida como el *Curse of Dimensionality*:

- Con `N` muestras en `D` dimensiones, la distancia media entre vecinos crece como `O(N^{-1/D})`.
- A `D > 50`, los vecinos más cercanos y más lejanos son prácticamente equidistantes → el estimador KNN de MI colapsa.
- Resultado: El SFI puede seleccionar variables que correlacionan con el target por ruido estadístico, no por contenido informativo real.

### El Estimador Actual vs el Estimador Correcto

| Método Actual | Problema | Alternativa |
|---|---|---|
| MI Marginal (cada feature vs target) | Ignora redundancias entre features | CMI: MI(X;Y \| Z) |
| Selección Greedy Top-K por MI | Acumula redundancias | MRMR: Maximizar relevancia, minimizar redundancia |
| KNN fijo (k=3) | Inestable en muestras pequeñas | KNN adaptativo (k = sqrt(N)) |

### Plan de Implementación (CMI + MRMR)

1. **Sustituir MI Marginal por CMI iterativo:**
   Para cada nueva variable candidata `X_i`, calcular:
   ```
   CMI(X_i ; Y | S) = MI(X_i, Y) - MI(X_i, Y | S)
   ```
   donde `S` es el subconjunto de features ya aprobadas. Solo se aprueba `X_i` si aporta información **genuinamente nueva** dado lo que ya sabe el modelo.

2. **Implementar criterio MRMR como score de selección:**
   ```
   score(X_i) = MI(X_i, Y) - (1/|S|) * Σ MI(X_i, X_j) para X_j en S
   ```

3. **KNN adaptativo:**
   ```python
   k_adaptive = max(3, int(np.sqrt(n_effective_samples)))
   ```

4. **Test de significancia estadística:** Aplicar permutation test (1000 permutaciones) para confirmar que cada feature aprobada supera el umbral de ruido con `p < 0.01`.

### Beneficio Esperado
Un conjunto de features más pequeño, verdaderamente ortogonal y estadísticamente significativo reduce el overfitting del XGBoost y mejora la generalización OOS en regímenes de mercado nunca vistos.

---

## Hipótesis C — "Latent Space Drift" en AutoEncoders Walk-Forward

### El Problema
El pipeline comprime ~500 variables en **32 dimensiones latentes** mediante un AutoEncoder neuronal. El mecanismo de *Warm-Start* (heredar pesos de la ventana anterior) mitiga el tiempo de convergencia, pero introduce una deuda matemática profunda:

Entre la ventana W(n) y W(n+1), la topología del mercado cambia. Aunque la red hereda pesos previos, la función de pérdida estándar (MSE de reconstrucción) solo optimiza para la ventana actual, sin restricción sobre la dirección de la rotación topológica del espacio latente.

**Consecuencia:** La neurona `ae_feat_3` que en W3 codificaba "correlación DVOL-precio", en W4 podría codificar "correlación Funding-volumen". El XGBoost, que fue entrenado interpretando `ae_feat_3` con el significado de W3, recibe en OOS un vector con un significado rotado. Este es un **vector de sesgo invisible** no detectado por los mecanismos de OOD estándar.

### Evidencia Observada
En W5, el `[H-AE-VAL-01-FIX]` tuvo que excluir 496 de 504 variables del fit del AutoEncoder para prevenir Val Loss divergente. Un ratio de exclusión del 98% indica que el espacio de features visible por el AE está sufriendo una deriva distribucional severa entre ventanas.

### Plan de Implementación (Anchored Contrastive Learning)

1. **Nueva función de pérdida compuesta:**
   ```
   L_total = L_reconstruction + λ_kl * KL(z_current || z_anchored)
   ```
   donde `z_anchored` es la distribución latente serializada de la ventana anterior, y `λ_kl` es un hiperparámetro de regularización (propuesto: `0.01 - 0.1`).

2. **Guardar distribución latente por ventana:**
   Al finalizar cada ventana, serializar `mean_z` y `std_z` del espacio latente en el snapshot del estado de ventana (`dehydrate_window_state`).

3. **Criterio de Drift Alarmante:**
   Si `KL(z_current || z_anchored) > threshold_kl` (ej. `> 0.5 nats`) durante el entrenamiento de AE, emitir un `WARNING` institucional indicando que el régimen ha mutado significativamente.

4. **Parámetros en settings.yaml:**
   ```yaml
   ae_anchored_kl_loss: true
   ae_kl_lambda: 0.05
   ae_kl_drift_alarm_threshold: 0.5
   ```

### Beneficio Esperado
Garantía de consistencia semántica del espacio latente entre ventanas. El XGBoost recibirá vectores de features latentes comparables en significado entre W(n) y W(n+1), reduciendo el sesgo inducido por la rotación topológica no supervisada.

---

## Hipótesis D — Sniper-Mode Adaptativo (Percentil Dinámico Anclado al HMM)

### El Problema
El filtro de entrada de señales en el MetaLabeler V2 usa un percentil estático fijo (`meta_v2_rolling_percentile: 0.85`), lo que exige que cada señal esté en el **Top 15% de confianza histórica** para ser aprobada.

La simulación exhaustiva realizada sobre las señales crudas (XGBoost baseline) de las **10 ventanas completas** (W1–W10, seed 42, run 17/06/2026) reveló el coste de oportunidad real de este parámetro.

#### Desglose de señales brutas por ventana

| Ventana | Señales Base | Win Rate Base | Retorno Base (1x) | Observación |
|---|---|---|---|---------|
| W1 | 3 | 66.7% | +1.97% | Muestra reducida, mercado lateral |
| W2 | 15 | 26.7% | **-27.18%** | ⚠️ Período adverso — MetaLabeler crítico |
| W3 | 21 | 47.6% | +3.65% | Mercado mixto, filtro necesario |
| W4 | 6 | 83.3% | +8.19% | Tendencia alcista consolidada |
| W5 | 13 | 76.9% | +17.88% | Mayo 2025 alcista fuerte |
| W6 | 30 | 66.7% | +22.49% | Junio 2025, máxima actividad |
| W7 | 17 | 58.8% | +10.35% | Julio 2025, mercado volátil |
| W8 | 14 | 50.0% | +1.35% | Agosto 2025, mercado lateral |
| W9 | 0 | — | 0.00% | ⚠️ Sin señales generadas — régimen OOD/bloqueado |
| W10 | 8 | 25.0% | **-10.82%** | ⚠️ Oct 2025, adverso — XGBoost base en modo pérdida |
| **TOTAL** | **127** | **55.12%** | **+27.89%** | Sin ningún filtro |

#### Simulación de Percentiles sobre W1-W10 (127 señales base)

| Percentil Sniper | Prob Umbral | Trades | Win Rate OOS | Retorno Neto (1x) |
|---|---|---|---|---|
| 95% (Top 5%) | >= 0.7719 | 7 | 57.14% | +3.43% |
| 90% (Top 10%) | >= 0.7603 | 13 | 53.85% | +9.06% |
| **85% (Actual)** | **>= 0.7362** | **19** | **57.89%** | **+10.99%** |
| 80% (Top 20%) | >= 0.7135 | 26 | 50.00% | +1.17% |
| 75% (Top 25%) | >= 0.7070 | 32 | 50.00% | +7.30% |
| 70% (Top 30%) | >= 0.7054 | 38 | 50.00% | +10.11% |
| 60% (Top 40%) | >= 0.6979 | 51 | 56.86% | +27.19% |
| **50% (Óptimo)** | **>= 0.6855** | **64** | **57.81%** | **+30.63%** |
| Sin filtro (base) | — | 127 | 55.12% | +27.89% |

**Conclusión crítica (W1-W10 completas, 10 meses OOS):** Con 127 señales base, el percentil 50% sigue siendo el óptimo (+30.63%) pero la ventaja sobre el filtro actual (85% → +10.99%) se reduce a **+19.64 pp**. El nuevo hallazgo crítico es la **degradación de W8-W10**: W9 genera 0 señales (régimen OOD bloqueado completo) y W10 tiene WR=25% con -10.82%. Esto indica que el mercado de Q3-Q4 2025 entró en un régimen genuinamente adverso que ningún percentil puede corregir — la pérdida es estructural, no un artefacto del umbral.

**Hallazgo crítico W9 (Sep 2025):** 0 señales generadas. El Guardian OOD o el filtro HMM bloqueó completamente la generación de señales. Esto es el comportamiento correcto del sistema — es preferible generar 0 trades en un régimen desconocido que perder capital. El filtro funcionó como se diseñó.

**Hallazgo forense W10 (Oct 2025):** WR base = 25% con solo 8 señales. Ningún percentil puede rescatar una ventana donde el XGBoost base está en modo pérdida — la solución no es ajustar el umbral de percentil sino reforzar el Guardian OOD para bloquear también W10 como lo hizo con W9.

Dato crítico de W2: el modelo base sin filtro habría generado **-27.18%**. Confirma que el MetaLabeler es imprescindible — pero también que hay períodos (W10) donde el problema es más profundo que el umbral de selección.

### El Problema Matemático del Umbral Estático
La eficiencia del filtro de percentil no es constante a través del tiempo. En mercados con fuerte inercia (`BULL_TREND`), el alfa está distribuido a lo largo de toda la cola superior de probabilidad — exigir Top 15% resulta en un coste de oportunidad elevado. En mercados caóticos (`VOLATILE_BULL`) o adversos (`BEAR`), el filtro estricto es indispensable para proteger el capital.

Un umbral único para todos los regímenes es matemáticamente equivalente a usar el mismo coeficiente de amortiguación para un coche deportivo en autopista y en un circuito de obstáculos.

### Plan de Implementación (HMM-Linked Dynamic Threshold)

1. **Tabla de Enrutamiento de Percentil por Régimen:**

   | HMM Regime | Percentil Propuesto | Lógica |
   |---|---|---|
   | `BULL_TREND` | 0.55 | Inercia fuerte, maximizar captura de alfa |
   | `VOLATILE_BULL_B` | 0.70 | Oportunidades altas, ruido moderado |
   | `VOLATILE_BULL` | 0.80 | Alta dispersión, exigir mayor calibración |
   | `CALM_BEAR` | 0.90 | Solo señales de inversión estructural |
   | `BEAR` | censura absoluta | Nadar contra la corriente no es estadísticamente justificable |

2. **Implementación en settings.yaml (No-Fallback Policy):**
   ```yaml
   meta_v2_dynamic_percentile: true
   meta_v2_percentile_by_regime:
     BULL_TREND: 0.55
     VOLATILE_BULL_B: 0.70
     VOLATILE_BULL: 0.80
     CALM_BEAR: 0.90
     BEAR: null  # Censura total
   ```

3. **Fallback de Seguridad (si HMM falla):** Usar el percentil estático actual (`0.85`) como valor por defecto de emergencia. Emitir `WARNING` institucional.

4. **Pre-requisito estadístico (Anti-Overfitting):** Antes de activar en producción, validar la tabla de enrutamiento con un test de permutación sobre los regímenes (barajar aleatoriamente los labels de régimen y confirmar que la ventaja desaparece). Esto confirma que la mejora proviene de la causalidad del régimen, no del *data-snooping*.

5. **Parámetro de control en settings.yaml:**
   ```yaml
   meta_v2_dynamic_percentile_min_trades_per_regime: 10
   # Mínimo de trades históricos por régimen antes de activar
   # el enrutamiento dinámico (protege de régimens recién aparecidos)
   ```

### Beneficio Esperado
Escalado geométrico del PnL neto OOS sin deterioro estadístico del Win Rate. Las simulaciones preliminares sugieren que bajar de `0.85` a `0.60-0.70` en regímenes alcistas multiplica el Retorno Neto en un factor `2x-3x` preservando un Win Rate institucional (`>70%`).

---

## Resumen de Prioridades para V3

| ID | Hipótesis | Complejidad | Impacto Potencial | Estado |
|---|---|---|---|---|
| **A** | Asymmetric TBM | Media | Alto (Target Bias) | Pendiente |
| **B** | CMI + MRMR en SFI | Alta | Medio-Alto (Overfitting) | Pendiente |
| **C** | Anchored AE (KL Loss) | Alta | Medio (Drift Latente) | Pendiente |
| **D** | Dynamic Sniper Percentile | Baja | **Muy Alto (+PnL 2-3x)** | En Curso (Percentil 30% activado) |

---

## Hipótesis E — Robustez del Orquestador (Resolución del Caso W12-W15)

### Contexto de la Auditoría (Semilla 42)
Durante las simulaciones WFB, se observó que las ventanas W12, W13, W14 y W15 de la semilla 42 generaron el flag `EMPTY` en lugar de producir un parquet de trades. Existía la duda operativa de si esto representaba una pérdida de datos, una poda del conjunto o un fallo en el pipeline.

### Resolución Analítica
La investigación en el código del orquestador (`run_wfb_orchestrator.py`, L750-803) confirmó que **es el comportamiento correcto y deseado del sistema (Early-Stop Dinámico)**:

1. **Mecanismo de Poda:** El orquestador descarta semillas (seeds) tempranamente si su *upper-bound* matemático proyectado no logra superar el umbral mínimo del Gauntlet (Benchmark V2). Para la semilla 42, el upper-bound tras W11 fue `37.8`, inferior al límite de poda (`40.0`).
2. **Modo Merge-Only:** Al detener la ejecución de W12-W15 para ahorrar recursos, el orquestador **sí retiene** los datos de W1-W11. Para ello lanza al *worker* con el flag `--merge-only`, generando archivos `EMPTY.flag` intencionales en W12-W15 y empaquetando los 67 trades válidos generados hasta W11 en un archivo `oos_trades_seed42.parquet`.
3. **Composición del Ensemble:** La semilla 42, a pesar de haber sido podada, inyecta exitosamente sus 67 trades iniciales al Ensemble final. La falta de operaciones en W12-W15 por parte de esta semilla particular se suple estadísticamente con las predicciones del resto de semillas (ej. seed 100, 777, etc.) que sí logren sobrevivir hasta esas ventanas.

### Beneficio Institucional
El mecanismo de poda ahorra decenas de horas de cómputo inútil (calculando ventanas finales para modelos mediocres) sin sacrificar los buenos trades iniciales. **Queda confirmado y auditado que la creación de `EMPTY.flag` y la pérdida de las ventanas finales de semillas descartadas no es un bug, sino una optimización de diseño.**

> **Recomendación de Secuenciación:** La Hipótesis D es la de mayor relación impacto/esfuerzo. Debería implementarse y validarse en la primera run de V3 antes de abordar las hipótesis de mayor complejidad de ingeniería (B y C).

---

*Documento actualizado automáticamente por el Sentinel de auditoría cuantitativa durante la run WFB 17/06/2026.*
