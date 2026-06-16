# 🏛️ Auditoría Institucional y Matriz de Mejoras (WFB Luna V2.5+)

**Fecha de Auditoría:** 2026-06-16
**Estado Actual del Pipeline:** Aprobado para Producción (`deploy_approved: true` | Sharpe 2.75 | WR 60.2%)
**Objetivo de la Auditoría:** Identificar puntos ciegos, ineficiencias matemáticas y fugas de Alpha en la arquitectura actual basándonos en los datos consolidados de la última run mensual.

---

## 1. Análisis de Puntos Débiles (Autopsia de Datos)

A pesar de los extraordinarios resultados globales, los logs internos del sistema revelan fallas estructurales y oportunidades de optimización masivas:

### 1.1. Toxicidad Interna en el Ensemble
El reporte del ensemble muestra que no todas las semillas aportan valor. 
- **La Semilla 94352** generó 61 trades con un **Sharpe de -0.077** y un retorno negativo. 
- **Problema:** El umbral de consenso (`0.55`) es ciego a la calidad de la semilla. Permite que semillas con esperanza matemática negativa diluyan las señales perfectas de semillas fuertes como la `34921` (Sharpe 2.39).

### 1.2. El "Falso Positivo" del MetaLabeler
El log del *Component Value Dashboard* (CVD) arroja un dato preocupante: `skip_metalabeler=true`.
- **Problema:** El pipeline calculó las probabilidades del MetaLabeler (demostrando que aporta un **+18.2pp** de Win Rate), pero **no lo usó como filtro duro**. Si hubiera operado con un umbral de `0.70`, el Sharpe de los trades individuales se habría disparado a `2.36`. Actualmente estamos asumiendo riesgo innecesario.

### 1.3. Sangrado por Tiempo en Mercado (Time-in-Market)
- **Dato:** El CVD revela una penalización de **-8.0pp** en el Win Rate para operaciones que dependen de la salida temporal (Time Barrier) en lugar del Take-Profit/Stop-Loss dinámico.
- **Conclusión:** Nuestro modelo predice shocks inminentes. Si el precio no se mueve en las primeras 24 horas, el *edge* predictivo colapsa. Esperar a que el *Time Barrier* cierre la operación drena capital (especialmente por el *Funding Rate*).

### 1.4. Inoperancia del Modelo LightGBM
- **Dato:** `LGBM Prob disponibles: 0/422`. 
- **Problema:** El orquestador entrenó modelos LGBM pero falló silenciosamente al re-evaluar los trades OOS. Estamos perdiendo una dimensión predictiva no lineal (basada en árboles) que podría descorrelacionar las predicciones del XGBoost.

---

## 2. Decálogo de Mejoras: 10 Hipótesis Estructurales

Basado empíricamente en los datos recolectados, propongo las siguientes 10 hipótesis accionables para la siguiente iteración de investigación:

> [!IMPORTANT]
> Estas hipótesis buscan extraer el "Alpha marginal" que el modelo actual está dejando en la mesa por configuraciones subóptimas o conservadurismo extremo.

### H1. Poda de Semillas Tóxicas (Pre-Ensemble)
**Hipótesis:** Implementar un *Gate* de admisión individual para el Ensemble (ej. exigir `Sharpe > 0.5` en OOS individual) eliminará el lastre de semillas como la `94352`. 
**Impacto Esperado:** Incremento inmediato del Sharpe global del Ensemble al no promediar señales con "ruido bajista".

### H2. Activación Forzada del MetaLabeler (Filtro Duro)
**Hipótesis:** Cambiar `skip_metalabeler=false` y fijar el umbral dinámico en `0.65 - 0.70` descartará el 30% inferior de las señales, incrementando el Win Rate global entre 5 y 10 puntos porcentuales.
**Impacto Esperado:** Menos trades, pero quirúrgicos. Máxima protección del capital.

### H3. Doctrina "Sniper Anomaly" (Cruce XGBoost + KL Guard)
**Hipótesis:** El CVD (Componente 9) demostró que cuando el XGBoost tiene confianza alta (`>= Q50`) en un mercado extremadamente anómalo (`KL <= Q25`), el Sharpe se dispara a **3.22**. Crear un "Fast-Track" que apruebe estos trades saltándose otras restricciones capturará ineficiencias profundas (liquidaciones en cascada).
**Impacto Esperado:** Captura de eventos de "Cisne Negro" rentables.

### H4. Reducción Drástica del TBM (Triple Barrier)
**Hipótesis:** Dado que el Holding Time prolongado resta -8.0pp de WR, reducir la barrera temporal máxima a 24 horas o implementar un *Time-Decay Stop Loss* (el Stop Loss sube a Break-Even tras X horas) cortará el riesgo de exposición pasiva.
**Impacto Esperado:** Mejora masiva del Calmar Ratio y drástica reducción del *Funding Drag*.

### H5. Censura de Regímenes HMM ("Calm Range")
**Hipótesis:** El estado HMM `2_CALM_RANGE` genera retorno negativo constante. Hardcodear una censura absoluta que bloquee el trading durante este estado HMM evitará operaciones de "chop" o lateralización.
**Impacto Esperado:** Curva de equity mucho más suave durante los veranos criptográficos.

### H6. Reactivación y Ensamblaje del LightGBM
**Hipótesis:** Reparar la fuga del pipeline OOS para el LightGBM y promediar geométricamente su probabilidad con la del XGBoost (`sqrt(prob_xgb * prob_lgbm)`) reducirá los falsos positivos por *overfitting* de un solo tipo de algoritmo.
**Impacto Esperado:** Aumento de la significancia estadística (p-binomial) de las señales.

### H7. Recalibración del SFI con Costos Reales (0.10%)
**Hipótesis:** Ahora que sabemos que OKX Futuros cuesta 0.10% (no 0.25%), relajar el costo en la función de selección de variables (SFI) permitirá que el sistema "descubra" features que antes descartaba por ser marginalmente no-rentables bajo el estándar punitivo del 0.25%.
**Impacto Esperado:** Aumento en la cantidad de trades válidos (solucionando el problema matemático de tener pocos trades para el DSR).

### H8. Ponderación de Capital Basada en "Alpha Triggers"
**Hipótesis:** Los trades iniciados por la variable `alpha_genetic_score` tuvieron un Win Rate del 65.6%. Asignar un multiplicador de Kelly (ej. x1.2) a los trades disparados por este *feature* maximizará el retorno del capital invertido.
**Impacto Esperado:** Mayor rentabilidad nominal aprovechando asimetrías demostradas.

### H9. Simetría Direccional (Módulos Cortos / Short)
**Hipótesis:** El modelo carece de datos para operaciones bajistas (`Short` = 0 trades). Entrenar un MetaLabeler simétrico para entornos HMM `3_CALM_BEAR` permitiría ganar dinero durante caídas prolongadas.
**Impacto Esperado:** Neutralidad de mercado y flujo de caja constante sin importar la dirección de BTC.

### H10. Optimización de la Asignación de Fracción Kelly (Apalancamiento)
**Hipótesis:** El sistema asignó una media del 10% del capital por trade. Sin embargo, con un MaxDD empírico del 0.19%, estamos enormemente sub-apalancados de cara al riesgo real. Permitir que el umbral de Kelly suba su límite artificial (respetando los márgenes del exchange) multiplicaría la rentabilidad sin amenazar el *Kill Switch* del 15% de DD.
**Impacto Esperado:** Escalar la rentabilidad base del 60% anual al +300% de manera controlada.
