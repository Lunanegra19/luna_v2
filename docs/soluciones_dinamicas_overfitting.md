# ⚡ Soluciones Dinámicas Anti-Overfitting y Adaptabilidad Temporal

Este documento detalla el diseño técnico, la validación empírica y los resultados cuantitativos de los mecanismos **dinámicos y auto-adaptativos** diseñados para evitar el sobreajuste (overfitting) temporal en Luna V2.

Estos mecanismos mitigan de raíz las tres problemáticas diagnosticadas el 13/06/2026:
1. El bloqueo estático de regímenes HMM profitables en **W1** (`1_VOLATILE_BULL_B`).
2. El drag asimétrico en mercados correctivos en **W4** (`3_BEAR_CRASH`).
3. El colapso predictivo por desfase Out-of-Distribution (OOD) en **W5** (Q1 2026).

---

## 🧠 1. HMM Dynamic Allowed Regimes (Habilitación In-Sample)

### Problemática Estática
Hardcodear `hmm_allowed_regimes` en [settings.yaml](file:///c:/Users/Usuario/Desktop/ia/luna_v2/config/settings.yaml) asume que la semántica de los regímenes es constante entre re-entrenamientos. Esto causa look-ahead bias (al forzar una lista optimizada a mano) y bloquea regímenes profitables como `1_VOLATILE_BULL_B` en W1 debido a cambios de etiqueta del HMM estocástico.

### Solución Dinámica
En lugar de una lista estática, el `SignalFilter` evalúa el rendimiento in-sample del clasificador (XGBoost) para cada régimen HMM en la ventana de **validación cruzada/validación in-sample**:
1. Si un régimen genera señales en validación, se calcula su **Profit Factor (PF) neto** (restando comisiones).
2. El régimen se habilita para la fase out-of-sample (Holdout) solo si cumple con los criterios de seguridad in-sample:
   - Mínimo de trades in-sample >= 3 y **Profit Factor > 1.05** con retorno neto positivo.
   - O si el régimen es semánticamente clasificado como alcista/rango (`BULL`/`RANGE`) y su Profit Factor in-sample no destruye capital (`PF > 0.95`).
3. Si el régimen no genera suficientes señales in-sample, se aplica un veto preventivo por defecto para regímenes catalogados históricamente como bajistas (`BEAR_CRASH`).

### Resultados de la Simulación ([test_dynamic_hmm_selection.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/tests/test_dynamic_hmm_selection.py))
La simulación demostró la adaptabilidad del algoritmo para auto-configurar el router:

*   **Ventana W1 (Alcista - Q1 2025)**:
    - Identificó que `1_VOLATILE_BULL_B` in-sample era rentable.
    - **Resultado**: Habilitó el régimen de forma 100% automatizada, desbloqueando las 500 señales rentables que el filtro estático había censurado.
*   **Ventana W4 (Correctiva - Q4 2025)**:
    - Evaluó in-sample los regímenes y determinó la viabilidad individual:
      * `3_BEAR_CRASH_B` (PF 1.11 in-sample) $\rightarrow$ **Habilitado** 🟢
      * `3_BEAR_CRASH` (PF 0.758 in-sample) $\rightarrow$ **Bloqueado** 🔴
      * `1_VOLATILE_BULL` (PF 0.060 in-sample) $\rightarrow$ **Bloqueado** 🔴
      * `1_BULL_TREND` (PF 1.209 in-sample) $\rightarrow$ **Habilitado** 🟢

---

## 🛡️ 2. Prediction Drift Sentinel (OOD Circuit Breaker)

### Problemática Estática
El desajuste poblacional (Feature Drift) en ventanas de transición macro (como W5 holdout en 2026) inutiliza los pesos del clasificador. Calcular el drift sobre variables en crudo (como precios) arroja falsos positivos constantes debido a que las variables no son estacionarias por naturaleza.

### Solución Dinámica
El **Prediction Drift Sentinel** actúa en el nivel jerárquico más alto, midiendo la estabilidad de la inferencia misma. Compara la distribución de las probabilidades calibradas predichas por el ensemble (`xgb_prob_cal`) en el Holdout (datos recientes de producción) contra la distribución in-sample (Validation):
1. Calcula el **Population Stability Index (PSI)** de las predicciones calibradas del modelo.
2. Si `pred_psi` supera un umbral crítico de deformación de probabilidad, el Sentinel atenúa dinámicamente el tamaño de capital Kelly asignado por el Position Sizer:
   - **PSI < 0.08**: Operativa normal (Exposición del 100% de la fracción Kelly).
   - **0.08 <= PSI < 0.20**: Operativa degradada. Se aplica una atenuación lineal al tamaño de posición:
     $$\text{Fracción Kelly Efectiva} = \text{Fracción Kelly Base} \times \left(\frac{0.20 - \text{PSI}}{0.20 - 0.08}\right)$$
   - **PSI >= 0.20**: Parada de Emergencia / Desconexión del Broker para esa ventana (Exposición = 0.0%).

### Resultados de la Simulación ([test_dynamic_prediction_drift.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/tests/test_dynamic_prediction_drift.py))
Validación cruzada del Sentinel en diferentes entornos estructurales:

| Ventana | Estado Estructural | PSI de Predicción | Status del Sentinel | Fracción Kelly | Impacto en Capital |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **W1** | Alcista Consistente | **0.0210** | **NORMAL OPERATIONAL** | **0.2500** | Exposición total (0% penalización) |
| **W3** | Tendencial Estable | **0.0000** | **NORMAL OPERATIONAL** | **0.2500** | Exposición total (0% penalización) |
| **W4** | Mercado Correctivo (Asimétrico) | **0.5960** | **EMERGENCY SHUTDOWN** 🚨 | **0.0000** | Protección total (**-100% exposición**) |
| **W5** | Out-of-Distribution (Q1 2026) | **0.1922** | **DEGRADED RISK** ⚠️ | **0.0163** | Mitigación extrema (**-93.5% exposición**) |

### Beneficios Clave
1. **Detección Ciega de OOD**: El Sentinel no necesita conocer qué variable causó el drift ni la fecha del train cutoff. Simplemente reacciona a la degradación matemática de la predictibilidad del ensemble.
2. **Cero Falsos Positivos**: W1 y W3 (ventanas altamente rentables de la run) mantuvieron el 100% del capital expuesto, operando con eficiencia nominal.
3. **Mitigación Pasiva**: En W4 y W5 (ventanas donde el modelo in-sample colapsó o tuvo asimetrías severas), el capital se puso automáticamente en cuarentena, evitando el Drawdown destructivo.

---

## 📋 Conclusiones Operativas para el SFI y el Position Sizer

1. **Integración en el Pipeline**: 
   - Proponemos integrar el algoritmo de **HMM Dynamic Selection** en el importador de configuraciones de `SignalFilter.apply_hmm` para eliminar la dependencia de `hmm_allowed_regimes` en settings.
   - Proponemos inyectar el **Prediction Drift Sentinel** dentro de `PositionSizer` (o la emisión de señales en `evaluate_ensemble_wfb.py`) para auto-regular la exposición según el PSI rodante.
2. **Trazabilidad**: 
   - Registrar estos nuevos disyuntores dinámicos en [docs/parametros_fijos.md](file:///c:/Users/Usuario/Desktop/ia/luna_v2/docs/parametros_fijos.md) tras la finalización de la run actual.
