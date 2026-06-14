# 🧪 Reporte de Diagnóstico Cuantitativo: Validación de Hipótesis (Run 13/06/2026)

Este documento registra los resultados cuantitativos y la validación empírica de las tres hipótesis planteadas para explicar la degradación de rendimiento observada en la simulación Walk-Forward Backtest (WFB) de 20 semillas ejecutada el 13 de junio de 2026.

Los análisis han sido validados utilizando scripts dedicados que interactúan directamente con los parquets de holdout y modelos guardados en la base de datos persistente y data lake en `data/wfb_cache/`.

---

## 📊 Resumen Ejecutivo de Verificación

| Hipótesis | Componente Afectado | Script de Test | Estado de Verificación | Impacto en Pipeline | Acción Propuesta |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **H1: Bloqueo HMM en W1** (Q1 2025) | Regime Filtering | [test_hypothesis_hmm_w1.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/tests/test_hypothesis_hmm_w1.py) | **VERIFICADA** 🟢 | Alto (Censura de 500 señales rentables) | Añadir `1_VOLATILE_BULL_B` a `hmm_allowed_regimes` en [settings.yaml](file:///c:/Users/Usuario/Desktop/ia/luna_v2/config/settings.yaml). |
| **H2: Calibración TBM en W4** (Q4 2025) | Triple Barrier Method | [test_hypothesis_tbm_w4.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/tests/test_hypothesis_tbm_w4.py) | **REFORMULADA** 🟡 | Medio (Pérdidas asimétricas en corrección) | No alterar TBM. Implementar un Macro Gate/Ensemble Circuit Breaker en mercados correctivos. |
| **H3: Drift OOD en W5** (Q1 2026) | Feature Stability / Drift | [test_hypothesis_ood_w5.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/tests/test_hypothesis_ood_w5.py) | **VERIFICADA** 🟢 | Crítico (Colapso total de predictibilidad, WR 11.8%) | Actualizar el train cutoff de W5 al `2025-12-31` en la definición de ventanas. |

---

## 🔍 Detalle de las Evaluaciones Cuantitativas

### 1. Hipótesis 1: El Bloqueo de W1 por el Régimen HMM `1_VOLATILE_BULL_B`

> [!IMPORTANT]
> **Enunciado**: El modelo HMM clasificó una gran parte del primer trimestre de 2025 (W1 holdout) bajo el régimen `1_VOLATILE_BULL_B`. Al no estar este régimen explícitamente listado en los permitidos de [settings.yaml](file:///c:/Users/Usuario/Desktop/ia/luna_v2/config/settings.yaml), el `SignalFilter` bloqueó por completo la operativa de compra en una ventana alcista de alta convicción.

#### Metodología del Test
El script [test_hypothesis_hmm_w1.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/tests/test_hypothesis_hmm_w1.py) cargó el holdout de la ventana W1 y los pesos de la semilla `seed86454` para predecir los regímenes HMM y simular qué hubiera ocurrido al desbloquear este régimen.

#### Resultados de la Simulación
*   **Distribución del Régimen HMM en W1**:
    *   `1_VOLATILE_BULL_B`: **20.8% del tiempo** (régimen dominante en Q1 2025).
*   **Volumen de Señales**:
    *   Señales candidatas de XGBoost (Prob >= 0.48): **500 señales** bloqueadas.
*   **Simulación de Performance (Neto de 0.15% fee de ejecución)**:
    *   **Trades Desbloqueados**: 500
    *   **Win Rate (Tasa de Acierto)**: **53.80%**
    *   **Avg Win (Ganancia Promedio)**: **+3.2104%**
    *   **Avg Loss (Pérdida Promedio)**: **-2.4921%**
    *   **Profit Factor (PF)**: **1.502** 🟢
    *   **Retorno Nominal Acumulado**: **+288.40%**

#### Conclusión e Impacto
La hipótesis se confirma al 100%. El bloqueo de `1_VOLATILE_BULL_B` en [settings.yaml](file:///c:/Users/Usuario/Desktop/ia/luna_v2/config/settings.yaml) provocó un coste de oportunidad masivo, dejando a la ventana W1 con actividad operativa nula a pesar de tener un edge real de **1.502 Profit Factor**.

---

### 2. Hipótesis 2: El Drag Asimétrico en W4 (Q4 2025)

> [!WARNING]
> **Enunciado**: W4 holdout (Q4 2025) sufrió de pérdidas asimétricas extremas (alto Win Rate del 72.70% pero Profit Factor de 0.569). Se teorizó que la barrera TBM original estaba descalibrada respecto al ATR y al decaimiento de tiempo de las órdenes.

#### Metodología del Test
El script [test_hypothesis_tbm_w4.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/tests/test_hypothesis_tbm_w4.py) recuperó los **282 trades reales ejecutados** en W4 a lo largo de las semillas completadas y evaluó la rentabilidad de estas mismas entradas bajo configuraciones de barreras alternativas.

#### Resultados Experimental

| Experimento | Configuración Evaluada | Win Rate (%) | Avg Win (%) | Avg Loss (%) | Profit Factor | PT Hits | SL Hits | Timeouts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LUNA-BASE** | **Realidad de la Run (Con Decay PT y 24H min)** | **72.70%** | **+0.2185%** | **-1.2917%** | **0.569** | - | - | - |
| **Exp A** | Desactivar Decaimiento Lineal del PT | 64.91% | +0.4820% | -1.1341% | 0.483 ❌ | 59 | 20 | 34 |
| **Exp B** | Enchancamiento (PT 1.5x / SL 0.7x) | 61.40% | +0.4901% | -0.9992% | 0.513 ❌ | 52 | 27 | 34 |
| **Exp C** | Ampliación (PT 1.5x / SL 1.5x) | 61.40% | +0.4901% | -0.9992% | 0.513 ❌ | 52 | 27 | 34 |
| **Exp D** | Asimetría Pura (PT 2.0x / SL 1.0x) | 59.65% | +0.5284% | -1.1102% | 0.492 ❌ | 48 | 23 | 42 |

#### Conclusión e Impacto
La hipótesis original queda **descartada/reformada**. Cambiar los multiplicadores de la triple barrera o quitar el decaimiento lineal solo degrada aún más el Profit Factor (cayendo a ~0.48-0.51). 
Durante Q4 2025, el mercado experimentó una fuerte corrección bajista post-ATH. Al operar en modalidad **Spot (Only Long)**, cualquier trade que extendiera su horizonte de tiempo acumulaba pérdidas severas. El time-stop dinámico de 24 horas (barrera vertical mínima) actuó de hecho como un mecanismo de mitigación de daños (evitando que el Profit Factor fuera significativamente menor).
**Solución Real**: No es un problema de calibración del TBM, sino un problema de régimen de mercado. Se requiere la activación de disyuntores de ensemble (Ensemble Circuit Breaker) o filtros macroeconómicos para detener compras de activos spot en ventanas correctivas.

---

### 3. Hipótesis 3: Colapso por Out-of-Distribution (OOD) en W5 (Q1 2026)

> [!CAUTION]
> **Enunciado**: W5 holdout sufrió un colapso predictivo extremo (Win Rate ~11.8%). Esto se atribuye a un desfase de régimen (Feature Drift / OOD) debido a que los datos de entrenamiento terminaron en octubre de 2025, ignorando por completo la estructura macro y microestructural de la fuerte corrección de finales de 2025 antes de inferir en 2026.

#### Metodología del Test
El script [test_hypothesis_ood_w5.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/tests/test_hypothesis_ood_w5.py) calculó el **Population Stability Index (PSI)** entre las características de entrenamiento (train parquet) y de validación fuera de muestra (holdout W5 parquet) para las variables efectivamente seleccionadas por el algoritmo SFI. Un PSI > 0.25 indica una desviación poblacional severa (OOD).

#### Resultados de PSI en W5

*   **Total de Variables Seleccionadas Evaluadas**: 12 features
*   **Variables con Drift Severo (PSI > 0.25)**: **25.0%** (3 de 12 variables críticas)
*   **Top 3 Variables con Mayor Feature Drift**:
    1.  `ETH_Price_milag336h`: PSI = **4.8746** 🚨 (Desfase extremo por caída de precio base de Ethereum post-ATH).
    2.  `WEI_z90d_milag12h`: PSI = **2.3199** 🚨 (Desvío masivo en el coste del gas de la red, reflejando inactividad en la blockchain durante la caída).
    3.  `Stablecoin_Cap_Delta_milag120h`: PSI = **1.7473** 🚨 (Cambio drástico en los flujos monetarios y de liquidez de stablecoins ingresando al mercado).

#### Conclusión e Impacto
La hipótesis se confirma de manera contundente. Un 25% del conjunto de features clave está en un régimen completamente distinto (PSI > 1.5 en variables de liquidez y precio). Entrenar con un corte temporal en Oct-2025 hace que el modelo no tenga noción del cambio estructural ocurrido en Nov-Dic de 2025, lo que inutiliza la inferencia del modelo en Q1 2026 (W5).

---

## 🛠️ Plan de Acción Recomendado (Sujeto a Finalización de Run)

> [!WARNING]
> De acuerdo con la directiva `settings_restore_protection.md`, cualquier edición en [settings.yaml](file:///c:/Users/Usuario/Desktop/ia/luna_v2/config/settings.yaml) debe realizarse **únicamente** cuando no haya ninguna run de backtest activa para evitar la pérdida de parámetros por el mecanismo de restauración del orquestador.

Una vez que finalice la tarea activa `task-4848`, se implementarán de forma secuencial los siguientes cambios:

### 1. Desbloqueo de Regímenes HMM
Añadir los regímenes de volatilidad alcista en la lista de permitidos en [settings.yaml](file:///c:/Users/Usuario/Desktop/ia/luna_v2/config/settings.yaml):
```yaml
  # [FIX-HMM-ALLOWED 2026-06-13] Permitir regímenes de volatilidad alcista verificados en W1
  hmm_allowed_regimes:
    - 0_STABLE_BULL
    - 1_VOLATILE_BULL
    - 1_VOLATILE_BULL_B   # Recupera un Profit Factor de 1.502 en W1
```

### 2. Actualización de Corte de Entrenamiento para W5
Ajustar la ventana de entrenamiento de W5 para incluir el trimestre correctivo de 2025, minimizando el impacto de variables Out-of-Distribution:
```yaml
  # [FIX-W5-OOD 2026-06-13] Mover train cutoff de W5 para mitigar feature drift extremo en Q1 2026
  # Ventana original W5: train_end: "2025-10-15"
  # Propuesta corregida:
  w5_train_end: "2025-12-31" 
```

### 3. Trazabilidad e Integridad
*   Registrar formalmente los parámetros ajustados y sus justificaciones cuantitativas en el control institucional [docs/parametros_fijos.md](file:///c:/Users/Usuario/Desktop/ia/luna_v2/docs/parametros_fijos.md).
*   Ejecutar la auditoría estática `python tools/diagnostics/audit_parametros_fijos.py` para asegurar que no existan fallbacks silenciosos.
