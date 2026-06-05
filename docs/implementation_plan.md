# Plan de Rediseño de Calibración, Kelly Sizer y Regularización Optuna (Phase P3 Update)

## Evidencia y Diagnóstico Cuantitativo (Test Scratch Ex-Post Seed 42)

Hemos ejecutado pruebas de hipótesis sobre el conjunto de trades OOS reales de la **semilla 42** (`oos_trades_seed42.parquet`) en el script [test_calibration_and_sizer.py](file:///g:/Mi%20unidad/ia/luna_v2/scratch/test_calibration_and_sizer.py), obteniendo la siguiente evidencia empírica:

1. **Colapso del Modelo XGBoost (Model Collapse)**:
   - En la **Ventana W2**, **38 de 39 trades (32% del dataset)** del régimen `1_BULL_TREND_WEAK` tienen exactamente la misma probabilidad cruda de `0.709399` hasta el sexto decimal.
   - En la **Ventana W5**, **20 de 20 trades (16.4% del dataset)** del régimen `1_BULL_TREND_WEAK` tienen exactamente la misma probabilidad cruda de `0.538462`.
   - **Causa Raíz:** El tamaño del dataset de entrenamiento para regímenes minoritarios es pequeño. Sin embargo, la cota inferior de regularización en Optuna `min_child_weight_min` es fija (e.g. 20 o 30 en `settings.yaml`). Optuna se ve forzado a elegir un `min_child_weight` excesivamente alto para el volumen de muestras, lo que impide que los árboles de XGBoost realicen cualquier partición. El modelo colapsa en un árbol raíz y predice el *base rate* del target en entrenamiento (exactamente `0.709399` en W2 y `0.538462` en W5).

2. **Colapso del Calibrador y Sizing de Kelly Desmedido**:
   - Al recibir probabilidades constantes, el calibrador de regresión isotónica no encuentra varianza en validación, colapsa (`std_cal < 1e-4`) y activa el fallback de `regime_router.py`, el cual revierte a las probabilidades crudas.
   - En W2, el sizer de Kelly interpretó `0.709399` como una señal de altísima confianza, asignando la fracción máxima permitida por el sizer dinámico (~13%) a todos los 39 trades. Esto sobre-dimensionó la exposición del portfolio en una señal con tasa de acierto OOS real de `58.9%` (edge positivo pero moderado).

---

## Resultados de la Run WFB Multi-Seed (2026-06-04/05) — 20 Seeds

Run lanzada el 2026-06-04 22:37 con `--nocache`. El orquestador completó automáticamente una búsqueda de hasta 20 seeds. Finalizada a las ~05:00 del 2026-06-05.

### Resumen de estados

| Estado | Count | Seeds |
|---|---|---|
| **Completadas (W1→W5)** | **2** | 72012, 72101 |
| Early-stop (W1+W4) | 15 | 42, 21782, 51853, 57163, 57872, 62327, 64837, 68843, 71348, 72318, 73346, 79386, 89744, 93636, 94221 |
| Fatal (error técnico) | 3 | 39369, 89871, 92949 |

> **Nota:** seed 92949 tenía WR=66.7% y ret=+0.084% pero murió por crash técnico del proceso, no por métricas. Es una pérdida de información.

### Tabla completa de resultados por seed

| Seed | W1 n | W1 WR | W4 n | W4 WR | Total | WR Global | Ret. total | Estado |
|---|---|---|---|---|---|---|---|---|
| **42** | 4 | 75.0% | 21 | 38.1% | 28 | 39.3% | -0.182% | EARLY-STOP |
| 21782 | 6 | 33.3% | 14 | 35.7% | 20 | 30.0% | -0.225% | EARLY-STOP |
| 39369 | 4 | 25.0% | — | — | 4 | 25.0% | -0.337% | FATAL |
| 51853 | 5 | 40.0% | 21 | 38.1% | 26 | 38.5% | -0.259% | EARLY-STOP |
| 57163 | 3 | 33.3% | 19 | 42.1% | 22 | 31.8% | -0.214% | EARLY-STOP |
| 57872 | 5 | 40.0% | 20 | 40.0% | 25 | 40.0% | -0.256% | EARLY-STOP |
| 62327 | 4 | 50.0% | 18 | 33.3% | 26 | 34.6% | -0.459% | EARLY-STOP |
| 64837 | 5 | 20.0% | 20 | 40.0% | 25 | 36.0% | -0.165% | EARLY-STOP |
| 68843 | 2 | 50.0% | 21 | 33.3% | 30 | 33.3% | -0.411% | EARLY-STOP |
| 71348 | 4 | 50.0% | 19 | 47.4% | 26 | 38.5% | -0.042% | EARLY-STOP |
| **72012** | 6 | 33.3% | 3 | 33.3% | 23 | 34.8% | -0.226% | COMPLETA |
| **72101** | 4 | 25.0% | 18 | **50.0%** | 35 | **45.7%** | **+0.097%** | COMPLETA |
| 72318 | 5 | 40.0% | 21 | 38.1% | 26 | 38.5% | -0.338% | EARLY-STOP |
| 73346 | 6 | 33.3% | 12 | 33.3% | 26 | 30.8% | -0.157% | EARLY-STOP |
| 79386 | 3 | 0.0% | 14 | 35.7% | 17 | 29.4% | -0.436% | EARLY-STOP |
| 89744 | 4 | 50.0% | 21 | 38.1% | 25 | 40.0% | -0.345% | EARLY-STOP |
| 89871 | 6 | 33.3% | — | — | 6 | 16.7% | -0.289% | FATAL |
| **92949** | 4 | **75.0%** | — | — | 6 | **66.7%** | **+0.084%** | FATAL (técnico) |
| 93636 | 5 | 20.0% | 20 | 40.0% | 28 | 32.1% | -0.540% | EARLY-STOP |
| 94221 | 5 | 20.0% | 14 | 42.9% | 19 | 36.8% | -0.186% | EARLY-STOP |

### Detalle ventana-a-ventana de seeds completadas

**Seed 72101 (mejor resultado global):**

| Ventana | Holdout | Trades | WR | Ret. Compuesto | MaxDD | Sharpe | Régimen |
|---|---|---|---|---|---|---|---|
| W1 | 2025 Q1 | 4 | 25.0% | -0.135% | -0.157% | -71.3 | BULL_TREND_WEAK |
| W4 | 2025 Q4 | 18 | **50.0%** | **+0.282%** | -0.166% | **+19.1** | BULL_TREND_WEAK |
| W5 | 2026 Q1 | 13 | 46.2% | -0.050% | -0.375% | -2.6 | BULL_TREND_WEAK |

**Seed 72012:**

| Ventana | Holdout | Trades | WR | Ret. Compuesto | MaxDD | Sharpe | Régimen |
|---|---|---|---|---|---|---|---|
| W1 | 2025 Q1 | 6 | 33.3% | -0.126% | -0.157% | -45.0 | BULL_TREND_WEAK |
| W2 | 2025 Q2 | 1 | 100% | +0.029% | 0% | — | CALM_RANGE |
| W4 | 2025 Q4 | 3 | 33.3% | -0.060% | -0.112% | -29.0 | BULL_TREND_WEAK |
| W5 | 2026 Q1 | 13 | 46.2% | -0.069% | -0.375% | -3.7 | BULL_TREND_WEAK |

### Gauntlet y ensemble

- **Gauntlet individual:** 0 de 3 seeds que completaron lo superaron.
- **Ensemble (seeds 72012 + 72101):** Portfolio unificado = 0 trades únicos (colisiones de timestamp). Gauntlet ensemble no ejecutado por portfolio insuficiente.

### CVD — Component Value Dashboard

| Componente | Delta WR medio | Veredicto |
|---|---|---|
| Alpha_Trigger | +66.7pp | APORTA EDGE |
| HMM_Regime | +55.6pp | APORTA EDGE |
| OOD_Guard | +30.0pp | APORTA EDGE |
| Signal_Threshold | +21.3pp | APORTA EDGE |
| XGBoost_prob_cal | +0.0pp | NEUTRAL/MARGINAL |
| MetaLabeler_V2 | N/A | DESACTIVADO |
| LGBM | N/A | DESACTIVADO |

---

## Diagnóstico de Problemas Detectados en esta Run

### P1 — Inanición operativa: muy pocos trades por seed (causa raíz)

**Observado:** Solo seed 72101 supera 30 trades (35). Mediana de trades = 25 por seed. Ninguna seed supera el Gauntlet estadístico individual.

**Causas identificadas en logs:**

1. **Embargo dinámico 96H filtra el 96% de señales candidatas.** De 215 señales post-momentum en W1, solo 9 sobreviven el embargo. Con holdouts de ~2.400H (~3 meses), un embargo mínimo de 96H impide más de ~25 trades por ventana incluso con señal perfecta. Log: `[EMBARGO DINAMICO] Modo=DINAMICO(72-168H) | 215 candidatos → 9 señales`.

2. **Agente BULL desactivado por gate DSR < 0.20.** El régimen `1_BULL_TREND_WEAK` cubre el 47-60% del holdout en Q1/Q2 2025. Con el agente bull en CASH forzado, se pierde la mitad del espacio temporal operativo. Log: `[FIX-BULL-GATE-01] DSR_CPCV=0.1754 <= 0.2000 → bull DESACTIVADO. 1128 barras forzadas a CASH`.

3. **W3 no aparece en ninguna seed.** El WFB evalúa early-stop en W1 y W4. W3 solo se ejecuta en seeds que llegan a W4, y ninguna early-stopped completó W3 antes del corte.

**Impacto:** El ensemble no alcanza masa crítica estadística. El Gauntlet (mínimo DSR > 0 con n>=30) no puede ejecutarse correctamente con menos de 30 trades únicos por seed.

### P2 — Régimen RANGE sin match en HMM (bug de configuración latente)

**Observado:** Los regímenes `2_CALM_RANGE`, `2_VOLATILE_RANGE` etc. están en `hmm_allowed_regimes` del settings pero el HMM entrenado (n_components=5) nunca los genera. Confirmado en todas las ventanas analizadas.

**Log:** `[FIX-HMM-004] 4 etiquetas en hmm_allowed_regimes sin match en state_map: ['2_CALM_RANGE', '2_CALM_RANGE_B', '2_VOLATILE_RANGE', '2_VOLATILE_RANGE_B']`.

**Impacto:** 489 barras por ventana sin régimen claro van a umbral fallback (0.50). El agente RANGE opera sobre barras que el HMM no identifica correctamente. No crashea gracias al FIX-HMM-004.

**Acción:** Verificar si alguna ventana genera estados RANGE. Si no, limpiar `hmm_allowed_regimes` del settings. Implementar **fuera de runs activas**.

### P3 — Features SHAP con importancia cero histórica (3 features CRITICAL)

**Observado en SHAP-AUDIT-01 (47 ventanas históricas):**

- `fear_greed_normalized`: < 0.02 en **7 ventanas consecutivas**, media = 0.000
- `yield_curve_pct_1y`: < 0.02 en **7 ventanas consecutivas**, media = 0.000
- `Exchange_Supply_Pct`: < 0.02 en **5 ventanas consecutivas**, media = 0.000

**Log:** `[SHAP-AUDIT-01] CANDIDATO A ELIMINAR — ACCIÓN: Evaluar eliminación de sfi_macro_features`.

**Acción:** Fuera de runs activas, eliminar estas 3 features de `sfi_macro_features`/`sfi_onchain_features` en settings.yaml respetando `settings_restore_protection`.

### P4 — Drift PSI crítico en 10/11 features (Kelly penalizado -50%)

**Observado:** PSI > 0.25 en 10 de 11 features. Valores entre 3.18 y 6.17 — extremadamente altos. Log: `[V2-P3-DRIFT] KELLY PENALTY ACTIVA: max_position reducido al 50%`.

**Causa estructural:** Features con lag largo (`_milag500h`) o precios absolutos (BITO, ETH) tienen PSI alto cuando el mercado alcanza rangos nuevos en 2025. Solución: usar features más estacionarias (retornos, z-scores sobre ventanas cortas).

### P5 — Agente BEAR peor que azar en validación IS

**Observado:** `Brier_VAL=0.268 > Brier_naive=0.230`. Causa: solo 775 muestras IS en régimen bear (rolling 5Y). El calibrador Platt lo rescata parcialmente (0.268→0.219).

**Log:** `[CAL-DIAG-01] FAIL (XGB PEOR QUE RANDOM). WR_real=64.1% | avg_prob=0.432 | overconf=-0.209`.

---

## Cambios Propuestos

### 1. Cota Inferior Adaptativa de `min_child_weight` en Optuna
Evitar el Model Collapse adaptando la cota inferior y superior del espacio de búsqueda de Optuna según el número de muestras reales del régimen actual.
- **Archivo:** [train_xgboost_v2.py](file:///g:/Mi%20unidad/ia/luna_v2/luna/models/train_xgboost_v2.py)
- **Modificación:**
  - En la función `objective(self, trial)`, redefinir la cota inferior `_mcw_min` y la superior `_mcw_max`:
    ```python
    _n_train_agent = len(self.X)
    _mcw_min = sp.min_child_weight_min
    # Cota inferior adaptativa: si el dataset es pequeño, permitir min_child_weight más bajos
    _mcw_min_adaptive = max(2, min(_mcw_min, _n_train_agent // 100))
    _mcw_min = _mcw_min_adaptive
    ```
    Esto garantiza que si el dataset del régimen tiene pocas muestras (e.g. 500 filas), Optuna pueda seleccionar un `min_child_weight` de 5 o 10, permitiendo que el árbol crezca y genere varianza predictiva real en lugar de una probabilidad constante.

### 2. Implementar Mitigación de Sobreconfianza (Probability Capping) en el Kelly Sizer
Prevenir que las probabilidades sobre-optimistas (o constantes por colapso) forcen exposiciones desmedidas.
- **Archivo:** [kelly_sizer.py](file:///g:/Mi%20unidad/ia/luna_v2/luna/risk/kelly_sizer.py)
- **Modificación:**
  - En `compute_kelly(self, p_win)` y `size_signals_dynamic()`, truncar cualquier probabilidad de ganar que supere un umbral configurable `probability_cap = 0.62` (el cual limita el tamaño máximo de posición en torno al 8%-10% de capital, alineado con el WR real de las colas en OOS).

### 3. Evitar Aplanamiento por Out-of-Bounds en la Calibración
Robustecer la calibración en validación mediante modelos paramétricos continuos.
- **Archivo:** [train_xgboost_v2.py](file:///g:/Mi%20unidad/ia/luna_v2/luna/models/train_xgboost_v2.py)
- **Modificación:**
  - En la lógica de validación, si `IsotonicRegression` colapsa o produce una desviación estándar menor a `1e-4`, priorizar `TemperatureCalibrator` o `PlattCalibrator` como fallback de calibración continuo que garantice que la probabilidad calibrada mantenga pendiente monótona no nula.

### 4. Limpiar regímenes RANGE en HMM state_map (P2)
- **Archivo:** `config/settings.yaml`
- **Acción:** Verificar ventanas W2-W5 para confirmar si el HMM genera estados RANGE. Si ninguna lo hace, eliminar `2_CALM_RANGE`, `2_CALM_RANGE_B`, `2_VOLATILE_RANGE`, `2_VOLATILE_RANGE_B` de `hmm_allowed_regimes`. Implementar **fuera de runs activas** siguiendo `settings_restore_protection`.

### 5. Eliminar features SHAP con importancia cero histórica (P3)
- **Archivo:** `config/settings.yaml`
- **Acción:** Eliminar `fear_greed_normalized`, `yield_curve_pct_1y`, `Exchange_Supply_Pct` de `sfi_macro_features`/`sfi_onchain_features`. Implementar **fuera de runs activas**.

---

## Plan de Verificación

### Pruebas Unitarias y Estáticas
1. Correr el validador sintáctico en Python tras realizar las modificaciones en el código.
2. Ejecutar la auditoría estática con `python tools/diagnostics/audit_parametros_fijos.py`.

### Verificación del Pipeline
1. Tras completar la run actual del WFB, entrenar el agente `bull_long` para la Ventana W2 de la semilla 42 con la cota adaptativa de `min_child_weight`.
2. Verificar que el número de valores únicos en la predicción cruda OOS de la Ventana W2 aumente significativamente (evitando la concentración constante del 32% en un solo decimal).
