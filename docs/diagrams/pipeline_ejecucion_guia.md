# Manual de Ejecución del Pipeline Cuantitativo de Producción — Luna V2

Este documento sirve como la guía maestra e institucional de ejecución del pipeline completo de **Luna V2**. El sistema está diseñado bajo el paradigma de separación de responsabilidades entre la **Ingeniería de Datos (Data Lake)**, la **Ingeniería de Modelos (Walk-Forward Backtesting & Ensamble)** y el **Motor de Ejecución en Vivo (Live Daemon)**.

---

## 1. Arquitectura de Carpetas y Organización del Proyecto

De acuerdo con las directrices institucionales de la **`RULE[estructuracarpetas.md]`**, el repositorio de Luna V2 está organizado rigurosamente bajo la siguiente estructura modular:

```text
📂 luna_v2/
├── 📂 luna/                    # Core Package: Lógica matemática, modelos, features y utilidades del pipeline.
│   ├── 📂 data/                # Captura de datos, fetchers específicos y transformaciones raw.
│   ├── 📂 features/            # Pipeline de features base y final (Smart Feature Isolation - SFI).
│   ├── 📂 models/              # Lógica de Modelos (HMM, XGBoost, LGBM, Autoencoders, Platt Calibrator).
│   └── 📂 live/                # Componentes en vivo (Position Sizer, Risk Monitor, OKX Broker).
├── 📂 scripts/                 # Orquestadores: Ejecución principal que inicia fases críticas del sistema.
├── 📂 config/                  # Configuración: settings.yaml y settings.py (carga dinámica sin números mágicos).
├── 📂 data/                    # Data Lake (Local/No Git): Base de datos persistente, descargas, modelos (.pt, .joblib).
├── 📂 logs/                    # Bitácoras de operaciones estructuradas producidas por Loguru.
├── 📂 tests/                   # Validación: Pruebas unitarias e integración de la librería core.
└── 📂 tools/                   # Mantenimiento y Diagnóstico.
    ├── 📂 diagnostics/         # Scripts de análisis y forense (audit_parquet, find_optimal_seeds, etc.).
    ├── 📂 refactor/            # Scripts de transformación y refactorización histórica de código.
    └── 📂 dumps/               # Volcados en crudo de texto para auditorías rápidas.
```

---

## 2. Diagrama de Flujo del Pipeline Completo

El siguiente diagrama ilustra el flujo cronológico y las dependencias de ejecución desde la sincronización inicial de los datos hasta el envío de órdenes a mercado en tiempo real:

```mermaid
flowchart TD
    subgraph Fase 1: Data Ingestion & Engineering
        A[scripts/sync_data_lake.py] -->|1. Pre-Flight Check| A1[scripts/pre_flight_check.py]
        A -->|2. Fetchers en Paralelo| A2[fetch_ohlcv.py, fetch_macro.py...]
        A -->|3. Data Integrity Check| A3[scripts/data_integrity_check.py]
        A -->|4. Anchor Temporal| A4[scripts/reconcile_external_data.py]
        A -->|5. Feature Isolation| A5[Smart Feature Isolation - SFI]
    end

    subgraph Fase 2: WFB & Multi-Seed Discovery
        A5 -->|features_train.parquet| B[scripts/run_wfb_orchestrator.py]
        B -->|Orquesta en Paralelo| B1[scripts/wfb_worker.py]
        B1 -->|Early Pruning| B2{¿Seed Competente?}
        B2 -->|No: Abortar Semilla| B3[Descarte Inmediato]
        B2 -->|Sí: Guardar Reporte| B4[data/reports/wfb/seed_*.csv]
        B4 --> B5[scripts/evaluate_ensemble_wfb.py]
    end

    subgraph Fase 3: Combinatorial Seed Optimization
        B5 -->|master_ensemble_probs.parquet| C[tools/diagnostics/find_optimal_seeds.py]
        C -->|Teorema Krogh & Vedelsby| C1[Evaluación de Diversidad vs Precisión]
        C1 -->|Consensus & Soft Embargo Sim| C2[Maximización de Calmar Ratio]
        C2 -->|Escribe a settings.yaml| D[config/settings.yaml: active_seeds]
    end

    subgraph Fase 4: Ensemble Production Training
        D --> E[scripts/train_production_ensemble.py]
        E -->|Entrenamiento Multisemilla| E1[LunaPipelineExecutor PROD]
        E1 -->|Fase 4 & 5| E2[HMM + XGBoost + LGBM + Autoencoder + Platt]
        E2 -->|Aislamiento Térmico| E3[data/models/prod/seed{seed}/]
        E2 -->|Manifiesto de Ensamble| E4[data/models/prod/ensemble_metadata.json]
    end

    subgraph Fase 5: Live Execution Daemon
        E3 & E4 --> F[scripts/run_live_trader.py]
        F -->|Heartbeat Loop| F1[Data Ingest & Feature Pipe Realtime]
        F1 -->|Quorum Consensus| F2[Evaluación de 5 Semillas Consecuentes]
        F2 -->|Kelly & Position Sizing| F3[Risk Monitor & Fallbacks]
        F3 -->|Ejecución Segura| F4[OKX Broker Connector API]
    end
```

---

## 3. Tabla Cronológica de Ejecución

| Orden | Script de Entrada (`.py`) | Rol del Componente | Entrada Crítica | Salida Crítica | Frecuencia Sugerida |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | [`scripts/sync_data_lake.py`](file:///g:/Mi%20unidad/ia/luna_v2/scripts/sync_data_lake.py) | Ingesta, Integridad y Minería de Features (SFI) | APIs Externas (Binance, FRED) | `data/features/features_train.parquet` | Diaria (Incremental) |
| **2** | [`scripts/run_wfb_orchestrator.py`](file:///g:/Mi%20unidad/ia/luna_v2/scripts/run_wfb_orchestrator.py) | Walk-Forward Backtesting Multi-Semilla y Pruning | Dataset final + `settings.yaml` | Reportes OOS en `data/reports/wfb/` | Semanal / Quincenal |
| **2.5** | [`scripts/evaluate_ensemble_wfb.py`](file:///g:/Mi%20unidad/ia/luna_v2/scripts/evaluate_ensemble_wfb.py) | Consolidación y Evaluación de Trades del Ensamble WFB | Reportes del WFB (`oos_trades_*.parquet`) | `master_ensemble_probs.parquet`, `wfb_ensemble_tearsheet_summary.md` | Al finalizar Fase 2 |
| **3** | [`tools/diagnostics/find_optimal_seeds.py`](file:///g:/Mi%20unidad/ia/luna_v2/tools/diagnostics/find_optimal_seeds.py) | Selección Combinatoria de Semillas Campeonas | Reportes del WFB | Selección de semillas óptimas en `settings.yaml` | Al finalizar Fase 2.5 |
| **4** | [`scripts/train_production_ensemble.py`](file:///g:/Mi%20unidad/ia/luna_v2/scripts/train_production_ensemble.py) | Entrenamiento y Serialización del Ensamble en $T_{now}$ | Semillas Campeonas + Dataset | `data/models/prod/` (Pesos & Metadatos) | Semanal / Mensual |
| **5** | [`scripts/run_live_trader.py`](file:///g:/Mi%20unidad/ia/luna_v2/scripts/run_live_trader.py) | Daemon de Trading en Vivo y Gestión de Riesgo | Modelos de Producción + `.env` | Operaciones en OKX + Alertas Telegram | Continuo (24/7 de forma inmortal) |

---

## 4. Descripción Detallada de las Fases del Pipeline

### Fase 1: Ingesta & Sincronización del Data Lake
* **Script Principal:** [`scripts/sync_data_lake.py`](file:///g:/Mi%20unidad/ia/luna_v2/scripts/sync_data_lake.py)
* **Contexto Técnico:** Esta fase se encarga de consolidar toda la información externa antes de realizar cualquier modelado de aprendizaje automático. Garantiza la calidad de los datos previniendo problemas comunes en finanzas cuantitativas como el sesgo de supervivencia y el sesgo de anticipación (look-ahead bias).
* **Flujo Secuencial Interno:**
  1. **Pre-Flight Check (`scripts/pre_flight_check.py`):** Valida la arquitectura del sistema, carpetas críticas y variables de entorno.
  2. **Fetchers en Paralelo:** Lanza múltiples hilos para capturar datos OHLCV, datos macroeconómicos (FRED), datos onchain, tasas de derivados, altcoins y mempools en paralelo.
  3. **Data Integrity Check (`scripts/data_integrity_check.py`):** Detecta corrupción, NaNs y calcula estadísticas de gaps temporales.
  4. **Reconcile External Data (`scripts/reconcile_external_data.py`):** Ancla temporalmente series macro y onchain a la frecuencia horaria de las velas OHLCV para evitar fugas de información.
  5. **Feature Pipeline (SFI):** Genera la suite base de 366 variables y aplica **Smart Feature Isolation (SFI)** para seleccionar las mejores subfamilias (TIPO-1 Mercado, TIPO-2 Macro, TIPO-3 AI Mining).
* **Comandos de Uso:**
  ```bash
  # Ejecución estándar (Fetch completo e ingeniería de variables)
  python scripts/sync_data_lake.py

  # Ejecución incremental omitiendo fetchers (Útil para pruebas rápidas de variables)
  python scripts/sync_data_lake.py --skip-fetch
  ```

---

### Fase 2: Walk-Forward Backtesting (WFB) & Multi-Seed Discovery
* **Script Principal:** [`scripts/run_wfb_orchestrator.py`](file:///g:/Mi%20unidad/ia/luna_v2/scripts/run_wfb_orchestrator.py)
* **Contexto Técnico:** En esta fase se realiza la validación robusta fuera de muestra de múltiples semillas candidatas. Luna V2 rechaza la optimización estática sobre datos históricos. En su lugar, utiliza un esquema de **Walk-Forward dinámico con ventanas superpuestas** (`W1` a `W5` definidas en `settings.yaml`).
* **Mecanismos Clave:**
  * **Spawn del Worker (`scripts/wfb_worker.py`):** El orquestador ejecuta los workers de forma aislada para cada semilla.
  * **Pruning Temprano (Early Stopping):** Utiliza el parámetro dinámico `wfb.prune_threshold` de `settings.yaml`. El orquestador calcula un upper-bound optimista del score de la semilla basados en las ventanas completadas. Si la semilla matemáticamente ya no puede batir al benchmark establecido (ej. Seed 777), se aborta su ejecución inmediatamente para ahorrar recursos y tiempo de cómputo.
* **Comandos de Uso:**
  ```bash
  # Ejecución estándar del WFB para la suite de semillas candidatas
  python scripts/run_wfb_orchestrator.py
  ```

* **Script de Consolidación Intermedia (Fase 2.5):** [`scripts/evaluate_ensemble_wfb.py`](file:///g:/Mi%20unidad/ia/luna_v2/scripts/evaluate_ensemble_wfb.py)
* **Contexto Técnico:** Este script actúa como el puente de agregación evaluadora entre los resultados de backtesting walk-forward crudos individuales y la posterior selección inteligente combinatoria del ensamble.
* **Mecanismos Clave:**
  * **Agregación por Soft Voting:** Toma los parquets de probabilidades out-of-sample (`oos_raw_probs_*.parquet`) de todas las semillas y genera el archivo maestro consolidado `master_ensemble_probs.parquet` por promedio simple de probabilidades.
  * **Carga Dinámica del Consensus Gate:** Lee de forma canónica `wfb.ensemble_consensus_threshold` de `settings.yaml` para filtrar operaciones del ensamble sobre el historial consolidado.
  * **Consensus-Soft Embargo:** Evalúa la atenuación temporal del embargo cuando coinciden $\ge 4$ semillas concurrentes según los parámetros en `settings.yaml`.
  * **Generación del Tearsheet Summary:** Compila métricas agregadas y desgloses individuales, guardando el reporte Markdown `wfb_ensemble_tearsheet_summary.md` en `data/reports/wfb/`.
* **Comandos de Uso:**
  ```bash
  # Consolidar los resultados de todas las semillas completadas en el WFB
  python scripts/evaluate_ensemble_wfb.py
  ```

---

### Fase 3: Combinatorial Seed Optimization & Diversity Analysis
* **Script Principal:** [`tools/diagnostics/find_optimal_seeds.py`](file:///g:/Mi%20unidad/ia/luna_v2/tools/diagnostics/find_optimal_seeds.py)
* **Contexto Técnico:** Frecuentemente en el trading cuantitativo se asume que solo se deben usar semillas individuales con retornos superlativos. Sin embargo, Luna V2 implementa el **Teorema de Krogh & Vedelsby (Dilema Diversidad-Precisión)**. Este teorema demuestra matemáticamente que un ensamble compuesto por modelos diversificados (incluso aquellos con un Sharpe menor individualmente) produce un error cuadrático menor y un Calmar consolidado superior debido a la cancelación de correlaciones de error.
* **Mecanismos Clave:**
  * **Simulación de Reglas de Consenso (Consensus Gate):** Simula cómo interactúan los modelos en conjunto (Votación Soft / Hard).
  * **Consensus-Soft Embargo:** Aplica penalizaciones y embargos temporales dinámicos (ej. 24 horas) cuando existe alta coincidencia de señales en regímenes de mercado específicos.
  * **Métricas Premium Completas (RULE[windowstats.md]):** De acuerdo con las especificaciones institucionales, el script calcula y muestra:
    * **Ganancias/Perdidas:** Absolutas, relativas y compuestas.
    * **Max Drawdown (Max DD):** Profundidad y duración de la racha de pérdidas en el ensamble consolidado.
    * **Optimal Kelly Sizer:** Fracción matemática de Kelly optimizada y calibrada (usualmente Half-Kelly limitado a un rango de prudencia del $[0.0, 0.40]$).
    * **Apalancamiento Óptimo:** Proyección de apalancamiento seguro de $x5$ a $x10$ ajustado a la volatilidad histórica.
    * **Calmar Ratio & DSR (Deflated Sharpe Ratio):** Significancia estadística que descuenta el sesgo de pruebas múltiples.
* **Comandos de Uso:**
  ```bash
  # Optimizar la selección combinatoria (Ej: Escoger las 5 mejores semillas a partir de un pool de 10)
  python tools/diagnostics/find_optimal_seeds.py --pool-size 10 --select-size 5 --leverage 10
  ```

---

### Fase 4: Multi-Seed Production Ensemble Training
* **Script Principal:** [`scripts/train_production_ensemble.py`](file:///g:/Mi%20unidad/ia/luna_v2/scripts/train_production_ensemble.py)
* **Contexto Técnico:** Tras identificar las 5 semillas campeonas (actualmente las semillas canónicas son `42, 100, 777, 1337, 2025` especificadas en la sección `wfb.active_seeds` de `config/settings.yaml`), esta fase entrena los modelos en todo el conjunto de datos disponible hasta la fecha actual ($T_{now}$) para dejarlos listos para la inferencia en producción en tiempo real.
* **Mecanismos Clave:**
  * **Aislamiento Térmico Absoluto:** Para evitar la fuga de datos o contaminación cruzada entre semillas, el script limpia secuencialmente el directorio de entrenamiento local `data/models/` y exporta los artefactos de cada semilla a directorios aislados `data/models/prod/seed{seed}/`.
  * **Entrenamiento en Cascada:**
    * **AutoEncoder (Bottleneck Dimensional):** Comprime variables continuas complejas de 366 a una representación densa de 32 dimensiones.
    * **Regime Router (HMM + GMM Fallback):** Clasifica en tiempo real los 4 regímenes macro de mercado.
    * **Classifiers (XGBoost CPCV + LGBM):** Estima probabilidades direccionales de subida del activo.
    * **Platt Calibrator:** Calibra las probabilidades crudas a probabilidades reales de ocurrencia.
    * **MetaLabeler V2:** Estima la probabilidad de que la operación supere los costos transaccionales y el slippage (limbral de fiabilidad).
  * **Manifiesto Consolidado (`ensemble_metadata.json`):** Escribe un archivo maestro que registra los umbrales de decisión validados, rutas de pesos del autoencoder, parámetros del modelo y la configuración del ensamble.
* **Comandos de Uso:**
  ```bash
  # Entrenamiento estándar de producción (Multisemilla)
  python scripts/train_production_ensemble.py --mode prod

  # Validación de estructura (Dry-Run) sin consumir GPU/CPU (Genera mocks estructurales)
  python scripts/train_production_ensemble.py --dry-run
  ```

---

### Fase 5: Real-Time Live Inference & Daemon Execution
* **Script Principal:** [`scripts/run_live_trader.py`](file:///g:/Mi%20unidad/ia/luna_v2/scripts/run_live_trader.py)
* **Contexto Técnico:** El motor de trading inmortal. Este script opera como un Daemon 24/7 en el servidor virtual (VPS). Carga el ensamble consolidado y ejecuta el ciclo de control cada hora sincronizadamente con el cierre de vela.
* **El Ciclo Inmortal de Control:**
  ```text
  [HEARTBEAT] ──> [RECONCILE] ──> [RISK MONITOR] ──> [ENSEMBLE INFERENCE] ──> [POSITION SIZER] ──> [OKX EXECUTION]
  ```
  1. **Heartbeat:** Confirmación de salud de las APIs externas, telemetría y base de datos local.
  2. **Reconcile:** Reconciliación de balances (OKX y base de datos) realizada por `BalanceReconciler`.
  3. **Risk Monitor:** El módulo `RiskMonitor` supervisa límites de drawdown diario, drawdown máximo global y apalancamiento. Cuenta con **resiliencia total a fallos de BD**, de modo que si la base de datos PostgreSQL local se cae, realiza un fallback crítico inmediato a telemetría en memoria y sigue operando de forma ultra-segura en lugar de colapsar.
  4. **Ensemble Inference:** Ejecuta la inferencia paralela de las 5 semillas a través de `LunaEnsembleLiveInference` sobre las variables generadas en vivo por `FeaturePipeline`. Si existe falta de archivos o firmas no disponibles en algún instante, activa un **fallback crítico** de inferencia numérica pura para garantizar la continuidad operativa.
  5. **Quorum Consensus & Embargo:** Realiza votación por consenso (Soft Voting). Si $\ge 3$ semillas coinciden, se autoriza la señal. Si se activan 4 o más semillas simultáneamente, evalúa si es necesario aplicar el **Consensus-Soft Embargo** temporal basado en la configuración de `wfb.soft_embargo_hours`.
  6. **Position Sizer:** Calcula la cantidad exacta de contratos perpetuos a abrir en OKX aplicando la fracción Half-Kelly calculada dinámicamente y limitada por el perfil de apalancamiento institucional.
  7. **OKX Execution:** Transmite la orden de forma segura a través de `OKXBrokerConnector` con límites estrictos de slippage y comisiones de mercado.
* **Comandos de Uso:**
  ```bash
  # Ejecución de prueba controlada en vivo (Modo Demo, una sola iteración - Dry run en vivo)
  python scripts/run_live_trader.py --once --demo

  # Ejecución en vivo de producción real (Daemon inmortal)
  python scripts/run_live_trader.py
  ```

---

## 5. Directrices Operativas Críticas (SOP)

Al operar e interactuar con el pipeline de Luna V2, es obligatorio seguir de forma estricta los siguientes estándares:

### A. Política de Lanzamiento de Runs (`RULE[inciorun.md]`)
Al iniciar una nueva run (ya sea para probar una nueva semilla experimental o para re-entrenar el ensamble):
1. **Verificar procesos zombis:** Ejecuta herramientas de sistema para asegurarte de que no existan hilos residuales o procesos zombis (`wfb_worker.py` o `run_live_trader.py`) ocupando puertos, sockets de base de datos o memoria de GPU.
2. **Lanzar la run:** Ejecuta el script correspondiente redirigiendo logs adecuadamente si es necesario.
3. **Validación de los 30 segundos:** Monitoriza ininterrumpidamente los logs generados durante los **primeros 30 segundos** de ejecución para confirmar la correcta inicialización de tensores PyTorch, la carga sin errores de `settings.yaml` y la ausencia de excepciones silenciosas.

### B. Vinculación a Settings y Evitación de Números Mágicos (`RULE[settingsyfallvack.md]`)
* **Prohibición de Valores Hardcodeados:** Absolutamente todos los umbrales de decisión (AUC mínimos, Brier targets, periodos de embargo, pesos del meta-labeler) deben estar vinculados dinámicamente al archivo canónico `config/settings.yaml`. 
* **Documentación en Parámetros Fijos:** Si por motivos estrictamente justificables o restricciones matemáticas de terceros se introduce un valor fijo (ej. la constante de decaimiento en la calibración Platt), esta debe documentarse exhaustivamente con su trasfondo teórico en el manual de [docs/parametros_fijos.md](file:///g:/Mi%20unidad/ia/luna_v2/docs/parametros_fijos.md).

### C. Trazabilidad mediante Logging Escrutable (`RULE[fixbugsprints.md]`)
* El pipeline está diseñado para operar de forma desasistida. Por ello, **cualquier corrección de bugs, parches aplicados o nuevas implementaciones deben incluir prints estructurados y llamadas a `logger.info()` / `logger.warning()` / `logger.success()`**. 
* Esto asegura que durante auditorías forenses posteriores de los archivos `.log` en `/logs/` sea posible rastrear exactamente el comportamiento ante caídas de red, fallos de calibración o activaciones de reglas de contingencia.
