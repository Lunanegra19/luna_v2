# Arquitectura Institucional de Procesos (Luna V2)

Este documento expone en máxima profundidad la arquitectura modular "End-to-End" de Luna V2. Se detallan las agrupaciones lógicas, los artefactos generados, las defensas institucionales, las tecnologías subyacentes y el ciclo de vida exacto de la información a través de la tubería algorítmica.

## 1. Diagrama de Flujo Principal

El siguiente diagrama detalla la orquestación del sistema, desde la extracción de datos en fuentes externas, pasando por las barreras de seguridad, hasta la generación de los entregables finales para producción y C#.

```mermaid
flowchart TD
    %% ==========================================
    %% Estilos de Nodos
    %% ==========================================
    classDef main fill:#1e3a8a,stroke:#60a5fa,color:#fff,font-weight:bold,stroke-width:2px;
    classDef subproc fill:#0f172a,stroke:#475569,stroke-width:1px,color:#e2e8f0;
    classDef phase fill:#064e3b,stroke:#34d399,color:#fff,stroke-width:1px;
    classDef data fill:#78350f,stroke:#fcd34d,color:#fef3c7;
    classDef out fill:#831843,stroke:#f9a8d4,color:#fce7f3;

    %% Entradas
    EXT["Fuentes: Binance, FRED, Glassnode, Coinglass"]:::data
    CFG[("settings.yaml")]:::data
    
    %% Orquestación
    subgraph ORCHESTRATION ["1. Terminales de Orquestación"]
        WFB["run_wfb_orchestrator.py<br/>(Modo Backtesting Institucional)"]:::main
        PROD["train_production_model.py<br/>(Modo Despliegue Producción)"]:::main
        SYNC["sync_data_lake.py<br/>(Data Updater)"]:::main
        EXEC(("LunaPipelineExecutor<br/>Cerebro de Control")):::main
        
        SYNC -->|1º Ejecuta Fases 1 a 3| EXEC
        WFB -->|2º Ejecuta Fases 4 y 5| EXEC
        PROD -->|2º Ejecuta Fases 4 y 5| EXEC
    end

    %% Pipeline Secuencial
    subgraph PIPELINE ["2. Pipeline Algorítmico Institucional"]
        direction TB
        
        subgraph F_PRE ["Fase 1: Extracción y Defensa (Vía sync_data_lake)"]
            direction TB
            F0["pre_flight_check.py"]:::subproc
            F1["fetch_*.py (8 Orígenes en paralelo)"]:::subproc
            F2["data_integrity_check.py"]:::subproc
            F3["reconcile_external_data.py"]:::subproc
            
            F0 --> F1 --> F2 --> F3
        end
        
        subgraph F_DATA ["Fase 2 y 3: Data Mining & Ingeniería Base"]
            direction TB
            D1["feature_pipeline.py (Pre-SFI)"]:::subproc
            D2["build_dataset.py (AI Mining)"]:::subproc
            D3["feature_selection_e.py (SFI)"]:::subproc
            D4["feature_pipeline.py (Post-SFI)"]:::subproc
            
            D1 --> D2 --> D3 --> D4
        end
        
        subgraph F_MODELS ["Fase 4: Ensemble Predictivo"]
            direction TB
            M1["hmm_regime.py (Identifica Regímenes HMM)"]:::subproc
            M2["train_xgboost_v2.py (Champion Model)"]:::subproc
            M3["ensemble_lgbm.py (Multi-Agent LGBM)"]:::subproc
            M4["ood_guard.py (Filtro de Anomalías)"]:::subproc
            M5["train_autoencoder.py (Reconstrucción)"]:::subproc
            M6["train_metalabeler_v2.py (Filtro L/S)"]:::subproc
            M7["calibrate_probabilities.py (Isotonic/Platt)"]:::subproc
            
            M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> M7
        end
        
        subgraph F_OOS ["Fase 5: Inferencia OOS y Validación"]
            direction TB
            V1["predict_oos.py (Inferencia Causal OOS)"]:::subproc
            V2["run_statistical_validation.py (Gauntlet)"]:::subproc
            
            V1 --> V2
        end

        F_PRE -.-> F_DATA
        F_DATA --> F_MODELS
        F_MODELS --> F_OOS
    end

    %% Entregables
    OUT["3. Entregables Finales:<br/>Production Bundle (C# cBot), Señales y Reportes"]:::out

    %% Conexiones Globales
    EXT -.-> F1
    CFG -.-> WFB
    CFG -.-> PROD
    CFG -.-> SYNC
    
    EXEC ==>|Controla Fases 1 a 5 según Terminal| D1
    V2 ==> OUT
```

## 2. Descripción de las Fases del Pipeline

El pipeline se encuentra dividido en una secuencia lógica para mantener el estricto control de datos, modelos y estabilidad institucional.

### Fase 0 y 1: Extracción y Defensa Institucional
Esta es la barrera de seguridad de Luna.
- Los **Fetchers** se ejecutan en paralelo para extraer miles de métricas desde Binance, Coinglass, FRED y plataformas On-chain.
- Los **Validadores Institucionales** (`data_integrity_check.py`, `reconcile_external_data.py` y `pre_flight_check.py`) garantizan que no existan NaNs críticos, gaps temporales o corrupción en las APIs.

### Fase 2 y 3: Data Mining & Ingeniería Base
Se encarga de generar los datasets y extraer la causalidad profunda.
- **Flujo Corregido:** Primero se ejecuta `feature_pipeline.py` para sentar las bases. Luego, el motor `build_dataset.py` (AI Mining) utiliza esa base para extraer las reglas Alpha (Bayesianas) y Tribus K-Means.
- Se cierra con **SFI (Smart Feature Isolation)** para depurar colinealidades severas antes de inyectar `features_train.parquet` a los modelos.

### Fase 4: Ensemble Predictivo
El núcleo de machine learning de Luna V2 representa una arquitectura en cascada:
- **HMM:** Identifica el régimen de mercado general (Bull, Bear, Calm, Crash).
- **XGBoost & LightGBM:** Champion Model direccional e inteligencia de conjunto (Ensemble Multi-Agente).
- **OOD Guard & AutoEncoder:** Descartan inferencias fuera de distribución (Out-of-Distribution) midiendo la anomalía estructural.
- **MetaLabeler:** Filtro final para reducir dramáticamente los falsos positivos en posiciones Long y Short.
- **Calibrador:** Ajusta las probabilidades generadas para reflejar una confianza estadística real (Isotonic / Platt).

### Fase 5 y 6: Inferencia OOS y Validación Final
Ejecuta las pruebas definitivas del conjunto de modelos en datos puros Out-of-Sample (OOS).
- **Gauntlet Estadístico:** Exige reglas institucionales muy estrictas para aprobar el modelo y enviarlo a la siguiente fase (WinRate > 45%, Riesgo/Recompensa > 1.2, Máx. Drawdown < 15%). Genera el Tearsheet final y el `oos_trades.parquet`.

## 3. Características Técnicas de esta Arquitectura

1. **Aislamiento Total por Semilla (Crash Resilience):** El `LunaPipelineExecutor` utiliza la caché (`wfb_cache`) para "hidratar" (cargar) el estado de una ventana temporal, ejecutar los subprocesos de manera confinada y luego "deshidratar" (guardar) los nuevos modelos/datasets.
2. **Dependencias Estrictamente Causales:** Todo el pipeline evita el *Look-Ahead Bias*. Los modelos nunca ven los datos de validación durante su ajuste.
3. **Múltiples Capas de Defensa (Risk Management):** Para que una señal llegue a cTrader (C#), debe haber sobrevivido a la cascada de filtros institucionales y estadísticos.
4. **Validación Institucional Automatizada:** Evaluaciones duras antes de generar el *Production Bundle* garantizan viabilidad matemática en mercado real.

## 4. Referencia de Comandos y Banderas (CLI)

Para interactuar con el pipeline existen tres puntos de entrada principales, separando estrictamente la gestión de datos del modelado:

### A. Para Actualización de Datos (Data Lake y Features)
`python scripts/sync_data_lake.py`
* `--skip-fetch`: Omite la descarga de datos si solo quieres regenerar las features sintéticas con datos cacheados.
* `--skip-sfi`: Salta el subproceso de aislamiento inteligente (SFI). Ahorra de 8 a 12 horas.
* `--skip-mining`: Evita recalcular las reglas bayesianas de AI Mining.

### B. Para Backtesting Institucional (WFB - Walk-Forward)
`python scripts/run_wfb_orchestrator.py`
*(Asume que `sync_data_lake.py` ya fue ejecutado)*
* `--seeds 42 777`: Define las semillas para el ensemble.
* `--resume`: Salta automáticamente las ventanas temporales (W1, W2...) que ya fueron completadas y cacheadas.
* `--force-resume`: Fuerza la reanudación incluso en el primer intento de la primera semilla.
* `--nocache`: Botón de seguridad. Elimina por completo la carpeta `data/wfb_cache/` para asegurar un test desde cero absoluto sin usar datos cacheados.
* `--smoke-test`: Modo ultrarrápido de depuración que inyecta `LUNA_SMOKE_TEST=1` para limitar iteraciones y probar causalidad.

### C. Para Entrenamiento de Producción (Deployment)
`python scripts/train_production_model.py`
*(Automáticamente invoca a `sync_data_lake.py --skip-sfi --skip-mining` por seguridad)*
* `--mode prod`: Usa el dataset completo hasta el día de hoy (evitando el truncamiento de development).
* `--skip-sync`: Omite la actualización automática del Data Lake al inicio del script.
* `--nocache`: Botón de seguridad. Elimina la carpeta `data/models/` para forzar a que todos los modelos se entrenen de forma limpia y desde cero.
* `--skip-hmm`: Omite el re-entrenamiento del régimen oculto de Markov.
* `--skip-validation`: Corta la ejecución antes del Gauntlet Estadístico si solo quieres exportar los binarios puros.
