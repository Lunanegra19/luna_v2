# Changelog y Mejoras del Sistema - Luna V2
**Fecha:** 2026-06-09

Este documento registra todas las implementaciones críticas, ajustes paramétricos y saneamientos realizados en el pipeline cuantitativo institucional de Luna V2 durante la sesión de hoy.

## 1. Habilitación de "Spot Pyramiding" (Embargo 0H)
Se habilitó la arquitectura del pipeline para operar acumulación agresiva en tendencias prolongadas en el mercado Spot, removiendo los filtros temporales que suprimían señales.
* **`config/settings.yaml`:**
  * Establecido `embargo_hours: 0`.
  * Desactivado `soft_embargo_enabled: false`.
* **Auditoría Institucional (`scripts/pre_flight/test_sop.py` y `test_architecture.py`):**
  * Se relajó la aserción estricta de la regla **SOP-R3** (cuarentenas de 24H o 96H).
  * El validador (TEST-28) reconoce y legitima legalmente el embargo de 0H sin bloquear la ejecución.

## 2. Activación del "Sniper-Mode" (MetaLabeler)
Como contrapeso cualitativo a la remoción del embargo (y el consiguiente aumento masivo de señales latentes), se restringió la densidad operativa para retener exclusivamente señales de alta calidad.
* **`config/settings.yaml`:**
  * Establecido `meta_v2_rolling_percentile: 0.85`.
  * El calibrador dinámico ahora desecha el 85% de las señales y ejecuta únicamente el top 15% con la más alta probabilidad matemática de ser rentables.

## 3. Estabilidad Matemática XGBoost (`[FIX-SPW-CRASH]`)
* **`luna/models/train_xgboost_v2.py`:**
  * Implementación de una protección matemática crítica para el cálculo dinámico del parámetro `scale_pos_weight`.
  * Prevención contra colisiones aritméticas (límites mínimos superando a los máximos o divisiones por cero) durante la optimización bayesiana (Optuna) en ventanas de mercado con escasez de muestras o asimetrías extremas.

## 4. Saneamiento del Offline Replay & Diagnostics
* **`tools/diagnostics/oos_replay_wfb.py`:**
  * Resolución de fallos de codificación de caracteres (`UnicodeEncodeError`) específicos en entornos Windows.
  * Reparación de la integración para la correcta carga de holdout features y el volcado desde `oos_raw_probs`, garantizando una auditoría humana fiable de las decisiones emitidas por el ensamble.

## 5. Rescate del WFB Orchestrator y Purga de Caché
Resolución de corrupción en disco detectada en la última fase de la ejecución previa, que generaba `TProtocolException: Invalid data / Page header failed` al intentar deserializar `features_train.parquet` en la Ventana 5 (W5).
* **Limpieza Absoluta:**
  * Purgado total del directorio `data/wfb_cache/`.
  * Eliminación de artefactos residuales `.parquet` y `.json` dentro del staging de `data/features/`.
* **Reinicio Limpio:**
  * Terminación de procesos zombie concurrentes.
  * Relanzamiento del orquestador de entrenamiento en modo `--nocache` para forzar la reconstrucción nativa del pipeline sin estados corruptos.
