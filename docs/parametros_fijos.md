# Registro Oficial de Parámetros Fijos y Dinámicos

Este documento registra los parámetros estructurales críticos del pipeline y las unificaciones automáticas realizadas en `config/settings.py` para asegurar que todo el pipeline utiliza parámetros consistentes y prevenir divergencias silenciosas.

## Unificaciones Automáticas (`config/settings.py::_unify_parameters`)

Para cumplir con la directiva estricta de **No-Fallback Silencioso** y asegurar la coherencia causal en todas las etapas, las siguientes familias de parámetros se resuelven y vinculan automáticamente en tiempo de ejecución:

| Familia de Parámetros | Nodos Vinculados (settings.yaml) | Resolución Matemática | Riesgo de Descorrelación |
|-----------------------|----------------------------------|-----------------------|--------------------------|
| **1. Ensayos HPO** | `xgboost.optuna_trials`<br>`stat.n_trials_total` | `stat.n_trials_total = xgboost.optuna_trials` | Previene cálculos erróneos del Deflated Sharpe Ratio (DSR) por usar un número diferente de ensayos en el validador estadístico. |
| **2. Barreras Temporales** | `xgboost.vertical_barrier_hours`<br>`sop.purge_hours` | `purge_hours = max(purge_hours, dynamic_horizon_max_h)` | **Leakage OOS**. Evita fugas de datos al asegurar que la purga de features siempre envuelva el tiempo de maduración de la barrera TBM. |
| **3. Bloque de Simulación** | `stat.mc_block_size_hours`<br>`sop.purge_hours` | `mc_block_size_hours = max(mc_block_size_hours, purge_hours)` | Bloques Monte Carlo (CSCV) deben incluir trayectorias completas de trades, no fractales. |
| **4. Embargo Temporal (SOP R3)** | `sop.embargo_hours`<br>`temporal_splits.embargo_hours` | `embargo_hours = max(embargo_hours, dyn_max_h)` *(si NO soft_embargo)* | Cumplimiento del Guardián Causal. Previene fugas temporales al forzar embargo ≥ 1x horizonte máximo. |
| **5. Umbrales DSR / PBO (SOP R5)** | `gauntlet.min_dsr`<br>`stat.min_dsr`<br>`gauntlet.max_pbo`<br>`stat.max_pbo` | `stat.min_dsr = gauntlet.min_dsr`<br>`stat.max_pbo = gauntlet.max_pbo` | **Gate Bypass**. Evitaba que el validador estadístico usara `min_dsr: 0.2` en lugar del oficial Bailey & LdP `0.75` definido en el Gauntlet. |
| **6. Gestión de Riesgo (SOP R17)** | `position_sizer.kelly_fraction`<br>`kelly_sizer.kelly_fraction` | `kelly_sizer.kelly_fraction = position_sizer.kelly_fraction` | **Ruin Risk**. Unifica el uso de *Fractional Kelly* al valor rector (Quarter Kelly = 0.25). Corrigió el uso paralelo de `0.5` en uno de los módulos. |
| **7. Costos Transaccionales (SOP R6)** | `costs.round_trip_pct`<br>`sop.cost_pct` | `sop.cost_pct = costs.round_trip_pct / 100.0` | **Optimismo de Retorno**. Traduce la configuración porcentual amigable (0.25%) al factor decimal (0.0025) exigido por todos los subsistemas (incluido MetaLabeler y PredictOOS). |
| **8. Brier Score Gates** | `stat.xgb_brier_hard_stop`<br>`stat.xgb_brier_warn`<br>`stat.brier_margin_range` | `brier_margin_range = hard_stop - warn` | Consistencia de la envolvente de alerta temprana de calibración. |
| **9. Decaimiento Ponderado** | `xgboost.weight_decay_alpha`<br>`metalabeler.weight_decay_alpha` | `metalabeler.weight_decay_alpha = xgb.weight_decay_alpha` | Asegura que toda la capa de inteligencia base (XGB + Meta) se deprecie temporalmente al mismo ritmo. |
| **10. Límites HPO Tree-based** | `xgboost.optuna_search_space`<br>`lightgbm.optuna_search_space` | Sincronización de `learning_rate` y `n_estimators` entre modelos base | Previene sobre-explotación del conjunto de búsqueda en uno de los estimadores que podría inducir sesgos de validación en LGBM. |

## Auditoría Continua

Cualquier nuevo parámetro o "Magic Number" descubierto durante auditorías posteriores debe ser:
1. Registrado en `config/settings.yaml`.
2. Extirpado del código estático mediante aserción `getattr(cfg, ...)`.
3. Sincronizado matemáticamente en `config/settings.py::_unify_parameters` si comparte dominio semántico con otros parámetros existentes.

---

## Parámetros de Mejoras Matemáticas V3 (2026-06-18)

Registrados tras la auditoría de implementación de Hipótesis A, B y C. Todos leen desde `settings.yaml` con fallback WARNING (no silencioso). Ninguno es un parámetro de Gauntlet, riesgo o embargo, por lo que según la política No-Fallback se permite fallback suave con trazabilidad obligatoria.

| ID | Parámetro (`settings.yaml`) | Valor Institucional | Sección YAML | Módulo Consumidor | Justificación |
|---|---|---|---|---|---|
| **A-1** | `xgboost.tbm_asymmetric` | `true` | `xgboost` | `luna/features/tbm.py::apply_triple_barrier` | Activa el cálculo de semi-varianza asimétrica para corregir el sesgo del TBM simétrico frente a la distribución de retornos con negative skew de BTC. |
| **A-2** | `xgboost.tbm_asymmetry_ratio_cap` | `2.0` | `xgboost` | `luna/features/tbm.py::apply_triple_barrier` | Cap institucional del ratio ATR_downside/ATR_upside. Permite máx 2x más amplitud en SL que en PT. Valor superior a 2.0 degradaría matemáticamente la esperanza matemática del trade. El clip interno de seguridad en `compute_asymmetric_ratio` es `10.0` para evitar propagación de valores explosivos en `bfill`. |
| **B-1** | `features.sfi_knn_adaptive` | `true` | `features` | `luna/features/feature_selection_e.py::AutoLagDiscovery.find_lag` | Activa KNN adaptativo `k = max(3, sqrt(N))` en el estimador de Información Mutua. Corrige la "Maldición de la Dimensionalidad" con `k=3` fijo en ventanas de alta dimensionalidad (D>50). |
| **B-2** | `features.sfi_mrmr_enabled` | `true` | `features` | `luna/features/feature_selection_e.py` | Flag de control para la lógica MRMR en el pipeline SFI. Pendiente de integración completa en la fase de scoring de clustering (actualmente informativo). |
| **C-1** | `autoencoder.ae_anchored_kl_loss` | `true` | `autoencoder` | `luna/models/train_autoencoder.py::train_autoencoder` | Activa la carga del modelo AE de la ventana anterior como ancla para el Contrastive Drift Loss. |
| **C-2** | `autoencoder.ae_kl_lambda` | `0.05` | `autoencoder` | `luna/models/train_autoencoder.py::train_autoencoder` | Peso de la penalización de deriva latente en la función de coste: `loss = MSE + 0.05 * MSE_latente`. A 0.05 se previene la rotación del espacio latente sin impedir que el modelo se adapte al nuevo régimen. |
| **C-3** | `autoencoder.ae_kl_drift_alarm_threshold` | `0.5` | `autoencoder` | `luna/models/train_autoencoder.py::train_autoencoder` | Umbral de MSE en espacio latente (Tanh bounded [-1,1]) que activa un WARNING de latent drift. 0.5 en escala Tanh equivale a una rotación semántica significativa del 50% del espacio. |

## Parámetros de Combo Estricto e Hipótesis de Threshold (2026-06-18)

| ID | Parámetro (`settings.yaml`) | Valor Institucional | Sección YAML | Módulo Consumidor | Justificación |
|---|---|---|---|---|---|
| **D-1** | `xgboost.conformal_gap_threshold` | `0.10` | `xgboost` | `luna/models/predict_oos.py` | Censor Estricto. Censura trades donde la diferencia entre la confianza predicha (probabilidad) y el Win Rate móvil excede el 10%, indicando sobreconfianza del modelo debido a Covariate Shift. |
| **D-2** | `xgboost.conformal_min_win_rate` | `0.55` | `xgboost` | `luna/models/predict_oos.py` | Censor Estricto. Win Rate mínimo móvil requerido para no censurar si se incumple el gap de calibración. |
| **D-3** | `xgboost.signal_threshold_modifier` | `-0.02` | `xgboost` | `luna/models/signal_filter.py::apply_model_threshold` | Modificador agresivo aplicado para reducir la censura extrema de señales, recuperando Alpha al capturar trades viables que quedaban a un 1-2% del umbral de decisión estático. |
