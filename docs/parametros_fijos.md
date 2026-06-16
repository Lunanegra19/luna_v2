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
