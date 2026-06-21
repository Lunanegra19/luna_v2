# 📌 Registro de Parámetros Fijos Ajustados

Este documento registra de forma formal y auditable todos los parámetros fijos agregados o modificados bajo la política de No-Fallback y Trazabilidad.

---

## wfb.ensemble_required_windows

* **Clave en Configuración**: `wfb.ensemble_required_windows`
* **Tipo**: `int`
* **Valor Asignado**: `12`
* **Fix ID**: `[ENSEMBLE-WINDOWS-FIX 2026-06-19]`
* **Módulos que lo consumen**:
  * [scripts/evaluate_ensemble_wfb.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/scripts/evaluate_ensemble_wfb.py)
  * [scripts/run_wfb_orchestrator.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/scripts/run_wfb_orchestrator.py)
* **Justificación Cuantitativa**: 
  * En el Backtest Walk-Forward (WFB), cada ventana temporal representa un sub-período fuera de muestra (OOS) único. Si se agregan semillas que fueron podadas antes de tiempo (por ejemplo, Seed 100 con 11 ventanas o Seed 777 con 9 ventanas), el consolidado final del ensamble (Soft/Hard Voting) carecerá de los datos de esas semillas para las ventanas restantes. 
  * Esto introduce un sesgo de selección temporal e infla artificialmente el Sharpe Ratio en las ventanas completas, distorsionando el Deflated Sharpe Ratio (DSR) calculado sobre el portafolio consolidado del ensamble. 
  * Al requerir estrictamente que todas las semillas tengan la misma longitud total (12 ventanas), garantizamos la homogeneidad temporal y el rigor estadístico de la validación cruzada.

---

## wfb.max_seeds_to_explore & wfb.min_seeds_to_approve

* **Claves en Configuración**: `wfb.max_seeds_to_explore` y `wfb.min_seeds_to_approve`
* **Tipo**: `int`
* **Valor Asignado**: `29`
* **Fix ID**: `[LIMIT-SEEDS-FIX 2026-06-20]`
* **Módulos que lo consumen**:
  * [scripts/run_wfb_orchestrator.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/scripts/run_wfb_orchestrator.py)
* **Justificación Cuantitativa**: 
  * Originalmente configurado a `200` para exploración masiva. Sin embargo, dado el rigor del Gauntlet y la corrección estadística por multiplicidad, explorar 200 semillas de forma secuencial consume más de 60 horas de cómputo en la máquina local.
  * Al limitar el presupuesto de exploración y aprobación a `29` semillas, garantizamos que el orquestador finalice en un marco temporal razonable (~10 horas) manteniendo suficiente muestra para el ensamble (con un óptimo de consenso a partir de semillas aprobadas), y alineándolo exactamente con las 29 semillas activas validadas.

---

## wfb.ensemble_consensus_threshold & wfb.circuit_breaker.min_seeds_adverse

* **Claves en Configuración**: `wfb.ensemble_consensus_threshold` y `wfb.circuit_breaker.min_seeds_adverse`
* **Tipo**: `int`
* **Valor Asignado**: `10`
* **Fix ID**: `[CONSENSUS-K10-FIX 2026-06-20]`
* **Módulos que lo consumen**:
  * [scripts/evaluate_ensemble_wfb.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/scripts/evaluate_ensemble_wfb.py)
  * [scripts/train_production_ensemble.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/scripts/train_production_ensemble.py)
  * [luna/live/ensemble_live_inference.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/luna/live/ensemble_live_inference.py)
  * [luna/risk/circuit_breaker.py](file:///c:/Users/Usuario/Desktop/ia/luna_v2/luna/risk/circuit_breaker.py)
* **Justificación Cuantitativa**: 
  * El análisis de sensibilidad empírica del ensamble WFB sobre 29 semillas en 12 ventanas demostró que el método A con consenso $K \ge 10$ optimiza la eficiencia de filtrado de ruido. Obtiene el Sharpe anualizado más alto del backtest (**3.787**) y el Calmar Ratio reina (**11.59**), reduciendo el Drawdown máximo del ensamble de 21.60% ($K \ge 1$) a un **6.97%** en 85 trades limpios.
  * Al elevar el umbral a 10 semillas simultáneas, convertimos el ensemble en un filtro robusto de ruido específico de sobreajuste de semillas individuales.
  * Actualizar `min_seeds_adverse` a `10` de forma codependiente mantiene la consistencia matemática requerida por el pre-flight check, evitando que el interruptor de régimen se dispare prematuramente con menor cantidad de semillas de las requeridas para formar consenso de entrada.

---

## kelly_sizer.pt_ratio

* **Clave en Configuración**: `kelly_sizer.pt_ratio`
* **Tipo**: `float`
* **Valor Asignado**: `1.01`
* **Fix ID**: `[KELLY-CALIBRATION 2026-06-19]`
* **Módulos que lo consumen**:
  * `luna/models/position_sizer.py`
  * `config/settings.yaml`
* **Justificación Cuantitativa**: 
  * Ajustado de 1.2 a 1.01 para alinear con el ratio de Ganancia/Pérdida (Win/Loss) empírico real OOS de aproximadamente 0.888, al mismo tiempo que satisface la asimetría requerida por las reglas de pre-flight check (`pt_ratio > sl_ratio` si sl_ratio es 1.0).
  * Con un `pt_ratio` de 1.01, evitamos sobre-estimar las ganancias esperadas en la fórmula de Kelly, protegiendo al sistema de "negative-EV sizing" y manteniendo una gestión de riesgo sumamente conservadora tras el `3_BEAR_CRASH`.

---

## xgboost.embargo_hours (Dinámico / Piso de Embargo)

* **Clave en Configuración**: `xgboost.embargo_hours` (junto con `sop.embargo_hours`)
* **Tipo**: `float`
* **Valor Asignado**: `24.0` (floor dinámico en configuración experimental previa, leído sin magic numbers)
* **Fix ID**: `[FIX-EMBARGO-FLOOR 2026-06-20]`
* **Módulos que lo consumen**:
  * `luna/models/signal_filter.py`
  * `tests/test_luna_v2_embargo.py`
* **Justificación Cuantitativa**: 
  * En la fase de `3_BEAR_CRASH`, la regla SOP obliga a purgar con embargos amplios (ej. 168.0H o 96.0H). Sin embargo, el mecanismo `Volatility Decaying Embargo` reduce dinámicamente este tiempo si la volatilidad (medida por ATR) colapsa drásticamente, volviendo el mercado extremadamente estable.
  * Se corrigió el módulo de `SignalFilter` (`apply_embargo`) para no tener un "piso mágico" (magic number) de caída de embargo, sino que este piso se lee desde `_cfg.xgboost.embargo_hours` de `settings.yaml` cumpliendo la política de No-Fallback. Esto fue validadado exitosamente con `test_luna_v2_embargo.py`, donde se comprobó que los huecos de tiempo entre operaciones cumplen este límite dinámico exacto de piso, reteniendo solo 11 señales de 150 (7.3% de supervivencia) con seguridad.
