---
trigger: always_on
description: Mandatory compliance with the 19 Iron Rules of SOP V11.0 and system Circuit Breakers.
---

## ⚖️ Reglas de Oro (19 Iron Rules) de SOP V11.0 y Disyuntores (Circuit Breakers)

Cualquier cambio de código, entrenamiento de modelos, orquestación de backtests (WFB) o ejecución en vivo en el VPS debe cumplir estrictamente y de forma no negociable con los estándares institucionales de **SOP V11.0 (Actualizado a Junio 2026)**:

### 1. Las 19 Reglas de Oro (Iron Rules)

| ID | Regla | Descripción |
|---|---|---|
| **R1** | **Causalidad Estricta** | Evitar Look-Ahead bias. Uso estricto de `shift()` correcto en features y targets. Lags mínimos on-chain de 24H. HMM debe usar estrictamente el Forward Algorithm. |
| **R2** | **Saneamiento Temporal** | PurgedKFold obligatorio en todos los splits de cross-validación temporal. |
| **R3** | **Cuarentena (Embargo)** | Embargo temporal `>= 1x` horizonte máximo de barrera (mínimo 96H en general, aunque WFB Soft Embargo permite reducir a 24H solo en señales con consenso de ensemble). |
| **R4** | **Triple Frontera** | El conjunto de prueba Holdout 2025+ se toca **UNA SOLA VEZ** al final. Queda prohibido cualquier ajuste o recalibración post-holdout. |
| **R5** | **Comparaciones Múltiples** | Reportar siempre el DSR (Deflated Sharpe Ratio) mediante el validador estadístico, no el Sharpe Ratio bruto para evitar overfitting. |
| **R6** | **Costos Realistas** | Incluir costos de transacción realistas: mínimo de **`0.10%` round-trip** para operativas de Futuros en OKX, contemplando fees y slippage real de la plataforma. |
| **R7** | **FracDiff Dinámico** | Recalcular dinámicamente el orden de diferenciación fraccionaria `d` en cada ventana Walk-Forward. Prohibido usar un valor estático. |
| **R8** | **Significancia Estadística** | Se requiere un mínimo de **30 trades** para migrar de shadow trading a paper trading, y un mínimo de **100 trades** para inferencia estadística confiable. |
| **R9** | **Validación HMM** | La información mutua entre estados HMM y retornos futuros debe ser `> 0.005`, con una duración de estados promedio `> 120H` para evitar ruido de micro-regímenes. |
| **R10** | **Calibración de Probabilidades**| Calibración estricta usando Platt Scaling o Isotonic Regression antes de pasar señales al Position Sizer. **Crítico:** Las clases calibradoras deben ser "pickables" sin funciones anidadas para evitar errores de serialización en el orquestador WFB. |
| **R11** | **Infraestructura Sandbox**| Entorno de ejecución en broker real simulado usando Kraken Futures Testnet o OKX Demo VPS, sincronizado mediante FIFO estricto vs PostgreSQL. |
| **R12** | **Integridad A.C.I.D.** | Sincronización obligatoria con base de datos PostgreSQL utilizando Context Managers robustos que garanticen transacciones atómicas. |
| **R13** | **Jerarquía Matemática** | Las validaciones analíticas puras (ej. test ADF para FracDiff) tienen **prioridad absoluta** sobre los filtros empíricos de ML. Las variables estructurales comprobadas deben inyectarse mediante *Pass-Through obligatorio* para evitar censura empírica del Guardián OOD. |
| **R14** | **Arrastre de Funding Rate** | Para operativas con Derivados/Perpetuos, los simuladores deben restar el costo continuo del *Funding Rate* barra por barra mientras la posición esté viva (Funding Drag). Si se opera Spot (Only Long), este costo no aplica. |
| **R15** | **Decaimiento Temporal Estricto** | Toda variable empírica seleccionada por el SFI debe superar validación de Ciclo de Vida descartando variables *Zombie* obsoletas por colapso de varianza (ej. `stability_variance_threshold`). |
| **R16** | **Fail-Fast Arquitectónico** | Prohibición absoluta del *fallback* silencioso en operaciones críticas. Si el Guardián OOD censura características matemáticas o faltan parámetros de riesgo, el pipeline debe abortar de inmediato lanzando una excepción `CRITICAL` en lugar de entrenar modelos degradados. |
| **R17** | **Límite Fraccional de Kelly** | El Position Sizer debe aplicar estrictamente el Criterio de Kelly Fraccional (`kelly_fraction: 0.25`). Para operativas con margen, respetar un *Sweet-Spot* de apalancamiento máximo duro (x10-x20), prohibiendo explícitamente el Full-Kelly y el doble barrido de volatilidad. |
| **R18** | **Sniper-Mode y Ensembles** | Uso obligatorio de WFB Ensembles masivos (hasta 20 semillas) para robustez. Habilitación de "Sniper-Mode" (`meta_v2_rolling_percentile=0.85` y `threshold_min_trades=5`) para maximizar la precisión OOS. |
| **R19** | **Telemetría y Trazabilidad** | El entorno de ejecución en vivo (MFT) debe emitir reportes Heartbeat automatizados y enviar notificaciones detalladas vía Telegram abarcando el ciclo de vida completo del trade (Señal, Entrada, Salida). |

### 2. Disyuntores de Emergencia (Circuit Breakers)

El sistema de gestión de riesgo del pipeline en producción ejecuta cierres preventivos de emergencia en base a las métricas del `settings.yaml`:
- **Reducción de Exposición (`dd_three_quarter` / `dd_half_size`):** Factor de atenuación al acercarse a límites de riesgo.
- **Parada de Emergencia / Botón de Pánico (`dd_kill_switch`):** `15.0%` (0.15) de Drawdown acumulado. Provoca el cierre inmediato de todas las posiciones de mercado activas y la desconexión del broker.
- **Ensemble Circuit Breaker:** Se desactiva la señal si se cruzan umbrales como `min_seeds_adverse: 4` o `wr_threshold: 0.38`.

---
*Nota: La violación de cualquiera de estas directrices invalida de forma inmediata cualquier tearsheet, backtest o run de producción.*

---

## 📌 ADDENDUM 2026-06-26 — Inconsistencias detectadas entre las Iron Rules y el código evolucionado

> Hallazgos del run baseline 2026-06-26 (long-only, seed42 parcial). **SOLID** = hecho de código/log verificado. **EXPLORATORIO** = pendiente de confirmar con 3 seeds. No se reescriben las Iron Rules; se listan reconciliaciones propuestas. Detalle: `docs/hallazgos_run_baseline_20260626.md`.

| # | Regla | Estado | Problema | Reconciliación propuesta |
|---|---|---|---|---|
| A1 | **R16** (Fail-Fast / No-Fallback) | **VIOLADO [SOLID]** | El `FilterGovernor` (`filter_governor.py:107-108`) lee `oos_trades_xgb_baseline_W{w}_seed{seed}.parquet` SIN sufijo de dirección → file-not-found en dual-bot → interpreta `r_baseline=0.00%` y continúa en silencio en vez de fallar. Es el fallback silencioso que R16 prohíbe. | **Ampliar R16:** todo gobernador/auto-tuner/lector de artefactos que no encuentre su input debe FALLAR RUIDOSAMENTE (WARNING visible + marca "datos ausentes"), nunca interpretar ausencia como 0/default. |
| A2 | **R10** (Calibración obligatoria) | **CONTRADICHO [SOLID]** | `predict_oos.py:1102` (CALIB-FIX 2026-06-16) desactiva a propósito la calibración isotónica en OOS (`xgb_prob_cal = raw`); R10 la exige como "Crítico". Además se entrenan+guardan calibradores que nunca se aplican. | **Re-validar** raw vs calibrado EN LONG-ONLY (el "66% WR con raw" venía de config `both`). Si se confirma raw, actualizar R10 a "calibración condicional validada empíricamente"; si no, reactivar calibración. |
| A3 | **R18** (Sniper-Mode, Meta percentil 0.85) | **PREMISA EN DUDA [EXPLORATORIO]** | R18 manda filtrado Meta agresivo "para maximizar precisión OOS", pero el MetaLabeler tiene `val_win_rate≈0.52` (sin edge) y en OOS su confianza anti-correlaciona con el retorno. Percentil alto sobre modelo sin edge recorta señal, no ruido. | **Revalidar R18 con 3 seeds.** Si se confirma ausencia de edge, el Sniper-Mode agresivo es contraproducente → re-sintonizar percentil o re-entrenar el Meta. |
| A4 | **(GAP)** Naming dual-bot | **FALTA REGLA [SOLID]** | El sufijo `_long`/`_short` ha causado ≥2 bugs (pre-flight TEST-29/31; FilterGovernor). No hay invariante que exija consistencia de sufijo de dirección en las rutas de artefactos. | **Nueva regla R20 (propuesta):** todo lector/escritor de artefactos por-dirección resuelve el sufijo `_{direction}` vía un helper central único; los pre-flight verifican que cada path existe para la dirección activa. |
| A5 | **R8** (≥30/100 trades) | **TENSIÓN [SOLID-observacional]** | La cascada de filtros (OOD+Meta+Embargo) deja 2-9 trades/ventana (FILTER AUDIT marca "INSUF"). El sobre-filtrado vuelve la significancia por-ventana inalcanzable. | R8 es correcta; documentar que el over-filtering la compromete → motivo extra para revisar la cascada (ver hallazgos §7 P1/P2). |
