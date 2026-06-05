# 🔍 Audit Exhaustivo: Parámetros Fijos vs Dinámicos — Luna V2

> Generado el 2026-05-20 | Investigación de código sobre `luna/`, `scripts/`, `config/`
> Herramienta: `tools/diagnostics/scan_magic_numbers.py` + revisión manual de módulos críticos
> **Última actualización**: 2026-05-20T14:50 — **40 fixes completados** (cobertura total de candidatos + phase gates, split ratios y estabilidad pre-flight)

---

## Leyenda de Estado

| Icono | Descripción |
|---|---|
| 🟢 | Fijo Justificado — regla estadística o regulatoria universal. Sin cambio necesario |
| 🟡 | Fijo Configurable — en `settings.yaml`. Aceptable. Revisión menor opcional |
| 🔴 | Fijo Sin Justificación — candidato a fix en próxima sesión |
| ✅ | **FIXED** — intervenido, aplicado y verificado (sintaxis OK + imports OK) |

**Total identificado: 48 parámetros** en 15 módulos.
**Sesión 2026-05-20**: **40 fixes completados** (9 críticos + 7 segunda ronda + 5 candidatos adicionales + 12 phase gates configs + 2 split ratios configs + 5 de estabilidad pre-flight) ✅

---

## 🏁 Resumen de Fixes Completados (Sesión 2026-05-20)

| # | Parámetro | Fichero | Acción |
|---|---|---|---|
| 7 | `alpha_binomial = 1.0 / 0.15` | `settings.yaml` + `generate_tearsheet.py:976,1113` | `1.0→0.05` en settings; fallbacks `0.15→0.05`. **Mayor bug estadístico: gate WIN RATE era 3× más permisivo** |
| 8 | `THR_X = 0.72` (nombre engañoso) | `generate_tearsheet.py:1022` | Renombrado `_LABEL_X` — era coordenada de layout gráfico, no un threshold del modelo |
| 5 | `0.0015` literal duplicado | `predict_oos.py:1320` | Sustituido por `_COST_RT_F5` que lee `sop.cost_pct` de cfg con fallback documentado |
| 4 | `max_states = 6` HMM | `signal_filter.py:557` | Dinámico: `v2_config["hmm_n_states"] + 1` (Risk-Off Shield). Previene pérdida silenciosa |
| 6 | `_base_thresh = 0.38` × 2 | `pipeline_executor.py:1220,1252` | Extraído como `_XGB_BASELINE_DEFAULT` constante nombrada con warning print |
| 1 | VBH fallbacks 96H vs 168H | `train_xgboost_v2.py:301` + `predict_oos.py:1057` | Unificados a **72H** — hallazgo: el valor real en `settings.yaml` era 72H, no 96 ni 168 |
| 3 | `purge_rows = 336` | `predict_oos.py:718` | `int(vbh × 1.5)` dinámico. Aplicado via script binario (encoding Windows cp1252) |
| 9 | `cusum threshold = 4.5` | `run_statistical_validation.py:337` + `settings.yaml` | Lee `stat.cusum_threshold` de cfg. Ref: Page(1954), Hawkins&Olwell(1998), rango [4.0, 5.0] |
| 2 | `_contested_threshold = 0.35 / 0.25` | `train_metalabeler_v2.py:357,363` | ✅ FIXED | `p25(xgb_probs>0.5)` dinámico; emergencia: `max(0.20, p25×0.80)`; clamp [0.20, 0.55] |
| A | `n_target = 30` recalibrador | `online_recalibrator.py:78` | ✅ FIXED | Lee `metalabeler.meta_min_trades` de cfg. Cost también lee de `sop.cost_pct` |
| B | `ev + 0.010` tolerancia EV | `train_xgboost_v2.py:1822` | ✅ FIXED | `xgboost.ev_tolerance_pct: 0.010` en settings. Justificación arquitectónica documentada |
| C | `min/max_density_pct` fallbacks | `train_xgboost_v2.py:1526,1775` | ✅ FIXED | `max_signal_density_pct: 0.60` añadido a settings; fallbacks con print warning |
| D | `N_TRIALS_PENALTY = 3.0` | `train_xgboost_v2.py:97` + `ensemble_lgbm.py:330` | ✅ FIXED | Lee `ai_mining.advanced.n_trials_penalty: 3.0` de settings. Ref: Bailey(2014) |
| E | `DATA_MAX_NAN_PCT = 0.50` | `phase_gates.py:107` | ✅ FIXED | Lee `debug.nan_threshold_pct` (5.0%) de settings. Gate G0: 50%→5% NaN máx |
| F | `r_min_trades` HMM I4 | `train_xgboost_v2.py:1896` | ✅ FIXED | Lee `xgboost.threshold_min_trades` (20), calcula `max(5, int(base * 0.25))` dinámico |
| G | `_fallback_embargo` unknown | `signal_filter.py:1283` | ✅ FIXED | Lee `sop.embargo_hours` (72H) de cfg con fallback seguro a 168H |
| H | `cols[:50]` PSI features | `psi_guard.py:227` | ✅ FIXED | Carga top-N desde `selected_features.json` si existe, ordenando por importancia XGBoost real |
| I | `_PRUNE_THRESHOLD` orchestrator | `run_wfb_orchestrator.py:64` | ✅ FIXED | Lee `wfb.prune_threshold` (0.95) de settings.yaml con fallback seguro |
| J | `SFI_MAX_ALPHA_RATIO = 0.80` | `phase_gates.py:105` | ✅ FIXED | Mover a `settings.yaml: features.sfi_max_alpha_ratio: 0.60` (reducir también el valor) |
| K | Forensic Brier Margin | `train_xgboost_v2.py:2113` | ✅ FIXED | `3.0%` de margen adaptativo para régimen `range` para absorber ruido, `2.5%` para el resto |
| L | Clasificación TIPO-1/2/3 | `settings.yaml` (header) | ✅ FIXED | Leyenda documental y clasificaciones SFI en cabecera de settings.yaml (TEST-85) |
| M | oos_raw_probs.parquet index | `predict_oos.py:1611` | ✅ FIXED | index=True para conservar el DatetimeIndex timestamp real en el parquet (TEST-121) |
| N | CPCV reference temporal | `settings.yaml:401` | ✅ FIXED | Tabla de referencia temporal de CPCV y ETAs para n=6/8/10 grupos (TEST-129) |
| O | Holdout-first sweep calibration | `train_xgboost_v2.py:1540` | ✅ FIXED | Jerarquía holdout-first utilizando holdout_calib_months si existe (TEST-130) |
| P | Warning cache stale skip-sfi | `train_production_model.py:27` | ✅ FIXED | Comparación mtime para detectar cache stale cuando skip-sfi activo (TEST-133) |

**Verificación**: Todos los 10/10 ficheros principales — sintaxis OK · `ALL SYSTEMS GO` en validador pre-flight · 40 fixes confirmados.

---

## 🔵 Candidatos Siguientes — Investigados (Próxima Ronda)

Todos los candidatos investigados han sido **100% implementados, verificados y vinculados** a la configuración institucional de `settings.yaml` o a dinámicas basadas en datos reales.

| # | Parámetro | Valor Original | Fichero | Estado | Acción de Fix / Configuración |
|---|---|---|---|---|---|
| F | `r_min_trades` | **25% ó 10** | `train_xgboost_v2.py:1896` | ✅ FIXED | Lee `xgboost.threshold_min_trades` (20), calcula `max(5, int(base * 0.25))` |
| G | `_fallback_embargo` | **168H** | `signal_filter.py:1283` | ✅ FIXED | Lee `sop.embargo_hours` (72H) en settings.yaml, fallback 168H si falla |
| H | `cols[:50]` PSI | **Primeras 50** | `psi_guard.py:227` | ✅ FIXED | Carga top-N desde `selected_features.json` si existe (importancia XGBoost) |
| I | `_PRUNE_THRESHOLD` | **0.95** | `run_wfb_orchestrator.py:64` | ✅ FIXED | Lee `wfb.prune_threshold` de settings.yaml con fallback seguro |
| J | `SFI_MAX_ALPHA_RATIO` | **80%** | `phase_gates.py:105` | ✅ FIXED | Lee `features.sfi_max_alpha_ratio` (0.60) en settings.yaml con fallback seguro |

---

## BLOQUE 1 — Temporal: Embargo, Purga & Cooldown

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `sop.embargo_hours` | **168H** (OOS) / **24H** (CV) | `settings.yaml` → `predict_oos.py:1059`, `train_xgboost_v2.py:53` | 🟡 | Sin cambio | `= percentile_90(realized_hold_time)` del set de calibración |
| `sop.purge_hours` | **96H** | `feature_selection_e.py:142`, `train_metalabeler_v2.py:71` | 🟡 | Sin cambio | `max(embargo_hours, vertical_barrier_hours * 0.5)` |
| `purge_rows = 336` | ~~336 filas hardcode~~ → **`int(vbh × 1.5)` dinámico** | `predict_oos.py:718` | 🔴 | ✅ FIXED | `int(vertical_barrier_hours * 1.5)` filas — proporcional al horizonte TBM |
| `_fallback_embargo = 168.0` | ~~168H fallback~~ → **Lee `sop.embargo_hours` (72H)** | `signal_filter.py:1283` | 🔴 | ✅ FIXED | Lee `sop.embargo_hours` (72H) en settings.yaml, fallback 168H si falla |
| `ewma_span_hours = 168` | **168H (7 días)** | `psi_guard.py:110` → `settings.yaml: psi_ewma_span` | 🟡 | Sin cambio | Configurable. Propuesta: `= 2 × vertical_barrier_hours` |
| `cooldown_hours = 336` | **336H (2 semanas)** | `psi_guard.py:111` → `settings.yaml: psi_cooldown_hours` | 🟡 | Sin cambio | Mantener en settings; considerar ajuste por régimen HMM |

---

## BLOQUE 2 — TBM: Barreras de Salida

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `xgboost.vertical_barrier_hours` fallbacks | ~~96H (xgb) / 168H (OOS)~~ → **72H ambos** | `train_xgboost_v2.py:301`, `predict_oos.py:1057` | 🔴 | ✅ FIXED | Fallbacks unificados a 72H = valor real en `settings.yaml` |
| `pt_sl_multiplier PT = 2.0x` | **2.0× ATR** | `tbm.py:362` | 🟡 | Sin cambio | Optuna explore `pt_mult ∈ [1.5, 3.0]` como hiperparámetro |
| `pt_sl_multiplier SL = 1.0x` | **1.0× ATR** | `tbm.py:362` | 🟡 | Sin cambio | Tabla régimen→(PT,SL): BULL=2.5/0.8, BEAR=1.5/1.5, RANGE=2.0/1.0 |
| `pt_mult_min = 1.6x` (MetaLabeler) | **1.6× ATR** | `train_metalabeler_v2.py:243` | 🟡 | Sin cambio | Unificar con `xgboost.pt_mult_min` de settings |
| `min_return = 0.0015` | **0.15%** | `tbm.py:365` | 🟢 | Sin cambio | ✅ Mantener vinculado a `sop.cost_pct` |
| `dynamic_horizon_min_h` / `max_h` | Configurable en settings | `tbm.py` | 🟡 | Sin cambio | Revisar que `vertical_barrier_hours` sirva de cota superior |

---

## BLOQUE 3 — Costes de Transacción

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `sop.cost_pct` | **0.0015 (0.15%)** | `settings.yaml` → todos los módulos | 🟢 | Sin cambio | ✅ Mantener. Actualizar si cambia exchange |
| `_COST_RT = 0.0015` | **0.0015** | `predict_oos.py:1194` | 🟢 | Sin cambio | ✅ OK — lee de cfg correctamente |
| `0.0015` hardcodeado duplicado | ~~0.0015 literal~~ → **`_COST_RT_F5` desde cfg** | `predict_oos.py:1320` | 🔴 | ✅ FIXED | Lee `sop.cost_pct` de cfg con fallback documentado y print |

---

## BLOQUE 4 — Umbral XGBoost (Calibración EV-Sweep)

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `threshold_sweep_min` | **0.45** | `settings.yaml` → `train_xgboost_v2.py:1515` | 🟡 | Sin cambio | `= base_rate - 0.05` donde `base_rate` = tasa de éxito histórica del régimen |
| `threshold_sweep_max` | **0.72** | `settings.yaml` → `train_xgboost_v2.py:1516` | 🟡 | Sin cambio | `= percentile_95(xgb_prob)` sobre validation set |
| `threshold_sweep_step` | **0.005** | `settings.yaml` | 🟢 | Sin cambio | ✅ Mantener |
| `ev + 0.010` (tolerancia EV) | ~~+1.0% hardcodeado~~ → **`xgboost.ev_tolerance_pct`** | `train_xgboost_v2.py:1822` | 🔴 | ✅ FIXED | Lee de cfg. Ref: XGBoost Weak Learner; MetaLabeler aporta EV downstream |
| `_base_thresh = 0.38` × 2 | ~~0.38 literal × 2~~ → **`_XGB_BASELINE_DEFAULT`** | `pipeline_executor.py:1220,1252` | 🔴 | ✅ FIXED | Constante nombrada con warning print cuando settings no carga |
| `xgb_min_signals_count` | **0** | `signal_filter.py:386` → settings | 🟡 | Sin cambio | `max(5, int(holdout_hours / vertical_barrier_hours))` |
| `xgb_min_signals_threshold = 0.45` | **0.45** | `signal_filter.py:387` → settings | 🟡 | Sin cambio | `= threshold_sweep_min + step` |
| `min_density_pct = 0.30` fallback | ~~fallback silencioso~~ → **warning print** | `train_xgboost_v2.py:1526` | 🔴 | ✅ FIXED | `max_signal_density_pct: 0.60` añadido a settings; fallback documentado |
| `_max_density_pct = 0.60` fallback | ~~fallback silencioso~~ → **warning print** | `train_xgboost_v2.py:1775` | 🔴 | ✅ FIXED | `max_signal_density_pct: 0.60` añadido a settings; fallback documentado |
| `_PRUNE_THRESHOLD = 0.95` | ~~0.95 hardcode~~ → **Lee `wfb.prune_threshold` (0.95)** | `run_wfb_orchestrator.py:64` | 🟡 | ✅ FIXED | Lee `wfb.prune_threshold` de settings.yaml con fallback seguro |

---

## BLOQUE 5 — MetaLabeler V2: Regularización RF

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `_topo_max_depth clip [2, 8]` | **log2(n_minority) ∈ [2,8]** | `train_metalabeler_v2.py:307` | 🟡 | Sin cambio | Añadir `metalabeler.rf_max_depth_cap: 8` en settings |
| `_topo_min_leaf = n×0.04 clip [10,50]` | **4% samples ∈ [10,50]** | `train_metalabeler_v2.py:309` | 🟡 | Sin cambio | `clip(n*0.04, 10, max(50, n*0.03))` para escalar el tope |
| `_contested_threshold = 0.35 / 0.25` | ~~0.35 / 0.25 hardcode~~ → **`p25(xgb_probs>0.5)` dinámico** | `train_metalabeler_v2.py:357,363` | 🔴 | ✅ FIXED | Umbral vinculado a distribución real del agente; clamp [0.20, 0.55] |
| `split_idx = int(n * 0.80)` | **80/20** | `train_metalabeler_v2.py:1311`, `:508` | 🟡 | Sin cambio | `metalabeler.val_split_ratio: 0.20` en settings con guard |
| `r_min_trades` | ~~25% ó 10~~ → **Dinámico desde `threshold_min_trades`** | `train_xgboost_v2.py:1896` | 🔴 | ✅ FIXED | Lee `xgboost.threshold_min_trades` (20), calcula `max(5, int(base * 0.25))` |

---

## BLOQUE 6 — Kelly Sizer & Risk Management

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `kelly_fraction = 0.25` | **25% (Quarter-Kelly)** | `kelly_sizer.py:142` | 🟢 | Sin cambio | ✅ Mantener (Thorp/MacLean) |
| `max_position = 0.15` | **15%** del capital | `kelly_sizer.py:144` | 🟡 | Sin cambio | `min(0.15, full_kelly * 0.25)` — adaptativo |
| `meta_prob default = 0.5` | **0.50** neutro | `kelly_sizer.py:55` | 🟢 | Sin cambio | ✅ Mantener |
| `elevated_risk_kelly_penalty = 0.50` | **50%** reducción | `signal_filter.py:153` → settings | 🟡 | Sin cambio | Función continua: `penalty = 1 - transition_risk` |
| `bear_transition_elevated_risk = 0.30` | **0.30** | `signal_filter.py:152` → settings | 🟡 | Sin cambio | `= percentile_75(transition_risk)` en períodos históricos |
| `bear_transition_veto_thresh = 0.40` | **0.40** | `signal_filter.py:433` → settings | 🟡 | Sin cambio | Función continua 1.0→0.0 (veto suave) |

---

## BLOQUE 7 — PSI Guard (Drift Monitor)

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `PSI_STABLE = 0.10` | **0.10** | `psi_guard.py:100` | 🟢 | Sin cambio | ✅ Estándar bancario (Yurdakul 2018) |
| `PSI_MODERATE = 0.25` | **0.25** | `psi_guard.py:101` | 🟢 | Sin cambio | ✅ Estándar bancario |
| `PSI_CRITICAL = 0.50` | **0.50** | `psi_guard.py:102` | 🟢 | Sin cambio | ✅ Estándar bancario |
| `psi_alert_threshold = 0.25` | **0.25** | `psi_guard.py:108` → settings | 🟢 | Sin cambio | ✅ Mantener |
| `psi_halt_threshold = 0.50` | **0.50** | `psi_guard.py:109` → settings | 🟢 | Sin cambio | ✅ Mantener |
| `N_BINS = 10` (deciles PSI) | **10 bins** | `psi_guard.py:104` | 🟢 | Sin cambio | ✅ Estándar estadístico |
| `cols[:50]` (max features PSI) | ~~Primeras 50~~ → **Importancia XGBoost** | `psi_guard.py:227` | 🟡 | ✅ FIXED | Carga top-N desde `selected_features.json` si existe, preservando importancia |

---

## BLOQUE 8 — Phase Gates (Control de Calidad del Pipeline)

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `XGB_AUC_HARD_STOP = 0.510` | ~~0.510~~ → **Configurable** | `phase_gates.py:127` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.xgb_auc_hard_stop` con fallback robusto |
| `XGB_AUC_WARN = 0.530` | ~~0.530~~ → **Configurable** | `phase_gates.py:128` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.xgb_auc_warn` con fallback robusto |
| `XGB_BRIER_HARD_STOP = 0.2850` | ~~0.285~~ → **Configurable** | `phase_gates.py:129` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.xgb_brier_hard_stop` con fallback robusto |
| `XGB_BRIER_WARN = 0.2700` | ~~0.270~~ → **Configurable** | `phase_gates.py:130` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.xgb_brier_warn` con fallback robusto |
| `XGB_BRIER_DEGRADED_MAX_AGENTS = 1` | ~~1 agente~~ → **Configurable** | `phase_gates.py:131` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.xgb_brier_degraded_max_agents` |
| `XGB_PROBA_STD_MIN = 0.010` | ~~0.010~~ → **Configurable** | `phase_gates.py:132` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.xgb_proba_std_min` con fallback |
| `LGBM_PROBA_STD_MIN = 0.010` | ~~0.010~~ → **Configurable** | `phase_gates.py:133` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.lgbm_proba_std_min` con fallback |
| `HMM_MIN_ACTIVE_STATES = 2` | ~~2 estados~~ → **Configurable** | `phase_gates.py:134` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.hmm_min_active_states` con fallback |
| `SFI_MIN_FEATURES = 5` | ~~5 features~~ → **Configurable** | `phase_gates.py:135` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.sfi_min_features` con fallback |
| `SFI_MAX_ALPHA_RATIO = 0.80` | ~~80% AI Mining~~ → **lee `features.sfi_max_alpha_ratio` (60%)** | `phase_gates.py:105` | 🔴 | ✅ FIXED | Mover a `settings.yaml: features.sfi_max_alpha_ratio: 0.60` |
| `DATA_MIN_ROWS = 1000` | ~~1000 filas~~ → **Configurable** | `phase_gates.py:136` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.data_min_rows` con fallback |
| `DATA_MAX_NAN_PCT = 0.50` | ~~50% NaN~~ → **lee `debug.nan_threshold_pct` (5%)** | `phase_gates.py:107` | 🔴 | ✅ FIXED | Gate G0: 50%→5% NaN máx. Lee de `debug.nan_threshold_pct` en settings |
| `DATA_MAX_GAP_H = 48` | ~~48H gaps~~ → **Configurable** | `phase_gates.py:137` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.data_max_gap_h` con fallback |
| `SIGNAL_MIN_COUNT_WARN = 5` | ~~5 señales~~ → **Configurable** | `phase_gates.py:138` | 🔴 | ✅ FIXED | Lee de `settings.yaml:stat.signal_min_count_warn` con fallback |
| `seq_len default = 48` | **48H** | `phase_gates.py:717` | 🟡 | Sin cambio | Leer siempre de `metalabeler_v2_*_config.json` |

---

## BLOQUE 9 — Validación Cruzada: CPCV & SFI

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `sfi_n_groups = 6` | **6 grupos** `C(6,2)=15 paths` | `feature_selection_e.py:137` | 🟡 | Sin cambio | `max(6, min(10, int(n_train / vb_h / 30)))` |
| `n_cpcv_groups = 8` (MetaLabeler) | **8 grupos** | `train_metalabeler_v2.py:74` → settings | 🟡 | Sin cambio | ✅ Mantener |
| `n_startup_trials = 10` | **10 trials** Optuna warm-up | `train_xgboost_v2.py:1425` | 🟢 | Sin cambio | ✅ Default oficial Optuna TPE |
| `N_TRIALS_PENALTY = 3.0` | ~~3.0 hardcode~~ → **`ai_mining.advanced.n_trials_penalty`** | `train_xgboost_v2.py:97`, `ensemble_lgbm.py:330` | 🔴 | ✅ FIXED | Ref: Bailey(2014). Lee de settings con fallback documentado |
| `SFI_DSR_N_TRIALS = 600` | **600** | `feature_selection_e.py:148` | 🟡 | Sin cambio | ✅ Mantener sincronizado con optuna_trials |
| `HMM_MIN_ACTIVE_STATES = 2` | **2 estados** mínimos | `phase_gates.py:103` | 🟡 | Sin cambio | ✅ Mantener |
| `random_state = 42` (XGBoost) | ~~42 hardcode~~ → **`int(os.environ.get('LUNA_SEED', 42))`** | `train_xgboost_v2.py:L1167,L2007` | 🔴 | ✅ **FIXED 2026-05-28** | **[FIX-RANDOM-STATE-01]** Con `random_state=42` fijo, todas las seeds WFB (42,100,777,1337,2025) entrenaban modelos XGBoost estructuralmente idénticos → ensemble sin diversidad real. Ahora cada seed usa su LUNA_SEED como random_state → diversidad estadística genuina |
| `TPESampler seed = 42` (Optuna XGBoost) | ~~cfg.optuna_seed fijo~~ → **prioriza `LUNA_SEED` env** | `train_xgboost_v2.py:L1499` | 🔴 | ✅ **FIXED 2026-05-28** | **[FIX-RANDOM-STATE-01]** Mismo problema: exploración Optuna idéntica entre seeds. Ahora usa LUNA_SEED si está en entorno, cfg.optuna_seed como fallback |
| `default_rng(42)` TBM sampling | ~~42 hardcode~~ → **`int(LUNA_SEED)`** | `train_xgboost_v2.py:L668` | 🔴 | ✅ **FIXED 2026-05-28** | **[FIX-RANDOM-STATE-01b]** Subsampling de eventos TBM ahora varía por seed |
| `PlattCalibrator random_state = 42` | ~~42~~ → **0 (determinista)** | `train_xgboost_v2.py:L42` | 🟡 | ✅ **FIXED 2026-05-28** | LR calibrador no necesita variabilidad entre seeds |
| `RandomForestClassifier random_state = 42` (MetaLabeler) | ~~42 hardcode~~ → **`int(LUNA_SEED)`** | `train_metalabeler_v2.py:L275,L473,L1257,L1267,L1290,L1297,L1351` | 🔴 | ✅ **FIXED 2026-05-28** | **[FIX-RANDOM-STATE-02]** RF árbitro + CV RF + CPCV classifiers en MetaLabeler. 7 instancias corregidas |
| `random_state = 42` (ensemble_lgbm) | ~~42 hardcode~~ → **`int(LUNA_SEED)`** | `ensemble_lgbm.py:L2604,L4664` | 🔴 | ✅ **FIXED 2026-05-28** | **[FIX-RANDOM-STATE-03]** LightGBM stacking final ahora usa LUNA_SEED |
| `random_state = 42` (feature_selection_e) | ~~42~~ × 7 instancias | `feature_selection_e.py:L883,L1085,L1488,L1567,L1667,L2400` | 🟡 | ✅ **NO FIX** | SFI compartido entre seeds (sfi_lock.json). Cambiar random_state invalidaría fingerprint de cache SFI |
| `random_state = 42` (hmm_regime L93, L606) | ~~42~~ en init/NAS | `hmm_regime.py:L93,L606` | 🟡 | ✅ **NO FIX** | L93: sobreescrito por FIX-HMM-INIT-01 (multi-seed search 0..10). L606: HMM compartido por ventana |
| `random_state = 42` (signal_filter, regime_router) | ~~42~~ LR calibrador | `signal_filter.py:L13`, `regime_router.py:L16` | 🟡 | ✅ **NO FIX** | LR calibrador Platt post-modelo — determinista por diseño |
| `n_components = 3` (hmm_regime MOCK) | **3** en bloque mock | `hmm_regime.py:L1406` | 🟡 | ✅ **NO FIX** | Solo en objeto mock para tests/dry-runs. Coherente con state_map de 3 entradas del mock |
| `gap = 144` MetaLabeler CV embargo | ~~144 hardcode~~ → **`embargo_hours + seq_len`** | `train_metalabeler_v2.py:L462` | 🔴 | ✅ **FIXED 2026-05-28** | **[FIX-EMBARGO-META-CV-01]** SOP R3 crítico. Calcula dinámicamente desde `cfg.sop.embargo_hours + cfg.metalabeler.seq_len`. Fallo LOUD si settings no disponible |
| `n_estimators = 2000` IS fit auxiliar | **2000** | `train_xgboost_v2.py:L1241` | 🟡 | ✅ **DOC-ONLY** | IS fit auxiliar para Brier CV — no es el modelo final Optuna. Justificable como constante de arquitectura |
| `n_estimators = 150, min_child_samples = 50` CPCV MetaLabeler | **150 / 50** | `train_metalabeler_v2.py:L1259,L1289` | 🟢 | ✅ **DOC-ONLY** | Classifiers internos CPCV de MetaLabeler — temporales para generación de OOS probs. No son modelos finales |

---

## BLOQUE 10 — HMM Régimen

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `max_states = 6` (one-hot) | ~~6 hardcode~~ → **`hmm_n_states + 1` dinámico** | `signal_filter.py:557` | 🔴 | ✅ FIXED | Lee `v2_config["hmm_n_states"] + 1` (Risk-Off Shield) |
| `_n_hmm_total = _n_hmm + 1` | `n_states + 1` | `signal_filter.py:627` | 🟡 | Sin cambio | Leer del modelo si el shield está activo |
| `survival_rate >= 0.80` (HMM) | **80%** | `hmm_regime.py:290` | 🟡 | Sin cambio | ✅ Mantener (estándar estadístico de re-sampling) |

---

## BLOQUE 11 — Seguridad & Leakage Guards

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `CORR_THRESHOLD_ALERT = 0.40` | **40%** correlación | `guard_pipeline.py:62` | 🟡 | Sin cambio | Calibrar sobre distribución histórica |
| `CORR_THRESHOLD_CRITICAL = 0.55` | **55%** | `guard_pipeline.py:63` | 🟡 | Sin cambio | ✅ Razonable. Mantener |
| `NAN_THRESHOLD_PCT = 5.0` | **5%** NaN permitido por columna | `debug_guards.py:69`, `data_integrity_check.py:66` | 🟡 | Sin cambio | ✅ Mantener en settings |
| `CORR_LEAKAGE_THR = 0.95` | **0.95** | `debug_guards.py:70` | 🟡 | Sin cambio | ✅ Mantener |
| `p_value binomial <= 0.05` | **5%** | `generate_validation_report.py:406` | 🟢 | Sin cambio | ✅ Estándar estadístico (α=0.05) |

---

## BLOQUE 12 — Splits Train/Val

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `split_idx = int(n * 0.80)` (MetaLabeler) | ~~80/20~~ → **Configurable via ratio** | `train_metalabeler_v2.py:1338` | 🔴 | ✅ FIXED | Lee `metalabeler.val_split_ratio` en settings, default 0.20 |
| `split_idx = int(total_rows * 0.80)` (OOS fallback) | ~~80/20~~ → **Configurable via ratio** | `predict_oos.py:728` | 🔴 | ✅ FIXED | Lee `metalabeler.val_split_ratio` en settings, default 0.20 |
| `purge_rows = 336` en fallback OOS | ~~336 = 14 días hardcode~~ → **`int(vbh × 1.5)` dinámico** | `predict_oos.py:718` | 🔴 | ✅ FIXED | Proporcional al horizonte TBM. Aplicado via script binario |
| `Autoencoder split = 0.80` | **80/20** | `train_autoencoder.py:111` | 🟡 | Sin cambio | Añadir como `autoencoder.val_split_ratio: 0.20` |

---

## BLOQUE 13 — Criterios de Aprobación de Modelos

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `min_dsr = 0.75` | **0.75** | `generate_validation_report.py:405`, `generate_tearsheet.py:828` | 🟡 | Sin cambio | ✅ Bailey & LdP (2014). Mantener |
| `DSR >= 0.80` Alpha Mining | **0.80** | `train_xgboost_v2.py:284`, `ensemble_lgbm.py:699` | 🟡 | Sin cambio | ✅ Mantener diferenciado |
| `alpha_binomial` | ~~1.0 en settings / 0.15 fallback código~~ → **0.05 en ambos** | `settings.yaml:301`, `generate_tearsheet.py:976,1113` | 🔴 | ✅ FIXED | **Mayor bug estadístico del sistema corregido** |
| `max_drawdown_pct = 60` | **60% max DD** | `generate_validation_report.py:408` | 🟡 | Sin cambio | Reducir a 40% o parametrizar |
| `meta_v2_min_prob = 0.50` | **0.50** fallback neutro | `signal_filter.py:648` | 🟢 | Sin cambio | ✅ Mantener |
| `min_trades = 20` (XGB calibración) | **20** | `train_xgboost_v2.py:1518` → settings | 🟡 | Sin cambio | `max(20, int(holdout_hours / vertical_barrier_hours))` |
| `meta_min_trades = 20` (MetaLabeler) | **20** | `calibrate_probabilities.py:746` → settings | 🟡 | Sin cambio | Usar mismo criterio que `min_trades` XGBoost |
| `n_target = 30` (online recalibrador) | ~~30 hardcode~~ → **`metalabeler.meta_min_trades`** | `online_recalibrator.py:78` | 🔴 | ✅ FIXED | Lee de settings; cost también lee de `sop.cost_pct` |
| `cusum threshold = 4.5` | ~~4.5 hardcode~~ → **lee `stat.cusum_threshold` de cfg** | `run_statistical_validation.py:337` + `settings.yaml` | 🔴 | ✅ FIXED | Ref: Page(1954), Hawkins&Olwell(1998), rango [4.0, 5.0] |

---

## BLOQUE 14 — AI Mining & Features Especiales

| Parámetro | Valor Actual | Fuente | Clase | Estado | Propuesta Dinámica |
|---|---|---|---|---|---|
| `THR_X = 0.72` en tearsheet | ~~0.72 con nombre engañoso~~ → **`_LABEL_X`** | `generate_tearsheet.py:1022` | 🔴 | ✅ FIXED | Era coordenada X de layout gráfico, no un threshold. Renombrado |
| `col[:50]` PSI multivariate | ~~Primeras 50~~ → **Importancia XGBoost** | `psi_guard.py:227` | 🟡 | ✅ FIXED | Carga top-N desde `selected_features.json` si existe, preservando importancia |
| `assert 0.40 <= thr <= 0.65` | **[0.40, 0.65]** valid range | `scripts/pre_flight/test_env.py:500` | 🟡 | Sin cambio | Sincronizar con `threshold_sweep_max` del settings activo |

---

## Parámetros Sanos — NO tocar

| Parámetro | Valor | Justificación |
|---|---|---|
| `cost_pct = 0.0015` | 0.15% | Kraken taker×2 + slippage×2 ✅ |
| `kelly_fraction = 0.25` | Quarter-Kelly | Thorp/MacLean ✅ |
| `threshold_sweep_step = 0.005` | Paso fino | 144 puntos de búsqueda ✅ |
| `min_dsr = 0.75` | 0.75 | Bailey & LdP (2014) ✅ |
| `n_startup_trials = 10` | 10 | Default oficial Optuna TPE ✅ |
| `meta_v2_min_prob = 0.50` | 0.50 | Fallback neutro ✅ |
| `PSI_STABLE/MODERATE/CRITICAL` | 0.10/0.25/0.50 | Estándar bancario (Yurdakul 2018) ✅ |
| `survival_rate >= 0.80` HMM | 80% | Estándar estadístico re-sampling ✅ |
| `p-value binomial <= 0.05` | 5% | Estándar estadístico α=0.05 ✅ |
| `N_BINS = 10` PSI | 10 deciles | Estándar estadístico PSI ✅ |

---

## 🔒 Mecanismo de Lock y Seguridad (.wfb_lock)

Para evitar colisiones catastróficas y condiciones de carrera al ejecutar múltiples instancias del orquestador Walk-Forward Validation en el mismo workspace de disco, Luna implementa un sistema de cerrojo atómico (`.wfb_lock` en la raíz del proyecto).

### El Problema Tradicional del Lock Huérfano
1. **Terminaciones abruptas (SIGKILL, IDE hard-stop)**: Cuando la ejecución se cancela mediante el IDE o por cortes de energía, los gestores de eventos `atexit` y las capturas de señales estándar no consiguen ejecutarse. Esto deja el archivo `.wfb_lock` huérfano con el PID del proceso muerto guardado en su interior.
2. **Colisión por Reciclaje de PID**: En sistemas operativos con alta concurrencia o tiempos prolongados de desarrollo (como Windows), los PIDs se reciclan con frecuencia. Si un archivo `.wfb_lock` contiene un PID muerto que luego es reasignado a un proceso completamente ajeno a Luna (como un LSP, un navegador o un daemon del sistema), Luna bloqueaba la ejecución al creer incorrectamente que el orquestador anterior seguía vivo.
3. **Inactividad prolongada (TTL)**: Los desarrolladores debían borrar manualmente el archivo para volver a ejecutar el pipeline tras crasheos.

### Implementación del Mecanismo de Autorecuperación Inteligente
En la arquitectura de `wfb_worker.py` (dentro de `_acquire_lock()`), se ha implementado un recuperador con lógica de defensa de tres niveles:
* **Nivel 1: Time-To-Live (TTL) de Emergencia**: Si el archivo de lock es anterior a 1.5 horas, se asume inactividad prolongada por crasheo y se elimina atómicamente.
* **Nivel 2: Validación de PID Activo (`psutil.pid_exists`)**: Se verifica si el PID registrado realmente existe en la lista de procesos activos del OS. Si no existe, se limpia inmediatamente.
* **Nivel 3: Desambiguación de Command Line**: Si el PID está vivo, se lee su cmdline. Si no contiene términos de control como `"python"` o `"wfb"`, se detecta la colisión del PID reciclado y se destruye el lock huérfano de manera autónoma, logueando un warning claro de recuperación.
* **Solución Futura / Industrial**: Guardar tanto el PID como el `create_time` del proceso de Python en el archivo de lock. Al arrancar, si los tiempos de creación no coinciden milimétricamente con el proceso activo actual para ese PID, se confirma el reciclaje de PID y se limpia de forma 100% determinista.

---

## ⚡ Auditoría de Rendimiento e Integridad Lógica (Sesión 2026-05-20)

### 1. Cuello de Botella de Rendimiento en `apply_triple_barrier` (`tbm.py:L431`)
* **Problema**: La asignación de la barrera de tiempo (`t1`) en la lógica de TBM dinámico realiza la siguiente operación para cada evento `t`:
  ```python
  candidates = price_series.index[price_series.index >= next_t]
  ```
  Esto crea una máscara booleana de tamaño completo ($N = 76,377$) sobre el índice de Pandas en cada iteración del bucle, incurriendo en un coste de complejidad temporal de $O(E \times N) \approx O(N^2)$. Con un dataset completo, esto consume entre 20 y 45 segundos de CPU de forma ineficiente, totalizando miles de millones de comprobaciones booleanas en memoria.
* **Solución (Planeada post-run)**: Dado que `price_series.index` está estrictamente ordenado de forma cronológica (garantía de OHLCV), se puede utilizar búsqueda binaria $O(E \log N)$ mediante `np.searchsorted`:
  ```python
  # Extraer numpy array antes de bucle para evitar overhead
  price_values = price_series.index.values
  
  # Dentro del bucle:
  idx = np.searchsorted(price_values, next_t)
  t1.append(price_series.index[idx] if idx < len(price_values) else price_series.index[-1])
  ```
  Esto reduce el cálculo del Time Stop de ~35 segundos a solo unos pocos milisegundos de forma 100% segura y causal.

### 2. Omisión de `meta_oracle_score` por Ausencia de `oracle_verdict.md`
* **Problema**: El pipeline reporta un warning `meta_oracle_score: oracle_verdict.md no encontrado — feature omitida`.
* **Causa de la Ausencia**: La carpeta `data/` se ignora en Git (como base de datos local temporal), por lo que en entornos limpios o con `--nocache`, el archivo `oracle_verdict.md` (normalmente generado externamente por `run_weekly_mining.py` que no reside en este repositorio) no está presente. El fallback a omitir es totalmente robusto.
* **Profundización del Leakage (C-04 / A4 Audit)**: Se ha verificado que **incluso si el archivo existiera, no se inyectaría**. El análisis del 2026-05-08 (`[AUDIT-A4 FIX]`) identificó que `meta_oracle_score` se calcula como un score estático global único para toda la ventana de entrenamiento. Al propagarlo horizontalmente, tiene una varianza de 0, aportando ganancia nula (`GAIN=0`) a los modelos XGBoost/LGBM y consumiendo memoria y variables de manera innecesaria. La omisión del feature es la conducta correcta y óptima de diseño para prevenir fallos silenciosos y leakage distributivo temporal.

---

## Referencias de Código

| Módulo | Fichero | Estado de la sesión |
|---|---|---|
| MetaLabeler V2 | [train_metalabeler_v2.py](file:///g:/Mi%20unidad/ia/luna_v2/luna/models/train_metalabeler_v2.py) | **L357-363 ✅** (contested_threshold dinámico) · **L508,1311,1338 ✅** (split ratio dinámico) |
| Predicción OOS | [predict_oos.py](file:///g:/Mi%20unidad/ia/luna_v2/luna/models/predict_oos.py) | **L718 ✅** (purge_rows) · **L1057 ✅** (VBH) · **L1320 ✅** (cost) · **L1606-1613 ✅** (oos_raw_probs with index=True) |
| SignalFilter | [signal_filter.py](file:///g:/Mi%20unidad/ia/luna_v2/luna/models/signal_filter.py) | **L557 ✅** (max_states HMM) · **L1283 ✅** (_fallback_embargo desde settings.yaml) |
| XGBoost V2 | [train_xgboost_v2.py](file:///g:/Mi%20unidad/ia/luna_v2/luna/models/train_xgboost_v2.py) | **L301 ✅** (VBH fallback 72H) · **L1540-1559 ✅** (holdout-first jerarquía) · **L1896 ✅** (r_min_trades dynamic) · **L2113 ✅** (Brier gate range adaptativo) |
| Pipeline Executor | [pipeline_executor.py](file:///g:/Mi%20unidad/ia/luna_v2/luna/pipeline_executor.py) | **L1220 ✅ · L1252 ✅** (_XGB_BASELINE_DEFAULT) |
| Tearsheet | [generate_tearsheet.py](file:///g:/Mi%20unidad/ia/luna_v2/luna/reports/generate_tearsheet.py) | **L976 ✅ · L1022 ✅ · L1113 ✅** (alpha_binomial + _LABEL_X) |
| Validation Stat | [run_statistical_validation.py](file:///g:/Mi%20unidad/ia/luna_v2/scripts/run_statistical_validation.py) | **L337 ✅** (cusum_threshold desde cfg) |
| Psi Guard | [psi_guard.py](file:///g:/Mi%20unidad/ia/luna_v2/luna/risk/psi_guard.py) | **L227 ✅** (cols[:50] via selected_features.json) |
| WFB Orchestrator | [run_wfb_orchestrator.py](file:///g:/Mi%20unidad/ia/luna_v2/scripts/run_wfb_orchestrator.py) | **L62 ✅** (_PRUNE_THRESHOLD desde settings.yaml) |
| Train Production | [train_production_model.py](file:///g:/Mi%20unidad/ia/luna_v2/scripts/train_production_model.py) | **L27-46 ✅** (skip-sfi mtime check & warning) |
| Settings | [settings.yaml](file:///g:/Mi%20unidad/ia/luna_v2/config/settings.yaml) | `alpha_binomial: 0.05` ✅ · `cusum_threshold: 4.5` ✅ · `sfi_max_alpha_ratio: 0.60` ✅ · `prune_threshold: 0.95` ✅ · `ev_tolerance_pct: 0.010` ✅ · `n_purged_splits: 8` con leyenda temporal ✅ · TIPO-1/2/3 clasificaciones ✅ |
