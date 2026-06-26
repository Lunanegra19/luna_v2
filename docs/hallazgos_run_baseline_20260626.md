# 🔎 Hallazgos — Run Baseline 2026-06-26 (long-only, 3 seeds, no-cache)

> **Documento vivo.** Se anota aquí todo lo que descubrimos DURANTE la run para revisarlo al cerrarla.
> No se implementa nada con la run activa (regla `settings_restore_protection` / runs detenidas).
> Estado al crear: seed42 ~W5, 0 crashes. Watcher `bag7t50bl` vigilando para parada manual.

---

## 0. Contexto de la run
- **Comando real:** `python scripts/run_wfb_orchestrator.py --seeds 42 100 777 --nocache` (PID 21616; worker `wfb_worker.py --seed N --nocache`).
- **Dirección:** `settings.yaml:131 direction_mode: long` → LONG-ONLY (verificado en SECURITY-SETTINGS-DUMP del log).
- **Caché:** `--nocache` (LUNA_NOCACHE=1) → sin reuso, todo recalculado.
- **Objetivo:** baseline HONESTO de 3 seeds para medir mejoras una a una contra él.

---

## 1. ⚠️ El run NO está capado en 3 seeds (control de seeds)
**Hallazgo:** `--seeds 42 100 777` **no es un límite**, es la cola inicial. El total lo manda `settings.yaml:568-569`:
```
max_seeds_to_explore: 29
min_seeds_to_approve: 29
```
En `run_wfb_orchestrator.py:633` → `target_complete = _max_seeds_to_explore` (29). Cuando la cola de 3 se agota (`run_wfb_orchestrator.py:647-654`), **genera seeds aleatorias** (`random.randint(10000,99999)`) hasta llegar a 29.

**RESUELTO (2026-06-26):** con la run detenida se capó estructuralmente en settings: `max_seeds_to_explore: 3`, `min_seeds_to_approve: 3`, `active_seeds: [42,100,777]` (la lista de 29 validadas queda en comentario para restaurar). Etiqueta `[CAP-3-TEST]`. Pre-flight TEST-82 cazó el choque `active_seeds=29 vs cap=3` → corregido. El orquestador ya NO genera seeds aleatorias.

> ⚠️ **Por qué 3 seeds — base honesta de la decisión.** NO es una elección estadística óptima; es una elección de **velocidad de iteración** + una **instrucción del usuario** ("no más de 3 por test"). Trade-off:
> - **A favor:** un run de 3 seeds tarda ~3-4h vs ~30h de 29 → permite probar palancas (Meta A/B, etc.) UNA a una sin esperar días. ~3×N trades (~90-140 baseline) basta para un go/no-go direccional.
> - **En contra (límites reconocidos):** la propia regla `diagnostico_cuantitativo.md` ("usar el máximo de seeds; el error cae con √N"), R8 (≥30/100 trades) y R18 (ensembles de hasta 20) empujan a MÁS seeds. Con 3 seeds + `consensus_threshold=10` el ENSAMBLE consolidado no produce nada → solo medimos **por-seed**, no el consenso.
> - **Protocolo:** 3 seeds para **SCREENING** de palancas (exploratorio) → **29 seeds + ensamble** para **SIGN-OFF** (confirmatorio, antes de cualquier deploy). Las conclusiones de 3 seeds son indicativas, no definitivas.

**Para futuros tests de N acotado:** editar `max_seeds_to_explore`/`min_seeds_to_approve`/`active_seeds` **con la run detenida** y relanzar (el valor se lee al arrancar; editarlo en caliente no afecta al run en curso).

---

## 2. ✅ Integridad del pipeline (W1-W4 verificadas)
Cada módulo se ejecuta COMPLETO y pasa sus datos al siguiente:
- **12 fases por ventana** con marker `Completada Exitosamente` en orden: Feature Pipeline (Base) → AI Mining → Pre-SFI → SFI → Post-SFI → HMM → XGBoost → OOD Guard → AutoEncoder → MetaLabeler(LONG) → Calibrador(LONG) → Predicciones OOS. (LGBM omitido = correcto, `use_lgbm_ensemble: false`.)
- **Handoffs poblados al 100%** en cada ventana: `HMM_Semantic`, `xgb_prob`, `meta_v2_prob`, `xgb_prob_cal`, `kelly_fraction_used`, `return_raw`. `lgbm_prob` NaN = esperado.
- **SFI→XGBoost** probado vía `shap_drivers`: features reales con lags causales correctos (`ETH_Return_1d_milag1h`, `M2_YoY_Chg_z90d_milag72h`, `MVRV_Proxy_z90d_milag96h` → cumple R1).
- **SFI sano:** circuit breaker `[FIX-MATH-SURVIVAL-01]` asigna Sharpe 0 a features de predicción constante (zombies) → R15 OK.

---

## 3. 🟡 Calibrador `xgb_prob_cal == xgb_prob` — passthrough INTENCIONAL (no es bug)
```
VEREDICTO: passthrough deliberado y documentado (NO fallo silencioso)
CAUSA RAÍZ: luna/models/predict_oos.py:1099-1102
  df_oos_iter["xgb_prob_cal"] = xgb_probs_df["raw"]
  # [CALIB-FIX 2026-06-16] Desactivamos la Calibración Isotónica Estática.
  # Usamos passthrough de la prob RAW (con SPW=1.0 el ranking natural es superior, 66% WR).
```
- Los 3 calibradores SÍ se entrenan (`PlattCalibrator` L1, cascada L1→L2 Temperature→GUARDIAN-02) y se guardan cada ventana (`xgboost_isotonic_calibrator_*_long.joblib`), pero `predict_oos` los ignora a propósito.
- `[FIX-CALIB-BINARY-01/DETECTION-3]` (`cal==raw con calibradores cargados`, `barras_modificadas=0/961`) = **guard obsoleto**; el código lo degrada a *"deshabilitadas por diseño (SPW=1.0 fix)"* y NO bloquea.
- `[PIPELINE-INTEGRITY] knots=0` = inspecciona calibradores que nunca se aplican (Platt no tiene knots).

**Deuda a limpiar (post-run, baja prioridad):**
- [ ] Dejar de entrenar+guardar calibradores que no se usan (cómputo desperdiciado por ventana).
- [ ] Silenciar el `logger.critical` falso de DETECTION-3.
- [ ] **Re-validar si raw sigue ganando a calibrado EN LONG-ONLY** — el "66% WR" venía de la config `both`; este run long-only va a ~44% WR, así que la premisa puede no sostenerse.

---

## 4. 🔴 MetaLabeler sin edge y, en OOS, INVERTIDO (palanca MAYOR)
```
VEREDICTO: corre bien (RF árbitro real, no passthrough, no mock) pero NO discrimina; señal anti-correlacionada en OOS.
```
**Evidencia 1 (entrenamiento, robusto):** `val_win_rate = 0.51–0.53` en las 4 ventanas → el meta-modelo (LSTM-extractor + RF-300, input_dim=19) predice ganadores apenas mejor que cara/cruz.

**Evidencia 2 (calibración):** meta-calibrador isotónico → `std_output 0.004–0.024` → aplasta la salida a `[0.70-0.77]` ≈ base-rate de la población que pasa XGBoost (~0.73).

**Evidencia 3 (OOS, N=27 W1-W4, EXPLORATORIO p=0.053):** `meta_v2_prob` vs return → `rho = -0.376` (INVERSO):
| Tercil meta_v2_prob | N | WR% | ret_medio/trade |
|---|---|---|---|
| ALTO (más confianza) | 9 | 33% | **−0.95%** |
| MEDIO | 9 | 56% | −0.17% |
| BAJO | 9 | 44% | +0.37% |

→ A mayor confianza del metalabeler, PEOR el trade. (xgb_prob crudo: rho=+0.10 p=0.63, no-monótono; el tercil bajo es el peor → algo de señal débil en el extremo bajo.)

**Causa raíz:** el meta-modelo no separa ganadores/perdedores (val WR 0.52); la calibración refleja esa ausencia de edge colapsando al base-rate; la variación residual es ruido que en esta muestra anti-correlaciona.

> ⚠️ **CORRECCIÓN (ver §6.1):** dije "sello pasa-todo". El FILTER AUDIT lo **refuta**: el Meta es un filtro agresivo (bloquea **5–88% por ventana, media ~48%**), no pasa-todo. Pero corrijo también la sobre-corrección: **NO bloquea 88% siempre** (eso era W6); en W4 solo 5%. Es un **sobre-filtro sin edge** co-dominante con OOD/Embargo, que en algunas ventanas mata regímenes ganadores (extinción de `1_BULL_TREND`).

**Tests pendientes (post-baseline):**
- [ ] **Confirmar la inversión con las 3 seeds completas** (subir N por encima de 30; p=0.053 es limítrofe).
- [ ] **A/B real:** desactivar el filtro MetaLabeler (dejar XGB + régimen + silenciador) vs mantenerlo, mismo baseline. Candidato a aportar más que cualquier veto de régimen.
- [ ] Investigar por qué no hay edge: features del meta (19), calidad de la meta-label TBM, aporte real del LSTM-extractor (rolling_stats) sobre xgb_prob solo.

---

## 4.5 Resultados preliminares (VIVO — seed42 parcial)
> ⚠️ 1 sola seed, parcial. N por ventana minúsculo (W3=3 trades). Exploratorio hasta tener las 3 seeds.

**seed42 W1-W5 agregado: N=31, WR 41.9%, ret_sum −10.43%, ret_medio −0.337% → BASELINE EN NEGATIVO.**
Confirma `env-recovery`: la run fresca no-cache long-only es el régimen degradado, NO la referencia cacheada (64 tr / 59% era `both`+caché). Este es el baseline honesto.

**Edge por régimen (nítido, accionable):**
| Régimen | N | WR% | ret_medio% |
|---|---|---|---|
| 1_BULL_TREND | 9 | 67 | **+0.948** |
| 2_CALM_RANGE | 4 | 75 | **+0.523** |
| 3_CALM_BEAR | 3 | 33 | −0.314 |
| 2_VOLATILE_RANGE | 2 | 50 | −0.549 |
| 1_VOLATILE_BULL | 8 | 12 | **−1.294** |
| 1_VOLATILE_BULL_B | 5 | 20 | **−1.734** |

→ Los 13 trades VOLATILE_BULL (8+5) suman ≈ −19%; BULL_TREND+CALM_RANGE ≈ +10.6%. **Vetar volátiles daría la vuelta al baseline.** Valida L1 (veto régimen) Y conecta con el MetaLabeler roto: debería haber filtrado esos volátiles perdedores y no lo hace.

**Forense (desde 09:38):** GUARDIAN FATAL 0 · Modo Degradado 0 · Tracebacks 0 · ALERTA cal==raw 5 (=passthrough esperado, 1/ventana). Run limpia.

---

## 6. 🚨 HALLAZGO MAYOR — el stack de filtros secundarios destruye un baseline rentable, y la red de seguridad está MUERTA
> Metodología SOP: hipótesis → test sobre datos guardados de la run → causa raíz en código. **1 seed, 6/12 ventanas → EXPLORATORIO** salvo donde se marca SOLID (hecho de código).

### 6.1 El medidor de eficacia (FILTER AUDIT) — dónde mueren las señales
Embudo por componente al cierre de CADA ventana (`[DATA-01 FILTER AUDIT]`). **Bloqueo del Meta por ventana (corrección de rigor: NO es 88% constante, eso era solo W6):**

| Ventana | XGBoost pasa | OOD bloquea | **Meta bloquea** | Trades final |
|---|---|---|---|---|
| W1 | 330 | 41% | 55% | 9 |
| W2 | 461 | 53% | 25% | 8 |
| W3 | 75 | 44% | **80%** | 5 |
| W4 | 504 | 66% | **5%** | 9 |
| W5 | 611 | 61% | 27% | 5 |
| W6 | 69 | 38% | **88%** | 4 |
| W7 | 464 | 64% | 57% | 2 |

→ El Meta bloquea **5–88% (media ~48%)** — es un filtro MAYOR pero **no el único ni siempre el dominante** (en W4 el OOD bloqueó 66% y el Meta solo 5%). Es un **cascada**: OOD + Meta + Embargo reducen las señales XGBoost al **0.4–2.7% final**. En W3/W6 el Meta es el cuello y causa extinción de régimen (`[FUNNEL-REGIME-01]`: `1_BULL_TREND XGB=4→FINAL=0`). **Ningún filtro aislado es "EL culpable" — el problema es la cascada acumulada + el gobernador muerto que debería relajarla.**

### 6.2 Contrafáctico Meta-OFF vs Meta-ON (datos guardados, sin re-correr)
`oos_trades_xgb_baseline_W*` = señales XGBoost + embargo + TBM **SIN** filtros secundarios (`predict_oos.py:1911-1975`, mismos costos/barreras). Comparado con los trades finales en las **mismas 6 ventanas** (seed42):

| | N | WR% | ret_sum% |
|---|---|---|---|
| **XGBoost-solo (filtros OFF)** | 48 | **54.2** | **+7.93** |
| **Sistema actual (filtros ON)** | 35 | 42.9 | **−7.52** |

→ **El stack de filtros secundarios (dominado por el MetaLabeler) convierte una estrategia GANADORA (+7.9%) en PERDEDORA (−7.5%).** (No es un A/B perfecto: el embargo es path-dependent y los filtros incluyen OOD; pero el funnel atribuye el grueso al Meta, 88% vs 38%.)

### 6.3 La red de seguridad existe… pero está MUERTA por un bug de path dual-bot **[SOLID]**
Existe un `FilterGovernor` (`luna/models/filter_governor.py`) cuyo ÚNICO propósito es *detectar si DVOL/MetaLabeler censuran en exceso las baselines rentables y relajar los filtros* (cableado real: `signal_filter.py:660-668` relaja DVOL, `:1068` + `max_relaxation_meta_prob_delta:0.08` relajan el Meta).

**Con nuestros datos calcularía relajación = 1.0 (máxima).** Pero en la run actual loguea, ventana tras ventana:
```
[FILTER-GOVERNOR] Baseline acumulada en pérdida o cero (0.00%). Manteniendo filtros en máxima restricción. Relaxation = 0.0
```
**CAUSA RAÍZ (verificada):** `filter_governor.py:107-108` busca el baseline en `oos_trades_xgb_baseline_W{w}_seed{seed}.parquet` (SIN sufijo de dirección), pero los archivos reales del dual-bot llevan `_long` → **file-not-found** → lee `r_baseline=0.00%` → entra en la rama "baseline en pérdida o cero → máxima restricción" → **relajación congelada en 0.0 para siempre**. Es el mismo tipo de bug de naming dual-bot que el de pre-flight (memoria `env-recovery`). **Fallo silencioso clásico:** loguea "baseline 0.00%" (parece benigno) cuando en realidad es "no encuentro los datos".

### 6.4 Cadena causal del baseline negativo
```
XGBoost-solo rentable (+7.9%)
   └─> filtros secundarios sobre-censuran (Meta bloquea 88%, sin edge val 0.52, mata BULL_TREND)
         └─> resultado −7.5%
               └─> el FilterGovernor DEBERÍA relajarlos (relax=1.0)…
                     └─> …pero lee baseline=0.00% por bug de path `_long` -> relax=0.0 -> nada se corrige
```

### 6.5 ✅ FIX APLICADO Y VALIDADO (2026-06-26) — 2 bugs en el FilterGovernor
Al validar el fix del path in-pipeline apareció un **SEGUNDO bug** en el mismo componente:

| Bug | Causa | Efecto | Fix |
|---|---|---|---|
| **#1 Path `_long`** | `filter_governor.py:107-108` buscaba `..._seed{seed}.parquet` sin sufijo dirección | file-not-found → `r_baseline=0.00%` | `[FIX-GOVERNOR-PATH-01]` resuelve `_{direction}` (env `LUNA_DIRECTION`→`cfg.fase2.direction_mode`) + fallback; fail-loud si no halla datos (R16/A1) |
| **#2 Doble-coste** | `return_raw` YA es neto (`predict_oos.py:1815/1948: ret_bruto = ret − _GLOBAL_COST_RT`); el governor restaba `cost_rt` otra vez | baseline real +7.93% → −4.07% (con N=48) → seguía sin relajar | `[FIX-GOVERNOR-COST-01]` usa `return_raw.sum()` sin restar coste |

**Validación in-pipeline** (no código muerto): instanciado `FilterGovernor(seed=42)` con los datos en disco →
```
R_baseline = 7.93%   R_filtered = -7.52%   Ratio = -1.35   relaxation_factor = 1.0
```
El governor reproduce EXACTAMENTE el contrafáctico de §6.2 (+7.93% / −7.52%) de forma independiente → **doble confirmación de que esos números eran correctos y netos** (mi −4.07% intermedio era el artefacto del doble-coste, no el baseline real). Ahora el governor detecta el sobre-filtrado y aplicará relajación (acotada: `meta_prob_delta 0.08`, `percentile_delta 0.20`).

**Run relanzada** (`20260626_115409`, 3 seeds long-only `--nocache`, governor vivo, settings capado a 3). Esta es la nueva referencia: el sistema operando como DEBÍA. NOTA: la relajación es acotada — puede no dar la vuelta total al baseline; la palanca grande (P2 Meta A/B) viene después.

---

## 7. Síntesis y palancas (por prioridad)
El baseline negativo **no es que el modelo no tenga señal** — XGBoost-solo es rentable. El problema es el **stack de filtrado secundario que destruye esa señal, sin la red de seguridad que debería auto-corregirlo**.

| # | Palanca | Tipo | Confianza | Acción (post-baseline, NO ahora) |
|---|---|---|---|---|
| **P1** | ~~Arreglar `FilterGovernor`~~ **✅ HECHO (2 bugs: path + doble-coste)** | bug-fix | **ALTA (SOLID)** | ver §6.5 — aplicado y validado (relaxation 0→1.0); en observación en la run nueva |
| **P2** | **A/B desactivar/aligerar filtro MetaLabeler** | pipeline | ALTA | re-run con Meta off (o umbral relajado) vs on, mismo baseline 3-seed |
| **P3** | Confirmar contrafáctico con 3 seeds (N≥30) | medición | — | repetir §6.2 con seed100/777 |
| **P4** | Investigar por qué el Meta no tiene edge | investigación | media | features(19)/meta-label TBM/aporte LSTM |
| **P5** | Limpieza calibrador (no-op intencional) | bug-fix menor | baja | §3 |

> El veto de régimen del plan original (L1) queda **subsumido**: si el Governor relaja y/o el Meta se arregla, el filtrado por régimen se auto-regula. Atacar la causa (filtro+governor) > parchear síntomas (veto manual).

---

## 8. Checklist de revisión al CERRAR la run
- [ ] Anotar baseline final por seed: trades, WR, Sharpe, Calmar, DSR, edge por régimen (panel `measure_long_run.py`).
- [ ] **Confirmar §6.2 contrafáctico (Meta-OFF +7.9% vs ON −7.5%) con las 3 seeds** (N≥30).
- [ ] Confirmar el bug del FilterGovernor en seed100/777 (mismo log "baseline 0.00%").
- [ ] Orden de tests propuesto: ~~P1 (fix Governor)~~ ✅ → **P2 (A/B Meta)** → P5 (calibrador).
- [ ] Verificar que el run paró en 3 seeds (no generó aleatorias) — ahora capado en settings.

---

## 9. Acciones tomadas (changelog 2026-06-26)
| # | Acción | Archivo / detalle | Estado |
|---|---|---|---|
| 1 | Detener run baseline rota (governor muerto) | orquestador PID 21616 + worker + watcher | ✅ |
| 2 | Auditoría de rigor + corrección sobre-ajuste Meta | §4/§6.1 (88% → 5-88% media ~48%) | ✅ |
| 3 | Review SOP → addendum A1-A5 | `.agents/rules/sop_v10_rules.md` | ✅ |
| 4 | Fix FilterGovernor **bug #1 (path `_long`)** | `filter_governor.py` `[FIX-GOVERNOR-PATH-01]` + fail-loud (R16/A1) | ✅ |
| 5 | Fix FilterGovernor **bug #2 (doble-coste)** | `filter_governor.py` `[FIX-GOVERNOR-COST-01]` | ✅ |
| 6 | Validación in-pipeline del governor | relaxation 0→1.0; baseline +7.93% reproducido | ✅ |
| 7 | Cap a 3 seeds (testeo) | `settings.yaml` `max/min/active_seeds=3` `[CAP-3-TEST]` | ✅ |
| 8 | Auditorías post-cambio | `audit_parametros_fijos` limpio; pre-flight 26/27 PASS (TEST-82 resuelto) | ✅ |
| 9 | Relanzar baseline con governor vivo | run `20260626_115409`, 3-seed long `--nocache` + watcher `bf8diwgvl` | ✅ |
| 10 | Pendiente menor | actualizar grafo graphify (toqué `filter_governor.py`) | ⏳ |

> **Decisión de 3 seeds:** ver caja en §1 — es velocidad de iteración + instrucción del usuario, NO óptimo estadístico. 3 seeds = SCREENING; 29 + ensamble = SIGN-OFF.
