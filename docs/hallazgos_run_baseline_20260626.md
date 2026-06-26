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

## 6.6 🚨🚨 HALLAZGO CRÍTICO — el pipeline NO es reproducible (el ruido domina todo)
> Investigación 2026-06-26 (post-relanzamiento). Comparación Run A (093845, governor roto) vs Run B (115409, governor fixed), MISMA seed 42, MISMA ventana W1 (julio 2025), ambas `--nocache`.

**Nuestros cambios tuvieron efecto causal CERO en los resultados:**
- Governor: `relaxation=0.0` en TODAS las ventanas de AMBAS runs (A por el bug; B porque filtrado adecuado). Mismo output → 0 trades cambiados.
- Cap a 3 seeds: solo afecta el bucle del orquestador, no el trading por-ventana.

**La causa real de que Run B (WR 75%) ≠ Run A (WR 29%) en la MISMA ventana/seed:**
| Evidencia | Run A | Run B |
|---|---|---|
| XGBoost `prob_bear` (744 barras) | — | difiere hasta **0.34** vs A (NO determinista) |
| SFI features seleccionadas | 12 (incl. `genetic_rule_1`) | 13 (incl. `ae_feat_0/49/51`) — solo **7 comunes** |
| W1 trades | N=7, WR 29% | N=8, WR 75% |

**Cadena:** features regeneradas distintas → SFI elige distinto → XGBoost entrena distinto → señales distintas → WR 29%↔75%.

**RAÍZ (verificada en código, investigación 2ª pasada — descartado que sea bug del SFI):**
- **NO es bug del SFI**: `feature_selection_e.py` es determinista (lag discovery ya seedeado por MI-FIX 2026-03-16; MI/RF con `random_state`; KMeans tribes seedeado). El SFI **propaga** el no-determinismo, no lo crea.
- **NO es la data** (idéntica entre runs), **NO es XGBoost** (`tree_method='hist'` CPU + `random_state=42`), **NO es el HMM** (`random_state=_LUNA_SEED`).
- **SÍ es el AutoEncoder**: `autoencoder_features.py` corre en CUDA (`device('cuda')`) y **NO tiene `torch.manual_seed`**. No existe `seed_everything` central → torch quedó fuera del seeding (que todo lo demás SÍ aplica). Cada run → AE distinto → `ae_feat_*` distintas (comprimen 158→64 features) → SFI elige distinto → modelo distinto → **13/14 trades difieren**. Amplificado por N=7-8 (WR salta 29%↔75%).

**MATIZ CONCEPTUAL (importante):** reproducibilidad ≠ "todas las seeds iguales". Determinismo POR-seed (seed42 repetible) + diversidad ENTRE seeds (42≠100≠777, base del ensamble) son ambos deseables. Seedear el AE NO es overfitting (es determinismo de cómputo, ortogonal a generalización). La varianza enorme también revela un **edge frágil** → la robustez real viene del ENSAMBLE de muchas seeds, no del determinismo solo. Plan = reproducibilidad **+** muchas seeds.

**FIX (seguro):** `seed_everything(LUNA_SEED)` central cubriendo torch (`manual_seed`+`cuda.manual_seed_all`+`cudnn.deterministic`), o correr el AE en CPU (minúsculo). Validar: 2 runs frescas seed42 → `ae_feat_*` idénticas → mismos trades.

**IMPLICACIÓN:** el "+13%/WR 73%" de Run B es un golpe de suerte, no mejora real. Hasta cerrar la reproducibilidad + medir con muchas seeds, ningún resultado de 1 run es fiable. → **P0.**

### 6.7 Datos de Run B (governor-fixed, 20260626_115409) — la "racha de suerte" regresa a la media
seed42, W1-W6 (detenida tras esto para aplicar P0). Ambas runs son draws aleatorios del mismo proceso no-reproducible.

| W | N | WR% | ret_sum | Sharpe~ |
|---|---|---|---|---|
| W1 | 8 | 75.0 | +5.76% | +1.43 |
| W2 | 7 | 71.4 | +7.23% | +2.06 |
| W3 | 5 | 80.0 | +3.26% | +1.24 |
| W4 | 8 | 50.0 | −1.23% | −0.24 |
| W5 | 5 | 20.0 | **−9.06%** | −2.18 |
| W6 | 4 | 50.0 | +3.24% | +0.97 |
| **Agreg.** | **37** | **59.5** | **+9.20%** | — |

**Observación clave:** el agregado cayó de **WR 73% (W1-W2) → 59.5% (W1-W6)** conforme creció N → la racha inicial regresa a la media. Refuerza que los resultados tempranos eran ruido. (Run A W1-W5 ≈ WR 42%; el gap A↔B se estrecha con N, como debe pasar si la diferencia es varianza.)

**Forense (investigado a fondo):** 0 crashes. 2 circuit breakers MANEJADOS (Modo Degradado = deshabilitar agentes de régimen, NO_OPERABLE):
- **`GUARDIAN-05` (OOD Covariate Shift, KL>2.0 en 2/5 features top) → ventana W4, agente `range`.** El OOS de W4 difiere radicalmente del train → "el modelo no generalizará". Se capturó y W4 igual produjo 8 trades (WR 50%, −1.23% = ventana floja). Sus trades salen de un modelo marcado por shift.
- **`GUARDIAN-01` (Top-WR 38.2% ≤ Bottom-WR 52.1% = inversión) → ventana W7 → 0 trades** (ventana muerta, fuera del panel W1-W6). **GUARDIAN-01 está SANO en W1-W6** (Top>Bottom casi siempre: n=414 37→72%, n=1095 46→65%); solo W7 se invirtió → el guardián cazó la única mala.

**2 conclusiones:** (1) los guardianes FUNCIONAN y cazan los problemas ya identificados (GUARDIAN-01 = la misma inversión del §4; GUARDIAN-05 = edge frágil/OOD). (2) Son DATA-DRIVEN: Run A y Run B tuvieron **exactamente 1+1** guardianes → hay ventanas genuinamente duras (W4 covariate shift, W7 inversión) que cualquier run encuentra. Robusto y preocupante. Impacto en el panel: solo W4 marcada (floja, no infló); W7 fuera del panel.

**Conclusión operativa:** ambas runs son draws no-reproducibles → ninguna es "el baseline". Se detiene la run para aplicar P0 (determinismo del AE) antes de medir nada más.

### 6.8 ✅ FIX DE DETERMINISMO APLICADO (P0, 2026-06-26)
Al implementar el fix se descubrió que NO era 1 componente torch sino **3** (el resto del pipeline ya estaba seedeado):

| | Componente torch | Rol | Estado | Validación |
|---|---|---|---|---|
| **A** | `autoencoder_features.py` (DeepFeatureAutoEncoder → `ae_feat_*`) | features (causa de la divergencia de señales) | ✅ seedeado + cudnn determinista + shuffle seedeado | **byte-idéntico en GPU** (`max\|lat1−lat2\|=0.0`; seed distinta → distinto) |
| **B** | `train_autoencoder.py` (DenoisingAutoEncoder) | OOD Guard (filtrado) | ✅ `seed_everything()` + shuffle seedeado | corre en CPU (determinismo total) |
| **C** | `train_metalabeler_v2.py` (`_TempClassifier` torch) | meta-filtrado | ✅ `seed_everything()` antes del init | loader shuffle=False (orden temporal) |

- Helper central nuevo: `luna/utils/determinism.py` (`seed_everything` / `seeded_generator`). Tag `[FIX-AE-DETERMINISM-01]`.
- **Validación dura hecha:** misma seed → AE byte-idéntico; seed distinta → distinto (diversidad entre seeds preservada → confirma reproducibilidad ≠ "todas las seeds iguales", y NO es overfitting).
- Imports OK, audit 0 CRÍTICO/ALTO.
- **Prueba end-to-end PENDIENTE (al relanzar):** 2 runs frescas de seed42 deben dar `ae_feat_*` idénticas y **mismos trades**. Eso cierra el "completamente seguro" a nivel pipeline.

---

## 7. Síntesis y palancas (por prioridad)
El baseline negativo **no es que el modelo no tenga señal** — XGBoost-solo es rentable. El problema es el **stack de filtrado secundario que destruye esa señal, sin la red de seguridad que debería auto-corregirlo**.

| # | Palanca | Tipo | Confianza | Acción (post-baseline, NO ahora) |
|---|---|---|---|---|
| **P0** 🚨 | **REPRODUCIBILIDAD del pipeline** (seedear torch+CUDA en AE; forzar determinismo) | bug-fix | **CRÍTICA (SOLID)** | sin esto el ruido run-a-run (WR 29%↔75%) aplasta toda palanca. Ver §6.6. **Bloquea P2-P5.** |
| **P1** | ~~Arreglar `FilterGovernor`~~ **✅ HECHO (2 bugs: path + doble-coste)** | bug-fix | **ALTA (SOLID)** | ver §6.5 — aplicado y validado (relaxation 0→1.0); efecto causal CERO en resultados (devolvió 0.0 en ambas runs) |
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

---

## 10. Próximos pasos y revisiones (plan vivo)

### 10.1 Secuencia inmediata (en curso)
1. ✅ **Verificar 0 runs activas** antes de tocar nada.
2. ⏳ **Commit del fix de determinismo** (`luna/utils/determinism.py` + A/B/C + este doc). NO se bundlea nada más (un fix a la vez).
3. ⏳ **VALIDACIÓN END-TO-END (antes del baseline completo):** lanzar seed42 **dos veces** (`--seeds 42 --nocache`), dejar que cada una cierre W1, y comparar `oos_raw_probs_W1` (744 barras) + `oos_trades_W1`:
   - **PASA** si byte-idénticas → determinismo confirmado a nivel pipeline → seguir a 10.2.
   - **FALLA** → hay un residual (candidato: ruido de desempate del SFI `feature_selection_e.py:403` con `np.random` global) → cazarlo y arreglarlo ANTES de gastar horas.
4. ⏳ Decidir nº de seeds del baseline (P0.5: subir de 3 → 5-8 ahora que cada seed es reproducible).

### 10.2 Después del baseline reproducible
- **P2 — A/B MetaLabeler** (desactivar/aligerar el filtro) vs baseline reproducible. Ahora SÍ es medible (la diferencia será por la palanca, no por ruido del AE).
- **P5 — Limpieza calibrador** (deuda menor): dejar de entrenar calibradores no usados + silenciar el CRITICAL falso de DETECTION-3. No afecta resultados.

### 10.3 Revisiones / investigaciones abiertas (no bloqueantes)
- **Mapa de salud por ventana (guardianes como métrica de 1ª clase):** W4 sufre covariate shift (OOD, GUARDIAN-05) y W7 invierte (GUARDIAN-01). Tras el determinismo serán reproducibles → investigar QUÉ tienen esos periodos (oct-2025 / ene-2026) que rompen el modelo.
- **Inversión del modelo (§4 + GUARDIAN-01):** confirmar con N≥30 si el MetaLabeler/XGBoost anti-correlaciona; conecta con P2.
- **Robustez = ensamble de muchas seeds diversas** (no solo determinismo). El determinismo hace cada seed repetible; la señal real emerge al promediar muchas.
- **SOP addendum (§sop A1-A5):** reconciliar R10 (calibración desactivada), R18 (Sniper-Mode asume edge del Meta), proponer R20 (naming dual-bot centralizado).

### 10.4 Checklist de validación del fix de determinismo
- [x] AE (A) validado byte-idéntico en GPU (misma seed → idéntico; seed distinta → distinto).
- [x] B y C seedeados con el mismo patrón + helper central. Imports OK, audit 0 crítico.
- [ ] **End-to-end: 2× seed42 W1 → `oos_raw_probs`/`oos_trades` idénticos** (sella el "completamente seguro").
- [ ] Confirmar diversidad entre seeds (42 ≠ 100) en el baseline (no colapso).

---

## 11. 🔒 BASE DETERMINISTA (2026-06-26) — tag `determinismo-base-20260626`

Tras descubrir que dos runs frescas `--nocache` de la MISMA seed daban resultados distintos (WR 29% vs 75% en la misma ventana — §6.6), se cazaron y arreglaron **todas** las fuentes de no-determinismo. Validado con 18 runs de instrumentación + comparación byte a byte.

### 11.1 Estado conseguido
**PROBADO byte-idéntico** (checksums de instrumentación, run-a-run): **AutoEncoder, TODAS las features, matriz C-MI, y el ORDEN de columnas.** El núcleo del pipeline es reproducible.

### 11.2 Fuentes arregladas (tag `[FIX-AE-DETERMINISM-01]`)
| # | Fuente | Fix |
|---|---|---|
| 1-3 | **torch sin seed** (AE features, OOD AE, MetaLabeler clf) | `seed_everything()` (manual_seed+cuda+cudnn) + DataLoader generator seedeado. Helper `luna/utils/determinism.py`. |
| 4 | **`np.random.normal` global** en binning C-MI del SFI | RNG seedeado (`np.random.default_rng(_LUNA_SEED)`). |
| 5 | **LightGBM SHAP-RFE** (multi-hilo no reproducible) | `deterministic=True` + `force_row_wise=True` + `n_jobs=1`. |
| 6 | **XGBoost en GPU** (`device="cuda"`) en MetaLabeler CV | → `device="cpu"` (GPU XGBoost no-determinista). |
| 7 | **XGBoost scoring DSR** (`n_jobs=-1`) en el SFI | → `n_jobs=1`. |
| 8 | **Cachés CROSS-RUN** (`_dsr_cache.json` + `_lag_cache.json`) | Orquestador los limpia al inicio + el SFI NO los reusa en `--nocache` (guards `LUNA_NOCACHE`). |

### 11.3 Test guardián
`scripts/pre_flight/test_determinism.py` (DET-1..7) — verifica estáticamente que los 8 puntos siguen en su sitio. **Integrado en el orquestador**: corre al inicio de cada WFB y alerta `*** DETERMINISMO ROTO ***` si un fix futuro lo desalinea. Concepto: reproducibilidad POR-seed + diversidad ENTRE seeds (no es overfitting).

### 11.4 ⚠️ Residual conocido (aceptado como base)
Persiste **1 feature de 13** que oscila entre 2-3 métricas on-chain **altamente correlacionadas** (Puell/MVRV/mt_vol, mismo cluster del SFI). Descartado de: torch, np.random, threading, GPU, cachés DSR/lag. Está enterrado en la lógica de selección del SFI (descubrimiento de lags / SHAP-RFE / cuotas) — sutil, NO RNG. **Efecto despreciable** (1-2 trades de diferencia, WR similar) → no contamina el screening de palancas fuertes. Pendiente para el sign-off de 29 seeds si una palanca sutil lo requiere. Decisión 2026-06-26: aceptar 12/13 como base y proceder con palancas.

### 11.5 Reversión
Esta es la **base determinista buena**. Tag git: `determinismo-base-20260626`. Para revertir: `git checkout determinismo-base-20260626`.
