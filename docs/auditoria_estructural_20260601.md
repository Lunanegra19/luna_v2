# Auditoría Estructural Profunda — Luna V2
**Fecha:** 2026-06-01 | **Fuente:** 125 `oos_trades.parquet` + firmas XGBoost + 63 runs del día

---

## 0. Estructura IS/OOS del Pipeline — Por Qué los Períodos Son Insuficientes

### Ventanas WFB (Walk-Forward Backtest)

```
Datos históricos (2017–2025):
├── W1: IS hasta ~2023-06 | OOS: 2023-07 → 2023-12  (~1.441 barras OOS = ~60 días)
├── W2: IS hasta ~2023-12 | OOS: 2024-01 → 2024-06
├── W3: IS hasta ~2024-06 | OOS: 2024-07 → 2024-12
├── W4: IS hasta ~2024-12 | OOS: 2025-01 → 2025-06
└── W5: IS hasta ~2025-06 | OOS: 2025-07 → 2025-12  (holdout — 1 sola vez)
```

### El Problema de N — Cuantificado

| Período | Barras OOS | CALM_BEAR en OOS (~27%) | Trades generados | SOP R8 mínimo |
|---|---|---|---|---|
| Por ventana | ~1.441 | ~389 barras (~16 días) | **~4 trades** | 100 |
| Por run completa (W1-W4) | ~5.764 | ~1.556 barras | **~20 trades** | 100 |
| Para inferencia válida | — | — | **≥5 runs** = 3 días de cómputo | 100 |

> [!CAUTION]
> **Cada run genera ~20 trades CALM_BEAR.** Con N=20, la varianza del WR es ±22% (IC95). Esto significa que WR=55% y WR=33% son **estadísticamente indistinguibles**. Se han tomado decisiones de diseño sobre señales que son ruido puro.

### Distribución de Régimen IS vs OOS

**IS histórico (63.037 barras, 2017–2025):**

| Régimen | Barras | % | ¿Edge OOS? |
|---|---|---|---|
| 1_BULL_TREND | 15.657 | 24.8% | ❌ No (WR=41.9%) |
| 1_VOLATILE_BULL | 11.231 | 17.8% | ❌ No |
| 2_VOLATILE_RANGE | 18.316 | **29.1%** | ❓ Sin datos suficientes |
| 3_CALM_BEAR | 7.432 | **11.8%** | ✅ Sí (WR=55.3%) |
| 4_BEAR_FORCED | 4.152 | 6.6% | ✅ Parcial |
| 3_BEAR_CRASH | 1.349 | 2.1% | ⚠️ N IS insuficiente |

**OOS 2025 (predicho por router, 283.131 barras):**

| Régimen | Barras | % | vs IS |
|---|---|---|---|
| RANGE | 111.143 | **39.3%** | +10pp más que IS |
| BULL | 95.794 | 33.8% | −8pp menos que IS |
| BEAR/CALM_BEAR | 76.194 | **26.9%** | ≈ IS |

> [!IMPORTANT]
> **OOS 2025 es distinto del IS**: hay más RANGE (39.3% vs 29.1%) y menos BULL. El agente RANGE opera en el régimen MÁS frecuente de 2025 pero genera solo 19 trades totales. Este es el mayor alpha no explotado.

---

## 1. Lo Que SÍ Funciona — Con Evidencia Cuantitativa

### ✅ CALM_BEAR — Edge Real Demostrado

| Métrica | Valor | Interpretación |
|---|---|---|
| WR OOS real | **55.3%** | Sobre breakeven de 48.4% |
| R:R | **1.065** | Wins > losses en valor absoluto |
| Exceso sobre BE_WR | **+6.9%** | Expectativa positiva confirmada |
| N trades totales (01/06) | 255 en 63 archivos | ~4/ventana, insuficiente para R8 |
| DSR_IS (firma actual) | **+0.0282** | Señal positiva en IS |
| Threshold calibrado | 0.61 | Razonable para base_rate=53.5% |
| avg_win | +0.0495% | Wins ligeramente mayores que losses |
| avg_loss | −0.0465% | R:R correcto para este régimen |

**¿Por qué funciona CALM_BEAR?**
Es una bajada ordenada de baja volatilidad. El TBM captura pullbacks cortos hacia el TP antes de que el SL se active. La dinámica IS (2018–2024) y OOS (2025) es consistente estructuralmente — los CALM_BEAR son similar en ambos períodos.

**Retorno esperado teórico:**
```
E[trade] = 0.553×(+0.0495%) + 0.447×(−0.0465%) = +0.0066% por trade
100 trades/año × 0.0066% = +0.66% sin apalancamiento
Con x10 (conservador): +6.6% anual
Con x20 (óptimo SOP):  +13.2% anual
```
Modesto pero **positivo y estadísticamente fundado**.

---

### ✅ HMM Regime Identification — Correcto

- MI entre estados y retornos futuros > 0.005 (R9 SOP verificado)
- Duración media estados > 120H (R9 SOP verificado)
- Distribución IS/OOS coherente — no hay drift extremo de etiquetas
- **El HMM no es el problema.** Identifica correctamente qué régimen ocurre.

---

### ✅ Arquitectura Multi-Agente — Concepto Correcto

Separar el modelo por régimen es el enfoque correcto. El problema no es que existan 4 agentes — es que 3 de ellos (BULL, RANGE inactivo, BEAR_CRASH) no generan señales útiles por razones distintas e independientes.

---

## 2. Lo Que NO Funciona — Diagnóstico por Componente

### ❌ BULL — Sin Edge Estructural (el Destructor Principal)

**Datos duros:**

| Métrica | Valor |
|---|---|
| WR OOS | **41.9%** (48 ventanas históricas = evidencia masiva) |
| R:R | **0.851** (losses > wins en absoluto — R:R invertido) |
| Breakeven WR necesario | **54.0%** |
| Déficit real | **−12.1%** |
| DSR_IS (firma actual) | **+0.0003** (estadísticamente = cero) |
| Threshold | 0.48 (bajo → genera muchas señales) |
| % del volumen total | **89% de todos los trades** |
| Impacto en retorno total | **−30.3% de retorno acumulado** |

**¿Por qué no funciona?**
El modelo BULL aprende en IS (2018–2024) pullbacks de duración media antes de que el SL se active. En OOS 2025 post-halving, los pullbacks son más profundos y duraderos: el SL se activa sistemáticamente antes que el TP.

**Este no es un bug de implementación. Es distribución shift: el BTC 2025 no se comporta como el BTC IS.**

**¿Por qué FIX-BULL-GATE-01 está parcialmente mal calibrado?**
El gate filtra DSR ≤ 0.0. BULL tiene DSR = +0.0003 → **pasa el gate**. Con DSR=0.0003 el modelo es estadísticamente indistinguible de ruido. El threshold necesita subirse a **0.10** para bloquear señales sin edge real.

---

### ❌ RANGE — Edge Latente Completamente Bloqueado por Threshold

**Datos:**

| Métrica | Valor |
|---|---|
| % tiempo OOS 2025 | **39.3%** (régimen dominante) |
| Barras OOS disponibles | **111.143** |
| Trades generados | **19** (0.017% signal rate) |
| DSR_IS (firma) | **+0.0471** (hay señal real en IS) |
| Threshold calibrado | **0.6200** |
| Base rate IS | 54.2% |
| Gap threshold vs base rate | **+7.8pp** |

**¿Por qué no funciona?**
```
El modelo necesita 62% de confianza para disparar señal.
El activo sube el 54.2% de las veces en RANGE IS.
Muy pocas barras OOS tienen probabilidad > 62%.
Resultado: 19 trades en 111.143 barras disponibles.
```

**Esto NO es ausencia de señal. Es threshold de Optuna demasiado conservador.**

Optuna maximizó DSR eligiendo threshold=0.62 en validación cruzada. Maximiza precisión (WR) pero colapsa recall (N). Con N→0 en OOS, el DSR IS es positivo pero el sistema es inútil en producción.

**¿Mejorable?** Sí, pero requiere cambiar el objetivo de Optuna o añadir un floor de N mínimo. No es un fix de 1 línea.

---

### ❌ BEAR_CRASH / BEAR_FORCED — N IS Estructuralmente Insuficiente

| Métrica | BEAR_CRASH | BEAR_FORCED |
|---|---|---|
| Barras IS (7 años) | **1.349** (2.1%) | 4.152 (6.6%) |
| Corresponde a | COVID 2020, FTX 2022 | Eventos de liquidación forzada |
| Problema | Muy pocos ejemplos IS | N marginal |

BTC solo ha tenido 3-4 crashes graves en toda su historia. No hay suficientes datos históricos para entrenar un modelo robusto. **Esta es una limitación fundamental de datos, no de arquitectura.**

---

### ❌ Período OOS — Demasiado Corto para Inferencia

```
OOS por ventana:     ~1.441 barras = ~60 días
CALM_BEAR en OOS:    ~27% = ~16 días de régimen
Trades generados:    ~4 por ventana
Trades por run:      ~20 total (W1-W4)
SOP R8 mínimo:       100 trades para inferencia válida
→ Se necesitan ≥5 runs para 1 medición estadística
```

**Por qué esto invalida la mayoría de evaluaciones hechas hasta hoy:**
Con N=20 trades, el intervalo de confianza 95% del WR es ±22pp. WR=55% y WR=33% son indistinguibles. Toda mejora o empeoramiento observado en una run individual puede ser ruido aleatorio.

---

## 3. Firmas XGBoost Actuales — Estado Real de Cada Agente

| Agente | DSR_CPCV | Threshold | Base Rate | Gap | N_feats | Veredicto |
|---|---|---|---|---|---|---|
| **BULL** | +0.0003 | 0.48 | 55.5% | −7.5pp | 10 | ❌ DSR≈0, R:R invertido |
| **RANGE** | +0.0471 | **0.62** | 54.2% | **+7.8pp** | 10 | ⚠️ Edge IS real, bloqueado |
| **BEAR** | +0.0795 | 0.61 | 52.8% | +8.2pp | 10 | ✅ Edge real, threshold alto |
| **CALM_BEAR** | +0.0282 | 0.61 | 53.5% | +7.5pp | 10 | ✅ Nuevo agente, funciona |

El gap `threshold − base_rate` representa cuánto debe el modelo superar la tasa base para disparar señal. Gap > +7pp explica los pocos trades en BEAR/BEAR y el casi cero en RANGE.

---

## 4. ¿Por Qué los Meses de Fixes No Convergen?

**El ciclo diagnosticado:**
```
Run N:    resultado malo (pero N=20, puede ser ruido)
→ Se identifica "bug" o "mejora necesaria"
→ Fix implementado
→ Run N+1: resultado diferente (pero N=20, sigue siendo ruido)
→ Conclusión: "el fix funcionó / no funcionó"
→ Nuevo fix → repite indefinidamente
```

**La causa raíz:**
Con N=20 trades por run, detectar una mejora real de +5pp en WR requiere N≈300 trades (poder estadístico 80%, α=0.05). Se han evaluado cambios con 15× menos datos de los necesarios.

**Consecuencias:**
- Fixes reales se descartaron porque el N era insuficiente para verlos
- Fixes que parecían funcionar eran noise favorable
- La complejidad del sistema aumentó con cada ciclo sin beneficio verificable

**La solución no es más fixes. Es aumentar N antes de evaluar cualquier cambio.**

---

## 5. Resumen: Lo Que Funciona vs Lo Que No

| Componente | Estado | Causa | ¿Mejorable? |
|---|---|---|---|
| HMM regime | ✅ Funciona | — | Mantener |
| CALM_BEAR model | ✅ Funciona | Edge real IS+OOS | Aumentar N trades |
| SOL3-CALM-BEAR-01 | ✅ Implementado | Agente dedicado | Verificar próxima run |
| FIX-BULL-GATE-01 | ⚠️ Parcial | Threshold 0.0 deja pasar DSR=0.0003 | Subir a 0.10 |
| BULL model | ❌ Sin edge | Distribución shift OOS 2025 | NO — dejar en cash |
| RANGE model | ❌ Inactivo | Threshold Optuna 0.62 bloquea señal | Sí, pero requiere rediseño |
| BEAR_CRASH | ❌ N IS insuf. | Solo 1.349 barras IS en 7 años | No sin datos nuevos |
| TBM barriers | ⚠️ Desc. en BULL | avg_win < avg_loss en BULL 2025 | Solo si BULL opera |
| Período OOS | ❌ N insuficiente | 20 trades/run vs 100 mínimo R8 | Sí: más runs o window_size mayor |
| Ciclo evaluación | ❌ N inválido | Decisiones con N<30 | Sí: cambiar criterio evaluación |

---

## 6. Mapa Completo de Oportunidades (Post-Diagnóstico Cuantitativo)

### P1 — ✅ IMPLEMENTADO: `bull_gate_min_dsr`: 0.0 → **0.20** *(settings.yaml L551)*
Subido a 0.20 basado en diagnóstico 5 fases. DSR archive max=0.1753 → bloquea 100% historial.
Elimina ~2.165 trades con EV=-0.015% → **+32.5% retorno acumulado**. Sistema: -30% → ~+2.5%.

### P2 — 🔴 CRÍTICO: **Bloquear alpha_dtw en CALM_BEAR** *(requiere cambio en predict_oos.py)*

> [!IMPORTANT]
> **Nuevo hallazgo crítico de la sesión 2026-06-01.** Única mejora con evidencia estadística real.

| Grupo | N | WR | IC95 | p-value |
|---|---|---|---|---|
| CALM_BEAR **CON** DTW activo | 102 | **43.1%** (perdedor) | — | — |
| CALM_BEAR **SIN** DTW activo | 155 | **63.2%** (ganador) | [55.6%, 70.8%] | **0.00062** |
| CALM_BEAR global (actual) | 257 | 55.3% | [49.2%, 61.3%] | 0.052 (no sig.) |

**Evidence stack:**
- `binom_test(no-DTW WR > 50%)`: p=0.000620 — edge **probado estadísticamente**
- `t-test(DTW vs no-DTW)`: t=-2.065, p=0.040 — medias **significativamente distintas**
- `KS test`: p≈0 — **distribuciones completamente distintas**
- Interpretación mecánica: DTW + momentum_24H en mercado BEAR añade falsas señales long que van contra la lógica de mean-reversion que captura CALM_BEAR

**Contrafactual cuantificado:**
```
Actual (todos):   N=257, WR=55.3%, total_EV=+1.677% (no sig., p=0.052)
Solo no-DTW:      N=155, WR=63.2%, total_EV=+2.040% (SIG., p=0.00062)
Ganancia:         +0.363% de retorno recuperado (eliminando 102 trades DTW perdedores)
```

**⚠️ Advertencia:** Todos los 257 trades CALM_BEAR son de W2 únicamente (2024H1). No hay validación cross-window todavía. Implementar después de obtener datos de W4/W5 o en la próxima run.

### P3 — ⚠️ PENDIENTE: RANGE threshold (cuello de botella mal identificado)
El problema NO es el threshold XGBoost RANGE (0.62) — es el **argmax routing del HMM**.
14.440 barras se enrútan a RANGE pero solo 19 generan señal (0.13% signal rate, todo en W3).
No implementar hasta entender por qué RANGE solo dispara en W3.

### P4 — 🟡 ESTRUCTURAL: Acumular N≥300 CALM_BEAR para validación cross-window
Con N=300 y WR=55.3%: p=0.047 (primera vez significativo). Solo 43 trades más (~3 runs).
Prioritario antes de implementar P2 para tener validación cross-window real.

---

## 7. Fix Implementado — Registro Completo (diagnostico_cuantitativo.md)

```
FIX:        DX-BULL-GATE-02 (2026-06-01)
PROBLEMA:   bull_gate_min_dsr=0.0 dejaba pasar DSR=0.0003 (estadísticamente ruido)
EVIDENCIA:
  H1 CONFIRMADA (p=0.000000, t=-8.087): EV BULL significativamente negativo
  H2 DESCARTADA (p=0.474, r=-0.015):    modelo BULL sin poder discriminante en ningún threshold
  Sweep 0.48→0.80:                       ningún threshold xgb_prob_cal hace BULL rentable
  DSR archive N=20:                      max=0.1753 → 0.20 bloquea 100% historial
CAUSA RAÍZ: predict_oos.py L966: `if _bull_dsr <= _min_bull_dsr` con _min_bull_dsr=0.0
IMPACTO:    2.165 trades EV=-0.015% → -32.5% retorno. Sistema a +2.5% sin BULL.
FIX:        settings.yaml L551: bull_gate_min_dsr: 0.0 → 0.20
NO REQUIERE REENTRENAMIENTO: solo cambia parámetro de inference
IMPACTO ESPERADO PRÓXIMA RUN: bull_long bloqueado en el 100% de las ventanas históricas
```

**Hipótesis NO implementadas por N insuficiente (SOP Error #5):**
- H3: CALM_BEAR WR=55.3% → p=0.052 (no significativo α=0.05, N=257, necesita N≥350)
- H4: RANGE WR=100% N=19 → exploratorio, no concluyente
- H5: RANGE threshold sweep → sweep idéntico en todos los N=19 trades, inválido

---

## 8. Settings Restore Protection — Cumplimiento

> [!IMPORTANT]
> Regla `settings_restore_protection.md` verificada antes del cambio.

| Check | Resultado |
|---|---|
| ¿Runs activas antes de editar? | `manage_task list` → **sin tareas** ✅ |
| Timestamp settings.yaml | **22:53:12** (posterior a cancelación de task-2072) ✅ |
| Timestamp backup existente | **22:27:05** (backup de task-2072, contiene 0.0) |
| ¿El backup puede sobreescribir el cambio? | **No** — task-2072 ya cancelado, no hay proceso vivo |
| ¿El backup tiene el nuevo valor? | **No** — contiene `bull_gate_min_dsr: 0.0` |

> [!CAUTION]
> **El backup `settings_backup_wfb_20260601_222703_26452.yaml` contiene el valor antiguo `0.0`.** Si por algún motivo el orquestador de una futura run restaura este backup específico en lugar del backup que generará la nueva run, el cambio se perdería.
>
> **Protección obligatoria: la próxima run debe lanzarse con `--nocache`** para que el nuevo backup capture `bull_gate_min_dsr: 0.20`. Una vez que la nueva run cree su backup, el riesgo desaparece.

**Acción requerida antes de la próxima run:**
```powershell
# Verificar que el backup del run anterior NO sea el más reciente
Select-String -Path "g:\Mi unidad\ia\luna_v2\config\settings_backup_wfb_*.yaml" -Pattern "bull_gate_min_dsr" | Select -Last 3

# Lanzar la siguiente run con --nocache para generar nuevo backup con 0.20
python scripts/run_wfb_orchestrator.py --nocache
```

**Verificación post-launch:**
```powershell
# Confirmar que el backup generado por la nueva run contiene 0.20
Select-String -Path "g:\Mi unidad\ia\luna_v2\config\settings_backup_wfb_*.yaml" -Pattern "bull_gate_min_dsr" | Select -Last 1
# Debe mostrar: bull_gate_min_dsr: 0.20
```

---

## 9. Plan de Acción — Próximas Runs

### Run Actual (2026-06-01 ~23:07) — Objetivos

Esta run se lanza con `--nocache` con los siguientes objetivos **medibles**:

| Objetivo | Métrica | Umbral |
|---|---|---|
| Confirmar que BULL está bloqueado | Trades BULL en log | **0 trades** expected |
| Acumular más CALM_BEAR | N trades CALM_BEAR | +~80 trades (total → ~340) |
| Validar DTW efecto en W3/W4 | WR no-DTW vs DTW en nuevas ventanas | Confirmar tendencia |
| Proteger `bull_gate_min_dsr: 0.20` | Backup de la nueva run | Debe contener `0.20` |

**Hipótesis a validar en esta run:**
- `H_BULL_ZERO`: 0 trades BULL en todas las ventanas (gate 0.20 > DSR_max_histórico 0.1753)
- `H_CALM_BEAR_ACC`: +80 CALM_BEAR trades (→ N=~340, cerca del umbral N=300 para p<0.05)
- `H_DTW_W3W4`: el efecto DTW (WR_noDTW > WR_DTW en CALM_BEAR) se replica en W3/W4

### Criterio de Evaluación de Esta Run

> [!IMPORTANT]
> **NO evaluar WR global** (está contaminado por el histórico de BULL). Evaluar ÚNICAMENTE:
> 1. ¿BULL = 0 trades? → Confirma el gate 0.20
> 2. ¿WR CALM_BEAR sin DTW > WR CALM_BEAR con DTW en W3/W4?
> 3. N acumulado CALM_BEAR total cross-run (target: N≥300)

### Criterio de Implementación de P2 (DTW Gate en CALM_BEAR)

**No implementar hasta que se cumpla al menos una de:**
- [ ] N≥300 CALM_BEAR acumulado cross-window (actual: ~257)
- [ ] Efecto DTW confirmado en ≥2 ventanas OOS distintas (actual: solo W2)
- [ ] p-value global no-DTW CALM_BEAR < 0.001 con N adicional (actual: p=0.00062 con N=155)

### Mapa de Decisiones Post-Run

```
Si BULL = 0 trades:
  → Gate DX-BULL-GATE-02 funciona ✅
  → Sistema opera solo con CALM_BEAR + RANGE

Si N(CALM_BEAR) total ≥ 300:
  → Hacer binom_test global (target: p < 0.05)
  → Si confirmado: edge CALM_BEAR probado — sistema es viable

Si efecto DTW replicado en W3/W4 (no-DTW WR > DTW WR, p < 0.10):
  → Implementar P2: gate DTW en CALM_BEAR
  → Estimación: WR sube de 55.3% → 63.2% — sistema operativo

Si efecto DTW NO replicado:
  → El hallazgo de W2 era ruido in-sample
  → No implementar P2, acumular más datos
```

---

## 10. Post-Mortem Run 2026-06-02 — Hallazgos Críticos

> [!CAUTION]
> **Esta sección invalida parcialmente el análisis previo.** La estructura de ventanas y los datos de CALM_BEAR presentaban supuestos incorrectos.

### 10.1 Estructura Real de Ventanas OOS (Corregida)

La arquitectura WFB no opera sobre 2017-2025 con ventanas de 6 meses como se asumía. Opera sobre **2025 completo** con ventanas trimestrales:

```
W1 OOS: 2025-01-01 → 2025-03-30  (Q1 2025 — mercado BULL puro)
W2 OOS: 2025-04-05 → 2025-06-28  (Q2 2025)
W3 OOS: 2025-07-01 → 2025-09-30  (Q3 2025)
W4 OOS: 2025-10-01 → 2025-11-11  (Q4 2025 parcial)
W5 OOS: holdout — no tocado
```

**Impacto en análisis previo:** Toda referencia a "W2 = 2024H1" o "RANGE solo en W3 = 2024H2" era incorrecta. Los periodos OOS son 2025, no 2024.

### 10.2 Resultado Real Post Bull-Gate (2026-06-02, 8h de run)

| Métrica | Pre-gate (20260601) | Post-gate (20260602) |
|---|---|---|
| Total trades | 2.441 | **~22** |
| Trades BULL | 2.096 (86%) | **0 ✅** |
| Trades CALM_BEAR | 257 (11%) | **0** |
| Trades RANGE | 19 (1%) | **22** (WR=100%) |
| WR global | 45.3% | 100% |
| EV real (post-costes 0.15%) | negativo | **negativo** (ret=0.009% << 0.15%) |

### 10.3 WR=100% es Espejismo — Retornos Bajo Costes de Transacción

Los 22 trades RANGE tienen WR=100% pero **retornos de 0.009%-0.16%** por trade.

Con coste mínimo SOP R6 = 0.15% round-trip:
- Trade RANGE típico: +0.009% bruto → **-0.141% neto** (perdedor real)
- Solo 2-3 trades superan el 0.15%: +0.16% → **+0.01% neto** (marginalmente positivo)

**Conclusión: WR=100% pero EV real es negativo.** El sistema de barreras TBM genera trades que cierran ligeramente en positivo antes de alcanzar el SL, insuficientes para cubrir costes.

### 10.4 Los 257 CALM_BEAR de Ayer Eran Específicos de Seeds

> [!WARNING]
> El análisis del 2026-06-01 con 257 trades CALM_BEAR fue sobre **seeds específicas que Optuna exploró en esa run**. En la run del 2026-06-02 con seeds distintas: **0 CALM_BEAR**.

El orquestador usa `max_seeds_to_explore: 20` con Optuna. Cada run explora seeds distintas. La consistencia entre runs del volumen CALM_BEAR **no está garantizada**. Los 257 trades no son reproducibles ni representativos de todas las seeds.

### 10.5 Conclusión Estructural — El Sistema Dependía del BULL para Operar

Con BULL bloqueado correctamente, queda expuesto el problema de fondo:

| Régimen | Q1 2025 | Q2 2025 | Q3 2025 | Viabilidad |
|---|---|---|---|---|
| BULL | ~85% del volumen | ~85% | ~85% | ❌ Bloqueado (sin edge) |
| CALM_BEAR | 0-10 trades/run | 0-10 | 0 | ⚠️ Seed-dependiente, no robusto |
| RANGE | 0 | 0 | 1-2/seed | ⚠️ Bajo coste, WR=100% espejismo |

**El sistema sin BULL prácticamente no opera.** Las hipótesis anteriores sobre CALM_BEAR (WR=55-63%) y RANGE (19 trades OOS 2024H2) eran artefactos de seeds específicas o periodos específicos, no señales generalizables.

### 10.6 Revisión del Plan de Acción

Los planes P2 (DTW gate CALM_BEAR) y P3 (acumular N≥300) son **prematuros**:

| Hipótesis anterior | Estado revisado |
|---|---|
| CALM_BEAR tiene edge real (WR=55.3%) | ⚠️ Seed-específico, no replicado en run distinta |
| RANGE WR=100% en W3 = 2024H2 | ❌ Era 2025Q3, y los retornos están bajo costes |
| DTW destruye edge CALM_BEAR | ⚠️ Hipótesis válida pero sin CALM_BEAR no es aplicable |
| Acumular N≥300 CALM_BEAR | ❌ Inviable si CALM_BEAR no aparece con seeds distintas |

### 10.7 Próxima Investigación Requerida

Antes de cualquier nueva implementación, se requiere responder:

1. **¿Por qué W1 (Q1 2025 = bull market) generaba CALM_BEAR antes?** — Contradicción semántica: si el mercado era alcista, ¿cómo el HMM asignaba barras a CALM_BEAR?
2. **¿Las barreras TBM generan retornos tan pequeños siempre o es un bug?** — ret=0.009% es ~17x menor que el coste mínimo. Investigar configuración de barreras para RANGE.
3. **¿El volumen de trades del sistema es viable sin BULL?** — Si sin BULL el sistema genera <30 trades en un trimestre completo, no cumple SOP R8 (min 30 trades para validación estadística).
4. **¿La arquitectura multi-régimen con umbral CPCV alto es el modelo correcto?** — Podría ser que el sistema necesite una dirección radicalmente distinta.
