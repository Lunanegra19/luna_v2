# 🛡️ Plan de Implementación: Sistema Dual Bidireccional (Long/Short XGBoost)
# Luna V2 — Documento de Diseño Institucional

> [!IMPORTANT]
> **Prerequisito de Lectura del Código:**  
> Este plan ha sido elaborado tras un análisis exhaustivo del código real de Luna V2:
> - `luna/models/train_xgboost_v2.py` (L303-L323): La infraestructura `direction_mode` **ya existe** en el `XGBoostTrainer` con soporte parcial de `"long"`, `"short"` y `"both"` vía `config/settings.yaml → fase2.direction_mode`.
> - `luna/models/regime_router.py` (L85-L95): El `RegimeRouter` **ya acepta** el parámetro `direction: str = "long"` y construye los nombres de artefactos como `xgboost_meta_bull_long.model` / `xgboost_meta_bull_short.model`.
> - El sistema **ya está parcialmente diseñado para esto**. Lo que se propone aquí es completar la arquitectura.

---

## 0. Clarificación Arquitectural: ¿Cuántos Modelos XGBoost?

> [!IMPORTANT]
> **Respuesta directa:** En el sistema Dual Bot, NO hay un único modelo XGBoost que predice "Long o Short". Hay **dos familias de modelos XGBoost completamente especializados e independientes**, cada uno entrenado con su propia etiqueta de mercado:
>
> - **Escuadrón Long (29 seeds × 4 agentes):** Entrenados para detectar movimientos alcistas. Target = `1` si el precio toca la barrera superior (Take Profit) antes que la inferior (Stop Loss). Especializados por régimen: `bull_long`, `range_long`, `calm_bear_long`, `bear_long`.
> - **Escuadrón Short (29 seeds × 4 agentes):** Entrenados para detectar movimientos bajistas. Target = `1` si el precio toca la barrera *inferior invertida* (Take Profit en short) antes que la superior (Stop Loss). Especializados por régimen: `bull_short`, `range_short`, `calm_bear_short`, `bear_short`.
>
> Matemáticamente, un agente `bull_long` y un agente `bull_short` son dos XGBoost completamente distintos, entrenados en el mismo período histórico de mercado alcista, pero con etiquetas opuestas. Uno busca rebotes al alza dentro del bull, el otro busca pullbacks (correcciones) dentro del bull.

---

## 1. Arquitectura General del Sistema Dual

### 1.1 Filosofía de Diseño
El sistema Dual Long/Short NO es un único modelo que predice "Long o Short". Es una **competición entre dos escuadrones de ensambles especializados**:

```
┌──────────────────────────────────────────────────────────────┐
│                  LUNA V2 — DUAL BOT ARCH                     │
│                                                              │
│  ENTRADA: Barra horaria con HMM_Semantic asignado            │
│                          │                                   │
│                    ┌─────▼──────┐                            │
│                    │ RegimeRouter│                            │
│                    │ (Árbitro)   │                            │
│                    └──┬──────┬──┘                            │
│                       │      │                               │
│          ┌────────────▼─┐  ┌─▼───────────────┐              │
│          │ Squad LONG   │  │  Squad SHORT     │              │
│          │ (29 seeds)   │  │  (29 seeds)      │              │
│          │              │  │                  │              │
│          │ bull_long    │  │  bear_short       │              │
│          │ range_long   │  │  calm_bear_short  │              │
│          │ calm_bear_l  │  │  range_short      │              │
│          └──────┬───────┘  └──────┬────────────┘              │
│                 │                 │                           │
│          ┌──────▼─────────────────▼──────┐                    │
│          │      Alpha Arbitrage Layer     │                   │
│          │  (¿Qué señal gana? o ¿ambas?) │                   │
│          └──────────────┬────────────────┘                   │
│                         │                                    │
│                    Execution Engine                          │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Modos de Operación (Configurable en `settings.yaml`)
```yaml
fase2:
  direction_mode: "both"   # "long" | "short" | "both"
```

- **`long` (Default actual):** Solo opera el escuadrón Long (comportamiento actual).
- **`short`:** Solo opera el escuadrón Short. Útil para periodos bajistas o testing aislado.
- **`both`:** Ambos escuadrones conviven. El Alpha Arbitrage Layer decide la posición final.

---

## 2. Fase 1 — Target Engineering (Labeling Simétrico)

El cambio más crítico es en la **etiqueta del Triple Barrier Method (TBM)**.

### 2.1 Problema del TBM Actual (Long-Only)
El TBM actual en `train_xgboost_v2.py` (L738) usa:
```python
_side_val = -1.0 if self.native_direction == "short" else 1.0
_sides_series = pd.Series(_side_val, index=events_idx)
```
La etiqueta `target=1` significa *"el precio tocó el PT antes que el SL"*.

Para Longs: `target=1` ≡ precio sube → ganamos.  
Para Shorts: `target=1` **ya** ≡ precio baja → ganamos (el TBM ya invierte la barrera gracias a `_side_val=-1.0`).

**El código ya lo contempla.** El problema es que los modelos short nunca se entrenan porque nunca se llaman desde el orquestador.

### 2.2 Perfil TBM para el Escuadrón Short
Los regímenes bear tienen una asimetría temporal: **las caídas son más rápidas que las subidas**.  
Ajustes en `settings.yaml`:
```yaml
xgboost:
  regime_tbm_profiles:
    bull:       {pt_mult_min: 1.5, sl_mult_min: 0.8}   # Long: 1.5/1 riesgo asimétrico
    range:      {pt_mult_min: 1.0, sl_mult_min: 1.0}   # Long/Short neutral en rango
    calm_bear:  {pt_mult_min: 1.2, sl_mult_min: 1.0}   # Short: caídas graduales
    bear:       {pt_mult_min: 0.8, sl_mult_min: 1.5}   # Short: caídas rápidas y violentas
```
> [!NOTE]
> Para shorts, el perfil `bear` con `pt_mult_min=0.8, sl_mult_min=1.5` significa:
> - Tomamos beneficio rápido (0.8x ATR) porque las correcciones se revierten.  
> - Stop Loss amplio (1.5x ATR) para aguantar el ruido inicial antes de la caída.

---

## 3. Fase 2 — Pre-XGBoost Filters (Filtros Previos al Modelo)

Antes de que una señal entre al XGBoost, debe pasar filtros de plausibilidad de dirección.

### 3.1 Filtro F1: HMM Semantic Gate — Desbloqueado en Modo Dual

> [!IMPORTANT]
> **Este es el cambio de mayor impacto en el sistema Dual.** En el modo Long-Only actual, el bot tiene regímenes completamente prohibidos. En modo `both`, esos regímenes se convierten en **oportunidades de Alpha adicional** para el escuadrón opuesto.

#### El Valor Oculto que Estábamos Ignorando

En el modo Long-Only actual, el sistema entra en `HOLD` (0 trades, 0 PnL) durante los regímenes bajistas. El mercado bajista de 2022 duró **12 meses**. En ese tiempo, Luna V2 simplemente no operó. Eso es capital dormido durante un año entero.

El modo Dual cambia esto radicalmente:

**Long en Mercados Bajistas ("Rebotes en el Bear"):**  
Dentro de una tendencia bajista, el mercado tiene rebotes alcistas violentos y muy rentables. En 2022, BTC rebotó desde $17,500 a $25,000 (+42%) en 6 semanas, antes de caer de nuevo. El agente `bear_long` (entrenado específicamente en esos rebotes técnicos dentro de bear markets) puede capturar exactamente esa clase de movimientos que actualmente tienen **prohibición absoluta**.

**Short en Mercados Alcistas ("Correcciones en el Bull"):**  
Dentro de un bull market, el mercado tiene correcciones del 20-40% antes de reanudar la tendencia. En 2023-2024, BTC corrigió de $73K a $56K (-23%) en pocas semanas antes de seguir subiendo. El agente `bull_short` (entrenado específicamente en esas correcciones dentro de bull markets) puede capturar esas caídas puntuales mientras el sistema principal sigue siendo alcista en general.

#### Tabla de Regímenes Desbloqueados (Modo Both)

| Régimen HMM | Long-Only (Actual) | Short-Only | **Dual Both (Propuesto)** |
|---|---|---|---|
| `1_BULL_TREND` | ✅ Long Activo | 🚫 Bloqueado | ✅ Long + ⚡ Short oportunista (correcciones) |
| `2_CALM_RANGE` | ✅ Long Activo | ✅ Short Activo | ✅ Long + ✅ Short (árbitro elige) |
| `3_CALM_BEAR` | 🚫 **PROHIBIDO (Long)** | ✅ Short Activo | ⚡ Long (rebotes) + ✅ Short activo |
| `3_BEAR_CRASH` | 🚫 **CASH TOTAL** | ✅ Short Activo | ⚡ Long (rebotes fuertes) + ✅ Short activo |
| `4_BEAR_FORCED` | 🚫 CASH (todos) | 🚫 CASH (todos) | 🚫 CASH (todos — seguridad operativa) |

*Leyenda: ✅ = Activo y prioritario | ⚡ = Activo pero subordinado (consenso mínimo más alto) | 🚫 = Bloqueado*

> [!NOTE]
> **Mecanismo de Seguridad en Regímenes Desbloqueados:**  
> Los regímenes que antes eran `PROHIBIDOS` para un escuadrón no se desbloquean a plena potencia. El Árbitro exige un consenso mínimo más alto (por ejemplo, `>= 0.65` en lugar de `>= 0.55`) para activar la señal subordinada. Esto garantiza que solo operamos rebotes de alta convicción, no rebotes de ruido.

> [!CAUTION]
> `4_BEAR_FORCED` (flash crashes, cascadas de liquidaciones) bloquea **ambas direcciones** incluso en modo dual. El riesgo de slippage catastrófico en esos momentos invalida cualquier señal.

#### Impacto Cuantitativo Estimado del Desbloqueo

Asumiendo que el modelo Short captura el 30% de las correcciones de mercado alcista y el modelo Long captura el 25% de los rebotes en mercado bajista (estimaciones conservadoras del WFB histórico):

| Escenario | Trades anuales actuales | Trades con Dual Bot | Incremento |
|---|---|---|---|
| Bull market (12 meses) | 56 Long trades | 56 Long + ~18 Short (correcciones) | +32% |
| Bear market (12 meses) | 0 trades (HOLD) | ~22 Short + ~14 Long (rebotes) | **+∞ (de 0 a 36)** |
| Rango lateral (12 meses) | 56 Long trades | 56 Long + 40 Short | +71% |

### 3.2 Filtro F2: Funding Rate Gate (Nuevo — Solo Futuros)
El Funding Rate es la señal de sesgo estructural del mercado. Se añade como filtro pre-señal:

```python
# Lógica del Funding Rate Gate
if direction == "long" and funding_rate < -0.0003:
    # El mercado paga a los shorts — sesgo bajista fuerte
    # BLOQUEAR Long o reducir Kelly al 50%
    kelly_factor *= 0.5

if direction == "short" and funding_rate > +0.0003:
    # El mercado paga a los longs — sesgo alcista fuerte  
    # BLOQUEAR Short o reducir Kelly al 50%
    kelly_factor *= 0.5
```
- **Parámetro a añadir en `settings.yaml`:** `funding_gate_threshold: 0.0003`

### 3.3 Filtro F3: Volatility Regime Coherence (Nuevo)
Asegurar que el `ATR` actual es coherente con la dirección de la señal:
- Si el ATR 24H supera 3x el ATR 168H (semana), estamos en régimen de alta volatilidad.
- En alta volatilidad, los modelos Short tienden a funcionar mejor (pánico = caída rápida).
- En baja volatilidad sostenida, los modelos Long dominan.
- **Acción:** Este filtro se codifica como feature del XGBoost, no como bloqueo duro. El SFI decidirá si tiene edge estadístico.

---

## 4. Fase 3 — Training de los Escuadrones (XGBoost Trainer)

### 4.1 Cambios en el Orquestador WFB (`run_wfb_orchestrator.py`)
El orquestador debe lanzar **dos pasadas de entrenamiento** por ventana:

**Pasada 1 (Long Squad):**
```python
env = {"LUNA_SEED": str(seed), "LUNA_DIRECTION": "long"}
subprocess.run(["python", "scripts/wfb_worker.py", ...], env=env)
```

**Pasada 2 (Short Squad):**
```python
env = {"LUNA_SEED": str(seed), "LUNA_DIRECTION": "short"}
subprocess.run(["python", "scripts/wfb_worker.py", ...], env=env)
```

### 4.2 Artefactos Generados por Semilla
Para la semilla `42`, con `direction_mode=both`, se generarían:
```
data/models/seed_42/
  ├── xgboost_meta_bull_long.model
  ├── xgboost_meta_bull_long_signature.json
  ├── xgboost_meta_bull_short.model      ← NUEVO
  ├── xgboost_meta_bull_short_signature.json ← NUEVO
  ├── xgboost_meta_range_long.model
  ├── xgboost_meta_range_short.model     ← NUEVO
  ├── xgboost_meta_calm_bear_long.model  ← NUEVA UTILIDAD
  ├── xgboost_meta_calm_bear_short.model ← NUEVO
  ├── xgboost_isotonic_calibrator_bull_long.joblib
  └── xgboost_isotonic_calibrator_bull_short.joblib ← NUEVO
```
> [!NOTE]
> El `RegimeRouter` **ya está preparado** para cargar estos artefactos. Solo hace falta que existan en disco (L196-L198 de `regime_router.py`).

### 4.3 Features Adicionales para el Escuadrón Short
El SFI (Selector de Features Institucional) debe evaluarlas en el ciclo de entrenamiento Short:

| Feature | Justificación para Shorts |
|---|---|
| `funding_rate_acum8h` (ya existe) | Negativo sostenido = mercado bajista real |
| `btc_drawdown_from_ath` (ya existe) | -30% o más = régimen bajista establecido |
| `open_interest_delta` (nueva) | Caída de OI = cierre de longs = presión bajista |
| `liquidation_cascade_score` (nueva) | Nº de liquidaciones acumuladas en 4H |
| `short_funding_divergence` (nueva) | `funding_rate` - `predicted_fair_funding` |

---

## 5. Fase 4 — Alpha Arbitrage Layer (El Árbitro)

Este es el componente matemáticamente más delicado del sistema dual.

### 5.1 El Problema de la Coexistencia
Cuando `direction_mode=both`, es posible que en una barra el Long Squad genere señal Long Y el Short Squad genere señal Short simultáneamente. Necesitamos un árbitro matemático.

### 5.2 Opciones de Arbitraje (De menor a mayor complejidad)

**Opción A: Prioridad por Régimen (Recomendada para V1)**
El HMM decide quién habla:
```python
if hmm_semantic in BULL_REGIMES:
    use_signal = long_signal
elif hmm_semantic in BEAR_REGIMES:
    use_signal = short_signal
elif hmm_semantic in RANGE_REGIMES:
    use_signal = signal_with_higher_consensus_pct
```
- **Ventaja:** Simple, predecible, alineado con el HMM actual.
- **Desventaja:** En regímenes de rango, el árbitro puede elegir mal.

**Opción B: Alpha Differential (Para V2)**
Calcular el "Alpha Neto" de cada escuadrón en la ventana actual:
```python
long_alpha  = long_consensus_pct  * long_calmar_wfb
short_alpha = short_consensus_pct * short_calmar_wfb

if long_alpha > short_alpha * 1.1:   # Umbral del 10%
    execute_long()
elif short_alpha > long_alpha * 1.1:
    execute_short()
else:
    hold()  # Empate estadístico → CASH
```
- **Ventaja:** Usa la evidencia histórica del Calmar de cada escuadrón.
- **Desventaja:** El Calmar de OOS puede ser ruidoso en ventanas cortas.

**Opción C: MetaLabeler de Dirección (Para V3 — Investigación Futura)**
Entrenar un tercer modelo XGBoost (el "Árbitro") que aprende qué dirección gana en cada contexto de mercado, con `target = sign(ret_long - ret_short)`.

### 5.3 Medición del Alpha Individual (Trazabilidad)
Cada escuadrón debe reportar sus KPIs **por separado** en los logs y en el Dashboard:

```python
long_alpha_report  = {
    "direction": "long",
    "n_trades": n_long_trades,
    "win_rate": wr_long,
    "calmar": calmar_long,
    "consensus_pct": consensus_long,
    "net_return_pct": ret_long,
}
short_alpha_report = {
    "direction": "short",
    "n_trades": n_short_trades,
    ...
}
combined_alpha_report = {
    "n_total_trades": n_long + n_short,
    "alpha_gain_vs_longonly": combined_ret - long_only_ret,
    ...
}
```

---

## 6. Fase 5 — Post-XGBoost Filters (Filtros Post-Señal)

### 6.1 Filtro G1: MetaLabeler V2 (Ya existe — extender)
El MetaLabeler actual valida Long. Necesita una copia especializada para Short, entrenada sobre los trades Short históricos del WFB. La firma del artefacto cambia:
```
data/models/seed_42/meta_labeler_long.joblib  ← existente
data/models/seed_42/meta_labeler_short.joblib ← nuevo
```

### 6.2 Filtro G2: Gate de Correlación entre Escuadrones (Nuevo)
Si el Long Squad y el Short Squad están en desacuerdo (uno señala Long, el otro señala Short con alta confianza), **ninguno opera**. El disenso entre expertos especializados en el mismo mercado es una señal de ruido.

```python
if long_consensus > 0.55 and short_consensus > 0.55:
    # Conflicto real → CASH
    logger.warning("[DUAL-GATE] Conflicto de señales: Long=%.2f Short=%.2f → HOLD")
    return "hold"
```

### 6.3 Filtro G3: Kelly Fraccional Diferenciado (Nuevo)
Los Shorts tienen mayor riesgo de liquidación (pérdidas ilimitadas teóricas) vs los Longs (pérdida máxima = 100% de la posición).  
**Propuesta:** `kelly_fraction` diferenciado en `settings.yaml`:
```yaml
sizing:
  kelly_fraction_long: 0.25   # Fracción institucional standard
  kelly_fraction_short: 0.15  # Fracción reducida por asimetría de riesgo en Shorts
```

---

## 7. Fase 6 — Evaluación Diferenciada (Dashboard y Reporting)

### 7.1 Métricas de Alpha por Escuadrón (Regla Obligatoria)
En el Dashboard y en todos los reportes, siempre mostrar:

| Métrica | Long Squad | Short Squad | Combined |
|---|---|---|---|
| Trades | 56 | 38 | 94 |
| Win Rate | 69.6% | 64.3% | 67.3% |
| Max DD | 0.74% | 1.12% | 0.89% |
| Calmar | 9.28 | 6.80 | 8.10 |
| Net Return (6M) | +11.82% | +7.44% | +19.26% |
| Alpha vs Long-Only | — | — | **+7.44% alpha adicional** |

### 7.2 Panel "Modo de Operación" en Dashboard
Añadir un badge configurable en el Header del Dashboard:
```
🟢 LONG-ONLY  |  🔴 SHORT-ONLY  |  🔵 BOTH (Dual Bot)
```

---

## 8. Orden de Implementación Recomendado

| Fase | Componente | Esfuerzo | Riesgo |
|---|---|---|---|
| **P0** | Activar `direction_mode: "both"` en settings.yaml | Bajo | Bajo |
| **P1** | Entrenar primer Short Squad en ventana histórica | Medio | Bajo |
| **P2** | Implementar Árbitro "Opción A" (Prioridad Régimen) | Bajo | Bajo |
| **P3** | Medición diferenciada en `evaluate_ensemble_wfb.py` | Medio | Bajo |
| **P4** | Funding Rate Gate (F2) como filtro pre-señal | Bajo | Bajo |
| **P5** | MetaLabeler Short especializado | Alto | Medio |
| **P6** | Kelly Fraccional Diferenciado | Bajo | Bajo |
| **P7** | Alpha Differential Árbitro (Opción B) | Alto | Medio |
| **P8** | MetaLabeler Árbitro (Opción C — V3) | Muy Alto | Alto |

---

## 9. Puntos Críticos de Causalidad (SOP Anti-Lookahead)

> [!CAUTION]
> Los siguientes puntos deben validarse meticulosamente para no introducir Look-Ahead Bias en el escuadrón Short:

1. **Funding Rate:** Usar siempre el Funding Rate de la barra `t`, nunca el de `t+1`. Ya existe `FundingRate` en el histórico con el `shift()` correcto.
2. **Open Interest Delta:** Calcular como `OI(t) - OI(t-1)` con `shift(1)` obligatorio.
3. **TBM Short Side:** Verificar que `_side_val = -1.0` se aplica ANTES de calcular las barreras, no después del lookback ATR.
4. **MetaLabeler Short:** Entrenarlo en IS con la misma ventana Purged K-Fold que el Long. No usar datos del período de prueba.
5. **HMM Régimen:** El HMM debe usar el Forward Algorithm estricto (SOP R1). No se puede "ver" el régimen futuro para clasificar si una barra es Bear/Bull.

---

## 10. Resumen Ejecutivo

### Lo que el Dual Bot Desbloquea Realmente

El mayor beneficio del sistema Dual no es solo operar en ambas direcciones en mercados neutrales. Es **eliminar los períodos de capital dormido**:

- **Bear markets de 12 meses** donde actualmente Luna V2 no opera: el Short Squad convierte ese tiempo en PnL activo.
- **Correcciones del 20-40% dentro de bull markets** que actualmente se ignoran: el Short Squad oportunista las captura.
- **Rebotes del 30-50% dentro de bear markets** que actualmente están prohibidos para Long: el Long Squad subordinado los captura.

Luna V2 **ya tiene el 60% de la infraestructura necesaria** para el Dual Bot:
- El `XGBoostTrainer` ya entiende `direction_mode` (L312-L323 de `train_xgboost_v2.py`).
- El `RegimeRouter` ya carga artefactos `_long` y `_short` (L196 de `regime_router.py`).
- El TBM ya invierte las barreras para Shorts usando `_side_val=-1.0` (L738).

Lo que **falta** es:
1. Lanzar el entrenamiento del escuadrón Short en el orquestador WFB.
2. Implementar el Alpha Arbitrage Layer (árbitro) con los umbrales de consenso diferenciados por régimen.
3. Reportar los KPIs de cada escuadrón por separado en el Dashboard.
4. Añadir los filtros F2 (Funding Rate Gate) y G2 (Correlación entre escuadrones).

> [!TIP]
> **Propuesta de Primer Hito Testeable:**  
> Lanzar una run con `direction_mode: "short"` sobre el histórico 2020-2025 con las mismas 29 semillas. Si el WFB del escuadrón Short supera DSR > 0.05 en 5+ ventanas, el Dual Bot tiene evidencia estadística suficiente para avanzar a la fase de coexistencia. El objetivo es ver si el modelo Short captura el 30% de las correcciones del bull market de 2023-2024 con una tasa de acierto > 60%.
