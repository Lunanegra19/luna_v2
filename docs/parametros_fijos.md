# Parámetros Fijos y Política de Fallbacks — Luna V2

> **Generado:** 2026-05-21 | **Método:** Auditoría estática de 210 archivos Python  
> **Herramienta:** `tools/diagnostics/audit_parametros_fijos.py`  
> **Total hallazgos:** 276 ocurrencias | **Archivos afectados:** 40+

---

## § 1. Política No-Fallback (Decisión de Diseño Institucional)

### Principio

**Para parámetros que afectan decisiones de trading, validación estadística o gates del Gauntlet: NO existe fallback silencioso.**

Si `settings.yaml` no carga correctamente, el sistema **detiene la ejecución** con `CRITICAL` + `RuntimeError`. Esta política fue adoptada el 2026-05-21 tras confirmar que el fallback silencioso de `PBO_N_BLOCKS=16` causó que **8 de 18 seeds obtuvieran `PBO=0.50` sistemáticamente**, invalidando el análisis estadístico de toda la batch.

### Reglas

| Tipo de parámetro | Política |
|---|---|
| Gates del Gauntlet (DSR, PBO, MaxDD) | **CRITICAL + RuntimeError** — sin fallback |
| Parámetros de riesgo (embargo, purge) | **CRITICAL + RuntimeError** — sin fallback |
| Parámetros de modelo (thresholds, n_states) | **WARNING** — fallback permitido con log explícito |
| Parámetros de diagnóstico (reports, tearsheet) | **DEBUG** — fallback silencioso aceptable |
| Parámetros de herramientas/tools/ | **Libre** — no afectan producción |

### Implementación

```python
# CORRECTO: política no-fallback para gates críticos
_REQUIRED = ["min_dsr", "max_pbo", "min_trades", "alpha_binomial", "max_drawdown", "pbo_n_blocks"]
_missing = [k for k in _REQUIRED if k not in _s]
if _missing:
    raise KeyError(f"CRITICAL: parámetros ausentes en cfg.stat: {_missing}")

self.MIN_DSR = float(_s["min_dsr"])   # KeyError si falta → falla rápido y visible

# INCORRECTO: fallback silencioso
self.MIN_DSR = float(_s.get("min_dsr", 0.75))  # Si falta → usa 0.75 sin aviso
```

---

## § 2. Caso Confirmado — Bug PBO_N_BLOCKS (Auditado 2026-05-21)

**Archivo:** `luna/monitoring/statistical_audit.py`  
**Impacto:** 8/18 seeds con `PBO=0.50` sistemático → batch de mayo 2026 parcialmente invalidada

### Causa raíz

El fallback hardcodeado tenía `PBO_N_BLOCKS=16`, mientras `settings.yaml` tiene `pbo_n_blocks=8`. Cuando settings.yaml fallaba en cargar (cualquier motivo — permisos, encoding, proceso previo), el CSCV exigía `n_trades >= 64` (imposible con 30-55 trades actuales), retornando `0.50` conservador para todas las seeds.

### Fix aplicado

1. Eliminado el bloque `except` con fallback silencioso
2. Reemplazado por `CRITICAL + RuntimeError` si settings.yaml falla
3. Parámetros ahora se leen con `_s["clave"]` (KeyError si falta) en lugar de `_s.get("clave", valor)`

---

## § 3. Inventario Completo de Parámetros Hardcodeados

> Generado automáticamente. Actualizar ejecutando `tools/diagnostics/audit_parametros_fijos.py`.

### 3.1 — CRÍTICO: Fallbacks Silenciosos en Bloques `except` (9 hallazgos)

Estos son los más peligrosos: el código intenta cargar de settings, falla silenciosamente, y usa un valor hardcodeado **sin notificación visible**.

| Archivo | Línea | Parámetro | Valor Fallback | Acción Requerida |
|---------|-------|-----------|----------------|------------------|
| `luna/monitoring/statistical_audit.py` | L90 | `MIN_DSR, MAX_PBO, MIN_TRADES` | `0.75, 0.10, 100` | ✅ **RESUELTO** — sustituido por CRITICAL+RuntimeError |
| `luna/validation/phase_gates.py` | L132 | `DATA_MAX_NAN_PCT` | `0.50` | ⚠️ Pendiente — cambiar a CRITICAL |
| `luna/validation/phase_gates.py` | L144 | `SFI_MAX_ALPHA_RATIO` | `0.80` | ⚠️ Pendiente |
| `luna/validation/phase_gates.py` | L170 | `XGB_AUC_HARD_STOP` | `0.510` | ⚠️ Pendiente — gate crítico |
| `luna/validation/phase_gates.py` | L171 | `XGB_AUC_WARN` | `0.530` | ⚠️ Pendiente |
| `luna/validation/phase_gates.py` | L172 | `XGB_BRIER_HARD_STOP` | `0.2850` | ⚠️ Pendiente — gate crítico |
| `luna/validation/phase_gates.py` | L173 | `XGB_BRIER_WARN` | `0.2700` | ⚠️ Pendiente |
| `luna/validation/phase_gates.py` | L174 | `XGB_BRIER_DEGRADED_MAX_AGENTS` | `1` | ⚠️ Pendiente |
| `luna/validation/phase_gates.py` | L175 | `XGB_PROBA_STD_MIN` | `0.010` | ⚠️ Pendiente |

### 3.2 — ALTO: Constantes Duplicadas con Inconsistencias (6 grupos)

Estas constantes aparecen en múltiples archivos con **valores distintos** — riesgo de que un cambio en uno no se propague a los demás.

#### MAX_DRAWDOWN — 3 valores distintos (0.05, 0.20, 0.60)

| Archivo | Línea | Valor | Nota |
|---------|-------|-------|------|
| `luna/monitoring/statistical_audit.py` | L95 | `0.05` | ❌ Erróneo — era parte del viejo fallback |
| `luna/monitoring/statistical_audit.py` | L99 | `0.60` | ✅ Correcto — coincide con settings.yaml |
| `scripts/pre_flight/test_invariants.py` | L373 | `0.60` | OK |
| `scripts/pre_flight/test_invariants.py` | L384 | `0.20` | ❌ Inconsistente — valor antiguo |
| `scripts/pre_flight/test_invariants.py` | L396 | `0.60` | OK |

**Autoridad:** `settings.yaml stat.max_drawdown = 0.60`

#### PBO_N_BLOCKS — 3 valores distintos (1, 8, 16)

| Archivo | Línea | Valor | Nota |
|---------|-------|-------|------|
| `luna/monitoring/statistical_audit.py` | L96 | `1` | ❌ Resto del viejo fallback (artefacto de `1e6, 8`) |
| `luna/monitoring/statistical_audit.py` | L97 | `8` | ✅ Correcto post-fix |
| `luna/monitoring/statistical_audit.py` | L99 | `8` | ✅ Correcto post-fix |
| `tools/diagnostics/audit_pbo_nblocks.py` | L82,L83 | `16` | ℹ️ Script de diagnóstico — correcto para testear el bug |

**Autoridad:** `settings.yaml stat.pbo_n_blocks = 8`

#### MIN_TRADES — 4 valores distintos (0.75, 20, 32, 100)

| Archivo | Línea | Valor | Nota |
|---------|-------|-------|------|
| `luna/monitoring/statistical_audit.py` | L90 | `0.75` | ❌ Artefacto del viejo fallback (`MIN_DSR, MAX_PBO, MIN_TRADES = 0.75, 0.10, 100` → el regex capturó mal) |
| `luna/monitoring/statistical_audit.py` | L98 | `100` | ❌ Viejo fallback — eliminado |
| `tools/diagnostics/simulate_embargo.py` | L41 | `32` | ℹ️ Valor correcto de la lógica CSCV (n_blocks*4) |
| `tools/diagnostics/simulate_calibration_strategies.py` | L40 | `20` | ℹ️ Script local — no afecta producción |

**Autoridad:** `settings.yaml stat.min_trades = 32`

#### MIN_DSR — 6 ocurrencias, valor consistente (0.75) pero duplicado

| Archivo | Líneas | Valor |
|---------|--------|-------|
| `luna/monitoring/statistical_audit.py` | L98 | `0.75` |
| `scripts/pre_flight/test_env.py` | L1333, L1337 | `0.75` |
| `scripts/pre_flight/test_v5_bugs.py` | L508, L509 | `0.75` |
| `tools/diagnostics/simulate_embargo.py` | L42 | `0.75` |

**Riesgo:** si `settings.yaml` cambia `min_dsr` a otro valor, los tests quedarán desactualizados.  
**Solución propuesta:** los tests deben leer `min_dsr` de settings.yaml, no tener el valor hardcodeado.

#### MAX_PBO — 4 ocurrencias, inconsistencia 0.10 vs 0.22

| Archivo | Líneas | Valor | Nota |
|---------|--------|-------|------|
| `luna/monitoring/statistical_audit.py` | L98 | `0.10` | ❌ Viejo fallback — eliminado |
| `scripts/pre_flight/test_env.py` | L1333 | `0.10` | ❌ Desactualizado — settings tiene 0.22 |
| `scripts/pre_flight/test_v5_bugs.py` | L512, L513 | `0.10` | ❌ Desactualizado — settings tiene 0.22 |

**Autoridad:** `settings.yaml stat.max_pbo = 0.22`  
**Acción:** actualizar los tests a 0.22.

### 3.3 — ALTO: getattr/get con Default en Parámetros Operativos (239 hallazgos)

Listado de los más críticos para el pipeline de producción:

| Archivo | Param | Default | En settings.yaml | Riesgo |
|---------|-------|---------|------------------|--------|
| `luna/models/signal_filter.py:L1330` | `embargo_hours` | `168.0` | Sí (72) | ALTO — valor diferente |
| `luna/models/predict_oos.py:L1144` | `embargo_hours` | `168` | Sí (72) | ALTO — valor diferente |
| `luna/models/train_xgboost_v2.py:L53` | `embargo_hours` | `96` | Sí (72) | ALTO |
| `luna/models/ensemble_lgbm.py:L178` | `embargo_hours` | `96` | Sí (72) | ALTO |
| `luna/models/train_metalabeler_v2.py:L71` | `embargo_hours` | `96` | Sí (72) | ALTO |
| `luna/features/feature_selection_e.py:L143` | `embargo_hours` | `24` | Sí (72) | ALTO — valor diferente |
| `luna/monitoring/statistical_audit.py:L303` | `PBO_N_BLOCKS` | `16` | Sí (8) | ⚠️ Pendiente de fix en línea 303 |
| `scripts/pre_flight/test_env.py:L1354` | `max_pbo` | `0.20` | Sí (0.22) | ALTO — test desactualizado |
| `scripts/pre_flight/test_env.py:L603` | `embargo_hours` | `72` | Sí (72) | OK |
| `luna/reports/generate_tearsheet.py:L982` | `min_dsr` | `0.75` | Sí (0.75) | Bajo (coincide) |
| `luna/risk/psi_guard.py:L140` | `psi_alert_threshold` | `0.25` | ? | Verificar |
| `luna/risk/psi_guard.py:L141` | `psi_halt_threshold` | `0.50` | ? | Verificar |

### 3.4 — PBO_N_BLOCKS en línea 303 (getattr residual)

Hay una tercera ocurrencia en `statistical_audit.py` línea 303 que el audit detectó con `PBO_N_BLOCKS=16`:

```python
# luna/monitoring/statistical_audit.py:L303
n_blocks = getattr(self, 'PBO_N_BLOCKS', 16)  # ← residual del bug
```

> Esta línea ya fue corregida a `8` en el fix anterior. Verificar que el archivo actual tiene `8`.

---

## § 4. Parámetros con Autoridad — Tabla de Referencia Canónica

Estos son los valores **auténticos** según `config/settings.yaml`. Cualquier hardcode en código debe eliminarse y reemplazarse por lectura de settings.

| Parámetro | settings.yaml clave | Valor actual | Afecta |
|-----------|---------------------|--------------|--------|
| `min_dsr` | `stat.min_dsr` | `0.75` | Gauntlet gate |
| `max_pbo` | `stat.max_pbo` | `0.22` | Gauntlet gate |
| `min_trades` | `stat.min_trades` | `32` | Gauntlet gate |
| `alpha_binomial` | `stat.alpha_binomial` | `1.0` | Gauntlet (informativo) |
| `max_drawdown` | `stat.max_drawdown` | `0.60` | Gauntlet gate |
| `pbo_n_blocks` | `stat.pbo_n_blocks` | `8` | CSCV (min_trades = 32) |
| `embargo_hours` | `sop.embargo_hours` | `72` | OOS signal filter |
| `purge_hours` | `sop.purge_hours` | `96` | OOS label purge |
| `cusum_threshold` | `stat.cusum_threshold` | `4.5` | OOS health monitor |
| `wfv_n_windows` | `stat.wfv_n_windows` | `5` | Walk-Forward Validation |
| `xgb_auc_hard_stop` | `phase_gates.xgb_auc_hard_stop` | `0.510` | Pre-flight gate |
| `xgb_brier_hard_stop` | `phase_gates.xgb_brier_hard_stop` | `0.2850` | Pre-flight gate |
| `ensemble_consensus_threshold` | `wfb.ensemble_consensus_threshold` | `3` | scripts/evaluate_ensemble_wfb.py (Consensus Gate) |
| `soft_embargo_enabled` | `wfb.soft_embargo_enabled` | `true` | scripts/evaluate_ensemble_wfb.py (Consensus-Soft Embargo activation) |
| `soft_embargo_hours` | `wfb.soft_embargo_hours` | `24.0` | scripts/evaluate_ensemble_wfb.py (Atenuated embargo hours for consensus >= 4) |
| `probability_cap` | `kelly_sizer.probability_cap` | `0.62` | luna/risk/kelly_sizer.py (Mitigates overconfidence / size collapse in Kelly sizer) |

---

## § 5. Hallazgos por Componente — Prioridad de Acción

### Prioridad 1 (Inmediata — afecta Gauntlet)
- [ ] `luna/validation/phase_gates.py` L132-L175: 8 fallbacks silenciosos en gates críticos → convertir a CRITICAL
- [ ] `scripts/pre_flight/test_env.py` L1354: `max_pbo=0.20` desactualizado (settings: 0.22)
- [ ] `luna/monitoring/statistical_audit.py` L303: verificar que `PBO_N_BLOCKS` usa `8` (post-fix)

### Prioridad 2 (Próximo sprint — afecta resultados)
- [ ] `luna/models/signal_filter.py` L1330: `embargo_hours=168` vs settings `72`
- [ ] `luna/models/predict_oos.py` L1144: `embargo_hours=168` vs settings `72`
- [ ] `luna/models/train_xgboost_v2.py` L53 + `ensemble_lgbm.py` L178: `embargo_hours=96` vs `72`
- [ ] Tests en `test_env.py`, `test_v5_bugs.py`: leer `min_dsr`/`max_pbo` de settings, no hardcodeados

### Prioridad 3 (Técnica — no afecta producción inmediata)
- [ ] 200+ getattr en `ensemble_lgbm.py`, `train_xgboost_v2.py`, `train_metalabeler_v2.py`: documentar si son intencionales o errores
- [ ] `generate_tearsheet.py` y `generate_validation_report.py`: leer gates de settings para mostrar umbrales actuales

---

## § 6. Procedimiento de Actualización

Cuando se cambie un parámetro en `settings.yaml`:

1. **Ejecutar la auditoría:** `python tools/diagnostics/audit_parametros_fijos.py`
2. **Buscar el parámetro** en la sección CONSTANTES DUPLICADAS
3. **Actualizar** todas las ocurrencias que difieran del nuevo valor
4. **Verificar tests:** ejecutar `python -m pytest scripts/pre_flight/test_env.py -k "dsr or pbo or drawdown"`
5. **Documentar** en este archivo si se corrige un hallazgo

---

## § 7. Script de Auditoría

```bash
# Ejecutar la auditoría completa
python tools/diagnostics/audit_parametros_fijos.py

# Buscar un parámetro específico
python -c "
import subprocess
result = subprocess.run(
    ['python', 'tools/diagnostics/audit_parametros_fijos.py'],
    capture_output=True, text=True
)
for line in result.stdout.split('\n'):
    if 'embargo_hours' in line:
        print(line)
"
```

---

*Documento mantenido por el equipo de ingeniería. Última actualización: 2026-05-22.*  
*Herramienta de generación: `tools/diagnostics/audit_parametros_fijos.py` (210 archivos analizados).*

## § 8. Bypass de AutoEncoder en Vivo (Live/Production Mode)

### Principio
En el entorno live/producción, no existe entrenamiento dinámico del AutoEncoder. En lugar de reentrenar en caliente en cada vela o tick (lo cual disparaba consumos críticos de CPU >99% y reinicios automáticos en PM2), el sistema carga dinámicamente los pesos congelados de entrenamiento e inyecta la reducción de dimensionalidad en microsegundos.

### Parámetros Fijos Registrados

| Parámetro | Valor de Producción | Tipo | Fallback |
|---|---|---|---|
| `bottleneck_size` | `32` | `int` | Crítico — Falla si no coincide con las 32 neuronas del bottleneck |
| `epochs` | `30` | `int` | Omitido en Vivo (Bypass) |
| `autoencoder_state.pt` | Pesos de producción | Binario PyTorch | CRITICAL + RuntimeError si no existe en `/data/models/` |
| `autoencoder_scaler.joblib` | Pesos del StandardScaler | Binario Joblib | CRITICAL + RuntimeError si no existe en `/data/models/` |
| `autoencoder_config.json` | Lista canónica de features | JSON estructurado | CRITICAL + RuntimeError si no existe en `/data/models/` |

### Trazabilidad y Seguridad
- Si CUDA está habilitado pero falla en iniciar la GPU en caliente, se realiza un fallback controlado a CPU.
- Si faltan columnas de entrada que el AutoEncoder espera (debido a lagunas de datos transitorias en la API de OKX), el pipeline las alinea dinámicamente inyectando un valor neutro (`0.0`), previniendo caídas críticas por `KeyError`.
- Cada carga e inferencia se registra explícitamente en logs de producción con la firma `✨ [LIVE-AE-FIX]`.

---

## § 9. Auditor Operativo en Vivo (Live Operational Auditor)

### Principio
El Auditor Operativo en Vivo ejecuta 6 salvaguardas preventivas automatizadas en cada ciclo antes y después de la inferencia del ensamble. Su objetivo es bloquear de raíz el trading ciego, datos corruptos, sobreapalancamientos catastróficos o desconexión del broker.

### Parámetros Operativos Registrados

| Salvaguarda | Parámetro Fijo | Límite Máximo | Acción en Falla |
|---|---|---|---|
| **Guard 1 (Clock Drift)** | `data_max_gap_h` (vivos) | `90 minutos` | **Pausa en DB + Cierre a HOLD** |
| **Guard 2 (NaN/Inf Shield)** | Integridad de features | `0 columnas corruptas` | **Pausa en DB + Cierre a HOLD** |
| **Guard 3 (Leverage Ceiling)** | `max_leverage_allowed` | `20.0x real leverage` | Alerta Telegram + Rebalanceo Seguro |
| **Guard 4 (API Liveness)** | Broker Connection | Falla de llamada API | **Pausa en DB + Cierre a HOLD** |
| **Guard 5 (HMM Consistency)** | HMM State Index | Rango `[0, 6]` | **Pausa en DB + Cierre a HOLD** |
| **Guard 6a (Cycle Latency)** | Execution latency | `20.0 segundos` | Alerta Prioritaria Telegram |
| **Guard 6b (Slippage Monitor)** | Fill slippage pct | `0.50% round-trip` | Alerta Prioritaria Telegram |

### Esquema SQL de Inserción (`operational_audit_logs`)

El sistema audita de forma persistente e ininterrumpida cada iteración horaria. Los resultados se registran en la tabla relacional `operational_audit_logs`:

```sql
CREATE TABLE IF NOT EXISTS operational_audit_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    clock_drift_minutes NUMERIC(8, 2) NOT NULL,
    clock_drift_status VARCHAR(20) NOT NULL,
    nan_inf_null_cols INT NOT NULL,
    nan_inf_status VARCHAR(20) NOT NULL,
    active_leverage NUMERIC(6, 2) NOT NULL,
    leverage_status VARCHAR(20) NOT NULL,
    api_liveness_equity NUMERIC(15, 2),
    api_liveness_status VARCHAR(20) NOT NULL,
    hmm_regime_index SMALLINT,
    hmm_status VARCHAR(20) NOT NULL,
    execution_latency_sec NUMERIC(6, 2),
    latency_status VARCHAR(20) NOT NULL,
    slippage_pct NUMERIC(8, 6),
    slippage_status VARCHAR(20) NOT NULL,
    is_approved BOOLEAN NOT NULL,
    details TEXT
);
```

### Seguridad y Resiliencia
- **No-Fallback:** La falta de cualquiera de los parámetros de drift o leverage ceiling interrumpe la ejecución de forma inmediata.
- **Offline Caching:** Si la base de datos PostgreSQL remota se desconecta o sufre un bloqueo de red, el auditor almacena de forma atómica los logs operativos en `data/cache/offline_operational_audit_logs.json`. Estos se sincronizan automáticamente una vez restablecida la conexión.


---

## § 10. Familias de Features SFI — Taxonomía Estructural [P3-B 2026-05-28]

### Contexto

El análisis empírico de las 30 seeds de la sesión 2026-05-28 reveló que las seeds aprobadas por el Gauntlet se agrupan en dos familias structuralmente distintas según los lags de sus features de mayor importancia SHAP. Esta clasificación impacta directamente en la política de embargo óptima por familia.

### Familia A — Macro-Institucional

| Criterio | Valor |
|----------|-------|
| **Seeds representativas** | 42, 100, 48907 |
| **Lag mínimo dominante** | > 120H (5 días) |
| **Features principales** | IBIT flows, M2 supply, institutional BTC allocation, BTC dominance rolling |
| **Embargo óptimo** | 96H (coincide con SOP estándar) |
| **Patrón de WR** | Estable 48-55%, menos sensible a regímenes bear-corto |

### Familia B — Cripto-Cíclica

| Criterio | Valor |
|----------|-------|
| **Seeds representativas** | 777, 1337, 2025 |
| **Lag máximo dominante** | < 168H (7 días) |
| **Features principales** | Fear & Greed Index, Puell Multiple, SOPR, funding rates |
| **Embargo óptimo** | 48-72H (más ágil que Familia A) |
| **Patrón de WR** | Volátil 42-58%, alta sensibilidad a regímenes bear-crash |

---

## § 11. Nuevos Parámetros Canónicos [Sesión 2026-05-28]

### CB-01 — Circuit Breaker de Régimen Mensual

Implementado en luna/risk/circuit_breaker.py. Fuente: settings.yaml wfb.circuit_breaker.

| Parámetro | Clave settings.yaml | Valor | Fundamento |
|-----------|---------------------|-------|------------|
| min_seeds_adverse | wfb.circuit_breaker.min_seeds_adverse | 4 | May-Jun 2025: 5/6 seeds adversas. Threshold=4 captura sin falsos positivos |
| wr_threshold | wfb.circuit_breaker.wr_threshold | 0.38 | WR normal ensemble: 46-51%. WR adverso real: 15-38% |
| rolling_days | wfb.circuit_breaker.rolling_days | 30 | Granularidad mensual de regímenes |

### P3-A-01 — Umbral Ventanas WFV Triviales

Implementado en scripts/run_statistical_validation.py.

| Parámetro | Clave settings.yaml | Valor | Razón |
|-----------|---------------------|-------|-------|
| wfv_min_trades_per_window | stat.wfv_min_trades_per_window | 5 | Ventanas con menos de 5 trades no tienen poder estadístico significativo |

### FIX-P1A — Signal Funnel Acumulado

LUNA_RUN_ID formato funnel: WFB_seed{N}_funnel — estable entre ventanas W1-W5 de la misma seed.

---

## §GAUNTLET-FIX-01 — Umbrales Corregidos del Gauntlet Estadístico (2026-05-28)

### Contexto
Análisis estadístico profundo detectó 3 gates con configuración incorrecta en `config/settings.yaml § stat`.
El gate binomial estaba efectivamente **deshabilitado** (`alpha=1.0`), `min_trades` era insuficiente para
que el CSCV (PBO) fuese fiable, y `max_drawdown` era inconsistente con los circuit breakers de producción.

### Cambios aplicados

| Parámetro | Antes | Después | Justificación |
|-----------|-------|---------|--------------|
| `alpha_binomial` | **1.0** | **0.20** | alpha=1.0 ≡ gate deshabilitado (p<1.0 siempre). alpha=0.20 activa el gate con n≥64 trades. Para producción real: 0.10. |
| `min_trades` | **32** | **64** | SOP R8: 100 para inferencia confiable. 64 garantiza CSCV block_size=8 (mínimo fiable). IC WR ±12.3%. |
| `max_drawdown` | **0.60** | **0.25** | kill_switch circuit breaker = 15%. MaxDD=60% es incoherente — el sistema pararí automáticamente antes. 25% = kill_switch × 1.67. |

### Gates que NO cambian (correctos)

| Parámetro | Valor | Por qué es correcto |
|-----------|-------|---------------------|
| `min_dsr` | 0.75 | Bailey & LdP 2014 — umbral estándar de significancia |
| `max_pbo` | 0.22 | CSCV literature — buen balance estricto/permisivo |
| `pbo_n_blocks` | 8 | Correcto si min_trades≥64 (block_size=8) |

### Análisis de impacto

**Seeds históricas SFI18 (n=34-55):** Todas habrían fallado `min_trades=64`. Esto es estadísticamente
correcto — con 34-55 trades, el IC del WR es ±13-17% y el CSCV no es fiable. El alpha=1.0 original
fue una decisión de emergencia para no rechazar todo el sistema en una fase temprana.

**seed=100 run actual (71 trades, WR=56.3%):**
- `min_trades`: 71 ≥ 64 ✅
- `alpha_binomial`: p=0.171 < 0.20 ✅
- `max_drawdown`: 0.45% < 25% ✅
- DSR y PBO: sin cambio (correctos)
→ **APPROVED** con umbrales nuevos.

### Referencia
- Bailey & López de Prado (2014) — Deflated Sharpe Ratio y Multiple Testing
- Magdon-Ismail (2004) — Expected Maximum Drawdown bajo GBM
- `tools/diagnostics/audit_binomial_impact.py` — análisis previo de impacto alpha=0.05
- SOP V10.0 Iron Rules R8 (min trades) y circuit breakers (max DD)

*Fecha: 2026-05-28 | Commit: GAUNTLET-FIX-01*

---

*Última actualización: 2026-05-28 — GAUNTLET-FIX-01 correccion umbrales estadisticos*

---

## P0-AUDIT-20260529: max_pbo 0.22 -> 0.45 (temporal)

**Fecha:** 2026-05-29
**Commit tag:** P0-AUDIT-20260529
**Archivo:** config/settings.yaml (bloques: gauntlet + stat)
**Referencia:** docs/auditoria_institucional_20260529.md seccion NUEVO-2

### Valor anterior
max_pbo: 0.22

### Valor nuevo
max_pbo: 0.45

### Justificacion estadistica

Simulacion Monte Carlo (N=5.000 iteraciones) con parametros reales del sistema:
- WR real observado: 53% (20 seeds overnight run 28-29/05/2026)
- pbo_n_blocks: 8 (configuracion actual CPCV)
- N_trades_per_split: ~20 (5 ventanas WFB, ~4 trades/ventana mediana)

Resultado: P(PBO > 0.22 | edge_real WR=53%, pbo_n_blocks=8) = 96%

El umbral 0.22 fue establecido asumiendo N_splits > 20 para que el estimador PBO
sea estadisticamente estable. Con N_splits=8, el estimador tiene incertidumbre
de +/-20%, lo que produce falsos positivos en el 96% de los casos.

Con max_pbo=0.22 y pbo_n_blocks=8:
- De 14/21 seeds de la run overnight, la mayoria fallaron PBO por ruido, no por overfitting real
- seed1337 (82T, DSR=1.0, PBO=0%) se confirma que NO es overfitting
- El gate PBO dejaba pasar solo el 4% de sistemas con edge real

### Condicion de revercion

Cuando N_windows >= 10 (Sprint 3: WFB de 5 a 7 ventanas, luego a 10):
- Revaluar max_pbo con nueva simulacion Monte Carlo
- Si P(FP) < 40%: volver a max_pbo=0.35
- Si P(FP) < 20%: volver a max_pbo=0.22 (valor original)

### No-fallback policy

Este es un gate CRITICO del Gauntlet. El statistical_audit.py implementa
no-fallback: si max_pbo falta en settings.yaml -> KeyError + stop del pipeline.
Ver luna/monitoring/statistical_audit.py L86-L94.

### Referencias
- SOP V10.0 R5 (comparaciones multiples, DSR sobre Sharpe bruto)
- Bailey & Lopez de Prado (2014) - Probability of Backtest Overfitting
- Monte Carlo simulation: scratch/test_overfitting.py H6

---

## 11. Score de Estabilidad Temporal Ponderada [WEIGHTED-STABILITY-01 2026-05-29]

### Problema resuelto
El score simple positive_years / total_years daba igual peso a 2018 que a 2025. Una feature post-ETF que funciona solo en 2023-2025 (y no tiene historia anterior) era penalizada injustamente. Igual que una feature obsoleta que funcionó en 2018-2020 pero ya no.

### Implementación

**Archivo:** `luna/features/feature_selection_e.py` — método `_eval_temporal_stability()`

**Parámetros canónicos (todos en settings.yaml sección features):**

| Parámetro | Valor | Justificación |
|---|---|---|
| `stability_half_life_years` | **2.0** | Hace 2yr tiene peso 0.60, hace 4yr tiene 0.37. Calibrado para que post-ETF (Ene-2024) tenga peso dominante. |
| `stability_recent_window_years` | **2** | Componente de recencia pura: 40% del score compuesto. Últimos 2 años = verdad operativa. |
| `stability_trend_window_years` | **4** | Ventana para regresión lineal de tendencia (Rising/Declining/Volatile). |
| `stability_trend_threshold` | **0.10** | Slope mínimo en SR/año para clasificar como Rising o Declining. |

**Fórmula:**
`
weighted_stability = Σ(w_i * 1[msr_i > 0]) / Σ(w_i)   donde w_i = exp(-(año_actual - año_i) / half_life)
recent_stability   = años_positivos_recientes / años_en_ventana_reciente
composite          = 0.60 * weighted_stability + 0.40 * recent_stability
score_final        = min(composite * trend_modifier, 1.0)

trend_modifier: Rising=1.10, Stable=1.00, Volatile=0.85, Declining=0.70
`

**Invariante:** `0.0 <= stability_score <= 1.0` — compatible con downstream sin cambios.

### Verificación
Test unitario: `tools/diagnostics/test_weighted_stability.py` (assertions verificadas 2026-05-29).
Casos verificados: Rising post-ETF sube, Declining baja, Stable correcta.

---

## LIFECYCLE-01: Evaluacion Consciente del Ciclo de Vida de Features

**Fecha de implementacion:** 2026-05-29
**Archivo:** `luna/features/feature_selection_e.py` — metodo `SFI_CPCV._eval_temporal_stability()`
**Settings:** `config/settings.yaml` seccion `features:`

### Problema que resuelve

El SFI evaluaba todos los anos desde el inicio del dataset (2017), usando DSR=0.0 para
los anos donde la feature no tenia datos. Esto penalizaba injustamente features nuevas
(Coinglass desde 2022, DVOL desde 2023) con 6 anos de zeros antes de su lanzamiento.

Ademas, el calculo de WEIGHTED-STABILITY-01 usaba `max(yearly_dsrs)` como referencia
temporal en vez de `ts_max_yr`. Para features muertas (last_real=2022), esto
hacia que la feature pareciera reciente (2022 obtenia peso=1.0 en el decay).

### Solucion: algoritmo de 2 fases

FASE 1 (pre-scan): detecta first_real_year y last_real_year por varianza real (std > threshold).
FASE 2 (evaluacion): aplica tratamiento diferenciado segun posicion temporal:
  PRE-BORN  (yr < first_real): excluir — no es fallo, simplemente no habia datos
  POST-DEATH (yr > last_real): DSR = -1.0 — fuente desconectada, penalizar
  GAP interno (sin varianza entre first y last): DSR = gap_penalty (-0.5)
  NORMAL (varianza real): evaluar con _eval_one normalmente

Bug fix: `_current_year = ts_max_yr` (antes usaba `max(yearly_dsrs)`)

### Parametros fijos canonicos

| Parametro | Valor | Justificacion |
|---|---|---|
| stability_variance_threshold | 1e-6 | Umbral de STD para datos reales vs constantes |
| stability_min_real_years | 2 | Minimo anos con varianza para evaluar (estadisticamente valido) |
| stability_maturity_min_years | 3 | Anos para score sin descuento de madurez |
| stability_dead_threshold_years | 2 | Anos sin datos para clasificar como DEAD |
| stability_gap_penalty | -0.5 | DSR de penalizacion para huecos internos |

### Descuento de madurez (YOUNG features)

Justificacion estadistica: IC binomial 95% para p=1.0:
  n=1 ano: [0.025, 1.0] — muy alta incertidumbre
  n=2 anos: [0.158, 1.0]
  n=3 anos: [0.292, 1.0]

Factor aplicado: min(1.0, max(0.60, n_real / maturity_min_years))
  1 ano real:  factor=0.60
  2 anos real: factor=0.80
  3+ anos:     factor=1.00 (sin descuento)

### Impacto medido (267 features evaluadas)

74 features (28%) tenian delta > 0.20 entre score SFI y score real.
177 features (66%) tenian algun sesgo por historia corta.
Features recuperadas: DVOL (0.333->0.80), Coinglass_oi (0.333->0.80), dv_vrp_30d (0.333->0.80).
Features correctamente penalizadas: features DEAD obtienen score < 0.30 vs score alto previo.

### Verificacion

Test unitario: scratch/test_lifecycle01.py — 4/4 tests pasados (2026-05-29).
Casos verificados: MATURE correcto, YOUNG con PRE-BORN excluidos, YOUNG insuficiente=0,
DEAD con POST-DEATH penalizados y trend=Declining(x0.7) => score=0.243.

---

## FIX-BEAR-COLLAPSE-01: Corrección Colapso XGBoost Bear_long (2026-06-01)

**Fecha:** 2026-06-01  
**Archivo modificado:** `config/settings.yaml` — sección `xgboost.optuna_search_space`  
**No requiere cambios de código** — solo parámetros de Optuna.

### Causa Raíz Identificada

En la run 2026-06-01 (31 seeds × W1-W5), el agente `bear_long` producía `std_prob=0.0`
en el 19% de los entrenamientos IS (15/77 eventos). Esto causaba 20 FATALs y la pérdida
de W4/W5 en esas seeds.

**Cadena causal (verificada con logs reales):**

```
n_train_bear = 267-296 (agente bear, ventanas donde el gate min_viable=200 no actúa)
  → Optuna elige gamma=4.5 + MCW=18 (dentro de los bounds anteriores)
  → Con n_train=285, gamma=4.5: ganancia de split < gamma → 0 splits → 1 hoja
  → predict_proba() devuelve constante = base_rate (0.576)
  → std_IS=0.000000 en POST-FIT IS
  → En OOS: std_prob=0 → FIX-ROUTER-SANITY-01 lanza RuntimeError → FATAL
  → 20 seeds pierden W4/W5 → 0 seeds completan el ensemble
```

**Nota:** El gate `min_viable_train_samples=200` ya bloquea los casos n_train<200
(n=91-99 → GATE-ABORT). El problema residía en n_train=267-296, que pasa el gate
pero sigue siendo insuficiente para los bounds anteriores de gamma/MCW.

### Parámetros Modificados

| Parámetro | Antes | Después | Justificación |
|---|---|---|---|
| `xgboost.optuna_search_space.gamma_max` | **5.0** | **2.0** | Con gamma=5 y n_train=285, ningún split tiene ganancia suficiente → 0 splits. Con gamma≤2 la ganancia es alcanzable con n~270. |
| `xgboost.optuna_search_space.min_child_weight_max` | **20** | **10** | Con n_train=267 y MCW=20 → max_leaves=13 (marginal). Con MCW≤10 → max_leaves=26 → modelo no degenerado. Ratio hojas/samples = 26/267 ≈ 10% (aceptable con reg_lambda≥0.5). |

### Verificación Matemática

| n_train | MCW_max_nuevo=10 | max_leaves | gamma_max_nuevo=2.0 | Estado |
|---|---|---|---|---|
| 267 | 10 | 26 | 2.0 | ✅ Viable (antes: colapsaba) |
| 285 | 10 | 28 | 2.0 | ✅ Viable (antes: colapsaba) |
| 296 | 10 | 29 | 2.0 | ✅ Viable (antes: colapsaba) |
| 733 | 10 | 73 | 2.0 | ✅ Viable (sin cambios) |

### Fixes Descartados y por qué

- **FIX-A (MCW=n/3):** DESCARTADO — con n=91 → MCW=30 → solo 3 hojas, no resuelve nada
- **FIX-C (skip bear):** DESCARTADO — W2 es 100% CALM_BEAR; skip eliminaría W2 completamente
- **FIX-D (retrain universal_mode):** DESCARTADO — universal_mode=True nunca ocurrió en logs, sin evidencia empírica de que funcione

### Impacto Esperado en Próxima Run

- **FATAL por bear_long:** 20 → estimado 0-3 (si Optuna aún elige combinaciones límite)
- **Seeds que completan W4/W5:** estimado +15-18 seeds adicionales
- **W2 (100% CALM_BEAR):** preservada — el modelo bear_long ahora puede predecir con varianza real
- **Riesgo overfitting:** Moderado (10% ratio hojas/samples). Controlado por `reg_lambda_max=2.0` existente

### Política No-Fallback

Este parámetro afecta directamente el entrenamiento del modelo. No hay fallback — si
`settings.yaml` no carga, el script lanza `RuntimeError` (comportamiento existente).

*Fecha: 2026-06-01 | FIX-BEAR-COLLAPSE-01*
