# BUG-PLATT-PICKLE-01 — PlattCalibrator no deserializable desde RegimeRouter

**Fecha detectado:** 2026-06-05  
**Severidad:** 🔴 CRÍTICO — Aborta el pipeline completo en la fase de Calibración  
**Estado:** 🟡 PENDIENTE DE FIX  

---

## Síntoma

La run de producción WFB aborta en la fase `Calibrador de Probabilidades (LONG)` con el siguiente error:

```
[FATAL] La Fase 'Calibrador de Probabilidades (LONG)' abortó con código de error 1
```

En el log de `calibrate_probabilities_*.log`:

```
WARNING | luna.models.regime_router:_load_models:263 - [RegimeRouter] Error cargando calibrador 
xgboost_isotonic_calibrator_bear_long.joblib: 
Can't get attribute 'PlattCalibrator' on <module '__main__' from 
'G:\\...\\luna\\models\\calibrate_probabilities.py'>

WARNING | [FIX-CALIB-BINARY-01/AUDIT-LOAD] ALERTA: 0/3 calibradores cargados. 
Agentes SIN calibrador: ['bear', 'bull', 'range']. 
Efecto: xgb_prob_cal == xgb_prob_raw → WR degradado (ej. W1: 32.9% vs 54% esperado).
```

---

## Causa Raíz

### Problema de pickling Python con `__main__`

Cuando `calibrate_probabilities.py` se ejecuta como **script** (`python calibrate_probabilities.py`), Python asigna `__module__ = '__main__'` a todas las clases definidas en ese archivo:

- `PlattCalibrator`
- `_RFWithAdapter`
- `_IdentityWrapper`
- `_TSAdapter`
- `_TemperatureScaler`

Cuando `joblib.dump()` serializa un objeto de estas clases, guarda `__main__.PlattCalibrator` como referencia de clase en el `.joblib`.

Cuando **posteriormente** `regime_router.py` intenta cargar ese `.joblib` con `joblib.load()`, lo hace desde un contexto distinto (módulo `luna.models.regime_router`), donde `__main__` ya **no** apunta a `calibrate_probabilities.py` sino al script orquestador del WFB. Como resultado, `PlattCalibrator` no existe en ese `__main__` y la deserialización falla con `AttributeError`.

```
# Lo que se serializó:
__module__ = '__main__'   # ← calibrate_probabilities.py como __main__

# Lo que encuentra al cargar (desde regime_router.py):
__main__ = run_wfb_orchestrator.py  # ← PlattCalibrator no existe aquí
```

---

## Impacto

| Consecuencia | Detalle |
|---|---|
| **Calibradores no cargados** | 0/3 calibradores cargados (bear, bull, range) |
| **Degradación de WR** | xgb_prob_cal = xgb_prob_raw (sin calibración), WR cae de ~54% a ~32.9% |
| **Aborto del pipeline** | `sys.exit(1)` desde la fase de calibración |
| **settings.yaml restaurado** | El orquestador WFB restaura el backup al abortar |

---

## Reproducción

1. Ejecutar una run WFB con `use_regime_agents=true`
2. El calibrador LONG se entrena y guarda el `.joblib` con `PlattCalibrator` como `__main__`
3. `regime_router.py` intenta cargar el `.joblib` → `AttributeError`

---

## Solución Propuesta

### Opción A (Recomendada): Mover clases a módulo compartido

Crear `luna/models/calibrators.py` con todas las clases serializables:

```python
# luna/models/calibrators.py  ← NUEVO FICHERO
class PlattCalibrator: ...
class _RFWithAdapter: ...
class _IdentityWrapper: ...
class _TSAdapter: ...
class _TemperatureScaler: ...
```

En `calibrate_probabilities.py` y `regime_router.py`, importar desde el módulo:

```python
from luna.models.calibrators import PlattCalibrator, _RFWithAdapter, _IdentityWrapper
```

Al ejecutarse como script, `PlattCalibrator.__module__` ahora será `luna.models.calibrators` (path absoluto), que **siempre** es importable desde cualquier contexto.

### Opción B (Parche rápido): Añadir `sys.modules` alias

En `regime_router.py`, antes de cargar el `.joblib`:

```python
import luna.models.calibrate_probabilities as _cp_mod
import sys
sys.modules['__main__'] = _cp_mod  # remap temporal
joblib.load(...)
sys.modules.pop('__main__')
```

> ⚠️ Opción B es frágil y no se recomienda en producción.

---

## Archivos Afectados

| Archivo | Rol |
|---|---|
| `luna/models/calibrate_probabilities.py` | Define y serializa las clases con `__main__` |
| `luna/models/regime_router.py` | Carga los `.joblib` desde contexto distinto |
| `data/models/*.joblib` | Artefactos serializados con referencia incorrecta |

---

## Notas Adicionales

- El bug **no ocurría** cuando `calibrate_probabilities.py` era invocado únicamente desde otros módulos (importado), porque en ese caso `__module__` toma el path real del paquete.
- El bug **se activa** específicamente porque el WFB orquestador lanza `calibrate_probabilities.py` como **subproceso script** (`subprocess.run(['python', 'calibrate_probabilities.py', ...])`).
- Los `.joblib` corruptos generados en ventanas anteriores siguen siendo inválidos hasta que se regeneren con el fix aplicado.
