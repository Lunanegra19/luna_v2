# 📈 Plan Paso a Paso — Mejora del Squad LONG (v2 · post-validación)

> **Reestructurado 2026-06-24** tras validar empíricamente las primeras palancas. Cambio de estrategia: de "volumen-primero" a **CALIDAD-PRIMERO** (la data refutó la palanca de volumen). Reglas SOP en `.agents/rules/` + `CLAUDE.md`.

---

## 0. Baseline VERIFICADO (la referencia buena)
- **Config buena = la run 022340/030336/112913** → seed42 long **WR 59.4%, Sharpe 1.79, Calmar 3.39, 64 trades**.
- **Claves de esa config:** `fase2.direction_mode: both` · banda DVOL `[-1.0, 1.5]` (default) · `Hash_Rate` en SFI · voting `hard` · ventanas mensuales 2025-07 → 2026-06.
- **Recuperable siempre** desde `data/runs/WFB_<id>_seed<N>/settings_snapshot.yaml` (el orquestador guarda un snapshot por run; **la config nunca se pierde**). Restaurada el 2026-06-24 desde `data/runs/WFB_20260624_022340_seed42/settings_snapshot.yaml`.
- **Trades archivados** de esa config: `data/predictions/archive/snapshot_20260624/oos_trades_seed*_long.parquet` → permiten retro-simular palancas post-hoc SIN re-entrenar.

---

## 1. Lo que el testeo REFUTÓ (no repetir)
| Cambio probado | Veredicto | Evidencia |
|---|---|---|
| **F1 — ensanchar banda DVOL** (`1.5/-1.0 → 2.0/-1.5`) | ❌ **REFUTADO** | Degrada el long: admite ruido de regímenes volátiles. `1_BULL_TREND` se mantuvo WR 57%, pero `1_VOLATILE_BULL`/`2_VOLATILE_RANGE` sangraron. El band original filtra bien. |
| **Long-only** (`direction_mode: long`) | ❌ Divergencia | El baseline bueno es `both`. En long-only el régimen bear entrena long → resultados peores en estas ventanas. |
| Hash_Rate → hashrate_chg_30d | ❌ Innecesario | `Hash_Rate` es candidato muerto inofensivo; la run buena corría con él. El "fix" desplazó el pool de features. |

> **Lección madre:** el cuello de los longs **NO es volumen vía DVOL**. Los filtros (DVOL, silenciador, OOD) están cortando ruido correctamente. **El edge se gana quitando trades malos (calidad), no añadiendo más.**

---

## 2. Estrategia: CALIDAD-PRIMERO
La data dice: el edge vive en regímenes concretos (`1_BULL_TREND` WR 75%, `CALM_BEAR`, `CALM_RANGE`) y **sangra** en otros (`2_VOLATILE_RANGE` WR ~0-36%, `1_VOLATILE_BULL` ~25-51%). Mejorar = **vetar/recortar lo que pierde**, no aflojar filtros.

**El volumen (para DSR) viene del MERGE long+short del dual-bot (`both`)** — que ya está activo — NO de ensanchar DVOL.

---

## 3. Marco de medición
### 3.1 Dos tipos de palanca
| Tipo | Ejemplos | Cómo se testea |
|---|---|---|
| **POST-HOC** (filtro sobre trades) | veto de régimen, floor de prob, censor | **Retro-sim sobre trades archivados** `data/predictions/archive/snapshot_20260624/` → **segundos, sin re-run** ✅ |
| **PIPELINE** (cambia modelo/labeling) | TBM hard-stop, retrain, fracdiff | Re-run con la config buena (horas) — el re-run ES la medición |

### 3.2 Métricas (panel estándar — `tools/diagnostics/measure_long_run.py`)
Trades, WR, retorno medio/sum, Sharpe-proxy, Calmar-proxy, `n_seeds≥30`, n negativos, edge por régimen. Diseño **pareado** (mismas seeds antes/después). Baseline: la config buena (WR 59.4%).

### 3.3 Seeds por confirmación
Smoke 1 · pre-validación post-hoc sobre TODAS las archivadas (gratis) · confirmación **12 seeds** (solo para palancas pipeline) · sign-off **29 + ensamble**.

---

## 4. Palancas ORDENADAS (calidad-primero)
> Orden por: confianza × (post-hoc antes que pipeline, porque post-hoc no cuesta re-runs).

### L1 — Veto LONG en `2_VOLATILE_RANGE` (+ revisar `1_VOLATILE_BULL`) · POST-HOC · **★ confianza ALTA**
- **Tipo A**, retro-sim sobre trades archivados → resultado en segundos, sin re-run.
- Evidencia previa: vetar `2_VOLATILE_RANGE` → WR 61.3→63.0%, Sharpe-proxy 1.12→1.35, mejora 20/22 seeds (jackknife-robusto).
- ⚠️ `1_VOLATILE_BULL`: NO vetar a ciegas (su pérdida fue 80% un seed). Re-validar sobre la config buena.
- **Implementación final** (tras validar): veto direccional long en `hmm_allowed_regimes` o param `long_forbidden_regimes`.

### L2 — Floor de prob calibrada (`xgb_prob_cal ≤ 0.5`) · POST-HOC · confianza BAJA
- Cortar el bucket perdedor, **conservando los sentinel `==0`** (meta-rescatados, ganadores). Floor naive = trampa.
- Retro-sim primero.

### L3 — TBM hard-stop regime-scoped (`0.025 → ~0.015` en CALM_BEAR/VOLATILE_BULL_B) · PIPELINE · confianza MEDIA
- **Pre-req: validar fill intrabar** en las velas-tope CALM_BEAR (66% de la ganancia depende de eso). Re-run con config buena.

### L4 — Investigar edge del modelo dual · PIPELINE · alto impacto, alta incertidumbre
- El squad long en `both` es la base buena (WR 59%). Cualquier retrain se compara contra esa base. Long-only quedó descartado como default.

### Volumen / DSR — **vía MERGE long+short (ya en `both`), NO DVOL**
- El conteo de trades para DSR sube fusionando ambos squads a nivel ensamble. Recalibrar el silenciador short (estaba invertido). Ver [[luna-v2-long-levers]].

---

## 5. Trampas (refutadas o peligrosas)
- ❌ **Ensanchar DVOL** (F1) — refutado, mete ruido volátil.
- ❌ **Aflojar silenciador/threshold/OOD** — re-añade la cola perdedora (WR 45%/−0.32%).
- ❌ **Long-only como default** — diverge del baseline bueno `both`.
- ❌ **Vetar `1_VOLATILE_BULL` a ciegas** — pérdida dominada por 1 seed.
- ❌ **Editar settings durante un run** — el orquestador restaura settings al terminar (ver `settings_restore_protection`).

---

## 6. Protocolo Operativo Estándar (POE)
```
POST-HOC:  retro-sim sobre archivo → decidir → (si GO) implementar en settings + re-run de confirmacion
PIPELINE:  [1] runs detenidas  [2] backup settings  [3] editar UN cambio + comentario [TAG fecha]
           [4] smoke 1 seed (vigilar 30s + forense de logs)  [5] 12 seeds  [6] medir vs baseline  [7] PASS/FAIL
SIGN-OFF:  29 seeds + ensamble + DSR real
```
Comandos: ver `CLAUDE.md` (§Comandos clave). Recuperar config buena: `cp data/runs/WFB_20260624_022340_seed42/settings_snapshot.yaml config/settings.yaml`.

---

## 7. Registro de experimentos
| Fecha | Palanca | Tipo | Método | Resultado | Veredicto |
|---|---|---|---|---|---|
| 2026-06-24 | Baseline (022340/112913, `both`) | — | run | seed42: 64 tr, WR 59.4%, Sh 1.79 | ref ✅ |
| 2026-06-24 | F1 ensanchar DVOL + long-only | pipeline | smoke 4 ventanas | WR 39% (vs 54% mismas ventanas) | ❌ REFUTADO |
| 2026-06-24 | L1 veto 2_VOLATILE_RANGE | post-hoc | retro-sim (config buena) | WR 61.3→63.0% (+1.6pp), Sharpe-proxy +20%, 20/22 seeds, −39 tr | ✅ VALIDADO |
| 2026-06-24 | L1+ veto VOL_RANGE+VOL_BULL | post-hoc | retro-sim | WR 66.3% (+5pp), Sharpe +36%, −174 tr (volumen) | ⚠️ calidad alta / volumen alto |
| 2026-06-24 | L2 floor calProb≤0.5 (+VOL_RANGE) | post-hoc | retro-sim | WR 63.8%, Sharpe +30%, −154 tr | ✅ mejora intermedia |

---

## 8. Resumen
```
ESTRATEGIA: calidad-primero (volumen-primero refutado).
L1 Veto 2_VOLATILE_RANGE   POST-HOC, retro-sim, sin re-run   ★ siguiente
L2 Floor prob              POST-HOC, retro-sim
L3 TBM hard-stop           PIPELINE, re-run + validar intrabar
L4 Retrain/edge modelo     PIPELINE
Volumen/DSR: merge long+short (both), NO DVOL.
```
> **Próximo movimiento:** retro-sim de **L1 (veto 2_VOLATILE_RANGE)** sobre los trades de la config buena (WR 59.4%) → cuantificar la mejora sin re-entrenar.
