## Metodología de Diagnóstico Cuantitativo y Gestión de Fixes

Esta regla codifica la metodología de análisis riguroso demostrada en la sesión 2026-05-31,
donde se identificaron y corrigieron bugs reales en Luna V2 basándose exclusivamente en
evidencia estadística de los datos OOS.

---

### PRINCIPIO FUNDAMENTAL: "Datos Primero, Hipótesis Después"

Antes de proponer cualquier fix, hipótesis o mejora, es **obligatorio**:

1. **Cargar los datos reales** (parquets OOS, firmas de modelos, logs de trades).
2. **Medir las estadísticas base** con scripts de análisis ejecutables.
3. **Formular hipótesis** solo sobre lo que los datos revelan, nunca sobre suposiciones.
4. **Descartar hipótesis** mediante tests estadísticos antes de implementar cualquier cambio.

**Prohibido:** Proponer fixes sin haber ejecutado código de diagnóstico sobre los datos reales.

---

### PROTOCOLO DE DIAGNÓSTICO (5 FASES OBLIGATORIAS)

#### FASE 1 — Carga y Estadísticas Base

Antes de cualquier análisis, ejecutar un script que produzca:
- N total de trades por agente/régimen/ventana/seed
- Distribución de retornos: mean, std, mediana, percentiles [10,25,50,75,90]
- Win Rate (WR), Average Win, Average Loss, **P/L ratio** (AvgWin / |AvgLoss|)
- **Expected Value (EV) por trade** = WR × AvgWin + (1-WR) × AvgLoss
- Distribución de la variable principal de interés (ej: `xgb_prob_cal`)

```python
# Patrón mínimo de carga
all_dfs = []
for f in sorted(wfb_dir.glob('oos_trades_W*_seed*.parquet')):
    df = pd.read_parquet(f)
    df['seed'] = int(f.stem.split('_seed')[1])
    df['window'] = f.stem.split('_')[2]
    all_dfs.append(df)
df_all = pd.concat(all_dfs, ignore_index=True)
```

#### FASE 2 — Formulación de Hipótesis (Máximo 3-5)

Cada hipótesis debe:
- Ser **falsable** con los datos disponibles
- Tener una **consecuencia medible** específica (ej: "si H es verdad, WR > X en el subgrupo Y")
- Ser **ordenadas por impacto esperado** en EV o Sharpe

**Anti-patrón prohibido:** "El problema podría ser A, o B, o C..." sin medir cuál.

#### FASE 3 — Tests Estadísticos por Hipótesis

Para cada hipótesis, aplicar el test apropiado y **reportar p-value**:

| Tipo de pregunta | Test correcto |
|---|---|
| ¿Son distintas dos distribuciones? | `scipy.stats.ks_2samp` |
| ¿Hay diferencia de medias? | `scipy.stats.ttest_ind` |
| ¿Hay correlación? | `scipy.stats.spearmanr` (no asume linealidad) |
| ¿El WR es significativamente > 50%? | `scipy.stats.binom_test` |
| ¿Hay diferencia entre grupos? | `pandas.groupby + apply` con métricas por grupo |

**Umbral mínimo:** p < 0.05 para aceptar una hipótesis. Si p > 0.05, la hipótesis se **descarta**.

Ejemplo de patrón correcto:
```python
# HIPÓTESIS: los trades cortos (<12h) tienen mayor xgb_prob que los largos (>36h)
t, p = stats.ttest_ind(df_fast['xgb_prob_cal'], df_slow['xgb_prob_cal'])
print(f't={t:.2f} p={p:.4f} → {"CONFIRMADA" if p < 0.05 else "DESCARTADA"}')
```

#### FASE 4 — Diagnóstico de Causa Raíz

Solo después de confirmar una hipótesis, trazar la causa raíz en el código:
- Leer el **código fuente exacto** de la función incriminada
- Identificar la **línea precisa** donde ocurre el bug
- Verificar que la causa raíz explica **todos** los síntomas observados
- Confirmar que no hay otras causas concurrentes

**Anti-patrón prohibido:** Proponer un fix antes de haber leído el código fuente.

#### FASE 5 — Counterfactual antes de Implementar

Antes de codificar el fix, estimar su impacto:
- ¿Cuántos trades se afectan?
- ¿El fix cambia señales (P&L) o solo trazabilidad?
- ¿Hay IS/OOS mismatch si se cambia un parámetro de entrenamiento en OOS?
- ¿El fix requiere reentrenar modelos o es aplicable en inference?

---

### CHECKLIST DE VALIDACIÓN POR FIX

Antes de marcar un fix como completado, verificar **todos** los puntos:

- [ ] **Script de investigación ejecutado** con datos reales (no simulados)
- [ ] **Hipótesis falsadas** estadísticamente (p-value reportado)
- [ ] **Causa raíz identificada** en el código fuente (archivo + línea)
- [ ] **Counterfactual estimado**: impacto en EV, Sharpe, N de trades
- [ ] **Fix implementado** en el archivo correcto (no en scripts de diagnóstico)
- [ ] **Print de trazabilidad añadido** según `fixbugsprints.md` (regla obligatoria)
- [ ] **Test unitario ejecutado** y pasado
- [ ] **No hay runs activas** al modificar `settings.yaml` (ver `settings_restore_protection.md`)
- [ ] **P/L ratio, WR, EV reportados** en el diagnóstico final (ver `windowstats.md`)

---

### PATRONES DE ANÁLISIS PROBADOS

#### Análisis de P/L ratio por subgrupo
```python
# Descomponer P/L por régimen, ventana, duración
pl_by_group = df.groupby('grupo').apply(lambda g: pd.Series({
    'n': len(g),
    'wr': (g['return_raw'] > 0).mean() * 100,
    'avg_win': g[g['return_raw'] > 0]['return_raw'].mean() * 100,
    'avg_loss': g[g['return_raw'] <= 0]['return_raw'].mean() * 100,
    'pl_ratio': abs(wins.mean() / losses.mean()),
    'ev_trade': g['return_raw'].mean() * 100,
}))
```

#### Correlación variable → retorno
```python
# Spearman (no asume linealidad ni normalidad)
for var in ['xgb_prob_cal', 'meta_v2_prob', 'kelly_fraction_used']:
    r, p = stats.spearmanr(df[var].dropna(), df.loc[df[var].notna(), 'return_raw'])
    print(f'{var}: r={r:+.4f} p={p:.4f} | {"SIGNIFICATIVO" if p < 0.05 else "ruido"}')
```

#### Sweep de umbrales (counterfactual)
```python
# Evaluar qué threshold óptimo hubiera generado mejor EV
for thresh in np.arange(0.50, 0.75, 0.02):
    sub = df[df['xgb_prob_cal'] >= thresh]
    if len(sub) < 20: continue
    print(f'thresh>={thresh:.2f}: N={len(sub)} WR={100*(sub.return_raw>0).mean():.1f}% '
          f'EV={sub.return_raw.mean()*100:+.3f}%')
```

---

### ERRORES METODOLÓGICOS PROHIBIDOS

1. **Sesgo de selección al reportar resultados**: No reportar solo los subgrupos con p < 0.05.
   Siempre reportar todos los subgrupos analizados, incluyendo los no significativos.

2. **Confundir correlación con causalidad**: Si variable X correlaciona con retorno,
   investigar si hay una causa común (ej: ambas correlacionan con el régimen HMM).

3. **Sweep de parámetros sobre datos OOS sin corrección**: Un sweep sobre los mismos datos
   que se usarán para evaluar genera overfitting. El sweep debe hacerse IS y validarse OOS
   en datos distintos. Reportar siempre si el sweep es IS o OOS.

4. **Cambiar parámetros de entrenamiento sin reentrenar**: Si un parámetro afecta la
   generación de labels TBM (ej: `vertical_barrier_hours`), cambiarlo en `settings.yaml`
   sin reentrenar crea IS/OOS mismatch. Documentar explícitamente en cada fix si requiere
   reentrenamiento.

5. **Extrapolación de N pequeño**: Si un subgrupo tiene N < 30, los resultados son
   exploratorios (no concluyentes). Siempre reportar N junto a cada métrica.

6. **Diagnóstico con datos de pocas seeds**: Resultados con 2-4 seeds pueden ser
   estadísticamente distintos a los de 12 seeds. Siempre usar el máximo de seeds disponibles.
   El error de muestreo disminuye con √N_seeds.

---

### COMUNICACIÓN DE RESULTADOS

Al reportar un diagnóstico o fix, siempre incluir:

```
BUG IDENTIFICADO: [nombre descriptivo del bug]
EVIDENCIA: [métricas concretas, ej: "72% de pares con std_cal < 1e-4"]
CAUSA RAÍZ: [archivo, función, línea]
IMPACTO MEDIDO: [EV/WR/N afectados]
FIX IMPLEMENTADO: [archivo modificado, tipo de cambio]
TESTS: [resultado de los tests unitarios]
IMPACTO ESPERADO EN PRÓXIMA RUN: [predicción cuantitativa]
```

**Prohibido:** Frases vagas como "podría mejorar el rendimiento" o "debería ayudar".
Toda afirmación de impacto debe ser cuantitativa o marcada como "estimación sin datos".
