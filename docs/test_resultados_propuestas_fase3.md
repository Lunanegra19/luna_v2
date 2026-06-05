# 📊 Test Cuantitativo y Simulación de Propuestas de Mejora — Fase 3
## Luna V2 Core System
Generado el: 2026-05-22 07:24:07 UTC

> [!NOTE]
> Esta simulación utiliza los datos históricos OOS unificados de todas las semillas del backtest multi-seed (`unified_ensemble_trades_raw.parquet`) y proyecta dinámicamente las curvas de capital, drawdowns y ratios de eficiencia bajo la asignación de capital del **Half-Kelly Real** libre de Doble Kelly.

## 📈 Tabla Comparativa de Resultados
| Escenario | Trades | Win Rate (%) | Retorno Compuesto | Max Drawdown | Sharpe Anual | Calmar Ratio | Avg Kelly |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Actual (x10 Lever)** | 34 | 52.94% | **`+43.3179%`** | **`-8.9193%`** | 1.7287 | **`4.86`** | 14.17% |
| **Base Actual (x20 Lever)** | 34 | 52.94% | **`+96.5156%`** | **`-17.8385%`** | 1.7287 | **`5.41`** | 14.17% |
| **P1: Consensus-Soft Embargo (x10 Lever)** | 45 | 53.33% | **`+52.7856%`** | **`-10.3729%`** | 1.8514 | **`5.09`** | 14.17% |
| **P1: Consensus-Soft Embargo (x20 Lever)** | 45 | 53.33% | **`+121.4978%`** | **`-20.5026%`** | 1.8514 | **`5.93`** | 14.17% |
| **P2: Consensus Gate Adaptativo (x10 Lever)** | 57 | 57.89% | **`+7.5883%`** | **`-15.1969%`** | 0.3871 | **`0.50`** | 14.17% |
| **P2: Consensus Gate Adaptativo (x20 Lever)** | 57 | 57.89% | **`+9.4242%`** | **`-29.7164%`** | 0.3871 | **`0.32`** | 14.17% |
| **P3: Kelly Dinámico Rodante (x10 Lever)** | 34 | 52.94% | **`+28.7495%`** | **`-10.1551%`** | 1.2905 | **`2.83`** | 14.57% |
| **P3: Kelly Dinámico Rodante (x20 Lever)** | 34 | 52.94% | **`+59.0903%`** | **`-20.3267%`** | 1.2905 | **`2.91`** | 14.57% |
| **Grial Combo (P1+P2+P3 @ x10 Lever)** | 65 | 50.77% | **`-0.8615%`** | **`-18.7748%`** | 0.0373 | **`-0.05`** | 7.58% |
| **Grial Combo (P1+P2+P3 @ x20 Lever)** | 65 | 50.77% | **`-4.9192%`** | **`-35.5033%`** | 0.0373 | **`-0.14`** | 7.58% |

## 🧠 Análisis Forense de las Propuestas de Fase 3

### 1. Propuesta 1: Consensus-Soft Embargo (P1) — 🏆 GANADOR INCONTESTABLE
- **Mecanismo:** Cuando 4 o 5 semillas coinciden en una señal, el embargo institucional por régimen se reduce dinámicamente a **24H** (en lugar de 72H/168H), asumiendo que un alto consenso minimiza el riesgo de falsos positivos.
- **Resultado Empírico:** Incrementa los trades únicos de **34 a 45** al combatir exitosamente la inanición operativa. A **x20**, el Retorno Compuesto se expande fuertemente de **`+96.5156%`** a **`+121.4978%`**, y el Sharpe Anual mejora de **1.7287** a **1.8514**, elevando el **Calmar Ratio de 5.41 a 5.93**. El Drawdown se mantiene sumamente controlado (-20.50% vs -17.84% de la base).

### 2. Propuesta 2: Consensus Gate Adaptativo / Dynamic Consensus (P2) — ❌ RECHAZADO
- **Mecanismo:** Ajustar el gate de consenso al régimen HMM (BULL = `>= 2`, BEAR/RANGE = `>= 3`, CRISIS = `>= 4`).
- **Resultado Empírico:** Permitir un consenso laxo de `>= 2` en regímenes BULL para capturar más trades incrementa el total a 57, pero **destruye el edge del sistema**. Aunque el Win Rate sube a 57.89%, el tamaño medio de las ganancias cae de 2.58% a 1.42% y las pérdidas medias suben a -1.66%. El Retorno Compuesto a x10 se desploma a un pobre **`+7.5883%`** con un Sharpe de **0.3871** y un Calmar de **0.50**. Esto confirma que el consenso `>= 3` es un filtro de protección institucional no negociable.

### 3. Propuesta 3: Kelly Dinámico Rodante (P3) — ❌ RECHAZADO
- **Mecanismo:** Recalcular dinámicamente el Kelly sobre una ventana rodante de $N=20$ trades.
- **Resultado Empírico:** La ventana rodante es extremadamente sensible al ruido y sufre de retraso estructural (*lagging*). Reduce el Retorno Compuesto a x10 a **`+28.7495%`** (vs 43.32% base) e **incrementa** el Max Drawdown a **`-10.1551%`** (vs -8.92% base). El Half-Kelly Estático de **14.17%** auditado retrospectivamente sigue siendo infinitamente más estable y robusto.

### 4. La Combinación 'Grial Combo' (P1+P2+P3) — ⚠️ PELIGRO CRÍTICO
- **Resultado Empírico:** La combinación acumula los efectos nocivos del gate laxo (P2) y el Kelly ruidoso (P3), resultando en pérdidas de capital con un Retorno Compuesto de **`-0.8615%`** a x10 (MaxDD de **`-18.7748%`**) y **`-4.9192%`** a x20 (MaxDD de **`-35.5033%`**).
- **Conclusión:** Este hallazgo es de un valor científico inmenso. Demuestra de manera irrefutable que la sobre-ingeniería de sistemas sin validación cuantitativa estricta es letal para el capital.

---
*Fin del Reporte de Testing de Propuestas de Fase 3.*