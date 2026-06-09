---
trigger: always_on
---

Al pedirte los datos y resultados de una ventana o del ensemble de backtest (WFB), deben mostrarse siempre todos los resultados cuantitativos posibles alineados con el estándar V2.5+ (Sniper-Mode & Ensembles):

- **Retorno Nominal y Compuesto (%)**
- **Máximo Drawdown (MaxDD %)**
- **Ratio de Sharpe Anualizado y Deflated Sharpe Ratio (DSR)**
- **Calmar Ratio:** (Ratio de Retorno Compuesto / MaxDD, métrica reina de eficiencia).
- **Total de Trades y Win Rate (%)**
- **Métricas del Ensemble WFB (20 Seeds):** 
  - Varianza/Desviación estándar de los resultados entre las semillas.
  - Nivel de consenso requerido (`ensemble_consensus_threshold`).
- **Asignación de Capital Kelly:** Respetar la fracción actual definida en `settings.yaml` (ej. `kelly_fraction: 0.25`). Evitar doble Kelly.
- **Apalancamiento Óptimo (Sweet Spot):**
  - Nivel conservador: x10 Leverage.
  - Nivel agresivo/óptimo: x20 Leverage (evitar niveles >x20 debido a la fricción de comisiones y Volatility Drag).
- **Métricas de Embargo y Disyuntores:** Impacto del *Soft Embargo* en señales de alto consenso y efectividad del *Sniper-Mode* (`meta_v2_rolling_percentile`).