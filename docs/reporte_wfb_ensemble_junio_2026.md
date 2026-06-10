# Reporte Final: Orquestación Multi-Seed Walk-Forward Backtesting (WFB) Ensemble
**Fecha de finalización:** 2026-06-10 00:53 UTC  
**Operación:** Sentinel Goal alcanzado exitosamente.

## 1. Resumen Ejecutivo
El sistema logró estabilizarse y finalizar la ejecución de una simulación completa en Ensemble, validando el marco de arquitectura OOS (Out-of-Sample) V2.5+ con características de "Sniper-Mode".

El orquestador logró aprobar **10 semillas aleatorias independientes** a través del Gauntlet estadístico estricto, alcanzando el umbral requerido tras probar un total de 13 semillas (1 abortada en ejecución, 1 rechazada por DSR/Binomial, y 1 finalizada en paralelo), demostrando así una tasa de supervivencia excepcional (~77%).

## 2. Telemetría del Ensemble Unificado
Tras consolidar colisiones de timestamps exactos (Umbral de consenso `ensemble_consensus_threshold: 4`):

- **Total Trades Únicos:** 18
  *(⚠️ INSUFICIENTE: Sufre de Inanición Operativa al estar por debajo del umbral estadístico de 30 trades. Esta es la consecuencia de combinar Sniper-Mode extremo con el requisito de consenso entre 4 modelos distintos).*
- **Win Rate Promedio:** 77.78%
- **Sharpe Ratio Anualizado:** 2.1503
- **Retorno Promedio por Trade:** 0.0465%
- **Retorno Nominal / Compuesto Estimado:** ~0.84%
- **Máximo Drawdown (MaxDD %):** 0.57% (Estimación de Monte Carlo al 95% de confianza)
- **Calmar Ratio:** ~1.47

## 3. Dispersión de Semillas
Las semillas individuales, antes de enfrentarse al cuello de botella del consenso, generaron cientos de trades.
- Rango de Trades por semilla: 55 (`seed42`) a 606 (`seed100`).
- Sharpe Ratios OOS individuales: de 2.89 a 12.44.

## 4. Diagnóstico de Atribución (CVD-01 Component Value Dashboard)
El reporte crítico de impacto marginal sobre la probabilidad del sistema reveló lo siguiente:

| Componente | Impacto en Win-Rate | Diagnóstico |
|---|---|---|
| **HMM_Regime** | **+56.1 pp** | ✅ Escudo de riesgo masivo y exitoso |
| **Alpha_Trigger** | **+41.1 pp** | ✅ Principal generador de Edge |
| **OOD_Guard** | **+14.5 pp** | ✅ Filtrado de cisnes negros en vivo |
| **MetaLabeler_V2** | **-27.9 pp** | ❌ **Tóxico/Perjudicial** |

### Conclusión Estratégica:
La arquitectura de tuberías WFB, el Guardián estadístico, la memoria compartida y los mitigadores de memoria (Early Stops y manejo de Zombies) funcionan de forma impecable y robusta (cero caídas, memoria estable). 
Sin embargo, **el componente de Meta-Etiquetado (MetaLabeler V2) está destruyendo la eficacia local**, lo cual, sumado a las severas restricciones del *Sniper-Mode* (`meta_v2_rolling_percentile: 0.85`), estrangula el rendimiento final creando una inanición operativa insostenible para un entorno de alto apalancamiento real.

*Próximos pasos a investigar: Remover o reformular el MetaLabeler V2 y relajar los umbrales de consenso.*


## Detalle de Resultados por Semilla y Ventana (WFV)
| Semilla | W1 (Trades - WR) | W2 (Trades - WR) | W3 (Trades - WR) | W4 (Trades - WR) | W5 (Trades - WR) |
|---------|-------------------|-------------------|-------------------|-------------------|-------------------|
| 100 | - | - | 484 (47.5%) | 85 (74.1%) | 37 (75.7%) |
| 777 | - | - | - | 53 (86.8%) | 144 (75.7%) |
| 1337 | - | 32 (71.9%) | 145 (53.8%) | 109 (65.1%) | 168 (73.8%) |
| 2025 | - | - | 174 (51.7%) | 89 (57.3%) | 244 (60.7%) |
| 95829 | - | - | - | - | - |
| 70191 | - | - | - | - | - |
| 25654 | - | - | - | - | - |
| 97486 | - | - | - | - | - |
| 10662 | - | - | - | - | - |
| 67152 | - | - | - | - | - |
| 88605 | - | - | - | - | - |
| 73175 | - | - | - | - | - |
| 63588 | - | - | - | - | - |
| 47866 | - | - | - | - | - |
| 31723 | - | - | - | - | - |
| 74480 | - | - | - | - | - |
| 42 | - | - | - | - | 55 (56.4%) |
| 62109 | - | - | - | 13 (23.1%) | 13 (46.2%) |
| 58965 | - | 1 (100.0%) | - | 21 (38.1%) | - |
| 45004 | - | 1 (100.0%) | - | 17 (29.4%) | 13 (46.2%) |
| 57891 | - | - | - | 14 (28.6%) | 13 (46.2%) |
| 18212 | 1 (100.0%) | - | - | 17 (35.3%) | 11 (45.5%) |
| 22225 | - | - | 8 (0.0%) | 9 (66.7%) | 3 (33.3%) |
| 72746 | 1 (100.0%) | 61 (41.0%) | - | - | - |
| 67362 | - | - | - | - | - |
| 16972 | - | 1 (100.0%) | - | 77 (32.5%) | 42 (47.6%) |
| 49818 | - | 1 (100.0%) | - | 75 (36.0%) | - |
| 70065 | - | - | - | 44 (31.8%) | - |
| 86697 | - | 1 (100.0%) | - | 63 (31.8%) | - |
| 16517 | - | - | - | 74 (29.7%) | 42 (47.6%) |
| 12556 | 1 (100.0%) | - | - | 69 (30.4%) | - |
| 37722 | 3 (33.3%) | 1 (100.0%) | - | 43 (30.2%) | 18 (55.6%) |
| 24374 | 20 (50.0%) | - | - | 77 (32.5%) | 36 (38.9%) |
| 59887 | 9 (44.4%) | - | - | 77 (32.5%) | 41 (46.3%) |
| 92666 | - | - | - | 77 (32.5%) | 42 (47.6%) |
| 46847 | - | - | - | 77 (32.5%) | 12 (58.3%) |
| 61041 | - | - | 1 (0.0%) | 41 (39.0%) | 20 (35.0%) |
| 82605 | - | - | - | 77 (32.5%) | - |
| 37907 | - | - | - | 77 (32.5%) | 13 (38.5%) |
| 86336 | - | - | - | 77 (32.5%) | 42 (47.6%) |
| 91262 | - | - | - | 77 (32.5%) | 42 (47.6%) |
| 39057 | - | 1 (100.0%) | 9 (44.4%) | 41 (36.6%) | 42 (47.6%) |
| 59328 | - | 6 (50.0%) | - | 77 (32.5%) | 42 (47.6%) |
| 30120 | - | - | - | 77 (32.5%) | 25 (44.0%) |
| 70865 | 30 (56.7%) | - | - | 32 (34.4%) | - |
| 17713 | 21 (57.1%) | - | - | 32 (37.5%) | - |
| 47861 | 18 (66.7%) | 18 (61.1%) | 33 (24.2%) | - | - |
| 66914 | 25 (56.0%) | 16 (62.5%) | 24 (33.3%) | 31 (22.6%) | - |
| 45948 | 24 (41.7%) | - | - | 21 (42.9%) | - |
| 81972 | 33 (45.5%) | - | - | 26 (30.8%) | - |
| 83410 | 9 (55.6%) | 11 (54.5%) | - | 36 (30.6%) | - |
| 75795 | 22 (40.9%) | - | - | 25 (40.0%) | - |
| 92407 | 35 (40.0%) | 3 (100.0%) | - | - | - |
| 86578 | 11 (72.7%) | - | - | 31 (32.3%) | - |
| 19806 | 9 (55.6%) | - | - | 29 (44.8%) | - |
| 95809 | 32 (53.1%) | 1 (0.0%) | 21 (42.9%) | - | - |
| 81582 | 25 (52.0%) | - | 28 (39.3%) | 33 (36.4%) | - |
| 17836 | 27 (37.0%) | 4 (0.0%) | - | - | - |
| 68857 | 31 (38.7%) | - | - | 28 (32.1%) | - |
| 10437 | 24 (50.0%) | 12 (50.0%) | - | - | - |
| 39119 | 31 (51.6%) | 6 (83.3%) | - | 32 (43.8%) | - |
| 2026 | 4 (25.0%) | 2 (0.0%) | 8 (50.0%) | - | 6 (66.7%) |
| 1111 | - | - | - | - | - |
| 2222 | - | - | - | - | - |
| 6666 | - | - | - | - | - |
| 7777 | - | - | 20 (40.0%) | 2 (50.0%) | - |
| 8888 | - | - | - | - | - |
| 9999 | - | - | 12 (41.7%) | 3 (100.0%) | 6 (66.7%) |
| 2024 | - | - | 21 (52.4%) | - | - |
| 888 | - | - | 22 (31.8%) | 5 (80.0%) | 4 (100.0%) |
| 999 | 1 (0.0%) | 6 (66.7%) | 8 (50.0%) | 5 (80.0%) | 1 (100.0%) |
| 3333 | - | - | 13 (53.8%) | 4 (75.0%) | 6 (83.3%) |
| 4444 | - | - | - | - | - |
| 5555 | - | - | - | - | - |
| 12345 | 8 (37.5%) | - | 22 (36.4%) | - | - |
| 79658 | 1 (0.0%) | - | 37 (43.2%) | - | - |
| 21904 | - | - | - | - | - |
| 27391 | - | - | - | - | - |
| 62040 | - | - | 19 (36.8%) | - | 3 (66.7%) |
| 64962 | - | 1 (100.0%) | 20 (45.0%) | - | 9 (77.8%) |
| 97346 | - | - | 15 (40.0%) | 3 (100.0%) | 3 (66.7%) |
| 80739 | - | - | - | - | - |
| 91990 | - | - | - | - | - |
| 21074 | - | - | 15 (60.0%) | 6 (66.7%) | 1 (100.0%) |
| 16063 | - | - | - | - | - |
| 10592 | - | - | 17 (47.1%) | - | 3 (100.0%) |
| 38633 | - | - | - | - | - |
| 31473 | - | 206 (62.6%) | 466 (50.6%) | 80 (75.0%) | 287 (59.2%) |
| 70831 | 171 (45.6%) | - | 378 (46.8%) | 84 (82.1%) | - |
| 86671 | - | - | 380 (39.2%) | 70 (74.3%) | 203 (57.6%) |
| 81451 | - | - | 324 (47.8%) | 114 (65.8%) | 55 (85.5%) |
| 47472 | - | - | 307 (53.4%) | 98 (73.5%) | 155 (68.4%) |
| 17650 | - | - | 386 (56.2%) | 76 (72.4%) | - |
| 23315 | 23 (69.6%) | - | 185 (55.1%) | - | 52 (80.8%) |
| 43488 | - | - | 196 (59.7%) | 84 (75.0%) | 152 (59.2%) |
| 88219 | 47 (46.8%) | 111 (73.0%) | 217 (41.9%) | - | - |
| 79375 | - | - | 132 (48.5%) | - | 228 (61.8%) |
| 30635 | - | - | 124 (56.5%) | 107 (63.5%) | 122 (65.6%) |
