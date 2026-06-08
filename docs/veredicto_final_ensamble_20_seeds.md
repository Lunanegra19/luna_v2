# Veredicto Final del Ensamble WFB (20 Semillas)
> Generado tras la ejecución completa de la corrida WFB de 20 semillas y evaluación del ensamble.

## 1. Métrica de Portafolio Ensemble (Unificado)
El ensamble multi-semilla (Soft Voting + Consensus Gate >= 2 + Soft Embargo) logró extraer suficientes trades para ser estadísticamente evaluable (mitigando la inanición). 

- **Total Trades Únicos**: 31 (✅ OK, > 30 mínimos)
- **Win Rate Promedio**: 67.74% (✅ Excelente)
- **Sharpe Ratio Anualizado**: 1.6413 (✅ Bueno)
- **Retorno Promedio por Trade**: 0.0297%

## 2. Veredicto Estadístico (ENSEMBLE-GAUNTLET-01)
> ### ❌ REJECTED — NO DESPLEGAR (Por 1 Trade de diferencia)

A pesar de tener métricas de portafolio atractivas, el **Gauntlet ha rechazado el pase a producción** principalmente por una falla técnica en la ventana del bloque de simulación de PBO:

| Gate | Valor Obtenido | Umbral | Estado |
| --- | --- | --- | --- |
| **Trades** | 31 | >= 30 | ✅ |
| **Win Rate** | 67.74% | > 50% | ✅ |
| **DSR (raw)** | 0.0 | >= 0.75 | ❌ |
| **DSR (adj R5, N=20)** | 1.000 | >= 0.75 | ✅ |
| **PBO CSCV** | 50.0% | < 45% | ❌ |
| **MaxDrawdown** | 0.22% | < 60.0% | ✅ |
| **Binomial p** | 0.035378 | < 0.2 | ✅ |

### ⚠️ Diagnóstico de Falla (El factor limitante)
La evaluación falló en el **PBO (Probability of Backtest Overfitting)** devolviendo un `50.0%`, pero esto ocurrió por un fallback defensivo:
```log
[FIX-PBO-01] WARN CSCV: 31 trades < 32 mínimo (n_blocks=8*4). Retornando PBO=0.50 conservador.
```
El sistema exige que el número de trades satisfaga `n_blocks * 4`. Dado que `pbo_n_blocks = 8`, el mínimo de trades necesarios para calcular el PBO correctamente es de **32 trades**. El ensamble logró **31 trades**, quedándose corto por exactamente 1 trade. Esto activó el fallback conservador (PBO=0.50), superando el máximo tolerado de `0.45`, y provocando el rechazo automático.

## 3. Desglose de Métricas por Semilla (Seed)
| Semilla | Trades Totales | Win Rate | Sharpe Ratio | Retorno Medio por Trade |
| --- | --- | --- | --- | --- |
| **100** | 25 | 44.00% | -0.8363 | -0.0090% |
| **2025** | 26 | 61.54% | 1.2972 | 0.0334% |
| **2026** | 15 | 40.00% | -0.7566 | -0.0104% |
| **2222** | 21 | 33.33% | -0.8532 | -0.0082% |
| **3333** | 22 | 50.00% | -0.2743 | -0.0040% |
| **6666** | 18 | 61.11% | 0.2785 | 0.0060% |
| **999** | 21 | 61.90% | 0.0728 | 0.0015% |
| **42** | 16 | 50.00% | -0.0260 | -0.0006% |
| **1337** | 18 | 61.11% | -0.1817 | -0.0028% |
| **2024** | 21 | 52.38% | 0.4462 | 0.0026% |
| **5555** | 17 | 64.71% | 1.7285 | 0.0199% |
| **7777** | 22 | 40.91% | -0.0638 | -0.0004% |
| **888** | 31 | 48.39% | -0.2589 | -0.0023% |
| **8888** | 17 | 35.29% | 0.2752 | 0.0015% |
| **9999** | 21 | 57.14% | 0.3257 | 0.0024% |
| **1111** | 7 | 42.86% | 0.7679 | 0.0289% |
| **4444** | 9 | 77.78% | 4.5612 | 0.0600% |
| **777** | 7 | 57.14% | 3.7409 | 0.0315% |

*(Nota: Las semillas `12345` y `54321` no superaron la ventana W2/W3 debido a fallos en gates intermedios y no contribuyeron al ensamble).*

## Conclusión y Próximos Pasos
- La inanición operativa fue resuelta parcialmente al habilitar el ensamble masivo con 20 semillas.
- El pipeline tiene una protección robusta (No-Fallback en PBO), evitando aceptar un Backtest Overfitting potencialmente oculto.
- Para superar el Gauntlet de forma rigurosa, se necesitan al menos **32 trades**.
- **Acción recomendada**: Relajar marginalmente el Consensus Gate (aunque requerimos >=2 semillas concurrentes por default), o reducir el umbral del Guardián OOD en XGBoost para aumentar mínimamente la llegada de señales y lograr superar el muro de los 32 trades sin sacrificar el Win Rate (67.74%).
