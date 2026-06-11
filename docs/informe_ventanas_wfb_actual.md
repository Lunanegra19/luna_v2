# Reporte de Diagnóstico WFB (Todas las semillas)
Este informe consolida todos los datos recopilados de las ventanas calculadas hasta el momento en la corrida de ensamble WFB actual.

## Resumen por Semilla
| Semilla | Estado | Trades (Total) | Win Rate (promedio) |
|---|---|---|---|
| 42 | 🛑 Early Stop | 16 | 53.97% |
| 100 | 🛑 Early Stop | 25 | 50.00% |
| 777 | 🛑 Early Stop | 7 | 57.14% |
| 888 | 🛑 Early Stop | 31 | 70.61% |
| 999 | 🛑 Early Stop | 21 | 59.33% |
| 1111 | ⏳ En Proceso | 7 | 25.00% |
| 1337 | 🛑 Early Stop | 18 | 57.22% |
| 2024 | 🛑 Early Stop | 21 | 52.38% |
| 2025 | 🛑 Early Stop | 26 | 73.86% |
| 2026 | 🛑 Early Stop | 15 | 40.00% |
| 2222 | 🛑 Early Stop | 21 | 65.00% |
| 3333 | 🛑 Early Stop | 22 | 47.22% |
| 4444 | 🛑 Early Stop | 9 | 83.33% |
| 5555 | 🛑 Early Stop | 17 | 65.71% |
| 6666 | ⏳ En Proceso | 18 | 59.37% |
| 7777 | 🛑 Early Stop | 22 | 45.00% |
| 8888 | ⏳ En Proceso | 17 | 35.29% |
| 9999 | 🛑 Early Stop | 21 | 69.44% |
| Semillas adicionales no listadas con 0 trades | 🛑 Early Stop (En fases previas) | 0 | 0.00% |

## Detalle de Early Stops (Mecanismos de Poda)
- **Seed 42**: [FIX-D2] Sharpe parcial=-1.5726 < -0.1 con 33 trades en W[1, 4]
- **Seed 100**: [FIX-D2] Sharpe parcial=-0.7083 < -0.1 con 25 trades en W[1, 3]
- **Seed 777**: [FIX-D2] Sharpe parcial=-2.8635 < -0.1 con 153 trades en W[2, 4]
- **Seed 888**: [FIX-D2] Sharpe parcial=-0.4905 < -0.1 con 27 trades en W[3, 4]
- **Seed 999**: upper_bound=50.0 < threshold=51.4 tras W[1, 2, 3, 4]
- **Seed 1337**: upper_bound=65.0 < threshold=68.8 tras W[2, 3, 4]
- **Seed 2024**: [FIX-D2] Sharpe parcial=-0.1612 < -0.1 con 31 trades en W[2, 3]
- **Seed 2025**: upper_bound=51.6 < threshold=51.8 tras W[2, 3, 4]
- **Seed 2026**: [FIX-D2] Sharpe parcial=-0.5899 < -0.1 con 15 trades en W[1, 3]
- **Seed 2222**: [FIX-D2] Sharpe parcial=-0.8031 < -0.1 con 20 trades en W[1, 3]
- **Seed 3333**: [FIX-D2] Sharpe parcial=-0.3224 < -0.1 con 16 trades en W[1, 3]
- **Seed 4444**: upper_bound=40.4 < threshold=68.8 tras W[1, 3, 4]
- **Seed 5555**: upper_bound=55.4 < threshold=68.8 tras W[2, 3, 4]
- **Seed 7777**: [FIX-D2] Sharpe parcial=-0.2502 < -0.1 con 20 trades en W[3, 4]
- **Seed 9999**: upper_bound=64.5 < threshold=68.8 tras W[1, 3, 4]
- Otras semillas fueron filtradas prematuramente en W1/W2 sin trades sobrevivientes.

## Rendimiento Desglosado por Ventana (Principales Semillas con Trades)

### Semilla 42
| Ventana | Trades | Win Rate | Retorno Medio |
|---|---|---|---|
| W2 | 3 | 66.67% | +0.0513% |
| W4 | 6 | 66.67% | -0.0165% |
| W5 | 7 | 28.57% | -0.0092% |

### Semilla 888
| Ventana | Trades | Win Rate | Retorno Medio |
|---|---|---|---|
| W3 | 22 | 31.82% | -0.0169% |
| W4 | 5 | 80.00% | +0.0438% |
| W5 | 4 | 100.00% | +0.0207% |

### Semilla 999
| Ventana | Trades | Win Rate | Retorno Medio |
|---|---|---|---|
| W1 | 1 | 0.00% | -0.1751% |
| W2 | 6 | 66.67% | +0.0204% |
| W3 | 8 | 50.00% | +0.0076% |
| W4 | 5 | 80.00% | +0.0011% |
| W5 | 1 | 100.00% | +0.0191% |

### Semilla 2025
| Ventana | Trades | Win Rate | Retorno Medio |
|---|---|---|---|
| W1 | 8 | 50.00% | -0.0177% |
| W3 | 11 | 45.45% | +0.0143% |
| W4 | 2 | 100.00% | +0.0425% |
| W5 | 5 | 100.00% | +0.1537% |

### Semilla 5555
| Ventana | Trades | Win Rate | Retorno Medio |
|---|---|---|---|
| W3 | 7 | 57.14% | -0.0048% |
| W4 | 5 | 80.00% | +0.0438% |
| W5 | 5 | 60.00% | +0.0305% |

### Semilla 6666 (En Proceso)
| Ventana | Trades | Win Rate | Retorno Medio |
|---|---|---|---|
| W1 | 5 | 40.00% | -0.0215% |
| W3 | 7 | 71.43% | +0.0221% |
| W5 | 6 | 66.67% | +0.0101% |

### Semilla 9999
| Ventana | Trades | Win Rate | Retorno Medio |
|---|---|---|---|
| W3 | 12 | 41.67% | -0.0088% |
| W4 | 3 | 100.00% | +0.0390% |
| W5 | 6 | 66.67% | +0.0065% |

## Conclusión Parcial
1. La gran mayoría de las semillas están siendo detenidas prematuramente por el mecanismo **`[FIX-D2] Sharpe parcial < -0.1`** (Early Stopping por rentabilidad inaceptable) o por el límite estadístico de **upper_bound de Win Rate** de CPCV. 
2. Esto indica que el sistema está siendo sumamente agresivo y riguroso protegiendo el capital, abortando el entrenamiento en cuanto se detecta que las estrategias degeneran a DSR negativo.
3. El ensamble global al final agrupará solo los trades de las ventanas que lograron superar estos estrictos controles.
