---
description: 
---

1. Integridad Matemática y Financiera (Quantitative Math)
Ajuste de Riesgo Penalizado: Validación estricta de la fórmula del Deflated Sharpe Ratio (Bailey & LdP) y Probability of Backtest Overfitting (PBO). ¿Se están penalizando correctamente los trials iterativos?
Estacionariedad de Datos (FracDiff): Comprobación de los tests Augmented Dickey-Fuller (ADF). ¿Se está aplicando la diferenciación fraccional de forma óptima sin borrar la memoria del mercado?
Métricas de Información (Mutual Information): Análisis causal de las features usadas en el HMM. ¿Los regímenes ocultos realmente anticipan distribuciones de retornos, o son solo clusters de ruido?
Calibración de Probabilidades: Revisión de las métricas de Brier Score y Platt Scaling (CalibratedClassifierCV). ¿Las probabilidades emitidas por el MetaLabeler reflejan frecuencias empíricas reales?
Dimensionamiento de Posiciones (Sizing): Evaluación del Fraccional Kelly adaptativo, Target Volatility (EWMA) y los Drawdown Kill-Switches. ¿La matemática de la función de coste y simulación de slippage incluye interés compuesto negativo?
2. Rigor Lógico y Anti-Leakage (Defensas contra Look-Ahead Bias)
Asincronía Macro y On-Chain: Verificación matemática de que la publicación de indicadores económicos (ej. M2 de la FED, IPC) o métricas de la blockchain sufran el retraso adecuado (lags) para simular la ignorancia del agente en tiempo real.
Fronteras Temporales (Splits): Demostración absoluta de que existe aislamiento de datos: Train < Validation < Holdout.
Purge & Embargo (CPCV): Validación algorítmica del Combinatorial Purged Cross-Validation. ¿Se están eliminando correctamente las barras superpuestas para evitar la filtración temporal entre grupos K-Fold?
Normalizaciones Cegadas: Los estimadores estadísticos (Kalman Z-Scores, medias móviles, StandardScalers) jamás deben "ver" la distribución futura. Toda normalización OOS debe usar los parámetros ajustados estrictamente In-Sample.
Dinámica de Umbrales: Asegurar que parámetros como xgb_signal_threshold o meta_v2_min_prob no estén hardcodeados mirando resultados pasados, sino que sean resueltos dinámicamente mediante barridos In-Sample.
3. Arquitectura de Machine Learning y Modelado
Robustez del Espacio de Búsqueda (Optuna): Auditoría de los hiperparámetros (profundidad, learning rate, penalizaciones L1/L2) para evitar corner cases donde el modelo colapse en arboles nulos o pesos infinitos.
Ortogonalidad en Feature Selection (SFI): Análisis de los clusters espectrales y la reducción de dimensionalidad (Autoencoders). ¿Las variables predictivas son verdaderamente independientes o el modelo está asumiendo multicolinealidad oculta?
Manejo de Desequilibrio de Clases: El mercado está en ruido el 90% del tiempo. ¿Cómo reaccionan los pesos muestrales (Sample Weights), las tasas de decaimiento temporal y funciones como Focal Loss ante distribuciones asimétricas?
Guardias Fuera de Distribución (OOD): Validación del umbral del Isolation Forest. Si el entorno de producción lanza datos anómalos que el modelo nunca ha visto, ¿se frena la inferencia correctamente?
4. Estructura y Flujo de Software (Ingeniería de Datos)
Invariantes de Ejecución (Orquestación): El pipeline jamás debe violar su orden: Extracción -> Feature Pipeline -> SFI -> HMM -> XGBoost -> MetaLabeler -> OOD -> Calibrador -> Inferencia OOS.
Gestión de Cadenas de Artefactos: Los modelos entrenados (.pkl, .pt) deben referenciar un hash o estampa de tiempo idéntica al archivo features_train.parquet y selected_features.json para garantizar que la inferencia no cruce modelos con datos incompatibles.
Higiene de Namespaces y Legacy: El código en rutas críticas de producción jamás debe invocar o heredar de scripts obsoletos o funciones experimentales marcadas como deprecated.
Gestión de Memoria y Rendimiento: Control exhaustivo de Memory Leaks al pasar matrices de pandas a numpy o cargar embeddings recurrentes en PyTorch, especialmente durante procesos pesados de Walk-Forward.
5. Resiliencia Funcional (Entorno en Producción y Live)
Manejo de Interrupciones de API (Fallbacks): ¿Qué ocurre funcionalmente si Coinglass o Kraken entran en mantenimiento o rechazan la conexión? ¿El sistema sustituye con NaNs, entra en modo espera, o crashea?
Alertas Dinámicas y Telemetría: Supervisar la latencia de ejecución del bot en vivo, estado de las rutinas asíncronas y reportes precisos vía Telegram ante fallos de umbral PSI (Population Stability Index).
Manejo de Reenganche (State Recovery): Si el servidor se apaga repentinamente a mitad del proceso, ¿el orquestador (run_wfb_orchestrator.py) reconoce qué ventanas (W1, W2) ya completó y retoma desde ahí, o sobreescribe y corrompe los archivos?