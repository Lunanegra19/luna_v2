# Investigación Forense: Colapso por Covariate Shift (WFB Seed 42 - W3)
**Fecha:** 2026-06-15
**Módulo:** SFI / XGBoost / MetaLabeler V2
**Autor:** Antigravity (Sistema Centinela)

## 1. Contexto del Colapso
Durante la ejecución masiva del Ensamble WFB (Walk-Forward Backtesting) con 20 semillas, se detectó una inanición operativa crítica y pérdida sistemática de *Edge* a partir de la Ventana 3 (Holdout 2025+). 
El `Component Value Dashboard (CVD)` evidenció que el modelo `XGBoost` y el `MetaLabeler V2` estaban operando con una correlación inversa a su entrenamiento (Covariate Shift). Específicamente, el sistema obtenía retornos positivos en las barras que el `OOD_Guard` (Isolation Forest) catalogaba como anomalías absolutas.

## 2. Metodología de Diagnóstico (Drift Analysis)
Para evitar el P-Hacking y el sobreajuste retrospectivo (SOP Regla R4), se ejecutó un script de análisis estadístico (`diagnose_covariate_shift.py`) directamente sobre el caché de datos que utilizó el orquestador (`wfb_cache/W3`). 
Se comparó la distribución estadística (Media y Varianza) de las 515 variables generadas por el pipeline entre el periodo In-Sample (`features_train.parquet`, 2017-2024) y el periodo Out-of-Sample (`features_holdout.parquet`, Q1 2025+).

Se calculó el **Z-Shift** de cada variable:
`Z_Shift = (Media_OOS - Media_IS) / Desviacion_Estandar_IS`

## 3. Hallazgos: Las Variables Tóxicas (Toxic Features)
El análisis reveló que las predicciones erróneas de XGBoost fueron causadas por su alta dependencia histórica en un conjunto de variables que sufrieron una ruptura estructural masiva (Z-Shift > 3.3).

| Feature (Variable) | Z-Shift OOS | Ratio Varianza | Diagnóstico Causal de Ruptura |
| :--- | :---: | :---: | :--- |
| `XRP_Price` | 4.21 | 0.26 | Desacople del precio base tras litigios y evolución independiente del ecosistema. |
| `Gold` | 4.02 | 0.25 | Ruptura de la correlación histórica / Cambio de régimen macroeconómico. |
| `SSR` | 3.44 | 0.02 | Stablecoin Supply Ratio obsoleto. Nuevas mecánicas de emisión (USDT/FDUSD) rompen el ratio. |
| `GBTC_Low` | 3.43 | 0.02 | **Catalizador Primario.** Conversión de Grayscale Trust a ETF Spot. Desaparición del premium/discount. |
| `pi_cycle_ma350` | 3.40 | 0.05 | Ruptura matemática de medias móviles absolutas de largo plazo. |
| `mc_btc_dxy_ratio`| 3.37 | 0.02 | Ruptura de la correlación inversa histórica entre Bitcoin y el Índice Dólar. |
| `close`, `high` | 3.37 | 0.02 | Precios absolutos sin normalización Z-Score, inyectando sesgo nominal al modelo. |

Estas variables eran sumamente predecibles y estables en 2017-2024, engañando al SFI para que les asignara un alto *Feature Importance*. En 2025, sus distribuciones colapsaron (Ratio de varianza tendiente a 0), convirtiendo la inferencia de XGBoost en ruido aleatorio.

## 4. Hallazgos: Las Variables Estables (Robust Features)
El diagnóstico validó la eficacia de la ingeniería de variables relativas (Z-scores transversales y momentum) creada por el equipo. Estas variables mantuvieron distribuciones idénticas entre IS y OOS (Z-Shift próximo a 0.000).

| Feature (Variable) | Z-Shift OOS | Ratio Varianza | Atributo Estructural |
| :--- | :---: | :---: | :--- |
| `DVOL_kz` | 0.0023 | 0.18 | Neutralidad al precio absoluto (Métricas de opciones relativas). |
| `cal_is_asia_session`| 0.0017 | 1.00 | Componentes de calendario inmutables. |
| `ofi_imb_delta_4h` | 0.0015 | 1.63 | Order Flow Imbalance de alta frecuencia y estacionariedad. |
| `Exchange_NetFlow` | 0.0013 | 1.19 | Flujos puros relativos. |

## 5. Propuesta de Refactorización (Fase SFI)
Para restaurar el *Edge* de la estrategia, se debe intervenir estructuralmente en la configuración institucional:

1. **Purga de Lista Negra (Blacklist):** Insertar `GBTC_*`, `SSR`, `XRP_Price`, `Gold`, `pi_cycle_*` y todas las métricas de precio absoluto (`open`, `high`, `low`, `close`) en la lista de exclusión obligatoria (`sfi_blacklist_features`) dentro de `settings.yaml`.
2. **Desactivación de MetaLabeler y OOD Guard:** Su entrenamiento está sesgado por el pasado pre-ETF. Deben ser apagados (`skip_metalabeler: true`) hasta que XGBoost recupere la causalidad usando exclusivamente el pool de variables estacionarias (`Stable Features`).

diagnose_covariant_sifht.py