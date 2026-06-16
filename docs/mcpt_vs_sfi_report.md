# Analisis Comparativo: SFI vs Tribunal MCPT

Este analisis cruza las caracteristicas seleccionadas por el SFI en la corrida actual (W4, W5, W6) con el filtro anti-ruido MCPT (P-Value <= 0.05).

## Ventana W4
- **Features SFI (moteadas a base)**: 11
- **Sobrevivientes MCPT (Genuinas)**: 1 (9.1%)
- **Censuradas por MCPT (Ruido)**: 10 (90.9%)

### Genuinas
- `ae_feat_42_milag12h` -> Base: `ae_feat_42` (IC: -0.0259, P-Value: 0.0396)
### Ruido Seleccionado por SFI
- `pi_cycle_distance_milag12h` -> Base: `pi_cycle_distance` (IC: -0.0277, P-Value: 0.0693) - Falso Positivo
- `NonEx_Supply_z90d_milag96h` -> Base: `NonEx_Supply` (IC: -0.0203, P-Value: 0.1386) - Falso Positivo
- `Whale_Proxy_Volume_USD_milag500h` -> Base: `Whale_Proxy_Volume_USD` (IC: 0.0176, P-Value: 0.2079) - Falso Positivo
- `EURUSD_milag500h` -> Base: `EURUSD` (IC: 0.0102, P-Value: 0.4356) - Falso Positivo
- `ae_feat_31_milag500h` -> Base: `ae_feat_31` (IC: -0.0095, P-Value: 0.4752) - Falso Positivo
- `WEI_z90d_milag6h` -> Base: `WEI` (IC: -0.0084, P-Value: 0.5941) - Falso Positivo
- `dv_dvol_pct_24h_milag12h` -> Base: `dv_dvol_pct_24h` (IC: -0.0081, P-Value: 0.6436) - Falso Positivo
- `ae_feat_30_milag48h` -> Base: `ae_feat_30` (IC: -0.0026, P-Value: 0.8218) - Falso Positivo
- `ae_feat_35_milag336h` -> Base: `ae_feat_35` (IC: 0.0014, P-Value: 0.9406) - Falso Positivo
- `FundingRate_30d_MA_milag336h` -> Base: `FundingRate_30d_MA` (IC: 0.0000, P-Value: 1.0000) - Falso Positivo

---
## Ventana W5
- **Features SFI (moteadas a base)**: 10
- **Sobrevivientes MCPT (Genuinas)**: 2 (20.0%)
- **Censuradas por MCPT (Ruido)**: 8 (80.0%)

### Genuinas
- `ae_feat_0_milag2h` -> Base: `ae_feat_0` (IC: -0.0566, P-Value: 0.0099)
- `ETH_Return_7d_milag168h` -> Base: `ETH_Return_7d` (IC: 0.0309, P-Value: 0.0396)
### Ruido Seleccionado por SFI
- `Whale_Proxy_Volume_USD_milag500h` -> Base: `Whale_Proxy_Volume_USD` (IC: 0.0176, P-Value: 0.2079) - Falso Positivo
- `DVOL_kz_milag1h` -> Base: `DVOL` (IC: 0.0149, P-Value: 0.3069) - Falso Positivo
- `ath_dist_pct_milag12h` -> Base: `ath_dist_pct` (IC: 0.0129, P-Value: 0.3069) - Falso Positivo
- `btc_cycle_position_milag6h` -> Base: `btc_cycle_position` (IC: -0.0125, P-Value: 0.3168) - Falso Positivo
- `CreditSpread_HY_IG_z90d_milag12h` -> Base: `CreditSpread_HY_IG` (IC: -0.0058, P-Value: 0.6139) - Falso Positivo
- `FearGreed_milag500h` -> Base: `FearGreed` (IC: 0.0068, P-Value: 0.6436) - Falso Positivo
- `dv_dvol_pct_24h_milag12h` -> Base: `dv_dvol_pct_24h` (IC: -0.0081, P-Value: 0.6436) - Falso Positivo
- `LTH_Accum_Signal_milag500h` -> Base: `LTH_Accum_Signal` (IC: 0.0038, P-Value: 0.8812) - Falso Positivo

---
## Ventana W6
- **Features SFI (moteadas a base)**: 10
- **Sobrevivientes MCPT (Genuinas)**: 2 (20.0%)
- **Censuradas por MCPT (Ruido)**: 8 (80.0%)

### Genuinas
- `ETH_Return_7d_milag336h` -> Base: `ETH_Return_7d` (IC: 0.0309, P-Value: 0.0396)
- `NASDAQ_Ret_milag24h` -> Base: `NASDAQ_Ret` (IC: 0.0478, P-Value: 0.0198)
### Ruido Seleccionado por SFI
- `NonEx_Supply_z90d_milag72h` -> Base: `NonEx_Supply` (IC: -0.0203, P-Value: 0.1386) - Falso Positivo
- `DVOL_kz_milag1h` -> Base: `DVOL` (IC: 0.0149, P-Value: 0.3069) - Falso Positivo
- `ae_feat_16_milag6h` -> Base: `ae_feat_16` (IC: 0.0134, P-Value: 0.3564) - Falso Positivo
- `DeFi_WBTC_TVL_z90d_milag72h` -> Base: `DeFi_WBTC_TVL` (IC: -0.0125, P-Value: 0.4158) - Falso Positivo
- `ae_feat_56_milag1h` -> Base: `ae_feat_56` (IC: 0.0082, P-Value: 0.5842) - Falso Positivo
- `Whale_Vol_30d_MA_milag12h` -> Base: `Whale_Vol_30d_MA` (IC: -0.0021, P-Value: 0.8614) - Falso Positivo
- `Tx_Fees_USD_milag48h` -> Base: `Tx_Fees_USD` (IC: 0.0026, P-Value: 0.9010) - Falso Positivo
- `FundingRate_ZScore_90d_milag336h` -> Base: `FundingRate` (IC: 0.0000, P-Value: 1.0000) - Falso Positivo

---
