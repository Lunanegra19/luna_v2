"""Audit all WARN features in features_live.parquet"""
import pandas as pd

df = pd.read_parquet('/root/luna_v2/data/features/features_live.parquet')
cols = set(df.columns)

# Check all groups from dashboard
groups = {
    "OHLCV + Derivadas": ["close", "open", "high", "low", "volume",
                           "returns_1h", "returns_24h", "atr_14h", "volatility_24h"],
    "Funding Rate": ["FundingRate", "FundingRate_EMA3", "FundingRate_Pct90d",
                     "dv_funding_rate", "funding_extreme_pos", "funding_extreme_neg"],
    "Open Interest": ["OI_USD", "OI_BTC", "OI_Open_USD", "OI_High_USD", "OI_Low_USD",
                      "OI_USD_z90d", "dv_oi_acceleration_24h"],
    "Long/Short Ratio": ["LongShortRatio", "LongAccount", "ShortAccount",
                          "Coinglass_long_ratio", "Coinglass_short_ratio"],
    "ETF Flows": ["ETF_Flow_Proxy", "dv_etf_flow_proxy", "ETF_IBIT_Flow_Proxy",
                   "BITO_Close", "BITO_Volume", "ETF_Total_Volume", "ETF_Volume_Spike"],
    "On-Chain + Macro": ["DXY_z90d", "CPI_YoY_kz", "M2_China_YoY", "Stablecoins_Delta_30d",
                          "Whale_Proxy_Volume_USD", "FearGreed"],
    "HMM (OLD)": ["hmm_regime_label", "hmm_prob_bull", "hmm_prob_bear",
                   "hmm_prob_volatile", "hmm_state_duration_h"],
    "HMM (FIXED)": ["HMM_Regime", "hmm_velocity_bull", "hmm_acceleration_bull"],
}

for group, features in groups.items():
    present = [f for f in features if f in cols]
    missing = [f for f in features if f not in cols]
    status = "OK" if not missing else "WARN" if present else "ERROR"
    print(f"\n{status} {group}: {len(present)}/{len(features)}")
    if missing:
        print(f"  MISSING: {missing}")
    if present:
        # Check last value for present cols
        last = df[present].iloc[-1]
        for col in present[:3]:  # show first 3
            print(f"  {col}: {last[col]:.4f}" if isinstance(last[col], float) else f"  {col}: {last[col]}")
