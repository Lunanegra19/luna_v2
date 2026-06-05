import re
with open("luna/data/fetch_macro.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. fetch_credit_spreads
code = re.sub(r'# Z-Score rolling 90d.*?df_cs\["CreditStress_Flag"\].*?\.astype\(int\)', 
r'''            df_cs = pd.DataFrame({
                "CreditSpread_HY_pct":     hy_d,
                "CreditSpread_IG_pct":     ig_d,
                "CreditSpread_BBB_pct":    bbb_d,
                "CreditSpread_HY_IG":      hy_ig_spread,
            })''', code, flags=re.DOTALL)

# 2. fetch_fed_net_liquidity
code = re.sub(r'net_liq = fed_assets - tga - rrp.*?logger\.info\(f"Fed Net Liquidity: {len\(df\)} dias"\)',
r'''net_liq = fed_assets - tga - rrp
        df = pd.DataFrame({
            "WALCL":               fed_assets,
            "Fed_Net_Liquidity":   net_liq,
        })
        logger.info(f"Fed Net Liquidity: {len(df)} dias")''', code, flags=re.DOTALL)

# 3. fetch_market_indicators
code = re.sub(r'# ── DXY Derivadas ──.*?return df\.sort_index\(\)',
r'''return df.sort_index()''', code, flags=re.DOTALL)

# 4. build_macro_dataset (remove BTC ratios, Macro Regime Flag, G3/G4 liquidity)
code = re.sub(r'# S2.4: BTC/Gold ratio y BTC/SP500 ratio.*?logger\.success\(f"Macro dataset',
r'''logger.success(f"Macro dataset''', code, flags=re.DOTALL)

# 5. Add apply_derived_features method
replacement_class = '''
    def apply_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica todos los indicadores rolling/derivados sobre el dataset completo ya ffill-ed."""
        df = df.copy()

        def zscore_90d(s):
            ma  = s.rolling(90).mean()
            std = s.rolling(90).std().clip(lower=1e-6)
            return (s - ma) / std

        if "CreditSpread_HY_IG" in df.columns:
            df["CreditSpread_HY_IG_z90d"] = zscore_90d(df["CreditSpread_HY_IG"])
            df["CreditStress_Flag"] = (df["CreditSpread_HY_IG"].rolling(252).rank(pct=True) > 0.70).astype(int)
        if "CreditSpread_HY_pct" in df.columns:
            df["CreditSpread_HY_z90d"] = zscore_90d(df["CreditSpread_HY_pct"])

        if "Fed_Net_Liquidity" in df.columns:
            ma90    = df["Fed_Net_Liquidity"].rolling(90).mean()
            std90   = df["Fed_Net_Liquidity"].rolling(90).std()
            df["Fed_Net_Liq_90d_MA"]  = ma90
            df["Fed_Net_Liq_90d_Std"] = std90
            df["Fed_Liq_ZScore"]      = (df["Fed_Net_Liquidity"] - ma90) / (std90 + 1e-8)
            df["Fed_Liq_Pct30d"]      = df["Fed_Net_Liquidity"].pct_change(30)

        if "DXY" in df.columns:
            df["DXY_Slope30d"] = df["DXY"].rolling(30).apply(
                lambda x: __import__('numpy').polyfit(range(len(x)), x, 1)[0] if len(x) == 30 else __import__('numpy').nan,
                raw=True,
            )
            df["DXY_Pct6m"]  = df["DXY"].pct_change(180)
            df["DXY_Ret"]    = df["DXY"].pct_change(1)
            ma90 = df["DXY"].rolling(90).mean()
            std90 = df["DXY"].rolling(90).std()
            df["DXY_Zscore"] = (df["DXY"] - ma90) / (std90 + 1e-8)

        if "VIX" in df.columns:
            df["VIX_Regime"]   = (df["VIX"] > 20).astype(int)
            df["VIX_Slope7d"]  = df["VIX"].pct_change(7)
            df["VIX_Spike"]    = (df["VIX"].pct_change(1) > 0.20).astype(int)
            ma252  = df["VIX"].rolling(252).mean()
            std252 = df["VIX"].rolling(252).std()
            df["VIX_Zscore"]   = (df["VIX"] - ma252) / (std252 + 1e-8)
            if "MOVE" in df.columns:
                df["Spread_VIX_MOVE"] = df["VIX"] - df["MOVE"]

        if "SP500" in df.columns:
            ma200 = df["SP500"].rolling(200).mean()
            df["SP500_AboveMA200"]     = (df["SP500"] > ma200).astype(int)
            df["SP500_vs_MA200_ratio"] = (df["SP500"] / ma200.replace(0, __import__('numpy').nan)).round(4)
            df["SP500_Ret"]            = df["SP500"].pct_change(1)
            df["SP500_Ret1m"]          = df["SP500"].pct_change(21)

        if "NASDAQ" in df.columns:
            df["NASDAQ_Ret"] = df["NASDAQ"].pct_change(1)
        if "Gold" in df.columns:
            df["Gold_Ret"]  = df["Gold"].pct_change(1)
        if "Oil" in df.columns:
            df["Oil_Ret"]   = df["Oil"].pct_change(1)
        if "RUSSELL2000" in df.columns:
            df["RUSSELL2000_Ret1m"] = df["RUSSELL2000"].pct_change(21)
        if "NatGas" in df.columns:
            df["Gas_Ret"]    = df["NatGas"].pct_change(1)
        if "Copper" in df.columns:
            df["Copper_Ret"] = df["Copper"].pct_change(1)
        if "HYG" in df.columns and "LQD" in df.columns:
            df["HY_Spread"] = (df["HYG"] / df["LQD"].replace(0, __import__('numpy').nan)).round(5)
            df.drop(columns=[c for c in ["HYG", "LQD"] if c in df.columns], inplace=True)

        try:
            import yfinance as yf
            btc_daily = yf.download("BTC-USD", start="2018-01-01", auto_adjust=True, progress=False)
            if not btc_daily.empty:
                btc_close = btc_daily["Close"].iloc[:, 0] if isinstance(btc_daily.columns, pd.MultiIndex) else btc_daily["Close"]
                btc_close.index = pd.to_datetime(btc_close.index, utc=True)
                btc_reindexed = btc_close.reindex(df.index, method="ffill")
                if "Gold" in df.columns:
                    df["BTC_Gold_Ratio"] = btc_reindexed / (df["Gold"] + 1e-8)
                if "SP500" in df.columns:
                    df["BTC_SP500_Ratio"] = btc_reindexed / (df["SP500"] + 1e-8)
        except Exception as e:
            pass

        regime_conditions = []
        if "T10Y2Y" in df.columns: regime_conditions.append((df["T10Y2Y"] > 0).astype(int))
        if "VIX" in df.columns: regime_conditions.append((df["VIX"] < 20).astype(int))
        if "SP500_AboveMA200" in df.columns: regime_conditions.append(df["SP500_AboveMA200"].fillna(0).astype(int))
        if regime_conditions:
            df["Macro_Risk_Score"] = sum(regime_conditions)
            df["Macro_Risk_On"]    = (df["Macro_Risk_Score"] >= 2).astype(int)

        if all(c in df.columns for c in ["WALCL", "ECBASSETS", "JPNASSETS", "EURUSD", "USDJPY"]):
            ecb_usd = df["ECBASSETS"] * df["EURUSD"].ffill()
            boj_usd = (df["JPNASSETS"] * 100) / df["USDJPY"].ffill()
            g3_liq = df["WALCL"] + ecb_usd + boj_usd
            g4_liq = None
            if "CHNASSETS" in df.columns and "USDCNY" in df.columns:
                pboc_usd = (df["CHNASSETS"] * 1000) / df["USDCNY"].ffill()
                g4_liq = g3_liq + pboc_usd

            ma90_3 = g3_liq.rolling(90).mean()
            std90_3 = g3_liq.rolling(90).std()
            df["G3_Net_Liquidity_USD"] = g3_liq
            df["G3_Net_Liquidity_ZScore_90d"] = (g3_liq - ma90_3) / (std90_3 + 1e-8)

            if g4_liq is not None:
                ma90_4 = g4_liq.rolling(90).mean()
                std90_4 = g4_liq.rolling(90).std()
                df["G4_Net_Liquidity_USD"] = g4_liq
                df["G4_Net_Liquidity_ZScore_90d"] = (g4_liq - ma90_4) / (std90_4 + 1e-8)

        return df

    def save(self, df: pd.DataFrame) -> Path:
'''
code = code.replace('    def save(self, df: pd.DataFrame) -> Path:', replacement_class)

# 6. Apply in __main__
main_replace = '''
        # FIX-MACRO-FFILL-01: Al inyectar nuevas columnas diarias a un histórico horario existente, 
        # debemos propagar los valores hacia adelante para rellenar los huecos intradiarios.
        df = df.ffill()
        logger.info("[fetch_macro] Merge OK: {} (prev) + {} (delta) → {} filas totales",
                    len(_existing), _rows_delta, len(df))

    # [FIX-ROLLING-01] Calcular TODAS las derivadas y rolling features sobre el DF fusionado
    df = fetcher.apply_derived_features(df)
'''
# find the block in main
code = re.sub(r'# FIX-MACRO-FFILL-01:.*?logger\.info\("\[fetch_macro\].*?len\(df\)\)', main_replace, code, flags=re.DOTALL)
# remove the VIX_ZSCORE fix blocks
code = re.sub(r'# \[FIX-VIX-ZSCORE-01\].*?_n_recalc\)', '', code, flags=re.DOTALL)
code = re.sub(r'if "DXY" in df\.columns:.*?notna\(\)\.sum\(\)\)', '', code, flags=re.DOTALL)

with open("luna/data/fetch_macro.py", "w", encoding="utf-8") as f:
    f.write(code)
