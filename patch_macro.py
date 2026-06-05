import re
with open("luna/data/fetch_macro.py", "r", encoding="utf-8") as f:
    code = f.read()

recalc_block = """
    # [FIX-ROLLING-01] Recalcular TODAS las derivadas y rolling features sobre el histórico COMPLETO post-merge.
    # Bug: en modo incremental, los features rolling (MA200, ZScore90d, Pct6m) se calculaban
    # sobre el delta de 5 días, devolviendo NaN que luego se congelaba con ffill.
    if "SP500" in df.columns:
        _ma200 = df["SP500"].rolling(200).mean()
        df["SP500_AboveMA200"]     = (df["SP500"] > _ma200).astype(int)
        df["SP500_vs_MA200_ratio"] = (df["SP500"] / _ma200.replace(0, float("nan"))).round(4)
        df["SP500_Ret1m"]          = df["SP500"].pct_change(21)
    
    if "DXY" in df.columns:
        import numpy as np
        df["DXY_Slope30d"] = df["DXY"].rolling(30).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 30 else np.nan, raw=True
        )
        df["DXY_Pct6m"] = df["DXY"].pct_change(180)
        _dxy_ma  = df["DXY"].rolling(90).mean()
        _dxy_std = df["DXY"].rolling(90).std()
        df["DXY_Zscore"] = (df["DXY"] - _dxy_ma) / (_dxy_std + 1e-8)

    if "VIX" in df.columns:
        _vix_ma   = df["VIX"].rolling(252).mean()
        _vix_std  = df["VIX"].rolling(252).std()
        df["VIX_Zscore"] = (df["VIX"] - _vix_ma) / (_vix_std + 1e-8)

    if "RUSSELL2000" in df.columns:
        df["RUSSELL2000_Ret1m"] = df["RUSSELL2000"].pct_change(21)

    if "CreditSpread_HY_IG" in df.columns:
        _hy_ig = df["CreditSpread_HY_IG"]
        _cs_ma  = _hy_ig.rolling(90).mean()
        _cs_std = _hy_ig.rolling(90).std().clip(lower=1e-6)
        df["CreditSpread_HY_IG_z90d"] = (_hy_ig - _cs_ma) / _cs_std
        df["CreditStress_Flag"] = (_hy_ig.rolling(252).rank(pct=True) > 0.70).astype(int)
    
    if "CreditSpread_HY_pct" in df.columns:
        _hy = df["CreditSpread_HY_pct"]
        _hy_ma  = _hy.rolling(90).mean()
        _hy_std = _hy.rolling(90).std().clip(lower=1e-6)
        df["CreditSpread_HY_z90d"] = (_hy - _hy_ma) / _hy_std

    if "Fed_Net_Liquidity" in df.columns:
        _fnl = df["Fed_Net_Liquidity"]
        _fnl_ma = _fnl.rolling(90).mean()
        _fnl_std = _fnl.rolling(90).std()
        df["Fed_Net_Liq_90d_MA"]  = _fnl_ma
        df["Fed_Net_Liq_90d_Std"] = _fnl_std
        df["Fed_Liq_ZScore"]      = (_fnl - _fnl_ma) / (_fnl_std + 1e-8)
        df["Fed_Liq_Pct30d"]      = _fnl.pct_change(30)

    if "G3_Net_Liquidity_USD" in df.columns:
        _g3 = df["G3_Net_Liquidity_USD"]
        _g3_ma = _g3.rolling(90).mean()
        _g3_std = _g3.rolling(90).std()
        df["G3_Net_Liquidity_ZScore_90d"] = (_g3 - _g3_ma) / (_g3_std + 1e-8)
        
    if "G4_Net_Liquidity_USD" in df.columns:
        _g4 = df["G4_Net_Liquidity_USD"]
        _g4_ma = _g4.rolling(90).mean()
        _g4_std = _g4.rolling(90).std()
        df["G4_Net_Liquidity_ZScore_90d"] = (_g4 - _g4_ma) / (_g4_std + 1e-8)
"""

# Replace the existing VIX and DXY block with the comprehensive one
code = re.sub(r'# \[FIX-VIX-ZSCORE-01\].*?notna\(\)\.sum\(\)\)', recalc_block, code, flags=re.DOTALL)

with open("luna/data/fetch_macro.py", "w", encoding="utf-8") as f:
    f.write(code)
