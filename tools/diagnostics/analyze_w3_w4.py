import pandas as pd
import json
from pathlib import Path

out = Path('c:/Users/Usuario/Desktop/ia/luna_v2/data/reports/wfb_analysis_w3_w4.txt')
wfb_dir = Path('c:/Users/Usuario/Desktop/ia/luna_v2/data/reports/wfb')

seeds = ['1337', '2025', '73469']
lines = []

for s in seeds:
    f_w3 = wfb_dir / f'oos_trades_W3_seed{s}.parquet'
    f_w4 = wfb_dir / f'oos_trades_W4_seed{s}.parquet'
    if not f_w3.exists() or not f_w4.exists():
        continue
    
    df3 = pd.read_parquet(f_w3)
    df4 = pd.read_parquet(f_w4)
    
    lines.append(f"--- SEED {s} ---")
    lines.append(f"W3 Trades: {len(df3)}, WR: {df3['is_win'].mean()*100:.2f}%")
    lines.append(f"W4 Trades: {len(df4)}, WR: {df4['is_win'].mean()*100:.2f}%")
    
    if 'xgb_prob' in df3.columns and 'lgbm_prob' in df3.columns:
        lines.append(f"W3 Avg Probas: XGB={df3['xgb_prob'].mean():.4f}, LGBM={df3['lgbm_prob'].mean():.4f}")
        lines.append(f"W4 Avg Probas: XGB={df4['xgb_prob'].mean():.4f}, LGBM={df4['lgbm_prob'].mean():.4f}")
    
    if 'hmm_regime' in df3.columns:
        lines.append(f"W3 Regimes:\n{df3['hmm_regime'].value_counts(normalize=True).to_string()}")
        lines.append(f"W4 Regimes:\n{df4['hmm_regime'].value_counts(normalize=True).to_string()}")
        
    if 'pnl_pct' in df3.columns:
        lines.append(f"W3 PnL: mean={df3['pnl_pct'].mean():.4f}, min={df3['pnl_pct'].min():.4f}, max={df3['pnl_pct'].max():.4f}")
        lines.append(f"W4 PnL: mean={df4['pnl_pct'].mean():.4f}, min={df4['pnl_pct'].min():.4f}, max={df4['pnl_pct'].max():.4f}")

    lines.append("")

out.write_text("\n".join(lines), encoding='utf-8')
print("Analysis written to data/reports/wfb_analysis_w3_w4.txt")
