import pandas as pd
import glob

seeds = [42, 100, 1337, 31473]
dfs = []
for s in seeds:
    files = glob.glob(f"data/reports/wfb/oos_trades_W*_seed{s}.parquet")
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)

if dfs:
    all_trades = pd.concat(dfs, ignore_index=True)
    n_trades = len(all_trades)
    wr = all_trades["is_win"].mean() * 100
    
    # buscar columna de retorno
    ret_col = [c for c in all_trades.columns if "ret" in c.lower() or "pnl" in c.lower() or "target" in c.lower()][0]
    ret_mean = all_trades[ret_col].mean() * 100
    
    print(f"Total trades brutos sumados: {n_trades}")
    print(f"Win Rate promedio bruto: {wr:.2f}%")
    print(f"Retorno medio por trade bruto ({ret_col}): {ret_mean:.4f}%")
else:
    print("No files found.")
