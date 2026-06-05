import pandas as pd
import json
import os

base_path = r"g:\Mi unidad\ia\luna_v2"
w2_parquet = os.path.join(base_path, "data", "reports", "wfb", "oos_trades_W2_seed42.parquet")
g2_json = os.path.join(base_path, "data", "reports", "wfb", "gate_G2_W2_seed42.json")

print("==== RESULTADOS W2 (SEED 42) ====")

if os.path.exists(g2_json):
    with open(g2_json, "r") as f:
        g2_data = json.load(f)
        print("GATE G2 (XGBoost):")
        print(f"  Passed: {g2_data.get('passed')}")
        print(f"  Hard Stop: {g2_data.get('is_hard_stop')}")
        print(f"  Disabled Agents: {g2_data.get('metrics', {}).get('disabled_agents', [])}")
        print(f"  Brier Scores: {g2_data.get('metrics', {}).get('brier_by_agent', {})}")

if os.path.exists(w2_parquet):
    df = pd.read_parquet(w2_parquet)
    print("\nOOS TRADES W2:")
    print(f"  Total Trades: {len(df)}")
    if len(df) > 0:
        print(f"  WinRate W2: {df['is_win'].mean():.2%}")
        print(f"  Retorno Total: {df['return_pct'].sum():.4%}")
        print("\n  Regímenes operativos:")
        print(df["hmm_regime"].value_counts())
        
        print("\n  Direcciones operativas:")
        print(df["direction"].value_counts())
else:
    print("\nOOS TRADES W2: Archivo no encontrado. Posiblemente abortado por Hard Stop.")
