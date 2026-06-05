import pandas as pd
from pathlib import Path

ROOT = Path("g:/Mi unidad/ia/luna_v2")
trades_w3_path = ROOT / "data" / "reports" / "wfb" / "oos_trades_W3_seed42.parquet"
trades_w4_path = ROOT / "data" / "reports" / "wfb" / "oos_trades_W4_seed42.parquet"

print("=== INSPECTING TRADES PARQUET ===")
for path, label in [(trades_w3_path, "W3"), (trades_w4_path, "W4")]:
    if path.exists():
        df = pd.read_parquet(path)
        print(f"\n{label} Trades:")
        print(f"  Total Trades: {len(df)}")
        if len(df) > 0:
            print("  Columns:", list(df.columns))
            # Mostrar los primeros trades y algunas métricas clave
            cols_to_show = [c for c in ["xgb_prob", "xgb_prob_cal", "meta_v2_prob", "hmm_regime", "hmm_semantic", "ret_bruto", "ret"] if c in df.columns]
            print(df[cols_to_show].head(10))
    else:
        print(f"\n{label} Trades file NOT found at: {path}")
