import pandas as pd
from pathlib import Path

ROOT = Path("c:/Users/Usuario/Desktop/ia/luna_v2")
WFB_DIR = ROOT / "data/reports/wfb"

def test_hypotheses():
    files = list(WFB_DIR.glob("oos_trades_xgb_baseline_*.parquet"))
    if not files: return
    
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs).sort_index()
    print(f"Total señales XGBoost Base: {len(df)}")
    
    if 'meta_v2_prob' in df.columns:
        for t in [0.45, 0.40, 0.38, 0.35, 0.33, 0.30]:
            print(f"Meta Prob >= {t}: {len(df[df['meta_v2_prob'] >= t])}")
            
    print("\nVamos a cargar las features para ver el momentum...")
    features_files = list((ROOT / "data/archive/W5_seed777/features_prev").glob("*.parquet"))
    if not features_files:
        print("No se encontro archivo de features para cruzar.")
    else:
        df_feat = pd.read_parquet(features_files[0])
        # Join
        df = df.join(df_feat[['dv_momentum_speed']], how='left')
        if 'dv_momentum_speed' in df.columns:
            for t in [-5.0, -10.0, -15.0, -20.0, -30.0, -100.0]:
                print(f"Momentum >= {t}: {len(df[df['dv_momentum_speed'] >= t])}")
                
        # Combinado:
        print("\nPrueba Combinada:")
        for m in [0.38, 0.35, 0.33]:
            for mo in [-15.0, -20.0, -100.0]:
                n = len(df[(df['meta_v2_prob'] >= m) & (df['dv_momentum_speed'] >= mo)])
                print(f"Meta >= {m} & Mom >= {mo}: {n} trades")

if __name__ == "__main__":
    test_hypotheses()
