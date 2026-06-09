import sys
from pathlib import Path
import pandas as pd
import numpy as np

def run_pure_hypotheses():
    base_dir = Path("g:/Mi unidad/ia/luna_v2/data/runs/WFB_20260609_093207_seed42/seed42")
    windows = {
        "W2": ("2025-06-06", "2025-06-21"),
        "W3": ("2025-07-02", "2025-09-21"),
        "W4": ("2025-10-01", "2025-10-09"),
        "W5": ("2026-03-10", "2026-03-27"),
    }
    
    dfs = []
    
    for w_name, (start_date, end_date) in windows.items():
        w_dir = base_dir / w_name
        if not w_dir.exists(): continue
            
        holdout_file = f"g:/Mi unidad/ia/luna_v2/data/features/features_holdout_{w_name}.parquet"
        probs_file = w_dir / "oos_raw_probs.parquet"
        
        if not Path(holdout_file).exists() or not probs_file.exists():
            continue
            
        df_oos = pd.read_parquet(holdout_file)
        if df_oos.index.tz is None:
            df_oos.index = df_oos.index.tz_localize("UTC")
        df_oos = df_oos.loc[start_date:end_date].copy()
        
        df_probs = pd.read_parquet(probs_file)
        
        # Merge exacto
        df_merged = df_oos.join(df_probs, how='inner')
        dfs.append(df_merged)
        
    if not dfs:
        print("No se pudieron cargar datos.")
        return
        
    df = pd.concat(dfs).sort_index()
    print(f"\n[DATOS] Cargadas {len(df)} horas OOS reales de W2 a W5")
    
    scenarios = [
        {"name": "SOP Actual (0.85 + Embargo 24H)", "percentile": 0.85, "embargo_h": 24},
        {"name": "Test A (Bajar a 0.80 + Embargo 24H)", "percentile": 0.80, "embargo_h": 24},
        {"name": "Test B (0.85 + SIN Embargo 0H)", "percentile": 0.85, "embargo_h": 0},
        {"name": "Test C (0.80 + SIN Embargo 0H)", "percentile": 0.80, "embargo_h": 0},
    ]
    
    results = []
    
    for s in scenarios:
        perc = s["percentile"]
        emb_h = s["embargo_h"]
        
        # 1. Umbral dinamico (rolling 100)
        roll_thresh = df['prob_bull'].rolling(window=100, min_periods=10).quantile(perc)
        raw_signals = (df['prob_bull'] > roll_thresh) & (df['prob_bull'] > 0.38)
        signal_indices = np.where(raw_signals)[0]
        
        # Simular Momentum (Sabemos que rechaza ~5% de las señales empíricamente, restamos 5% al final o al azar)
        # Para precision exacta, aplicaremos el embargo estrictamente:
        trades = []
        last_trade_idx = -9999
        
        for idx in signal_indices:
            if idx >= last_trade_idx + emb_h:
                trades.append(idx)
                last_trade_idx = idx
                
        # 2. Descontar 5% por Momentum (aprox basado en log oficial)
        n_trades_raw = len(trades)
        n_trades = int(n_trades_raw * 0.95) 
        
        if len(trades) > 0:
            df_trades = df.iloc[trades]
            wins = df_trades['Target_TBM_Bin'].sum()
            wr = wins / len(trades) * 100
            
            # EV = (WinRate * 2.0%) - (LossRate * 1.5%) - comisiones
            ev = (wr / 100.0) * 0.02 - ((100 - wr) / 100.0) * 0.015
            net_ret = (ev - 0.0025) * n_trades * 100
        else:
            wr, ev, net_ret = 0, 0, 0
            
        results.append({
            "Escenario": s["name"],
            "Trades (Aprox con Momentum)": n_trades,
            "WinRate Real (%)": f"{wr:.1f}%",
            "EV por Trade": f"{ev*100:.2f}%",
            "Retorno Neto Total": f"{net_ret:.2f}%"
        })
        
    df_res = pd.DataFrame(results)
    print("\n================ RESULTADOS FINALES EXACTOS SEED 42 ================")
    print(df_res.to_string(index=False))
    print("====================================================================")

if __name__ == "__main__":
    run_pure_hypotheses()
