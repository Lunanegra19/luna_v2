import pandas as pd
import numpy as np
import glob
import os

def run_simulation():
    print("Cargando predicciones puras de todas las ventanas WFB (seed42)...")
    path_pattern = "g:/Mi unidad/ia/luna_v2/data/runs/WFB_20260609_093207_seed42/seed42/W*/oos_raw_probs.parquet"
    files = glob.glob(path_pattern)
    
    if not files:
        print("No se encontraron archivos de probabilidad cruda.")
        return
        
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df_probs = pd.concat(dfs).sort_index()
    print(f"Probabilidades crudas cargadas: {len(df_probs)} horas OOS.")
    
    print("Cargando features_train.parquet para obtener precios Close...")
    df_features = pd.read_parquet("g:/Mi unidad/ia/luna_v2/data/features/features_train.parquet", columns=["Target_TBM_Bin"])
    
    # Unir datos
    df = df_probs.join(df_features, how='inner')
    
    # Calcular retorno futuro a 48H como proxy simplificado del TBM para evaluar EV
    df['Target'] = df['Target_TBM_Bin']
    df = df.dropna(subset=['Target'])
    
    scenarios = [
        {"name": "SOP Actual (0.85 + Embargo 24H)", "percentile": 0.85, "embargo_h": 24},
        {"name": "Test A (Bajar a 0.80 + Embargo 24H)", "percentile": 0.80, "embargo_h": 24},
        {"name": "Test B (0.85 + SIN Embargo 0H)", "percentile": 0.85, "embargo_h": 0},
        {"name": "Test C (0.80 + SIN Embargo 0H)", "percentile": 0.80, "embargo_h": 0},
        {"name": "Test D (0.75 + Embargo 24H)", "percentile": 0.75, "embargo_h": 24},
    ]
    
    results = []
    
    for s in scenarios:
        perc = s["percentile"]
        emb_h = s["embargo_h"]
        
        # 1. Calcular umbral dinamico (rolling 100)
        roll_thresh = df['prob_bull'].rolling(window=100, min_periods=10).quantile(perc)
        
        # 2. Generar Senales
        # Ademas de superar el percentil, debe superar la base minima (0.38)
        raw_signals = (df['prob_bull'] > roll_thresh) & (df['prob_bull'] > 0.38)
        
        # 3. Aplicar Embargo (simulacion secuencial)
        trades = []
        last_trade_idx = -9999
        
        # Obtener indices integer de las senales
        signal_indices = np.where(raw_signals)[0]
        
        for idx in signal_indices:
            if idx >= last_trade_idx + emb_h:
                trades.append(idx)
                last_trade_idx = idx
                
        # 4. Calcular metricas
        if len(trades) > 0:
            df_trades = df.iloc[trades]
            wins = df_trades['Target'].sum()
            wr = wins / len(trades) * 100
            
            # Asumimos riesgo asimétrico (2% profit, 1.5% stop loss) menos fees para simular EV OOS
            ev = (wr / 100.0) * 0.02 - ((100 - wr) / 100.0) * 0.015
            net_ret = (ev - 0.0025) * len(trades) * 100
            ev = ev * 100
        else:
            wr = 0
            net_ret = 0
            ev = 0
            
        results.append({
            "Escenario": s["name"],
            "Percentil": perc,
            "Embargo": f"{emb_h}H",
            "Trades": len(trades),
            "WinRate": f"{wr:.1f}%",
            "EV (48h)": f"{ev:.2f}%",
            "Ret Neto": f"{net_ret:.2f}%"
        })
        
    df_res = pd.DataFrame(results)
    print("\n--- RESULTADOS DEL TEST DE HIPOTESIS EN LA CORRIDA ACTUAL ---")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_simulation()
