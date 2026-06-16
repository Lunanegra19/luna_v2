import pandas as pd
import json
import glob
import os
import numpy as np

def run_analysis():
    print("="*60)
    print("HIPOTESIS 1: Auditoría de SFI Features y Covariate Shift")
    print("="*60)
    
    # Check SFI components
    sfi_files = glob.glob("data/features/canonical/sfi_selected_features*.json")
    if sfi_files:
        latest_sfi = max(sfi_files, key=os.path.getmtime)
        with open(latest_sfi, 'r') as f:
            sfi_data = json.load(f)
        
        features = sfi_data.get('selected_features', [])
        print(f"Total SFI Features Selected: {len(features)}")
        tech_count = sum(1 for f in features if any(x in f.lower() for x in ['rsi', 'macd', 'ema', 'bb', 'atr', 'mom', 'roc', 'close', 'price', 'volatility', 'donchian', 'keltner', 'std']))
        macro_count = sum(1 for f in features if any(x in f.lower() for x in ['macro', 'cpi', 'dxy', 'fed', 'bond', 'rate']))
        onchain_count = sum(1 for f in features if any(x in f.lower() for x in ['onchain', 'nvt', 'mvrv', 'hash', 'sopr', 'funding']))
        print(f"- Técnicos / Precio puro: {tech_count}")
        print(f"- Macro / Económicos: {macro_count}")
        print(f"- On-Chain / Sentimiento: {onchain_count}")
        print(f"- Otros: {len(features) - tech_count - macro_count - onchain_count}")
    else:
        print("No SFI files found.")

    print("\n" + "="*60)
    print("HIPOTESIS 2: Análisis del OOD Guard y Feature Drift Crítico")
    print("="*60)
    # Read the signal funnel
    funnel_path = "data/reports/signal_funnel_WFB_seed2025_funnel.json"
    if os.path.exists(funnel_path):
        with open(funnel_path, 'r') as f:
            funnel = json.load(f)
        total = funnel.get('after_xgb', 1)
        ood_survivors = funnel.get('after_ood', 1)
        print(f"XGBoost Signals: {total}")
        print(f"OOD Survivors: {ood_survivors}")
        print(f"OOD Censorship Rate: {(1 - ood_survivors/total)*100:.1f}%")
        if (1 - ood_survivors/total) > 0.5:
            print("=> CONFIRMADO: El Covariate Shift es masivo. El Guardián OOD bloqueó más del 50% de la operativa.")
    else:
        print("Funnel not found.")

    print("\n" + "="*60)
    print("HIPOTESIS 3: Decaimiento Temporal OOS (¿Embargo muy largo?)")
    print("="*60)
    
    oos_file = "data/predictions/oos_trades_seed2025.parquet"
    if os.path.exists(oos_file):
        df = pd.read_parquet(oos_file)
        # Sort by index (timestamp)
        df = df.sort_index()
        active_trades = df[df.get('kelly_fraction_used', 1.0) > 0.0]
        
        print(f"Total Active Trades (Seed 2025): {len(active_trades)}")
        if len(active_trades) > 0:
            # Divide in two halves chronologically
            mid_idx = len(active_trades) // 2
            first_half = active_trades.iloc[:mid_idx]
            second_half = active_trades.iloc[mid_idx:]
            
            wr1 = first_half['is_win_kelly'].mean() if 'is_win_kelly' in first_half.columns else 0.0
            wr2 = second_half['is_win_kelly'].mean() if 'is_win_kelly' in second_half.columns else 0.0
            ret1 = first_half['return_pct'].sum()
            ret2 = second_half['return_pct'].sum()
            
            print(f"Primera Mitad de OOS ({len(first_half)} trades): WR = {wr1:.1%} | Retorno Neto = {ret1:.2%}")
            print(f"Segunda Mitad de OOS ({len(second_half)} trades): WR = {wr2:.1%} | Retorno Neto = {ret2:.2%}")
            
            if wr1 > wr2 and ret1 > ret2:
                print("=> CONFIRMADO: El Alpha decae fuertemente con el tiempo. El modelo se vuelve obsoleto rápidamente.")
            elif wr1 < wr2 and ret1 < ret2:
                print("=> RECHAZADO: La segunda mitad fue mejor. El decaimiento temporal inmediato no es la causa.")
            else:
                print("=> MIXTO: No hay una degradación temporal lineal clara.")
    else:
        print("OOS parquet not found.")

if __name__ == '__main__':
    run_analysis()
