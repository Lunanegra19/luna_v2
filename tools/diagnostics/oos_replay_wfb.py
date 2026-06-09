import sys, os
from pathlib import Path
import pandas as pd
import numpy as np

# Fix stdout encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure luna is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import cfg
from luna.models.signal_filter import SignalFilter

def run_replay(percentile: float, embargo: float):
    print(f"\n=======================================================")
    print(f"[REPLAY] Test: Percentile={percentile} | Embargo={embargo}H")
    print(f"=======================================================")
    
    # 1. Parcheamos el cfg en memoria
    cfg.xgboost.meta_v2_rolling_percentile = percentile
    cfg.sop.embargo_hours = embargo
    if embargo == 0.0:
        cfg.sop.soft_embargo_enabled = False
    else:
        cfg.sop.soft_embargo_enabled = True

    # Carpetas de los modelos WFB (W1 a W5)
    base_dir = Path("g:/Mi unidad/ia/luna_v2/data/runs/WFB_20260609_093207_seed42/seed42")
    
    # Las ventanas OOS reales (aproximadas de los logs)
    windows = {
        "W2": ("2025-06-06", "2025-06-21"),
        "W3": ("2025-07-02", "2025-09-21"),
        "W4": ("2025-10-01", "2025-10-09"),
        "W5": ("2026-03-10", "2026-03-27"),
    }
    
    total_trades = 0
    all_trade_indices = []
    
    for w_name, (start_date, end_date) in windows.items():
        w_dir = base_dir / w_name
        if not w_dir.exists():
            continue
            
        print(f"\n--- Procesando {w_name} ({start_date} a {end_date}) ---")
        
        try:
            holdout_file = f"g:/Mi unidad/ia/luna_v2/data/features/features_holdout_{w_name}.parquet"
            if not Path(holdout_file).exists():
                print(f"  No se encontró {holdout_file}")
                continue
            df_oos = pd.read_parquet(holdout_file)
            if df_oos.index.tz is None:
                df_oos.index = df_oos.index.tz_localize("UTC")
            # Slice by the exact start_date and end_date
            df_oos = df_oos.loc[start_date:end_date].copy()
            
            # Cargar las probabilidades del modelo (esencial para SignalFilter)
            probs_file = w_dir / "oos_raw_probs.parquet"
            if probs_file.exists():
                df_probs = pd.read_parquet(probs_file)
                df_oos = df_oos.join(df_probs, how='inner')
            else:
                print(f"  No se encontró oos_raw_probs en {w_name}")
                continue
                
        except Exception as e:
            print(f"  Slicing error: {e}")
            continue
            
        print(f"  Barras OOS filtradas: {len(df_oos)}")
        if len(df_oos) == 0:
            print(f"  Index min/max: {df_oos.index.min()} / {df_oos.index.max()}")
            continue
            
        # Instanciar el SignalFilter oficial con los modelos entrenados
        try:
            sf = SignalFilter(models_dir=w_dir)
            # Ejecutar el Pipeline oficial 100% real
            trades_idx = sf.filter_signals(df_oos, available_feats=list(df_oos.columns), direction="long")
            
            n_trades = len(trades_idx)
            total_trades += n_trades
            all_trade_indices.extend(trades_idx)
            print(f"  Trades en {w_name}: {n_trades}")
            print(f"  Embudo oficial: {sf.funnel_stats}")
            
        except Exception as e:
            print(f"  Error al procesar {w_name}: {e}")
            import traceback
            traceback.print_exc()

    # Evaluar la rentabilidad proxy con la base de datos de los trades generados
    if len(all_trade_indices) > 0:
        print(f"\n[RESULTADO TOTAL] Percentil: {percentile} | Embargo: {embargo}H")
        print(f"Total Trades OOS: {total_trades}")
        print("NOTA: El EV requeriria reconstruir el DataFrame consolidado.")
    else:
        print(f"\n[RESULTADO TOTAL] Percentil: {percentile} | Embargo: {embargo}H")
        print(f"Total Trades OOS: 0")

if __name__ == "__main__":
    print("Iniciando Replay Institucional Luna V2...")
    # 1. Base SOP
    run_replay(percentile=0.85, embargo=24.0)
    # 2. Bajar threshold
    run_replay(percentile=0.80, embargo=24.0)
    # 3. Quitar embargo (Piramidar)
    run_replay(percentile=0.85, embargo=0.0)
    # 4. Quitar embargo y bajar threshold
    run_replay(percentile=0.80, embargo=0.0)
