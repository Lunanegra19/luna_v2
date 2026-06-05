import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = ROOT / "data" / "predictions"

print("================================================================================")
print("   ANALISIS DEL PORTAFOLIO ENSEMBLE CONJUNTO ")
print("================================================================================")

files = ["ensemble_portfolio_trades.parquet", "unified_ensemble_trades_raw.parquet"]

for fn in files:
    fpath = PREDICTIONS_DIR / fn
    if fpath.exists():
        df = pd.read_parquet(fpath)
        print(f"\nArchivo: {fn}")
        print(f"Número de trades: {len(df)}")
        if "is_win" in df.columns:
            print(f"Win Rate: {df['is_win'].mean():.2%}")
        
        # Calcular retornos
        if "return_pct" in df.columns:
            ret_simple = df["return_pct"].sum()
            equity = (1 + df["return_pct"]).cumprod()
            ret_comp = equity.iloc[-1] - 1.0 if not equity.empty else 0.0
            peaks = equity.cummax()
            dd = (equity - peaks) / peaks
            max_dd = dd.min() if not dd.empty else 0.0
            
            print(f"Retorno Simple: {ret_simple:.2%}")
            print(f"Retorno Compuesto: {ret_comp:.2%}")
            print(f"Max Drawdown: {max_dd:.2%}")
            
            # Sharpe anualizado aprox si tenemos timestamps
            # Asumamos 1 año
            if len(df) > 5:
                # Estimar Sharpe
                std = df["return_pct"].std()
                mean = df["return_pct"].mean()
                if std > 0:
                    sharpe_diario = mean / std
                    sharpe_anual = sharpe_diario * np.sqrt(252) # aprox diaria
                    print(f"Sharpe Anualizado Estimado: {sharpe_anual:.4f}")
                    print(f"Calmar Ratio Estimado: {abs(ret_comp / max_dd) if max_dd != 0 else 0.0:.2f}")
        else:
            print("No se encontró columna return_pct")
    else:
        print(f"No existe: {fn}")
