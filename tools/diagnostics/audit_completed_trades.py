import os
import sys
import io
import pandas as pd
from pathlib import Path

# Configurar encoding UTF-8 para consola de Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent.parent

def audit():
    print("[AUDIT-TRADES] Iniciando auditoria de trades WFB...")
    wfb_dir = _ROOT / "data" / "reports" / "wfb"
    if not wfb_dir.exists():
        print(f"[AUDIT-TRADES] ERROR: Directorio {wfb_dir} no existe.")
        return

    # Buscar archivos de trades para la run activa
    trade_files = sorted(list(wfb_dir.glob("oos_trades_W*_seed*.parquet")))
    flag_files = sorted(list(wfb_dir.glob("oos_trades_W*_seed*_EMPTY.flag")))

    if not trade_files and not flag_files:
        print("[AUDIT-TRADES] No se encontraron archivos de trades ni flags.")
        return

    print(f"[AUDIT-TRADES] Encontrados {len(trade_files)} archivos de trades y {len(flag_files)} flags de vacio.")

    # Agrupar por semilla
    seed_data = {}
    for f in trade_files:
        stem = f.stem
        # oos_trades_W{window}_seed{seed}
        parts = stem.split("_")
        window = parts[2]
        seed = parts[3].replace("seed", "")
        if seed not in seed_data:
            seed_data[seed] = []
        seed_data[seed].append((window, f, False))

    for f in flag_files:
        stem = f.stem
        # oos_trades_W{window}_seed{seed}_EMPTY
        parts = stem.split("_")
        window = parts[2]
        seed = parts[3].replace("seed", "")
        if seed not in seed_data:
            seed_data[seed] = []
        seed_data[seed].append((window, f, True))

    for seed, windows in sorted(seed_data.items()):
        print(f"\n==================================================")
        print(f"RESULTADOS PARA SEMILLA (SEED): {seed}")
        print(f"==================================================")
        
        rows = []
        for window, path, is_empty in sorted(windows, key=lambda x: x[0]):
            if is_empty:
                rows.append({
                    "Ventana": window,
                    "Trades": 0,
                    "Win Rate (%)": "N/A",
                    "Sharpe": "N/A",
                    "Retorno Medio (%)": "N/A",
                    "Status": "CASH (0 trades)"
                })
            else:
                try:
                    df = pd.read_parquet(path)
                    n_trades = len(df)
                    if n_trades == 0:
                        rows.append({
                            "Ventana": window,
                            "Trades": 0,
                            "Win Rate (%)": "0.00%",
                            "Sharpe": "0.0000",
                            "Retorno Medio (%)": "0.0000%",
                            "Status": "CASH"
                        })
                        continue

                    wr = df['is_win'].mean() * 100 if 'is_win' in df.columns else 0.0
                    mean_ret = df['return_pct'].mean() * 100 if 'return_pct' in df.columns else 0.0
                    
                    # Sharpe
                    sharpe = 0.0
                    if n_trades > 1 and 'return_pct' in df.columns:
                        std_r = df['return_pct'].std()
                        if std_r > 1e-10:
                            if 'timestamp' in df.columns:
                                df = df.set_index('timestamp')
                            df.index = pd.to_datetime(df.index, utc=True)
                            days = (df.index.max() - df.index.min()).days
                            n_per_year = n_trades / (days / 365.25) if days > 0 else n_trades * 365.25
                            sharpe = (df['return_pct'].mean() / std_r) * (n_per_year ** 0.5)

                    rows.append({
                        "Ventana": window,
                        "Trades": n_trades,
                        "Win Rate (%)": f"{wr:.2f}%",
                        "Sharpe": f"{sharpe:.4f}",
                        "Retorno Medio (%)": f"{mean_ret:.4f}%",
                        "Status": "OPERATIVA"
                    })
                except Exception as e:
                    rows.append({
                        "Ventana": window,
                        "Trades": "Error",
                        "Win Rate (%)": "Error",
                        "Sharpe": "Error",
                        "Retorno Medio (%)": "Error",
                        "Status": f"Error: {e}"
                    })

        df_summary = pd.DataFrame(rows)
        print(df_summary.to_string(index=False))

if __name__ == "__main__":
    audit()
