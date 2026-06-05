import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# FIX-BUGS-PRINTS: siempre imprimir para trazar el comportamiento en los logs
print("[DIAGNOSTIC] Iniciando analisis profundo y robusto de los parquets de la ultima run...")

_ROOT = Path("g:/Mi unidad/ia/luna_v2")
_WFB_DIR = _ROOT / "data" / "reports" / "wfb"

seeds = [42, 100, 777, 1337, 2025]
windows = ["W1", "W2", "W3", "W4", "W5"]

def calculate_stats(returns):
    if len(returns) == 0:
        return {}
    
    # 1. Ganancias/perdidas normales (acumulado simple)
    normal_return = float(np.sum(returns))
    
    # 2. Ganancias/perdidas compuestas
    comp_series = (1.0 + returns).cumprod()
    comp_return = float(comp_series[-1] - 1.0)
    
    # 3. Max Drawdown (compuesto)
    peaks = np.maximum.accumulate(comp_series)
    drawdowns = (comp_series - peaks) / peaks
    max_dd = float(abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0
    
    # 4. Win Rate
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    n_wins = len(wins)
    n_losses = len(losses)
    n_trades = len(returns)
    win_rate = float(n_wins / n_trades) if n_trades > 0 else 0.0
    
    # 5. Win/Loss Ratio
    avg_win = float(np.mean(wins)) if n_wins > 0 else 0.0
    avg_loss = float(abs(np.mean(losses))) if n_losses > 0 else 0.0
    wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    
    # 6. Optimal Kelly Fraction
    if wl_ratio > 0:
        kelly_fraction = win_rate - (1.0 - win_rate) / wl_ratio
    else:
        kelly_fraction = 0.0
    kelly_fraction = max(0.0, kelly_fraction)
    
    # 7. Leverage Analysis (5x and 10x)
    # 5x Leverage
    returns_5x = returns * 5.0
    returns_5x = np.clip(returns_5x, -0.999, None)
    comp_series_5x = (1.0 + returns_5x).cumprod()
    comp_return_5x = float(comp_series_5x[-1] - 1.0)
    peaks_5x = np.maximum.accumulate(comp_series_5x)
    drawdowns_5x = (comp_series_5x - peaks_5x) / peaks_5x
    max_dd_5x = float(abs(np.min(drawdowns_5x))) if len(drawdowns_5x) > 0 else 0.0
    
    # 10x Leverage
    returns_10x = returns * 10.0
    returns_10x = np.clip(returns_10x, -0.999, None)
    comp_series_10x = (1.0 + returns_10x).cumprod()
    comp_return_10x = float(comp_series_10x[-1] - 1.0)
    peaks_10x = np.maximum.accumulate(comp_series_10x)
    drawdowns_10x = (comp_series_10x - peaks_10x) / peaks_10x
    max_dd_10x = float(abs(np.min(drawdowns_10x))) if len(drawdowns_10x) > 0 else 0.0
    
    # 8. Optimal Leverage Sweep
    best_lev = 1.0
    best_ret = comp_return
    best_dd = max_dd
    
    best_lev_unconstrained = 1.0
    best_ret_unconstrained = comp_return
    
    for lev in np.arange(1.0, 30.0, 0.2):
        ret_lev = returns * lev
        ret_lev = np.clip(ret_lev, -0.999, None)
        series_lev = (1.0 + ret_lev).cumprod()
        final_ret = series_lev[-1] - 1.0
        
        p_lev = np.maximum.accumulate(series_lev)
        dd_lev = (series_lev - p_lev) / p_lev
        max_dd_lev = float(abs(np.min(dd_lev))) if len(dd_lev) > 0 else 0.0
        
        if final_ret > best_ret_unconstrained:
            best_ret_unconstrained = float(final_ret)
            best_lev_unconstrained = float(lev)
            
        if max_dd_lev < 0.50:
            if final_ret > best_ret:
                best_ret = float(final_ret)
                best_lev = float(lev)
                best_dd = max_dd_lev
                
    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "wl_ratio": wl_ratio,
        "normal_return": normal_return,
        "comp_return": comp_return,
        "max_dd": max_dd,
        "kelly_fraction": kelly_fraction,
        "comp_return_5x": comp_return_5x,
        "max_dd_5x": max_dd_5x,
        "comp_return_10x": comp_return_10x,
        "max_dd_10x": max_dd_10x,
        "optimal_leverage_constrained": best_lev,
        "optimal_return_constrained": best_ret,
        "optimal_dd_constrained": best_dd,
        "optimal_leverage_unconstrained": best_lev_unconstrained,
        "optimal_return_unconstrained": best_ret_unconstrained
    }

results = {}

for seed in seeds:
    results[seed] = {}
    print(f"\n[SEED {seed}] Analizando ventanas...")
    for w in windows:
        parquet_path = _WFB_DIR / f"oos_trades_{w}_seed{seed}.parquet"
        
        # Ignoramos el .flag y simplemente miramos si el parquet existe y tiene filas
        if not parquet_path.exists():
            print(f"  - Ventana {w}: 0 trades (no existe parquet)")
            results[seed][w] = {
                "n_trades": 0,
                "win_rate": 0.0,
                "wl_ratio": 0.0,
                "normal_return": 0.0,
                "comp_return": 0.0,
                "max_dd": 0.0,
                "kelly_fraction": 0.0,
                "comp_return_5x": 0.0,
                "max_dd_5x": 0.0,
                "comp_return_10x": 0.0,
                "max_dd_10x": 0.0,
                "optimal_leverage_constrained": 1.0,
                "optimal_return_constrained": 0.0,
                "optimal_dd_constrained": 0.0,
                "optimal_leverage_unconstrained": 1.0,
                "optimal_return_unconstrained": 0.0
            }
        else:
            try:
                df = pd.read_parquet(parquet_path)
                if len(df) == 0:
                    print(f"  - Ventana {w}: 0 trades (parquet vacio)")
                    stats = {
                        "n_trades": 0,
                        "win_rate": 0.0,
                        "wl_ratio": 0.0,
                        "normal_return": 0.0,
                        "comp_return": 0.0,
                        "max_dd": 0.0,
                        "kelly_fraction": 0.0,
                        "comp_return_5x": 0.0,
                        "max_dd_5x": 0.0,
                        "comp_return_10x": 0.0,
                        "max_dd_10x": 0.0,
                        "optimal_leverage_constrained": 1.0,
                        "optimal_return_constrained": 0.0,
                        "optimal_dd_constrained": 0.0,
                        "optimal_leverage_unconstrained": 1.0,
                        "optimal_return_unconstrained": 0.0
                    }
                else:
                    returns = df["return_pct"].values.astype(float)
                    stats = calculate_stats(returns)
                
                results[seed][w] = stats
                if stats["n_trades"] > 0:
                    print(f"  - Ventana {w}: {stats['n_trades']} trades | Ret_simple={stats['normal_return']*100:.4f}% | Ret_comp={stats['comp_return']*100:.4f}% | MaxDD={stats['max_dd']*100:.4f}% | Kelly={stats['kelly_fraction']*100:.2f}%")
                    print(f"    Leverage 5x: Ret={stats['comp_return_5x']*100:.4f}% | MaxDD={stats['max_dd_5x']*100:.4f}%")
                    print(f"    Leverage 10x: Ret={stats['comp_return_10x']*100:.4f}% | MaxDD={stats['max_dd_10x']*100:.4f}%")
                    print(f"    Leverage Optimo (MaxDD < 50%): Factor={stats['optimal_leverage_constrained']:.1f}x | Ret={stats['optimal_return_constrained']*100:.4f}% | MaxDD={stats['optimal_dd_constrained']*100:.4f}%")
            except Exception as e:
                print(f"  - Ventana {w}: ERROR al procesar parquet {parquet_path}: {e}")

# Guardar los resultados detallados en JSON para poder cargarlos
out_json_path = _ROOT / "tools" / "diagnostics" / "recent_runs_analysis_results.json"
with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
print(f"\n[DIAGNOSTIC] Analisis finalizado con éxito. Resultados persistidos en {out_json_path}")
