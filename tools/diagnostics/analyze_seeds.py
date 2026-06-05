import pandas as pd
import numpy as np
from pathlib import Path
import json

# Definir la ruta base del proyecto
ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = ROOT / "data" / "predictions"
REPORTS_DIR = ROOT / "data" / "reports"

# Lista de semillas objetivo del 22 de mayo de 2026
TARGET_SEEDS = [42, 100, 777, 1337, 2025, 10, 99, 888]

print("================================================================================")
print("   ESTUDIO DE RESULTADOS DE LAS SEMILLAS COMPLETAS (WFB Y AUDITORIA ESTADISTICA) ")
print("================================================================================")

results = []

for seed in TARGET_SEEDS:
    trade_file = PREDICTIONS_DIR / f"oos_trades_seed{seed}.parquet"
    verdict_files = list(REPORTS_DIR.glob(f"*_seed{seed}_FINAL_statistical_verdict.json"))
    
    if not verdict_files:
        verdict_files = list(REPORTS_DIR.glob(f"*seed{seed}_FINAL_statistical_verdict.json"))
        
    if not trade_file.exists():
        print(f"ERROR: No se encontro el archivo de trades para la semilla {seed}")
        continue
        
    # Cargar trades
    df = pd.read_parquet(trade_file)
    
    # Cargar veredicto json si existe
    verdict_data = {}
    if verdict_files:
        verdict_files.sort(key=lambda x: x.stat().st_mtime)
        v_path = verdict_files[-1]
        try:
            with open(v_path, encoding="utf-8") as f:
                verdict_data = json.load(f)
        except Exception as e:
            print(f"Error al leer veredicto para semilla {seed}: {e}")
            
    # Extraer métricas básicas
    n_trades = len(df)
    win_rate = df["is_win"].mean() if "is_win" in df.columns else (verdict_data.get("metrics", {}).get("win_rate", 0))
    
    # Calcular retornos normales (aritméticos) con el sizing de Kelly real usado
    return_normal_sum = df["return_pct"].sum()
    
    # Calcular retorno compuesto real con el sizing de Kelly real usado
    equity_curve_real = (1 + df["return_pct"]).cumprod()
    return_compuesto_real = equity_curve_real.iloc[-1] - 1.0 if not equity_curve_real.empty else 0.0
    
    # Drawdown máximo real
    peaks = equity_curve_real.cummax()
    drawdowns_real = (equity_curve_real - peaks) / peaks
    max_dd_real = drawdowns_real.min() if not drawdowns_real.empty else 0.0
    
    # Kelly puro y Quarter-Kelly teórico basado en el Win Rate real
    p = win_rate
    f_star = 1.5 * p - 0.5
    f_quarter = f_star * 0.25
    f_quarter_capped = np.clip(f_quarter, 0.01, 0.15) if f_star > 0 else 0.0
    
    # Simulación de Apalancamiento x5 y x10
    # A x5
    equity_x5 = (1 + 5 * df["return_pct"]).cumprod()
    ret_comp_x5 = equity_x5.iloc[-1] - 1.0 if not equity_x5.empty else 0.0
    peaks_x5 = equity_x5.cummax()
    dd_x5 = (equity_x5 - peaks_x5) / peaks_x5
    max_dd_x5 = dd_x5.min() if not dd_x5.empty else 0.0
    
    # A x10
    equity_x10 = (1 + 10 * df["return_pct"]).cumprod()
    ret_comp_x10 = equity_x10.iloc[-1] - 1.0 if not equity_x10.empty else 0.0
    peaks_x10 = equity_x10.cummax()
    dd_x10 = (equity_x10 - peaks_x10) / peaks_x10
    max_dd_x10 = dd_x10.min() if not dd_x10.empty else 0.0
    
    # Sharpe, Calmar, DSR, PBO del veredicto
    metrics = verdict_data.get("metrics", {})
    audit = verdict_data.get("statistical_audit", {})
    
    sharpe = metrics.get("sharpe_crudo", 0.0)
    calmar = metrics.get("calmar_ratio", 0.0)
    dsr = audit.get("dsr", 0.0)
    pbo = audit.get("estimated_pbo", 0.0)
    binomial_p = audit.get("binomial_p_value", 1.0)
    approved = verdict_data.get("deploy_approved", False)
    
    results.append({
        "seed": seed,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "ret_normal": return_normal_sum,
        "ret_compound": return_compuesto_real,
        "max_dd": max_dd_real,
        "f_star": f_star,
        "f_quarter": f_quarter_capped,
        "ret_x5": ret_comp_x5,
        "dd_x5": max_dd_x5,
        "ret_x10": ret_comp_x10,
        "dd_x10": max_dd_x10,
        "sharpe": sharpe,
        "calmar": calmar,
        "dsr": dsr,
        "pbo": pbo,
        "binomial_p": binomial_p,
        "approved": approved
    })
    
    status = "APROBADA" if approved else "RECHAZADA"
    print(f"Semilla {seed:4d} | Trades: {n_trades:3d} | WR: {win_rate:.2%} | Ret. Compuesto: {return_compuesto_real:.2%} | Max DD: {max_dd_real:.2%} | Status: {status}")

# Guardar los resultados en formato JSON estructurado
Path(ROOT / "tools" / "dumps").mkdir(parents=True, exist_ok=True)
with open(ROOT / "tools" / "dumps" / "analyzed_seeds_report.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
    
# Generar tabla en formato Markdown sin emojis problemáticos para consolas
print("\nTABLA COMPARATIVA EN FORMATO MARKDOWN PARA INFORME DE RENDIMIENTO:\n")
print("| Semilla | Trades | Win Rate | Ret. Simple (Arit.) | Ret. Compuesto (Real) | Max DD Real | Sharpe | Calmar | DSR | PBO | Binomial p | Veredicto |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in results:
    v_icon = "APROBADO" if r["approved"] else "RECHAZADO"
    print(f"| **{r['seed']}** | {r['n_trades']} | {r['win_rate']:.2%} | {r['ret_normal']:.2%} | {r['ret_compound']:.2%} | {r['max_dd']:.2%} | {r['sharpe']:.4f} | {r['calmar']:.2f} | {r['dsr']:.4f} | {r['pbo']:.2%} | {r['binomial_p']:.4f} | {v_icon} |")

print("\nTABLA DE POLITICA DE KELLY Y APALANCAMIENTO SIMULADO (x5 a x10):\n")
print("| Semilla | Win Rate | Kelly Puro (f*) | Quarter-Kelly (f_applied) | Ret. x5 (Compuesto) | Max DD x5 | Ret. x10 (Compuesto) | Max DD x10 | Exposicion Max. Permitida |")
print("|---|---|---|---|---|---|---|---|---|")
for r in results:
    f_star_str = f"{r['f_star']:.2%}" if r['f_star'] > 0 else "0.00% (Negativo)"
    f_q_str = f"{r['f_quarter']:.2%}" if r['f_quarter'] > 0 else "0.00% (No operar)"
    print(f"| **{r['seed']}** | {r['win_rate']:.2%} | {f_star_str} | {f_q_str} | {r['ret_x5']:.2%} | {r['dd_x5']:.2%} | {r['ret_x10']:.2%} | {r['dd_x10']:.2%} | [1.0%, 15.0%] (SOP Cap) |")

print("================================================================================")
print("   DIAGNOSTICO FINALIZADO CORRECTAMENTE. DATOS EXPORTADOS A dumps/analyzed_seeds_report.json")
print("================================================================================")
