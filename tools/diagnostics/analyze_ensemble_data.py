import json
import glob
from pathlib import Path
import statistics
from collections import defaultdict

def analyze_ensemble_data():
    base_dir = Path("C:/Users/Usuario/Desktop/ia/luna_v2")
    reports_dir = base_dir / "data" / "reports"
    wfb_reports_dir = reports_dir / "wfb"

    json_files = list(reports_dir.glob("*_FINAL_statistical_verdict.json"))
    gate_files = list(wfb_reports_dir.glob("gate_*.json"))

    print(f"Encontrados {len(json_files)} veredictos estadísticos y {len(gate_files)} reportes de gates.")

    # 1. Analizar veredictos
    verdicts = []
    failed_reasons = defaultdict(int)
    total_trades = []
    win_rates = []
    sharpes = []

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
                verdicts.append(data)
                summ = data.get("summary", {})
                total_trades.append(summ.get("total_trades", 0))
                win_rates.append(summ.get("win_rate_pct", 0))
                sharpes.append(summ.get("sharpe_crudo", 0))

                flags = data.get("flags", {})
                if not flags.get("pass_trades"): failed_reasons["Trades Insuficientes"] += 1
                if not flags.get("pass_dsr"): failed_reasons["DSR Fallido"] += 1
                if not flags.get("pass_pbo"): failed_reasons["PBO Fallido"] += 1
                if not flags.get("pass_binomial"): failed_reasons["Binomial Fallido"] += 1
        except Exception as e:
            pass

    print("\n=== RESUMEN DE VEREDICTOS INDIVIDUALES ===")
    print(f"Total semillas evaluadas: {len(verdicts)}")
    print(f"Trades medios por semilla: {statistics.mean(total_trades):.2f}")
    print(f"Win Rate medio: {statistics.mean(win_rates):.2f}%")
    print(f"Sharpe medio crudo: {statistics.mean(sharpes):.4f}")
    print("Razones de fallo en semillas individuales:")
    for k, v in failed_reasons.items():
        print(f"  - {k}: {v} semillas")

    # 2. Analizar embudo (Gates)
    gates_survival = defaultdict(list)
    initial_signals = 0
    final_signals = 0

    for gf in gate_files:
        try:
            with open(gf, "r") as f:
                data = json.load(f)
                gate_id = data.get("gate_id")
                metrics = data.get("metrics", {})
                
                if gate_id == "G5":
                    xgb = metrics.get("n_after_xgb", 0)
                    final = metrics.get("n_final", 0)
                    if xgb > 0:
                        survival = final / xgb
                        gates_survival["G5 (Final vs XGB)"].append(survival)
        except:
            pass

    print("\n=== SUPERVIVENCIA EN EL EMBUDO DE SEÑALES (GATES) ===")
    for k, v in gates_survival.items():
        if v:
            print(f"{k}: Promedio {statistics.mean(v)*100:.2f}% de señales sobreviven")

if __name__ == "__main__":
    analyze_ensemble_data()
