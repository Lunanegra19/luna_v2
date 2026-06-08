import os
import json
import glob
import pandas as pd
from pathlib import Path

def main():
    reports_dir = Path("data/reports/wfb")
    
    seeds_data = {}
    
    # 1. Gather early stop info
    early_stops = glob.glob(str(reports_dir / "early_stop_*.json"))
    for f in early_stops:
        with open(f, 'r') as fp:
            data = json.load(fp)
            seed = data.get("seed")
            if seed not in seeds_data: seeds_data[seed] = {}
            seeds_data[seed]["early_stop"] = data.get("reason", "Unknown")
            
    # 2. Gather trades info
    trade_files = glob.glob(str(reports_dir / "oos_trades_W*_seed*.parquet"))
    for f in trade_files:
        p = Path(f)
        parts = p.stem.split("_seed")
        if len(parts) == 2:
            seed = int(parts[1])
            window = p.stem.split("_")[2]
            
            if seed not in seeds_data: seeds_data[seed] = {}
            if "windows" not in seeds_data[seed]: seeds_data[seed]["windows"] = {}
            if window not in seeds_data[seed]["windows"]: seeds_data[seed]["windows"][window] = {}
            
            try:
                df = pd.read_parquet(f)
                seeds_data[seed]["windows"][window]["trades"] = len(df)
                if len(df) > 0 and 'return_pct' in df.columns:
                    seeds_data[seed]["windows"][window]["return_mean"] = df['return_pct'].mean() * 100
                    seeds_data[seed]["windows"][window]["win_rate"] = (df['return_pct'] > 0).mean() * 100
                else:
                    seeds_data[seed]["windows"][window]["return_mean"] = 0
                    seeds_data[seed]["windows"][window]["win_rate"] = 0
            except Exception as e:
                pass
                
    # 3. Gather final verdicts
    verdicts = glob.glob(str(reports_dir / "*_FINAL_statistical_verdict.json"))
    for f in verdicts:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                seed = data.get("seed")
                if seed is None:
                    # try to parse from filename
                    parts = Path(f).stem.split("_seed")
                    if len(parts) == 2:
                        seed = int(parts[1].split("_")[0])
                
                if seed is not None:
                    if seed not in seeds_data: seeds_data[seed] = {}
                    seeds_data[seed]["final_verdict"] = data
        except Exception:
            pass

    # 4. Generate Markdown
    md = []
    md.append("# Reporte de Diagnóstico WFB (Todas las semillas)")
    md.append("Este informe consolida todos los datos recopilados de las ventanas calculadas hasta el momento.\n")
    
    # Tabla resumen de semillas
    md.append("## Resumen por Semilla")
    md.append("| Semilla | Estado | Trades (Total) | Win Rate (promedio) |")
    md.append("|---|---|---|---|")
    
    for seed, data in sorted(seeds_data.items()):
        status = "🟢 Finalizado" if "final_verdict" in data else ("🛑 Early Stop" if "early_stop" in data else "⏳ En Proceso")
        total_trades = sum([w.get("trades", 0) for w in data.get("windows", {}).values()])
        wr_list = [w.get("win_rate", 0) for w in data.get("windows", {}).values() if w.get("trades", 0) > 0]
        avg_wr = sum(wr_list)/len(wr_list) if wr_list else 0.0
        
        md.append(f"| {seed} | {status} | {total_trades} | {avg_wr:.2f}% |")
        
    md.append("\n## Detalle de Early Stops")
    for seed, data in sorted(seeds_data.items()):
        if "early_stop" in data:
            md.append(f"- **Seed {seed}**: {data['early_stop']}")
            
    md.append("\n## Rendimiento por Ventana")
    for seed, data in sorted(seeds_data.items()):
        if "windows" in data and len(data["windows"]) > 0:
            md.append(f"### Semilla {seed}")
            md.append("| Ventana | Trades | Win Rate | Retorno Medio |")
            md.append("|---|---|---|---|")
            for w, wdata in sorted(data["windows"].items()):
                md.append(f"| {w} | {wdata.get('trades', 0)} | {wdata.get('win_rate', 0):.2f}% | {wdata.get('return_mean', 0):.4f}% |")
            md.append("")
            
    md.append("\n## Veredictos Finales")
    for seed, data in sorted(seeds_data.items()):
        if "final_verdict" in data:
            v = data["final_verdict"]
            md.append(f"### Semilla {seed}")
            metrics = v.get("metrics", {})
            md.append(f"- **DSR**: {metrics.get('sharpe_crudo', 'N/A')}")
            md.append(f"- **Aprobado**: {'Sí' if v.get('deploy_approved') else 'No'}")
            if "rejection_reason" in v:
                md.append(f"- **Razón Rechazo**: {v['rejection_reason']}")
            md.append("")

    with open("data/reports/wfb_full_diagnostics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print("Report written to data/reports/wfb_full_diagnostics.md")

if __name__ == "__main__":
    main()
