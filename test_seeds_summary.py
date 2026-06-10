import json
import glob
from pathlib import Path

files = sorted(glob.glob("data/reports/*_statistical_verdict.json"))
if not files:
    files = sorted(glob.glob("data/reports/wfb/*_statistical_verdict.json"))

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            seed = data.get("seed", "Unknown")
            appr = "? APROBADA" if data.get("deploy_approved") else "? RECHAZADA"
            metrics = data.get("metrics", {})
            audit = data.get("statistical_audit", {})
            
            wr = metrics.get("win_rate_pct", 0)
            trades = metrics.get("n_trades", 0)
            calmar = metrics.get("calmar_ratio", 0)
            dsr = audit.get("dsr", 0)
            pbo = audit.get("pbo", 0)
            
            print(f"Seed {seed}: {appr}")
            print(f"  Trades: {trades} | WR: {wr:.1f}% | Calmar: {calmar:.2f}")
            print(f"  DSR: {dsr:.4f} | PBO: {pbo:.1f}%")
            print("-" * 40)
    except Exception as e:
        print(f"Error reading {f}: {e}")
