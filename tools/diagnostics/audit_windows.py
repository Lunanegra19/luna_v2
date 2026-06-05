import glob
import json
import os

print("==== AUDITORIA DE VENTANAS W1-W5 (SEED 42) ====")
g2_files = sorted(glob.glob("g:/Mi unidad/ia/luna_v2/data/reports/wfb/gate_G2_W*_seed42.json"))

for f in g2_files:
    w = os.path.basename(f).split('_')[2]
    with open(f, 'r') as file:
        data = json.load(file)
        passed = data.get("passed")
        hard_stop = data.get("is_hard_stop")
        disabled = data.get("metrics", {}).get("disabled_agents", [])
        brier = data.get("metrics", {}).get("brier_by_agent", {})
        print(f"[{w}] Passed: {passed} | Hard Stop: {hard_stop} | Disabled: {disabled}")
        print(f"     Brier: {brier}")

print("\n==== VEREDICTO FINAL ====")
verdict_file = "g:/Mi unidad/ia/luna_v2/data/reports/statistical_verdict.json"
if os.path.exists(verdict_file):
    with open(verdict_file, 'r') as file:
        v = json.load(file)
        print(f"Gauntlet Passed: {v.get('passed_gauntlet')}")
        print(f"Total Trades: {v.get('metrics', {}).get('total_trades')}")
        print(f"Win Rate: {v.get('metrics', {}).get('win_rate')}")
