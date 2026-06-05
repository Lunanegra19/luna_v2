import json
from pathlib import Path
from collections import defaultdict

wfb_dir = Path("data/reports/wfb")

# 1) Seeds podadas
pruned = []
for f in sorted(wfb_dir.glob("early_stop_seed*.json")):
    seed = f.stem.replace("early_stop_seed", "")
    data = json.loads(f.read_text())
    pruned.append({
        "seed": seed,
        "windows": data.get("windows_evaluated", []),
        "reason": data.get("reason", "")[:70]
    })

print(f"SEEDS PODADAS: {len(pruned)}")
for p in sorted(pruned, key=lambda x: x["seed"]):
    print(f"  seed={p['seed']:>6} | W{p['windows']} | {p['reason']}")

# 2) Seeds con trades
trades_per_seed = defaultdict(list)
for f in wfb_dir.glob("oos_trades_W*_seed*.parquet"):
    seed = f.stem.split("seed")[1]
    w = f.stem.split("_")[2]
    trades_per_seed[seed].append(w)

print()
print(f"SEEDS CON TRADES (al menos 1 ventana): {len(trades_per_seed)}")
for s, ws in sorted(trades_per_seed.items()):
    ws_sorted = sorted(ws)
    complete = len(ws_sorted) == 5
    flag = "<<< COMPLETA W1-W5" if complete else f"parcial {ws_sorted}"
    print(f"  seed={s:>6} | {flag}")

# 3) Totales
pruned_seeds = set(p["seed"] for p in pruned)
seeds_with_trades = set(trades_per_seed.keys())
all_explored = pruned_seeds | seeds_with_trades

print()
print("=" * 55)
print(f"TOTAL SEEDS UNICAS EXPLORADAS en esta run: {len(all_explored)}")
print(f"  - Podadas (early stop):          {len(pruned)}")
print(f"  - Con al menos 1 ventana trades: {len(seeds_with_trades)}")
print(f"  - Completas W1-W5:               {sum(1 for ws in trades_per_seed.values() if len(ws)==5)}")
print(f"  - Actualmente en proceso:        {len(seeds_with_trades) - sum(1 for ws in trades_per_seed.values() if len(ws)==5)}")

# 4) Benchmark
bm = wfb_dir / "dynamic_benchmark.json"
if bm.exists():
    d = json.loads(bm.read_text())
    print()
    print(f"BENCHMARK: seed={d['champion_seed']} | score={d['champion_score']}/100 | DSR={d['champion_dsr']}")
