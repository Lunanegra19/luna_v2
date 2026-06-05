"""
tools/diagnostics/check_seed_completion.py
Verifica que ventanas (W1-W5) tiene completadas cada seed en la ultima run.
[CHECK-SEED-COMPLETION 2026-05-30]
"""
import re
from pathlib import Path
from collections import defaultdict

wfb_dir = Path(r"g:\Mi unidad\ia\luna_v2\data\reports\wfb")

parquets = list(wfb_dir.glob("oos_trades_W*_seed*.parquet"))
empties  = list(wfb_dir.glob("oos_trades_W*_seed*_EMPTY.flag"))
gates    = list(wfb_dir.glob("gate_G*_W*_seed*.json"))

seed_windows_trades  = defaultdict(set)
seed_windows_empty   = defaultdict(set)
seed_windows_gate    = defaultdict(set)

for f in parquets:
    m = re.search(r"(W\d)_seed(\d+)", f.stem)
    if m:
        seed_windows_trades[int(m.group(2))].add(m.group(1))

for f in empties:
    m = re.search(r"(W\d)_seed(\d+)", f.stem)
    if m:
        seed_windows_empty[int(m.group(2))].add(m.group(1))

for f in gates:
    m = re.search(r"(W\d)_seed(\d+)", f.name)
    if m:
        seed_windows_gate[int(m.group(2))].add(m.group(1))

ALL_WINDOWS = {"W1", "W2", "W3", "W4", "W5"}
all_seeds = sorted(set(list(seed_windows_trades.keys()) + list(seed_windows_empty.keys())))

print("[CHECK-SEED-COMPLETION] Estado de ventanas por seed:")
print("-" * 75)
print(f"{'Seed':6} | {'Trades':20} | {'Empty':12} | {'Completa?':12} | Faltantes")
print("-" * 75)

completas = []
incompletas = []

for seed in all_seeds:
    trades_w  = seed_windows_trades[seed]
    empty_w   = seed_windows_empty[seed]
    cubiertas = trades_w | empty_w
    faltantes = ALL_WINDOWS - cubiertas
    completa  = len(faltantes) == 0
    if completa:
        completas.append(seed)
    else:
        incompletas.append(seed)
    estado = "SI (5/5)" if completa else "NO (" + str(len(cubiertas)) + "/5)"
    falt_str = "-" if completa else str(sorted(faltantes))
    print(f"{seed:6} | {str(sorted(trades_w)):20} | {str(sorted(empty_w)):12} | {estado:12} | {falt_str}")

print("-" * 75)
print()
print("[CHECK-SEED-COMPLETION] Seeds COMPLETAS (5/5 ventanas):", completas)
print("[CHECK-SEED-COMPLETION] Seeds INCOMPLETAS:", incompletas)
print()

# Cuantos trades tiene cada seed completa
if completas:
    print("[CHECK-SEED-COMPLETION] Trades por seed completa:")
    import pandas as pd
    for seed in completas:
        seed_files = list(wfb_dir.glob(f"oos_trades_W*_seed{seed}.parquet"))
        total = 0
        for f in seed_files:
            try:
                df = pd.read_parquet(f)
                total += len(df)
            except Exception:
                pass
        print(f"  Seed {seed}: {total} trades en {len(seed_files)} ventanas con datos")
