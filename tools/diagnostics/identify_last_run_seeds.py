"""
tools/diagnostics/identify_last_run_seeds.py
Identifica exactamente que seeds pertenecen a la ultima run
analizando los timestamps de los archivos en data/reports/wfb/
[IDENTIFY-LAST-RUN 2026-05-30]
"""
import re
import datetime
from pathlib import Path
from collections import defaultdict, Counter

wfb_dir = Path(r"g:\Mi unidad\ia\luna_v2\data\reports\wfb")

# Recopilar todos los archivos relevantes con su mtime
patterns = [
    "oos_trades_W*_seed*.parquet",
    "oos_trades_W*_seed*_EMPTY.flag",
    "gate_G*_W*_seed*.json",
    "oos_raw_probs_W*_seed*.parquet",
]
all_files = []
for p in patterns:
    all_files.extend(wfb_dir.glob(p))

mtimes = []
for f in all_files:
    dt = datetime.datetime.fromtimestamp(f.stat().st_mtime)
    mtimes.append((f, dt))

mtimes.sort(key=lambda x: x[1])

if not mtimes:
    print("[IDENTIFY-LAST-RUN] ERROR: No se encontraron archivos.")
    exit(1)

oldest = mtimes[0][1]
newest = mtimes[-1][1]
print(f"[IDENTIFY-LAST-RUN] Rango total: {oldest.strftime('%Y-%m-%d %H:%M')} -> {newest.strftime('%Y-%m-%d %H:%M')}")
print()

# Agrupar por hora para detectar clusters de actividad
hora_counts = Counter()
for f, dt in mtimes:
    hora_counts[dt.strftime("%Y-%m-%d %H:00")] += 1

print("[IDENTIFY-LAST-RUN] Actividad por hora (detecta runs distintas):")
for hora in sorted(hora_counts.keys())[-30:]:
    bar = "#" * min(hora_counts[hora], 50)
    print(f"  {hora}  {hora_counts[hora]:4d}  {bar}")

print()

# La ultima run = archivos escritos en la ultima hora de actividad densa
# Buscar el cluster mas reciente (gap > 2 horas = run distinta)
last_cluster_start = newest
for f, dt in reversed(mtimes):
    gap = (newest - dt).total_seconds() / 3600.0
    if gap > 6.0:  # mas de 6h de diferencia = run anterior
        break
    last_cluster_start = dt

print(f"[IDENTIFY-LAST-RUN] Cluster de la ultima run: {last_cluster_start.strftime('%Y-%m-%d %H:%M')} -> {newest.strftime('%Y-%m-%d %H:%M')}")
print()

# Extraer seeds de los archivos del ultimo cluster
last_run_seeds_trades = defaultdict(set)
last_run_seeds_empty  = defaultdict(set)

for f, dt in mtimes:
    if dt >= last_cluster_start - datetime.timedelta(minutes=5):
        m = re.search(r"(W\d)_seed(\d+)", f.name)
        if m:
            win, seed = m.group(1), int(m.group(2))
            if "_EMPTY.flag" in f.name:
                last_run_seeds_empty[seed].add(win)
            elif "oos_trades_" in f.name and f.suffix == ".parquet":
                last_run_seeds_trades[seed].add(win)

ALL_WINDOWS = {"W1", "W2", "W3", "W4", "W5"}
all_seeds = sorted(set(list(last_run_seeds_trades.keys()) + list(last_run_seeds_empty.keys())))

print(f"[IDENTIFY-LAST-RUN] Seeds detectadas en la ULTIMA RUN: {len(all_seeds)} seeds")
print()

completas_con_trades = []
print(f"{'Seed':7} | {'Ventanas trades':20} | {'Ventanas empty':20} | {'Total':5} | {'Status'}")
print("-" * 80)
for seed in all_seeds:
    t = last_run_seeds_trades[seed]
    e = last_run_seeds_empty[seed]
    cubiertas = t | e
    faltantes = ALL_WINDOWS - cubiertas
    completa = len(faltantes) == 0
    tiene_trades = len(t) >= 1
    total = len(cubiertas)

    if completa and tiene_trades:
        status = "OK completa+trades"
        completas_con_trades.append(seed)
    elif completa:
        status = "completa SOLO EMPTY (0 trades)"
    else:
        status = "INCOMPLETA falta " + str(sorted(faltantes))

    print(f"  {seed:5} | {str(sorted(t)):20} | {str(sorted(e)):20} | {total:5} | {status}")

print("-" * 80)
print()
print(f"[IDENTIFY-LAST-RUN] Seeds COMPLETAS CON TRADES de la ultima run: {completas_con_trades}")
print(f"[IDENTIFY-LAST-RUN] Total: {len(completas_con_trades)} seeds validas")
