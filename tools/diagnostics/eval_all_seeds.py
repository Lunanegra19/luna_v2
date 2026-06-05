"""
tools/diagnostics/eval_all_seeds.py
====================================
Evalua el ensemble WFB sobre las seeds COMPLETAS (5/5 ventanas cubiertas)
con trades reales de la ultima run. No toca settings.yaml.

Deteccion automatica:
  - Completa = W1..W5 cubiertos por parquet de trades O flag EMPTY
  - Con trades = al menos 1 parquet de trades (no solo empties)

[EVAL-ALL-SEEDS-DIAG 2026-05-30]
"""
import sys
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

wfb_dir = _ROOT / "data" / "reports" / "wfb"
ALL_WINDOWS = {"W1", "W2", "W3", "W4", "W5"}

# ── 1. Clasificar ventanas por seed ──────────────────────────────────────────
seed_windows_trades = defaultdict(set)
seed_windows_empty  = defaultdict(set)

for f in wfb_dir.glob("oos_trades_W*_seed*.parquet"):
    m = re.search(r"(W\d)_seed(\d+)", f.stem)
    if m:
        seed_windows_trades[int(m.group(2))].add(m.group(1))

for f in wfb_dir.glob("oos_trades_W*_seed*_EMPTY.flag"):
    m = re.search(r"(W\d)_seed(\d+)", f.stem)
    if m:
        seed_windows_empty[int(m.group(2))].add(m.group(1))

# ── 2. Filtrar: completas (5/5) Y con al menos 1 trade real ─────────────────
seeds_validas = []
all_seeds_seen = sorted(set(list(seed_windows_trades.keys()) + list(seed_windows_empty.keys())))

print("[EVAL-ALL-SEEDS-DIAG] Analizando seeds de la ultima run...")
print(f"{'Seed':7} | {'Completa':8} | {'Trades':6} | {'Estado'}")
print("-" * 50)

for seed in all_seeds_seen:
    trades_w  = seed_windows_trades[seed]
    empty_w   = seed_windows_empty[seed]
    cubiertas = trades_w | empty_w
    completa  = ALL_WINDOWS.issubset(cubiertas)
    n_trades  = len(trades_w)  # ventanas con datos reales

    if completa and n_trades >= 1:
        estado = "OK - incluida"
        seeds_validas.append(seed)
    elif completa and n_trades == 0:
        estado = "SKIP - completa pero 0 trades (todo EMPTY)"
    else:
        faltantes = sorted(ALL_WINDOWS - cubiertas)
        estado = f"SKIP - incompleta, faltan {faltantes}"

    print(f"  {seed:5} | {'SI':8} | {n_trades:6} | {estado}" if completa
          else f"  {seed:5} | {'NO':8} | {n_trades:6} | {estado}")

print("-" * 50)
print(f"[EVAL-ALL-SEEDS-DIAG] Seeds validas para evaluacion: {seeds_validas}")
print(f"[EVAL-ALL-SEEDS-DIAG] Total: {len(seeds_validas)} seeds completas con trades")

if not seeds_validas:
    print("[EVAL-ALL-SEEDS-DIAG] ERROR: No hay seeds completas con trades. Abortando.")
    sys.exit(1)

# ── 3. Contar trades reales por seed valida ───────────────────────────────────
print()
print("[EVAL-ALL-SEEDS-DIAG] Trades por seed valida:")
for seed in seeds_validas:
    total = 0
    for f in wfb_dir.glob(f"oos_trades_W*_seed{seed}.parquet"):
        try:
            total += len(pd.read_parquet(f))
        except Exception:
            pass
    print(f"  Seed {seed}: {total} trades")

# ── 4. Parchear cfg EN MEMORIA (no toca settings.yaml) ───────────────────────
from config.settings import cfg as _cfg

original_seeds     = list(_cfg.wfb.active_seeds)
original_CUTOFF = int(_cfg.wfb.ensemble_consensus_threshold)

_cfg.wfb.active_seeds = seeds_validas

# Threshold adaptativo segun numero de seeds completas
n = len(seeds_validas)
NEW_CUTOFF = 4 if n >= 8 else 3 if n >= 5 else 2
_cfg.wfb.ensemble_consensus_CUTOFF = NEW_THRESHOLD

print()
print(f"[EVAL-ALL-SEEDS-DIAG] active_seeds  : {original_seeds} -> {seeds_validas}")
print(f"[EVAL-ALL-SEEDS-DIAG] consensus_thr : {original_threshold} -> {NEW_THRESHOLD} (adaptativo para {n} seeds)")
print(f"[EVAL-ALL-SEEDS-DIAG] stat.min_trades (sin cambio): {_cfg.stat.min_trades}")

# ── 5. Ejecutar el evaluador institucional ────────────────────────────────────
from scripts.evaluate_ensemble_wfb import main

print()
print("=" * 80)
print(f"[EVAL-ALL-SEEDS-DIAG] EVALUANDO {n} SEEDS COMPLETAS CON TRADES")
print("=" * 80)

sys.exit(main())
