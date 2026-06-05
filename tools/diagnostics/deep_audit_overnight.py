"""
deep_audit_overnight.py
Auditoria profunda de la run nocturna 2026-06-01/02
Analiza: completitud de ventanas, errores, OOS trades, degradados, warnings
"""
import os, glob, re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path(r"g:\Mi unidad\ia\luna_v2")
RUNS_DIR = BASE / "data" / "runs"
LOGS_DIR = BASE / "logs"
REPORTS_DIR = BASE / "data" / "reports"
PREDICTIONS_DIR = BASE / "data" / "predictions"

print("=" * 80)
print("  AUDITORIA PROFUNDA — RUN NOCTURNA 2026-06-01/02")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. INVENTARIO DE RUNS Y VENTANAS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] INVENTARIO DE RUNS Y VENTANAS")
print("-" * 80)

runs = sorted([d for d in RUNS_DIR.iterdir()
               if d.is_dir() and (d.name.startswith("WFB_20260602") or d.name.startswith("WFB_20260601_2"))])

run_summary = []
for run in runs:
    seed = run.name.split("seed")[-1] if "seed" in run.name else "UNK"
    seed_dir = run / seed
    if not seed_dir.exists():
        # try to find any subdir
        subdirs = [d for d in run.iterdir() if d.is_dir() and d.name != "FINAL"]
        seed_dir = subdirs[0] if subdirs else None

    windows = {}
    for w in ["W1","W2","W3","W4","W5"]:
        if seed_dir is None:
            windows[w] = "NO_DIR"
            continue
        w_dir = seed_dir / w
        if not w_dir.exists():
            windows[w] = "MISSING"
            continue
        has_oos = (w_dir / "oos_trades.parquet").exists()
        has_gate_disabled = (w_dir / "gate_g2_disabled_agents.json").exists()
        has_xgb_sig = any((w_dir / f"xgboost_meta_{a}_long_signature.json").exists()
                         for a in ["bull","bear","range","calm_bear"])
        n_trades = 0
        oos_regimes = []
        if has_oos:
            try:
                df = pd.read_parquet(w_dir / "oos_trades.parquet")
                n_trades = len(df)
                if "hmm_regime" in df.columns:
                    oos_regimes = df["hmm_regime"].value_counts().to_dict()
            except Exception as e:
                n_trades = -1

        if has_oos:
            status = f"OOS({n_trades}t)"
        elif has_xgb_sig:
            status = "TRAIN_OK/NO_OOS"
        elif (w_dir / "hmm_regime_labels.parquet").exists():
            status = "HMM_ONLY"
        else:
            status = "INCOMPLETE"
        if has_gate_disabled:
            status += "+DEGRADED"
        windows[w] = status

    final_dir = run / seed / "FINAL" if seed_dir else None
    has_final = final_dir.exists() if final_dir else False

    run_summary.append({
        "run": run.name,
        "seed": seed,
        "W1": windows.get("W1","?"),
        "W2": windows.get("W2","?"),
        "W3": windows.get("W3","?"),
        "W4": windows.get("W4","?"),
        "W5": windows.get("W5","?"),
        "FINAL": "YES" if has_final else "NO",
    })
    print(f"  {run.name:<45} | W1:{windows.get('W1','?'):<20} W2:{windows.get('W2','?'):<20} W3:{windows.get('W3','?'):<20} W4:{windows.get('W4','?'):<20} | FINAL:{('YES' if has_final else 'NO')}")

print(f"\n  Total runs encontradas: {len(runs)}")
complete = sum(1 for r in run_summary if "OOS" in r["W3"] or "OOS" in r["W4"])
print(f"  Runs con al menos W3 OOS: {complete}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SCAN DE ERRORES EN WORKER LOGS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] SCAN DE ERRORES EN WORKER LOGS")
print("-" * 80)

worker_logs = sorted(LOGS_DIR.glob("wfb_worker_*.log"),
                     key=lambda x: x.stat().st_mtime)
worker_logs = [l for l in worker_logs if l.stat().st_mtime > datetime(2026,6,1,22,0).timestamp()]

error_patterns = ["ERROR", "CRITICAL", "Traceback", "Exception", "FAILED",
                  "aborted", "crash", "KeyError", "RuntimeError", "MemoryError",
                  "GAUNTLET RECHAZADO", "0 se", "DEGRADED"]
warning_patterns = ["WARNING", "WARN"]

total_errors = 0
total_warnings = 0
error_types = {}

for log in worker_logs:
    log_errors = []
    log_warnings = []
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                is_err = any(p in line for p in error_patterns[:8])  # real errors
                is_warn = "WARNING" in line or "WARN" in line
                is_gauntlet = "GAUNTLET RECHAZADO" in line
                is_degraded = "DEGRADED" in line
                if is_err or is_gauntlet:
                    log_errors.append((lineno, line.strip()[:120]))
                    # Categorize
                    if "GAUNTLET" in line:
                        error_types["GAUNTLET_REJECTED"] = error_types.get("GAUNTLET_REJECTED",0)+1
                    elif "DEGRADED" in line:
                        error_types["GATE_G2_DEGRADED"] = error_types.get("GATE_G2_DEGRADED",0)+1
                    elif "0 se" in line or "0 trade" in line:
                        error_types["ZERO_SIGNALS"] = error_types.get("ZERO_SIGNALS",0)+1
                    elif "Traceback" in line or "Exception" in line:
                        error_types["EXCEPTION"] = error_types.get("EXCEPTION",0)+1
                    else:
                        error_types["OTHER_ERROR"] = error_types.get("OTHER_ERROR",0)+1
                if is_warn:
                    log_warnings.append(lineno)
    except Exception as e:
        print(f"  ERROR leyendo {log.name}: {e}")
        continue

    total_errors += len(log_errors)
    total_warnings += len(log_warnings)

    if log_errors:
        print(f"\n  [{log.name[:40]}]  {len(log_errors)} errores | {len(log_warnings)} warnings")
        for lineno, err in log_errors[:5]:  # show first 5
            print(f"    L{lineno}: {err}")
        if len(log_errors) > 5:
            print(f"    ... (+{len(log_errors)-5} more)")

print(f"\n  RESUMEN ERRORES:")
for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
    print(f"    {etype:<30}: {count}")
print(f"  Total warnings en todos los logs: {total_warnings}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ANÁLISIS DETALLADO POR SEMILLA — RESULTADOS OOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] RESULTADOS OOS POR SEMILLA Y VENTANA")
print("-" * 80)

all_oos = []
for run in runs:
    seed = run.name.split("seed")[-1] if "seed" in run.name else "UNK"
    seed_dir = run / seed
    if not seed_dir.exists():
        subdirs = [d for d in run.iterdir() if d.is_dir() and d.name != "FINAL"]
        seed_dir = subdirs[0] if subdirs else None
    if seed_dir is None:
        continue
    for w in ["W1","W2","W3","W4","W5"]:
        oos_file = seed_dir / w / "oos_trades.parquet"
        if oos_file.exists():
            try:
                df = pd.read_parquet(oos_file)
                df["seed"] = seed
                df["window"] = w
                df["run"] = run.name
                all_oos.append(df)
            except Exception as e:
                print(f"  ERROR leyendo {oos_file}: {e}")

if all_oos:
    combined = pd.concat(all_oos, ignore_index=True)
    print(f"  Total trades en runs nocturnas: {len(combined)}")
    print(f"  Seeds con datos OOS: {combined['seed'].nunique()}")

    print("\n  Por seed × window:")
    pivot = combined.groupby(["seed","window"])["is_win"].agg(["count","mean"]).rename(
        columns={"count":"N","mean":"WR"})
    pivot["WR"] = pivot["WR"].apply(lambda x: f"{x*100:.0f}%")
    print(pivot.to_string())

    print("\n  Por régimen HMM:")
    if "hmm_regime" in combined.columns:
        reg = combined.groupby("hmm_regime")["is_win"].agg(["count","mean"])
        reg.columns = ["N","WR"]
        reg["WR"] = reg["WR"].apply(lambda x: f"{x*100:.1f}%")
        print(reg.to_string())

    print("\n  Por alpha_trigger:")
    if "alpha_trigger" in combined.columns:
        at = combined.groupby("alpha_trigger")["is_win"].agg(["count","mean"])
        at.columns = ["N","WR"]
        at["WR"] = at["WR"].apply(lambda x: f"{x*100:.1f}%")
        print(at.to_string())

    # Estadísticas de retorno
    r = combined["return_pct"]
    wins = combined[combined["is_win"]==1]["return_pct"]
    losses = combined[combined["is_win"]==0]["return_pct"]
    print(f"\n  WinRate global: {combined['is_win'].mean()*100:.1f}% (N={len(combined)})")
    print(f"  Retorno total: {r.sum()*100:.4f}%")
    print(f"  avg_win:  {wins.mean()*100:.4f}%  (N={len(wins)})")
    print(f"  avg_loss: {losses.mean()*100:.4f}%  (N={len(losses)})")
    if len(losses) > 0 and losses.mean() != 0:
        print(f"  R:R: {abs(wins.mean()/losses.mean()):.3f}")
    # Equity
    equity = (1 + r).cumprod()
    roll_max = equity.cummax()
    mdd = ((equity - roll_max)/roll_max).min() * 100
    print(f"  MaxDrawdown: {mdd:.2f}%")
else:
    print("  NO SE ENCONTRARON OOS TRADES en runs nocturnas")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ANÁLISIS DE DEGRADADOS (Gate-G2) POR SEMILLA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] DEGRADADOS (GATE-G2) POR SEMILLA")
print("-" * 80)

for run in runs:
    seed = run.name.split("seed")[-1] if "seed" in run.name else "UNK"
    seed_dir = run / seed
    if not seed_dir.exists():
        subdirs = [d for d in run.iterdir() if d.is_dir() and d.name != "FINAL"]
        seed_dir = subdirs[0] if subdirs else None
    if seed_dir is None:
        continue
    for w in ["W1","W2","W3","W4","W5"]:
        disabled_file = seed_dir / w / "gate_g2_disabled_agents.json"
        if disabled_file.exists():
            try:
                import json
                with open(disabled_file) as f:
                    data = json.load(f)
                print(f"  {run.name[-25:]}/{w} — DEGRADED: {data}")
            except Exception as e:
                print(f"  {run.name[-25:]}/{w} — DEGRADED (error leyendo: {e})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANÁLISIS DE CADA WORKER LOG EN DETALLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] RESUMEN DETALLADO POR WORKER LOG")
print("-" * 80)

for log in worker_logs:
    lines_all = []
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            lines_all = f.readlines()
    except:
        continue

    seed_match = re.search(r"seed(\w+)", log.name)
    seed = seed_match.group(1) if seed_match else "?"

    # Extraer info clave
    windows_done = re.findall(r"CICLO VENTANA: (W\d)", "\n".join(lines_all))
    windows_oos = re.findall(r"aislada con (\d+) trades.*?ventana (W\d)", "\n".join(lines_all))
    gate_g5_ok = [l.strip()[:100] for l in lines_all if "GATE-G5" in l and "fallback_level" in l]
    gate_g5_fail = [l.strip()[:100] for l in lines_all if "GATE-G5" in l and "0 se" in l]
    errors = [l.strip()[:100] for l in lines_all if "| ERROR" in l or "CRITICAL" in l or "Traceback" in l]
    degraded = [l.strip()[:100] for l in lines_all if "DEGRADED" in l and "agente" in l.lower()]
    gauntlet = [l.strip()[:80] for l in lines_all if "GAUNTLET" in l]
    bull_gate = [l.strip()[:100] for l in lines_all if "FIX-BULL-GATE" in l or "bull.*DESACTIVADO" in l.lower()]

    print(f"\n  [{log.name[:45]}]  seed={seed}  lines={len(lines_all)}")
    print(f"    Ventanas iniciadas: {windows_done}")
    print(f"    OOS trades aislados: {windows_oos}")
    if gate_g5_ok:
        for g in gate_g5_ok[:2]: print(f"    G5 OK: {g[-80:]}")
    if gate_g5_fail:
        print(f"    G5 FAIL (0 señales): {len(gate_g5_fail)} ventanas")
    if degraded:
        for d in degraded[:2]: print(f"    DEGRADED: {d[-80:]}")
    if bull_gate:
        for b in bull_gate[:1]: print(f"    BULL_GATE: {b[-80:]}")
    if gauntlet:
        for g in gauntlet[:1]: print(f"    GAUNTLET: {g}")
    if errors:
        print(f"    ERRORS ({len(errors)}):")
        for e in errors[:3]: print(f"      {e[-100:]}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. RUN INTERRUMPIDA — seed58530
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] RUN INTERRUMPIDA — seed58530")
print("-" * 80)
run_58530 = RUNS_DIR / "WFB_20260602_070624_seed58530"
if run_58530.exists():
    print(f"  Directorio: {run_58530}")
    for item in sorted(run_58530.rglob("*")):
        if item.is_file():
            print(f"    {item.relative_to(run_58530)} ({item.stat().st_size} bytes)")
else:
    print("  Directorio NO encontrado")

# Buscar log de este seed en generate_oos
log_58530 = sorted(LOGS_DIR.glob("*58530*.log"))
print(f"  Logs relacionados con seed58530: {[l.name for l in log_58530]}")
for log in log_58530[:3]:
    print(f"\n  [{log.name}]")
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            content = f.read()
        for line in content.split("\n")[-20:]:
            if line.strip():
                print(f"    {line.strip()[:120]}")
    except Exception as e:
        print(f"  Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SEÑALES POR EMBUDO (signal_funnel)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] EMBUDO DE SEÑALES — RESUMEN GLOBAL")
print("-" * 80)

funnel_files = list(REPORTS_DIR.glob("signal_funnel_WFB_seed*.json"))
funnel_overnight = [f for f in funnel_files
                    if f.stat().st_mtime > datetime(2026,6,1,22,0).timestamp()]

import json
funnel_data = []
for ff in sorted(funnel_overnight, key=lambda x: x.stat().st_mtime):
    try:
        with open(ff) as f:
            data = json.load(f)
        # puede ser lista de ventanas acumuladas
        if isinstance(data, list):
            for item in data:
                item["seed_file"] = ff.stem
                funnel_data.append(item)
        elif isinstance(data, dict):
            data["seed_file"] = ff.stem
            funnel_data.append(data)
    except Exception as e:
        print(f"  Error leyendo {ff.name}: {e}")

print(f"  Funnel files encontrados: {len(funnel_overnight)}")

if funnel_data:
    # Estadísticas del embudo
    totals = {"raw": 0, "after_xgb": 0, "after_meta": 0, "after_hmm": 0,
              "after_embargo": 0, "final": 0, "zero_signal_windows": 0}
    for d in funnel_data:
        totals["raw"] += d.get("raw", d.get("n_raw", 0))
        totals["after_xgb"] += d.get("after_xgb", 0)
        totals["after_hmm"] += d.get("after_hmm", 0)
        totals["after_embargo"] += d.get("after_embargo", 0)
        totals["final"] += d.get("n_trades", d.get("final", 0))
        if d.get("status") == "zero_signals" or d.get("n_trades", 1) == 0:
            totals["zero_signal_windows"] += 1

    print(f"  Señales raw totales:    {totals['raw']}")
    print(f"  Después XGBoost:        {totals['after_xgb']}")
    print(f"  Después HMM:            {totals['after_hmm']}")
    print(f"  Después Embargo:        {totals['after_embargo']}")
    print(f"  FINAL (trades):         {totals['final']}")
    print(f"  Ventanas 0 señales:     {totals['zero_signal_windows']}")
    if totals["raw"] > 0:
        pct = totals["final"] / totals["raw"] * 100
        print(f"  Pass-through rate:      {pct:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. TEARSHEETS GENERADOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] TEARSHEETS GENERADOS")
print("-" * 80)
tearsheets = sorted(REPORTS_DIR.glob("*tearsheet_oos.png"),
                    key=lambda x: x.stat().st_mtime, reverse=True)
tearsheets_ov = [t for t in tearsheets if t.stat().st_mtime > datetime(2026,6,1,22,0).timestamp()]
print(f"  Tearsheets generados en la noche: {len(tearsheets_ov)}")
for t in tearsheets_ov:
    size_kb = t.stat().st_size // 1024
    print(f"    {t.name[:80]}  [{size_kb}KB]")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SEEDS COMPLETADAS VS FALLIDAS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] RESUMEN FINAL — SEEDS COMPLETADAS VS INCOMPLETAS")
print("-" * 80)

# Seeds con FINAL verdict
final_seeds = []
no_final_seeds = []
for run in runs:
    seed = run.name.split("seed")[-1] if "seed" in run.name else "UNK"
    seed_dir = run / seed
    if not seed_dir.exists():
        subdirs = [d for d in run.iterdir() if d.is_dir()]
        seed_dir = subdirs[0] if subdirs else None
    has_final = (seed_dir / "FINAL").exists() if seed_dir else False
    stat_val_logs = list(LOGS_DIR.glob(f"run_statistical_validation_*{run.name.split('_')[1]}_{run.name.split('_')[2]}*{seed}*.log"))
    has_stat_val = len(stat_val_logs) > 0
    if has_final or has_stat_val:
        final_seeds.append(f"{run.name[-30:]}")
    else:
        no_final_seeds.append(f"{run.name[-30:]}")

print(f"  Seeds con FINAL: {len(final_seeds)}")
for s in final_seeds: print(f"    {s}")
print(f"\n  Seeds SIN FINAL (incompletas/interrumpidas): {len(no_final_seeds)}")
for s in no_final_seeds: print(f"    {s}")

print("\n" + "=" * 80)
print("  FIN AUDITORIA")
print("=" * 80)
