"""
audit_signal_funnel.py
======================
Reconstruye el embudo de senales (signal funnel) ventana a ventana
para seeds 42 y 100 leyendo directamente los parquets OOS canonicos.

Objetivo: entender que gate bloquea las senales en W1, W2, W5.

[AUDIT-FUNNEL-01] Script de diagnostico - no modifica nada.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RUNS_DIR = ROOT / "data" / "runs"
REPORTS_DIR = ROOT / "data" / "reports"
LOG_PATH = None  # se detecta automáticamente

# Seeds y ventanas a auditar
SEEDS = [42, 100]
WINDOWS = ["W1", "W2", "W3", "W4", "W5"]

# Run IDs conocidos (los más recientes de la run activa)
RUN_IDS = {
    42:  "WFB_20260528_202402_seed42",
    100: "WFB_20260528_214002_seed100",
}

# ─────────────────────────────────────────────────────────────────────
def load_raw_probs(seed: int, window: str) -> pd.DataFrame | None:
    run_id = RUN_IDS.get(seed)
    if not run_id:
        return None
    path = RUNS_DIR / run_id / f"seed{seed}" / window / "oos_raw_probs.parquet"
    if not path.exists():
        print(f"  [AUDIT-FUNNEL-01] MISSING: {path.relative_to(ROOT)}")
        return None
    df = pd.read_parquet(path)
    print(f"  [AUDIT-FUNNEL-01] LOADED: seed{seed}/{window} → {len(df)} rows | cols={list(df.columns)[:8]}")
    return df


def analyze_window(seed: int, window: str) -> dict:
    """Analiza un parquet OOS y devuelve las columnas disponibles y counts."""
    df = load_raw_probs(seed, window)
    result = {"seed": seed, "window": window, "raw_bars": 0, "available": False, "columns": []}

    if df is None or len(df) == 0:
        result["note"] = "EMPTY/MISSING — parquet vacío (Risk-Off Shield activado antes de inference)"
        return result

    result["raw_bars"] = len(df)
    result["available"] = True
    result["columns"] = list(df.columns)

    print(f"\n  [AUDIT-FUNNEL-01] seed{seed}/{window} — {len(df)} barras OOS")
    print(f"    Columnas: {list(df.columns)}")

    # Intentar reconstruir el embudo desde las columnas disponibles
    funnel = {}

    # Gate 0: barras totales del período OOS
    funnel["raw_oos_bars"] = len(df)

    # Gate XGBoost: columnas de probabilidad XGBoost
    xgb_cols = [c for c in df.columns if "xgb" in c.lower() or "prob" in c.lower() or "signal" in c.lower()]
    meta_cols = [c for c in df.columns if "meta" in c.lower() or "labeler" in c.lower()]
    hmm_cols  = [c for c in df.columns if "hmm" in c.lower() or "regime" in c.lower() or "shield" in c.lower()]
    trade_cols = [c for c in df.columns if "trade" in c.lower() or "entry" in c.lower() or "signal_final" in c.lower()]

    print(f"    XGB cols: {xgb_cols}")
    print(f"    Meta cols: {meta_cols}")
    print(f"    HMM cols: {hmm_cols}")
    print(f"    Trade cols: {trade_cols}")

    # Contar señales que superaron cada gate si la columna existe
    for col in xgb_cols[:3]:
        if col in df.columns:
            n_pos = (df[col] > 0.48).sum() if df[col].dtype in [float, np.float64] else (df[col] == 1).sum()
            funnel[f"after_{col}"] = int(n_pos)
            print(f"    {col} > 0.48: {n_pos} / {len(df)}")

    for col in meta_cols[:2]:
        if col in df.columns:
            n_pos = (df[col] == 1).sum() if df[col].dtype != float else (df[col] > 0.5).sum()
            funnel[f"after_{col}"] = int(n_pos)
            print(f"    {col} == 1: {n_pos}")

    for col in trade_cols[:2]:
        if col in df.columns:
            n_pos = df[col].notna().sum() if df[col].dtype == object else (df[col] == 1).sum()
            funnel[f"after_{col}"] = int(n_pos)
            print(f"    {col} signals: {n_pos}")

    # Estadísticas básicas del parquet
    for col in df.columns:
        if df[col].dtype in [np.float64, float]:
            print(f"    {col}: min={df[col].min():.4f} mean={df[col].mean():.4f} max={df[col].max():.4f} non-null={df[col].notna().sum()}")
        elif df[col].dtype in [np.int64, int, bool]:
            vc = df[col].value_counts().head(5).to_dict()
            print(f"    {col}: {vc}")
        else:
            vc = df[col].value_counts().head(5).to_dict()
            print(f"    {col} (str): {vc}")

    result["funnel"] = funnel
    return result


def extract_log_funnel(log_path: Path, seed: int, window: str) -> dict:
    """Extrae del log el desglose de gates para una ventana específica."""
    if not log_path or not log_path.exists():
        return {}

    print(f"\n  [AUDIT-FUNNEL-01] Extrayendo log para seed{seed}/{window}...")
    results = {}

    # Patrones a buscar
    patterns = {
        "raw_oos_bars":    r"raw_oos_bars.*?(\d+)",
        "after_xgb":       r"after_xgb.*?(\d+)|XGBoost.*?(\d+).*?señ",
        "after_meta":      r"after_meta.*?(\d+)|MetaLabeler.*?(\d+).*?señ",
        "after_hmm":       r"after_hmm.*?(\d+)",
        "after_embargo":   r"after_embargo.*?(\d+)",
        "momentum_blocked": r"MOMENTUM.*?(\d+) BLOQUEADAS",
        "shield_forced":   r"total_forced=(\d+)",
        "trades_oos":      r"(\d+) trades OOS",
        "zero_signals":    r"zero_signals",
    }

    # Buscar en el log las líneas relevantes para esta seed+window
    seed_tag = f"seed{seed}"
    window_tag = window

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Dividir en secciones por ventana/seed para no mezclar contextos
        # Buscamos bloques entre marcadores de ventana
        window_marker = f"VENTANA: {window}"
        blocks = content.split(window_marker)

        for i, block in enumerate(blocks[1:], 1):  # skip before first marker
            # Verificar que este bloque es de la seed correcta
            if seed_tag not in block[:2000] and f"seed={seed}" not in block[:2000]:
                continue

            # Solo tomar las primeras 50.000 chars del bloque (hasta la próxima ventana)
            next_block = block[:50000]

            for key, pattern in patterns.items():
                match = re.search(pattern, next_block, re.IGNORECASE)
                if match:
                    groups = [g for g in match.groups() if g is not None]
                    if groups:
                        results[key] = int(groups[0])
                    elif key == "zero_signals":
                        results[key] = True

            if results:
                break  # Encontramos el bloque correcto

    except Exception as e:
        print(f"    [AUDIT-FUNNEL-01] Error leyendo log: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("[AUDIT-FUNNEL-01] ANÁLISIS DE EMBUDO DE SEÑALES — seeds 42 y 100")
    print("="*70)

    # Detectar log más reciente
    log_candidates = sorted(Path(
        r"C:\Users\Usuario\.gemini\antigravity-ide\brain\238b39a8-3d3a-4951-b5b6-0f0693aa5072\.system_generated\tasks"
    ).glob("task-137.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    log_path = log_candidates[0] if log_candidates else None
    print(f"\n[AUDIT-FUNNEL-01] Log: {log_path}")

    all_results = []

    for seed in SEEDS:
        print(f"\n{'─'*70}")
        print(f"[AUDIT-FUNNEL-01] SEED {seed}")
        print(f"{'─'*70}")

        for window in WINDOWS:
            print(f"\n{'─'*50}")
            print(f"  [AUDIT-FUNNEL-01] {window}")
            print(f"{'─'*50}")

            r = analyze_window(seed, window)
            r["log_funnel"] = extract_log_funnel(log_path, seed, window)
            all_results.append(r)

    # ─── RESUMEN CONSOLIDADO ──────────────────────────────────────────
    print("\n\n" + "="*70)
    print("[AUDIT-FUNNEL-01] RESUMEN CONSOLIDADO — EMBUDO DE SEÑALES")
    print("="*70)
    print(f"{'Seed':<6} {'Window':<8} {'Raw Bars':<12} {'Disponible':<12} {'Nota'}")
    print("-"*70)
    for r in all_results:
        disponible = "✅ SÍ" if r["available"] else "❌ NO"
        nota = r.get("note", "")
        print(f"  {r['seed']:<6} {r['window']:<8} {r['raw_bars']:<12} {disponible:<12} {nota}")

    # ─── ANÁLISIS POR GATE ─────────────────────────────────────────
    print("\n" + "="*70)
    print("[AUDIT-FUNNEL-01] COLUMNAS DISPONIBLES EN PARQUETS OOS")
    print("="*70)
    for r in all_results:
        if r["available"] and r.get("columns"):
            print(f"  seed{r['seed']}/{r['window']}: {r['columns']}")

    print("\n[AUDIT-FUNNEL-01] Diagnóstico completado.\n")


if __name__ == "__main__":
    main()
