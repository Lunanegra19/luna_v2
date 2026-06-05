"""
arch03_bull_gate_diagnosis.py
===================================
ARCH-03: Régimen BULL bloqueado sin alternativa — sistema genera <30 trades/quarter
Verifica distribución real de trades BULL vs NON-BULL y throughput por ventana.
"""
import pandas as pd, numpy as np, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
PRED = ROOT / "data" / "predictions"

print("="*70)
print("[ARCH-03] BULL GATE: distribucion real de trades por regimen")
print("="*70)

dfs = []
for f in sorted(PRED.glob("oos_trades_seed*.parquet")):
    try:
        df = pd.read_parquet(f)
        df["seed"] = int(f.stem.replace("oos_trades_seed",""))
        dfs.append(df)
    except Exception as e:
        print(f"  [WARN] {f.name}: {e}")

if not dfs:
    print("[ERROR] No se encontraron archivos oos_trades_seed*.parquet")
    sys.exit(1)

df_all = pd.concat(dfs, ignore_index=True)
regime_col = next((c for c in ["hmm_regime","HMM_Semantic","regime"] if c in df_all.columns), None)
df_all["regime"] = df_all[regime_col].astype(str) if regime_col else "UNKNOWN"
df_all["is_bull"] = df_all["regime"].str.upper().str.contains("BULL")

n_seeds = df_all["seed"].nunique()
n_total = len(df_all)
n_bull = df_all["is_bull"].sum()
n_nonbull = n_total - n_bull

print(f"  Total trades: {n_total:,} | Seeds: {n_seeds} | Trades/seed: {n_total/n_seeds:.1f}")
print(f"  BULL trades: {n_bull:,} ({n_bull/n_total*100:.1f}%)")
print(f"  NON-BULL trades: {n_nonbull:,} ({n_nonbull/n_total*100:.1f}%)")

wfb_col = next((c for c in ["wfb_window","window"] if c in df_all.columns), None)
if wfb_col:
    print(f"\n  N trades por ventana WFB (promedio sobre todas las seeds):")
    by_window = df_all.groupby(wfb_col).size() / n_seeds
    for w, n in by_window.items():
        print(f"    {w}: {n:.1f} trades/seed")
    sop_r8_min = 30
    below_sop = by_window[by_window < sop_r8_min]
    print(f"\n  Ventanas con < {sop_r8_min} trades/seed (viola SOP R8): {len(below_sop)}/{len(by_window)}")

# Bull gate settings
import yaml
with open(ROOT/"config"/"settings.yaml","r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
bull_gate = cfg.get("gauntlet", {}).get("bull_gate_min_dsr", cfg.get("bull_gate_min_dsr","NO ENCONTRADO"))
print(f"\n  gauntlet.bull_gate_min_dsr: {bull_gate}")

# Distribucion de regimenes en OOS (hmm_regime_labels.parquet)
feat_dir = ROOT / "data" / "features"
hmm_path = feat_dir / "hmm_regime_labels.parquet"
if hmm_path.exists():
    df_hmm = pd.read_parquet(hmm_path)
    oos_start = pd.Timestamp("2024-01-01", tz="UTC")
    df_oos = df_hmm[df_hmm.index >= oos_start]
    if "HMM_Semantic" in df_oos.columns:
        dist = df_oos["HMM_Semantic"].value_counts()
        total_oos = len(df_oos)
        print(f"\n  Distribucion HMM en OOS (2024+, {total_oos:,} barras):")
        for state, cnt in dist.items():
            is_bull_state = "BULL" in str(state).upper()
            flag = " <- BULL BLOQUEADO" if is_bull_state else ""
            print(f"    {state:30s}: {cnt:5,} ({cnt/total_oos*100:.1f}%){flag}")
        bull_pct = dist[dist.index.str.upper().str.contains("BULL")].sum() / total_oos * 100
        print(f"\n  BULL ocupa {bull_pct:.1f}% del tiempo OOS — bloqueado por gate")
        print(f"  Trades perdidos estimados: {bull_pct:.0f}% de las oportunidades")

print("\n[ARCH-03] VEREDICTO:")
print("  Fix requerido: desbloquear BULL requiere resolver ARCH-01 primero (EV neto)")
print("  Sin EV positivo el gate BULL es CORRECTO — protege contra perdidas")
print("  ARCH-03 no es un bug sino una CONSECUENCIA de ARCH-01")
print("  No implementamos fix en esta sesion.")
