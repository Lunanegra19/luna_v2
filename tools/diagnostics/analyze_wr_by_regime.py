"""
[MEJORA-WR-01] analyze_wr_by_regime.py
=======================================
Análisis forense de seed2025 WR=40%.

La auditoría detectó que seed2025 tiene WR=40% — potencialmente señal invertida
en algún régimen HMM. Este script calcula WR, Sharpe y MaxDD por régimen HMM
y compara seed2025 vs seed1337 (la única aprobada) para identificar la diferencia.

Uso:
    python tools/diagnostics/analyze_wr_by_regime.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Fix encoding Windows cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
RUNS_DIR = PROJECT_ROOT / "data" / "runs"

print("=" * 70)
print("[MEJORA-WR-01] Análisis forense WR por régimen HMM — seed2025 vs seed1337")
print("=" * 70)


def load_seed_trades(seed: int) -> pd.DataFrame:
    """Carga y concatena todos los parquets OOS de una seed."""
    frames = []
    for p in sorted(RUNS_DIR.rglob(f"oos_trades*.parquet")):
        if f"seed{seed}" in str(p):
            try:
                df = pd.read_parquet(p)
                # Extraer ventana del path
                parts = Path(p).parts
                window = next((x for x in parts if x.startswith("W") and len(x) == 2), "?")
                run_id = next((x for x in parts if x.startswith("WFB_")), "?")
                df["window"] = window
                df["run_id"] = run_id
                frames.append(df)
                print(f"  Cargado: seed{seed}/{window} — {len(df)} trades")
            except Exception as e:
                print(f"  [WARN] Error: {p}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def analyze_by_regime(df: pd.DataFrame, seed: int) -> None:
    """Calcula métricas por régimen HMM."""
    if df.empty:
        print(f"  [!] No hay datos para seed{seed}")
        return

    print(f"\n[seed{seed}] Total trades: {len(df)}")
    print(f"  Columnas disponibles: {list(df.columns)[:15]}")

    # WR global
    if "is_win" in df.columns:
        wr_global = df["is_win"].mean()
        print(f"  WR global: {wr_global*100:.1f}%")
    if "return_pct" in df.columns:
        rets = df["return_pct"].values
        sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(len(rets))
        cum = (1 + rets).cumprod()
        peaks = np.maximum.accumulate(cum)
        maxdd = float(np.abs(np.min((cum - peaks) / peaks)))
        print(f"  Sharpe: {sharpe:.3f} | MaxDD: {maxdd*100:.1f}%")
    print()

    # Por régimen HMM si disponible
    regime_col = None
    for c in ["hmm_regime", "HMM_Regime", "hmm_state", "regime", "HMM_Semantic"]:
        if c in df.columns:
            regime_col = c
            break

    if regime_col:
        print(f"  Columna régimen: '{regime_col}'")
        print(f"  {'Régimen':<25} | {'N trades':>8} | {'WR%':>7} | {'Sharpe':>8} | {'MaxDD%':>8}")
        print("  " + "-" * 65)
        for regime, grp in df.groupby(regime_col):
            n = len(grp)
            wr = grp["is_win"].mean() * 100 if "is_win" in grp.columns else float("nan")
            if "return_pct" in grp.columns and n >= 3:
                r = grp["return_pct"].values
                sh = np.mean(r) / (np.std(r) + 1e-10) * np.sqrt(n)
                cum = (1 + r).cumprod()
                pk = np.maximum.accumulate(cum)
                mdd = float(np.abs(np.min((cum - pk) / pk))) * 100
            else:
                sh, mdd = float("nan"), float("nan")
            flag = " ⚠️  WR<50" if wr < 50 else ""
            print(f"  {str(regime):<25} | {n:>8} | {wr:>7.1f} | {sh:>8.3f} | {mdd:>8.1f}{flag}")
    else:
        print("  [INFO] No hay columna de régimen HMM en los trades.")
        print("  Columnas disponibles:", list(df.columns))

    # Por ventana
    print()
    print(f"  {'Ventana':<10} | {'N trades':>8} | {'WR%':>7} | {'Run ID'}")
    print("  " + "-" * 55)
    for _, grp in df.groupby(["window", "run_id"]):
        if grp.empty:
            continue
        window = grp["window"].iloc[0]
        run_id = grp["run_id"].iloc[0][:20]
        n = len(grp)
        wr = grp["is_win"].mean() * 100 if "is_win" in grp.columns else float("nan")
        print(f"  {window:<10} | {n:>8} | {wr:>7.1f} | {run_id}")


# ── Análisis seed2025 ─────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("SEED 2025")
print("─" * 70)
df_2025 = load_seed_trades(2025)
analyze_by_regime(df_2025, 2025)

# ── Análisis seed1337 (referencia aprobada) ───────────────────────────────────
print("\n" + "─" * 70)
print("SEED 1337 (referencia — única aprobada en SFI16)")
print("─" * 70)
df_1337 = load_seed_trades(1337)
analyze_by_regime(df_1337, 1337)

# ── Comparativa directa ───────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("COMPARATIVA seed2025 vs seed1337")
print("─" * 70)
if "is_win" in df_2025.columns and "is_win" in df_1337.columns:
    print(f"  WR global:  seed2025={df_2025['is_win'].mean()*100:.1f}%  |  seed1337={df_1337['is_win'].mean()*100:.1f}%")
if "return_pct" in df_2025.columns and "return_pct" in df_1337.columns:
    for seed, df_s in [(2025, df_2025), (1337, df_1337)]:
        r = df_s["return_pct"].values
        total_ret = float(np.prod(1 + r) - 1) * 100
        mean_r = np.mean(r) * 100
        std_r = np.std(r) * 100
        print(f"  seed{seed}: total_ret={total_ret:.1f}% | mean_trade={mean_r:.2f}% | std_trade={std_r:.2f}%")

print()
print("[MEJORA-WR-01] Análisis forense completado.")
