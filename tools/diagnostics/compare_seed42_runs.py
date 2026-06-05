"""
Diagnóstico forense: compara todos los archivos oos_trades de seed42 W1
para detectar inconsistencias entre runs distintas.
"""
import pandas as pd
import glob
import os
import json

print("=" * 70)
print("FORENSE: Comparacion de oos_trades seed42 W1 entre runs")
print("=" * 70)

patterns = [
    "data/runs/*/seed42/W1/oos_trades.parquet",
    "data/runs/WFB_*seed42*/seed42/W1/oos_trades.parquet",
]
files = set()
for p in patterns:
    files.update(glob.glob(p))
files = sorted(files, key=os.path.getmtime)

print(f"\n[1] Archivos oos_trades W1/seed42 encontrados: {len(files)}")
for f in files:
    df = pd.read_parquet(f)
    wins = int(df["is_win"].sum()) if "is_win" in df.columns else -1
    total = len(df)
    wr = wins / total if total > 0 and wins >= 0 else 0
    mtime = pd.Timestamp(os.path.getmtime(f), unit="s").strftime("%Y-%m-%d %H:%M:%S")
    ret = df["return_pct"].sum() if "return_pct" in df.columns else 0
    print(f"\n  [{mtime}] {f}")
    print(f"    Trades={total} | Wins={wins} | WR={wr:.1%} | Return={ret:.4%}")

print("\n" + "=" * 70)
print("[2] Veredicto estadistico final (statistical_verdict.json)")
verdict_path = "data/reports/statistical_verdict.json"
if os.path.exists(verdict_path):
    v = json.load(open(verdict_path))
    print(f"  total_trades : {v['metrics']['total_trades']}")
    print(f"  win_rate     : {v['metrics']['win_rate']:.4f} ({v['metrics']['win_rate']*100:.1f}%)")
    print(f"  wfv_results  : {json.dumps(v.get('wfv_results', {}), indent=4)}")
else:
    print("  NO ENCONTRADO")

print("\n" + "=" * 70)
print("[3] Parquets de predicciones finales")
candidates = [
    "data/predictions/oos_trades_seed42.parquet",
    "data/oos/oos_trades.parquet",
    "data/predictions/oos_trades.parquet",
]
for c in candidates:
    if os.path.exists(c):
        df = pd.read_parquet(c)
        wins = int(df["is_win"].sum()) if "is_win" in df.columns else -1
        total = len(df)
        wr = wins / total if total > 0 and wins >= 0 else 0
        print(f"\n  {c}")
        print(f"    Trades={total} | Wins={wins} | WR={wr:.1%}")
        if hasattr(df.index, "min"):
            print(f"    Index: {df.index.min()} -> {df.index.max()}")

print("\n" + "=" * 70)
print("[4] Detalle trades W1 seed42 (run mas reciente)")
if files:
    df = pd.read_parquet(files[-1])
    cols = ["direction", "return_pct", "is_win", "xgb_prob_cal", "hmm_regime"]
    cols_ok = [c for c in cols if c in df.columns]
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df[cols_ok].to_string())
    wins = int(df["is_win"].sum()) if "is_win" in df.columns else -1
    print(f"\nTotal={len(df)} | Wins={wins} | WR={df['is_win'].mean():.1%}")
    print(f"\n[BUG-HUNT] xgb_prob_cal stats:")
    if "xgb_prob_cal" in df.columns:
        print(df["xgb_prob_cal"].describe())
