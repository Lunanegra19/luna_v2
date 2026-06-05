import pandas as pd, numpy as np
from pathlib import Path

PREDS = Path("g:/Mi unidad/ia/luna_v2/data/predictions")

for seed in ["42", "53929"]:
    p = PREDS / f"oos_trades_seed{seed}.parquet"
    df = pd.read_parquet(p)
    print(f"=== seed{seed}: {len(df)} trades ===")
    if "is_win" in df.columns:
        print(f"  WR (is_win): {df['is_win'].mean():.1%}")
    if "return_pct" in df.columns:
        wr = (df["return_pct"] > 0).mean()
        ev = df["return_pct"].mean()
        print(f"  WR (ret_pct>0): {wr:.1%}  EV={ev:.5f}")
    if "return_raw" in df.columns:
        ev_raw = df["return_raw"].mean()
        print(f"  EV return_raw: {ev_raw:.5f}")
    if "hmm_regime" in df.columns:
        print(f"  Regimenes: {df['hmm_regime'].value_counts().to_dict()}")
    if "wfb_window" in df.columns:
        print(f"  Por ventana: {df['wfb_window'].value_counts().sort_index().to_dict()}")
    if "direction" in df.columns:
        print(f"  Direcciones: {df['direction'].value_counts().to_dict()}")
    if "signal_threshold" in df.columns:
        print(f"  Thr media: {df['signal_threshold'].mean():.3f}")
    print()

all_trades = []
for p in sorted(PREDS.glob("oos_trades_seed*.parquet")):
    try:
        df = pd.read_parquet(p)
        df["seed"] = p.stem.replace("oos_trades_seed","")
        all_trades.append(df)
    except Exception:
        pass

if all_trades:
    combined = pd.concat(all_trades, ignore_index=True)
    print(f"=== GLOBAL ({len(combined)} trades, {len(all_trades)} seeds) ===")
    if "is_win" in combined.columns:
        print(f"  WR global (is_win)  : {combined['is_win'].mean():.1%}")
    if "return_pct" in combined.columns:
        print(f"  WR global (ret>0)   : {(combined['return_pct']>0).mean():.1%}")
        print(f"  EV global (ret_pct) : {combined['return_pct'].mean():.5f}")
        q = combined["return_pct"].quantile([0.05,0.25,0.5,0.75,0.95])
        print(f"  P05={q[0.05]:.4f} P25={q[0.25]:.4f} P50={q[0.5]:.4f} P75={q[0.75]:.4f} P95={q[0.95]:.4f}")
    # Reduccion downstream real
    total_oos_bars = 5 * 2401  # aprox media de barras por ventana
    print(f"  Media trades/seed : {combined.groupby('seed').size().mean():.0f}")
    print(f"  Reduccion downstream real: {1 - combined.groupby('seed').size().mean()/total_oos_bars:.1%}")
