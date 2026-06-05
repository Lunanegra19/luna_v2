import pandas as pd
from pathlib import Path

wfb_dir = Path("data/reports/wfb")

print("=== SEEDS CON W5 (holdout critico Q1-2026) ===")
w5_seeds = []
for f in sorted(wfb_dir.glob("oos_trades_W5_seed*.parquet")):
    seed = f.stem.split("seed")[1]
    df = pd.read_parquet(f)
    n = len(df)
    wr = df["is_win"].mean() * 100 if "is_win" in df.columns else float("nan")
    ret = df["return_pct"].mean() * 100 if "return_pct" in df.columns else 0
    w5_seeds.append({"seed": seed, "N": n, "WR_W5": round(wr, 1), "AvgRet": round(ret, 4)})

for r in sorted(w5_seeds, key=lambda x: -x["WR_W5"]):
    print(f"  seed={r['seed']:>6} | N={r['N']:>3} | WR_W5={r['WR_W5']}% | AvgRet={r['AvgRet']}%")

print()
print("=== DEGRADACION W5 vs W1-W4 por seed ===")
rows = []
for f in sorted(wfb_dir.glob("oos_trades_W5_seed*.parquet")):
    seed = f.stem.split("seed")[1]
    df5 = pd.read_parquet(f)
    wr5 = df5["is_win"].mean() * 100 if "is_win" in df5.columns else float("nan")
    n5 = len(df5)

    prev_dfs = []
    for w in ["W1", "W2", "W3", "W4"]:
        p = wfb_dir / f"oos_trades_{w}_seed{seed}.parquet"
        if p.exists():
            prev_dfs.append(pd.read_parquet(p))

    if prev_dfs:
        prev = pd.concat(prev_dfs, ignore_index=True)
        wr_prev = prev["is_win"].mean() * 100
        n_prev = len(prev)
        delta = wr5 - wr_prev
        rows.append((seed, wr_prev, n_prev, wr5, n5, delta))
        flag = "<<< MEJORA" if delta > 0 else ("XXX colapso" if delta < -10 else "")
        print(f"  seed={seed:>6} | WR_W1-W4={wr_prev:.1f}% (N={n_prev:>3}) | WR_W5={wr5:.1f}% (N={n5:>2}) | Delta={delta:+.1f}pp {flag}")

if rows:
    avg_delta = sum(r[5] for r in rows) / len(rows)
    avg_wr_prev = sum(r[1] for r in rows) / len(rows)
    avg_wr5 = sum(r[3] for r in rows) / len(rows)
    print()
    print(f"MEDIA: WR_W1-W4={avg_wr_prev:.1f}% -> WR_W5={avg_wr5:.1f}% | Delta={avg_delta:+.1f}pp")
    print()
    print("REFERENCIA historica (run anterior sin fixes):")
    print("  WR W1-W4 ~ 58%  |  WR W5 = 27%  |  Delta = -31pp")
    print()
    print(f"MEJORA NETA del fix vs run anterior: {avg_wr5 - 27:.1f}pp en W5")

print()
print("=== WR GLOBAL por seed (todas las ventanas disponibles) ===")
all_seeds = set()
for f in wfb_dir.glob("oos_trades_W*_seed*.parquet"):
    all_seeds.add(f.stem.split("seed")[1])

for seed in sorted(all_seeds):
    dfs = []
    ws = []
    for w in ["W1", "W2", "W3", "W4", "W5"]:
        p = wfb_dir / f"oos_trades_{w}_seed{seed}.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
            ws.append(w)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        wr = combined["is_win"].mean() * 100
        n = len(combined)
        ret = combined["return_pct"].mean() * 100 if "return_pct" in combined.columns else 0
        print(f"  seed={seed:>6} | {str(ws):30s} | N={n:>3} | WR={wr:.1f}% | AvgRet={ret:.4f}%")
