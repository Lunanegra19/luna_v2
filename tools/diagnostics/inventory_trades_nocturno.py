import pandas as pd
import numpy as np
from pathlib import Path
import datetime

BASE = Path(r"g:\Mi unidad\ia\luna_v2")
RUNS = BASE / "data" / "runs"
cutoff = datetime.datetime(2026, 6, 1, 22, 0, 0).timestamp()
overnight_runs = [d for d in sorted(RUNS.iterdir()) if d.is_dir() and d.stat().st_mtime >= cutoff]

all_trades = []
for run in overnight_runs:
    seed_label = run.name.split("seed")[-1] if "seed" in run.name else None
    if seed_label is None:
        continue
    seed_sub = run / seed_label
    if not seed_sub.exists():
        subs = [d for d in run.iterdir() if d.is_dir()]
        seed_sub = subs[0] if subs else None
    if seed_sub is None:
        continue
    for w in ["W1", "W2", "W3", "W4", "W5"]:
        pq = seed_sub / w / "oos_trades.parquet"
        if pq.exists():
            try:
                df = pd.read_parquet(pq)
                df["seed"] = seed_label
                df["window"] = w
                all_trades.append(df)
            except Exception:
                pass

df_all = pd.concat(all_trades, ignore_index=True)
print("=== INVENTARIO DATOS DISPONIBLES PARA HIPOTESIS ADICIONALES ===")
print(f"Total trades: {len(df_all)}")
print()

print("Por ventana:")
for w in ["W1", "W2", "W3", "W4", "W5"]:
    sub = df_all[df_all["window"] == w]
    if len(sub) > 0:
        wr = sub["is_win"].mean()
        ev = sub["return_pct"].mean() * 100
        print(f"  {w}: N={len(sub):3d}  WR={wr*100:.0f}%  EV={ev:+.4f}%")
print()

print("Por regimen:")
for reg in sorted(df_all["hmm_regime"].dropna().unique()):
    sub = df_all[df_all["hmm_regime"] == reg]
    wr = sub["is_win"].mean()
    ev = sub["return_pct"].mean() * 100
    print(f"  {reg}: N={len(sub):3d}  WR={wr*100:.0f}%  EV={ev:+.4f}%")
print()

print("Columnas disponibles:")
print([c for c in df_all.columns if not c.startswith("_")])
print()

# Analisis de retornos para ver si hay hipotesis de EV worth testing
print("Estadisticas globales de retorno:")
r = df_all["return_pct"] * 100
print(f"  Media:   {r.mean():.4f}%")
print(f"  Std:     {r.std():.4f}%")
print(f"  Sharpe:  {r.mean()/r.std() * (365*24)**0.5:.2f} (anualizado aprox)")
print(f"  Max:     {r.max():.4f}%")
print(f"  Min:     {r.min():.4f}%")
print()

# Retorno por dia de semana
if "entry_time" in df_all.columns:
    df_all["entry_dt"] = pd.to_datetime(df_all["entry_time"], utc=True)
    df_all["dow"] = df_all["entry_dt"].dt.dayofweek
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    print("WR por dia de la semana:")
    for d, name in dow_names.items():
        sub = df_all[df_all["dow"] == d]
        if len(sub) > 0:
            print(f"  {name}: N={len(sub):2d}  WR={sub['is_win'].mean()*100:.0f}%")
