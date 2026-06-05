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
    if seed_label is None: continue
    seed_sub = run / seed_label
    if not seed_sub.exists():
        subs = [d for d in run.iterdir() if d.is_dir()]
        seed_sub = subs[0] if subs else None
    if seed_sub is None: continue
    pq = seed_sub / "W3" / "oos_trades.parquet"
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
            df["seed"] = seed_label
            all_trades.append(df)
        except Exception:
            pass

df_w3 = pd.concat(all_trades, ignore_index=True)
df_range = df_w3[df_w3["hmm_regime"].str.contains("RANGE", na=False)].copy()
df_range["entry_time"] = pd.to_datetime(df_range["entry_time"], utc=True)
df_range["entry_date"] = df_range["entry_time"].dt.date
df_range["entry_hour"] = df_range["entry_time"].dt.hour

print("=== ANALISIS CONCENTRACION 07H UTC ===")
n_seeds_total = df_range["seed"].nunique()
print(f"Total trades RANGE W3: {len(df_range)}")
print(f"Seeds distintas: {n_seeds_total}")
print()

dates_unique = df_range["entry_date"].value_counts().sort_index()
print("Fechas de entrada (unicas):")
for date, n in dates_unique.items():
    print(f"  {date}: {n} trades")
print()

print("¿Son distintas seeds en la misma barra de mercado?")
same_bar = df_range.groupby("entry_time").agg(
    n_seeds=("seed", "nunique"),
    seeds_list=("seed", lambda x: ", ".join(sorted(set(x))))
).reset_index()
print(same_bar.to_string())
print()

unique_bars = df_range["entry_time"].nunique()
total_trades = len(df_range)
avg_per_bar = total_trades / unique_bars if unique_bars > 0 else 0
print(f"Trades totales:          {total_trades}")
print(f"Barras unicas de entrada: {unique_bars}")
print(f"Trades por barra promedio: {avg_per_bar:.1f}")
print(f"Seeds por run: {n_seeds_total}")
print()

if avg_per_bar > n_seeds_total * 0.5:
    print("DIAGNOSTICO: Los trades son en barras MUY diversas.")
    print("El edge es robusto — no es una sola barra replicada.")
elif unique_bars < 5:
    print("DIAGNOSTICO: Muy pocas barras unicas. El edge depende de 1-4 barras especificas.")
    print("WR=100% con N=23 puede ser N seeds x 1 barra = estadistica engañosa.")
    print("Esa barra a 07h puede ser un evento especifico (no un edge sistematico).")
else:
    print(f"DIAGNOSTICO: {unique_bars} barras unicas en {n_seeds_total} seeds.")

# Causalidad economica: por que todas a las 07h?
print()
print("CAUSALIDAD ECONOMICA:")
print("07:00 UTC = Apertura de Frankfurt Stock Exchange (08:00 CEST)")
print("Es la hora de mayor volumen institucional europeo en BTC-USD.")
print("El edge a 07h puede ser real (liquidez institucional) o puede ser")
print("que el gate [7-13] hace que TODOS los trades empiecen en la primera hora permitida.")
print()
print("Si el model genera senal a las 05h pero el gate la bloquea,")
print("la proxima vez que puede entrar es a las 07h -> sesgo sistematico.")
print("Con gate desactivado, las entradas se distribuirian naturalmente.")
