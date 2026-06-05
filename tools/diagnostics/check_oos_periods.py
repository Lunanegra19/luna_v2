"""
tools/diagnostics/check_oos_periods.py
Muestra los periodos OOS reales de cada seed en cada ventana.
Responde: todas las seeds tienen el mismo OOS?
[CHECK-OOS-PERIODS 2026-05-30]
"""
import re, json
import pandas as pd
from pathlib import Path
from collections import defaultdict

wfb_dir = Path(r"g:\Mi unidad\ia\luna_v2\data\reports\wfb")
SEEDS = [42, 100, 777, 1337, 2025, 12751, 28020, 30915, 34324, 43610, 77542]

print("[CHECK-OOS-PERIODS] Leyendo timestamps reales de los parquets de trades...")
print()

seed_window_dates = defaultdict(dict)

for seed in SEEDS:
    for f in sorted(wfb_dir.glob(f"oos_trades_W*_seed{seed}.parquet")):
        win = re.search(r"(W\d)", f.stem).group(1)
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
            t_min = df.index.min().strftime("%Y-%m-%d")
            t_max = df.index.max().strftime("%Y-%m-%d")
            seed_window_dates[seed][win] = (t_min, t_max)
        except Exception as e:
            seed_window_dates[seed][win] = ("ERR", str(e)[:20])

# Tabla por ventana
for win in ["W2", "W3", "W4", "W5"]:
    print(f"--- Ventana {win} ---")
    dates_seen = set()
    for seed in SEEDS:
        if win in seed_window_dates[seed]:
            t0, t1 = seed_window_dates[seed][win]
            dates_seen.add((t0, t1))
            print(f"  seed {seed:5}: {t0} -> {t1}")
        else:
            print(f"  seed {seed:5}: EMPTY / sin trades")
    if len(dates_seen) == 1:
        print(f"  [OK] Todas las seeds con trades tienen el MISMO OOS: {list(dates_seen)[0]}")
    elif len(dates_seen) > 1:
        print(f"  [WARN] Periodos OOS DISTINTOS: {dates_seen}")
    print()

# Ver los G0 para entender los rangos de datos completos (train+oos)
print("=== RANGOS DE DATOS COMPLETOS (gate_G0) ===")
print("(Muestra: seeds 42, 43610, 34324 en todas las ventanas)")
for seed in [42, 43610, 34324]:
    for win in ["W1", "W2", "W3", "W4", "W5"]:
        gf = wfb_dir / f"gate_G0_{win}_seed{seed}.json"
        if gf.exists():
            data = json.load(open(gf))
            m = data.get("metrics", {})
            d_start = m.get("date_start", "?")
            d_end   = m.get("date_end", "?")
            n_rows  = m.get("n_rows", "?")
            print(f"  seed{seed:5} {win}: {d_start} -> {d_end} ({n_rows} rows)")
    print()
