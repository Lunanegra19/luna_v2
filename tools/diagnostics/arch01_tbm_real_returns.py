"""arch01_tbm_real_returns.py
Lee los oos_trades.parquet de todas las runs de las ultimas 24H y calcula
los retornos reales por trade para diagnosticar el gap ret vs coste.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("[ARCH-01] RETORNOS REALES OOS — ULTIMAS 24H DE RUNS")
print("="*70)

# ── 1. Encontrar todos los oos_trades.parquet de hoy ─────────────────────────
cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
runs_dir = ROOT / "data" / "runs"

trade_files = []
for f in runs_dir.rglob("oos_trades.parquet"):
    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
    if mtime >= cutoff:
        # Extraer info del path: runs/WFB_FECHA_seedXX/seedXX/WN/oos_trades.parquet
        parts = f.parts
        run_id = parts[-4] if len(parts) >= 4 else "?"
        window = parts[-2] if len(parts) >= 2 else "?"
        trade_files.append({"path": f, "run_id": run_id, "window": window, "mtime": mtime})

trade_files.sort(key=lambda x: x["mtime"])
print(f"\n[1] ARCHIVOS oos_trades.parquet ENCONTRADOS ({len(trade_files)} en ultimas 24H)")
print("-"*60)
for tf in trade_files:
    size = tf["path"].stat().st_size
    print(f"  {tf['run_id']}/{tf['window']}  ({tf['mtime'].strftime('%H:%M')})  {size:,}B")

if not trade_files:
    print("  NO se encontraron archivos oos_trades.parquet recientes")
    sys.exit(0)

# ── 2. Cargar y consolidar todos los trades ───────────────────────────────────
print(f"\n[2] ANALISIS DE TRADES POR RUN")
print("-"*60)

all_dfs = []
for tf in trade_files:
    try:
        df = pd.read_parquet(tf["path"])
        df["_run_id"] = tf["run_id"]
        df["_window"] = tf["window"]
        all_dfs.append(df)
        print(f"\n  {tf['run_id']}/{tf['window']}: {len(df)} trades")
        print(f"    Columnas: {list(df.columns)[:12]}")
    except Exception as e:
        print(f"  ERROR leyendo {tf['path'].name}: {e}")

if not all_dfs:
    print("  No se pudieron cargar trades")
    sys.exit(0)

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"\n  TOTAL consolidado: {len(df_all)} trades de {len(trade_files)} ventanas")

# ── 3. Identificar columna de retorno ─────────────────────────────────────────
print(f"\n[3] COLUMNAS DE RETORNO DISPONIBLES")
print("-"*60)
print(f"  {list(df_all.columns)}")

ret_col = None
for candidate in ["ret","return","xgb_ret","trade_ret","pnl","pnl_pct","ret_raw","ret_net","tbm_ret"]:
    if candidate in df_all.columns:
        ret_col = candidate
        break
# Buscar cualquier columna con 'ret'
if not ret_col:
    ret_cols = [c for c in df_all.columns if "ret" in c.lower() or "pnl" in c.lower()]
    if ret_cols:
        ret_col = ret_cols[0]
        print(f"  Usando primera columna de retorno encontrada: {ret_col}")

agent_col = None
for candidate in ["agent","regime","agent_name","HMM_Semantic","regime_name"]:
    if candidate in df_all.columns:
        agent_col = candidate
        break

# ── 4. Estadisticas de retornos reales ───────────────────────────────────────
print(f"\n[4] ESTADISTICAS DE RETORNOS REALES")
print("-"*60)

COST_PCT = 0.0015  # 0.15% round-trip

if ret_col:
    r = df_all[ret_col].dropna()
    print(f"\n  Columna usada: '{ret_col}' ({len(r)} trades validos)")
    print(f"\n  --- RETORNO BRUTO (pre-coste) ---")
    print(f"    Media:   {r.mean()*100:.4f}%")
    print(f"    Mediana: {r.median()*100:.4f}%")
    print(f"    Std:     {r.std()*100:.4f}%")
    print(f"    Min:     {r.min()*100:.4f}%")
    print(f"    Max:     {r.max()*100:.4f}%")

    r_net = r - COST_PCT
    wr = (r > 0).mean()
    wr_net = (r_net > 0).mean()
    avg_win = r[r > 0].mean() if (r > 0).any() else 0
    avg_loss = abs(r[r < 0].mean()) if (r < 0).any() else 0
    ev_gross = wr * avg_win - (1 - wr) * avg_loss
    ev_net = ev_gross - COST_PCT

    print(f"\n  --- METRICAS CLAVE ---")
    print(f"    Win Rate bruto:     {wr*100:.2f}%")
    print(f"    Win Rate neto:      {wr_net*100:.2f}%")
    print(f"    Avg Win:            {avg_win*100:.4f}%")
    print(f"    Avg Loss:           {avg_loss*100:.4f}%")
    print(f"    Ratio Win/Loss:     {avg_win/avg_loss:.2f}x" if avg_loss > 0 else "    Ratio Win/Loss:     INF")
    print(f"    EV bruto/trade:     {ev_gross*100:.4f}%")
    print(f"    Coste round-trip:   {COST_PCT*100:.4f}%")
    print(f"    EV neto/trade:      {ev_net*100:.4f}%")

    if ev_net < 0:
        print(f"\n  ⚠️  EV NEGATIVO: el sistema pierde {abs(ev_net)*100:.4f}% por trade de media")
        print(f"     Para breakeven necesita EV bruto >= {COST_PCT*100:.4f}%")
        print(f"     Gap actual: {(ev_net)*100:.4f}% ({abs(ev_net)/COST_PCT*100:.0f}% del coste)")
    else:
        print(f"\n  ✅ EV POSITIVO: {ev_net*100:.4f}% por trade de media")

    # Distribucion de retornos
    print(f"\n  --- DISTRIBUCION DE RETORNOS ---")
    bins = [-np.inf, -0.005, -0.003, -0.0015, 0, 0.0015, 0.003, 0.005, 0.01, np.inf]
    labels = ["<-0.5%", "-0.5/-0.3%", "-0.3/-0.15%", "-0.15/0%",
              "0/+0.15%", "+0.15/+0.3%", "+0.3/+0.5%", "+0.5/+1%", ">+1%"]
    hist = pd.cut(r, bins=bins, labels=labels).value_counts().sort_index()
    for label, count in hist.items():
        bar = "█" * min(40, int(count / max(hist) * 40))
        print(f"    {label:15} | {count:3d} | {bar}")

    # Por agente si existe la columna
    if agent_col:
        print(f"\n  --- POR AGENTE ('{agent_col}') ---")
        for ag, grp in df_all.groupby(agent_col):
            r_ag = grp[ret_col].dropna()
            if len(r_ag) == 0:
                continue
            wr_ag = (r_ag > 0).mean()
            ev_ag = r_ag.mean() - COST_PCT
            print(f"    {str(ag):20} | N={len(r_ag):3d} | "
                  f"WR={wr_ag*100:.1f}% | "
                  f"ret_medio={r_ag.mean()*100:.4f}% | "
                  f"EV_neto={ev_ag*100:.4f}%"
                  f"{'  ⚠️' if ev_ag < 0 else '  ✅'}")

    # Por ventana
    print(f"\n  --- POR VENTANA ---")
    for win, grp in df_all.groupby("_window"):
        r_w = grp[ret_col].dropna()
        if len(r_w) == 0:
            continue
        ev_w = r_w.mean() - COST_PCT
        print(f"    {win} | N={len(r_w):3d} | "
              f"ret_medio={r_w.mean()*100:.4f}% | "
              f"EV_neto={ev_w*100:.4f}%"
              f"{'  ⚠️' if ev_w < 0 else '  ✅'}")
else:
    print("  No se encontró columna de retorno en los parquets")
    print(f"  Columnas disponibles: {list(df_all.columns)}")

print("\n[ARCH-01] Analisis de retornos reales completado.")
