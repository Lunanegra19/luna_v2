#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
measure_long_run.py — Panel de medicion estandar para el escuadron LONG (Luna V2).
Read-only. No toca settings/codigo/modelos ni lanza runs.

Uso:
    python tools/diagnostics/measure_long_run.py                 # todas las verdicts long
    python tools/diagnostics/measure_long_run.py --run WFB_20260624_112913
    python tools/diagnostics/measure_long_run.py --run WFB_20260624_112913 --json data/reports/baseline_long_20260624.json
    python tools/diagnostics/measure_long_run.py --direction short

Panel (ver docs/plan_mejora_long_paso_a_paso.md §1.1):
  - Por-seed (fuente autoritativa = verdict JSON): trades, WR, ret, maxDD, Sharpe, Calmar, DSR, PBO, binomial, flags.
  - Trade-level (fuente = data/predictions, TRANSITORIA): activos vs silenciados, edge por regimen.
  - Viabilidad: n_seeds>=30, n_seeds negativos.

NOTA: data/predictions/*.parquet se SOBRESCRIBEN en cada run. Para baseline inmutable
      se usa el verdict JSON (archivado por run). El bloque trade-level refleja
      el ultimo estado en disco, no necesariamente el run pedido.
"""
import argparse, json, glob, os, re, sys, io
import numpy as np

# Forzar UTF-8 en consola Windows (cp1252 revienta con ASCII art/simbolos)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS = os.path.join(ROOT, "data", "reports")
PRED = os.path.join(ROOT, "data", "predictions")
MIN_TRADES = 30


def load_verdicts(run=None, direction="long"):
    rows = []
    for f in sorted(glob.glob(os.path.join(REPORTS, f"*{direction}_FINAL_statistical_verdict.json"))):
        b = os.path.basename(f)
        if run and run not in b:
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        m = d.get("metrics", {}); sa = d.get("statistical_audit", {}); fl = d.get("flags", {})
        rid = re.search(r"(WFB_\d+_\d+)_seed(\d+)", d.get("run_id", b))
        rows.append(dict(
            run=rid.group(1) if rid else b,
            seed=int(rid.group(2)) if rid else -1,
            trades=m.get("total_trades"), wr=m.get("win_rate"),
            ret=m.get("total_return_pct"), dd=m.get("max_drawdown_pct"),
            sharpe=m.get("sharpe_crudo"), calmar=m.get("calmar_ratio"),
            dsr=sa.get("dsr"), pbo=sa.get("estimated_pbo"),
            binom=sa.get("binomial_p_value"), nwin=d.get("n_windows_with_trades"),
            p_trades=fl.get("pass_trades"), p_dsr=fl.get("pass_dsr"),
        ))
    return rows


def trade_level(direction="long"):
    """Pooled trade-level desde data/predictions (transitorio). Devuelve dict o None."""
    try:
        import pandas as pd
    except ImportError:
        return None
    frames = []
    for f in glob.glob(os.path.join(PRED, f"oos_trades_seed*_{direction}.parquet")):
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue
        if "direction" in d:
            d = d[d["direction"] == direction]
        if "kelly_fraction_used" not in d or "return_raw" not in d or len(d) == 0:
            continue
        seed = os.path.basename(f).split("seed")[1].split("_")[0]
        frames.append(d.assign(seed=seed))
    if not frames:
        return None
    df = pd.concat(frames)
    act = df[df["kelly_fraction_used"] > 0]
    sil = df[df["kelly_fraction_used"] == 0]

    def stats(x):
        if len(x) == 0:
            return dict(n=0, wr=float("nan"), mean_ret=float("nan"), ir=float("nan"))
        r = x["return_raw"]
        ir = (r.mean() / r.std()) if r.std() else float("nan")
        return dict(n=int(len(x)), wr=float(x["is_win"].mean()),
                    mean_ret=float(r.mean()), ir=float(ir))

    reg = (df.groupby("HMM_Semantic")
             .apply(lambda x: pd.Series({
                 "n": len(x),
                 "n_act": int((x["kelly_fraction_used"] > 0).sum()),
                 "wr_act": x.loc[x["kelly_fraction_used"] > 0, "is_win"].mean() if (x["kelly_fraction_used"] > 0).any() else np.nan,
                 "ret_act": x.loc[x["kelly_fraction_used"] > 0, "return_raw"].mean() if (x["kelly_fraction_used"] > 0).any() else np.nan,
             }), include_groups=False)
             .sort_values("n", ascending=False))
    return dict(active=stats(act), silenced=stats(sil),
                regime=reg.to_dict("index"), n_seeds=df["seed"].nunique())


def fnum(v, n=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{n}f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="substring del run_id (ej. WFB_20260624_112913)")
    ap.add_argument("--direction", default="long", choices=["long", "short"])
    ap.add_argument("--json", default=None, help="ruta para volcar el snapshot (baseline)")
    args = ap.parse_args()

    V = load_verdicts(args.run, args.direction)
    if not V:
        print(f"[!] Sin verdicts {args.direction} para run={args.run!r}")
        return
    V.sort(key=lambda x: x["seed"])

    print(f"\n=== PANEL {args.direction.upper()} | run={args.run or 'ALL'} | {len(V)} seeds (fuente: verdict JSON) ===")
    hdr = ["seed", "trades", "WR%", "ret%", "DD%", "Sharpe", "Calmar", "DSR", "PBO%", "binom", "n>=30", "neg"]
    print("  ".join(f"{h:>7}" for h in hdr))
    for r in V:
        print("  ".join([
            f"{r['seed']:>7}", f"{fnum(r['trades'],0):>7}",
            f"{(r['wr'] or 0)*100:>7.1f}", f"{fnum(r['ret'],2):>7}", f"{fnum(r['dd'],2):>7}",
            f"{fnum(r['sharpe'],3):>7}", f"{fnum(r['calmar'],2):>7}", f"{fnum(r['dsr'],3):>7}",
            f"{(r['pbo'] or 0)*100:>7.0f}", f"{fnum(r['binom'],3):>7}",
            f"{'Y' if (r['trades'] or 0)>=MIN_TRADES else 'n':>7}",
            f"{'Y' if (r['sharpe'] or 0)<0 or (r['ret'] or 0)<0 else 'n':>7}",
        ]))

    def col(k):
        return [r[k] for r in V if isinstance(r[k], (int, float))]
    agg = {}
    for k in ["trades", "wr", "ret", "dd", "sharpe", "calmar", "dsr"]:
        c = col(k)
        if c:
            mult = 100 if k == "wr" else 1
            agg[k] = dict(mean=float(np.mean(c)) * mult, median=float(np.median(c)) * mult,
                          min=float(np.min(c)) * mult, max=float(np.max(c)) * mult)
    n_ge30 = sum(1 for r in V if (r["trades"] or 0) >= MIN_TRADES)
    n_neg = sum(1 for r in V if (r["sharpe"] or 0) < 0 or (r["ret"] or 0) < 0)
    print("\n--- Agregados (pooled) ---")
    for k, a in agg.items():
        print(f"  {k:<8} mean={a['mean']:.3f} median={a['median']:.3f} min={a['min']:.3f} max={a['max']:.3f}")
    print(f"  n_seeds_ge_30 = {n_ge30}/{len(V)}   n_seeds_negativos = {n_neg}/{len(V)}")

    TL = trade_level(args.direction)
    if TL:
        a, s = TL["active"], TL["silenced"]
        print(f"\n--- Trade-level (data/predictions, TRANSITORIO, {TL['n_seeds']} seeds) ---")
        print(f"  ACTIVOS    n={a['n']:5d}  WR={a['wr']*100:5.1f}%  mean_ret={a['mean_ret']*100:+.3f}%  IR(mean/std)={fnum(a['ir'],3)}")
        print(f"  SILENCIADOS n={s['n']:5d}  WR={s['wr']*100:5.1f}%  mean_ret={s['mean_ret']*100:+.3f}%  IR(mean/std)={fnum(s['ir'],3)}")
        print("  Edge por regimen (activos):")
        for reg, v in TL["regime"].items():
            wr = v["wr_act"]; rt = v["ret_act"]
            wr = f"{wr*100:5.1f}%" if wr == wr else "  -- "
            rt = f"{rt*100:+.3f}%" if rt == rt else "  --  "
            print(f"    {reg:<20} n={int(v['n']):3d} act={int(v['n_act']):3d}  WR={wr}  ret={rt}")
    else:
        print("\n--- Trade-level: sin parquets en data/predictions (o falta pandas) ---")

    if args.json:
        out = dict(run=args.run, direction=args.direction, n_seeds=len(V),
                   per_seed=V, aggregates=agg, n_seeds_ge_30=n_ge30,
                   n_seeds_negativos=n_neg, trade_level=TL)
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w", encoding="utf-8"), indent=2, default=float)
        print(f"\n[OK] Snapshot guardado en {args.json}")


if __name__ == "__main__":
    main()
