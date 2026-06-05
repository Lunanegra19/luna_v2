"""
tools/diagnostics/audit_document_proposals.py
Auditoria institucional del documento wfb_calibration_forensic_20260519.md
Corrobora o refuta cada proposicion con datos reales. Sin implementar nada.
"""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("g:/Mi unidad/ia/luna_v2")
MODELS = ROOT / "data" / "models"
FEATURES = ROOT / "data" / "features"
PREDS = ROOT / "data" / "predictions"
WFB_CACHE = ROOT / "data" / "wfb_cache"

results = {}

# ─── AUDIT-01: FIX-FUNNEL-ACCUM-01 correctamente implementado ────────────────
print("\n[AUDIT-01] FIX-FUNNEL-ACCUM-01 en signal_filter.py")
sf = Path(ROOT / "luna/models/signal_filter.py").read_text(encoding="utf-8")
ok = all(x in sf for x in ["FIX-FUNNEL-ACCUM-01", "run_id", "accum"])
results["AUDIT-01"] = "PASS" if ok else "FAIL"
print(f"  Label presente: {'FIX-FUNNEL-ACCUM-01' in sf}")
print(f"  run_id logic  : {'run_id' in sf}")
print(f"  => {results['AUDIT-01']}")

# ─── AUDIT-02: FIX-BRIER-GATE-RANGE-01 en lugar correcto ─────────────────────
print("\n[AUDIT-02] FIX-BRIER-GATE-RANGE-01 en train_xgboost_v2.py")
src = Path(ROOT / "luna/models/train_xgboost_v2.py").read_text(encoding="utf-8")
ok2 = ("FIX-BRIER-GATE-RANGE-01" in src and
       "_brier_margin = 0.030 if _regime_str_gate" in src and
       "_brier_adaptive_gate = round(_brier_naive_true + 0.010, 4)" not in src)
# Verificar que esta DENTRO del bloque else (no en IDEA-G)
idx = src.find("FIX-BRIER-GATE-RANGE-01")
before = src[max(0,idx-500):idx]
in_else_block = "brier_naive_true" in before and "IDEA-A" in src[max(0,idx-800):idx]
results["AUDIT-02"] = "PASS" if (ok2 and in_else_block) else "WARN"
print(f"  Label presente : {'FIX-BRIER-GATE-RANGE-01' in src}")
print(f"  Logica margin  : {'_brier_margin = 0.030 if _regime_str_gate' in src}")
print(f"  Vieja borrada  : {'round(_brier_naive_true + 0.010' not in src}")
print(f"  En bloque IDEA-A: {in_else_block}")
print(f"  => {results['AUDIT-02']}")

# ─── AUDIT-03: EV real en TBM vs proxy 1H ────────────────────────────────────
print("\n[AUDIT-03] EV TBM real (calibration_report) vs EV proxy 1H")
print("  PROPOSICION: 'EV negativo en todas las ventanas validation con TBM'")
for agent in ["bull", "range", "bear"]:
    sp = MODELS / f"xgboost_meta_{agent}_long_signature.json"
    if not sp.exists():
        print(f"  {agent}: signature no encontrada")
        continue
    sig = json.loads(sp.read_text(encoding="utf-8"))
    rep = sig.get("calibration_report", [])
    evs = [r.get("ev") for r in rep if r.get("ev") is not None]
    n_pos = sum(1 for e in evs if e > 0)
    ev_min = min(evs) if evs else None
    ev_max = max(evs) if evs else None
    best_wr = max((r.get("win_rate", r.get("wr", 0)) for r in rep), default=0)
    cal_src = sig.get("cal_source", "?")
    ev_min_s = f"{ev_min:.5f}" if ev_min is not None else "N/A"
    ev_max_s = f"{ev_max:.5f}" if ev_max is not None else "N/A"
    print(f"  {agent}: n={len(evs)} EV_min={ev_min_s} "
          f"EV_max={ev_max_s} EV>0={n_pos} "
          f"best_WR={best_wr:.1%} src={cal_src}")

print()
print("  PROPOSICION: 'El proxy 1H subestima el EV real TBM'")
print("  Razon: TBM usa PT/SL simetrico con max 72h horizonte -> asimetria positiva")
print("  El 1H-return es simetrico y no captura el PT/SL advantage")
print("  => Esta proposicion NO fue testada con datos. PENDIENTE verificacion.")

# ─── AUDIT-04: WR real de los trades que pasaron el pipeline completo ─────────
print("\n[AUDIT-04] WR real de oos_trades vs WR estimada en el documento")
print("  PROPOSICION: 'Sistema genera WR=68% en los 23 trades reales'")
for seed in ["42", "53929"]:
    p = PREDS / f"oos_trades_seed{seed}.parquet"
    if not p.exists():
        print(f"  seed{seed}: no encontrado")
        continue
    df = pd.read_parquet(p)
    print(f"  seed{seed}: {len(df)} trades | cols={list(df.columns)}")
    if "ret" in df.columns or "pnl" in df.columns or "return" in df.columns:
        rcol = next(c for c in ["ret","pnl","return","ret_pct"] if c in df.columns)
        wr = (df[rcol] > 0).mean()
        ev = df[rcol].mean()
        print(f"    WR={wr:.1%} EV={ev:.5f} n={len(df)}")
    else:
        print(f"    Columnas ret/pnl no encontradas: {list(df.columns)[:8]}")

# ─── AUDIT-05: Verificar la afirmacion '40% del tiempo en regimen RANGE' ─────
print("\n[AUDIT-05] Porcentaje de barras en regimen RANGE vs afirmacion 40%")
print("  PROPOSICION: 'RANGE silenciaba ~40% del tiempo de mercado'")
hmm_path = FEATURES / "hmm_regime_labels.parquet"
for fpath in [FEATURES / "features_holdout_W3.parquet", FEATURES / "features_holdout.parquet"]:
    if not fpath.exists():
        continue
    df = pd.read_parquet(fpath, columns=["close"] + (
        ["HMM_Semantic"] if "HMM_Semantic" in pd.read_parquet(fpath, columns=["close"]).columns
        else []))
    if "HMM_Semantic" not in df.columns:
        # intentar join desde hmm_labels
        if hmm_path.exists():
            df_hmm = pd.read_parquet(hmm_path)
            df = df.join(df_hmm[["HMM_Semantic"]] if "HMM_Semantic" in df_hmm.columns else pd.DataFrame(), how="left")
    if "HMM_Semantic" in df.columns:
        counts = df["HMM_Semantic"].value_counts(normalize=True)
        range_pct = counts[counts.index.str.contains("RANGE|range|Range", case=False)].sum()
        bull_pct  = counts[counts.index.str.contains("BULL|bull|Bull", case=False)].sum()
        bear_pct  = counts[counts.index.str.contains("BEAR|bear|Bear", case=False)].sum()
        print(f"  {fpath.name}: RANGE={range_pct:.1%} BULL={bull_pct:.1%} BEAR={bear_pct:.1%}")
        for regime, pct in counts.head(8).items():
            print(f"    {regime}: {pct:.1%}")
    break

# ─── AUDIT-06: WFB-PRIOR disponibilidad real en cache ────────────────────────
print("\n[AUDIT-06] WFB-PRIOR: datos reales disponibles en cache")
print("  PROPOSICION: '23 ventanas con EV>0 para BULL, 9 para RANGE'")
for agent in ["bull", "range", "bear"]:
    sigs_ev = []
    for seed_dir in WFB_CACHE.glob("seed*"):
        for w in ["W1","W2","W3","W4"]:
            sp = seed_dir / w / "models" / f"xgboost_meta_{agent}_long_signature.json"
            if not sp.exists():
                continue
            try:
                s = json.loads(sp.read_text(encoding="utf-8"))
                rep = s.get("calibration_report", [])
                has_ev = any(r.get("ev", -1) > 0 for r in rep)
                if has_ev:
                    thr = s.get("optimal_threshold")
                    max_ev = max(r.get("ev", -1) for r in rep)
                    sigs_ev.append({"seed": seed_dir.name, "w": w, "thr": thr, "ev": max_ev})
            except Exception:
                pass
    thrs = [x["thr"] for x in sigs_ev if x["thr"]]
    median = float(np.median(thrs)) if thrs else None
    median_s = f"{median:.3f}" if median else "N/A"
    rango_s = f"[{min(thrs):.3f},{max(thrs):.3f}]" if thrs else "sin datos"
    print(f"  {agent}: {len(sigs_ev)} ventanas con EV>0 | "
          f"median_thr={median_s} | rango={rango_s}")


# ─── AUDIT-07: Verificar afirmacion 'cadena causal' - periodo validacion ─────
print("\n[AUDIT-07] Periodos de validacion reales por ventana WFB")
print("  PROPOSICION: 'Validacion = 2024-H1 (adverso)'")
for w in ["W1","W2","W3","W4","W5"]:
    p = FEATURES / f"features_validation_{w}.parquet"
    if p.exists():
        df = pd.read_parquet(p, columns=["close"])
        print(f"  {w}: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} barras)")

# ─── AUDIT-08: Verificar 'downstream reduce 70-85%' ──────────────────────────
print("\n[AUDIT-08] Reduccion real downstream (XGB signals -> final trades)")
print("  PROPOSICION: 'downstream reduce 70-85%'")
# Calcular con datos reales: cuantas barras total vs cuantos trades en oos_trades
all_seeds_data = []
for p in PREDS.glob("oos_trades_seed*.parquet"):
    try:
        df = pd.read_parquet(p)
        all_seeds_data.append({"seed": p.stem.replace("oos_trades_seed",""), "n_trades": len(df)})
    except Exception:
        pass
if all_seeds_data:
    ns = [x["n_trades"] for x in all_seeds_data]
    print(f"  Seeds analizadas: {len(all_seeds_data)}")
    print(f"  Trades por seed: min={min(ns)} max={max(ns)} median={np.median(ns):.0f} mean={np.mean(ns):.0f}")
    print(f"  Total barras OOS (5 ventanas x ~2400): ~12000")
    print(f"  Reduccion real media: {(1-np.median(ns)/12000)*100:.1f}% (si todas son XGB bruto)")
    print(f"  NOTA: oos_trades contiene los trades finales post-todos-los-filtros")
    for x in sorted(all_seeds_data, key=lambda x: x["n_trades"])[:5]:
        print(f"    seed{x['seed']}: {x['n_trades']} trades finales")

# ─── AUDIT-09: HMM nomenclatura - verificar mismatch ────────────────────────
print("\n[AUDIT-09] Nomenclatura HMM: mismatch IS vs pipeline actual")
print("  PROPOSICION: 'IS usa nombres distintos que el mapa configurado'")
hmm_path = FEATURES / "hmm_regime_labels.parquet"
if hmm_path.exists():
    df_hmm = pd.read_parquet(hmm_path)
    print(f"  Columnas hmm_labels: {list(df_hmm.columns)}")
    if "HMM_Semantic" in df_hmm.columns:
        uniq = df_hmm["HMM_Semantic"].dropna().unique()
        print(f"  Regimenes IS: {sorted(uniq)}")
# Buscar el regime_map en el codigo
src_sf = Path(ROOT / "luna/models/signal_filter.py").read_text(encoding="utf-8")
import re
maps = re.findall(r'regime_map\s*=\s*\{[^}]+\}', src_sf, re.DOTALL)
if maps:
    print(f"  regime_map en signal_filter: {maps[0][:200]}")

# ─── AUDIT-10: Verificar claim de brier_naive en Gate-G2 ─────────────────────
print("\n[AUDIT-10] Valores Brier por agente desde signatures reales")
print("  PROPOSICION: RANGE brier=0.2773, naive=0.2501, gate=0.2601 (17bp de margen)")
for agent in ["bull", "range", "bear"]:
    sp = MODELS / f"xgboost_meta_{agent}_long_signature.json"
    if not sp.exists():
        continue
    sig = json.loads(sp.read_text(encoding="utf-8"))
    brier_raw = sig.get("xgb_brier_raw", sig.get("xgb_brier_calibrated"))
    brier_gate = sig.get("brier_adaptive_gate")
    base_rate = sig.get("base_rate_is", sig.get("base_rate"))
    thr = sig.get("optimal_threshold")
    print(f"  {agent}: brier_raw={brier_raw} brier_gate={brier_gate} "
          f"base_rate={base_rate} thr={thr}")
    if brier_raw and brier_gate:
        margin = brier_gate - (base_rate*(1-base_rate) if base_rate else 0)
        diff = brier_raw - brier_gate
        print(f"    => brier_raw vs gate: {diff:+.4f} ({'DEGRADED' if diff>0 else 'PASA'})")

print("\n\n=== RESUMEN DE AUDITORIAS ===")
for k, v in results.items():
    print(f"  {k}: {v}")
