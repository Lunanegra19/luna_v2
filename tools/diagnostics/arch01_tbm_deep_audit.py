"""arch01_tbm_deep_audit.py
Auditoria profunda de los parametros TBM y el mecanismo de generacion de retornos.
Objetivo: entender por que ret_medio=0.009% < coste=0.15% y encontrar el fix correcto.
"""
import sys, yaml, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("[ARCH-01] TBM DEEP AUDIT — ret=0.009% vs coste=0.15%")
print("="*70)

# ── 1. Leer todos los parametros TBM ─────────────────────────────────────────
print("\n[1] PARAMETROS TBM EN SETTINGS.YAML")
print("-"*60)
with open(ROOT/"config"/"settings.yaml","r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

xgb = cfg.get("xgboost", {})
sop = cfg.get("sop", {})
fase2 = cfg.get("fase2", {})

# Grupos de parametros TBM
tbm_groups = {
    "Multiplicadores PT/SL": ["pt_mult_min","pt_mult_max","sl_mult_min","sl_mult_max","pt_mult","sl_mult"],
    "Barreras temporales": ["vertical_barrier_hours","atr_period","atr_multiplier","tbm_mode"],
    "Retorno minimo": ["tbm_min_return","min_return"],
    "Barreras dinamicas": ["dynamic_barrier","dynamic_barrier_atr","pt_decay_fraction","linear_decay_pt"],
    "Costes": ["cost_pct","ev_tolerance_pct"],
}

for group, keys in tbm_groups.items():
    found = {}
    for section_name, section in [("xgboost", xgb), ("sop", sop), ("fase2", fase2)]:
        for k in keys:
            if k in section:
                found[f"{section_name}.{k}"] = section[k]
    if found:
        print(f"\n  {group}:")
        for k, v in found.items():
            print(f"    {k}: {v}")

# Buscar regime_tbm_profiles
print("\n  Regime TBM Profiles:")
rtp = xgb.get("regime_tbm_profiles", {})
if rtp:
    for regime, params in rtp.items():
        print(f"    {regime}: {params}")
else:
    # Buscar en subniveles
    for k, v in xgb.items():
        if "profile" in k.lower() or "regime_tbm" in k.lower():
            print(f"    xgboost.{k}: {v}")
    print("    (No encontrado en xgboost — buscando en otras secciones...)")
    for section_name, section in [("sop", sop), ("fase2", fase2)]:
        if isinstance(section, dict):
            for k, v in section.items():
                if "profile" in k.lower() or "tbm" in k.lower():
                    print(f"    {section_name}.{k}: {v}")

# ── 2. Calcular el retorno minimo teorico dado PT/SL ──────────────────────────
print("\n[2] ANALISIS MATEMATICO DE LAS BARRERAS")
print("-"*60)

# Leer parametros
pt_min = xgb.get("pt_mult_min", None)
pt_max = xgb.get("pt_mult_max", None)
sl_min = xgb.get("sl_mult_min", None)
sl_max = xgb.get("sl_mult_max", None)
tbm_min_return = xgb.get("tbm_min_return", 0.003)
vbh = xgb.get("vertical_barrier_hours", 96)
cost_pct = sop.get("cost_pct", 0.0015)

print(f"\n  ATR-based barriers:")
print(f"    pt_mult_min: {pt_min} | pt_mult_max: {pt_max}")
print(f"    sl_mult_min: {sl_min} | sl_mult_max: {sl_max}")
print(f"    tbm_min_return: {tbm_min_return} ({tbm_min_return*100:.2f}%)")
print(f"    vertical_barrier_hours: {vbh}H")
print(f"    cost_pct (round-trip): {cost_pct} ({cost_pct*100:.3f}%)")

if pt_min and sl_min:
    print(f"\n  Matematica de la barrera:")
    print(f"    TP = ATR * {pt_min}x  → retorno si TP toca: {tbm_min_return*100:.2f}% minimo")
    print(f"    SL = ATR * {sl_min}x  → retorno si SL toca: -{tbm_min_return*100:.2f}% (simetrico)")
    print(f"    Ratio PT/SL teorico: {pt_min}/{sl_min} = {pt_min/sl_min:.2f}x")
    print(f"\n  Para ser rentable con ratio {pt_min}/{sl_min}:")
    ratio = float(pt_min)/float(sl_min)
    # EV = WR * PT - (1-WR) * SL = 0 → WR_min = SL/(PT+SL) = 1/(1+ratio)
    wr_breakeven = 1 / (1 + ratio)
    print(f"    WR breakeven (sin costes): {wr_breakeven*100:.1f}%")
    wr_breakeven_cost = (1 + cost_pct/tbm_min_return) / (1 + ratio) if tbm_min_return > 0 else 0.5
    print(f"    WR breakeven (con costes {cost_pct*100:.3f}%): ~{wr_breakeven_cost*100:.1f}%")

# ── 3. Analizar el parquet de features para retornos reales ───────────────────
print("\n[3] RETORNOS TBM REALES EN FEATURES_TRAIN")
print("-"*60)
train_path = ROOT / "data" / "features" / "features_train.parquet"
if train_path.exists():
    # Leer solo columnas relevantes
    cols_to_try = ["Target_TBM_Bin", "Target_TBM_Ret", "tbm_ret", "ret", "TBM_return", "close"]
    available_cols = pd.read_parquet(train_path, columns=None).columns.tolist()
    ret_cols = [c for c in available_cols if any(k in c.lower() for k in ["ret","return","tbm","target","close"])]
    print(f"  Columnas de retorno disponibles: {ret_cols[:15]}")
    
    df_sample = pd.read_parquet(train_path, columns=ret_cols[:10] if ret_cols else available_cols[:5])
    
    if "Target_TBM_Ret" in df_sample.columns:
        r = df_sample["Target_TBM_Ret"].dropna()
        print(f"\n  Target_TBM_Ret stats ({len(r)} filas):")
        print(f"    Media:  {r.mean()*100:.4f}%")
        print(f"    Mediana:{r.median()*100:.4f}%")
        print(f"    Std:    {r.std()*100:.4f}%")
        print(f"    P25:    {r.quantile(0.25)*100:.4f}%")
        print(f"    P75:    {r.quantile(0.75)*100:.4f}%")
        wr = (r > 0).mean()
        avg_win = r[r > 0].mean() if (r > 0).any() else 0
        avg_loss = r[r < 0].mean() if (r < 0).any() else 0
        ev_gross = wr * avg_win + (1 - wr) * avg_loss
        ev_net = ev_gross - float(cost_pct)
        print(f"    Win Rate: {wr*100:.2f}%")
        print(f"    Avg Win:  {avg_win*100:.4f}%")
        print(f"    Avg Loss: {avg_loss*100:.4f}%")
        print(f"    EV bruto: {ev_gross*100:.4f}%")
        print(f"    EV neto (post-coste {cost_pct*100:.3f}%): {ev_net*100:.4f}%")
        if ev_net < 0:
            print(f"    ⚠️ EV NEGATIVO: el sistema pierde dinero en promedio IS")
        else:
            print(f"    ✅ EV POSITIVO en IS")
    
    if "Target_TBM_Bin" in df_sample.columns:
        b = df_sample["Target_TBM_Bin"].dropna()
        print(f"\n  Target_TBM_Bin base rate: {b.mean()*100:.2f}%")
else:
    print("  features_train.parquet no encontrado")

# ── 4. Posibles causas del ret=0.009% ─────────────────────────────────────────
print("\n[4] HIPOTESIS PARA ret=0.009% < 0.15%")
print("-"*60)
print("""
HIPOTESIS A — tbm_min_return demasiado bajo:
  Si tbm_min_return=0.003 (0.3%), el retorno minimo para que un trade
  sea etiquetado como 1 es 0.3%, que es solo 2x el coste (0.15%).
  En muchos casos el modelo aprendera a predecir trades que tocan TP
  en 0.3-0.5% — EV bruto positivo pero EV neto negativo.

HIPOTESIS B — vertical_barrier_hours demasiado corto:
  Con vbh=96H (4 dias), muchos trades se cierran por barrera temporal
  con retorno cercano a 0 (ni TP ni SL tocados). Estos trades arrastran
  el retorno medio hacia 0.

HIPOTESIS C — multiplicadores PT/SL demasiado conservadores:
  Si PT=1.8x ATR y SL=1.5x ATR, el TP es apenas 20% mayor que el SL.
  Con un win rate de ~52%, el EV bruto es muy pequenyo.

HIPOTESIS D — ATR de BTC en 2025 es alto:
  Con BTC volatile, ATR alto implica TP/SL en terminos absolutos grandes,
  pero los multiplicadores no se ajustan -> barreras muy amplias ->
  mas trades cerrados por barrera temporal (ret~0).
""")

print("[ARCH-01] Audit completado.")
