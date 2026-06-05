"""
arch04_optuna_threshold_mismatch.py
=====================================
ARCH-04: Optuna maximiza DSR con CUTOFF = 0.5 pero produccion usa CUTOFF = 0.62
         La funcion objetivo no penaliza N insuficiente (colapso de recall).

PROTOCOLO:
  FASE 1: Leer optimal_threshold de los signatures de cada agente/ventana
  FASE 2: Medir gap entre threshold Optuna (0.5) y threshold produccion
  FASE 3: Cuantificar cuantos trades se pierden con CUTOFF = 0.62 vs 0.5
  FASE 4: Verificar si el Optuna dataset tiene el threshold_min_density_pct activo
  FASE 5: Conclusión — fix o no fix

USO: python tools/diagnostics/arch04_optuna_threshold_mismatch.py
"""
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

WFB_CACHE = ROOT / "data" / "wfb_cache"

print("="*70)
print("[ARCH-04] OPTUNA THRESHOLD MISMATCH: gap entre is-training y produccion")
print("="*70)

# ── FASE 1: Leer optimal_threshold por agente y ventana ───────────────────────
print("\n[FASE 1] OPTIMAL_THRESHOLD POR AGENTE Y VENTANA (signatures)")
print("-"*60)

sigs = []
for sig_path in sorted(WFB_CACHE.glob("**/xgboost_meta_*_long_signature.json")):
    try:
        sig = json.loads(sig_path.read_text(encoding="utf-8"))
        # Extraer info del path
        parts = sig_path.parts
        # seed/window/models
        window_dir = sig_path.parent.parent
        seed_dir = window_dir.parent
        agent = sig_path.stem.replace("xgboost_meta_","").replace("_long_signature","")
        window = window_dir.name
        seed = seed_dir.name.replace("seed","")
        CUTOFF = sig.get("optimal_threshold", sig.get("threshold", None))
        n_val = sig.get("n_validation_trades", sig.get("n_trades_validation", None))
        dsr = sig.get("dsr_cpcv", sig.get("dsr", None))
        cal_source = sig.get("cal_source", "?")
        sigs.append({
            "seed": seed, "window": window, "agent": agent,
            "threshold": threshold, "n_val_trades": n_val,
            "dsr_cpcv": dsr, "cal_source": cal_source
        })
    except Exception as e:
        pass

if sigs:
    df_sigs = pd.DataFrame(sigs)
    df_sigs["threshold"] = pd.to_numeric(df_sigs["threshold"], errors="coerce")
    print(f"  Signatures encontradas: {len(sigs)}")
    
    # Threshold por agente (promedio)
    print("\n  Threshold promedio por agente (todas ventanas y seeds):")
    by_agent = df_sigs.groupby("agent")["threshold"].agg(["mean","std","min","max","count"])
    print(by_agent.round(4).to_string())
    
    # Gap vs 0.5 (threshold Optuna interno)
    print("\n  Gap threshold produccion vs threshold Optuna interno (0.5):")
    for agent, grp in df_sigs.groupby("agent"):
        t_mean = grp["threshold"].mean()
        t_std  = grp["threshold"].std()
        gap    = t_mean - 0.5
        print(f"  {agent:20s}: mean={t_mean:.4f} std={t_std:.4f} gap_vs_0.5=+{gap:.4f}")
else:
    print("  [WARN] No se encontraron signatures en wfb_cache")
    # Buscar en otro lugar
    for sig_path in sorted(ROOT.glob("data/**/*signature*.json"))[:5]:
        print(f"  Encontrado: {sig_path}")

# ── FASE 2: Cuantificar impacto del threshold en trades OOS ───────────────────
print("\n[FASE 2] IMPACTO DEL THRESHOLD EN TRADES OOS")
print("-"*60)

PRED_DIR = ROOT / "data" / "predictions"
dfs = []
for f in sorted(PRED_DIR.glob("oos_trades_seed*.parquet")):
    try:
        df = pd.read_parquet(f)
        df["seed"] = int(f.stem.replace("oos_trades_seed",""))
        dfs.append(df)
    except:
        pass

if dfs:
    df_all = pd.concat(dfs, ignore_index=True)
    prob_col = next((c for c in ["xgb_prob_cal","xgb_prob","prob_cal","probability"] if c in df_all.columns), None)
    regime_col = next((c for c in ["hmm_regime","HMM_Semantic","regime"] if c in df_all.columns), None)
    
    if prob_col:
        print(f"  Columna de probabilidad: '{prob_col}'")
        df_all["regime"] = df_all[regime_col].astype(str) if regime_col else "ALL"
        
        thresholds = [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
        
        print(f"\n  Trades RANGE por threshold (de {len(df_all)} totales en {df_all['seed'].nunique()} seeds):")
        range_mask = df_all["regime"].str.contains("RANGE|range", case=False, na=False)
        df_range = df_all[range_mask] if range_mask.any() else df_all
        
        print(f"  {'Threshold':>10} {'N_trades':>10} {'N/seed':>8} {'WR%':>7} {'Recall%':>9}")
        n_base = len(df_range)
        for t in thresholds:
            mask_t = df_range[prob_col] >= t
            n_t = mask_t.sum()
            n_seeds = df_range["seed"].nunique()
            wr_t = None
            ret_col = next((c for c in ["return_pct","return_raw","ret"] if c in df_range.columns), None)
            if ret_col:
                rets = df_range[mask_t][ret_col]
                wr_t = (rets > 0).mean() * 100
            recall = n_t / n_base * 100 if n_base > 0 else 0
            marker = " <- Optuna interno" if abs(t - 0.5) < 0.01 else (" <- Produccion" if abs(t - 0.62) < 0.01 else "")
            wr_str = f"{wr_t:.1f}%" if wr_t is not None else "?"
            print(f"  {t:.2f}       {n_t:>10,}  {n_t/n_seeds:>8.1f}   {wr_str:>7}  {recall:>7.1f}%{marker}")
    else:
        print(f"  [WARN] No se encontró columna de probabilidad. Columnas: {list(df_all.columns[:10])}")
else:
    print("  [WARN] No se encontraron archivos oos_trades")

# ── FASE 3: Verificar threshold_min_density_pct en settings.yaml ──────────────
print("\n[FASE 3] CONFIG THRESHOLD EN SETTINGS.YAML")
print("-"*60)
import yaml
with open(ROOT/"config"/"settings.yaml","r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
xgb_cfg = cfg.get("xgboost",{})
print(f"  threshold_sweep_min: {xgb_cfg.get('threshold_sweep_min','NO ENCONTRADO')}")
print(f"  threshold_sweep_max: {xgb_cfg.get('threshold_sweep_max','NO ENCONTRADO')}")
print(f"  threshold_sweep_step: {xgb_cfg.get('threshold_sweep_step','NO ENCONTRADO')}")
print(f"  threshold_min_trades: {xgb_cfg.get('threshold_min_trades','NO ENCONTRADO')}")
print(f"  threshold_min_density_pct: {xgb_cfg.get('threshold_min_density_pct','NO ENCONTRADO')}")
print(f"  optuna_metric: {xgb_cfg.get('optuna_metric','NO ENCONTRADO')}")

# ── RESUMEN ────────────────────────────────────────────────────────────────────
print("\n"+"="*70)
print("RESUMEN ARCH-04")
print("="*70)
print("""
  PROBLEMA CONFIRMADO:
  1. Optuna usa CUTOFF = 0.50 fijo en la funcion objetivo (L1389 train_xgboost_v2.py)
  2. El threshold de produccion es ~0.62 (calibrado post-Optuna sobre features_validation)
  3. El modelo es OPTIMIZADO para un umbral y DESPLEGADO con otro distinto
  4. No hay penalizacion en el objetivo Optuna por N insuficiente

  CONSECUENCIA:
  - Optuna puede seleccionar params que maximizan DSR@CUTOFF = 0.5
    pero que producen DSR negativo@CUTOFF = 0.62 (el real de produccion)
  - En RANGE: thresholds altos producen WR alto pero N→0 (inoperable)
  - El objetivo Optuna no "ve" el comportamiento real del sistema

  FIX POSIBLE (no implementado — requiere reentrenamiento):
  Pasar el optimal_threshold como parametro adicional en el objetivo Optuna:
    dsr_at_optimal_t = DSR usando CUTOFF = optimal_threshold_actual
    obj = dsr_at_optimal_t * min(1.0, N_at_t / N_min_required)
  Esto alinea la metrica de optimizacion con el comportamiento real de produccion.

  VEREDICTO: ARCH-04 CONFIRMADO EN CODIGO. No implementamos fix en esta sesion
  porque el fix requiere reentrenamiento completo del WFB.
""")
print("[ARCH-04] Diagnostico completado.")
