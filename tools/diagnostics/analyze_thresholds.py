"""
analyze_thresholds.py — Auditoría de thresholds de los agentes XGBoost
Investiga el Bug 1: ¿Por qué el XGBoost filtra de 2425 → 23 señales?
"""
import json
from pathlib import Path

models_dir = Path("g:/Mi unidad/ia/luna_v2/data/models")

def analyze_agent(name, path):
    sig = json.loads(path.read_text(encoding="utf-8"))
    thr = sig.get("optimal_threshold", "?")
    cal = sig.get("cal_source", "N/A")
    cal_report = sig.get("calibration_report", [])
    dsr = sig.get("dsr_oos", "N/A")
    brier = sig.get("xgb_brier_raw", "N/A")
    brier_gate = sig.get("brier_adaptive_gate", "N/A")

    print(f"\n{'='*55}")
    print(f"AGENTE: {name.upper()}")
    print(f"{'='*55}")
    print(f"  optimal_threshold : {thr}")
    print(f"  cal_source        : {cal}")
    print(f"  dsr_oos           : {dsr}")
    print(f"  brier_raw         : {brier}")
    print(f"  brier_gate        : {brier_gate}")

    if cal_report:
        ev_positivos = [r for r in cal_report if r.get("ev", -999) > 0]
        print(f"  calibration_report: {len(cal_report)} thresholds evaluados")
        print(f"  EV > 0 encontrados: {len(ev_positivos)}")
        if ev_positivos:
            best = max(ev_positivos, key=lambda x: x.get("ev", 0))
            worst_ev_pos = min(ev_positivos, key=lambda x: x.get("ev", 0))
            print(f"  Mejor EV positivo: thr={best['threshold']}  ev={best['ev']:.6f}  wr={best['wr']:.3f}  n={best['n_trades']}")
            print(f"  Peor EV positivo:  thr={worst_ev_pos['threshold']}  ev={worst_ev_pos['ev']:.6f}")
        else:
            print("  -> SIN EV POSITIVO en ningún threshold evaluado")
            print("  -> FIX-THRESH-01: threshold forzado a 0.95 (SILENCIADOR ACTIVO)")
            if cal_report:
                best_neg = max(cal_report, key=lambda x: x.get("ev", -999))
                print(f"  -> Mejor EV negativo fue: thr={best_neg['threshold']}  ev={best_neg['ev']:.6f}  wr={best_neg['wr']:.3f}  n={best_neg['n_trades']}")
    else:
        print(f"  calibration_report: VACIO")
        print(f"  -> threshold asignado por fallback directo: {thr}")

    # Identificar el período de calibración
    train_period = sig.get("training_period", sig.get("val_period", "N/A"))
    n_train = sig.get("n_train_samples", sig.get("n_samples_val", "N/A"))
    print(f"  n_samples_train   : {n_train}")
    print(f"  training_period   : {train_period}")


agents = [
    ("bull",  models_dir / "xgboost_meta_bull_long_signature.json"),
    ("range", models_dir / "xgboost_meta_range_long_signature.json"),
    ("bear",  models_dir / "xgboost_meta_bear_long_signature.json"),
]

print("AUDITORIA DE THRESHOLDS XGBoost — Bug 1")
print("Objetivo: entender por que 2425 barras OOS -> solo 23 señales XGBoost")

for name, path in agents:
    if path.exists():
        analyze_agent(name, path)
    else:
        print(f"\n[MISSING] {path.name} no encontrado")

# También leer el calibrador del MetaLabeler
print(f"\n{'='*55}")
print("CALIBRADOR MetaLabeler (Platt Scaling / Isotónico)")
print(f"{'='*55}")
cal_path = models_dir / "calibrator_long_signature.json"
if cal_path.exists():
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    print(f"  optimal_meta_threshold      : {cal.get('optimal_meta_threshold', '?')}")
    print(f"  optimal_meta_threshold_/reg : {cal.get('optimal_meta_threshold_per_regime', {})}")
    print(f"  brier_raw                   : {cal.get('brier_score_raw', '?'):.4f}")
    print(f"  brier_calibrado             : {cal.get('brier_score_calibrated', '?'):.4f}")
    print(f"  mejora_pct                  : {cal.get('mejora_pct', '?'):.1f}%")
    print(f"  n_samples_val               : {cal.get('n_samples_val', '?')}")

print()
print("RESUMEN CRITICO:")
print("  Thresholds efectivos en OOS (basados en firmas):")
print("  BULL  -> thr=? (verificar en signature)")
print("  RANGE -> thr=? (verificar en signature - posible FIX-THRESH-01)")
print("  BEAR  -> thr=0.65 (fallback_dynamic)")
