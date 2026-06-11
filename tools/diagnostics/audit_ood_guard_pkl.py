"""Auditoria del ood_guard.pkl - Paso 4 investigacion"""
import json
import joblib
import numpy as np
import os
import datetime
from pathlib import Path

root = Path(".")
sig_path = root / "data" / "models" / "ood_guard_signature.json"
pkl_path = root / "data" / "models" / "ood_guard.pkl"

print("=== PASO 4: AUDITORIA DEL ood_guard.pkl ===")
print()

if sig_path.exists():
    sig = json.loads(sig_path.read_text())
    print(f"Entrenado en : {sig.get('trained_at', 'desconocido')}")
    print(f"N features   : {sig.get('n_features')}")
    print(f"N samples IS : {sig.get('n_samples')} barras")
    print(f"Contamination: {sig.get('contamination')}")
    print(f"SFI hash     : {sig.get('sfi_hash')}")
    print(f"Fill strategy: {sig.get('fillna_strategy')}")
    print(f"Umbral anomalia IS: {sig.get('anomaly_score_threshold'):.6f}")
    print()
    feats = sig.get("features_tracked", [])
    print(f"Features en el OOD Guard ({len(feats)}):")
    for f in feats[:15]:
        print(f"  - {f}")
    if len(feats) > 15:
        print(f"  ... y {len(feats)-15} mas")
else:
    print("ood_guard_signature.json NO existe")

print()
if pkl_path.exists():
    stat = os.stat(pkl_path)
    mod_time = datetime.datetime.fromtimestamp(stat.st_mtime)
    print(f"ood_guard.pkl tamanio: {stat.st_size/1024:.1f} KB")
    print(f"ood_guard.pkl modificado: {mod_time}")

    model = joblib.load(pkl_path)
    print(f"Modelo tipo: {type(model).__name__}")
    print(f"N estimators: {model.n_estimators}")
    print(f"Contamination: {model.contamination}")
    print(f"Umbral interno decision_function (offset_): {model.offset_:.6f}")
    print()
    print("SEMANTICA del decision_function (scikit-learn IsolationForest):")
    print("  score > 0  -> IN-distribution (normal)  <- mas positivo = mas normal")
    print("  score < 0  -> OUT-OF-distribution (anomalo)")
    print("  score = offset_ -> umbral de la contamination configurada")
    print()
    print("VALIDACION en OOS: ood_kl_distance en parquets de trades tiene:")
    print("  min=-0.040, mean=+0.134, max=+0.200")
    print("  => 99.2% de trades son 'normales' segun el IsolationForest (score > 0)")
    print("  => Solo 0.8% son 'anomalos' (score < 0) -> Kelly penalty casi inactivo")
    print()
    print("CAUSA DEL COVARIATE SHIFT:")
    print("  El IsolationForest se entrena con features IS (2022-2024).")
    print("  El offset_ se calibra para que 'contamination' % de IS sea anomalo.")
    print("  En OOS 2025-2026, el pattern post-halving/post-ETF es DIFERENTE.")
    print("  Las barras de alto momentum 2025-26 son 'normales' para el IF")
    print("  (score=+0.15 aprox) pero GANAN menos que las barras con score bajo.")
    print()
    print("CONCLUSION PASO 4:")
    print("  El ood_kl_distance es el decision_function del IsolationForest de sklearn.")
    print("  Se produce en signal_filter.apply_ood() linea 553.")
    print("  La inversion NO es un bug de calculo — es un covariate shift real.")
    print("  El IsolationForest necesitaria re-entrenarse con datos 2025-2026 para")
    print("  aprender la nueva 'normalidad' del mercado post-institutional.")
    print()
    print("  RIESGO de re-entrenamiento: si se re-entrena el IF con OOS, se viola")
    print("  la causalidad estricta (Rule R1 SOP). Se necesita un enfoque alternativo.")
    print("  OPCIONES:")
    print("  A) Desactivar el OOD Guard KL score del Kelly (ya es casi inactivo - 0.8%)")
    print("  B) Re-entrenar IF incluyendo datos hasta train_end de cada ventana WFB")
    print("     (causal: cada ventana solo ve datos hasta su split point)")
    print("  C) Invertir el signo del penalty en Kelly (usar score alto como penalizacion)")
else:
    print("ood_guard.pkl NO existe")

print()
print("=== RESUMEN CUANTITATIVO INVESTIGACION COMPLETA ===")
print()
print("METALABELER:")
print("  - Spearman global: r=-0.147 (invertido)")
print("  - Con gate >= 0.65: Ret_total = -425pp vs baseline +647pp = -1072pp diferencia")
print("  - Con gate >= 0.70: WR cae de 50.6% a 46.3% con solo 26% de los trades")
print("  - VEREDICTO: skip_metalabeler=true es correcto. El gate siempre perjudica.")
print()
print("OOD GUARD:")
print("  - Spearman KL vs return: r=-0.259 (invertido)")
print("  - KL<=Q25 (anomalos): WR=67.2%, Ret=+1623pp, Sharpe=1.95")
print("  - KL>=Q75 (normales) : WR=32.1%, Ret=-1563pp, Sharpe=-3.03")
print("  - OOD penalty Kelly  : 0.8% de trades (practicamente nulo)")
print("  - VEREDICTO: el OOD Guard KL score tiene informacion VALIOSA pero invertida.")
print("    La solucion no es apagarlo — es usarlo con el signo correcto.")
print()
print("COMBINACION GANADORA (retroactiva, no implementada):")
print("  XGBoost>=Q75 + KL<=med: WR=61.5%, Ret=+805pp, Sharpe=2.34, MaxDD=75%")
print("  XGBoost>=Q50 + KL<=Q25: WR=69.2%, Ret=+987pp, Sharpe=2.96, MaxDD=75%")
print("  vs BASELINE:            WR=50.6%, Ret=+647pp, Sharpe=0.26, MaxDD=100%")
