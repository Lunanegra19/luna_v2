import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")

# Resolve project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

def calculate_psi(expected, actual, num_buckets=10):
    expected = expected.dropna().values
    actual = actual.dropna().values
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    percentiles = np.linspace(0, 100, num_buckets + 1)
    try:
        buckets = np.unique(np.percentile(expected, percentiles))
        if len(buckets) < 2:
            return 0.0
    except Exception:
        return 0.0
    buckets[0] = -np.inf
    buckets[-1] = np.inf
    expected_counts = np.histogram(expected, bins=buckets)[0]
    actual_counts = np.histogram(actual, bins=buckets)[0]
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)
    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

def test_dynamic_drift_gate():
    print("[TEST-DYN-DRIFT] Iniciando simulación de OOD Sentinel Simétrico (Libre de Sesgo Muestral)...")
    
    # Analizar W3, W4 y W5
    for window in ["W3", "W4", "W5"]:
        print(f"\n=================== Window {window} ===================")
        feat_dir = ROOT / "data" / "wfb_cache" / window / "features"
        train_path = feat_dir / "features_train.parquet"
        val_path = feat_dir / "features_validation.parquet"
        holdout_path = feat_dir / f"features_holdout_{window}.parquet"
        sel_feats_path = feat_dir / "selected_features.json"
        
        if not (train_path.exists() and val_path.exists() and holdout_path.exists() and sel_feats_path.exists()):
            print(f"[TEST-DYN-DRIFT] SKIP: Archivos para ventana {window} no encontrados.")
            continue
            
        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path)
        df_holdout = pd.read_parquet(holdout_path)
        
        with open(sel_feats_path, "r", encoding="utf-8") as f:
            selected_features_data = json.load(f)
            
        if isinstance(selected_features_data, dict) and "selected_features" in selected_features_data:
            selected_features = selected_features_data["selected_features"]
        else:
            selected_features = selected_features_data
            
        # Para evitar el sesgo de tamaño muestral:
        # Extraemos una submuestra del final del dataset de entrenamiento que sea del MISMO tamaño que el holdout.
        n_holdout = len(df_holdout)
        
        # Muestra de referencia in-sample equivalente (últimas N barras del train set)
        ref_in_sample = df_train.tail(n_holdout)
        
        print(f"Comparando muestras simétricas de tamaño N = {n_holdout}")
        
        # 1. Medir Drift de Holdout (Inferencia actual) contra Referencia In-Sample Equivalente
        drifted_count = 0
        extreme_drifted_count = 0
        drift_scores = {}
        
        for feat in selected_features:
            if feat in ref_in_sample.columns and feat in df_holdout.columns:
                psi = calculate_psi(ref_in_sample[feat], df_holdout[feat])
                drift_scores[feat] = psi
                if psi > 0.25:
                    drifted_count += 1
                if psi > 1.0:
                    extreme_drifted_count += 1
                    
        total_feats = len(drift_scores)
        drift_ratio = drifted_count / max(1, total_feats)
        extreme_ratio = extreme_drifted_count / max(1, total_feats)
        
        # Mostrar las features con drift
        print("  Top drifted features en Holdout contra Referencia In-Sample:")
        sorted_drift = sorted(drift_scores.items(), key=lambda x: x[1], reverse=True)
        for feat, score in sorted_drift[:5]:
            print(f"    {feat}: PSI = {score:.4f}")
            
        print(f"  Ratio de Features Desviadas (PSI > 0.25): {drift_ratio:.1%}")
        
        # 2. Mitigación Dinámica adaptada al Drift Simétrico
        # - Si drift_ratio < 15%, operativa normal.
        # - Si drift_ratio >= 15% y < 40%, aplicamos una atenuación lineal a la fracción Kelly.
        # - Si drift_ratio >= 40%, emergencia (fuerte anomalía estructural).
        
        base_kelly_fraction = 0.25
        effective_kelly = base_kelly_fraction
        status = "NORMAL OPERATIONAL"
        
        if drift_ratio >= 0.40:
            effective_kelly = 0.0
            status = "🚨 EMERGENCY SHUTDOWN (Extreme Drift)"
        elif drift_ratio >= 0.15:
            factor = (0.40 - drift_ratio) / (0.40 - 0.15)
            effective_kelly = base_kelly_fraction * factor
            status = f"⚠️ DEGRADED RISK (Atenuación Kelly a {factor*100:.1f}%)"
            
        print(f"  Drift Status:      {status}")
        print(f"  Base Kelly Frac:   {base_kelly_fraction:.4f}")
        print(f"  Effective Kelly:   {effective_kelly:.4f} (-{(1.0 - effective_kelly/max(1e-5, base_kelly_fraction))*100:.1f}% capital allocation)")
        
        # Validación de protección
        if window == "W5":
            assert drift_ratio >= 0.15, "El Drift Sentinel no se activó en W5!"
            print("  [OK] Sentinel reaccionó correctamente en W5 ante exceso de drift.")
        elif window == "W3":
            print("  [OK] Validación en W3 completada.")
            
if __name__ == "__main__":
    test_dynamic_drift_gate()
