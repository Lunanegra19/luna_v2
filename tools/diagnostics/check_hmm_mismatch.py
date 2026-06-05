import joblib
import json
from pathlib import Path

seeds = [99, 1337, 2025]
root = Path('/root/luna_v2')
print("=" * 60)
print("[CHECK-HMM-MISMATCH] Inspeccionando alineacion HMM vs MetaLabeler")
print("=" * 60)

for seed in seeds:
    rf_path = root / f'data/models/prod/seed{seed}/metalabeler_v2_long_rf.joblib'
    cfg_path = root / f'data/models/prod/seed{seed}/metalabeler_v2_long_config.json'
    hmm_path = root / f'data/models/prod/seed{seed}/hmm_model_config.json'
    
    if not rf_path.exists():
        print(f"Seed {seed}: NO EXISTE RF en {rf_path}")
        continue
    
    rf = joblib.load(rf_path)
    n_features = rf.n_features_in_
    
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    
    hmm_n_states_cfg = cfg.get('hmm_n_states', 'MISSING')
    
    # Calcular n_hmm_states segun la formula del ensemble_live_inference.py
    # n_features_in = rolling_stats(n_feat*3) + xgb(1) + hmm_states
    # => hmm_states = n_features_in - n_feat*3 - 1
    # pero necesitamos conocer n_feat para calcular n_feat*3
    # El campo seq_features en el config nos da n_feat
    seq_features = cfg.get('seq_features', [])
    n_seq_feat = len(seq_features)
    if n_seq_feat > 0:
        expected_hmm_states = n_features - (n_seq_feat * 3) - 1
    else:
        expected_hmm_states = "DESCONOCIDO (sin seq_features en config)"
    
    print(f"\nSeed {seed}:")
    print(f"  RF n_features_in_  = {n_features}")
    print(f"  hmm_n_states_cfg   = {hmm_n_states_cfg} (guardado en el momento del entrenamiento)")
    print(f"  n_seq_features     = {n_seq_feat}")
    print(f"  hmm_states_deducido= {expected_hmm_states} (calculado: {n_features} - {n_seq_feat}*3 - 1)")
    
    if isinstance(expected_hmm_states, int):
        status = "OK" if expected_hmm_states == hmm_n_states_cfg else "MISMATCH"
        print(f"  Estado             = {status}")
    
    # Inspeccionar cuantos estados tiene el HMM live cargado
    hmm_cfg_path = root / f'data/models/prod/seed{seed}/hmm_model_config.json'
    if hmm_cfg_path.exists():
        hmm_cfg = json.loads(hmm_cfg_path.read_text())
        hmm_n_states_live = hmm_cfg.get('n_states', 'MISSING')
        print(f"  HMM n_states_live  = {hmm_n_states_live}")
        if isinstance(expected_hmm_states, int) and isinstance(hmm_n_states_live, int):
            if hmm_n_states_live != expected_hmm_states:
                print(f"  *** DISCREPANCIA: HMM live tiene {hmm_n_states_live} estados pero MetaLabeler espera {expected_hmm_states} ***")
    else:
        print(f"  hmm_model_config.json no encontrado para seed {seed}")

print("\n[CHECK-HMM-MISMATCH] Completado.")
