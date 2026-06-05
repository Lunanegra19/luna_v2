"""Audit HMM pkl and XGB feature set for aspirational feature mismatch"""
import joblib
import sys

# Check HMM pkl
hmm_path = '/root/luna_v2/data/models/hmm_regime.pkl'
try:
    data = joblib.load(hmm_path)
    print("=== HMM PKL ===")
    print(f"Keys: {list(data.keys())}")
    print(f"State map: {data.get('state_map', {})}")
    features = data.get('features', [])
    print(f"HMM input features ({len(features)}): {features[:10]}...")
    hmm_prob_feats = [f for f in features if 'prob' in f or 'duration' in f or 'label' in f]
    print(f"Prob/duration/label in HMM features: {hmm_prob_feats}")
except Exception as e:
    print(f"HMM PKL error: {e}")

# Check XGB models for aspirational features
import glob, os
xgb_models = glob.glob('/root/luna_v2/data/models/*/xgb_model_seed*.pkl')
aspirational = ['hmm_regime_label', 'hmm_prob_bull', 'hmm_prob_bear', 
                'hmm_prob_volatile', 'hmm_state_duration_h']

print("\n=== XGB MODEL FEATURE CHECK ===")
for mp in xgb_models[:3]:
    try:
        xgb_data = joblib.load(mp)
        feat_names = []
        if hasattr(xgb_data, 'feature_names_in_'):
            feat_names = list(xgb_data.feature_names_in_)
        elif isinstance(xgb_data, dict):
            model = xgb_data.get('model', xgb_data)
            if hasattr(model, 'feature_names_in_'):
                feat_names = list(model.feature_names_in_)
        
        found_asp = [f for f in aspirational if f in feat_names]
        print(f"  {os.path.basename(mp)}: {len(feat_names)} features | Aspirational found: {found_asp}")
    except Exception as e:
        print(f"  {os.path.basename(mp)}: ERROR {e}")

# Check SFI/features_canonical
try:
    import pandas as pd
    sfi = pd.read_parquet('/root/luna_v2/data/features/features_train.parquet')
    found_in_train = [f for f in aspirational if f in sfi.columns]
    print(f"\nAspirational in features_train.parquet: {found_in_train}")
    hmm_in_train = [c for c in sfi.columns if 'hmm' in c.lower() or 'HMM' in c]
    print(f"HMM cols in train parquet: {hmm_in_train}")
except Exception as e:
    print(f"features_train check error: {e}")
