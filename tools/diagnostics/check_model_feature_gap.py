"""Check if XGB/Meta models use the renamed/missing features"""
import json, glob, os

models_dir = '/root/luna_v2/data/models'
critical_features = [
    'FundingRate_EMA3', 'FundingRate_Pct90d',
    'OI_Open_USD', 'OI_High_USD', 'OI_Low_USD',
    'ETF_Flow_Proxy', 'dv_etf_flow_proxy',
    'genetic_rule_5', 'genetic_rule_6', 'genetic_rule_7', 'genetic_rule_8', 'genetic_rule_9'
]

# Check XGB active seeds (99, 1337, 2025)
active_seeds = ['seed99', 'seed1337', 'seed2025']
print("=== XGB MODEL FEATURE CHECK (Active Seeds) ===")

for seed in active_seeds:
    seed_dir = f"{models_dir}/prod/{seed}"
    # Check signature files
    sigs = glob.glob(f"{seed_dir}/*signature*.json")
    for sig_path in sigs:
        try:
            with open(sig_path) as f:
                sig = json.load(f)
            features = sig.get('feature_names', sig.get('features', sig.get('columns', [])))
            if not features:
                continue
            missing_crit = [f for f in critical_features if f in features]
            if missing_crit:
                print(f"\n  CRITICAL - {seed}/{os.path.basename(sig_path)}")
                print(f"    Uses renamed/missing features: {missing_crit}")
            else:
                print(f"  OK - {seed}/{os.path.basename(sig_path)} ({len(features)} features, none critical)")
        except Exception as e:
            print(f"  ERR {sig_path}: {e}")

# Also check the xgb model files directly
print("\n=== XGB BOOSTER FILES ===")
xgb_files = glob.glob(f"{models_dir}/prod/*/xgboost*.json")
for xf in xgb_files[:6]:
    try:
        with open(xf) as f:
            content = f.read(50000)  # Read first 50KB
        # Look for feature names in XGBoost JSON format
        if '"feature_names"' in content:
            import json as _json
            data = _json.loads(content)
            learner = data.get('learner', {})
            feat_names = learner.get('feature_names', [])
            missing_crit = [f for f in critical_features if f in feat_names]
            print(f"  {os.path.basename(xf)} seed {os.path.basename(os.path.dirname(xf))}: {len(feat_names)} feats | missing critical: {missing_crit}")
        else:
            print(f"  {os.path.basename(xf)}: no feature_names found in JSON")
    except Exception as e:
        print(f"  ERR {xf}: {e}")
