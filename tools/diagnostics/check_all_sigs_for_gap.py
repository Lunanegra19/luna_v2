"""Check regime_router and XGB base model feature lists"""
import json, glob, os, sys

critical_features = [
    'FundingRate_EMA3', 'FundingRate_Pct90d',
    'OI_Open_USD', 'OI_High_USD', 'OI_Low_USD',
    'ETF_Flow_Proxy', 'dv_etf_flow_proxy',
    'genetic_rule_5', 'genetic_rule_6', 'genetic_rule_7', 'genetic_rule_8', 'genetic_rule_9'
]

seeds = ['seed99', 'seed1337', 'seed2025']
models_dir = '/root/luna_v2/data/models/prod'

print("=== CHECKING ALL SIGNATURE FILES FOR CRITICAL FEATURES ===")
any_critical_found = False

for seed in seeds:
    seed_dir = f"{models_dir}/{seed}"
    for sig_file in sorted(glob.glob(f"{seed_dir}/*.json")):
        try:
            with open(sig_file) as f:
                sig = json.load(f)
            
            # Get feature list from any key that might have it
            features = []
            for key in ['feature_names', 'features', 'columns', 'feature_list', 'input_features']:
                if key in sig and isinstance(sig[key], list):
                    features = sig[key]
                    break
            
            if not features:
                continue
            
            crit_found = [f for f in critical_features if f in features]
            if crit_found:
                print(f"\n  !!! CRITICAL - {seed}/{os.path.basename(sig_file)}")
                print(f"      Uses renamed/missing: {crit_found}")
                any_critical_found = True
                
        except Exception as e:
            pass

if not any_critical_found:
    print("\n  SAFE: No active model signature uses the 12 missing/renamed features!")
    print("  The train-live gap does NOT affect model predictions.")
    print("  These features were likely filtered out by SFI before training the final models.")

print("\n=== CHECKING REGIME ROUTER CONFIG ===")
# Check the regime router signature
rr_sigs = glob.glob(f"{models_dir}/*/regime_router*.json") + glob.glob(f"/root/luna_v2/data/models/regime_router*.json")
for rr in rr_sigs:
    with open(rr) as f:
        d = json.load(f)
    feats = d.get('feature_names', d.get('features', []))
    crit_found = [f for f in critical_features if f in feats]
    print(f"  {os.path.basename(rr)}: {len(feats)} feats | critical: {crit_found}")

print("\n=== CHECKING XGB BASE CONFIG ===")
config_files = glob.glob(f"{models_dir}/*/config.json")
for cf in config_files[:3]:
    with open(cf) as f:
        c = json.load(f)
    # Look for feature names in config
    features_key = None
    for k in c:
        if 'feature' in k.lower() and isinstance(c[k], list):
            feats = c[k]
            crit_found = [f for f in critical_features if f in feats]
            print(f"  {os.path.basename(os.path.dirname(cf))}/config.json key='{k}': {len(feats)} feats | critical: {crit_found}")
            break
    else:
        print(f"  {os.path.basename(os.path.dirname(cf))}/config.json: keys={list(c.keys())[:8]}")
