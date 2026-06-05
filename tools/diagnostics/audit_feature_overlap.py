import os
import sys
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
wfb_out_dir = _ROOT / "data" / "reports" / "wfb"

def audit_features():
    print("="*90)
    print("      DIAGNOSTIC AUDIT: FEATURE OVERLAP AND VARIABILITY ANALYSIS      ")
    print("="*90)
    
    feature_files = list(wfb_out_dir.glob("selected_features_W*.json"))
    if not feature_files:
        print("ERROR: No selected_features_W*.json files found.")
        return
        
    all_features = {}
    for f in sorted(feature_files):
        print(f"\n=================== FEATURE SELECTION FOR {f.stem} ===================")
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                
            features = data.get('selected_features', [])
            passthrough = data.get('pass_through_features', [])
            total_features = list(set(features + passthrough))
            
            print(f"Total features: {len(total_features)}")
            print(f"SFI Selected:  {features}")
            print(f"Passthrough:   {passthrough}")
            
            all_features[f.stem] = total_features
        except Exception as e:
            print(f"Error parsing {f.name}: {e}")
            
    # Calculate overlap matrix between windows
    print("\n" + "="*90)
    print("      FEATURE OVERLAP MATRIX BETWEEN WINDOWS      ")
    print("="*90)
    windows = sorted(all_features.keys())
    for i, w1 in enumerate(windows):
        for j, w2 in enumerate(windows):
            if i <= j:
                set1 = set(all_features[w1])
                set2 = set(all_features[w2])
                overlap = len(set1.intersection(set2))
                pct = (overlap / len(set1)) * 100 if len(set1) > 0 else 0
                print(f"Overlap {w1:3s} vs {w2:3s} | Overlapping Features: {overlap:2d} | % of {w1}: {pct:5.1f}%")

if __name__ == "__main__":
    audit_features()
