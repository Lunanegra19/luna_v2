"""Audit features_live.parquet HMM columns"""
import pandas as pd
import sys

p = '/root/luna_v2/data/features/features_live.parquet'
try:
    df = pd.read_parquet(p)
    print(f"Shape: {df.shape}")
    print(f"Last index: {df.index[-1]}")
    print(f"Total cols: {len(df.columns)}")
    
    hmm_cols = [c for c in df.columns if 'hmm' in c.lower() or 'HMM' in c]
    print(f"\nHMM columns found ({len(hmm_cols)}): {hmm_cols}")
    
    # Check which dashboard-expected HMM features exist
    expected = ['hmm_regime_label', 'hmm_prob_bull', 'hmm_prob_bear', 
                'hmm_prob_volatile', 'hmm_state_duration_h']
    print("\nDashboard expected vs actual:")
    for col in expected:
        exists = col in df.columns
        print(f"  {'OK' if exists else 'MISSING'}: {col}")
    
    # Also check HMM_Regime (what pipeline actually produces)
    if 'HMM_Regime' in df.columns:
        print(f"\nHMM_Regime found: {df['HMM_Regime'].value_counts().head()}")
        print(f"HMM_Regime NaN: {df['HMM_Regime'].isna().sum()} / {len(df)}")
    
    # Show all columns with hmm for context
    print(f"\nAll HMM-related cols: {hmm_cols}")
    
except FileNotFoundError:
    print(f"ERROR: {p} no existe!", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
