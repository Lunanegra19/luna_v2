import os
import glob
import pandas as pd

def find_and_inspect():
    print("Searching for W4 oos_raw_probs parquet files...")
    search_path = "g:/Mi unidad/ia/luna_v2/**/oos_raw_probs*W4*.parquet"
    files = glob.glob(search_path, recursive=True)
    if not files:
        # try a broader search
        search_path_broad = "g:/Mi unidad/ia/luna_v2/**/oos_raw_probs*.parquet"
        files = glob.glob(search_path_broad, recursive=True)
        
    print(f"Found {len(files)} raw probs files:")
    for f in files:
        print(f" - {f} ({os.path.getsize(f)} bytes)")
        
    if files:
        # Load the most relevant one (preferably containing seed42 and W4)
        target_file = None
        for f in files:
            if "seed42" in f and "W4" in f:
                target_file = f
                break
        if not target_file:
            target_file = files[0]
            
        print(f"\nLoading {target_file}...")
        df = pd.read_parquet(target_file)
        print(f"Shape: {df.shape}")
        print("Columns:")
        print(list(df.columns))
        print("\nHead sample:")
        print(df.head())
        
        # If we have probability columns, let's check values
        prob_cols = [c for c in df.columns if "prob" in c or "pred" in c or "signal" in c]
        print(f"\nProbability columns: {prob_cols}")
        for c in prob_cols:
            print(f" - {c}: min={df[c].min()}, max={df[c].max()}, mean={df[c].mean()}")

if __name__ == "__main__":
    find_and_inspect()
