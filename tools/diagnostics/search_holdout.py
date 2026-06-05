import glob
import os

def find_holdout():
    print("Searching for holdout and features_train files...")
    patterns = [
        "g:/Mi unidad/ia/luna_v2/**/features_holdout*.parquet",
        "g:/Mi unidad/ia/luna_v2/**/features_validation*.parquet",
        "g:/Mi unidad/ia/luna_v2/**/ohlcv*.parquet"
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        print(f"\nPattern: {pattern}")
        for f in files:
            print(f" - {f} ({os.path.getsize(f)} bytes)")

if __name__ == "__main__":
    find_holdout()
