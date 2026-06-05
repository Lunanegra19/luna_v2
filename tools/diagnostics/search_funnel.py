import glob
import os
import json

def find_funnel():
    print("Searching for signal_funnel.json...")
    files = glob.glob("g:/Mi unidad/ia/luna_v2/**/signal_funnel*.json", recursive=True)
    for f in files:
        print(f"\nFound: {f} ({os.path.getsize(f)} bytes)")
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)
                print(json.dumps(data, indent=2)[:1000])
        except Exception as e:
            print(f"Error reading: {e}")

if __name__ == "__main__":
    find_funnel()
