import glob
from pathlib import Path

def search_text():
    print("Searching for '33' or 'W4' in all log files...")
    log_files = glob.glob("g:/Mi unidad/ia/luna_v2/logs/*.log")
    for f in log_files:
        path = Path(f)
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            if "33" in content:
                for line in content.splitlines():
                    if "33" in line and ("W4" in line or "signal" in line or "trade" in line or "block" in line or "meta" in line):
                        print(f"Found in {path.name}: {line}")
        except Exception as e:
            pass

    # Also check brain logs if possible
    print("\nSearching in brain/77197b66-ef1d-4d80-9e47-a78575aa4e5e/ logs...")
    brain_logs = glob.glob("C:/Users/Usuario/.gemini/antigravity-ide/brain/77197b66-ef1d-4d80-9e47-a78575aa4e5e/**/*", recursive=True)
    for f in brain_logs:
        path = Path(f)
        if path.is_file() and path.suffix in ('.log', '.txt', '.md', '.json'):
            try:
                content = path.read_text(encoding='utf-8', errors='ignore')
                if "33" in content:
                    for line in content.splitlines():
                        if "33" in line and ("W4" in line or "signal" in line or "trade" in line or "block" in line or "meta" in line):
                            print(f"Found in brain/{path.relative_to('C:/Users/Usuario/.gemini/antigravity-ide/brain/77197b66-ef1d-4d80-9e47-a78575aa4e5e')}: {line}")
            except Exception as e:
                pass

if __name__ == "__main__":
    search_text()
