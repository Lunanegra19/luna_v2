import re
import sys
from pathlib import Path

# Configure UTF-8 for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

p = Path("g:/Mi unidad/ia/luna_v2/luna/models/predict_oos.py")
content = p.read_text(encoding="utf-8")

# Search for class definitions
print("=== CLASES ===")
for m in re.finditer(r"class \w+.*:", content):
    print(m.group())

# Search for function/method definitions
print("\n=== FUNCIONES ===")
for m in re.finditer(r"def (\w+)\(self", content):
    line = content[:m.start()].count("\n") + 1
    print(f"L{line}: {m.group(1)}")

# Search for where signal filter or filtering is mentioned
print("\n=== MATCHES FOR 'filter' / 'signal' / 'meta' ===")
for i, line in enumerate(content.splitlines()):
    if any(k in line.lower() for k in ["filter", "signal_filter", "meta_v2", "apply_"]):
        # Strip accents or non-ascii to avoid win32 encoding issues
        clean_line = line.encode("ascii", "replace").decode("ascii")
        print(f"L{i+1}: {clean_line}")
