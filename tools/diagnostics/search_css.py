import os
import re

css_path = r"g:\Mi unidad\ia\luna_v2\dashboard\index.css"
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    print("Matches for 'seed':")
    for m in re.finditer(r"\.[a-zA-Z0-9_-]*seed[a-zA-Z0-9_-]*", content):
        print(m.group(0))
        
    print("\nMatches for 'badge':")
    for m in re.finditer(r"\.[a-zA-Z0-9_-]*badge[a-zA-Z0-9_-]*", content):
        print(m.group(0))
else:
    print("File not found")
