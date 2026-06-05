import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
hmm_path = root / "luna" / "models" / "hmm_regime.py"
dump_path = root / "tools" / "dumps" / "hmm_inspect.txt"

print(f"Reading {hmm_path}...")
content = hmm_path.read_text(encoding="utf-8", errors="replace")
lines = content.splitlines()

target = "def enrich_validation_and_holdout"
found = False
output_lines = []

for idx, line in enumerate(lines):
    if target in line:
        found = True
        output_lines.append(f"Found '{target}' on line {idx + 1}:")
        start = max(0, idx - 5)
        end = min(len(lines), idx + 120)
        for i in range(start, end):
            output_lines.append(f"{i+1:4d}: {lines[i]}")
        output_lines.append("-" * 50)

if not found:
    # Buscar ocurrencia parcial
    target_partial = "enrich_validation_and_holdout"
    for idx, line in enumerate(lines):
        if target_partial in line:
            found = True
            output_lines.append(f"Found partial match '{target_partial}' on line {idx + 1}:")
            start = max(0, idx - 5)
            end = min(len(lines), idx + 120)
            for i in range(start, end):
                output_lines.append(f"{i+1:4d}: {lines[i]}")
            output_lines.append("-" * 50)

if not found:
    output_lines.append(f"Substring '{target}' not found in {hmm_path}")

dump_path.parent.mkdir(parents=True, exist_ok=True)
dump_path.write_text("\n".join(output_lines), encoding="utf-8", errors="replace")
print(f"Dumped results to {dump_path}")
