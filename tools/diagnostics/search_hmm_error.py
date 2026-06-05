from pathlib import Path

log_path = Path(r"C:\Users\Usuario\.gemini\antigravity-ide\brain\fd276a25-382c-463b-a633-6e14e3db0da1\.system_generated\tasks\task-1162.log")
dump_path = Path(r"g:\Mi unidad\ia\luna_v2\tools\dumps\hmm_error_context.txt")

if log_path.exists():
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    found = False
    output_lines = []
    
    for idx, line in enumerate(lines):
        if "hmm_train" in line or "Parquet magic bytes" in line or "Could not open Parquet" in line:
            found = True
            output_lines.append(f"Match on line {idx + 1}:")
            start = max(0, idx - 15)
            end = min(len(lines), idx + 25)
            for i in range(start, end):
                output_lines.append(f"{i+1:5d}: {lines[i]}")
            output_lines.append("=" * 60)
            
    if found:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text("\n".join(output_lines), encoding="utf-8")
        print(f"Dumped matches to {dump_path}")
    else:
        print("No HMM matches found in the log.")
else:
    print(f"Log path does not exist: {log_path}")
