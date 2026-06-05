from pathlib import Path

log_path = Path(r"C:\Users\Usuario\.gemini\antigravity-ide\brain\fd276a25-382c-463b-a633-6e14e3db0da1\.system_generated\tasks\task-1162.log")
dump_path = Path(r"g:\Mi unidad\ia\luna_v2\tools\dumps\task_1162_tail.txt")

if log_path.exists():
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-300:]
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text("\n".join(tail), encoding="utf-8")
    print(f"Dumped {len(tail)} lines to {dump_path}")
else:
    print(f"Log path does not exist: {log_path}")
