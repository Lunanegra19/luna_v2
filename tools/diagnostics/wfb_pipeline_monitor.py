import os
import sys
import time
import re
from pathlib import Path
import psutil

# Ensure stdout is in UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"

def get_active_processes():
    orchestrators = []
    workers = []
    sfi_rankers = []
    other_py = []
    
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = p.info.get('cmdline') or []
            cmd_str = ' '.join(cmdline).lower()
            if 'python' in p.info.get('name', '').lower() or any('python' in cmd.lower() for cmd in cmdline):
                pid = p.info['pid']
                if 'run_wfb_orchestrator.py' in cmd_str:
                    orchestrators.append((pid, cmdline))
                elif 'wfb_worker.py' in cmd_str:
                    workers.append((pid, cmdline))
                elif 'feature_selection_e.py' in cmd_str:
                    sfi_rankers.append((pid, cmdline))
                else:
                    other_py.append((pid, cmd_str))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    return orchestrators, workers, sfi_rankers, other_py

def get_latest_log_file(prefix: str) -> tuple[Path | None, str]:
    if not LOGS_DIR.exists():
        return None, "Logs directory not found"
    
    log_files = list(LOGS_DIR.glob(f"{prefix}*.log"))
    if not log_files:
        return None, f"No logs starting with '{prefix}' found"
        
    latest_file = max(log_files, key=lambda f: f.stat().st_mtime)
    mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_file.stat().st_mtime))
    return latest_file, mtime_str

def parse_wfb_worker_log(path: Path) -> dict:
    info = {
        "seed": "Unknown",
        "window": "Unknown",
        "active_phase": "Unknown",
        "last_lines": [],
        "errors": [],
        "gates": []
    }
    
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            
        # Parse last 12 lines for context
        info["last_lines"] = [line.strip() for line in lines[-12:]]
        
        # Scan for key phrases
        for line in reversed(lines):
            line_str = line.strip()
            # Seed and window identification
            if "LUNA V2 - SECUENCIA DE ENTRENAMIENTO" in line_str:
                pass
            if "Semilla:" in line_str and "Ventana:" in line_str:
                match = re.search(r"Semilla:\s*(\d+)\s*\|\s*Ventana:\s*(W\d+)", line_str)
                if match:
                    if info["seed"] == "Unknown":
                        info["seed"] = match.group(1)
                    if info["window"] == "Unknown":
                        info["window"] = match.group(2)
            if "--- INICIANDO CICLO VENTANA:" in line_str:
                match = re.search(r"CICLO VENTANA:\s*(W\d+)", line_str)
                if match and info["window"] == "Unknown":
                    info["window"] = match.group(1)
            
            # Active phase identification
            if "--- Iniciando Fase:" in line_str or "--- Iniciando Fase Compartida:" in line_str:
                match = re.search(r"Fase(?:\s+Compartida)?:\s*([^-]+)", line_str)
                if match and info["active_phase"] == "Unknown":
                    info["active_phase"] = match.group(1).strip()
            
            # Phase gates audits
            if "[GATE-" in line_str:
                info["gates"].append(line_str)
                
            # Errors/Exceptions
            if "ERROR" in line_str or "CRITICAL" in line_str or "Traceback" in line_str:
                info["errors"].append(line_str)
                
    except Exception as e:
        info["errors"].append(f"Error reading worker log: {e}")
        
    return info

def parse_sfi_log(path: Path) -> dict:
    info = {
        "features_total": 0,
        "features_done": 0,
        "active_feature": "Unknown",
        "last_log_time": "Unknown",
        "last_completed": [],
        "errors": []
    }
    
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            
        for line in lines:
            line_str = line.strip()
            
            # Find total features in parallel SFI run
            if "Iniciando procesamiento PARALELO de" in line_str:
                match = re.search(r"PARALELO de\s*(\d+)\s*features", line_str)
                if match:
                    info["features_total"] = int(match.group(1))
            if "ADVERTENCIA: tiempo estimado en CPU: ~8 min (" in line_str:
                match = re.search(r"\((\d+)\s*candidatos", line_str)
                if match and info["features_total"] == 0:
                    info["features_total"] = int(match.group(1))
            
            # Find progress
            if "_thread_worker" in line_str and "[OK]" in line_str:
                match = re.search(r"\[\s*(\d+)/(\d+)\]\s*([a-zA-Z0-9_]+)", line_str)
                if match:
                    done_idx = int(match.group(1))
                    tot_idx = int(match.group(2))
                    feat_name = match.group(3)
                    
                    info["features_done"] = max(info["features_done"], done_idx)
                    if tot_idx > info["features_total"]:
                        info["features_total"] = tot_idx
                        
                    # Save last completed features
                    info["last_completed"].append(f"#{done_idx}/{tot_idx}: {feat_name}")
            
            # Errors/Exceptions
            if "ERROR" in line_str or "CRITICAL" in line_str or "Traceback" in line_str:
                info["errors"].append(line_str)
                
        # Parse last lines for active evaluations
        for line in reversed(lines[-15:]):
            line_str = line.strip()
            if "_eval_temporal_stability" in line_str:
                match = re.search(r"stability for feature:\s*([a-zA-Z0-9_]+)", line_str)
                if match:
                    info["active_feature"] = match.group(1)
                    break
                
        if info["last_completed"]:
            info["last_completed"] = info["last_completed"][-5:] # last 5 only
            
        if lines:
            # Extract timestamp from last line
            match = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", lines[-1])
            if match:
                info["last_log_time"] = match.group(1)
                
    except Exception as e:
        info["errors"].append(f"Error reading SFI log: {e}")
        
    return info

def print_dashboard():
    print("=" * 80)
    print("           LUNA V2 - WALK-FORWARD BACKTESTING PIPELINE MONITOR")
    print("=" * 80)
    
    # 1. Processes Audit
    print("\n--- ACTIVE ENVIRONMENT PROCESSES ---")
    orchs, wrks, sfis, others = get_active_processes()
    
    lock_file = PROJECT_ROOT / ".wfb_lock"
    lock_held = lock_file.exists()
    lock_val = ""
    if lock_held:
        try:
            lock_val = f" (Held by PID: {lock_file.read_text().strip()})"
        except Exception:
            pass
            
    print(f"Active WFB Lock  : {lock_held}{lock_val}")
    print(f"Orchestrator runs: {len(orchs)} active")
    for pid, cmd in orchs:
        print(f"  - PID {pid}: {' '.join(cmd)[:70]}...")
        
    print(f"Worker runs      : {len(wrks)} active")
    for pid, cmd in wrks:
        print(f"  - PID {pid}: {' '.join(cmd)[:70]}...")
        
    print(f"SFI Rankers      : {len(sfis)} active")
    for pid, cmd in sfis:
        print(f"  - PID {pid}: {' '.join(cmd)[:70]}...")
        
    if not orchs and not wrks:
        print(">> WARNING: No active WFB orchestrator or worker python process found!")
        
    # 2. Worker Logs Progress
    print("\n--- PIPELINE WORKER PROGRESS ---")
    latest_worker, worker_time = get_latest_log_file("wfb_worker_")
    if latest_worker:
        print(f"Latest Worker Log: {latest_worker.name} (Last modified: {worker_time})")
        w_info = parse_wfb_worker_log(latest_worker)
        print(f"Active Seed      : {w_info['seed']}")
        print(f"Active Window    : {w_info['window']}")
        print(f"Active Stage     : {w_info['active_phase']}")
        
        # SFI Details
        if "SFI Feature Selection" in w_info["active_phase"] or sfis:
            latest_sfi, sfi_time = get_latest_log_file("feature_selection_")
            if latest_sfi:
                s_info = parse_sfi_log(latest_sfi)
                print(f"  └ SFI Log      : {latest_sfi.name} (Last modified: {sfi_time})")
                print(f"  └ SFI Candidates: {s_info['features_total']}")
                print(f"  └ Completed     : {s_info['features_done']} / {s_info['features_total']}")
                if s_info['features_total'] > 0:
                    pct = s_info['features_done'] / s_info['features_total'] * 100
                    print(f"  └ SFI Progress  : {pct:.1f}%")
                if s_info['last_completed']:
                    print(f"  └ Last Done     : {', '.join(s_info['last_completed'])}")
                print(f"  └ Last Log time : {s_info['last_log_time']}")
                
        print("\n--- PHASE GATES AUDIT (GATE-G0 to G5) ---")
        if w_info["gates"]:
            for gate in w_info["gates"][-6:]:
                print(f"  {gate}")
        else:
            print("  No phase gates recorded in current worker session log yet.")
            
        print("\n--- LATEST LOG TRACES ---")
        for line in w_info["last_lines"]:
            print(f"  {line}")
            
        if w_info["errors"]:
            print("\n--- WORKER ALERTS / EXCEPTIONS ---")
            for err in w_info["errors"][-5:]:
                print(f"  [ALERT] {err}")
    else:
        print("No active worker logs found.")
        
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_dashboard()
