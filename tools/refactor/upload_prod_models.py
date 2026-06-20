import os
import sys
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime

# Remote connection details
VPS_IP = "178.105.197.191"
VPS_USER = "root"
VPS_ROOT = "/root/luna_v2"
VPS_PYTHON = "/root/miniconda3/envs/luna_env/bin/python"

_ROOT = Path(__file__).resolve().parent.parent.parent

def run_local_cmd(cmd):
    print(f"[LOCAL RUN] {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[LOCAL ERROR] code {res.returncode}")
        print(res.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return res.stdout

def run_remote_cmd(cmd_str):
    print(f"[REMOTE RUN] {cmd_str}")
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        f"{VPS_USER}@{VPS_IP}",
        cmd_str
    ]
    res = subprocess.run(ssh_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"[REMOTE ERROR] code {res.returncode}")
        print(res.stderr)
        raise RuntimeError(f"Remote command failed: {cmd_str}")
    return res.stdout

def zip_models(zip_path):
    print(f"Creating local models zip at: {zip_path}")
    models_prod_dir = _ROOT / "data" / "models" / "prod"
    if not models_prod_dir.exists():
        raise FileNotFoundError(f"Local models production directory does not exist: {models_prod_dir}")
        
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(models_prod_dir):
            for file in files:
                file_path = Path(root) / file
                # Save inside the 'prod' folder in the zip
                rel_path = Path("prod") / file_path.relative_to(models_prod_dir)
                zipf.write(file_path, rel_path)
                
    print(f"Zip created successfully. File size: {zip_path.stat().st_size / (1024*1024):.2f} MB")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"luna_models_{timestamp}.zip"
    local_zip_path = _ROOT / "scratch" / zip_name
    local_zip_path.parent.mkdir(exist_ok=True)
    
    # 1. Zip local models
    try:
        zip_models(local_zip_path)
    except Exception as e:
        print(f"Error zipping models: {e}")
        print("Please ensure models are trained and present in data/models/prod/.")
        sys.exit(1)
        
    # 2. SCP zip file to VPS data/models/ directory
    print("\n--- STAGE 1: Uploading models zip to VPS ---")
    run_remote_cmd(f"mkdir -p {VPS_ROOT}/data/models")
    scp_cmd = [
        "scp", "-o", "StrictHostKeyChecking=no",
        str(local_zip_path),
        f"{VPS_USER}@{VPS_IP}:{VPS_ROOT}/data/models/{zip_name}"
    ]
    run_local_cmd(scp_cmd)
    
    # 3. Create remote backup of old models
    print("\n--- STAGE 2: Creating remote backup of old models ---")
    backup_dir = f"/root/luna_v2_backups/models_backup_{timestamp}"
    run_remote_cmd(f"mkdir -p {backup_dir}")
    run_remote_cmd(f"[ -d {VPS_ROOT}/data/models/prod ] && cp -r {VPS_ROOT}/data/models/prod {backup_dir}/ || true")
    print(f"Remote models backup created at: {backup_dir}")
    
    # 4. Stop PM2 live trader
    print("\n--- STAGE 3: Stopping PM2 live trader daemon ---")
    try:
        run_remote_cmd("pm2 stop luna-v2-live-demo || true")
    except Exception as e:
        print(f"Warning stopping PM2: {e}")
        
    # 5. Extract models on VPS
    print("\n--- STAGE 4: Extracting new models on VPS ---")
    # Clean previous prod directory to avoid orphan seeds
    run_remote_cmd(f"rm -rf {VPS_ROOT}/data/models/prod")
    extract_cmd = f"unzip -o {VPS_ROOT}/data/models/{zip_name} -d {VPS_ROOT}/data/models/"
    run_remote_cmd(extract_cmd)
    
    # Clean up zip on VPS
    run_remote_cmd(f"rm {VPS_ROOT}/data/models/{zip_name}")
    
    # 6. Verify model load and execution with a single operational cycle
    print("\n--- STAGE 5: Running post-upload live trader validation (once) ---")
    live_trader_cmd = f"cd {VPS_ROOT} && {VPS_PYTHON} scripts/run_live_trader.py --once --demo"
    try:
        run_remote_cmd(live_trader_cmd)
        print("\n[MODELS SUCCESS] Live trader successfully loaded the new 29-seed ensemble and executed one cycle.")
    except Exception as e:
        print(f"\n[MODELS FAILURE] Live trader failed on new ensemble: {e}")
        print("Restoring old models backup...")
        run_remote_cmd(f"rm -rf {VPS_ROOT}/data/models/prod && cp -r {backup_dir}/prod {VPS_ROOT}/data/models/")
        print("Backup restored successfully. Please inspect live trader logs.")
        sys.exit(1)
        
    # 7. Restart PM2 live trader
    print("\n--- STAGE 6: Restarting PM2 live trader ---")
    run_remote_cmd("pm2 start luna-v2-live-demo")
    run_remote_cmd("pm2 list")
    
    # 8. Clean up local zip
    if local_zip_path.exists():
        local_zip_path.unlink()
        
    print("\n[SUCCESS] New 29-seed models successfully deployed and verified on VPS.")

if __name__ == "__main__":
    main()
