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

def zip_codebase(zip_path):
    print(f"Creating local codebase zip at: {zip_path}")
    folders_to_include = ["luna", "scripts", "dashboard", "tools"]
    files_to_include = [Path("config/settings.yaml")]
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Include directories
        for folder in folders_to_include:
            folder_path = _ROOT / folder
            if not folder_path.exists():
                continue
            for root, dirs, files in os.walk(folder_path):
                # Skip pycache and git folders
                if "__pycache__" in Path(root).parts or ".git" in Path(root).parts:
                    continue
                for file in files:
                    # Skip binary pyc or raw data files
                    if file.endswith(".pyc") or file.endswith(".log") or file.endswith(".tmp"):
                        continue
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(_ROOT)
                    zipf.write(file_path, rel_path)
                    
        # Include individual files
        # [MEJORA-DEPLOY-SETTINGS 2026-06-20] Prefer settings_vps.yaml mapped to settings.yaml
        vps_settings = _ROOT / "config" / "settings_vps.yaml"
        if vps_settings.exists():
            zipf.write(vps_settings, "config/settings.yaml")
            print("[DEPLOY] Zipped config/settings_vps.yaml as config/settings.yaml")
        else:
            for f in files_to_include:
                f_path = _ROOT / f
                if f_path.exists():
                    zipf.write(f_path, f)
                
    print(f"Zip created successfully. File size: {zip_path.stat().st_size / (1024*1024):.2f} MB")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"luna_code_{timestamp}.zip"
    local_zip_path = _ROOT / "scratch" / zip_name
    local_zip_path.parent.mkdir(exist_ok=True)
    
    # 1. Zip local codebase
    zip_codebase(local_zip_path)
    
    # 2. SCP zip file to VPS scratch directory
    print("\n--- STAGE 1: Uploading codebase to VPS ---")
    run_remote_cmd(f"mkdir -p {VPS_ROOT}/scratch")
    scp_cmd = [
        "scp", "-o", "StrictHostKeyChecking=no",
        str(local_zip_path),
        f"{VPS_USER}@{VPS_IP}:{VPS_ROOT}/scratch/{zip_name}"
    ]
    run_local_cmd(scp_cmd)
    
    # 3. Create remote backup
    print("\n--- STAGE 2: Creating remote backup of old codebase ---")
    backup_dir = f"/root/luna_v2_backups/backup_{timestamp}"
    run_remote_cmd(f"mkdir -p {backup_dir}")
    
    # Backup directories and settings if they exist on VPS
    backup_cmds = []
    for d in ["luna", "scripts", "dashboard", "tools", "config/settings.yaml"]:
        backup_cmds.append(f"[ -e {VPS_ROOT}/{d} ] && cp -r {VPS_ROOT}/{d} {backup_dir}/ || true")
    run_remote_cmd(" && ".join(backup_cmds))
    print(f"Remote backup created at: {backup_dir}")
    
    # 4. Stop PM2 processes to ensure clean replacement
    print("\n--- STAGE 3: Stopping PM2 live trader & dashboard ---")
    try:
        run_remote_cmd("pm2 stop luna-v2-live-demo || true")
        run_remote_cmd("pm2 stop luna-dashboard || true")
    except Exception as e:
        print(f"Warning stopping PM2 processes: {e}")
        
    # 5. Extract codebase on VPS
    print("\n--- STAGE 4: Extracting new codebase on VPS ---")
    extract_cmd = f"unzip -o {VPS_ROOT}/scratch/{zip_name} -d {VPS_ROOT}"
    run_remote_cmd(extract_cmd)
    
    # Clean up zip on VPS
    run_remote_cmd(f"rm {VPS_ROOT}/scratch/{zip_name}")
    
    # 6. Validate VPS installation with train production --dry-run
    print("\n--- STAGE 5: Running pre-flight dry-run on VPS ---")
    dry_run_cmd = f"cd {VPS_ROOT} && {VPS_PYTHON} scripts/train_production_ensemble.py --dry-run"
    try:
        run_remote_cmd(dry_run_cmd)
        print("\n[DEPLOYMENT SUCCESS] Pre-flight dry-run completed successfully on the VPS.")
    except Exception as e:
        print(f"\n[DEPLOYMENT FAILURE] Pre-flight dry-run failed on the VPS: {e}")
        print("Restoring backup...")
        restore_cmds = []
        for d in ["luna", "scripts", "dashboard", "tools"]:
            restore_cmds.append(f"rm -rf {VPS_ROOT}/{d} && cp -r {backup_dir}/{d} {VPS_ROOT}/")
        restore_cmds.append(f"cp {backup_dir}/settings.yaml {VPS_ROOT}/config/settings.yaml")
        run_remote_cmd(" && ".join(restore_cmds))
        print("Backup restored successfully. Please inspect dry-run errors.")
        sys.exit(1)
        
    # 7. Restart PM2 processes
    print("\n--- STAGE 6: Restarting PM2 services ---")
    run_remote_cmd("pm2 start luna-v2-live-demo")
    run_remote_cmd("pm2 start luna-dashboard")
    run_remote_cmd("pm2 list")
    
    # 8. Clean up local zip
    if local_zip_path.exists():
        local_zip_path.unlink()
        
    print("\n[SUCCESS] Codebase successfully deployed and verified on VPS.")

if __name__ == "__main__":
    main()
