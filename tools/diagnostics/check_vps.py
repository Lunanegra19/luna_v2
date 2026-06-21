import subprocess
import sys

def run_remote(cmd):
    try:
        res = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10", "root@178.105.197.191", cmd], capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
        return res.stdout
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr}"

if __name__ == "__main__":
    print("=== VPS LIVE TRADER LOGS (OUT) ===")
    out_logs = run_remote("pm2 describe luna-v2-live-demo")
    print(out_logs)
    
    print("\n=== PM2 LIST ===")
    err_logs = run_remote("pm2 list")
    print(err_logs)
