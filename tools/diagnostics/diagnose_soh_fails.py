"""
Fix 3 issues in run_sop_health_checks():
1. CHK-09: s usado antes de definirse -> mover s = line.strip() arriba del loop
2. CHK-06/07: pm2 no en PATH del server -> usar /usr/bin/pm2 o leer dump.pm2
3. Investigar dv_etf_flow_proxy
"""

# ─────────────────────────────────────────────────────
# Check 1: dv_etf_flow_proxy
# ─────────────────────────────────────────────────────
import pandas as pd
df = pd.read_parquet('/root/luna_v2/data/features/features_live.parquet')
etf_cols = [c for c in df.columns if 'etf' in c.lower() or 'dv_etf' in c.lower()]
print(f"[DV-ETF] Columnas ETF en parquet: {etf_cols}")
print(f"[DV-ETF] dv_etf_flow_proxy presente: {'dv_etf_flow_proxy' in df.columns}")
if 'dv_etf_flow_proxy' in df.columns:
    last = df['dv_etf_flow_proxy'].iloc[-1]
    nan_count = df['dv_etf_flow_proxy'].isna().sum()
    print(f"[DV-ETF] Ultimo valor: {last}, NaN count: {nan_count}")
else:
    print("[DV-ETF] NO EXISTE en parquet")

# ─────────────────────────────────────────────────────
# Check 2: pm2 path
# ─────────────────────────────────────────────────────
import subprocess, shutil
pm2_path = shutil.which('pm2')
print(f"\n[PM2] PATH del server: {pm2_path}")
# Test with explicit PATH
env_with_path = {'PATH': '/usr/bin:/bin:/usr/local/bin:/root/.nvm/versions/node/v18.20.4/bin'}
r = subprocess.run(['pm2', 'list'], capture_output=True, text=True, timeout=5, env=env_with_path)
print(f"[PM2] pm2 list con PATH extendido: returncode={r.returncode}, output len={len(r.stdout)}")
print(f"[PM2] 'luna-v2-live-demo' en output: {'luna-v2-live-demo' in r.stdout}")

# Test dump.pm2
import json
with open('/root/.pm2/dump.pm2') as f:
    dump = json.load(f)
procs = {p.get('name'): p.get('pm2_env', {}).get('status', 'unknown') for p in dump.get('list', [])}
print(f"[PM2-DUMP] Procesos en dump.pm2: {procs}")
