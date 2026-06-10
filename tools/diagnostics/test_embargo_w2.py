import os
import sys
import subprocess
import yaml
from pathlib import Path
import json

ROOT = Path('g:/Mi unidad/ia/luna_v2')
SETTINGS_PATH = ROOT / 'config' / 'settings.yaml'

def modify_settings(embargo, floor):
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Modify n_estimators_min_floor
    if 'xgboost' in data:
        data['xgboost']['n_estimators_min_floor'] = floor
    
    # Modify wfb_windows to ONLY have W2
    if 'wfb_windows' in data:
        w2 = data['wfb_windows'].get('W2')
        if w2:
            data['wfb_windows'] = {'W2': w2}
            
    # Modify embargo
    # There are multiple embargo_hours in different phases, we will change all of them for the test
    if 'pipeline' in data and 'windows' in data['pipeline']:
        data['pipeline']['embargo_hours'] = embargo
    if 'fase2' in data:
        data['fase2']['embargo_hours'] = embargo

    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)
        
    # Agregamos manualmente para asegurar que quede (yaml dump elimina comentarios pero es solo para test)
    # wait, yaml dump is fine for a temporary test

def run_wfb():
    cmd = [sys.executable, 'scripts/wfb_worker.py', '--seed', '12345', '--nocache']
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    
    # extract results from W2 telemetry
    trades = 0
    dsr = 0.0
    for line in result.stdout.split('\n') + result.stderr.split('\n'):
        if "W2 OOS_Trades" in line or "[W2] DSR" in line:
            print(line)
        if "ventana producirá" in line:
            print(line)
    return result

print("=== TEST 1: Embargo=72h, Floor=100 (Baseline actual) ===")
modify_settings(72, 100)
run_wfb()

print("\n=== TEST 2: Embargo=72h, Floor=5 (Fix early stopping) ===")
modify_settings(72, 5)
run_wfb()

print("\n=== TEST 3: Embargo=24h, Floor=5 (Fix ES + Embargo 24h) ===")
modify_settings(24, 5)
run_wfb()

# Restore settings using git
subprocess.run(['git', 'checkout', str(SETTINGS_PATH)], cwd=str(ROOT))
