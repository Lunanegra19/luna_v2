"""Diagnóstico: inspeccionar por qué los modelos multi-agente son MOCK en W1."""
import sys, numpy as np, pandas as pd
from pathlib import Path

cache_dir = Path('g:/Mi unidad/ia/luna_v2/data/wfb_cache')

print("=== ESTADO DE MODELOS POR VENTANA ===")
print()
for w in ['W1', 'W2', 'W3', 'W4', 'W5']:
    wdir = cache_dir / w / 'models'
    if not wdir.exists():
        continue
    print(f"{w}:")
    for agent in ['bull_long', 'range_long', 'bear_long']:
        mpath = wdir / f"xgboost_meta_{agent}.model"
        if not mpath.exists():
            print(f"  {agent}: NO EXISTE")
            continue
        size_kb = mpath.stat().st_size // 1024
        with open(mpath, 'rb') as f:
            header = f.read(4)
        is_json = header[0:1] == b'{'
        if is_json:
            try:
                with open(mpath, 'r', encoding='latin-1', errors='replace') as f:
                    content = f.read(400)
                reason = ''
                n_is = '?'
                n_val = '?'
                if '"reason"' in content:
                    start = content.find('"reason"') + 9
                    reason = content[start:start+100].strip().lstrip(':').strip().strip('"').split('"')[0]
                if 'n_is_samples' in content:
                    s = content.find('n_is_samples') + 14
                    n_is = content[s:s+10].strip().lstrip(':').strip().split(',')[0].split('}')[0].strip()
                if 'n_val_samples' in content:
                    s = content.find('n_val_samples') + 15
                    n_val = content[s:s+10].strip().lstrip(':').strip().split(',')[0].split('}')[0].strip()
                print(f"  {agent}: MOCK ({size_kb}KB) | reason=\"{reason[:60]}\" n_is={n_is} n_val={n_val}")
            except Exception as e:
                print(f"  {agent}: MOCK ({size_kb}KB) | error leyendo: {e}")
        else:
            print(f"  {agent}: REAL ({size_kb}KB) | header={header.hex()}")
    print()

print()
print("=== RAZON DE LOS MOCKS: buscar en train_xgboost_v2.py ===")
print()

# Buscar el codigo que genera los mocks
src = Path('g:/Mi unidad/ia/luna_v2/luna/models/train_xgboost_v2.py')
with open(src, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Buscar las condiciones que generan mocks
print("Condiciones en train_xgboost_v2.py que generan MOCK:")
in_block = False
for i, line in enumerate(lines):
    if 'mocked' in line.lower() or 'mock' in line.lower() or 'null_model' in line.lower():
        if 'json' in line.lower() or 'save' in line.lower() or 'write' in line.lower() or 'reason' in line.lower() or 'n_' in line.lower():
            print(f"  L{i+1}: {line.rstrip()}")

print()
print("=== BUSCAR UMBRALES MINIMOS DE MUESTRAS ===")
keywords = ['min_samples', 'n_samples', 'n_train', 'insufficient', 'not enough', 'too few', 'skip', 'fallback']
for i, line in enumerate(lines):
    if any(k in line.lower() for k in keywords):
        if any(x in line for x in ['<', '>=', 'if ', 'len(']):
            print(f"  L{i+1}: {line.rstrip()}")
