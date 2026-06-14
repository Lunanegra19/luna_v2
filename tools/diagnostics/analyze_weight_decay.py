import pandas as pd
import numpy as np
from pathlib import Path

# Need to load the parquet for W4 training data
# Usually training data is built inside train_xgboost_v2 or similar.
# Since we don't have the exact training parquet, we can build a proxy 
# by reading the baseline features and applying HMM.
# Actually, let's just check the code in train_xgboost_v2.py for the weight decay.

out = []
with open('c:/Users/Usuario/Desktop/ia/luna_v2/luna/models/train_xgboost_v2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'WEIGHT_DECAY_ALPHA' in line or 'sample_weight' in line:
            out.append(f"{i}: {line.strip()}")

Path('c:/Users/Usuario/Desktop/ia/luna_v2/data/reports/weight_decay_analysis.txt').write_text("\n".join(out), encoding='utf-8')
