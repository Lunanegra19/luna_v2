import pandas as pd
import numpy as np
import sys
import logging
sys.path.append("g:/Mi unidad/ia/luna_v2")

logging.basicConfig(level=logging.INFO)
from luna.models.train_metalabeler_v2 import MetaLabelerV2Trainer
from pathlib import Path

# Create a mock trainer
trainer = MetaLabelerV2Trainer("seed_42_W1")

# Mock dataframe
print("[BUG-FIX-PANDAS] Aplicando corrección de freq='H' a 'h' para compatibilidad con Pandas 2.0+")
dates = pd.date_range("2020-01-01", periods=1000, freq="h")
df = pd.DataFrame(index=dates)
df["Target_TBM_Bin"] = np.random.randint(0, 2, size=1000)
df["Target_TBM_Dir"] = np.random.choice([-1, 1], size=1000)
df["HMM_Regime"] = np.random.randint(0, 4, size=1000)
df["HMM_Semantic"] = np.random.choice(["BULL", "BEAR", "CRAB"], size=1000)
df["feature1"] = np.random.randn(1000)
df["feature2"] = np.random.randn(1000)
df["close"] = np.linspace(10000, 20000, 1000)

xgb_features = ["feature1", "feature2"]
valid_idx = np.arange(1000)

print("Testeando _load_data...")
# _load_data mock
try:
    trainer.df_val = df.copy()
    trainer.df_labeled = df.copy()
    trainer.y_all = df["Target_TBM_Bin"].values
except Exception as e:
    print("Error:", e)

print("Testeando Generación CPCV temporal...")
# CPCV C(10,2) inline mock
try:
    n = 1000
    N_CPCV_GROUPS = 10
    n_groups = N_CPCV_GROUPS
    group_size = n // n_groups
    groups = [list(range(i * group_size, min((i + 1) * group_size, n))) for i in range(n_groups)]
    cpcv_splits = []
    X_xgb_df = df[xgb_features]
    timestamps_xgb = X_xgb_df.index
    EMBARGO_H = 96
    
    from itertools import combinations
    for test_gidxs in combinations(range(n_groups), 2):
        test_flat = [i for gi in test_gidxs for i in groups[gi]]
        test_set = set(test_flat)
        
        # [MEJORA-EMBARGO-01] Purge temporal
        train_idx_all = np.array([i for i in range(n) if i not in test_set])
        if len(train_idx_all) == 0:
            continue
            
        train_mask = np.ones(len(train_idx_all), dtype=bool)
        for gi in test_gidxs:
            block = groups[gi]
            if not block:
                continue
            block_start = timestamps_xgb[block[0]]
            block_end   = timestamps_xgb[block[-1]]
            purge_lo    = block_start - pd.Timedelta(hours=EMBARGO_H)
            purge_hi    = block_end   + pd.Timedelta(hours=EMBARGO_H)
            
            in_purge_zone = (
                (timestamps_xgb[train_idx_all] >= purge_lo) &
                (timestamps_xgb[train_idx_all] <= purge_hi)
            )
            train_mask &= ~in_purge_zone
            
        train_idx = train_idx_all[train_mask].tolist()

        if len(train_idx) >= 100:
            cpcv_splits.append((train_idx, test_flat))

    print(f"[TEST] CPCV generado con {len(cpcv_splits)} splits usando purge temporal de {EMBARGO_H}H.")
except Exception as e:
    print("[ERROR CPCV] Falló la generación de CPCV:", e)

print("Test completado.")
