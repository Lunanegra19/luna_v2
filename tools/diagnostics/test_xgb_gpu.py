import xgboost as xgb
try:
    xgb.XGBClassifier(tree_method="hist", device="cuda").fit([[0]], [0])
    print("XGBoost CUDA: AVAILABLE")
except Exception as e:
    print(f"XGBoost CUDA: FAILED - {e}")
