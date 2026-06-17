
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

def test_single_class():
    print("Testing XGBClassifier with a single class in the training fold...")
    
    # Create dataset with 100 samples
    X = np.random.rand(100, 5)
    
    # Fold 1 train: indices 0 to 24. Make them all class 1.
    y = np.ones(100)
    y[25:] = 0  # Rest are class 0
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold}: train size {len(train_idx)}, test size {len(test_idx)}")
        X_train, y_train = X[train_idx], y[train_idx]
        # [FIX-SINGLE-CLASS-FOLD] Guard matemático.
        if len(np.unique(y_train)) < 2:
            print(f"Fold {fold} ignorado matemáticamente: solo contiene la clase {y_train[0]}. Predicción directa = {y_train[0]}")
            continue
            
        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        try:
            clf.fit(X_train, y_train)
            print(f"Fold {fold} fit successful.")
        except Exception as e:
            print(f"Fold {fold} fit FAILED: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    test_single_class()
