import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import interp1d

class ExtrapolatingIsotonicRegression:
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds='clip')
        self.interpolator = None
        self.x_min = None
        self.x_max = None
        
    def fit(self, X, y):
        self.ir.fit(X, y)
        
        # Obtener los puntos de soporte (knots)
        X_knots = self.ir.X_thresholds_
        y_knots = self.ir.y_thresholds_
        
        self.x_min, self.x_max = X_knots[0], X_knots[-1]
        
        # Crear interpolador con extrapolación lineal
        # Eliminar duplicados en X (interp1d no lo soporta bien)
        _, unique_idx = np.unique(X_knots, return_index=True)
        X_unique = X_knots[unique_idx]
        y_unique = y_knots[unique_idx]
        
        if len(X_unique) > 1:
            self.interpolator = interp1d(X_unique, y_unique, kind='linear', fill_value='extrapolate')
        else:
            self.interpolator = None
            
        return self

    def predict(self, X):
        if self.interpolator is not None:
            preds = self.interpolator(X)
            # CRITICAL: Forzar que las probabilidades no superen [0, 1]
            return np.clip(preds, 0.0, 1.0)
        else:
            return self.ir.predict(X)

def test_extrapolation():
    print("="*50)
    print("TEST: ISOTONIC EXTRAPOLATION VS CLIP")
    print("="*50)
    
    np.random.seed(42)
    # raw_probs_train va de 0.51 a 0.85
    X_train = np.random.uniform(0.51, 0.85, size=500)
    y_train = (np.random.rand(500) < X_train**2).astype(int)
    
    # 1. Clip nativo
    iso_clip = IsotonicRegression(out_of_bounds='clip').fit(X_train, y_train)
    
    # 2. Nuestra extrapolación
    iso_extrap = ExtrapolatingIsotonicRegression().fit(X_train, y_train)
    
    X_test = np.array([0.70, 0.80, 0.84, 0.85, 0.88, 0.95, 0.99])
    
    y_clip = iso_clip.predict(X_test)
    y_extrap = iso_extrap.predict(X_test)
    
    print(f"{'Raw OOS':<10} | {'Sklearn Clip':<15} | {'Extrapolated (Clipped 0-1)':<25}")
    print("-" * 55)
    for x, y_c, y_e in zip(X_test, y_clip, y_extrap):
        print(f"{x:<10.2f} | {y_c:<15.4f} | {y_e:<25.4f}")

if __name__ == '__main__':
    test_extrapolation()
