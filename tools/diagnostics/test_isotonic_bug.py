import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

def test_isotonic_clipping():
    print("="*50)
    print("TEST: ISOTONIC REGRESSION OOS CLIPPING BUG")
    print("="*50)
    
    # 1. Simular datos de calibración In-Sample (validation set)
    # Por la naturaleza del modelo y el filtro XGBoost previo, 
    # supongamos que el XGBoost filtró señales de modo que raw_probs van de 0.51 a 0.85
    np.random.seed(42)
    raw_probs_train = np.random.uniform(0.51, 0.85, size=500)
    
    # y = 1 con más probabilidad si raw_prob es más alto
    y_train = (np.random.rand(500) < raw_probs_train).astype(int)
    
    # 2. Ajustar IsotonicRegression (Como lo hace calibrate_probabilities.py)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_probs_train, y_train)
    
    # 3. Simular datos Out-of-Sample (ej. Ventana W4 donde hay un rally obvio)
    # Aquí el modelo de ML (RandomForest/XGB) escupe una probabilidad brutal de 0.99
    raw_probs_oos = np.array([0.70, 0.80, 0.85, 0.90, 0.95, 0.99])
    
    cal_probs_oos = iso.predict(raw_probs_oos)
    
    print(f"{'Raw Prob (OOS)':<15} | {'Calibrated Prob':<15}")
    print("-" * 35)
    for raw, cal in zip(raw_probs_oos, cal_probs_oos):
        print(f"{raw:<15.2f} | {cal:<15.4f}")
        
    print("\nCONCLUSION ISOTONIC REGRESSION:")
    print("Como puedes ver, a partir de 0.85 (el máximo visto en entrenamiento),")
    print("IsotonicRegression DESTRUYE la capacidad discriminativa del MetaLabeler.")
    print("Un trade con confianza 0.85, 0.90, 0.95 y 0.99 reciben la MISMA probabilidad calibrada.")
    print("El MetaLabeler queda ciego ante las oportunidades más asimétricas y extremas.")
    print("Solución: cambiar a out_of_bounds='nan' con fallback inteligente, o usar extrapolación lineal.")

if __name__ == '__main__':
    test_isotonic_clipping()
